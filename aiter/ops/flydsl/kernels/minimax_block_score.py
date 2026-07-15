# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 sparse-prefill block-score (lightning indexer) -- FlyDSL gfx950.

Drop-in candidate for sglang's Triton
``_flash_attn_fwd_with_block_score_kernel`` in the DEPLOYED *score-only* path
(``disable_index_value=True`` -> v_cache=None, sink=None, no attention output).
For each query row ``r`` (head ``h``, batch ``b``) and each 128-key block ``j``::

    score[h, seq_start+r, j] = max_{key in block j, key <= q_abs, key < seq_len}
                                   (<Q[r], K_phys(key)> * sm_scale) * log2e   (SCORE_TYPE=max)

or the log2-sum-exp over that block (SCORE_TYPE=lse). ``K_phys(key)`` is gathered
paged via ``req_to_token[sid, key] -> slot -> k_cache[slot, 0, :]``. Blocks whose
keys are entirely in the causal future of a query stay ``-inf`` (the caller
prefills ``score`` with ``-inf``; the kernel writes only real maxima).

Structural difference vs the Triton kernel (why this can be faster): in the
score-only path each 128-key block's score is INDEPENDENT (no cross-block
online-softmax dependency), but the Triton kernel keeps the serial K-walk from
the fused-attention path, so its grid is only ``(ceil(Sq/BQ), batch*heads)`` --
64 thread blocks for the deployment shape (batch=1, 1 head, Sq=8192) on a
256-CU MI355X. This kernel parallelizes over (q-tile x k-group) so the grid is
thousands of blocks and the device is filled. Same total K bytes, higher
achieved HBM bandwidth.

MFMA: bf16 16x16x32 (CDNA4/gfx950). A=Q[query,d] (M=query,K=d), B=K[key,d]
(N=key,K=d) -> C[query,key], reduced over key to one score per query. The
per-lane 16x16 output holds C[m=(lane//16)*4+i, n=lane%16]; the max over the 128
keys is a register-max across the 8 N-tiles (keys sharing lane%16) followed by a
shuffle_xor reduce over the 16 lanes of each aligned group.
"""

# No `from __future__ import annotations`: FlyDSL arg typing needs real objects.

from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.numeric import ArithValue
from flydsl.expr.typing import T
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP, SmemAllocator, SmemPtr
from flydsl._mlir.dialects import scf
from flydsl._mlir import ir

from .tensor_shim import GTensor, STensor, _run_compiled, _to_raw

Vec = fx.Vector

MFMA_M = 16
MFMA_N = 16
MFMA_K = 32  # bf16 K per MFMA on gfx950
LOG2E = 1.4426950409

DEFAULT_COMPILE_HINTS = {"waves_per_eu": 2, "fast_fp_math": True}

NEG_SENTINEL = -1.0e30


def _build_block_score_kernel(
    *,
    head_dim: int,
    block_q: int,
    block_size: int,   # pooling / page size == N keys per score block (128)
    k_group: int,      # score-blocks processed per thread block (Q-reuse loop)
    score_type: str,
    waves_per_block: int = 1,
    use_lds: bool = False,
):
    assert score_type in ("max", "lse")
    D = head_dim
    BQ = block_q
    BS = block_size
    G = k_group
    W = waves_per_block
    assert D % MFMA_K == 0, f"head_dim {D} must be a multiple of {MFMA_K}"
    assert BQ % MFMA_M == 0, f"block_q {BQ} must be a multiple of {MFMA_M}"
    assert BS % MFMA_N == 0, f"block_size {BS} must be a multiple of {MFMA_N}"
    assert G % W == 0, f"k_group {G} must be divisible by waves_per_block {W}"
    if use_lds:
        assert W == 1, "use_lds path requires waves_per_block==1 (shared LDS K tile)"
    M_TILES = BQ // MFMA_M
    N_TILES = BS // MFMA_N
    K_STEPS = D // MFMA_K
    G_PER_WAVE = G // W       # k-blocks each wave owns (disjoint slice, MLP)
    BLOCK_THREADS = 64 * W

    # LDS staging of the [BS, D] bf16 K tile (coalesced global load once per
    # k-block, reused across all M_TILES; fixes the uncoalesced per-lane global
    # MFMA-operand gather that caps the direct-from-global path).
    K_TILE_ELEMS = BS * D
    LDS_VEC = 8
    N_STAGE_LOADS = K_TILE_ELEMS // (BLOCK_THREADS * LDS_VEC)
    allocator = None
    smem_k_offset = 0
    if use_lds:
        assert K_TILE_ELEMS % (BLOCK_THREADS * LDS_VEC) == 0
        GPU_ARCH = get_rocm_arch()
        allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
        smem_k_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = smem_k_offset + K_TILE_ELEMS * 2  # bf16 bytes
        assert allocator.ptr <= SMEM_CAPACITY_MAP[GPU_ARCH]

    fm_fast = arith.FastMathFlags.fast
    _kname = (
        f"minimax_block_score_D{D}_bq{BQ}_bs{BS}_g{G}_w{W}"
        f"_{'lds' if use_lds else 'reg'}_{score_type}_flydsl"
    )

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        q: fx.Tensor,            # [total_q, num_heads, D] bf16
        k_cache: fx.Tensor,      # [max_slots, num_kv_heads, D] bf16
        score: fx.Tensor,        # [num_heads, total_q, max_seqblock_k] f32 (-inf prefilled)
        req_to_token: fx.Tensor,  # [max_reqs, max_kv_len] i32
        slot_ids: fx.Tensor,     # [batch] i32
        cu_seqlens: fx.Tensor,   # [batch+1] i32
        seq_lens: fx.Tensor,     # [batch] i32
        prefix_lens: fx.Tensor,  # [batch] i32
        num_heads: fx.Int32,
        gqa_group_size: fx.Int32,
        max_slots: fx.Int32,
        sm_scale_log2e: fx.Float32,
        stride_q_n: fx.Int32,
        stride_q_h: fx.Int32,
        stride_k_s: fx.Int32,
        stride_k_h: fx.Int32,
        stride_r2t_b: fx.Int32,
        stride_s_h: fx.Int32,
        stride_s_q: fx.Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        neg = arith.constant(NEG_SENTINEL, type=f32)
        scale_c = _to_raw(sm_scale_log2e)
        mfma_res_ty = T.vec(4, f32)

        tid = fx.thread_idx.x
        wave = fx.Int32(arith.divui(_to_raw(tid), _to_raw(fx.Int32(64))))       # 0..W-1
        lane = fx.Int32(arith.remui(_to_raw(tid), _to_raw(fx.Int32(64))))
        m_group = fx.Int32(arith.divui(_to_raw(lane), _to_raw(fx.Int32(16))))  # 0..3
        n_lane = fx.Int32(arith.remui(_to_raw(lane), _to_raw(fx.Int32(16))))   # 0..15

        pid_q = fx.Int32(fx.block_idx.x)
        pid_kg = fx.Int32(fx.block_idx.y)
        pid_bh = fx.Int32(fx.block_idx.z)
        pid_b = fx.Int32(arith.divui(_to_raw(pid_bh), _to_raw(num_heads)))
        pid_h = fx.Int32(arith.remui(_to_raw(pid_bh), _to_raw(num_heads)))
        pid_kh = fx.Int32(arith.divui(_to_raw(pid_h), _to_raw(gqa_group_size)))

        cu_t = GTensor(cu_seqlens, dtype=i32, shape=(-1,))
        sl_t = GTensor(seq_lens, dtype=i32, shape=(-1,))
        pl_t = GTensor(prefix_lens, dtype=i32, shape=(-1,))
        sid_t = GTensor(slot_ids, dtype=i32, shape=(-1,))
        ind_t = GTensor(req_to_token, dtype=i32, shape=(-1,))
        q_t = GTensor(q, dtype=T.bf16, shape=(-1,))
        k_t = GTensor(k_cache, dtype=T.bf16, shape=(-1,))
        sc_t = GTensor(score, dtype=f32, shape=(-1,))

        if const_expr(use_lds):
            ks_lds = STensor(
                SmemPtr(allocator.get_base(), smem_k_offset, T.bf16,
                        shape=(K_TILE_ELEMS,)),
                T.bf16, shape=(K_TILE_ELEMS,),
            )

        seq_start = fx.Int32(cu_t[pid_b])
        q_len = fx.Int32(cu_t[pid_b + fx.Int32(1)]) - seq_start
        seq_len = fx.Int32(sl_t[pid_b])
        prefix_len = fx.Int32(pl_t[pid_b])
        sid = fx.Int32(sid_t[pid_b])

        q_tile_base = pid_q * BQ  # query index within the chunk
        # max absolute query position handled by this q-tile
        max_qpos = prefix_len + q_tile_base + fx.Int32(BQ - 1)
        kg_block0 = pid_kg * G          # first score-block index of this group
        kg_key0 = kg_block0 * BS        # first key position

        # Active iff: q-tile in range AND k-group not entirely causal-future AND
        # k-group starts before seq_len. All real work is guarded by `active`.
        active = arith.andi(
            arith.cmpi(arith.CmpIPredicate.slt, _to_raw(q_tile_base), _to_raw(q_len)),
            arith.andi(
                arith.cmpi(arith.CmpIPredicate.sle, _to_raw(kg_key0), _to_raw(max_qpos)),
                arith.cmpi(arith.CmpIPredicate.slt, _to_raw(kg_key0), _to_raw(seq_len)),
            ),
        )

        # Preload Q fragments for all M-tiles (A operand: lane holds
        # Q[m=n_lane, d=k_step*32 + m_group*8 + 0..7]), reused across the whole
        # k-group. Preload beats per-M-tile reload (measured) because the reload's
        # extra L2 traffic outweighs the VGPR pressure at these tile sizes.
        d_lane = m_group * fx.Int32(8)
        q_row_hi = seq_start + q_len - fx.Int32(1)
        a_frag = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
        for mt in range_constexpr(M_TILES):
            q_row = seq_start + q_tile_base + fx.Int32(mt * 16) + n_lane
            q_row_c = fx.Int32(arith.minsi(_to_raw(q_row), _to_raw(q_row_hi)))
            base = q_row_c * stride_q_n + pid_h * stride_q_h
            for ks in range_constexpr(K_STEPS):
                a_frag[mt][ks] = q_t.vec_load(
                    (_to_raw(base + fx.Int32(ks * MFMA_K) + d_lane),), vec_size=8
                )

        # ---- k-group loop: each of the W waves owns a disjoint slice of
        # G_PER_WAVE score-blocks (memory-level parallelism); wave w handles
        # blocks [w*G_PER_WAVE, (w+1)*G_PER_WAVE). Each score-block -> one column.
        wave_g0 = wave * fx.Int32(G_PER_WAVE)
        for gl in range_constexpr(G_PER_WAVE):
            kb = kg_block0 + wave_g0 + fx.Int32(gl)  # score-block index
            key_block0 = kb * fx.Int32(BS)   # first key position of this block
            in_range = arith.andi(
                arith.cmpi(arith.CmpIPredicate.slt, _to_raw(key_block0), _to_raw(seq_len)),
                arith.cmpi(arith.CmpIPredicate.sle, _to_raw(key_block0), _to_raw(max_qpos)),
            )
            blk_if = scf.IfOp(arith.andi(in_range, active))
            with ir.InsertionPoint(blk_if.then_block):
                if const_expr(use_lds):
                    # Coalesced cooperative stage of the [BS, D] K tile into LDS.
                    # Each vec8 covers 8 contiguous d of one key (D % 8 == 0), so
                    # the global load is contiguous within a key (256 B/key), and
                    # the LDS store is contiguous. Reused across all M_TILES below.
                    for it in range_constexpr(N_STAGE_LOADS):
                        flat = fx.Int32((it * BLOCK_THREADS + 0)) + tid
                        flat = flat * fx.Int32(LDS_VEC)
                        key_l = fx.Int32(arith.divui(_to_raw(flat), _to_raw(fx.Int32(D))))
                        d_l = fx.Int32(arith.remui(_to_raw(flat), _to_raw(fx.Int32(D))))
                        key = key_block0 + key_l
                        key_c = fx.Int32(
                            arith.minsi(_to_raw(key), _to_raw(seq_len - fx.Int32(1)))
                        )
                        slot = fx.Int32(ind_t[sid * stride_r2t_b + key_c])
                        slot_c = fx.Int32(
                            arith.remui(_to_raw(slot + max_slots), _to_raw(max_slots))
                        )
                        goff = slot_c * stride_k_s + pid_kh * stride_k_h + d_l
                        v8 = k_t.vec_load((_to_raw(goff),), vec_size=LDS_VEC)
                        ks_lds.vec_store((fx.Index(flat),), v8, vec_size=LDS_VEC)
                    gpu.barrier()
                    k_off_nt = None
                else:
                    # Direct-from-global: precompute per-lane paged K byte offsets
                    # for the 8 N-tiles (slot for key = key_block0 + nt*16 + n_lane).
                    k_off_nt = [None] * N_TILES
                    for nt in range_constexpr(N_TILES):
                        key = key_block0 + fx.Int32(nt * 16) + n_lane
                        key_c = fx.Int32(
                            arith.minsi(_to_raw(key), _to_raw(seq_len - fx.Int32(1)))
                        )
                        r2t_off = sid * stride_r2t_b + key_c
                        slot = fx.Int32(ind_t[r2t_off])
                        slot_c = fx.Int32(
                            arith.remui(
                                _to_raw(slot + max_slots), _to_raw(max_slots)
                            )
                        )
                        k_off_nt[nt] = slot_c * stride_k_s + pid_kh * stride_k_h

                for mt in range_constexpr(M_TILES):
                    # per-query running values (4 queries per lane: m_group*4+i)
                    red = [_to_raw(neg) for _ in range_constexpr(4)]
                    if const_expr(score_type == "lse"):
                        qk_store = [[None] * 4 for _ in range_constexpr(N_TILES)]

                    for nt in range_constexpr(N_TILES):
                        acc = Vec.filled(4, 0.0, fx.Float32)
                        for ks in range_constexpr(K_STEPS):
                            if const_expr(use_lds):
                                # LDS layout [key, d]; b_frag = K[key=nt*16+n_lane,
                                # d=ks*32+m_group*8 .. +8].
                                lds_off = (
                                    (fx.Int32(nt * 16) + n_lane) * fx.Int32(D)
                                    + fx.Int32(ks * MFMA_K) + d_lane
                                )
                                b_frag = ks_lds.vec_load((fx.Index(lds_off),), vec_size=8)
                            else:
                                d0 = k_off_nt[nt] + fx.Int32(ks * MFMA_K) + d_lane
                                b_frag = k_t.vec_load((_to_raw(d0),), vec_size=8)
                            acc = rocdl.mfma_f32_16x16x32_bf16(
                                mfma_res_ty, [a_frag[mt][ks], b_frag, acc, 0, 0, 0]
                            )
                        # acc[i] = qk[m=m_group*4+i, n=nt*16+n_lane]
                        key_abs = key_block0 + fx.Int32(nt * 16) + n_lane
                        key_valid = arith.cmpi(
                            arith.CmpIPredicate.slt, _to_raw(key_abs), _to_raw(seq_len)
                        )
                        for i in range_constexpr(4):
                            qv = Vec(acc)[i].ir_value()
                            qv = arith.MulFOp(qv, scale_c, fastmath=fm_fast).result
                            q_abs = (
                                prefix_len + q_tile_base + fx.Int32(mt * 16)
                                + m_group * fx.Int32(4) + fx.Int32(i)
                            )
                            causal_ok = arith.cmpi(
                                arith.CmpIPredicate.sge, _to_raw(q_abs), _to_raw(key_abs)
                            )
                            ok = arith.andi(key_valid, causal_ok)
                            masked = arith.select(ok, qv, _to_raw(neg))
                            red[i] = arith.maximumf(red[i], masked)
                            if const_expr(score_type == "lse"):
                                qk_store[nt][i] = masked

                    # cross-lane max over the 16 lanes of each aligned group
                    gmax = [None] * 4
                    for i in range_constexpr(4):
                        v = red[i]
                        for sh in (1, 2, 4, 8):
                            peer = _to_raw(ArithValue(v).shuffle_xor(sh, 64))
                            v = arith.maximumf(v, peer)
                        gmax[i] = v

                    if const_expr(score_type == "max"):
                        out_val = gmax
                    else:
                        # lse = gmax + log2(sum_k exp2(qk - gmax))
                        out_val = [None] * 4
                        # AMD v_exp_f32 == 2^x and v_log_f32 == log2(x), so the
                        # log2-sum-exp stays entirely in base-2 (no ln<->log2 scale).
                        for i in range_constexpr(4):
                            psum = arith.constant(0.0, type=f32)
                            for nt in range_constexpr(N_TILES):
                                d = arith.SubFOp(
                                    qk_store[nt][i], gmax[i], fastmath=fm_fast
                                ).result
                                psum = arith.AddFOp(
                                    psum, rocdl.exp2(f32, d), fastmath=fm_fast
                                ).result
                            s = psum
                            for sh in (1, 2, 4, 8):
                                peer = _to_raw(ArithValue(s).shuffle_xor(sh, 64))
                                s = arith.AddFOp(s, peer, fastmath=fm_fast).result
                            log2_s = rocdl.log(f32, s)
                            lse = arith.AddFOp(gmax[i], log2_s, fastmath=fm_fast).result
                            out_val[i] = lse

                    # writer lanes (n_lane==0) store 4 queries for their m_group
                    is_writer = arith.cmpi(
                        arith.CmpIPredicate.eq, _to_raw(n_lane), _to_raw(fx.Int32(0))
                    )
                    with ir.InsertionPoint(scf.IfOp(is_writer).then_block):
                        for i in range_constexpr(4):
                            query = (
                                q_tile_base + fx.Int32(mt * 16)
                                + m_group * fx.Int32(4) + fx.Int32(i)
                            )
                            # write only in-bounds queries with a real (non-sentinel) max
                            qin = arith.cmpi(
                                arith.CmpIPredicate.slt, _to_raw(query), _to_raw(q_len)
                            )
                            real = arith.cmpf(
                                arith.CmpFPredicate.OGT,
                                out_val[i],
                                arith.constant(NEG_SENTINEL * 0.5, type=f32),
                            )
                            do_w = arith.andi(qin, real)
                            with ir.InsertionPoint(scf.IfOp(do_w).then_block):
                                s_off = (
                                    pid_h * stride_s_h
                                    + (seq_start + query) * stride_s_q
                                    + kb
                                )
                                sc_t[_to_raw(s_off)] = fx.Float32(out_val[i])
                                scf.YieldOp([])
                        scf.YieldOp([])
                scf.YieldOp([])

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k_cache: fx.Tensor,
        score: fx.Tensor,
        req_to_token: fx.Tensor,
        slot_ids: fx.Tensor,
        cu_seqlens: fx.Tensor,
        seq_lens: fx.Tensor,
        prefix_lens: fx.Tensor,
        grid_x: fx.Int32,
        grid_y: fx.Int32,
        grid_z: fx.Int32,
        num_heads: fx.Int32,
        gqa_group_size: fx.Int32,
        max_slots: fx.Int32,
        sm_scale_log2e: fx.Float32,
        stride_q_n: fx.Int32,
        stride_q_h: fx.Int32,
        stride_k_s: fx.Int32,
        stride_k_h: fx.Int32,
        stride_r2t_b: fx.Int32,
        stride_s_h: fx.Int32,
        stride_s_q: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        if const_expr(use_lds):
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()
        gx = arith.index_cast(T.index, _to_raw(grid_x))
        gy = arith.index_cast(T.index, _to_raw(grid_y))
        gz = arith.index_cast(T.index, _to_raw(grid_z))
        kernel._func.__name__ = _kname
        kernel(
            q, k_cache, score, req_to_token, slot_ids, cu_seqlens, seq_lens,
            prefix_lens, num_heads, gqa_group_size, max_slots, sm_scale_log2e,
            stride_q_n, stride_q_h, stride_k_s, stride_k_h, stride_r2t_b,
            stride_s_h, stride_s_q,
        ).launch(grid=(gx, gy, gz), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch


@lru_cache(maxsize=64)
def compile_block_score(
    *,
    head_dim: int,
    block_q: int = 64,
    block_size: int = 128,
    k_group: int = 8,
    score_type: str = "max",
    waves_per_block: int = 1,
    use_lds: bool = False,
):
    launcher = _build_block_score_kernel(
        head_dim=head_dim,
        block_q=block_q,
        block_size=block_size,
        k_group=k_group,
        score_type=score_type,
        waves_per_block=waves_per_block,
        use_lds=use_lds,
    )
    launcher.compile_hints = dict(DEFAULT_COMPILE_HINTS)
    return launcher


def flydsl_minimax_block_score(
    q,               # [total_q, num_heads, head_dim] bf16
    k_cache,         # [max_slots, num_kv_heads, head_dim] bf16
    req_to_token,    # [max_reqs, max_kv_len] int32
    slot_ids,        # [batch] int32
    cu_seqlens,      # [batch+1] int32
    seq_lens,        # [batch] int32
    prefix_lens,     # [batch] int32
    max_seqlen_q,
    max_seqlen_k,
    block_size_k=128,
    sm_scale=None,
    score_type="max",
    score=None,
    block_q=64,
    k_group=16,
    waves_per_block=1,
    use_lds=True,
    stream=None,
):
    """FlyDSL block-score (score-only path). Drop-in for the deployed Triton
    ``_flash_attn_fwd_with_block_score_kernel`` (disable_index_value=True).

    Default config (bq=64, k_group=16, use_lds=True) beats the Triton reference
    at the deployment shapes on gfx950: coalesced LDS staging of the [128,128]
    K tile (reused across M-tiles) removes the uncoalesced per-lane global gather
    that caps the direct-from-global MFMA path."""
    assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16
    total_q, num_heads, head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    gqa_group_size = num_heads // num_kv_heads
    batch = cu_seqlens.shape[0] - 1
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    sm_scale_log2e = float(sm_scale) * LOG2E
    max_seqblock_k = (max_seqlen_k + block_size_k - 1) // block_size_k
    if score is None:
        score = torch.full(
            (num_heads, total_q, max_seqblock_k),
            float("-inf"), dtype=torch.float32, device=q.device,
        )
    else:
        score.fill_(float("-inf"))

    launcher = compile_block_score(
        head_dim=head_dim,
        waves_per_block=waves_per_block,
        use_lds=use_lds,
        block_q=block_q,
        block_size=block_size_k,
        k_group=k_group,
        score_type=score_type,
    )

    n_q_tiles = (max_seqlen_q + block_q - 1) // block_q
    n_k_blocks = max_seqblock_k
    n_k_groups = (n_k_blocks + k_group - 1) // k_group
    grid = (n_q_tiles, n_k_groups, batch * num_heads)

    if stream is None:
        stream = torch.cuda.current_stream()

    with torch.cuda.device(q.device.index):
        _run_compiled(
            launcher, q, k_cache, score, req_to_token, slot_ids, cu_seqlens,
            seq_lens, prefix_lens,
            int(grid[0]), int(grid[1]), int(grid[2]),
            int(num_heads), int(gqa_group_size), int(max_slots),
            float(sm_scale_log2e),
            int(q.stride(0)), int(q.stride(1)),
            int(k_cache.stride(0)), int(k_cache.stride(1)),
            int(req_to_token.stride(0)),
            int(score.stride(0)), int(score.stride(1)),
            stream,
        )
    return score
