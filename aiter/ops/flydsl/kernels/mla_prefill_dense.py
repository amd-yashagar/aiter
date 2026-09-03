# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dense absorbed MLA prefill (FlyDSL) — bf16 MFMA on CDNA.

Absorb math (shared latent KV, online softmax, ``v_mfma_f32_16x16x16bf16_1k``)
with a dense prefill schedule:

  Grid ``(num_q_tiles, H, num_kv_splits)`` — Q-seq on M, one head per CTA,
    KV sequence split across Z for CU fill on long ``Skv``
  CTA = ``NUM_WARPS`` waves; each wave owns ``MFMA`` Q rows
  Streams KV tiles of ``BLOCK_N`` through LDS; reuses across the Q tile
  GEMM1 ``S = K @ Q^T`` so ``P`` stays in MFMA B-layout registers
  GEMM2 ``O += V @ P`` reading V from K LDS layout (no VT buffer —
    measured win: ~+16% vs staged VT on 512×4096 / gfx942)
  ``num_kv_splits>1``: write unnormalized fp32 ``(m,l,O)`` partials + GPU
    online-softmax combine (same KV traffic, more CTAs)

Default product path (gfx942, H=12 absorb, measured keep/reject):
  * ``buffer_load`` bf16×4 for Q/K; Q reload-at-use (``q_preload`` off, ~+2%)
  * K-only LDS ~18 KiB (``stage_vt`` off); optional VT buffer rejected (~−16%)
  * ``BLOCK_N=16`` (``BN=32`` rejected ~−9%); ``k_double_buffer`` off (~+2%)
  * Split-KV when ``ntiles*H << num_CU``
  * Vector ``buffer_store`` f32×4 / bf16×4 epilogue (``vec_epilogue``) —
    measured ~+4.6% kn / ~+4% e2e on 512×4096; default on

Contract:
  Q  : bf16 [total_q, H, qk_dim]
  KV : bf16 rows of qk_dim (paged/dense); V = leading v_dim
  O  : bf16 [total_q, H, v_dim]

NOTE: Do NOT use ``from __future__ import annotations``.
"""

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops, vector

MFMA = 16
WARP_SIZE = 64
VEC = 4
DEFAULT_NUM_WARPS = 4
DEFAULT_BLOCK_M = DEFAULT_NUM_WARPS * MFMA  # 64
DEFAULT_BLOCK_N = MFMA  # 16
DEFAULT_WAVES_PER_EU = 2
DEFAULT_QK_DIM = 576
DEFAULT_V_DIM = 512
_LOG2E = math.log2(math.e)
_NEG_INF = -3.4028234663852886e38
SUPPORTED_GFX = ("gfx942", "gfx950")

QK_HEAD_DIM = DEFAULT_QK_DIM
V_HEAD_DIM = DEFAULT_V_DIM
BLOCK_M = DEFAULT_BLOCK_M
BLOCK_N = DEFAULT_BLOCK_N

# Split-KV: aim to keep ~2 WGs/CU busy when the Q×H grid is small.
_SPLIT_WG_PER_CU = 2
_MAX_KV_SPLITS = 64


def _validate_geometry(
    qk_dim,
    v_dim,
    block_m,
    block_n,
    num_warps,
    k_double_buffer=True,
    stage_vt=True,
):
    if qk_dim % MFMA or qk_dim <= 0:
        raise ValueError(f"qk_dim must be a positive multiple of {MFMA}, got {qk_dim}")
    if v_dim % MFMA or v_dim <= 0 or v_dim > qk_dim:
        raise ValueError(f"v_dim invalid: {v_dim} (qk_dim={qk_dim})")
    if block_m != num_warps * MFMA:
        raise ValueError(
            f"block_m must be num_warps*{MFMA} (got {block_m}, warps={num_warps})"
        )
    if block_n % MFMA or block_n <= 0:
        raise ValueError(f"block_n must be a positive multiple of {MFMA}, got {block_n}")
    if qk_dim % VEC or v_dim % VEC:
        raise ValueError(f"qk_dim/v_dim must be multiples of {VEC}")
    # VT stages one MFMA-wide N-subtile (BN may be 2×MFMA); K holds the full BN tile.
    k_bufs = 2 if k_double_buffer else 1
    vt = (v_dim * MFMA) if stage_vt else 0
    lds = (k_bufs * block_n * qk_dim + vt) * 2
    if lds > 64 * 1024:
        raise ValueError(
            f"LDS {lds} B exceeds 64 KiB for bn={block_n} qk={qk_dim} v={v_dim} "
            f"k_dbuf={k_double_buffer} stage_vt={stage_vt}"
        )
    if k_double_buffer and block_n > MFMA:
        raise ValueError("k_double_buffer unsupported for block_n > MFMA")


def choose_num_kv_splits(
    *,
    num_q_tiles: int,
    num_heads: int,
    max_kv_len: int,
    block_n: int = DEFAULT_BLOCK_N,
    num_cu=None,
    max_splits: int = _MAX_KV_SPLITS,
) -> int:
    """Pick Z-splits so ``ntiles*H*splits`` approaches ``num_cu * WG_PER_CU``.

    Skips splitting on short KV / already-full grids — split overhead (Q reload +
    partial combine) dominates there.
    """
    base = max(int(num_q_tiles) * int(num_heads), 1)
    kv = int(max_kv_len)
    bn = int(block_n)
    if kv < 1024:
        return 1
    if num_cu is None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        num_cu = int(getattr(props, "multi_processor_count", 304) or 304)
    if base >= int(num_cu):
        return 1
    target = max(int(num_cu) * _SPLIT_WG_PER_CU, base)
    splits = (target + base - 1) // base
    # Prefer ~512 tokens/split; avoid oversplitting into tiny CTA work.
    by_chunk = max(1, kv // 512)
    max_by_tiles = max(1, (kv + bn - 1) // bn)
    return max(1, min(int(splits), int(by_chunk), int(max_by_tiles), int(max_splits)))


def _bf16x4_from_vec(vec):
    return vector.bitcast(T.i16x4, fx.Vector(vec))


def _mfma(a, b, c):
    return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a, b, c, 0, 0, 0])


@lru_cache(maxsize=64)
def compile_mla_prefill_dense(
    *,
    qk_dim: int = DEFAULT_QK_DIM,
    v_dim: int = DEFAULT_V_DIM,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    num_warps: int = DEFAULT_NUM_WARPS,
    is_causal: bool = False,
    waves_per_eu: int = DEFAULT_WAVES_PER_EU,
    k_double_buffer: bool = False,
    q_preload: bool = False,
    stage_vt: bool = False,
    vec_epilogue: bool = True,
):
    _validate_geometry(
        qk_dim,
        v_dim,
        block_m,
        block_n,
        num_warps,
        k_double_buffer=k_double_buffer,
        stage_vt=stage_vt,
    )
    QK_DIM, V_DIM = int(qk_dim), int(v_dim)
    BM, BN = int(block_m), int(block_n)
    NW = int(num_warps)
    NT = NW * WARP_SIZE
    CAUSAL = bool(is_causal)
    K_DBUF = bool(k_double_buffer)
    Q_PRELOAD = bool(q_preload)
    STAGE_VT = bool(stage_vt)
    VEC_EPI = bool(vec_epilogue)
    NUM_QK_K = QK_DIM // MFMA
    NUM_V_K = V_DIM // MFMA
    NUM_N = BN // MFMA  # 1 for BN=16; 2 for BN=32 (N-subtile MFMA loop)
    LDS_K = BN * QK_DIM
    # Stage one MFMA-wide N strip of V^T so BN=32 fits in 64 KiB LDS.
    LDS_VT = V_DIM * MFMA if STAGE_VT else 0
    LDS_K_TOTAL = LDS_K * (2 if K_DBUF else 1)
    if LDS_K % NT or (STAGE_VT and LDS_VT % NT):
        raise ValueError(f"LDS must divide NT={NT}: k={LDS_K} vt={LDS_VT}")
    USE_VEC_K = LDS_K % (NT * VEC) == 0
    USE_VEC_VT = STAGE_VT and LDS_VT % (NT * VEC) == 0
    K_NVEC = LDS_K // (NT * VEC) if USE_VEC_K else 0
    VT_NVEC = LDS_VT // (NT * VEC) if USE_VEC_VT else 0
    K_PER = LDS_K // NT
    VT_PER = LDS_VT // NT if STAGE_VT else 0

    attrs = {"rocdl.waves_per_eu": int(waves_per_eu)} if waves_per_eu >= 1 else {}
    hints = {"waves_per_eu": int(waves_per_eu)} if waves_per_eu >= 1 else {}

    if STAGE_VT:

        @fx.struct
        class SharedStorage:
            k: fx.Array[fx.BFloat16, LDS_K_TOTAL, 16]
            vt: fx.Array[fx.BFloat16, LDS_VT, 16]

    else:

        @fx.struct
        class SharedStorage:
            k: fx.Array[fx.BFloat16, LDS_K_TOTAL, 16]

    @flyc.kernel(known_block_size=[NT, 1, 1])
    def kn(
        q: fx.Tensor,
        kv: fx.Tensor,
        o: fx.Tensor,
        partial_o: fx.Tensor,
        partial_ml: fx.Tensor,
        tile_q_start: fx.Tensor,
        tile_batch: fx.Tensor,
        qo_indptr: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_indices: fx.Tensor,
        page_size: fx.Int32,
        num_heads: fx.Int32,
        num_tiles: fx.Int32,
        num_splits: fx.Int32,
        sm_scale: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        tile = fx.Int32(gpu.block_id("x"))
        head = fx.Int32(gpu.block_id("y"))
        split = fx.Int32(gpu.block_id("z"))
        warp = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)

        q_rsrc = buffer_ops.create_buffer_resource(q, max_size=True)
        kv_rsrc = buffer_ops.create_buffer_resource(kv, max_size=True)
        o_rsrc = buffer_ops.create_buffer_resource(o, max_size=True)
        po_rsrc = buffer_ops.create_buffer_resource(partial_o, max_size=True)
        tqs = fx.rocdl.make_buffer_tensor(tile_q_start)
        tb = fx.rocdl.make_buffer_tensor(tile_batch)
        qo = fx.rocdl.make_buffer_tensor(qo_indptr)
        kip = fx.rocdl.make_buffer_tensor(kv_indptr)
        kix = fx.rocdl.make_buffer_tensor(kv_indices)
        o_buf = fx.rocdl.make_buffer_tensor(o)
        po_buf = fx.rocdl.make_buffer_tensor(partial_o)
        pml_buf = fx.rocdl.make_buffer_tensor(partial_ml)
        sm = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_k = sm.k
        # Alias only for type-checker paths; GEMM2 uses K layout when STAGE_VT=False.
        lds_vt = sm.vt if STAGE_VT else sm.k

        ab_row = lane % fx.Int32(MFMA)
        ab_col0 = (lane // fx.Int32(MFMA)) * fx.Int32(VEC)
        c_row0, c_col = ab_col0, ab_row

        q0 = fx.Int32(tqs[tile])
        b = fx.Int32(tb[tile])
        q_seq0 = fx.Int32(qo[b])
        q_seq1 = fx.Int32(qo[b + 1])
        q_len = q_seq1 - q_seq0
        kv0 = fx.Int32(kip[b])
        kv1 = fx.Int32(kip[b + 1])
        kv_len = kv1 - kv0

        chunk = (kv_len + num_splits - fx.Int32(1)) // num_splits
        n_begin = split * chunk
        n_end_raw = n_begin + chunk
        n_end = (n_end_raw < kv_len).select(n_end_raw, kv_len)
        split_len_raw = n_end - n_begin
        split_len = (split_len_raw > fx.Int32(0)).select(split_len_raw, fx.Int32(0))

        q_row = q0 + warp * fx.Int32(MFMA) + ab_row
        q_base = (q_row * num_heads + head) * fx.Int32(QK_DIM)

        log2e = fx.Float32(_LOG2E)
        neg_inf = fx.Float32(_NEG_INF)
        z = fx.Float32(0.0)
        one = fx.Float32(1.0)
        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        bf0 = fx.BFloat16(0.0)
        lds_k_elems = fx.Int32(LDS_K)

        def load_k_tile(base, n0i):
            if const_expr(USE_VEC_K):
                for e in range_constexpr(K_NVEC):
                    flat0 = (tid * fx.Int32(K_NVEC) + fx.Int32(e)) * fx.Int32(VEC)
                    row = flat0 // fx.Int32(QK_DIM)
                    col = flat0 % fx.Int32(QK_DIM)
                    jj = n0i + row
                    ok = jj < n_end
                    safe_jj = ok.select(jj, fx.Int32(0))
                    page = fx.Int32(kix[kv0 + safe_jj // page_size])
                    phys = page * page_size + (safe_jj % page_size)
                    g = ok.select(phys * fx.Int32(QK_DIM) + col, fx.Int32(0))
                    vec = fx.Vector(
                        buffer_ops.buffer_load(
                            kv_rsrc, g, vec_width=VEC, dtype=T.bf16
                        )
                    )
                    lds_k[base + flat0 + 0] = ok.select(fx.BFloat16(vec[0]), bf0)
                    lds_k[base + flat0 + 1] = ok.select(fx.BFloat16(vec[1]), bf0)
                    lds_k[base + flat0 + 2] = ok.select(fx.BFloat16(vec[2]), bf0)
                    lds_k[base + flat0 + 3] = ok.select(fx.BFloat16(vec[3]), bf0)
            else:
                for e in range_constexpr(K_PER):
                    flat = tid + fx.Int32(e) * fx.Int32(NT)
                    row = flat // fx.Int32(QK_DIM)
                    col = flat % fx.Int32(QK_DIM)
                    jj = n0i + row
                    ok = jj < n_end
                    safe_jj = ok.select(jj, fx.Int32(0))
                    page = fx.Int32(kix[kv0 + safe_jj // page_size])
                    phys = page * page_size + (safe_jj % page_size)
                    g = ok.select(phys * fx.Int32(QK_DIM) + col, fx.Int32(0))
                    raw = buffer_ops.buffer_load(
                        kv_rsrc, g, vec_width=1, dtype=T.bf16
                    )
                    lds_k[base + flat] = ok.select(fx.BFloat16(raw), bf0)

        def transpose_k_to_vt(base, n_sub):
            """K[BN,QK] → VT[V,MFMA] for N-subtile ``n_sub`` (cols n_sub*MFMA:+MFMA)."""
            n_off = fx.Int32(n_sub * MFMA)
            if const_expr(USE_VEC_VT):
                for e in range_constexpr(VT_NVEC):
                    flat0 = (tid * fx.Int32(VT_NVEC) + fx.Int32(e)) * fx.Int32(VEC)
                    d = flat0 // fx.Int32(MFMA)
                    n0_v = flat0 % fx.Int32(MFMA)
                    lds_vt[flat0 + 0] = lds_k[
                        base + (n_off + n0_v) * fx.Int32(QK_DIM) + d
                    ]
                    lds_vt[flat0 + 1] = lds_k[
                        base + (n_off + n0_v + 1) * fx.Int32(QK_DIM) + d
                    ]
                    lds_vt[flat0 + 2] = lds_k[
                        base + (n_off + n0_v + 2) * fx.Int32(QK_DIM) + d
                    ]
                    lds_vt[flat0 + 3] = lds_k[
                        base + (n_off + n0_v + 3) * fx.Int32(QK_DIM) + d
                    ]
            else:
                for e in range_constexpr(VT_PER):
                    flat = tid + fx.Int32(e) * fx.Int32(NT)
                    d = flat // fx.Int32(MFMA)
                    n = flat % fx.Int32(MFMA)
                    lds_vt[flat] = lds_k[base + (n_off + n) * fx.Int32(QK_DIM) + d]

        def v_a_vals(k_base, n_sub, di):
            """MFMA-A bf16×4 for V dim strip ``di`` (from VT LDS or K layout)."""
            vals = []
            for v in range_constexpr(4):
                d = fx.Int32(di * MFMA) + ab_row
                kv_r = ab_col0 + fx.Int32(v)
                if const_expr(STAGE_VT):
                    vals.append(fx.BFloat16(lds_vt[d * fx.Int32(MFMA) + kv_r]))
                else:
                    vals.append(
                        fx.BFloat16(
                            lds_k[
                                k_base
                                + (fx.Int32(n_sub * MFMA) + kv_r) * fx.Int32(QK_DIM)
                                + d
                            ]
                        )
                    )
            return vals

        m_i = neg_inf
        l_i = z
        o_acc = [zero4 for _ in range_constexpr(NUM_V_K)]

        q_frags = []
        if const_expr(Q_PRELOAD):
            for ki in range_constexpr(NUM_QK_K):
                q_frags.append(
                    _bf16x4_from_vec(
                        buffer_ops.buffer_load(
                            q_rsrc,
                            q_base + fx.Int32(ki * MFMA) + ab_col0,
                            vec_width=VEC,
                            dtype=T.bf16,
                        )
                    )
                )

        if const_expr(K_DBUF):
            load_k_tile(fx.Int32(0), n_begin)
            fx.gpu.barrier()
            if const_expr(STAGE_VT):
                transpose_k_to_vt(fx.Int32(0), 0)
                fx.gpu.barrier()
            init = [m_i, l_i, fx.Int32(0)] + o_acc
            final = init

            for n0, st in range(0, split_len, BN, init=init):
                m_cur = fx.Float32(st[0])
                l_cur = fx.Float32(st[1])
                phase = fx.Int32(st[2])
                o_cur = [fx.Vector(st[3 + di]) for di in range_constexpr(NUM_V_K)]
                n0i = n_begin + fx.Int32(n0)
                cur_base = phase * lds_k_elems
                nxt_base = (fx.Int32(1) - phase) * lds_k_elems

                # Prefetch next BN tile into idle K buffer (masked past n_end).
                load_k_tile(nxt_base, n0i + fx.Int32(BN))

                s = zero4
                for ki in range_constexpr(NUM_QK_K):
                    k_base = (
                        cur_base
                        + ab_row * fx.Int32(QK_DIM)
                        + fx.Int32(ki * MFMA)
                        + ab_col0
                    )
                    k_frag = vector.bitcast(
                        T.i16x4,
                        fx.Vector.from_elements(
                            [
                                fx.BFloat16(lds_k[k_base + 0]),
                                fx.BFloat16(lds_k[k_base + 1]),
                                fx.BFloat16(lds_k[k_base + 2]),
                                fx.BFloat16(lds_k[k_base + 3]),
                            ],
                            dtype=fx.BFloat16,
                        ),
                    )
                    if const_expr(Q_PRELOAD):
                        q_frag = q_frags[ki]
                    else:
                        q_frag = _bf16x4_from_vec(
                            buffer_ops.buffer_load(
                                q_rsrc,
                                q_base + fx.Int32(ki * MFMA) + ab_col0,
                                vec_width=VEC,
                                dtype=T.bf16,
                            )
                        )
                    s = _mfma(k_frag, q_frag, s)

                scores = []
                for v in range_constexpr(4):
                    sc = fx.Float32(s[v]) * sm_scale
                    q_pos = (q0 + warp * fx.Int32(MFMA) + c_col) - q_seq0
                    kv_col = n0i + c_row0 + fx.Int32(v)
                    ok = kv_col < n_end
                    if const_expr(CAUSAL):
                        lim = kv_len - q_len + q_pos + fx.Int32(1)
                        ok = ok & (kv_col < lim)
                    scores.append(ok.select(sc, neg_inf))

                m_tile = scores[0]
                for v in range_constexpr(3):
                    m_tile = m_tile.maximumf(scores[v + 1])
                for xor_m in (16, 32):
                    m_tile = m_tile.maximumf(
                        m_tile.shuffle_xor(fx.Int32(xor_m), fx.Int32(64))
                    )
                m_new = m_cur.maximumf(m_tile)
                empty = m_cur <= fx.Float32(_NEG_INF * 0.5)
                alpha = empty.select(z, fmath.exp2((m_cur - m_new) * log2e))

                p_f = []
                p_sum = z
                for v in range_constexpr(4):
                    pv = fmath.exp2((scores[v] - m_new) * log2e)
                    pv = (scores[v] > fx.Float32(_NEG_INF * 0.5)).select(pv, z)
                    p_f.append(pv)
                    p_sum = p_sum + pv
                for xor_m in (16, 32):
                    p_sum = p_sum + p_sum.shuffle_xor(fx.Int32(xor_m), fx.Int32(64))
                l_new = l_cur * alpha + p_sum
                p_frag = vector.bitcast(
                    T.i16x4,
                    fx.Vector.from_elements(
                        [p_f[v].to(fx.BFloat16) for v in range_constexpr(4)],
                        dtype=fx.BFloat16,
                    ),
                )

                o_new = []
                for di in range_constexpr(NUM_V_K):
                    scaled = fx.Vector.from_elements(
                        [
                            fx.Float32(o_cur[di][v]) * alpha
                            for v in range_constexpr(4)
                        ],
                        dtype=fx.Float32,
                    )
                    o_new.append(
                        _mfma(
                            vector.bitcast(
                                T.i16x4,
                                fx.Vector.from_elements(
                                    v_a_vals(cur_base, 0, di), dtype=fx.BFloat16
                                ),
                            ),
                            p_frag,
                            scaled,
                        )
                    )

                fx.gpu.barrier()
                if const_expr(STAGE_VT):
                    transpose_k_to_vt(nxt_base, 0)
                    fx.gpu.barrier()
                phase_new = fx.Int32(1) - phase
                final = yield [m_new, l_new, phase_new] + o_new

            final_m = fx.Float32(final[0])
            final_l = fx.Float32(final[1])
            final_o = [fx.Vector(final[3 + di]) for di in range_constexpr(NUM_V_K)]
        else:
            init = [m_i, l_i] + o_acc
            final = init
            for n0, st in range(0, split_len, BN, init=init):
                m_cur = fx.Float32(st[0])
                l_cur = fx.Float32(st[1])
                o_cur = [fx.Vector(st[2 + di]) for di in range_constexpr(NUM_V_K)]
                n0i = n_begin + fx.Int32(n0)
                load_k_tile(fx.Int32(0), n0i)
                fx.gpu.barrier()

                for n_sub in range_constexpr(NUM_N):
                    if const_expr(STAGE_VT):
                        transpose_k_to_vt(fx.Int32(0), n_sub)
                        fx.gpu.barrier()

                    s = zero4
                    row_off = fx.Int32(n_sub * MFMA)
                    for ki in range_constexpr(NUM_QK_K):
                        k_base = (
                            (ab_row + row_off) * fx.Int32(QK_DIM)
                            + fx.Int32(ki * MFMA)
                            + ab_col0
                        )
                        k_frag = vector.bitcast(
                            T.i16x4,
                            fx.Vector.from_elements(
                                [
                                    fx.BFloat16(lds_k[k_base + 0]),
                                    fx.BFloat16(lds_k[k_base + 1]),
                                    fx.BFloat16(lds_k[k_base + 2]),
                                    fx.BFloat16(lds_k[k_base + 3]),
                                ],
                                dtype=fx.BFloat16,
                            ),
                        )
                        if const_expr(Q_PRELOAD):
                            q_frag = q_frags[ki]
                        else:
                            q_frag = _bf16x4_from_vec(
                                buffer_ops.buffer_load(
                                    q_rsrc,
                                    q_base + fx.Int32(ki * MFMA) + ab_col0,
                                    vec_width=VEC,
                                    dtype=T.bf16,
                                )
                            )
                        s = _mfma(k_frag, q_frag, s)

                    scores = []
                    for v in range_constexpr(4):
                        sc = fx.Float32(s[v]) * sm_scale
                        q_pos = (q0 + warp * fx.Int32(MFMA) + c_col) - q_seq0
                        kv_col = n0i + row_off + c_row0 + fx.Int32(v)
                        ok = kv_col < n_end
                        if const_expr(CAUSAL):
                            lim = kv_len - q_len + q_pos + fx.Int32(1)
                            ok = ok & (kv_col < lim)
                        scores.append(ok.select(sc, neg_inf))

                    m_tile = scores[0]
                    for v in range_constexpr(3):
                        m_tile = m_tile.maximumf(scores[v + 1])
                    for xor_m in (16, 32):
                        m_tile = m_tile.maximumf(
                            m_tile.shuffle_xor(fx.Int32(xor_m), fx.Int32(64))
                        )
                    m_new = m_cur.maximumf(m_tile)
                    empty = m_cur <= fx.Float32(_NEG_INF * 0.5)
                    alpha = empty.select(z, fmath.exp2((m_cur - m_new) * log2e))

                    p_f = []
                    p_sum = z
                    for v in range_constexpr(4):
                        pv = fmath.exp2((scores[v] - m_new) * log2e)
                        pv = (scores[v] > fx.Float32(_NEG_INF * 0.5)).select(pv, z)
                        p_f.append(pv)
                        p_sum = p_sum + pv
                    for xor_m in (16, 32):
                        p_sum = p_sum + p_sum.shuffle_xor(
                            fx.Int32(xor_m), fx.Int32(64)
                        )
                    l_new = l_cur * alpha + p_sum
                    p_frag = vector.bitcast(
                        T.i16x4,
                        fx.Vector.from_elements(
                            [p_f[v].to(fx.BFloat16) for v in range_constexpr(4)],
                            dtype=fx.BFloat16,
                        ),
                    )

                    o_new = []
                    for di in range_constexpr(NUM_V_K):
                        scaled = fx.Vector.from_elements(
                            [
                                fx.Float32(o_cur[di][v]) * alpha
                                for v in range_constexpr(4)
                            ],
                            dtype=fx.Float32,
                        )
                        o_new.append(
                            _mfma(
                                vector.bitcast(
                                    T.i16x4,
                                    fx.Vector.from_elements(
                                        v_a_vals(fx.Int32(0), n_sub, di),
                                        dtype=fx.BFloat16,
                                    ),
                                ),
                                p_frag,
                                scaled,
                            )
                        )

                    m_cur = m_new
                    l_cur = l_new
                    o_cur = o_new
                    fx.gpu.barrier()

                final = yield [m_cur, l_cur] + o_cur

            final_m = fx.Float32(final[0])
            final_l = fx.Float32(final[1])
            final_o = [fx.Vector(final[2 + di]) for di in range_constexpr(NUM_V_K)]

        local_q = warp * fx.Int32(MFMA) + c_col
        use_split = num_splits > fx.Int32(1)

        if use_split:
            base_ml = (
                ((tile * num_heads + head) * num_splits + split) * fx.Int32(BM)
                + local_q
            ) * fx.Int32(2)
            pml_buf[base_ml + 0] = final_m
            pml_buf[base_ml + 1] = final_l
            base_o = (
                ((tile * num_heads + head) * num_splits + split) * fx.Int32(BM)
                + local_q
            ) * fx.Int32(V_DIM)
            if const_expr(VEC_EPI):
                for di in range_constexpr(NUM_V_K):
                    buffer_ops.buffer_store(
                        final_o[di],
                        po_rsrc,
                        base_o + fx.Int32(di * MFMA) + c_row0,
                    )
            else:
                for di in range_constexpr(NUM_V_K):
                    for v in range_constexpr(4):
                        d = fx.Int32(di * MFMA) + c_row0 + fx.Int32(v)
                        po_buf[base_o + d] = fx.Float32(final_o[di][v])
        else:
            inv = (final_l > z).select(one / final_l, z)
            qr = q0 + local_q
            o_row = (qr * num_heads + head) * fx.Int32(V_DIM)
            if const_expr(VEC_EPI):
                for di in range_constexpr(NUM_V_K):
                    bf = fx.Vector.from_elements(
                        [
                            (fx.Float32(final_o[di][v]) * inv).to(fx.BFloat16)
                            for v in range_constexpr(4)
                        ],
                        dtype=fx.BFloat16,
                    )
                    buffer_ops.buffer_store(
                        bf, o_rsrc, o_row + fx.Int32(di * MFMA) + c_row0
                    )
            else:
                for di in range_constexpr(NUM_V_K):
                    for v in range_constexpr(4):
                        val = (fx.Float32(final_o[di][v]) * inv).to(fx.BFloat16)
                        d = fx.Int32(di * MFMA) + c_row0 + fx.Int32(v)
                        o_buf[o_row + d] = val

    stream0 = fx.Stream(None)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        kv: fx.Tensor,
        o: fx.Tensor,
        partial_o: fx.Tensor,
        partial_ml: fx.Tensor,
        tile_q_start: fx.Tensor,
        tile_batch: fx.Tensor,
        qo_indptr: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_indices: fx.Tensor,
        page_size: fx.Int32,
        num_heads: fx.Int32,
        num_tiles: fx.Int32,
        num_splits: fx.Int32,
        sm_scale: fx.Float32,
        stream: fx.Stream = stream0,
    ):
        kn(
            q,
            kv,
            o,
            partial_o,
            partial_ml,
            tile_q_start,
            tile_batch,
            qo_indptr,
            kv_indptr,
            kv_indices,
            page_size,
            num_heads,
            num_tiles,
            num_splits,
            sm_scale,
            value_attrs=attrs,
        ).launch(
            grid=(num_tiles, num_heads, num_splits),
            block=(NT, 1, 1),
            stream=stream,
        )

    launch.compile_hints = dict(hints)
    return launch



def _pad_sequences_to_block_m(q, o, qo_indptr, block_m: int):
    """Pad each sequence length to a multiple of ``block_m``.

    When every sequence is already aligned, returns views (no ``cat``).
    """
    device = q.device
    batch = int(qo_indptr.numel()) - 1
    # Fast path: single check via device lengths (one sync) then no-copy.
    lens = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)
    if bool((lens % block_m == 0).all().item()):
        real = [int(x) for x in lens.tolist()]
        return q, o, qo_indptr, real

    nhead, qk, vdim = q.shape[1], q.shape[2], o.shape[2]
    qp, op, ind, real = [], [], [0], []
    for b in range(batch):
        s, e = int(qo_indptr[b]), int(qo_indptr[b + 1])
        slen = e - s
        real.append(slen)
        pad = (block_m - slen % block_m) % block_m
        qp.append(q[s:e])
        op.append(o[s:e])
        if pad:
            qp.append(torch.zeros(pad, nhead, qk, dtype=q.dtype, device=device))
            op.append(torch.zeros(pad, nhead, vdim, dtype=o.dtype, device=device))
        ind.append(ind[-1] + slen + pad)
    return (
        torch.cat(qp, 0),
        torch.cat(op, 0),
        torch.tensor(ind, dtype=torch.int32, device=device),
        real,
    )


def _build_q_tiles(qo_indptr, block_m: int):
    """Build per-CTA Q tile starts. Prefers a pure-device arange when possible."""
    device = qo_indptr.device
    batch = int(qo_indptr.numel()) - 1
    if batch == 1:
        s = int(qo_indptr[0].item())
        e = int(qo_indptr[1].item())
        assert (e - s) % block_m == 0
        if e == s:
            z = torch.zeros(1, dtype=torch.int32, device=device)
            return z, z, 0
        starts = torch.arange(s, e, block_m, dtype=torch.int32, device=device)
        batches = torch.zeros_like(starts)
        return starts, batches, int(starts.numel())

    starts, batches = [], []
    for b in range(batch):
        s, e = int(qo_indptr[b]), int(qo_indptr[b + 1])
        assert (e - s) % block_m == 0
        for q0 in range(s, e, block_m):
            starts.append(q0)
            batches.append(b)
    if not starts:
        z = torch.zeros(1, dtype=torch.int32, device=device)
        return z, z, 0
    return (
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(batches, dtype=torch.int32, device=device),
        len(starts),
    )


def _combine_kv_splits(partial_o, partial_ml, o_pad, tile_q_start, num_heads, block_m):
    """Online-softmax merge of split partials into ``o_pad`` (bf16).

    partial_o  : fp32 [ntiles, H, S, BM, V]
    partial_ml : fp32 [ntiles, H, S, BM, 2]  (m, l)

    Assumes ``tile_q_start`` enumerates ``o_pad`` in contiguous BM tiles
    (true for ``_build_q_tiles`` after ``_pad_sequences_to_block_m``).
    """
    m = partial_ml[..., 0]
    l = partial_ml[..., 1]
    m_max = m.amax(dim=2, keepdim=True)
    valid = m > (_NEG_INF * 0.5)
    alpha = torch.where(valid, torch.exp(m - m_max), torch.zeros_like(m))
    o_sum = (partial_o * alpha.unsqueeze(-1)).sum(dim=2)
    l_sum = (l * alpha).sum(dim=2)
    inv = torch.where(l_sum > 0, 1.0 / l_sum, torch.zeros_like(l_sum))
    merged = (o_sum * inv.unsqueeze(-1)).to(o_pad.dtype)
    o_pad.copy_(merged.permute(0, 2, 1, 3).reshape(-1, num_heads, merged.shape[-1]))


class MlaPrefillWorkspace:
    """Prepared buffers for an e2e-style device-only launch (no host rebuild)."""

    __slots__ = (
        "q_pad",
        "o_pad",
        "o_out",
        "kv_flat",
        "qo_pad",
        "kv_indptr",
        "kv_indices",
        "tile_q_start",
        "tile_batch",
        "partial_o",
        "partial_ml",
        "real_lens",
        "page_size",
        "num_heads",
        "ntiles",
        "num_kv_splits",
        "sm_scale",
        "block_m",
        "launch",
        "needs_unpad",
    )


def prepare_mla_prefill_workspace(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale=None,
    *,
    is_causal=False,
    block_m=DEFAULT_BLOCK_M,
    block_n=DEFAULT_BLOCK_N,
    num_warps=DEFAULT_NUM_WARPS,
    waves_per_eu=DEFAULT_WAVES_PER_EU,
    num_kv_splits=None,
    k_double_buffer=False,
    q_preload=False,
    stage_vt=False,
    vec_epilogue=True,
):
    """Build metadata + scratch once (serving-style). Not part of device bench."""
    if q.dtype != torch.bfloat16 or kv_buffer.dtype != torch.bfloat16:
        raise TypeError("bf16 required")
    total_q, num_heads, qk_dim = q.shape
    v_dim = o.shape[-1]
    if o.shape[:2] != (total_q, num_heads):
        raise ValueError("o/q shape mismatch")
    _validate_geometry(
        qk_dim,
        v_dim,
        block_m,
        block_n,
        num_warps,
        k_double_buffer=k_double_buffer,
        stage_vt=stage_vt,
    )

    if kv_buffer.ndim == 4:
        P, page_size, nk, kd = kv_buffer.shape
        if nk != 1 or kd != qk_dim:
            raise ValueError(f"bad kv {tuple(kv_buffer.shape)}")
        kv_flat = kv_buffer.reshape(P * page_size, qk_dim).contiguous()
    elif kv_buffer.ndim == 3:
        page_size = 1
        kv_flat = kv_buffer.reshape(-1, qk_dim).contiguous()
    elif kv_buffer.ndim == 2:
        page_size = 1
        kv_flat = kv_buffer.contiguous()
    else:
        raise ValueError(f"kv ndim {kv_buffer.ndim}")

    if sm_scale is None:
        sm_scale = qk_dim**-0.5

    q = q.contiguous()
    o = o.contiguous()
    qo_indptr = qo_indptr.contiguous()
    kv_indptr = kv_indptr.contiguous()
    kv_indices = kv_indices.contiguous()

    q_pad, o_pad, qo_pad, real = _pad_sequences_to_block_m(q, o, qo_indptr, block_m)
    tqs, tba, ntiles = _build_q_tiles(qo_pad, block_m)
    ws = MlaPrefillWorkspace()
    ws.o_out = o
    ws.real_lens = real
    ws.needs_unpad = q_pad.data_ptr() != q.data_ptr() or any(
        (block_m - sl % block_m) % block_m for sl in real
    )
    if ntiles == 0:
        ws.ntiles = 0
        return ws

    # Prefer host-visible max without per-call .item() in the hot path: compute
    # once here during prepare.
    max_kv = int((kv_indptr[1:] - kv_indptr[:-1]).max().item())
    if num_kv_splits is None:
        num_kv_splits = choose_num_kv_splits(
            num_q_tiles=ntiles,
            num_heads=num_heads,
            max_kv_len=max_kv,
            block_n=block_n,
        )
    num_kv_splits = max(1, int(num_kv_splits))

    ws.q_pad = q_pad
    ws.o_pad = o_pad
    ws.kv_flat = kv_flat
    ws.qo_pad = qo_pad
    ws.kv_indptr = kv_indptr
    ws.kv_indices = kv_indices
    ws.tile_q_start = tqs
    ws.tile_batch = tba
    ws.page_size = int(page_size)
    ws.num_heads = int(num_heads)
    ws.ntiles = int(ntiles)
    ws.num_kv_splits = num_kv_splits
    ws.sm_scale = float(sm_scale)
    ws.block_m = int(block_m)
    ws.launch = compile_mla_prefill_dense(
        qk_dim=qk_dim,
        v_dim=v_dim,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        is_causal=is_causal,
        waves_per_eu=waves_per_eu,
        k_double_buffer=k_double_buffer,
        q_preload=q_preload,
        stage_vt=stage_vt,
        vec_epilogue=vec_epilogue,
    )
    if num_kv_splits > 1:
        ws.partial_o = torch.empty(
            ntiles,
            num_heads,
            num_kv_splits,
            block_m,
            v_dim,
            dtype=torch.float32,
            device=q.device,
        )
        ws.partial_ml = torch.empty(
            ntiles,
            num_heads,
            num_kv_splits,
            block_m,
            2,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        ws.partial_o = torch.empty(1, dtype=torch.float32, device=q.device)
        ws.partial_ml = torch.empty(1, dtype=torch.float32, device=q.device)
    return ws


def run_mla_prefill_prepared(ws, stream=None):
    """Device-only e2e step: ``kn_0`` (+ GPU split combine). No host rebuild."""
    if ws.ntiles == 0:
        return ws.o_out
    if stream is None:
        stream = torch.cuda.current_stream()
    ws.launch(
        ws.q_pad.view(-1),
        ws.kv_flat.view(-1),
        ws.o_pad.view(-1),
        ws.partial_o.view(-1),
        ws.partial_ml.view(-1),
        ws.tile_q_start,
        ws.tile_batch,
        ws.qo_pad,
        ws.kv_indptr,
        ws.kv_indices,
        ws.page_size,
        ws.num_heads,
        ws.ntiles,
        ws.num_kv_splits,
        ws.sm_scale,
        fx.Stream(stream),
    )
    if ws.num_kv_splits > 1:
        _combine_kv_splits(
            ws.partial_o,
            ws.partial_ml,
            ws.o_pad,
            ws.tile_q_start,
            ws.num_heads,
            ws.block_m,
        )
    if ws.needs_unpad:
        dst = src = 0
        o = ws.o_out
        for slen in ws.real_lens:
            o[dst : dst + slen].copy_(ws.o_pad[src : src + slen])
            dst += slen
            src += slen + (ws.block_m - slen % ws.block_m) % ws.block_m
    elif ws.o_pad.data_ptr() != ws.o_out.data_ptr():
        ws.o_out.copy_(ws.o_pad[: ws.o_out.shape[0]])
    return ws.o_out


def flydsl_mla_prefill_dense_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale=None,
    *,
    is_causal=False,
    block_m=DEFAULT_BLOCK_M,
    block_n=DEFAULT_BLOCK_N,
    num_warps=DEFAULT_NUM_WARPS,
    waves_per_eu=DEFAULT_WAVES_PER_EU,
    num_kv_splits=None,
    stream=None,
):
    """Convenience wrapper: prepare + run (includes host setup — not e2e-fair)."""
    ws = prepare_mla_prefill_workspace(
        q,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        is_causal=is_causal,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        num_kv_splits=num_kv_splits,
    )
    return run_mla_prefill_prepared(ws, stream=stream)
