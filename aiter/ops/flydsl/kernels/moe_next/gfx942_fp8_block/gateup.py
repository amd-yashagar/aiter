# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 blockscale (per_1x128) MoE Gate/Up for prefill.

A is FP8 e4m3fnuz with one FP32 scale per row and 128 K values, gathered by
sorted token id; W1 is aiter's shuffle_weight(w1, (16, 16)) layout with one FP32
scale per 128x128 block ([E, 2I/128, K/128]). TILE_K is 128, so every K tile is
one scale group: its MFMAs accumulate into a fresh partial that is folded into
the FP32 accumulator with sa[row, k] * sb[n_block, k].
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import common as fxh
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import (
    _f32_to_bf16,
    make_gemm_helpers,
)

SCALE_BLOCK = 128
TILE_K = 128


B_STEP = (1, 2)
VMEM_EVERY = 4


def compile_gateup(
    *, N, K, TOPK, BLOCK_M, BLOCK_N, activation="silu", OUT_FP8=False, A_BF16=False
):
    """N = 2 * inter_dim (gate rows then up rows), K = model_dim.

    OUT_FP8 writes the intermediate as aiter's per_1x128 quant of the bf16 result
    would ([M * TOPK, I] fp8 plus [M * TOPK, I / 128] fp32 scales), computed in the
    epilogue; it needs BLOCK_N = 256 so a workgroup owns one 128-column group.
    A_BF16 reads bf16 hidden states and applies aiter's per_1x128 activation quant
    to each K tile while loading it (the scale goes to LDS for the fold), so no
    separate quant kernel and no activation-scale tensor are needed.
    """
    assert activation in ("silu", "gelu")
    # Split each sort block (the caller's BLOCK_M rows, one expert) into SUB
    # 64-row tiles: half the accumulators and A LDS, more workgroups, and
    # padding-only sub-tiles are skipped by the valid-id check.
    SORT_BM = BLOCK_M
    BLOCK_M = min(BLOCK_M, 64)
    SUB = SORT_BM // BLOCK_M
    assert BLOCK_N in (128, 256), "each wave owns BLOCK_N // 8 gate and up channels"
    assert 32 <= BLOCK_M <= 256 and BLOCK_M % 32 == 0
    assert K % (2 * TILE_K) == 0, "the 2-stage pipeline needs an even K-tile count"
    assert (N // 2) % (BLOCK_N // 2) == 0
    contiguous_n = BLOCK_N // 2
    assert not OUT_FP8 or contiguous_n == SCALE_BLOCK
    IB = (N // 2) // SCALE_BLOCK
    assert SCALE_BLOCK % contiguous_n == 0 or contiguous_n % SCALE_BLOCK == 0
    assert contiguous_n <= SCALE_BLOCK, "one weight scale per gate/up sub-tile"
    KB = K // SCALE_BLOCK
    NB = N // SCALE_BLOCK
    weight_dtype = fx.Float8E4M3FNUZ
    a_lds_size = BLOCK_M * TILE_K
    assert 2 * a_lds_size <= 64 * 1024
    assert BLOCK_M * contiguous_n * 2 <= 2 * a_lds_size, "CShuffle reuses A LDS"

    @fx.struct
    class GemmBuffers:
        a_ping: fx.Array[weight_dtype, a_lds_size, 16]
        a_pong: fx.Array[weight_dtype, a_lds_size, 16]
        sa_ping: fx.Array[fx.Float32, BLOCK_M, 16]
        sa_pong: fx.Array[fx.Float32, BLOCK_M, 16]

    if OUT_FP8:
        # The epilogue quantizes from the FP32 activation output, as aiter's ASM
        # 1-stage path does (no bf16 rounding before the per_1x128 quant).
        assert BLOCK_M * contiguous_n * 4 <= 64 * 1024

        @fx.union
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]
            gemm: GemmBuffers
            cshuf: fx.Array[fx.Float32, BLOCK_M * contiguous_n, 16]

    else:

        @fx.union
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]
            gemm: GemmBuffers

    TensorWithIndex, _read_sorted_index, _ = make_gemm_helpers(
        K, weight_dtype, BLOCK_M, TOPK
    )

    # Same operations and order as aiter's silu_kernel / gelu_kernel followed by
    # act_and_mul: x * rcp(1 + exp(-x)) and x * 0.5 * (1 + erf(x / sqrt(2))).
    def _act(g, u):
        if const_expr(activation == "gelu"):
            return g * 0.5 * (1.0 + fx.erf(g * 0.7071067811865476)) * u
        return (g * rocdl.rcp(T.f32, _raw(1.0 + fx.exp(-g)))) * u

    def _gateup_pair_bf16(gate_frag, up_frag):
        out_bf16 = fx.make_fragment_like(gate_frag, dtype=fx.BFloat16)
        m_reps = fx.size(fx.get_shape(gate_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(gate_frag)[2]).to_py_value()
        for m in range_constexpr(m_reps):
            for n in range_constexpr(n_reps):
                gate = gate_frag[None, m, n].load()
                up = up_frag[None, m, n].load()
                acc = [_act(gate[j], up[j]) for j in range_constexpr(gate.numel)]
                acc = Vec.from_elements(acc, fx.Float32)
                out_bf16[None, m, n].store(_f32_to_bf16(acc))
        return out_bf16

    def _gateup_pair_f32(gate_frag, up_frag):
        out = fx.make_fragment_like(gate_frag, dtype=fx.Float32)
        m_reps = fx.size(fx.get_shape(gate_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(gate_frag)[2]).to_py_value()
        for m in range_constexpr(m_reps):
            for n in range_constexpr(n_reps):
                gate = gate_frag[None, m, n].load()
                up = up_frag[None, m, n].load()
                acc = [_act(gate[j], up[j]) for j in range_constexpr(gate.numel)]
                out[None, m, n].store(Vec.from_elements(acc, fx.Float32))
        return out

    def _make_gateup_weight_view(p_weight, expert_id):
        group_layout = fx.make_layout(
            ((contiguous_n, 2, N // (contiguous_n * 2)), K),
            ((1, N // 2, contiguous_n), N),
        )
        element_num = 16
        return fx.make_view(
            p_weight + fx.Int64(expert_id * N * K),
            fx.composition(
                fx.make_layout(
                    ((16, N // 16), (element_num, K // element_num)),
                    ((element_num, 16 * K), (1, 16 * element_num)),
                ),
                group_layout,
            ),
        )

    def _make_tiled_mma():
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, weight_dtype))
        k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((4, 1, 1), (1, 0, 0)),
            fx.make_tile(None, None, k_perm),
        )
        return mma_atom, tiled_mma

    @flyc.jit
    def gemm_blockscale(
        blk_n,
        expert_id,
        arg_p_input: fx.Tensor,
        arg_p_weight: fx.Tensor,
        p_w_scale,
        a_scale_tensor,
        row_toks,
        lds,
        b_bits,
    ):
        tid = gpu.thread_idx.x
        lane = tid % 64
        TILE_M = BLOCK_M

        a_tensor = fx.rocdl.make_buffer_tensor(arg_p_input, max_size=False)
        b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)
        mma_atom, tiled_mma = _make_tiled_mma()

        a_size_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((TILE_M, K), (K, 1))),
            max_size=False,
        )
        a_tile = fx.flat_divide(a_size_buf, fx.make_tile(TILE_M, TILE_K))[
            None, None, 0, None
        ]
        buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_dtype)
        _val_per_thr = 16
        _thrs_k = TILE_K // _val_per_thr
        _thrs_m = 256 // _thrs_k
        g2r_tv_layout = fx.make_layout(
            ((_thrs_k, _thrs_m), (1, _val_per_thr)),
            ((_thrs_m * _val_per_thr, 1), (1, _thrs_m)),
        )
        a_mem_cp_g2r = fx.make_tiled_copy(
            buf_cp_atom_r, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
        )
        _m_per_wave = _thrs_m // 4
        tiled_copy_sortid_a = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
            fx.make_layout(((_thrs_k, _m_per_wave, 4), 1), ((0, 1, _m_per_wave), 0)),
            fx.make_tile(_thrs_m),
        )
        a_index_frag = _read_sorted_index(
            tiled_copy_sortid_a, tid, lds.sorted_lds, index_size=TILE_M
        )
        a_idx = TensorWithIndex(
            a_tensor, TILE_M, TILE_K, a_index_frag, a_mem_cp_g2r, tid
        )
        a_mem_thr = a_mem_cp_g2r.get_slice(tid).partition_S(a_tile)
        a_cp_frag = fx.make_fragment_like(a_mem_thr[None, None, None, 0])
        if const_expr(A_BF16):
            a_bf_view = fx.make_view(
                fx.recast_iter(fx.BFloat16, fx.get_iter(arg_p_input)),
                fx.get_layout(arg_p_input),
            )
            a_bf_tensor = fx.rocdl.make_buffer_tensor(a_bf_view, max_size=False)
            bf_cp_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
            a_bf_g2r = fx.make_tiled_copy(
                bf_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
            )
            a_bf_idx = TensorWithIndex(
                a_bf_tensor, TILE_M, TILE_K, a_index_frag, a_bf_g2r, tid
            )
            a_bf_tile = fx.flat_divide(
                fx.rocdl.make_buffer_tensor(
                    fx.make_view(
                        fx.get_iter(a_bf_view), fx.make_layout((TILE_M, K), (K, 1))
                    ),
                    max_size=False,
                ),
                fx.make_tile(TILE_M, TILE_K),
            )[None, None, 0, None]
            a_bf_frag = fx.make_fragment_like(
                a_bf_g2r.get_slice(tid).partition_S(a_bf_tile)[None, None, None, 0]
            )

        # All indices must be captured before a_ping reuses sorted_lds.
        gpu.barrier()

        swz = fx.SwizzleType.get(3, 4, 3)
        a_lds_layout = fx.make_composed_layout(
            fx.static(swz), fx.make_ordered_layout((TILE_M, TILE_K), order=(1, 0))
        )
        a_ping = fx.make_view(lds.gemm.a_ping.ptr, a_lds_layout)
        a_pong = fx.make_view(lds.gemm.a_pong.ptr, a_lds_layout)
        sa_lds = [
            fx.make_view(lds.gemm.sa_ping.ptr, fx.make_layout(BLOCK_M, 1)),
            fx.make_view(lds.gemm.sa_pong.ptr, fx.make_layout(BLOCK_M, 1)),
        ]

        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), weight_dtype)
        a_r2s = fx.make_tiled_copy(
            uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
        )
        a_lds_w = [
            a_r2s.get_slice(tid).partition_D(a_ping),
            a_r2s.get_slice(tid).partition_D(a_pong),
        ]
        a_cp_frag_retile = a_r2s.get_slice(tid).retile(a_cp_frag)
        b_copy_lds = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid)
        a_lds_r = [b_copy_lds.partition_S(a_ping), b_copy_lds.partition_S(a_pong)]
        a_frag = tiled_mma.make_fragment_B(a_ping)
        a_frag_retile = b_copy_lds.retile(a_frag)

        bl_tile = fx.flat_divide(b_tensor, fx.make_tile(contiguous_n, TILE_K))[
            None, None, blk_n * 2 + 0, None
        ]
        br_tile = fx.flat_divide(b_tensor, fx.make_tile(contiguous_n, TILE_K))[
            None, None, blk_n * 2 + 1, None
        ]
        # Weight loads take this block's cache policy (b_bits is a Python int:
        # 0 cached, 2 non-temporal); A keeps the default.
        buf_cp_atom_b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_bits), weight_dtype)
        b_g2r = fx.make_tiled_copy_A(buf_cp_atom_b, tiled_mma).get_slice(tid)
        bl_g2r = b_g2r.partition_S(bl_tile)
        br_g2r = b_g2r.partition_S(br_tile)
        bl_frag_st = [
            tiled_mma.make_fragment_A(bl_tile[None, None, 0]),
            tiled_mma.make_fragment_A(bl_tile[None, None, 0]),
        ]
        br_frag_st = [
            tiled_mma.make_fragment_A(br_tile[None, None, 0]),
            tiled_mma.make_fragment_A(br_tile[None, None, 0]),
        ]
        bl_ret_st = [b_g2r.retile(bl_frag_st[0]), b_g2r.retile(bl_frag_st[1])]
        br_ret_st = [b_g2r.retile(br_frag_st[0]), b_g2r.retile(br_frag_st[1])]

        c_fake_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_layout((contiguous_n, TILE_M), (TILE_M, 1)),
            ),
            max_size=False,
        )
        c_fake = fx.flat_divide(c_fake_buf, fx.make_tile(contiguous_n, TILE_M))[
            None, None, 0, 0
        ]
        c_gate = tiled_mma.make_fragment_C(c_fake)
        c_up = tiled_mma.make_fragment_C(c_fake)
        p_gate = tiled_mma.make_fragment_C(c_fake)
        p_up = tiled_mma.make_fragment_C(c_fake)
        c_gate.fill(0)
        c_up.fill(0)

        num_tiles = K // TILE_K
        k_per_mma = 32
        _m_reps = fx.size(fx.get_shape(c_gate)[1]).to_py_value()
        _n_reps = fx.size(fx.get_shape(c_gate)[2]).to_py_value()
        k_iters = TILE_K // (2 * k_per_mma)

        # Weight scales: wave-uniform, one scalar load per K tile and gate/up half.
        ws_view = fx.make_view(
            fxh._as_ptr(p_w_scale) + expert_id * (NB * KB), fx.make_layout(NB * KB, 1)
        )
        ws_rsrc = fx.rocdl.get_buffer_rsrc(
            fx.get_iter(fx.rocdl.make_buffer_tensor(ws_view, max_size=False))
        )
        gate_off = (blk_n * contiguous_n) // SCALE_BLOCK * KB
        up_off = (N // 2 + blk_n * contiguous_n) // SCALE_BLOCK * KB
        # Activation scales: lane l loads the scale of tile row l (+ 64 j); the MFMA
        # fragment's token column n * 16 + lane % 16 is then read with ds_bpermute.
        n_row_loads = len(row_toks)
        bperm_addr = [
            (((n * 16) % 64) + lane % 16) * 4 for n in range_constexpr(_n_reps)
        ]
        mfma_per_gemm = _m_reps * _n_reps * (TILE_K // k_per_mma)
        mem_a_cnt = a_cp_frag.load().numel * weight_dtype.width // 8 // 16
        mem_b_cnt = bl_frag_st[0].load().numel * weight_dtype.width // 8 // 16
        lds_a_cnt = a_frag.load().numel * weight_dtype.width // 8 // 16

        def hot_loop_scheduler():
            mfma_cnt = 2 * mfma_per_gemm
            n_vmem = mem_a_cnt + 2 * mem_b_cnt
            n_dswr = mem_a_cnt
            n_dsrd = lds_a_cnt
            used = 0
            rocdl.sched_dsrd(2)
            for _ in range_constexpr(n_vmem):
                rocdl.sched_dsrd(1)
                rocdl.sched_vmem(1)
                rocdl.sched_mfma(2)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(2)
                used += 4
            for _ in range_constexpr(n_dsrd - 2 * n_vmem - 2):
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                used += 1
            if const_expr(mfma_cnt - n_dswr * 2 - used > 0):
                rocdl.sched_mfma(mfma_cnt - n_dswr * 2 - used)
            for _ in range_constexpr(n_dswr):
                rocdl.sched_dswr(1)
                rocdl.sched_mfma(2)
                used += 2
            if const_expr(mfma_cnt - used > 0):
                rocdl.sched_mfma(mfma_cnt - used)

        a_rows_per_rep = _thrs_m
        a_my_row = tid // _thrs_k

        def load_a(k):
            if const_expr(A_BF16):
                a_bf_idx.copy(bf_cp_atom, k, a_bf_frag)
            else:
                a_idx.copy(buf_cp_atom_r, k, a_cp_frag)

        def quant_a(write_i):
            """aiter's per_1x128 quant of the bf16 tile rows this thread holds (16
            values; _thrs_k consecutive lanes share a row): amax floor 1e-10, scale =
            amax * fp32(1/240), q = med3(x * (1 / scale), -240, 240) -> fp8."""
            rep_m = fx.size(fx.get_shape(a_bf_frag)[1]).to_py_value()
            for m in range_constexpr(rep_m):
                v = Vec(a_bf_frag[None, m, 0].load())
                xs = [v[j].to(fx.Float32) for j in range_constexpr(_val_per_thr)]
                amax = fx.Float32(1e-10)
                for x in xs:
                    amax = amax.maximumf(fx.absf(x))
                for sh in range_constexpr(_thrs_k.bit_length() - 1):
                    amax = amax.maximumf(
                        amax.shuffle_xor(fx.Int32(1 << sh), fx.Int32(64))
                    )
                scale = amax * fx.Float32(1.0 / 240.0)
                inv = fx.Float32(1.0) / scale
                words = []
                for w in range_constexpr(_val_per_thr // 4):
                    q = [
                        (xs[w * 4 + j] * inv)
                        .maximumf(fx.Float32(-240.0))
                        .minimumf(fx.Float32(240.0))
                        for j in range_constexpr(4)
                    ]
                    pk = rocdl.cvt_pk_fp8_f32(
                        T.i32, _raw(q[0]), _raw(q[1]), _raw(fx.Int32(0)), False
                    )
                    pk = rocdl.cvt_pk_fp8_f32(T.i32, _raw(q[2]), _raw(q[3]), pk, True)
                    words.append(fx.Int32(pk))
                a_cp_frag[None, m, 0].store(
                    Vec.from_elements(words, fx.Int32).bitcast(weight_dtype)
                )
                # Every lane of the row writes the same value.
                sa_lds[write_i][a_my_row + m * a_rows_per_rep] = scale

        def _sload(off):
            v = buffer_ops.buffer_load(
                ws_rsrc,
                rocdl.readfirstlane(T.i32, _raw(fx.Int32(off))),
                vec_width=1,
                is_scalar=True,
            )
            return fx.Int32(v).bitcast(fx.Float32)

        def load_scales(k_cur):
            sb_gate = _sload(gate_off + k_cur)
            sb_up = _sload(up_off + k_cur)
            if const_expr(A_BF16):
                return [sb_gate, sb_up]
            rows = [
                a_scale_tensor[row_toks[j] + k_cur]
                for j in range_constexpr(n_row_loads)
            ]
            return [sb_gate, sb_up] + rows

        def fold_prep(scales, read_i):
            sb_gate, sb_up = fx.Float32(scales[0]), fx.Float32(scales[1])
            prods = []
            for n in range_constexpr(_n_reps):
                if const_expr(A_BF16):
                    sa = fx.Float32(sa_lds[read_i][n * 16 + lane % 16])
                else:
                    src = fx.Float32(scales[2 + (n * 16) // 64]).bitcast(fx.Int32)
                    sa = fx.Int32(
                        rocdl.ds_bpermute(
                            T.i32, _raw(fx.Int32(bperm_addr[n])), _raw(src)
                        )
                    ).bitcast(fx.Float32)
                prods.append((sa * sb_gate, sa * sb_up))
            return prods

        def fold_one(prods, n):
            for m in range_constexpr(_m_reps):
                for c, pp, sc in (
                    (c_gate, p_gate, prods[n][0]),
                    (c_up, p_up, prods[n][1]),
                ):
                    pv = Vec(pp[None, m, n].load())
                    cv = Vec(c[None, m, n].load())
                    vals = [fx.fma(pv[j], sc, cv[j]) for j in range_constexpr(4)]
                    c[None, m, n].store(Vec.from_elements(vals, fx.Float32))

        # Stage = 4 steps (one n group each), closed by sched_barrier(0) with
        # {MFMA 1, VALU 1} groups; fold of group n-1 lags under group n, and group
        # 3's partial is carried into the next stage's step 0.
        MFMA_M, VALU_M, VMEM_M, DSRD_M, DSWR_M = 0x008, 0x002, 0x020, 0x100, 0x200
        mfma_per_step = 4 * k_iters * _m_reps
        n_b_vmem = mem_b_cnt
        n_sc_vmem = 0 if const_expr(A_BF16) else n_row_loads

        def step_plan(n_vmem, n_dsrd, n_dswr, n_valu):
            if n_dsrd > 0:
                rocdl.sched_group_barrier(DSRD_M, n_dsrd, 0)
            vm = n_vmem
            dw = n_dswr
            va = n_valu
            for i in range_constexpr(mfma_per_step):
                rocdl.sched_group_barrier(MFMA_M, 1, 0)
                if va > 0:
                    rocdl.sched_group_barrier(VALU_M, 1, 0)
                    va -= 1
                if vm > 0 and (i % VMEM_EVERY == 0):
                    rocdl.sched_group_barrier(VMEM_M, 1, 0)
                    vm -= 1
                if dw > 0 and i >= mfma_per_step - 2 * n_dswr and (i % 2 == 0):
                    rocdl.sched_group_barrier(DSWR_M, 1, 0)
                    dw -= 1
            rocdl.sched_barrier(0)

        def mfma_group(read_i, n):
            for ki in range_constexpr(k_iters):
                for k in range_constexpr(2):
                    for m in range_constexpr(_m_reps):
                        fx.mma_atom_call(
                            mma_atom,
                            p_gate[None, m, n],
                            bl_frag_st[read_i][None, m, (k, ki)],
                            a_frag[None, n, (k, ki)],
                            p_gate[None, m, n],
                        )
                        fx.mma_atom_call(
                            mma_atom,
                            p_up[None, m, n],
                            br_frag_st[read_i][None, m, (k, ki)],
                            a_frag[None, n, (k, ki)],
                            p_up[None, m, n],
                        )

        def fold_carry(carry):
            nl = _n_reps - 1
            for m in range_constexpr(_m_reps):
                for c, pv, sc in (
                    (c_gate, carry[m], carry[2 * _m_reps]),
                    (c_up, carry[_m_reps + m], carry[2 * _m_reps + 1]),
                ):
                    pv = Vec(pv)
                    cv = Vec(c[None, m, nl].load())
                    vals = [
                        fx.fma(pv[j], fx.Float32(sc), cv[j]) for j in range_constexpr(4)
                    ]
                    c[None, m, nl].store(Vec.from_elements(vals, fx.Float32))

        def take_carry(prods):
            nl = _n_reps - 1
            return (
                [p_gate[None, m, nl].load() for m in range_constexpr(_m_reps)]
                + [p_up[None, m, nl].load() for m in range_constexpr(_m_reps)]
                + [prods[nl][0], prods[nl][1]]
            )

        def pipeline_stage(read_i, scales, k_next, do_prefetch, carry):
            write_i = read_i ^ 1
            next_scales = None
            if const_expr(do_prefetch):
                load_a(k_next)
                if const_expr(B_STEP[0] == 0):
                    fx.copy(
                        buf_cp_atom_b,
                        bl_g2r[None, None, None, k_next],
                        bl_ret_st[write_i],
                    )
                if const_expr(B_STEP[1] == 0):
                    fx.copy(
                        buf_cp_atom_b,
                        br_g2r[None, None, None, k_next],
                        br_ret_st[write_i],
                    )
                next_scales = load_scales(k_next)
            prods = fold_prep(scales, read_i)
            p_gate.fill(0)
            p_up.fill(0)
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_cp_atom,
                    a_lds_r[read_i][None, None, ki],
                    a_frag_retile[None, None, ki],
                )
            mfma_group(read_i, 0)
            fold_carry(carry)
            if const_expr(do_prefetch):
                nb0 = (n_b_vmem if B_STEP[0] == 0 else 0) + (
                    n_b_vmem if B_STEP[1] == 0 else 0
                )
                step_plan(mem_a_cnt + nb0 + n_sc_vmem, lds_a_cnt, 0, 2 * 4 * _m_reps)
            else:
                rocdl.sched_barrier(0)
            for n in range_constexpr(1, _n_reps):
                nb = 0
                if const_expr(do_prefetch):
                    if const_expr(B_STEP[0] == n):
                        fx.copy(
                            buf_cp_atom_b,
                            bl_g2r[None, None, None, k_next],
                            bl_ret_st[write_i],
                        )
                        nb += n_b_vmem
                    if const_expr(B_STEP[1] == n):
                        fx.copy(
                            buf_cp_atom_b,
                            br_g2r[None, None, None, k_next],
                            br_ret_st[write_i],
                        )
                        nb += n_b_vmem
                mfma_group(read_i, n)
                fold_one(prods, n - 1)
                ndw = 0
                if const_expr(do_prefetch and n == _n_reps - 1):
                    if const_expr(A_BF16):
                        quant_a(write_i)
                    fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w[write_i])
                    ndw = mem_a_cnt
                if const_expr(do_prefetch):
                    step_plan(nb, 0, ndw, 2 * 4 * _m_reps)
                else:
                    rocdl.sched_barrier(0)
            if const_expr(do_prefetch):
                carry_out = take_carry(prods)
            else:
                fold_one(prods, _n_reps - 1)
                carry_out = None
            rocdl.sched_barrier(0)
            gpu.barrier()
            return next_scales, carry_out

        load_a(fx.Int32(0))
        fx.copy(buf_cp_atom_b, bl_g2r[None, None, None, fx.Int32(0)], bl_ret_st[0])
        fx.copy(buf_cp_atom_b, br_g2r[None, None, None, fx.Int32(0)], br_ret_st[0])
        scales0 = load_scales(fx.Int32(0))
        rocdl.s_waitcnt(vmcnt=0)
        if const_expr(A_BF16):
            quant_a(0)
        fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w[0])
        p_gate.fill(0)
        p_up.fill(0)
        carry0 = take_carry([(fx.Float32(0.0), fx.Float32(0.0))] * _n_reps)
        gpu.barrier()

        n_sc = 2 if const_expr(A_BF16) else 2 + n_row_loads
        n_cy = 2 * _m_reps + 2
        acc_init = [c_gate.load(), c_up.load()] + scales0 + carry0
        results = acc_init
        for iv, state in range(0, num_tiles // 2 - 1, 1, init=acc_init):
            c_gate.store(state[0])
            c_up.store(state[1])
            kb = fx.Int32(iv * 2)
            cy = [state[2 + n_sc + i] for i in range_constexpr(n_cy)]
            s1, cy1 = pipeline_stage(
                0, [state[2 + i] for i in range_constexpr(n_sc)], kb + 1, True, cy
            )
            s2, cy2 = pipeline_stage(1, s1, kb + 2, True, cy1)
            results = yield [c_gate.load(), c_up.load()] + s2 + cy2
        c_gate.store(results[0])
        c_up.store(results[1])
        kb = fx.Int32(num_tiles - 2)
        cy = [results[2 + n_sc + i] for i in range_constexpr(n_cy)]
        s1, cy1 = pipeline_stage(
            0, [results[2 + i] for i in range_constexpr(n_sc)], kb + 1, True, cy
        )
        pipeline_stage(1, s1, fx.Int32(0), False, cy1)
        return c_gate, c_up

    INV_FP8_MAX = 1.0 / 240.0
    COLS = SCALE_BLOCK // (256 // BLOCK_M)

    def _quant_store(lds_c, row, part, sid, blk_n, p_output, p_out_scale, M):
        """One row of 128 FP32 values is read by 256 / BLOCK_M threads; same math
        and rounding as aiter's per_1x128 quant on gfx942: amax (floor 1e-10),
        scale = amax * fp32(1/240), q = med3(x * (1 / scale), -240, 240) -> fp8."""
        TPR = 256 // BLOCK_M
        vals = []
        for c in range_constexpr(COLS // 4):
            col = part * COLS + c * 4
            v = fxh.atom_tensor(lds_c, (row, col), 128).load()
            for j in range_constexpr(4):
                vals.append(fx.Float32(Vec(v)[j]))
        amax = fx.Float32(1e-10)
        for x in vals:
            amax = amax.maximumf(fx.absf(x))
        for sh in range_constexpr(TPR.bit_length() - 1):
            amax = amax.maximumf(amax.shuffle_xor(fx.Int32(1 << sh), fx.Int32(64)))
        scale = amax * fx.Float32(INV_FP8_MAX)
        inv = fx.Float32(1.0) / scale
        hi = fx.Float32(240.0)
        lo = fx.Float32(-240.0)
        words = []
        for w in range_constexpr(COLS // 4):
            q = []
            for j in range_constexpr(4):
                y = vals[w * 4 + j] * inv
                q.append(y.maximumf(lo).minimumf(hi))
            pk = rocdl.cvt_pk_fp8_f32(
                T.i32, _raw(q[0]), _raw(q[1]), _raw(fx.Int32(0)), False
            )
            pk = rocdl.cvt_pk_fp8_f32(T.i32, _raw(q[2]), _raw(q[3]), pk, True)
            words.append(fx.Int32(pk))
        tok = fx.Int32(sid & 0xFFFFFF)
        slot = fx.Int32(sid >> 24)
        valid = (tok < M) & (slot < TOPK)
        out_row = tok * TOPK + slot
        out_rsrc = fx.rocdl.get_buffer_rsrc(
            fx.get_iter(
                fx.rocdl.make_buffer_tensor(
                    fx.make_view(
                        fx.recast_iter(fx.Uint8, fxh._as_ptr(p_output)),
                        fx.make_layout(M * TOPK * (N // 2), 1),
                    ),
                    max_size=False,
                )
            )
        )
        scale_rsrc = fx.rocdl.get_buffer_rsrc(
            fx.get_iter(
                fx.rocdl.make_buffer_tensor(
                    fx.make_view(
                        fx.recast_iter(fx.Float32, fxh._as_ptr(p_out_scale)),
                        fx.make_layout(M * TOPK * IB, 1),
                    ),
                    max_size=False,
                )
            )
        )
        base = out_row * (N // 2) + blk_n * SCALE_BLOCK + part * COLS
        for c in range_constexpr(COLS // 16):
            buffer_ops.buffer_store(
                Vec.from_elements(words[c * 4 : c * 4 + 4], fx.Int32),
                out_rsrc,
                base + c * 16,
                mask=valid,
                offset_is_bytes=True,
            )
        buffer_ops.buffer_store(
            scale, scale_rsrc, out_row * IB + blk_n, mask=valid & (part == 0)
        )

    @flyc.jit
    def _gemm_and_store(
        b_bits,
        tid,
        blk_n,
        expert_id,
        arg_p_input,
        arg_p_weight,
        p_w_scale,
        a_scale_tensor,
        row_toks,
        lds,
        q_row,
        q_part,
        q_sid,
        p_output,
        p_out_scale,
        M,
        c_out,
        c_rw_copy,
    ):
        c_gate_frag, c_up_frag = gemm_blockscale(
            blk_n,
            expert_id,
            arg_p_input,
            arg_p_weight,
            p_w_scale,
            a_scale_tensor,
            row_toks,
            lds,
            b_bits,
        )

        _, _tiled_mma = _make_tiled_mma()
        if const_expr(OUT_FP8):
            c_act = _gateup_pair_f32(c_gate_frag, c_up_frag)
            cs_ptr = lds.cshuf.peek().ptr
            swz32 = fx.static(fx.SwizzleType.get(3, 2, 5))
            lds_c32_store = fx.make_view(
                cs_ptr,
                fx.make_composed_layout(
                    swz32,
                    fx.make_ordered_layout((contiguous_n, BLOCK_M), order=(0, 1)),
                ),
            )
            lds_c32 = fx.make_view(
                cs_ptr,
                fx.make_composed_layout(
                    swz32,
                    fx.make_ordered_layout((BLOCK_M, contiguous_n), order=(1, 0)),
                ),
            )
            gpu.barrier()
            c32_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
            store_c32 = fx.make_tiled_copy_C(c32_atom, _tiled_mma).get_slice(tid)
            fx.copy(
                c32_atom,
                store_c32.retile(c_act),
                store_c32.partition_D(lds_c32_store),
            )
            gpu.barrier()
            _quant_store(lds_c32, q_row, q_part, q_sid, blk_n, p_output, p_out_scale, M)
        else:
            c_out_bf16 = _gateup_pair_bf16(c_gate_frag, c_up_frag)
            cshuf_atom_w = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
            cshuf_atom_r = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            cshuf_ptr = fx.recast_iter(fx.BFloat16, lds.gemm.a_ping.ptr)
            swz_c = fx.SwizzleType.get(3, 3, 3)
            lds_c_store = fx.make_view(
                cshuf_ptr,
                fx.make_composed_layout(
                    fx.static(swz_c),
                    fx.make_ordered_layout((contiguous_n, BLOCK_M), order=(0, 1)),
                ),
            )
            lds_c = fx.make_view(
                cshuf_ptr,
                fx.make_composed_layout(
                    fx.static(swz_c),
                    fx.make_ordered_layout((BLOCK_M, contiguous_n), order=(1, 0)),
                ),
            )
            gpu.barrier()
            store_c = fx.make_tiled_copy_C(cshuf_atom_w, _tiled_mma).get_slice(tid)
            fx.copy(
                cshuf_atom_w,
                store_c.retile(c_out_bf16),
                store_c.partition_D(lds_c_store),
            )
            gpu.barrier()
            rd = fx.make_fragment_like(c_rw_copy.get_slice(tid).partition_S(lds_c))
            fx.copy(cshuf_atom_r, c_rw_copy.get_slice(tid).partition_S(lds_c), rd)
            buf_atom_w128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
            c_out.copy(buf_atom_w128, blk_n, rd)

    @flyc.kernel
    def gateup_blockscale(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        p_out_scale: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        # XCD-aware remap: blocks are dealt round-robin to the 8 XCDs; give each
        # XCD a contiguous chunk so same-expert tiles share one L2.
        gx = N // BLOCK_N
        gy = fx.Int32(gpu.grid_dim.y)
        bid = fx.Int32(gpu.block_idx.y) * gx + fx.Int32(gpu.block_idx.x)
        total = gy * gx
        RR = 32
        full = (total // (8 * RR)) * (8 * RR)
        j = bid // 8
        remap = (j // RR) * (8 * RR) + (bid % 8) * RR + j % RR
        new_id = fx.arith.select(bid < full, remap, bid)
        blk_n = new_id % gx
        e_idx = new_id // gx

        in_ptr = fx.recast_iter(weight_dtype, fxh._as_ptr(p_input))
        arg_p_input = fx.make_view(in_ptr, fx.make_layout((M, K), (K, 1)))
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        p_weight = fx.recast_iter(weight_dtype, fxh._as_ptr(p_weight))
        first_sid = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids))[e_idx * BLOCK_M]
        live = (e_idx * BLOCK_M < max_valid_id) & ((first_sid & 0xFFFFFF) < M)
        if live:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.gemm = lds.gemm.peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids) + e_idx * BLOCK_M),
                fx.make_layout(BLOCK_M, 1),
            )
            expert_id = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_expert_ids))[
                e_idx // SUB
            ]
            arg_p_weight = _make_gateup_weight_view(p_weight, expert_id)

            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            if tid < BLOCK_M:
                lds_view = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BLOCK_M, 1))
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            arg_p_output = fx.make_view(
                fxh._as_ptr(p_output),
                fx.make_layout((M, TOPK, N // 2), (TOPK * N // 2, N // 2, 1)),
            )
            out_tensor = fx.rocdl.make_buffer_tensor(
                arg_p_output,
                max_size=False,
                num_records_bytes=fx.Int64(M) * (TOPK * N // 2 * 2),
            )
            buf_atom_w128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
            c_rw_copy = fx.make_tiled_copy(
                buf_atom_w128,
                fx.make_layout(((4, 16, 2, 2), 8), ((256, 1, 16, 1024), 32)),
                fx.make_tile(32, 64),
            )
            c_index_copy = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((4, 16, 2, 2), 1), ((0, 1, 16, 0), 0)),
                fx.make_tile(32),
            )
            c_out_index_frag = _read_sorted_index(c_index_copy, tid, lds.sorted_lds)
            c_out = TensorWithIndex(
                out_tensor,
                BLOCK_M,
                contiguous_n,
                c_out_index_frag,
                c_rw_copy,
                tid,
                is_read_from_mem=False,
                TOPK=TOPK,
            )

            # Row token of tile row (lane % BLOCK_M) + 64 j, read before the GEMM
            # reuses sorted_lds; times KB, the activation-scale row offset.
            sorted_view = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(256, 1))
            lane = tid % 64
            row_toks = [
                (sorted_view[(lane + 64 * j) % BLOCK_M] & 0xFFFFFF) * KB
                for j in range_constexpr(max(1, BLOCK_M // 64))
            ]

            a_scale_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout(M * KB, 1),
                ),
                max_size=False,
            )
            TPR = 256 // BLOCK_M
            q_row = tid // TPR
            q_part = tid % TPR
            q_sid = sorted_view[q_row].bitcast(fx.Uint32)

            # Weight cache policy per block: cached when another live 64-row
            # sub-tile of the same expert reads the same weights (an L2/MALL hit
            # for the second reader), non-temporal when this sub-tile is the
            # expert's only reader, so its read-once stream does not occupy L2.
            sub_i = e_idx % SUB
            sb = e_idx // SUB
            eids_ptr = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_expert_ids))
            prev_e = eids_ptr[fx.arith.select(sb > 0, sb - 1, sb)]
            prev_diff = (sb == 0) | (prev_e != expert_id)
            if const_expr(SUB > 1):
                nidx = (e_idx + 1) * BLOCK_M
                nsid = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids))[nidx]
                next_dead = (nidx >= max_valid_id) | ((nsid & 0xFFFFFF) >= M)
            else:
                nidx = (sb + 1) * SORT_BM
                next_e = eids_ptr[fx.arith.select(nidx < max_valid_id, sb + 1, sb)]
                next_dead = (nidx >= max_valid_id) | (next_e != expert_id)
            alone = (sub_i == 0) & prev_diff & next_dead
            if alone:
                _gemm_and_store(
                    2,
                    tid,
                    blk_n,
                    expert_id,
                    arg_p_input,
                    arg_p_weight,
                    p_w_scale,
                    a_scale_tensor,
                    row_toks,
                    lds,
                    q_row,
                    q_part,
                    q_sid,
                    p_output,
                    p_out_scale,
                    M,
                    c_out,
                    c_rw_copy,
                )
            else:
                _gemm_and_store(
                    0,
                    tid,
                    blk_n,
                    expert_id,
                    arg_p_input,
                    arg_p_weight,
                    p_w_scale,
                    a_scale_tensor,
                    row_toks,
                    lds,
                    q_row,
                    q_part,
                    q_sid,
                    p_output,
                    p_out_scale,
                    M,
                    c_out,
                    c_rw_copy,
                )

    @flyc.jit
    def launch(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        p_out_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        gateup_blockscale(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            p_out_scale,
            M,
        ).launch(
            grid=(N // BLOCK_N, task_num * SUB, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    launch.compile_hints["llvm_options"] = {"slp-threshold": 100000}
    return launch
