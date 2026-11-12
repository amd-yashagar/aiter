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


def compile_gateup(*, N, K, TOPK, BLOCK_M, BLOCK_N, activation="silu"):
    """N = 2 * inter_dim (gate rows then up rows), K = model_dim."""
    assert activation in ("silu", "gelu")
    assert BLOCK_N in (128, 256), "each wave owns BLOCK_N // 8 gate and up channels"
    assert 32 <= BLOCK_M <= 256 and BLOCK_M % 32 == 0
    assert K % (2 * TILE_K) == 0, "the 2-stage pipeline needs an even K-tile count"
    assert (N // 2) % (BLOCK_N // 2) == 0
    contiguous_n = BLOCK_N // 2
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

        # All indices must be captured before a_ping reuses sorted_lds.
        gpu.barrier()

        swz = fx.SwizzleType.get(3, 4, 3)
        a_lds_layout = fx.make_composed_layout(
            fx.static(swz), fx.make_ordered_layout((TILE_M, TILE_K), order=(1, 0))
        )
        a_ping = fx.make_view(lds.gemm.a_ping.ptr, a_lds_layout)
        a_pong = fx.make_view(lds.gemm.a_pong.ptr, a_lds_layout)

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
        b_g2r = fx.make_tiled_copy_A(buf_cp_atom_r, tiled_mma).get_slice(tid)
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
            rows = [
                a_scale_tensor[row_toks[j] + k_cur]
                for j in range_constexpr(n_row_loads)
            ]
            return [sb_gate, sb_up] + rows

        def fold(scales):
            sb_gate, sb_up = fx.Float32(scales[0]), fx.Float32(scales[1])
            for n in range_constexpr(_n_reps):
                src = fx.Float32(scales[2 + (n * 16) // 64]).bitcast(fx.Int32)
                sa = fx.Int32(
                    rocdl.ds_bpermute(T.i32, _raw(fx.Int32(bperm_addr[n])), _raw(src))
                ).bitcast(fx.Float32)
                sg = Vec.filled(4, sa * sb_gate, fx.Float32)
                su = Vec.filled(4, sa * sb_up, fx.Float32)
                for m in range_constexpr(_m_reps):
                    c_gate[None, m, n].store(
                        fx.fma(p_gate[None, m, n].load(), sg, c_gate[None, m, n].load())
                    )
                    c_up[None, m, n].store(
                        fx.fma(p_up[None, m, n].load(), su, c_up[None, m, n].load())
                    )

        # The scales of tile k_next are loaded with its A/B prefetch, so the fold of
        # the current tile never waits behind the in-order prefetch queue.
        def pipeline_stage(read_i, scales, k_next, do_prefetch):
            write_i = read_i ^ 1
            next_scales = None
            if const_expr(do_prefetch):
                a_idx.copy(buf_cp_atom_r, k_next, a_cp_frag)
                fx.copy(
                    buf_cp_atom_r, bl_g2r[None, None, None, k_next], bl_ret_st[write_i]
                )
                fx.copy(
                    buf_cp_atom_r, br_g2r[None, None, None, k_next], br_ret_st[write_i]
                )
                next_scales = load_scales(k_next)
            p_gate.fill(0)
            p_up.fill(0)
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_cp_atom,
                    a_lds_r[read_i][None, None, ki],
                    a_frag_retile[None, None, ki],
                )
                for n in range_constexpr(_n_reps):
                    for m in range_constexpr(_m_reps):
                        for k in range_constexpr(2):
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
            fold(scales)
            if const_expr(do_prefetch):
                fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w[write_i])
                hot_loop_scheduler()
            rocdl.sched_barrier(0)
            gpu.barrier()
            return next_scales

        a_idx.copy(buf_cp_atom_r, fx.Int32(0), a_cp_frag)
        fx.copy(buf_cp_atom_r, bl_g2r[None, None, None, fx.Int32(0)], bl_ret_st[0])
        fx.copy(buf_cp_atom_r, br_g2r[None, None, None, fx.Int32(0)], br_ret_st[0])
        scales0 = load_scales(fx.Int32(0))
        rocdl.s_waitcnt(vmcnt=0)
        fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w[0])
        gpu.barrier()

        n_sc = 2 + n_row_loads
        acc_init = [c_gate.load(), c_up.load()] + scales0
        results = acc_init
        for iv, state in range(0, num_tiles // 2 - 1, 1, init=acc_init):
            c_gate.store(state[0])
            c_up.store(state[1])
            kb = fx.Int32(iv * 2)
            s1 = pipeline_stage(
                0, [state[2 + i] for i in range_constexpr(n_sc)], kb + 1, True
            )
            s2 = pipeline_stage(1, s1, kb + 2, True)
            results = yield [c_gate.load(), c_up.load()] + s2
        c_gate.store(results[0])
        c_up.store(results[1])
        kb = fx.Int32(num_tiles - 2)
        s1 = pipeline_stage(
            0, [results[2 + i] for i in range_constexpr(n_sc)], kb + 1, True
        )
        pipeline_stage(1, s1, fx.Int32(0), False)
        return c_gate, c_up

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
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        in_ptr = fx.recast_iter(weight_dtype, fxh._as_ptr(p_input))
        arg_p_input = fx.make_view(in_ptr, fx.make_layout((M, K), (K, 1)))
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        p_weight = fx.recast_iter(weight_dtype, fxh._as_ptr(p_weight))
        if e_idx * BLOCK_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.gemm = lds.gemm.peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids) + e_idx * BLOCK_M),
                fx.make_layout(BLOCK_M, 1),
            )
            expert_id = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_expert_ids))[
                e_idx
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

            c_gate_frag, c_up_frag = gemm_blockscale(
                blk_n,
                expert_id,
                arg_p_input,
                arg_p_weight,
                p_w_scale,
                a_scale_tensor,
                row_toks,
                lds,
            )

            c_out_bf16 = _gateup_pair_bf16(c_gate_frag, c_up_frag)

            _, _tiled_mma = _make_tiled_mma()
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
            c_out.copy(buf_atom_w128, blk_n, rd)

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
            M,
        ).launch(
            grid=(N // BLOCK_N, task_num, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch
