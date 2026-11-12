# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 per-token / per-tensor MoE Down for prefill.

One workgroup per sorted M tile keeps the tile's whole A ([BLOCK_M, K] FP8, K =
inter_dim, gathered by sorted token id from [M, TOPK, K]) in LDS and walks every
64-column N tile of the expert's W2 (aiter's shuffle_weight(w2, (16, 16)) layout).
The FP32 accumulator is scaled once after the K loop by the row factor
a2_scale[row] * route_weight[row] and the column factor w2_scale[expert, n]
(per-tensor: one scalar each), rounded to bf16 and added into out[token] with
packed bf16 atomics; the sort zero-fills the output.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr

from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import common as fxh
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import _f32_to_bf16


def compile_down(*, N, K, TOPK, BLOCK_M, PER_TENSOR=False, ATOMIC=True):
    """N = model_dim, K = inter_dim. ATOMIC adds into out[token]; otherwise each
    sorted row is stored to out[sorted_row] for a separate per-token sum."""
    assert K % 64 == 0
    assert N % 128 == 0, "pairs of 64-column N tiles"
    assert BLOCK_M % 32 == 0 and 32 <= BLOCK_M <= 128
    assert BLOCK_M * K <= 48 * 1024, "A tile must fit LDS beside the C buffers"
    weight_dtype = fx.Float8E4M3FNUZ
    BLOCK_N = 64
    BLOCK_K = 64
    KT = K // BLOCK_K
    A_ROWS_PER_PASS = 256 // (BLOCK_N // 2)
    flyobj = fxh.FlyObjCache()

    @flyc.kernel
    def down_token(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
    ):
        # Cached builders hold IR of one trace; a second compile of this
        # kernel (warmup, then a new specialization) must not reuse them.
        flyobj.clear_all()
        e_idx = fx.gpu.block_idx.y
        tid = fx.gpu.thread_idx.x
        flyobj.bid = e_idx

        max_valid_id = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]

        if e_idx * BLOCK_M < max_valid_id:
            arg_p_input = fxh.view_as_torch_tensor(p_input, (M, TOPK, K), weight_dtype)
            arg_p_sorted_ids = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_ids) + e_idx * BLOCK_M, (BLOCK_M,), fx.Int32
            )
            arg_p_sorted_weights = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_weights) + e_idx * BLOCK_M,
                (BLOCK_M,),
                fx.Float32,
            )
            expert_id = fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[
                e_idx
            ]

            arg_p_weight = fx.make_view(
                fxh._as_ptr(p_weight, weight_dtype) + fx.Int64(expert_id * N * K),
                fx.make_layout(
                    ((16, N // 16), (16, K // 16)), ((16, 16 * K), (1, 16 * 16))
                ),
            )
            arg_p_weight = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)

            swz = fx.SwizzleType.get(3, 4, 3)

            # Both views start with the same row-info arrays, so the C buffers can
            # reuse the A tile's storage while the row info stays live.
            @fx.struct
            class WithA:
                row_fac: fx.Array[fx.Float32, BLOCK_M, 16]
                row_sid: fx.Array[fx.Int32, BLOCK_M, 16]
                A: fx.Array[weight_dtype, BLOCK_M * K, 16]

            @fx.struct
            class WithC:
                row_fac: fx.Array[fx.Float32, BLOCK_M, 16]
                row_sid: fx.Array[fx.Int32, BLOCK_M, 16]
                C: fx.Array[fx.BFloat16, 2 * BLOCK_M * BLOCK_N, 16]

            @fx.union
            class SharedStorage:
                a: WithA
                c: WithC

            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds_a = lds.a.peek()
            lds_c = lds.c.peek()
            ldsA0 = lds_a.A.view(
                fx.make_composed_layout(fx.static(swz), fxh.torch_layout(BLOCK_M, K))
            )
            layoutC = fx.make_composed_layout(
                fx.static(swz),
                fx.make_ordered_layout((BLOCK_M, BLOCK_N, 2), (1, 0, 2)),
            )
            layoutCt = fx.make_composed_layout(
                fx.static(swz), fx.make_ordered_layout((BLOCK_N, BLOCK_M, 2), (0, 1, 2))
            )
            ldsC = lds_c.C.view(layoutC)
            ldsCt = lds_c.C.view(layoutCt)
            row_fac = fx.make_view(lds_a.row_fac.ptr, fx.make_layout(BLOCK_M, 1))
            row_sid = fx.make_view(lds_a.row_sid.ptr, fx.make_layout(BLOCK_M, 1))

            arg_p_input = fx.rocdl.make_buffer_tensor(
                arg_p_input,
                max_size=False,
                num_records_bytes=fx.Int64(M) * (TOPK * K),
            )
            cp_atom = flyobj.get_buffer_copy_atom(arg_p_input.dtype, 128)

            def flatten_A(x):
                x = fx.select(x, [1, 0])
                return fx.group(x, 0, -1)

            cp_ldsA0 = flatten_A(ldsA0)
            cp_rows = flatten_A(
                fxh.make_1d_coord_tensor(ldsA0, 0, fx.get_iter(arg_p_sorted_ids))
            )
            cp_cols = flatten_A(
                fxh.make_1d_coord_tensor(ldsA0, 1, fx.make_int_tuple(0))
            )
            for dst, row, col in fxh.all_copy_atoms(
                cp_ldsA0, cp_rows, cp_cols, atom_bits=128, num_threads=256
            ):
                sorted_id = row[0].bitcast(fx.Uint32)
                atom_A = fxh.atom_tensor(
                    arg_p_input, (sorted_id & 0xFFFFFF, sorted_id >> 24, col[0]), 128
                )
                fx.copy(cp_atom, atom_A, dst)

            # Row factor a2_scale[row] * route weight, and the row's sorted id for
            # the atomic epilogue; padding rows get factor 0 and are skipped.
            if tid < BLOCK_M:
                sid = arg_p_sorted_ids[tid].bitcast(fx.Uint32)
                tok = fx.Int32(sid & 0xFFFFFF)
                slot = fx.Int32(sid >> 24)
                valid = (tok < M) & (slot < TOPK)
                row = tok * TOPK + slot
                a_sc = fxh.view_as_torch_tensor(
                    p_a_scale, (1 if PER_TENSOR else M * TOPK,), fx.Float32
                )
                a2 = a_sc[0] if PER_TENSOR else a_sc[valid.select(row, fx.Int32(0))]
                fac = a2 * arg_p_sorted_weights[tid]
                row_fac[tid] = valid.select(fac, fx.Float32(0.0))
                row_sid[tid] = valid.select(fx.Int32(sid), fx.Int32(-1))
            fx.gpu.barrier()

            weight = fx.flat_divide(arg_p_weight, (BLOCK_N, BLOCK_K))
            ldsA = fx.flat_divide(ldsA0, (BLOCK_M, BLOCK_K))
            nBN = N // BLOCK_N

            mm = flyobj.create_thr_mma(weight_dtype, (4, 1, 1))
            c_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_ordered_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            fragC = [
                mm.make_fragment_C(c_fake_tensor),
                mm.make_fragment_C(c_fake_tensor),
            ]
            fragC_bf16 = fx.make_fragment_like(fragC[0], fx.BFloat16)

            frag_act = flyobj.load_tiled_mma_fragB(mm, ldsA, copy_atom_bits=128)
            frag_row = flyobj.load_tiled_mma_fragC(
                mm,
                fx.make_view(
                    fx.get_iter(row_fac), fx.make_layout((BLOCK_N, BLOCK_M), (0, 1))
                ),
                copy_atom_bits=32,
            )
            fx.gpu.barrier()  # Finish reading ldsA before ldsC reuses its storage.

            def gemm_compute(n, fragW, fragC):
                fragC.fill(0)
                for k in range_constexpr(KT):
                    fx.gemm(
                        mm,
                        fragC,
                        fragW[None, None, None, k],
                        frag_act[None, None, None, 0, k],
                        fragC,
                    )

            tcopyLDS, cp_ldsc = flyobj.get_tiled_copy_coalesced_mn(
                ldsC[None, None, 0], copy_atom_bits=128, num_threads=256
            )
            thrv_ldsC = tcopyLDS.partition_S(ldsC)
            copy_atom_ = flyobj.get_universal_copy_atom(fragC_bf16.dtype, 64)
            tcopy = flyobj.get_tiled_mma_copy(copy_atom_, mm, "C")
            fragC_bf16r = flyobj.get_retile(tcopy, fragC_bf16)
            thrv_ldsCt = flyobj.get_partition_D(tcopy, ldsCt)

            def col_scale(n):
                if const_expr(PER_TENSOR):
                    return None
                ws = fx.make_view(
                    fxh._as_ptr(p_w_scale, fx.Float32) + expert_id * N + n * BLOCK_N,
                    fx.make_layout((BLOCK_N, BLOCK_M), (1, 0)),
                )
                return flyobj.load_tiled_mma_fragC(mm, ws, copy_atom_bits=32)

            def postprocess_store2lds(n, fragC, ldsc_idx):
                if const_expr(PER_TENSOR):
                    w_e = fx.make_view(
                        fxh._as_ptr(p_w_scale, fx.Float32) + expert_id,
                        fx.make_layout(1, 1),
                    )[0]
                    for fc, fr in fxh.all_elements(fragC, frag_row):
                        fc.store(fc.load() * fr.load() * w_e)
                else:
                    frag_col = col_scale(n)
                    for fc, fr, fs in fxh.all_elements(fragC, frag_row, frag_col):
                        fc.store(fc.load() * fr.load() * fs.load())
                fragC_bf16.store(_f32_to_bf16(fragC.load()))
                fx.copy(copy_atom_, fragC_bf16r, thrv_ldsCt[None, None, None, ldsc_idx])

            out_bf16 = fxh._as_ptr(p_output, fx.BFloat16)
            a_row = tid // (BLOCK_N // 2)
            a_col = (tid % (BLOCK_N // 2)) * 2

            if const_expr(not ATOMIC):
                rows_out = fx.rocdl.make_buffer_tensor(
                    fxh.view_as_torch_tensor(
                        out_bf16 + fx.Int64(e_idx) * (BLOCK_M * N), (BLOCK_M, N)
                    ),
                    max_size=False,
                )
                thrv_out = tcopyLDS.partition_D(
                    fx.flat_divide(rows_out, (BLOCK_M, BLOCK_N))
                )
                fragOut = fx.make_fragment_like(thrv_ldsC[None, None, None, 0])
                cp_atom_out_128b = flyobj.get_buffer_copy_atom(fx.BFloat16, 128)

            def postprocess_atomic(n, ldsc_idx):
                if const_expr(not ATOMIC):
                    fx.copy(cp_ldsc, thrv_ldsC[None, None, None, ldsc_idx], fragOut)
                    fx.copy(cp_atom_out_128b, fragOut, thrv_out[None, None, None, 0, n])
                else:
                    # 32 consecutive lanes add one row's 32 consecutive dwords, so
                    # each atomic instruction covers two whole 128-byte row tiles.
                    for p in range_constexpr(BLOCK_M // A_ROWS_PER_PASS):
                        r = a_row + p * A_ROWS_PER_PASS
                        v = fxh.atom_tensor(
                            ldsC[None, None, ldsc_idx], (r, a_col), 32
                        ).load()
                        sid = row_sid[r]
                        if sid >= 0:
                            tok = fx.Int32(sid.bitcast(fx.Uint32) & 0xFFFFFF)
                            dst = out_bf16 + (fx.Int64(tok) * N + n * BLOCK_N + a_col)
                            fxh.atomic_add_bf16(dst, v)

            frag_weights = [None, None]
            frag_weights[0] = flyobj.load_tiled_mma_fragA(
                mm, weight, [None, None, 0, None]
            )
            gemm_compute(0, frag_weights[0], fragC[0])
            frag_weights[1] = flyobj.load_tiled_mma_fragA(
                mm, weight, [None, None, 1, None]
            )
            postprocess_store2lds(0, fragC[0], 0)
            fx.gpu.barrier()
            for n, state in range(0, nBN - 2, 2, init=[]):
                postprocess_atomic(n, 0)
                flyobj.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 2, None], frag_weights[0]
                )
                gemm_compute(n + 1, frag_weights[1], fragC[1])
                postprocess_store2lds(n + 1, fragC[1], 1)
                fx.gpu.barrier()

                postprocess_atomic(n + 1, 1)
                flyobj.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 3, None], frag_weights[1]
                )
                gemm_compute(n + 2, frag_weights[0], fragC[0])
                postprocess_store2lds(n + 2, fragC[0], 0)
                fx.gpu.barrier()

            postprocess_atomic(nBN - 2, 0)
            gemm_compute(nBN - 1, frag_weights[1], fragC[1])
            postprocess_store2lds(nBN - 1, fragC[1], 1)
            fx.gpu.barrier()
            postprocess_atomic(nBN - 1, 1)

    @flyc.jit
    def launch(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        down_token(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            M,
        ).launch(grid=(1, task_num, 1), block=(256, 1, 1), stream=stream)

    return launch
