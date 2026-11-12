# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 blockscale (per_1x128) MoE Down for prefill.

One workgroup per sorted M tile keeps the tile's whole A ([BLOCK_M, K] FP8, K =
inter_dim) in LDS and walks every 64-column N tile of the expert's W2
(shuffle_weight(w2, (16, 16)) layout, scales [E, N/128, K/128]). Each 128-K
group accumulates into a fresh partial folded with sa[row, g] * sb[n_block, g];
the route weight is applied after the K loop. Rows go to a per-sorted-row
output that aiter's sorted_sum combines.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import common as fxh
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import _f32_to_bf16

SCALE_BLOCK = 128


def compile_down(*, N, K, TOPK, BLOCK_M):
    """N = model_dim, K = inter_dim."""
    assert K % SCALE_BLOCK == 0
    assert N % 128 == 0, "pairs of 64-column N tiles"
    assert BLOCK_M % 16 == 0 and 16 <= BLOCK_M <= 128
    assert BLOCK_M * K + 0 <= 64 * 1024, "A tile must fit LDS"
    KG = K // SCALE_BLOCK
    NB = N // SCALE_BLOCK
    weight_dtype = fx.Float8E4M3FNUZ
    BLOCK_N = 64
    BLOCK_K = 64
    K_PER_GROUP = SCALE_BLOCK // BLOCK_K
    flyobj = fxh.FlyObjCache()

    @flyc.kernel
    def down_blockscale(
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
        e_idx = fx.gpu.block_idx.y
        tid = fx.gpu.thread_idx.x
        lane = tid % 64
        flyobj.bid = e_idx

        max_valid_id = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]

        if e_idx * BLOCK_M < max_valid_id:
            arg_p_input = fxh.view_as_torch_tensor(p_input, (M, TOPK, K), weight_dtype)
            arg_p_output = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_output, fx.BFloat16) + fx.Int64(e_idx) * (BLOCK_M * N),
                (BLOCK_M, N),
            )
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
            arg_p_output = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False)

            swz = fx.SwizzleType.get(3, 4, 3)

            @fx.union
            class SharedStorage:
                A: fx.Array[weight_dtype, BLOCK_M * K]
                C: fx.Array[fx.BFloat16, 2 * BLOCK_M * BLOCK_N]

            lds = fx.SharedAllocator().allocate(SharedStorage)
            ldsA0 = lds.A.peek().view(
                fx.make_composed_layout(fx.static(swz), fxh.torch_layout(BLOCK_M, K))
            )
            layoutC = fx.make_composed_layout(
                fx.static(swz),
                fx.make_ordered_layout((BLOCK_M, BLOCK_N, 2), (1, 0, 2)),
            )
            layoutCt = fx.make_composed_layout(
                fx.static(swz), fx.make_ordered_layout((BLOCK_N, BLOCK_M, 2), (0, 1, 2))
            )
            ldsC = lds.C.peek().view(layoutC)
            ldsCt = lds.C.peek().view(layoutCt)

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

            # Activation scales of this tile: lane l loads row l % BLOCK_M (and
            # l + 64), then MFMA token column n * 16 + lane % 16 is read by
            # ds_bpermute. One value per (group, n) per lane; rows are
            # (token, slot) of the sorted id.
            a_scale_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout(M * TOPK * KG, 1),
                ),
                max_size=False,
            )
            n_reps = BLOCK_M // 16
            row_ids = []
            for j in range_constexpr(max(1, BLOCK_M // 64)):
                sid = arg_p_sorted_ids[(lane + 64 * j) % BLOCK_M].bitcast(fx.Uint32)
                row_ids.append(fx.Int32(((sid & 0xFFFFFF) * TOPK + (sid >> 24)) * KG))
            sa = []
            for g in range_constexpr(KG):
                loaded = [
                    a_scale_tensor[row_ids[j] + g].bitcast(fx.Int32)
                    for j in range_constexpr(len(row_ids))
                ]
                per_n = []
                for n in range_constexpr(n_reps):
                    addr = fx.Int32(((n * 16) % 64 + lane % 16) * 4)
                    per_n.append(
                        fx.Int32(
                            rocdl.ds_bpermute(
                                T.i32, _raw(addr), _raw(loaded[(n * 16) // 64])
                            )
                        ).bitcast(fx.Float32)
                    )
                sa.append(per_n)
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
            fragP = mm.make_fragment_C(c_fake_tensor)
            fragC_bf16 = fx.make_fragment_like(fragC[0], fx.BFloat16)

            frag_act = flyobj.load_tiled_mma_fragB(mm, ldsA, copy_atom_bits=128)
            fx.gpu.barrier()  # Finish reading ldsA before ldsC reuses its storage.

            sorted_weights = fx.make_view(
                fx.get_iter(arg_p_sorted_weights),
                fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            frag_sorted_weight = flyobj.load_tiled_mma_fragC(
                mm, sorted_weights, copy_atom_bits=32
            )

            ws_view = fx.make_view(
                fxh._as_ptr(p_w_scale) + expert_id * (NB * KG),
                fx.make_layout(NB * KG, 1),
            )
            ws_rsrc = fx.rocdl.get_buffer_rsrc(
                fx.get_iter(fx.rocdl.make_buffer_tensor(ws_view, max_size=False))
            )

            def load_sb(n):
                base = (n * BLOCK_N) // SCALE_BLOCK * KG
                out = []
                for g in range_constexpr(KG):
                    v = buffer_ops.buffer_load(
                        ws_rsrc,
                        rocdl.readfirstlane(T.i32, _raw(fx.Int32(base + g))),
                        vec_width=1,
                        is_scalar=True,
                    )
                    out.append(fx.Int32(v).bitcast(fx.Float32))
                return out

            def gemm_compute(n, fragW, fragC):
                sb = load_sb(n)
                fragC.fill(0)
                for g in range_constexpr(KG):
                    fragP.fill(0)
                    for kk in range_constexpr(K_PER_GROUP):
                        k = g * K_PER_GROUP + kk
                        fx.gemm(
                            mm,
                            fragP,
                            fragW[None, None, None, k],
                            frag_act[None, None, None, 0, k],
                            fragP,
                        )
                    m_reps = fx.size(fx.get_shape(fragC)[1]).to_py_value()
                    for nr in range_constexpr(n_reps):
                        scale = Vec.filled(4, sa[g][nr] * sb[g], fx.Float32)
                        for m in range_constexpr(m_reps):
                            fragC[None, m, nr].store(
                                fx.fma(
                                    fragP[None, m, nr].load(),
                                    scale,
                                    fragC[None, m, nr].load(),
                                )
                            )

            tcopyLDS, cp_ldsc = flyobj.get_tiled_copy_coalesced_mn(
                ldsC[None, None, 0], copy_atom_bits=128, num_threads=256
            )
            thrv_ldsC = tcopyLDS.partition_S(ldsC)
            copy_atom_ = flyobj.get_universal_copy_atom(fragC_bf16.dtype, 64)
            tcopy = flyobj.get_tiled_mma_copy(copy_atom_, mm, "C")
            fragC_bf16r = flyobj.get_retile(tcopy, fragC_bf16)
            thrv_ldsCt = flyobj.get_partition_D(tcopy, ldsCt)

            def postprocess_store2lds(fragC, ldsc_idx):
                for fc, fsw in fxh.all_elements(fragC, frag_sorted_weight):
                    fc.store(fc.load() * fsw.load())
                fragC_bf16.store(_f32_to_bf16(fragC.load()))
                fx.copy(copy_atom_, fragC_bf16r, thrv_ldsCt[None, None, None, ldsc_idx])

            arg_p_output = fx.flat_divide(arg_p_output, (BLOCK_M, BLOCK_N))
            cp_atom_out_128b = flyobj.get_buffer_copy_atom(fx.BFloat16, 128)
            thrv_out = tcopyLDS.partition_D(arg_p_output)
            fragOut = fx.make_fragment_like(thrv_ldsC[None, None, None, 0])

            def postprocess_store2vmem(n, ldsc_idx):
                fx.copy(cp_ldsc, thrv_ldsC[None, None, None, ldsc_idx], fragOut)
                fx.copy(cp_atom_out_128b, fragOut, thrv_out[None, None, None, 0, n])

            frag_weights = [None, None]
            frag_weights[0] = flyobj.load_tiled_mma_fragA(
                mm, weight, [None, None, 0, None]
            )
            gemm_compute(0, frag_weights[0], fragC[0])
            frag_weights[1] = flyobj.load_tiled_mma_fragA(
                mm, weight, [None, None, 1, None]
            )
            postprocess_store2lds(fragC[0], 0)
            fx.gpu.barrier()
            for n, state in range(0, nBN - 2, 2, init=[]):
                postprocess_store2vmem(n, 0)
                flyobj.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 2, None], frag_weights[0]
                )
                gemm_compute(n + 1, frag_weights[1], fragC[1])
                postprocess_store2lds(fragC[1], 1)
                fx.gpu.barrier()

                postprocess_store2vmem(n + 1, 1)
                flyobj.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 3, None], frag_weights[1]
                )
                gemm_compute(n + 2, frag_weights[0], fragC[0])
                postprocess_store2lds(fragC[0], 0)
                fx.gpu.barrier()

            postprocess_store2vmem(nBN - 2, 0)
            gemm_compute(nBN - 1, frag_weights[1], fragC[1])
            postprocess_store2lds(fragC[1], 1)
            fx.gpu.barrier()
            postprocess_store2vmem(nBN - 1, 1)

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
        down_blockscale(
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
