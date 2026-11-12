# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Portions Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 fp8 blockscale (per_1x128) MoE stage 2: down GEMM, route weight and combine.

Built on FlyDSL's kernels/moe/moe_gemm_2stage/gemm2.py (upstream 19418894): B-first
MFMA with the shuffle_weight(16, 16) W2 loaded straight to VGPRs, A through LDS,
CShuffle epilogue with channel-major pk-bf16 buffer atomics.

A is core's per_1x128 intermediate: values [rows, I] fp8 and scales [rows, I/128] fp32,
row = sorted position, so a tile's rows are contiguous (no gather). Per 128-wide K group
g: P = A[:, g] . W2[:, g]^T, acc += P * (inter_scale[row, g] * w2_scale[block, g]).
Epilogue: acc * sorted_weight[row] (unless doweight_stage1), then by `combine`:
  atomic_bf16  pk-bf16 atomics into out[token], zeroed by the sort (as aiter's CK stage 2)
  atomic_f32   f32 atomics into a zeroed f32 [M, N] buffer, rounded to bf16 once after
  rows_f32     f32 per-route rows [M, topk, N] (row = token * topk + slot), summed over
               topk in fp32 and rounded once after (deterministic)
Padding rows (token >= M) are skipped.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from . import helpers as fxh

# keep fp32 folds as VOP2: v_pk_fma_f32 does not issue under an MFMA (lag-scale-fold-behind-mfma)
_NO_PACKED_F32 = {"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]}

COMBINES = ("atomic_bf16", "atomic_f32", "rows_f32")
_BUF_COPY = {
    32: fx.rocdl.BufferCopy32b,
    64: fx.rocdl.BufferCopy64b,
    128: fx.rocdl.BufferCopy128b,
}
_UNI_COPY = {
    32: fx.UniversalCopy32b,
    64: fx.UniversalCopy64b,
    128: fx.UniversalCopy128b,
}


def _build_stage2(
    *,
    model_dim,
    inter_dim,
    experts,
    topk,
    tile_m,
    tile_n,
    doweight_stage1,
    combine,
    wide_k,
):
    elem_t = fx.Float8E4M3FN
    K = int(inter_dim)
    KB = K // 128
    N = int(model_dim)
    NB = N // 128
    BM = int(tile_m)
    BN = int(tile_n)
    TILE_K = 128
    TOPK = int(topk)
    assert combine in COMBINES, combine
    out_is_f32 = combine != "atomic_bf16"
    atomic = combine.startswith("atomic")
    out_elem = fx.Float32 if out_is_f32 else fx.BFloat16
    out_bytes = 4 if out_is_f32 else 2
    assert K % TILE_K == 0, f"inter_dim={K} must be a multiple of 128"
    assert (
        BN in (64, 128, 256) and N % BN == 0
    ), f"tile_n={BN} must divide model_dim={N}"
    assert BM in (16, 32, 64, 128), f"tile_m={BM}"
    a_lds_bytes = max(BM, 256 // (TILE_K // 16)) * TILE_K  # the A copy covers 32 rows
    cshuf_bytes = BM * BN * out_bytes
    region_bytes = max(2 * a_lds_bytes, cshuf_bytes)

    @fx.struct
    class GemmBuffers:
        region: fx.Array[fx.Int8, region_bytes, 16]

    @fx.struct
    class Index:
        pos: fx.Array[fx.Int32, 256, 16]
        ids: fx.Array[fx.Int32, 256, 16]

    @fx.struct
    class SharedStorage:
        index: Index
        gemm: GemmBuffers

    _val_per_thr = 16
    _swz_params = (3, 4, 3)
    _thrs_k = TILE_K // _val_per_thr
    _thrs_m = 256 // _thrs_k
    _m_per_wave = _thrs_m // 4

    def _gemm(
        blk_n,
        arg_p_input,
        arg_p_weight,
        lds,
        rows,
        a_scale_t,
        w_scale_t,
        expert_id,
        row0,
    ):
        tid = gpu.thread_idx.x
        mma_atom, tiled_mma = fxh.make_1x4_tiled_mma(elem_t, wide_k)
        a_tensor = fx.rocdl.make_buffer_tensor(
            arg_p_input, max_size=False, num_records_bytes=fx.Int64(rows) * fx.Int64(K)
        )
        b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)
        a_size_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((BM, K), (K, 1))),
            max_size=False,
        )
        a_tile = fx.flat_divide(a_size_buf, fx.make_tile(BM, TILE_K))[
            None, None, 0, None
        ]
        buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_t)
        g2r_tv_layout = fx.make_layout(
            ((_thrs_k, _thrs_m), (1, _val_per_thr)),
            ((_thrs_m * _val_per_thr, 1), (1, _thrs_m)),
        )
        a_mem_cp_g2r = fx.make_tiled_copy(
            buf_cp_atom_r, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
        )
        tiled_copy_pos = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
            fx.make_layout(((_thrs_k, _m_per_wave, 4), 1), ((0, 1, _m_per_wave), 0)),
            fx.make_tile(_thrs_m),
        )
        a_index_frag = fxh.read_sorted_index(tiled_copy_pos, tid, lds.index.pos, BM)
        a_idx = fxh.TensorWithIndex(
            a_tensor, BM, TILE_K, a_index_frag, a_mem_cp_g2r, tid
        )
        a_mem_thr = a_mem_cp_g2r.get_slice(tid).partition_S(a_tile)
        a_cp_frag = fx.make_fragment_like(a_mem_thr[None, None, None, 0])
        pos_frag = fxh.token_rep_frag(lds.index.pos, BM, tid)
        gpu.barrier()

        swz = fx.SwizzleType.get(*_swz_params)
        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_t)
        a_lds = fx.make_view(
            fx.recast_iter(elem_t, lds.gemm.region.ptr),
            fx.make_composed_layout(
                fx.static(swz), fx.make_ordered_layout((BM, TILE_K), order=(1, 0))
            ),
        )
        a_r2s = fx.make_tiled_copy(
            uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
        ).get_slice(tid)
        a_lds_w = a_r2s.partition_D(a_lds)
        a_lds_r = (
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma)
            .get_slice(tid)
            .partition_S(a_lds)
        )
        a_frag = tiled_mma.make_fragment_B(a_lds)
        a_frag_retile = (
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).retile(a_frag)
        )
        a_cp_frag_retile = (
            fx.make_tiled_copy(
                uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
            )
            .get_slice(tid)
            .retile(a_cp_frag)
        )

        b_tile = fx.flat_divide(b_tensor, fx.make_tile(BN, TILE_K))[
            None, None, blk_n, None
        ]
        b_g2r = fx.make_tiled_copy_A(buf_cp_atom_r, tiled_mma).get_slice(tid)
        b_g2r_s = b_g2r.partition_S(b_tile)
        b_frag = tiled_mma.make_fragment_A(b_tile[None, None, 0])
        b_ret = b_g2r.retile(b_frag)
        _b_loads_per_tile = fx.size(fx.get_shape(b_frag)).to_py_value() // _val_per_thr

        c_fake_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((BN, BM), (BM, 1))),
            max_size=False,
        )
        c_fake = fx.flat_divide(c_fake_buf, fx.make_tile(BN, BM))[None, None, 0, 0]
        c_frag = tiled_mma.make_fragment_C(c_fake)
        p_frag = tiled_mma.make_fragment_C(c_fake)
        c_frag.fill(0)
        m_reps = fxh.reps(c_frag, 1)
        n_reps = fxh.reps(c_frag, 2)
        k_iters = fxh.reps(a_lds_r, 2)
        # w2_scale [E, N/128, KB]; m rep m covers channels blk_n*BN + m*64 + [0, 64)
        sw_base = [
            (
                expert_id * fx.Int32(NB)
                + (blk_n * fx.Int32(BN) + fx.Int32(m * 64)) // fx.Int32(128)
            )
            * fx.Int32(KB)
            for m in range_constexpr(m_reps)
        ]
        pos_rows = [pos_frag[0, n] * fx.Int32(KB) for n in range_constexpr(n_reps)]

        for iv in range_constexpr(KB):
            kb = fx.Int32(iv)
            a_idx.copy(buf_cp_atom_r, kb, a_cp_frag)
            fx.copy(buf_cp_atom_r, b_g2r_s[None, None, None, kb], b_ret)
            sa = [a_scale_t[pos_rows[n] + kb] for n in range_constexpr(n_reps)]
            sb = [w_scale_t[sw_base[m] + kb] for m in range_constexpr(m_reps)]
            rocdl.s_waitcnt(fxh.encode_waitcnt(vmcnt=_b_loads_per_tile))
            gpu.barrier()
            fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w)
            gpu.barrier()
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_cp_atom, a_lds_r[None, None, ki], a_frag_retile[None, None, ki]
                )
            p_frag.fill(0)
            fx.gemm(mma_atom, p_frag, b_frag, a_frag, p_frag)
            for n in range_constexpr(n_reps):
                for m in range_constexpr(m_reps):
                    f = sa[n] * sb[m]
                    p = p_frag[None, m, n].load()
                    c = c_frag[None, m, n].load()
                    c_frag[None, m, n].store(
                        Vec.from_elements(
                            [fx.fma(p[v], f, c[v]) for v in range(4)], fx.Float32
                        )
                    )
        return c_frag

    _gemm = ASTRewriter.transform(_gemm)

    def _weight_and_cast(c_frag, tw_frag):
        out_frag = fx.make_fragment_like(c_frag, dtype=out_elem)
        m_reps = fxh.reps(c_frag, 1)
        n_reps = fxh.reps(c_frag, 2)
        for n in range_constexpr(n_reps):
            for m in range_constexpr(m_reps):
                acc = c_frag[None, m, n].load()
                if const_expr(not doweight_stage1):
                    acc = acc * tw_frag[0, n]
                if const_expr(out_is_f32):
                    out_frag[None, m, n].store(acc)
                else:
                    out_frag[None, m, n].store(acc.to(out_elem))
        return out_frag

    _weight_and_cast = ASTRewriter.transform(_weight_and_cast)

    @flyc.kernel
    def fp8b_stage2(
        arg_out: fx.Tensor,
        arg_inter: fx.Tensor,
        arg_inter_scale: fx.Tensor,
        arg_w: fx.Tensor,
        arg_w_scale: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_rows: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y
        tokens = i32_tokens
        rows = i32_rows
        in_ptr = fx.recast_iter(elem_t, fx.get_iter(arg_inter))
        arg_p_input = fx.make_view(
            in_ptr, fx.make_layout((rows, fx.Int32(K)), (fx.Int32(K), 1))
        )
        num_valid = fx.make_view(
            fx.recast_iter(fx.Int32, fx.get_iter(arg_num_valid_ids)),
            fx.make_layout(2, 1),
        )[0]
        if e_idx * fx.Int32(BM) < num_valid:
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            row0 = e_idx * fx.Int32(BM)
            sorted_ids = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Int32, fx.get_iter(arg_sorted_token_ids) + row0),
                    fx.make_layout(BM, 1),
                ),
                max_size=False,
            )
            expert_id = fx.make_view(
                fx.recast_iter(fx.Int32, fx.get_iter(arg_expert_ids)),
                fx.make_layout(1 << 20, 1),
            )[e_idx]
            w_ptr = fx.recast_iter(elem_t, fx.get_iter(arg_w))
            arg_p_weight = fxh.make_weight_view(w_ptr, expert_id, N, K)
            pos_view = fx.make_view(lds.index.pos.ptr, fx.make_layout(256, 1))
            ids_view = fx.make_view(lds.index.ids.ptr, fx.make_layout(256, 1))
            # Slots past BM read by the copy layouts point at row `rows` / token `tokens` (OOB).
            pos_view[tid] = rows
            ids_view[tid] = tokens
            gpu.barrier()
            if tid < fx.Int32(BM):
                pos_view[tid] = row0 + tid
                ids_view[tid] = sorted_ids[tid]
            gpu.barrier()
            a_scale_t = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Float32, fx.get_iter(arg_inter_scale)),
                    fx.make_layout(rows * fx.Int32(KB), 1),
                ),
                max_size=False,
                num_records_bytes=fx.Int64(rows) * fx.Int64(KB * 4),
            )
            w_scale_t = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Float32, fx.get_iter(arg_w_scale)),
                    fx.make_layout(fx.Int32(experts * NB * KB), 1),
                ),
                max_size=False,
            )
            tw_frag = fxh.load_sorted_weight_frag(arg_sorted_weights, row0, BM, tid)

            out_rows = tokens if const_expr(atomic) else tokens * fx.Int32(TOPK)
            out_atomic_rsrc = fx.rocdl.get_buffer_rsrc(
                fx.get_iter(
                    fx.rocdl.make_buffer_tensor(
                        arg_out,
                        max_size=False,
                        num_records_bytes=fx.Int64(out_rows) * fx.Int64(N * out_bytes),
                    )
                )
            )
            if const_expr(atomic):
                arg_p_output = fx.make_view(
                    fx.recast_iter(out_elem, fx.get_iter(arg_out)),
                    fx.make_layout((tokens, fx.Int32(N)), (fx.Int32(N), 1)),
                )
            else:
                # per-route rows [tokens, topk, N]: row = token * topk + slot
                arg_p_output = fx.make_view(
                    fx.recast_iter(out_elem, fx.get_iter(arg_out)),
                    fx.make_layout(
                        (tokens, fx.Int32(TOPK), fx.Int32(N)),
                        (fx.Int32(TOPK * N), fx.Int32(N), 1),
                    ),
                )
            out_tensor = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False)
            _cshuffle_nlane = 32
            _n_row_thr = 256 // _cshuffle_nlane
            # Atomics: two values per lane keep the 32 lanes on contiguous channels
            # (one pk pair, or two f32 atomics, per instruction). Rows: 128-bit stores.
            _e_vec = 2 if const_expr(atomic) else 128 // out_elem.width
            _n_chan_tile = _cshuffle_nlane * _e_vec
            _epi_bits = _e_vec * out_elem.width
            buf_atom = fx.make_copy_atom(_BUF_COPY[_epi_bits](), out_elem)
            c_rw_copy = fx.make_tiled_copy(
                buf_atom,
                fx.make_layout(
                    ((_cshuffle_nlane, _n_row_thr), _e_vec),
                    ((_n_row_thr * _e_vec, 1), _n_row_thr),
                ),
                fx.make_tile(_n_row_thr, _n_chan_tile),
            )
            c_index_copy = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((_cshuffle_nlane, _n_row_thr), 1), ((0, 1), 0)),
                fx.make_tile(_n_row_thr),
            )
            c_out_index_frag = fxh.read_sorted_index(
                c_index_copy, tid, lds.index.ids, BM
            )
            c_out = fxh.TensorWithIndex(
                out_tensor,
                BM,
                BN,
                c_out_index_frag,
                c_rw_copy,
                tid,
                is_read_from_mem=False,
            )

            c_frag = _gemm(
                blk_n,
                arg_p_input,
                arg_p_weight,
                lds,
                rows,
                a_scale_t,
                w_scale_t,
                expert_id,
                row0,
            )
            c_out_frag = _weight_and_cast(c_frag, tw_frag)

            _, _tiled_mma = fxh.make_1x4_tiled_mma(elem_t, wide_k)
            cshuf_atom_w = fx.make_copy_atom(fx.UniversalCopy64b(), out_elem)
            cshuf_atom_r = fx.make_copy_atom(_UNI_COPY[_epi_bits](), out_elem)
            cshuf_ptr = fx.recast_iter(out_elem, lds.gemm.region.ptr)
            swz_c = fx.SwizzleType.get(3, 2 if const_expr(out_is_f32) else 3, 3)
            lds_c_store = fx.make_view(
                cshuf_ptr,
                fx.make_composed_layout(
                    fx.static(swz_c), fx.make_ordered_layout((BN, BM), order=(0, 1))
                ),
            )
            lds_c = fx.make_view(
                cshuf_ptr,
                fx.make_composed_layout(
                    fx.static(swz_c), fx.make_ordered_layout((BM, BN), order=(1, 0))
                ),
            )
            gpu.barrier()
            store_c = fx.make_tiled_copy_C(cshuf_atom_w, _tiled_mma).get_slice(tid)
            fx.copy(
                cshuf_atom_w,
                store_c.retile(c_out_frag),
                store_c.partition_D(lds_c_store),
            )
            gpu.barrier()
            rd = fx.make_fragment_like(c_rw_copy.get_slice(tid).partition_S(lds_c))
            fx.copy(cshuf_atom_r, c_rw_copy.get_slice(tid).partition_S(lds_c), rd)
            if const_expr(atomic):
                c_out.copy(
                    buf_atom,
                    blk_n,
                    rd,
                    atomic_rsrc=out_atomic_rsrc,
                    out_bytes=out_bytes,
                    row_stride=N,
                    row_limit=tokens,
                )
            else:
                c_out.copy(buf_atom, blk_n, rd)

    @flyc.jit
    def launch_fp8b_stage2(
        arg_out: fx.Tensor,
        arg_inter: fx.Tensor,
        arg_inter_scale: fx.Tensor,
        arg_w: fx.Tensor,
        arg_w_scale: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_rows: fx.Int32,
        i32_tiles: fx.Int32,
        stream: fx.Stream,
    ):
        fp8b_stage2(
            arg_out,
            arg_inter,
            arg_inter_scale,
            arg_w,
            arg_w_scale,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            i32_tokens,
            i32_rows,
            value_attrs=_NO_PACKED_F32,
        ).launch(
            grid=(N // BN, fx.Int64(i32_tiles), 1), block=(256, 1, 1), stream=stream
        )

    return launch_fp8b_stage2


@functools.lru_cache(maxsize=256)
def compile_stage2(
    *,
    model_dim,
    inter_dim,
    experts,
    topk,
    tile_m,
    tile_n=128,
    doweight_stage1=False,
    combine="atomic_bf16",
    wide_k=True,
):
    return _build_stage2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        doweight_stage1=doweight_stage1,
        combine=combine,
        wide_k=wide_k,
    )
