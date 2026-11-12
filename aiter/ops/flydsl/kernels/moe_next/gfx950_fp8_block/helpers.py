# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Portions Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout helpers for the gfx950 fp8 blockscale MoE stages.

Adapted from FlyDSL's kernels/moe/moe_gemm_2stage/layout_helpers.py (upstream
19418894): the B-first 1x4 tiled MMA, the shuffle_weight(16, 16) weight views,
the sorted-index gather/scatter and the pk-bf16 buffer atomics.
"""

import flydsl.expr as fx
import numpy as np
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

FP8_MAX = 448.0  # OCP e4m3fn on gfx950
FP8_MAX_RCP = float(np.float32(1.0) / np.float32(FP8_MAX))
AMAX_FLOOR = 1e-10  # csrc/include/quant_amax_floor.h kAmaxFloor


def reps(tensor, mode):
    return fx.size(fx.get_shape(tensor)[mode]).to_py_value()


def encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def make_1x4_tiled_mma(elem_t, wide_k=False):
    """B-first: weights are MFMA operand A (channels on M), tokens operand B.

    wide_k: gfx950's 16x16x128 f8f6f4 MFMA (unit scales, as aiter's dense blockscale
    GEMM uses it): one MFMA per 128-K scale group. Each lane takes 32 consecutive K of
    both operands; the same permutation on A and B leaves the product unchanged."""
    if wide_k:
        mma_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, elem_t))
        mma_atom = fx.atom_set_value(mma_atom, "scale_a", fx.Int32(0))
        mma_atom = fx.atom_set_value(mma_atom, "scale_b", fx.Int32(0))
        k_perm = fx.make_layout((32, 4), (1, 32))
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((4, 1, 1), (1, 0, 0)), (None, None, k_perm)
        )
        return mma_atom, tiled_mma
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, elem_t))
    k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((4, 1, 1), (1, 0, 0)),
        fx.make_tile(None, None, k_perm),
    )
    return mma_atom, tiled_mma


def make_gateup_weight_view(p_weight, expert_id, contiguous_n, N, K):
    """Per-expert (N, K) view over shuffle_weight(16, 16) W1, grouped so channel
    block b of width contiguous_n is gate[b] and block b+1 is the matching up."""
    group_layout = fx.make_layout(
        ((contiguous_n, 2, N // (contiguous_n * 2)), K),
        ((1, N // 2, contiguous_n), N),
    )
    element_num = 16 // (p_weight.dtype.width // 8)
    return fx.make_view(
        p_weight + fx.Int64(expert_id) * fx.Int64(N * K),
        fx.composition(
            fx.make_layout(
                ((16, N // 16), (element_num, K // element_num)),
                ((element_num, 16 * K), (1, 16 * element_num)),
            ),
            group_layout,
        ),
    )


def make_weight_view(p_weight, expert_id, N, K):
    """Per-expert (N, K) view over shuffle_weight(16, 16) W2."""
    element_num = 16 // (p_weight.dtype.width // 8)
    return fx.make_view(
        p_weight + fx.Int64(expert_id) * fx.Int64(N * K),
        fx.make_layout(
            ((16, N // 16), (element_num, K // element_num)),
            ((element_num, 16 * K), (1, 16 * element_num)),
        ),
    )


def read_sorted_index(tiled_copy_index, tid, lds_index, index_size):
    lds = fx.make_view(lds_index.ptr, fx.make_layout(index_size, 1))
    cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
    lds_thr = tiled_copy_index.get_slice(tid).partition_S(lds)
    index_frag = fx.make_fragment_like(lds_thr)
    fx.copy(cp_atom_lds, lds_thr, index_frag)
    return index_frag


def token_rep_frag(lds_index, BM, tid, dtype=fx.Int32):
    """One value per token column of the B-first C fragment (n reps), from LDS."""
    cp = fx.make_tiled_copy(
        fx.make_copy_atom(fx.UniversalCopy32b(), dtype),
        fx.make_layout(((16, 4, 4), 1), ((1, 0, 0), 0)),
        fx.make_tile(16),
    )
    view = fx.make_view(lds_index.ptr, fx.make_layout(BM, 1))
    thr = cp.get_slice(tid).partition_S(view)
    frag = fx.make_fragment_like(thr)
    fx.copy(fx.make_copy_atom(fx.UniversalCopy32b(), dtype), thr, frag)
    return frag


def load_sorted_weight_frag(arg_sorted_weights, row0, BM, tid):
    sw_ptr = fx.recast_iter(fx.Float32, fx.get_iter(arg_sorted_weights) + row0)
    tw_view = fx.make_view(sw_ptr, fx.make_layout(BM, 1))
    tw_copy = fx.make_tiled_copy(
        fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32),
        fx.make_layout(((16, 4, 4), 1), ((1, 0, 0), 0)),
        fx.make_tile(16),
    )
    tw_thr = tw_copy.get_slice(tid).partition_S(tw_view)
    tw_frag = fx.make_fragment_like(tw_thr)
    fx.copy(fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32), tw_thr, tw_frag)
    return tw_frag


def buffer_atomic_pk(rsrc, elem_idx, reg_vec, elem_bytes):
    """Pairwise buffer atomic add of a bf16 vector (one pk pair per instruction)."""
    z = fx.Int32(0)
    for i in range_constexpr(reg_vec.numel // 2):
        pair = Vec.from_elements([reg_vec[i * 2], reg_vec[i * 2 + 1]], reg_vec.dtype)
        byte_off = (elem_idx + fx.Int32(i * 2)) * fx.Int32(elem_bytes)
        rocdl.raw_ptr_buffer_atomic_fadd(pair, rsrc, byte_off, z, z)


def buffer_atomic_f32(rsrc, elem_idx, reg_vec):
    """Scalar buffer atomic add of an f32 vector."""
    z = fx.Int32(0)
    for i in range_constexpr(reg_vec.numel):
        byte_off = (elem_idx + fx.Int32(i)) * fx.Int32(4)
        rocdl.raw_ptr_buffer_atomic_fadd(reg_vec[i], rsrc, byte_off, z, z)


class TensorWithIndex:
    """Gather/scatter of per-thread tiles by packed sorted ids (token | slot << 24)."""

    def __init__(
        self, view, tile_m, tile_k, index_frag, tiled_copy, tid, is_read_from_mem=True
    ):
        self.view = view
        self.tile_m = tile_m
        self.tile_k = tile_k
        self.is_read_from_mem = is_read_from_mem
        self.index_frag = index_frag
        rank = fx.get_shape(self.view).rank
        dims = [1] * (rank - 1)
        self.tensor_blocks_in_k = fx.zipped_divide(view, fx.make_tile(*dims, tile_k))
        dtype = fx.PointerType.get(fx.Int8.ir_type, 1, 512)
        ptr = fx.inttoptr(dtype, fx.Int32(0))
        self.fake_tensor = fx.make_view(
            ptr, fx.make_layout((tile_m, tile_k), (1, tile_m))
        )
        slc = tiled_copy.get_slice(tid)
        self.fake_tensor_thr = (
            slc.partition_S(self.fake_tensor)
            if is_read_from_mem
            else slc.partition_D(self.fake_tensor)
        )
        self.offset_thread_k = (
            fx.Int32(fx.ptrtoint(fx.get_iter(self.fake_tensor_thr))) // tile_m
        )
        self._guard_rows = 256
        guard_fake = fx.make_view(
            ptr, fx.make_layout((self._guard_rows, tile_k), (1, self._guard_rows))
        )
        guard_thr = (
            slc.partition_S(guard_fake)
            if is_read_from_mem
            else slc.partition_D(guard_fake)
        )
        self.guard_offset = fx.Int32(fx.ptrtoint(fx.get_iter(guard_thr)))
        self.guard_layout = fx.get_layout(guard_thr)

    def copy(
        self,
        copy_atom,
        k_idx,
        frag,
        atomic_rsrc=None,
        out_bytes=None,
        row_stride=None,
        row_limit=None,
    ):
        layout = fx.get_layout(self.fake_tensor_thr)
        rep_m = reps(self.fake_tensor_thr, 1)
        rep_k = reps(self.fake_tensor_thr, 2)
        value_size = fx.get_shape(frag)[0].to_py_value()
        stride_size = fx.get_stride(frag)[0].to_py_value()
        rank = fx.get_shape(self.view).rank
        tensor_block = self.tensor_blocks_in_k[None, (*([None] * (rank - 1)), k_idx)]
        for m in range_constexpr(rep_m):
            if const_expr(atomic_rsrc is not None):
                tok = self.index_frag[0, m] & 0xFFFFFF
                if tok < fx.Int32(row_limit):
                    row_base = tok * fx.Int32(row_stride) + fx.Int32(k_idx) * fx.Int32(
                        self.tile_k
                    )
                    for k in range_constexpr(rep_k):
                        offset_block = fx.crd2idx((0, m, k), layout).to_py_value()
                        chan_off = offset_block // self.tile_m + self.offset_thread_k
                        guard_full = (
                            fx.crd2idx((0, m, k), self.guard_layout).to_py_value()
                            + self.guard_offset
                        )
                        g_row = guard_full % fx.Int32(self._guard_rows)
                        g_col = guard_full // fx.Int32(self._guard_rows)
                        valid = (g_row < fx.Int32(self.tile_m)) & (
                            g_col < fx.Int32(self.tile_k)
                        )
                        reg_vec = frag[None, m, k].load()
                        aligned = (row_base + chan_off) & fx.Int32(~(reg_vec.numel - 1))
                        elem_idx = valid.select(
                            aligned, fx.Int32(row_limit) * fx.Int32(row_stride)
                        )
                        if const_expr(out_bytes == 4):
                            buffer_atomic_f32(atomic_rsrc, elem_idx, reg_vec)
                        else:
                            buffer_atomic_pk(atomic_rsrc, elem_idx, reg_vec, out_bytes)
                continue
            if const_expr(rank == 3):
                # [tokens, topk, N] rows: (token, slot) from the packed sorted id
                tensor_sub_block = tensor_block[
                    None, self.index_frag[0, m] & 0xFFFFFF, self.index_frag[0, m] >> 24
                ]
            else:
                tensor_sub_block = tensor_block[None, self.index_frag[0, m] & 0xFFFFFF]
            for k in range_constexpr(rep_k):
                offset_block = fx.crd2idx((0, m, k), layout).to_py_value()
                offset_k_in_tile = offset_block // self.tile_m + self.offset_thread_k
                reg = frag[None, m, k]
                mem = fx.make_view(
                    fx.get_iter(tensor_sub_block) + offset_k_in_tile,
                    fx.make_layout(value_size, stride_size),
                )
                if const_expr(self.is_read_from_mem):
                    fx.copy(copy_atom, mem, reg)
                else:
                    fx.copy(copy_atom, reg, mem)


TensorWithIndex.copy = ASTRewriter.transform(TensorWithIndex.copy)
