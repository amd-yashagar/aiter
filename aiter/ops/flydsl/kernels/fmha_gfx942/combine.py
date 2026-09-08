# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused LSE-weighted split-K combine for gfx942 packed-varlen FMHA."""

from __future__ import annotations

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import format_kernel_name
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    _run_compiled,
    _to_raw as _raw,
    ptr_arg,
    ptr_rsrc,
)

_LOG2E = 1.4426950408889634
_BLOCK = 256
_VEC = 4  # dwords (8 bf16) per load/store
_DENOM_MIN = 1.0e-20


def _unpack_bf16_pair(raw_dw):
    lo16 = raw_dw & 0xFFFF
    hi16 = (raw_dw >> 16) & 0xFFFF
    return (lo16 << 16).bitcast(fx.Float32), (hi16 << 16).bitcast(fx.Float32)


def _pack_bf16_pair(acc_lo, acc_hi):
    lo_i32 = fx.Uint32(acc_lo.to(fx.BFloat16).bitcast(fx.Uint16))
    hi_i32 = fx.Uint32(acc_hi.to(fx.BFloat16).bitcast(fx.Uint16))
    return lo_i32 | (hi_i32 << 16)


def build_combine_split_kv_module(num_splits: int, head_dim: int):
    """One thread per ``(t, h)`` row; fused amax/exp/sum/madd/store."""
    if num_splits < 2:
        raise ValueError(f"fused combine needs S>=2, got {num_splits}")
    if head_dim % 8 != 0:
        raise ValueError(f"fused combine needs D%8==0, got {head_dim}")
    n_vec = (head_dim // 2) // _VEC
    row_dwords = head_dim // 2
    module_name = format_kernel_name(f"fmha_gfx942_combine_s{num_splits}_d{head_dim}")

    @flyc.kernel(name=module_name, known_block_size=[_BLOCK, 1, 1])
    def combine_split_kv_kernel(
        o_parts: fx.Pointer,
        lse_parts: fx.Pointer,
        out: fx.Pointer,
        lse_out: fx.Pointer,
        seqlen_q: fx.Int32,
        num_heads: fx.Int32,
    ):
        gid = fx.Int32(fx.block_idx.x) * fx.Int32(_BLOCK) + fx.Int32(fx.thread_idx.x)
        n_rows = seqlen_q * num_heads
        if gid < n_rows:
            token = gid // num_heads
            head = gid % num_heads
            o_rsrc = ptr_rsrc(o_parts)
            lse_rsrc = ptr_rsrc(lse_parts)
            out_rsrc = ptr_rsrc(out)
            lse_out_rsrc = ptr_rsrc(lse_out)
            ht = num_heads * seqlen_q

            lses = []
            for split in range_constexpr(num_splits):
                lse_idx = split * ht + head * seqlen_q + token
                lses.append(
                    fx.Float32(
                        buffer_ops.buffer_load(
                            lse_rsrc, lse_idx, vec_width=1, dtype=fx.Float32
                        )
                    )
                )
            lse_max = lses[0]
            for split in range_constexpr(num_splits - 1):
                lse_max = fx.Float32(
                    arith.MaxNumFOp(
                        _raw(lse_max), _raw(lses[split + 1])
                    ).result
                )
            weights = []
            denom = fx.Float32(0.0)
            for split in range_constexpr(num_splits):
                scaled = (lses[split] - lse_max) * fx.Float32(_LOG2E)
                w = fx.Float32(rocdl.exp2(T.f32, _raw(scaled)))
                weights.append(w)
                denom = denom + w
            denom = fx.Float32(
                arith.MaxNumFOp(
                    _raw(denom), _raw(fx.Float32(_DENOM_MIN))
                ).result
            )
            inv_denom = fx.Float32(rocdl.rcp(T.f32, _raw(denom)))
            weights = [w * inv_denom for w in weights]
            merged_lse = lse_max + fx.Float32(fmath.log(_raw(denom)))
            lse_out_idx = head * seqlen_q + token
            buffer_ops.buffer_store(merged_lse, lse_out_rsrc, lse_out_idx)

            row_dw = fx.Int32(row_dwords)
            for vec_i in range_constexpr(n_vec):
                acc = [fx.Float32(0.0) for _ in range(2 * _VEC)]
                dw_off = fx.Int32(vec_i * _VEC)
                for split in range_constexpr(num_splits):
                    o_dw = (
                        (fx.Int32(split) * seqlen_q + token) * num_heads + head
                    ) * row_dw + dw_off
                    raw_vec = buffer_ops.buffer_load(
                        o_rsrc, o_dw, vec_width=_VEC, dtype=fx.Int32
                    )
                    w = weights[split]
                    for lane in range_constexpr(_VEC):
                        raw_dw = fx.Uint32(fx.Vector(raw_vec)[lane])
                        lo_f32, hi_f32 = _unpack_bf16_pair(raw_dw)
                        acc[2 * lane] = acc[2 * lane] + w * lo_f32
                        acc[2 * lane + 1] = acc[2 * lane + 1] + w * hi_f32
                packed = [
                    _pack_bf16_pair(acc[2 * lane], acc[2 * lane + 1])
                    for lane in range(_VEC)
                ]
                out_vec = fx.Vector.from_elements(packed, fx.Uint32)
                out_dw = (token * num_heads + head) * row_dw + dw_off
                buffer_ops.buffer_store(out_vec, out_rsrc, out_dw)

    @flyc.jit
    def launch_combine_split_kv(
        o_parts: fx.Pointer,
        lse_parts: fx.Pointer,
        out: fx.Pointer,
        lse_out: fx.Pointer,
        seqlen_q: fx.Int32,
        num_heads: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        n_rows = seqlen_q * num_heads
        grid_x = (n_rows + fx.Int32(_BLOCK - 1)) // fx.Int32(_BLOCK)
        combine_split_kv_kernel(
            o_parts,
            lse_parts,
            out,
            lse_out,
            seqlen_q,
            num_heads,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(_BLOCK, 1, 1),
            stream=stream,
        )

    launch_combine_split_kv.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_combine_split_kv


@lru_cache(maxsize=32)
def _get_combine_launcher(num_splits: int, head_dim: int):
    return build_combine_split_kv_module(num_splits, head_dim)


def run_fused_combine(
    o_parts: torch.Tensor,
    lse_parts: torch.Tensor,
    out: torch.Tensor,
    lse_out: torch.Tensor,
) -> None:
    splits, seqlen_q, num_heads, head_dim = o_parts.shape
    exe = _get_combine_launcher(int(splits), int(head_dim))
    _run_compiled(
        exe,
        ptr_arg(o_parts),
        ptr_arg(lse_parts),
        ptr_arg(out),
        ptr_arg(lse_out),
        int(seqlen_q),
        int(num_heads),
        torch.cuda.current_stream(o_parts.device),
    )
