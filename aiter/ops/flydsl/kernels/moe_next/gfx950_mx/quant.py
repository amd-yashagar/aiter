# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-token MX activation quant (bf16 -> fp4 or fp8, group 32) for the MX MoE path.

One thread per (token, 32-element group): read 64 B of bf16, RoundUp e8m0 from the
bf16 amax, convert with the hardware scaled conversion. Values are written token-major
([M, K / 2] fp4x2 or [M, K] fp8); scales in the lane-native order of common.py.
Prototype of core change request CR-MX-2 (to be fused into core's sort launch).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from . import common as cm

BLOCK = 256


def _build_quant(*, model_dim, a_dtype):
    K = int(model_dim)
    G = K // 32
    GP = cm.token_scale_cols(K)
    GQ = GP // 4
    row_bytes = K // 2 if a_dtype == "fp4" else K
    maxpos = cm.max_pos(a_dtype)
    assert K % 32 == 0, f"model_dim={K} must be a multiple of 32"

    @flyc.kernel
    def mx_token_quant(
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        arg_x: fx.Int64,
        i32_rows: fx.Int32,
        i32_row_stride_bytes: fx.Int32,
    ):
        gid = fx.block_idx.x * fx.Int32(BLOCK) + fx.thread_idx.x
        t = gid // fx.Int32(G)
        g = gid - t * fx.Int32(G)
        if t < i32_rows:
            src = t * i32_row_stride_bytes + g * fx.Int32(64)
            dw = []
            for i in range_constexpr(4):
                v = cm.global_vec_load(arg_x, src + fx.Int32(16 * i), 4)
                dw.extend([v[j] for j in range(4)])
            amax = cm.bf16_group_amax(dw)
            e = cm.e8m0_ceil(amax, maxpos)
            qs = (e << fx.Int32(23)).bitcast(fx.Float32)
            if const_expr(a_dtype == "fp4"):
                out = [cm.quant_bf16_fp4(dw[4 * i : 4 * i + 4], qs) for i in range(4)]
                cm.global_vec_store(
                    fx.Vector.from_elements(out, fx.Int32),
                    arg_q,
                    t * fx.Int32(row_bytes) + g * fx.Int32(16),
                    4,
                )
            else:
                out = []
                for i in range_constexpr(4):
                    out.extend(cm.quant_bf16_fp8(dw[4 * i : 4 * i + 4], qs))
                for h in range_constexpr(2):
                    cm.global_vec_store(
                        fx.Vector.from_elements(out[4 * h : 4 * h + 4], fx.Int32),
                        arg_q,
                        t * fx.Int32(row_bytes) + g * fx.Int32(32) + fx.Int32(16 * h),
                        4,
                    )
            sp = cm.global_ptr(arg_s, fx.Int8.ir_type, 1)
            off = (
                t * fx.Int32(GP) + (g & fx.Int32(3)) * fx.Int32(GQ) + (g >> fx.Int32(2))
            )
            sp[off] = fx.Int8(e)

    @flyc.jit
    def launch_mx_token_quant(
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        arg_x: fx.Int64,
        i32_rows: fx.Int32,
        i32_row_stride_bytes: fx.Int32,
        i32_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        mx_token_quant(arg_q, arg_s, arg_x, i32_rows, i32_row_stride_bytes).launch(
            grid=(fx.Int64(i32_blocks), 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    return launch_mx_token_quant


@functools.lru_cache(maxsize=64)
def compile_token_quant(*, model_dim, a_dtype):
    return _build_quant(model_dim=model_dim, a_dtype=a_dtype)


def token_quant_blocks(rows, model_dim):
    return (rows * (model_dim // 32) + BLOCK - 1) // BLOCK
