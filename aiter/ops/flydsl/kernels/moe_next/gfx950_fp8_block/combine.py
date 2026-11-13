# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Combine for stage 2's `rows_bf16` mode: out[t] = bf16(sum over k of fp32(rows[t, k])).

Stage 2 writes one bf16 row per route, token-major ([tokens, topk, N]: row = token * topk + slot), route weight
already applied, with plain stores. This kernel sums a token's topk rows in fp32 in slot order (deterministic)
and rounds once to bf16. A route the sort drops (id -1, an expert outside expert_mask, a token at or past
num_local_tokens) has no row: its load is moved past the buffer's end, which reads zeros, so its row needs no
zero-fill. Rows at or past num_local_tokens are not written, as aiter's dispatch path leaves them.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

VEC = 8  # bf16 per thread per row: one 16-byte load
THREADS = 256


def _rsrc(t, num_bytes):
    return fx.rocdl.get_buffer_rsrc(
        fx.get_iter(fx.rocdl.make_buffer_tensor(t, max_size=False, num_records_bytes=num_bytes))
    )


def _i32_view(t):
    return fx.make_view(fx.recast_iter(fx.Int32, fx.get_iter(t)), fx.make_layout(1 << 30, 1))


@functools.lru_cache(maxsize=64)
def compile_rows_reduce(*, model_dim, topk, has_mask, has_local, chunks=1, threads=THREADS):
    """Launcher(out [M, N] bf16, rows [M * topk, N] bf16, topk_ids [M, topk] int32, expert_mask int32 (or any
    tensor when has_mask is False), num_local_tokens int32 [1] (or any tensor), M, stream). chunks: 8-column groups per
    thread, strided by the block's width (more independent loads in flight per thread)."""
    N, TOPK = int(model_dim), int(topk)
    assert N % VEC == 0, f"model_dim={N} must be a multiple of {VEC}"
    CH, TH = int(chunks), int(threads)
    GX = -(-N // (TH * VEC * CH))

    @flyc.kernel(known_block_size=[TH, 1, 1])
    def fp8b_rows_reduce(
        arg_out: fx.Tensor,
        arg_rows: fx.Tensor,
        arg_ids: fx.Tensor,
        arg_mask: fx.Tensor,
        arg_local: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        t = gpu.block_idx.y
        col0 = (gpu.block_idx.x * fx.Int32(TH * CH) + tid) * fx.Int32(VEC)
        cols = [col0 + fx.Int32(c * TH * VEC) for c in range(CH)]
        tokens = i32_tokens
        rows_dw = tokens * fx.Int32(TOPK * N // 2)  # the rows buffer's end, in dwords
        rows_rsrc = _rsrc(arg_rows, fx.Int64(tokens) * fx.Int64(TOPK * N * 2))
        out_rsrc = _rsrc(arg_out, fx.Int64(tokens) * fx.Int64(N * 2))
        ids = _i32_view(arg_ids)
        live = t < tokens
        if const_expr(has_local):
            live = live & (t < _i32_view(arg_local)[0])
        acc = [[fx.Float32(0.0) for _ in range(VEC)] for _ in range(CH)]
        for k in range_constexpr(TOPK):
            e = ids[t * fx.Int32(TOPK) + fx.Int32(k)]
            ok = live & (e >= fx.Int32(0))
            if const_expr(has_mask):
                ok = ok & (_i32_view(arg_mask)[(e >= fx.Int32(0)).select(e, fx.Int32(0))] != fx.Int32(0))
            for c in range_constexpr(CH):
                off = ((t * fx.Int32(TOPK) + fx.Int32(k)) * fx.Int32(N) + cols[c]) // fx.Int32(2)
                raw = buffer_ops.buffer_load(rows_rsrc, ok.select(off, rows_dw), vec_width=VEC // 2, dtype=T.i32)
                v = Vec(raw).bitcast(fx.BFloat16)
                acc[c] = [acc[c][i] + v[i].to(fx.Float32) for i in range(VEC)]
        for c in range_constexpr(CH):
            packed = Vec.from_elements(acc[c], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
            # rows at or past num_local_tokens stay untouched, as aiter's dispatch leaves them
            if live & (cols[c] < fx.Int32(N)):
                buffer_ops.buffer_store(packed, out_rsrc, (t * fx.Int32(N) + cols[c]) // fx.Int32(2))

    @flyc.jit
    def launch_fp8b_rows_reduce(
        arg_out: fx.Tensor,
        arg_rows: fx.Tensor,
        arg_ids: fx.Tensor,
        arg_mask: fx.Tensor,
        arg_local: fx.Tensor,
        i32_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        fp8b_rows_reduce(arg_out, arg_rows, arg_ids, arg_mask, arg_local, i32_tokens).launch(
            grid=(GX, fx.Int64(i32_tokens), 1), block=(TH, 1, 1), stream=stream
        )

    return launch_fp8b_rows_reduce
