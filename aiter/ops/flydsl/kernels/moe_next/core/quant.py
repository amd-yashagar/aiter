# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Activation quant for the routed tokens, bit-exact with aiter's own quant ops.

fp8 per row (per_token_fp8: a row is a token, as aiter's
dynamic_per_token_scaled_quant; per_1x128_fp8: a row is 128 columns of a
token, as aiter's dynamic_per_group_scaled_quant):
    amax  = max |x| over the row (fp32, from 0 per token, from kAmaxFloor =
            1e-10 per group)
    scale = amax * float(1 / fp8_max)                       -> scales (fp32)
    q     = cvt_pk_fp8_f32(med3(x * rcp(scale), -fp8_max, fp8_max))
fp8_max is the arch's (240 for gfx942 e4m3fnuz, 448 for gfx950 e4m3fn).

The helpers take a workgroup-relative block index, so a quant pass can run as
standalone kernels or in extra workgroups of the sort launch.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

BLOCK = 256
WAVE = 64
GROUP_LANES = 16  # lanes per 128-column group (8 bf16 each)
OOB = 0x7FFFFFFF
AMAX_FLOOR = 1e-10  # aiter::kAmaxFloor (csrc/include/quant_amax_floor.h)


def _raw(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _it(t):
    return fx.get_iter(fx.rocdl.make_buffer_tensor(t, max_size=True))


def _bf16x8_f32(v):
    """8 bf16 (4 dwords) -> 8 f32 (exact)."""
    out = []
    for d in v:
        out.append((d << fx.Int32(16)).bitcast(fx.Float32))
        out.append((d & fx.Int32(-65536)).bitcast(fx.Float32))
    return out


def _absmax(vals, acc):
    for x in vals:
        a = (x.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)).bitcast(fx.Float32)
        acc = (a > acc).select(a, acc)
    return acc


def _fp8x4(vals, inv, fp8_max):
    """4 f32 -> one dword of 4 fp8, as aiter's fp32_to_fp8_scaled_x4."""
    hi, lo = _raw(fx.Float32(fp8_max)), _raw(fx.Float32(-fp8_max))
    clamped = [fx.Float32(rocdl.fmed3(T.f32, _raw(x * inv), lo, hi)) for x in vals]
    pk = rocdl.cvt_pk_fp8_f32(
        T.i32, _raw(clamped[0]), _raw(clamped[1]), _raw(fx.Int32(0)), 0
    )
    pk = rocdl.cvt_pk_fp8_f32(T.i32, _raw(clamped[2]), _raw(clamped[3]), pk, 1)
    return fx.Int32(pk)


def _load8(x_it, elem):
    """8 bf16 at element offset ``elem`` (16-byte aligned) as 4 dwords; ``x_it``
    iterates the bf16 input viewed as int32."""
    v = fx.Vector(
        fx.ptr_load(
            x_it + fx.Int64(elem >> fx.Int32(1)),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    return [v[j] for j in range(4)]


def _fshuffle_xor(x, off, width):
    return (
        x.bitcast(fx.Int32)
        .shuffle_xor(fx.Int32(off), fx.Int32(width))
        .bitcast(fx.Float32)
    )


@flyc.jit
def quant_1x128_block(x_it, q_it, s_it, blk, n_groups, fp8_max):
    """16 groups of 128 columns per workgroup, from group ``blk * 16``; 16
    lanes per group, 8 columns per lane. Groups at or past n_groups are skipped
    (rows past the token count included)."""
    tid = gpu.thread_idx.x
    lane16 = tid % fx.Int32(GROUP_LANES)
    grp = blk * fx.Int32(BLOCK // GROUP_LANES) + tid // fx.Int32(GROUP_LANES)
    live = grp < n_groups
    col = grp * fx.Int32(128) + lane16 * fx.Int32(8)
    x = _bf16x8_f32(_load8(x_it, live.select(col, fx.Int32(0))))
    amax = _absmax(x, fx.Float32(AMAX_FLOOR))
    for sh in range_constexpr(4):
        other = _fshuffle_xor(amax, 1 << sh, GROUP_LANES)
        amax = (other > amax).select(other, amax)
    scale = amax * fx.Float32(1.0 / fp8_max)
    inv = fx.Float32(rocdl.rcp(T.f32, _raw(scale)))
    d0 = _fp8x4(x[0:4], inv, fp8_max)
    d1 = _fp8x4(x[4:8], inv, fp8_max)
    q = live.select(col >> fx.Int32(2), fx.Int32(OOB))
    fx.ptr_store(d0, q_it + fx.Int64(q))
    fx.ptr_store(d1, q_it + fx.Int64(live.select(q + fx.Int32(1), fx.Int32(OOB))))
    s = (live & (lane16 == fx.Int32(0))).select(grp, fx.Int32(OOB))
    fx.ptr_store(scale, s_it + fx.Int64(s))


@flyc.jit
def quant_token_block(x_it, q_it, s_it, red, row, n_rows, cols, fp8_max):
    """One token row per workgroup: 8 columns per thread per pass, max over the
    row through wave shuffles and ``red`` (WAVES floats of LDS)."""
    tid = gpu.thread_idx.x
    lane, wave = tid % fx.Int32(WAVE), tid // fx.Int32(WAVE)
    live = row < n_rows
    passes = -(-cols // (BLOCK * 8))
    vals, cols_at = [], []
    amax = fx.Float32(0.0)
    for p in range_constexpr(passes):
        c = fx.Int32(p * BLOCK * 8) + tid * fx.Int32(8)
        ok = live & (c < fx.Int32(cols))
        elem = row * fx.Int32(cols) + c
        x = _bf16x8_f32(_load8(x_it, ok.select(elem, fx.Int32(0))))
        x = [ok.select(v, fx.Float32(0.0)) for v in x]
        amax = _absmax(x, amax)
        vals.append(x)
        cols_at.append((ok, elem))
    for sh in range_constexpr(6):
        other = _fshuffle_xor(amax, 1 << sh, WAVE)
        amax = (other > amax).select(other, amax)
    gpu.barrier()
    fx.ptr_store(amax, red + fx.Int64((lane == fx.Int32(0)).select(wave, fx.Int32(4))))
    gpu.barrier()
    amax = fx.ptr_load(red)
    for w in range_constexpr(1, BLOCK // WAVE):
        other = fx.ptr_load(red + fx.Int64(w))
        amax = (other > amax).select(other, amax)
    scale = amax * fx.Float32(1.0 / fp8_max)
    inv = fx.Float32(rocdl.rcp(T.f32, _raw(scale)))
    for p in range_constexpr(passes):
        ok, elem = cols_at[p]
        d0 = _fp8x4(vals[p][0:4], inv, fp8_max)
        d1 = _fp8x4(vals[p][4:8], inv, fp8_max)
        q = ok.select(elem >> fx.Int32(2), fx.Int32(OOB))
        fx.ptr_store(d0, q_it + fx.Int64(q))
        fx.ptr_store(d1, q_it + fx.Int64(ok.select(q + fx.Int32(1), fx.Int32(OOB))))
    s = (live & (tid == fx.Int32(0))).select(row, fx.Int32(OOB))
    fx.ptr_store(scale, s_it + fx.Int64(s))


@functools.lru_cache(maxsize=64)
def build_standalone(mode, cols, fp8_max, has_local):
    """A standalone launcher for one mode (tests, and calls without a sort)."""

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, 8, 16]

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def quant_kernel(
        x: fx.Tensor, q: fx.Tensor, s: fx.Tensor, ltok: fx.Tensor, m_cap: fx.Int32
    ):
        bid = gpu.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        rows = m_cap
        if has_local:
            t = fx.ptr_load(_it(ltok) + fx.Int64(0))
            rows = (t < m_cap).select(t, m_cap)
        if mode == "per_1x128_fp8":
            quant_1x128_block(
                _it(x), _it(q), _it(s), bid, rows * fx.Int32(cols // 128), fp8_max
            )
        else:
            quant_token_block(
                _it(x), _it(q), _it(s), lds.red.ptr, bid, rows, cols, fp8_max
            )

    @flyc.jit
    def launch(
        x: fx.Tensor,
        q: fx.Tensor,
        s: fx.Tensor,
        ltok: fx.Tensor,
        m_cap: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = None,
    ):
        quant_kernel(x, q, s, ltok, m_cap).launch(
            grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    return launch


def blocks(mode, m, cols):
    """Workgroups a quant pass needs for ``m`` token rows."""
    if mode == "per_1x128_fp8":
        return -(-(m * (cols // 128)) // (BLOCK // GROUP_LANES))
    return m
