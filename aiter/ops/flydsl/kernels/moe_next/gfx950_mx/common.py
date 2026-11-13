# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared device helpers and layout arithmetic for the gfx950 MX MoE kernels.

Numerics follow aiter's default MX path (csrc/include/mx_quant_utils.h, RoundUp):
e8m0 = ceil_pow2(amax / max_pos) over a group of 32, values converted by the
hardware scaled conversion (round to nearest even).

Layouts (host and device agree on these):
- token scales ("lane-native", written by quant.py, read by stage 1): the e8m0 of
  token t, group g sits at byte t * GP + (g % 4) * (GP // 4) + g // 4, with GP the
  group count padded to 16. A lane holding K group (4 * k + j) of a 16x16x128 MFMA
  step k reads one dword per four steps.
- intermediate and weight scales: core's ``mfma_native`` order (aiter's e8m0_shuffle
  over sorted rows, ``core.intermediate.scale_offset_mfma``); stage 2 reads one dword
  per lane of a 32-row by 8-group block.
"""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import as_ir_value as _raw

FP4_MAX = 6.0
FP8_MAX = 448.0


def max_pos(dtype):
    return FP4_MAX if dtype == "fp4" else FP8_MAX


def pad_up(x, m):
    return (x + m - 1) // m * m


def token_scale_cols(k):
    """Bytes per token row of the lane-native token scales."""
    return pad_up(k // 32, 16)


def inter_scale_bytes(rows, inter):
    """Bytes of the mfma_native intermediate scales for `rows` sorted rows."""
    return pad_up(rows, 32) * pad_up(inter // 32, 8)


def global_ptr(addr_i64, elem_ty, align=4):
    pt = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    return fx.inttoptr(pt, fx.Int64(addr_i64))


def global_vec_load(base_i64, byte_off, n_i32):
    """n_i32 dwords from global base + byte_off (16 B aligned when n_i32 == 4)."""
    p = global_ptr(fx.Int64(base_i64) + fx.Int64(byte_off), T.i32, 4 * min(n_i32, 4))
    if n_i32 == 1:
        return fx.ptr_load(p, result_type=fx.Int32)
    return fx.Vector(fx.ptr_load(p, result_type=fx.Vector.make_type(n_i32, fx.Int32)))


def global_vec_store(vec, base_i64, byte_off, n_i32):
    p = global_ptr(fx.Int64(base_i64) + fx.Int64(byte_off), T.i32, 4 * min(n_i32, 4))
    fx.ptr_store(vec, p)


def global_vec_store_nt(vec, base_i64, byte_off, n_i32):
    """global_vec_store with the non-temporal hint (data nobody re-reads)."""
    from flydsl._mlir.dialects import llvm

    p = global_ptr(fx.Int64(base_i64) + fx.Int64(byte_off), T.i32, 4 * min(n_i32, 4))
    llvm.StoreOp(
        _raw(vec), fx.to_llvm_ptr(p), alignment=4 * min(n_i32, 4), nontemporal=True
    )


def e8m0_ceil(amax_f32, maxpos, cap=0xFF):
    """aiter RoundUp: biased exponent of ceil_pow2(amax / maxpos), at most cap
    (0xFF keeps inf/nan; MXFP8 activations cap at 254 = 2^127 as aiter does)."""
    u = (amax_f32 * fx.Float32(1.0 / maxpos)).bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)
    e = (u + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23))
    return (fx.Uint32(e) < fx.Uint32(cap)).select(e, fx.Int32(cap))


def umax_i32(a, b):
    return (fx.Uint32(a) > fx.Uint32(b)).select(a, b)


def bf16x2(dw):
    return fx.Vector.from_elements([dw], fx.Int32).bitcast(fx.BFloat16)


AMAX_FLOOR = 1e-10  # csrc/include/quant_amax_floor.h kAmaxFloor


def bf16_group_amax(dwords):
    """aiter's group amax of the bf16 pairs in `dwords`, as an f32: NaN skipped,
    floored at kAmaxFloor."""
    m = fx.Int32(0)
    for d in dwords:
        for h in (d & fx.Int32(0x7FFF), d.shrui(fx.Int32(16)) & fx.Int32(0x7FFF)):
            m = umax_i32(m, (h <= fx.Int32(0x7F80)).select(h, fx.Int32(0)))
    amax = (m << fx.Int32(16)).bitcast(fx.Float32)
    return amax.maximumf(fx.Float32(AMAX_FLOOR))


def quant_bf16_fp4(dwords, qscale):
    """8 bf16 (4 dwords) -> one dword of 8 fp4, scaled by 1 / qscale (RNE)."""
    pk = _raw(fx.Int32(0))
    qs = _raw(qscale)
    for j, d in enumerate(dwords):
        pk = rocdl.cvt_scalef32_pk_fp4_bf16(T.i32, pk, _raw(bf16x2(d)), qs, j)
    return fx.Int32(pk)


def quant_bf16_fp8(dwords, qscale):
    """8 bf16 (4 dwords) -> two dwords of fp8 e4m3fn, scaled by 1 / qscale (RNE)."""
    i16x2 = fx.Vector.make_type(2, fx.Int16)
    out = []
    qs = _raw(qscale)
    for p in range(2):
        z = _raw(fx.full(2, 0, fx.Int16))
        v = rocdl.cvt_scalef32_pk_fp8_bf16(i16x2, z, _raw(bf16x2(dwords[2 * p])), qs, 0)
        v = rocdl.cvt_scalef32_pk_fp8_bf16(
            i16x2, v, _raw(bf16x2(dwords[2 * p + 1])), qs, 1
        )
        out.append(fx.Vector(v).bitcast(fx.Int32)[0])
    return out


def quant_f32_fp4(vals, qscale):
    """8 f32 -> one dword of 8 fp4."""
    pk = _raw(fx.Int32(0))
    qs = _raw(qscale)
    for j in range(4):
        pk = rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32, pk, _raw(vals[2 * j]), _raw(vals[2 * j + 1]), qs, j
        )
    return fx.Int32(pk)


def quant_f32_fp8(vals, qscale):
    """8 f32 -> two dwords of fp8 e4m3fn."""
    i16x2 = fx.Vector.make_type(2, fx.Int16)
    out = []
    qs = _raw(qscale)
    for p in range(2):
        z = _raw(fx.full(2, 0, fx.Int16))
        v = rocdl.cvt_scalef32_pk_fp8_f32(
            i16x2, z, _raw(vals[4 * p]), _raw(vals[4 * p + 1]), qs, 0
        )
        v = rocdl.cvt_scalef32_pk_fp8_f32(
            i16x2, v, _raw(vals[4 * p + 2]), _raw(vals[4 * p + 3]), qs, 1
        )
        out.append(fx.Vector(v).bitcast(fx.Int32)[0])
    return out


def buffer_load16(addr_i64, base_dw, lane_dw, nt=False):
    """16 bytes per lane by buffer load: wave-uniform dword base folded into the
    descriptor, per-lane dword offset; nt sets the streaming cache policy."""
    from aiter.ops.flydsl.kernels.mxfp4_gemm_common import flat_buffer_view

    view = flat_buffer_view(
        addr_i64, _raw(base_dw), T.i32, align=16, elem_bytes=4, fold=True
    )
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if nt else 0), 32)
    frag = fx.make_rmem_tensor(4, fx.Int32)
    fx.copy(atom, view[lane_dw, None], frag)
    return fx.Vector(frag.load())


def xcd_tile(bx, by, nx, group=1):
    """(column block, tile) for a grid of nx column blocks by a multiple of 8 * group
    tiles: workgroup id w is dispatched to XCD w % 8, and every column block of tiles
    group * k .. group * k + group - 1 runs on XCD k % 8, back to back. With group 1 a
    tile's token rows are fetched into one XCD's L2; a larger group keeps an expert's
    consecutive tiles there too, so its weights are read from HBM once."""
    pid = by * fx.Int32(nx) + bx
    xc = pid & fx.Int32(7)
    q = pid >> fx.Int32(3)
    s = q % fx.Int32(group)
    n = (q // fx.Int32(group)) % fx.Int32(nx)
    m = ((q // fx.Int32(group * nx)) * fx.Int32(8) + xc) * fx.Int32(group) + s
    return n, m


def xcd_slice(bx, by, nx, nvalid, bm):
    """(column block, tile) with each XCD owning one contiguous slice of the live tiles
    (all column blocks of a tile back to back), so an expert's tiles share one L2.
    Workgroups past the slice get a tile past nvalid (they exit)."""
    pid = by * fx.Int32(nx) + bx
    xc = pid & fx.Int32(7)
    i = pid >> fx.Int32(3)
    live = (nvalid + fx.Int32(bm - 1)) // fx.Int32(bm)
    per = (live + fx.Int32(7)) >> fx.Int32(3)
    mi = i // fx.Int32(nx)
    n = i - mi * fx.Int32(nx)
    m = (mi < per).select(xc * per + mi, fx.Int32(1 << 20))
    return n, m


def xcd_rows(m_blocks):
    """Grid rows for xcd_tile: the tile count rounded up to a multiple of 8."""
    return (m_blocks + 7) // 8 * 8
