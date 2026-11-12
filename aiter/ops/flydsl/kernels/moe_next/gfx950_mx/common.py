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
- intermediate scales ("mfma_native", written by stage 1, read by stage 2): aiter's
  e8m0_shuffle order over sorted rows, byte (r, g) at
  (r // 32) * GP8 * 32 + (g // 8) * 256 + (g % 4) * 64 + (r % 16) * 4
  + ((g % 8) // 4) * 2 + (r % 32) // 16, GP8 = groups padded to 8.
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


def e8m0_ceil(amax_f32, maxpos):
    """aiter RoundUp: biased exponent of ceil_pow2(amax / maxpos), 0xFF kept for inf/nan."""
    u = (amax_f32 * fx.Float32(1.0 / maxpos)).bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)
    e = (u + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23))
    return (fx.Uint32(e) < fx.Uint32(0xFF)).select(e, fx.Int32(0xFF))


def umax_i32(a, b):
    return (fx.Uint32(a) > fx.Uint32(b)).select(a, b)


def bf16x2(dw):
    return fx.Vector.from_elements([dw], fx.Int32).bitcast(fx.BFloat16)


def bf16_group_amax(dwords):
    """|max| of the bf16 pairs in `dwords` as an f32 (bf16 bits in the high half)."""
    m = dwords[0] & fx.Int32(0x7FFF7FFF)
    for d in dwords[1:]:
        m = _pkmax_u16(m, d & fx.Int32(0x7FFF7FFF))
    lo = m & fx.Int32(0xFFFF)
    hi = m.shrui(fx.Int32(16))
    return (umax_i32(lo, hi) << fx.Int32(16)).bitcast(fx.Float32)


def _pkmax_u16(a, b):
    va = fx.Vector.from_elements([a], fx.Int32).bitcast(fx.Uint16)
    vb = fx.Vector.from_elements([b], fx.Int32).bitcast(fx.Uint16)
    return fx.max(va, vb).bitcast(fx.Int32)[0]


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
