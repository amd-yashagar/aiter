# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared device helpers for the mid-band kernels.

MFMA operand mapping used by both GEMMs (v_mfma_f32_16x16x16_bf16): A is a
16-channel block of aiter's shuffle_weight(w, (16, 16)) layout, where channel c's
16-byte chunk for K block kb sits at nblk * 16K + kb * 256 + c * 16. Lane l
(c = l % 16, q = l // 16) loads the chunk of K step g (64 K values) at
kb = 4g + q, and MFMA j of that step uses bytes 4j..4j+3, so its K slot
4q + i is K value 64g + 16q + 4j + i. B (tokens) uses the same mapping: lane
(t = l % 16, q) holds bf16 elements 16q + 4j .. 16q + 4j + 3 of token t's 64-wide
step. C lane l holds rows (channels) 4q .. 4q + 3 of column (token) l % 16.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import common as fxh


def rsrc(ptr, nbytes=None):
    """Buffer descriptor at a typed pointer; ``nbytes`` bounds it (out-of-range
    loads return 0), otherwise it spans the maximum size."""
    view = fx.make_view(ptr, fx.make_layout(1, 1))
    if nbytes is None:
        t = fx.rocdl.make_buffer_tensor(view, max_size=True)
    else:
        t = fx.rocdl.make_buffer_tensor(
            view, max_size=False, num_records_bytes=fx.Int64(nbytes)
        )
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(t))


def load_x4(rsrc_, byte_off, cache=0):
    """Four dwords (16 bytes) at a byte offset."""
    v = buffer_ops.buffer_load(
        rsrc_, byte_off >> 2, vec_width=4, dtype=T.i32, cache_modifier=cache
    )
    return Vec(v)


def load_s1(p, idx):
    """One wave-uniform int32 through the scalar cache."""
    r = rsrc(fxh._as_ptr(p, fx.Int32))
    return fx.Int32(buffer_ops.buffer_load(r, idx, vec_width=1, is_scalar=True))


def fp8_dword_to_bf16x4(word):
    """Four fp8 e4m3fnuz bytes as an MFMA operand of four bf16 (exact: fp8 fits
    bf16, so the upper half of the fp32 conversion is the bf16 value)."""
    w = _raw(word)
    lo = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, w, word_sel=False))
    hi = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, w, word_sel=True))
    # Bytes 2-3 of a (low half) and of b (high half) in one v_perm_b32.
    sel = fx.Int32(0x07060302)

    def pack(a, b):
        return fx.Int32(rocdl.perm_b32(b.bitcast(fx.Int32), a.bitcast(fx.Int32), sel))

    d0 = pack(lo[0], lo[1])
    d1 = pack(hi[0], hi[1])
    return Vec.from_elements([d0, d1], fx.Int32).bitcast(fx.Int16)


def lds_ptr16(base, dword_off):
    """A 16-byte-aligned int32 pointer ``dword_off`` dwords past an LDS array."""
    p = fx.recast_iter(fx.Int32, base) + dword_off
    return fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, p.address_space, 16), p)


def lds_load_x4(base, dword_off):
    return fx.generic_load(lds_ptr16(base, dword_off), dtype=fx.Int32, count=4)


def lds_store_x4(base, dword_off, value):
    fx.generic_store(lds_ptr16(base, dword_off), value)


def bf16_pair(words, j):
    """Dwords 2j and 2j+1 of eight as an MFMA operand of four bf16."""
    return Vec.from_elements([words[2 * j], words[2 * j + 1]], fx.Int32).bitcast(
        fx.Int16
    )


def mfma(a, b, acc):
    return Vec(rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a, b, acc, 0, 0, 0]))


def make_act(activation, swiglu_limit):
    """SiLU as aiter's silu_kernel: x * rcp(1 + exp(-x)); Swiglu as aiter's gfx9
    FlyDSL MoE: clamped gate * sigmoid(1.702 gate) * (clamped up + 1)."""
    limit = float(swiglu_limit) if swiglu_limit else 7.0

    def act(g, u):
        if const_expr(activation == "swiglu"):
            neg = fx.Float32(-limit)
            g = -((-g).maximumf(neg))
            u = (-((-u).maximumf(neg))).maximumf(neg)
            t = rocdl.exp2(T.f32, _raw(g * (-1.702 * 1.4426950408889634)))
            return (g * rocdl.rcp(T.f32, _raw(1.0 + t))) * (u + 1.0)
        return (g * rocdl.rcp(T.f32, _raw(1.0 + fx.exp(-g)))) * u

    return act


def bf16x4_words(vals):
    """Four fp32 values rounded to bf16 (RNE, as aiter) and packed in two dwords."""
    pair0 = fxh._f32_to_bf16(Vec.from_elements([vals[0], vals[1]], fx.Float32))
    pair1 = fxh._f32_to_bf16(Vec.from_elements([vals[2], vals[3]], fx.Float32))
    return Vec.from_elements(
        [pair0.bitcast(fx.Int32)[0], pair1.bitcast(fx.Int32)[0]], fx.Int32
    )
