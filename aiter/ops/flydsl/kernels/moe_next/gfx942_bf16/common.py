# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared device helpers for the gfx942 bf16 MoE kernels.

MFMA operand mapping used by both GEMMs (v_mfma_f32_16x16x16_bf16): A is a
16-row block of aiter's shuffle_weight(w, (16, 16)) bf16 layout, where row c's
16-byte chunk kb (bf16 K values 8kb .. 8kb + 7) sits at
nblk * 32K + kb * 256 + c * 16 bytes (K = the weight's row length). A K step of
64 values is chunks 8g .. 8g + 7; lane l (c = l % 16, q = l // 16) loads chunks
8g + q and 8g + 4 + q, and MFMA j of the step (j = 2h + i2) takes elements
4 i2 .. 4 i2 + 3 of chunk 8g + 4h + q, so its K slot 4q + i is K value
64g + 32h + 8q + 4 i2 + i. B (tokens) uses the same mapping: lane (t = l % 16, q)
holds bf16 elements 32h + 8q .. 32h + 8q + 7 of token t's 64-wide step. C lane l
holds rows (channels) 4q .. 4q + 3 of column (token) l % 16.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import rocdl as mlir_rocdl
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import common as fxh

OOB = 0x7FFFFFF0  # an out-of-range buffer offset: the load returns 0


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


def lds_barrier():
    """Workgroup barrier that orders LDS only. ``gpu.barrier()`` also waits for every
    global load in flight (vmcnt(0)), which drains the weight prefetch each K chunk;
    loads whose registers are read later still get their own vmcnt wait."""
    fx.memory_fence(ordering=fx.AtomicOrdering.SeqCst, syncscope="singlethread")
    rocdl.s_waitcnt(lgkmcnt=0)
    mlir_rocdl.s_barrier()
    fx.memory_fence(ordering=fx.AtomicOrdering.SeqCst, syncscope="singlethread")


def lds_ptr16(base, dword_off, align=16):
    """An ``align``-byte-aligned int32 pointer ``dword_off`` dwords past an LDS array."""
    p = fx.recast_iter(fx.Int32, base) + dword_off
    return fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, p.address_space, align), p
    )


def lds_load_x4(base, dword_off):
    return fx.generic_load(lds_ptr16(base, dword_off), dtype=fx.Int32, count=4)


def lds_store_x4(base, dword_off, value):
    fx.generic_store(lds_ptr16(base, dword_off), value)


def lds_store_x2(base, dword_off, value):
    fx.generic_store(lds_ptr16(base, dword_off, 8), value)


def half(words, i2):
    """Dwords 2 i2 and 2 i2 + 1 of a 16-byte chunk as an MFMA operand of four bf16."""
    return Vec.from_elements([words[2 * i2], words[2 * i2 + 1]], fx.Int32).bitcast(
        fx.Int16
    )


def mfma(a, b, acc):
    return Vec(rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a, b, acc, 0, 0, 0]))


def make_act(activation):
    """aiter's CK a16w16 activations in FP32 (element_wise::Silu and ::Gelu):
    silu(g) = g * (1 / (1 + exp(-g))), gelu(g) = 0.5 g (1 + erf(0.70710678118 g))."""

    def act(g):
        if const_expr(activation == "gelu"):
            return 0.5 * g * (1.0 + fx.math.erf(g * 0.70710678118))
        return g * (fx.Float32(1.0) / (1.0 + fx.exp(-g)))

    return act


def bf16_bits(v):
    """fp32 -> bf16 bits in the low 16 bits of a dword: round to nearest even, NaN
    to 0x7FFF, as aiter's CK conversion (ck_tile float_to_bf16_rtn_asm). Integer
    arithmetic only: fxh._f32_to_bf16's inline-asm form returns zeros on gfx942
    when it is fed MFMA accumulator elements directly."""
    bits = v.bitcast(fx.Uint32)
    rounded = (bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))) >> 16
    nan = (bits & fx.Uint32(0x7FFFFFFF)) > fx.Uint32(0x7F800000)
    return nan.select(fx.Uint32(0x7FFF), rounded)


def bf16x2_word(lo, hi):
    """Two fp32 values as one dword of two bf16 (``lo`` in the low half)."""
    return (bf16_bits(lo) | (bf16_bits(hi) << 16)).bitcast(fx.Int32)


def bf16x4_words(vals):
    """Four fp32 values rounded to bf16 (as bf16_bits) and packed in two dwords."""
    return Vec.from_elements(
        [bf16x2_word(vals[0], vals[1]), bf16x2_word(vals[2], vals[3])], fx.Int32
    )
