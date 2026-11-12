# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Buffer atomic adds with cache-policy bits that reach the instruction.

FlyDSL 0.5.0's ``rocdl.raw_ptr_buffer_atomic_fadd`` (and ``_fmax``,
``raw_ptr_buffer_load``, ``raw_ptr_buffer_load_lds``) keeps ``aux`` only when
it is a Python int or an ``ir.Attribute``; any other value, an ``fx.Int32``
included, is silently replaced by no attribute, so the bits never reach the
ISA. These helpers take the bits as a Python int and raise on anything else.

Cache-policy bits (CDNA3 gfx942, CDNA4 gfx950; LLVM CPol): SC0 = 1 (GLC),
NT = 2 (SLC), SC1 = 16 (SCC).
"""

import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl

SC0, NT, SC1 = 1, 2, 16


def cache_bits(bits):
    """``bits`` as the ``aux`` FlyDSL keeps: a Python int of SC0 | NT | SC1."""
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError(
            f"cache bits must be a Python int (SC0 | NT | SC1), not {type(bits).__name__}:"
            " rocdl.raw_ptr_buffer_* drops any other aux without an error"
        )
    if bits & ~(SC0 | NT | SC1):
        raise ValueError(f"cache bits {bits:#x} outside SC0 | NT | SC1")
    return bits


def buffer_atomic_add_pk(rsrc, elem_idx, reg_vec, elem_bytes, cache=0):
    """Pairwise buffer atomic add of a bf16/f16 vector (one packed pair per
    instruction) at element ``elem_idx``, with cache-policy bits ``cache``."""
    aux = cache_bits(cache)
    z = fx.Int32(0)
    for i in range_constexpr(reg_vec.numel // 2):
        pair = fx.Vector.from_elements(
            [reg_vec[i * 2], reg_vec[i * 2 + 1]], reg_vec.dtype
        )
        byte_off = (elem_idx + fx.Int32(i * 2)) * fx.Int32(elem_bytes)
        rocdl.raw_ptr_buffer_atomic_fadd(pair, rsrc, byte_off, z, aux)


def buffer_atomic_add_f32(rsrc, elem_idx, reg_vec, cache=0):
    """Scalar buffer atomic adds of an f32 vector, with cache-policy bits."""
    aux = cache_bits(cache)
    z = fx.Int32(0)
    for i in range_constexpr(reg_vec.numel):
        byte_off = (elem_idx + fx.Int32(i)) * fx.Int32(4)
        rocdl.raw_ptr_buffer_atomic_fadd(reg_vec[i], rsrc, byte_off, z, aux)
