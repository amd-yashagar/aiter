# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inline addressing for the sorted-row intermediate (see moe_next/core/intermediate.py).

Element offsets into the values and scales tensors; ``row_elems``,
``groups`` and ``gp8`` are the layout's compile-time constants.
"""

import flydsl.expr as fx


def value_offset(row, col, row_elems):
    """Offset of value (row, col); for fp4x2 ``col`` counts bytes."""
    return row * fx.Int32(row_elems) + col


def scale_offset(row, group, groups):
    """Offset of the scale of column group ``group`` in sorted row ``row``."""
    return row * fx.Int32(groups) + group


def mfma_native_offset(row, group, gp8, c=int):
    """Byte of the e8m0 scale (row, group) in the ``mfma_native`` order,
    aiter's mx_scale_shuffle_idx / e8m0_shuffle: 32-row by 8-group blocks of
    256 bytes. ``gp8`` is the group count padded to 8. Plain arithmetic, so
    kernels (``c`` = fx.Int32) and host code (``c`` = int, or tensors) share it."""
    return (
        (row // c(32)) * c(gp8 * 32)
        + (group // c(8)) * c(256)
        + (group % c(4)) * c(64)
        + (row % c(16)) * c(4)
        + ((group % c(8)) // c(4)) * c(2)
        + (row % c(32)) // c(16)
    )


def scale_offset_mfma(row, group, gp8):
    """``mfma_native`` scale byte of (row, group) in a kernel."""
    return mfma_native_offset(row, group, gp8, fx.Int32)


def tile_rows(tile, tile_m):
    """First sorted row of tile ``tile``."""
    return tile * fx.Int32(tile_m)
