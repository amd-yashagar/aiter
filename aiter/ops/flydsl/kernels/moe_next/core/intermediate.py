# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inline addressing for the sorted-row intermediate (see moe_next/core/intermediate.py).

Element offsets into the values and scales tensors of a ``row_major`` layout;
``row_elems`` and ``groups`` are the layout's compile-time constants.
"""

import flydsl.expr as fx


def value_offset(row, col, row_elems):
    """Offset of value (row, col); for fp4x2 ``col`` counts bytes."""
    return row * fx.Int32(row_elems) + col


def scale_offset(row, group, groups):
    """Offset of the scale of column group ``group`` in sorted row ``row``."""
    return row * fx.Int32(groups) + group


def tile_rows(tile, tile_m):
    """First sorted row of tile ``tile``."""
    return tile * fx.Int32(tile_m)
