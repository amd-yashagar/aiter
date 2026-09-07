# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from .config import (
    BLOCK_M,
    BLOCK_N,
    SUPPORTED_GFX,
    WAVES_PER_EU,
    validate_fmha_gfx942_tiles,
)
from .fmha_kernel import (
    build_fmha_varlen_gfx942_module,
    flash_attn_varlen_gfx942,
)
from .fmha_pingpong_kernel import (
    build_fmha_pingpong_gfx942_module,
    flash_attn_varlen_gfx942_pingpong,
)

__all__ = [
    "BLOCK_M",
    "BLOCK_N",
    "SUPPORTED_GFX",
    "WAVES_PER_EU",
    "build_fmha_pingpong_gfx942_module",
    "build_fmha_varlen_gfx942_module",
    "flash_attn_varlen_gfx942",
    "flash_attn_varlen_gfx942_pingpong",
    "validate_fmha_gfx942_tiles",
]
