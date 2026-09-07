# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile-time tile and LDS budget for gfx942 packed-varlen FMHA.

``BLOCK_M`` / ``BLOCK_N`` / ``WAVES_PER_EU`` here are fallback defaults. Host
wrappers forward overrides; do not silently pin them one layer down.

K/V LDS match the native fused layouts:
* K: padded 64x32 planes so ``buffer_load_dword … lds`` can deposit through
  the hardware ``m0 + lane*4`` fan-out.
* V: 8x8 tiles padded to 8x9 after the register ``v_perm`` transpose.
"""

from __future__ import annotations

# Fallback defaults. Override via the host ``block_m`` / ``block_n`` /
# ``waves_per_eu`` knobs (see ``flash_attn_varlen_gfx942``).
BLOCK_M = 128
BLOCK_N = 64
WAVES_PER_EU = 2  # staged-K path measures 232 combined VGPR/AGPR and 27 KiB LDS
WARP_SIZE = 64
BLOCK_SIZE = 256  # 4 waves; matches fallback BLOCK_M=128 (32 rows/wave)
VEC_WIDTH = 16
K_SUB_N = 32  # MFMA32 splits each BLOCK_N=64 tile into lo/hi K halves
D_CHUNK = 32
MFMA_K = 8  # gfx942 CDNA3 32x32x8
# One padded 64x32 K plane (ASM / op_lds.hpp kSingleSmemElements).
K_PLANE_ELEMS = 2304
K_STAGE_COUNT = 2
# Per-wave / per-issue M0 steps for buffer_load_dword…lds (bytes).
K_DMA_WARP_STRIDE_BYTES = 0x110  # 272
K_DMA_ISSUE_STRIDE_BYTES = 0x440  # 1088
LDS_LIMIT_BYTES = 65536
SUPPORTED_GFX = ("gfx942",)
SUPPORTED_DTYPES = ("bf16", "f16")


def k_lds_elems(head_dim_qk: int) -> int:
    """Elements for two ping-pong padded 64x32 K staging planes."""
    if head_dim_qk % 32 != 0:
        raise ValueError(f"head_dim_qk must be a multiple of 32, got {head_dim_qk}")
    return K_PLANE_ELEMS * K_STAGE_COUNT


def v_lds_elems(head_dim_v: int, block_n: int) -> int:
    """Elements for native V LDS: padded 8x8 tiles (72 elements each)."""
    if block_n % 32 != 0 or head_dim_v % 8 != 0:
        raise ValueError(
            f"V LDS requires block_n%32==0 and head_dim_v%8==0, "
            f"got block_n={block_n} head_dim_v={head_dim_v}"
        )
    return (block_n // 8) * (head_dim_v // 8) * 72


def lds_elems(head_dim_qk: int, head_dim_v: int, block_n: int) -> int:
    return k_lds_elems(head_dim_qk) + v_lds_elems(head_dim_v, block_n)


def lds_bytes(head_dim_qk: int, head_dim_v: int, block_n: int) -> int:
    return lds_elems(head_dim_qk, head_dim_v, block_n) * 2


def lds_elem_offset(j: int, d: int) -> int:
    """ASM/native K LDS element offset (bf16). ``j`` = seqlen_k row, ``d`` = headdim."""
    return (
        (j % 4) * 136
        + ((j // 4) % 4) * 32
        + (j // 16) * 544
        + (d % 32)
        + (d // 32) * K_PLANE_ELEMS
    )


def swizzle_a(x: int) -> int:
    """Swap bits 2 and 3 (ASM SwizzleA on the K N-column index)."""
    b2 = (x >> 2) & 1
    b3 = (x >> 3) & 1
    return (x & ~0xC) | (b2 << 3) | (b3 << 2)


def validate_fmha_gfx942_tiles(
    head_dim_qk: int,
    head_dim_v: int,
    *,
    block_m: int = BLOCK_M,
    block_n: int = BLOCK_N,
    block_size: int = BLOCK_SIZE,
    vec_width: int = VEC_WIDTH,
) -> None:
    """Reject configs that cannot map onto the DMA K + linearized V coop-load."""
    if block_m % (block_size // WARP_SIZE) != 0:
        raise ValueError(
            f"block_m ({block_m}) must be divisible by num_waves "
            f"({block_size // WARP_SIZE})"
        )
    if block_n != BLOCK_N:
        raise ValueError(
            f"DMA K path requires block_n={BLOCK_N} (got {block_n})"
        )
    if block_n % K_SUB_N != 0:
        raise ValueError(f"block_n ({block_n}) must be a multiple of {K_SUB_N}")
    if head_dim_qk % 32 != 0 or head_dim_v % 64 != 0:
        raise ValueError(
            f"QK must be a multiple of 32 and V a multiple of 64, "
            f"got QK={head_dim_qk} V={head_dim_v}"
        )
    if head_dim_qk % vec_width != 0 or head_dim_v % vec_width != 0:
        raise ValueError(
            f"head dims must be multiples of vec_width={vec_width}, "
            f"got QK={head_dim_qk} V={head_dim_v}"
        )
    elems_per_pass = block_size * vec_width
    if (block_n * head_dim_v) % elems_per_pass != 0:
        raise ValueError(
            f"BLOCK_N*V ({block_n}*{head_dim_v}) must divide "
            f"{elems_per_pass} linearized load elems/pass"
        )
    nbytes = lds_bytes(head_dim_qk, head_dim_v, block_n)
    if nbytes > LDS_LIMIT_BYTES:
        raise ValueError(
            f"LDS {nbytes} bytes exceeds gfx942 {LDS_LIMIT_BYTES} "
            f"(QK={head_dim_qk} V={head_dim_v} block_n={block_n})"
        )
