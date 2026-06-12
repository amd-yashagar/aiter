# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unified config schema for the jagged_dense_bmm (jdbba) v3 variant framework.

One ``JdbbaConfig`` describes a fully-specified kernel instance across *all*
registered variant types. Each variant reads only the knobs it understands
(``variant.relevant_keys``); the rest are carried inertly so a single table /
autotune record can describe any variant without a schema change.

The op (HSTU grouped jagged x dense BMM + broadcast bias), in bench naming:

    Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]   per group b
      (M_b, N)     (M_b, K)        (K, N)       (1, N)

with reduction dim ``K`` (bench "D"), output dim ``N`` (bench "Kout"), and the
per-group row envelope ``max_seq_len`` (bench "Mi"). Dense is supplied in the
genrec ``(B, K, N)`` convention; variants transpose as needed.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class JdbbaConfig:
    """A fully-specified, variant-agnostic kernel instance.

    Knob applicability by variant (see each variant's ``relevant_keys``):
      - lowlevel : tile_m/n/k, stages, block_{m,n,k}_warps, b_to_lds,
                   waves_per_eu, xcd_c
      - highlevel: tile_m, tile_n, xcd_c, xcd_w, use_mfma_k32, waves_per_eu,
                   uniform_seqlen

    ``None`` means "let the variant/kernel pick its own default" for that knob.
    """

    variant: str = "lowlevel"

    # --- tiling ---
    tile_m: Optional[int] = None
    tile_n: Optional[int] = None
    tile_k: Optional[int] = None
    stages: int = 2

    # --- warp layout (lowlevel) ---
    block_m_warps: int = 2
    block_n_warps: int = 2
    block_k_warps: int = 1

    # --- operand staging (lowlevel: stage B through LDS vs registers) ---
    b_to_lds: bool = False

    # --- occupancy ---
    waves_per_eu: int = 0

    # --- chiplet (XCD) L2-aware grid remap ---
    # xcd_c = group/tile chunk per XCD (0 = remap off). xcd_w = window height
    # (highlevel 2-phase remap only).
    xcd_c: int = 0
    xcd_w: Optional[int] = None

    # --- MFMA atom ---
    use_mfma_k32: Optional[bool] = None  # None -> arch auto (gfx950 True)

    # --- regime hint (highlevel remap gating); None -> caller's runtime flag ---
    uniform_seqlen: Optional[bool] = None

    def merged(self, **overrides: Any) -> "JdbbaConfig":
        """Return a copy with non-None overrides applied."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return dataclasses.replace(self, **clean)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JdbbaConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


def shape_id(*, n_groups: int, reduction_k: int, output_n: int, max_seq_len: int) -> str:
    """Per-shape dispatch key. ``reduction_k`` is bench-D, ``output_n`` is bench-K."""
    return f"B{n_groups}D{reduction_k}K{output_n}N{max_seq_len}"


def d_bucket(reduction_k: int) -> str:
    if reduction_k <= 256:
        return "d_le_256"
    if reduction_k <= 512:
        return "d_le_512"
    if reduction_k <= 1024:
        return "d_le_1024"
    return "d_big"
