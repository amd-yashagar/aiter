# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The one stage1 -> stage2 intermediate layout, addressed by sorted row.

Row r of the intermediate is the route at sorted position r of the tile map
(``sorted_token_ids[r]``), so stage 1 writes each tile's rows contiguously and
stage 2 reads them with no gather. Rows past a tile's real rows (padding) hold
unspecified data: a kernel must drop their results by the padding sentinel in
``sorted_token_ids``, never by multiplying with their zero weight (0 * NaN).
"""

from dataclasses import dataclass

import torch

from aiter import QuantType, dtypes

from .tile_map import TileMap, max_sorted_rows

SCALE_ORDERS = ("row_major", "mfma_native")


@dataclass(frozen=True)
class Layout:
    """Shapes and addressing of one intermediate.

    ``dtype`` is the value dtype (bf16, the arch's fp8, or fp4x2 with two
    values per byte). ``group`` is the scale group along the columns: an int
    (128 blockscale, 32 MX), ``"row"`` (one scale per row) or ``"none"``.
    ``scale_order`` ``row_major`` stores scales ``[rows, groups]``;
    ``mfma_native`` (MX e8m0 in scaled-MFMA operand order) is specified in
    docs/aiter-moe-api.md and not implemented yet. ``row_pad`` rounds the row
    count up; ``col_pad`` rounds each value row (in elements) up.
    """

    rows: int
    cols: int
    dtype: torch.dtype
    group: int | str
    scale_dtype: torch.dtype | None
    scale_order: str = "row_major"
    row_pad: int = 1
    col_pad: int = 1

    def __post_init__(self):
        if self.scale_order not in SCALE_ORDERS:
            raise ValueError(f"scale_order must be one of {SCALE_ORDERS}")
        if self.scale_order == "mfma_native":
            raise NotImplementedError("mfma_native scale order is not implemented yet")
        if isinstance(self.group, int) and self.cols % self.group:
            raise ValueError(
                f"cols {self.cols} is not a multiple of group {self.group}"
            )

    @property
    def padded_rows(self) -> int:
        return -(-self.rows // self.row_pad) * self.row_pad

    @property
    def row_elems(self) -> int:
        """Value elements per row, padded."""
        return -(-self.cols // self.col_pad) * self.col_pad

    @property
    def values_shape(self) -> tuple[int, int]:
        per_byte = 2 if self.dtype == dtypes.fp4x2 else 1
        return (self.padded_rows, self.row_elems // per_byte)

    @property
    def groups(self) -> int:
        if self.group == "none":
            return 0
        if self.group == "row":
            return 1
        return self.cols // self.group

    @property
    def scales_shape(self) -> tuple[int, int] | None:
        if self.group == "none":
            return None
        return (self.padded_rows, self.groups)

    def value_offset(self, row: int, col: int) -> int:
        """Element offset of (row, col) in the values tensor's storage."""
        return row * self.values_shape[1] + col

    def scale_offset(self, row: int, group: int) -> int:
        return row * self.groups + group

    def alloc(self, device) -> "Intermediate":
        values = torch.empty(self.values_shape, dtype=self.dtype, device=device)
        shape = self.scales_shape
        scales = (
            torch.empty(shape, dtype=self.scale_dtype, device=device) if shape else None
        )
        return Intermediate(self, values, scales)

    @classmethod
    def for_spec(cls, spec, m: int, topk: int, tile_m: int) -> "Layout":
        """The layout aiter's path quantizes the intermediate to, for this
        layer (same dtype, group and quantization points)."""
        rows = max_sorted_rows(m, topk, spec.global_experts, tile_m)
        if spec.quant == QuantType.per_1x128:
            return cls(rows, spec.inter_dim, dtypes.fp8, 128, dtypes.fp32)
        raise NotImplementedError(f"no intermediate layout for {spec.quant} yet")


@dataclass
class Intermediate:
    layout: Layout
    values: torch.Tensor
    scales: torch.Tensor | None


def reference_pack(a2: torch.Tensor, tile_map: TileMap, layout: Layout) -> Intermediate:
    """The intermediate aiter's path would hand stage 2, in sorted-row order.

    ``a2`` is stage 1's token-major output ``[M, topk, cols]`` before the
    inter-stage quant. It is quantized with aiter's own per_1x128 quant (one
    scale per 128 columns of each token-slot row), then gathered by sorted row.
    Padding rows are zero. For tests only.
    """
    from aiter import get_hip_quant

    if layout.group != 128 or layout.dtype != dtypes.fp8:
        raise NotImplementedError("reference_pack covers per_1x128 only")
    m, topk, cols = a2.shape
    q, s = get_hip_quant(QuantType.per_1x128)(a2, quant_dtype=dtypes.fp8)
    q = q.view(m * topk, cols)
    s = s.view(m * topk, cols // 128)
    out = layout.alloc(a2.device)
    out.values.view(torch.uint8).zero_()
    out.scales.zero_()
    nv = int(tile_map.num_valid_ids[0])
    sid = tile_map.sorted_token_ids[:nv]
    tok, slot = sid & 0xFFFFFF, sid >> 24
    real = slot < topk
    rows = torch.nonzero(real).flatten()
    src = (tok[real] * topk + slot[real]).long()
    out.values[rows] = q[src]
    out.scales[rows] = s[src]
    return out
