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
    ``mfma_native`` (MX: group 32, e8m0) stores them in scaled-MFMA operand
    order, aiter's e8m0_shuffle over sorted rows: ``[rows padded to 32,
    groups padded to 8]`` bytes, byte (r, g) at ``scale_offset(r, g)``, the
    padding bytes 0x7F (written by stage 1; ``alloc`` is ``torch.empty``).
    ``row_pad`` rounds the row count up; ``col_pad`` rounds each value row
    (in elements) up.
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
        if self.scale_order == "mfma_native" and (
            self.group != 32 or self.scale_dtype != dtypes.fp8_e8m0
        ):
            raise ValueError("mfma_native needs group 32 and fp8_e8m0 scales")
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
        if self.scale_order == "mfma_native":
            return (-(-self.padded_rows // 32) * 32, -(-self.groups // 8) * 8)
        return (self.padded_rows, self.groups)

    @property
    def nbytes(self) -> int:
        n = self.values_shape[0] * self.values_shape[1] * self.dtype.itemsize
        if self.scales_shape is not None:
            rows, cols = self.scales_shape
            n += rows * cols * self.scale_dtype.itemsize
        return n

    def value_offset(self, row: int, col: int) -> int:
        """Element offset of (row, col) in the values tensor's storage."""
        return row * self.values_shape[1] + col

    def scale_offset(self, row, group):
        """Element offset of the scale of (row, group); ints or tensors."""
        if self.scale_order == "mfma_native":
            from aiter.ops.flydsl.kernels.moe_next.core.intermediate import (
                mfma_native_offset,
            )

            return mfma_native_offset(row, group, self.scales_shape[1])
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
        if spec.quant == QuantType.per_1x32:
            # a4w4 keeps fp4 between the stages; MXFP8 weights keep fp8
            dtype = dtypes.fp4x2 if spec.w_dtype == dtypes.fp4x2 else dtypes.fp8
            return cls(rows, spec.inter_dim, dtype, 32, dtypes.fp8_e8m0, "mfma_native")
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

    if layout.scale_order == "mfma_native":
        return _reference_pack_mx(a2, tile_map, layout)
    if layout.group != 128 or layout.dtype != dtypes.fp8:
        raise NotImplementedError("reference_pack covers per_1x128 and MX only")
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


def _sorted_rows(tile_map: TileMap, topk: int):
    nv = int(tile_map.num_valid_ids[0])
    sid = tile_map.sorted_token_ids[:nv]
    tok, slot = sid & 0xFFFFFF, sid >> 24
    real = slot < topk
    return torch.nonzero(real).flatten(), (tok[real] * topk + slot[real]).long()


def _reference_pack_mx(a2, tile_map, layout):
    """MX: aiter's per_1x32_mx_quant_hip (RoundUp) per token-slot row, values
    gathered by sorted row (padding rows zero), scales in the mfma_native
    order with 0x7F padding."""
    from aiter.ops.quant import per_1x32_mx_quant_hip

    m, topk, cols = a2.shape
    q, s = per_1x32_mx_quant_hip(
        a2.reshape(m * topk, cols),
        quant_dtype=layout.dtype,
        scale_type=dtypes.fp8_e8m0,
    )
    out = layout.alloc(a2.device)
    out.values.view(torch.uint8).zero_()
    rows, src = _sorted_rows(tile_map, topk)
    out.values.view(torch.uint8)[rows] = q.view(torch.uint8).view(m * topk, -1)[src]
    rm = torch.full(layout.scales_shape, 0x7F, dtype=torch.uint8, device=a2.device)
    rm[rows, : layout.groups] = s.view(torch.uint8)[src, : layout.groups]
    out.scales.view(torch.uint8).copy_(to_mfma_native(rm, layout))
    return out


def to_mfma_native(row_major: torch.Tensor, layout: Layout) -> torch.Tensor:
    """Row-major e8m0 bytes ``[*layout.scales_shape]`` in the mfma_native
    order (same shape)."""
    r = torch.arange(row_major.shape[0], device=row_major.device)[:, None]
    g = torch.arange(row_major.shape[1], device=row_major.device)[None, :]
    out = torch.empty_like(row_major).view(-1)
    out[layout.scale_offset(r, g).flatten()] = row_major.flatten()
    return out.view(row_major.shape)
