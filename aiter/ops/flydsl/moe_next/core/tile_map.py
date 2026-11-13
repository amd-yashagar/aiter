# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routing to tiles: the sort every tile-mode kernel reads, with the output
zero-fill fused in. See kernels/moe_next/core/sort.py for the array contract."""

import functools
import os
from dataclasses import dataclass

import torch

from aiter.jit.utils.chip_info import get_cu_num

TILE_M = (16, 32, 64, 128)
# Up to this many route slots (M * topk) one workgroup sorts them all from
# registers; above, the sort is count + scatter.
ONESHOT_MAX_ROUTES = 3072  # one-shot rounds bound (12 x 256 routes)


def oneshot_max_routes(num_experts: int) -> int:
    """Routes up to which the one-workgroup sort beats the two-launch one
    (MI300X, both zero-filling: break-even near 3072 routes at E <= 256 and
    2560 at E = 512)."""
    return ONESHOT_MAX_ROUTES if num_experts <= 256 else 2560


CHUNK_MIN_ROUTES = 512
MAX_CHUNKS = 128
ZERO_BYTES_PER_WG = 64 * 1024
# A zero-fill this large runs in the count launch (it would compete with long
# scatter chunks); a smaller one overlaps the scatter launch.
ZERO_IN_COUNT_BYTES = 512 << 20


@dataclass
class TileMap:
    """Device-side tile map for one call. The host never reads its contents.

    ``live_tiles[t] = (local expert id, first sorted row, real rows)`` for
    ``t < num_live_tiles[0]``; rows ``>= num_valid_ids[0]`` and tiles
    ``>= num_live_tiles[0]`` hold garbage. ``loc_ids[t * topk + k]`` (when
    requested) is route (t, k)'s sorted row, -1 for a dropped route (masked,
    -1 id, or a repeated expert's earlier slot: an expert repeated within a
    token keeps only its last slot, deterministically; aiter's sorts keep one
    slot by another, unspecified rule, so which slot and weight survive can
    differ from aiter there, MOEBUG-035); tokens at or past the token
    count are not written. ``ready_flags`` is reserved for a single persistent
    launch over both stages (not allocated today). ``a_q`` / ``a_scale`` are
    the quantized activations when ``quant_mode`` was given (QUANT_MODES).
    """

    tile_m: int
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    live_tiles: torch.Tensor
    num_live_tiles: torch.Tensor
    loc_ids: torch.Tensor | None = None
    a_q: torch.Tensor | None = None
    a_scale: torch.Tensor | None = None
    ready_flags: torch.Tensor | None = None

    @property
    def rows(self) -> int:
        return self.sorted_token_ids.numel()

    @property
    def max_tiles(self) -> int:
        return self.rows // self.tile_m


# Activation quant fused into the sort launch: mode -> (column multiple, archs).
QUANT_MODES = {
    "per_token_fp8": (8, None),
    "per_1x128_fp8": (128, None),
    "per_tensor_fp8": (8, None),
    "mx_fp4_round_up": (32, ("gfx950",)),
    "mx_fp8_round_up": (32, ("gfx950",)),
    # tests only: MX addressing with tagged output dwords in place of values
    "mx_fp4_index_debug": (32, None),
    "mx_fp8_index_debug": (32, None),
}


def _quant_buffers(mode, m, k, dev):
    """(a_q, a_scale, kernel views of both, fp8_max). MX scales are e8m0 bytes,
    [M, k/32 padded to 16], lane-native (see core/quant.py)."""
    from aiter import dtypes
    from aiter.ops.flydsl.kernels.moe_next.core.quant import mx_scale_cols

    if mode.startswith("mx_"):
        fp4 = mode.startswith("mx_fp4")
        if mode.endswith("_index_debug"):
            a_q = torch.empty(
                (m, k // (8 if fp4 else 4)), dtype=torch.int32, device=dev
            )
        elif fp4:
            a_q = torch.empty((m, k // 2), dtype=dtypes.fp4x2, device=dev)
        else:
            a_q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=dev)
        s = torch.empty((m, mx_scale_cols(k)), dtype=torch.uint8, device=dev)
        return (
            a_q,
            s.view(dtypes.fp8_e8m0),
            a_q.view(torch.int32),
            s.view(torch.int8),
            0.0,
        )
    a_q = torch.empty((m, k), dtype=dtypes.fp8, device=dev)
    fp8_max = float(torch.finfo(dtypes.fp8).max)
    if mode == "per_tensor_fp8":
        return a_q, None, a_q.view(torch.int32), None, fp8_max
    cols = 1 if mode == "per_token_fp8" else k // 128
    s = torch.empty((m, cols), dtype=torch.float32, device=dev)
    return a_q, s, a_q.view(torch.int32), s, fp8_max


def max_sorted_rows(m: int, topk: int, num_experts: int, tile_m: int) -> int:
    """Padded sorted-row bound, the one aiter's sort allocates."""
    rows = m * topk + num_experts * tile_m - topk
    return -(-rows // tile_m) * tile_m


def max_tiles(m: int, topk: int, num_experts: int, tile_m: int) -> int:
    """Host upper bound on live tiles: a graph-safe grid size."""
    return max_sorted_rows(m, topk, num_experts, tile_m) // tile_m


@functools.cache
def _cu_num() -> int:
    return get_cu_num()


def persistent_grid(k: int = 1) -> int:
    """A persistent grid of ``k`` workgroups per CU, looping over the live tiles."""
    return k * _cu_num()


def _launch(fn, *args, device):
    """Launch through aiter's compiled-handle cache: compiling (and launching)
    once per device, then dispatching the cached handle. Calling the
    ``@flyc.jit`` launcher directly re-traces on every call (tens of ms of
    host time). Every other launch input is fixed by the build key: dtypes,
    ranks and unit strides of the tensors, and the constexprs."""
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    _run_compiled(fn, *args, specialization_key=(device.index,))


@functools.cache
def _debug() -> bool:
    return os.getenv("AITER_MOE_DEBUG", "0") == "1"


def _empty_i32(device):
    return torch.empty(1, dtype=torch.int32, device=device)


def route_to_tiles(
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    *,
    num_experts: int,
    tile_m: int,
    expert_mask: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    zero_output: torch.Tensor | None = None,
    with_loc_ids: bool = False,
    quant_input: torch.Tensor | None = None,
    quant_mode: str | None = None,
    quant_scale: torch.Tensor | None = None,
    pad_rows: bool = True,
) -> TileMap:
    """Sort routes into tiles of ``tile_m`` rows and zero ``zero_output``'s rows
    below the token count (``num_local_tokens`` when given), as aiter's sort does.

    ``num_experts`` is the global expert count (the size of ``expert_mask``
    under EP). ``zero_output`` is the output an atomic epilogue accumulates
    into; pass None for a reduce epilogue. ``quant_input`` / ``quant_mode``
    quantize the token rows in the same launch; ``per_tensor_fp8`` takes its
    static scale as ``quant_scale`` (fp32, one value; ``a_scale`` is that
    tensor). ``pad_rows=False`` skips writing the padding rows (each expert's
    tail up to tile_m) of sorted_token_ids and sorted_weights, which then hold
    garbage there; a kernel that passes it must take each tile's real rows
    from ``live_tiles`` and never read ids or weights past them. Under
    AITER_MOE_DEBUG=1 those rows get a poison instead (id 0x7FFFFFFF, weight
    NaN) so a consumer that reads them is caught. Graph-capture safe: launch
    shape depends only on shapes; every buffer is a per-call ``torch.empty``.
    """
    from aiter.ops.flydsl.kernels.moe_next.core.sort import build

    if tile_m not in TILE_M:
        raise ValueError(f"tile_m must be one of {TILE_M}, got {tile_m}")
    if topk_ids.dtype != torch.int32 or topk_weight.dtype != torch.float32:
        raise TypeError("topk_ids must be int32 and topk_weight fp32")
    m, topk = topk_ids.shape
    if m >= 1 << 24 or topk > 255:
        raise ValueError("sorted ids pack token < 2^24 and topk < 256")
    dev = topk_ids.device
    ids = topk_ids.contiguous()
    w = topk_weight.contiguous()
    rows = max_sorted_rows(m, topk, num_experts, tile_m)
    tiles = rows // tile_m
    tm = TileMap(
        tile_m=tile_m,
        sorted_token_ids=torch.empty(rows, dtype=torch.int32, device=dev),
        sorted_weights=torch.empty(rows, dtype=torch.float32, device=dev),
        sorted_expert_ids=torch.empty(tiles, dtype=torch.int32, device=dev),
        num_valid_ids=torch.empty(2, dtype=torch.int32, device=dev),
        live_tiles=torch.empty((tiles, 3), dtype=torch.int32, device=dev),
        num_live_tiles=torch.empty(1, dtype=torch.int32, device=dev),
    )
    if with_loc_ids:
        tm.loc_ids = torch.empty((m, topk), dtype=torch.int32, device=dev)
    quant, qw, dyn = None, 0, None
    qx = qq = qs = _empty_i32(dev)
    if quant_mode is not None:
        from aiter.jit.utils.chip_info import get_gfx_runtime
        from aiter.ops.flydsl.kernels.moe_next.core.quant import (
            SINGLE_WG_ELEMS,
            blocks,
            tensor_partials,
        )

        if quant_mode not in QUANT_MODES:
            raise ValueError(f"quant_mode must be one of {sorted(QUANT_MODES)}")
        mult, archs = QUANT_MODES[quant_mode]
        if archs is not None and get_gfx_runtime() not in archs:
            raise ValueError(
                f"quant_mode {quant_mode} needs {archs}, not {get_gfx_runtime()}"
            )
        k = quant_input.shape[-1]
        if (
            quant_input.dtype != torch.bfloat16
            or not quant_input.is_contiguous()
            or quant_input.shape != (m, k)
            or k % mult
        ):
            raise ValueError(
                f"quant_input must be contiguous bf16 [M, K], K % {mult} == 0"
            )
        tm.a_q, tm.a_scale, qq, qs, fp8_max = _quant_buffers(quant_mode, m, k, dev)
        quant = (quant_mode, k, fp8_max)
        qw = blocks(quant_mode, m, k)
        if quant_mode == "per_tensor_fp8" and quant_scale is not None:
            if quant_scale.dtype != torch.float32 or quant_scale.numel() != 1:
                raise ValueError("quant_scale must be one fp32 value")
            tm.a_scale = qs = quant_scale.view(1)
        elif quant_mode == "per_tensor_fp8":
            tm.a_scale = qs = torch.empty(1, dtype=torch.float32, device=dev)
            if m * topk <= oneshot_max_routes(num_experts) and m * k <= SINGLE_WG_ELEMS:
                quant, qw = ("per_tensor_fp8_single", k, fp8_max), 1
            else:
                quant = ("per_tensor_fp8_dynamic", k, fp8_max)
                dyn = (qw, tensor_partials(m, k))
        qx = quant_input.view(torch.int32)
    loc = tm.loc_ids if with_loc_ids else _empty_i32(dev)
    mask = expert_mask if expert_mask is not None else _empty_i32(dev)
    ltok = num_local_tokens if num_local_tokens is not None else _empty_i32(dev)
    if zero_output is not None:
        row_bytes = zero_output.nbytes // max(m, 1)
        if (
            not zero_output.is_contiguous()
            or zero_output.shape[0] != m
            or row_bytes % 16
        ):
            raise ValueError(
                "zero_output must be a contiguous [M, ...] tensor with rows a "
                "multiple of 16 bytes"
            )
        zbuf = zero_output.view(torch.int32).view(-1)
        zero_v4, row_v4 = zero_output.nbytes // 16, row_bytes // 16
    else:
        zbuf, zero_v4, row_v4 = _empty_i32(dev), 0, 0
    z_wg = min(-(-zero_v4 * 16 // ZERO_BYTES_PER_WG), 8 * _cu_num()) if zero_v4 else 0
    n_routes = m * topk
    oneshot, count, scatter = build(
        num_experts,
        topk,
        tile_m,
        expert_mask is not None,
        num_local_tokens is not None,
        rounds=max(1, -(-min(n_routes, oneshot_max_routes(num_experts)) // 256)),
        with_loc=with_loc_ids,
        quant=quant,
        pad="sentinel" if pad_rows else ("poison" if _debug() else "none"),
    )
    stream = torch.cuda.current_stream(dev)
    if n_routes <= oneshot_max_routes(num_experts):
        if dyn is not None:  # partial amaxes here, the quant in its own launch
            qw2, n_part = dyn
            qs = torch.empty(n_part, dtype=torch.int32, device=dev)
            qw = n_part
        _launch(
            oneshot,
            ids,
            w,
            tm.sorted_token_ids,
            tm.sorted_weights,
            tm.sorted_expert_ids,
            tm.num_valid_ids,
            tm.live_tiles,
            tm.num_live_tiles,
            loc,
            mask,
            ltok,
            zbuf,
            qx,
            qq,
            qs,
            m,
            row_v4,
            qw,
            1 + qw + z_wg,
            stream,
            device=dev,
        )
        if dyn is not None:
            from aiter.ops.flydsl.kernels.moe_next.core.quant import (
                build_tensor_quant,
            )

            _launch(
                build_tensor_quant(quant[1], quant[2], num_local_tokens is not None),
                qx,
                qq,
                tm.a_scale,
                qs,
                ltok,
                m,
                n_part,
                qw2,
                stream,
                device=dev,
            )
        return tm
    n_chunks = min(MAX_CHUNKS, -(-n_routes // CHUNK_MIN_ROUTES))
    per_chunk = -(-n_routes // n_chunks)
    chunk = -(-per_chunk // 256) * 256
    n_chunks = -(-n_routes // chunk)
    # dynamic per-tensor: the count launch's partial amaxes follow the counts
    qw2, n_part = dyn if dyn is not None else (0, 0)
    counts = torch.empty(n_chunks * num_experts + n_part, dtype=torch.int32, device=dev)
    qp = counts[n_chunks * num_experts :] if dyn is not None else _empty_i32(dev)
    q_count = (qx, qq, qp, n_part) if dyn is not None else (qx, qq, qs, qw)
    early = zero_v4 * 16 >= ZERO_IN_COUNT_BYTES
    _launch(
        count,
        ids,
        mask,
        ltok,
        counts,
        zbuf,
        *q_count[:3],
        m,
        chunk,
        n_chunks,
        row_v4 if early else 0,
        q_count[3],
        max(n_chunks + q_count[3], z_wg) if early else n_chunks + q_count[3],
        stream,
        device=dev,
    )
    _launch(
        scatter,
        ids,
        w,
        tm.sorted_token_ids,
        tm.sorted_weights,
        tm.sorted_expert_ids,
        tm.num_valid_ids,
        tm.live_tiles,
        tm.num_live_tiles,
        loc,
        mask,
        ltok,
        counts,
        zbuf,
        qx,
        qq,
        qs,
        qp,
        m,
        chunk,
        n_chunks,
        0 if early else row_v4,
        qw2,
        n_part,
        n_chunks + qw2 + (0 if early else z_wg),
        stream,
        device=dev,
    )
    return tm


def reference_tile_map(
    topk_ids,
    topk_weight,
    *,
    num_experts,
    tile_m,
    expert_mask=None,
    num_local_tokens=None,
    zero_output=None,
) -> TileMap:
    """aiter's Opus sort, with the live-tile list derived on the device."""
    from aiter import dtypes
    from aiter.fused_moe import _moe_sorting_impl

    topk = topk_ids.shape[1]
    model_dim = zero_output.shape[-1] if zero_output is not None else 0
    sids, sw, seids, nv, _ = _moe_sorting_impl(
        topk_ids,
        topk_weight,
        num_experts,
        max(model_dim, 1),
        zero_output.dtype if zero_output is not None else dtypes.bf16,
        tile_m,
        expert_mask,
        num_local_tokens,
        0,
        True,
        accumulate=zero_output is not None,
        output=zero_output,
    )
    tiles = seids.numel()
    t = torch.arange(tiles, device=topk_ids.device, dtype=torch.int32)
    real = (sids.view(tiles, tile_m) >> 24) < topk
    live = torch.stack([seids, t * tile_m, real.sum(1, dtype=torch.int32)], 1)
    return TileMap(tile_m, sids, sw, seids, nv, live, nv[:1] // tile_m)
