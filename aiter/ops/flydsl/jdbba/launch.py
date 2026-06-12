# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared launch helpers for the jdbba framework.

Single home for the lowlevel-kernel launch and the dense-panel transpose cache,
used by both the ``lowlevel``/``highlevel`` variants and the public genrec API
wrapper — so there is exactly one launch path and one weight-transpose cache.
"""

from __future__ import annotations

import functools
import weakref
from typing import Optional

import torch
from torch import Tensor

import flydsl.expr as fx

from ..gemm_kernels import _normalize_launch_stream, _ptr_view_safe
from ..kernels.jagged_dense_bmm import compile_jagged_dense_bmm_kernel
from ..kernels.tensor_shim import _run_compiled

# data_ptr -> (weakref(dense), subkey, panel): ONE cached transposed panel per
# live tensor address, weakref-identity guarded (a hit is valid only if the same
# live tensor still occupies that address). One entry (not a per-version dict)
# keeps it bounded under changing weights.
_DENSE_B_CACHE: dict[int, tuple] = {}


def dense_b_panel(
    dense: Tensor,
    *,
    batch_size: int,
    n_dim: int,
    k_dim: int,
    transpose_dense: bool,
) -> Tensor:
    """Lay dense out as ``(B*N, K)`` row-major (transpose of the ``(B, D, K)``
    reference layout), cached per live ``dense`` so static weights transpose once."""
    if not transpose_dense:
        return dense.contiguous().reshape(batch_size * n_dim, k_dim)
    ptr = dense.data_ptr()
    subkey = (batch_size, n_dim, k_dim, int(getattr(dense, "_version", 0)))
    entry = _DENSE_B_CACHE.get(ptr)
    if entry is not None:
        wr, cached_key, cached_panel = entry
        if wr() is dense and cached_key == subkey:
            return cached_panel
        # same live tensor mutated, address reused, or shape changed -> replace.
    dense_b = dense.transpose(1, 2).contiguous().reshape(batch_size * n_dim, k_dim)
    try:
        _DENSE_B_CACHE[ptr] = (weakref.ref(dense), subkey, dense_b)
    except TypeError:
        pass  # non-weakrefable; skip caching (always correct)
    return dense_b


@functools.lru_cache(maxsize=256)
def get_launcher(
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    block_m_warps: int,
    block_n_warps: int,
    block_k_warps: int,
    b_to_lds: bool,
    waves_per_eu: int,
    has_bias: bool,
    elementwise: bool,
    xcd_remap: int = 0,
    m_tiles: int = 0,
):
    return compile_jagged_dense_bmm_kernel(
        "bf16", n, k,
        TILE_M=tile_m, TILE_N=tile_n, TILE_K=tile_k, STAGES=stages,
        BLOCK_M_WARPS=block_m_warps, BLOCK_N_WARPS=block_n_warps,
        BLOCK_K_WARPS=block_k_warps, B_TO_LDS=b_to_lds, WAVES_PER_EU=waves_per_eu,
        HAS_BIAS=has_bias, ELEMENTWISE=elementwise, XCD_REMAP=xcd_remap,
        M_TILES=m_tiles,
    )


def run_lowlevel(
    *,
    out: Tensor,
    jagged: Tensor,
    dense: Tensor,
    bias: Optional[Tensor],
    seq_offsets: Tensor,
    max_seq_len: int,
    cfg,  # JdbbaConfig (lowlevel), already resolved for this shape
    has_bias: bool = True,
    elementwise: bool = False,
    transpose_dense: bool = True,
    out_n: Optional[int] = None,
    red_k: Optional[int] = None,
    stream=None,
) -> Tensor:
    """Launch the lowlevel kernel into ``out`` with a resolved ``cfg``.

    Supports the broadcast/elementwise forward and the transposed, bias-free
    d_jagged backward (``transpose_dense=False``, ``out_n``/``red_k`` swapped).
    """
    B = dense.shape[0]
    n_dim = out_n if out_n is not None else dense.shape[2]
    k_dim = red_k if red_k is not None else dense.shape[1]
    tile_m = cfg.tile_m or 128
    tile_n = cfg.tile_n or 128
    tile_k = cfg.tile_k or 32

    if n_dim % tile_n != 0:
        raise ValueError(
            f"output dim N={n_dim} not divisible by tile_n={tile_n} (B{B} D{k_dim} K{n_dim})"
        )
    if k_dim % tile_k != 0 or k_dim // tile_k < cfg.stages:
        raise ValueError(
            f"reduction dim K={k_dim} incompatible with tile_k={tile_k}/"
            f"stages={cfg.stages} (B{B} D{k_dim} K{n_dim})"
        )

    dense_b = dense_b_panel(dense, batch_size=B, n_dim=n_dim, k_dim=k_dim,
                            transpose_dense=transpose_dense)
    if seq_offsets.dtype != torch.int32 or not seq_offsets.is_contiguous():
        seq_offsets = seq_offsets.to(torch.int32).contiguous()

    launch_stream = _normalize_launch_stream(jagged.device, stream)
    dummy = torch.zeros(1, dtype=torch.int32, device=jagged.device)
    if has_bias:
        bias_arg = _ptr_view_safe(bias if elementwise else bias.reshape(B * n_dim))
    else:
        bias_arg = _ptr_view_safe(dummy)

    m_tiles = (max_seq_len + tile_m - 1) // tile_m
    xcd = cfg.xcd_c if (cfg.xcd_c > 0 and B % (8 * cfg.xcd_c) == 0) else 0
    m_tiles_hint = m_tiles if xcd > 0 else 0

    launcher = get_launcher(
        n_dim, k_dim, tile_m, tile_n, tile_k, cfg.stages,
        cfg.block_m_warps, cfg.block_n_warps, cfg.block_k_warps,
        bool(cfg.b_to_lds), int(cfg.waves_per_eu), has_bias, elementwise,
        xcd, m_tiles_hint,
    )
    _run_compiled(
        launcher,
        _ptr_view_safe(out),
        _ptr_view_safe(jagged),
        _ptr_view_safe(dense_b),
        bias_arg,
        _ptr_view_safe(seq_offsets),
        max_seq_len,
        B,
        m_tiles,
        fx.Stream(launch_stream),
    )
    return out
