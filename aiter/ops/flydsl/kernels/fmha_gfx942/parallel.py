# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CU-filling split-K mapping for gfx942 packed-varlen FMHA.

Independent work items are ``(batch, q_tile, head, kv_split)``. Each workgroup
owns ``BLOCK_M`` query rows of one head and a contiguous slice of KV tiles
(``BLOCK_N`` aligned). Online softmax stays inside the workgroup; splits are
combined with the usual LSE logsumexp.

gfx942 has 304 CUs and these kernels are one workgroup per CU (LDS). Launching
fewer workgroups than CUs leaves compute idle. Launching just over 304 causes a
second dispatch wave whose duration equals a full tile and can erase the split.

``choose_kv_splits`` minimizes ``ceil(N / cu_count) / S`` with ``N`` the
workgroup count, preferring fewer splits on a tie. ``BLOCK_M`` stays a multiple
of ``32 * num_waves`` so each wave keeps one 32x32 MFMA M-tile.
"""

from __future__ import annotations

import math

import torch

# MI300X / MI325X. Overridden at call time from ``get_cu_num()``.
GFX942_CU_COUNT = 304
# Compile-cache bound. KV tiles already cap S for short sequences.
MAX_KV_SPLITS = 16
_EMPTY_LSE = -1.0e30


# Prefer fewer splits when modeled time is within this relative band of the
# minimum. Prologue, ping-pong stagger, and the host combine dominate once
# each workgroup's KV walk is short, so a 5–10% arithmetic win is not real.
_SPLIT_COST_REL_TOL = 0.10


def choose_kv_splits(
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    block_m: int,
    block_n: int,
    cu_count: int = GFX942_CU_COUNT,
    batch: int = 1,
) -> int:
    """Pick the split-K factor that fills CUs without a wasteful tail wave."""
    if (
        max_seqlen_q <= 0
        or num_heads <= 0
        or block_m <= 0
        or block_n <= 0
        or cu_count <= 0
        or batch <= 0
    ):
        return 1
    nq = (max_seqlen_q + block_m - 1) // block_m
    base = batch * nq * num_heads
    kv_tiles = max(1, (max(max_seqlen_k, 0) + block_n - 1) // block_n)
    max_s = min(kv_tiles, MAX_KV_SPLITS)
    costs = []
    best_cost = math.inf
    for splits in range(1, max_s + 1):
        n_wg = base * splits
        dispatch_waves = (n_wg + cu_count - 1) // cu_count
        cost = dispatch_waves / splits
        costs.append((splits, cost))
        if cost < best_cost:
            best_cost = cost
    threshold = best_cost * (1.0 + _SPLIT_COST_REL_TOL)
    for splits, cost in costs:
        if cost <= threshold:
            return splits
    return 1


def combine_split_kv(
    o_parts: torch.Tensor,
    lse_parts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge normalized split outputs.

    ``o_parts`` is ``[S, T, H, D]`` (already divided by each split's ``l``).
    ``lse_parts`` is ``[S, H, T]`` with empty splits at ``_EMPTY_LSE``.
    """
    lse_f = lse_parts.float()
    o_f = o_parts.float()
    lse_max = lse_f.amax(dim=0)
    alpha = (lse_f - lse_max).exp()
    denom = alpha.sum(dim=0).clamp_min(1e-20)
    weight = (alpha / denom).permute(0, 2, 1).unsqueeze(-1)
    out = (o_f * weight).sum(dim=0).to(o_parts.dtype)
    lse = lse_max + denom.log()
    return out, lse
