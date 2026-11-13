# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A token whose topk_ids repeat an expert, through the opus, CK and FlyDSL sorts.

The sorts keep exactly one slot of a repeated expert and drop the rest with their
weights (see the contract in ``moe_sorting``'s docstring). The surviving slot is
unspecified, so these tests pin what is guaranteed and nothing else: every route of a
token without repeats is kept, a repeated expert keeps exactly one of its slots, no
other route appears, and a repeat of a masked expert changes nothing.
"""

import pytest
import torch

import aiter.fused_moe as fused_moe_module
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or get_gfx() not in ("gfx942", "gfx950"),
    reason="Needs gfx942 or gfx950",
)

BLOCK = 32
D = 64
SORTS = ["opus", "ck", "flydsl"]
# topk 8 divides the 64-lane wave; topk 9 makes some tokens span two waves.
CASES = [(256, 128, 8), (256, 128, 9), (4096, 128, 8), (4096, 128, 9)]


def _routes(M, E, topk):
    g = torch.Generator().manual_seed(M + topk)
    ids = torch.rand(M, E, generator=g).argsort(dim=1)[:, :topk].to(torch.int32)
    pairs = [(a, b) for a in range(topk) for b in range(a + 1, topk)]
    dup = {}
    for t in range(0, M, 2):
        a, b = pairs[(t // 2) % len(pairs)]
        ids[t, b] = ids[t, a]
        dup[t] = (a, b)
    return ids.cuda(), dup


def _sort(kind, ids, weights, E, expert_mask=None):
    old = (
        fused_moe_module._USE_CK_MOE_SORTING,
        fused_moe_module._USE_FLYDSL_MOE_SORTING,
    )
    fused_moe_module._USE_CK_MOE_SORTING = kind == "ck"
    fused_moe_module._USE_FLYDSL_MOE_SORTING = kind == "flydsl"
    try:
        sorted_ids, _, _, num_valid, _ = fused_moe_module.moe_sorting(
            ids, weights, E, D, dtypes.bf16, BLOCK, expert_mask
        )[:5]
    finally:
        (
            fused_moe_module._USE_CK_MOE_SORTING,
            fused_moe_module._USE_FLYDSL_MOE_SORTING,
        ) = old
    entries = sorted_ids[: int(num_valid[0])].cpu().numpy().astype("int64")
    token, slot = entries & 0xFFFFFF, entries >> 24
    keep = token < ids.shape[0]
    return {(int(t), int(s)) for t, s in zip(token[keep], slot[keep])}, int(keep.sum())


@pytest.mark.parametrize("kind", SORTS)
@pytest.mark.parametrize("M,E,topk", CASES)
def test_repeated_expert_keeps_exactly_one_slot(kind, M, E, topk):
    ids, dup = _routes(M, E, topk)
    weights = torch.rand(M, topk, device="cuda") + 0.5
    kept, count = _sort(kind, ids, weights, E)
    assert count == len(kept), "a route was emitted twice"
    for token in range(M):
        pair = dup.get(token)
        for slot in range(topk):
            if pair is not None and slot in pair:
                continue
            assert (token, slot) in kept, f"route ({token}, {slot}) was lost"
        if pair is not None:
            survivors = [(token, s) in kept for s in pair]
            assert (
                sum(survivors) == 1
            ), f"token {token} repeats an expert in slots {pair}"
    assert count == M * topk - len(dup)


@pytest.mark.parametrize("kind", ["opus", "ck"])
def test_repeats_of_a_masked_expert_are_dropped(kind):
    M, E, topk = 256, 128, 8
    ids, _ = _routes(M, E, topk)
    ids[:, 1] = ids[:, 0]
    mask = (torch.arange(E, device="cuda") % 2).to(dtypes.i32)
    weights = torch.rand(M, topk, device="cuda") + 0.5
    kept, count = _sort(kind, ids, weights, E, mask)
    assert count == len(kept)
    rows = ids.cpu().tolist()
    local = mask.cpu().tolist()
    per_expert = {}
    for token, slot in kept:
        per_expert.setdefault((token, rows[token][slot]), []).append(slot)
    assert all(local[expert] for _, expert in per_expert), "a masked expert was kept"
    wanted = {(t, e) for t in range(M) for e in rows[t] if local[e]}
    assert set(per_expert) == wanted
    assert all(len(slots) == 1 for slots in per_expert.values())
