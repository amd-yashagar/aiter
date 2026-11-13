# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""core ``route_to_tiles`` padding ids under a device ``num_local_tokens``.

A padding row's token id must be the capacity M (rows of ``topk_ids``), as aiter's
sort writes it without a local count. With ``num_local_tokens`` = n < M the sort
wrote n, which is below M, so every consumer's ``token < M`` check took the
padding rows for real routes (the CK stage 2 then accumulated all of them into one
row, 11-28x slower, and n = M - 1 pointed one row past the stage-1 output).
"""

import numpy as np
import pytest
import torch

import aiter.ops.flydsl.moe_next.core.tile_map as tmap
from aiter.jit.utils.chip_info import get_gfx_runtime

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or get_gfx_runtime() not in ("gfx942", "gfx950"),
    reason="Needs gfx942 or gfx950",
)

E, TOPK, TILE = 128, 8, 32
# (tokens, local tokens): the one-shot sort (<= 3072 routes) and the two-launch sort.
CASES = [
    (64, 48),
    (64, 63),
    (64, 64),
    (64, 1),
    (256, 192),
    (256, 255),
    (1024, 768),
    (1024, 1023),
    (4096, 3072),
    (4096, 4095),
]


@pytest.mark.parametrize("M,local", CASES)
def test_padding_rows_carry_the_capacity(M, local):
    g = torch.Generator().manual_seed(M)
    ids = torch.rand(M, E, generator=g).argsort(dim=1)[:, :TOPK].to(torch.int32).cuda()
    weights = (torch.rand(M, TOPK) + 0.5).cuda()
    nlt = torch.tensor([local], dtype=torch.int32, device="cuda")
    tm = tmap.route_to_tiles(
        ids, weights, num_experts=E, tile_m=TILE, num_local_tokens=nlt
    )
    rows = int(tm.num_valid_ids[0])
    entries = tm.sorted_token_ids[:rows].cpu().numpy().astype(np.int64)
    token, slot = entries & 0xFFFFFF, entries >> 24
    real = token < local
    gap = (token >= local) & (token < M)
    assert not gap.any(), (
        f"{int(gap.sum())} padding rows have an id in [{local}, {M}) "
        f"(first {int(token[gap][0])})"
    )
    assert (token[~real] == M).all()
    got = sorted(int(t) * TOPK + int(s) for t, s in zip(token[real], slot[real]))
    assert got == sorted(t * TOPK + s for t in range(local) for s in range(TOPK))
    assert int(tm.num_valid_ids[1]) == local
