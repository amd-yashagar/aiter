# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter.moe core pieces against aiter: the sort and tile map (exact against
Opus, with its zero-fill), graph capture, timing, and the intermediate layout."""

import argparse
import csv
import functools

import pandas as pd
import torch

import aiter
from aiter import QuantType, dtypes
from aiter.fused_moe import _flydsl_moe_sorting, _moe_sorting_impl, fused_topk
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.moe import MoeSpec
from aiter.ops.flydsl.moe_next.core.intermediate import (
    Layout,
    reference_pack,
    to_mfma_native,
)
from aiter.ops.flydsl.moe_next.core.tile_map import reference_tile_map, route_to_tiles

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]


def routing(token, E, topk, case="random", mask=False, seed=0):
    """topk ids/weights as fused_topk makes them, then the routing case."""
    g = torch.Generator(device="cuda").manual_seed(seed + token)
    x = torch.randn((token, 64), dtype=dtypes.bf16, generator=g)
    score = torch.rand((token, E), dtype=dtypes.bf16, generator=g)
    if case == "onehot":
        score[:, 0] += 5
    elif case == "zipf":
        rank = torch.arange(1, E + 1, dtype=torch.float32)
        score = score + (10.0 / rank).to(dtypes.bf16)
    tw, ti = fused_topk(x, score, topk, True)
    if case == "mixed":
        inv = (torch.arange(token * topk).view(token, topk) % 3 == 1).cuda()
        ti, tw = ti.masked_fill(inv, -1), tw.masked_fill(inv, 0)
    elif case == "all_empty":
        ti, tw = torch.full_like(ti, -1), torch.zeros_like(tw)
    elif case == "dup":
        ti = ti[:, :1].expand_as(ti).contiguous()
    elif case == "pad":
        # CommFused-style padding rows: expert 0 in every slot, weight 0
        pad = torch.arange(token).cuda() % 4 == 3
        ti, tw = ti.masked_fill(pad[:, None], 0), tw.masked_fill(pad[:, None], 0)
    em = torch.randint(0, 2, (E,), dtype=torch.int32, generator=g) if mask else None
    return ti, tw, em


def compare(a, b, tile_m, out_a, out_b, slots=True):
    """Field names that differ between two tile maps (valid regions only).

    ``slots=False`` ignores which topk slot a row came from (and its weight):
    for an expert repeated within one token, aiter keeps one slot by its own
    unspecified rule (opus and CK prefer the larger slot % 4, then the lower
    slot // 4; FlyDSL another; all nondeterministic when the token spans two
    64-lane waves: MOEBUG-035); core keeps the last slot (test_repeat_keep_last)."""
    nv = int(b.num_valid_ids[0])
    nt = nv // tile_m
    bad = []
    if not torch.equal(a.num_valid_ids, b.num_valid_ids):
        bad.append("num_valid_ids")
    ida, idb = a.sorted_token_ids[:nv], b.sorted_token_ids[:nv]
    if not slots:
        ida, idb = ida & 0xFFFFFF, idb & 0xFFFFFF
    if not torch.equal(ida, idb):
        bad.append("sorted_token_ids")
    if slots and not torch.equal(
        a.sorted_weights[:nv].view(torch.int32), b.sorted_weights[:nv].view(torch.int32)
    ):
        bad.append("sorted_weights")
    if not torch.equal(a.sorted_expert_ids[:nt], b.sorted_expert_ids[:nt]):
        bad.append("sorted_expert_ids")
    if not torch.equal(a.live_tiles[:nt], b.live_tiles[:nt]):
        bad.append("live_tiles")
    if int(a.num_live_tiles[0]) != nt:
        bad.append("num_live_tiles")
    if out_a is not None and not torch.equal(out_a, out_b):
        bad.append("zero_fill")
    return bad


def _repeat_routing(token, E, topk, seed):
    """topk ids with repeated experts in most tokens (pairs, runs, a token all
    one expert), weights all distinct, so the kept slot is visible."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ti = torch.stack(
        [torch.randperm(E, generator=g, device="cpu")[:topk] for _ in range(token)]
    )
    for t in range(token):
        kind = t % 4
        if kind == 1:  # one pair
            a, b = torch.randperm(topk, generator=g, device="cpu")[:2].tolist()
            ti[t, b] = ti[t, a]
        elif kind == 2:  # every slot the same expert
            ti[t] = ti[t, 0]
        elif kind == 3:  # runs of three
            for k in range(2, topk, 3):
                ti[t, k] = ti[t, k - 2]
                ti[t, k - 1] = ti[t, k - 2]
    tw = torch.arange(1, token * topk + 1, dtype=torch.float32, device="cpu")
    tw = tw.view(token, topk)
    return ti.int().cuda(), (tw / (token * topk)).cuda()


def test_repeat_keep_last():
    """Core's rule for an expert repeated within one token: exactly one row per
    (token, expert), from the token's last slot of that expert, with that
    slot's weight; earlier slots are dropped (loc_ids -1). On the one-shot and
    two-launch paths, at topk 2, 8, 9 and 11 (9 and 11: tokens span two
    64-lane waves), tile_m 16 and 32. Not compared with aiter, whose kept slot
    follows another, unspecified rule (MOEBUG-035)."""
    rows_out = []
    for token, E, topk, tile_m in [
        (8, 16, 2, 32),
        (64, 32, 8, 16),
        (100, 64, 9, 32),
        (93, 257, 11, 16),
        (300, 64, 9, 32),  # 2700 routes
        (2000, 257, 11, 32),  # 22000 routes: two launches
        (1000, 16, 9, 16),  # 9000 routes: two launches
    ]:
        ti, tw = _repeat_routing(token, E, topk, token + topk)
        tm = route_to_tiles(ti, tw, num_experts=E, tile_m=tile_m, with_loc_ids=True)
        nv = int(tm.num_valid_ids[0])
        sid = tm.sorted_token_ids[:nv].cpu().long()
        sw = tm.sorted_weights[:nv].cpu()
        tok, slot = sid & 0xFFFFFF, sid >> 24
        real = slot < topk
        tok, slot, sw = tok[real], slot[real], sw[real]
        tic, twc = ti.cpu().long(), tw.cpu()
        expert = tic[tok, slot]
        keys = (tok * (E + 1) + expert).tolist()
        bad = []
        if len(keys) != len(set(keys)):
            bad.append("more than one row per (token, expert)")
        want = {}
        for t in range(token):
            for k in range(topk):
                want[(t, int(tic[t, k]))] = k  # the last slot of each expert wins
        got = {(int(t), int(e)): int(k) for t, e, k in zip(tok, expert, slot)}
        if got != want:
            bad.append("kept slots")
        if not torch.equal(sw, twc[tok, slot]):
            bad.append("weights of the kept slots")
        loc = tm.loc_ids.cpu().long()
        for t in range(token):
            for k in range(topk):
                keep = want[(t, int(tic[t, k]))] == k
                if (loc[t, k] >= 0) != keep:
                    bad.append(f"loc_ids ({t}, {k})")
                    break
        assert not bad, f"keep-last M={token} E={E} topk={topk}: {bad[:3]}"
        rows_out.append(
            {
                "check": f"repeat keeps the last slot M={token} E={E} topk={topk}",
                "exact": True,
            }
        )
    return rows_out


def check_loc_ids(tm, ti, expert_mask, num_local_tokens):
    """loc_ids against the tile map itself: each kept route's row holds that
    route; dropped routes (masked, -1, a repeated expert's earlier slot) are -1."""
    m, topk = ti.shape
    rows = m if num_local_tokens is None else int(num_local_tokens[0])
    loc = tm.loc_ids[:rows].flatten().long()
    ids = ti[:rows].flatten()
    kept = loc >= 0
    route = torch.arange(rows * topk, device=ti.device)
    packed = ((route % topk) << 24) | (route // topk)
    if not torch.equal(tm.sorted_token_ids[loc[kept]], packed[kept].int()):
        return ["loc_ids rows"]
    valid = ids >= 0
    if expert_mask is not None:
        valid &= expert_mask[ids.clamp(min=0).long()] != 0
    later = torch.zeros_like(valid)
    v = ti[:rows]
    for j in range(1, topk):
        later.view(rows, topk)[:, :-j] |= v[:, j:] == v[:, :-j]
    if not torch.equal(kept, valid & ~later):
        return ["loc_ids kept set"]
    return []


def test_path_boundary():
    """The one-shot and two-launch sorts give bit-identical outputs on the
    same input (sorted ids and weights with their padding, expert ids, counts,
    live_tiles, loc_ids, zero-fill, fused quant), at each E's one-shot limit
    and just past it, so a routed kernel sees the same inputs whichever path
    the limit picks."""
    import aiter.ops.flydsl.moe_next.core.tile_map as tmap

    lim = tmap.oneshot_max_routes
    rows_out = []
    for E, topk in (
        (64, 8),
        (128, 8),
        (129, 8),
        (256, 8),
        (257, 9),
        (385, 7),
        (513, 11),
    ):
        cap = lim(E)
        for token in (cap // topk, cap // topk + 1):
            for case, mask, local in (
                ("random", False, None),
                ("dup", False, None),
                ("mixed", True, token - 3),
            ):
                ti, tw, em = routing(token, E, topk, case, mask)
                nl = None if local is None else torch.tensor([local], dtype=torch.int32)
                x = _quant_input(token, 1024, token)
                outs = [
                    torch.full((token, 512), 3.0, dtype=dtypes.bf16) for _ in range(2)
                ]
                maps = []
                for force, out in zip((1 << 13, 0), outs):
                    tmap.oneshot_max_routes = lambda e, f=force: f
                    try:
                        maps.append(
                            route_to_tiles(
                                ti,
                                tw,
                                num_experts=E,
                                tile_m=32,
                                expert_mask=em,
                                num_local_tokens=nl,
                                zero_output=out,
                                with_loc_ids=True,
                                quant_input=x,
                                quant_mode="per_token_fp8",
                            )
                        )
                    finally:
                        tmap.oneshot_max_routes = lim
                a, b = maps
                nv = int(a.num_valid_ids[0])
                nlive = int(a.num_live_tiles[0])
                rows = token if local is None else local
                bad = [
                    name
                    for name, x1, x2 in (
                        ("num_valid_ids", a.num_valid_ids, b.num_valid_ids),
                        ("num_live_tiles", a.num_live_tiles, b.num_live_tiles),
                        (
                            "sorted_token_ids",
                            a.sorted_token_ids[:nv],
                            b.sorted_token_ids[:nv],
                        ),
                        (
                            "sorted_weights",
                            a.sorted_weights[:nv].view(torch.int32),
                            b.sorted_weights[:nv].view(torch.int32),
                        ),
                        (
                            "sorted_expert_ids",
                            a.sorted_expert_ids[: nv // 32],
                            b.sorted_expert_ids[: nv // 32],
                        ),
                        ("live_tiles", a.live_tiles[:nlive], b.live_tiles[:nlive]),
                        ("loc_ids", a.loc_ids[:rows], b.loc_ids[:rows]),
                        ("zero_fill", outs[0], outs[1]),
                        (
                            "a_q",
                            a.a_q[:rows].view(torch.uint8),
                            b.a_q[:rows].view(torch.uint8),
                        ),
                        (
                            "a_scale",
                            a.a_scale[:rows].view(torch.int32),
                            b.a_scale[:rows].view(torch.int32),
                        ),
                    )
                    if not torch.equal(x1, x2)
                ]
                assert (
                    not bad
                ), f"one-shot vs two-launch E={E} topk={topk} M={token} {case}: {bad}"
                rows_out.append(
                    {
                        "check": f"paths bit-identical E={E} topk={topk} M={token} {case}",
                        "exact": True,
                    }
                )
    return rows_out


def test_tile_map(token, E, topk, tile_m, case, mask, local, model_dim=4096):
    """route_to_tiles against Opus (via reference_tile_map), output zero-fill included."""
    ti, tw, em = routing(token, E, topk, case, mask)
    nl = None
    if local is not None:
        nl = torch.tensor([local], dtype=torch.int32)
    out_a = torch.full((token, model_dim), 3.0, dtype=dtypes.bf16)
    out_b = out_a.clone()
    kw = {"num_experts": E, "tile_m": tile_m, "expert_mask": em, "num_local_tokens": nl}
    a = route_to_tiles(ti, tw, zero_output=out_a, with_loc_ids=True, **kw)
    b = reference_tile_map(ti, tw, zero_output=out_b, **kw)
    srt = ti.sort(dim=1).values
    repeats = bool(((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).any())
    bad = compare(a, b, tile_m, out_a, out_b, slots=not repeats)
    bad += check_loc_ids(a, ti, em, nl)
    assert not bad, f"M={token} E={E} topk={topk} tile_m={tile_m} {case}: {bad}"
    return {"exact": True}


def _real_rows(tm):
    """Bool [rows]: rows that hold a route, from live_tiles alone."""
    n_live = int(tm.num_live_tiles[0])
    lt = tm.live_tiles[:n_live].cpu()
    real = torch.zeros(tm.sorted_token_ids.numel(), dtype=torch.bool, device="cpu")
    for _, first, rows in lt.tolist():
        real[first : first + rows] = True
    return real


@functools.cache
def _tail_consumer(tile_m):
    """A consumer shaped like the fp8-token mid kernels: one workgroup per
    live tile from live_tiles (expert, first row, real rows); every row of the
    tile reads its sorted id and weight and gathers x[token] through an
    M-bounded buffer (garbage tokens read 0 or a real row), and only rows below
    the tile's real rows store out[token * topk + slot] = x[token] * weight."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    def it(t, n=None):
        if n is None:
            return fx.get_iter(fx.rocdl.make_buffer_tensor(t, max_size=True))
        return fx.get_iter(
            fx.rocdl.make_buffer_tensor(t, max_size=False, num_records_bytes=n)
        )

    @flyc.kernel(known_block_size=[128, 1, 1])
    def consume(
        x: fx.Tensor,
        sids: fx.Tensor,
        sw: fx.Tensor,
        live: fx.Tensor,
        nlive: fx.Tensor,
        out: fx.Tensor,
        m: fx.Int32,
        topk: fx.Int32,
    ):
        t = fx.gpu.block_idx.x
        row = fx.gpu.thread_idx.x
        if (t < fx.ptr_load(it(nlive))) & (row < fx.Int32(tile_m)):
            lt = it(live)
            r0 = fx.ptr_load(lt + fx.Int64(t * fx.Int32(3) + fx.Int32(1)))
            real = fx.ptr_load(lt + fx.Int64(t * fx.Int32(3) + fx.Int32(2)))
            sid = fx.ptr_load(it(sids) + fx.Int64(r0 + row)).bitcast(fx.Uint32)
            tok = fx.Int32(sid & fx.Uint32(0xFFFFFF))
            slot = fx.Int32(sid >> fx.Uint32(24))
            w = fx.ptr_load(it(sw) + fx.Int64(r0 + row))
            xv = fx.ptr_load(it(x, fx.Int64(m) * fx.Int64(4)) + fx.Int64(tok))
            if row < real:
                fx.ptr_store(xv * w, it(out) + fx.Int64(tok * topk + slot))

    @flyc.jit
    def launch(
        x: fx.Tensor,
        sids: fx.Tensor,
        sw: fx.Tensor,
        live: fx.Tensor,
        nlive: fx.Tensor,
        out: fx.Tensor,
        m: fx.Int32,
        topk: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = None,
    ):
        consume(x, sids, sw, live, nlive, out, m, topk).launch(
            grid=(grid, 1, 1), block=(128, 1, 1), stream=stream
        )

    return launch


def test_pad_rows_consumer():
    """pad_rows=False with a consumer that masks the tile tail by live_tiles
    (as the fp8-token mid kernels do): with the padding skipped (garbage) or
    poisoned (AITER_MOE_DEBUG), its output equals the padded map's and the
    reference, x[token] * weight for every kept route."""
    import os

    import aiter.ops.flydsl.moe_next.core.tile_map as tmap

    rows_out = []
    for token, E, topk, tile_m, mask, local in [
        (8, 256, 8, 32, False, None),
        (100, 513, 11, 16, True, 90),
        (300, 385, 7, 64, False, None),
        (1024, 256, 8, 128, False, 1000),
    ]:
        ti, tw, em = routing(token, E, topk, mask=mask)
        nl = None if local is None else torch.tensor([local], dtype=torch.int32)
        x = torch.randn(token, dtype=torch.float32) + 2
        kw = {"num_experts": E, "tile_m": tile_m, "expert_mask": em}
        kw["num_local_tokens"] = nl
        outs = {}
        for name, pad_rows, debug in (
            ("padded", True, False),
            ("unpadded", False, False),
            ("poisoned", False, True),
        ):
            old = os.environ.get("AITER_MOE_DEBUG")
            os.environ["AITER_MOE_DEBUG"] = "1" if debug else "0"
            tmap._debug.cache_clear()
            try:
                tm = route_to_tiles(ti, tw, pad_rows=pad_rows, **kw)
            finally:
                if old is None:
                    del os.environ["AITER_MOE_DEBUG"]
                else:
                    os.environ["AITER_MOE_DEBUG"] = old
                tmap._debug.cache_clear()
            out = torch.full((token * topk,), -7.0, dtype=torch.float32)
            _tail_consumer(tile_m)(
                x,
                tm.sorted_token_ids,
                tm.sorted_weights,
                tm.live_tiles,
                tm.num_live_tiles,
                out,
                token,
                topk,
                tm.live_tiles.shape[0],
            )
            outs[name] = (out, tm)
        ref = torch.full((token * topk,), -7.0, dtype=torch.float32)
        tm = outs["padded"][1]
        nv = int(tm.num_valid_ids[0])
        sid = tm.sorted_token_ids[:nv].long()
        tok, slot = sid & 0xFFFFFF, sid >> 24
        kept = slot < topk
        u = tok[kept] * topk + slot[kept]
        ref[u] = x[tok[kept]] * tm.sorted_weights[:nv][kept]
        bad = [
            n
            for n, (o, _) in outs.items()
            if not torch.equal(o.view(torch.int32), ref.view(torch.int32))
        ]
        assert not bad, f"pad_rows consumer M={token} E={E} topk={topk}: {bad}"
        rows_out.append(
            {
                "check": f"pad_rows=False tail consumer M={token} E={E} tile_m={tile_m}",
                "exact": True,
            }
        )
    return rows_out


def test_pad_rows():
    """pad_rows=False: everything a consumer may read (expert ids, counts,
    live_tiles, ids and weights of real rows, loc_ids, zero-fill, quant)
    matches the padded tile map; under AITER_MOE_DEBUG every padding row is
    poisoned, so a consumer that reads past a tile's real rows sees NaN."""
    import os

    import aiter.ops.flydsl.moe_next.core.tile_map as tmap

    rows_out = []
    for token, E, topk, tile_m, mask, local in [
        (8, 256, 8, 32, False, None),
        (8, 896, 16, 16, True, 6),
        (300, 385, 7, 32, True, 250),
        (512, 256, 8, 64, False, None),
        (3000, 513, 11, 32, False, 2900),
    ]:
        ti, tw, em = routing(token, E, topk, mask=mask)
        nl = None if local is None else torch.tensor([local], dtype=torch.int32)
        x = _quant_input(token, 1024, token)
        outs = [torch.full((token, 256), 3.0, dtype=dtypes.bf16) for _ in range(3)]
        kw = {
            "num_experts": E,
            "tile_m": tile_m,
            "expert_mask": em,
            "num_local_tokens": nl,
            "with_loc_ids": True,
            "quant_input": x,
            "quant_mode": "per_token_fp8",
        }
        a = route_to_tiles(ti, tw, zero_output=outs[0], **kw)
        b = route_to_tiles(ti, tw, zero_output=outs[1], pad_rows=False, **kw)
        old = os.environ.get("AITER_MOE_DEBUG")
        os.environ["AITER_MOE_DEBUG"] = "1"
        tmap._debug.cache_clear()
        try:
            c = route_to_tiles(ti, tw, zero_output=outs[2], pad_rows=False, **kw)
        finally:
            if old is None:
                del os.environ["AITER_MOE_DEBUG"]
            else:
                os.environ["AITER_MOE_DEBUG"] = old
            tmap._debug.cache_clear()
        real = _real_rows(a)
        nv = int(a.num_valid_ids[0])
        pad = ~real
        pad[nv:] = False
        n_live = int(a.num_live_tiles[0])
        bad = []
        for name, t in (("b", b), ("poisoned", c)):
            if not torch.equal(t.num_valid_ids, a.num_valid_ids):
                bad.append(f"{name} num_valid_ids")
            if not torch.equal(t.num_live_tiles, a.num_live_tiles):
                bad.append(f"{name} num_live_tiles")
            if not torch.equal(t.live_tiles[:n_live], a.live_tiles[:n_live]):
                bad.append(f"{name} live_tiles")
            tiles = -(-nv // tile_m)
            if not torch.equal(
                t.sorted_expert_ids[:tiles], a.sorted_expert_ids[:tiles]
            ):
                bad.append(f"{name} sorted_expert_ids")
            if not torch.equal(
                t.sorted_token_ids.cpu()[real], a.sorted_token_ids.cpu()[real]
            ):
                bad.append(f"{name} real-row ids")
            if not torch.equal(
                t.sorted_weights.cpu()[real].view(torch.int32),
                a.sorted_weights.cpu()[real].view(torch.int32),
            ):
                bad.append(f"{name} real-row weights")
            rows = token if local is None else local  # rows past it: unwritten
            if not torch.equal(t.loc_ids[:rows], a.loc_ids[:rows]):
                bad.append(f"{name} loc_ids")
            if not torch.equal(
                t.a_q[:rows].view(torch.uint8), a.a_q[:rows].view(torch.uint8)
            ):
                bad.append(f"{name} a_q")
        if not (torch.equal(outs[1], outs[0]) and torch.equal(outs[2], outs[0])):
            bad.append("zero-fill")
        # the poison, and a consumer that reads whole tiles against one that
        # takes real rows from live_tiles
        cid, cw = c.sorted_token_ids.cpu(), c.sorted_weights.cpu()
        if pad.any() and not (
            (cid[pad] == 0x7FFFFFFF).all() and torch.isnan(cw[pad]).all()
        ):
            bad.append("poison")
        whole = cw[: (nv // tile_m) * tile_m].sum()
        good = cw[real].sum()
        if pad.any() and not torch.isnan(whole):
            bad.append("whole-tile read not caught")
        if not torch.equal(good, a.sorted_weights.cpu()[real].sum()):
            bad.append("real-row read")
        assert not bad, f"pad_rows M={token} E={E} topk={topk} tile_m={tile_m}: {bad}"
        rows_out.append(
            {
                "check": f"pad_rows=False M={token} E={E} topk={topk} tile_m={tile_m}",
                "exact": True,
            }
        )
    return rows_out


def test_graph_capture(token=256, E=256, topk=8, tile_m=32):
    """Captured once, replayed with new routing written into the same inputs."""
    ti, tw, _ = routing(token, E, topk, seed=1)
    out = torch.empty((token, 4096), dtype=dtypes.bf16)
    route_to_tiles(ti, tw, num_experts=E, tile_m=tile_m, zero_output=out)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        tm = route_to_tiles(ti, tw, num_experts=E, tile_m=tile_m, zero_output=out)
    ti2, tw2, _ = routing(token, E, topk, seed=7)
    ti.copy_(ti2)
    tw.copy_(tw2)
    out.fill_(3.0)
    g.replay()
    ref_out = torch.full_like(out, 3.0)
    ref = reference_tile_map(ti, tw, num_experts=E, tile_m=tile_m, zero_output=ref_out)
    bad = compare(tm, ref, tile_m, out, ref_out)
    assert not bad, f"graph replay: {bad}"
    return [{"check": f"graph capture and replay M={token} E={E}"}]


QUANT_TYPES = {
    "per_token_fp8": QuantType.per_Token,
    "per_1x128_fp8": QuantType.per_1x128,
    "per_tensor_fp8": QuantType.per_Tensor,
}
STATIC_SCALE = 0.0173  # the 400 outlier saturates


def _quant_input(token, K, seed, specials=False):
    """Random rows, an outlier, an all-zero token 1, and with ``specials`` a NaN
    (group 1 of token 2) and an inf (group 2 of token 3)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((token, K), dtype=dtypes.bf16, generator=g) * 3
    x[0, 5] = 400.0
    if token > 2:
        x[1].zero_()
    if specials and token > 3:
        x[2, 40] = float("nan")
        x[3, 70] = float("inf")
    return x


MX_TYPES = {"mx_fp4_round_up": dtypes.fp4x2, "mx_fp8_round_up": dtypes.fp8}


def _mx_lane_native(rs, m, k):
    """aiter's row-major [M, G] e8m0 scales in core's lane-native order."""
    from aiter.ops.flydsl.kernels.moe_next.core.quant import mx_scale_cols

    groups, gp = k // 32, mx_scale_cols(k)
    out = torch.zeros((m, gp), dtype=torch.uint8, device=rs.device)
    g = torch.arange(groups, device=rs.device)
    out[:, (g % 4) * (gp // 4) + g // 4] = rs.view(torch.uint8)[:m, :groups]
    return out


def _check_quant(tm, x, mode, local, scale=None):
    rows = x.shape[0]
    nr = None
    if local is not None:
        nr = torch.tensor([local], dtype=torch.int32)
        rows = local
    if mode == "per_tensor_fp8":
        # aiter's per-tensor quant takes no num_rows; compare the rows below it
        rq, _ = aiter.get_hip_quant(QuantType.per_Tensor)(
            x, scale=scale, quant_dtype=dtypes.fp8
        )
        bad = []
        if not torch.equal(
            tm.a_q[:rows].view(torch.uint8), rq[:rows].view(torch.uint8)
        ):
            bad.append("a_q")
        if tm.a_scale.data_ptr() != scale.data_ptr():
            bad.append("a_scale is not the given scale")
        return bad
    if mode in MX_TYPES:
        from aiter.ops.quant import per_1x32_mx_quant_hip

        rq, rs = per_1x32_mx_quant_hip(
            x, quant_dtype=MX_TYPES[mode], scale_type=dtypes.fp8_e8m0, num_rows=nr
        )
        rs = _mx_lane_native(rs, x.shape[0], x.shape[1])
        bad = []
        # Token-major: one scale row per token and none for the sentinel token
        # that padding slots name, so stage 1 must mask those rows. Padding
        # groups (K/32 up to a multiple of 16) are 0, the byte aiter's sorted
        # Triton quant writes for its padding, finite so that a padding K-step
        # times zero activations stays 0.
        if tm.a_scale.shape[0] != x.shape[0]:
            bad.append("a_scale rows")
        gp, groups = tm.a_scale.shape[1], x.shape[1] // 32
        g = torch.arange(groups, gp, device=x.device)
        if tm.a_scale[:rows].view(torch.uint8)[:, (g % 4) * (gp // 4) + g // 4].any():
            bad.append("a_scale padding")
        if not torch.equal(
            tm.a_q[:rows].view(torch.uint8), rq[:rows].view(torch.uint8)
        ):
            bad.append("a_q")
        if not torch.equal(tm.a_scale[:rows].view(torch.uint8), rs[:rows]):
            bad.append("a_scale")
        return bad
    kw = {"quant_dtype": dtypes.fp8}
    if nr is not None:
        kw["num_rows"] = nr
    rq, rs = aiter.get_hip_quant(QUANT_TYPES[mode])(x, **kw)
    bad = []
    if not torch.equal(tm.a_q[:rows].view(torch.uint8), rq[:rows].view(torch.uint8)):
        bad.append("a_q")
    rs = rs.view(x.shape[0], -1)
    if not torch.equal(
        tm.a_scale[:rows].view(torch.int32), rs[:rows].view(torch.int32)
    ):
        bad.append("a_scale")
    return bad


def _mx_e8m0_model(x, fp4, fp8_max=448.0):
    """aiter's per_1x32_mx_quant_hip scales: amax over 32 columns from
    kAmaxFloor with NaN skipped, RoundUp e8m0 of amax * float(1 / max_pos),
    fp8 capped at 254 (mxfp8_cap_scale). max_pos is 6 for fp4 and the fp8
    max (448 on gfx950; aiter's gfx942 build uses 240)."""
    a = x.float().abs().view(x.shape[0], -1, 32)
    a = torch.where(a.isnan(), torch.zeros_like(a), a).amax(-1).clamp_min(1e-10)
    inv = torch.tensor(1.0 / (6.0 if fp4 else fp8_max), dtype=torch.float32)
    u = (a * inv.to(a.device)).view(torch.int32) & 0x7FFFFFFF
    e = ((u + 0x7FFFFF) >> 23).clamp(max=255 if fp4 else 254)
    return e.to(torch.uint8)


def test_mx_index():
    """The MX pass's addressing on any arch: the index-debug modes write each
    output dword as t*4096 + g*16 + j and the real e8m0 scales, checked against
    a torch model of the lane-native layout, the padding groups and the
    num_local_tokens bound (rows past it must stay untouched)."""
    from aiter.ops.flydsl.kernels.moe_next.core.quant import mx_offsets, mx_scale_cols

    rows_out = []
    for fp4 in (True, False):
        mode = f"mx_fp{4 if fp4 else 8}_index_debug"
        for token, E, topk, K, local in [
            (8, 256, 8, 7168, None),
            (200, 128, 8, 4096, 150),
            (3000, 256, 8, 7168, None),
            (2050, 385, 7, 2048, 2000),
            (8, 128, 4, 2880, None),
            (300, 128, 4, 2880, 250),
            (1, 64, 2, 2880, None),
        ]:
            ti, tw, _ = routing(token, E, topk)
            x = _quant_input(token, K, token, specials=True)
            nl = torch.tensor([local], dtype=torch.int32) if local is not None else None
            tm = route_to_tiles(
                ti,
                tw,
                num_experts=E,
                tile_m=32,
                num_local_tokens=nl,
                quant_input=x,
                quant_mode=mode,
            )
            rows = token if local is None else local
            groups, gp = K // 32, mx_scale_cols(K)
            per = 4 if fp4 else 8
            u = torch.arange(rows * gp, device=x.device)
            t, g, _, q, sb = mx_offsets(u, K, fp4, int)
            ok = g < groups
            want_s = torch.zeros(rows * gp, dtype=torch.uint8, device=x.device)
            want_s[sb[ok]] = _mx_e8m0_model(x[:rows], fp4).flatten()
            want_q = torch.full((rows * K // 32 * per,), -1, dtype=torch.int32)
            want_q = want_q.to(x.device)
            j = torch.arange(per, device=x.device)
            want_q[(q[ok, None] + j).flatten()] = (
                ((t[ok] * 4096 + g[ok] * 16)[:, None] + j).flatten().int()
            )
            bad = []
            if (want_q < 0).any() or (want_s.numel() != rows * gp):
                bad.append("model coverage")
            if not torch.equal(tm.a_q[:rows].flatten(), want_q):
                bad.append("a_q placement")
            if not torch.equal(tm.a_scale[:rows].view(torch.uint8).flatten(), want_s):
                bad.append("a_scale")
            assert not bad, f"{mode} M={token} K={K} local={local}: {bad}"
            rows_out.append(
                {"mode": mode, "token": token, "K": K, "local": local, "exact": True}
            )
    return rows_out


def _dyn_input(token, K, seed, kind, local):
    x = _quant_input(token, K, seed)
    if kind == "zero":
        x.zero_()
    elif kind == "nan" and token > 2:
        x[2] = float("nan")
    elif kind == "inf" and token > 3:
        x[3, :5] = float("inf")
    if local is not None:
        x[local:] = 3e38  # past num_local_tokens: must not reach the scale
    return x


def _check_dynamic_tensor(tm, x, rows):
    rq, rs = aiter.get_hip_quant(QuantType.per_Tensor)(
        x[:rows].contiguous(), quant_dtype=dtypes.fp8
    )
    bad = []
    if not torch.equal(tm.a_q[:rows].view(torch.uint8), rq.view(torch.uint8)):
        bad.append("a_q")
    if not torch.equal(tm.a_scale.view(torch.int32), rs.view(torch.int32)):
        bad.append(f"a_scale {tm.a_scale.item()} vs {rs.item()}")
    return bad


def _dyn_path(token, E, topk, K):
    from aiter.ops.flydsl.kernels.moe_next.core.quant import SINGLE_WG_ELEMS
    from aiter.ops.flydsl.moe_next.core.tile_map import oneshot_max_routes

    if token * topk > oneshot_max_routes(E):
        return "count + scatter"
    if token * K <= SINGLE_WG_ELEMS:
        return "one workgroup"
    return "one-shot + quant launch"


def test_dynamic_tensor():
    """Dynamic per-tensor fp8 against aiter's dynamic_per_tensor_quant (on the
    rows below num_local_tokens; aiter's op takes no row count): the three
    paths (one workgroup in the one-shot launch, one-shot plus a quant launch,
    count plus scatter) and their boundaries, NaN and inf rows,
    all-zero input (scale 0, NaN outputs), huge rows past num_local_tokens,
    and graph replay."""
    from aiter.ops.flydsl.moe_next.core.tile_map import oneshot_max_routes

    lim64 = oneshot_max_routes(64) // 8
    rows_out = []
    cases = [
        (8, 256, 8, 7168, None, "plain"),
        (8, 64, 8, 4096, None, "plain"),  # M x K = SINGLE_WG_ELEMS (32K)
        (8, 64, 8, 4104, None, "plain"),  # 32K + 64, the next size (K % 8)
        (8, 64, 8, 4096, None, "nan"),
        (8, 64, 8, 4096, None, "inf"),
        (8, 64, 8, 4096, None, "zero"),
        (8, 64, 8, 8192, None, "plain"),
        (lim64, 64, 8, 7168, None, "plain"),  # last one-shot token count
        (lim64 + 1, 64, 8, 7168, None, "plain"),  # first count + scatter
        (4, 64, 4, 2048, 2, "plain"),
        (8, 256, 8, 7168, None, "nan"),
        (8, 256, 8, 7168, None, "inf"),
        (8, 256, 8, 7168, None, "zero"),
        (200, 128, 8, 4096, 150, "plain"),
        (200, 128, 8, 4096, 150, "nan"),
        (3000, 256, 8, 7168, None, "plain"),
        (3000, 256, 8, 7168, None, "zero"),
        (2050, 385, 7, 2048, 2000, "inf"),
    ]
    for token, E, topk, K, local, kind in cases:
        ti, tw, _ = routing(token, E, topk)
        x = _dyn_input(token, K, token, kind, local)
        nl = torch.tensor([local], dtype=torch.int32) if local is not None else None
        out_a = torch.full((token, 1024), 3.0, dtype=dtypes.bf16)
        out_b = out_a.clone()
        kw = {"num_experts": E, "tile_m": 32, "num_local_tokens": nl}
        a = route_to_tiles(
            ti, tw, zero_output=out_a, quant_input=x, quant_mode="per_tensor_fp8", **kw
        )
        b = reference_tile_map(ti, tw, zero_output=out_b, **kw)
        rows = token if local is None else local
        bad = compare(a, b, 32, out_a, out_b) + _check_dynamic_tensor(a, x, rows)
        assert not bad, f"dynamic per-tensor M={token} K={K} {kind}: {bad}"
        rows_out.append(
            {
                "token": token,
                "K": K,
                "local": local,
                "input": kind,
                "path": _dyn_path(token, E, topk, K),
                "exact": True,
            }
        )
    for token, K in [(8, 4096), (8, 7168), (512, 7168)]:
        ti, tw, _ = routing(token, 256, 8)
        x = _dyn_input(token, K, 1, "plain", None)
        kw = {"num_experts": 256, "tile_m": 32, "quant_input": x}
        route_to_tiles(ti, tw, quant_mode="per_tensor_fp8", **kw)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            tm = route_to_tiles(ti, tw, quant_mode="per_tensor_fp8", **kw)
        x.copy_(_dyn_input(token, K, 9, "nan", None) * 7)
        g.replay()
        torch.cuda.synchronize()
        bad = _check_dynamic_tensor(tm, x, token)
        assert not bad, f"dynamic per-tensor graph replay M={token}: {bad}"
        rows_out.append(
            {
                "token": token,
                "K": K,
                "local": None,
                "input": "graph replay, new x",
                "path": _dyn_path(token, 256, 8, K),
                "exact": True,
            }
        )
    return rows_out


def test_fused_quant():
    """Activation quant fused into the sort launch, bit-exact with aiter's
    quant op for each mode, on the one-shot and two-launch paths, and with the
    tile map still exact."""
    rows = []
    modes = [*QUANT_TYPES, *(MX_TYPES if get_gfx_runtime() == "gfx950" else ())]
    for mode in modes:
        cases = [
            (8, 256, 8, 7168, None),
            (200, 128, 8, 4096, 150),
            (3000, 256, 8, 7168, None),
            (2050, 385, 7, 2048, 2000),
        ]
        if mode in MX_TYPES:
            # K = 2880: 90 groups padded to 96, and a 128-column step half padding
            cases += [(8, 128, 4, 2880, None), (300, 128, 4, 2880, 250)]
        for token, E, topk, K, local in cases:
            ti, tw, _ = routing(token, E, topk)
            x = _quant_input(token, K, token, specials=mode in MX_TYPES)
            nl = torch.tensor([local], dtype=torch.int32) if local is not None else None
            out_a = torch.full((token, 1024), 3.0, dtype=dtypes.bf16)
            out_b = out_a.clone()
            kw = {"num_experts": E, "tile_m": 32, "num_local_tokens": nl}
            scale = None
            if mode == "per_tensor_fp8":
                scale = torch.tensor([STATIC_SCALE], dtype=torch.float32)
            a = route_to_tiles(
                ti,
                tw,
                zero_output=out_a,
                quant_input=x,
                quant_mode=mode,
                quant_scale=scale,
                **kw,
            )
            b = reference_tile_map(ti, tw, zero_output=out_b, **kw)
            bad = compare(a, b, 32, out_a, out_b)
            bad += _check_quant(a, x, mode, local, scale)
            assert not bad, f"{mode} M={token} K={K}: {bad}"
            rows.append(
                {
                    "mode": mode,
                    "token": token,
                    "E": E,
                    "K": K,
                    "local": local,
                    "exact": True,
                }
            )
    # graph capture: replay with new activations (and a new static scale)
    # written into the same inputs
    ti, tw, _ = routing(256, 256, 8)
    for mode in ("per_1x128_fp8", "per_tensor_fp8"):
        x = _quant_input(256, 7168, 1)
        scale = torch.tensor([STATIC_SCALE], dtype=torch.float32)
        kw = {"num_experts": 256, "tile_m": 32, "quant_input": x, "quant_mode": mode}
        kw["quant_scale"] = scale if mode == "per_tensor_fp8" else None
        route_to_tiles(ti, tw, **kw)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            tm = route_to_tiles(ti, tw, **kw)
        x.copy_(_quant_input(256, 7168, 9))
        scale.fill_(0.5)
        g.replay()
        torch.cuda.synchronize()
        bad = _check_quant(tm, x, mode, None, scale)
        assert not bad, f"{mode} graph replay: {bad}"
        rows.append(
            {
                "mode": f"{mode} graph replay",
                "token": 256,
                "E": 256,
                "K": 7168,
                "local": None,
                "exact": True,
            }
        )
    return rows


def test_host_time(calls=300):
    """Eager host time per route_to_tiles call stays in microseconds: launches go
    through cached compiled handles, not a re-trace per call."""
    import time

    rows = []
    for token, E, topk, tile_m in [
        (8, 256, 8, 16),
        (512, 256, 8, 32),
        (4096, 512, 11, 64),
    ]:
        ti, tw, _ = routing(token, E, topk)
        out = torch.empty((token, 4096), dtype=dtypes.bf16)
        kw = {"num_experts": E, "tile_m": tile_m, "zero_output": out}
        for _ in range(5):
            route_to_tiles(ti, tw, **kw)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(calls):
            route_to_tiles(ti, tw, **kw)
        host_us = (time.perf_counter() - t) * 1e6 / calls
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(calls):
            _moe_sorting_impl(
                ti, tw, E, 4096, dtypes.bf16, tile_m, None, None, 0, True, output=out
            )
        opus_us = (time.perf_counter() - t) * 1e6 / calls
        torch.cuda.synchronize()
        assert host_us < 300, f"route_to_tiles host time {host_us:.0f} us per call"
        rows.append(
            {
                "token": token,
                "E": E,
                "topk": topk,
                "tile_m": tile_m,
                "new host us": host_us,
                "opus host us": opus_us,
            }
        )
    return rows


def _graph_us(fn, iters=20, per_graph=20):
    for _ in range(3):
        fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(per_graph):
            fn()
    g.replay()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / (iters * per_graph)


def time_sorts(token, E, topk, mask, model_dim, tile_m=32):
    """us per call, captured 20 per graph, for the new sort (also with
    pad_rows=False) and today's sorts, each zero-filling an [M, model_dim] bf16
    output as an atomic stage 2 needs."""
    ti, tw, em = routing(token, E, topk, mask=mask)
    buf = torch.empty((token, model_dim), dtype=dtypes.bf16)
    bf16 = dtypes.bf16
    fns = {
        "new": lambda: route_to_tiles(
            ti, tw, num_experts=E, tile_m=tile_m, expert_mask=em, zero_output=buf
        ),
        "new nopad": lambda: route_to_tiles(
            ti,
            tw,
            num_experts=E,
            tile_m=tile_m,
            expert_mask=em,
            zero_output=buf,
            pad_rows=False,
        ),
        "opus": lambda: _moe_sorting_impl(
            ti, tw, E, model_dim, bf16, tile_m, em, None, 0, True, output=buf
        ),
        "ck": lambda: _moe_sorting_impl(
            ti, tw, E, model_dim, bf16, tile_m, em, None, 0, False, output=buf
        ),
        "flydsl": lambda: _flydsl_moe_sorting(
            ti, tw, E, model_dim, bf16, tile_m, em, None, output=buf
        ),
    }
    ret = {f"{k} us": _graph_us(fn) for k, fn in fns.items()}
    best = min(("opus", "ck", "flydsl"), key=lambda k: ret[f"{k} us"])
    ret["best today"] = best
    ret["new / best"] = ret["new us"] / ret[f"{best} us"]
    return ret


@functools.cache
def _inline_quant_probe(mode, cols, fp8_max):
    """A direct-kernel prologue: every workgroup quantizes all M rows with
    the inline block of ``mode`` (``none``: no quant, the kernel alone)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.moe_next.core import quant as cq

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, 8, 16]

    def _iter(t):
        return fx.get_iter(fx.rocdl.make_buffer_tensor(t, max_size=True))

    @flyc.kernel(known_block_size=[256, 1, 1])
    def inline_kernel(x: fx.Tensor, q: fx.Tensor, s: fx.Tensor, m: fx.Int32):
        lds = fx.SharedAllocator().allocate(Lds).peek()
        xi, qi, si = _iter(x), _iter(q), _iter(s)
        if mode == "per_token_fp8":
            for r in range(fx.Int32(0), m, fx.Int32(1)):
                cq.quant_token_block(
                    xi, qi, si, lds.red.ptr, fx.Int32(r), m, cols, fp8_max
                )
        elif mode == "per_1x128_fp8":
            groups = m * fx.Int32(cols // 128)
            for b in range(
                fx.Int32(0), (groups + fx.Int32(15)) // fx.Int32(16), fx.Int32(1)
            ):
                cq.quant_1x128_block(xi, qi, si, fx.Int32(b), groups, fp8_max)
        elif mode in cq.MX_MODES:
            units = m * fx.Int32(cq.mx_scale_cols(cols))
            for b in range(
                fx.Int32(0), (units + fx.Int32(255)) // fx.Int32(256), fx.Int32(1)
            ):
                cq.quant_mx_block(xi, qi, si, fx.Int32(b), m, cols, *cq.MX_MODES[mode])

    @flyc.jit
    def launch(
        x: fx.Tensor,
        q: fx.Tensor,
        s: fx.Tensor,
        m: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = None,
    ):
        inline_kernel(x, q, s, m).launch(
            grid=(grid, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


def _bound_launch(fn, *args):
    """A call of a compiled launcher on the stream current when it runs."""
    from aiter.ops.flydsl.moe_next.core.tile_map import _launch

    dev = torch.cuda.current_device()

    def run():
        _launch(
            fn, *args, torch.cuda.current_stream(), device=torch.device("cuda", dev)
        )

    return run


def _in_order(*calls):
    def run():
        for call in calls:
            call()

    return run


def time_inline_quant(m=8, cols=7168, wgs=(1, 2, 4, 8, 16, 32)):
    """us per call (graph of 20): a direct kernel of ``wgs`` column-block
    workgroups that each quantize all M rows inline, against the same kernel
    without quant plus one separate quant launch (core's standalone launch,
    and aiter's own quant op)."""
    from aiter.ops.flydsl.kernels.moe_next.core.quant import (
        blocks,
        build_standalone,
        mx_scale_cols,
    )

    modes = ["per_token_fp8", "per_1x128_fp8"]
    if get_gfx_runtime() == "gfx950":
        modes += ["mx_fp4_round_up", "mx_fp8_round_up"]
    fp8_max = float(torch.finfo(dtypes.fp8).max)
    x = torch.randn((m, cols), dtype=dtypes.bf16)
    none = _inline_quant_probe("none", cols, fp8_max)
    q8 = torch.empty((m, cols), dtype=torch.uint8)
    rows = []
    for mode in modes:
        if mode.startswith("mx_"):
            q = torch.empty(
                (m, cols // 2 if "fp4" in mode else cols), dtype=torch.uint8
            )
            s = torch.empty((m, mx_scale_cols(cols)), dtype=torch.int8)
            ref = functools.partial(
                aiter.get_hip_quant(QuantType.per_1x32),
                quant_dtype=dtypes.fp4x2 if "fp4" in mode else dtypes.fp8,
                scale_type=dtypes.fp8_e8m0,
            )
        else:
            q = q8
            s = torch.empty(
                (m, 1 if mode == "per_token_fp8" else cols // 128), dtype=torch.float32
            )
            ref = functools.partial(
                aiter.get_hip_quant(QUANT_TYPES[mode]), quant_dtype=dtypes.fp8
            )
        qi, lt = q.view(torch.int32), torch.empty(1, dtype=torch.int32)
        inline = _inline_quant_probe(mode, cols, fp8_max)
        alone = build_standalone(mode, cols, fp8_max, False)
        grid = blocks(mode, m, cols)
        for c in wgs:
            kernel = _bound_launch(none, x, qi, s, m, c)
            t_none = _graph_us(kernel)
            t_in = _graph_us(_bound_launch(inline, x, qi, s, m, c))
            t_core = _graph_us(
                _in_order(
                    _bound_launch(alone, x, qi, s, lt, m, grid),
                    kernel,
                )
            )
            t_aiter = _graph_us(_in_order(functools.partial(ref, x), kernel))
            rows.append(
                {
                    "mode": mode,
                    "M": m,
                    "K": cols,
                    "workgroups": c,
                    "kernel alone us": round(t_none, 2),
                    "inline us": round(t_in, 2),
                    "core launch + kernel us": round(t_core, 2),
                    "aiter quant + kernel us": round(t_aiter, 2),
                }
            )
    return rows


QUANT_TIMED = ("per_token_fp8", "per_1x128_fp8", "per_tensor_fp8", "static")


def time_sort_quant(token, E, topk, mask, model_dim, tile_m=32):
    """us per call (20 per graph): the sort with activation quant fused
    against Opus's sort plus aiter's quant op, both zero-filling an
    [M, model_dim] bf16 output; per_tensor_fp8 is dynamic, static is
    per_tensor_fp8 with a given scale."""
    ti, tw, em = routing(token, E, topk, mask=mask)
    buf = torch.empty((token, model_dim), dtype=dtypes.bf16)
    x = torch.randn((token, model_dim), dtype=dtypes.bf16)
    scale = torch.tensor([0.02], dtype=torch.float32)
    bf16 = dtypes.bf16
    rows = []
    for mode in QUANT_TIMED:
        qmode = "per_tensor_fp8" if mode == "static" else mode
        qscale = scale if mode == "static" else None
        qtype = QUANT_TYPES[qmode]

        def fused(qmode=qmode, qscale=qscale):
            route_to_tiles(
                ti,
                tw,
                num_experts=E,
                tile_m=tile_m,
                expert_mask=em,
                zero_output=buf,
                quant_input=x,
                quant_mode=qmode,
                quant_scale=qscale,
            )

        def separate(qtype=qtype, qscale=qscale):
            _moe_sorting_impl(
                ti, tw, E, model_dim, bf16, tile_m, em, None, 0, True, output=buf
            )
            aiter.get_hip_quant(qtype)(x, scale=qscale, quant_dtype=dtypes.fp8)

        f, sep = _graph_us(fused), _graph_us(separate)
        rows.append(
            {
                "mode": mode,
                "fused_us": round(f, 2),
                "opus_plus_quant_us": round(sep, 2),
                "ratio": round(f / sep, 3),
            }
        )
    return rows


@functools.cache
def _direct_probe(E, topk, has_mask, has_local):
    """A one-workgroup kernel running group_routes and zero_rows, dumping the
    LDS block."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.moe_next.core.direct import (
        group_routes,
        lds_words,
        zero_rows,
    )

    words = -(-lds_words(E) // 256) * 256

    @fx.struct
    class Lds:
        w: fx.Array[fx.Int32, words, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def probe(
        ids: fx.Tensor,
        weights: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        dump: fx.Tensor,
        zout: fx.Tensor,
        m_cap: fx.Int32,
        row_v4: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(Lds).peek()
        group_routes(
            lds.w.ptr, ids, weights, mask, ltok, m_cap, E, topk, has_mask, has_local
        )
        tid = fx.gpu.thread_idx.x
        it = fx.get_iter(fx.rocdl.make_buffer_tensor(dump, max_size=True))
        for k in fx.range_constexpr(0, words, 256):
            idx = fx.Int32(k) + tid
            fx.ptr_store(fx.ptr_load(lds.w.ptr + fx.Int64(idx)), it + fx.Int64(idx))
        tokens = m_cap
        if has_local:
            lt = fx.get_iter(fx.rocdl.make_buffer_tensor(ltok, max_size=True))
            tokens = fx.ptr_load(lt)
        zero_rows(zout, tid, fx.Int32(256), tokens, m_cap, row_v4)

    @flyc.jit
    def launch(
        ids: fx.Tensor,
        weights: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        dump: fx.Tensor,
        zout: fx.Tensor,
        m_cap: fx.Int32,
        row_v4: fx.Int32,
        stream: fx.Stream = None,
    ):
        probe(ids, weights, mask, ltok, dump, zout, m_cap, row_v4).launch(
            grid=(1, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch, words


@functools.cache
def _slot_probe(E, topk, tile_m, has_mask, has_local):
    """One workgroup per route slot running route_slot; thread 0 dumps
    (expert, rank, leader), a leader dumps its tile list."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.moe_next.core.direct import (
        route_slot,
        slot_lds_words,
    )

    words = -(-slot_lds_words(tile_m) // 16) * 16

    @fx.struct
    class Lds:
        w: fx.Array[fx.Int32, words, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def probe(
        ids: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        meta: fx.Tensor,
        tiles: fx.Tensor,
        m_cap: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(Lds).peek()
        bid = fx.gpu.block_idx.x
        tid = fx.gpu.thread_idx.x
        expert, rank, leader = route_slot(
            lds.w.ptr, ids, mask, ltok, m_cap, bid, E, topk, tile_m, has_mask, has_local
        )
        mi = fx.get_iter(fx.rocdl.make_buffer_tensor(meta, max_size=True))
        ti = fx.get_iter(fx.rocdl.make_buffer_tensor(tiles, max_size=True))
        if tid == fx.Int32(0):
            fx.ptr_store(expert, mi + fx.Int64(bid * fx.Int32(3)))
            fx.ptr_store(rank, mi + fx.Int64(bid * fx.Int32(3) + fx.Int32(1)))
            fx.ptr_store(
                leader.select(fx.Int32(1), fx.Int32(0)),
                mi + fx.Int64(bid * fx.Int32(3) + fx.Int32(2)),
            )
        if leader & (tid < fx.Int32(tile_m)):
            fx.ptr_store(
                fx.ptr_load(lds.w.ptr + fx.Int64(tid)),
                ti + fx.Int64(bid * fx.Int32(tile_m) + tid),
            )

    @flyc.jit
    def launch(
        ids: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        meta: fx.Tensor,
        tiles: fx.Tensor,
        m_cap: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = None,
    ):
        probe(ids, mask, ltok, meta, tiles, m_cap).launch(
            grid=(grid, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


def _slot_expected(tm, topk, n_slots):
    """Per route: (expert, rank) from a tile map, and each tile's routes."""
    nv = int(tm.num_valid_ids[0])
    sid = tm.sorted_token_ids[:nv].cpu()
    eid = tm.sorted_expert_ids.cpu()
    want = {u: (-1, None) for u in range(n_slots)}
    tiles = {}
    start_of = {}
    for r in range(nv):
        tok, k = int(sid[r]) & 0xFFFFFF, int(sid[r]) >> 24
        if k >= topk:
            continue
        e = int(eid[r // tm.tile_m])
        start_of.setdefault(e, r)
        u = tok * topk + k
        want[u] = (e, r - start_of[e])
        tiles.setdefault(r // tm.tile_m, []).append(u)
    leaders = {v[0]: (want[v[0]][0], v) for v in tiles.values()}
    return want, leaders


def test_route_slot():
    """direct.route_slot (one workgroup per slot) against the core sort's tile
    map: every slot's expert and rank, which slots lead a tile, and each
    leader's route list; M * topk up to 256, EP, -1 ids, num_local_tokens,
    repeats, tile_m 16 and 32."""
    rows_out = []
    shapes = [(1, 256, 8), (8, 256, 8), (8, 385, 7), (8, 896, 16), (8, 513, 11)]
    shapes += [(16, 64, 4), (13, 128, 5), (16, 512, 11), (32, 256, 8), (2, 16, 2)]
    for token, E, topk in shapes:
        for case in ("random", "mixed", "dup", "pad", "all_empty", "onehot"):
            for mask in (False, True):
                for local in (None, max(0, token - 2)):
                    for tile_m in (16, 32):
                        ti, tw, em = routing(token, E, topk, case, mask)
                        nl = (
                            None
                            if local is None
                            else torch.tensor([local], dtype=torch.int32)
                        )
                        n = token * topk
                        meta = torch.full((n * 3,), -7, dtype=torch.int32)
                        tiles = torch.full((n * tile_m,), -7, dtype=torch.int32)
                        launch = _slot_probe(E, topk, tile_m, mask, local is not None)
                        launch(
                            ti,
                            em if mask else ti,
                            nl if nl is not None else ti,
                            meta,
                            tiles,
                            token,
                            n,
                        )
                        tm = route_to_tiles(
                            ti,
                            tw,
                            num_experts=E,
                            tile_m=tile_m,
                            expert_mask=em,
                            num_local_tokens=nl,
                        )
                        want, leaders = _slot_expected(tm, topk, n)
                        m = meta.cpu().view(n, 3).tolist()
                        tl = tiles.cpu().view(n, tile_m).tolist()
                        bad = []
                        for u in range(n):
                            e, rank, lead = m[u]
                            we, wr = want[u]
                            if e != we or (we >= 0 and rank != wr):
                                bad.append(f"slot {u}: {e, rank} vs {we, wr}")
                            if bool(lead) != (u in leaders):
                                bad.append(f"slot {u}: leader {lead}")
                            elif lead:
                                routes = leaders[u][1]
                                got = tl[u][: len(routes)]
                                rest = tl[u][len(routes) :]
                                if got != routes or any(x != -1 for x in rest):
                                    bad.append(f"slot {u}: tile {tl[u]} vs {routes}")
                        assert not bad, (
                            f"route_slot M={token} E={E} topk={topk} {case} "
                            f"mask={mask} local={local} tile_m={tile_m}: {bad[:3]}"
                        )
                        rows_out.append({"token": token, "E": E, "exact": True})
    return rows_out


def _direct_expected(tm, topk):
    """(order, weight bits, group experts, group starts) from a tile map."""
    nv = int(tm.num_valid_ids[0])
    sid = tm.sorted_token_ids[:nv].cpu()
    sw = tm.sorted_weights[:nv].cpu().view(torch.int32)
    eid = tm.sorted_expert_ids.cpu()[torch.arange(nv, device="cpu") // tm.tile_m]
    real = (sid >> 24) < topk
    order, wts, ex = sid[real], sw[real], eid[real]
    starts = [i for i in range(len(ex)) if i == 0 or ex[i] != ex[i - 1]]
    starts_t = torch.tensor(starts + [len(ex)], dtype=torch.int32, device="cpu")
    return order, wts, ex[starts], starts_t


def test_direct_grouping():
    """direct.group_routes against the core sort (and Opus where no expert
    repeats within a token) for M * topk up to 256 (64, 65, 128 and 256
    included), with EP masks, -1 ids, num_local_tokens and repeats; zero_rows
    on bf16 and fp32 outputs."""
    from aiter.ops.flydsl.kernels.moe_next.core.direct import offsets

    rows_out = []
    shapes = [(1, 256, 8), (8, 256, 8), (4, 128, 6), (16, 64, 4), (64, 32, 1)]
    shapes += [(7, 385, 9), (3, 513, 5), (2, 16, 2)]
    # 64, 65, 128 and 256 routes, and decode shapes past one wave
    shapes += [(13, 128, 5), (65, 64, 1), (8, 512, 11), (16, 512, 11)]
    shapes += [(8, 896, 16), (16, 256, 8), (32, 256, 8), (128, 64, 2)]
    for token, E, topk in shapes:
        for case in ("random", "mixed", "dup", "pad", "all_empty", "onehot"):
            for mask in (False, True):
                for local in (None, max(0, token - 2)):
                    ti, tw, em = routing(token, E, topk, case, mask)
                    nl = (
                        None
                        if local is None
                        else torch.tensor([local], dtype=torch.int32)
                    )
                    launch, words = _direct_probe(E, topk, mask, local is not None)
                    dump = torch.empty(words, dtype=torch.int32)
                    for dt in (dtypes.bf16, torch.float32):
                        zout = torch.full((token, 72), 7.0, dtype=dt)
                        row_v4 = zout.nbytes // token // 16
                        launch(
                            ti,
                            tw,
                            em if mask else ti,
                            nl if nl is not None else ti,
                            dump,
                            zout.view(torch.int32),
                            token,
                            row_v4,
                        )
                        rows = token if local is None else local
                        assert (zout[:rows] == 0).all() and (zout[rows:] == 7).all()
                    o_order, o_w, o_ge, o_gs, o_meta = offsets(E)[:5]
                    d = dump.cpu()
                    n_valid, n_groups = int(d[o_meta]), int(d[o_meta + 1])
                    got = (
                        d[o_order : o_order + n_valid],
                        d[o_w : o_w + n_valid],
                        d[o_ge : o_ge + n_groups],
                        d[o_gs : o_gs + n_groups + 1],
                    )
                    kw = {"num_experts": E, "tile_m": 32, "num_local_tokens": nl}
                    kw["expert_mask"] = em
                    # aiter keeps a repeated expert's slot by its own rule (MOEBUG-035)
                    refs = [("core sort", route_to_tiles(ti, tw, **kw))]
                    if case not in ("dup", "pad"):
                        refs.append(("opus", reference_tile_map(ti, tw, **kw)))
                    for ref_name, ref in refs:
                        want = _direct_expected(ref, topk)
                        names = ("order", "weight", "group expert", "group start")
                        bad = [
                            n
                            for n, a, b in zip(names, got, want)
                            if not torch.equal(a, b)
                        ]
                        assert not bad, (
                            f"direct M={token} E={E} topk={topk} {case} "
                            f"mask={mask} local={local} vs {ref_name}: {bad}"
                        )
                    rows_out.append(
                        {
                            "token": token,
                            "E": E,
                            "topk": topk,
                            "case": case,
                            "mask": mask,
                            "local": local,
                            "exact": True,
                        }
                    )
    return rows_out


def _offsets_on_device(rows, groups, gp8):
    """scale_offset_mfma evaluated by a kernel for every (row, group)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.moe_next.core.intermediate import (
        scale_offset_mfma,
    )

    @flyc.kernel(known_block_size=[256, 1, 1])
    def offsets_kernel(out: fx.Tensor, n: fx.Int32):
        u = fx.gpu.block_idx.x * fx.Int32(256) + fx.gpu.thread_idx.x
        r = u // fx.Int32(groups)
        g = u - r * fx.Int32(groups)
        it = fx.get_iter(fx.rocdl.make_buffer_tensor(out, max_size=True))
        if u < n:
            fx.ptr_store(scale_offset_mfma(r, g, gp8), it + fx.Int64(u))

    @flyc.jit
    def launch(out: fx.Tensor, n: fx.Int32, grid: fx.Int32, stream: fx.Stream = None):
        offsets_kernel(out, n).launch(
            grid=(grid, 1, 1), block=(256, 1, 1), stream=stream
        )

    out = torch.empty(rows * groups, dtype=torch.int32)
    launch(out, rows * groups, -(-rows * groups // 256))
    return out.view(rows, groups)


def test_mfma_native():
    """The MX mfma_native scale order: byte for byte against aiter's
    shuffle_scale (e8m0_shuffle), padding 0x7F included; the kernel helper
    against the host formula; and reference_pack against aiter's sorted MX
    quant (fused_dynamic_mx_quant_moe_sort) on the valid bytes."""
    from aiter.ops.shuffle import shuffle_scale

    rows_out = []
    for R, G in [(256, 12), (512, 14), (256, 64), (512, 224), (100, 90), (33, 7)]:
        lay = Layout(R, G * 32, dtypes.fp4x2, 32, dtypes.fp8_e8m0, "mfma_native")
        assert lay.scales_shape == (-(-R // 32) * 32, -(-G // 8) * 8)
        src = torch.randint(0, 255, (R, G), dtype=torch.uint8)
        ref = shuffle_scale(src.view(dtypes.fp8_e8m0)).view(torch.uint8).flatten()
        rm = torch.full(lay.scales_shape, 0x7F, dtype=torch.uint8)
        rm[:R, :G] = src
        mine = to_mfma_native(rm, lay).flatten()
        assert torch.equal(mine, ref[: mine.numel()]), f"mfma_native R={R} G={G}"
        r = torch.arange(lay.scales_shape[0])[:, None]
        g = torch.arange(lay.scales_shape[1])[None, :]
        host = lay.scale_offset(r, g).int()
        dev = _offsets_on_device(*lay.scales_shape, lay.scales_shape[1])
        assert torch.equal(host, dev), f"device offsets R={R} G={G}"
        rows_out.append({"check": f"mfma_native R={R} G={G}", "exact": True})
    mx = [dtypes.fp8] + ([dtypes.fp4x2] if get_gfx_runtime() == "gfx950" else [])
    for dt in mx:
        for token, E, topk, inter in [(16, 32, 4, 512), (300, 64, 8, 2880)]:
            from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort

            ti, tw, _ = routing(token, E, topk)
            tm = route_to_tiles(ti, tw, num_experts=E, tile_m=32)
            lay = Layout(tm.rows, inter, dt, 32, dtypes.fp8_e8m0, "mfma_native")
            a2 = torch.randn((token, topk, inter), dtype=dtypes.bf16) * 2
            a2[0, 0].zero_()
            packed = reference_pack(a2, tm, lay)
            aq, asc = fused_dynamic_mx_quant_moe_sort(
                a2.view(token * topk, inter),
                tm.sorted_token_ids,
                tm.num_valid_ids,
                token,
                topk,
                32,
                quant_dtype=dt,
            )
            nv = int(tm.num_valid_ids[0])
            sid = tm.sorted_token_ids[:nv]
            tok, slot = sid & 0xFFFFFF, sid >> 24
            real = slot < topk
            rows = torch.nonzero(real).flatten()
            src = (tok[real] * topk + slot[real]).long()
            bad = []
            if not torch.equal(
                packed.values.view(torch.uint8)[rows],
                aq.view(torch.uint8).view(token * topk, -1)[src],
            ):
                bad.append("values")
            gg = torch.arange(lay.groups)[None, :]
            off = lay.scale_offset(rows[:, None], gg).flatten()
            mine = packed.scales.view(torch.uint8).flatten()[off]
            theirs = asc.view(torch.uint8).flatten()[off]
            if not torch.equal(mine, theirs):
                bad.append(f"scales ({int((mine != theirs).sum())} bytes)")
            assert not bad, f"reference_pack {dt} M={token} I={inter}: {bad}"
            rows_out.append(
                {"check": f"reference_pack {dt} M={token} I={inter}", "exact": True}
            )
    return rows_out


def test_intermediate(token=64, E=32, topk=4, inter_dim=512, tile_m=32):
    """per_1x128 layout shapes, and reference_pack against aiter's own quant."""
    spec = MoeSpec(
        experts=E,
        global_experts=E,
        model_dim=1024,
        inter_dim=inter_dim,
        act=aiter.ActivationType.Silu,
        quant=QuantType.per_1x128,
        w_dtype=dtypes.fp8,
        out_dtype=dtypes.bf16,
        g1u1=True,
        doweight_stage1=False,
        gate_mode=aiter.ops.flydsl.moe_common.GateMode.SEPARATED,
        w_shuffled=True,
    )
    lay = Layout.for_spec(spec, token, topk, tile_m)
    assert lay.values_shape == (lay.rows, inter_dim) and lay.dtype == dtypes.fp8
    assert lay.scales_shape == (lay.rows, inter_dim // 128)
    ti, tw, _ = routing(token, E, topk)
    tm = route_to_tiles(ti, tw, num_experts=E, tile_m=tile_m)
    a2 = torch.randn((token, topk, inter_dim), dtype=dtypes.bf16)
    packed = reference_pack(a2, tm, lay)
    q, s = aiter.get_hip_quant(QuantType.per_1x128)(a2, quant_dtype=dtypes.fp8)
    nv = int(tm.num_valid_ids[0])
    sid = tm.sorted_token_ids[:nv]
    tok, slot = sid & 0xFFFFFF, sid >> 24
    real = slot < topk
    rows = torch.nonzero(real).flatten()
    assert torch.equal(
        packed.values[rows].view(torch.uint8),
        q[tok[real], slot[real]].view(torch.uint8),
    ), "values follow sorted rows, bit for bit"
    assert torch.equal(packed.scales[rows], s[tok[real], slot[real]])
    return [{"check": "per_1x128 intermediate: shapes, sorted-row packing exact"}]


EDGE_CASES = [
    # token, E, topk, tile_m, case, mask, local
    (1, 256, 8, 32, "random", False, None),
    (8, 256, 8, 16, "random", True, None),
    (33, 257, 9, 64, "mixed", False, None),
    (64, 256, 8, 128, "onehot", False, None),
    (40, 64, 6, 32, "dup", False, None),
    (3000, 256, 8, 32, "dup", True, None),
    (2000, 128, 6, 16, "dup", False, None),
    (500, 256, 8, 64, "pad", False, None),
    (100, 256, 8, 32, "random", False, 50),
    (127, 40, 6, 16, "all_empty", False, None),
    (300, 513, 11, 32, "zipf", True, None),
    (2050, 385, 7, 64, "mixed", True, 2000),
    (4096, 256, 8, 32, "onehot", False, None),
    (8192, 128, 8, 128, "zipf", False, 5000),
    (16384, 512, 10, 32, "random", False, None),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier", help="tier2-sorting.csv: exactness and timing per row"
    )
    parser.add_argument("--time", action="store_true", help="time the --tier rows")
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument(
        "--quant-time",
        action="store_true",
        help="time fused sort+quant against Opus sort + aiter quant per --tier row",
    )
    parser.add_argument("--out", help="CSV for the --quant-time rows")
    parser.add_argument(
        "--inline-quant-time",
        action="store_true",
        help="time inline quant in a direct kernel against a separate launch, then exit",
    )
    args = parser.parse_args()
    gfx = get_gfx_runtime()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.info("skipped: %s not in %s", gfx, SUPPORTED_GFX)
        return
    if args.inline_quant_time:
        aiter.logger.info(
            "inline quant in a direct kernel:\n%s",
            pd.DataFrame(time_inline_quant()).to_markdown(index=False),
        )
        return
    rows = []
    names = ("token", "E", "topk", "tile_m", "case", "mask", "local")
    for case in EDGE_CASES:
        row = dict(zip(names, case))
        row.update(test_tile_map(*case))
        rows.append(row)
    aiter.logger.info(
        "tile map edge cases:\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )
    aiter.logger.info(
        "fused quant:\n%s", pd.DataFrame(test_fused_quant()).to_markdown(index=False)
    )
    aiter.logger.info(
        "dynamic per-tensor:\n%s",
        pd.DataFrame(test_dynamic_tensor()).to_markdown(index=False),
    )
    slots = pd.DataFrame(test_route_slot())
    aiter.logger.info(
        "direct-mode route_slot: %d cases exact (M x topk up to 256)", len(slots)
    )
    direct = pd.DataFrame(test_direct_grouping())
    aiter.logger.info(
        "direct-mode grouping: %d cases exact (M x topk up to 256; EP, -1 ids, "
        "num_local_tokens, repeats)",
        len(direct),
    )
    aiter.logger.info(
        "MX addressing:\n%s", pd.DataFrame(test_mx_index()).to_markdown(index=False)
    )
    aiter.logger.info(
        "eager host time:\n%s", pd.DataFrame(test_host_time()).to_markdown(index=False)
    )
    checks = test_graph_capture() + test_intermediate() + test_mfma_native()
    checks += test_pad_rows() + test_pad_rows_consumer() + test_repeat_keep_last()
    checks += test_path_boundary()
    aiter.logger.info("checks:\n%s", pd.DataFrame(checks).to_markdown(index=False))
    if args.tier:
        rows, qrows = [], []
        with open(args.tier) as f:
            tier = list(csv.DictReader(f))
        for r in tier:
            token, E, topk = int(r["token"]), int(r["expert"]), int(r["topk"])
            md = int(r["model_dim"] or 4096)
            mask = "expert_mask=1" in r["extra"]
            row = {
                "case": r["case_id"],
                "token": token,
                "model_dim": md,
                "E": E,
                "topk": topk,
                "mask": mask,
            }
            row.update(
                test_tile_map(token, E, topk, args.block, "random", mask, None, md)
            )
            if args.time:
                row.update(time_sorts(token, E, topk, mask, md, args.block))
            if args.quant_time:
                for q in time_sort_quant(token, E, topk, mask, md, args.block):
                    qrows.append({**row, **q})
            rows.append(row)
        aiter.logger.info("tier rows:\n%s", pd.DataFrame(rows).to_markdown(index=False))
        if qrows and args.out:
            pd.DataFrame(qrows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
