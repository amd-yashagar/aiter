# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter.moe core pieces against aiter: the sort and tile map (exact against
Opus, with its zero-fill), graph capture, timing, and the intermediate layout."""

import argparse
import csv

import pandas as pd
import torch

import aiter
from aiter import QuantType, dtypes
from aiter.fused_moe import _flydsl_moe_sorting, _moe_sorting_impl, fused_topk
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.moe import MoeSpec
from aiter.ops.flydsl.moe_next.core.intermediate import Layout, reference_pack
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
    for an expert repeated within one token, aiter keeps one slot, and which
    one is a cross-wave LDS write race when the token's slots span two waves;
    ours keeps the last slot."""
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
    """us per call, captured 20 per graph, for the new sort and today's sorts,
    each zero-filling an [M, model_dim] bf16 output as an atomic stage 2 needs."""
    ti, tw, em = routing(token, E, topk, mask=mask)
    buf = torch.empty((token, model_dim), dtype=dtypes.bf16)
    bf16 = dtypes.bf16
    fns = {
        "new": lambda: route_to_tiles(
            ti, tw, num_experts=E, tile_m=tile_m, expert_mask=em, zero_output=buf
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
    args = parser.parse_args()
    gfx = get_gfx_runtime()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.info("skipped: %s not in %s", gfx, SUPPORTED_GFX)
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
        "eager host time:\n%s", pd.DataFrame(test_host_time()).to_markdown(index=False)
    )
    checks = test_graph_capture() + test_intermediate()
    aiter.logger.info("checks:\n%s", pd.DataFrame(checks).to_markdown(index=False))
    if args.tier:
        rows = []
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
            rows.append(row)
        aiter.logger.info("tier rows:\n%s", pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
