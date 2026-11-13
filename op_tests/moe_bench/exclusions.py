# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Candidates a bar must not count (pure data, shared by the worker and the offline audit).

* MOEBUG-064: the ASM fp8 stage-1 kernels with tile_n = 64
  (``fmoe_stage1_bf16_pertokenFp8_g1u1_<tile_m>x64_...``) give wrong results, some at every
  launch and some only at large M, and the single accuracy run of a tuner or of a bar can
  miss the intermittent ones. A candidate whose stage 1 is one of them is not eligible, on
  every arch; dispatch is not eligible either when its route's stage 1 is one.
* Stage-1 taint: when any pair with a given stage-1 kernel fails the accuracy check in a
  case, every pair with that stage 1 is excluded for that case (a stage 1 that is wrong for
  some stage-2 partner is not trusted with the others), dispatch included.

The bar row's ``bar_exclusions`` column names what was excluded.
"""

import csv
import functools
import os
import re

TILE64 = re.compile(r"fmoe_stage1_bf16_pertokenFp8_g1u1_\d+x64_")
MOEBUG_064 = "MOEBUG-064: ASM fp8 stage-1 kernel with tile_n=64 (wrong results, some at every launch)"


_TILE = re.compile(r"pertokenFp8_(?:doweight_)?g1u1_(\S+?)E?$")


def stage1_tile(name):
    """``128x128_pf2`` of ``..._pertokenFp8_g1u1_128x128_pf2E`` (None for any other name)."""
    m = _TILE.search(name or "")
    return m.group(1) if m else None


@functools.lru_cache(maxsize=1)
def unsafe_table():
    """{(model_dim, inter_dim, expert, topk, tile): "<source> <row>: <evidence>"} from
    ``unsafe_stage1.csv``: the stage-1 kernels that gave wrong results at a shape in the
    MOEBUG-064 verification runs (``results/smci355-moebug064-gfx950.csv`` FAIL rows,
    ``meta-moebug064-tuned-rows.csv`` FAIL rows), including shipped non-tile-64 kernels with
    intermittent garbage. Such a pair is not eligible at that shape, at any M, on any arch.
    """
    out = {}
    path = os.path.join(os.path.dirname(__file__), "unsafe_stage1.csv")
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = (
                    int(r["model_dim"]),
                    int(r["inter_dim"]),
                    int(r["expert"]),
                    int(r["topk"]),
                    r["stage1_tile"],
                )
                out.setdefault(
                    key, f"{r['source']} {r['row_id']} ({r['gfx']}: {r['evidence']})"
                )
    except (OSError, KeyError, ValueError):
        pass
    return out


def unsafe_reason(shape, stage1):
    """Why ``stage1`` is not allowed at ``shape`` = (model_dim, inter_dim, expert, topk), or ""."""
    tile = stage1_tile(stage1)
    if tile is None or shape is None:
        return ""
    try:
        key = (*(int(v) for v in shape), tile)
    except (TypeError, ValueError):
        return ""
    hit = unsafe_table().get(key)
    return (
        f"MOEBUG-064 unsafe pair: stage 1 {tile} gave wrong results at this shape in {hit}"
        if hit
        else ""
    )


def stage1_of(cand_id, kernel1=""):
    """The stage-1 kernel name of a candidate: the second field of a forced-pair id
    (``fused_moe|<stage 1>|<stage 2>|bm..``), else the route's ``kernel1`` (dispatch).
    """
    parts = (cand_id or "").split("|")
    if (
        parts[0] == "fused_moe"
        and len(parts) >= 6
        and parts[1] not in ("1stage", "graph")
    ):
        return parts[1]
    return kernel1 or ""


def tile64(stage1):
    return bool(TILE64.search(stage1 or ""))


def excluded(entries, shape=None):
    """``entries``: iterable of (key, stage1 name, outcome) for one case, outcome "fail" for a
    candidate that failed the accuracy check. Returns {key: reason} for every entry that must
    not count: tile_n=64 stage 1, or a stage 1 that failed accuracy in some pair of the case.
    """
    entries = list(entries)
    tainted = {}
    for key, s1, outcome in entries:
        if outcome == "fail" and s1 and s1 not in tainted:
            tainted[s1] = key
    out = {}
    for key, s1, _ in entries:
        if tile64(s1):
            out[key] = MOEBUG_064
        elif unsafe_reason(shape, s1):
            out[key] = unsafe_reason(shape, s1)
        elif s1 in tainted:
            out[key] = f"stage-1 taint: {s1} failed accuracy in {tainted[s1]}"
    return out


def note(reasons):
    """The ``bar_exclusions`` text for a row from {key: reason}."""
    if not reasons:
        return ""
    n064 = sum(1 for r in reasons.values() if r == MOEBUG_064)
    nunsafe = sum(1 for r in reasons.values() if r.startswith("MOEBUG-064 unsafe pair"))
    taint = {
        r.split(": ", 1)[1].split(" failed")[0]
        for r in reasons.values()
        if r.startswith("stage-1 taint")
    }
    parts = []
    if n064:
        parts.append(f"{n064} candidates excluded, {MOEBUG_064}")
    if nunsafe:
        parts.append(
            f"{nunsafe} candidates excluded, MOEBUG-064 unsafe pair at this shape"
        )
    if taint:
        n = len(reasons) - n064 - nunsafe
        parts.append(
            f"{n} candidates excluded by stage-1 taint ({len(taint)} stage-1 kernels failed accuracy)"
        )
    return "; ".join(parts)
