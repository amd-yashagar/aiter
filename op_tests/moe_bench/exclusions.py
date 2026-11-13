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

import re

TILE64 = re.compile(r"fmoe_stage1_bf16_pertokenFp8_g1u1_\d+x64_")
MOEBUG_064 = "MOEBUG-064: ASM fp8 stage-1 kernel with tile_n=64 (wrong results, some at every launch)"


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


def excluded(entries):
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
        elif s1 in tainted:
            out[key] = f"stage-1 taint: {s1} failed accuracy in {tainted[s1]}"
    return out


def note(reasons):
    """The ``bar_exclusions`` text for a row from {key: reason}."""
    if not reasons:
        return ""
    n064 = sum(1 for r in reasons.values() if r == MOEBUG_064)
    taint = {
        r.split(": ", 1)[1].split(" failed")[0]
        for r in reasons.values()
        if r.startswith("stage-1 taint")
    }
    parts = []
    if n064:
        parts.append(f"{n064} candidates excluded, {MOEBUG_064}")
    if taint:
        n = len(reasons) - n064
        parts.append(
            f"{n} candidates excluded by stage-1 taint ({len(taint)} stage-1 kernels failed accuracy)"
        )
    return "; ".join(parts)
