# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Candidates a bar must not count (pure data, shared by the worker and the offline audit).

* MOEBUG-064: the ASM fp8 stage-1 kernels with tile_n = 64
  (``fmoe_stage1_bf16_pertokenFp8_g1u1_<tile_m>x64_...``) give wrong results, some at every
  launch and some only at large M, and the single accuracy run of a tuner or of a bar can
  miss the intermittent ones. A candidate whose stage 1 is one of them is not eligible, on
  every arch; dispatch is not eligible either when its route's stage 1 is one.
* Unsafe kernels (``unsafe_stage1.csv``): every ASM fp8 stage-1 kernel that gave wrong results
  at any shape in the verification runs. They are nondeterministic (identical inputs give
  different wrong blocks across replays), so a pair passing in a bar run proves nothing: the
  kernel is out of all bars at every shape on its arch and on the archs sharing its ``.co``.
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


_MANGLE = re.compile(r"^_ZN5aiter\d+")
_STAGE1_PLAIN = re.compile(r"fmoe_stage1_bf16_pertokenFp8_g1u1_(.+)")
ONE_STAGE_PT_FP8 = "fmoe_bf16_pertokenFp8_g1u1_"
ONE_STAGE_BLOCKSCALE = "fmoe_bf16_blockscaleFp8_g1u1_"


def kernel_key(name):
    """The list key of an ASM kernel name, or None for a name that is not an ASM fmoe kernel.
    The 2-stage plain per-token fp8 stage 1 is keyed by its tile (``128x128_pf2`` of
    ``..._pertokenFp8_g1u1_128x128_pf2E``, the form the verification files use); any other
    ASM fmoe kernel (1-stage ``fmoe_bf16_*``, other stage-1 families) by its whole
    demangled name without the trailing ``E``, so ``doweight`` and ``blockscale`` binaries
    are never taken for the plain kernel of the same tile."""
    d = _MANGLE.sub("", name or "")
    d = d.removesuffix("E")
    m = _STAGE1_PLAIN.fullmatch(d)
    if m:
        return m.group(1)
    return d if d.startswith(("fmoe_bf16_", "fmoe_stage1_bf16_")) else None


def stage1_tile(name):
    """Alias of ``kernel_key`` (the 2-stage plain tile for that family)."""
    return kernel_key(name)


UNSAFE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "unsafe_stage1.csv"
)
UNSAFE_COLS = [
    "gfx",
    "stage1_tile",
    "also_on",
    "source",
    "row_id",
    "evidence",
    "status",
]


@functools.lru_cache(maxsize=1)
def unsafe_table():
    """{(gfx, tile): "<source> <row>: <evidence>"} from ``unsafe_stage1.csv``, the one list of
    ASM fp8 stage-1 kernels that gave wrong results at any shape. The kernels are
    nondeterministic (identical inputs, different wrong blocks across replays: a bar run that
    passed proves nothing), so a listed kernel is out of every bar at every shape and M on its
    arch, and on every arch in ``also_on`` (archs that share its ``.co``). Append new verdicts
    with ``python -m op_tests.moe_bench.exclusions add|ingest``."""
    out = {}
    try:
        with open(UNSAFE_CSV, newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("status") or "exclude") != "exclude":
                    continue  # watch rows are recorded, not excluded
                why = f"{r['source']} {r['row_id']} ({r['gfx']}: {r['evidence']})"
                for g in [r["gfx"], *filter(None, (r.get("also_on") or "").split(","))]:
                    out.setdefault((g.strip(), r["stage1_tile"]), why)
    except (OSError, KeyError, ValueError):
        pass
    return out


def unsafe_reason(gfx, stage1):
    """Why the ASM kernel ``stage1`` (a 2-stage stage 1 or a 1-stage kernel) is not allowed on
    ``gfx`` (any shape), or ""."""
    tile = kernel_key(stage1)
    if tile is None or not gfx:
        return ""
    hit = unsafe_table().get((gfx, tile))
    return (
        f"MOEBUG-064 unsafe kernel: stage 1 {tile} gave wrong results (nondeterministic) in {hit}"
        if hit
        else ""
    )


def append_verdicts(rows, path=None):
    """Append ``rows`` (dicts with UNSAFE_COLS keys) that are not listed yet for their gfx
    and tile; returns the number added. The table is read again afterwards."""
    path = path or UNSAFE_CSV
    have = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            have = {
                (r["gfx"], r["stage1_tile"], r.get("status") or "exclude")
                for r in csv.DictReader(f)
            }
    new = []
    for r in rows:
        r = {**r, "stage1_tile": r["stage1_tile"].rstrip("E")}
        key = (
            r["gfx"],
            r["stage1_tile"],
        )  # a watch entry is not re-added as an exclusion: edit its status
        if key not in have:
            have.add(key)
            new.append({c: r.get(c, "") for c in UNSAFE_COLS})
    if new:
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=UNSAFE_COLS)
            if not exists:
                w.writeheader()
            w.writerows(new)
        unsafe_table.cache_clear()
    return len(new)


def ingest(path, gfx, also_on="", prefix=None):
    """The FAIL rows of a verification result file as table rows: ``results/smci355-moebug064-
    gfx950.csv`` (``row_id``, ``stage1_kernel``, ``verdict``) or ``meta-moebug064-tuned-rows.csv``
    (``row_id``, ``stage1_tile``, ``verdict`` FAIL*). Returns the rows."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            tile = (r.get("stage1_kernel") or r.get("stage1_tile") or "").rstrip("E")
            if not r.get("verdict", "").startswith("FAIL") or not tile:
                continue
            kind = r.get("kind", "")
            if prefix is not None or "1stage" in kind:
                # a 1-stage row names the kernel without its family: tkw1_silu_1tg_32x64
                tile = (
                    prefix
                    if prefix is not None
                    else (
                        ONE_STAGE_BLOCKSCALE
                        if "blockscale" in kind
                        else ONE_STAGE_PT_FP8
                    )
                ) + tile
            ev = (
                f"uniform {r.get('uniform_bad_reps') or r.get('mi300x_uniform_fused_bad', '')}, "
                f"hot1 {r.get('hot1_bad_reps') or r.get('mi300x_hot1_fused_bad', '')} bad reps"
            )
            out.append(
                {
                    "gfx": gfx,
                    "stage1_tile": tile,
                    "also_on": also_on,
                    "source": os.path.basename(os.path.dirname(path))
                    + "/"
                    + os.path.basename(path),
                    "row_id": r["row_id"],
                    "evidence": ev,
                }
            )
    return out


def stage1_of(cand_id, kernel1=""):
    """The ASM kernel to judge of a candidate: the second field of a forced-pair id
    (``fused_moe|<stage 1>|<stage 2>|bm..``), the kernel of a 1-stage id
    (``fused_moe|1stage|<kernel>|bm..``), else the route's ``kernel1`` (dispatch)."""
    parts = (cand_id or "").split("|")
    if (
        parts[0] == "fused_moe"
        and len(parts) >= 6
        and parts[1] not in ("1stage", "graph")
    ):
        return parts[1]
    if parts[0] == "fused_moe" and len(parts) >= 3 and parts[1] == "1stage":
        return parts[2]
    return kernel1 or ""


def tile64(stage1):
    return bool(TILE64.search(stage1 or ""))


def excluded(entries, gfx=None):
    """``entries``: iterable of (key, kernel name, outcome) for one case, outcome "fail" for a
    candidate that failed the accuracy check. Returns {key: reason} for every entry that must
    not count: a tile_n=64 stage 1, a kernel on the unsafe list for ``gfx`` (2-stage stage 1 or
    1-stage kernel, any shape), or a stage 1 that failed accuracy in some pair of the case
    (taint: 2-stage pairs only, a 1-stage id is not a pair of a stage 1 with others)."""
    entries = list(entries)
    tainted = {}
    for key, s1, outcome in entries:
        if (
            outcome == "fail"
            and s1
            and "|1stage|" not in str(key)
            and s1 not in tainted
        ):
            tainted[s1] = key
    out = {}
    for key, s1, _ in entries:
        if tile64(s1):
            out[key] = MOEBUG_064
        elif unsafe_reason(gfx, s1):
            out[key] = unsafe_reason(gfx, s1)
        elif s1 in tainted and "|1stage|" not in str(key):
            out[key] = f"stage-1 taint: {s1} failed accuracy in {tainted[s1]}"
    return out


def note(reasons):
    """The ``bar_exclusions`` text for a row from {key: reason}."""
    if not reasons:
        return ""
    n064 = sum(1 for r in reasons.values() if r == MOEBUG_064)
    nunsafe = sum(
        1 for r in reasons.values() if r.startswith("MOEBUG-064 unsafe kernel")
    )
    taint = {
        r.split(": ", 1)[1].split(" failed")[0]
        for r in reasons.values()
        if r.startswith("stage-1 taint")
    }
    parts = []
    if n064:
        parts.append(f"{n064} candidates excluded, {MOEBUG_064}")
    if nunsafe:
        parts.append(f"{nunsafe} candidates excluded, MOEBUG-064 unsafe kernel")
    if taint:
        n = len(reasons) - n064 - nunsafe
        parts.append(
            f"{n} candidates excluded by stage-1 taint ({len(taint)} stage-1 kernels failed accuracy)"
        )
    return "; ".join(parts)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="the MOEBUG-064 unsafe stage-1 kernel list"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.set_defaults(fn="list")
    p = sub.add_parser("add")
    p.add_argument("--gfx", required=True)
    p.add_argument(
        "--tile",
        required=True,
        help="2-stage plain tile (64x128_2tg_pf3) or a whole 1-stage kernel name "
        "(fmoe_bf16_pertokenFp8_g1u1_tkw1_silu_1tg_32x64)",
    )
    p.add_argument(
        "--also-on", default="", help="archs sharing the .co, comma separated"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--row", required=True)
    p.add_argument("--evidence", default="")
    p.set_defaults(fn="add")
    p = sub.add_parser("ingest")
    p.add_argument("file")
    p.add_argument("--gfx", required=True)
    p.add_argument("--also-on", default="")
    p.add_argument(
        "--prefix",
        help="kernel family prefix of 1-stage rows (default: from the row kind)",
    )
    p.set_defaults(fn="ingest")
    a = ap.parse_args(argv)
    if a.fn == "list":
        for (g, t), why in sorted(unsafe_table().items()):
            print(g, t, why)
        return 0
    if a.fn == "add":
        rows = [
            {"gfx": a.gfx, "stage1_tile": a.tile.rstrip("E"), "also_on": a.also_on,
             "source": a.source, "row_id": a.row, "evidence": a.evidence}
        ]  # fmt: skip
    else:
        rows = ingest(a.file, a.gfx, a.also_on, a.prefix)
    print(f"{append_verdicts(rows)} added, {len(rows)} given")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
