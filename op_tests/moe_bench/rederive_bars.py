# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Re-derive existing bar files from their candidate files under the bar exclusions (no GPU).

A bar is the fastest correct candidate of a case and mode. This reads each bar file and its
candidates file, takes the confirmed (``phase=confirm``) ``status=ok`` rows, applies
``exclusions`` (MOEBUG-064 tile_n=64 ASM fp8 stage 1, and stage-1 taint) and lists every point
whose winner or ``bar_us`` changes, with the shipped route rows and gate sets that sit on it.

* A bar row whose case has no candidate rows, and a bar file with no candidates file, are
  ``unverifiable``.
* Rows are read through ``schema.map_row`` (the header whose values fit), so rows written under
  another column set are still read; rows no header fits unambiguously are counted and not used.
* The derivation is first checked against the file: where the fastest ok confirmed row is not
  the recorded ``bar_us`` the row says so (``derived_matches`` false) and its change is
  measured against the derived bar.

    python -m op_tests.moe_bench.rederive_bars --root DIR [--root DIR] [--pair BAR=CANDS,CANDS]
        [--routes-tree TREE] --csv changes.csv --md report.md
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

from . import bar_runner, exclusions, schema

csv.field_size_limit(1 << 28)

ROUTE_FILES = {"mi300x.csv": "gfx942", "mi325x.csv": "gfx942", "mi355x.csv": "gfx950"}


def read_mapped(path, kind):
    """([row dicts], rejected count) of a bar or candidate CSV, each row under the header
    whose values fit (the file's, the current columns, an earlier head's)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], 0
    cur = bar_runner.BAR_COLS if kind == "bar" else bar_runner.CAND_COLS
    known = [rows[0], list(cur), *bar_runner._history(kind)]
    out, rej = [], 0
    for r in rows[1:]:
        m = schema.map_row(r, known)
        if m is None:
            rej += 1
            continue
        out.append(dict(zip(m[0], r, strict=False)))
    return out, rej


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def meta_of(bar):
    mp = bar[:-4] + ".meta.json"
    try:
        with open(mp) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def discover(roots):
    """[(bar csv, [candidate csvs])] for every bar-*.csv under the roots; the candidates file
    is the sibling candidates-<same name>.csv (none: the list is empty)."""
    seen, out = set(), []
    for root in roots:
        for bar in sorted(
            glob.glob(os.path.join(root, "**", "bar-*.csv"), recursive=True)
        ):
            real = os.path.realpath(bar)
            if real in seen:
                continue
            seen.add(real)
            cand = os.path.join(
                os.path.dirname(bar), "candidates-" + os.path.basename(bar)[4:]
            )
            out.append((bar, [cand] if os.path.exists(cand) else []))
    return out


def derive(bars, cands):
    """Per (case, mode) of the bar rows: old and re-derived winners. Returns (rows,
    unverifiable) where rows are dicts with the old (recorded), derived and new bars."""
    by_case = {}
    for c in cands:
        by_case.setdefault(c.get("case_id"), []).append(c)
    res, unver = [], []
    seen = set()
    for b in bars:
        key = (b.get("case_id"), b.get("mode"))
        if key in seen or b.get("mode") not in ("warm", "cold"):
            continue
        seen.add(key)
        if b.get("status") != "ok":
            continue
        rows = by_case.get(b["case_id"])
        if not rows:
            unver.append((b["case_id"], b["mode"], "no candidate rows for the case"))
            continue
        ok = [
            c
            for c in rows
            if c.get("mode") == b["mode"]
            and c.get("phase") == "confirm"
            and c.get("status") == "ok"
            and _num(c.get("median_us")) is not None
        ]
        if not ok:
            unver.append(
                (b["case_id"], b["mode"], "no confirmed ok candidate rows in the mode")
            )
            continue
        # a stage 1 that failed accuracy in any pair of the case, in any mode or phase, taints
        entries = [
            (
                c["cand_id"],
                exclusions.stage1_of(c["cand_id"], c.get("kernel1", "")),
                "fail" if c.get("status") == "fail" else "ok",
            )
            for c in rows
            if c.get("status") in ("ok", "fail")
        ]
        skip = exclusions.excluded(entries)
        best0 = min(ok, key=lambda c: float(c["median_us"]))
        keep = [c for c in ok if c["cand_id"] not in skip]
        best1 = min(keep, key=lambda c: float(c["median_us"])) if keep else None
        res.append(
            {
                "bar": b,
                "old_id": b.get("bar_id", ""),
                "old_us": _num(b.get("bar_us")),
                "derived_id": best0["cand_id"],
                "derived_us": float(best0["median_us"]),
                "new_id": best1["cand_id"] if best1 else "",
                "new_us": float(best1["median_us"]) if best1 else None,
                "why": skip.get(best0["cand_id"], ""),
                "n_excluded": sum(1 for c in ok if c["cand_id"] in skip),
            }
        )
    return res, unver


def load_route_files(specs):
    """[(file name, gfx, row)] of route CSVs given as ``path:gfx`` (rows with a ``kernel``)."""
    out = []
    for spec in specs:
        path, _, gfx = spec.partition(":")
        try:
            with open(path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if "m_min" in r]
        except OSError:
            continue
        out += [
            (
                os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path),
                gfx,
                r,
            )
            for r in rows
        ]
    return out


def load_routes(trees):
    """[(arch file, gfx, row)] of the shipped route tables under each tree."""
    seen, out = set(), []
    for t in trees:
        for name, gfx in ROUTE_FILES.items():
            p = os.path.join(t, "aiter", "moe", "routes", name)
            if not os.path.exists(p):
                continue
            with open(p, newline="") as f:
                for r in csv.DictReader(f):
                    k = (name, tuple(sorted(r.items())))
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append((name, gfx, r))
    return out


def route_hits(bar, routes, gfx):
    hits = []
    for name, rgfx, r in routes:
        if gfx and rgfx != gfx:
            continue
        try:
            if (
                r["family"] == bar.get("family")
                and int(r["model_dim"]) == int(bar["model_dim"])
                and int(r["inter_dim"]) == int(bar["inter_dim"])
                and int(r["experts"]) == int(bar["expert"])
                and int(r["topk"]) == int(bar["topk"])
                and r["act"].lower() == str(bar["act_type"]).lower()
                and int(r["m_min"]) <= int(bar["token"]) <= int(r["m_max"])
            ):
                hits.append(f"{name}:{r['kernel']}")
        except (KeyError, ValueError, TypeError):
            continue
    return hits


COLS = [
    "bar_set", "harness", "gfx", "case_id", "family", "token", "mode", "old_id", "old_us",
    "derived_matches", "new_id", "new_us", "delta_pct", "why_old_winner_excluded", "n_excluded",
    "shipped_routes", "gate_set", "note",
]  # fmt: skip


def run(roots, pairs, route_trees, rel="", route_files=()):
    routes = load_routes(route_trees) + load_route_files(route_files)
    sets = discover(roots) + [(b, c) for b, c in pairs]
    changes, unverifiable, summary = [], [], []
    for bar, cand_paths in sets:
        name = os.path.relpath(bar, rel) if rel and bar.startswith(rel) else bar
        meta = meta_of(bar)
        if not cand_paths:
            unverifiable.append((name, "", "", "no candidates file"))
            summary.append(
                {
                    "set": name,
                    "cases": "?",
                    "rows": "?",
                    "changed": "?",
                    "unverifiable": "file",
                }
            )
            continue
        bars, rej_b = read_mapped(bar, "bar")
        cands, rej_c = [], 0
        for cp in cand_paths:
            rows, rj = read_mapped(cp, "cand")
            cands += rows
            rej_c += rj
        derived, unver = derive(bars, cands)
        gfx = meta.get("gfx") or next((b.get("gfx") for b in bars if b.get("gfx")), "")
        harness = meta.get("harness_sha") or next(
            (b.get("harness_sha") for b in bars if schema.column_ok("harness_sha", b.get("harness_sha") or "") and b.get("harness_sha")), ""
        )  # fmt: skip
        n_changed = 0
        for d in derived:
            b = d["bar"]
            matches = d["old_us"] is not None and abs(d["derived_us"] - d["old_us"]) <= 1e-6 * max(1.0, abs(d["old_us"]))  # fmt: skip
            if d["new_id"] == d["derived_id"] and d["new_us"] == d["derived_us"]:
                continue
            n_changed += 1
            delta = (
                (d["new_us"] - d["derived_us"]) / d["derived_us"] * 100
                if d["new_us"] is not None
                else None
            )
            note = ""
            if not matches:
                note = (
                    f"derived bar {d['derived_us']:.2f} us differs from the recorded {d['old_us']}: "
                    "change is against the derived bar"
                )
            if not d["new_id"]:
                note = (
                    note + "; " if note else ""
                ) + "no eligible candidate left: no bar"
            changes.append(
                dict(  # noqa: C408
                    bar_set=name, harness=harness, gfx=gfx, case_id=b["case_id"],
                    family=b.get("family", ""), token=b.get("token", ""), mode=b["mode"],
                    old_id=d["derived_id"], old_us=f"{d['derived_us']:.4f}",
                    derived_matches=str(matches), new_id=d["new_id"],
                    new_us="" if d["new_us"] is None else f"{d['new_us']:.4f}",
                    delta_pct="" if delta is None else f"{delta:.2f}",
                    why_old_winner_excluded=d["why"], n_excluded=d["n_excluded"],
                    shipped_routes="; ".join(route_hits(b, routes, gfx)),
                    gate_set="gate" if "gate" in os.path.basename(bar) else "", note=note,
                )
            )  # fmt: skip
        for case_id, mode, why in unver:
            unverifiable.append((name, case_id, mode, why))
        summary.append(
            {
                "set": name,
                "harness": harness,
                "gfx": gfx,
                "rows": len(derived),
                "changed": n_changed,
                "unverifiable": len(unver),
                "unreadable_rows": rej_b + rej_c,
            }
        )
    return changes, unverifiable, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument(
        "--pair",
        action="append",
        default=[],
        help="BAR=CANDS[,CANDS...] for a bar file named otherwise",
    )
    ap.add_argument("--routes-tree", action="append", default=[])
    ap.add_argument(
        "--routes", action="append", default=[], help="route CSV as path[:gfx]"
    )
    ap.add_argument("--rel", default="")
    ap.add_argument("--csv")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    pairs = []
    for p in a.pair:
        bar, cands = p.split("=", 1)
        pairs.append((bar, cands.split(",")))
    changes, unver, summary = run(a.root, pairs, a.routes_tree, a.rel, a.routes)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(changes)
    lines = [
        "| bar set | harness | gfx | derived rows | changed | unverifiable rows | unreadable rows |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['set']} | {s.get('harness', '')} | {s.get('gfx', '')} | {s['rows']} | {s['changed']} | {s['unverifiable']} | {s.get('unreadable_rows', '')} |"
        )
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text + "\n")
    print(
        f"{len(changes)} changed points, {len(unver)} unverifiable rows",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
