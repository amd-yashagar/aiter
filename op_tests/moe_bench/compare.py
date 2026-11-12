# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``compare`` and ``merge``: speedups, regressions, correctness, lost support.

A case regresses when it is slower than its baseline by more than
max(2%, 3x its run-to-run spread). The spread comes from the merged baseline
(``merge`` of the three tier-1 repeats); without one, half the p10-p90 width of the
baseline run is used. When the drift check flagged anchors, a regression is
``unconfirmed`` until an A/B interleaved re-run on both trees confirms it.
"""

import collections
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path

from . import runner, schema

BAD_STATUS = {"unsupported", "crash", "timeout", "error", "oom"}


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "tier1.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def key(r):
    return (r["sig"], r["mode"])


def bucket(tok):
    t = int(_f(tok, 0))
    for hi, name in (
        (16, "1-16"),
        (128, "17-128"),
        (1024, "129-1024"),
        (8192, "1025-8192"),
    ):
        if t <= hi:
            return name
    return ">8192"


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def band(base):
    spread = _f(base.get("spread"))
    if spread is None:
        p10, p90, med = (
            _f(base.get("p10_us")),
            _f(base.get("p90_us")),
            _f(base.get("median_us")),
        )
        spread = (p90 - p10) / 2 / med if med and p10 and p90 else 0.0
    return max(0.02, 3.0 * spread), spread


def fp_mismatch(b, n):
    try:
        fb, fn = json.loads(b.get("fp") or "{}"), json.loads(n.get("fp") or "{}")
    except ValueError:
        return False
    if not fb or not fn or fb.get("sha") == fn.get("sha"):
        return False
    rel = abs(fb["l2"] - fn["l2"]) / max(fb["l2"], 1e-9)
    worst = 0.0
    for rb, rn in zip(fb["sample"], fn["sample"]):
        for x, y in zip(rb, rn):
            worst = max(worst, abs(x - y) / (fb["absmean"] + 1e-6))
    return rel > 0.02 or worst > 0.5


def compare_rows(base_rows, new_rows):
    base = {key(r): r for r in base_rows}
    out = []
    for n in new_rows:
        b = base.get(key(n))
        row = {
            "case_id": n["case_id"],
            "sig": n["sig"],
            "mode": n["mode"],
            "family": n["family"],
            "token": n["token"],
            "gpu": n.get("gpu", ""),
            "route_family": n.get("route_family", ""),
            "dtype": f"{n['q_dtype_a']}/{n['q_dtype_w']}/{n['q_type']}",
            "base_status": b["status"] if b else "missing",
            "new_status": n["status"],
            "base_us": "",
            "new_us": "",
            "speedup": "",
            "band": "",
            "verdict": "",
        }
        if b is None:
            row["verdict"] = "no-baseline"
        elif b["status"] == "ok" and n["status"] in BAD_STATUS:
            row["verdict"] = f"lost-support:{n['status']}"
        elif n["status"] == "fail" or n.get("ref_pass") == "False":
            row["verdict"] = "correctness-fail"
        elif b["status"] == "ok" and n["status"] == "ok":
            bu, nu = _f(b["median_us"]), _f(n["median_us"])
            bd, _ = band(b)
            row.update(base_us=bu, new_us=nu, speedup=bu / nu, band=bd)
            if fp_mismatch(b, n):
                row["verdict"] = "fingerprint-mismatch"
            elif nu > bu * (1 + bd):
                row["verdict"] = "regression"
            elif nu < bu * (1 - bd):
                row["verdict"] = "faster"
            else:
                row["verdict"] = "same"
        else:
            row["verdict"] = "n/a"
        out.append(row)
    return out


def summary_tables(rows):
    groups = {
        "gpu": lambda r: r["gpu"],
        "backend": lambda r: r["route_family"],
        "dtype/quant": lambda r: r["dtype"],
        "token bucket": lambda r: bucket(r["token"]),
    }
    lines = []
    for name, fn in groups.items():
        for mode in sorted({r["mode"] for r in rows}):
            acc = collections.defaultdict(list)
            for r in rows:
                if r["mode"] == mode and r["speedup"] != "":
                    acc[fn(r)].append(r["speedup"])
            if not acc:
                continue
            lines.append(f"geomean speedup by {name} ({mode}):")
            for k, xs in sorted(acc.items()):
                lines.append(f"  {k or '-':50s} {geomean(xs):7.4f}  n={len(xs)}")
    return lines


def ab_confirm(cands, args):
    """Re-run candidate cases interleaved base/new; return {sig,mode: ratio}."""
    from argparse import Namespace

    ratios = collections.defaultdict(list)
    ids = {r["case_id"] for r in cands}
    tmp = args.tmpdir or "/tmp/moe-bench-tmp"
    os.makedirs(tmp, exist_ok=True)
    medians = {
        "base": collections.defaultdict(list),
        "new": collections.defaultdict(list),
    }
    for rnd in range(args.rounds):
        for side, tree in (("base", args.ab_base_tree), ("new", args.ab_new_tree)):
            out = os.path.join(tmp, f"ab-{side}-{rnd}.csv")
            ns = Namespace(
                aiter=tree,
                cases=args.cases,
                filter=None,
                modes="warm,cold",
                out=out,
                resume=False,
                shard=None,
                jobs=1,
                launcher=args.launcher,
                node=None,
                tmpdir=tmp,
                case_timeout=1800,
                setup_timeout=3600,
                baseline_tree=None,
                baseline_results=None,
                drift_anchors=0,
            )
            ns.ids = ids
            runner.cmd_run(ns)
            for r in schema.read_csv(out):
                if r["status"] == "ok":
                    medians[side][key(r)].append(float(r["median_us"]))
    for k, nv in medians["new"].items():
        bv = medians["base"].get(k)
        if bv:
            ratios[k] = statistics.median(nv) / statistics.median(bv)
    return ratios


def cmd_compare(args):
    base_rows, new_rows = load(args.base), load(args.new)
    rows = compare_rows(base_rows, new_rows)
    drift = {}
    if args.drift and os.path.exists(args.drift):
        drift = json.loads(Path(args.drift).read_text())
    cands = [r for r in rows if r["verdict"] == "regression"]
    if cands and drift.get("flagged") is not None and (drift["flagged"] or args.ab):
        if args.ab_base_tree and args.ab_new_tree:
            ratios = ab_confirm(cands, args)
            for r in cands:
                ratio = ratios.get((r["sig"], r["mode"]))
                if ratio is None:
                    r["verdict"] = "regression-unconfirmed"
                elif ratio > 1 + r["band"]:
                    r["verdict"] = "regression"
                    r["speedup"] = 1 / ratio
                else:
                    r["verdict"] = "same-after-ab"
                    r["speedup"] = 1 / ratio
        else:
            for r in cands:
                r["verdict"] = "regression-unconfirmed"
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=list(rows[0].keys()) if rows else ["case_id"]
            )
            w.writeheader()
            w.writerows(rows)
    counts = collections.Counter(r["verdict"] for r in rows)
    print("verdicts:", dict(counts))
    if drift:
        print(
            f"drift: {len(drift.get('flagged', []))} anchors moved over {drift.get('threshold')}"
        )
    for line in summary_tables(rows):
        print(line)
    bad = [
        r
        for r in rows
        if r["verdict"].startswith(
            ("regression", "correctness", "lost-support", "fingerprint")
        )
    ]
    for r in bad[:50]:
        print(
            f"  {r['verdict']}: {r['case_id']} {r['family']} M={r['token']} {r['mode']} "
            f"base={r['base_us']} new={r['new_us']}"
        )
    return 1 if bad else 0


def cmd_merge(args):
    """Merge repeats: median of medians, spread = (max-min)/median across repeats."""
    runs = [load(p) for p in args.runs]
    by = collections.defaultdict(list)
    for run in runs:
        for r in run:
            by[(r["case_id"], r["mode"])].append(r)
    cols = runner.RESULT_COLS + ["spread", "noise_band", "repeats"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rs in by.values():
            ok = [r for r in rs if r["status"] == "ok"]
            row = dict(ok[0] if ok else rs[0])
            if ok:
                meds = [float(r["median_us"]) for r in ok]
                med = statistics.median(meds)
                spread = (max(meds) - min(meds)) / med if med else 0.0
                row.update(
                    median_us=med,
                    spread=spread,
                    noise_band=max(0.02, 3 * spread),
                    repeats=len(ok),
                )
            else:
                row.update(spread="", noise_band="", repeats=0)
            w.writerow(row)
    print(f"merged {len(runs)} runs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(0)
