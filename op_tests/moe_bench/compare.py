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
import re
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
        rows = list(csv.DictReader(f))
    for r in rows:
        # new-<tier>.csv rows carry new_us/new_id; read them like dispatch rows
        if "new_us" in r and not r.get("median_us"):
            r["median_us"] = r["new_us"]
            r.setdefault("ref_pass", "")
            if (
                r["status"] == "no-correct-candidate"
            ):  # no-reference stays its own state
                r["status"] = "fail"
    return rows


def load_csv(path):
    return schema.read_csv(path)


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


def read_weights(path):
    """model-weights.csv: first column model, second column weight; absent -> 1.0."""
    w = {}
    if path and os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.reader(f):
                if len(r) >= 2 and _f(r[1]) is not None:
                    w[r[0].strip()] = float(r[1])
    return w


def model_weight(model, weights):
    """Weight of a case's model string. The whole string is looked up first (a listed name
    may itself contain ``|``: "Mixtral-8x22B|Grok-1"), then its ``|`` parts; a name with a
    trailing note in parentheses ("DeepSeek-V3/R1 (default shape; shared expert)") falls
    back to the name before it, as the listed name has no note."""
    whole = (model or "").replace("?", "").strip()
    if whole in weights:
        return weights[whole]
    hits = []
    for part in [p.strip() for p in whole.split("|") if p.strip()]:
        for name in (part, re.sub(r"\s+\([^)]*\)\s*$", "", part)):
            if name in weights:
                hits.append(weights[name])
                break
    return max(hits) if hits else 1.0


def gpu_key(name):
    from . import roofline

    return roofline.sku_of(name) or (name or "unknown").replace(" ", "_")


INTER_ROUNDING = {"bf16": 0, "fp32_direct": 1}  # bf16_then_<quant> ranks 2
COMBINE_ROUNDING = {
    "fp32_reduce": 0,
    "fp32_atomic": 0,
    "bf16_rows_fp32_sum": 1,
    "bf16_atomic": 2,
}


def _numerics_fields(key):
    return dict(p.split("=", 1) for p in (key or "").split(";") if "=" in p)


def rounds_more(bar_key, new_key):
    """What the new kernel rounds more than the bar's path, from fewest roundings up.

    act_quant: none < quantized. inter: bf16 (unquantized) < fp32_direct < bf16_then_<q>.
    combine: fp32_reduce = fp32_atomic < bf16_rows_fp32_sum < bf16_atomic. Informational:
    the verdict stays on the error against the bar; bf16_atomic matches aiter's own paths.
    """
    b, n = _numerics_fields(bar_key), _numerics_fields(new_key)
    out = []
    if (
        b.get("act_quant")
        and n.get("act_quant")
        and (n["act_quant"] != "none") > (b["act_quant"] != "none")
    ):
        out.append("act_quant")

    def inter_rank(v):
        return 2 if v.startswith("bf16_then_") else INTER_ROUNDING.get(v)

    bi, ni = inter_rank(b.get("inter", "")), inter_rank(n.get("inter", ""))
    if bi is not None and ni is not None and ni > bi:
        out.append("inter")
    bc = COMBINE_ROUNDING.get(b.get("combine", "").removesuffix(" (inferred)"))
    nc = COMBINE_ROUNDING.get(n.get("combine", "").removesuffix(" (inferred)"))
    if bc is not None and nc is not None and nc > bc:
        out.append("combine")
    return ",".join(out)


ACC_GATE = [f"{ref}_{k}" for ref in ("path", "exact") for k in ("rel_l2", "max_norm")]


def _same_rounding(a, b):
    """Equal act_quant, inter and inter_quant; ``combine`` is not modeled by the
    references. Bars from before ``inter_quant`` existed compare on what they have."""
    if not a or not b:
        return False
    fa, fb = _numerics_fields(a), _numerics_fields(b)
    keys = [k for k in ("act_quant", "inter", "inter_quant") if k in fa and k in fb]
    return bool(keys) and all(fa[k] == fb[k] for k in keys)


def _mx_rule(q):
    for fam in ("mx_fp4_", "mx_fp8_"):
        if (q or "").startswith(fam):
            return fam[:-1], q[len(fam) :]
    return None, None


def rule_differs(bar_key, new_key):
    """ "different rule (MX scale)" when both quantize MX with the same format but another
    scale rule (round_up, even, in_kernel). Informational: not "rounds more"; the verdict
    stays on ref_exact against the bar."""
    b, n = _numerics_fields(bar_key), _numerics_fields(new_key)
    for k in ("act_quant", "inter_quant"):
        bf, br = _mx_rule(b.get(k))
        nf, nr = _mx_rule(n.get(k))
        if bf and bf == nf and br != nr:
            return "different rule (MX scale)"
    return ""


def acc_ok(bar, new):
    """Accuracy rule of the plan: mean(new) <= mean(bar) + 2 * std(bar) + 1% of mean(bar)
    for rel_l2 and max_norm (mean and std over seeds and repeats). When the new kernel's
    rounding points equal the bar's, against both references; when they differ, against
    ref_exact only. Also needs the new kernel to pass aiter's rule.

    Returns (ok, form): ok True/False/None (numbers missing), form ``both`` or ``exact``.
    """
    if str(new.get("new_acc_pass", "")) != "True":
        return (None if new.get("new_acc_pass", "") == "" else False), ""
    same = _same_rounding(bar.get("bar_numerics"), new.get("new_numerics"))
    form = "both" if same else "exact"
    names = ACC_GATE if same else [n for n in ACC_GATE if n.startswith("exact_")]
    ok = True
    for name in names:
        bm, bs = _f(bar.get(f"bar_{name}_mean")), _f(bar.get(f"bar_{name}_std"))
        nm = _f(new.get(f"new_{name}_mean"))
        if bm is None or bs is None or nm is None:
            return None, form
        ok = ok and nm <= bm + 2 * bs + 0.01 * bm
    return ok, form


def bar_view(rows, bar_rows, new_rows, weights):
    """Per case and mode: dispatch, bar, roofline, new, gap and roofline efficiency.

    gap = new_us / bar_us - 1; roofline_eff = roofline_us / new_us. A case is won
    only when gap < -max(2%, 3x noise) in every mode it runs (cold and warm).
    """
    bars = {key(r): r for r in bar_rows}
    news = {key(r): r for r in new_rows}
    base_by = {key(r): r for r in rows}
    out = []
    for k, b in bars.items():
        n = news.get(k)
        d = base_by.get(k)
        bar_us = _f(b.get("bar_us"))
        roof = _f(b.get("roofline_us"))
        new_us = _f(n.get("median_us")) if n and n.get("status") == "ok" else None
        band_v = band(d)[0] if d else 0.02
        err_ok, acc_form = acc_ok(b, n or {}) if (new_us and bar_us) else (None, "")
        gap = new_us / bar_us - 1 if (new_us and bar_us) else None
        if new_us and not bar_us and b.get("status") == "no-bar":
            gap = -1.0  # no existing backend computes it: any correct kernel wins
        out.append(
            {
                "case_id": b["case_id"],
                "sig": b["sig"],
                "family": b["family"],
                "model": b["model"],
                "token": b["token"],
                "mode": b["mode"],
                "gpu": b.get("gpu", ""),
                "dispatch_us": _f(b.get("dispatch_us")) or "",
                "bar_us": bar_us or "",
                "bar_source": b.get("bar_source", ""),
                "bar_kernels": b.get("bar_kernels", ""),
                "roofline_us": roof or "",
                "bound": b.get("bound", ""),
                "new_us": new_us or "",
                "new_id": (
                    (n.get("new_id") or f"tree:{n.get('aiter_sha', '')}") if n else ""
                ),
                "gap": gap if gap is not None else "",
                "roofline_eff": roof / new_us if (roof and new_us) else "",
                "band": band_v,
                "new_kernel_id": (n.get("new_kernel_id") or "") if n else "",
                **{
                    f"{who}_{name}_{st}": (b if who != "new" else (n or {})).get(
                        f"{who}_{name}_{st}", ""
                    )
                    for who in ("dispatch", "bar", "new")
                    for name in ACC_GATE
                    for st in ("mean", "std")
                },
                "bar_acc_pass": b.get("bar_acc_pass", ""),
                "new_acc_pass": (n or {}).get("new_acc_pass", ""),
                "acc_ok": "" if err_ok is None else err_ok,
                "acc_form": acc_form,
                "bar_incomplete": b.get("bar_incomplete", ""),
                "bar_numerics": b.get("bar_numerics", ""),
                "new_numerics": (n or {}).get("new_numerics", ""),
                "rounds_more": rounds_more(
                    b.get("bar_numerics", ""), (n or {}).get("new_numerics", "")
                ),
                "mx_rule": rule_differs(
                    b.get("bar_numerics", ""), (n or {}).get("new_numerics", "")
                ),
                "dispatch_exact_rel_l2_mean": b.get("dispatch_exact_rel_l2_mean", ""),
                "bar_exact_rel_l2_mean_": b.get("bar_exact_rel_l2_mean", ""),
                "won_mode": bool(
                    gap is not None
                    and gap < -band_v
                    and err_ok is not False
                    and not b.get("bar_incomplete")
                    and (err_ok is True or not bar_us or b.get("status") == "no-bar")
                    and not (n and n.get("new_kernel_id") == "aiter")
                ),
                "weight": model_weight(b["model"], weights),
            }
        )
    return out


def case_won(rows):
    """sig -> True when every mode of the case is won."""
    acc = {}
    for r in rows:
        acc.setdefault(r["sig"], []).append(r["won_mode"])
    return {sig: all(v) for sig, v in acc.items()}


def write_queue(view, queue_dir):
    """Rebuild queue/<gpu>.csv: every case not yet won, ranked by gap x model weight."""
    won = case_won(view)
    acc_fail = {
        r["sig"]
        for r in view
        if r["acc_ok"] is False or str(r["new_acc_pass"]) == "False"
    }
    by_gpu = collections.defaultdict(dict)
    acc_by_gpu = collections.defaultdict(dict)
    for r in view:
        if r["sig"] in acc_fail:
            acc_by_gpu[gpu_key(r["gpu"])].setdefault(r["sig"], r)
            continue
        if won.get(r["sig"]):
            continue
        cur = by_gpu[gpu_key(r["gpu"])].get(r["sig"])
        score_gap = (
            r["gap"]
            if r["gap"] != ""
            else (
                (r["bar_us"] / r["roofline_us"] - 1)
                if (r["bar_us"] and r["roofline_us"])
                else 0.0
            )
        )
        score = score_gap * r["weight"]
        if cur is None or score > cur["score"]:
            by_gpu[gpu_key(r["gpu"])][r["sig"]] = dict(
                r,
                score=score,
                ranked_by="gap" if r["gap"] != "" else "bar_us/roofline_us",
            )
    os.makedirs(queue_dir, exist_ok=True)
    cols = [
        "rank",
        "gpu",
        "case_id",
        "family",
        "model",
        "token",
        "mode",
        "weight",
        "score",
        "ranked_by",
        "dispatch_us",
        "bar_us",
        "bar_source",
        "roofline_us",
        "bound",
        "new_us",
        "new_id",
        "gap",
        "roofline_eff",
    ]
    for gpu, cases in by_gpu.items():
        ordered = sorted(cases.values(), key=lambda r: -r["score"])
        with open(os.path.join(queue_dir, f"{gpu}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for i, r in enumerate(ordered, 1):
                w.writerow(dict(r, rank=i, gpu=gpu))
    acc_cols = [
        "gpu",
        "case_id",
        "family",
        "model",
        "token",
        "mode",
        "weight",
        "new_id",
        "new_acc_pass",
        "acc_ok",
        *[
            f"{w}_{n}_{st}"
            for w in ("bar", "new")
            for n in ACC_GATE
            for st in ("mean", "std")
        ],
    ]
    for gpu, cases in acc_by_gpu.items():
        ordered = sorted(cases.values(), key=lambda r: -r["weight"])
        with open(os.path.join(queue_dir, f"{gpu}-accuracy.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=acc_cols, extrasaction="ignore")
            w.writeheader()
            for r in ordered:
                w.writerow(dict(r, gpu=gpu))
    return {
        "speed": {g: len(c) for g, c in by_gpu.items()},
        "accuracy": {g: len(c) for g, c in acc_by_gpu.items()},
    }


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
    if getattr(args, "bar", None):
        view = bar_view(
            base_rows, load_csv(args.bar), new_rows, read_weights(args.weights)
        )
        if args.bar_out:
            with open(args.bar_out, "w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=list(view[0].keys()) if view else ["case_id"]
                )
                w.writeheader()
                w.writerows(view)
        won = case_won(view)
        print(f"vs bar: {sum(won.values())} of {len(won)} cases won in every mode")
        if args.queue_dir:
            print("queue rebuilt:", write_queue(view, args.queue_dir))
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
        last = {}
        for r in run:  # a resumed run may hold a case twice: keep its latest row
            last[(r["case_id"], r["mode"])] = r
        for k, r in last.items():
            by[k].append(r)
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
