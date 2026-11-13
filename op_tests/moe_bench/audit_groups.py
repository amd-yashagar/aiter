# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Audit existing bar outputs for dropped block_m groups (no GPU).

Until 946a02adc the marginal sweep of the bar (full screen since the first harness head
21b9ca0c3, two-phase rank since 488919f53) built its base pair from the first stage-1 kernel
of a block_m group: that kernel lacking a partner raised KeyError (the case failed), and that
kernel failing with every partner dropped the whole group without a record. This reads bar and
candidate CSVs and flags, per case:

* ``keyerror``: the bar row is an error/crash whose detail names a KeyError;
* ``base_failed``: a group with candidate rows, none ok, all of them one stage-1 kernel (its
  base never ran: the group was dropped);
* ``never_tried`` (with ``--enumerate``): a group the tuner enumerates for the case that has
  no candidate row at all;
* ``first_no_partner`` (with ``--enumerate``): the group's first stage-1 kernel has no pair
  with some stage-2 kernel, the old KeyError condition.

``--routes`` CSVs (route tables) mark flagged cases whose shape has a route row.

    python -m op_tests.moe_bench.audit_groups --root <dir> [--root ...] [--enumerate]
        [--routes routes.csv ...] [--md out.md] [--csv flagged.csv]
"""

import argparse
import collections
import csv
import glob
import json
import os
import re
import subprocess
import sys

from . import schema

GROUP = re.compile(r"\|(bm\d+)\|(ks\d+)\|(nt\d+)$")
OLD_SCREEN_HEAD = "21b9ca0c3"  # the first harness head with the base-pair logic
FIXED_HEAD = "946a02adc"


def discover(roots):
    """[(bar csv, candidates csv, meta dict)] for every bar set under the roots."""
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
            if not os.path.exists(cand):
                continue
            meta = {}
            mp = bar[:-4] + ".meta.json"
            if os.path.exists(mp):
                try:
                    with open(mp) as f:
                        meta = json.load(f)
                except (OSError, ValueError):
                    meta = {}
            out.append((bar, cand, meta))
    return out


csv.field_size_limit(1 << 28)


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def unreadable(path):
    """[(case_id, why)] of every row of a bar or candidate CSV that cannot be read under the
    file's own header: a wrong field count, or a ruled column (sha, version, timing or
    accuracy number) holding something else, i.e. a row written under another column set.
    Such a row is invisible to every other check of this audit, so each one is a hard flag.
    A file repaired by ``bar_runner.repair_bar_csv`` reads clean."""
    out = []
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return out
    case_col = rows[0].index("case_id") if "case_id" in rows[0] else 0
    by_record = {n: r for n, r in enumerate(rows[1:], 1)}
    for n, _line, why in schema.row_problems(path):
        r = by_record.get(n, [])
        out.append((r[case_col] if len(r) > case_col else "?", why))
    return out


def group_of(cand_id):
    m = GROUP.search(cand_id or "")
    return m.groups() if m else None


def parse_pair(cand_id):
    parts = (cand_id or "").split("|")
    return (parts[1], parts[2]) if len(parts) == 6 and parts[0] == "fused_moe" else None


def data_flags(bars, cands, mode="warm"):
    """{case_id: [flag, ...]} from the bar and candidate rows alone."""
    flags = collections.defaultdict(list)
    for b in bars:
        if b.get("mode") != mode and b.get("mode") is not None:
            continue
        if b.get("status") in ("error", "crash") and "KeyError" in (
            b.get("detail") or ""
        ):
            flags[b["case_id"]].append(("keyerror", "", b["detail"][:120]))
    by_case = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in cands:
        g, pair = group_of(r["cand_id"]), parse_pair(r["cand_id"])
        if g and pair:
            by_case[r["case_id"]][g].append((pair, r.get("status", "")))
    for case_id, groups in by_case.items():
        for g, rows in groups.items():
            if any(s == "ok" for _, s in rows):
                continue
            s1s = {p[0] for p, _ in rows}
            if len(s1s) == 1 and all(s in ("fail", "error", "crash") for _, s in rows):
                flags[case_id].append(
                    (
                        "base_failed?",
                        "/".join(g),
                        f"{len(rows)} rows, one stage-1 kernel",
                    )
                )
    return flags, by_case


def enumerate_cases(bars, gfx, cu, aiter_tree, jobs=24):
    """{case_id: {group: {"pairs", "first_no_partner"}}} by enumerating on a GPU-less host
    standing in for the arch (children with AITER_AOT_REPLAY)."""
    rows = {}
    for b in bars:
        rows.setdefault(b["case_id"], b)
    ids = sorted(rows)
    chunks = [ids[i::jobs] for i in range(min(jobs, len(ids)))]
    procs = []
    for ch in chunks:
        env = dict(
            os.environ, PYTHONPATH=aiter_tree, HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
            CUDA_VISIBLE_DEVICES="", GPU_ARCHS=gfx, CU_NUM=str(cu), AITER_AOT_REPLAY="1",
            MOE_BENCH_STATE_DIR=os.environ.get("TMPDIR", "/tmp"),
        )  # fmt: skip
        p = subprocess.Popen(
            [sys.executable, "-m", "op_tests.moe_bench.audit_groups", "--enum-child", gfx, str(cu)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            env=env, cwd=aiter_tree,
        )  # fmt: skip
        p.stdin.write(json.dumps([rows[i] for i in ch]))
        p.stdin.close()
        procs.append(p)
    out = {}
    for p in procs:
        for line in p.stdout.read().splitlines():
            if line.startswith("ENUM "):
                out.update(json.loads(line[5:]))
        p.wait()
    return out


def _enum_child(gfx, cu):
    from . import flydsl_prebuild, schema, workload
    from .candidates.fused_cfg import FusedCfgSource

    rows = json.load(sys.stdin)
    out = {}
    for r in rows:
        row = {c: r.get(c, "") for c in schema.COLS}
        case = schema.Case(row, schema.parse_extra(row.get("extra", "")))
        try:
            S = workload.resolve(case)
            ctx = flydsl_prebuild.make_ctx(
                S, case, os.environ.get("TMPDIR", "/tmp"), gfx, int(cu)
            )
            cands = FusedCfgSource().enumerate(ctx)
        except Exception as e:  # noqa: BLE001
            out[case.id] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
            continue
        groups = {}
        for c in cands:
            if not c.group.startswith("pair-"):
                continue
            g = group_of(c.id)
            e = groups.setdefault(
                "/".join(g), {"s1": set(), "s2": set(), "pairs": set()}
            )
            e["s1"].add(c.meta["s1"])
            e["s2"].add(c.meta["s2"])
            e["pairs"].add((c.meta["s1"], c.meta["s2"]))
        res = {}
        for g, e in groups.items():
            first = min(e["s1"])
            missing = any((first, k2) not in e["pairs"] for k2 in sorted(e["s2"]))
            res[g] = {
                "pairs": len(e["pairs"]),
                "s1": len(e["s1"]),
                "first_no_partner": missing,
            }
        out[case.id] = res
    print("ENUM " + json.dumps(out))


def enum_flags(enum, by_case):
    flags = collections.defaultdict(list)
    for case_id, groups in enum.items():
        if "error" in groups:
            continue
        seen = by_case.get(case_id, {})
        seen_groups = {"/".join(g) for g in seen}
        for g, e in groups.items():
            rows = seen.get(tuple(g.split("/")), [])
            # confirmed only where the group has other stage-1 kernels that were never tried
            if (
                rows
                and not any(st == "ok" for _, st in rows)
                and len({p[0] for p, _ in rows}) == 1
                and e["s1"] > 1
            ):
                flags[case_id].append(
                    (
                        "base_dropped",
                        g,
                        f"{e['s1'] - 1} other stage-1 kernels never tried ({e['pairs']} pairs)",
                    )
                )
            if g not in seen_groups:
                flags[case_id].append(
                    (
                        "never_tried",
                        g,
                        f"{e['pairs']} pairs enumerated, no candidate row",
                    )
                )
            if e["first_no_partner"]:
                flags[case_id].append(
                    ("first_no_partner", g, "old base-pair search raises KeyError here")
                )
    return flags


def settle_soft(flags, enum):
    """Where the replay enumerated the group, its verdict replaces the data-only guess:
    ``base_failed?`` goes (a group that enumerates one stage-1 kernel has nothing else to
    try, and one that enumerates more is ``base_dropped`` when it applies). A group the
    replay does not know (not in today's enumeration) keeps the soft flag."""
    for case_id in list(flags):
        groups = enum.get(case_id)
        if not groups or "error" in groups:
            continue
        kept = [
            f for f in flags[case_id] if not (f[0] == "base_failed?" and f[1] in groups)
        ]
        if kept:
            flags[case_id] = kept
        else:
            del flags[case_id]


def _show(path, rel):
    for base in (rel, "/workspaces/meta/wt/GM-123"):
        if base and path.startswith(base + os.sep):
            return os.path.relpath(path, base)
    return path


def load_routes(paths):
    rows = []
    for p in paths:
        try:
            rows += _read(p)
        except OSError:
            pass
    return rows


def routed(case, routes):
    """The route rows a case's shape matches (family, dims, experts, topk, act, M range)."""
    hits = []
    for r in routes:
        try:
            if (
                r["family"] == case.get("family")
                and int(r["model_dim"]) == int(case["model_dim"])
                and int(r["inter_dim"]) == int(case["inter_dim"])
                and int(r["experts"]) == int(case["expert"])
                and int(r["topk"]) == int(case["topk"])
                and r["act"].lower() == str(case["act_type"]).lower()
                and int(r["m_min"]) <= int(case["token"]) <= int(r["m_max"])
            ):
                hits.append(r["kernel"])
        except (KeyError, ValueError):
            continue
    return hits


def audit(
    bar_path,
    cand_path,
    meta,
    routes=(),
    enumerate_=False,
    aiter_tree=None,
    routes_gfx="",
):
    bars, cands = _read(bar_path), _read(cand_path)
    flags, by_case = data_flags(bars, cands)
    bad_rows = unreadable(bar_path) + (unreadable(cand_path) if cand_path else [])
    for case_id, why in bad_rows:
        flags[case_id].append(("unreadable_row", "", why))
    if enumerate_ and meta.get("gfx"):
        enum = enumerate_cases(
            [b for b in bars if b.get("mode") == "warm"] or bars,
            meta["gfx"], meta.get("cu_num", 304), aiter_tree,
        )  # fmt: skip
        settle_soft(flags, enum)
        for k, v in enum_flags(enum, by_case).items():
            flags[k] += v
    case_rows = {b["case_id"]: b for b in bars}
    bar_of = {b["case_id"]: b for b in bars if b.get("mode") == "warm"}
    res = []
    for case_id, fl in sorted(flags.items()):
        c = case_rows.get(case_id, {})
        res.append(
            {
                "case_id": case_id,
                "token": c.get("token", ""),
                "flags": sorted({f[0] for f in fl}),
                "groups": sorted({f[1] for f in fl if f[1]}),
                "detail": "; ".join(f"{a}:{b}:{d}" for a, b, d in fl[:3]),
                "items": fl,
                "routed": (
                    routed(c, routes)
                    if c and (not routes_gfx or meta.get("gfx") == routes_gfx)
                    else []
                ),
                "bar_status": bar_of.get(case_id, {}).get("status", ""),
                "bar_id": bar_of.get(case_id, {}).get("bar_id", "")[-80:],
            }
        )
    harness = meta.get("harness_sha", "")
    return {
        "enumerated": bool(enumerate_ and meta.get("gfx")),
        "bar": bar_path,
        "harness": harness,
        "gfx": meta.get("gfx", ""),
        "cases": len({b["case_id"] for b in bars}),
        "unreadable": len(bad_rows),
        "flagged": res,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--routes", action="append", default=[])
    ap.add_argument("--enumerate", action="store_true")
    ap.add_argument(
        "--routes-gfx", default="", help="the arch the route tables are for"
    )
    ap.add_argument("--aiter", default=os.getcwd())
    ap.add_argument("--csv")
    ap.add_argument("--md")
    ap.add_argument("--enum-child", nargs=2)
    ap.add_argument(
        "--rel", default="", help="show bar paths relative to this directory"
    )
    a = ap.parse_args(argv)
    if a.enum_child:
        _enum_child(*a.enum_child)
        return 0
    routes = load_routes(a.routes)
    sets = [
        audit(b, c, m, routes, a.enumerate, a.aiter, a.routes_gfx)
        for b, c, m in discover(a.root)
    ]
    lines = [
        (
            "| bar set | harness | enum | cases | cases with a lost group (bar ok) | groups lost "
            "| stage-1 kernels never tried | groups never tried | first-kernel no-partner cases "
            "| keyerror cases | unreadable rows (hard) | routed rows among the lost-group cases |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sets:
        kind = "base_dropped" if s["enumerated"] else "base_failed?"
        drop = [r for r in s["flagged"] if kind in r["flags"]]
        drop_ok = [r for r in drop if r["bar_status"] == "ok"]
        groups = sum(1 for r in drop for it in r["items"] if it[0] == kind)
        untried = sum(
            int(m.group(1))
            for r in drop
            for it in r["items"]
            if it[0] == kind
            for m in [re.search(r"(\d+) other stage-1", it[2])]
            if m
        )
        never = sum(
            1 for r in s["flagged"] for it in r["items"] if it[0] == "never_tried"
        )
        nopart = sum(1 for r in s["flagged"] if "first_no_partner" in r["flags"])
        keyerr = sum(1 for r in s["flagged"] if "keyerror" in r["flags"])
        n_routed = sum(1 for r in drop_ok if r["routed"])
        en = s["enumerated"]
        lines.append(
            f"| {_show(s['bar'], a.rel)} | {s['harness'] or '?'} | {'yes' if en else 'no'} "
            f"| {s['cases']} | {len(drop_ok)} | {groups} | {untried if en else '?'} "
            f"| {never if en else '?'} | {nopart if en else '?'} | {keyerr} "
            f"| {s['unreadable']} | {n_routed} |"
        )
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text + "\n")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "bar_set",
                    "harness",
                    "case_id",
                    "token",
                    "flags",
                    "groups",
                    "routed",
                    "bar_status",
                    "detail",
                ]
            )
            for s in sets:
                for r in s["flagged"]:
                    w.writerow(
                        [
                            s["bar"],
                            s["harness"],
                            r["case_id"],
                            r["token"],
                            ",".join(r["flags"]),
                            ",".join(r["groups"]),
                            ",".join(r["routed"]),
                            r["bar_status"],
                            r["detail"],
                        ]
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
