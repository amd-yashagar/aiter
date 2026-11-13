# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``run --mode bar|new``: orchestrate bar workers and write bar/candidates/new CSVs.

Outputs in ``--out-dir`` (the baseline folder): ``bar-<tier>.csv`` (one row per case and
mode: dispatch_us, bar_us, bar_source, roofline), ``candidates-<tier>.csv`` (every
candidate's times and errors) and, for ``--kind new``, ``new-<tier>.csv``. Requires the
dispatch results of the same tier, which supply each case's dispatch status.
"""

import csv
import json
import os
import socket
import tempfile
import time

from . import accuracy, preflight, runner, schema

BAR_COLS = (
    [
        "case_id",
        "sig",
        "kind",
        "family",
        "model",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
        "extra",
        "mode",
        "status",
        "detail",
        "dispatch_us",
        "dispatch_status",
        "dispatch_route_key",
        "dispatch_kernel1",
        "dispatch_kernel2",
        "dispatch_route_family",
        "dispatch_rounds_more",
        "bar_rounds_more",
        "review",
        "lr_us",
        "lr_id",
        "lr_kernels",
        "lr_confirmed",
        "lr_status",
        "lr_numerics",
        "lr_acc_pass",
        "bar_us",
        "bar_id",
        "bar_source",
        "bar_entry",
        "bar_kernels",
        "bar_timing_mode",
        "bar_tag",
        "bar_raw_us",
        "bar_topk_us",
        "n_candidates",
        "n_correct",
        "skipped_sources",
        "main_failure",
        "bar_incomplete",
        "gate_fail",
        "bar_exclusions",
        "gate_fail",
        "cand_accounting",
        "new_us",
        "new_id",
        "new_kernel_id",
        "new_host_us",
        "new_reason",
        "route_changed",
        "off_us",
        "off_kernel_id",
        "off_host_us",
        "roofline_us",
        "roofline_staged_us",
        "roofline_serial_us",
        "bound",
        "mfma",
        "bar_roofline_eff",
        "dispatch_roofline_eff",
        "new_roofline_eff",
        "dispatch_numerics",
        "bar_numerics",
        "new_numerics",
        "dispatch_numerics",
        "bar_numerics",
        "new_numerics",
        *accuracy.gate_columns("dispatch"),
        *accuracy.gate_columns("bar"),
        *accuracy.gate_columns("new"),
    ]
    + ["node", "gpu", "gfx", "cu_num", "rocm", "torch", "flydsl"]
    + ["aiter_sha", "aiter_dirty", "harness_sha", "env", "date", "routes_disabled"]
)
CAND_COLS = [
    "case_id",
    "sig",
    "family",
    "token",
    "mode",
    "cand_id",
    "source",
    "entry",
    "kernels",
    "layout",
    "phase",
    "status",
    "detail",
    "timing_mode",
    "median_us",
    "p10_us",
    "p90_us",
    "std_us",
    "iters",
    "raw_us",
    "topk_us",
    "topk_alone_us",
    "tag",
    "err_ratio",
    "logits_diff",
    "ref_pass",
    "route_key",
    "kernel1",
    "kernel2",
    "new_kernel_id",
    "reason",
    "host_us",
    "numerics",
    "numerics",
    *accuracy.seed_columns(),
    "node",
    "gpu",
    "aiter_sha",
    "date",
    "routes_disabled",
    "rounds_more",
]


BAR_COLS = list(dict.fromkeys(BAR_COLS))
CAND_COLS = list(dict.fromkeys(CAND_COLS))


def _history(kind):
    """Column lists of earlier harness heads (``col_history.json``): a resume maps rows of
    those widths exactly instead of rejecting them."""
    try:
        with open(os.path.join(os.path.dirname(__file__), "col_history.json")) as f:
            return json.load(f)[kind]
    except (OSError, ValueError, KeyError):
        return []


class Table:
    """Append-only CSV with a flush per row."""

    def __init__(self, path, cols, resume, key=("case_id", "mode"), history=()):
        self.key = key
        self.done = set()
        self.out = schema.Appender(path, cols, resume, history)
        self.cols = self.out.cols
        if os.path.exists(path) and resume:
            for r in schema.read_csv(path):
                self.done.add(tuple(r[k] for k in key))

    def add(self, row):
        self.out.add_row(row)
        self.done.add(tuple(row.get(k, "") for k in self.key))

    def close(self):
        self.out.close()


def _row(case, ev, meta, cols):
    r = {k: case.row.get(k, "") for k in cols if k in case.row}
    r["sig"] = runner.case_sig(case)
    for k in cols:
        if k in ev and k != "event":
            v = ev[k]
            r[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    for k in ("node", "gfx", "cu_num", "rocm", "torch", "flydsl", "aiter_sha", "date"):
        r.setdefault(k, meta.get(k, ""))
    for k in ("aiter_dirty", "harness_sha", "env", "routes_disabled"):
        r[k] = meta.get(k, "")
    r["gpu"] = ev.get("gpu") or meta.get("gpu", "")
    return r


def dispatch_lookup(path):
    """Dispatch results -> ({case_id: status}, {(case_id, mode): median_us})."""
    status, us = {}, {}
    for r in schema.read_csv(path):
        prev = status.get(r["case_id"])
        if prev is None or r["status"] == "ok":
            status[r["case_id"]] = r["status"]
        if r["status"] == "ok" and r.get("median_us"):
            us[(r["case_id"], r["mode"])] = float(r["median_us"])
    return status, us


def fill_dispatch(ev, us_map, case_id):
    """Put the base tree's dispatch_us on a new/off row (they never time dispatch)."""
    if ev.get("dispatch_us") in (None, ""):
        v = us_map.get((case_id, ev.get("mode")))
        if v is not None:
            ev["dispatch_us"] = v
    return ev


class BarGroupJob(runner.GroupJob):
    worker_name = "bar_worker"

    def __init__(
        self,
        *a,
        cand_table=None,
        dispatch_status=None,
        kind="bar",
        sources=None,
        source_modules=(),
        screen="full",
        **k,
    ):
        super().__init__(*a, **k)
        self.cand_table = cand_table
        self.dispatch_status = dispatch_status or {}
        self.kind = kind
        self.sources = sources
        self.source_modules = list(source_modules)
        self.screen = screen
        self.bad = {}
        self.faults = {}  # (case id, cand id) -> the error the worker died of
        self.main_failures = {}
        self.dispatch_us = {}

    def extra_spec(self):
        return {
            "kind": self.kind,
            "dispatch_status": {
                c.id: self.dispatch_status.get(c.id) for c in self.cases
            },
            "bad_cands": {k: sorted(v) for k, v in self.bad.items()},
            "main_failures": {
                c.id: self.main_failures[c.id]
                for c in self.cases
                if c.id in self.main_failures
            },
            "sources": self.sources,
            "source_modules": self.source_modules,
            "screen": self.screen,
            "tmpdir": self.tmpdir,
        }

    def handle_row(self, c, ev):
        self.results.add(
            _row(c, fill_dispatch(ev, self.dispatch_us, c.id), self.meta, BAR_COLS)
        )

    def handle_other(self, ev):
        if ev.get("event") == "gpu_fault":
            self.faults[(ev["case_id"], ev["cand_id"])] = ev.get("detail", "")
            return
        if ev.get("event") != "cand" or self.cand_table is None:
            return
        c = next((x for x in self.cases if x.id == ev["case_id"]), None)
        if c is not None:
            self.cand_table.add(_row(c, ev, self.meta, CAND_COLS))

    def _maybe_done(self, c):
        wanted = [m for m in c.modes if m in self.modes] or ["warm"]
        if all((c.id, m) in self.results.done for m in wanted):
            self.pending.pop(c.id, None)

    def _fail(self, status, detail):
        if self.phase.startswith("cand:") and self.inflight in self.pending:
            cid = self.phase[len("cand:") :]
            c = self.pending[self.inflight]
            self.bad.setdefault(c.id, set()).add(cid)
            fault = self.faults.get((c.id, cid))
            if fault:
                detail = f"{detail}; worker error before exit: {fault}"
            ev = {
                "case_id": c.id,
                "cand_id": cid,
                "mode": "warm",
                "phase": "confirm",
                "status": status,
                "detail": detail,
            }
            self.cand_table.add(_row(c, ev, self.meta, CAND_COLS))
            return
        ids = (
            list(self.pending)
            if self.inflight is None
            or self.phase
            in (
                "weights",
                "start",
                "reference",
            )
            else [self.inflight]
        )
        for cid in ids:
            c = self.pending.pop(cid, None)
            if c is None:
                continue
            for m in [m for m in c.modes if m in self.modes] or ["warm"]:
                if (c.id, m) not in self.results.done:
                    self.results.add(
                        _row(
                            c,
                            {"mode": m, "status": status, "detail": detail},
                            self.meta,
                            BAR_COLS,
                        )
                    )


def read_main_failures(args, cases_arg):
    """Cases known to fail on main (tiers/main-failures.csv): {case_id: reason}."""
    path = getattr(args, "main_failures", None) or os.path.join(
        os.path.dirname(os.path.abspath(cases_arg.split(",")[0])), "main-failures.csv"
    )
    out = {}
    if os.path.exists(path):
        for r in schema.read_csv(path):
            cid = r.get("case_id")
            if cid:
                out[cid] = (
                    r.get("reason")
                    or r.get("detail")
                    or r.get("status")
                    or "known failure"
                )[:200]
    return out


def tier_name(cases_arg):
    return os.path.splitext(os.path.basename(cases_arg.split(",")[0]))[0]


def check_discarded_sources(out_dir, tier, prefix):
    """Flag a candidate source (source and backend) the router discarded on every row.

    One discard is a mismatch between what was forced and what ran; a whole source
    discarded everywhere is a harness defect that leaves the bar missing a backend. The
    finding goes into the bar's meta and the log; rerun the run after the fix.
    """
    path = os.path.join(out_dir, f"candidates-{tier}.csv")
    if not os.path.exists(path):
        return []
    from . import routes

    tally = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("source") in ("dispatch", "off", "new"):
                continue
            be = routes.backend_of((r.get("kernels") or "").split(" ")[0]) or "?"
            t = tally.setdefault(f"{r.get('source')}:{be}", [0, 0])
            t[0] += 1
            t[1] += r.get("status") == "router-discarded"
    bad = {k: v for k, v in tally.items() if v[0] > 0 and v[1] == v[0]}
    if bad:
        print(
            f"WARNING {prefix}: candidate sources discarded by the router on every "
            f"case: {sorted(bad)}. The bar lacks these backends; do not use it as a gate.",
            flush=True,
        )
        mp = os.path.join(out_dir, f"{prefix}-{tier}.meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                meta = json.load(f)
            meta["router_discarded_sources"] = {k: v[0] for k, v in bad.items()}
            with open(mp, "w") as f:
                json.dump(meta, f, indent=1)
    return sorted(bad)


def mark_router_discards(bar_csv, candidates_csv, reason):
    """Add/extend ``bar_incomplete`` on the cases of an existing bar that have a candidate
    the router discarded and no other candidate of that backend that ran (bars taken
    before the whole-graph naming fix). Returns the case ids marked."""
    from . import routes

    bad = {}
    with open(candidates_csv, newline="") as f:
        for r in csv.DictReader(f):
            be = routes.backend_of((r.get("kernels") or "").split(" ")[0]) or "?"
            t = bad.setdefault((r["case_id"], be), [0, 0])
            t[0] += r.get("status") == "router-discarded" and "'<lambda>'" in (
                r.get("detail") or ""
            )
            t[1] += r.get("status") == "ok"
    hit = {c for (c, _), (d, ok) in bad.items() if d and not ok}
    with open(bar_csv, newline="") as f:
        rd = csv.DictReader(f)
        cols = list(rd.fieldnames)
        rows = list(rd)
    if "bar_incomplete" not in cols:
        cols.append("bar_incomplete")
    for r in rows:
        if r["case_id"] in hit:
            old = r.get("bar_incomplete") or ""
            r["bar_incomplete"] = f"{old}; {reason}" if old else reason
    with open(bar_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    return sorted(hit)


def repair_bar_csv(path, out=None):
    """Rewrite a bar CSV whose rows were appended by runs with different BAR_COLS (a
    column added between a run and its resume): rows are told apart by field count and
    mapped onto the current columns; the file gets the union header."""
    csv.field_size_limit(1 << 28)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    old = rows[0]
    new = list(BAR_COLS)
    cols = new + [c for c in old if c not in new]
    fixed = []
    for r in rows[1:]:
        if len(r) == len(new):
            fixed.append(dict(zip(new, r)))
        elif len(r) == len(old):
            fixed.append(dict(zip(old, r)))
        else:
            raise ValueError(f"{path}: row of {len(r)} fields fits neither header")
    with open(out or path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(fixed)
    return len(fixed)


def cmd_bar(args):
    """bar mode, or new mode: routing on, routing off (AITER_MOE_DISABLE), first calls."""
    if args.kind != "new":
        return _run_pass(args, args.kind)
    if getattr(args, "new_routes", None):
        os.environ["AITER_MOE_ROUTES"] = args.new_routes
        os.environ["AITER_MOE_STRICT"] = "1"
    rc = _run_pass(args, "new")
    if not args.no_off:
        os.environ["AITER_MOE_DISABLE"] = "1"
        try:
            rc = _run_pass(args, "off") or rc
        finally:
            os.environ.pop("AITER_MOE_DISABLE", None)
        merge_off(args)
    if args.first:
        run_first_calls(args)
    summarize_new(args)
    if getattr(args, "inner", False):
        open(os.path.join(args.out_dir, ".done"), "w").close()
    return rc


def _tier(args):
    return args.tier_name or tier_name(args.cases)


def merge_off(args):
    """Add the routing-off timings to new-<tier>.csv as off_us, off_kernel_id, off_host_us."""
    tier = _tier(args)
    new_path = os.path.join(args.out_dir, f"new-{tier}.csv")
    off = {
        (r["case_id"], r["mode"]): r
        for r in schema.read_csv(os.path.join(args.out_dir, f"off-{tier}.csv"))
    }
    rows = schema.read_csv(new_path)
    for r in rows:
        o = off.get((r["case_id"], r["mode"]))
        if o:
            r["off_us"] = o.get("off_us", "")
            r["off_kernel_id"] = o.get("off_kernel_id", "")
            r["off_host_us"] = o.get("off_host_us", "")
    tmp = f"{new_path}.{os.getpid()}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BAR_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, new_path)


def run_first_calls(args):
    """first-call.csv: first eager call per weight set (lazy state and compiles), with a
    warm and an empty FlyDSL cache. Outside the timed region: in serving it is warmup.
    """
    tree = os.path.abspath(args.new_aiter or args.aiter)
    tmpdir = args.tmpdir or os.path.join(tempfile.gettempdir(), "moe-bench-tmp")
    cases = runner.select_cases(args.cases.split(","), args.filter)
    seen, picked = set(), []
    for c in cases:
        if c.kind != "fused" or c.token <= 0:
            continue
        g = schema.group_id(schema.weight_group_key(c))
        if g in seen:
            continue
        seen.add(g)
        c.row["modes"] = "first"
        picked.append(c)
        if len(picked) >= args.first:
            break
    meta = runner.probe_meta(
        tree, args.node or socket.gethostname(), args.launcher, tmpdir
    )
    res = runner.Results(os.path.join(args.out_dir, "first-call.csv"), args.resume)
    runner.run_first(picked, tree, tmpdir, args.launcher, res, meta, ["warm", "empty"])
    res.close()


def summarize_new(args):
    """Run summary: routing-off vs the base tree's dispatch, hook overhead, route counts."""
    tier = _tier(args)
    new_rows = schema.read_csv(os.path.join(args.out_dir, f"new-{tier}.csv"))
    summary = {"tier": tier, "cases": len({r["case_id"] for r in new_rows})}
    routed = [r for r in new_rows if r.get("new_kernel_id") not in ("", "aiter")]
    summary["served_by_kernel"] = len({r["case_id"] for r in routed})
    noref = [r for r in new_rows if r.get("status") == "no-reference"]
    summary["no_reference"] = len(noref)
    summary["no_reference_routed_gate_fail"] = sorted(
        {r["case_id"] for r in noref if r.get("gate_fail") == "1"}
    )
    summary["gate_pass"] = not summary["no_reference_routed_gate_fail"]
    if noref:
        print(
            f"GATE: {len(noref)} rows have no reference (never a pass); routed rows that "
            f"fail the gate: {summary['no_reference_routed_gate_fail']}",
            flush=True,
        )
    summary["route_changed_in_capture"] = [
        r["case_id"] for r in new_rows if r.get("route_changed") == "1"
    ]
    if args.dispatch_results:
        path = args.dispatch_results
        if os.path.isdir(path):
            path = os.path.join(path, f"{tier}.csv")
        disp = {
            (r["case_id"], r["mode"]): r
            for r in schema.read_csv(path)
            if r["status"] == "ok"
        }
        off_bad = []
        for r in new_rows:
            d = disp.get((r["case_id"], r["mode"]))
            if d and r.get("off_us"):
                ratio = float(r["off_us"]) / float(d["median_us"])
                if abs(ratio - 1) > 0.05:
                    off_bad.append((r["case_id"], r["mode"], round(ratio, 3)))
        summary["off_vs_dispatch_over_5pct"] = off_bad
    ov = {}
    for label, pick in (
        (
            "routed",
            [r for r in routed if r.get("new_host_us") and r.get("off_host_us")],
        ),
        (
            "not_routed",
            [
                r
                for r in new_rows
                if r.get("new_kernel_id") == "aiter"
                and r.get("new_host_us")
                and r.get("off_host_us")
            ],
        ),
    ):
        if pick:
            r = pick[0]
            ov[label] = {
                "case_id": r["case_id"],
                "on_host_us": float(r["new_host_us"]),
                "off_host_us": float(r["off_host_us"]),
                "overhead_us": float(r["new_host_us"]) - float(r["off_host_us"]),
            }
    summary["hook_overhead"] = ov
    with open(os.path.join(args.out_dir, f"summary-{tier}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("new-mode summary:", json.dumps(summary), flush=True)


def _run_pass(args, kind):
    node = args.node or socket.gethostname()
    tmpdir = args.tmpdir or os.path.join(tempfile.gettempdir(), "moe-bench-tmp")
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tree = os.path.abspath(
        args.new_aiter if kind in ("new", "off") and args.new_aiter else args.aiter
    )
    tier = _tier(args)
    cases = runner.select_cases(args.cases.split(","), args.filter)
    if getattr(args, "ids", None):
        cases = [c for c in cases if c.id in args.ids]
    modes_req = [m for m in args.modes.split(",") if m and m != "first"]
    dispatch_status, dispatch_us = {}, {}
    if args.dispatch_results:
        path = args.dispatch_results
        if os.path.isdir(path):
            path = os.path.join(path, f"{tier}.csv")
        dispatch_status, dispatch_us = dispatch_lookup(path)
    main_failures = read_main_failures(args, args.cases)
    prefix = kind if kind in ("new", "off") else "bar"
    bar = Table(
        os.path.join(args.out_dir, f"{prefix}-{tier}.csv"),
        BAR_COLS,
        args.resume,
        history=_history("bar"),
    )
    cand = Table(
        os.path.join(
            args.out_dir,
            f"candidates-{tier}.csv" if kind != "off" else f"candidates-off-{tier}.csv",
        ),
        CAND_COLS,
        args.resume,
        key=("case_id", "cand_id", "mode", "phase"),
        history=_history("cand"),
    )
    meta = runner.probe_meta(tree, node, args.launcher, tmpdir)
    # bar mode measures aiter's own paths with the router hook off (AITER_MOE_DISABLE=1),
    # which the workers set and assert; the tree's sha is in aiter_sha
    meta["routes_disabled"] = 1 if kind == "bar" else 0
    if kind in ("bar", "new"):
        shimmed = runner.needs_shim(tree)
        pf = preflight.ensure(
            tree,
            tmpdir,
            args.launcher,
            {
                **runner.worker_env(
                    tree,
                    tmpdir,
                    extra_path=runner.make_shim(tmpdir) if shimmed else None,
                ),
                "MOE_BENCH_OUT_DIR": os.path.abspath(args.out_dir),
            },
            pkg="moe_bench_shim" if shimmed else "op_tests.moe_bench",
        )
        meta["preflight"] = {
            "rocm_path_env": pf["rocm_path_env"],
            "flydsl_version": pf["versions"].get("flydsl"),
            "flydsl_compile_probe": next(
                (
                    c["detail"]
                    for c in pf["checks"]
                    if c["name"] == "flydsl compile probe"
                ),
                "",
            ),
            "versions": pf["versions"],
            "checks": {c["name"]: c["ok"] for c in pf["checks"]},
        }
    with open(os.path.join(args.out_dir, f"{prefix}-{tier}.meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    for c in cases:
        if c.kind == "fused" and c.token <= 0:
            for m in [m for m in c.modes if m in modes_req]:
                if (c.id, m) not in bar.done:
                    bar.add(
                        _row(
                            c,
                            {
                                "mode": m,
                                "status": "invalid",
                                "detail": f"token={c.token}",
                            },
                            meta,
                            BAR_COLS,
                        )
                    )
    cases = [c for c in cases if c.kind == "fused" and c.token > 0]
    groups, _ = runner.make_groups(cases, modes_req, bar.done)
    groups = runner.shard_groups(groups, runner.parse_shard(args.shard))
    opts = {"case_timeout": args.case_timeout, "setup_timeout": args.setup_timeout}
    sources = args.sources.split(",") if args.sources else None
    jobs = [
        BarGroupJob(
            k,
            cs,
            modes_req,
            tree,
            tmpdir,
            args.launcher,
            bar,
            meta,
            opts,
            cand_table=cand,
            dispatch_status=dispatch_status,
            kind=kind,
            sources=sources,
            source_modules=args.source_module or (),
            screen=args.screen,
        )
        for k, cs in groups
    ]
    for j in jobs:
        j.main_failures = main_failures
        j.dispatch_us = dispatch_us
    print(
        f"{prefix}: {sum(len(cs) for _, cs in groups)} cases in {len(jobs)} groups, modes {modes_req}",
        flush=True,
    )
    t0 = time.time()
    if runner.needs_shim(tree):
        shim = runner.make_shim(tmpdir)
        for j in jobs:
            j.pkg = "moe_bench_shim"
            j.extra_path = shim
    runner.run_jobs(jobs, args.jobs)
    bar.close()
    cand.close()
    if kind == "bar":
        check_discarded_sources(args.out_dir, tier, prefix)
    if getattr(args, "inner", False):
        open(os.path.join(args.out_dir, ".done"), "w").close()
    print(f"{prefix} finished in {time.time() - t0:.0f}s -> {args.out_dir}", flush=True)
    return 0
