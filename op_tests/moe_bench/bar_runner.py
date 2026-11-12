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

from . import runner, schema

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
        "bar_us",
        "bar_id",
        "bar_source",
        "bar_entry",
        "bar_kernels",
        "bar_timing_mode",
        "n_candidates",
        "n_correct",
        "skipped_sources",
        "main_failure",
        "new_us",
        "new_id",
        "roofline_us",
        "roofline_staged_us",
        "roofline_serial_us",
        "bound",
        "mfma",
        "bar_roofline_eff",
        "dispatch_roofline_eff",
        "new_roofline_eff",
    ]
    + ["node", "gpu", "gfx", "cu_num", "rocm", "torch", "flydsl"]
    + ["aiter_sha", "aiter_dirty", "harness_sha", "env", "date"]
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
    "err_ratio",
    "logits_diff",
    "ref_pass",
    "route_key",
    "kernel1",
    "kernel2",
    "node",
    "gpu",
    "aiter_sha",
    "date",
]


class Table:
    """Append-only CSV with a flush per row."""

    def __init__(self, path, cols, resume, key=("case_id", "mode")):
        self.cols, self.key = cols, key
        self.done = set()
        new = not os.path.exists(path) or not resume
        if not new:
            for r in schema.read_csv(path):
                self.done.add(tuple(r[k] for k in key))
        self.fh = open(path, "w" if new else "a", newline="")  # noqa: SIM115
        self.w = csv.DictWriter(self.fh, fieldnames=cols, extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.fh.flush()

    def add(self, row):
        self.w.writerow(row)
        self.fh.flush()
        self.done.add(tuple(row.get(k, "") for k in self.key))

    def close(self):
        self.fh.close()


def _row(case, ev, meta, cols):
    r = {k: case.row.get(k, "") for k in cols if k in case.row}
    r["sig"] = runner.case_sig(case)
    for k in cols:
        if k in ev and k != "event":
            v = ev[k]
            r[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    for k in ("node", "gfx", "cu_num", "rocm", "torch", "flydsl", "aiter_sha", "date"):
        r.setdefault(k, meta.get(k, ""))
    for k in ("aiter_dirty", "harness_sha", "env"):
        r[k] = meta.get(k, "")
    r["gpu"] = ev.get("gpu") or meta.get("gpu", "")
    return r


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
        **k,
    ):
        super().__init__(*a, **k)
        self.cand_table = cand_table
        self.dispatch_status = dispatch_status or {}
        self.kind = kind
        self.sources = sources
        self.source_modules = list(source_modules)
        self.bad = {}
        self.main_failures = {}

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
            "tmpdir": self.tmpdir,
        }

    def handle_row(self, c, ev):
        self.results.add(_row(c, ev, self.meta, BAR_COLS))

    def handle_other(self, ev):
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


def cmd_bar(args):
    node = args.node or socket.gethostname()
    tmpdir = args.tmpdir or os.path.join(tempfile.gettempdir(), "moe-bench-tmp")
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tree = os.path.abspath(
        args.new_aiter if args.kind == "new" and args.new_aiter else args.aiter
    )
    if args.new_entry:
        os.environ["MOE_BENCH_NEW_ENTRY"] = args.new_entry
    tier = args.tier_name or tier_name(args.cases)
    cases = runner.select_cases(args.cases.split(","), args.filter)
    if getattr(args, "ids", None):
        cases = [c for c in cases if c.id in args.ids]
    modes_req = [m for m in args.modes.split(",") if m and m != "first"]
    dispatch_status = {}
    if args.dispatch_results:
        path = args.dispatch_results
        if os.path.isdir(path):
            path = os.path.join(path, f"{tier}.csv")
        for r in schema.read_csv(path):
            prev = dispatch_status.get(r["case_id"])
            if prev is None or r["status"] == "ok":
                dispatch_status[r["case_id"]] = r["status"]
    kind = args.kind
    main_failures = read_main_failures(args, args.cases)
    prefix = "new" if kind == "new" else "bar"
    bar = Table(
        os.path.join(args.out_dir, f"{prefix}-{tier}.csv"), BAR_COLS, args.resume
    )
    cand = Table(
        os.path.join(args.out_dir, f"candidates-{tier}.csv"),
        CAND_COLS,
        args.resume,
        key=("case_id", "cand_id", "mode", "phase"),
    )
    meta = runner.probe_meta(tree, node, args.launcher, tmpdir)
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
        )
        for k, cs in groups
    ]
    for j in jobs:
        j.main_failures = main_failures
    print(
        f"{prefix}: {sum(len(cs) for _, cs in groups)} cases in {len(jobs)} groups, modes {modes_req}",
        flush=True,
    )
    t0 = time.time()
    if kind == "new" and tree != os.path.abspath(args.aiter):
        shim = runner.make_shim(tmpdir)
        for j in jobs:
            j.pkg = "moe_bench_shim"
            j.extra_path = shim
    runner.run_jobs(jobs, args.jobs)
    bar.close()
    cand.close()
    if getattr(args, "inner", False):
        open(os.path.join(args.out_dir, ".done"), "w").close()
    print(f"{prefix} finished in {time.time() - t0:.0f}s -> {args.out_dir}", flush=True)
    return 0
