# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Prebuild batch N+1 on the CPU while batch N runs in its lease.

A bar sweep is split into batches of whole weight groups sized for about
``--batch-minutes`` of pure measurement. The driver prebuilds batch 0, then for every batch
starts the CPU prebuild of the next one and runs the current one (``run --mode bar``, the
leased step, which refuses to compile), so a lease only ever measures. Batches are resumable:
a rerun with the same ``--out-dir`` skips rows already written.

    python -m op_tests.moe_bench pipeline --aiter . --cases tier1.csv \\
        --out-dir out/bar-tier1 --tier-name tier1 --batch-minutes 15 \\
        --history out/bar-tier1/candidates-tier1.csv -- --launcher gpuq --screen two-phase

Everything after ``--`` is passed to every ``run`` (bar) call.
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time

DEFAULT_ROWS = {  # candidate rows per case when there is no history (rough)
    "a4w4": 600,
    "a8w4": 300,
    "mxfp8": 500,
    "a16w4": 500,
    "a16wi4": 150,
    "fp8-block": 150,
    "fp8-token": 100,
    "fp8-tensor": 100,
    "a16w16": 100,
    "int8": 60,
}
FALLBACK_ROWS = 100


def history_rows(path):
    """{family: median candidate rows per case} from a candidates CSV of a finished run."""
    if not path or not os.path.exists(path):
        return {}
    csv.field_size_limit(1 << 28)
    per_case, fam = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            per_case[r["case_id"]] = per_case.get(r["case_id"], 0) + 1
            fam[r["case_id"]] = r.get("family", "")
    out = {}
    for cid, n in per_case.items():
        out.setdefault(fam[cid], []).append(n)
    return {f: statistics.median(v) for f, v in out.items() if f}


def estimate_seconds(case, history, sec_per_row):
    """Pure measurement time of one case: candidate rows times seconds per row."""
    fam = case.row["family"]
    rows = history.get(fam) or DEFAULT_ROWS.get(fam, FALLBACK_ROWS)
    return (
        rows
        * sec_per_row
        * max(1, len([m for m in case.modes if m in ("warm", "cold")]))
    )


def make_batches(cases, minutes, history=None, sec_per_row=1.2):
    """Batches of whole weight groups, each estimated at about ``minutes`` of measurement.

    A group is never split (its weights are built once per batch); a group larger than the
    budget is a batch of its own.
    """
    from . import schema

    history = history or {}
    groups = {}
    for c in cases:
        if c.kind != "fused" or c.token <= 0:
            continue
        groups.setdefault(schema.weight_group_key(c), []).append(c)
    budget = minutes * 60.0
    batches, cur, cur_s = [], [], 0.0
    for g in groups.values():
        s = sum(estimate_seconds(c, history, sec_per_row) for c in g)
        if cur and cur_s + s > budget:
            batches.append((cur, cur_s))
            cur, cur_s = [], 0.0
        cur += g
        cur_s += s
    if cur:
        batches.append((cur, cur_s))
    return batches


def write_batch(path, cases):
    from . import schema

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=schema.COLS)
        w.writeheader()
        for c in cases:
            w.writerow({k: c.row.get(k, "") for k in schema.COLS})


def prebuild_cmd(args, batch_csv):
    cmd = [
        sys.executable, "-m", "op_tests.moe_bench", "run", "--mode", "prebuild",
        "--aiter", args.aiter, "--cases", batch_csv, "--min-free-gb", str(args.min_free_gb),
    ]  # fmt: skip
    if args.prebuild_jobs:
        cmd += ["--prebuild-jobs", str(args.prebuild_jobs)]
    if args.prebuild_cores:
        cmd += ["--prebuild-cores", str(args.prebuild_cores)]
    if args.prebuild_retry:
        cmd += ["--prebuild-retry"]
    return cmd


def run_cmd(args, batch_csv, extra):
    return [
        sys.executable, "-m", "op_tests.moe_bench", "run", "--mode", "bar",
        "--aiter", args.aiter, "--cases", batch_csv, "--out-dir", args.out_dir,
        "--tier-name", args.tier_name, "--resume", *extra,
    ]  # fmt: skip


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser()
    ap.add_argument("--aiter", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--filter")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tier-name", required=True)
    ap.add_argument("--batch-minutes", type=float, default=15.0)
    ap.add_argument("--history", help="candidates CSV of an earlier run, for sizing")
    ap.add_argument("--sec-per-row", type=float, default=1.2)
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    ap.add_argument("--prebuild-jobs", type=int)
    ap.add_argument("--prebuild-cores", type=int)
    ap.add_argument("--prebuild-retry", action="store_true")
    ap.add_argument(
        "--plan-only", action="store_true", help="print the batches and stop"
    )
    a = ap.parse_args(argv)

    from . import runner

    cases = runner.select_cases(a.cases.split(","), a.filter)
    batches = make_batches(
        cases, a.batch_minutes, history_rows(a.history), a.sec_per_row
    )
    work = os.path.join(a.out_dir, "pipeline")
    os.makedirs(work, exist_ok=True)
    paths = []
    for i, (cs, secs) in enumerate(batches):
        p = os.path.join(work, f"batch-{i:03d}.csv")
        write_batch(p, cs)
        paths.append(p)
        print(
            f"batch {i:03d}: {len(cs):4d} cases, about {secs / 60:5.1f} min", flush=True
        )
    if a.plan_only or not batches:
        return 0

    def start_prebuild(i):
        if i >= len(paths):
            return None
        return subprocess.Popen(
            prebuild_cmd(a, paths[i]), stdout=sys.stdout, stderr=sys.stderr
        )

    t0 = time.time()
    pre = start_prebuild(0)
    waited = 0.0
    for i, p in enumerate(paths):
        w = time.time()
        if pre is not None and pre.wait() != 0:
            print(
                f"prebuild of batch {i:03d} reported failures (modules it could not "
                "build are recorded; the run fails those candidates)",
                flush=True,
            )
        waited += time.time() - w
        pre = start_prebuild(i + 1)
        s = time.time()
        rc = subprocess.run(run_cmd(a, p, extra), check=False).returncode
        print(
            f"batch {i:03d} ran in {time.time() - s:.0f}s (exit {rc}); lease waited "
            f"{waited:.0f}s for prebuilds so far",
            flush=True,
        )
        if rc != 0:
            if pre is not None:
                pre.terminate()
            return rc
    print(
        f"pipeline done in {time.time() - t0:.0f}s; waited {waited:.0f}s on prebuild",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
