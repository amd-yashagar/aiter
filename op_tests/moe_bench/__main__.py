# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE regression gate: ``python -m op_tests.moe_bench <command>``.

Commands: cases, routes, coverage, run, affected, compare, merge, gpus.
The harness measures any aiter tree given as ``--aiter <path>``.
"""

import argparse
import os
import sys
import tempfile

DEFAULT_TIERS = os.environ.get("MOE_BENCH_TIERS", "")


def add_launcher(p):
    p.add_argument(
        "--launcher",
        choices=("claim", "gpuq", "none"),
        default=os.environ.get("MOE_BENCH_LAUNCHER", "claim"),
        help="claim: claim file + HIP_VISIBLE_DEVICES on an idle GPU (meta, dell300x); "
        "gpuq: GPUQ_USER=yashagar gpuq --num-gpus=1 per worker (smci355); "
        "none: inherit the environment",
    )
    p.add_argument(
        "--exclude-gpus",
        default=os.environ.get("MOE_BENCH_EXCLUDE_GPUS", ""),
        help="comma separated HIP indexes the claim launcher never takes (e.g. 0,3); "
        "also MOE_BENCH_EXCLUDE_GPUS",
    )
    p.add_argument("--tmpdir", default=None, help="private TMPDIR for workers")
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=15.0,
        help="pause at start while the work disk has less "
        "free space than this; 0 disables",
    )
    p.add_argument(
        "--wait", type=int, default=0, help="seconds to wait for an idle GPU"
    )


def cmd_cases(a):
    from . import schema

    paths = schema.tier_paths(a.tiers_dir)
    problems, warnings, total = [], [], {}
    for name, p in paths.items():
        cases = schema.load_tier(p)
        total[name] = len(cases)
        errs, warns = schema.validate(cases, name)
        problems += errs
        warnings += warns
    for name, n in total.items():
        print(f"{name}: {n} rows")
    for pr in problems[:100]:
        print("PROBLEM", pr)
    for pr in warnings[:100]:
        print("WARN", pr)
    if a.aiter:
        from .catalogue import regen

        rows = regen.build_tier3(regen.build_matrix(os.path.abspath(a.aiter)))
        new_sigs = {
            schema.case_signature(schema.Case(r, schema.parse_extra(r["extra"]))): r
            for r in rows
        }
        old = schema.load_tier(paths["tier3"]) if "tier3" in paths else []
        old_sigs = {schema.case_signature(c) for c in old}
        added = [r for s, r in new_sigs.items() if s not in old_sigs]
        gone = len(old_sigs - set(new_sigs))
        print(
            f"tier3 from {a.aiter}: {len(rows)} cases; {len(added)} not in the shared file, "
            f"{gone} of the shared file's rows absent from this tree (never dropped)"
        )
        if a.out:
            schema.write_tier(a.out, rows)
            print("wrote", a.out)
        if a.update and added and "tier3" in paths:
            n = len(old)
            for r in added:
                r["case_id"] = f"tier3-{n:05d}"
                r["reason"] = f"new in tree {os.path.basename(a.aiter.rstrip('/'))}"
                r["added_by"] = a.added_by
                n += 1
            with open(paths["tier3"], "a", newline="") as f:
                import csv

                w = csv.DictWriter(f, fieldnames=schema.COLS)
                for r in added:
                    w.writerow({k: r.get(k, "") for k in schema.COLS})
            print(f"appended {len(added)} rows to {paths['tier3']}")
    return 1 if problems else 0


def _guard_builds(a, runner):
    """A leased run must never compile or plan: it reads the plan the host prebuild wrote
    (refusing at once, naming the cases, when a case has none), refuses when a module that
    should exist is missing, and makes workers fail a candidate instead of building. A run
    that is not a lease does not plan at all; it only warns when a stored plan says modules
    are missing. Everything is looked up in the --aiter tree under test."""
    from . import prebuild

    prebuild.use_tree(a.aiter)
    run_only_env = (
        os.environ.get(prebuild.NO_BUILD_ENV) == "1"
        or os.environ.get("FLYDSL_RUNTIME_RUN_ONLY") == "1"
    )
    lease = (a.launcher == "gpuq" or a.no_build or run_only_env) and not a.allow_build
    cases = runner.select_cases(a.cases.split(","), a.filter)
    miss = []
    if lease:
        # the parent of a lease must not JIT-build either (planning used to)
        os.environ[prebuild.NO_BUILD_ENV] = "1"
        prebuild.forbid_builds()
        need = prebuild.leased_plan(cases)
        prebuild.require_prebuilt(cases, need)
        fly = prebuild.flydsl_cases_missing(cases)
        if fly:
            raise SystemExit(
                f"{len(fly)} cases have FlyDSL candidates that no host build covered "
                f"(cache {prebuild.flydsl_cache_dir()}); a leased run must not compile. "
                "Run `run --mode prebuild` first, or pass --allow-build:\n  "
                + "\n  ".join(fly[:20])
            )
        prebuild.arm_flydsl_run_only()
    else:
        if os.path.exists(prebuild.manifest_path()):
            os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = prebuild.flydsl_cache_dir()
        try:
            need = prebuild.leased_plan(cases)
        except SystemExit:
            need = (
                {}
            )  # no current plan: nothing is known about the modules, nothing planned
        bad = prebuild.known_failed()
        miss = [m for m in prebuild.missing(need) if m not in bad]
    if not miss:
        return
    print(
        f"WARNING: {len(miss)} JIT modules are not built; this run will compile them "
        f"on its GPU: {miss[:5]}{' ...' if len(miss) > 5 else ''}",
        flush=True,
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["pipeline"]:
        from . import pipeline

        return pipeline.main(args[1:])
    ap = argparse.ArgumentParser(prog="moe_bench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "cases", help="validate tier files; regenerate tier 3 from a tree"
    )
    p.add_argument("--tiers-dir", default=DEFAULT_TIERS)
    p.add_argument("--aiter", help="tree to regenerate tier 3 from")
    p.add_argument("--out", help="write the regenerated tier 3 here")
    p.add_argument(
        "--update",
        action="store_true",
        help="append new tier-3 rows to the shared file",
    )
    p.add_argument("--added-by", default="moe-bench-harness")
    p.set_defaults(fn=cmd_cases)

    p = sub.add_parser("routes", help="dispatch decision per case, no timing")
    p.add_argument("--aiter", required=True)
    p.add_argument(
        "--cases", required=True, help="tier CSV (comma separated for several)"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--filter")
    add_launcher(p)
    p.set_defaults(fn="routes")

    p = sub.add_parser("routes-worker", help=argparse.SUPPRESS)
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--filter")
    p.set_defaults(fn="routes-worker")

    p = sub.add_parser("coverage", help="every live route key at least once in tier 1")
    p.add_argument("--tier1", required=True)
    p.add_argument("--tiers-dir", default=DEFAULT_TIERS)
    p.add_argument("--routes-tier1", required=True)
    p.add_argument(
        "--routes-other", required=True, help="tier2-x=routes.csv,tier3=routes.csv"
    )
    p.add_argument("--arch", default="gfx942")
    p.add_argument("--added-by", default="meta moe-bench-harness")
    p.add_argument("--append", action="store_true")
    p.set_defaults(fn="coverage")

    p = sub.add_parser("run", help="correctness and timing")
    p.add_argument("--aiter", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--filter")
    p.add_argument("--modes", default="warm,cold")
    p.add_argument("--out", help="result CSV (dispatch mode)")
    p.add_argument(
        "--mode",
        choices=("dispatch", "bar", "new", "prebuild"),
        default="dispatch",
        help="dispatch: fused_moe as dispatched; bar: best of every existing candidate; "
        "new: the new entry point (MOE_BENCH_NEW_ENTRY); prebuild: JIT-build the modules "
        "the cases need on the CPU (no GPU lease), then exit",
    )
    p.add_argument(
        "--screen",
        choices=("full", "two-phase"),
        default="full",
        help="bar mode: full = marginal-effect pair screening with a one-seed accuracy "
        "check per screened candidate; two-phase = a timing-only warm screen of every "
        "candidate, then the full measurement and accuracy for the top 8 and anything "
        "within 5%% of the best (widened if every finalist fails)",
    )
    p.add_argument(
        "--prebuild-only",
        action="store_true",
        help="build the modules the cases need (as --mode prebuild) and exit",
    )
    p.add_argument(
        "--prebuild-jobs",
        type=int,
        help="modules built at once (default cores/16; each gets MAX_JOBS=cores/jobs)",
    )
    p.add_argument(
        "--no-flydsl-prebuild",
        action="store_true",
        help="prebuild: skip the FlyDSL host build (CK and static modules only)",
    )
    p.add_argument(
        "--flydsl-only",
        action="store_true",
        help="prebuild: only the FlyDSL host build and the vecadd probe (no CK or static "
        "JIT modules: those are built for GPU_ARCHS and must not be built for another "
        "arch into a shared AITER_JIT_DIR)",
    )
    p.add_argument(
        "--flydsl-procs",
        type=int,
        help="FlyDSL host-build processes at once (default cores/4)",
    )
    p.add_argument(
        "--flydsl-verify",
        type=int,
        default=80,
        help="per case, run-only check of this many FlyDSL candidates the cover did "
        "not build (a miss means the cover is incomplete; 0 skips)",
    )
    p.add_argument(
        "--flydsl-per-proc",
        type=int,
        default=40,
        help="FlyDSL candidates per host-build process slice",
    )
    p.add_argument(
        "--prebuild-retry",
        action="store_true",
        help="try again modules a previous prebuild could not build",
    )
    p.add_argument(
        "--prebuild-cores",
        type=int,
        help="cores the prebuild may use (default: all); lower it on a node that is "
        "measuring at the same time",
    )
    p.add_argument(
        "--no-build",
        action="store_true",
        help="run as a lease does whatever the launcher: refuse to start if a needed JIT "
        "module or FlyDSL case is not prebuilt, and skip (log `not prebuilt: <module>`) "
        "any candidate that would compile",
    )
    p.add_argument(
        "--allow-build",
        action="store_true",
        help="let the run compile missing modules (the default under --launcher gpuq "
        "is to refuse: a leased run must only measure)",
    )
    p.add_argument(
        "--out-dir", help="bar/new modes: baseline folder for bar-<tier>.csv etc."
    )
    p.add_argument(
        "--tier-name", help="bar/new modes: file suffix, default: cases file name"
    )
    p.add_argument(
        "--dispatch-results", help="bar/new modes: dispatch results of the tier"
    )
    p.add_argument("--sources", help="bar mode: comma list of candidate sources")
    p.add_argument(
        "--main-failures",
        help="known main failures CSV (default: next to the cases file)",
    )
    p.add_argument(
        "--new-aiter",
        help="new mode: tree with the new entry point (workers import aiter from it)",
    )
    p.add_argument(
        "--new-entry", help="new mode: pkg.module (prepare/forward) or pkg.module:fn"
    )
    p.add_argument(
        "--new-routes",
        help="new mode: kernel id or route file for AITER_MOE_ROUTES; also sets "
        "AITER_MOE_STRICT=1 so a fallback raises instead of posing as the kernel",
    )
    p.add_argument(
        "--no-off",
        action="store_true",
        help="new mode: skip the routing-off pass (AITER_MOE_DISABLE=1, off_us)",
    )
    p.add_argument(
        "--first",
        type=int,
        default=0,
        help="new mode: first-call.csv for this many weight sets (warm and empty FlyDSL cache)",
    )
    p.add_argument(
        "--source-module", action="append", help="module that registers sources"
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--shard", help="i/n")
    p.add_argument(
        "--batches",
        type=int,
        default=1,
        help="gpuq launcher: split the groups into this many batches, one lease each",
    )
    p.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--jobs", type=int, default=1, help="concurrent workers, one GPU each"
    )
    p.add_argument("--node")
    p.add_argument("--case-timeout", type=int, default=1200)
    p.add_argument("--setup-timeout", type=int, default=3600)
    p.add_argument("--baseline-tree", help="drift check: re-time anchors on this tree")
    p.add_argument("--baseline-results", help="drift check: stored baseline CSV")
    p.add_argument("--drift-anchors", type=int, default=10)
    add_launcher(p)
    p.set_defaults(fn="run")

    p = sub.add_parser("affected", help="cases whose route or route sources changed")
    p.add_argument("--base", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--base-routes")
    p.add_argument("--new-routes")
    p.add_argument("--out")
    add_launcher(p)
    p.set_defaults(fn="affected")

    p = sub.add_parser(
        "compare", help="speedups, regressions, correctness; exit 1 on any"
    )
    p.add_argument("base")
    p.add_argument("new")
    p.add_argument("--out")
    p.add_argument("--drift", help="drift.json written by run")
    p.add_argument(
        "--bar", help="bar-<tier>.csv: also report against bar_us and roofline"
    )
    p.add_argument("--bar-out", help="write the per-case bar/gap/roofline table here")
    p.add_argument("--queue-dir", help="rebuild <dir>/<gpu>.csv, cases not yet won")
    p.add_argument(
        "--weights", help="model-weights.csv (model, weight); unlisted = 1.0"
    )
    p.add_argument("--ab", action="store_true")
    p.add_argument("--ab-base-tree")
    p.add_argument("--ab-new-tree")
    p.add_argument("--cases")
    p.add_argument("--rounds", type=int, default=3)
    add_launcher(p)
    p.set_defaults(fn="compare")

    p = sub.add_parser("merge", help="merge repeats into medians with per-case spread")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.set_defaults(fn="merge")

    p = sub.add_parser("gpus", help="idle check and claim state")
    p.set_defaults(fn="gpus")

    a = ap.parse_args(argv)
    if getattr(a, "exclude_gpus", ""):
        os.environ["MOE_BENCH_EXCLUDE_GPUS"] = a.exclude_gpus
    if a.cmd == "cases":
        return a.fn(a)
    if a.fn == "routes":
        from . import routes_cmd

        return routes_cmd.cmd_routes(a)
    if a.fn == "routes-worker":
        from . import routes_cmd

        return routes_cmd.routes_worker(a)
    if a.fn == "coverage":
        from . import routes_cmd

        return routes_cmd.cmd_coverage(a)
    if a.fn == "run":
        from . import prebuild as _prebuild
        from . import runner as _runner

        _prebuild.use_tree(a.aiter)

        _runner.wait_for_disk(tempfile.gettempdir(), a.min_free_gb)
        if a.mode == "prebuild" or a.prebuild_only:
            from . import prebuild

            prebuild.use_tree(a.aiter)
            res = {"failed": []}
            if not a.flydsl_only:
                res = prebuild.run_prebuild(
                    _runner.select_cases(a.cases.split(","), a.filter),
                    tree=a.aiter,
                    tmpdir=a.tmpdir,
                    jobs=a.prebuild_jobs,
                    min_free_gb=a.min_free_gb,
                    cores=a.prebuild_cores,
                    retry_failed=a.prebuild_retry,
                )
            bad = bool(res["failed"])
            if not a.no_flydsl_prebuild:
                cs = _runner.select_cases(a.cases.split(","), a.filter)
                bad |= not prebuild.prebuild_probe(tree=a.aiter, tmpdir=a.tmpdir)
                fly = prebuild.run_flydsl_prebuild(
                    cs,
                    procs=a.flydsl_procs,
                    per_proc=a.flydsl_per_proc,
                    verify=a.flydsl_verify,
                    tree=a.aiter,
                    tmpdir=a.tmpdir,
                )
                bad |= any(not r["built"] for r in fly.values())
            return 1 if bad else 0
        _guard_builds(a, _runner)
        if a.launcher == "gpuq" and not a.inner:
            from . import batch

            return batch.run_batched(a, list(sys.argv[1:] if argv is None else argv))
        if a.mode != "dispatch":
            from . import bar_runner

            if not a.out_dir:
                ap.error("--out-dir is required for bar and new modes")
            a.kind = a.mode
            return bar_runner.cmd_bar(a)
        if not a.out:
            ap.error("--out is required for dispatch mode")
        from . import runner

        return runner.cmd_run(a)
    if a.fn == "affected":
        from . import affected

        return affected.cmd_affected_main(a)
    if a.fn == "compare":
        from . import compare

        return compare.cmd_compare(a)
    if a.fn == "merge":
        from . import compare

        return compare.cmd_merge(a)
    if a.fn == "gpus":
        from . import gpu_claim

        return gpu_claim.main(["--list"])
    return 2


if __name__ == "__main__":
    sys.exit(main())
