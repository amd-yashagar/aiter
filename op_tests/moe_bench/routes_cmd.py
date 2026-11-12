# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``routes`` and route-coverage: dispatch per case, route keys per family."""

import collections
import csv
import os
import subprocess
import sys
import tempfile

from . import runner, schema
from .routes import ROUTE_COLS

OUT_COLS = ["case_id", "sig", "kind", "family", "token"] + ROUTE_COLS


def routes_worker(args):
    """Runs under the launcher (needs a GPU context); never launches a kernel."""
    from . import routes

    cases = runner.select_cases(args.cases.split(","), getattr(args, "filter", None))
    cache = {}
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS, extrasaction="ignore")
        w.writeheader()
        for i, c in enumerate(cases):
            if c.kind == "fused":
                sig = schema.case_signature(c)
                if sig not in cache:
                    try:
                        cache[sig] = routes.dry_route(c)
                    except Exception as e:  # noqa: BLE001
                        cache[sig] = {
                            "route_error": f"{type(e).__name__}: {str(e)[:200]}",
                            "route_key": "unsupported:error",
                        }
                r = cache[sig]
            elif c.kind in ("sorting", "routing"):
                r = routes.aux_route(c)
            else:
                r = {"route_key": f"not-run:{c.kind}", "route_family": c.kind}
            w.writerow(
                dict(
                    r,
                    case_id=c.id,
                    sig=schema.case_signature(c),
                    kind=c.kind,
                    family=c["family"],
                    token=c.row["token"],
                )
            )
            if i % 200 == 0:
                f.flush()
                print(f"routes {i}/{len(cases)}", flush=True)
    return 0


def cmd_routes(args):
    tree = os.path.abspath(args.aiter)
    tmpdir = args.tmpdir or os.path.join(tempfile.gettempdir(), "moe-bench-tmp")
    os.makedirs(tmpdir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "moe_bench_shim",
        "routes-worker",
        "--cases",
        args.cases,
        "--out",
        args.out,
    ]
    if args.filter:
        cmd += ["--filter", args.filter]
    env = runner.worker_env(tree, tmpdir, extra_path=runner.make_shim(tmpdir))
    if args.launcher == "gpuq":
        cmd = ["gpuq", "--num-gpus=1", *cmd]
        rc = subprocess.run(cmd, env=env, cwd=tree, check=False).returncode
    elif args.launcher == "claim":
        from . import gpu_claim

        gpus = gpu_claim.acquire(1, cmd, wait=args.wait)
        if not gpus:
            print("no idle unclaimed GPU", file=sys.stderr)
            return 3
        env["HIP_VISIBLE_DEVICES"] = str(gpus[0])
        try:
            rc = subprocess.run(cmd, env=env, cwd=tree, check=False).returncode
        finally:
            for g in gpus:
                gpu_claim.release_claim(g)
    else:
        rc = subprocess.run(cmd, env=env, cwd=tree, check=False).returncode
    if rc == 0:
        summarize(args.out)
    return rc


def read_routes(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def is_live(r):
    return not r["route_key"].startswith(("unsupported:", "not-run:"))


def summarize(path):
    rows = read_routes(path)
    live = [r for r in rows if is_live(r)]
    keys = collections.Counter((r["family"], r["route_key"]) for r in live)
    uns = collections.Counter(r["family"] for r in rows if not is_live(r))
    fams = sorted({r["family"] for r in live})
    print(
        f"{path}: {len(rows)} cases, {len(live)} with a route, {len(keys)} (family, route) keys "
        f"over {len(fams)} families; no route: {dict(uns)}"
    )


def weight_cost(c):
    try:
        e = schema.ep_local_experts(c)
        return e * int(c["model_dim"]) * int(c["inter_dim"]) * 3 + int(c["token"])
    except (ValueError, TypeError):
        return 1 << 60


def cmd_coverage(args):
    """Tier 1 must hold every (family, route key) live on this arch at least once.

    ``--routes-other`` lists ``<tier>=<routes csv>`` pairs for the larger tiers;
    each missing key gets one representative (the cheapest case with that key)
    appended to tier 1 with reason ``route-coverage <arch>``.
    """
    t1_cases = schema.load_tier(args.tier1)
    covered = {
        (r["family"], r["route_key"])
        for r in read_routes(args.routes_tier1)
        if is_live(r)
    }
    reps = {}
    for item in args.routes_other.split(","):
        name, rp = item.split("=", 1)
        by_id = {
            c.id: c
            for c in schema.load_tier(os.path.join(args.tiers_dir, f"{name}.csv"))
        }
        for r in read_routes(rp):
            if not is_live(r) or r["kind"] != "fused":
                continue
            key = (r["family"], r["route_key"])
            c = by_id.get(r["case_id"])
            if key in covered or c is None:
                continue
            cur = reps.get(key)
            if cur is None or weight_cost(c) < weight_cost(cur):
                reps[key] = c
    print(f"covered keys {len(covered)}, missing {len(reps)}")
    if not args.append:
        for (fam, key), c in sorted(reps.items()):
            print(f"  missing {fam}: {key[:140]}  via {c.id}")
        return 0
    n = len(t1_cases)
    sigs = {schema.case_signature(c) for c in t1_cases}
    rows = []
    for _key, c in sorted(reps.items()):
        if schema.case_signature(c) in sigs:
            continue
        r = dict(c.row)
        r["case_id"] = f"tier1-{n:05d}"
        n += 1
        r["reason"] = f"route-coverage {args.arch}"
        r["added_by"] = args.added_by
        r["modes"] = "warm,cold" if weight_cost(c) <= (1 << 33) else "warm"
        sigs.add(schema.case_signature(c))
        rows.append(r)
    with open(args.tier1, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=schema.COLS)
        for r in rows:
            w.writerow({k: r.get(k, "") for k in schema.COLS})
    print(f"appended {len(rows)} rows to {args.tier1}")
    return 0
