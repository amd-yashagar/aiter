# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``affected``: cases whose route key changed, or whose route's sources changed.

The gate re-runs only these plus tier 1. Changed files come from git (the base
commit against the new working tree) and fall back to a file-hash walk when the
base commit is not in the new tree's repository.
"""

import csv
import hashlib
import os
import subprocess

from . import routes_cmd, runner, schema

COMMON = ["aiter/fused_moe.py", "aiter/jit/core.py", "aiter/ops/moe_op.py"]
SORT = [
    "csrc/py_itfs_cu/moe_sorting_opus_kernels.cu",
    "csrc/kernels/moe_align_block_size_kernels.cu",
    "csrc/include/moe_sorting",
    "aiter/ops/moe_sorting",
    "aiter/ops/opus/moe_sorting",
]
QUANT = [
    "csrc/kernels/quant_kernels.cu",
    "csrc/include/quant",
    "aiter/ops/quant.py",
    "csrc/kernels/mxfp4_moe/",
    "csrc/kernels/quant_mxfp4.cu",
    "aiter/ops/moe_mxfp4_aux.py",
    "csrc/include/moe_mxfp4_aux.h",
]
BACKEND_SOURCES = {
    "ck": [
        "csrc/ck_gemm_moe_2stages_codegen/",
        "csrc/py_itfs_ck/moe_ck",
        "csrc/include/moe_ck.h",
        "3rdparty/composable_kernel",
    ],
    "ck_tile": ["csrc/ck_tile_gemm_moe_2stages/", "3rdparty/composable_kernel"],
    "asm": [
        "hsa/codegen.py",
        "csrc/py_itfs_cu/asm_fmoe.cu",
        "csrc/py_itfs_cu/asm_moe_2stage.cu",
    ],
    "flydsl": ["aiter/ops/flydsl/", "aiter/aot/flydsl/"],
    "flydsl-graph": ["aiter/ops/flydsl/", "aiter/aot/flydsl/"],
    "opus": ["csrc/opus_moe/", "aiter/ops/opus/"],
}
ROUTING = [
    "csrc/kernels/topk_softmax_kernels",
    "csrc/kernels/topk_gating",
    "csrc/kernels/topk_plain_kernels.cu",
    "csrc/kernels/moe_fused_gate.cu",
    "aiter/ops/topk.py",
    "aiter/ops/moe_topk",
]


def route_sources(route, kind):
    """Source path prefixes a route depends on, for the changed-file match."""
    if kind == "routing":
        return ROUTING
    if kind == "sorting":
        return SORT + QUANT
    out = list(COMMON) + list(SORT) + list(QUANT)
    for b in (route.get("backend1", ""), route.get("backend2", "")):
        out += BACKEND_SOURCES.get(b, [])
    fam = route.get("route_family", "")
    for b, srcs in BACKEND_SOURCES.items():
        if b in fam:
            out += srcs
    if route.get("backend1") == "asm" or "asm" in fam:
        arch = "gfx942"
        out += [f"hsa/{arch}/fmoe", "hsa/gfx950/fmoe"]
    return sorted(set(out))


def changed_files(base, new):
    """Paths that differ between two trees (relative, forward slashes)."""
    base_sha = runner.git(["rev-parse", "HEAD"], base)
    r = subprocess.run(
        ["git", "diff", "--name-only", base_sha],
        cwd=new,
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode == 0 and base_sha:
        files = set(r.stdout.split())
        files |= set(
            subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=new,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.split()
        )
        base_dirty = runner.git(["diff", "--name-only"], base).split()
        return sorted(files | set(base_dirty))
    return walk_diff(base, new)


def walk_diff(base, new, roots=("aiter", "csrc", "hsa", "op_tests")):
    def digest(p):
        h = hashlib.sha1()
        with open(p, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    changed = set()
    for root in roots:
        seen = set()
        for tree, other in ((new, base), (base, new)):
            for dp, dn, fns in os.walk(os.path.join(tree, root)):
                dn[:] = [d for d in dn if d not in ("__pycache__", "build", "jit")]
                for fn in fns:
                    rel = os.path.relpath(os.path.join(dp, fn), tree)
                    if rel in seen or "moe_bench" in rel:
                        continue
                    seen.add(rel)
                    a, b = os.path.join(base, rel), os.path.join(new, rel)
                    if not (os.path.exists(a) and os.path.exists(b)) or digest(
                        a
                    ) != digest(b):
                        changed.add(rel)
    return sorted(changed)


def touches(changed, prefixes, kernel_names=()):
    for p in changed:
        if any(p.startswith(s) or s in p for s in prefixes):
            return True
        base = os.path.basename(p)
        if any(k and k in base for k in kernel_names):
            return True
    return False


def cmd_affected(args):
    cases = runner.select_cases(args.cases.split(","))
    by_id = {c.id: c for c in cases}
    base_r = {r["case_id"]: r for r in routes_cmd.read_routes(args.base_routes)}
    new_r = {r["case_id"]: r for r in routes_cmd.read_routes(args.new_routes)}
    changed = changed_files(os.path.abspath(args.base), os.path.abspath(args.new))
    out, why = [], {}
    for cid, c in by_id.items():
        b, n = base_r.get(cid), new_r.get(cid)
        if b is None or n is None:
            out.append(c)
            why[cid] = "no route record"
        elif b["route_key"] != n["route_key"]:
            out.append(c)
            why[cid] = "route key changed"
        elif touches(
            changed,
            route_sources(n, c.kind),
            (n.get("kernel1", ""), n.get("kernel2", "")),
        ):
            out.append(c)
            why[cid] = "route sources changed"
    print(f"{len(changed)} changed files; {len(out)} of {len(cases)} cases affected")
    reasons = {}
    for w in why.values():
        reasons[w] = reasons.get(w, 0) + 1
    print(reasons)
    if args.out:
        rows = [dict(c.row) for c in out]
        schema.write_tier(args.out, rows)
        with open(os.path.splitext(args.out)[0] + ".why.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "why"])
            for c in out:
                w.writerow([c.id, why[c.id]])
    else:
        for c in out:
            print(c.id, why[c.id])
    return 0


def _routes_for(tree, args, tag):
    """Route CSV for one tree (computed through the launcher when not given)."""
    from argparse import Namespace

    tmp = args.tmpdir or os.path.join("/tmp", "moe-bench-tmp")
    os.makedirs(tmp, exist_ok=True)
    sha = runner.git(["rev-parse", "HEAD"], tree)[:12]
    out = os.path.join(tmp, f"routes-{tag}-{sha}.csv")
    ns = Namespace(
        aiter=tree,
        cases=args.cases,
        out=out,
        filter=None,
        launcher=args.launcher,
        tmpdir=tmp,
        wait=args.wait,
    )
    rc = routes_cmd.cmd_routes(ns)
    if rc != 0:
        raise SystemExit(f"routes failed for {tree}")
    return out


def cmd_affected_main(args):
    if not args.base_routes:
        args.base_routes = _routes_for(os.path.abspath(args.base), args, "base")
    if not args.new_routes:
        args.new_routes = _routes_for(os.path.abspath(args.new), args, "new")
    return cmd_affected(args)
