# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only JIT prebuild of the aiter modules a tier's candidates need.

A leased GPU run should only measure. ``plan`` lists the modules the cases need
(deduplicated across the whole batch), ``run_prebuild`` builds the missing ones with no GPU
visible, several at a time, into the shared ``AITER_JIT_DIR`` (each module is built once;
aiter's own per-module lock makes concurrent builders safe), and ``require_prebuilt`` /
``forbid_builds`` make a leased run refuse to start, or fail a candidate, instead of
compiling.

What builds without a GPU: the CK 2-stage modules (``module_moe_ck2stages_*``, named by
dtypes, activation, quant type, routed-weight stage, split-k and runtime swiglu limit), the
CK-Tile module (``module_moe_cktile2stages``, a8w4) and the static modules the ASM and
quant paths load (``module_moe_asm``, ``module_moe_fmoe_asm``, ``module_quant``,
``module_aiter_core``). What cannot: FlyDSL and Triton kernels compile when first launched
for a shape (they need a device context) and are cached afterwards, so their first call
stays on the lease; ASM kernels are prebuilt code objects with nothing to build.

    python -m op_tests.moe_bench run --mode prebuild --aiter . --cases tier1.csv
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

STATIC = ("module_aiter_core", "module_moe_asm", "module_moe_fmoe_asm", "module_quant")
NO_BUILD_ENV = "MOE_BENCH_NO_BUILD"
COMPILE_FAILED = "failed !!!!!!"  # aiter.jit.core.raise_build_error


def jit_dir():
    from aiter.jit.core import get_user_jit_dir

    return get_user_jit_dir()


def module_path(md_name):
    return os.path.join(jit_dir(), md_name + ".so")


def _ck(S, stage, splitk=False, runtime_limit=False):
    from aiter.ops.moe_op import get_moe_stage_module

    if stage == 1:
        a, mul = S.aq, (1 if S.doweight else 2)
    else:
        a, mul = S.aq, (2 if not S.doweight else 1)
    if S.qtype.name == "No":
        a = S.dtype
    out = S.dtype
    md, cmd = get_moe_stage_module(
        a, S.wq, out, S.act, S.qtype, mul, True, splitk, runtime_limit
    )
    return {"md_name": md, "blob_gen_cmd": cmd, "kind": "ck2stages"}


def _tuner_info(S):
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    return (
        get_gfx(),
        get_cu_num(),
        S.M,
        S.D,
        S.I,
        S.E,
        S.topk,
        S.act,
        S.dtype,
        S.aq,
        S.wq,
        S.qtype,
        bool(S.g1u1),
        bool(S.doweight),
    )


def modules_for_case(S):
    """{md_name: spec} the case's candidates can load, from the case alone.

    Whether CK 2-stage candidates exist at all is aiter's tuner's answer
    (``gen_2stages_task``: CK has no SwiGLU for MX or per_1x128, no a16wi4, ...), so the
    modules follow the tasks it generates and nothing is built for a combination it would
    never run. Stage 1 and stage 2 modules follow the dtypes the case implies; a task
    named ``Splitk`` adds the split-k stage 1; per_1x128 SiLU with an active swiglu limit
    adds the runtime-limit stage 1. a8w4 uses the CK-Tile module instead.
    """
    from aiter import dtypes
    from aiter.ops.moe_limits import normalize_swiglu_limit

    from .candidates.fused_cfg import BLOCK_MS, tuner

    mods = {}
    info = _tuner_info(S)
    if S.qtype.name == "per_1x32" and S.aq == dtypes.fp8 and S.wq == dtypes.fp4x2:
        mods["module_moe_cktile2stages"] = {
            "md_name": "module_moe_cktile2stages",
            "kind": "cktile",
        }
    else:
        try:
            tasks = tuner().gen_2stages_task(info, BLOCK_MS)
        except Exception:  # noqa: BLE001  (the codegen refuses the case: no CK module)
            tasks = []
        if tasks:
            stage1 = [_ck(S, 1)]
            if (
                S.qtype.name == "per_1x128"
                and S.aq == dtypes.fp8
                and S.wq == dtypes.fp8
                and any(S.D % k == 0 and (S.D // k) % 128 == 0 for k in range(2, 9))
            ):
                stage1.append(_ck(S, 1, splitk=True))
            if (
                S.act.name == "Silu"
                and S.qtype.name == "per_1x128"
                and normalize_swiglu_limit(S.swiglu_limit)
            ):
                stage1.append(_ck(S, 1, runtime_limit=True))
            for spec in [*stage1, _ck(S, 2)]:
                mods[spec["md_name"]] = spec
    for name in STATIC:
        mods[name] = {"md_name": name, "kind": "static"}
    return mods


def plan(cases):
    """Deduplicated {md_name: spec} over every runnable fused case, with the case ids that
    need each module."""
    from . import workload

    need = {}
    for c in cases:
        if c.kind != "fused" or c.token <= 0:
            continue
        try:
            S = workload.resolve(c)
        except Exception:  # noqa: BLE001, S112  (invalid rows are reported by run)
            continue
        for md, spec in modules_for_case(S).items():
            need.setdefault(md, dict(spec, cases=[]))["cases"].append(c.id)
    return need


def failed_path():
    return os.path.join(jit_dir(), "prebuild_failed.json")


def known_failed():
    """{md_name: error tail} of modules a prebuild tried and could not build: the
    candidates that load them fail the same way on the lease, so they do not block it.
    """
    try:
        with open(failed_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def missing(need):
    return sorted(md for md in need if not os.path.exists(module_path(md)))


def require_prebuilt(cases):
    """Raise SystemExit naming every missing module: a leased run must never compile."""
    bad = known_failed()
    miss = [m for m in missing(plan(cases)) if m not in bad]
    if miss:
        raise SystemExit(
            f"{len(miss)} JIT modules are not built in {jit_dir()}; a leased run must "
            "not compile. Run `run --mode prebuild` (CPU, no lease) first, or pass "
            "--allow-build:\n  " + "\n  ".join(miss)
        )


def forbid_builds():
    """In a worker: turn any JIT build into an error (``MOE_BENCH_NO_BUILD=1``)."""
    if os.environ.get(NO_BUILD_ENV) != "1":
        return
    from aiter.jit import core

    def refuse(md_name, *a, **k):
        raise RuntimeError(
            f"build forbidden in a leased run ({NO_BUILD_ENV}=1): {md_name} is not "
            "prebuilt; run `run --mode prebuild` first"
        )

    core.build_module = refuse


def free_gb(path):
    return shutil.disk_usage(path).free / 2**30


def build_one(spec):
    """Build one module in this process (no GPU needed); returns True if its .so exists."""
    from aiter.jit import core

    md = spec["md_name"]
    if os.path.exists(module_path(md)):
        return True
    base = "module_moe_ck2stages" if spec["kind"] == "ck2stages" else md
    d = core.get_args_of_build(base)
    if spec["kind"] == "ck2stages":
        d.update(md_name=md, blob_gen_cmd=spec["blob_gen_cmd"])
    core.build_module(
        md,
        d["srcs"],
        d["flags_extra_cc"],
        d["flags_extra_hip"],
        d["blob_gen_cmd"],
        d["extra_include"],
        d["extra_ldflags"],
        d["verbose"],
        d["is_python_module"],
        d["is_standalone"],
        d["torch_exclude"],
        d.get("third_party", []),
        d.get("hipify", False),
        flags_extra_hip_per_source=d.get("flags_extra_hip_per_source", {}),
    )
    return os.path.exists(module_path(md))


def _child_env(max_jobs):
    from aiter.jit.utils.chip_info import get_gfx

    env = dict(os.environ)
    # Empty values hide the GPU on every ROCm stack; "-1" aborts the runtime on TheRock
    # wheels ("Sequence '-1' not recognized"). With ROCR_VISIBLE_DEVICES empty rocminfo
    # sees no chip, so the arch is named explicitly and the child never has to look.
    env.update(
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        CUDA_VISIBLE_DEVICES="",
        GPU_ARCHS=os.environ.get("GPU_ARCHS") or get_gfx(),
        MAX_JOBS=str(max_jobs),
    )
    env.pop(NO_BUILD_ENV, None)
    return env


def run_prebuild(
    cases,
    jobs=None,
    min_free_gb=10.0,
    log=print,
    timeout=3 * 3600,
    cores=None,
    retry_failed=False,
):
    """Build every missing module of ``cases``, ``jobs`` at a time, GPU hidden.

    ``cores`` (default: every core) is split between the builders: each gets
    ``MAX_JOBS = cores // jobs`` for its own compile parallelism. Waits
    while the disk has under ``min_free_gb`` free. Returns {"built", "present", "failed"}.
    """
    need = plan(cases)
    bad = {} if retry_failed else known_failed()
    todo = [need[m] for m in missing(need) if m not in bad]
    for m in missing(need):
        if m in bad:
            log(
                f"prebuild: skip {m}: failed before ({bad[m][:80]!r}); --prebuild-retry"
            )
    present = sorted(set(need) - {s["md_name"] for s in todo})
    log(
        f"prebuild: {len(need)} modules needed, {len(present)} present, "
        f"{len(todo)} to build in {jit_dir()}"
    )
    cores = cores or os.cpu_count() or 8
    jobs = max(1, min(jobs or max(1, cores // 16), len(todo) or 1))
    per = max(4, cores // jobs)
    built, failed, tails = [], [], {}
    t0 = time.time()

    def one(spec):
        while free_gb(jit_dir()) < min_free_gb:
            log(
                f"prebuild: paused, {free_gb(jit_dir()):.1f} GiB free < {min_free_gb:g}"
            )
            time.sleep(60)
        t = time.time()
        p = subprocess.run(
            [
                sys.executable,
                "-m",
                "op_tests.moe_bench.prebuild",
                "--build",
                json.dumps(spec),
            ],
            env=_child_env(per),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        ok = os.path.exists(module_path(spec["md_name"]))
        return spec["md_name"], ok, time.time() - t, p.stdout[-400:] + p.stderr[-600:]

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = [pool.submit(one, s) for s in todo]
        for f in as_completed(futs):
            md, ok, sec, tail = f.result()
            (built if ok else failed).append(md)
            tails[md] = tail[-300:]
            log(f"prebuild: {'built' if ok else 'FAILED'} {md} in {sec:.0f}s")
            if not ok:
                log("  " + tail.replace("\n", "\n  "))
    # only a compile that ran and failed is a property of the module; anything else
    # (the environment, a killed child) is retried by the next prebuild
    record = known_failed()
    genuine = {m for m in failed if COMPILE_FAILED in tails.get(m, "")}
    record = {
        m: v
        for m, v in record.items()
        if m not in built and not os.path.exists(module_path(m))
    }
    record.update({m: tails[m] for m in genuine})
    if record != known_failed():
        with open(failed_path(), "w") as f:
            json.dump(record, f, indent=1)
    transient = sorted(set(failed) - genuine)
    if transient:
        log(
            f"prebuild: {len(transient)} failures were not compile errors (not recorded, "
            "retried next time): " + ", ".join(transient[:5])
        )
    log(
        f"prebuild: {len(built)} built, {len(failed)} failed, {len(present)} already "
        f"present, {time.time() - t0:.0f}s ({jobs} at a time, MAX_JOBS={per} each)"
    )
    return {"built": built, "present": present, "failed": failed}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", help="JSON module spec: build it in this process")
    a = ap.parse_args(argv)
    if a.build:
        spec = json.loads(a.build)
        return 0 if build_one(spec) else 1
    ap.error("--build is required (use `run --mode prebuild` for a tier)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
