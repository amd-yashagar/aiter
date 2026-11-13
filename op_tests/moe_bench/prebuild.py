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
import contextlib
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .candidates import HarnessError

STATIC = (
    "module_aiter_core",
    "module_moe_asm",
    "module_moe_fmoe_asm",
    "module_quant",
    "module_moe_sorting_opus",  # the sort every fused_moe call starts with (CK sorting is opt-in)
)
NO_BUILD_ENV = "MOE_BENCH_NO_BUILD"
COMPILE_FAILED = "failed !!!!!!"  # aiter.jit.core.raise_build_error


def jit_dir():
    from aiter.jit.core import get_user_jit_dir

    return get_user_jit_dir()


def state_dir():
    """Where the prebuild's bookkeeping files live (``prebuild_failed.json``,
    ``prebuild_missing.json``, ``prebuild_keymods.json``, ``prebuild_attempted.json``): the
    JIT dir, unless ``MOE_BENCH_STATE_DIR`` names another (the tests do, so they never
    write into a shared JIT dir)."""
    return os.environ.get("MOE_BENCH_STATE_DIR") or jit_dir()


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


def modules_from_names(S, k1, k2):
    """Modules a route's kernel names load: a CK 2-stage name loads that stage's
    ``module_moe_ck2stages_*`` (dtypes, activation and quant type from the case, split-k
    from the name), a CK-Tile name loads ``module_moe_cktile2stages``."""
    mods = {}
    for stage, k in ((1, k1 or ""), (2, k2 or "")):
        if k.startswith("moe_ck2stages"):
            spec = _ck(S, stage, splitk=stage == 1 and "splitk" in k.lower())
            mods[spec["md_name"]] = spec
        elif "cktile" in k.lower():
            mods["module_moe_cktile2stages"] = {
                "md_name": "module_moe_cktile2stages",
                "kind": "cktile",
            }
    return mods


def _ck_or_none(S, stage, splitk):
    try:
        return _ck(S, stage, splitk=splitk)
    except (ValueError, KeyError):  # no CK module exists for this combination
        return None


def _unwrap(fn):
    """(function, merged keyword arguments) of a possibly nested functools.partial."""
    import functools

    kw = {}
    while isinstance(fn, functools.partial):
        kw = {**fn.keywords, **kw}
        fn = fn.func
    return fn, kw


def modules_from_metadata(S, md):
    """Modules a resolved ``MOEMetadata`` loads when it runs: read off its stage functions,
    not off the kernel names. A name aiter's resolver does not understand (a CK-Tile stage
    1 next to a FlyDSL stage 2 falls into the branch that wraps every other stage-1 name
    in ``ck_moe_stage1``) loads whatever function it was wrapped in."""
    import aiter
    import aiter.fused_moe as fm

    mods = {}
    for stage, fn in ((1, md.stage1), (2, md.stage2)):
        if fn is None:
            continue
        base, kw = _unwrap(fn)
        if base is fm.ck_moe_stage1 or base is aiter.ck_moe_stage2_fwd:
            kn = kw.get("kernelName") or ""
            splitk = stage == 1 and (
                "Splitk" in kn
                or (
                    not kn
                    and S.qtype.name == "per_1x128"
                    and int(kw.get("splitk") or 0) > 1
                )
            )
            spec = _ck_or_none(S, stage, splitk)
            if spec:
                mods[spec["md_name"]] = spec
        elif base in (fm.cktile_moe_stage1, fm.cktile_moe_stage2):
            mods["module_moe_cktile2stages"] = {
                "md_name": "module_moe_cktile2stages",
                "kind": "cktile",
            }
    return mods


def cand_key(c):
    """What decides a candidate's modules: its block_m group and the families of its two
    stages (the split-k stage 1 is its own module)."""
    import re

    m = getattr(c, "meta", None) or {}
    s1, s2 = m.get("s1") or c.kernels or "", m.get("s2") or ""
    return "|".join(
        [
            str(getattr(c, "group", "")),
            re.split(r"[_ ]", s1)[0],
            re.split(r"[_ ]", s2)[0],
            "splitk" if "splitk" in s1.lower() else "",
        ]
    )


def fallback_modules(S, case):
    """Modules of aiter's own default route for the case.

    A forced candidate row that aiter does not accept makes ``fused_moe`` fall back to its
    default heuristics, and that route's modules would be JIT-built inside the lease (the
    candidate is discarded only afterwards). The default route is the one an empty tuned
    table gives: resolved on meta tensors, no launch."""
    import tempfile

    from . import routes
    from .candidates import fused_cfg

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "empty_fmoe.csv")
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(fused_cfg.CSV_COLS)
        old = os.environ.get("AITER_CONFIG_FMOE")
        os.environ["AITER_CONFIG_FMOE"] = path
        fused_cfg.reset_router()
        try:
            route = routes.dry_route(case, S)
        finally:
            if old is None:
                os.environ.pop("AITER_CONFIG_FMOE", None)
            else:
                os.environ["AITER_CONFIG_FMOE"] = old
            fused_cfg.reset_router()
    return modules_from_names(S, route.get("kernel1", ""), route.get("kernel2", ""))


def candidate_modules(S, cands):
    """Modules of an enumeration's CK / CK-Tile candidates (by their kernel names)."""
    mods = {}
    for c in cands:
        meta = getattr(c, "meta", None) or {}
        if "s1" in meta:
            mods.update(modules_from_names(S, meta["s1"], meta["s2"]))
        else:
            mods.update(modules_from_names(S, c.kernels or "", ""))
    return mods


def routed_modules(S, case, cands, ctx):
    """Modules aiter loads for each kind of enumerated candidate, after its own decision.

    One representative per ``cand_key`` is forced and resolved on meta tensors
    (``routes.dry_route``: the metadata aiter ends up with, nothing launched), and the
    modules are read off the resolved stage functions (``modules_from_metadata``). That is
    the resolution the runtime does, including the fallbacks when aiter declines the
    forced row. Returns (modules, per-key modules {key: [md_name]}, representatives)."""
    from . import routes
    from .candidates import fused_cfg

    reps = {}
    for c in cands:
        if getattr(c, "row", None):
            reps.setdefault(cand_key(c), c)
    mods, by_key = {}, {}
    # aiter's own dispatch (the bar's "dispatch" candidate): shipped tuned rows included
    fused_cfg.reset_router()
    got = []
    routes.dry_route(case, S, capture=got)
    for md in got[-1:]:
        mods.update(modules_from_metadata(S, md))
    try:
        for key, c in reps.items():
            c.prepare(ctx)
            got = []
            routes.dry_route(case, S, capture=got)
            c.release(ctx)
            found = modules_from_metadata(S, got[-1]) if got else {}
            by_key[key] = sorted(found)
            mods.update(found)
    finally:
        os.environ.pop("AITER_CONFIG_FMOE", None)
        fused_cfg.reset_router()
    return mods, by_key, len(reps)


def keymods_path():
    return os.path.join(state_dir(), "prebuild_keymods.json")


def read_keymods():
    """{case_id: {cand_key: [md_name]}} written by the plan: which modules each kind of
    candidate loads."""
    try:
        with open(keymods_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


SPECS_KEY = "_specs"


def scope():
    """The tree under test and the chip: the part of a plan that two workers on one node
    may differ in. Keymods entries are kept per scope."""
    import aiter

    tree = os.path.dirname(os.path.dirname(os.path.abspath(aiter.__file__)))
    try:
        arch, cu = gpu_identity()
    except SystemExit:
        arch, cu = "", ""
    return f"{os.path.realpath(tree)}|{arch}/{cu}"


def _kmkey(case_id):
    return f"{case_id}|{scope()}"


_KEYMODS_LOCK = threading.Lock()


def record_keymods(case_id, by_key, specs=None):
    with _KEYMODS_LOCK:
        _record_keymods(case_id, by_key, specs)


def _record_keymods(case_id, by_key, specs=None):
    rec = read_keymods()
    rec[_kmkey(case_id)] = by_key
    if specs:
        rec.setdefault(f"{SPECS_KEY}|{scope()}", {}).update(specs)
    tmp = keymods_path() + f".{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, keymods_path())


def excluded_keys(case_id):
    """{cand_key: (module, error)} of the candidate kinds whose module failed to build on
    a host prebuild: they are marked ``build_failed`` and never reach a lease."""
    failed = known_failed()
    out = {}
    for key, mods in read_keymods().get(_kmkey(case_id), {}).items():
        for md in mods:
            if md in failed:
                out[key] = (md, failed[md])
                break
    return out


def routed_modules_in_child(case):
    """``routed_modules`` for one case, run as a GPU-less child that stands in for the
    target arch (``AITER_AOT_REPLAY``): the arch aiter's runtime dispatch sees follows
    ``GPU_ARCHS`` only there, and this process may be another arch's host."""
    import aiter

    from . import runner

    tree = os.path.dirname(os.path.dirname(os.path.abspath(aiter.__file__)))
    shim, pkg = None, "op_tests.moe_bench"
    if runner.needs_shim(tree):
        shim = runner.make_shim(os.environ.get("TMPDIR") or "/tmp")
        pkg = "moe_bench_shim"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            f"{pkg}.prebuild",
            "--routed-modules",
            json.dumps(case.row),
        ],
        env=_child_env(1, tree, shim, flydsl=True),
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in r.stdout.splitlines():
        if line.startswith("ROUTED "):
            got = json.loads(line[len("ROUTED ") :])
            record_keymods(case.id, got["by_key"], got["modules"])
            for e in got.get("errors", []):
                PLAN_ERRORS.setdefault(case.id, []).append(e)
            return got["modules"]
    raise RuntimeError(f"routed-modules child failed: {(r.stderr or r.stdout)[-400:]}")


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


def missing_record_path():
    return os.path.join(state_dir(), "prebuild_missing.json")


def read_missing_record():
    """{md_name: spec} modules a leased run asked for and did not find (``not prebuilt``);
    the next prebuild builds them."""
    try:
        with open(missing_record_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


CURRENT = {"candidate": ""}  # "<case>|<candidate id>" the worker is measuring


def record_missing(md_name, spec=None, candidate=None):
    """Note a module a leased run asked for: its build spec (so the next prebuild builds it
    whether or not the plan names it) and the candidates that reached it."""
    rec = read_missing_record()
    ent = rec.setdefault(md_name, spec or {"md_name": md_name, "kind": "unknown"})
    if spec and "blob_gen_cmd" in spec and "blob_gen_cmd" not in ent:
        ent.update(spec)
    cand = candidate or CURRENT["candidate"]
    if cand:
        seen = ent.setdefault("candidates", [])
        if cand not in seen and len(seen) < 200:
            seen.append(cand)
    tmp = missing_record_path() + f".{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, missing_record_path())


PLAN_ERRORS = {}  # case id -> [what failed while planning it]; run_prebuild logs them


_PLAN_CACHE = (
    {}
)  # (case id, row) -> {md_name: spec} resolved for the case in this process


def _case_key(c):
    return (c.id, json.dumps(c.row, sort_keys=True, default=str))


def plan(cases):
    """Deduplicated {md_name: spec} over every runnable fused case, with the case ids that
    need each module.

    Each case is resolved once per process (the resolution is a GPU-less child, about 3 s)
    and the children of the cases not yet seen run side by side. Every step is its own try:
    a failing step is recorded in ``PLAN_ERRORS`` (and logged by the prebuild) but never
    hides the modules of the others. The modules an earlier plan resolved for a case
    (``prebuild_keymods.json``) are part of the plan too, so a module the keymods file
    lists is a module the prebuild attempts."""
    from concurrent.futures import ThreadPoolExecutor

    from . import workload

    need = {}
    PLAN_ERRORS.clear()
    keymods = read_keymods()
    specs = keymods.get(f"{SPECS_KEY}|{scope()}", {})
    todo = []
    for c in cases:
        if c.kind != "fused" or c.token <= 0:
            continue
        try:
            todo.append((c, workload.resolve(c)))
        except Exception:  # noqa: BLE001, S112  (invalid rows are reported by run)
            continue
    fresh = [(c, S) for c, S in todo if _case_key(c) not in _PLAN_CACHE]
    children = {}
    if fresh:
        workers = max(1, min(len(fresh), 16, (os.cpu_count() or 8) // 8))

        def resolve_in_child(c):
            try:
                return routed_modules_in_child(c), None
            except Exception as e:  # noqa: BLE001
                return {}, e

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {c.id: pool.submit(resolve_in_child, c) for c, _ in fresh}
            children = {cid: f.result() for cid, f in futs.items()}
    for c, S in todo:
        key = _case_key(c)
        if key not in _PLAN_CACHE:
            mods = dict(modules_for_case(S))

            def step(name, fn, mods=mods, c=c):
                try:
                    mods.update(fn() or {})
                except Exception as e:  # noqa: BLE001
                    PLAN_ERRORS.setdefault(c.id, []).append(
                        f"{name}: {type(e).__name__}: {str(e)[:200]}"
                    )

            def routed(c=c):
                found, err = children[c.id]
                if err is not None:
                    raise err
                return found

            step("resolved routes", routed)
            _PLAN_CACHE[key] = mods
        mods = dict(_PLAN_CACHE[key])
        for key_mods in keymods.get(_kmkey(c.id), {}).values():
            for md in key_mods:
                if md in specs:
                    mods.setdefault(md, specs[md])
        for md, spec in mods.items():
            need.setdefault(md, dict(spec, cases=[]))["cases"].append(c.id)
    for md, spec in read_missing_record().items():
        need.setdefault(md, dict(spec, cases=[]))["cases"].append("lease-recorded")
    return need


def _enumerate_all(S, case):
    """Every candidate the bar sweep would try for the case (no GPU)."""
    from .candidates.fused_cfg import FusedCfgSource
    from .flydsl_prebuild import make_ctx

    arch, cu = gpu_identity()
    ctx = make_ctx(S, case, os.environ.get("TMPDIR") or "/tmp", arch, int(cu))
    return ctx, FusedCfgSource().enumerate(ctx)


def failed_path():
    return os.path.join(state_dir(), "prebuild_failed.json")


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


def attempted_path():
    return os.path.join(state_dir(), "prebuild_attempted.json")


def read_attempted():
    try:
        with open(attempted_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def record_attempted(results):
    """{md_name: "built" | "failed"} of the modules a prebuild tried: what separates a
    module that cannot be built (recorded in prebuild_failed.json) from one nobody tried.
    """
    rec = read_attempted()
    rec.update(results)
    tmp = attempted_path() + f".{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, attempted_path())


PLAN_FILE = "prebuild_plan.json"
STAMP_FIELDS = ("tree", "tree_sha", "arch", "harness")


def plan_path():
    return os.path.join(state_dir(), PLAN_FILE)


def _plan_stamp():
    """What a plan depends on besides the case: the tree under test, the target chip and
    the harness (its enumeration decides the candidate kinds)."""
    import aiter

    from . import runner

    tree = os.path.dirname(os.path.dirname(os.path.abspath(aiter.__file__)))
    sha, _dirty = runner.tree_info(tree)
    try:
        arch, cu = gpu_identity()
    except SystemExit:
        arch, cu = "", ""
    return {
        "tree": os.path.realpath(tree),
        "tree_sha": sha,
        "arch": f"{arch}/{cu}",
        "harness": runner.harness_sha(),
    }


def _plan_key(case_id, stamp):
    return "|".join([case_id, *(stamp[k] for k in STAMP_FIELDS)])


@contextlib.contextmanager
def _plan_lock():
    """Workers that share a state dir update the manifest one at a time."""
    import fcntl

    os.makedirs(state_dir(), exist_ok=True)
    with open(plan_path() + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read_plan():
    try:
        with open(plan_path()) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        rec = {}
    if rec.get("version") != 2:
        rec = {"version": 2, "entries": {}}
    return rec


def save_plan(cases, need):
    """Write the host's plan for ``cases`` so a leased run reads it instead of planning.

    Entries are keyed by (case id, tree, tree sha, chip, harness sha), each with the case
    row's signature and the modules, specs included (a spec names paths of its own tree),
    so workers planning the same case ids on other trees or chips share the file without
    overwriting each other. An older entry of the same case on the same tree (another sha
    or harness) is replaced."""
    from . import runner

    stamp = _plan_stamp()
    by_case = {}
    for md, spec in need.items():
        for cid in spec.get("cases", []):
            by_case.setdefault(cid, {})[md] = {
                k: v for k, v in spec.items() if k != "cases"
            }
    with _plan_lock():
        rec = _read_plan()
        entries = rec["entries"]
        for c in cases:
            if c.kind != "fused" or c.token <= 0:
                continue
            for k in [
                k
                for k, e in entries.items()
                if e["case"] == c.id
                and e["tree"] == stamp["tree"]
                and e["arch"] == stamp["arch"]
            ]:
                del entries[k]
            entries[_plan_key(c.id, stamp)] = dict(
                stamp,
                case=c.id,
                sig=runner.case_sig(c),
                modules=by_case.get(c.id, {}),
            )
        tmp = plan_path() + f".{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(tmp, plan_path())


def leased_plan(cases):
    """The plan of a leased run: read from the host's manifest, never computed (planning
    takes minutes and imports aiter). Raises SystemExit at once, naming the cases, when the
    manifest has no entry for a case under this tree, chip and harness, or one made for
    another case row."""
    from . import runner

    entries = _read_plan()["entries"]
    stamp = _plan_stamp()
    problems, need = [], {}
    for c in cases:
        if c.kind != "fused" or c.token <= 0:
            continue
        e = entries.get(_plan_key(c.id, stamp))
        if e is None:
            others = [x for x in entries.values() if x["case"] == c.id]
            if not others:
                problems.append((c.id, "no plan"))
            else:
                near = max(
                    others, key=lambda x: sum(x[k] == stamp[k] for k in STAMP_FIELDS)
                )
                why = [k for k in STAMP_FIELDS if near[k] != stamp[k]]
                problems.append((c.id, "planned for another " + ", ".join(why)))
            continue
        if e.get("sig") != runner.case_sig(c):
            problems.append((c.id, "planned for another case row"))
            continue
        for md, spec in e["modules"].items():
            need.setdefault(md, dict(spec, cases=[]))["cases"].append(c.id)
    if problems:
        shown = "; ".join(f"{cid} ({why})" for cid, why in problems[:12])
        raise SystemExit(
            f"{len(problems)} of {len(cases)} cases have no current prebuild plan in "
            f"{plan_path()} (tree {stamp['tree']}, {stamp['arch']}): {shown}"
            f"{' ...' if len(problems) > 12 else ''}\n"
            "A leased run never plans (minutes of CPU, aiter imports). Run `run --mode "
            "prebuild --aiter <tree>` on the host for these cases first, or pass "
            "--allow-build."
        )
    for md, spec in read_missing_record().items():
        need.setdefault(md, dict(spec, cases=[]))["cases"].append("lease-recorded")
    return need


def require_prebuilt(cases, need=None):
    """Raise SystemExit when a module that should exist is missing: a leased run must never
    compile. A module a prebuild tried and could not build is not missing (its candidates
    are ``build_failed``, see ``excluded_keys``); one no prebuild tried is, and the cases
    that need it are named."""
    bad = known_failed()
    need = need if need is not None else plan(cases)
    miss = [m for m in missing(need) if m not in bad]
    if miss:
        tried = read_attempted()
        lines = []
        for m in miss:
            why = (
                f"prebuilt then lost ({tried[m]})"
                if m in tried
                else "no prebuild tried it"
            )
            cs = [c for c in need[m].get("cases", []) if c != "lease-recorded"]
            lines.append(f"{m}  [{why}; cases {', '.join(cs[:6]) or 'lease-recorded'}]")
        raise SystemExit(
            f"{len(miss)} JIT modules are not built in {jit_dir()}; a leased run must "
            "not compile. Run `run --mode prebuild` (CPU, no lease) for the cases that "
            "need them, or pass --allow-build:\n  " + "\n  ".join(lines)
        )


class NotPrebuilt(RuntimeError):
    """A leased run reached a JIT module that was not built beforehand."""

    def __init__(self, md_name):
        super().__init__(
            f"build forbidden in a leased run ({NO_BUILD_ENV}=1): {md_name} is not "
            "prebuilt; run `run --mode prebuild` first"
        )
        self.md_name = md_name


def not_prebuilt_module(exc):
    """The module a NotPrebuilt names, also after a custom-op wrapper re-raised it."""
    import re

    if isinstance(exc, NotPrebuilt):
        return exc.md_name
    m = re.search(r"\): (\S+) is not prebuilt", str(exc))
    return m.group(1) if m and "build forbidden in a leased run" in str(exc) else None


def forbid_builds():
    """In a worker (or the parent of a leased run): turn any JIT build into an error. Active under ``MOE_BENCH_NO_BUILD=1``
    and whenever FlyDSL is in run-only mode (``FLYDSL_RUNTIME_RUN_ONLY=1``): a lease is
    run-only for every compiler, not just FlyDSL."""
    if (
        os.environ.get(NO_BUILD_ENV) != "1"
        and os.environ.get("FLYDSL_RUNTIME_RUN_ONLY") != "1"
    ):
        return
    from aiter.jit import core

    def refuse(
        md_name, srcs=None, flags_cc=None, flags_hip=None, blob_gen_cmd=None, *a, **k
    ):
        # the one place every JIT build goes through (compile_ops, ctypes modules): record
        # the module with what it takes to build it and the candidate that asked
        spec = {
            "md_name": md_name,
            "kind": (
                "ck2stages" if md_name.startswith("module_moe_ck2stages") else "static"
            ),
        }
        if spec["kind"] == "ck2stages":
            spec["blob_gen_cmd"] = blob_gen_cmd
        if md_name not in known_failed():
            record_missing(md_name, spec)
        raise NotPrebuilt(md_name)

    core.build_module = refuse


TRITON_COMPILE_ENV = "MOE_BENCH_TRITON_COMPILE"


def triton_cache_dir():
    """The persistent Triton cache every bar worker and the warm lease share:
    ``TRITON_CACHE_DIR`` when set, else ``triton-cache`` next to the prebuild state."""
    return os.environ.get("TRITON_CACHE_DIR") or os.path.join(
        state_dir(), "triton-cache"
    )


def use_triton_cache():
    d = triton_cache_dir()
    os.makedirs(d, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = d
    return d


def triton_run_only():
    """A leased (run-only) run that was not told to compile Triton kernels."""
    run_only = (
        os.environ.get(NO_BUILD_ENV) == "1"
        or os.environ.get("FLYDSL_RUNTIME_RUN_ONLY") == "1"
    )
    return run_only and os.environ.get(TRITON_COMPILE_ENV) != "1"


class TritonNotCached(HarnessError):
    """A leased run reached a Triton kernel that is not in the persistent cache."""

    def __init__(self, kernel):
        super().__init__(
            f"Triton kernel {kernel} is not in the cache {triton_cache_dir()} and a leased "
            f"run must not compile: run `python -m op_tests.moe_bench.triton_warm` first "
            f"(one short lease under timeout)"
        )
        self.kernel = kernel


_TRITON_GUARD = {"saved": None}


def forbid_triton_compiles():
    """Make a Triton cache miss raise ``TritonNotCached`` instead of compiling. Triton
    compiles through ``ASTSource.make_ir`` only after its cache lookup missed, so refusing
    there leaves every cached kernel untouched. Returns False when this Triton has no such
    hook (the caller must not run unguarded). Undo with ``restore_triton_compiles``."""
    if _TRITON_GUARD["saved"] is not None:
        return True
    try:
        from triton.compiler import compiler as tc
    except ImportError:
        return False
    classes = [tc.ASTSource]
    try:
        from triton.experimental.gluon._runtime import GluonASTSource

        classes.append(GluonASTSource)
    except ImportError:
        pass
    if not all(callable(getattr(c, "make_ir", None)) for c in classes):
        return False
    saved = [(c, c.__dict__.get("make_ir")) for c in classes]

    def refuse(self, *a, **k):
        raise TritonNotCached(getattr(self, "name", "?"))

    for c in classes:
        c.make_ir = refuse
    _TRITON_GUARD["saved"] = saved
    return True


def restore_triton_compiles():
    saved = _TRITON_GUARD["saved"]
    if saved is None:
        return
    for c, orig in saved:
        if orig is None:
            del c.make_ir
        else:
            c.make_ir = orig
    _TRITON_GUARD["saved"] = None


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


def gpu_identity():
    """(GPU_ARCHS, CU_NUM) of the target chip, from the environment or from the chip.

    With no GPU visible aiter cannot ask the chip (rocminfo shows nothing), so a build
    host with its GPU hidden must name both: ``GPU_ARCHS=gfx950 CU_NUM=256``.
    """
    arch, cu = os.environ.get("GPU_ARCHS"), os.environ.get("CU_NUM")
    if arch and cu:
        return arch, cu
    try:
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx

        return arch or get_gfx(), cu or str(get_cu_num())
    except Exception as e:
        raise SystemExit(
            "prebuild cannot read the GPU's identity with no GPU visible: set "
            "GPU_ARCHS (e.g. gfx950) and CU_NUM (e.g. 256 on MI355X, 304 on MI300X). "
            f"({type(e).__name__}: {str(e)[:100]})"
        ) from e


def use_tree(tree):
    """Make this process import aiter from ``tree`` (the --aiter tree), as the runner's
    workers do: put it first on sys.path before aiter is imported."""
    tree = os.path.abspath(tree)
    mod = sys.modules.get("aiter")
    if mod is not None:
        have = os.path.realpath(os.path.dirname(os.path.dirname(mod.__file__)))
        if have != os.path.realpath(tree):
            raise SystemExit(
                f"aiter was already imported from {have}, not the --aiter tree {tree}"
            )
        return tree
    sys.path.insert(0, tree)
    return tree


def flydsl_cache_dir():
    """Where FlyDSL artifacts go: the host build and the lease must use the same one."""
    return os.environ.get("FLYDSL_RUNTIME_CACHE_DIR") or os.path.join(
        jit_dir(), "flycache"
    )


def _child_env(max_jobs, tree=None, shim=None, flydsl=False):
    arch, cu = gpu_identity()
    env = dict(os.environ)
    if tree:
        paths = [tree] + ([shim] if shim else [])
        env["PYTHONPATH"] = (
            os.pathsep.join(paths) + os.pathsep + env.get("PYTHONPATH", "")
        )
    # Empty values hide the GPU on every ROCm stack; "-1" aborts the runtime on TheRock
    # wheels ("Sequence '-1' not recognized"). With ROCR_VISIBLE_DEVICES empty rocminfo
    # sees no chip, so the arch is named explicitly and the child never has to look.
    env.update(
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        CUDA_VISIBLE_DEVICES="",
        GPU_ARCHS=arch,
        CU_NUM=cu,
        MAX_JOBS=str(max_jobs),
    )
    env.pop(NO_BUILD_ENV, None)
    env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    # the build lands where this process (which imports aiter from the --aiter tree) would
    # put it, whatever the child's own import resolves to
    env["AITER_JIT_DIR"] = jit_dir()
    if flydsl:
        env.update(
            COMPILE_ONLY="1",
            FLYDSL_GPU_ARCH=arch,
            AITER_AOT_REPLAY="1",
            FLYDSL_RUNTIME_CACHE_DIR=flydsl_cache_dir(),
        )
    return env


def run_prebuild(
    cases,
    jobs=None,
    min_free_gb=10.0,
    log=print,
    timeout=3 * 3600,
    cores=None,
    retry_failed=False,
    tree=None,
    tmpdir=None,
):
    """Build every missing module of ``cases``, ``jobs`` at a time, GPU hidden.

    ``cores`` (default: every core) is split between the builders: each gets
    ``MAX_JOBS = cores // jobs`` for its own compile parallelism. Waits
    while the disk has under ``min_free_gb`` free. Returns {"built", "present", "failed"}.
    """
    if tree:
        tree = use_tree(tree)
    need = plan(cases)
    for cid, errs in PLAN_ERRORS.items():
        for e in errs:
            log(f"prebuild: PLAN ERROR {cid}: {e}")
    try:
        save_plan(cases, need)
    except Exception as e:  # noqa: BLE001
        log(
            f"prebuild: WARNING plan manifest not written ({type(e).__name__}: {e}); leases will refuse"
        )
    bad = {} if retry_failed else known_failed()
    todo = [need[m] for m in missing(need) if m not in bad]
    for m in missing(need):
        if m in bad:
            log(
                f"prebuild: skip {m}: failed before ({bad[m][:80]!r}); --prebuild-retry"
            )
    present = sorted(m for m in need if os.path.exists(module_path(m)))
    log(
        f"prebuild: {len(need)} modules needed, {len(present)} present, "
        f"{len(todo)} to build in {jit_dir()}"
    )
    cores = cores or os.cpu_count() or 8
    jobs = max(1, min(jobs or max(1, cores // 16), len(todo) or 1))
    per = max(4, cores // jobs)
    shim = pkg = None
    if tree:
        from . import runner

        if runner.needs_shim(tree):
            shim = runner.make_shim(tmpdir or os.environ.get("TMPDIR") or "/tmp")
            pkg = "moe_bench_shim"
    pkg = pkg or "op_tests.moe_bench"
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
                f"{pkg}.prebuild",
                "--build",
                json.dumps(spec),
            ],
            env=_child_env(per, tree, shim),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=tree or None,
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
    record_attempted({m: "built" for m in built} | {m: "failed" for m in genuine})
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


def manifest_path():
    return os.path.join(flydsl_cache_dir(), "prebuilt.json")


def read_manifest():
    try:
        with open(manifest_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# manifest entries written before a candidate's forced row was checked against the route
# aiter took (padded-M lookup, see flydsl_prebuild.dispatch_args_of) may cover nothing
FORCED_VERIFIED = 3


def flydsl_covered(entry):
    """A manifest entry counts only when its candidates' forced rows were verified and at
    least one built."""
    return bool(
        entry
        and entry.get("forced_verified") == FORCED_VERIFIED
        and (entry.get("built", 0) > 0 or entry.get("candidates", 0) == 0)
        # the run-only check of candidates the cover did not build found them all
        and entry.get("n_misses", 1) == 0
        and (entry.get("verify_tried", 0) > 0 or entry.get("verify_pool", 1) == 0)
    )


def flydsl_cases_missing(cases):
    """Case ids with FlyDSL bar candidates that no FlyDSL prebuild has covered."""
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    from . import workload
    from .flydsl_prebuild import enumerate_flydsl

    done = read_manifest()
    tmp = os.environ.get("TMPDIR") or "/tmp"
    gfx, cu = get_gfx(), get_cu_num()
    out = []
    for c in cases:
        if c.kind != "fused" or c.token <= 0 or flydsl_covered(done.get(c.id)):
            continue
        try:
            _ctx, cands = enumerate_flydsl(workload.resolve(c), c, tmp, gfx, cu)
        except Exception:  # noqa: BLE001, S112  (the bar reports enumeration errors)
            continue
        if cands:
            out.append(c.id)
    return out


def arm_flydsl_run_only():
    """In a leased run: the same cache dir as the host build, and a miss is an error."""
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = flydsl_cache_dir()
    os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"


def run_flydsl_prebuild(
    cases,
    procs=None,
    per_proc=40,
    log=print,
    tree=None,
    tmpdir=None,
    cases_csv=None,
    timeout=6 * 3600,
    arch=None,
    verify=80,
):
    """Host-compile the FlyDSL candidates of ``cases`` with no GPU, ``procs`` at a time.

    Each case's FlyDSL candidates are split in slices of about ``per_proc`` candidates, one
    child process per slice (compiles are mostly single-threaded). Results go into the
    cache's ``prebuilt.json`` manifest; returns {case_id: {...}}.
    """
    import tempfile

    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    from . import runner, schema, workload
    from .flydsl_prebuild import elements, enumerate_flydsl

    covered_by_case = {}
    tmpdir = tmpdir or os.environ.get("TMPDIR") or "/tmp"
    os.makedirs(flydsl_cache_dir(), exist_ok=True)
    gfx, cu = get_gfx(), get_cu_num()
    shim = pkg = None
    if tree:
        tree = use_tree(tree)
        if runner.needs_shim(tree):
            shim = runner.make_shim(tmpdir)
            pkg = "moe_bench_shim"
    pkg = pkg or "op_tests.moe_bench"
    cores = os.cpu_count() or 8
    procs = max(1, procs or cores // 4)
    plan_ = []
    for c in cases:
        if c.kind != "fused" or c.token <= 0:
            continue
        try:
            _ctx, cands = enumerate_flydsl(workload.resolve(c), c, tmpdir, gfx, cu)
        except Exception as e:  # noqa: BLE001
            log(f"flydsl prebuild: {c.id}: cannot enumerate ({type(e).__name__})")
            continue
        if cands:
            parts = max(1, -(-len(cands) // per_proc))
            plan_.append((c, len(cands), parts))
            covered_by_case[c.id] = sorted({e for x in cands for e in elements(x)})
    log(
        f"flydsl prebuild: {len(plan_)} cases with FlyDSL candidates, "
        f"{sum(n for _, n, _ in plan_)} candidates, {sum(p for _, _, p in plan_)} slices, "
        f"{procs} at a time, cache {flydsl_cache_dir()}"
    )
    if cases_csv is None:
        fd, cases_csv = tempfile.mkstemp(suffix=".csv", dir=tmpdir)
        os.close(fd)
        with open(cases_csv, "w", newline="") as f:
            import csv as _csv

            w = _csv.DictWriter(f, fieldnames=schema.COLS)
            w.writeheader()
            for c, _, _ in plan_:
                w.writerow({k: c.row.get(k, "") for k in schema.COLS})
    tasks = [(c, i, parts) for c, _, parts in plan_ for i in range(parts)]
    started, ended, rows = {}, {}, {}

    def child(c, part, extra, run_only=False):
        env = _child_env(1, tree, shim, flydsl=True)
        if run_only:
            env["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
        p = subprocess.run(
            [
                sys.executable,
                "-m",
                f"{pkg}.flydsl_prebuild",
                "--cases",
                cases_csv,
                "--id",
                c.id,
                "--part",
                part,
                "--tmpdir",
                tmpdir,
                *extra,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=tree or None,
        )
        for line in p.stdout.splitlines():
            if line.startswith("FLYDSL "):
                return json.loads(line[len("FLYDSL ") :]), ""
        return None, (p.stderr or p.stdout)[-300:]

    counts = {c.id: n for c, n, _ in plan_}

    def one(task):
        c, i, parts = task
        started.setdefault(c.id, time.time())
        res, _err = child(c, f"{i}/{parts}", [])
        if res is None:
            # a compiler crash (the tracing can segfault) loses the whole slice: redo its
            # candidates one by one so one bad kernel costs one candidate
            res = {"built": 0, "failed": [], "seconds": 0.0}
            for k in range(i, counts[c.id], parts):
                r1, e1 = child(c, f"{i}/{parts}", ["--indices", str(k)])
                if r1 is None:
                    res["failed"].append([f"candidate #{k}", f"child crashed: {e1}"])
                else:
                    res["built"] += r1["built"]
                    res["failed"] += r1["failed"]
                    res["seconds"] += r1["seconds"]
        ended[c.id] = time.time()
        return c.id, res, ""

    with ThreadPoolExecutor(max_workers=procs) as pool:
        for fut in as_completed([pool.submit(one, t) for t in tasks]):
            cid, res, err = fut.result()
            r = rows.setdefault(cid, {"built": 0, "failed": [], "child_seconds": 0.0})
            if res is None:
                r["failed"].append(["slice", err])
            else:
                r["built"] += res["built"]
                r["failed"] += res["failed"]
                r["child_seconds"] += res["seconds"]
    verified = {}
    if verify:

        def check(c):
            res, err = child(c, "0/1", ["--verify", str(verify)], run_only=True)
            return c.id, res, err

        with ThreadPoolExecutor(max_workers=procs) as pool:
            for cid, res, err in pool.map(check, [c for c, _, _ in plan_]):
                verified[cid] = res or {
                    "verify_tried": 0,
                    "verify_pool": 1,
                    "n_misses": 1,
                    "misses": [["verify", f"child failed: {err}"]],
                }
    manifest = read_manifest()
    for c, n, _ in plan_:
        r = rows.get(c.id, {"built": 0, "failed": [], "child_seconds": 0.0})
        v = verified.get(c.id, {})
        r.update(
            covered=covered_by_case.get(c.id, []),
            verify_tried=v.get("verify_tried", 0),
            verify_pool=v.get("verify_pool", 0),
            n_misses=v.get("n_misses", 0),
            misses=v.get("misses", []),
        )
        r.update(
            candidates=n,
            wall_seconds=round(ended.get(c.id, 0) - started.get(c.id, 0), 1),
            arch=gfx,
            forced_verified=FORCED_VERIFIED,
        )
        manifest[c.id] = r
        log(
            f"flydsl prebuild: {c.id}: {n} candidates, {r['built']} compiled, "
            f"{len(r['failed'])} could not, {r['child_seconds']:.0f} CPU-s, "
            f"{r['wall_seconds']:.0f}s wall; run-only check {r['verify_tried']}/"
            f"{r['verify_pool']} uncovered candidates, {r['n_misses']} missed"
        )
        for cid, err in r["misses"][:5]:
            log(f"flydsl prebuild:   MISS {cid}: {err[:160]}")
    with open(manifest_path(), "w") as f:
        json.dump(manifest, f, indent=1)
    return {c.id: manifest[c.id] for c, _, _ in plan_}


def prebuild_probe(log=print, tree=None, tmpdir=None):
    """Host-compile the preflight's FlyDSL vecadd probe so the lease finds it cached."""
    from . import runner

    shim = pkg = None
    if tree:
        tree = use_tree(tree)
        if runner.needs_shim(tree):
            shim = runner.make_shim(tmpdir or os.environ.get("TMPDIR") or "/tmp")
            pkg = "moe_bench_shim"
    pkg = pkg or "op_tests.moe_bench"
    os.makedirs(flydsl_cache_dir(), exist_ok=True)
    p = subprocess.run(
        [sys.executable, "-m", f"{pkg}.preflight"],
        env=_child_env(1, tree, shim, flydsl=True),
        capture_output=True,
        text=True,
        check=False,
        cwd=tree or None,
    )
    ok = False
    for line in p.stdout.splitlines():
        if line.startswith("PREFLIGHT "):
            res = json.loads(line[len("PREFLIGHT ") :])
            ok = any(
                c["name"] == "flydsl compile probe" and c["ok"] for c in res["checks"]
            )
    log(f"flydsl prebuild: vecadd probe {'host-compiled' if ok else 'FAILED'}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", help="JSON module spec: build it in this process")
    ap.add_argument("--routed-modules", help="JSON case row: print its routed modules")
    a = ap.parse_args(argv)
    if a.routed_modules:
        from . import schema, workload

        case = schema.Case(json.loads(a.routed_modules), {})
        S = workload.resolve(case)
        mods, by_key, errors = {}, {}, []

        def step(name, fn):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {type(e).__name__}: {str(e)[:200]}")
                return None

        enum = step("enumerate", lambda: _enumerate_all(S, case))
        ctx, cands = enum if enum else (None, [])
        mods.update(step("fallback", lambda: fallback_modules(S, case)) or {})
        mods.update(step("candidate names", lambda: candidate_modules(S, cands)) or {})
        got = step("resolved routes", lambda: routed_modules(S, case, cands, ctx))
        if got:
            mods.update(got[0])
            by_key = got[1]
        print(
            "ROUTED "
            + json.dumps({"modules": mods, "by_key": by_key, "errors": errors})
        )
        return 0
    if a.build:
        spec = json.loads(a.build)
        return 0 if build_one(spec) else 1
    ap.error("--build is required (use `run --mode prebuild` for a tier)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
