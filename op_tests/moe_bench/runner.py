# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``run``: shard, group, launch workers, collect rows, resume, drift check.

This process never touches a GPU. Workers (``worker.py``) run in subprocesses
pinned to one GPU each through the launcher (``claim``: claim file plus
HIP_VISIBLE_DEVICES; ``gpuq``: ``gpuq --num-gpus=1``; ``none``: inherit the
environment). A crash or hang marks the case ``crash`` or ``timeout`` and the run
continues; results are flushed after every row.
"""

import csv
import datetime
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import gpu_claim, schema
from .routes import ENV_KEYS, ROUTE_COLS
from .worker import UNSUPPORTED_FAMILIES_NON_GFX950

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_COLS = (
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
        "timing_mode",
        "capture_error",
        "median_us",
        "p10_us",
        "p90_us",
        "std_us",
        "iters",
        "tflops",
        "gbs",
        "ref_pass",
        "ref_warn",
        "err_ratio",
        "logits_diff",
        "nan",
        "inf",
        "sentinel_ok",
        "empty_rows_ok",
        "fp",
        "new_kernel_id",
        "first_call_s",
        "second_call_s",
        "weight_build_seconds",
        "ref_seconds",
    ]
    + ROUTE_COLS
    + [
        "node",
        "gpu",
        "gfx",
        "cu_num",
        "rocm",
        "torch",
        "flydsl",
        "aiter_sha",
        "aiter_dirty",
        "harness_sha",
        "env",
        "date",
    ]
)


def git(args, cwd):
    r = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def tree_info(tree):
    sha = git(["rev-parse", "HEAD"], tree)
    dirty = bool(git(["status", "--porcelain", "--untracked-files=no"], tree))
    return sha, dirty


def harness_sha():
    root = git(["rev-parse", "--show-toplevel"], HERE)
    sha = git(["log", "-1", "--format=%H", "--", "op_tests/moe_bench"], root)
    dirty = bool(git(["status", "--porcelain", "--", "op_tests/moe_bench"], root))
    return (sha[:12] if sha else "none") + ("+dirty" if dirty else "")


def make_shim(tmpdir):
    """A directory holding this harness under the package name ``moe_bench_shim``.

    Lets a worker import aiter from one tree and the harness from another: put the
    aiter tree first on PYTHONPATH and the shim directory after it.
    """
    shim = os.path.join(tmpdir, "shim")
    os.makedirs(shim, exist_ok=True)
    link = os.path.join(shim, "moe_bench_shim")
    # Re-point on every run: a shim left by another harness tree under the same TMPDIR
    # would silently import that tree's harness.
    if not os.path.islink(link) or os.path.realpath(link) != os.path.realpath(HERE):
        if os.path.lexists(link):
            os.unlink(link)
        os.symlink(HERE, link)
    return shim


def needs_shim(tree):
    """True unless the tree's own op_tests/moe_bench is exactly this harness."""
    own = os.path.join(tree, "op_tests", "moe_bench")
    return not os.path.isdir(own) or os.path.realpath(own) != os.path.realpath(HERE)


def worker_env(tree, tmpdir, extra=None, extra_path=None):
    env = dict(os.environ)
    paths = [tree] + ([extra_path] if extra_path else [])
    env["PYTHONPATH"] = os.pathsep.join(paths) + os.pathsep + env.get("PYTHONPATH", "")
    env["TMPDIR"] = tmpdir
    env.update(extra or {})
    return env


def probe_meta(tree, node, launcher, tmpdir):
    """GPU and software versions, from a short worker on a claimed GPU.

    Also asserts that the harness a worker imports is the one running this command.
    """
    shimmed = needs_shim(tree)
    pkg = "moe_bench_shim" if shimmed else "op_tests.moe_bench"
    code = (
        "import json,torch,importlib.metadata as m;"
        "import aiter;from aiter.jit.utils.chip_info import get_gfx,get_cu_num;"
        "p=torch.cuda.get_device_properties(0);"
        "print('META'+json.dumps({'gpu':p.name,'gfx':get_gfx(),'cu_num':get_cu_num(),"
        "'rocm':torch.version.hip,'torch':torch.__version__,"
        "'flydsl':m.version('flydsl')}));"
        "import importlib,os;"
        f"print('HARNESS'+os.path.dirname(importlib.import_module('{pkg}').__file__))"
    )
    cmd = [sys.executable, "-c", code]
    env = worker_env(tree, tmpdir, extra_path=make_shim(tmpdir) if shimmed else None)
    proc = _launch(cmd, launcher, env, capture=True, wait=600)
    info, imported = {}, ""
    for line in (proc or "").splitlines():
        if line.startswith("META"):
            info = json.loads(line[4:])
        elif line.startswith("HARNESS"):
            imported = os.path.realpath(line[7:])
    check_harness(HERE, imported)
    info.update(harness_expected=os.path.realpath(HERE), harness_imported=imported)
    sha, dirty = tree_info(tree)
    info.update(
        node=node,
        aiter_sha=sha[:12],
        aiter_dirty=int(dirty),
        harness_sha=harness_sha(),
        env=";".join(f"{k}={os.environ[k]}" for k in ENV_KEYS if k in os.environ),
        date=datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    )
    return info


def check_harness(expected, imported):
    """Refuse to run when workers would import another tree's harness."""
    if not imported or os.path.realpath(expected) != os.path.realpath(imported):
        raise RuntimeError(
            f"worker imports the harness from {imported or '(nowhere)'}, but this run uses "
            f"{os.path.realpath(expected)}; a stale shim or a harness copy inside the tree "
            "under test would measure the wrong code"
        )


def _launch(cmd, launcher, env, capture=False, wait=0):
    """Run one command pinned by the launcher; returns captured stdout if asked."""
    if launcher == "gpuq":
        env = dict(env, GPUQ_USER=env.get("GPUQ_USER", "yashagar"))
        full = ["gpuq", "--num-gpus=1", *cmd]
        r = subprocess.run(
            full, env=env, capture_output=capture, text=True, check=False
        )
        return r.stdout if capture else r.returncode
    gpus = []
    if launcher == "claim":
        gpus = gpu_claim.acquire(1, cmd, wait=wait)
        if not gpus:
            raise RuntimeError("no idle unclaimed GPU")
        env = dict(env, HIP_VISIBLE_DEVICES=str(gpus[0]), MOE_BENCH_GPU=str(gpus[0]))
    try:
        r = subprocess.run(cmd, env=env, capture_output=capture, text=True, check=False)
        return r.stdout if capture else r.returncode
    finally:
        for g in gpus:
            gpu_claim.release_claim(g)


def case_sig(c):
    return schema.case_signature(c)


def make_row(case, ev, meta):
    r = {k: case.row.get(k, "") for k in RESULT_COLS if k in case.row}
    r["sig"] = case_sig(case)
    for k in RESULT_COLS:
        if k in ev and k not in ("event",):
            v = ev[k]
            r[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    for k in ("node", "gpu", "gfx", "cu_num", "rocm", "torch", "flydsl"):
        r[k] = meta.get(k, "")
    for k in ("aiter_sha", "aiter_dirty", "harness_sha", "env", "date"):
        r[k] = meta.get(k, "")
    return r


class Results:
    """Append-only CSV with a flush per row; remembers what is finished."""

    def __init__(self, path, resume):
        self.path = path
        self.done = set()
        new = not os.path.exists(path) or not resume
        if resume and os.path.exists(path):
            for r in schema.read_csv(path):
                self.done.add((r["case_id"], r["mode"]))
        self.fh = open(path, "w" if new else "a", newline="")  # noqa: SIM115
        self.w = csv.DictWriter(self.fh, fieldnames=RESULT_COLS, extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.fh.flush()

    def add(self, row):
        self.w.writerow(row)
        self.fh.flush()
        self.done.add((row["case_id"], row["mode"]))

    def close(self):
        self.fh.close()


def select_cases(paths, case_filter=None):
    cases = []
    for p in paths:
        cases += schema.load_tier(p)
    if case_filter:
        pat = case_filter
        cases = [
            c for c in cases if pat in c.id or pat in c["family"] or pat in c["model"]
        ]
    return cases


def make_groups(cases, modes_req, done):
    groups = {}
    skipped = []
    for c in cases:
        if c.kind not in schema.RUNNABLE_KINDS:
            skipped.append(c)
            continue
        modes = [m for m in c.modes if m in modes_req and m != "first"]
        if not modes or all((c.id, m) in done for m in modes):
            continue
        key = schema.weight_group_key(c) if c.kind == "fused" else (c.kind,)
        groups.setdefault(key, []).append(c)
    return list(groups.items()), skipped


def shard_groups(groups, shard):
    if not shard:
        return groups
    i, n = shard
    ordered = sorted(groups, key=lambda g: schema.group_id(g[0]))
    return [g for k, g in enumerate(ordered) if k % n == i]


def parse_shard(s):
    if not s:
        return None
    i, n = s.split("/")
    return int(i), int(n)


class GroupJob:
    """One weight group on one worker process, with restarts after crash or hang."""

    worker_name = "worker"
    pkg = __package__
    extra_path = None

    def __init__(self, key, cases, modes, tree, tmpdir, launcher, results, meta, opts):
        self.key, self.cases, self.modes = key, list(cases), modes
        self.tree, self.tmpdir, self.launcher = tree, tmpdir, launcher
        self.results, self.meta, self.opts = results, meta, opts
        self.pending = {c.id: c for c in cases}
        self.proc = None
        self.gpu = None
        self.outfile = None
        self.pos = 0
        self.last_event = time.time()
        self.phase = ""
        self.inflight = None
        self.done = False

    def _spawn(self):
        todo = [c for c in self.cases if c.id in self.pending]
        spec = {"cases": [c.row for c in todo], "modes": self.modes}
        spec.update(self.extra_spec())
        fd, gpath = tempfile.mkstemp(suffix=".json", dir=self.tmpdir)
        with os.fdopen(fd, "w") as f:
            json.dump(spec, f)
        self.outfile = gpath + ".out"
        self.pos = 0
        self.inflight = None
        self.phase = "start"
        cmd = [
            sys.executable,
            "-m",
            f"{self.pkg}.{self.worker_name}",
            "--group",
            gpath,
            "--out",
            self.outfile,
        ]
        env = worker_env(self.tree, self.tmpdir, extra_path=self.extra_path)
        if self.launcher == "gpuq":
            env["GPUQ_USER"] = env.get("GPUQ_USER", "yashagar")
            cmd = ["gpuq", "--num-gpus=1", *cmd]
        elif self.launcher == "claim":
            got = gpu_claim.acquire(1, cmd)
            if not got:
                return False
            self.gpu = got[0]
            env.update(HIP_VISIBLE_DEVICES=str(self.gpu), MOE_BENCH_GPU=str(self.gpu))
        self.log = open(self.outfile + ".log", "w")  # noqa: SIM115
        self.proc = subprocess.Popen(
            cmd, env=env, cwd=self.tree, stdout=self.log, stderr=subprocess.STDOUT
        )
        self.last_event = time.time()
        return True

    def extra_spec(self):
        return {}

    def handle_row(self, c, ev):
        self.results.add(make_row(c, ev, self.meta))

    def handle_other(self, ev):
        pass

    def _release(self):
        if self.gpu is not None:
            gpu_claim.release_claim(self.gpu)
            self.gpu = None

    def _drain(self):
        if not self.outfile or not os.path.exists(self.outfile):
            return
        with open(self.outfile, "rb") as f:
            f.seek(self.pos)
            data = f.read()
        end = data.rfind(b"\n") + 1
        self.pos += end
        lines = data[:end].decode().split("\n")
        for line in lines:
            if not line.strip():
                continue
            ev = json.loads(line)
            self.last_event = time.time()
            kind = ev.get("event")
            if kind == "worker_start":
                check_harness(HERE, ev.get("harness", ""))
            elif kind == "case_start":
                self.inflight = ev["case_id"]
                self.phase = ev.get("phase", "")
            elif kind == "row":
                c = next((x for x in self.cases if x.id == ev["case_id"]), None)
                if c is None:
                    continue
                self.handle_row(c, ev)
                self._maybe_done(c)
            else:
                self.handle_other(ev)

    def _maybe_done(self, c):
        wanted = [m for m in c.modes if m in self.modes]
        if all((c.id, m) in self.results.done for m in wanted):
            self.pending.pop(c.id, None)

    def _fail(self, status, detail):
        """Mark the in-flight case (or the whole group in setup) and go on."""
        ids = (
            list(self.pending)
            if self.phase in ("probe", "weights", "start") or self.inflight is None
            else [self.inflight]
        )
        for cid in ids:
            c = self.pending.get(cid)
            if c is None:
                continue
            st, det = status, detail
            if (
                status == "crash"
                and c.kind == "sorting"
                and c["q_type"] == "per_1x32"
                and self.meta.get("gfx") != "gfx950"
            ):
                st = "unsupported"
                det = "MXFP4 sort+quant aborted the process on this arch: " + detail
            if (
                status == "crash"
                and "LLVM ERROR" in detail
                and self.meta.get("gfx") != "gfx950"
            ):
                st = "unsupported"
                det = "aiter aborted the process on this arch: " + detail
            if (
                status == "crash"
                and self.phase == "probe"
                and self.meta.get("gfx") != "gfx950"
                and c["family"] in UNSUPPORTED_FAMILIES_NON_GFX950
            ):
                st = "unsupported"
                det = "aiter aborted the process on this arch: " + detail
            for m in [m for m in c.modes if m in self.modes]:
                if (c.id, m) not in self.results.done:
                    self.results.add(
                        make_row(
                            c,
                            {"mode": m, "status": st, "detail": det},
                            self.meta,
                        )
                    )
            self.pending.pop(cid, None)

    def poll(self):
        """Advance the job; returns True while it still has work."""
        if self.done:
            return False
        if self.proc is None:
            if not self.pending:
                self.done = True
                return False
            if not self._spawn():
                return True
        self._drain()
        rc = self.proc.poll()
        limit = (
            self.opts["setup_timeout"]
            if self.phase in ("probe", "weights", "reference", "start")
            else self.opts["case_timeout"]
        )
        if rc is None and time.time() - self.last_event > limit:
            self.proc.kill()
            self.proc.wait()
            self._drain()
            self._fail("timeout", f"no progress for {limit}s in phase {self.phase}")
            self._finish_proc()
            return True
        if rc is not None:
            self._drain()
            if self.pending:
                tail = ""
                try:
                    text = Path(self.outfile + ".log").read_text()
                    errs = [
                        x for x in text.splitlines() if "ERROR" in x or "Error" in x
                    ]
                    tail = (errs[-1] if errs else text[-400:])[:400].replace(
                        "\n", " | "
                    )
                except OSError:
                    tail = "(no log)"
                self._fail("crash", f"worker exit {rc} in phase {self.phase}: {tail}")
            self._finish_proc()
        return True

    def _finish_proc(self):
        if getattr(self, "log", None) is not None:
            self.log.close()
        self._release()
        self.proc = None
        if not self.pending:
            self.done = True


def run_jobs(jobs, concurrency):
    active = []
    queue = list(jobs)
    while queue or active:
        while queue and len(active) < concurrency:
            active.append(queue.pop(0))
        still = []
        for j in active:
            if j.poll():
                still.append(j)
        if len(still) == len(active) and not any(j.proc for j in active) and queue:
            time.sleep(5)
        active = still
        time.sleep(1.0)


def run_first(cases, tree, tmpdir, launcher, results, meta, variants):
    for c in cases:
        if "first" not in c.modes or c.kind != "fused":
            continue
        for v in variants:
            mode = f"first-{v}"
            if (c.id, mode) in results.done:
                continue
            fd, gpath = tempfile.mkstemp(suffix=".json", dir=tmpdir)
            with os.fdopen(fd, "w") as f:
                json.dump({"cases": [c.row], "modes": ["first"], "first": v}, f)
            out = gpath + ".out"
            extra = {"FLYDSL_RUNTIME_ENABLE_CACHE": "0"} if v == "empty" else {}
            shimmed = needs_shim(tree)
            cmd = [
                sys.executable,
                "-m",
                "moe_bench_shim.worker" if shimmed else "op_tests.moe_bench.worker",
                "--group",
                gpath,
                "--out",
                out,
            ]
            env = worker_env(
                tree, tmpdir, extra, extra_path=make_shim(tmpdir) if shimmed else None
            )
            gpus = []
            if launcher == "gpuq":
                cmd = ["gpuq", "--num-gpus=1", *cmd]
            elif launcher == "claim":
                gpus = gpu_claim.acquire(1, cmd, wait=3600)
                env.update(HIP_VISIBLE_DEVICES=str(gpus[0]))
            try:
                subprocess.run(cmd, env=env, cwd=tree, timeout=3600, check=False)
            except subprocess.TimeoutExpired:
                pass
            finally:
                for g in gpus:
                    gpu_claim.release_claim(g)
            got = False
            if os.path.exists(out):
                for line in Path(out).read_text().splitlines():
                    ev = json.loads(line)
                    if ev.get("event") == "row":
                        results.add(make_row(c, ev, meta))
                        got = True
            if not got:
                results.add(make_row(c, {"mode": mode, "status": "crash"}, meta))


def anchors(rows, n=10):
    """Deterministic anchor cases: one ok warm fused case per family, by hash."""
    by_fam = {}
    for r in rows:
        if r["status"] != "ok" or r["mode"] != "warm" or r["kind"] != "fused":
            continue
        h = schema.short_hash(r["sig"])
        cur = by_fam.get(r["family"])
        if cur is None or h < cur[0]:
            by_fam[r["family"]] = (h, r["case_id"])
    return [cid for _, cid in sorted(by_fam.values())][:n]


def drift_check(args, base_rows, meta, tmpdir):
    """Re-time anchor cases on the baseline tree; flag any that moved over 2%."""
    ids = anchors(base_rows, args.drift_anchors)
    cases = [c for c in select_cases(args.cases_list) if c.id in ids]
    out = os.path.join(os.path.dirname(args.out) or ".", "drift-anchors.csv")
    res = Results(out, resume=False)
    groups, _ = make_groups(cases, ["warm"], set())
    opts = {"case_timeout": args.case_timeout, "setup_timeout": args.setup_timeout}
    jobs = [
        GroupJob(
            k, cs, ["warm"], args.baseline_tree, tmpdir, args.launcher, res, meta, opts
        )
        for k, cs in groups
    ]
    run_jobs(jobs, 1)
    res.close()
    base = {(r["case_id"], r["mode"]): r for r in base_rows}
    flagged, report = [], []
    for r in schema.read_csv(out):
        b = base.get((r["case_id"], r["mode"]))
        if not b or r["status"] != "ok":
            continue
        ratio = float(r["median_us"]) / float(b["median_us"])
        report.append({"case_id": r["case_id"], "ratio": ratio})
        if abs(ratio - 1.0) > 0.02:
            flagged.append(r["case_id"])
    drift = {"anchors": report, "flagged": flagged, "threshold": 0.02}
    with open(os.path.join(os.path.dirname(args.out) or ".", "drift.json"), "w") as f:
        json.dump(drift, f, indent=1)
    return drift


def cmd_run(args):
    node = args.node or socket.gethostname()
    tmpdir = args.tmpdir or os.path.join(tempfile.gettempdir(), "moe-bench-tmp")
    os.makedirs(tmpdir, exist_ok=True)
    tree = os.path.abspath(args.aiter)
    args.cases_list = args.cases.split(",")
    cases = select_cases(args.cases_list, args.filter)
    if getattr(args, "ids", None):
        cases = [c for c in cases if c.id in args.ids]
    modes_req = [m for m in args.modes.split(",") if m]
    res = Results(args.out, args.resume)
    meta = probe_meta(tree, node, args.launcher, tmpdir)
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    for c in cases:
        if c.kind in schema.RUNNABLE_KINDS and c.token <= 0:
            for m in [m for m in c.modes if m in modes_req]:
                if (c.id, m) not in res.done:
                    ev = {"mode": m, "status": "invalid", "detail": f"token={c.token}"}
                    res.add(make_row(c, ev, meta))
    cases = [c for c in cases if c.token > 0 or c.kind not in schema.RUNNABLE_KINDS]
    groups, _ = make_groups(cases, modes_req, res.done)
    groups = shard_groups(groups, parse_shard(args.shard))
    opts = {"case_timeout": args.case_timeout, "setup_timeout": args.setup_timeout}
    jobs = [
        GroupJob(k, cs, modes_req, tree, tmpdir, args.launcher, res, meta, opts)
        for k, cs in groups
    ]
    if needs_shim(tree):
        shim = make_shim(tmpdir)
        for j in jobs:
            j.pkg = "moe_bench_shim"
            j.extra_path = shim
    n_cases = sum(len(cs) for _, cs in groups)
    print(
        f"{n_cases} cases in {len(jobs)} weight groups, modes {modes_req}", flush=True
    )
    t0 = time.time()
    run_jobs(jobs, args.jobs)
    if "first" in modes_req:
        run_first(cases, tree, tmpdir, args.launcher, res, meta, ["warm", "empty"])
    res.close()
    if getattr(args, "inner", False):
        Path(args.out + ".done").touch()
    print(f"run finished in {time.time() - t0:.0f}s -> {args.out}", flush=True)
    if args.baseline_tree and args.baseline_results:
        base_rows = schema.read_csv(args.baseline_results)
        d = drift_check(args, base_rows, meta, tmpdir)
        print(
            f"drift: {len(d['flagged'])} of {len(d['anchors'])} anchors moved over 2%"
        )
    return 0
