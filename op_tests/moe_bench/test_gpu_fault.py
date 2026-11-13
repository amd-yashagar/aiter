# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Configs that cannot run on the arch are never tried (MOEBUG-061); a launch failure that
poisons the context is recorded with its error before the worker restarts."""

import pytest

from op_tests.moe_bench import bar_runner, schema
from op_tests.moe_bench.candidates.fused_cfg import (
    flydsl_stage1_unsupported as unsupported,
)

# configs that need 128 KiB of LDS: t32x{16,32,64}x256_kw4, t64x{16,32}x128_kw4, t64x32x256_kw2
BIG = [
    "flydsl_moe1_abf16_wbf16_bf16_t32x16x256_bnt0_kw4",
    "flydsl_moe1_abf16_wbf16_bf16_t32x32x256_bnt0_kw4",
    "flydsl_moe1_abf16_wbf16_bf16_t32x64x256_bnt0_kw4",
    "flydsl_moe1_abf16_wbf16_bf16_t64x16x128_bnt0_kw4",
    "flydsl_moe1_abf16_wbf16_bf16_t64x32x128_bnt0_kw4",
    "flydsl_moe1_abf16_wbf16_bf16_t64x32x256_bnt0_kw2",
]
SMALL = "flydsl_moe1_abf16_wbf16_bf16_t16x16x128_bnt0_kw4"


def test_stage_1_configs_over_the_arch_lds_budget_are_unsupported_with_the_reason():
    for name in BIG:
        assert (
            unsupported(name, 3072, "gfx942")
            == "LDS 128 KiB > budget 64 KiB (MOEBUG-061)"
        ), name
        assert unsupported(name, 3072, "gfx950") is None  # 160 KiB budget
    assert unsupported(SMALL, 3072, "gfx942") is None
    assert unsupported(BIG[0], 3072, None) is None  # arch unknown: not guessed
    # stage 2 and non-a16 kernels have no LDS rule here
    assert (
        unsupported("flydsl_moe2_abf16_wbf16_bf16_t32x128x128_atomic", 3072, "gfx942")
        is None
    )


def test_every_enumerated_a16w16_config_over_64_kib_is_dropped_on_gfx942():
    from aiter.ops.flydsl import moe_kernels as mk

    names = mk.get_flydsl_stage1_kernels_a16w16("bf16")
    dropped = {n for n in names if unsupported(n, 3072, "gfx942")}
    assert set(BIG) - dropped == {b for b in BIG if b not in names}
    for n in names:
        p = mk.get_flydsl_kernel_params(n)
        need = mk._gemm1_lds_bytes(p["tile_m"], p["tile_n"], p["tile_k"], p["k_wave"])
        over = need > 64 * 1024 or (need == 64 * 1024 and p["tile_m"] >= 128)
        assert (n in dropped) == over, n
        assert unsupported(n, 3072, "gfx950") is None


def _job():
    class Table:
        def __init__(self):
            self.rows = []

        def add(self, r):
            self.rows.append(r)

    row = {c: "" for c in schema.COLS}
    row.update(case_id="case1", kind="fused", token="8", modes="warm")
    case = schema.Case(row, {})
    job = bar_runner.BarGroupJob.__new__(bar_runner.BarGroupJob)
    job.cases, job.pending = [case], {"case1": case}
    job.bad, job.faults = {}, {}
    job.cand_table, job.meta = Table(), {}
    return job


def test_a_poisoning_fault_is_recorded_with_its_error_before_the_restart():
    job = _job()
    job.handle_other(
        {
            "event": "gpu_fault",
            "case_id": "case1",
            "cand_id": "k1",
            "detail": "hipErrorIllegalState",
        }
    )
    job.phase, job.inflight = "cand:k1", "case1"
    job._fail("crash", "worker exit 70")
    row = job.cand_table.rows[0]
    assert row["status"] == "crash"
    assert "worker exit 70" in row["detail"] and "hipErrorIllegalState" in row["detail"]
    assert job.bad["case1"] == {"k1"}  # the restarted worker skips it
    # a crash with no word from the worker keeps the plain detail
    job.phase = "cand:k2"
    job._fail("crash", "worker exit -11")
    assert job.cand_table.rows[1]["detail"] == "worker exit -11"


def test_exactly_at_the_budget_runs_below_tile_m_128_and_is_dropped_at_128_on_gfx942():
    """The code objects give 64 KiB for all of these; the launches split by tile_m."""
    ok = [
        "flydsl_moe1_abf16_wbf16_bf16_t64x128x256_bnt0",  # 64 KiB, tile_m 64: runs
        "flydsl_moe1_abf16_wbf16_bf16_t32x16x128_bnt0_kw4",  # 64 KiB, tile_m 32: runs
        "flydsl_moe1_abf16_wbf16_bf16_t16x16x256_bnt0_kw4",  # 64 KiB, tile_m 16: runs
    ]
    for name in ok:
        assert unsupported(name, 3072, "gfx942") is None, name
    for name in (
        "flydsl_moe1_abf16_wbf16_bf16_t128x128x128",
        "flydsl_moe1_abf16_wbf16_bf16_t128x128x128_bnt0",
        "flydsl_moe1_abf16_wbf16_bf16_t128x64x128",
        "flydsl_moe1_abf16_wbf16_bf16_t128x64x128_bnt0",
    ):
        why = unsupported(name, 3072, "gfx942")
        assert why and "LDS 64 KiB = budget 64 KiB" in why and "MOEBUG-061" in why, name
        assert unsupported(name, 3072, "gfx950") is None
    # over the budget stays over
    assert unsupported(BIG[0], 3072, "gfx942").startswith("LDS 128 KiB > budget 64 KiB")


def test_the_lds_model_is_the_code_objects_size_where_artifacts_exist():
    import glob
    import os

    from op_tests.moe_bench import lds_real

    roots = [
        d
        for d in (
            os.path.expanduser("~/.flydsl/cache"),
            os.path.join(os.environ.get("AITER_JIT_DIR", ""), "flycache"),
        )
        if glob.glob(os.path.join(d, "launch_gemm1*", "*.pkl"))
    ]
    if not roots:
        pytest.skip("no FlyDSL gemm1 artifacts on this node")
    rows = lds_real.scan(roots)
    assert rows and lds_real.mismatches(rows) == []
    assert (
        lds_real.model_bytes(128, 128, 128, 1, 7168) == 65536
    )  # t128x128x128, 2 stages
    assert lds_real.model_bytes(128, 128, 256, 1, 7168) == 131072


# --- the claim launcher -------------------------------------------------------------------


def _idle(monkeypatch, state, claims=None, flow=False, users=None):
    from op_tests.moe_bench import gpu_claim

    monkeypatch.setattr(gpu_claim, "gpu_state", lambda: state)
    monkeypatch.setattr(gpu_claim, "read_claims", lambda: claims or {})
    monkeypatch.setattr(gpu_claim, "attlas_flow_running", lambda: flow)
    monkeypatch.setattr(gpu_claim, "kfd_gpu_users", lambda: users or {})
    return gpu_claim.idle_gpus()[0]


def _gpu(procs=(), util=0.0, vram=0.0):
    return {"smi": 0, "util": util, "vram_gb": vram, "procs": list(procs)}


def test_a_gpu_with_a_process_the_harness_did_not_start_is_never_taken(monkeypatch):
    """GPU 0 had a foreign job and no claim file; amd-smi shows a process on every GPU, the
    kernel's per-GPU accounting shows where it really runs."""
    state = {i: _gpu(procs=[4242, 777]) for i in range(3)}  # amd-smi: everywhere
    assert _idle(monkeypatch, state, users={0: [(4242, 200 * 2**20, 3)]}) == [1, 2]
    assert _idle(monkeypatch, state, users={}) == [0, 1, 2]


def _fake_kfd(tmp_path, procs):
    """procs: {pid: {gpu_id: (vram bytes, queues)}}; nodes: cpu 0, gpus 111, 222."""
    for n, gid in ((0, 0), (1, 111), (2, 222)):
        d = tmp_path / "topology" / "nodes" / str(n)
        d.mkdir(parents=True)
        (d / "gpu_id").write_text(str(gid))
    for pid, per in procs.items():
        d = tmp_path / "proc" / str(pid)
        (d / "queues").mkdir(parents=True)
        k = 0
        for gid, (vram, queues) in per.items():
            (d / f"vram_{gid}").write_text(str(vram))
            for _ in range(queues):
                (d / "queues" / str(k)).mkdir()
                (d / "queues" / str(k) / "gpuid").write_text(str(gid))
                k += 1


def test_the_kernels_per_gpu_accounting_says_which_gpu_a_process_uses(tmp_path):
    from op_tests.moe_bench import gpu_claim

    mb = 2**20
    _fake_kfd(
        tmp_path,
        {
            10: {111: (900 * mb, 2), 222: (0, 0)},  # a job on the first GPU only
            11: {111: (1 * mb, 0), 222: (0, 3)},  # a queue on the second, no memory
            12: {111: (8 * mb, 0), 222: (8 * mb, 0)},  # contexts only: not in use
        },
    )
    users = gpu_claim.kfd_gpu_users(str(tmp_path))
    assert sorted(users) == [0, 1]
    assert [u[0] for u in users[0]] == [10] and [u[0] for u in users[1]] == [11]
    assert gpu_claim.kfd_gpu_users(str(tmp_path / "missing")) == {}  # no KFD: no data


def test_excluded_gpus_are_never_claimed(monkeypatch):
    monkeypatch.setenv("MOE_BENCH_EXCLUDE_GPUS", "0, 3")
    state = {i: _gpu() for i in range(5)}
    assert _idle(monkeypatch, state) == [1, 2, 4]
    monkeypatch.delenv("MOE_BENCH_EXCLUDE_GPUS")
    assert _idle(monkeypatch, state) == [0, 1, 2, 3, 4]


def test_the_exclude_flag_reaches_every_subcommand_that_launches(monkeypatch):
    from op_tests.moe_bench import __main__ as cli

    monkeypatch.delenv("MOE_BENCH_EXCLUDE_GPUS", raising=False)
    monkeypatch.setattr(cli, "main", cli.main)
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers()
    p = sub.add_parser("run")
    cli.add_launcher(p)
    a = ap.parse_args(["run", "--exclude-gpus", "0,3"])
    assert a.exclude_gpus == "0,3"


def test_a_respawned_worker_stays_on_the_gpu_the_job_started_on(monkeypatch, tmp_path):
    import subprocess

    from op_tests.moe_bench import gpu_claim, runner

    acquired, released = [], []
    monkeypatch.setattr(
        gpu_claim, "acquire", lambda n, cmd, **k: acquired.append(1) or [5]
    )
    monkeypatch.setattr(gpu_claim, "release_claim", lambda g: released.append(g))
    monkeypatch.setattr(runner, "worker_env", lambda *a, **k: {"PATH": "/usr/bin"})

    class Proc:
        returncode = None

        def poll(self):
            return self.returncode

    spawned = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda cmd, **kw: spawned.append(kw["env"]) or Proc()
    )
    row = {c: "" for c in schema.COLS}
    row.update(case_id="c1", kind="fused", token="8", modes="warm")
    case = schema.Case(row, {})
    job = runner.GroupJob.__new__(runner.GroupJob)
    job.cases, job.pending, job.modes = [case], {"c1": case}, ["warm"]
    job.tree, job.tmpdir, job.launcher = str(tmp_path), str(tmp_path), "claim"
    job.pkg, job.gpu, job.proc, job.done = "op_tests.moe_bench", None, None, False
    job.opts = {}
    assert job._spawn() and job._spawn()  # a respawn after the first worker died
    job._finish_proc()
    assert acquired == [1]  # one claim for the job
    assert [e["HIP_VISIBLE_DEVICES"] for e in spawned] == ["5", "5"]
    assert released == []  # kept between workers
    job.pending.clear()
    job._finish_proc()
    assert released == [5] and job.done


def _alive(pid):
    import os

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    with open(f"/proc/{pid}/stat") as f:
        return f.read().rsplit(")", 1)[1].split()[0] != "Z"


def test_a_worker_dies_with_its_parent_whatever_kills_the_parent():
    import os
    import signal
    import subprocess
    import sys
    import time

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = (
        "import os, signal, subprocess, sys\n"
        "from op_tests.moe_bench import gpu_claim\n"
        "p = subprocess.Popen(['sleep', '300'], preexec_fn=gpu_claim.die_with_parent())\n"
        "print('CHILD', p.pid, flush=True)\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, PYTHONPATH=here),
        cwd=here,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert r.returncode == -signal.SIGKILL
    pid = int(
        next(x for x in r.stdout.splitlines() if x.startswith("CHILD")).split()[1]
    )
    for _ in range(50):
        if not _alive(pid):
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)
        raise AssertionError("the worker outlived its parent")


def _claim(tmp_path, gpu, pid, worker="meta-harness"):
    import json

    (tmp_path / f"gpu{gpu}").write_text(json.dumps({"worker": worker, "pid": pid}))


def test_a_claim_names_its_live_owner_or_the_worker_must_not_run(monkeypatch, tmp_path):
    import os

    from op_tests.moe_bench import gpu_claim

    monkeypatch.setattr(gpu_claim, "CLAIM_DIR", str(tmp_path))
    me = os.getpid()
    assert not gpu_claim.claim_is_live(3, me)  # no claim file
    _claim(tmp_path, 3, me)
    assert gpu_claim.claim_is_live(3, me) and gpu_claim.claim_is_live(
        3, me, "meta-harness"
    )
    assert not gpu_claim.claim_is_live(3, me + 1)  # another owner
    assert not gpu_claim.claim_is_live(3, me, "someone-else")
    _claim(tmp_path, 3, 2**22 + 12345)  # a dead pid
    assert not gpu_claim.claim_is_live(3, 2**22 + 12345)


def test_a_worker_exits_when_its_gpu_has_no_live_claim_of_its_parent(tmp_path):
    import os
    import subprocess
    import sys
    import time

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = (
        "import time\n"
        "from op_tests.moe_bench import gpu_claim\n"
        "gpu_claim.watch_claim(period=0.2)\n"
        "time.sleep(8)\n"
        "print('SURVIVED')\n"
    )

    def run(env_extra):
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-c", script],
            env=dict(os.environ, PYTHONPATH=here, MOE_BENCH_CLAIM_DIR=str(tmp_path), **env_extra),
            cwd=here, capture_output=True, text=True, timeout=60, check=False,
        )  # fmt: skip
        return r, time.time() - t0

    # no GPU named (launcher none / gpuq): nothing to watch
    r, _ = run({})
    assert "SURVIVED" in r.stdout
    # the claim launcher named gpu 3 but no claim file exists: the worker stops at once
    r, took = run({"MOE_BENCH_GPU": "3"})
    assert r.returncode == 71 and took < 6, (r.returncode, r.stderr[-300:])
    assert "no live claim on gpu3" in r.stderr
