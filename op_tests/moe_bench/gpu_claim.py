# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: PLW1509  (preexec_fn only calls prctl; every spawn is single-threaded)
"""Claim idle GPUs on a shared node, run commands on them, release the claims.

Claim files live in /tmp/gm123-gpu-claims/gpu<N> (override with MOE_BENCH_CLAIM_DIR),
one per GPU, N being the HIP index. They tell other agents on the node which GPU is
taken. A GPU counts as idle when amd-smi shows at most 5% utilization over three
samples, under 1 GB of VRAM in use (amd-smi lists one process on every GPU inside a
container, so processes are ignored), and no live claim
file exists for it. amd-smi indexes are mapped to HIP indexes by PCI bus id.

    python -m op_tests.moe_bench.gpu_claim --list
    python -m op_tests.moe_bench.gpu_claim [--gpu N] [--wait S] -- <cmd> ...

Shard launchers (``run_shards``):
  claim: one claim file per GPU, at most as many concurrent shards as idle GPUs.
  gpuq:  ``GPUQ_USER=yashagar gpuq --num-gpus=1 <cmd>`` per shard (smci355).
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time

CLAIM_DIR = os.environ.get("MOE_BENCH_CLAIM_DIR", "/tmp/gm123-gpu-claims")
MAX_VRAM_GB = 1.0
MAX_UTIL_PCT = 5
WORKER = os.environ.get("MOE_BENCH_WORKER", "meta-harness")
# Other GM-123 workers share the node: this harness never holds more than this many
# GPUs at once (a sharded tier-3 run raises it with MOE_BENCH_MAX_GPUS).
MAX_OWN_GPUS = int(os.environ.get("MOE_BENCH_MAX_GPUS", "2"))


def _smi(*args):
    out = subprocess.run(
        ["amd-smi", *args, "--json"], capture_output=True, text=True, check=False
    )
    try:
        data = json.loads(out.stdout) if out.stdout.strip() else []
    except ValueError:
        return []
    if isinstance(data, dict):
        data = data.get("gpu_data", [])
    return data


def _val(d, key, default=0.0):
    v = d.get(key, {})
    v = v.get("value", default) if isinstance(v, dict) else v
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bdf_key(domain, bus, dev=0):
    return (int(domain), int(bus), int(dev))


def smi_to_hip_index():
    """Map amd-smi GPU index -> HIP index through the PCI bus id."""
    smi = {}
    for g in _smi("static", "--bus"):
        bdf = (g.get("bus") or {}).get("bdf", "")
        if bdf:
            dom, bus, rest = bdf.split(":")
            smi[int(g["gpu"])] = _bdf_key(
                int(dom, 16), int(bus, 16), int(rest.split(".")[0], 16)
            )
    code = (
        "import torch,json;print(json.dumps([[p.pci_domain_id,p.pci_bus_id,"
        "p.pci_device_id] for p in (torch.cuda.get_device_properties(i) "
        "for i in range(torch.cuda.device_count()))]))"
    )
    env = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    hip = {}
    try:
        for i, (dom, bus, dev) in enumerate(
            json.loads(out.stdout.strip().splitlines()[-1])
        ):
            hip[_bdf_key(dom, bus, dev)] = i
            hip[_bdf_key(dom, bus, 0)] = i
    except (ValueError, IndexError):
        pass
    mapping = {}
    for s, key in smi.items():
        mapping[s] = hip.get(key, hip.get((key[0], key[1], 0), s))
    return mapping or None


def gpu_state():
    """Return {hip_index: {util, vram_gb, procs}} from three monitor samples."""
    util, vram = {}, {}
    for _ in range(3):
        for g in _smi("monitor"):
            i = int(g["gpu"])
            util[i] = max(util.get(i, 0.0), _val(g, "gfx"))
            vram[i] = _val(g, "vram_used")
        time.sleep(0.3)
    procs = {}
    for g in _smi("process"):
        i = int(g["gpu"])
        procs[i] = [
            p["process_info"]["pid"]
            for p in g.get("process_list", [])
            if isinstance(p, dict)
            and isinstance(p.get("process_info"), dict)
            and isinstance(p["process_info"].get("memory_usage"), dict)
            and _val(p["process_info"]["memory_usage"], "vram_mem") > 0
        ]
    mapping = smi_to_hip_index() or {i: i for i in util}
    return {
        mapping.get(i, i): {
            "smi": i,
            "util": util[i],
            "vram_gb": vram.get(i, 0.0),
            "procs": procs.get(i, []),
        }
        for i in sorted(util)
    }


def read_claims():
    claims = {}
    if not os.path.isdir(CLAIM_DIR):
        return claims
    for name in os.listdir(CLAIM_DIR):
        if not name.startswith("gpu") or not name[3:].isdigit():
            continue
        try:
            with open(os.path.join(CLAIM_DIR, name)) as f:
                c = json.load(f)
        except (OSError, ValueError):
            c = {"pid": 0}
        pid = int(c.get("pid") or 0)
        claims[int(name[3:])] = dict(
            c, alive=pid > 0 and os.path.exists(f"/proc/{pid}")
        )
    return claims


def attlas_flow_running():
    """A live attlas flow (``flow.py``) on this node: its GPUs are claimed for the flow
    even where the claim file records only a launcher pid."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "flow.py"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return False
    return any(p.strip().isdigit() and int(p) != os.getpid() for p in out.split())


FLOW_GPUS = (0, 3)  # the gfx942 prefill flow's GPUs: never started on while it runs


def die_with_parent():
    """``preexec_fn`` for every process this harness starts on a GPU: the kernel kills it
    when its parent dies (PR_SET_PDEATHSIG), so a worker never outlives the process whose
    claim file holds its GPU."""
    parent = os.getpid()

    def preexec():
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)  # PR_SET_PDEATHSIG
        if os.getppid() != parent:  # the parent died before the call
            os._exit(1)

    return preexec


def claim_is_live(gpu, parent_pid, worker=None):
    """Whether the claim file of ``gpu`` names ``parent_pid`` (alive) as its owner."""
    try:
        with open(claim_path(gpu)) as f:
            c = json.load(f)
    except (OSError, ValueError):
        return False
    if int(c.get("pid") or 0) != parent_pid or not os.path.exists(
        f"/proc/{parent_pid}"
    ):
        return False
    return worker is None or c.get("worker") == worker


def watch_claim(period=15.0, exit_code=71):
    """In a worker: when ``MOE_BENCH_GPU`` names a GPU (the claim launcher set it), exit if
    the claim file of that GPU stops naming this worker's parent: a worker never runs on a
    GPU without a live claim. Returns the thread (None when there is no claim to watch).
    """
    import threading

    gpu = os.environ.get("MOE_BENCH_GPU")
    if not gpu or not gpu.isdigit():
        return None
    parent = os.getppid()
    gpu = int(gpu)

    def loop():
        while True:
            time.sleep(period)
            if os.getppid() != parent or not claim_is_live(gpu, parent):
                print(
                    f"moe_bench worker: no live claim on gpu{gpu} (parent {parent}); exiting",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(exit_code)

    t = threading.Thread(target=loop, daemon=True, name="claim-watch")
    t.start()
    return t


def excluded_gpus():
    """HIP indexes never claimed (``MOE_BENCH_EXCLUDE_GPUS=0,3`` or ``--exclude-gpus``),
    read at call time so a CLI flag set after import applies."""
    out = set()
    for tok in os.environ.get("MOE_BENCH_EXCLUDE_GPUS", "").replace(" ", "").split(","):
        if tok.isdigit():
            out.add(int(tok))
    return out


KFD = "/sys/class/kfd/kfd"
MIN_FOREIGN_VRAM = 64 * 2**20  # bytes of a GPU's memory held by a process: in use


def kfd_gpu_users(kfd=KFD):
    """{HIP index: [(pid, vram bytes, queues)]} of the processes using each GPU, from the
    kernel's per-process, per-GPU accounting (``/sys/class/kfd/kfd/proc``): a process counts
    on a GPU when it holds memory there (over 64 MiB) or has a compute queue on it. This is
    what the node's processes really do, whatever their pid namespace and whatever claim
    files exist (amd-smi lists a process on every GPU with its total memory). The HIP index
    is the position among the GPU nodes of the KFD topology."""
    try:
        nodes = sorted(
            int(n) for n in os.listdir(f"{kfd}/topology/nodes") if n.isdigit()
        )
        gpu_ids = []
        for n in nodes:
            with open(f"{kfd}/topology/nodes/{n}/gpu_id") as f:
                gid = int(f.read().strip() or 0)
            if gid:
                gpu_ids.append(gid)
        procs = os.listdir(f"{kfd}/proc")
    except (OSError, ValueError):
        return {}
    out = {}
    for pid in procs:
        if not pid.isdigit():
            continue
        for idx, gid in enumerate(gpu_ids):
            try:
                with open(f"{kfd}/proc/{pid}/vram_{gid}") as f:
                    vram = int(f.read().strip() or 0)
            except (OSError, ValueError):
                vram = 0
            queues = 0
            try:
                qdir = f"{kfd}/proc/{pid}/queues"
                for q in os.listdir(qdir):
                    with open(f"{qdir}/{q}/gpuid") as f:
                        queues += f.read().strip() == str(gid)
            except OSError:
                pass
            if vram > MIN_FOREIGN_VRAM or queues:
                out.setdefault(idx, []).append((int(pid), vram, queues))
    return out


def idle_gpus():
    state = gpu_state()
    claims = read_claims()
    flow = attlas_flow_running()
    out = []
    skip = excluded_gpus()
    users = kfd_gpu_users()
    for i, s in sorted(state.items()):
        busy = (
            i in skip
            or s["util"] > MAX_UTIL_PCT
            or s["vram_gb"] > MAX_VRAM_GB
            # a process holds memory or a queue on it, claim file or not (this harness
            # only runs on the GPUs it claimed, so on an unclaimed GPU any is foreign)
            or bool(users.get(i))
            # another worker's claim holds the GPU even when its pid looks dead: the
            # recorded pid can be a launcher while the job lives on
            or (
                i in claims
                and (claims[i]["alive"] or claims[i].get("worker") != WORKER)
            )
            or (flow and i in FLOW_GPUS)
        )
        if not busy:
            out.append(i)
    return out, state, claims


def claim_path(gpu):
    return os.path.join(CLAIM_DIR, f"gpu{gpu}")


def write_claim(gpu, cmd, worker=WORKER):
    os.makedirs(CLAIM_DIR, exist_ok=True)
    path = claim_path(gpu)
    body = {
        "worker": worker,
        "pid": os.getpid(),
        "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(cmd),
    }
    if os.path.exists(path):
        # Another agent's claim is never removed. Ours or a dead one is replaced.
        try:
            with open(path) as f:
                old = json.load(f)
        except (OSError, ValueError):
            old = {}
        if old.get("pid") not in (None, os.getpid()) and os.path.exists(
            f"/proc/{old.get('pid')}"
        ):
            raise FileExistsError(path)
        if old.get("worker") != worker:
            raise FileExistsError(path)
        os.remove(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(body, f)
    return path


def release_claim(gpu, pid=None):
    path = claim_path(gpu)
    try:
        with open(path) as f:
            mine = json.load(f).get("pid") == (pid or os.getpid())
        if mine:
            os.remove(path)
    except (OSError, ValueError):
        pass


def own_live_claims(worker=WORKER):
    return sum(
        1 for c in read_claims().values() if c.get("worker") == worker and c["alive"]
    )


def acquire(n, cmd, wait=0, prefer=None, worker=WORKER):
    """Claim up to n idle GPUs (at least one); returns the claimed HIP indexes.

    Never takes a claimed GPU and never lets this worker hold more than
    MAX_OWN_GPUS claims.
    """
    deadline = time.time() + wait
    while True:
        room = MAX_OWN_GPUS - own_live_claims(worker)
        if room <= 0:
            if time.time() >= deadline:
                return []
            time.sleep(10)
            continue
        n = min(n, room)
        idle, _, _ = idle_gpus()
        if prefer is not None:
            idle = [g for g in idle if g in prefer]
        got = []
        for g in idle:
            if len(got) >= n:
                break
            try:
                write_claim(g, cmd, worker)
                got.append(g)
            except FileExistsError:
                continue
        if got:
            return got
        if time.time() >= deadline:
            return []
        time.sleep(10)


def _gpuq_cmd(cmd):
    return ["gpuq", "--num-gpus=1", *cmd]


def run_shards(cmds, launcher="claim", jobs=None, wait=0, env_extra=None):
    """Run shard commands, each on its own GPU. Returns the list of exit codes.

    claim: shards run concurrently on idle GPUs, never more than are idle at the
    time a shard starts; the rest wait for a GPU to come free.
    gpuq: each shard is submitted through gpuq with one GPU; ``jobs`` bounds the
    number submitted at once (default: all).
    """
    pending = list(enumerate(cmds))
    running = {}
    codes = [None] * len(cmds)
    limit = jobs or len(cmds)
    started = time.time()
    while pending or running:
        for idx, (proc, gpu) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                codes[idx] = rc
                if gpu is not None:
                    release_claim(gpu)
                del running[idx]
        while pending and len(running) < limit:
            idx, cmd = pending[0]
            env = dict(os.environ, **(env_extra or {}))
            if launcher == "gpuq":
                env.setdefault("GPUQ_USER", "yashagar")
                proc = subprocess.Popen(
                    _gpuq_cmd(cmd),
                    env=env,
                    preexec_fn=die_with_parent(),
                )
                gpu = None
            else:
                got = acquire(1, cmd, wait=0)
                if not got:
                    break
                gpu = got[0]
                env.update(HIP_VISIBLE_DEVICES=str(gpu), MOE_BENCH_GPU=str(gpu))
                proc = subprocess.Popen(cmd, env=env, preexec_fn=die_with_parent())
            pending.pop(0)
            running[idx] = (proc, gpu)
        if pending and not running and time.time() - started > wait:
            for idx, _ in pending:
                codes[idx] = 3
            break
        time.sleep(2)
    return codes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", type=int, default=None, help="claim this GPU if idle")
    ap.add_argument("--worker", default=WORKER)
    ap.add_argument(
        "--exclude-gpus", help="comma separated HIP indexes never claimed (e.g. 0,3)"
    )
    ap.add_argument("--list", action="store_true", help="print state and exit")
    ap.add_argument("--wait", type=int, default=0, help="seconds to wait for a GPU")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if a.exclude_gpus:
        os.environ["MOE_BENCH_EXCLUDE_GPUS"] = a.exclude_gpus

    if a.list:
        idle, state, claims = idle_gpus()
        for i, s in state.items():
            print(
                f"gpu{i} (smi {s['smi']}) util={s['util']:.0f}% vram={s['vram_gb']:.1f}GB "
                f"procs={s['procs']} claim={claims.get(i)}"
            )
        print("idle:", idle)
        return 0
    if not cmd:
        ap.error("no command")
    prefer = [a.gpu] if a.gpu is not None else None
    got = acquire(1, cmd, wait=a.wait, prefer=prefer, worker=a.worker)
    if not got:
        print("no idle unclaimed GPU", file=sys.stderr)
        return 3
    gpu = got[0]
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu), MOE_BENCH_GPU=str(gpu))
    proc = None

    def _stop(*_):
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        print(f"claimed gpu{gpu}: {shlex.join(cmd)}", file=sys.stderr)
        proc = subprocess.Popen(cmd, env=env, preexec_fn=die_with_parent())
        return proc.wait()
    finally:
        release_claim(gpu)


if __name__ == "__main__":
    sys.exit(main())
