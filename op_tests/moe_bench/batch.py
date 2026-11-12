# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gpuq launcher: one lease per batch, never per case.

``run ... --launcher gpuq --batches N`` splits the weight groups into N batches
(``--shard i/N``) and submits one ``GPUQ_USER=... gpuq --num-gpus=1`` job per batch. The
job runs the whole batch (``--launcher none --inner``) inside its lease. A lease that is
evicted or dies is resubmitted and the batch resumes (``--resume``) from the rows
already flushed. Each batch writes its own files; they are merged when all finish.
"""

import csv
import glob
import os
import shutil
import subprocess
import sys
import time

MAX_LEASES = 8


def _swap(argv, flag, value):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2
            continue
        if argv[i].startswith(flag + "="):
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return out + [flag, value]


def inner_argv(argv, args, i, n):
    a = _swap(argv, "--launcher", "none")
    if n > 1:
        a = _swap(a, "--shard", f"{i}/{n}")
    if args.mode == "dispatch":
        a = _swap(a, "--out", f"{args.out}.batch{i}")
    else:
        a = _swap(a, "--out-dir", os.path.join(args.out_dir, f".batch{i}"))
    base = args.tmpdir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "moe-bench-tmp"
    )
    a = _swap(a, "--tmpdir", os.path.join(base, f"batch{i}"))
    a = _swap(a, "--jobs", "1")
    a = [x for x in a if x not in ("--resume", "--inner")]
    return a + ["--resume", "--inner"]


def marker(args, i):
    if args.mode == "dispatch":
        return f"{args.out}.batch{i}.done"
    return os.path.join(args.out_dir, f".batch{i}", ".done")


def _merge_csv(parts, dest):
    rows, header = [], None
    for p in parts:
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            rd = csv.DictReader(f)
            header = header or rd.fieldnames
            rows += list(rd)
    if header is None:
        return
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def merge(args, n):
    if args.mode == "dispatch":
        _merge_csv([f"{args.out}.batch{i}" for i in range(n)], args.out)
        src = f"{args.out}.meta.json"
        if os.path.exists(src):
            shutil.copy(src, os.path.splitext(args.out)[0] + ".meta.json")
        return
    seen = set()
    for i in range(n):
        for p in glob.glob(os.path.join(args.out_dir, f".batch{i}", "*.csv")):
            seen.add(os.path.basename(p))
    for name in sorted(seen):
        _merge_csv(
            [os.path.join(args.out_dir, f".batch{i}", name) for i in range(n)],
            os.path.join(args.out_dir, name),
        )
    metas = sorted(glob.glob(os.path.join(args.out_dir, ".batch0", "*.meta.json")))
    for m in metas:
        shutil.copy(m, args.out_dir)


def run_batched(args, argv):
    n = max(1, args.batches)
    env = dict(os.environ, GPUQ_USER=os.environ.get("GPUQ_USER", "yashagar"))
    if args.mode != "dispatch":
        os.makedirs(args.out_dir, exist_ok=True)
    state = {
        i: {
            "proc": None,
            "leases": 0,
            "done": False,
            "cmd": inner_argv(argv, args, i, n),
        }
        for i in range(n)
    }
    active_limit = max(1, args.jobs)
    while not all(s["done"] for s in state.values()):
        running = [i for i, s in state.items() if s["proc"] is not None]
        for i in running:
            s = state[i]
            rc = s["proc"].poll()
            if rc is None:
                continue
            s["proc"] = None
            if rc == 0 and os.path.exists(marker(args, i)):
                s["done"] = True
                print(
                    f"batch {i}/{n} finished after {s['leases']} lease(s)", flush=True
                )
            elif s["leases"] >= MAX_LEASES:
                s["done"] = True
                print(f"batch {i}/{n} gave up after {MAX_LEASES} leases", flush=True)
            else:
                print(
                    f"batch {i}/{n} lease ended (rc={rc}); resubmitting with --resume",
                    flush=True,
                )
        running = [i for i, s in state.items() if s["proc"] is not None]
        for i, s in state.items():
            if len(running) >= active_limit:
                break
            if s["done"] or s["proc"] is not None:
                continue
            cmd = [
                "gpuq",
                "--num-gpus=1",
                sys.executable,
                "-m",
                "op_tests.moe_bench",
                *s["cmd"],
            ]
            s["leases"] += 1
            s["proc"] = subprocess.Popen(cmd, env=env)
            running.append(i)
        time.sleep(5)
    merge(args, n)
    ok = all(os.path.exists(marker(args, i)) for i in range(n))
    return 0 if ok else 1
