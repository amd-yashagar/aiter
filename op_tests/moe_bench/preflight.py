# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Preflight for bar and new runs: fail loudly when a candidate source cannot work.

A bar built while FlyDSL cannot compile (no ROCM_PATH with ld.lld and bitcode) or while
the tuner's generators cannot import (einops missing) silently shrinks to "dispatch only"
and looks valid. This module imports every candidate source, checks the ROCm install
FlyDSL compiles against, and compiles and runs one tiny FlyDSL kernel, on the GPU the run
will use. ``python -m <pkg>.preflight`` prints one JSON line (``PREFLIGHT {...}``); the
orchestrator (``ensure``) stops with a message naming the source and the fix.
"""

import json
import os
import sys

REQUIRED_VERSIONS = ("einops", "pandas", "numpy", "flydsl", "torch", "triton")


def _rocm_root():
    for cand in (os.environ.get("ROCM_PATH"), "/opt/rocm"):
        if cand and os.path.isdir(cand):
            return cand, os.environ.get("ROCM_PATH") is not None
    return os.environ.get("ROCM_PATH") or "", os.environ.get("ROCM_PATH") is not None


def _check(name, fn, fix, required=True):
    try:
        detail = fn()
        return {"name": name, "ok": True, "detail": detail or "", "required": required}
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {str(e).strip().splitlines()[0][:300] if str(e).strip() else ''}"
        return {
            "name": name,
            "ok": False,
            "detail": msg,
            "fix": fix,
            "required": required,
        }


def _rocm():
    root, from_env = _rocm_root()
    missing = [
        p
        for p in ("llvm/bin/ld.lld", "amdgcn/bitcode")
        if not os.path.exists(os.path.join(root, p))
    ]
    if missing:
        raise RuntimeError(f"{root or '(no ROCm root)'} lacks {', '.join(missing)}")
    return f"{root} (ROCM_PATH {'set' if from_env else 'unset, using /opt/rocm'})"


def _tuner_generators():
    from .candidates import fused_cfg

    t = fused_cfg.tuner()
    for g in ("gen_2stages_task", "gen_flydsl_2stages_task", "gen_1stage_asm_task"):
        getattr(t, g)
    return "gemm_moe_tune FmoeTuner generators import"


def _triton_moe():
    import aiter.ops.triton.moe.moe_op_gemm_a8w8  # noqa: F401

    return "aiter.ops.triton.moe imports"


def _flydsl_probe():
    """Compile and run FlyDSL's simplest kernel (examples/00-simple-vecadd-universal.py)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import torch

    @flyc.kernel
    def vadd_kernel(
        A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, block_dim: fx.Constexpr[int]
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
        tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
        tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))
        tA = fx.logical_divide(fx.slice(tA, (None, bid)), fx.make_layout(1, 1))
        tB = fx.logical_divide(fx.slice(tB, (None, bid)), fx.make_layout(1, 1))
        tC = fx.logical_divide(fx.slice(tC, (None, bid)), fx.make_layout(1, 1))
        copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        rA = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        rB = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        rC = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        fx.copy_atom_call(copy, fx.slice(tA, (None, tid)), rA)
        fx.copy_atom_call(copy, fx.slice(tB, (None, tid)), rB)
        vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
        fx.memref_store_vec(vC, rC)
        fx.copy_atom_call(copy, rC, fx.slice(tC, (None, tid)))

    @flyc.jit
    def vadd(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        n: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        block_dim = 64
        grid_x = (n + block_dim - 1) // block_dim
        vadd_kernel(A, B, C, block_dim).launch(
            grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream
        )

    n = 4096
    a = torch.randn(n, dtype=torch.float32, device="cuda")
    b = torch.randn(n, dtype=torch.float32, device="cuda")
    c = torch.zeros(n, dtype=torch.float32, device="cuda")
    vadd(a, b, c, n)
    torch.cuda.synchronize()
    if not torch.allclose(c, a + b):
        raise RuntimeError("FlyDSL vecadd compiled but returned a wrong result")
    return "FlyDSL vecadd compiled and ran"


def versions():
    import importlib.metadata as m

    out = {}
    for p in REQUIRED_VERSIONS:
        try:
            out[p] = m.version(p)
        except m.PackageNotFoundError:
            out[p] = None
    return out


def run_probe():
    checks = [
        _check(
            "rocm",
            _rocm,
            "set ROCM_PATH to a ROCm install that has llvm/bin/ld.lld and amdgcn/bitcode "
            "(for example ROCM_PATH=/opt/rocm-7.2.4)",
        ),
        _check(
            "einops",
            lambda: __import__("einops").__version__,
            "pip install einops into the GM-123 venv",
        ),
        _check(
            "tuner generators (fused_moe source)",
            _tuner_generators,
            "pip install einops into the GM-123 venv; the tree needs csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py",
        ),
        _check(
            "flydsl compile probe",
            _flydsl_probe,
            "set ROCM_PATH (see the rocm check) and pip install flydsl==0.5.0 into the GM-123 venv",
        ),
        _check(
            "triton moe api",
            _triton_moe,
            "pip install triton for this ROCm (the Triton MoE source is not wired yet, so this is only a warning)",
            required=False,
        ),
    ]
    return {
        "checks": checks,
        "rocm_path_env": os.environ.get("ROCM_PATH", ""),
        "versions": versions(),
        "ok": all(c["ok"] for c in checks if c["required"]),
    }


def ensure(
    tree, tmpdir, launcher="none", worker_env=None, pkg="op_tests.moe_bench", cwd=None
):
    """Run the probe on the run's GPU; raise SystemExit naming the source and the fix."""
    cmd = [sys.executable, "-m", f"{pkg}.preflight"]
    from . import runner

    stdout = runner._launch(cmd, launcher, worker_env, capture=True, wait=600) or ""
    result = None
    for line in stdout.splitlines():
        if line.startswith("PREFLIGHT "):
            result = json.loads(line[len("PREFLIGHT ") :])
    if result is None:
        raise SystemExit(
            "preflight did not run (no result line): the probe subprocess died, run "
            f"`{' '.join(cmd)}` by hand on a claimed GPU to see why"
        )
    bad = [c for c in result["checks"] if not c["ok"] and c["required"]]
    for c in result["checks"]:
        if not c["ok"] and not c["required"]:
            print(
                f"preflight warning: {c['name']}: {c['detail']} -- {c['fix']}",
                flush=True,
            )
    if bad:
        lines = [f"  - {c['name']}: {c['detail']}\n    fix: {c['fix']}" for c in bad]
        raise SystemExit(
            "preflight failed; a bar built without these candidate sources would look valid "
            "and not be:\n" + "\n".join(lines)
        )
    return result


if __name__ == "__main__":
    print("PREFLIGHT " + json.dumps(run_probe()), flush=True)
