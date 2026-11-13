# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One short warm lease that fills the persistent Triton cache for the gluon a16w4 candidate.

A leased (run-only) bar never compiles; the Triton kernels of the gluon side API
(routing, the two gluon GEMMs, the combine, the top-k timed alone) compile the first time a
shape runs and are then cached. This runs the candidate's exact call once per distinct shape
of the cases that apply (random fp4 weights of the case's E, D, I; the kernels depend on
shapes, strides and dtypes, not values), then runs every shape again with Triton compiles
forbidden to prove the cache covers the run-only call. Run it under a timeout on a gfx950
lease with ``TRITON_CACHE_DIR`` (or the default next to the prebuild state) shared with the
bar workers:

    timeout 1800 python -m op_tests.moe_bench.triton_warm --aiter . --cases tier2.csv

It exits 0 when every shape is warm and verified, 1 otherwise, and says which.
"""

import argparse
import json
import os
import sys
import time

COMPILE_ENV = "MOE_BENCH_TRITON_COMPILE"


def shape_key(S, M):
    return (S.E, S.D, S.I, M, bool(S.bias), float(S.swiglu_limit or 0))


def _resolve(c):
    from . import workload

    try:
        return workload.resolve(c)
    except (KeyError, ValueError, AttributeError):
        return None


def applicable(cases, gfx):
    """[(case, spec)] the gluon candidate enumerates on this arch."""
    from .candidates import gluon_a16w4 as g

    out = []
    for c in cases:
        if c.kind != "fused":
            continue
        S = _resolve(c)
        if S is None:
            continue
        if gfx == g.GFX and g.is_a16w4_swiglu(S) and g.unsupported(S) is None:
            out.append((c, S))
    return out


def dummy_group(S, torch):
    """Random fp4 weights and e8m0 scales shaped like a case's unshuffled reference."""
    dev = "cuda"
    E, D, I = S.E, S.D, S.I
    rw = {
        "w1": torch.randint(0, 256, (E, 2 * I, D // 2), dtype=torch.uint8, device=dev),
        "w2": torch.randint(0, 256, (E, D, I // 2), dtype=torch.uint8, device=dev),
        "w1_scale": torch.randint(118, 126, (E * 2 * I, D // 32), dtype=torch.uint8, device=dev),
        "w2_scale": torch.randint(118, 126, (E * D, I // 32), dtype=torch.uint8, device=dev),
    }  # fmt: skip
    bias = None
    if S.bias:
        bias = (
            torch.randn((E, 2 * I), dtype=S.dtype, device=dev),
            torch.randn((E, D), dtype=S.dtype, device=dev),
        )
    return rw, bias


class _Ctx:
    def __init__(self, S, M, rw, bias, gfx):
        from . import workload

        self.S, self.case = S, None
        self.aw = object()
        self.ref_weights, self.bias = rw, bias
        self.tw, self.ti = workload.make_routing(S, M)
        self.hidden_static = workload.make_inputs(S, M)
        self.group_cache = {}
        self.dispatch_meta = {"gfx": gfx}


def warm(cases, gfx, budget_s=None, log=print):
    """Warm and verify; returns (done, errors) as lists of shape keys / messages."""
    import torch

    from . import prebuild
    from .candidates import gluon_a16w4 as g

    todo = {}
    for c, S in applicable(cases, gfx):
        todo.setdefault((S.E, S.D, S.I, bool(S.bias), float(S.swiglu_limit or 0)), []).append(
            (S, c.token)
        )  # fmt: skip
    t0 = time.time()
    done, errors = [], []
    for key, items in todo.items():
        S = items[0][0]
        rw, bias = dummy_group(S, torch)
        shapes = sorted({m for _, m in items})
        for M in shapes:
            if budget_s and time.time() - t0 > budget_s:
                errors.append(f"budget {budget_s}s spent before {key} M={M}")
                continue
            t1 = time.time()
            try:
                ctx = _Ctx(S, M, rw, bias, gfx)
                cand = g.GluonA16W4()
                cand.prepare(ctx)
                cand.call(ctx)()
                cand.topk_call()()
                torch.cuda.synchronize()
                log(f"warm {key} M={M}: {time.time() - t1:.1f}s")
                done.append(list(shape_key(S, M)))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{key} M={M}: {type(e).__name__}: {str(e)[:200]}")
                log(f"FAILED {errors[-1]}")
        del rw, bias
        torch.cuda.empty_cache()
    if not prebuild.forbid_triton_compiles():
        errors.append("cannot verify: this triton has no compile hook")
        return done, errors
    try:
        for key, items in todo.items():
            S = items[0][0]
            rw, bias = dummy_group(S, torch)
            for M in sorted({m for _, m in items}):
                if list(shape_key(S, M)) not in done:
                    continue
                try:
                    ctx = _Ctx(S, M, rw, bias, gfx)
                    cand = g.GluonA16W4()
                    cand.prepare(ctx)
                    cand.call(ctx)()
                    cand.topk_call()()
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001
                    done.remove(list(shape_key(S, M)))
                    errors.append(
                        f"not cached after warming {key} M={M}: {str(e)[:200]}"
                    )
            del rw, bias
            torch.cuda.empty_cache()
    finally:
        prebuild.restore_triton_compiles()
    return done, errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aiter", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--filter")
    ap.add_argument(
        "--budget-s", type=float, help="stop starting shapes after this long"
    )
    a = ap.parse_args(argv)
    sys.path.insert(0, a.aiter)
    os.environ[COMPILE_ENV] = "1"
    from . import prebuild, runner
    from .bar_worker import _gpu_identity

    prebuild.use_tree(a.aiter)
    cache = prebuild.use_triton_cache()
    cases = runner.select_cases(a.cases.split(","), a.filter)
    gfx = _gpu_identity()["gfx"]
    done, errors = warm(cases, gfx, a.budget_s)
    rec = {
        "gfx": gfx,
        "cache": cache,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warm": done,
        "errors": errors,
    }
    path = os.path.join(prebuild.state_dir(), "triton_warm.json")
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(path + ".tmp", path)
    print(
        f"triton_warm: {len(done)} shapes warm and verified, {len(errors)} errors; "
        f"cache {cache}; record {path}",
        flush=True,
    )
    for e in errors:
        print("  ", e, flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
