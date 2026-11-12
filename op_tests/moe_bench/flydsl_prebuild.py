# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-compile (no GPU) every FlyDSL kernel a case's bar candidates reach.

FlyDSL kernels compile when a ``@flyc.jit`` launcher is first called, not at import, and the
artifact goes to ``FLYDSL_RUNTIME_CACHE_DIR``. Under aiter's AOT ``dry_run`` the same host
code that picks kernels and sizes buffers runs on FakeTensors with ``COMPILE_ONLY=1``, so
each launch compiles and persists its artifact without a device, and the keys are the ones the
runtime looks up (method: the smci355 MX worker's ``host_dry_run.py``, store note
``flydsl-host-build.md``). Every FlyDSL candidate of a case (the forced single-row configs the
bar enumerates, plus the dispatch itself) is replayed that way at the case's real M.

A leased run then sets ``FLYDSL_RUNTIME_RUN_ONLY=1`` with the same cache dir: a missing
artifact raises instead of compiling, and the candidate fails.

Children are started by ``prebuild.run_flydsl_prebuild``:
    python -m op_tests.moe_bench.flydsl_prebuild --cases tier.csv --id tier1-00102 --part 0/8
"""

import argparse
import json
import os
import sys
import time

FLYDSL_MARKERS = ("flydsl", "impl__flydsl")


def is_flydsl(cand):
    k = (getattr(cand, "kernels", "") or "").lower()
    return any(m in k for m in FLYDSL_MARKERS)


def dispatch_args_of(S):
    """The get_2stage_cfgs arguments of the case, as the harness's route recorder gets them."""
    return {
        "token": S.M,
        "model_dim": S.D,
        "inter_dim": S.I,
        "expert": S.E,
        "topk": S.topk,
        "dtype": S.dtype,
        "q_dtype_a": S.aq,
        "q_dtype_w": S.wq,
        "q_type": S.qtype,
        "use_g1u1": S.g1u1,
        "activation": S.act,
        "doweight_stage1": S.doweight,
    }


def make_ctx(S, case, tmpdir, gfx, cu):
    import types

    return types.SimpleNamespace(
        S=S,
        case=case,
        dispatch_args=dispatch_args_of(S),
        dispatch_meta={"gfx": gfx, "cu_num": cu},
        tmpdir=tmpdir,
        enum_errors=[],
    )


def enumerate_flydsl(S, case, tmpdir, gfx, cu):
    """The FlyDSL candidates to compile for ``case`` (no GPU needed): the bar's FlyDSL
    singles plus a covering set of its stage pairs (``covering_pairs``)."""
    from .candidates.fused_cfg import FusedCfgSource

    ctx = make_ctx(S, case, tmpdir, gfx, cu)
    cands = FusedCfgSource().enumerate(ctx)
    return ctx, covering_pairs([c for c in cands if is_flydsl(c)])


def covering_pairs(cands):
    """Candidates to compile so every stage kernel is compiled at least once.

    Pair candidates are the product of stage-1 and stage-2 kernels per block_m group (55k
    for one a4w4 case), but a stage's artifact does not depend on its partner: its key is
    that stage's own lowered code and config. So a smallest cover of the stage-1 and stage-2
    kernels (about max(n1, n2) pairs per group) fills the cache for every pair and every
    pair the bar's marginal screening will time. Singles (1-stage, whole-graph) are all kept.
    """
    pairs = [c for c in cands if c.group.startswith("pair-")]
    keep = [c for c in cands if not c.group.startswith("pair-")]
    by_group = {}
    for c in pairs:
        by_group.setdefault(c.group, []).append(c)
    for lst in by_group.values():
        by_s1 = {}
        for c in lst:
            by_s1.setdefault(c.meta["s1"], []).append(c)
        need1 = set(by_s1)
        need2 = {c.meta["s2"] for c in lst}
        while need1 or need2:
            best, gain = None, -1
            for c in lst:
                g = (c.meta["s1"] in need1) + (c.meta["s2"] in need2)
                if g > gain:
                    best, gain = c, g
                    if g == 2:
                        break
            keep.append(best)
            need1.discard(best.meta["s1"])
            need2.discard(best.meta["s2"])
    return keep


def _patch_generators():
    import torch

    real_gen = torch.Generator
    torch.Generator = lambda device=None: real_gen()
    for name in ("randn", "rand", "randint", "randperm", "normal"):
        real = getattr(torch, name)

        def wrap(*a, _real=real, **k):
            k.pop("generator", None)
            return _real(*a, **k)

        setattr(torch, name, wrap)


def build_slice(case, part, parts, tmpdir, indices=None):
    """Compile slice ``part`` of ``parts`` of the case's FlyDSL candidates (plus the
    dispatch route in slice 0), or exactly the candidates at ``indices``. Returns
    [(candidate id, ok, error)]."""
    import inspect

    from aiter.aot.flydsl.dry_run import dry_run
    from aiter.aot.flydsl.moe import _without_device_queries
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    from . import accuracy, workload

    _patch_generators()
    gfx, cu = get_gfx(), get_cu_num()
    S = workload.resolve(case)
    out = []
    with dry_run(gfx, cu), _without_device_queries():
        import aiter.fused_moe as fm

        ctx, cands = enumerate_flydsl(S, case, tmpdir, gfx, cu)
        if indices is not None:
            mine = [cands[i] for i in indices if i < len(cands)]
        else:
            mine = [c for i, c in enumerate(cands) if i % parts == part]
        import torch

        # values never matter for a build key; the quantizers' scalar reads (.item())
        # have no value on a FakeTensor and raise (mxfp8's scale rule does)
        real_item = torch.Tensor.item
        torch.Tensor.item = lambda self: 0.0
        try:
            aw, _rw, bias = workload.build_group(S)
            hs, tw, ti = accuracy.acc_inputs(S, case.token, 0)
        finally:
            torch.Tensor.item = real_item
        mask = workload.make_expert_mask(S) if S.is_ep else None
        kw = workload.fused_kwargs(S, aw, bias, mask)
        params = inspect.signature(fm._fused_moe_impl).parameters
        args = {
            k: (
                v.value
                if k in ("activation", "quant_type", "quant_type_a", "gate_mode")
                and hasattr(v, "value")
                else v
            )
            for k, v in kw.items()
            if k in params
        }

        def one(cid, prepare):
            try:
                prepare()
                fm._fused_moe_impl(hs.clone(), aw["w1"], aw["w2"], tw, ti, **args)
                out.append((cid, True, ""))
            except Exception as e:  # noqa: BLE001
                out.append((cid, False, f"{type(e).__name__}: {str(e)[:200]}"))

        if part == 0 and indices is None:
            from .candidates.fused_cfg import unforce

            unforce()
            one("dispatch", lambda: None)
        for c in mine:
            one(c.id, lambda c=c: c.prepare(ctx))
        from .candidates.fused_cfg import unforce

        unforce()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--indices", help="comma-separated candidate indexes (a retry)")
    ap.add_argument("--tmpdir", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args(argv)
    from . import runner

    case = next(c for c in runner.select_cases([a.cases]) if c.id == a.id)
    part, parts = (int(x) for x in a.part.split("/"))
    t0 = time.time()
    idx = [int(x) for x in a.indices.split(",")] if a.indices else None
    res = build_slice(case, part, parts, a.tmpdir, idx)
    bad = [r for r in res if not r[1]]
    print(
        "FLYDSL "
        + json.dumps(
            {
                "case": a.id,
                "part": a.part,
                "built": len(res) - len(bad),
                "failed": [[c, e] for c, _, e in bad],
                "seconds": round(time.time() - t0, 1),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
