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
import re
import sys
import time

FLYDSL_MARKERS = ("flydsl", "impl__flydsl")


def is_flydsl(cand):
    k = (getattr(cand, "kernels", "") or "").lower()
    return any(m in k for m in FLYDSL_MARKERS)


def dispatch_args_of(S, case=None):
    """The get_2stage_cfgs arguments of the case, as the harness's route recorder gets them
    from a real dispatch call.

    They are read from a dry resolution of the case when it can be made (meta tensors, no
    launch): ``fused_moe`` looks the tuned row up by the *padded* token count
    (``get_padded_M``: 5 -> 8, 333 -> 512), so a forced one-row config built from the case's
    own token count is never found for any M that is not a power of two, and the host build
    then compiled the router's default route instead of the forced pair. The static
    fallback pads the same way."""
    if case is not None:
        try:
            from . import routes

            got = []
            routes.dry_route(case, S, args_out=got)
            if got:
                return got[-1]
        except Exception:  # noqa: BLE001, S110  (the static form below)
            pass
    import aiter.fused_moe as fm

    return {
        "token": fm.get_padded_M(S.M),
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
        dispatch_args=dispatch_args_of(S, case),
        dispatch_meta={"gfx": gfx, "cu_num": cu},
        tmpdir=tmpdir,
        enum_errors=[],
    )


def enumerate_flydsl(S, case, tmpdir, gfx, cu, cover=True):
    """The FlyDSL candidates to compile for ``case`` (no GPU needed): the bar's FlyDSL
    singles plus a covering set of its stage pairs (``covering_pairs``), or every FlyDSL
    candidate with ``cover=False`` (what a lease may try)."""
    from .candidates.fused_cfg import FusedCfgSource

    ctx = make_ctx(S, case, tmpdir, gfx, cu)
    cands = [c for c in FusedCfgSource().enumerate(ctx) if is_flydsl(c)]
    return ctx, covering_pairs(cands) if cover else cands


def _family(name):
    return re.split(r"_t\d+x\d+x\d+", name or "")[0]


def stage2_class(name):
    """What of a stage-2 kernel changes the stage-1 launch beside stage 1's own name: the
    v2 layout epilogue (scatter writes the aux sort output; its input dtype sets stage 1's
    output dtype) and the sort block of the sbm variants."""
    from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel

    v2 = parse_flydsl_v2_gemm2_kernel(name)
    if v2:
        return f"v2:{v2['a_dtype']}:{v2['epilog']}:{v2['sort_block_m']}:{v2['spart']}"
    sbm = re.search(r"_sbm(\d+)", name or "")
    return f"{_family(name)}:{sbm.group(1) if sbm else ''}"


def stage1_class(name):
    """What of a stage-1 kernel changes the stage-2 launch beside stage 2's own name: the
    quantization fused into stage 1 decides whether stage 2 reads scales and which dtype.
    """
    tail = (name or "").split("_t")[-1]
    fuse = "fp4" if "_fp4" in tail else "fp8" if "_fp8" in tail else ""
    return f"{_family(name)}:{fuse}"


def elements(c):
    """The artifacts a candidate launches, as keys: each stage kernel with the class of
    its partner (a stage's artifact key follows its partner's class, not its tile), or
    the candidate itself for a single."""
    m = getattr(c, "meta", None) or {}
    if c.group.startswith("pair-") and "s1" in m:
        return (
            f"1|{c.group}|{m['s1']}|{stage2_class(m['s2'])}",
            f"2|{c.group}|{m['s2']}|{stage1_class(m['s1'])}",
        )
    return (f"single|{c.id}",)


def covering_pairs(cands):
    """Candidates to compile so every artifact any candidate launches exists.

    Pair candidates are the product of stage-1 and stage-2 kernels per block_m group (55k
    for one a4w4 case); a stage's artifact does not depend on its partner's tile but does on
    its class (``stage2_class``, ``stage1_class``): the same stage-1 kernel is a different
    artifact next to a layout stage 2 than next to an atomic one. So the cover has one pair
    for every (stage 1, partner class) and every (stage 2, partner class). Singles
    (1-stage, whole-graph) are all kept.
    """
    keep = [c for c in cands if not c.group.startswith("pair-")]
    seen, chosen = set(), {}
    for c in cands:
        if not c.group.startswith("pair-"):
            continue
        els = elements(c)
        if any(e not in seen for e in els):
            chosen[c.id] = c
            seen.update(els)
    return keep + list(chosen.values())


def uncovered(cands, covered):
    """Candidates of ``cands`` whose artifacts the manifest's ``covered`` set lacks."""
    return [c for c in cands if any(e not in covered for e in elements(c))]


def verify_sample(case, tmpdir, k, only=None):
    """Run ``k`` FlyDSL candidates the cover did not compile in FlyDSL's run-only mode
    (the environment sets FLYDSL_RUNTIME_RUN_ONLY=1): each finds its artifact or raises
    "no usable AOT cache". Returns {"verify": k_tried, "misses": [...]}."""
    import random

    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    from . import workload

    S = workload.resolve(case)
    _ctx, everything = enumerate_flydsl(
        S, case, tmpdir, get_gfx(), get_cu_num(), cover=False
    )
    built = {c.id for c in covering_pairs(everything)}
    rest = [
        i
        for i, c in enumerate(everything)
        if c.id not in built and (only is None or only(c))
    ]
    random.Random(case.id).shuffle(rest)
    pick = rest[:k]
    res = build_slice(case, 0, 1, tmpdir, indices=pick, cover=False) if pick else []
    # only a missing artifact is a miss: a candidate aiter declines (its forced row is not
    # honored) or whose kernel raises is not an incomplete cover
    misses = [(c, e) for c, ok, e in res if not ok and "no usable AOT cache" in e]
    other = [(c, e) for c, ok, e in res if not ok and "no usable AOT cache" not in e]
    return {
        "case": case.id,
        "verify_tried": len(res),
        "verify_pool": len(rest),
        "misses": [[c, e] for c, e in misses[:20]],
        "n_misses": len(misses),
        "n_other_errors": len(other),
    }


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


def build_slice(case, part, parts, tmpdir, indices=None, cover=True):
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

        ctx, cands = enumerate_flydsl(S, case, tmpdir, gfx, cu, cover=cover)
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

        def one(cid, prepare, cand=None):
            from . import routes

            try:
                prepare()
                with routes.RouteRecorder() as rec:
                    fm._fused_moe_impl(hs.clone(), aw["w1"], aw["w2"], tw, ti, **args)
                # a forced row the router did not take compiled something else (its
                # default route): that is a candidate nothing was built for
                if cand is not None and hasattr(cand, "verify") and rec.records:
                    why = cand.verify(routes.route_from_record(rec.last()))
                    if why:
                        out.append((cid, False, f"forced row not honored: {why}"[:200]))
                        return
                out.append((cid, True, ""))
            except Exception as e:  # noqa: BLE001
                from .bar_worker import aiter_error_detail

                why = aiter_error_detail(e)
                out.append(
                    (
                        cid,
                        False,
                        (
                            f"aiter_error: {why}"
                            if why
                            else f"{type(e).__name__}: {str(e)[:200]}"
                        ),
                    )
                )

        if part == 0 and indices is None:
            from .candidates.fused_cfg import unforce

            unforce()
            one("dispatch", lambda: None)
        for c in mine:
            one(c.id, lambda c=c: c.prepare(ctx), c)
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
    ap.add_argument(
        "--verify",
        type=int,
        default=0,
        help="run-only check of this many candidates the cover did not build "
        "(FLYDSL_RUNTIME_RUN_ONLY=1): a miss means the cover is incomplete",
    )
    a = ap.parse_args(argv)
    from . import runner

    case = next(c for c in runner.select_cases([a.cases]) if c.id == a.id)
    part, parts = (int(x) for x in a.part.split("/"))
    t0 = time.time()
    if a.verify:
        out = verify_sample(case, a.tmpdir, a.verify)
        print("FLYDSL " + json.dumps(out), flush=True)
        return 0
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
