# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which reference does a kernel's output agree with? One short GPU call per case.

When a kernel and aiter's dispatch agree with each other and both disagree with the
harness reference, either the reference computes something else than the kernels or the
case is not what the harness thinks. This runs ``fused_moe`` once on the case's seed-0
inputs and scores that output against the harness reference and against one-at-a-time
variants of it: SwiGLU limit off, no ``+1`` on the linear branch, gate and up swapped,
bias off, bias read in the interleaved tile order, the other MX scale rule, bf16 activations,
an unquantized intermediate. The variant with the smallest error is what the kernel computes.

    python -m op_tests.moe_bench.diagnose --aiter . --cases tier1.csv --ids tier1-00166
"""

import argparse
import contextlib
import sys

import torch


def _tree_sha():
    import subprocess

    import aiter

    root = __import__("os").path.dirname(__import__("os").path.dirname(aiter.__file__))
    try:
        return subprocess.run(
            ["git", "-C", root, "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def interleave_position(S):
    """Interleaved row of each separated row of w1: tiles of 16 gate rows then 16 up."""
    i = S.I
    pos = torch.empty(2 * i, dtype=torch.long, device="cpu")
    r = torch.arange(i, device="cpu")
    pos[:i] = (r // 16) * 32 + (r % 16)
    pos[i:] = (r // 16) * 32 + 16 + (r % 16)
    return pos


@contextlib.contextmanager
def swiglu_variant(mode="default", plus1=True, swap=False, alpha=1.702):
    import aiter.fused_moe as fm

    real = fm.swiglu

    def variant(x_glu, x_linear, alpha_=alpha, limit=None):
        if swap:
            x_glu, x_linear = x_linear, x_glu
        lim = {"default": 7.0 if limit is None else limit, "none": None}[mode]
        if lim is not None:
            x_glu = x_glu.clamp(min=None, max=lim)
            x_linear = x_linear.clamp(min=-lim, max=lim)
        out_glu = x_glu * torch.sigmoid(alpha_ * x_glu)
        return out_glu * (x_linear + 1 if plus1 else x_linear)

    fm.swiglu = variant
    try:
        yield
    finally:
        fm.swiglu = real


def _score(out, ref):
    o, r = out.double(), ref.double()
    rel = float((o - r).norm() / r.norm())
    den = float((o * o + r * r).sum())
    return rel, 1.0 - 2.0 * float((o * r).sum()) / den


def diagnose(case, rows=16, seed=0, log=print):
    """Score the dispatch output against references, in two stages.

    1. *Numerics.* The rounding points of the path that ran are read from its route
       (``numerics_from_route``); where the route does not name them (in-kernel MX quants) or
       as a cross-check, the reference is built at each combination of scale rule and
       rounding before the quant (fp32_direct and bf16_then). The closest becomes the
       baseline. A SwiGLU semantics variant scored against the wrong rounding point can
       match by coincidence (an fp32_direct a4w4 kernel read "no +1" against a bf16_then
       baseline), so stage 2 never mixes them.
    2. *Semantics* at that baseline: SwiGLU limit off, no ``+1``, gate and up swapped, bias
       off or read in interleaved order, bf16 activations, an unquantized intermediate.
    """
    from . import accuracy, routes, workload
    from .bar_worker import assert_no_route, disable_routes

    # the dispatch side is aiter's own path: the router hook off, asserted per call
    has_hook = disable_routes()
    S = workload.resolve(case)
    M = min(case.token, rows)
    aw, rw, bias = workload.build_group(S)
    hs, tw, ti = accuracy.acc_inputs(S, M, seed)
    mask = workload.make_expert_mask(S) if S.is_ep else None
    kw = workload.fused_kwargs(S, aw, bias, mask)
    import aiter.fused_moe as fm

    with routes.RouteRecorder() as rec:
        out = fm.fused_moe(hs.clone(), aw["w1"], aw["w2"], tw, ti, **kw).clone()
    torch.cuda.synchronize()
    assert_no_route()
    route = routes.route_from_record(rec.last()) if rec.records else {}
    ridx = torch.arange(M, device="cuda")
    fam = accuracy.quant_name(S)
    mx = fam in ("mx_fp4", "mx_fp8")

    results = {}

    def score(name, fn, group):
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            log(f"  {name:<44} not computable: {type(e).__name__}: {str(e)[:80]}")
            return None
        results[name] = _score(out[:M], r) + (group,)
        log(
            f"  {name:<44} rel_l2 {results[name][0]:.4f}  "
            f"logits_diff {results[name][1]:.5f}"
        )
        return results[name]

    log(
        f"{case.id} {S.act.name} {fam} bias={S.bias} gate_mode={S.gate_mode} M={M} "
        f"route={route.get('kernel1', '')[:40]} routes_disabled=1"
        f"{' (aiter.moe hook present)' if has_hook else ''} aiter={_tree_sha()}"
    )
    exact = accuracy.exact_ref(S, rw, bias, hs, tw, ti, mask, ridx)
    results["ref_exact"] = _score(out[:M], exact) + ("exact",)
    log(
        f"  {'ref_exact':<44} rel_l2 {results['ref_exact'][0]:.4f}  "
        f"logits_diff {results['ref_exact'][1]:.5f}"
    )

    def ref(n, b=bias, **sw):
        with swiglu_variant(**sw):
            return accuracy.path_ref(S, rw, b, hs, tw, ti, mask, ridx, n)

    # stage 1: rounding points
    n0 = accuracy.numerics_from_route(S, route)
    bases = {}
    if accuracy.has_in_kernel(n0):
        bases = {
            f"numerics {k}": v for k, v in accuracy.in_kernel_variants(S, n0).items()
        }
    else:
        bases = {"numerics from the route": n0}
        for rounding in ("fp32_direct", "bf16_then"):
            iq = n0["inter_quant"]
            if iq == "none" or n0["inter"] == "bf16":
                continue
            v = dict(
                n0,
                inter="fp32_direct" if rounding == "fp32_direct" else f"bf16_then_{iq}",
            )
            bases[f"numerics {rounding} (route rule)"] = v
        if mx:
            for rule in ("round_up", "even"):
                v = dict(n0, act_quant=f"{fam}_{rule}", inter_quant=f"{fam}_{rule}")
                if n0["inter"].startswith("bf16_then_"):
                    v["inter"] = f"bf16_then_{fam}_{rule}"
                bases[f"numerics route rounding, {rule}"] = v
    for name, n in bases.items():
        score(name, lambda n=n: ref(n), "numerics")
    nums = {k: v for k, v in results.items() if v[2] == "numerics"}
    base_name = min(nums, key=lambda k: nums[k][0]) if nums else None
    base = bases[base_name] if base_name else n0
    if base_name:
        log(f"  -> baseline for the semantics variants: {base_name}")

    # stage 2: semantics at that baseline
    sem = {}
    if S.act.name == "Swiglu":
        sem["no swiglu limit"] = lambda: ref(base, mode="none")
        sem["no +1 on the linear branch"] = lambda: ref(base, plus1=False)
        sem["gate and up swapped"] = lambda: ref(base, swap=True)
    if bias is not None:
        sem["bias off"] = lambda: ref(base, b=None)
        pos = interleave_position(S).to("cuda")
        sem["bias1 read in interleaved order"] = lambda: ref(
            base, b=(bias[0][:, pos], bias[1])
        )
        sem["bias2 off"] = lambda: ref(base, b=(bias[0], None))
    if mx:
        sem["bf16 activations"] = lambda: ref(dict(base, act_quant="none"))
        sem["unquantized intermediate"] = lambda: ref(
            dict(base, inter="bf16", inter_quant="none")
        )
    for name, fn in sem.items():
        score(name, fn, "semantics")
    pick = {k: v for k, v in results.items() if k != "ref_exact"}
    if pick:
        best = min(pick, key=lambda k: pick[k][0])
        log(f"  -> closest: {best} (rel_l2 {pick[best][0]:.4f})")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--aiter", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--rows", type=int, default=16)
    a = ap.parse_args(argv)
    from . import schema

    cases = {c.id: c for c in schema.load_tier(a.cases)}
    for cid in a.ids:
        diagnose(cases[cid], rows=a.rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
