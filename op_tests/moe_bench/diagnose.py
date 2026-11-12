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


def interleave_position(S):
    """Interleaved row of each separated row of w1: tiles of 16 gate rows then 16 up."""
    i = S.I
    pos = torch.empty(2 * i, dtype=torch.long)
    r = torch.arange(i)
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
    from . import accuracy, workload

    S = workload.resolve(case)
    M = min(case.token, rows)
    aw, rw, bias = workload.build_group(S)
    hs, tw, ti = accuracy.acc_inputs(S, M, seed)
    mask = workload.make_expert_mask(S) if S.is_ep else None
    kw = workload.fused_kwargs(S, aw, bias, mask)
    import aiter.fused_moe as fm

    out = fm.fused_moe(hs.clone(), aw["w1"], aw["w2"], tw, ti, **kw).clone()
    torch.cuda.synchronize()
    ridx = torch.arange(M, device="cuda")
    fam = accuracy.quant_name(S)
    mx = fam in ("mx_fp4", "mx_fp8")

    def numerics(act_rule="round_up", inter_rule="round_up", act=True, inter=True):
        a = f"{fam}_{act_rule}" if (mx and act) else ("none" if mx else fam or "none")
        if not mx and not act:
            a = "none"
        iq = f"{fam}_{inter_rule}" if (mx and inter) else (fam or "none")
        n = {
            "act_quant": a,
            "inter": f"bf16_then_{iq}" if inter and iq != "none" else "bf16",
            "combine": "bf16_atomic",
        }
        return accuracy.normalize_numerics(S, n)

    def ref(b=bias, n=None, **sw):
        n = n or numerics()
        with swiglu_variant(**sw):
            return accuracy.path_ref(S, rw, b, hs, tw, ti, mask, ridx, n)

    results = {}

    def add(name, fn):
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            log(f"  {name:<34} not computable: {type(e).__name__}: {str(e)[:80]}")
            return
        results[name] = _score(out[:M], r)
        log(
            f"  {name:<34} rel_l2 {results[name][0]:.4f}  logits_diff {results[name][1]:.5f}"
        )

    log(f"{case.id} {S.act.name} {fam} bias={S.bias} gate_mode={S.gate_mode} M={M}")
    exact = accuracy.exact_ref(S, rw, bias, hs, tw, ti, mask, ridx)
    results["ref_exact"] = _score(out[:M], exact)
    log(
        f"  {'ref_exact':<34} rel_l2 {results['ref_exact'][0]:.4f}  logits_diff {results['ref_exact'][1]:.5f}"
    )
    add("baseline (limit 7, alpha 1.702, +1)", lambda: ref())
    if S.act.name == "Swiglu":
        add("no swiglu limit", lambda: ref(mode="none"))
        add("no +1 on the linear branch", lambda: ref(plus1=False))
        add("gate and up swapped", lambda: ref(swap=True))
    if bias is not None:
        add("bias off", lambda: ref(b=None))
        pos = interleave_position(S).to("cuda")
        b1 = bias[0][:, pos]
        add("bias1 read in interleaved order", lambda: ref(b=(b1, bias[1])))
        add("bias2 off", lambda: ref(b=(bias[0], None)))
    if mx:
        other = "even"
        add("MX activation rule even", lambda: ref(n=numerics(act_rule=other)))
        add("MX intermediate rule even", lambda: ref(n=numerics(inter_rule=other)))
        add("MX both rules even", lambda: ref(n=numerics(other, other)))
        add("bf16 activations", lambda: ref(n=numerics(act=False)))
        add("unquantized intermediate", lambda: ref(n=numerics(inter=False)))
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
