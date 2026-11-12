# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""An in-kernel MX path scores as passing at the rounding point it really has."""

import types

import pytest

torch = pytest.importorskip("torch")

from op_tests.moe_bench import accuracy, schema, workload

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _spec(token, d=512, i=128, e=16, topk=4, act="Swiglu"):
    row = {c: "" for c in schema.COLS}
    row.update(
        case_id=f"t-m-{token}", kind="fused", family="a4w4", token=str(token),
        model_dim=str(d), inter_dim=str(i), expert=str(e), topk=str(topk), act_type=act,
        dtype="bf16", q_dtype_a="fp4x2", q_dtype_w="fp4x2", q_type="per_1x32",
        use_g1u1="1", doweight_stage1="0", modes="warm",
    )  # fmt: skip
    return workload.resolve(schema.Case(row, {}))


def _score(S, kernel_variant, token):
    """Score a fake kernel that computes ``kernel_variant`` of the reference."""
    _aw, rw, bias = workload.build_group(S)
    fam = "mx_fp4"
    route_numerics = {
        "act_quant": f"{fam}_round_up",
        "inter": "fp32_direct",
        "inter_quant": f"{fam}_in_kernel",
        "combine": "bf16_rows_fp32_sum",
        "out_dtype": "bf16",
    }
    variants = accuracy.in_kernel_variants(S, route_numerics)
    rows = accuracy.sample_rows("t-case", token)
    refs, exact = {}, {}
    for seed in accuracy.SEEDS[:1]:
        hs, tw, ti = accuracy.acc_inputs(S, token, seed)
        exact[seed] = accuracy.exact_ref(S, rw, bias, hs, tw, ti, None, rows)
        d = {"exact": exact[seed]}
        for name, n in variants.items():
            d[f"path_{name}"] = accuracy.path_ref(
                S, rw, bias, hs, tw, ti, None, rows, n
            )
        refs[seed] = d
    ref = refs[0][f"path_{kernel_variant}"]

    def call():
        out = torch.zeros((token, S.D), dtype=torch.bfloat16, device="cuda")
        out[rows] = ref.to(
            torch.bfloat16
        )  # the kernel's output, bf16 like the real one
        return out

    ctx = types.SimpleNamespace(
        hidden_static=torch.zeros((token, S.D), dtype=torch.bfloat16, device="cuda"),
        tw=None,
        ti=None,
    )
    acc = accuracy.evaluate(
        S,
        ctx,
        call,
        refs,
        rows,
        token,
        seeds=accuracy.SEEDS[:1],
        sentinel=False,
        repeats=1,
    )
    matched = accuracy.match_rule(acc, tuple(variants))
    return matched, accuracy.rule(acc), variants


@pytest.mark.parametrize("token", [8, 64])
@pytest.mark.parametrize("kernel", ["round_up", "round_up+bf16"])
def test_each_rounding_point_matches_itself_and_passes(token, kernel):
    """00164 (M=8, bf16 first) and 00165/00256 (M=64/4096, fp32 direct) are different
    flydsl_moe1 configs of one kernel family: whichever it is, it scores as passing."""
    S = _spec(token)
    matched, passed, variants = _score(S, kernel, token)
    assert matched == kernel
    assert passed is True
    other = "round_up+bf16" if kernel == "round_up" else "round_up"
    assert variants[other]["inter"] != variants[kernel]["inter"]


def test_a_kernel_on_a_different_scale_rule_is_named_not_failed():
    S = _spec(64)
    matched, passed, _ = _score(S, "even", 64)
    # it passes against the reference it matches, and the numerics record that rule
    # (rule_source=matched), so compare reports "different rule (MX scale)"
    assert matched == "even" and passed is True


def test_a_wrong_kernel_matches_nothing_and_fails():
    S = _spec(64)
    _aw, rw, bias = workload.build_group(S)
    rows = accuracy.sample_rows("t-case", 64)
    hs, tw, ti = accuracy.acc_inputs(S, 64, 0)
    n = {
        "act_quant": "mx_fp4_round_up",
        "inter": "fp32_direct",
        "inter_quant": "mx_fp4_in_kernel",
        "combine": "bf16_atomic (inferred)",
        "out_dtype": "bf16",
    }
    variants = accuracy.in_kernel_variants(S, n)
    d = {"exact": accuracy.exact_ref(S, rw, bias, hs, tw, ti, None, rows)}
    for name, v in variants.items():
        d[f"path_{name}"] = accuracy.path_ref(S, rw, bias, hs, tw, ti, None, rows, v)

    def call():
        g = torch.Generator(device="cuda").manual_seed(5)
        return torch.randn((64, S.D), device="cuda", generator=g).to(torch.bfloat16) * 5

    ctx = types.SimpleNamespace(
        hidden_static=torch.zeros((64, S.D), dtype=torch.bfloat16, device="cuda"),
        tw=None,
        ti=None,
    )
    acc = accuracy.evaluate(
        S,
        ctx,
        call,
        {0: d},
        rows,
        64,
        seeds=accuracy.SEEDS[:1],
        sentinel=False,
        repeats=1,
    )
    accuracy.match_rule(acc, tuple(variants))
    assert accuracy.rule(acc) is False
