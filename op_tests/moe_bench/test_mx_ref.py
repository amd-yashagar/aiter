# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MX reference against aiter's own ops, bit for bit (values and e8m0 bytes)."""

import pytest

torch = pytest.importorskip("torch")

from op_tests.moe_bench import mx_ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _x(rows=257, k=512, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn((rows, k), device="cuda", generator=g)
    spread = torch.rand((rows, k), device="cuda", generator=g).exp() * 3
    return (base * spread).to(torch.bfloat16)


def _gfx():
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx()


def _eq(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("rule", mx_ref.RULES)
def test_fp4_matches_aiter_torch_reference(rule):
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.utility.mx_types import MxScaleRoundModeInt as M

    x = _x()
    q, s = per_1x32_f4_quant(
        x, round_mode={"round_up": M.RoundUp, "even": M.Even}[rule]
    )
    r = mx_ref.mx_quant_ref(x, "fp4", rule)
    assert _eq(q, r["q"]) and _eq(s, r["scale"])


def test_fp8_round_up_matches_the_hip_op():
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    x = _x()
    q, s = per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
    r = mx_ref.mx_quant_ref(x, "fp8", "round_up")
    assert _eq(q, r["q"]) and _eq(s, r["scale"])
    assert not _eq(s, mx_ref.mx_quant_ref(x, "fp8", "even")["scale"])


def test_fp4_round_up_matches_the_hip_op():
    if _gfx() != "gfx950":
        pytest.skip("the HIP fp4 quant kernel exists on gfx950 only")
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    x = _x()
    q, s = per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp4x2)
    r = mx_ref.mx_quant_ref(x, "fp4", "round_up")
    assert _eq(q, r["q"]) and _eq(s, r["scale"])


def test_fp4_even_matches_the_triton_op():
    from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant

    x = _x()
    q, s = dynamic_mxfp4_quant(x, scaling_mode="even")
    r = mx_ref.mx_quant_ref(x, "fp4", "even")
    assert _eq(q, r["q"]) and _eq(s[: x.shape[0], : x.shape[1] // 32], r["scale"])
    assert not _eq(q, mx_ref.mx_quant_ref(x, "fp4", "round_up")["q"])


@pytest.mark.parametrize("tokens", [64, 1024])
def test_fused_moe_quant_call_uses_the_rule_the_harness_derives(tokens):
    """The quant call fused_moe makes for a1 (fused HIP at small M, quant + sort split at
    large M): its values equal the reference at the rule ``mx_rule_of_route`` reads."""
    import aiter.fused_moe as fm
    from op_tests.moe_bench import accuracy

    fmt = "fp4" if _gfx() == "gfx950" else "fp8"
    fam = "mx_fp4" if fmt == "fp4" else "mx_fp8"
    fn = (
        fm.fused_dynamic_mxfp4_quant_moe_sort
        if fmt == "fp4"
        else fm.fused_dynamic_mxfp8_quant_moe_sort
    )
    experts, topk, d = 16, 4, 512
    x = _x(tokens, d, seed=3)
    ids = torch.stack(
        [torch.randperm(experts, device="cuda")[:topk] for _ in range(tokens)]
    ).to(torch.int32)
    w = torch.softmax(torch.randn(tokens, topk, device="cuda"), -1)
    sorted_ids, sw, _sei, nv, _buf = fm.moe_sorting(
        ids, w, experts, d, torch.bfloat16, 32
    )
    out, _scale = fn(
        x,
        sorted_ids=sorted_ids,
        num_valid_ids=nv,
        token_num=tokens,
        topk=topk,
        block_size=32,
        num_rows=None,
        sorted_weights=sw,
        num_experts_upper_bound=experts,
    )
    rule = accuracy.mx_rule_of_route(fam)
    assert _eq(out, mx_ref.mx_quant_ref(x, fmt, rule)["q"])
    other = "even" if rule == "round_up" else "round_up"
    assert not _eq(out, mx_ref.mx_quant_ref(x, fmt, other)["q"])


def test_mxfp8_path_ref_differs_by_rule_and_in_kernel_has_none():
    from op_tests.moe_bench import accuracy, schema, workload

    row = {c: "" for c in schema.COLS}
    row.update(
        {
            "case_id": "t-mxfp8",
            "kind": "fused",
            "family": "mxfp8",
            "token": "32",
            "model_dim": "512",
            "inter_dim": "256",
            "expert": "8",
            "topk": "2",
            "act_type": "Silu",
            "dtype": "bf16",
            "q_dtype_a": "fp8",
            "q_dtype_w": "fp8",
            "q_type": "per_1x32",
            "use_g1u1": "1",
            "doweight_stage1": "0",
            "modes": "warm",
        }
    )
    S = workload.resolve(schema.Case(row, {}))
    _aw, rw, bias = workload.build_group(S)
    hs, tw, ti = accuracy.acc_inputs(S, 32, 0)
    rows = torch.arange(16, device="cuda")
    out = {}
    for rule in ("round_up", "even"):
        n = accuracy.normalize_numerics(
            S,
            {
                "act_quant": f"mx_fp8_{rule}",
                "inter": "bf16_then_" + f"mx_fp8_{rule}",
                "combine": "bf16_atomic",
            },
        )
        out[rule] = accuracy.path_ref(S, rw, bias, hs, tw, ti, None, rows, n)
        assert out[rule].dtype == torch.float64 and torch.isfinite(out[rule]).all()
    assert not torch.equal(out["round_up"], out["even"])
    kern = {
        "act_quant": "mx_fp8_in_kernel",
        "inter": "fp32_direct",
        "inter_quant": "mx_fp8_in_kernel",
        "combine": "bf16_atomic (inferred)",
        "out_dtype": "bf16",
    }
    with pytest.raises(NotImplementedError, match="n/a"):
        accuracy.path_ref(S, rw, bias, hs, tw, ti, None, rows, kern)
    refs = accuracy.references(S, rw, bias, hs, tw, ti, None, rows, kern)
    assert "path" not in refs and "exact" in refs
