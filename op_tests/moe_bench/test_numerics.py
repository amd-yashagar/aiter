# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerics a kernel declares reach the reference builder, in core's vocabulary."""

import sys
import types
from typing import NamedTuple

import pytest

from op_tests.moe_bench import accuracy
from op_tests.moe_bench.bar_worker import RoutedFusedMoe


class LastRoute(NamedTuple):
    kernel: str
    family: str
    gpu: str
    reason: str
    numerics: dict | None


class OldLastRoute(NamedTuple):
    kernel: str
    family: str
    gpu: str
    reason: str


def _S(qtype="per_1x128", aq="fp8", dtype="bfloat16"):
    import torch

    import aiter
    from aiter import dtypes

    return types.SimpleNamespace(
        qtype=getattr(aiter.QuantType, qtype),
        aq=getattr(dtypes, aq),
        dtype=getattr(torch, dtype),
    )


def _decl(**kw):
    d = {"act_quant": "per_1x128_fp8", "inter": "fp32_direct", "combine": "fp32_reduce"}
    d.update(kw)
    return d


def test_core_vocabulary_is_accepted_and_recorded():
    S = _S()
    n = accuracy.normalize_numerics(S, _decl())
    assert accuracy.numerics_key(n) == (
        "act_quant=per_1x128_fp8;inter=fp32_direct;inter_quant=per_1x128_fp8;"
        "combine=fp32_reduce;out_dtype=bf16"
    )
    n = accuracy.normalize_numerics(
        S, _decl(inter="bf16_then_per_1x128_fp8", combine="bf16_atomic")
    )
    assert n["inter"] == "bf16_then_per_1x128_fp8"
    n = accuracy.normalize_numerics(
        _S("per_Token"), _decl(act_quant="none", inter="bf16")
    )
    assert n["act_quant"] == "none"


@pytest.mark.parametrize(
    "combine", ["bf16_atomic", "fp32_atomic", "bf16_rows_fp32_sum", "fp32_reduce"]
)
def test_every_core_combine_is_accepted_and_recorded(combine):
    S = _S()
    n = accuracy.normalize_numerics(S, _decl(combine=combine))
    assert n["combine"] == combine
    cand = types.SimpleNamespace(declared_numerics=_decl(combine=combine))
    assert f"combine={combine};" in accuracy.numerics_key(
        accuracy.numerics_of(S, cand, {})
    )
    # combine does not change the rounding points the references are built at
    other = accuracy.normalize_numerics(S, _decl(combine="bf16_atomic"))
    assert accuracy.same_rounding(
        accuracy.numerics_key(n), accuracy.numerics_key(other)
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"act_quant": "hip"},
        {"inter": "fp32"},
        {"inter": "bf16_then_mx_fp4"},
        {"act_quant": "per_token_fp8"},  # case is per_1x128
        {"combine": "atomic"},
        {"extra_key": 1},
    ],
)
def test_unknown_or_misfit_values_fail_loudly(bad):
    with pytest.raises(accuracy.NumericsError):
        accuracy.normalize_numerics(_S(), _decl(**bad))


def test_missing_key_fails_loudly():
    with pytest.raises(accuracy.NumericsError):
        accuracy.normalize_numerics(_S(), {"act_quant": "none"})


def test_numerics_of_uses_declared_then_same_as_then_route():
    S = _S()
    cand = types.SimpleNamespace(declared_numerics=_decl())
    assert accuracy.numerics_of(S, cand, {})["inter"] == "fp32_direct"
    same = types.SimpleNamespace(declared_numerics={"same_as": "aiter dispatched path"})
    ctx = types.SimpleNamespace(dispatch_route={"stages": "1-stage"})
    n = accuracy.numerics_of(S, same, {}, ctx)
    assert n["inter"] == "fp32_direct" and n["act_quant"] == "per_1x128_fp8"
    with pytest.raises(accuracy.NumericsError):
        accuracy.numerics_of(S, same, {}, types.SimpleNamespace(dispatch_route=None))
    fell_back = types.SimpleNamespace(declared_numerics=None)
    n = accuracy.numerics_of(S, fell_back, {"kernel1": "some_ck_stage1"})
    assert n["inter"] == "bf16_then_per_1x128_fp8"


def test_probe_route_reads_last_route_numerics(monkeypatch):
    import aiter

    decl = _decl()
    lr = LastRoute("k1", "f", "g", "r", decl)
    fake = types.ModuleType("aiter.moe")
    fake.last_route = lambda: lr
    monkeypatch.setitem(sys.modules, "aiter.moe", fake)
    monkeypatch.setattr(aiter, "moe", fake, raising=False)
    cand = RoutedFusedMoe()
    cand.probe_route("eager")
    assert cand.declared_numerics == decl

    # a 4-field last_route (older core): fall back to aiter.moe.numerics(id)
    lr = OldLastRoute("k1", "f", "g", "r")
    fake.numerics = lambda kid: {
        "act_quant": "none",
        "inter": "bf16",
        "combine": "bf16_atomic",
    }
    cand.probe_route("eager")
    assert cand.declared_numerics["inter"] == "bf16"

    # declined: aiter's own path served the call, the route-derived numerics apply
    lr = LastRoute("aiter", "", "", "declined", None)
    cand.probe_route("eager")
    assert cand.declared_numerics is None


def test_fp32_direct_reference_builds_and_differs_from_bf16_round():
    """fp64 stage-1 output went straight into aiter's quant op on the fp32_direct path
    ("Unsupported dtype: torch.float64"); the reference must hand it fp32."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    import aiter
    from op_tests.moe_bench import schema, workload

    row = {c: "" for c in schema.COLS}
    row.update(
        case_id="t-fp32direct",
        kind="fused",
        family="fp8-block",
        token="64",
        model_dim="512",
        inter_dim="256",
        expert="8",
        topk="2",
        act_type="Silu",
        dtype="bf16",
        q_dtype_a="fp8",
        q_dtype_w="fp8",
        q_type="per_1x128",
        use_g1u1="1",
        doweight_stage1="0",
        modes="warm",
    )
    S = workload.resolve(schema.Case(row, {}))
    _aw, rw, bias = workload.build_group(S)
    hs, tw, ti = accuracy.acc_inputs(S, 64, 0)
    rows = torch.arange(32, device="cuda")
    refs = {}
    for inter in ("fp32_direct", "bf16_then_per_1x128_fp8"):
        n = accuracy.normalize_numerics(
            S, {"act_quant": "per_1x128_fp8", "inter": inter, "combine": "bf16_atomic"}
        )
        refs[inter] = accuracy.path_ref(S, rw, bias, hs, tw, ti, None, rows, n)
    assert all(
        r.dtype == torch.float64 and torch.isfinite(r).all() for r in refs.values()
    )
    assert not torch.equal(*refs.values())
    import aiter.fused_moe as fm

    assert fm.dtypes.fp32 == torch.float32  # fp64_math restored
    assert torch.get_default_dtype() == torch.float32
    assert aiter is not None


def test_combine_rounding_rank_is_informational_in_compare():
    from op_tests.moe_bench import compare

    def key(combine, act="per_1x128_fp8", inter="fp32_direct"):
        return f"act_quant={act};inter={inter};combine={combine}"

    bar = key("fp32_atomic")
    assert compare.rounds_more(bar, key("fp32_reduce")) == ""
    assert compare.rounds_more(bar, key("bf16_rows_fp32_sum")) == "combine"
    assert compare.rounds_more(bar, key("bf16_atomic")) == "combine"
    assert (
        compare.rounds_more(key("bf16_rows_fp32_sum"), key("bf16_atomic")) == "combine"
    )
    assert compare.rounds_more(key("bf16_atomic"), key("fp32_reduce")) == ""
    # act_quant and inter are reported in the same column
    both = compare.rounds_more(
        key("fp32_reduce", act="none", inter="bf16"),
        key("bf16_atomic", inter="bf16_then_per_1x128_fp8"),
    )
    assert both == "act_quant,inter,combine"
    # "n/a" (a route that says nothing) is never judged
    assert compare.rounds_more(key("n/a"), key("bf16_atomic")) == ""
    # rounds_more does not touch the verdict: acc_ok still compares errors only
    assert compare._same_rounding(key("fp32_reduce"), key("bf16_atomic"))


def test_aiter_route_combine_from_stage2_name():
    r = {"stages": "2-stage", "backend2": "ck", "kernel2": "moe_ck2stages_x"}
    assert accuracy.combine_of_route(r) == "bf16_atomic (inferred)"
    f = {
        "stages": "2-stage",
        "backend2": "flydsl",
        "kernel2": "flydsl_moe2_afp8_t32x128_reduce",
    }
    assert accuracy.combine_of_route(f) == "bf16_rows_fp32_sum"
    f["kernel2"] = "flydsl_moe2_afp8_t32x128_atomic"
    assert accuracy.combine_of_route(f) == "bf16_atomic"
    assert accuracy.combine_of_route({"stages": "1-stage"}) == "bf16_atomic (inferred)"
    assert accuracy.combine_of_route({"stages": "whole-graph"}) == "n/a"
    n = accuracy.numerics_from_route(_S(), r)
    assert n["combine"] == "bf16_atomic (inferred)"
    accuracy.normalize_numerics(_S(), n, derived=True)


def test_no_reference_is_its_own_state_and_fails_a_routed_row(tmp_path):
    import csv
    import json

    from op_tests.moe_bench import bar_worker

    cand = types.SimpleNamespace(
        kernel_id="gfx942_fp8_block.prefill", id="new|fused_moe"
    )
    aiter_cand = types.SimpleNamespace(kernel_id="aiter", id="new|fused_moe")
    S = types.SimpleNamespace(kind="fused")
    case = types.SimpleNamespace(
        id="c1", row={"model": "m", "family": "f", "sig": "s"}, modes=["warm"], token=8
    )
    for c, want_fail in ((cand, "1"), (aiter_cand, "")):
        import io

        fh = io.StringIO()
        results = {"new|fused_moe": {"ok": False, "outcome": "noref", "cand": c}}
        try:
            bar_worker._emit_summary(
                fh,
                case,
                S,
                "warm",
                results,
                {"sku": "", "gpu": "", "gfx": ""},
                [],
                "new",
            )
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"summary needs more of the case than this fake has: {e}")
        row = json.loads(fh.getvalue().splitlines()[-1])
        assert row["status"] == "no-reference"
        assert row.get("gate_fail", "") == want_fail
    assert csv and tmp_path


def test_whole_graph_combine_follows_its_epilogue():
    pytest.importorskip("aiter.ops.flydsl.fused_moe_gfx942")
    from aiter.ops.flydsl.fused_moe_gfx942 import Config, get_tune_space

    seen = {}
    for cs in get_tune_space():
        c = Config.from_string(cs)
        route = {
            "stages": "whole-graph",
            "kernel1": f"impl__flydsl_gfx942__{cs}",
        }
        seen.setdefault(bool(c.use_prefill), accuracy.combine_of_route(route))
    assert seen.get(True) == "bf16_rows_fp32_sum"  # prefill ends with sorted_sum
    assert seen.get(False) == "bf16_atomic"  # batch1 / decode: pairwise bf16 atomics


def test_out_dtype_is_recorded_for_fp16_cases():
    n = accuracy.normalize_numerics(_S(dtype="float16"), _decl())
    assert n["out_dtype"] == "fp16"
    assert "out_dtype=fp16" in accuracy.numerics_key(n)


def test_inferred_combine_label_is_not_a_declaration():
    S = _S()
    with pytest.raises(accuracy.NumericsError):
        accuracy.normalize_numerics(S, _decl(combine="bf16_atomic (inferred)"))
    ok = accuracy.normalize_numerics(
        S, _decl(combine="bf16_atomic (inferred)"), derived=True
    )
    assert ok["combine"] == "bf16_atomic (inferred)"


def _mx(qtype="per_1x32", aq="fp4x2"):
    return _S(qtype, aq)


def test_mx_rules_are_in_the_vocabulary():
    S = _mx()
    for rule in ("round_up", "even"):
        n = accuracy.normalize_numerics(
            S,
            {
                "act_quant": f"mx_fp4_{rule}",
                "inter": "fp32_direct",
                "inter_quant": f"mx_fp4_{rule}",
                "combine": "bf16_atomic",
            },
        )
        assert (
            n["act_quant"] == f"mx_fp4_{rule}" and n["inter_quant"] == f"mx_fp4_{rule}"
        )
    # a rule that does not fit the format, an unknown rule, the harness-only value
    for bad in ("mx_fp8_even", "mx_fp4_floor", "mx_fp4_in_kernel"):
        with pytest.raises(accuracy.NumericsError):
            accuracy.normalize_numerics(
                S,
                {"act_quant": bad, "inter": "bf16", "combine": "bf16_atomic"},
            )
    # an MX intermediate quantized from fp32 with act_quant none needs its rule
    with pytest.raises(accuracy.NumericsError, match="inter_quant"):
        accuracy.normalize_numerics(
            S, {"act_quant": "none", "inter": "fp32_direct", "combine": "bf16_atomic"}
        )
    # inter_quant defaults to act_quant (core's registry sets it the same way)
    n = accuracy.normalize_numerics(
        S,
        {"act_quant": "mx_fp4_even", "inter": "fp32_direct", "combine": "bf16_atomic"},
    )
    assert n["inter_quant"] == "mx_fp4_even"


def test_bare_mx_is_accepted_once_with_a_warning_that_names_the_qualifier():
    S = _mx()
    with pytest.warns(UserWarning, match="mx_fp4_round_up or mx_fp4_even"):
        n = accuracy.normalize_numerics(
            S, {"act_quant": "mx_fp4", "inter": "bf16", "combine": "bf16_atomic"}
        )
    assert n["act_quant"] == "mx_fp4_round_up"


def test_mx_rule_of_route_is_read_from_the_quant_call(monkeypatch):
    import aiter.fused_moe as fm

    def fake(module):
        f = lambda *a, **k: None
        f.__module__ = module
        return f

    monkeypatch.setattr(
        fm, "fused_dynamic_mxfp4_quant_moe_sort", fake("aiter.ops.quant")
    )
    assert accuracy.mx_rule_of_route("mx_fp4") == "round_up"
    monkeypatch.setattr(
        fm,
        "fused_dynamic_mxfp4_quant_moe_sort",
        fake("aiter.ops.triton.quant.fused_mxfp4_quant"),
    )
    assert accuracy.mx_rule_of_route("mx_fp4") == "even"
    monkeypatch.setattr(
        fm, "fused_dynamic_mxfp8_quant_moe_sort", fake("somewhere.else")
    )
    with pytest.raises(accuracy.NumericsError):
        accuracy.mx_rule_of_route("mx_fp8")


def test_mx_route_numerics_follow_the_call_and_in_kernel_for_asm_one_stage(monkeypatch):
    import aiter.fused_moe as fm

    def fake(module):
        f = lambda *a, **k: None
        f.__module__ = module
        return f

    S = _mx()
    monkeypatch.setattr(
        fm, "fused_dynamic_mxfp4_quant_moe_sort", fake("aiter.ops.triton.x")
    )
    two = accuracy.numerics_from_route(
        S, {"stages": "2-stage", "backend1": "ck", "backend2": "ck", "kernel1": "k"}
    )
    assert two["act_quant"] == "mx_fp4_even" and two["inter_quant"] == "mx_fp4_even"
    assert two["inter"] == "bf16_then_mx_fp4_even"
    one = accuracy.numerics_from_route(
        S, {"stages": "1-stage", "backend1": "asm", "kernel1": "fmoe_x"}
    )
    assert one["act_quant"] == "mx_fp4_in_kernel" and one["inter"] == "fp32_direct"
    assert one["combine"] == "bf16_atomic (inferred)"
    # the harness-only value is accepted from a route, never from a declaration
    accuracy.normalize_numerics(S, one, derived=True)
    # a in-kernel path has no ref_path; the references say so
    with pytest.raises(NotImplementedError, match="n/a"):
        accuracy.path_ref(S, None, None, None, None, None, None, None, one)


def test_compare_reports_mx_rule_and_rounding_separately():
    from op_tests.moe_bench import compare

    def key(act, inter="bf16_then_mx_fp4_even", iq=None, combine="bf16_atomic"):
        iq = iq or act
        return (
            f"act_quant={act};inter={inter};inter_quant={iq};combine={combine};"
            "out_dtype=bf16"
        )

    bar = key("mx_fp4_even")
    new = key("mx_fp4_round_up", inter="bf16_then_mx_fp4_round_up")
    assert compare.rule_differs(bar, new) == "different rule (MX scale)"
    assert compare.rounds_more(bar, new) == ""  # not "rounds more"
    assert not compare._same_rounding(bar, new)
    assert compare.rule_differs(bar, bar) == ""
    # a bar against an in-kernel path is a different rule too
    inkernel = key("mx_fp4_in_kernel", inter="fp32_direct", iq="mx_fp4_in_kernel")
    assert compare.rule_differs(bar, inkernel) == "different rule (MX scale)"
    # bars from before inter_quant existed still compare on what they have
    old = "act_quant=mx_fp4_even;inter=bf16_then_mx_fp4_even;combine=bf16_atomic"
    assert compare._same_rounding(old, bar)
    # "(inferred)" does not change the combine rank
    assert (
        compare.rounds_more(
            key("none", inter="bf16", combine="fp32_reduce"),
            key("none", inter="bf16", combine="bf16_atomic (inferred)"),
        )
        == "combine"
    )


def _exact_only(logits, rel=0.05, nan=0, sentinel=True, seeds=3):
    per = [{"exact_logits_diff": logits, "exact_rel_l2": rel} for _ in range(seeds)]
    return {
        "per_seed": per,
        "nan_inf": nan,
        "sentinel_ok": sentinel,
        "mean": {"exact_rel_l2": rel, "exact_logits_diff": logits},
    }


def test_in_kernel_path_must_pass_the_strict_rule_against_ref_exact():
    # EMSORT-class (logits_diff about 6e-4) is eligible as a bar
    assert accuracy.rule(_exact_only(6e-4)) is True
    assert accuracy.rule(_exact_only(0.01)) is True
    # a loose kernel must not set the bar, even below the garbage guard
    assert accuracy.rule(_exact_only(0.02, rel=0.2)) is False
    # one bad seed fails it
    res = _exact_only(6e-4)
    res["per_seed"][1]["exact_logits_diff"] = 0.05
    assert accuracy.rule(res) is False
    # NaN/Inf and the sentinel still fail; the rel_l2 garbage guard stays
    assert accuracy.rule(_exact_only(6e-4, nan=1)) is False
    assert accuracy.rule(_exact_only(6e-4, sentinel=False)) is False
    assert accuracy.rule(_exact_only(6e-4, rel=0.9)) is False
    # no exact metrics at all: undecided, not a pass
    assert (
        accuracy.rule({"per_seed": [{}], "nan_inf": 0, "sentinel_ok": True, "mean": {}})
        is None
    )


def _whole_graph_routes():
    pytest.importorskip("aiter.ops.flydsl.fused_moe_gfx942")
    from aiter.ops.flydsl.fused_moe_gfx942 import Config, get_tune_space

    out = {True: None, False: None}
    for cs in get_tune_space():
        c = Config.from_string(cs)
        if out[bool(c.use_prefill)] is None:
            name = f"impl__flydsl_gfx942__{cs}"
            out[bool(c.use_prefill)] = {
                "stages": "whole-graph",
                "kernel1": name,
                "full_impl": name,
                "backend1": "flydsl-graph",
            }
    return out


def test_whole_graph_numerics_follow_the_sub_path_on_minimax_m3_ptpc():
    """MiniMax-M3 PTPC (fp8-token, per_Token, Swiglu): the prefill sub-path quantizes
    activations per token and re-quantizes the intermediate to fp8 per token; batch1 and
    decode keep bf16 activations and do not re-quantize (MOEBUG-043)."""
    S = _S("per_Token", "fp8")
    routes = _whole_graph_routes()
    pre = accuracy.numerics_from_route(S, routes[True])
    assert pre["act_quant"] == "per_token_fp8"
    assert pre["inter"] == "bf16_then_per_token_fp8"
    assert pre["inter_quant"] == "per_token_fp8"
    assert pre["combine"] == "bf16_rows_fp32_sum"
    dec = accuracy.numerics_from_route(S, routes[False])
    assert dec["act_quant"] == "none"
    assert dec["inter"] == "bf16" and dec["inter_quant"] == "none"
    assert dec["combine"] == "bf16_atomic"
    # both round-trip through the validator that path_ref applies
    for n in (pre, dec):
        accuracy.normalize_numerics(S, n, derived=True)
    # the two score against different references: they must not share a key
    assert accuracy.numerics_key(pre) != accuracy.numerics_key(dec)


def test_whole_graph_without_a_config_in_its_name_is_an_error():
    S = _S("per_Token", "fp8")
    bad = {
        "stages": "whole-graph",
        "kernel1": "impl__flydsl_gfx942__nonsense",
        "full_impl": "impl__flydsl_gfx942__nonsense",
    }
    with pytest.raises(accuracy.NumericsError, match="carries no config"):
        accuracy.numerics_from_route(S, bad)


def test_whole_graph_on_a_bf16_case_has_no_quant_in_either_sub_path():
    S = _S("No", "bf16")
    for route in _whole_graph_routes().values():
        n = accuracy.numerics_from_route(S, route)
        assert n["act_quant"] == "none" and n["inter"] == "bf16"


def test_in_kernel_paths_are_scored_against_the_scale_rule_they_match():
    S = _mx()
    in_kernel = {
        "act_quant": "mx_fp4_round_up",
        "inter": "fp32_direct",
        "inter_quant": "mx_fp4_in_kernel",
        "combine": "bf16_atomic (inferred)",
        "out_dtype": "bf16",
    }
    assert accuracy.has_in_kernel(in_kernel)
    v = accuracy.in_kernel_variants(S, in_kernel)
    assert set(v) == {"round_up", "even", "round_up+bf16", "even+bf16"}
    # the activation went through the quant call: only the intermediate varies
    assert v["even"]["act_quant"] == "mx_fp4_round_up"
    assert v["even"]["inter_quant"] == "mx_fp4_even"
    assert v["even"]["inter"] == "fp32_direct"
    # the activation output rounded to bf16 before the quant is its own variant
    assert v["round_up+bf16"]["inter"] == "bf16_then_mx_fp4_round_up"
    assert v["even+bf16"]["inter"] == "bf16_then_mx_fp4_even"
    assert not accuracy.has_in_kernel(v["even"])
    # a 1-stage kernel quantizes both in-kernel: both rules vary together
    one = dict(in_kernel, act_quant="mx_fp4_in_kernel")
    v1 = accuracy.in_kernel_variants(S, one)
    assert v1["even"]["act_quant"] == "mx_fp4_even"
    assert v1["round_up+bf16"]["inter_quant"] == "mx_fp4_round_up"


def test_fp32_direct_reference_does_not_round_the_activation_to_bf16():
    """ref_path at fp32_direct hands the fp32 activation output to the MX quant, so it
    differs from the bf16_then reference by the bf16 rounding (0.023 logits_diff on
    a4w4 SwiGLU at tier1-00102, measured) and not from an fp32 emulation."""
    import torch

    x = torch.randn(4, 64, device="cpu") * 3
    fp32 = accuracy._inter_values(
        types.SimpleNamespace(dtype=torch.bfloat16), x.double(), "fp32_direct"
    )
    assert fp32.dtype == torch.float32 and torch.equal(
        fp32.double(), x.double().float().double()
    )
    bf = accuracy._inter_values(
        types.SimpleNamespace(dtype=torch.bfloat16),
        x.double(),
        "bf16_then_mx_fp4_round_up",
    )
    assert bf.dtype == torch.bfloat16 and not torch.equal(bf.float(), fp32)


def test_match_rule_picks_the_closer_reference_and_applies_the_strict_rule():
    def seed(up, even):
        return {
            "seed": 0,
            "rep": 0,
            "path_round_up_rel_l2": up,
            "path_even_rel_l2": even,
            "path_round_up_allclose": 0.2 if up > 0.01 else 0.0,
            "path_even_allclose": 0.2 if even > 0.01 else 0.0,
            "path_round_up_logits_diff": up * up / 2,
            "path_even_logits_diff": even * even / 2,
            "exact_rel_l2": 0.5,
            "exact_logits_diff": 0.12,
        }

    for matches, small, big in (("even", 1e-3, 0.27), ("round_up", 1e-3, 0.27)):
        per = [seed(big, small) if matches == "even" else seed(small, big)] * 3
        res = {
            "per_seed": per,
            "nan_inf": 0,
            "sentinel_ok": True,
            **accuracy.aggregate(per),
        }
        assert accuracy.match_rule(res, ("round_up", "even")) == matches
        assert accuracy.rule(res) is True
    # neither rule matches: both references are far, aiter's rule fails the path
    per = [seed(0.3, 0.25)] * 3
    res = {
        "per_seed": per,
        "nan_inf": 0,
        "sentinel_ok": True,
        **accuracy.aggregate(per),
    }
    accuracy.match_rule(res, ("round_up", "even"))
    assert accuracy.rule(res) is False
    assert accuracy.match_rule({"mean": {}, "per_seed": []}, ("even",)) is None


def test_mx_flydsl_moe1_is_matched_not_assumed_bf16_then():
    """flydsl_moe1 quantizes from fp32 at a4w4 M=64/4096 and rounds to bf16 first at
    M=8 (MI355X): the route cannot say which, so the output picks it."""
    import aiter.fused_moe as fm

    S = _mx()
    fn = lambda *a, **k: None
    fn.__module__ = "aiter.ops.quant"
    fm.fused_dynamic_mxfp4_quant_moe_sort, real = (
        fn,
        fm.fused_dynamic_mxfp4_quant_moe_sort,
    )
    try:
        n = accuracy.numerics_from_route(
            S,
            {
                "stages": "2-stage",
                "backend1": "flydsl",
                "backend2": "flydsl",
                "kernel1": "flydsl_moe1_afp4_wfp4_bf16_t64x256x256",
                "kernel2": "flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce",
            },
        )
    finally:
        fm.fused_dynamic_mxfp4_quant_moe_sort = real
    assert accuracy.has_in_kernel(n)
    assert n["act_quant"] == "mx_fp4_round_up"  # through the quant call
    v = accuracy.in_kernel_variants(S, n)
    assert v["round_up"]["inter"] == "fp32_direct"
    assert v["round_up+bf16"]["inter"] == "bf16_then_mx_fp4_round_up"
    assert n["combine"] == "bf16_rows_fp32_sum"
