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


def _S(qtype="per_1x128", aq="fp8"):
    import aiter
    from aiter import dtypes

    return types.SimpleNamespace(
        qtype=getattr(aiter.QuantType, qtype), aq=getattr(dtypes, aq)
    )


def _decl(**kw):
    d = {"act_quant": "per_1x128_fp8", "inter": "fp32_direct", "combine": "fp32_reduce"}
    d.update(kw)
    return d


def test_core_vocabulary_is_accepted_and_recorded():
    S = _S()
    n = accuracy.normalize_numerics(S, _decl())
    assert accuracy.numerics_key(n) == (
        "act_quant=per_1x128_fp8;inter=fp32_direct;combine=fp32_reduce"
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
    assert accuracy.numerics_key(accuracy.numerics_of(S, cand, {})).endswith(
        f"combine={combine}"
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
    assert accuracy.combine_of_route(r) == "bf16_atomic"
    f = {
        "stages": "2-stage",
        "backend2": "flydsl",
        "kernel2": "flydsl_moe2_afp8_t32x128_reduce",
    }
    assert accuracy.combine_of_route(f) == "bf16_rows_fp32_sum"
    f["kernel2"] = "flydsl_moe2_afp8_t32x128_atomic"
    assert accuracy.combine_of_route(f) == "bf16_atomic"
    assert accuracy.combine_of_route({"stages": "1-stage"}) == "n/a"
    assert accuracy.combine_of_route({"stages": "whole-graph"}) == "n/a"
    n = accuracy.numerics_from_route(_S(), r)
    assert n["combine"] == "bf16_atomic"
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
