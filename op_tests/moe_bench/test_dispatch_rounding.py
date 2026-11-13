# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch that rounds the activations more than the case asks for: scored on the requested
math, with the least-rounding aiter path (the activation-dtype bound raised) as a second bar."""

import os
import types

from op_tests.moe_bench import accuracy, bar_worker, schema, workload


def _case(cid, token, aq, wq, extra="", act="Swiglu", qt="per_1x32"):
    row = {c: "" for c in schema.COLS}
    row.update(
        case_id=cid, kind="fused", family="a16w4", token=str(token), model_dim="3072",
        inter_dim="3072", expert="128", topk="4", act_type=act, dtype="bf16",
        q_dtype_a=aq, q_dtype_w=wq, q_type=qt, use_g1u1="1", doweight_stage1="0",
        modes="warm", extra=extra,
    )  # fmt: skip
    return schema.Case(row, schema.parse_extra(extra))


SEP = "gate_mode=SEPARATED;bias=s1+s2"


def test_separated_swiglu_a16w4_is_rounded_more_from_the_bound_up():
    for m, rounds in ((8, False), (255, False), (256, True), (4096, True)):
        S = workload.resolve(_case("c", m, "bf16", "fp4x2", SEP))
        why = accuracy.dispatch_rounds_more(S, m, "gfx950")
        assert bool(why) is rounds, (m, why)
    assert "bf16" in why and "fp4x2" in why and "requested math" in why


def test_rows_that_ask_for_the_quantized_activations_are_not_rounded_more():
    a4w4 = workload.resolve(_case("c", 4096, "fp4x2", "fp4x2"))
    assert accuracy.dispatch_rounds_more(a4w4, 4096, "gfx950") == ""
    fp8 = workload.resolve(_case("c", 4096, "fp8", "fp8", act="Silu", qt="per_Token"))
    assert accuracy.dispatch_rounds_more(fp8, 4096, "gfx950") == ""
    a16 = workload.resolve(_case("c", 4096, "bf16", "bf16", act="Silu", qt="No"))
    assert accuracy.dispatch_rounds_more(a16, 4096, "gfx950") == ""


def test_requested_numerics_are_the_unquantized_math():
    S = workload.resolve(_case("c", 256, "bf16", "fp4x2", SEP))
    n = accuracy.requested_numerics(
        S, {"stages": "2-stage", "kernel2": "flydsl_moe2_x"}
    )
    assert (n["act_quant"], n["inter"], n["inter_quant"]) == ("none", "bf16", "none")
    assert n["out_dtype"] == "bf16"


def test_a_kernel_that_is_the_same_as_dispatch_matches_the_requested_math_here():
    S = workload.resolve(_case("c", 256, "bf16", "fp4x2", SEP))
    route = {"stages": "2-stage", "kernel1": "flydsl_moe1_afp4_wfp4_bf16_t32x32x256", "kernel2": "flydsl_moe2_afp4_wfp4_bf16_t32x128x128_atomic", "backend1": "flydsl"}  # fmt: skip
    cand = types.SimpleNamespace(declared_numerics={"same_as": "aiter dispatched path"})
    plain = types.SimpleNamespace(dispatch_route=route, dispatch_rounds_more="")
    rounded = types.SimpleNamespace(dispatch_route=route, dispatch_rounds_more="x")
    n = accuracy.numerics_of(S, cand, {}, rounded)
    assert n == accuracy.requested_numerics(S, route)
    assert (n["act_quant"], n["inter"]) == ("none", "bf16")
    # the case's own label (bf16 activations) already makes the dispatched route's
    # numerics the unquantized math: the flag states it instead of leaving it implicit
    assert accuracy.numerics_of(S, cand, {}, plain) == n


def test_the_bound_override_is_scoped_and_confirmed_from_what_aiter_logged(monkeypatch):
    import aiter.fused_moe as fm
    from aiter import dtypes

    monkeypatch.delenv("AITER_BF16_FP8_MOE_BOUND", raising=False)
    S = workload.resolve(_case("c", 256, "bf16", "fp4x2", SEP))
    cand = bar_worker.Bf16BoundCandidate()
    before = fm._SWIGLU_MXFP4_BF16_BOUND
    cand.prepare(None)
    assert fm._SWIGLU_MXFP4_BF16_BOUND > 10**12
    assert (
        fm.resolve_activation_dtype(
            S.qtype, S.wq, activation=S.act, gate_mode=S.gate_mode, M=4096, gfx="gfx950"
        )
        == dtypes.bf16
    )  # the switch is out of reach
    cand.release(None)
    assert fm._SWIGLU_MXFP4_BF16_BOUND == before
    assert "AITER_BF16_FP8_MOE_BOUND" not in os.environ
    route = {"kernel1": "cktile_moe_stage1", "kernel2": "cktile_moe_stage2"}
    assert cand.confirm({"q_dtype_a": dtypes.fp4x2}, route, S).startswith(
        "activation dtype is still"
    )
    assert cand.confirm(None, route, S).startswith("no get_2stage_cfgs call")
    assert cand.confirm({"q_dtype_a": dtypes.bf16}, route, S) is None
    assert cand.confirmed == (
        "q_dtype_a=bfloat16 kernel1=cktile_moe_stage1 kernel2=cktile_moe_stage2"
    )
    assert cand.describe()["kernels"] == "cktile_moe_stage1 + cktile_moe_stage2"


def test_the_bar_row_carries_dispatch_rounds_more_and_the_second_bar():
    import io
    import json

    S = workload.resolve(_case("c", 256, "bf16", "fp4x2", SEP))
    c = _case("c", 256, "bf16", "fp4x2", SEP)
    lr = bar_worker.Bf16BoundCandidate()
    lr.confirmed = (
        "q_dtype_a=bfloat16 kernel1=cktile_moe_stage1 kernel2=cktile_moe_stage2"
    )
    lr.kernels = "cktile_moe_stage1 + cktile_moe_stage2"
    results = {
        "dispatch": {"ok": False, "rounds_more": "activations requested bf16, ...",
                     "route": {"kernel1": "flydsl_moe1_x"}},
        lr.id: {"ok": True, "cand": lr, "by_mode": {"warm": {"median_us": 41.0}},
                "numerics": "act=none", "acc": None},
    }  # fmt: skip
    fh = io.StringIO()
    ident = {"gpu": "MI355X", "sku": None, "gfx": "gfx950"}
    bar_worker._emit_summary(fh, c, S, "warm", results, ident, [], "bar")
    row = json.loads(fh.getvalue().splitlines()[-1])
    assert row["dispatch_status"] == "rounds-more"
    assert row["dispatch_rounds_more"].startswith("activations requested")
    assert row["lr_us"] == 41.0 and row["lr_status"] == "ok"
    assert row["lr_kernels"] == "cktile_moe_stage1 + cktile_moe_stage2"
    assert row["lr_confirmed"].startswith("q_dtype_a=bfloat16")
    assert row["bar_id"] == lr.id  # the only correct candidate here


def test_the_logged_dtype_settles_it():
    from aiter import dtypes

    S = workload.resolve(_case("c", 256, "bf16", "fp4x2", SEP))
    assert accuracy.measured_rounds_more(S, dtypes.fp4x2, 256).startswith(
        "activations requested bf16, dispatched as fp4x2"
    )
    assert accuracy.measured_rounds_more(S, dtypes.bf16, 8) == ""
    silu = workload.resolve(_case("c", 4096, "bf16", "fp4x2", act="Silu"))
    assert accuracy.dispatch_rounds_more(silu, 4096, "gfx950") == ""  # not guessed
    assert accuracy.measured_rounds_more(silu, dtypes.fp4x2, 4096)  # but measured
