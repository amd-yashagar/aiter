# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side numeric, binding, split-K and tuned-instance contract tests."""

import csv
import importlib
from pathlib import Path

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.ops.moe_limits import normalize_swiglu_limit

moe = importlib.import_module("aiter.fused_moe")
ops = importlib.import_module("aiter.ops.moe_op")


@pytest.mark.parametrize("limit", [None, 0, -0.0, float("inf")])
def test_disabled_limit(limit):
    assert normalize_swiglu_limit(limit) is None
    assert moe.swiglu_limit_tag(limit) == ""


@pytest.mark.parametrize("limit", [1e-100, 1.0, 5.0, 7.0, 1e100])
def test_positive_limit(limit):
    assert normalize_swiglu_limit(limit) == limit
    assert moe.swiglu_limit_tag(limit) == f"swiglu_limit={limit:g}"


def test_dispatch_normalizes_infinite_limit_for_per_128x128_alias():
    quant_type, quant_type_a, limit = moe._normalize_moe_dispatch_contract(
        QuantType.per_128x128, ActivationType.Silu, float("inf")
    )
    assert quant_type == QuantType.per_1x128
    assert quant_type_a is None
    assert limit is None


@pytest.mark.parametrize("quant_type_a", [QuantType.per_128x128, QuantType.per_1x128])
def test_dispatch_accepts_matching_blockscale_aliases(quant_type_a):
    quant_type, canonical_quant_type_a, _ = moe._normalize_moe_dispatch_contract(
        QuantType.per_128x128, ActivationType.Silu, None, quant_type_a
    )
    assert quant_type == QuantType.per_1x128
    assert canonical_quant_type_a == QuantType.per_1x128


def test_torch_reference_clamps_large_finite_limit_without_overflow():
    gate = torch.tensor([2.0], dtype=torch.float32)
    up = torch.tensor([-3.0], dtype=torch.float32)
    gate_ref, up_ref = moe._clamp_swiglu_gate_up(gate, up, 1e100)
    torch.testing.assert_close(gate_ref, gate)
    torch.testing.assert_close(up_ref, up)


def test_torch_reference_applies_finite_limit():
    gate = torch.tensor([2.0], dtype=torch.float32)
    up = torch.tensor([-3.0], dtype=torch.float32)
    gate_ref, up_ref = moe._clamp_swiglu_gate_up(gate, up, 1.0)
    torch.testing.assert_close(gate_ref, torch.tensor([1.0]))
    torch.testing.assert_close(up_ref, torch.tensor([-1.0]))


@pytest.mark.parametrize("limit", [-1.0, float("nan"), -float("inf")])
def test_invalid_limit(limit):
    with pytest.raises(ValueError, match="swiglu_limit"):
        normalize_swiglu_limit(limit)
    with pytest.raises(ValueError, match="swiglu_limit"):
        moe.swiglu_limit_tag(limit)


@pytest.mark.parametrize("limit", [None, 0.0, 1.0, float("inf")])
@pytest.mark.parametrize("splitk", [0, 1, 4])
def test_ck_stage1_routes_without_split_for_override(monkeypatch, limit, splitk):
    calls = []
    monkeypatch.setattr(
        aiter, "ck_moe_stage1_fwd", lambda *a, **k: calls.append((a, k))
    )
    activations = []
    monkeypatch.setattr(aiter, "silu_and_mul", lambda *a: activations.append(a))
    x = torch.empty(1, 128)
    out = torch.empty(1, 1, 128)
    result = moe.ck_moe_stage1(
        x,
        torch.empty(1, 256, 128),
        None,
        torch.empty(16),
        None,
        None,
        out,
        1,
        16,
        None,
        None,
        quant_type=QuantType.per_1x128,
        activation=ActivationType.Silu,
        splitk=splitk,
        swiglu_limit=limit,
    )
    args, kwargs = calls.pop()
    assert result is out
    is_split = normalize_swiglu_limit(limit) is None and splitk > 1
    assert args[15] == (splitk if is_split else 0)
    assert bool(activations) is is_split
    assert kwargs["swiglu_limit_enabled"] is (limit == 1.0)
    assert kwargs["swiglu_limit"] == (limit if limit == 1.0 else 0.0)
    assert not kwargs.get("swiglustep", False)


def test_ck_stage1_canonicalizes_per_128x128_alias(monkeypatch):
    calls = []
    monkeypatch.setattr(
        aiter, "ck_moe_stage1_fwd", lambda *a, **k: calls.append((a, k))
    )
    moe.ck_moe_stage1(
        torch.empty(1, 128),
        torch.empty(1, 256, 128),
        None,
        torch.empty(16),
        None,
        None,
        torch.empty(1, 1, 128),
        1,
        16,
        None,
        None,
        quant_type=QuantType.per_128x128,
        activation=ActivationType.Silu,
        swiglu_limit=1.0,
    )
    args, kwargs = calls.pop()
    assert args[13] == QuantType.per_1x128
    assert kwargs["swiglu_limit_enabled"] is True


@pytest.mark.parametrize("kernel", ["old_swiglustep_F8", "old_silu_Splitk_F8"])
def test_stale_explicit_ck_kernel_rejected(kernel):
    with pytest.raises(ValueError, match="non-split CK SiLU"):
        moe.ck_moe_stage1(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            16,
            None,
            None,
            quant_type=QuantType.per_1x128,
            activation=ActivationType.Silu,
            kernelName=kernel,
            splitk=4,
            swiglu_limit=1.0,
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_binding_default_and_enabled_flag(monkeypatch, enabled):
    calls = []
    monkeypatch.setattr(ops, "ck_moe_stage1", lambda *a, **k: calls.append((a, k)))
    ops.ck_moe_stage1_fwd(
        *([None] * 7),
        1,
        quant_type=QuantType.per_1x128,
        activation=ActivationType.Silu,
        splitk=4 if enabled else 1,
        swiglu_limit=1.0,
        swiglu_limit_enabled=enabled,
    )
    args, kwargs = calls.pop()
    assert kwargs["swiglu_limit_enabled"] is enabled
    assert args[15] == (0 if enabled else 1)


def test_binding_canonicalizes_per_128x128_alias(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "ck_moe_stage1", lambda *a, **k: calls.append((a, k)))
    ops.ck_moe_stage1_fwd(
        *([None] * 7),
        1,
        quant_type=QuantType.per_128x128,
        activation=ActivationType.Silu,
        swiglu_limit=1.0,
        swiglu_limit_enabled=True,
    )
    args, _ = calls.pop()
    assert args[13] == QuantType.per_1x128.value


def test_silu_and_legacy_codegen_modules_are_distinct():
    args = (
        aiter.dtypes.fp8,
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        ActivationType.Silu,
        QuantType.per_1x128,
        2,
    )
    silu, silu_cmd = ops.get_moe_stage_module(*args)
    legacy, legacy_cmd = ops.get_moe_stage_module(*args, swiglustep=True)
    assert "runtime_limit_v2" in silu
    assert silu != legacy
    assert "-act silu " in silu_cmd[0]
    assert "-act swiglustep " in legacy_cmd[0]


def test_runtime_module_name_matches_prebuild_recipe():
    recipe = importlib.import_module("aiter.jit.utils.moe_recipes")
    runtime_name, _ = ops.get_moe_stage_module(
        aiter.dtypes.fp8,
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        ActivationType.Silu,
        QuantType.per_1x128,
        2,
        preshuffle_mode=True,
    )
    prebuild_name, _ = recipe._build_moe_variant(
        aiter_csrc_dir="csrc",
        a_dtype="f8",
        b_dtype="f8",
        c_dtype="b16",
        activation="silu",
        quant_type="per_1x128",
        mul_routed_weight_stage=2,
        preshuffle_mode=True,
        is_splitk=False,
    )
    assert runtime_name == prebuild_name


@pytest.fixture
def tuner(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "csrc/ck_gemm_moe_2stages_codegen")
    )
    return importlib.import_module("gemm_moe_tune")


@pytest.mark.parametrize("limit", [None, 0.0, 1.0, float("inf")])
def test_tuner_passes_independent_enable_flag(monkeypatch, tuner, limit):
    calls = []
    monkeypatch.setattr(
        tuner, "ck_moe_stage1_fwd", lambda *a, **k: calls.append((a, k))
    )
    monkeypatch.setattr(aiter, "get_hip_quant", lambda _: lambda out, **k: (out, None))
    tuner.FmoeTuner.ck_moe_stage1_fwd_out(
        torch.empty(1, 128),
        torch.empty(1, 256, 128),
        *([None] * 7),
        torch.bfloat16,
        1,
        "known_silu_F8",
        16,
        QuantType.per_1x128,
        ActivationType.Silu,
        swiglu_limit=limit,
    )
    args, kwargs = calls.pop()
    assert args[15] == 0
    assert kwargs["swiglu_limit_enabled"] is (limit == 1.0)
    assert "swiglustep" not in kwargs


def test_tuner_rejects_splitk_with_limit(tuner):
    with pytest.raises(ValueError, match="non-split CK"):
        tuner.FmoeTuner.ck_moe_stage1_fwd_out(
            torch.empty(1, 128),
            torch.empty(1, 256, 128),
            *([None] * 7),
            torch.bfloat16,
            1,
            "known_silu_F8",
            16,
            QuantType.per_1x128,
            ActivationType.Silu,
            splitk=4,
            swiglu_limit=1.0,
        )


def test_tuned_limit_rows_match_runtime_silu():
    path = Path(__file__).resolve().parents[1] / (
        "aiter/configs/model_configs/"
        "a8w8_blockscale_tuned_swiglu_limit_fmoe_gfx942.csv"
    )
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 16
    for row in rows:
        assert "_silu_" in row["kernelName1"]
        assert "Splitk" not in row["kernelName1"]
        assert int(row["ksplit"]) == 0
        assert int(row["run_1stage"]) == 0
        assert row["_tag"] in ("swiglu_limit=5", "swiglu_limit=7")


@pytest.mark.parametrize("ksplit", ["-1", "1", "4", "16"])
@pytest.mark.parametrize("tokens", [1, 32, 1024])
def test_dispatch_overrides_splitk_environment(monkeypatch, ksplit, tokens):
    monkeypatch.setenv("AITER_KSPLIT", ksplit)
    monkeypatch.setattr(moe, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(moe, "get_gfx_runtime", lambda: "gfx942")
    monkeypatch.setattr(moe, "get_cu_num", lambda: 304)
    moe.get_2stage_cfgs.cache_clear()
    try:
        meta = moe.get_2stage_cfgs(
            tokens,
            4096,
            1024,
            289,
            9,
            aiter.dtypes.bf16,
            aiter.dtypes.fp8,
            aiter.dtypes.fp8,
            QuantType.per_1x128,
            True,
            ActivationType.Silu,
            False,
            0,
            0,
            swiglu_limit=1.0,
        )
        assert not meta.run_1stage
        assert int(meta.ksplit or 0) == 0
    finally:
        moe.get_2stage_cfgs.cache_clear()
