# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side numeric, binding, split-K and tuned-instance contract tests."""

import ast
import csv
import functools
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.moe_limits import normalize_swiglu_limit

moe = importlib.import_module("aiter.fused_moe")
ops = importlib.import_module("aiter.ops.moe_op")


@pytest.fixture(autouse=True)
def fresh_dispatch(monkeypatch):
    monkeypatch.setattr(moe, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(moe, "get_gfx_runtime", lambda: "gfx942")
    monkeypatch.setattr(moe, "get_cu_num", lambda: 304)
    get_cfg = moe.get_2stage_cfgs
    get_cfg.cache_clear()
    moe.AITER_CONFIGS.get_config_file.cache_clear()
    moe.cfg_2stages_by_limit_tag.clear()
    yield
    get_cfg.cache_clear()
    moe.AITER_CONFIGS.get_config_file.cache_clear()


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
    assert kwargs["swiglu_limit"] == (limit if limit == 1.0 else 0.0)


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
    assert kwargs["swiglu_limit"] == 1.0


@pytest.mark.parametrize("kernel", ["stale_activation_F8", "old_silu_Splitk_F8"])
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


@pytest.mark.parametrize("limit", [None, 0.0, 1.0, float("inf")])
def test_binding_selects_variant_from_limit(monkeypatch, limit):
    enabled = normalize_swiglu_limit(limit) is not None
    calls = []
    monkeypatch.setattr(ops, "ck_moe_stage1", lambda *a, **k: calls.append((a, k)))
    ops.ck_moe_stage1_fwd(
        *([None] * 7),
        1,
        quant_type=QuantType.per_1x128,
        activation=ActivationType.Silu,
        splitk=4 if enabled else 1,
        swiglu_limit=limit,
    )
    args, kwargs = calls.pop()
    assert kwargs["swiglu_limit"] == (1.0 if enabled else 0.0)
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
    )
    args, _ = calls.pop()
    assert args[13] == QuantType.per_1x128.value


def test_default_and_limit_module_names_match_prebuild_recipes():
    recipe = importlib.import_module("aiter.jit.utils.moe_recipes")
    args = (
        aiter.dtypes.fp8,
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        ActivationType.Silu,
        QuantType.per_1x128,
        2,
    )
    default_name, _ = ops.get_moe_stage_module(
        *args,
        preshuffle_mode=True,
    )
    default_prebuild_name, _ = recipe._build_moe_variant(
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
    limit_name, limit_cmd = ops.get_moe_stage_module(
        *args,
        preshuffle_mode=True,
        runtime_swiglu_limit=True,
    )
    limit_prebuild_name, limit_prebuild_cmd = recipe._build_moe_variant(
        aiter_csrc_dir="csrc",
        a_dtype="f8",
        b_dtype="f8",
        c_dtype="b16",
        activation="silu",
        quant_type="per_1x128",
        mul_routed_weight_stage=2,
        preshuffle_mode=True,
        is_splitk=False,
        runtime_swiglu_limit=True,
    )
    assert default_name == default_prebuild_name
    assert limit_name == limit_prebuild_name
    assert default_name != limit_name
    assert "--runtime-swiglu-limit" in limit_cmd[0]
    assert "--runtime-swiglu-limit" in limit_prebuild_cmd[0]


def test_limit_tag_prebuilds_default_stage2_module(monkeypatch):
    recipe = importlib.import_module("aiter.jit.utils.moe_recipes")
    monkeypatch.setattr(
        recipe,
        "_get_tuned_fmoe_rows",
        lambda: [
            {
                "dtype": "torch.bfloat16",
                "q_dtype_a": "torch.float8_e4m3fn",
                "q_dtype_w": "torch.float8_e4m3fn",
                "q_type": "per_1x128",
                "act_type": "silu",
                "_tag": "swiglu_limit=5",
                "ksplit": "0",
            }
        ],
    )
    variants = recipe.get_moe_ck2stages_prebuild_variants("csrc")
    names = {variant["md_name"] for variant in variants}
    stage2_name, _ = ops.get_moe_stage_module(
        aiter.dtypes.fp8,
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        ActivationType.Silu,
        QuantType.per_1x128,
        2,
        preshuffle_mode=True,
    )
    limit_name, _ = ops.get_moe_stage_module(
        aiter.dtypes.fp8,
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        ActivationType.Silu,
        QuantType.per_1x128,
        2,
        preshuffle_mode=True,
        runtime_swiglu_limit=True,
    )
    assert names == {stage2_name, limit_name}


def test_runtime_limit_codegen_heuristic_uses_enabled_specialization(tmp_path):
    generator = (
        Path(__file__).resolve().parents[1]
        / "csrc/ck_gemm_moe_2stages_codegen/gen_instances.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(generator),
            "-a",
            "f8",
            "-b",
            "f8",
            "-c",
            "b16",
            "-q",
            "per_1x128",
            "-act",
            "silu",
            "-m",
            "2",
            "-w",
            str(tmp_path),
            "--runtime-swiglu-limit",
        ],
        check=True,
        cwd=generator.parent,
        capture_output=True,
        text=True,
    )
    dispatch = (tmp_path / "ck2stages_moe_stage1_heuristic_dispatch.hpp").read_text()
    references = re.findall(r"ck_moe_stage1_gemm<([^;]+)>;", dispatch)
    assert references
    assert all(reference.strip().endswith(", true") for reference in references)


def test_runtime_module_requires_fp8_blockscale():
    with pytest.raises(ValueError, match="non-split FP8 per_1x128"):
        ops.get_moe_stage_module(
            aiter.dtypes.bf16,
            aiter.dtypes.fp8,
            aiter.dtypes.bf16,
            ActivationType.Silu,
            QuantType.per_1x128,
            2,
            runtime_swiglu_limit=True,
        )


@pytest.fixture
def tuner(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "csrc/ck_gemm_moe_2stages_codegen")
    )
    return importlib.import_module("gemm_moe_tune")


@pytest.mark.parametrize("limit", [None, 0.0, 1.0, float("inf")])
def test_tuner_passes_normalized_limit(monkeypatch, tuner, limit):
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
    assert kwargs["swiglu_limit"] == (1.0 if limit == 1.0 else 0.0)


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
    with path.open() as config_file:
        rows = list(csv.DictReader(config_file))
    assert len(rows) == 16
    for row in rows:
        assert "_silu_" in row["kernelName1"]
        assert "Splitk" not in row["kernelName1"]
        assert int(row["ksplit"]) == 0
        assert int(row["run_1stage"]) == 0
        assert row["_tag"] in ("swiglu_limit=5", "swiglu_limit=7")

    recipe = importlib.import_module("aiter.jit.utils.moe_recipes")
    found = recipe._get_tuned_fmoe_rows()
    assert all(
        any(all(r.get(k) == v for k, v in row.items()) for r in found) for row in rows
    )
    prebuild_names = {
        variant["md_name"]
        for variant in recipe.get_moe_ck2stages_prebuild_variants("csrc")
    }
    default_name, _ = ops.get_moe_stage_module(
        aiter.dtypes.fp8,
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        ActivationType.Silu,
        QuantType.per_1x128,
        2,
        preshuffle_mode=True,
    )
    runtime_name, _ = ops.get_moe_stage_module(
        aiter.dtypes.fp8,
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        ActivationType.Silu,
        QuantType.per_1x128,
        2,
        preshuffle_mode=True,
        runtime_swiglu_limit=True,
    )
    assert {default_name, runtime_name} <= prebuild_names


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


def test_asm_stage1_rejects_limit_without_splitk():
    with pytest.raises(NotImplementedError, match="swiglu_limit"):
        moe.asm_stage1(
            *([None] * 6),
            torch.empty(1, 1, 1, device="cpu"),
            9,
            32,
            ksplit=0,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x128,
            swiglu_limit=5.0,
        )


@pytest.mark.parametrize("quant_type", [QuantType.No, QuantType.per_Token])
def test_ck_stage1_rejects_non_blockscale_limit_without_splitk(quant_type):
    with pytest.raises(NotImplementedError, match="swiglu_limit"):
        moe.ck_moe_stage1(
            torch.empty(1, 4096, device="cpu"),
            *([None] * 6),
            9,
            32,
            None,
            None,
            quant_type=quant_type,
            activation=ActivationType.Silu,
            splitk=1,
            swiglu_limit=5.0,
        )


def _dispatch(token, model_dim, inter_dim, expert, topk, limit):
    return moe.get_2stage_cfgs(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtypes.bf16,
        dtypes.fp8,
        dtypes.fp8,
        QuantType.per_1x128,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        swiglu_limit=limit,
    )


def test_limit_5_uses_the_epilogue():
    meta = _dispatch(32, 4096, 1024, 289, 9, 5.0)
    assert meta.run_1stage is False
    assert int(meta.ksplit or 0) == 0


def _swiglu_limit_csv():
    return (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "aiter/configs/model_configs/a8w8_blockscale_tuned_swiglu_limit_fmoe_gfx942.csv"
    )


def _capture_warnings(monkeypatch):
    messages = []
    monkeypatch.setattr(
        "aiter.fused_moe.logger.warning",
        lambda msg, *args, **kwargs: messages.append(str(msg)),
    )
    return messages


def test_tuned_cohere_row_uses_the_epilogue(monkeypatch):
    monkeypatch.setenv("AITER_CONFIG_FMOE_SWIGLU_LIMIT", str(_swiglu_limit_csv()))
    messages = _capture_warnings(monkeypatch)
    meta = _dispatch(32, 2048, 768, 128, 8, 5.0)
    assert not int(meta.run_1stage or 0)
    assert int(meta.ksplit or 0) == 0
    assert "_silu_" in meta.stage1.keywords["kernelName"]
    assert not any("no row" in msg for msg in messages)


def test_missing_limit_row_logs_at_default_level(monkeypatch):
    monkeypatch.setenv("AITER_CONFIG_FMOE_SWIGLU_LIMIT", str(_swiglu_limit_csv()))
    messages = _capture_warnings(monkeypatch)
    infos = []
    monkeypatch.setattr(moe.logger, "info", lambda msg, *a, **k: infos.append(str(msg)))
    # Token 16 is not one of the tuned buckets (1, 32, 1024, 8192).
    _dispatch(16, 2048, 768, 128, 8, 5.0)
    assert not any("no row" in msg for msg in messages)
    assert any("no row" in msg for msg in infos)


@pytest.mark.parametrize("quant_type", [QuantType.No, QuantType.per_Token])
@pytest.mark.parametrize("stage", ["ck_moe_stage1", "asm_stage1"])
@pytest.mark.parametrize("limit", [None, 0.0, float("inf"), 5.0])
def test_two_stage_forwards_limit_for_every_quant_type(
    monkeypatch, quant_type, stage, limit
):
    launches = []
    stage1 = getattr(moe, stage)
    metadata = moe.MOEMetadata(
        functools.partial(
            stage1, quant_type=quant_type, activation=ActivationType.Silu
        ),
        lambda *a, **k: a[6],
        16,
        0,
        prequant=False,
        skip_inter_quant=True,
    )
    monkeypatch.setattr(moe, "get_2stage_cfgs", lambda *a, **k: metadata)
    monkeypatch.setattr(moe, "get_quant", lambda _: lambda x, **k: (x, None))
    monkeypatch.setattr(
        aiter, "ck_moe_stage1_fwd", lambda *a, **k: launches.append((a, k))
    )
    monkeypatch.setattr(
        aiter, "moe_stage1_g1u1", lambda *a, **k: launches.append((a, k))
    )
    x = torch.empty(1, 128, dtype=dtypes.bf16)
    call = functools.partial(
        moe.fused_moe_2stages,
        x,
        torch.empty(1, 256, 128, dtype=dtypes.bf16),
        torch.empty(1, 128, 128, dtype=dtypes.bf16),
        1,
        torch.zeros(16, dtype=torch.int32),
        None,
        torch.zeros(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        torch.empty(1, 128, dtype=dtypes.bf16),
        True,
        16,
        quant_type=quant_type,
        activation=ActivationType.Silu,
        swiglu_limit=limit,
    )
    if limit == 5.0:
        with pytest.raises(NotImplementedError, match="swiglu_limit"):
            call()
        assert not launches
    else:
        call()
        assert len(launches) == 1
        if stage == "ck_moe_stage1":
            assert launches[0][1]["swiglu_limit"] == 0.0


@pytest.mark.parametrize("limit", [None, 0.0, float("inf"), 5.0])
def test_asm_splitk_normalizes_limit(monkeypatch, limit):
    calls = []
    monkeypatch.setattr(aiter, "moe_stage1_g1u1", lambda *a, **k: None)
    monkeypatch.setattr(aiter, "silu_and_mul", lambda *a: calls.append(a))
    moe.asm_stage1(
        torch.empty(1, 128),
        torch.empty(1, 256, 128),
        torch.empty(1, 128, 128),
        None,
        None,
        None,
        torch.empty(1, 1, 128, dtype=dtypes.bf16),
        1,
        16,
        ksplit=1,
        quant_type=QuantType.No,
        activation=ActivationType.Silu,
        swiglu_limit=limit,
    )
    assert calls[0][-1] == (5.0 if limit == 5.0 else 0.0)


@pytest.mark.parametrize("limit", [-1.0, float("nan"), -float("inf")])
def test_binding_wrapper_rejects_invalid_limit(limit):
    with pytest.raises(ValueError, match="swiglu_limit"):
        ops.ck_moe_stage1_fwd(*([None] * 7), 1, swiglu_limit=limit)


def test_default_limit_config_keeps_both_tags_without_warnings(monkeypatch):
    import pandas as pd

    monkeypatch.delenv("AITER_CONFIG_FMOE_SWIGLU_LIMIT", raising=False)
    messages = _capture_warnings(monkeypatch)
    path = moe.AITER_CONFIGS.AITER_CONFIG_FMOE_SWIGLU_LIMIT_FILE
    rows = pd.read_csv(path)
    assert len(rows) == 16
    assert set(rows["_tag"]) == {"swiglu_limit=5", "swiglu_limit=7"}
    assert not messages
    meta = _dispatch(32, 2048, 768, 128, 8, 5.0)
    assert "_silu_" in meta.stage1.keywords["kernelName"]


def test_limit_config_pathsep_override(monkeypatch, tmp_path):
    import pandas as pd

    rows = pd.read_csv(_swiglu_limit_csv())
    paths = [tmp_path / "five.csv", tmp_path / "seven.csv"]
    for tag, path in zip(("swiglu_limit=5", "swiglu_limit=7"), paths):
        rows[rows["_tag"] == tag].to_csv(path, index=False)
    monkeypatch.setenv(
        "AITER_CONFIG_FMOE_SWIGLU_LIMIT", os.pathsep.join(map(str, paths))
    )
    result = pd.read_csv(moe.AITER_CONFIGS.AITER_CONFIG_FMOE_SWIGLU_LIMIT_FILE)
    assert len(result) == 16


def test_prebuild_import_without_aiter_package():
    root = Path(__file__).resolve().parents[1]
    package = "aiter"
    code = (
        f"import sys; sys.path.insert(0, {str(root / package)!r}); "
        "from jit.utils.moe_recipes import _get_tuned_fmoe_rows; "
        f"assert _get_tuned_fmoe_rows(); assert {package!r} not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "quant_type", [QuantType.No, QuantType.per_Token, QuantType.per_1x128]
)
@pytest.mark.parametrize("limit", [None, 0.0, float("inf"), 5.0])
def test_one_stage_limit_guard_for_every_quant_type(quant_type, limit):
    # Exercise the early guard without sorting or launching any GPU kernels.
    tree = ast.parse(Path(moe.__file__).read_text())
    impl = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_fused_moe_impl"
    )
    branch = next(
        n
        for n in ast.walk(impl)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "metadata.run_1stage"
    )
    guard = next(
        n
        for n in branch.body
        if isinstance(n, ast.If) and "normalize_swiglu_limit" in ast.unparse(n.test)
    )
    code = compile(ast.Module(body=[guard], type_ignores=[]), moe.__file__, "exec")
    scope = {
        "activation": ActivationType.Silu,
        "ActivationType": ActivationType,
        "quant_type": quant_type,
        "swiglu_limit": limit,
        "normalize_swiglu_limit": normalize_swiglu_limit,
    }
    if limit == 5.0:
        with pytest.raises(NotImplementedError, match="1-stage MoE"):
            exec(code, scope)  # noqa: S102 - the local early guard, without GPU setup
    else:
        exec(code, scope)  # noqa: S102 - the local early guard, without GPU setup


@pytest.mark.parametrize("kernel", ["stale_activation_F8", "known_silu_Splitk_F8"])
def test_stale_tuned_limit_kernel_rejected(monkeypatch, tmp_path, kernel):
    import pandas as pd

    rows = pd.read_csv(_swiglu_limit_csv())
    rows.loc[:, "kernelName1"] = kernel
    path = tmp_path / "stale.csv"
    rows.to_csv(path, index=False)
    monkeypatch.setenv("AITER_CONFIG_FMOE_SWIGLU_LIMIT", str(path))
    with pytest.raises(ValueError, match="Incompatible runtime SiLU limit tuning row"):
        _dispatch(32, 2048, 768, 128, 8, 5.0)
