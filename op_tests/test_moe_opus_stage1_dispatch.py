# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A tuned Opus A8W4 stage1 row must be rejected for shapes the kernel can't take.

The Opus stage1 host code asserts its shape support and aborts the process (not
a Python exception) when the assert fails. The tuned rows are keyed without the
intermediate padding, so a caller that pads (gpt-oss style, e.g. 192/128) hits a
row tuned for the unpadded shape. Stage2 already discards unsupported rows with
``cfg_is_supported``; stage1 must do the same before the kernel is launched.
"""

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.opus import moe_stage1_a8w4 as opus_stage1
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

TOKEN = 8192
MODEL_DIM = 5120
INTER_DIM = 2304
E = 48
TOPK = 6
HIDDEN_PAD = 192
INTER_PAD = 128

# The shipped tuned row for this shape (model_configs/dsv4_fp8fp4_tuned_fmoe.csv).
KERNEL1 = "opus_moe1_afp8_wfp4_bf16_t64x384_gs_qgb3_aff32_p2_fp8"
KERNEL2 = "flydsl_moe2_layout_afp8_wfp4_bf16_t64x256x128_atomic_sbm64"


@pytest.mark.parametrize(
    ("kwargs", "supported"),
    [
        ({}, True),
        ({"kernel_name": KERNEL1[: -len("_fp8")]}, True),
        ({"inter_dim_pad": INTER_PAD}, False),
        ({"inter_dim_pad": 0, "inter_dim": 2304 + 64}, False),
        ({"block_m": 32}, False),
        ({"model_dim": 5120 + 64}, False),
        ({"gfx": "gfx942"}, False),
        ({"kernel_name": "opus_moe1_does_not_exist"}, False),
    ],
)
def test_stage1_cfg_is_supported(kwargs, supported):
    args = {
        "kernel_name": KERNEL1,
        "gfx": "gfx950",
        "model_dim": MODEL_DIM,
        "inter_dim": INTER_DIM,
        "inter_dim_pad": 0,
        "block_m": 64,
    }
    args.update(kwargs)
    ok, reason = opus_stage1.cfg_is_supported(args.pop("kernel_name"), **args)
    assert ok is supported
    assert bool(reason) is not supported


@pytest.fixture
def gfx950_opus_row(monkeypatch):
    key = (
        "gfx950",
        256,
        TOKEN,
        MODEL_DIM,
        INTER_DIM,
        E,
        TOPK,
        str(ActivationType.Silu),
        str(dtypes.bf16),
        str(dtypes.fp8),
        str(dtypes.fp4x2),
        str(QuantType.per_1x32),
        True,
        False,
    )
    cfg = {
        "block_m": 64,
        "ksplit": 0,
        "kernelName1": KERNEL1,
        "kernelName2": KERNEL2,
        "run_1stage": False,
    }
    monkeypatch.setattr(fused_moe_module, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe_module, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "cfg_2stages", ({key: cfg}, {}))
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def _dispatch(hidden_pad, intermediate_pad):
    return get_2stage_cfgs(
        TOKEN,
        MODEL_DIM,
        INTER_DIM,
        E,
        TOPK,
        dtypes.bf16,
        dtypes.fp8,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        hidden_pad,
        intermediate_pad,
        is_shuffled=True,
        gate_mode=GateMode.INTERLEAVE.value,
        opus_weights_shuffled=True,
    )


def _stage1_kernel(meta):
    return meta.stage1.keywords.get("kernelName", "")


def test_tuned_opus_stage1_row_is_used_when_supported(gfx950_opus_row):
    assert _stage1_kernel(_dispatch(0, 0)) == KERNEL1


def test_tuned_opus_stage1_row_is_discarded_when_padding_is_unsupported(
    gfx950_opus_row, monkeypatch
):
    warnings = []
    monkeypatch.setattr(
        fused_moe_module.logger,
        "warning",
        lambda message: warnings.append(str(message)),
    )
    meta = _dispatch(HIDDEN_PAD, INTER_PAD)

    assert any("Opus stage1 config unsupported" in w for w in warnings)
    assert not _stage1_kernel(meta).startswith("opus_")


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 a8w4 MoE required",
)
def test_padded_a8w4_call_on_tuned_opus_shape_matches_reference():
    torch.manual_seed(0)
    dev = "cuda"
    dtype = dtypes.bf16
    x = torch.randn(TOKEN, MODEL_DIM, dtype=dtype, device=dev)
    w1 = torch.randn(E, INTER_DIM * 2, MODEL_DIM, dtype=dtype, device=dev)
    w2 = torch.randn(E, MODEL_DIM, INTER_DIM, dtype=dtype, device=dev)
    w1[:, :, -HIDDEN_PAD:] = 0
    w1[:, -INTER_PAD:, :] = 0
    w1[:, INTER_DIM - INTER_PAD : INTER_DIM, :] = 0
    w2[:, :, -INTER_PAD:] = 0
    w2[:, -HIDDEN_PAD:, :] = 0
    bias1 = torch.clamp(torch.randn(E, INTER_DIM * 2, device=dev), -1.0, 1.0)
    bias2 = torch.clamp(torch.randn(E, MODEL_DIM, device=dev), -1.0, 1.0)
    topk_weights, topk_ids = fused_topk(
        x, torch.randn(TOKEN, E, dtype=dtype, device=dev), TOPK, True
    )

    quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q, w1_scale = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(E, INTER_DIM * 2, MODEL_DIM // 2)
    w2_q = w2_q.view(E, MODEL_DIM, INTER_DIM // 2)

    a1, a1_scale = per_1x32_f8_scale_f8_quant(
        x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    out1 = fused_moe_module.torch_moe_stage1(
        a1,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        w1_bias=bias1.to(dtype),
        situ_beta=1.0,
        situ_linear_beta=1.0,
    )
    a2, a2_scale = per_1x32_f8_scale_f8_quant(
        out1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    ref = fused_moe_module.torch_moe_stage2(
        a2.view(TOKEN, TOPK, -1),
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        w2_bias=bias2.to(dtype),
        doweight=True,
    )

    out = fused_moe(
        x,
        shuffle_weight_a16w4(w1_q, 16, True),
        shuffle_weight_a16w4(w2_q, 16, False),
        topk_weights,
        topk_ids,
        w1_scale=shuffle_scale_a16w4(w1_scale, E, True),
        w2_scale=shuffle_scale_a16w4(w2_scale, E, False),
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Silu,
        gate_mode=GateMode.INTERLEAVE.value,
        bias1=bias1,
        bias2=bias2,
        hidden_pad=HIDDEN_PAD,
        intermediate_pad=INTER_PAD,
    )

    assert not out.isnan().any().item()
    ref, out = ref.double(), out.double()
    logits_diff = 1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum()
    assert logits_diff < 1e-3, f"logits_diff={logits_diff.item():.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
