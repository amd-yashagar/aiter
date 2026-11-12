# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Silu a8w4 MoE must apply the expert bias, not drop it.

Dispatch enabled the FlyDSL and Opus stage bias only for Swiglu, so a caller that
passed ``bias1``/``bias2`` with Silu and fp8 activations got the bias-free result
with no error. The kernels apply the bias for Silu as well.
"""

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

TOKEN = 256
MODEL_DIM = 512
INTER_DIM = 256
E = 8
TOPK = 2
BIAS_SCALE = 50.0


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(fused_moe_module, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "get_cu_num", lambda: 256)
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def _dispatch(activation, q_dtype_a=dtypes.fp8):
    get_2stage_cfgs.cache_clear()
    return get_2stage_cfgs(
        TOKEN,
        MODEL_DIM,
        INTER_DIM,
        E,
        TOPK,
        dtypes.bf16,
        q_dtype_a,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        activation,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode=GateMode.INTERLEAVE.value,
        has_stage1_bias=True,
        has_stage2_bias=True,
        opus_weights_shuffled=True,
    )


@pytest.mark.parametrize("activation", [ActivationType.Silu, ActivationType.Swiglu])
def test_heuristic_flydsl_route_enables_bias(gfx950, activation):
    meta = _dispatch(activation)
    assert meta.has_bias and meta.stage2_has_bias


def test_tuned_flydsl_row_enables_silu_bias(gfx950, monkeypatch):
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
        "block_m": 32,
        "ksplit": 0,
        "kernelName1": "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w2_gui",
        "kernelName2": "flydsl_moe2_afp8_wfp4_bf16_t32x128x256_atomic_bnt2",
        "run_1stage": False,
    }
    monkeypatch.setattr(fused_moe_module, "cfg_2stages", ({key: cfg}, {}))
    meta = _dispatch(ActivationType.Silu)
    assert meta.has_bias and meta.stage2_has_bias


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 a8w4 MoE required",
)
def test_silu_a8w4_applies_bias():
    torch.manual_seed(0)
    dev, dtype = "cuda", dtypes.bf16
    x = torch.randn(TOKEN, MODEL_DIM, dtype=dtype, device=dev)
    w1 = torch.randn(E, INTER_DIM * 2, MODEL_DIM, dtype=dtype, device=dev)
    w2 = torch.randn(E, MODEL_DIM, INTER_DIM, dtype=dtype, device=dev)
    # A large bias: a dropped one moves the result far outside the quantization noise.
    bias1 = BIAS_SCALE * torch.clamp(torch.randn(E, INTER_DIM * 2, device=dev), -1, 1)
    bias2 = BIAS_SCALE * torch.clamp(torch.randn(E, MODEL_DIM, device=dev), -1, 1)
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
    ref1 = fused_moe_module.torch_moe_stage1(
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
    )
    a2, a2_scale = per_1x32_f8_scale_f8_quant(
        ref1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
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
    )

    assert torch.isfinite(out).all()
    ref, out = ref.double(), out.double()
    logits_diff = 1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum()
    assert logits_diff < 1e-3, f"logits_diff={logits_diff.item():.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
