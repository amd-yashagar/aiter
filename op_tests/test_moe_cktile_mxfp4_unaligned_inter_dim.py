# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK-Tile MXFP4 MoE stage 2 with inter_dim a multiple of 128 but not of 256 (384, 640).

``e8m0_shuffle`` pads the weight-scale K groups to a multiple of 8 (K to 256) per
expert. The CK-Tile kernel strode and counted scale packs with the unpadded K, so
K = 384 or 640 read the wrong scales (every expert, truncated pack count) and
returned garbage. composable_kernel now uses the padded K layout and aiter pads the
A tail, and the build reports it through ``moe_cktile2stages_padded_scale_k``; an
older module keeps raising ``NotImplementedError`` for these shapes.
"""

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

MODEL_DIM = 1024


def test_unpatched_module_keeps_the_guard(monkeypatch):
    monkeypatch.setattr(fused_moe_module, "_cktile_mxfp4_padded_scale_k", lambda: False)
    assert fused_moe_module._cktile_mxfp4_stage2_unsafe(384)
    assert fused_moe_module._cktile_mxfp4_stage2_unsafe(640)
    assert not fused_moe_module._cktile_mxfp4_stage2_unsafe(512)


def test_patched_module_takes_multiples_of_128(monkeypatch):
    monkeypatch.setattr(fused_moe_module, "_cktile_mxfp4_padded_scale_k", lambda: True)
    assert not fused_moe_module._cktile_mxfp4_stage2_unsafe(384)
    assert not fused_moe_module._cktile_mxfp4_stage2_unsafe(640)
    assert fused_moe_module._cktile_mxfp4_stage2_unsafe(192)
    assert fused_moe_module._cktile_mxfp4_stage2_unsafe(160)


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 MXFP4 MoE required",
)


def _reference_and_output(inter_dim, num_experts, topk, a8):
    torch.manual_seed(0)
    dev, dtype, token = "cuda", dtypes.bf16, 64
    x = torch.randn(token, MODEL_DIM, dtype=dtype, device=dev)
    w1 = torch.randn(num_experts, inter_dim * 2, MODEL_DIM, dtype=dtype, device=dev)
    w2 = torch.randn(num_experts, MODEL_DIM, inter_dim, dtype=dtype, device=dev)
    topk_weights, topk_ids = fused_topk(
        x, torch.randn(token, num_experts, dtype=dtype, device=dev), topk, True
    )
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q, w1_scale = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(num_experts, inter_dim * 2, MODEL_DIM // 2)
    w2_q = w2_q.view(num_experts, MODEL_DIM, inter_dim // 2)

    def act_quant(t):
        if a8:
            return per_1x32_f8_scale_f8_quant(
                t, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
            )
        return t, None

    a1, a1_scale = act_quant(x)
    ref1 = fused_moe_module.torch_moe_stage1(
        a1,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
    )
    a2, a2_scale = act_quant(ref1)
    ref = fused_moe_module.torch_moe_stage2(
        a2.view(token, topk, -1),
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
    )
    out = fused_moe(
        x,
        shuffle_weight_a16w4(w1_q, 16, True),
        shuffle_weight_a16w4(w2_q, 16, False),
        topk_weights,
        topk_ids,
        w1_scale=shuffle_scale_a16w4(w1_scale, num_experts, True),
        w2_scale=shuffle_scale_a16w4(w2_scale, num_experts, False),
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Swiglu,
        gate_mode=GateMode.INTERLEAVE.value,
    )
    return ref.double(), out.double()


@_GPU
@pytest.mark.parametrize("a8", [False, True], ids=["a16w4", "a8w4"])
@pytest.mark.parametrize(("num_experts", "topk"), [(1, 1), (8, 2)])
@pytest.mark.parametrize("inter_dim", [256, 384, 512, 640])
def test_cktile_stage2_matches_reference(monkeypatch, inter_dim, num_experts, topk, a8):
    if inter_dim % 256 and not fused_moe_module._cktile_mxfp4_padded_scale_k():
        pytest.skip("the CK-Tile module predates the padded-K scale layout")
    if a8:
        monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    fused_moe_module.get_2stage_cfgs.cache_clear()
    ref, out = _reference_and_output(inter_dim, num_experts, topk, a8)

    assert torch.isfinite(out).all()
    logits_diff = 1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum()
    assert logits_diff < 1e-3, f"logits_diff={logits_diff.item():.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
