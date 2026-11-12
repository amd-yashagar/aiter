# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK-Tile MXFP4 MoE has bf16 instances only.

``moe_cktile2stages_gemm1/2`` dispatch on the output dtype, and the fp16 branches
are commented out, so an fp16 output matched no branch and the call returned the
untouched output buffer: ``fused_moe`` on fp16 hidden states returned all zeros
through the CK-Tile route, without an error. The wrappers now compute in bf16
(the kernels also read the activation as bf16) and return fp16.
"""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import cktile_moe_stage1, cktile_moe_stage2, fused_moe, fused_topk
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

MODEL_DIM = 512
INTER_DIM = 256
E = 8
TOPK = 2


@pytest.fixture
def launches(monkeypatch):
    seen = []

    def stage1(xq, wq, y, *args, **kwargs):
        seen.append((xq.dtype, y.dtype))
        y.fill_(1.0)

    def stage2(xq, wq, y, *args, **kwargs):
        seen.append((xq.dtype, y.dtype))
        y.fill_(1.0)

    monkeypatch.setattr(aiter, "moe_cktile2stages_gemm1", stage1)
    monkeypatch.setattr(aiter, "moe_cktile2stages_gemm2", stage2)
    return seen


def _fp4(*shape):
    return torch.empty(shape, dtype=dtypes.fp4x2, device="cpu")


def test_stage1_fp16_runs_the_bf16_kernel(launches):
    out = cktile_moe_stage1(
        torch.empty(4, MODEL_DIM, dtype=dtypes.fp16),
        _fp4(E, INTER_DIM * 2, MODEL_DIM // 2),
        _fp4(E, MODEL_DIM, INTER_DIM // 2),
        None,
        None,
        None,
        None,
        TOPK,
        16,
        None,
        None,
        dtype=dtypes.fp16,
    )
    assert launches == [(dtypes.bf16, dtypes.bf16)]
    assert out.dtype == dtypes.fp16 and bool((out == 1).all())


def test_stage2_fp16_runs_the_bf16_kernel(launches):
    out = torch.zeros(4, MODEL_DIM, dtype=dtypes.fp16)
    ret = cktile_moe_stage2(
        torch.empty(4, TOPK, INTER_DIM, dtype=dtypes.fp16),
        _fp4(E, INTER_DIM * 2, MODEL_DIM // 2),
        _fp4(E, MODEL_DIM, INTER_DIM // 2),
        None,
        None,
        None,
        out,
        TOPK,
        None,
        None,
        16,
    )
    assert launches == [(dtypes.bf16, dtypes.bf16)]
    assert ret is out and bool((out == 1).all())


def test_other_output_dtypes_are_rejected(launches):
    with pytest.raises(NotImplementedError, match="bf16 instances only"):
        cktile_moe_stage2(
            torch.empty(4, TOPK, INTER_DIM // 2, dtype=torch.uint8),
            _fp4(E, INTER_DIM * 2, MODEL_DIM // 2),
            _fp4(E, MODEL_DIM, INTER_DIM // 2),
            None,
            None,
            None,
            torch.empty(4, MODEL_DIM, dtype=torch.float32),
            TOPK,
            None,
            None,
            16,
        )
    assert launches == []


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 a16w4 MoE required",
)
def test_swiglu_a16w4_fp16_matches_bf16():
    torch.manual_seed(0)
    x = torch.randn(32, MODEL_DIM, dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(E, INTER_DIM * 2, MODEL_DIM, dtype=dtypes.bf16, device="cuda")
    w2 = torch.randn(E, MODEL_DIM, INTER_DIM, dtype=dtypes.bf16, device="cuda")
    topk_weights, topk_ids = fused_topk(
        x, torch.randn(32, E, dtype=dtypes.bf16, device="cuda"), TOPK, True
    )
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q, w1_scale = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = shuffle_weight_a16w4(w1_q.view(E, INTER_DIM * 2, MODEL_DIM // 2), 16, True)
    w2_q = shuffle_weight_a16w4(w2_q.view(E, MODEL_DIM, INTER_DIM // 2), 16, False)
    kwargs = {
        "w1_scale": shuffle_scale_a16w4(w1_scale, E, True),
        "w2_scale": shuffle_scale_a16w4(w2_scale, E, False),
        "quant_type": QuantType.per_1x32,
        "activation": ActivationType.Swiglu,
        "gate_mode": GateMode.INTERLEAVE.value,
    }
    out_bf16 = fused_moe(x, w1_q, w2_q, topk_weights, topk_ids, **kwargs)
    out_fp16 = fused_moe(x.half(), w1_q, w2_q, topk_weights, topk_ids, **kwargs)

    assert out_fp16.dtype == dtypes.fp16
    ref, out = out_bf16.double(), out_fp16.double()
    assert ref.abs().max().item() > 0
    logits_diff = 1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum()
    assert logits_diff < 1e-3, f"logits_diff={logits_diff.item():.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
