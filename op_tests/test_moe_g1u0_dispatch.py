# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression: non-gated (g1u0) MoE weights must never run on a 2-stage kernel.

Stage 1 of the CK and asm 2-stage paths computes gate and up together (it halves
the weight's N). With g1u0 weights it returned NaN or garbage (logits diff 1.0)
at every M above the 1-stage bucket, and also when forced at M = 16. g1u0 now
runs on the 1-stage kernel at every M where one exists (bf16/fp16 no-quant,
int8 per-token on gfx942) and is refused everywhere else."""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, torch_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in ("gfx942", "gfx950"),
    reason="Needs a gfx942 or gfx950 GPU",
)

E, MODEL_DIM, INTER_DIM, TOPK = 32, 1024, 512, 4


def _inputs(token, dtype):
    torch.manual_seed(0)
    x = torch.randn(token, MODEL_DIM, dtype=dtype, device="cuda")
    w1 = torch.randn(E, INTER_DIM, MODEL_DIM, dtype=dtype, device="cuda") * 0.1
    w2 = torch.randn(E, MODEL_DIM, INTER_DIM, dtype=dtype, device="cuda") * 0.1
    topk_weights, topk_ids = fused_topk(
        x, torch.randn(token, E, dtype=dtype, device="cuda"), TOPK, True
    )
    return x, w1, w2, topk_weights, topk_ids


def _logits_diff(a, b):
    a, b = a.double(), b.double()
    return float(1 - 2 * (a * b).sum() / (a * a + b * b).sum())


@pytest.mark.parametrize("dtype", [dtypes.bf16, dtypes.fp16])
@pytest.mark.parametrize("token", [16, 200, 4096])
def test_g1u0_no_quant_is_correct_or_refused(token, dtype):
    x, w1, w2, topk_weights, topk_ids = _inputs(token, dtype)
    ref = torch_moe(
        x,
        w1.float(),
        w2.float(),
        topk_weights,
        topk_ids,
        activation=ActivationType.Silu,
    )
    try:
        out = fused_moe(
            x,
            shuffle_weight(w1, layout=(16, 16)),
            shuffle_weight(w2, layout=(16, 16)),
            topk_weights,
            topk_ids,
            quant_type=QuantType.No,
            activation=ActivationType.Silu,
        )
    except NotImplementedError:
        assert get_gfx() != "gfx942", "gfx942 has a 1-stage g1u0 kernel"
        return
    assert torch.isfinite(out).all()
    assert _logits_diff(ref, out) < 1e-2


@pytest.mark.parametrize("token", [16, 200, 4096])
def test_g1u0_fp8_per_tensor_is_refused(token):
    x, w1, w2, topk_weights, topk_ids = _inputs(token, dtypes.bf16)
    w1_q, w1_scale = aiter.pertoken_quant(w1.view(E, -1), quant_dtype=dtypes.fp8)
    w2_q, w2_scale = aiter.pertoken_quant(w2.view(E, -1), quant_dtype=dtypes.fp8)
    with pytest.raises(NotImplementedError, match="g1u0"):
        fused_moe(
            x,
            shuffle_weight(w1_q.view(w1.shape), layout=(16, 16)),
            shuffle_weight(w2_q.view(w2.shape), layout=(16, 16)),
            topk_weights,
            topk_ids,
            quant_type=QuantType.per_Tensor,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )


@pytest.mark.parametrize("token", [16, 200, 4096])
def test_g1u0_int8_per_token_is_refused(token):
    """fmoe_int8_g1u0 needs smooth-quant scales that fused_moe cannot pass."""
    x, w1, w2, topk_weights, topk_ids = _inputs(token, dtypes.bf16)
    w1_q, w1_scale = aiter.pertoken_quant(w1, quant_dtype=dtypes.i8)
    w2_q, w2_scale = aiter.pertoken_quant(w2, quant_dtype=dtypes.i8)
    with pytest.raises(NotImplementedError, match="g1u0"):
        fused_moe(
            x,
            shuffle_weight(w1_q, layout=(16, 16)),
            shuffle_weight(w2_q, layout=(16, 16)),
            topk_weights,
            topk_ids,
            quant_type=QuantType.per_Token,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
