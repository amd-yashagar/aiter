# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Situv2 a16w4 (bf16 A x MXFP4 W) with interleaved gate/up weights must be refused.

Under the default environment Situv2 resolves to bf16 activations, which run on the
FlyDSL a16w4 kernels. Those read separated gate/up rows. Weights tagged
``gate_up_interleave=True`` (the a8w4 layout) were not checked on this path, so the
call, tuned row or heuristic, returned uncorrelated output (logits_diff about 0.98)
without an error.
"""

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in ("gfx942", "gfx950"),
    reason="CDNA MXFP4 MoE required",
)

MODEL_DIM, INTER_DIM, E, TOPK = 1024, 384, 8, 2


def _call(token, interleave, gate_mode=None, seed=0):
    torch.manual_seed(seed)
    dev, dtype = "cuda", dtypes.bf16
    x = torch.randn(token, MODEL_DIM, dtype=dtype, device=dev)
    w1 = torch.randn(E, INTER_DIM * 2, MODEL_DIM, dtype=dtype, device=dev)
    w2 = torch.randn(E, MODEL_DIM, INTER_DIM, dtype=dtype, device=dev)
    topk_weights, topk_ids = fused_topk(
        x, torch.randn(token, E, dtype=dtype, device=dev), TOPK, True
    )
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q, w1_scale = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(E, INTER_DIM * 2, MODEL_DIM // 2)
    w2_q = w2_q.view(E, MODEL_DIM, INTER_DIM // 2)
    ref1 = fused_moe_module.torch_moe_stage1(
        x,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_scale,
        situ_beta=DEFAULT_SITUV2_BETA,
        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
    )
    ref = fused_moe_module.torch_moe_stage2(
        ref1.view(token, TOPK, -1),
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=None,
    )
    if gate_mode is None:
        gate_mode = GateMode.INTERLEAVE if interleave else GateMode.SEPARATED
    fused_moe_module.get_2stage_cfgs.cache_clear()
    out = fused_moe(
        x,
        shuffle_weight_a16w4(w1_q, 16, interleave),
        shuffle_weight_a16w4(w2_q, 16, False),
        topk_weights,
        topk_ids,
        w1_scale=shuffle_scale_a16w4(w1_scale, E, interleave),
        w2_scale=shuffle_scale_a16w4(w2_scale, E, False),
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Situv2,
        gate_mode=gate_mode.value,
    )
    return ref.double(), out.double()


@pytest.mark.parametrize("token", [16, 4096])
def test_interleaved_weights_are_refused(token):
    with pytest.raises(NotImplementedError, match="interleaved gate/up weights"):
        _call(token, interleave=True)


def test_interleaved_gate_mode_with_separated_tag_is_refused():
    with pytest.raises(NotImplementedError, match="interleaved gate/up weights"):
        _call(16, interleave=False, gate_mode=GateMode.INTERLEAVE)


@pytest.mark.parametrize("token", [16, 4096])
def test_separated_weights_still_run_and_match_the_reference(token):
    ref, out = _call(token, interleave=False)
    assert torch.isfinite(out).all()
    logits_diff = 1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum()
    assert logits_diff < 1e-3, f"logits_diff={logits_diff.item():.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
