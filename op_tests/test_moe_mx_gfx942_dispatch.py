# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression: MXFP4/MXFP8 MoE dispatch on gfx942.

gfx942 has no kernel for a4w4, a8w4 or mxfp8: the heuristic FlyDSL MX route
used to be picked anyway and aborted the process (LLVM ERROR), and a16w4
Swiglu at M >= 256 switched its activations to fp4 and took the same route.
Dispatch must keep a16w4 on bf16 activations and refuse the rest cleanly."""

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, resolve_activation_dtype
from aiter.jit.utils.chip_info import get_gfx


@pytest.mark.parametrize("token", [1, 255, 256, 4096])
def test_a16w4_swiglu_keeps_bf16_activations_on_gfx942(token):
    q_dtype_a = resolve_activation_dtype(
        QuantType.per_1x32,
        dtypes.fp4x2,
        activation=ActivationType.Swiglu,
        M=token,
        gfx="gfx942",
    )
    assert q_dtype_a == dtypes.bf16


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx942",
    reason="Needs a gfx942 GPU",
)
@pytest.mark.parametrize(
    "weight_dtype, quant_dtype_a",
    [
        (dtypes.fp4x2, dtypes.fp4x2),
        (dtypes.fp4x2, dtypes.fp8),
        (dtypes.fp8, dtypes.fp8),
    ],
    ids=["a4w4", "a8w4", "mxfp8"],
)
def test_mx_quantized_activations_are_refused_on_gfx942(weight_dtype, quant_dtype_a):
    token, model_dim, inter_dim, E, topk = 4, 256, 128, 8, 2
    packed = 2 if weight_dtype == dtypes.fp4x2 else 1
    w1 = torch.zeros(
        (E, inter_dim * 2, model_dim // packed), dtype=torch.uint8, device="cuda"
    ).view(weight_dtype)
    w2 = torch.zeros(
        (E, model_dim, inter_dim // packed), dtype=torch.uint8, device="cuda"
    ).view(weight_dtype)
    w1.is_shuffled = w2.is_shuffled = True
    hidden = torch.randn((token, model_dim), dtype=dtypes.bf16, device="cuda")
    topk_weights = torch.ones((token, topk), dtype=dtypes.fp32, device="cuda")
    topk_ids = torch.zeros((token, topk), dtype=dtypes.i32, device="cuda")
    with pytest.raises(NotImplementedError, match="gfx942"):
        fused_moe(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Silu,
            quant_dtype_a=quant_dtype_a,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
