# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression: CK fp8 2-stage MoE stage2 with inter_dim <= 192 (KPerBlock = 64)
must match the dequantized reference.

The 256-thread KPerBlock = 64 fp8 instances used AK1 = BK1 = 8, which does not
match the 16-byte-per-lane preshuffled B layout, so stage2 returned garbage
(logits diff about 0.93)."""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, torch_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="Needs a ROCm GPU",
)

MODEL_DIM, E, TOPK = 1024, 8, 2


@pytest.mark.parametrize(
    "quant_type, inter_dim, token, block_size_M",
    [
        (aiter.QuantType.per_Tensor, 64, 8, None),
        (aiter.QuantType.per_Tensor, 128, 8, None),
        (aiter.QuantType.per_Tensor, 192, 8, None),
        (aiter.QuantType.per_Tensor, 192, 256, 64),
        (aiter.QuantType.per_Tensor, 192, 512, 128),
        (aiter.QuantType.per_Tensor, 192, 1024, 256),
        (aiter.QuantType.per_Token, 128, 8, None),
    ],
)
def test_ck_fp8_stage2_small_inter_dim(quant_type, inter_dim, token, block_size_M):
    if get_gfx() == "gfx950" and inter_dim % 128 != 0:
        pytest.skip("gfx950 a8w8 stage2 requires inter_dim to be a multiple of 128")
    torch.manual_seed(0)
    hidden = torch.randn((token, MODEL_DIM), dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn((E, inter_dim * 2, MODEL_DIM), device="cuda") / 10
    w2 = torch.randn((E, MODEL_DIM, inter_dim), device="cuda") / 10
    score = torch.randn((token, E), device="cuda")
    topk_weights, topk_ids = torch.softmax(score, dim=-1).topk(TOPK, dim=-1)
    topk_ids = topk_ids.to(dtypes.i32)

    if quant_type == aiter.QuantType.per_Tensor:
        w1_qt, w1_scale = aiter.pertoken_quant(w1.view(E, -1), quant_dtype=dtypes.fp8)
        w2_qt, w2_scale = aiter.pertoken_quant(w2.view(E, -1), quant_dtype=dtypes.fp8)
    else:
        w1_qt, w1_scale = aiter.pertoken_quant(w1, quant_dtype=dtypes.fp8)
        w2_qt, w2_scale = aiter.pertoken_quant(w2, quant_dtype=dtypes.fp8)
    w1_qt = w1_qt.view(w1.shape)
    w2_qt = w2_qt.view(w2.shape)

    ref = torch_moe(
        hidden,
        w1_qt.float() * w1_scale.view(E, -1, 1),
        w2_qt.float() * w2_scale.view(E, -1, 1),
        topk_weights,
        topk_ids,
    )

    out = fused_moe(
        hidden,
        shuffle_weight(w1_qt, layout=(16, 16)),
        shuffle_weight(w2_qt, layout=(16, 16)),
        topk_weights,
        topk_ids,
        quant_type=quant_type,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_size_M=block_size_M,
    )

    assert torch.isfinite(out).all()
    out, ref = out.double(), ref.double()
    diff = 1 - 2 * (out * ref).sum() / (out * out + ref * ref).sum()
    assert diff < 1e-2, f"{diff=}, mean|out|={out.abs().mean():.6f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
