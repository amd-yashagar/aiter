# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression: a16wi4 (bf16 A x int4 W) fused_moe with fp16 hidden states.

The a16w-mix FlyDSL kernels read bf16 activations and store bf16 whatever
dtype is requested, so an fp16 call used to return garbage (cos diff about
0.997 against the bf16 call). The fp16 result must match the bf16 one."""

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import per_1x32_i4_quant
from aiter.ops.shuffle import (
    pack_int8_to_packed_int4,
    shuffle_scale_for_int4,
    shuffle_weight,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in ("gfx942", "gfx950"),
    reason="Needs a gfx942 or gfx950 GPU",
)

TOKEN, MODEL_DIM, INTER_DIM, E, TOPK = 32, 512, 256, 16, 4


def _prep(weight):
    quantized, scale = per_1x32_i4_quant(weight)
    shuffled = pack_int8_to_packed_int4(
        shuffle_weight(quantized.view(dtypes.i8), (16, 16))
    )
    shuffled = shuffled.view(E, weight.shape[1], weight.shape[2] // 2)
    return (
        shuffled.view(dtypes.i4x2),
        shuffle_scale_for_int4(scale, group_size=32).view(-1).contiguous(),
    )


def test_a16wi4_fp16_output_matches_bf16():
    torch.manual_seed(0)
    w1, w1_scale = _prep(
        torch.randn((E, INTER_DIM * 2, MODEL_DIM), dtype=dtypes.bf16, device="cuda")
    )
    w2, w2_scale = _prep(
        torch.randn((E, MODEL_DIM, INTER_DIM), dtype=dtypes.bf16, device="cuda")
    )
    hidden = torch.randn((TOKEN, MODEL_DIM), dtype=dtypes.bf16, device="cuda")
    score = torch.randn((TOKEN, E), dtype=dtypes.bf16, device="cuda")

    outs = {}
    for dtype in (dtypes.bf16, dtypes.fp16):
        topk_weights, topk_ids = fused_topk(
            hidden.to(dtype), score.to(dtype), TOPK, True
        )
        outs[dtype] = fused_moe(
            hidden.to(dtype),
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Silu,
        )

    assert outs[dtypes.fp16].dtype == dtypes.fp16
    ref, out = outs[dtypes.bf16].double(), outs[dtypes.fp16].double()
    assert torch.isfinite(out).all()
    diff = 1 - 2 * (out * ref).sum() / (out * out + ref * ref).sum()
    assert diff < 1e-3, f"{diff=}, mean|out|={out.abs().mean():.6f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
