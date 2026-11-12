# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A quantized ``quant_type`` on bf16/fp16 weights must raise, not abort the process.

``quant_type=per_Token`` with bf16 weights reached ``dynamic_per_token_scaled_quant`` with a bf16
output type, whose AITER_CHECK aborts the process (SIGABRT, rc 134). Small M took a 1-stage ASM
branch and ended in a TypeError instead. ``fused_moe`` now validates the pair first.
"""

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    _validate_quant_type_vs_weight_dtype,
    fused_moe,
    fused_topk,
)
from aiter.jit.utils.chip_info import get_gfx

QUANTIZED = [
    QuantType.per_Tensor,
    QuantType.per_Token,
    QuantType.per_1x128,
    QuantType.per_1x32,
]


@pytest.mark.parametrize("quant_type", QUANTIZED)
@pytest.mark.parametrize("weight_dtype", [dtypes.bf16, dtypes.fp16, dtypes.fp32])
def test_quantized_quant_type_on_unquantized_weights_is_refused(
    quant_type, weight_dtype
):
    with pytest.raises(ValueError, match="needs quantized expert weights"):
        _validate_quant_type_vs_weight_dtype(quant_type, weight_dtype)


@pytest.mark.parametrize(
    ("quant_type", "weight_dtype"),
    [
        (QuantType.No, dtypes.bf16),
        (QuantType.No, dtypes.fp16),
        (QuantType.per_Token, dtypes.fp8),
        (QuantType.per_Token, dtypes.i8),
        (QuantType.per_Token, torch.uint32),
        (QuantType.per_Tensor, dtypes.fp8),
        (QuantType.per_1x128, dtypes.fp8),
        (QuantType.per_1x32, dtypes.fp4x2),
        (QuantType.per_1x32, dtypes.i4x2),
        (QuantType.per_1x32, dtypes.fp8),
    ],
)
def test_valid_pairs_are_accepted(quant_type, weight_dtype):
    _validate_quant_type_vs_weight_dtype(quant_type, weight_dtype)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in ("gfx942", "gfx950"),
    reason="CDNA GPU required",
)
@pytest.mark.parametrize("token", [16, 512])
@pytest.mark.parametrize("quant_type", [QuantType.per_Token, QuantType.per_Tensor])
def test_fused_moe_raises_instead_of_aborting(quant_type, token):
    dev, dtype, D, I, E, topk = "cuda", dtypes.bf16, 1024, 256, 8, 2
    x = torch.randn(token, D, dtype=dtype, device=dev)
    w1 = torch.randn(E, 2 * I, D, dtype=dtype, device=dev) * 0.05
    w2 = torch.randn(E, D, I, dtype=dtype, device=dev) * 0.05
    tw, ti = fused_topk(x, torch.randn(token, E, dtype=dtype, device=dev), topk, True)
    with pytest.raises(ValueError, match="needs quantized expert weights"):
        fused_moe(
            x,
            w1,
            w2,
            tw,
            ti,
            quant_type=quant_type,
            activation=ActivationType.Silu,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
