# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``aiter.fused_moe_bf16_asm.ck_moe_2stages`` must call the current CK stage entry points.

It still passed the old positional arguments (``fc1_scale, a1_scale, block_size, weights,
act_op``) to ``aiter.ck_moe_stage1/2``, which now take ``kernelName`` right after ``topk``,
so the call failed with "Expected a value of type 'Optional[Tensor]' for argument
'a1_scale' but instead found type 'int'" (and ``act_op`` had Silu/Gelu swapped).
"""

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes, get_torch_quant
from aiter.fused_moe import fused_moe, fused_topk, torch_moe
from aiter.fused_moe_bf16_asm import ck_moe_2stages
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or get_gfx() not in ("gfx942", "gfx950"),
    reason="Needs gfx942 or gfx950",
)

M, D, I, E, TOPK = 16, 512, 256, 8, 2


def logits_diff(ref, out):
    r, o = ref.double(), out.double()
    return (1 - 2 * (r * o).sum() / (r * r + o * o).sum()).item()


def _inputs():
    torch.manual_seed(0)
    x = torch.randn(M, D, dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(E, 2 * I, D, dtype=dtypes.bf16, device="cuda") * 0.05
    w2 = torch.randn(E, D, I, dtype=dtypes.bf16, device="cuda") * 0.05
    weights, ids = fused_topk(
        x, torch.randn(M, E, dtype=dtypes.bf16, device="cuda"), TOPK, True
    )
    return x, w1, w2, weights, ids


@pytest.mark.parametrize("activation", [ActivationType.Silu, ActivationType.Gelu])
def test_bf16_matches_the_reference(activation):
    x, w1, w2, weights, ids = _inputs()
    out = ck_moe_2stages(
        x,
        shuffle_weight(w1, layout=(16, 16)),
        shuffle_weight(w2, layout=(16, 16)),
        weights,
        ids,
        quant_type=QuantType.No,
        activation=activation,
    )
    ref = torch_moe(x, w1, w2, weights, ids, activation=activation)
    assert logits_diff(ref.float(), out.float()) < 1e-3


def test_fp8_per_tensor_matches_fused_moe():
    x, w1, w2, weights, ids = _inputs()
    quant = get_torch_quant(QuantType.per_Tensor)
    w1q, w1s = quant(w1.view(E, -1), quant_dtype=dtypes.fp8)
    w2q, w2s = quant(w2.view(E, -1), quant_dtype=dtypes.fp8)
    W1 = shuffle_weight(w1q.view(E, 2 * I, D), layout=(16, 16))
    W2 = shuffle_weight(w2q.view(E, D, I), layout=(16, 16))
    args = (x, W1, W2, weights, ids)
    out = ck_moe_2stages(
        *args, quant_type=QuantType.per_Tensor, fc1_scale=w1s, fc2_scale=w2s
    )
    ref = fused_moe(
        *args,
        quant_type=QuantType.per_Tensor,
        activation=ActivationType.Silu,
        w1_scale=w1s,
        w2_scale=w2s,
    )
    assert logits_diff(ref.float(), out.float()) < 1e-3
