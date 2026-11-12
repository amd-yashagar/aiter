# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""a16w4 Swiglu on CK-Tile with separated gate/up weights and a padded intermediate dim.

CK-Tile stage 1 skips the last ``n_pad_zeros`` rows of N. That removes the padded tail of
both halves only for the tile-interleaved layout. On the standard ``[gate; up]`` layout
(``swiglu_mxfp4_bf16_cktile``) it cut real up rows, so an a16w4 call with
``intermediate_pad`` > 0 and separated weights returned a wrong result (logits_diff
0.02 to 0.03) with no error.
"""

import functools

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

_ARCH = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"), reason="CDNA MXFP4 MoE dispatch required"
)
_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 MXFP4 MoE required",
)
LIMIT = 7.0


def _stage1_n_pad(interleave, intermediate_pad):
    get_2stage_cfgs.cache_clear()
    meta = get_2stage_cfgs(
        16,
        1024,
        512,
        8,
        2,
        dtypes.bf16,
        dtypes.bf16,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Swiglu,
        False,
        0,
        intermediate_pad,
        is_shuffled=True,
        gate_mode=GateMode.INTERLEAVE if interleave else GateMode.SEPARATED,
        gate_up_interleave=interleave,
        opus_weights_shuffled=True,
    )
    assert isinstance(meta.stage1, functools.partial)
    assert meta.stage1.func is fused_moe_module.cktile_moe_stage1
    return meta.stage1.keywords["n_pad_zeros"]


@_ARCH
@pytest.mark.parametrize("intermediate_pad", [64, 128, 192])
def test_separated_layout_computes_the_padded_rows(intermediate_pad):
    assert _stage1_n_pad(False, intermediate_pad) == 0


@_ARCH
def test_interleaved_layout_keeps_skipping_the_padded_tail():
    assert _stage1_n_pad(True, 128) == 256


def _run(interleave, intermediate_pad, bias, token=16, D=1024, I=512, E=8, topk=2):
    torch.manual_seed(0)
    dev, dtype = "cuda", dtypes.bf16
    x = torch.randn(token, D, dtype=dtype, device=dev)
    w1 = torch.randn(E, 2 * I, D, dtype=dtype, device=dev) * 0.5
    w2 = torch.randn(E, D, I, dtype=dtype, device=dev) * 0.5
    if intermediate_pad:
        w1[:, I - intermediate_pad : I, :] = 0
        w1[:, -intermediate_pad:, :] = 0
        w2[:, :, -intermediate_pad:] = 0
    topk_weights, topk_ids = fused_topk(
        x, torch.randn(token, E, dtype=dtype, device=dev), topk, True
    )
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q, w1_s = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_s = quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(E, 2 * I, D // 2)
    w2_q = w2_q.view(E, D, I // 2)
    b1 = b2 = None
    if bias:
        b1 = torch.randn(E, 2 * I, dtype=torch.float32, device=dev).clamp(-2, 2)
        b2 = torch.randn(E, D, dtype=torch.float32, device=dev).clamp(-2, 2)
        if intermediate_pad:
            b1[:, I - intermediate_pad : I] = 0
            b1[:, -intermediate_pad:] = 0
    ref1 = fused_moe_module.torch_moe_stage1(
        x,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_s,
        w1_bias=b1,
        swiglu_limit=LIMIT,
    )
    ref = fused_moe_module.torch_moe_stage2(
        ref1.view(token, topk, -1),
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_s,
        a2_scale=None,
        w2_bias=b2,
    )
    get_2stage_cfgs.cache_clear()
    out = fused_moe(
        x,
        shuffle_weight_a16w4(w1_q.view(dtypes.fp4x2), 16, interleave),
        shuffle_weight_a16w4(w2_q.view(dtypes.fp4x2), 16, False),
        topk_weights,
        topk_ids,
        w1_scale=shuffle_scale_a16w4(w1_s.view(-1, D // 32), E, interleave),
        w2_scale=shuffle_scale_a16w4(w2_s.view(-1, I // 32), E, False),
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Swiglu,
        gate_mode=(GateMode.INTERLEAVE if interleave else GateMode.SEPARATED).value,
        swiglu_limit=LIMIT,
        # vLLM passes the bias as [gate; up] (de-interleaved) next to interleaved weights.
        bias1=b1,
        bias2=b2,
        intermediate_pad=intermediate_pad,
    )
    return ref.double(), out.double()


def _logits_diff(ref, out):
    return float(1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum())


@_GPU
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("intermediate_pad", [0, 64, 128])
def test_separated_a16w4_with_intermediate_pad_matches_the_reference(
    intermediate_pad, bias
):
    ref, out = _run(False, intermediate_pad, bias)
    assert torch.isfinite(out).all()
    assert _logits_diff(ref, out) < 1e-3


@_GPU
@pytest.mark.parametrize("intermediate_pad", [0, 128])
def test_interleaved_a16w4_with_intermediate_pad_still_matches_the_reference(
    intermediate_pad,
):
    ref, out = _run(True, intermediate_pad, True)
    assert torch.isfinite(out).all()
    assert _logits_diff(ref, out) < 1e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
