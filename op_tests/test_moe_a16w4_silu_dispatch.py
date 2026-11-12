# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Silu (non-Swiglu) MoE with gate/up-interleaved MXFP4 weights below the fp8 bound.

``resolve_activation_dtype`` returns bf16 activations below
``AITER_BF16_FP8_MOE_BOUND`` tokens for any activation with interleaved MXFP4
weights, but only Swiglu and SiTUv2 have a16w4 kernels. Silu used to fall to the
CK 2-stage heuristic, which has no such instance and failed at launch with
``RuntimeError: Unsupported kernel config for moe heuristic dispatch``. Dispatch
now takes CK-Tile split-k (bf16 output), and raises ``NotImplementedError``
where no backend exists (stage1 bias with Silu, fp16 output, or non-interleaved
weights).
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

TOKEN = 32
MODEL_DIM = 512
INTER_DIM = 256
E = 8
TOPK = 2


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(fused_moe_module, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "get_cu_num", lambda: 256)
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def _dispatch(
    activation=ActivationType.Silu,
    gate_mode=GateMode.INTERLEAVE,
    q_dtype_a=dtypes.bf16,
    has_bias=False,
    dtype=dtypes.bf16,
):
    get_2stage_cfgs.cache_clear()
    return get_2stage_cfgs(
        TOKEN,
        MODEL_DIM,
        INTER_DIM,
        E,
        TOPK,
        dtype,
        q_dtype_a,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        activation,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode=gate_mode.value,
        has_stage1_bias=has_bias,
        has_stage2_bias=has_bias,
        opus_weights_shuffled=True,
    )


def _stage1_name(meta):
    fn = meta.stage1
    return (fn.func if isinstance(fn, functools.partial) else fn).__name__


def test_silu_interleaved_a16w4_uses_cktile_split_k(gfx950):
    meta = _dispatch()
    assert _stage1_name(meta) == "cktile_moe_stage1"
    assert meta.stage1.keywords["split_k"] >= 2


def test_silu_interleaved_a16w4_with_stage1_bias_is_not_implemented(gfx950):
    with pytest.raises(NotImplementedError, match="bias"):
        _dispatch(has_bias=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gate_mode": GateMode.SEPARATED},
        {"activation": ActivationType.Gelu},
        {"dtype": dtypes.fp16},
    ],
)
def test_a16w4_without_any_backend_is_not_implemented(gfx950, kwargs):
    with pytest.raises(NotImplementedError, match="No backend"):
        _dispatch(**kwargs)


@pytest.mark.parametrize(
    ("activation", "q_dtype_a"),
    [
        (ActivationType.Silu, dtypes.fp8),
        (ActivationType.Swiglu, dtypes.bf16),
    ],
)
def test_supported_a8w4_and_swiglu_a16w4_dispatch_is_unchanged(
    gfx950, activation, q_dtype_a
):
    meta = _dispatch(activation=activation, q_dtype_a=q_dtype_a)
    assert "ck_moe_stage1" != _stage1_name(meta)


def _build(dtype, has_bias):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(TOKEN, MODEL_DIM, dtype=dtype, device=dev)
    w1 = torch.randn(E, INTER_DIM * 2, MODEL_DIM, dtype=dtype, device=dev)
    w2 = torch.randn(E, MODEL_DIM, INTER_DIM, dtype=dtype, device=dev)
    bias1 = torch.clamp(torch.randn(E, INTER_DIM * 2, device=dev), -1, 1)
    topk_weights, topk_ids = fused_topk(
        x, torch.randn(TOKEN, E, dtype=dtype, device=dev), TOPK, True
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
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w1_bias=bias1.to(dtype) if has_bias else None,
    )
    ref = fused_moe_module.torch_moe_stage2(
        ref1.view(TOKEN, TOPK, -1),
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
    )
    args = (
        x,
        shuffle_weight_a16w4(w1_q, 16, True),
        shuffle_weight_a16w4(w2_q, 16, False),
        topk_weights,
        topk_ids,
    )
    kwargs = {
        "w1_scale": shuffle_scale_a16w4(w1_scale, E, True),
        "w2_scale": shuffle_scale_a16w4(w2_scale, E, False),
        "quant_type": QuantType.per_1x32,
        "activation": ActivationType.Silu,
        "gate_mode": GateMode.INTERLEAVE.value,
        "bias1": bias1 if has_bias else None,
    }
    return args, kwargs, ref


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 a16w4 MoE required",
)


@_GPU
def test_silu_interleaved_a16w4_matches_reference():
    args, kwargs, ref = _build(dtypes.bf16, has_bias=False)
    out = fused_moe(*args, **kwargs)

    assert not out.isnan().any().item()
    ref, out = ref.double(), out.double()
    logits_diff = 1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum()
    assert logits_diff < 1e-3, f"logits_diff={logits_diff.item():.3e}"


@_GPU
def test_silu_interleaved_a16w4_with_bias_raises_not_implemented():
    args, kwargs, _ = _build(dtypes.bf16, has_bias=True)
    with pytest.raises(NotImplementedError, match="bias"):
        fused_moe(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
