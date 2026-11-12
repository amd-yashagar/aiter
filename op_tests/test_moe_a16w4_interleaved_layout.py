# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""a16w4 (bf16 A x MXFP4 W) Swiglu with gate/up interleaved weights must not reach FlyDSL.

The FlyDSL a16w4 stage-1 kernels read separated gate/up rows. Weights tagged
``gate_up_interleave=True`` (the fp8-activation layout, which fused_moe also runs with
bf16 activations below ``AITER_BF16_FP8_MOE_BOUND`` tokens) were accepted by the layout
check, so both the untuned reroute (inter_dim not a multiple of 256) and the tuned
``flydsl_moe1_abf16_wfp4`` rows computed uncorrelated output without an error
(logits_diff about 0.97).
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


def _kernel(fn):
    target = fn.func if isinstance(fn, functools.partial) else fn
    name = getattr(target, "__name__", str(target))
    if isinstance(fn, functools.partial) and fn.keywords.get("kernelName"):
        return fn.keywords["kernelName"]
    return name


def _dispatch(token, model_dim, inter_dim, expert, topk, gate_mode, tag):
    get_2stage_cfgs.cache_clear()
    return get_2stage_cfgs(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtypes.bf16,
        dtypes.bf16,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Swiglu,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode=gate_mode,
        gate_up_interleave=tag,
        opus_weights_shuffled=True,
    )


@pytest.mark.parametrize(
    ("q_dtype_a", "gate_mode", "tag", "want"),
    [
        (dtypes.bf16, GateMode.INTERLEAVE, True, False),
        (dtypes.bf16, GateMode.SEPARATED, False, True),
        (dtypes.bf16, GateMode.SEPARATED, None, True),
        (dtypes.bf16, GateMode.INTERLEAVE, None, False),
        (dtypes.fp8, GateMode.INTERLEAVE, True, True),
        (dtypes.fp8, GateMode.INTERLEAVE, None, True),
        (dtypes.fp4x2, GateMode.INTERLEAVE, True, False),
    ],
)
def test_layout_check(q_dtype_a, gate_mode, tag, want):
    assert (
        fused_moe_module._flydsl_mxfp4_layout_is_compatible(q_dtype_a, gate_mode, tag)
        is want
    )


@_ARCH
@pytest.mark.parametrize("inter_dim", [384, 640])
def test_untuned_interleaved_weights_do_not_take_the_a16w4_reroute(inter_dim):
    meta = _dispatch(32, 1024, inter_dim, 8, 2, GateMode.INTERLEAVE, True)
    assert not _kernel(meta.stage1).startswith("flydsl")
    assert not _kernel(meta.stage2).startswith("flydsl")


@_ARCH
@pytest.mark.parametrize("inter_dim", [384, 640])
def test_untuned_separated_weights_keep_the_a16w4_reroute(inter_dim):
    meta = _dispatch(32, 1024, inter_dim, 8, 2, GateMode.SEPARATED, False)
    assert _kernel(meta.stage1).startswith("flydsl_moe1_abf16_wfp4")


@_ARCH
@pytest.mark.parametrize("token", [1, 16, 64, 256])
def test_tuned_a16w4_flydsl_rows_are_skipped_for_interleaved_weights(token):
    sep = _dispatch(token, 6144, 768, 128, 4, GateMode.SEPARATED, False)
    if not _kernel(sep.stage1).startswith("flydsl_moe1_abf16_wfp4"):
        pytest.skip("no tuned FlyDSL a16w4 row for this device")
    meta = _dispatch(token, 6144, 768, 128, 4, GateMode.INTERLEAVE, True)
    assert not _kernel(meta.stage1).startswith("flydsl")


def _reference_and_output(inter_dim, num_experts, topk, tag):
    torch.manual_seed(0)
    dev, dtype, token, model_dim = "cuda", dtypes.bf16, 64, 1024
    x = torch.randn(token, model_dim, dtype=dtype, device=dev)
    w1 = torch.randn(num_experts, inter_dim * 2, model_dim, dtype=dtype, device=dev)
    w2 = torch.randn(num_experts, model_dim, inter_dim, dtype=dtype, device=dev)
    topk_weights, topk_ids = fused_topk(
        x, torch.randn(token, num_experts, dtype=dtype, device=dev), topk, True
    )
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q, w1_scale = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(num_experts, inter_dim * 2, model_dim // 2)
    w2_q = w2_q.view(num_experts, model_dim, inter_dim // 2)
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
        w1_scale=w1_scale,
    )
    ref = fused_moe_module.torch_moe_stage2(
        ref1.view(token, topk, -1),
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=None,
    )
    get_2stage_cfgs.cache_clear()
    out = fused_moe(
        x,
        shuffle_weight_a16w4(w1_q, 16, tag),
        shuffle_weight_a16w4(w2_q, 16, False),
        topk_weights,
        topk_ids,
        w1_scale=shuffle_scale_a16w4(w1_scale, num_experts, tag),
        w2_scale=shuffle_scale_a16w4(w2_scale, num_experts, False),
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Swiglu,
        gate_mode=(GateMode.INTERLEAVE if tag else GateMode.SEPARATED).value,
    )
    return ref.double(), out.double()


@_GPU
@pytest.mark.parametrize(("num_experts", "topk"), [(1, 1), (8, 2)])
@pytest.mark.parametrize("inter_dim", [256, 384, 512, 640])
def test_interleaved_a16w4_swiglu_is_correct_or_refused(inter_dim, num_experts, topk):
    try:
        ref, out = _reference_and_output(inter_dim, num_experts, topk, True)
    except NotImplementedError:
        assert inter_dim % 256 != 0
        return
    assert torch.isfinite(out).all()
    logits_diff = 1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum()
    assert logits_diff < 1e-3, f"logits_diff={logits_diff.item():.3e}"


@_GPU
@pytest.mark.parametrize("inter_dim", [384, 640])
def test_separated_a16w4_swiglu_still_runs_on_flydsl_and_is_correct(inter_dim):
    ref, out = _reference_and_output(inter_dim, 8, 2, False)
    logits_diff = 1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum()
    assert logits_diff < 1e-3, f"logits_diff={logits_diff.item():.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
