# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""a4w4 (fp4 A x MXFP4 W) with interleaved gate/up weights.

The FlyDSL ``flydsl_moe1_afp4_wfp4`` stage-1 kernels read separated gate/up rows, the CK,
CK-Tile and ASM kernels never read the interleaved layout, and only the MXMOE kernels
take it. Weights tagged ``gate_up_interleave=True`` were accepted by the FlyDSL layout
check and by tuned ``flydsl_moe1_afp4_wfp4`` rows, and the fallbacks ran the other
backends on them, so the call returned uncorrelated output (logits_diff about 0.98).
It must be refused unless an MXMOE row serves it.
"""

import functools

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

_ARCH = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"), reason="CDNA MXFP4 MoE dispatch required"
)
_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 MXFP4 MoE required",
)


def _kernel(fn):
    if isinstance(fn, functools.partial) and fn.keywords.get("kernelName"):
        return fn.keywords["kernelName"]
    target = fn.func if isinstance(fn, functools.partial) else fn
    return getattr(target, "__name__", str(target))


def _dispatch(token, model_dim, inter_dim, expert, topk, interleave):
    get_2stage_cfgs.cache_clear()
    return get_2stage_cfgs(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Swiglu,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode=GateMode.INTERLEAVE if interleave else GateMode.SEPARATED,
        gate_up_interleave=interleave,
        opus_weights_shuffled=True,
    )


@_ARCH
@pytest.mark.parametrize("token", [16, 64])
def test_tuned_mixed_rows_are_skipped_for_interleaved_weights(token):
    sep = _dispatch(token, 6144, 768, 128, 4, False)
    if not _kernel(sep.stage1).startswith("flydsl_moe1_afp4_wfp4"):
        pytest.skip("no tuned FlyDSL mixed row for this device")
    meta = _dispatch(token, 6144, 768, 128, 4, True)
    assert not _kernel(meta.stage1).startswith("flydsl_moe1_afp4_wfp4")


def _case(act, token, model_dim, inter_dim, experts, topk, interleave, seed=0):
    torch.manual_seed(seed)
    dev, dtype = "cuda", dtypes.bf16
    x = torch.randn(token, model_dim, dtype=dtype, device=dev)
    w1 = torch.randn(experts, inter_dim * 2, model_dim, dtype=dtype, device=dev)
    w2 = torch.randn(experts, model_dim, inter_dim, dtype=dtype, device=dev)
    topk_weights, topk_ids = fused_topk(
        x, torch.randn(token, experts, dtype=dtype, device=dev), topk, True
    )
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q, w1_scale = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(experts, inter_dim * 2, model_dim // 2)
    w2_q = w2_q.view(experts, model_dim, inter_dim // 2)
    ref1 = fused_moe_module.torch_moe_stage1(
        x,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=act,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_scale,
        situ_beta=DEFAULT_SITUV2_BETA,
        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
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
        shuffle_weight_a16w4(w1_q, 16, interleave),
        shuffle_weight_a16w4(w2_q, 16, False),
        topk_weights,
        topk_ids,
        w1_scale=shuffle_scale_a16w4(w1_scale, experts, interleave),
        w2_scale=shuffle_scale_a16w4(w2_scale, experts, False),
        quant_type=QuantType.per_1x32,
        activation=act,
        quant_dtype_a=dtypes.fp4x2,
        gate_mode=(GateMode.INTERLEAVE if interleave else GateMode.SEPARATED).value,
    )
    return ref.double(), out.double()


def _logits_diff(ref, out):
    return float(1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum())


_ACTS = [ActivationType.Situv2, ActivationType.Silu, ActivationType.Swiglu]


@_GPU
@pytest.mark.parametrize("act", _ACTS, ids=lambda a: a.name)
@pytest.mark.parametrize("inter_dim", [256, 384, 512])
@pytest.mark.parametrize("token", [16, 512])
def test_interleaved_a4w4_is_refused(act, inter_dim, token):
    # Swiglu at inter_dim % 256 != 0 is already refused by the CK-Tile stage-2 check,
    # and Situv2 by the CK 2-stage guard once it falls through to CK.
    with pytest.raises(
        NotImplementedError,
        match="a4w4.*interleaved gate/up|No safe MXFP4|Situv2 has no CK 2-stage",
    ):
        _case(act, token, 1024, inter_dim, 8, 2, True)


@_GPU
@pytest.mark.parametrize("token", [16, 64])
def test_interleaved_a4w4_tuned_mixed_shape_is_refused(token):
    with pytest.raises(NotImplementedError, match="a4w4.*interleaved gate/up"):
        _case(ActivationType.Swiglu, token, 6144, 768, 128, 4, True)


@_GPU
@pytest.mark.parametrize("act", _ACTS, ids=lambda a: a.name)
@pytest.mark.parametrize("inter_dim", [256, 384])
def test_separated_a4w4_still_matches_the_reference(act, inter_dim):
    ref, out = _case(act, 16, 1024, inter_dim, 8, 2, False)
    assert torch.isfinite(out).all()
    # fp4 activation and intermediate quantization put the reference at 0.04 to 0.11
    assert _logits_diff(ref, out) < 0.2


@_GPU
def test_interleaved_a4w4_served_by_mxmoe_rows_still_matches_the_reference():
    ref, out = _case(ActivationType.Swiglu, 1024, 6144, 768, 128, 4, True)
    assert torch.isfinite(out).all()
    assert _logits_diff(ref, out) < 0.2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
