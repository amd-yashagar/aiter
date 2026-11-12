# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK blockscale (per_1x128 fp8) Swiglu.

Dispatch sends per_1x128 fp8 Swiglu to the CK blockscale 2-stage module. In a CK checkout whose
``gridwise_moe_gemm_blockscale.hpp`` lacks ``swiglu_oai_and_mul``, every stage-1 instance fails a
``static_assert`` after several minutes of JIT build. aiter now refuses that call up front, and with
the epilogue in CK the call runs and matches the torch reference.
"""

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight

_ARCH = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"), reason="CDNA dispatch required"
)
_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in ("gfx942", "gfx950"),
    reason="CDNA GPU required",
)


def _dispatch(swiglu_limit=None):
    get_2stage_cfgs.cache_clear()
    return get_2stage_cfgs(
        8,
        6144,
        384,
        129,
        5,
        dtypes.bf16,
        dtypes.fp8,
        dtypes.fp8,
        QuantType.per_1x128,
        True,
        ActivationType.Swiglu,
        False,
        0,
        0,
        is_shuffled=True,
        swiglu_limit=swiglu_limit,
    )


@_ARCH
def test_checkout_without_the_epilogue_is_refused_before_any_build(monkeypatch):
    monkeypatch.setattr(
        fused_moe_module, "_ck_blockscale_swiglu_oai_supported", lambda: False
    )
    with pytest.raises(NotImplementedError, match="no Swiglu epilogue"):
        _dispatch()


@_ARCH
def test_checkout_with_the_epilogue_dispatches_to_ck(monkeypatch):
    monkeypatch.setattr(
        fused_moe_module, "_ck_blockscale_swiglu_oai_supported", lambda: True
    )
    meta = _dispatch()
    assert meta.stage1.func is fused_moe_module.ck_moe_stage1


@_ARCH
@pytest.mark.parametrize("limit", [None, 0.0, 7.0])
def test_limits_the_epilogue_implements_are_accepted(monkeypatch, limit):
    monkeypatch.setattr(
        fused_moe_module, "_ck_blockscale_swiglu_oai_supported", lambda: True
    )
    _dispatch(limit)


@_ARCH
@pytest.mark.parametrize("limit", [10.0, 1.0, float("inf")])
def test_other_limits_are_refused(monkeypatch, limit):
    monkeypatch.setattr(
        fused_moe_module, "_ck_blockscale_swiglu_oai_supported", lambda: True
    )
    with pytest.raises(NotImplementedError, match="clamps at 7.0"):
        _dispatch(limit)


def _quant_block_128(weight):
    E, d1, d2 = weight.shape
    wb = weight.view(E, d1 // 128, 128, d2 // 128, 128).permute(0, 1, 3, 2, 4)
    wb = wb.contiguous().view(E, -1, 128 * 128)
    wq, ws = aiter.pertoken_quant(wb, quant_dtype=dtypes.fp8)
    wq = wq.view(E, d1 // 128, d2 // 128, 128, 128).permute(0, 1, 3, 2, 4)
    return wq.contiguous().view(E, d1, d2), ws.view(E, d1 // 128, d2 // 128)


@_GPU
@pytest.mark.skipif(
    not fused_moe_module._ck_blockscale_swiglu_oai_supported(),
    reason="this composable_kernel checkout has no blockscale Swiglu epilogue",
)
# (8, 64) takes the split-k stage 1 (get_ksplit > 1), the others the plain one.
@pytest.mark.parametrize(("token", "E"), [(8, 8), (64, 8), (8, 64)])
def test_blockscale_swiglu_matches_the_reference(token, E):
    torch.manual_seed(0)
    D, I, topk, dev, dtype = 1024, 384, 2, "cuda", dtypes.bf16
    if (token, E) == (8, 64):
        assert fused_moe_module.get_ksplit(token, topk, E, I, D) > 1
    x = torch.randn(token, D, dtype=dtype, device=dev)
    w1 = torch.randn(E, 2 * I, D, dtype=dtype, device=dev) * 0.3
    w2 = torch.randn(E, D, I, dtype=dtype, device=dev) * 0.3
    topk_weights, topk_ids = fused_topk(
        x, torch.randn(token, E, dtype=dtype, device=dev), topk, True
    )
    w1q, w1s = _quant_block_128(w1)
    w2q, w2s = _quant_block_128(w2)
    a1q, a1s = aiter.pertoken_quant(x.view(token, -1, 128), quant_dtype=dtypes.fp8)
    ref1 = fused_moe_module.torch_moe_stage1(
        a1q.view(token, D),
        w1q,
        w2q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x128,
        a1_scale=a1s.squeeze(-1),
        w1_scale=w1s,
    )
    a2q, a2s = aiter.pertoken_quant(
        ref1.view(token, topk, -1, 128).view(token, -1, 128), quant_dtype=dtypes.fp8
    )
    ref = fused_moe_module.torch_moe_stage2(
        a2q.view(token, topk, -1),
        w1q,
        w2q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x128,
        w2_scale=w2s,
        a2_scale=a2s.view(token, topk, -1),
    )
    w1_sh, w2_sh = shuffle_weight(w1q, (16, 16)), shuffle_weight(w2q, (16, 16))
    get_2stage_cfgs.cache_clear()
    out = fused_moe(
        x,
        w1_sh,
        w2_sh,
        topk_weights,
        topk_ids,
        quant_type=QuantType.per_1x128,
        activation=ActivationType.Swiglu,
        w1_scale=w1s,
        w2_scale=w2s,
    )
    r, o = ref.double(), out.double()
    assert torch.isfinite(o).all()
    assert float(1 - 2 * (r * o).sum() / (r * r + o * o).sum()) < 1e-2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
