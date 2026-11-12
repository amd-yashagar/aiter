# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""fp4 x fp4 MoE at an inter_dim FlyDSL can't tile (160, 320, 576).

The heuristic FlyDSL MXFP4 route tiles stage1 N in 128s and stage2 K = inter_dim
in 128 or 256, so inter_dim 160 and 320 read out of bounds: the result was wrong
(1 - cosine about 1.0) with no error whenever no tuned row applied (tokens above
the tuned 512, or any M with the rows bypassed). Silu now takes the ASM no-sort
kernel the tuned rows already use at small M (it takes any multiple of 32), and
any other call that FlyDSL can't tile raises ``NotImplementedError``.
"""

import functools

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import AsmMoeMode, fused_moe, fused_topk, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

MODEL_DIM = 2560
E = 512
TOPK = 10
ASM_KERNEL = "_ZN5aiter49fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x32E"


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    monkeypatch.setattr(fused_moe_module, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "get_cu_num", lambda: 256)
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def _dispatch(
    inter_dim,
    token=1024,
    dtype=dtypes.bf16,
    has_bias=False,
    model_dim=MODEL_DIM,
    num_experts=E,
    topk=TOPK,
):
    get_2stage_cfgs.cache_clear()
    return get_2stage_cfgs(
        token,
        model_dim,
        inter_dim,
        num_experts,
        topk,
        dtype,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode=GateMode.SEPARATED.value,
        has_stage1_bias=has_bias,
        has_stage2_bias=has_bias,
        opus_weights_shuffled=True,
    )


@pytest.mark.parametrize("inter_dim", [160, 192, 320, 576])
@pytest.mark.parametrize("token", [1024, 4096])
def test_flydsl_untileable_inter_dim_takes_asm_no_sort(
    gfx950, monkeypatch, inter_dim, token
):
    warnings = []
    monkeypatch.setattr(
        fused_moe_module.logger, "warning", lambda msg: warnings.append(str(msg))
    )
    meta = _dispatch(inter_dim, token)
    assert any(
        f"inter_dim={inter_dim}" in w and "intermediate_pad" in w for w in warnings
    )
    assert meta.run_1stage
    assert meta.flat == AsmMoeMode.NO_SORT
    assert meta.stage1.keywords["kernelName"] == ASM_KERNEL


def test_deepseek_v41_576_takes_asm_no_sort_instead_of_faulting(gfx950):
    # tier1-00520: the heuristic FlyDSL kernel t128x128x128_atomic faulted here.
    meta = _dispatch(576, 8192, model_dim=5120, num_experts=384, topk=6)
    assert meta.run_1stage
    assert meta.stage1.keywords["kernelName"] == ASM_KERNEL


def test_flydsl_tileable_inter_dim_is_unchanged(gfx950):
    meta = _dispatch(256)
    assert not meta.run_1stage
    stage1 = meta.stage1.func if isinstance(meta.stage1, functools.partial) else None
    assert stage1.__name__ == "_flydsl_stage1_wrapper"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inter_dim": 112},
        {"inter_dim": 160, "has_bias": True},
        {"inter_dim": 160, "dtype": dtypes.fp16},
    ],
)
def test_no_backend_raises_not_implemented(gfx950, kwargs):
    with pytest.raises(NotImplementedError, match="No FlyDSL MXFP4 kernel"):
        _dispatch(**kwargs)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 a4w4 MoE required",
)
@pytest.mark.parametrize(
    ("model_dim", "inter_dim", "num_experts", "topk", "token"),
    [
        (MODEL_DIM, 160, E, TOPK, 600),
        (MODEL_DIM, 160, E, TOPK, 4096),
        (MODEL_DIM, 320, E, TOPK, 600),
        (MODEL_DIM, 320, E, TOPK, 4096),
        (MODEL_DIM, 576, E, TOPK, 600),
        (MODEL_DIM, 256, E, TOPK, 600),
        # DeepSeek-V4.1 (tier1-00520): the FlyDSL heuristic faulted here.
        (5120, 576, 384, 6, 600),
        (5120, 576, 384, 6, 8192),
    ],
)
def test_a4w4_silu_matches_reference(
    monkeypatch, model_dim, inter_dim, num_experts, topk, token
):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    torch.manual_seed(0)
    dev, dtype = "cuda", dtypes.bf16
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

    a1, a1_scale = quant(x, quant_dtype=dtypes.fp4x2)
    ref1 = fused_moe_module.torch_moe_stage1(
        a1,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
    )
    a2, a2_scale = quant(ref1, quant_dtype=dtypes.fp4x2)
    ref = fused_moe_module.torch_moe_stage2(
        a2.view(token, topk, -1),
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
    )

    out = fused_moe(
        x,
        shuffle_weight(w1_q, layout=(16, 16)),
        shuffle_weight(w2_q, layout=(16, 16)),
        topk_weights,
        topk_ids,
        w1_scale=fp4_utils.e8m0_shuffle(w1_scale),
        w2_scale=fp4_utils.e8m0_shuffle(w2_scale),
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Silu,
    )

    assert torch.isfinite(out).all()
    ref, out = ref.double(), out.double()
    logits_diff = 1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum()
    assert logits_diff < 1e-2, f"logits_diff={logits_diff.item():.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
