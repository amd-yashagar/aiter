# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression: per-token fp8 (gfx942, gfx950) and int8 (gfx942) asm 1-stage MoE needs
model_dim % 256 == 0.

The `fmoe_bf16_pertoken{Fp8,Int8}_g1u1_*` kernels walk model_dim in steps of 256.
For any other model_dim (including multiples of 128 such as 2688 or 2944, and gpt-oss's
2880) they return wrong results (logits diff 0.1 to 1.0) with no error. Kernel
selection was called without model_dim, so no constraint was ever checked. The
heuristic now keeps those shapes off the 1-stage kernel, and the kernel selection
refuses them."""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, torch_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or get_gfx() not in ("gfx942", "gfx950"),
    reason="Needs gfx942 or gfx950",
)

E, TOPK = 8, 2
SILU, GELU = ActivationType.Silu, ActivationType.Gelu


def _run(quant, act, model_dim, inter_dim, token):
    if quant == "i8" and get_gfx() != "gfx942":
        pytest.skip("per-token int8 asm 1-stage is only checked on gfx942")
    qd = dtypes.fp8 if quant == "fp8" else dtypes.i8
    torch.manual_seed(0)
    x = torch.randn((token, model_dim), dtype=dtypes.bf16, device="cuda")
    w1 = (
        torch.randn((E, inter_dim * 2, model_dim), dtype=dtypes.bf16, device="cuda")
        / 10
    )
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16, device="cuda") / 10
    score = torch.randn((token, E), dtype=dtypes.bf16, device="cuda")
    topk_weights, topk_ids = torch.softmax(score.float(), dim=-1).topk(TOPK, dim=-1)
    topk_ids = topk_ids.to(dtypes.i32)
    w1q, w1s = aiter.pertoken_quant(w1, quant_dtype=qd)
    w2q, w2s = aiter.pertoken_quant(w2, quant_dtype=qd)
    ref = torch_moe(
        x,
        w1q.float() * w1s.float(),
        w2q.float() * w2s.float(),
        topk_weights,
        topk_ids,
        activation=act,
    )
    out = fused_moe(
        x,
        shuffle_weight(w1q, layout=(16, 16)),
        shuffle_weight(w2q, layout=(16, 16)),
        topk_weights,
        topk_ids,
        quant_type=QuantType.per_Token,
        activation=act,
        w1_scale=w1s,
        w2_scale=w2s,
    )
    torch.cuda.synchronize()
    return ref, out


def _logits_diff(ref, out):
    ref, out = ref.double(), out.double().nan_to_num()
    return float(1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum())


@pytest.mark.parametrize(
    "quant, act, model_dim, inter_dim, token",
    [
        ("fp8", SILU, 2688, 256, 64),
        ("fp8", SILU, 2688, 512, 512),
        ("fp8", GELU, 2944, 256, 64),
        ("i8", SILU, 2688, 512, 64),
        ("i8", GELU, 2944, 256, 512),
    ],
)
def test_pertoken_asm_non_256_model_dim_is_correct(
    quant, act, model_dim, inter_dim, token
):
    ref, out = _run(quant, act, model_dim, inter_dim, token)
    assert _logits_diff(ref, out) < 1e-2


@pytest.mark.parametrize(
    "model_dim, inter_dim, token", [(2880, 256, 64), (2880, 320, 512), (3040, 320, 17)]
)
def test_pertoken_asm_unsupported_model_dim_is_not_silently_wrong(
    model_dim, inter_dim, token
):
    try:
        ref, out = _run("fp8", SILU, model_dim, inter_dim, token)
    except (RuntimeError, NotImplementedError):
        return
    assert _logits_diff(ref, out) < 1e-2


@pytest.mark.parametrize("quant", ["fp8", "i8"])
@pytest.mark.parametrize("model_dim", [2560, 3072])
def test_pertoken_asm_256_multiple_keeps_1stage(monkeypatch, quant, model_dim):
    import aiter.fused_moe as fm

    seen = []
    real = fm.get_2stage_cfgs

    def spy(*args, **kwargs):
        metadata = real(*args, **kwargs)
        seen.append(metadata.run_1stage)
        return metadata

    monkeypatch.setattr(fm, "get_2stage_cfgs", spy)
    ref, out = _run(quant, SILU, model_dim, 256, 512)
    assert seen and all(seen), "expected the asm 1-stage route"
    assert _logits_diff(ref, out) < 1e-2


@pytest.mark.parametrize("model_dim", [2688, 2880])
def test_pertoken_asm_tkw1_refuses_non_256_model_dim(model_dim):
    from aiter.fused_moe import moe_sorting

    inter_dim, token = 256, 64
    torch.manual_seed(0)
    x = torch.randn((token, model_dim), dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtypes.bf16, device="cuda")
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16, device="cuda")
    w1q, w1s = aiter.pertoken_quant(w1 / 10, quant_dtype=dtypes.fp8)
    w2q, w2s = aiter.pertoken_quant(w2 / 10, quant_dtype=dtypes.fp8)
    topk_ids = torch.stack(
        [torch.randperm(E, device="cuda")[:1] for _ in range(token)]
    ).to(dtypes.i32)
    topk_weights = torch.ones((token, 1), dtype=dtypes.fp32, device="cuda")
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtypes.bf16, 32
    )
    a8 = torch.empty((token, model_dim), dtype=w1q.dtype, device="cuda")
    a8_scale = torch.empty(token, dtype=dtypes.fp32, device="cuda")
    aiter.dynamic_per_token_scaled_quant(a8, x, a8_scale)
    with pytest.raises(RuntimeError, match="No suitable kernel found"):
        aiter.fmoe_g1u1_tkw1(
            moe_buf,
            a8,
            shuffle_weight(w1q, layout=(16, 16)),
            shuffle_weight(w2q, layout=(16, 16)),
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            1,
            a8_scale,
            w1s,
            w2s,
            "",
            None,
            ActivationType.Silu.value,
        )
        torch.cuda.synchronize()
