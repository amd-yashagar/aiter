# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""bias1/bias2 on paths that do not apply them.

Only the MXFP4 (per_1x32, bf16/fp16 output) stages take the expert bias. Every other path
(a16w16, fp8 per-tensor, per-token and blockscale, int8, mxfp8) dropped it: the call returned
the result without the bias, bit for bit, even for a bias 100x the signal. This variant warns
once per path and argument; ``fix-moe-bias-ignored-raise`` raises instead.
"""

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in ("gfx942", "gfx950"),
    reason="CDNA GPU required",
)

D, I, E, TOPK = 1024, 512, 8, 2


def _quant_block_128(weight):
    n, d1, d2 = weight.shape
    wb = weight.view(n, d1 // 128, 128, d2 // 128, 128).permute(0, 1, 3, 2, 4)
    wb = wb.contiguous().view(n, -1, 128 * 128)
    wq, ws = aiter.pertoken_quant(wb, quant_dtype=dtypes.fp8)
    wq = wq.view(n, d1 // 128, d2 // 128, 128, 128).permute(0, 1, 3, 2, 4)
    return wq.contiguous().view(n, d1, d2), ws.view(n, d1 // 128, d2 // 128)


def _inputs(token):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(token, D, dtype=dtypes.bf16, device=dev)
    w1 = torch.randn(E, 2 * I, D, dtype=dtypes.bf16, device=dev) * 0.05
    w2 = torch.randn(E, D, I, dtype=dtypes.bf16, device=dev) * 0.05
    tw, ti = fused_topk(
        x, torch.randn(token, E, dtype=dtypes.bf16, device=dev), TOPK, True
    )
    return x, w1, w2, tw, ti


def _a16w16(x, w1, w2):
    return {
        "w1": shuffle_weight(w1, (16, 16)),
        "w2": shuffle_weight(w2, (16, 16)),
        "kwargs": {"quant_type": QuantType.No},
    }


def _fp8(quant_type):
    def build(x, w1, w2):
        quant = aiter.get_torch_quant(quant_type)
        w1q, w1s = quant(w1, quant_dtype=dtypes.fp8)
        w2q, w2s = quant(w2, quant_dtype=dtypes.fp8)
        return {
            "w1": shuffle_weight(w1q, (16, 16)),
            "w2": shuffle_weight(w2q, (16, 16)),
            "kwargs": {"quant_type": quant_type, "w1_scale": w1s, "w2_scale": w2s},
        }

    return build


def _fp8_block(x, w1, w2):
    w1q, w1s = _quant_block_128(w1)
    w2q, w2s = _quant_block_128(w2)
    return {
        "w1": shuffle_weight(w1q, (16, 16)),
        "w2": shuffle_weight(w2q, (16, 16)),
        "kwargs": {
            "quant_type": QuantType.per_1x128,
            "w1_scale": w1s,
            "w2_scale": w2s,
        },
    }


FAMILIES = {
    "a16w16": _a16w16,
    "fp8_per_token": _fp8(QuantType.per_Token),
    "fp8_per_tensor": _fp8(QuantType.per_Tensor),
    "fp8_blockscale": _fp8_block,
}


def _call(family, token, bias):
    x, w1, w2, tw, ti = _inputs(token)
    spec = FAMILIES[family](x, w1, w2)
    b1 = b2 = None
    if bias:
        b1 = torch.randn(E, 2 * I, dtype=torch.float32, device="cuda") * 100
        b2 = torch.randn(E, D, dtype=torch.float32, device="cuda") * 100
    return fused_moe(
        x,
        spec["w1"],
        spec["w2"],
        tw,
        ti,
        activation=ActivationType.Silu,
        bias1=b1,
        bias2=b2,
        **spec["kwargs"],
    )


@pytest.fixture
def warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(
        fused_moe_module.logger, "warning", lambda msg, *a, **k: seen.append(msg)
    )
    fused_moe_module._warn_bias_not_applied.cache_clear()
    return seen


@pytest.mark.parametrize("token", [16, 512])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_a_bias_the_path_drops_is_warned_about_once(family, token, warnings):
    out = _call(family, token, True)
    _call(family, token, True)
    dropped = [w for w in warnings if "is not applied by this MoE path" in w]
    # one warning per argument (bias1, bias2) and path, not one per call
    assert len(dropped) == 2
    assert any("bias1" in w for w in dropped) and any("bias2" in w for w in dropped)
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("family", list(FAMILIES))
def test_calls_without_a_bias_do_not_warn(family, warnings):
    _call(family, 16, False)
    assert not [w for w in warnings if "is not applied" in w]


@pytest.mark.parametrize("token", [16, 512])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_calls_without_bias_still_run(family, token):
    out = _call(family, token, False)
    assert torch.isfinite(out.float()).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
