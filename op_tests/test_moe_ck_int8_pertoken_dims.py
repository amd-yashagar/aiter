# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression: int8 per-token 2-stage MoE on gfx942 (asm stage 1 + CK stage 2).

* inter_dim 128 and 192 select the 256-thread KPerBlock = 64 stage-2 instances, which set
  BK1 = 8 bytes while the preshuffled B layout is written for 16 bytes per lane, so the
  result was wrong (logits diff about 0.94) at every model_dim and token count.
* model_dim not a multiple of 128 (2880): the asm stage-1 kernel consumes model_dim in
  blocks of 128 and returned wrong results (logits diff 0.02 to 0.1) with no error.
  Dispatch now raises NotImplementedError."""

import pytest
import torch

import aiter
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, torch_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip or get_gfx() != "gfx942",
    reason="Needs gfx942 (MI300X/MI325X)",
)

E, TOPK = 8, 2


def _run(act, model_dim, inter_dim, token):
    torch.manual_seed(0)
    x = torch.randn((token, model_dim), dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtypes.bf16, device="cuda")
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16, device="cuda")
    score = torch.randn((token, E), dtype=dtypes.bf16, device="cuda")
    topk_weights, topk_ids = fused_topk(x, score, TOPK, True)
    w1q, w1s = aiter.pertoken_quant(w1 / 10, quant_dtype=dtypes.i8)
    w2q, w2s = aiter.pertoken_quant(w2 / 10, quant_dtype=dtypes.i8)
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


@pytest.fixture
def two_stage(monkeypatch):
    monkeypatch.setitem(fm.fused_moe_1stage_dict, "gfx942", {})
    fm.get_2stage_cfgs.cache_clear()
    yield
    fm.get_2stage_cfgs.cache_clear()


@pytest.mark.parametrize("act", [ActivationType.Silu, ActivationType.Gelu])
@pytest.mark.parametrize("model_dim", [2560, 2688])
@pytest.mark.parametrize("inter_dim", [128, 192, 256])
@pytest.mark.parametrize("token", [8, 256])
def test_int8_pertoken_2stage_small_inter_dim(
    two_stage, act, model_dim, inter_dim, token
):
    ref, out = _run(act, model_dim, inter_dim, token)
    assert _logits_diff(ref, out) < 1e-2


@pytest.mark.parametrize("model_dim", [2880, 3040])
@pytest.mark.parametrize("token", [8, 256])
def test_int8_pertoken_2stage_refuses_model_dim_not_multiple_of_128(
    two_stage, model_dim, token
):
    with pytest.raises(NotImplementedError, match="model_dim % 128"):
        _run(ActivationType.Silu, model_dim, 256, token)
