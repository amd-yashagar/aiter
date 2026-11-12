# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuned fp4 ASM 1-stage rows must be reachable and correct.

* ``fused_moe_1stage`` viewed the weight scales as ``(E, -1)``. ``e8m0_shuffle``
  pads the total row count to a multiple of 256, so for E=513 and inter_dim 160 or
  320 the element count is not a multiple of E and every call raised before the
  kernel ran. The 10 Qwen3.8-flash E=513 EMSORT rows could never be used.
* ``kimik3_a16w4_tuned_fmoe.csv`` named Situv2 a16w4 ASM kernels. ``get_2stage_cfgs``
  discards every 1-stage Situv2 row for preshuffled MXFP4 weights and raises for
  unpreshuffled ones, so those rows never ran.
"""

import glob
import os

import pandas as pd
import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

MODEL_DIM = 2560
E = 513
TOPK = 11
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "aiter", "configs")


def test_no_situv2_a16w4_asm_1stage_rows_are_shipped():
    dead = []
    for path in glob.glob(os.path.join(CONFIG_DIR, "**", "*fmoe*.csv"), recursive=True):
        df = pd.read_csv(path)
        if "kernelName1" not in df:
            continue
        rows = df[
            (df["act_type"] == str(ActivationType.Situv2))
            & (df["q_dtype_w"] == str(dtypes.fp4x2))
            & df["kernelName1"].astype(str).str.startswith("_ZN5aiter")
        ]
        dead += [(os.path.basename(path), i + 2) for i in rows.index]
    assert not dead, f"Situv2 ASM 1-stage rows are always discarded: {dead}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 a4w4 MoE required",
)
@pytest.mark.parametrize(("inter_dim", "token"), [(160, 32), (320, 64)])
def test_e513_emsort_row_matches_reference(monkeypatch, inter_dim, token):
    kernels = []
    one_stage = fused_moe_module.fused_moe_1stage

    def spy(*args, **kwargs):
        kernels.append(kwargs.get("kernelName"))
        return one_stage(*args, **kwargs)

    monkeypatch.setattr(fused_moe_module, "fused_moe_1stage", spy)
    fused_moe_module.get_2stage_cfgs.cache_clear()
    torch.manual_seed(0)
    dev, dtype = "cuda", dtypes.bf16
    x = torch.randn(token, MODEL_DIM, dtype=dtype, device=dev)
    w1 = torch.randn(E, inter_dim * 2, MODEL_DIM, dtype=dtype, device=dev)
    w2 = torch.randn(E, MODEL_DIM, inter_dim, dtype=dtype, device=dev)
    topk_weights, topk_ids = fused_topk(
        x, torch.randn(token, E, dtype=dtype, device=dev), TOPK, True
    )
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q, w1_scale = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(E, inter_dim * 2, MODEL_DIM // 2)
    w2_q = w2_q.view(E, MODEL_DIM, inter_dim // 2)

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
        a2.view(token, TOPK, -1),
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

    assert kernels and "emsort_silu" in kernels[0], kernels
    assert torch.isfinite(out).all()
    ref, out = ref.double(), out.double()
    logits_diff = 1 - 2 * (ref * out).sum() / (ref * ref + out * out).sum()
    assert logits_diff < 1e-2, f"logits_diff={logits_diff.item():.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
