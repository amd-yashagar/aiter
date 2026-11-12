# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A tuned flat (NO_SORT / EMSORT) asm 1-stage row must not serve an expert-parallel call.

The flat kernels read the raw top-k ids and skip the host sort that maps global
expert ids to local ones under an expert mask, so with an EP mask they index past
the local experts: a GPU memory fault (the process aborts). Dispatch now discards
such a row when an expert mask is passed.
"""

import pytest
import torch

import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import AsmMoeMode, fused_moe, get_2stage_cfgs, torch_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight

TOKEN, MODEL_DIM, INTER_DIM, E_TOTAL, E_LOCAL, TOPK = 64, 2048, 128, 512, 256, 11
BLK = 128
KERNEL = "_ZN5aiter48fmoe_bf16_blockscaleBf16_g1u1_flat_vs_silu_1x128E"


@pytest.fixture
def flat_row(monkeypatch):
    key = (
        "gfx950",
        256,
        TOKEN,
        MODEL_DIM,
        INTER_DIM,
        E_LOCAL,
        TOPK,
        str(ActivationType.Silu),
        str(dtypes.bf16),
        str(dtypes.fp8),
        str(dtypes.fp8),
        str(QuantType.per_1x128),
        True,
        False,
    )
    cfg = {
        "block_m": 1,
        "ksplit": 0,
        "kernelName1": KERNEL,
        "kernelName2": "",
        "run_1stage": True,
        "xbf16": 1,
        "flat": int(AsmMoeMode.NO_SORT),
    }
    monkeypatch.setattr(fused_moe_module, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fused_moe_module, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe_module, "cfg_2stages", ({key: cfg}, {}))
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def _dispatch(is_ep):
    get_2stage_cfgs.cache_clear()
    return get_2stage_cfgs(
        TOKEN,
        MODEL_DIM,
        INTER_DIM,
        E_LOCAL,
        TOPK,
        dtypes.bf16,
        dtypes.fp8,
        dtypes.fp8,
        QuantType.per_1x128,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        is_shuffled=True,
        is_ep=is_ep,
        opus_weights_shuffled=True,
    )


def test_flat_row_is_kept_without_an_expert_mask(flat_row):
    meta = _dispatch(is_ep=False)
    assert meta.run_1stage and meta.flat == AsmMoeMode.NO_SORT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs a ROCm GPU")
def test_flat_row_is_discarded_with_an_expert_mask(flat_row, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        fused_moe_module.logger, "warning", lambda msg: warnings.append(str(msg))
    )
    meta = _dispatch(is_ep=True)
    assert any("flat 1-stage asm config" in w for w in warnings)
    assert meta.flat != AsmMoeMode.NO_SORT
    assert getattr(meta.stage1, "keywords", {}).get("kernelName") != KERNEL


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 MI355X required",
)
def test_ep_call_with_a_flat_row_runs_and_matches_reference(flat_row):
    torch.manual_seed(0)
    dev = "cuda"
    hidden = torch.randn((TOKEN, MODEL_DIM), dtype=dtypes.bf16, device=dev)
    w1 = (torch.randn((E_LOCAL, INTER_DIM * 2, MODEL_DIM), device=dev) / 10).to(
        dtypes.fp8
    )
    w2 = (torch.randn((E_LOCAL, MODEL_DIM, INTER_DIM), device=dev) / 10).to(dtypes.fp8)
    w1_scale = torch.ones(
        (E_LOCAL, INTER_DIM * 2 // BLK, MODEL_DIM // BLK), dtype=dtypes.fp32, device=dev
    )
    w2_scale = torch.ones(
        (E_LOCAL, MODEL_DIM // BLK, INTER_DIM // BLK), dtype=dtypes.fp32, device=dev
    )
    expert_mask = torch.zeros(E_TOTAL, dtype=dtypes.i32, device=dev)
    expert_mask[E_TOTAL // 4 : E_TOTAL // 4 + E_LOCAL] = 1
    score = torch.randn((TOKEN, E_TOTAL), device=dev)
    topk_weights, topk_ids = torch.softmax(score, dim=-1).topk(TOPK, dim=-1)
    topk_ids = topk_ids.to(dtypes.i32)

    ref = torch_moe(hidden, w1, w2, topk_weights, topk_ids, expert_mask=expert_mask)
    out = fused_moe(
        hidden,
        shuffle_weight(w1, layout=(16, 16)),
        shuffle_weight(w2, layout=(16, 16)),
        topk_weights,
        topk_ids,
        expert_mask=expert_mask,
        quant_type=QuantType.per_1x128,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    assert torch.isfinite(out).all()
    out, ref = out.double(), ref.double()
    diff = 1 - 2 * (out * ref).sum() / (out * out + ref * ref).sum()
    assert diff < 1e-2, f"{diff=}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
