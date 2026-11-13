# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 fp8 blockscale: use the 64-row asm 1-stage tile once an expert sees ~32 tokens.

The default 1-stage path always sorted with block_m 32. On MI300X the 64x256 tile is
faster once the estimated tokens per expert reach about 32 (DeepSeek-V3 7168/256 E257
top9 at M >= 1024, Qwen3.5-397B 4096/256 E513 top11 at M >= 2048: 1.2x to 1.8x, warm and
cold, same accuracy) and slower below that (Qwen3.5 at M <= 1024 is 10 to 15 % slower), so
the tile follows the estimate."""

import pytest
import torch
import torch.nn.functional as F

import aiter
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight

DSV3 = (7168, 256, 257, 9)
QWEN35 = (4096, 256, 513, 11)


def _metadata(monkeypatch, gfx, token, shape, act=ActivationType.Silu):
    monkeypatch.setattr(fm, "get_gfx", lambda: gfx)
    monkeypatch.setattr(fm, "get_gfx_runtime", lambda: gfx)
    monkeypatch.setattr(fm, "get_cu_num", lambda: 304 if gfx == "gfx942" else 256)
    fm.get_2stage_cfgs.cache_clear()
    D, I, E, topk = shape
    try:
        return fm.get_2stage_cfgs(
            token,
            D,
            I,
            E,
            topk,
            dtypes.bf16,
            dtypes.fp8,
            dtypes.fp8,
            QuantType.per_1x128,
            True,
            act,
            False,
            0,
            0,
            is_shuffled=True,
        )
    finally:
        fm.get_2stage_cfgs.cache_clear()


@pytest.mark.parametrize(
    "shape,token,block_m",
    [
        (DSV3, 512, 32),  # 18 tokens per expert
        (DSV3, 1024, 64),  # 36
        (DSV3, 4096, 64),
        (QWEN35, 1024, 32),  # 22
        (QWEN35, 2048, 64),  # 44
        (QWEN35, 8192, 64),
    ],
)
def test_gfx942_tile_follows_the_tokens_per_expert(monkeypatch, shape, token, block_m):
    meta = _metadata(monkeypatch, "gfx942", token, shape)
    assert meta.run_1stage
    assert meta.block_m == block_m


@pytest.mark.parametrize("token", [1024, 8192])
def test_gelu_has_no_gfx942_1stage_and_is_untouched(monkeypatch, token):
    # Gelu blockscale takes CK 2-stage on gfx942: no asm tile to choose.
    assert not _metadata(
        monkeypatch, "gfx942", token, DSV3, ActivationType.Gelu
    ).run_1stage


@pytest.mark.parametrize("shape,token", [(DSV3, 4096), (QWEN35, 8192)])
def test_gfx950_is_unchanged(monkeypatch, shape, token):
    meta = _metadata(monkeypatch, "gfx950", token, shape)
    assert meta.block_m == 32 or not meta.run_1stage


def _blockq(w):
    E, N, K = w.shape
    b = (
        w.view(E, N // 128, 128, K // 128, 128)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(E, -1, 128 * 128)
    )
    q, s = aiter.pertoken_quant(b, quant_dtype=dtypes.fp8)
    q = (
        q.view(E, N // 128, K // 128, 128, 128)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(E, N, K)
    )
    return q, s.view(E, N // 128, K // 128)


def _deq(q, s):
    E, N, K = q.shape
    return (
        q.float().view(E, N // 128, 128, K // 128, 128)
        * s.view(E, N // 128, 1, K // 128, 1)
    ).view(E, N, K)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or get_gfx_runtime() != "gfx942",
    reason="Needs gfx942 (MI300X/MI325X)",
)
def test_64_row_tile_result_matches_the_reference(monkeypatch):
    D, I, E, topk, M = 2560, 256, 16, 4, 4096
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * I, D, dtype=dtypes.bf16, device="cuda") * 0.05
    w2 = torch.randn(E, D, I, dtype=dtypes.bf16, device="cuda") * 0.05
    w1q, w1s = _blockq(w1)
    w2q, w2s = _blockq(w2)
    x = torch.randn(M, D, dtype=dtypes.bf16, device="cuda")
    tw, ti = fused_topk(
        x, torch.randn(M, E, dtype=dtypes.bf16, device="cuda"), topk, True
    )
    w1d, w2d = _deq(w1q, w1s), _deq(w2q, w2s)
    ref = torch.zeros(M, D, device="cuda")
    for e in range(E):
        tok, slot = (ti == e).nonzero(as_tuple=True)
        h = x[tok].float() @ w1d[e].T
        a = F.silu(h[:, :I]) * h[:, I:]
        ref.index_add_(0, tok, (a @ w2d[e].T) * tw[tok, slot].unsqueeze(1).float())

    seen = []
    real = fm.get_2stage_cfgs

    def spy(*args, **kwargs):
        meta = real(*args, **kwargs)
        seen.append(meta)
        return meta

    real.cache_clear()
    monkeypatch.setattr(fm, "get_2stage_cfgs", spy)
    out = fused_moe(
        x,
        shuffle_weight(w1q, layout=(16, 16)),
        shuffle_weight(w2q, layout=(16, 16)),
        tw,
        ti,
        quant_type=QuantType.per_1x128,
        activation=ActivationType.Silu,
        w1_scale=w1s,
        w2_scale=w2s,
    )
    torch.cuda.synchronize()
    assert seen[-1].run_1stage and seen[-1].block_m == 64
    r, o = ref.double(), out.double()
    assert (1 - 2 * (r * o).sum() / (r * r + o * o).sum()).item() < 2e-3
