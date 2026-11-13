# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL a16w16 / a16wi4 MoE configs must fit the LDS of the arch they are registered for.

The registry filtered configs against the gfx950 budget (160 KiB) on every arch. On gfx942
(64 KiB per workgroup) the over-budget stage-1 configs (``t32x{16,32,64}x256_kw4``,
``t64x{16,32}x128_kw4``, ``t64x32x256_kw2``, ``t128x{32}x128_kw2``, ``t128x{64,128}x256``) and the
tile_m=128 stage-2 configs (``t128x128x128_atomic`` needs 96 KiB, ``t128x128x256_atomic`` 128 KiB;
stage 2 had no LDS filter at all) built, and then failed their first launch with
hipErrorIllegalState, which poisons the HIP context. A pair with tile_m=128 therefore died at
its stage-2 launch even though its stage-1 kernel (exactly 64 KiB) launches fine."""

import pytest
import torch

from aiter.jit.utils import chip_info
from aiter.ops.flydsl import moe_kernels as mk

KIB = 1024


def _registry(monkeypatch, gfx):
    monkeypatch.setattr(chip_info, "get_gfx", lambda: gfx)
    return {
        "a16w16 s1": mk.get_flydsl_stage1_kernels_a16w16("bf16"),
        "a16w16 s2": mk.get_flydsl_stage2_kernels_a16w16("bf16"),
        "int4 s1": mk.get_flydsl_stage1_kernels_int4_bf16("bf16"),
        "int4 s2": mk.get_flydsl_stage2_kernels_int4_bf16("bf16"),
    }


def _lds(label, params):
    if label.endswith("s1"):
        return mk._gemm1_lds_bytes(
            params["tile_m"], params["tile_n"], params["tile_k"], params["k_wave"]
        )
    return mk._gemm2_lds_bytes(
        params["tile_m"], params["tile_n"], params["tile_k"], params["mode"]
    )


def test_gfx942_registry_fits_64_kib(monkeypatch):
    for label, kernels in _registry(monkeypatch, "gfx942").items():
        assert kernels, label
        too_big = {
            n: _lds(label, p) for n, p in kernels.items() if _lds(label, p) > 64 * KIB
        }
        assert not too_big, f"{label}: {too_big}"


def test_gfx942_drops_the_configs_that_cannot_launch(monkeypatch):
    reg = _registry(monkeypatch, "gfx942")
    prefix1 = "flydsl_moe1_abf16_wbf16_bf16_"
    for tile in (
        "t32x16x256_kw4",
        "t32x32x256_kw4",
        "t32x64x256_kw4",
        "t64x16x128_kw4",
        "t64x32x128_kw4",
        "t64x32x256_kw2",
        "t128x32x128_kw2",
        "t128x128x256",
        "t128x64x256",
    ):
        assert prefix1 + tile not in reg["a16w16 s1"], tile
    prefix2 = "flydsl_moe2_abf16_wbf16_bf16_"
    for tile in ("t128x128x128_atomic", "t128x128x256_atomic"):
        assert prefix2 + tile not in reg["a16w16 s2"], tile
        assert "flydsl_moe2_abf16_wint4_bf16_" + tile not in reg["int4 s2"], tile


def test_gfx942_keeps_configs_at_exactly_64_kib(monkeypatch):
    # Not a size rule: configs at exactly 64 KiB launch (767 launches observed, tile_m 16/32/64,
    # and every stage-1 tile_m=128 config at 64 KiB launches on its own).
    reg = _registry(monkeypatch, "gfx942")
    prefix1 = "flydsl_moe1_abf16_wbf16_bf16_"
    for tile in (
        "t16x16x256_kw4",
        "t16x64x256_kw4",
        "t32x32x256_kw2",
        "t64x64x256",
        "t128x128x128",
    ):
        assert prefix1 + tile in reg["a16w16 s1"], tile
        assert _lds("a16w16 s1", reg["a16w16 s1"][prefix1 + tile]) <= 64 * KIB
    assert "flydsl_moe2_abf16_wbf16_bf16_t64x128x256_atomic" in reg["a16w16 s2"]


def test_gfx950_registry_keeps_its_160_kib_budget(monkeypatch):
    reg = _registry(monkeypatch, "gfx950")
    assert "flydsl_moe1_abf16_wbf16_bf16_t128x128x256" in reg["a16w16 s1"]
    assert "flydsl_moe1_abf16_wbf16_bf16_t32x16x256_kw4" in reg["a16w16 s1"]
    assert "flydsl_moe2_abf16_wbf16_bf16_t128x128x256_atomic" in reg["a16w16 s2"]
    for label, kernels in reg.items():
        assert all(_lds(label, p) <= 160 * KIB for p in kernels.values()), label


def _launch_stage(stage, name, tile_m):
    from aiter import ActivationType, dtypes
    from aiter.fused_moe import (
        _flydsl_stage1_wrapper,
        _flydsl_stage2_wrapper,
        fused_topk,
        moe_sorting,
    )
    from aiter.ops.shuffle import shuffle_weight

    E, topk, D, I, M = 8, 2, 4096, 1024, 64
    torch.manual_seed(0)
    x = torch.randn(M, D, dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(E, 2 * I, D, dtype=dtypes.bf16, device="cuda") * 0.05
    w2 = torch.randn(E, D, I, dtype=dtypes.bf16, device="cuda") * 0.05
    tw, ti = fused_topk(
        x, torch.randn(M, E, dtype=dtypes.bf16, device="cuda"), topk, True
    )
    sid, sw, seid, nv, _ = moe_sorting(ti, tw, E, D, dtypes.bf16, tile_m)
    if stage == 1:
        out = torch.zeros(M, topk, I, dtype=dtypes.bf16, device="cuda")
        _flydsl_stage1_wrapper(
            x,
            shuffle_weight(w1, layout=(16, 16)),
            w2,
            sid,
            seid,
            nv,
            out,
            topk,
            kernelName=name,
            activation=ActivationType.Silu,
        )
    else:
        out = torch.zeros(M, D, dtype=dtypes.bf16, device="cuda")
        inter = torch.randn(sid.shape[0], I, dtype=dtypes.bf16, device="cuda") * 0.1
        _flydsl_stage2_wrapper(
            inter,
            w1,
            shuffle_weight(w2, layout=(16, 16)),
            sid,
            seid,
            nv,
            out,
            topk,
            kernelName=name,
            sorted_weights=sw,
            block_m=tile_m,
        )
    torch.cuda.synchronize()
    # A failed launch is reported by the next HIP call that reads the context, not by the sync.
    assert torch.isfinite(out.float().abs().sum()).item()
    return out


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="Needs a ROCm GPU"
)
@pytest.mark.parametrize(
    "stage, name, tile_m",
    [
        (1, "flydsl_moe1_abf16_wbf16_bf16_t16x64x256_kw4", 16),
        (1, "flydsl_moe1_abf16_wbf16_bf16_t128x128x128", 128),
        (2, "flydsl_moe2_abf16_wbf16_bf16_t64x128x256_atomic", 64),
    ],
)
def test_registered_configs_launch_at_64_kib(stage, name, tile_m):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx942":
        pytest.skip("gfx942 LDS budget")
    assert mk.get_flydsl_kernel_params(name) is not None
    _launch_stage(stage, name, tile_m)
