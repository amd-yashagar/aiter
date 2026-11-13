# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK 2-stage modules that cannot be built must be refused before the JIT build.

With no tuned row, ``get_2stage_cfgs`` falls to the default CK pair, whose module is
built at the first call. Three combinations never build (checked with host builds for
gfx950 and gfx942, no GPU):

* any activation Situv2: ``gen_instances.py`` rejects it;
* mxfp8 (fp8 x fp8 per_1x32), any activation: every instance fails to compile;
* MXFP4 weights with Swiglu on gfx950: ``static_assert`` in ``gridwise_moe_mx_gemm``
  (on gfx942 the MXFP4 instances are compiled out, so that case is left alone).

The call used to die minutes later inside the build; it now raises NotImplementedError
at dispatch. Every neighbouring combination that does build keeps its route.
"""

import csv

import pytest

import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import get_2stage_cfgs
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx, get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

pytestmark = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"), reason="CDNA dispatch required"
)

HDR = (
    "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,"
    "q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,"
    "kernelName2,err2,us,run_1stage,tflops,bw,_tag"
)
FLY_S1 = "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w4_gui_fp8"
CK_S2 = (
    "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_"
    "Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
)
D, I, E, TOPK = 6144, 1024, 128, 4
ARCHS = ["native", "gfx942"]


def _use_tuned_rows(monkeypatch, tmp_path, rows):
    path = tmp_path / "tuned_fmoe.csv"
    with open(path, "w", newline="") as f:
        f.write(HDR + "\n")
        csv.writer(f).writerows(rows)
    monkeypatch.setattr(
        type(AITER_CONFIGS),
        "AITER_CONFIG_FMOE_FILE",
        property(lambda self: str(path)),
    )
    monkeypatch.setattr(fused_moe_module, "cfg_2stages", None)
    get_2stage_cfgs.cache_clear()


def _arch(monkeypatch, arch):
    if arch != "native":
        monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: arch)
    return get_gfx_runtime() if arch == "native" else arch


def _default_route(monkeypatch, tmp_path, arch, qa, qw, qt, act, shuffled=False):
    _use_tuned_rows(monkeypatch, tmp_path, [])
    _arch(monkeypatch, arch)
    try:
        return get_2stage_cfgs(
            8,
            D,
            I,
            E,
            TOPK,
            dtypes.bf16,
            qa,
            qw,
            qt,
            True,
            act,
            False,
            0,
            0,
            is_shuffled=shuffled,
            gate_mode=GateMode.SEPARATED,
        )
    finally:
        get_2stage_cfgs.cache_clear()
        monkeypatch.setattr(fused_moe_module, "cfg_2stages", None)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize(
    "act", [ActivationType.Silu, ActivationType.Gelu, ActivationType.Swiglu]
)
def test_mxfp8_default_ck_route_is_refused(monkeypatch, tmp_path, arch, act):
    with pytest.raises(NotImplementedError, match="mxfp8"):
        _default_route(
            monkeypatch,
            tmp_path,
            arch,
            dtypes.fp8,
            dtypes.fp8,
            QuantType.per_1x32,
            act,
            shuffled=True,
        )


@pytest.mark.parametrize("arch", ARCHS)
def test_situv2_default_ck_route_is_refused(monkeypatch, tmp_path, arch):
    with pytest.raises(NotImplementedError, match="Situv2"):
        _default_route(
            monkeypatch,
            tmp_path,
            arch,
            dtypes.bf16,
            dtypes.fp4x2,
            QuantType.per_1x32,
            ActivationType.Situv2,
        )


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize(
    "qa,qw,qt,act,shuffled",
    [
        (dtypes.fp8, dtypes.fp4x2, QuantType.per_1x32, ActivationType.Silu, False),
        (dtypes.fp8, dtypes.fp4x2, QuantType.per_1x32, ActivationType.Gelu, False),
        (dtypes.fp4x2, dtypes.fp4x2, QuantType.per_1x32, ActivationType.Silu, False),
        (dtypes.fp8, dtypes.fp8, QuantType.per_Token, ActivationType.Silu, True),
        (dtypes.fp8, dtypes.fp8, QuantType.per_Token, ActivationType.Swiglu, True),
        (dtypes.fp8, dtypes.fp8, QuantType.per_1x128, ActivationType.Silu, True),
    ],
)
def test_buildable_default_ck_routes_are_unchanged(
    monkeypatch, tmp_path, arch, qa, qw, qt, act, shuffled
):
    meta = _default_route(monkeypatch, tmp_path, arch, qa, qw, qt, act, shuffled)
    assert meta.stage1.func is fused_moe_module.ck_moe_stage1


def _mixed(monkeypatch, tmp_path, arch):
    _use_tuned_rows(
        monkeypatch,
        tmp_path,
        [
            [
                get_cu_num(),
                256,
                D,
                I,
                E,
                TOPK,
                "ActivationType.Swiglu",
                "torch.bfloat16",
                str(dtypes.fp8),
                "torch.float4_e2m1fn_x2",
                "QuantType.per_1x32",
                1,
                0,
                32,
                0,
                0,
                FLY_S1,
                "0.0%",
                0,
                CK_S2,
                "0.0%",
                0,
                0,
                0,
                0,
                "",
            ]
        ],
    )
    _arch(monkeypatch, arch)
    try:
        return get_2stage_cfgs(
            256,
            D,
            I,
            E,
            TOPK,
            dtypes.bf16,
            dtypes.fp8,
            dtypes.fp4x2,
            QuantType.per_1x32,
            True,
            ActivationType.Swiglu,
            False,
            0,
            0,
            is_shuffled=True,
            gate_mode=GateMode.INTERLEAVE,
            gate_up_interleave=True,
            opus_weights_shuffled=True,
        )
    finally:
        get_2stage_cfgs.cache_clear()
        monkeypatch.setattr(fused_moe_module, "cfg_2stages", None)


def test_tuned_row_with_a_ck_stage2_for_mxfp4_swiglu_is_refused_on_gfx950(
    monkeypatch, tmp_path
):
    if get_gfx_runtime() != "gfx950":
        pytest.skip("native gfx950 only")
    with pytest.raises(NotImplementedError, match="Swiglu epilogue"):
        _mixed(monkeypatch, tmp_path, "native")


def test_same_row_is_left_alone_on_gfx942(monkeypatch, tmp_path):
    assert not _mixed(monkeypatch, tmp_path, "gfx942").run_1stage


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
