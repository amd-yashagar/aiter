# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuned rows that pair a CK-Tile or ASM stage 1 with a FlyDSL or opus stage 2.

``get_2stage_cfgs`` sent every tuned row with a FlyDSL stage 2 through one branch that
built stage 1 as opus, FlyDSL or, for any other name, ``ck_moe_stage1(kernelName=...)``.
A CK-Tile name (``cktile_a8w4_bm32``) or an ASM name (``_ZN5aiter...``) therefore asked
the CK 2-stage codegen for an instance it does not have (for swiglu per_1x32 the module
cannot even build). Stage 1 is now the CK-Tile function for CK-Tile names, and the pairs
that have no correct stage-1/stage-2 combination are refused with a clear error.
"""

import csv
import functools

import pytest

import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import cktile_moe_stage1, get_2stage_cfgs
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.moe_common import GateMode

pytestmark = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"), reason="CDNA dispatch required"
)

TOKEN, D, I, E, TOPK = 128, 3072, 3072, 128, 4
HDR = (
    "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,"
    "q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,"
    "kernelName2,err2,us,run_1stage,tflops,bw,_tag"
)
CKTILE = "cktile_a8w4_bm32"
ASM_STAGE1 = "_ZN5aiter56fmoe_stage1_bf16_pertokenFp8_doweight_g1u1_32x64_4tg_pf3E"
FLY_S1 = "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w4_gui_fp8"
FLY_S2 = "flydsl_moe2_afp8_wfp4_bf16_t32x128x128_atomic_bnt2"
FLY_S2_LAYOUT = "flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_nt_sbm32"
CK_S1 = (
    "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v1_"
    "Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
)


def _dispatch(monkeypatch, tmp_path, kernel1, kernel2):
    path = tmp_path / "tuned_fmoe.csv"
    row = [
        get_cu_num(),
        TOKEN,
        D,
        I,
        E,
        TOPK,
        "ActivationType.Swiglu",
        "torch.bfloat16",
        "torch.float8_e4m3fn",
        "torch.float4_e2m1fn_x2",
        "QuantType.per_1x32",
        1,
        0,
        32,
        0,
        0,
        kernel1,
        "0.0%",
        0,
        kernel2,
        "0.0%",
        0,
        0,
        0,
        0,
        "",
    ]
    with open(path, "w", newline="") as f:
        f.write(HDR + "\n")
        csv.writer(f).writerow(row)
    monkeypatch.setattr(
        type(AITER_CONFIGS),
        "AITER_CONFIG_FMOE_FILE",
        property(lambda self: str(path)),
    )
    monkeypatch.setattr(fused_moe_module, "cfg_2stages", None)
    get_2stage_cfgs.cache_clear()
    try:
        return get_2stage_cfgs(
            TOKEN,
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
            192,
            192,
            is_shuffled=True,
            gate_mode=GateMode.INTERLEAVE,
            gate_up_interleave=True,
            opus_weights_shuffled=True,
        )
    finally:
        get_2stage_cfgs.cache_clear()
        monkeypatch.setattr(fused_moe_module, "cfg_2stages", None)


def _func(partial):
    return partial.func if isinstance(partial, functools.partial) else partial


def test_cktile_stage1_with_flydsl_stage2_runs_cktile_stage1(monkeypatch, tmp_path):
    meta = _dispatch(monkeypatch, tmp_path, CKTILE, FLY_S2)
    assert _func(meta.stage1) is cktile_moe_stage1
    assert meta.stage1.keywords["n_pad_zeros"] == 192 // 64 * 64 * 2
    assert meta.stage1.keywords["k_pad_zeros"] == 192 // 128 * 128
    assert _func(meta.stage2) is fused_moe_module._flydsl_stage2_wrapper


@pytest.mark.parametrize(
    "kernel1,kernel2",
    [
        (CKTILE, FLY_S2_LAYOUT),
        (ASM_STAGE1, FLY_S2),
        (ASM_STAGE1, FLY_S2_LAYOUT),
        (CK_S1, FLY_S2_LAYOUT),
    ],
)
def test_pairs_without_a_correct_combination_are_refused(
    monkeypatch, tmp_path, kernel1, kernel2
):
    with pytest.raises(NotImplementedError, match="cannot feed|no verified pairing"):
        _dispatch(monkeypatch, tmp_path, kernel1, kernel2)


@pytest.mark.parametrize(
    "kernel1,kernel2,stage1,stage2",
    [
        (FLY_S1, FLY_S2, "_flydsl_stage1_wrapper", "_flydsl_stage2_wrapper"),
        (FLY_S1, CKTILE, "_flydsl_stage1_wrapper", "cktile_moe_stage2"),
        (CKTILE, CKTILE, "cktile_moe_stage1", "cktile_moe_stage2"),
    ],
)
def test_existing_pairs_are_unchanged(
    monkeypatch, tmp_path, kernel1, kernel2, stage1, stage2
):
    meta = _dispatch(monkeypatch, tmp_path, kernel1, kernel2)
    assert _func(meta.stage1).__name__ == stage1
    assert _func(meta.stage2).__name__ == stage2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
