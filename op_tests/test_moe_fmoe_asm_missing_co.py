# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""1-stage a16w16 ASM rows on an arch that does not ship ``fmoe_b16.co``.

``fused_moe_1stage`` runs ``aiter.fmoe`` for a pure a16w16 Silu g1u0 call, which loads
``hsa/<gfx>/fmoe_{b16,f16}.co``. gfx950 ships neither file, so a tuned row with
``run_1stage`` for that call (a user's tuned file; no shipped row has one) failed at launch with
"failed to open ... fmoe_b16.co". Dispatch now discards such a row and takes the CK 2-stage
heuristic.
"""

import csv
import os

import pytest

import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import get_2stage_cfgs
from aiter.jit.core import AITER_CONFIGS, AITER_ROOT_DIR
from aiter.jit.utils.chip_info import get_cu_num, get_gfx, get_gfx_runtime

pytestmark = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"), reason="CDNA dispatch required"
)

TOKEN, D, I, E, TOPK = 16, 1024, 256, 8, 2
HDR = (
    "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,"
    "q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,"
    "kernelName2,err2,us,run_1stage,tflops,bw,_tag"
)


def _tuned_file(tmp_path, dtype_name):
    path = tmp_path / "tuned_fmoe.csv"
    row = [
        get_cu_num(),
        TOKEN,
        D,
        I,
        E,
        TOPK,
        "ActivationType.Silu",
        dtype_name,
        dtype_name,
        dtype_name,
        "QuantType.No",
        0,
        0,
        32,
        0,
        0,
        "",
        "0.0%",
        0,
        "",
        "0.0%",
        0,
        1,
        0,
        0,
        "",
    ]
    with open(path, "w", newline="") as f:
        f.write(HDR + "\n")
        csv.writer(f).writerow(row)
    return str(path)


def _dispatch(monkeypatch, tmp_path, dtype):
    tuned = _tuned_file(tmp_path, str(dtype))
    monkeypatch.setattr(
        type(AITER_CONFIGS),
        "AITER_CONFIG_FMOE_FILE",
        property(lambda self: tuned),
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
            dtype,
            dtype,
            dtype,
            QuantType.No,
            False,
            ActivationType.Silu,
            False,
            0,
            0,
            is_shuffled=True,
        )
    finally:
        get_2stage_cfgs.cache_clear()
        monkeypatch.setattr(fused_moe_module, "cfg_2stages", None)


@pytest.mark.skipif(
    get_gfx_runtime() != "gfx950", reason="gfx950 does not ship the a16w16 ASM kernel"
)
@pytest.mark.parametrize("dtype", [dtypes.bf16, dtypes.fp16])
def test_gfx950_has_no_a16w16_fmoe_asm_kernel(dtype):
    tag = "b16" if dtype == dtypes.bf16 else "f16"
    assert not os.path.exists(
        os.path.join(AITER_ROOT_DIR, "hsa", "gfx950", f"fmoe_{tag}.co")
    )
    assert not fused_moe_module._fmoe_asm_bf16_kernel_exists(dtype)


@pytest.mark.skipif(
    get_gfx_runtime() != "gfx950", reason="gfx950 does not ship the a16w16 ASM kernel"
)
@pytest.mark.parametrize("dtype", [dtypes.bf16, dtypes.fp16])
def test_tuned_1stage_row_without_the_kernel_is_discarded(monkeypatch, tmp_path, dtype):
    # With the g1u0 dispatch guard, a discarded row ends in a refusal instead of
    # a 2-stage meta; either way the row must not be returned as 1-stage.
    try:
        meta = _dispatch(monkeypatch, tmp_path, dtype)
    except NotImplementedError:
        return
    assert not meta.run_1stage


@pytest.mark.parametrize("dtype", [dtypes.bf16, dtypes.fp16])
def test_tuned_1stage_row_with_the_kernel_is_kept(monkeypatch, tmp_path, dtype):
    monkeypatch.setattr(
        fused_moe_module, "_fmoe_asm_bf16_kernel_exists", lambda d: True
    )
    meta = _dispatch(monkeypatch, tmp_path, dtype)
    assert meta.run_1stage


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
