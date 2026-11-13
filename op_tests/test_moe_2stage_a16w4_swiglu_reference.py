# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression: the test_moe_2stage.py reference must follow the dispatch's
activation switch.

a16w4 Swiglu with separated gate/up weights is dispatched as a4w4 (fp4 activations,
fp4 intermediate) from GPTOSS_SWIGLU_MXFP4_BF16_BOUND tokens on. The reference of a
declared-bf16 case stayed unquantized, so every one of those cases failed the
accuracy check at logits_diff 0.076 to 0.080 although the kernels sit at the bf16
floor (1e-5) of their own rounding point."""

import runpy
import sys
from pathlib import Path

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip or get_gfx() != "gfx950",
    reason="Needs a gfx950 GPU",
)

SCRIPT = str(Path(__file__).with_name("test_moe_2stage.py"))


@pytest.fixture(scope="module")
def sweep_namespace():
    argv = sys.argv
    sys.argv = [SCRIPT, "-q", "6", "--no-legacy", "--no-flydsl-csv"]
    try:
        return runpy.run_path(SCRIPT, run_name="__main__")
    finally:
        sys.argv = argv


def _case(ns, token):
    quant = (aiter.QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2)
    return {
        "dtype": dtypes.bf16,
        "token": token,
        "model_dim": 7168,
        "inter_dim": 256,
        "E": 257,
        "topk": 9,
        "actType": aiter.ActivationType.Swiglu,
        "gateMode": ns["_effective_gate_mode"](*quant),
        "qType": quant[0],
        "AQDType": quant[1],
        "WQDType": quant[2],
        "use_g1u1": True,
        "doweight_stage1": False,
        "hidden_pad": 192,
        "intermediate_pad": 128,
        "strict_accuracy": True,
        "check_aot_cache": False,
        "swiglu_limit": ns["_effective_swiglu_limit"](
            *quant, ns["args"].swiglu_limit, aiter.ActivationType.Swiglu
        ),
    }


# 256 and 1024 take tuned rows whose stage 1 fuses the fp4 quant (reference R0);
# 4096 takes the heuristic route whose stage 2 quantizes the bf16 intermediate (R1).
@pytest.mark.parametrize("token", [256, 1024, 4096])
def test_a4w4_dispatch_of_a16w4_swiglu_is_checked_against_its_own_reference(
    sweep_namespace, token
):
    ret = sweep_namespace["test_fmoe"](**_case(sweep_namespace, token))
    assert ret["logits_diff"] < 1e-3, ret["logits_diff"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
