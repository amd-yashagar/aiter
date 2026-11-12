# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression: fp8 per_1x128 at token 16, 6144x256, E 257, top-9 on gfx942
(304 CU) must give correct results, also when a tuned row asks for the ASM
1-stage EMSORT kernel fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_emsort_16x128_pf3.

That kernel returned wrong results (1 - cos about 0.99) for every shape and
token count tried on gfx942, and a shipped tuned row selected it."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, torch_moe
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="Needs a ROCm GPU",
)

TOKEN, MODEL_DIM, INTER_DIM, E, TOPK = 16, 6144, 256, 257, 9
BLK = 128


HEADER = (
    "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,"
    "q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,"
    "err1,us2,kernelName2,err2,us,run_1stage,xbf16,flat,tflops,bw,_tag"
)
EMSORT_ROW = (
    "gfx942,304,16,6144,256,257,9,ActivationType.Silu,torch.bfloat16,"
    "torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_1x128,1,0,16,0,"
    "123.614,_ZN5aiter62fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_emsort_16x128_pf3E,"
    "0.0%,0.0,,0.0%,123.614,1,1,2,0.0,0.0,"
)


def _check():
    torch.manual_seed(0)
    hidden = torch.randn((TOKEN, MODEL_DIM), dtype=dtypes.bf16, device="cuda")
    w1 = (torch.randn((E, INTER_DIM * 2, MODEL_DIM), device="cuda") / 10).to(dtypes.fp8)
    w2 = (torch.randn((E, MODEL_DIM, INTER_DIM), device="cuda") / 10).to(dtypes.fp8)
    w1_scale = torch.ones(
        (E, INTER_DIM * 2 // BLK, MODEL_DIM // BLK), dtype=dtypes.fp32, device="cuda"
    )
    w2_scale = torch.ones(
        (E, MODEL_DIM // BLK, INTER_DIM // BLK), dtype=dtypes.fp32, device="cuda"
    )
    score = torch.randn((TOKEN, E), device="cuda")
    topk_weights, topk_ids = torch.softmax(score, dim=-1).topk(TOPK, dim=-1)
    topk_ids = topk_ids.to(dtypes.i32)

    ref = torch_moe(hidden, w1, w2, topk_weights, topk_ids)

    out = fused_moe(
        hidden,
        shuffle_weight(w1, layout=(16, 16)),
        shuffle_weight(w2, layout=(16, 16)),
        topk_weights,
        topk_ids,
        quant_type=aiter.QuantType.per_1x128,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    assert torch.isfinite(out).all()
    out, ref = out.double(), ref.double()
    diff = 1 - 2 * (out * ref).sum() / (out * out + ref * ref).sum()
    assert diff < 1e-2, f"{diff=}, mean|out|={out.abs().mean():.6f}"


def test_per_1x128_token16_e257():
    _check()


def test_per_1x128_emsort_tuned_row_is_not_used(tmp_path):
    cfg = tmp_path / "emsort_fmoe.csv"
    cfg.write_text(HEADER + "\n" + EMSORT_ROW + "\n")
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    repo = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        AITER_CONFIG_FMOE=str(cfg),
        TMPDIR=str(tmp),
        PYTHONPATH=f"{repo}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    )
    code = (
        "from op_tests.test_moe_blockscale_tuned_row_accuracy import _check; _check()"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr[-2000:]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
