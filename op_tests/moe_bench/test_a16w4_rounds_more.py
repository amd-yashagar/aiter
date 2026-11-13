# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""a16w4 SEPARATED Swiglu from M 256: aiter's dispatch flips the activations to fp4x2, and
enumerating from the recorded dispatch arguments gave the a4w4 registry (the a16w4 FlyDSL
pairs never entered the bar). The candidates are the requested dtype's, run under the raised
activation-dtype bound."""

import json
import os
import subprocess
import sys

import pytest

from op_tests.moe_bench import schema

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = """
import json, os, sys
from op_tests.moe_bench import flydsl_prebuild as fp, runner, workload, routes
from op_tests.moe_bench.candidates.fused_cfg import FusedCfgSource
import aiter.fused_moe as fm
case = runner.select_cases([sys.argv[1]])[0]
S = workload.resolve(case)
out = {}
ctx = fp.make_ctx(S, case, os.environ["TMPDIR"], "gfx950", 256)
out["rounds_more"] = bool(ctx.dispatch_rounds_more)
out["dispatch_q"] = str(ctx.dispatch_args["q_dtype_a"])
cands = FusedCfgSource().enumerate(ctx)
pairs = [c for c in cands if c.group.startswith("pair")]
names = " ".join(c.kernels for c in pairs)
out["n_pairs"] = len(pairs)
out["a4w4"] = "afp4_wfp4" in names
out["a16w4"] = "abf16_wfp4" in names
# the old behaviour: enumerating as dispatch resolved it
ctx.dispatch_rounds_more = ""
old = " ".join(c.kernels for c in FusedCfgSource().enumerate(ctx) if c.group.startswith("pair"))
out["old_a4w4"] = "afp4_wfp4" in old
ctx.dispatch_rounds_more = "x"
c = pairs[len(pairs) // 2]
before = fm._SWIGLU_MXFP4_BF16_BOUND
c.prepare(ctx)
row = open(os.environ["AITER_CONFIG_FMOE"]).read().splitlines()
out["row_q"] = row[1].split(",")[9]
out["bound_raised"] = fm._SWIGLU_MXFP4_BF16_BOUND > 10**12
r = routes.dry_route(case, S)
out["verify"] = c.verify(r)
out["resolved"] = [r["kernel1"], r["kernel2"]]
c.release(ctx)
out["bound_restored"] = fm._SWIGLU_MXFP4_BF16_BOUND == before
print("RESULT " + json.dumps(out))
"""


def test_a16w4_at_m256_enumerates_a16w4_pairs_and_forces_them_under_the_raised_bound(
    tmp_path,
):
    pytest.importorskip("aiter")
    row = {c: "" for c in schema.COLS}
    row.update(
        case_id="a16-256", kind="fused", family="a16w4", token="256", model_dim="6144",
        inter_dim="768", expert="128", topk="4", act_type="Swiglu", dtype="bf16",
        q_dtype_a="bf16", q_dtype_w="fp4x2", q_type="per_1x32", use_g1u1="1",
        doweight_stage1="0", modes="warm", extra="gate_mode=SEPARATED;bias=s1+s2",
    )  # fmt: skip
    tier = tmp_path / "t.csv"
    schema.write_tier(str(tier), [row])
    env = dict(
        os.environ, PYTHONPATH=HERE, TMPDIR=str(tmp_path),
        MOE_BENCH_STATE_DIR=str(tmp_path / "state"), HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", GPU_ARCHS="gfx950",
        CU_NUM="256", AITER_AOT_REPLAY="1",
    )  # fmt: skip
    r = subprocess.run(
        [sys.executable, "-c", CHILD, str(tier)],
        env=env, cwd=HERE, capture_output=True, text=True, timeout=900, check=False,
    )  # fmt: skip
    line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
    assert line, r.stderr[-1500:]
    out = json.loads(line[7:])
    assert out["rounds_more"] and "float4" in out["dispatch_q"]  # dispatch flipped it
    assert out["old_a4w4"], "the premise: enumerating as dispatch resolved gives a4w4"
    assert out["n_pairs"] > 0 and out["a16w4"] and not out["a4w4"], out
    assert out["row_q"] == "torch.bfloat16" and out["bound_raised"], out
    assert out["verify"] is None, out  # the router took the forced a16w4 pair
    assert out["bound_restored"], out
