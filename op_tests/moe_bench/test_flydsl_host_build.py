# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The FlyDSL host build must really compile the forced stage pairs (no GPU), so a leased
run with FLYDSL_RUNTIME_RUN_ONLY=1 compiles nothing."""

import json
import os
import subprocess
import sys

import pytest

from op_tests.moe_bench import flydsl_prebuild, prebuild, schema, workload

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _case(token):
    row = {c: "" for c in schema.COLS}
    row.update(
        case_id=f"fly-{token}", kind="fused", family="a16w16", token=str(token),
        model_dim="7168", inter_dim="256", expert="257", topk="9", act_type="Silu",
        dtype="bf16", q_dtype_a="bf16", q_dtype_w="bf16", q_type="No", use_g1u1="1",
        doweight_stage1="0", modes="warm", extra="bias=none",
    )  # fmt: skip
    return schema.Case(row, schema.parse_extra(row["extra"]))


def test_the_forced_row_is_looked_up_by_the_padded_token_count():
    """get_2stage_cfgs is keyed by get_padded_M(M): a one-row config for M=5 never matched
    (the router took its default route) and the host build compiled nothing for the pair.
    """
    pytest.importorskip("aiter")
    for m, padded in ((5, 8), (8, 8), (333, 512), (4096, 4096), (1, 1), (129, 256)):
        S = workload.resolve(_case(m))
        assert flydsl_prebuild.dispatch_args_of(S)["token"] == padded, m


CHILD = """
import json, sys, os
from op_tests.moe_bench import flydsl_prebuild as fp, runner
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
case = runner.select_cases([sys.argv[1]])[0]
from op_tests.moe_bench import workload
S = workload.resolve(case)
ctx, cands = fp.enumerate_flydsl(S, case, os.environ["TMPDIR"], get_gfx(), get_cu_num())
pairs = [i for i, c in enumerate(cands) if c.group.startswith("pair")][:2]
print("RESULT " + json.dumps(fp.build_slice(case, 0, 1, os.environ["TMPDIR"], pairs)))
"""


def _child(tmp_path, tier, cache, run_only):
    env = dict(
        os.environ,
        PYTHONPATH=HERE,
        TMPDIR=str(tmp_path),
        AITER_JIT_DIR=os.environ.get("AITER_JIT_DIR", ""),
        MOE_BENCH_STATE_DIR=str(tmp_path / "state"),
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        CUDA_VISIBLE_DEVICES="",
        GPU_ARCHS="gfx942",
        CU_NUM="304",
        COMPILE_ONLY="1",
        FLYDSL_GPU_ARCH="gfx942",
        AITER_AOT_REPLAY="1",
        FLYDSL_RUNTIME_CACHE_DIR=str(cache),
    )
    if run_only:
        env["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", CHILD, str(tier)],
        env=env, cwd=HERE, capture_output=True, text=True, timeout=900, check=False,
    )  # fmt: skip
    line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
    assert line, r.stderr[-1500:]
    return json.loads(line[7:])


def test_a_host_built_pair_leaves_artifacts_and_a_run_only_run_compiles_nothing(
    tmp_path,
):
    pytest.importorskip("aiter")
    tier = tmp_path / "t.csv"
    case = _case(5)  # M=5: padded to 8, the case that built nothing
    schema.write_tier(str(tier), [case.row])
    cache = tmp_path / "flycache"
    built = _child(tmp_path, tier, cache, run_only=False)
    assert built and all(ok for _cid, ok, _err in built), built
    artifacts = [f for _r, _d, fs in os.walk(cache) for f in fs if f.endswith(".pkl")]
    assert artifacts, "the host build compiled nothing for the forced pairs"
    # the lease: same cache, FlyDSL run-only: every artifact is found, nothing compiles
    leased = _child(tmp_path, tier, cache, run_only=True)
    assert all(ok for _cid, ok, _err in leased), leased
    after = [f for _r, _d, fs in os.walk(cache) for f in fs if f.endswith(".pkl")]
    assert sorted(after) == sorted(artifacts)


def test_a_manifest_entry_without_verified_forced_rows_does_not_count_as_covered():
    ok = {"built": 5, "candidates": 5, "forced_verified": prebuild.FORCED_VERIFIED}
    assert prebuild.flydsl_covered(ok)
    assert not prebuild.flydsl_covered({"built": 5, "candidates": 5})  # an older build
    assert not prebuild.flydsl_covered(
        {"built": 0, "candidates": 5, "forced_verified": prebuild.FORCED_VERIFIED}
    )
    assert not prebuild.flydsl_covered(None)
    assert prebuild.flydsl_covered(
        {"built": 0, "candidates": 0, "forced_verified": prebuild.FORCED_VERIFIED}
    )
