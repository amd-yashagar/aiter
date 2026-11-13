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
    ok = {
        "built": 5, "candidates": 5, "forced_verified": prebuild.FORCED_VERIFIED,
        "verify_tried": 3, "verify_pool": 10, "n_misses": 0,
    }  # fmt: skip
    assert prebuild.flydsl_covered(ok)
    assert not prebuild.flydsl_covered({"built": 5, "candidates": 5})  # an older build
    assert not prebuild.flydsl_covered(dict(ok, built=0))
    assert not prebuild.flydsl_covered(None)
    assert prebuild.flydsl_covered(dict(ok, built=0, candidates=0))


def _pair(i, s1, s2, group="pair-bm32-0-0-0"):
    import types

    return types.SimpleNamespace(
        id=f"{i}", kernels=f"{s1} + {s2}", group=group, meta={"s1": s1, "s2": s2}
    )


S1 = [
    "flydsl_moe1_afp8_wfp4_bf16_t32x32x256_w2_gui",
    "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w2_kw2_gui",
    "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w2_kw4_gui_fp8",
]
S2_LAYOUT = [
    "flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_nt_sbm32",
    "flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x128_atomic_nt_sbm32",
]
S2_PLAIN = [
    "flydsl_moe2_afp8_wfp4_bf16_t16x128x128_atomic",
    "flydsl_moe2_afp8_wfp4_bf16_t16x128x256_atomic",
]


def _group():
    out = []
    for a in S1:
        for b in S2_LAYOUT + S2_PLAIN:
            out.append(_pair(len(out), a, b))
    return out


def test_a_stage_1_kernel_is_covered_per_class_of_its_partner_stage_2():
    """00166: the stage-1 gui kernels next to moe2_layout stage 2 were never built, the
    cover had picked each of them a partner of the other class."""
    pytest.importorskip("aiter")
    cands = _group()
    cover = flydsl_prebuild.covering_pairs(cands)
    assert len(cover) < len(cands)
    covered = {e for c in cover for e in flydsl_prebuild.elements(c)}
    assert flydsl_prebuild.uncovered(cands, covered) == []
    # every stage-1 kernel has a layout partner and a plain partner in the cover
    for s1 in S1:
        partners = {c.meta["s2"] for c in cover if c.meta["s1"] == s1}
        assert partners & set(S2_LAYOUT) and partners & set(S2_PLAIN), s1
    # a cover that ignored the class (one partner per stage kernel) leaves candidates out
    naive = [c for c in cands if c.meta["s2"] in (S2_PLAIN[0], S2_LAYOUT[0])][:3]
    naive_cov = {e for c in naive for e in flydsl_prebuild.elements(c)}
    assert flydsl_prebuild.uncovered(cands, naive_cov)


def test_the_fused_quant_of_stage_1_is_a_class_of_stage_2():
    pytest.importorskip("aiter")
    assert flydsl_prebuild.stage1_class(S1[2]) != flydsl_prebuild.stage1_class(S1[0])
    assert flydsl_prebuild.stage2_class(S2_LAYOUT[0]).startswith("v2:")
    assert flydsl_prebuild.stage2_class(S2_PLAIN[0]) != flydsl_prebuild.stage2_class(
        S2_LAYOUT[0]
    )


def test_a_lease_reports_candidates_the_cover_lacks_as_aot_missing(
    monkeypatch, tmp_path
):
    import io

    from op_tests.moe_bench import bar_worker

    pytest.importorskip("aiter")
    monkeypatch.setattr(prebuild, "jit_dir", lambda: str(tmp_path))
    monkeypatch.setattr(prebuild, "flydsl_cache_dir", lambda: str(tmp_path / "fly"))
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    cands = _group()
    keep = [c for c in cands if c.meta["s2"] not in S2_LAYOUT]  # a cover without layout
    covered = sorted({e for c in keep for e in flydsl_prebuild.elements(c)})
    os.makedirs(tmp_path / "fly")
    (tmp_path / "fly" / "prebuilt.json").write_text(
        json.dumps({"case1": {"covered": covered}})
    )
    fh = io.StringIO()
    kept, n = bar_worker.drop_aot_uncovered(fh, "case1", "fused_moe", "warm", cands)
    assert n == len(cands) - len(keep) and {c.id for c in kept} == {c.id for c in keep}
    rows = [json.loads(x) for x in fh.getvalue().splitlines()]
    assert all(r["status"] == "aot-missing" for r in rows) and len(rows) == n
    # not a lease, or no cover recorded: nothing is dropped
    monkeypatch.delenv("FLYDSL_RUNTIME_RUN_ONLY")
    monkeypatch.delenv("MOE_BENCH_NO_BUILD", raising=False)
    assert (
        bar_worker.drop_aot_uncovered(fh, "case1", "fused_moe", "warm", cands)[1] == 0
    )


def test_a_candidate_that_hits_no_usable_aot_cache_makes_the_bar_incomplete():
    from op_tests.moe_bench import bar_worker

    stats = {"fused_moe:flydsl": {"total": 30, "ok": 20, "aotmiss": 10}}
    _, why = bar_worker._accounting(
        {"fused_moe:flydsl": 30, "fused_moe:*": 30}, stats, [], "bar"
    )
    assert "10 candidates hit no usable AOT cache" in why and "harness error" in why


def test_the_entry_needs_a_clean_run_only_check_to_count_as_covered():
    ok = {
        "built": 5, "candidates": 5, "forced_verified": prebuild.FORCED_VERIFIED,
        "verify_tried": 40, "verify_pool": 900, "n_misses": 0,
    }  # fmt: skip
    assert prebuild.flydsl_covered(ok)
    assert not prebuild.flydsl_covered(dict(ok, n_misses=2))
    assert not prebuild.flydsl_covered(dict(ok, verify_tried=0))
    assert prebuild.flydsl_covered(dict(ok, verify_tried=0, verify_pool=0))
    assert not prebuild.flydsl_covered({k: v for k, v in ok.items() if k != "n_misses"})


COVER_CHILD = """
import json, sys, os
from op_tests.moe_bench import flydsl_prebuild as fp, runner
case = runner.select_cases([sys.argv[1]])[0]
if sys.argv[2] == "build":
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    from op_tests.moe_bench import workload
    _c, cover = fp.enumerate_flydsl(workload.resolve(case), case, os.environ["TMPDIR"], get_gfx(), get_cu_num())
    # the whole-graph singles trace into a segfault on this node; the pairs are the point
    pairs = [i for i, c in enumerate(cover) if c.group.startswith("pair")]
    res = fp.build_slice(case, 0, 1, os.environ["TMPDIR"], indices=pairs)
    print("RESULT " + json.dumps({"built": sum(1 for r in res if r[1]), "bad": [r for r in res if not r[1]]}))
else:
    print("RESULT " + json.dumps(fp.verify_sample(case, os.environ["TMPDIR"], 12)))
"""


def test_the_cover_leaves_nothing_a_run_only_run_of_the_other_candidates_misses(
    tmp_path,
):
    """The host build of the cover, then FlyDSL's run-only mode over candidates the cover
    did not build: each finds its artifact (nothing compiles in a lease)."""
    pytest.importorskip("aiter")
    tier = tmp_path / "t.csv"
    case = _case(5)
    schema.write_tier(str(tier), [case.row])
    cache = tmp_path / "flycache"

    def run(mode, run_only):
        env = dict(
            os.environ, PYTHONPATH=HERE, TMPDIR=str(tmp_path),
            MOE_BENCH_STATE_DIR=str(tmp_path / "state"), HIP_VISIBLE_DEVICES="",
            ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", GPU_ARCHS="gfx942",
            CU_NUM="304", COMPILE_ONLY="1", FLYDSL_GPU_ARCH="gfx942",
            AITER_AOT_REPLAY="1", FLYDSL_RUNTIME_CACHE_DIR=str(cache),
        )  # fmt: skip
        if run_only:
            env["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
        r = subprocess.run(
            [sys.executable, "-c", COVER_CHILD, str(tier), mode],
            env=env, cwd=HERE, capture_output=True, text=True, timeout=1500, check=False,
        )  # fmt: skip
        line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
        assert line, r.stderr[-1500:]
        return json.loads(line[7:])

    built = run("build", run_only=False)
    assert built["built"] > 0 and not built["bad"], built
    checked = run("verify", run_only=True)
    assert checked["verify_tried"] > 0, checked
    assert checked["n_misses"] == 0, checked["misses"]


WG_CHILD = """
import json, sys, os
from op_tests.moe_bench import flydsl_prebuild as fp, runner, workload
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
case = runner.select_cases([sys.argv[1]])[0]
_c, cover = fp.enumerate_flydsl(workload.resolve(case), case, os.environ["TMPDIR"], get_gfx(), get_cu_num())
whole = [i for i, c in enumerate(cover) if not c.group.startswith("pair")]
if sys.argv[2] == "n":
    print("RESULT", len(whole))
else:
    k = whole[int(sys.argv[2])]
    print("RESULT", json.dumps(fp.build_slice(case, 0, 1, os.environ["TMPDIR"], indices=[k])))
"""


def test_the_gfx942_whole_graph_kernels_are_host_built_and_found_run_only(tmp_path):
    """tier1-00209 failed in a lease with 'no usable AOT cache for launch_splitk' and
    'for launch_prefill_1x4': the whole-graph candidates compile their launchers on the
    host (one process each: their prefill tracing can crash a slice)."""
    pytest.importorskip("aiter")
    row = {c: "" for c in schema.COLS}
    row.update(
        case_id="wg-33", kind="fused", family="a16w16", token="33", model_dim="256",
        inter_dim="192", expert="8", topk="2", act_type="Silu", dtype="bf16",
        q_dtype_a="bf16", q_dtype_w="bf16", q_type="No", use_g1u1="1",
        doweight_stage1="0", modes="warm", extra="bias=none",
    )  # fmt: skip
    tier = tmp_path / "t.csv"
    schema.write_tier(str(tier), [row])
    cache = tmp_path / "flycache"

    def run(arg, run_only):
        env = dict(
            os.environ, PYTHONPATH=HERE, TMPDIR=str(tmp_path),
            MOE_BENCH_STATE_DIR=str(tmp_path / "state"), HIP_VISIBLE_DEVICES="",
            ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", GPU_ARCHS="gfx942",
            CU_NUM="304", COMPILE_ONLY="1", FLYDSL_GPU_ARCH="gfx942",
            AITER_AOT_REPLAY="1", FLYDSL_RUNTIME_CACHE_DIR=str(cache),
        )  # fmt: skip
        if run_only:
            env["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
        r = subprocess.run(
            [sys.executable, "-c", WG_CHILD, str(tier), str(arg)],
            env=env, cwd=HERE, capture_output=True, text=True, timeout=900, check=False,
        )  # fmt: skip
        line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
        assert line, r.stderr[-1000:]
        return json.loads(line[7:])

    n = run("n", False)
    assert n >= 2
    for k in range(n):
        res = run(k, False)
        assert all(ok for _c, ok, _e in res), res
    launchers = {d for d in os.listdir(cache) if os.listdir(cache / d)}
    assert {"launch_splitk", "launch_prefill_1x4"} <= launchers, launchers
    for k in range(n):  # the lease: run-only finds every artifact
        res = run(k, True)
        assert all(ok for _c, ok, _e in res), res
