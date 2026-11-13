# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The CPU-only prebuild: which modules a tier needs, once each, and what a lease may do."""

import pytest

from op_tests.moe_bench import prebuild, schema, workload


def _case(cid, family, token, d, i, e, topk, act, aq, wq, qt, extra=""):
    row = {c: "" for c in schema.COLS}
    row.update(
        case_id=cid, kind="fused", family=family, token=str(token), model_dim=str(d),
        inter_dim=str(i), expert=str(e), topk=str(topk), act_type=act, dtype="bf16",
        q_dtype_a=aq, q_dtype_w=wq, q_type=qt, use_g1u1="1", doweight_stage1="0",
        modes="warm", extra=extra,
    )  # fmt: skip
    return schema.Case(row, schema.parse_extra(extra))


FP8BLOCK = _case(
    "c-blk", "fp8-block", 8, 7168, 256, 256, 8, "Silu", "fp8", "fp8", "per_1x128"
)
FP8BLOCK2 = _case(
    "c-blk2", "fp8-block", 4096, 7168, 256, 256, 8, "Silu", "fp8", "fp8", "per_1x128"
)
PTPC = _case(
    "c-ptpc", "fp8-token", 8, 4096, 128, 513, 11, "Silu", "fp8", "fp8", "per_Token"
)
A16 = _case("c-a16", "a16w16", 8, 4096, 1024, 64, 4, "Silu", "bf16", "bf16", "No")
A8W4 = _case("c-a8w4", "a8w4", 256, 6144, 1536, 129, 5, "Swiglu", "fp8", "fp4x2", "per_1x32",
             "gate_mode=INTERLEAVE")  # fmt: skip
A4W4_SWIGLU = _case(
    "c-a4w4", "a4w4", 8, 6144, 384, 129, 5, "Swiglu", "fp4x2", "fp4x2", "per_1x32"
)


def test_modules_dedupe_across_a_batch_and_name_the_cases_that_need_them():
    need = prebuild.plan([FP8BLOCK, FP8BLOCK2, PTPC, A16])
    ck = [m for m in need if m.startswith("module_moe_ck2stages")]
    # two fp8-block cases share every module: each is built once
    both = [m for m in ck if "per_1x128" in m]
    assert both and all(set(need[m]["cases"]) == {"c-blk", "c-blk2"} for m in both)
    assert len(ck) == len(set(ck))
    assert not (set(prebuild.STATIC) - set(need))


def test_ck_variants_follow_the_case():
    need = prebuild.plan([FP8BLOCK, PTPC, A16])
    names = set(need)
    assert any(
        "f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2" in m for m in names
    )
    # per_1x128 fp8 with a hidden size that splits into 128-wide k tiles has split-k
    assert any(m.endswith("per_1x128_mulWeightStage2_splitk") for m in names)
    # without doweight_stage1 both stages load the same routed-weight-stage-2 module
    assert any(
        "f8_f8_preshuffle_on_b16_silu_per_token_mulWeightStage2" in m for m in names
    )
    dw = _case(
        "c-dw", "fp8-token", 8, 4096, 128, 513, 11, "Silu", "fp8", "fp8", "per_Token"
    )
    dw.row["doweight_stage1"] = "1"
    dw_names = set(prebuild.plan([dw]))
    assert any("per_token_mulWeightStage1" in m for m in dw_names)
    assert not any("per_token_mulWeightStage2" in m for m in dw_names)
    assert any("b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2" in m for m in names)
    # no per-token split-k, no a16 split-k
    assert not any("per_token" in m and "splitk" in m for m in names)


def test_cktile_for_a8w4_and_nothing_ck_for_mx_swiglu():
    need = prebuild.plan([A8W4])
    assert "module_moe_cktile2stages" in need
    # the only CK module is the one aiter's resolver loads for a CK-Tile stage 1 next to a
    # FlyDSL stage 2 (ck_moe_stage1 wraps the name); it cannot build, so the host prebuild
    # records the failure and those candidates are excluded
    assert [m for m in need if m.startswith("module_moe_ck2stages")] == [
        "module_moe_ck2stages_abi2_f8_fp4x2_preshuffle_on_b16_swiglu_per_1x32_mulWeightStage2"
    ]
    # CK 2-stage codegen has no SwiGLU for MX: the tuner offers no CK task, so no module
    need = prebuild.plan([A4W4_SWIGLU])
    assert not [m for m in need if m.startswith("module_moe_ck2stages")]


def test_a_leased_run_refuses_when_a_module_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(prebuild, "jit_dir", lambda: str(tmp_path))
    with pytest.raises(SystemExit) as e:
        prebuild.require_prebuilt([PTPC])
    assert "must not compile" in str(e.value)
    assert "module_moe_ck2stages" in str(e.value)
    # once every module exists, or is recorded as unbuildable, it does not refuse
    for md in prebuild.plan([PTPC]):
        (tmp_path / f"{md}.so").write_bytes(b"")
    prebuild.require_prebuilt([PTPC])
    (tmp_path / "module_moe_asm.so").unlink()
    (tmp_path / "prebuild_failed.json").write_text('{"module_moe_asm": "x"}')
    prebuild.require_prebuilt([PTPC])


def test_workers_cannot_compile_under_the_guard(monkeypatch):
    from aiter.jit import core

    monkeypatch.setattr(core, "build_module", core.build_module)
    monkeypatch.setenv(prebuild.NO_BUILD_ENV, "1")
    prebuild.forbid_builds()
    with pytest.raises(RuntimeError, match="build forbidden"):
        core.build_module(
            "module_x", [], [], [], "", [], None, False, True, False, False
        )


def test_children_build_with_the_gpu_hidden_and_the_arch_named():
    env = prebuild._child_env(16)
    # empty hides the GPU everywhere; "-1" aborts the runtime on TheRock wheels
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        assert env[var] == "", var
    assert env["GPU_ARCHS"].startswith("gfx") and env["MAX_JOBS"] == "16"
    assert prebuild.NO_BUILD_ENV not in env
    assert workload is not None


def test_only_compile_failures_are_remembered_and_built_modules_are_forgotten(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(prebuild, "jit_dir", lambda: str(tmp_path))
    need = prebuild.plan([PTPC])
    target = min(m for m in need if m.startswith("module_moe_ck2stages"))

    class P:
        def __init__(self, out):
            self.stdout, self.stderr = out, ""

    def fake_run(cmd, **kw):
        return P(kw_out["text"])

    monkeypatch.setattr(prebuild.subprocess, "run", fake_run)
    monkeypatch.setattr(prebuild, "plan", lambda cases: {target: need[target]})
    # an environment failure ("Sequence '-1' not recognized") is not recorded
    kw_out = {"text": "agent.cpp:208 Sequence '-1' not recognized"}
    res = prebuild.run_prebuild([], jobs=1, log=lambda *_: None)
    assert res["failed"] == [target] and prebuild.known_failed() == {}
    # a compile that ran and failed is
    kw_out = {"text": f"build [{target}] under /x failed !!!!!!"}
    prebuild.run_prebuild([], jobs=1, log=lambda *_: None)
    assert target in prebuild.known_failed()

    # --prebuild-retry tries it again; once it builds, the record is dropped
    def build_it(cmd, **kw):
        (tmp_path / f"{target}.so").write_bytes(b"")
        return P("")

    monkeypatch.setattr(prebuild.subprocess, "run", build_it)
    res = prebuild.run_prebuild([], jobs=1, log=lambda *_: None, retry_failed=True)
    assert res["built"] == [target] and prebuild.known_failed() == {}


def test_children_need_the_chip_identity_and_the_target_tree_first(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GPU_ARCHS", "gfx950")
    monkeypatch.setenv("CU_NUM", "256")
    env = prebuild._child_env(8, tree="/some/aiter", shim=str(tmp_path))
    assert env["GPU_ARCHS"] == "gfx950" and env["CU_NUM"] == "256"
    # the --aiter tree first, then the harness shim, as the runner's workers have it
    assert env["PYTHONPATH"].split(":")[:2] == ["/some/aiter", str(tmp_path)]


def test_without_a_readable_chip_prebuild_asks_for_arch_and_cu_num(monkeypatch):
    monkeypatch.delenv("GPU_ARCHS", raising=False)
    monkeypatch.delenv("CU_NUM", raising=False)
    import aiter.jit.utils.chip_info as chip

    def boom():
        raise RuntimeError("No gfx arch found in rocminfo output.")

    monkeypatch.setattr(chip, "get_gfx", boom)
    with pytest.raises(SystemExit, match="GPU_ARCHS.*CU_NUM"):
        prebuild.gpu_identity()


def test_use_tree_refuses_an_aiter_already_imported_from_elsewhere(tmp_path):
    with pytest.raises(SystemExit, match="already imported"):
        prebuild.use_tree(str(tmp_path))


def test_children_follow_the_aiter_tree_whatever_the_cwd(monkeypatch, tmp_path):
    """Run from a harness checkout that is itself a full aiter tree, a child's cwd leads
    its sys.path: its own aiter, its JIT dir and its missing CK submodule. The children
    run in the --aiter tree and are told the JIT dir the parent resolved."""
    tree = tmp_path / "target-aiter"
    tree.mkdir()
    jit = tmp_path / "jit"
    jit.mkdir()
    monkeypatch.setattr(prebuild, "jit_dir", lambda: str(jit))
    monkeypatch.setattr(prebuild, "use_tree", lambda t: str(t))
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)
    need = prebuild.plan([PTPC])
    target = min(m for m in need if m.startswith("module_moe_ck2stages"))
    seen = []

    class P:
        stdout, stderr = "", ""

    def fake_run(cmd, **kw):
        seen.append(kw)
        return P()

    monkeypatch.setattr(prebuild.subprocess, "run", fake_run)
    monkeypatch.setattr(prebuild, "plan", lambda cases: {target: need[target]})
    prebuild.run_prebuild([], jobs=1, log=lambda *_: None, tree=str(tree))
    assert seen and all(kw["cwd"] == str(tree) for kw in seen)
    assert all(kw["env"]["AITER_JIT_DIR"] == str(jit) for kw in seen)
    assert seen[0]["env"]["PYTHONPATH"].split(":")[0] == str(tree)
    seen.clear()
    prebuild.prebuild_probe(log=lambda *_: None, tree=str(tree))
    assert seen and seen[0]["cwd"] == str(tree)
    assert seen[0]["env"]["AITER_JIT_DIR"] == str(jit)


def test_a_candidate_that_would_compile_is_skipped_logged_and_recorded(
    monkeypatch, tmp_path
):
    from op_tests.moe_bench import bar_worker

    monkeypatch.setattr(prebuild, "jit_dir", lambda: str(tmp_path))
    status, detail, outcome = bar_worker.not_prebuilt_row("module_moe_x")
    assert (status, outcome) == ("not-prebuilt", "notprebuilt")
    assert detail == "not prebuilt: module_moe_x"
    assert (
        "module_moe_x" in prebuild.read_missing_record()
    )  # the next prebuild builds it
    # a module that failed to build on the CPU carries its recorded error
    (tmp_path / "prebuild_failed.json").write_text(
        '{"module_moe_y": "compile error z"}'
    )
    status, detail, outcome = bar_worker.not_prebuilt_row("module_moe_y")
    assert status == "build_failed" and "compile error z" in detail
    assert "module_moe_y" not in prebuild.read_missing_record()


def test_the_build_guard_names_the_module_even_after_a_wrapper_reraised_it():
    e = prebuild.NotPrebuilt("module_moe_ck2stages_abi2_f8")
    assert prebuild.not_prebuilt_module(e) == "module_moe_ck2stages_abi2_f8"
    wrapped = RuntimeError(
        str(e)
    )  # a torch custom-op wrapper re-raises as RuntimeError
    assert prebuild.not_prebuilt_module(wrapped) == "module_moe_ck2stages_abi2_f8"
    assert prebuild.not_prebuilt_module(RuntimeError("something else")) is None


def test_flydsl_run_only_alone_turns_builds_into_errors(monkeypatch):
    from aiter.jit import core

    real = core.build_module
    monkeypatch.delenv(prebuild.NO_BUILD_ENV, raising=False)
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    try:
        prebuild.forbid_builds()
        with pytest.raises(prebuild.NotPrebuilt):
            core.build_module(
                "module_x", [], [], [], "", [], None, False, True, False, False
            )
    finally:
        core.build_module = real


def test_the_plan_includes_the_default_route_and_candidate_modules(monkeypatch):
    """A forced row aiter declines falls back to its default route; those modules are
    planned too, as are the CK modules the enumeration's own candidates name."""
    mods = prebuild.modules_from_names(
        PTPC_S(), "moe_ck2stages_gemm1_256x32x64x128_1x4_v1", "moe_ck2stages_gemm2_x"
    )
    assert any("per_token" in m for m in mods)
    ct = prebuild.modules_from_names(PTPC_S(), "cktile_a8w4_bm32", "cktile_a8w4_bm32")
    assert set(ct) == {"module_moe_cktile2stages"}
    assert prebuild.modules_from_names(PTPC_S(), "flydsl_moe1_x", "flydsl_moe2_y") == {}
    sk = prebuild.modules_from_names(PTPC_S(), "moe_ck2stages_gemm1_Splitk_x", "")
    assert any(m.endswith("_splitk") for m in sk)


def PTPC_S():
    from op_tests.moe_bench import workload

    return workload.resolve(PTPC)


def test_modules_are_read_off_the_resolved_stage_functions_not_the_names():
    """A CK-Tile stage 1 next to a FlyDSL stage 2 is wrapped by aiter's resolver in
    ck_moe_stage1 (the flydsl/opus branch treats every other stage-1 name as CK), so the
    CK swiglu module, which cannot build, is what a call loads."""
    import functools
    import types

    import aiter
    import aiter.fused_moe as fm

    S = workload.resolve(A8W4)
    md = types.SimpleNamespace(
        stage1=functools.partial(
            fm.ck_moe_stage1, kernelName="cktile_a8w4_bm32", splitk=0
        ),
        stage2=functools.partial(functools.partial(fm._flydsl_stage2_wrapper)),
    )
    mods = prebuild.modules_from_metadata(S, md)
    assert list(mods) == [
        "module_moe_ck2stages_abi2_f8_fp4x2_preshuffle_on_b16_swiglu_per_1x32_mulWeightStage2"
    ]
    # the same pair resolved honestly (both CK-Tile) loads the CK-Tile module only
    md = types.SimpleNamespace(
        stage1=functools.partial(fm.cktile_moe_stage1),
        stage2=functools.partial(fm.cktile_moe_stage2),
    )
    assert list(prebuild.modules_from_metadata(S, md)) == ["module_moe_cktile2stages"]
    # a CK stage 2 loads its own module; split-k only follows the name
    md = types.SimpleNamespace(
        stage1=functools.partial(fm.ck_moe_stage1, kernelName="x_Splitk_y"),
        stage2=functools.partial(aiter.ck_moe_stage2_fwd, kernelName="y"),
    )
    got = prebuild.modules_from_metadata(workload.resolve(FP8BLOCK), md)
    assert any(m.endswith("_splitk") for m in got) and len(got) == 2


def test_every_build_goes_through_one_hook_that_records_module_spec_and_candidate(
    monkeypatch, tmp_path
):
    from aiter.jit import core

    monkeypatch.setattr(prebuild, "jit_dir", lambda: str(tmp_path))
    monkeypatch.setenv(prebuild.NO_BUILD_ENV, "1")
    real = core.build_module
    try:
        prebuild.forbid_builds()
        prebuild.CURRENT["candidate"] = "c1|fused_moe|cktile_a8w4_bm32|flydsl_x"
        with pytest.raises(prebuild.NotPrebuilt):
            core.build_module(
                "module_moe_ck2stages_abi2_f8_fp4x2_x", ["s.cpp"], [], [], ["gen -w {}"]
            )
        with pytest.raises(prebuild.NotPrebuilt):  # a build entry that is not a CK one
            core.build_module("module_odd", ["s.cpp"], [], [], [])
    finally:
        core.build_module = real
        prebuild.CURRENT["candidate"] = ""
    rec = prebuild.read_missing_record()
    ck = rec["module_moe_ck2stages_abi2_f8_fp4x2_x"]
    assert ck["blob_gen_cmd"] == ["gen -w {}"] and ck["kind"] == "ck2stages"
    assert ck["candidates"] == ["c1|fused_moe|cktile_a8w4_bm32|flydsl_x"]
    assert rec["module_odd"]["kind"] == "static"
    # the next prebuild builds what a lease asked for, named by the plan or not
    assert "module_odd" in prebuild.plan([])
    assert prebuild.plan([])["module_moe_ck2stages_abi2_f8_fp4x2_x"]["blob_gen_cmd"]


def test_candidates_whose_module_failed_on_the_host_are_build_failed_before_any_lease(
    monkeypatch, tmp_path
):
    import io
    import json
    import types

    from op_tests.moe_bench import bar_worker

    monkeypatch.setattr(prebuild, "jit_dir", lambda: str(tmp_path))
    bad_md = "module_moe_ck2stages_abi2_f8_fp4x2_swiglu"
    mk = lambda i, s1, s2: types.SimpleNamespace(
        id=i, kernels=f"{s1} + {s2}", group="pair-bm32-0-0-0", meta={"s1": s1, "s2": s2}
    )
    cands = [
        mk("a", "cktile_a8w4_bm32", "flydsl_moe2_x"),
        mk("b", "cktile_a8w4_bm32", "flydsl_moe2_y"),
        mk("c", "flydsl_moe1_x", "flydsl_moe2_x"),
        mk("d", "cktile_a8w4_bm32", "cktile_a8w4_bm32"),
    ]
    keymods = {}
    for c in cands:
        keymods[prebuild.cand_key(c)] = (
            [bad_md] if c.id in ("a", "b") else ["module_moe_cktile2stages"]
        )
    prebuild.record_keymods("case1", keymods)
    assert prebuild.cand_key(cands[0]) == prebuild.cand_key(cands[1])
    assert prebuild.excluded_keys("case1") == {}  # nothing failed yet
    (tmp_path / "prebuild_failed.json").write_text(
        json.dumps({bad_md: "static assertion: swiglu is act 3"})
    )
    fh = io.StringIO()
    keep = bar_worker.drop_build_failed(fh, "case1", "fused_moe", "warm", cands)
    assert [c.id for c in keep] == ["c", "d"]
    ev = [json.loads(line) for line in fh.getvalue().splitlines()]
    rows = [e for e in ev if e["event"] == "cand"]
    assert {e["cand_id"] for e in rows} == {"a", "b"}
    assert all(
        e["status"] == "build_failed" and "static assertion" in e["detail"]
        for e in rows
    )
    # another case, or another source, is untouched
    assert (
        bar_worker.drop_build_failed(fh, "case2", "fused_moe", "warm", cands) == cands
    )


def test_flydsl_stage1_kernels_aiter_rejects_for_the_shape_are_unsupported_not_failures():
    from op_tests.moe_bench.candidates.fused_cfg import flydsl_stage1_unsupported as f

    kw4 = "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w2_kw4_fp4"
    assert f(kw4, 3584) == "K per group=896 not divisible by tile_k=256"
    assert f(kw4, 2560) == "K per group=640 not divisible by tile_k=256"
    assert f(kw4, 4096) is None  # 1024 is whole tiles
    assert f(kw4, 3586) == "model_dim=3586 not divisible by k_wave=4"
    assert f("flydsl_moe1_afp4_wfp4_bf16_t32x32x256_bnt0_gui", 3584) is None
    assert f("flydsl_moe2_afp4_wfp4_bf16_t32x128x128_atomic", 3584) is None  # stage 2
    assert f("moe_ck2stages_gemm1_x", 3584) is None
