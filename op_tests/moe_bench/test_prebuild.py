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
    assert not [m for m in need if m.startswith("module_moe_ck2stages")]
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
