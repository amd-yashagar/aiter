# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The pipeline driver: batches of whole weight groups sized for a lease."""

from op_tests.moe_bench import pipeline, schema


def _case(cid, family, token, d, i, e, qt="per_Token", modes="warm,cold"):
    row = {c: "" for c in schema.COLS}
    row.update(
        case_id=cid, kind="fused", family=family, token=str(token), model_dim=str(d),
        inter_dim=str(i), expert=str(e), topk="4", act_type="Silu", dtype="bf16",
        q_dtype_a="fp8", q_dtype_w="fp8", q_type=qt, use_g1u1="1", doweight_stage1="0",
        modes=modes,
    )  # fmt: skip
    return schema.Case(row, {})


def test_batches_hold_whole_weight_groups_near_the_budget():
    # three weight groups (different shapes), two cases each
    cases = []
    for g, (d, i) in enumerate([(4096, 128), (4096, 256), (7168, 256)]):
        for tok in (8, 64):
            cases.append(_case(f"g{g}-{tok}", "fp8-token", tok, d, i, 64))
    # 100 rows x 1.0 s x 2 modes = 200 s per case, 400 s per group
    batches = pipeline.make_batches(cases, minutes=14, history={}, sec_per_row=1.0)
    sizes = [len(cs) for cs, _ in batches]
    assert sizes == [
        4,
        2,
    ], sizes  # 1200 s would pass the 840 s budget: the third group starts a batch
    for cs, _ in batches:  # a group is never split
        groups = {schema.weight_group_key(c) for c in cs}
        assert all(
            sum(1 for c in cs if schema.weight_group_key(c) == k) == 2 for k in groups
        )
    assert pipeline.make_batches(cases, 0.1, {}, 1.0)[0][0][0].id == "g0-8"
    assert [len(cs) for cs, _ in pipeline.make_batches(cases, 0.1, {}, 1.0)] == [
        2,
        2,
        2,
    ]


def test_history_replaces_the_default_row_count_by_family(tmp_path):
    p = tmp_path / "candidates.csv"
    p.write_text(
        "case_id,family\n"
        + "".join("a,fp8-token\n" for _ in range(40))
        + "".join("b,fp8-token\n" for _ in range(60))
    )
    h = pipeline.history_rows(str(p))
    assert h == {"fp8-token": 50}
    c = _case("x", "fp8-token", 8, 4096, 128, 64)
    assert pipeline.estimate_seconds(c, h, 2.0) == 50 * 2.0 * 2
    assert pipeline.estimate_seconds(c, {}, 2.0) == 100 * 2.0 * 2
    mx = _case("y", "a4w4", 8, 4096, 128, 64, "per_1x32", "warm")
    assert pipeline.estimate_seconds(mx, {}, 1.0) == 600


def test_commands_prebuild_the_next_batch_on_the_cpu_and_run_the_current_one_leased(
    tmp_path,
):
    class A:
        aiter, out_dir, tier_name = "/t", "/o", "tier1"
        min_free_gb, prebuild_jobs, prebuild_cores, prebuild_retry = 12.0, 2, 32, True

    pre = pipeline.prebuild_cmd(A, "b1.csv")
    assert pre[3:5] == ["run", "--mode"] and "prebuild" in pre
    assert pre[pre.index("--cases") + 1] == "b1.csv"
    assert "--prebuild-retry" in pre and pre[pre.index("--prebuild-cores") + 1] == "32"
    run = pipeline.run_cmd(A, "b0.csv", ["--launcher", "gpuq", "--screen", "two-phase"])
    assert "bar" in run and "--resume" in run
    assert run[run.index("--launcher") + 1] == "gpuq" and "--screen" in run
    assert run[run.index("--cases") + 1] == "b0.csv"


def test_plan_only_writes_the_batches(tmp_path, capsys):
    t = tmp_path / "tier.csv"
    cases = [_case(f"c{i}", "fp8-token", 8, 4096, 128 + 128 * i, 64) for i in range(5)]
    schema.write_tier(str(t), [dict(c.row) for c in cases])
    rc = pipeline.main(
        [
            "--aiter", str(tmp_path), "--cases", str(t), "--out-dir", str(tmp_path / "o"),
            "--tier-name", "t", "--batch-minutes", "7", "--sec-per-row", "1.0",
            "--plan-only",
        ]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0 and "batch 000" in out and "batch 001" in out
    assert (tmp_path / "o" / "pipeline" / "batch-000.csv").exists()


def test_covering_pairs_compile_every_stage_kernel_once_not_the_product():
    import types

    from op_tests.moe_bench import flydsl_prebuild as fp

    def pair(g, a, b):
        return types.SimpleNamespace(
            id=f"{g}|{a}|{b}", group=g, kernels="flydsl_x", meta={"s1": a, "s2": b}
        )

    cands = [pair("pair-bm32", f"a{i}", f"b{j}") for i in range(6) for j in range(11)]
    cands += [pair("pair-bm64", f"a{i}", f"b{j}") for i in range(3) for j in range(4)]
    single = types.SimpleNamespace(
        id="g", group="whole-graph", kernels="impl__flydsl", meta={}
    )
    keep = fp.covering_pairs(cands + [single])
    bm32 = [c for c in keep if c.group == "pair-bm32"]
    bm64 = [c for c in keep if c.group == "pair-bm64"]
    assert single in keep
    assert len(bm32) == 11 and len(bm64) == 4  # max(n1, n2), not 66 and 12
    for lst, n1, n2 in ((bm32, 6, 11), (bm64, 3, 4)):
        assert len({c.meta["s1"] for c in lst}) == n1
        assert len({c.meta["s2"] for c in lst}) == n2
