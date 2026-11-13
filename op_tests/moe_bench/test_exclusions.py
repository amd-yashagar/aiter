# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MOEBUG-064: tile_n=64 ASM fp8 stage-1 kernels and tainted stage-1 kernels never set a bar."""

import io
import json
from types import SimpleNamespace

from op_tests.moe_bench import bar_runner, bar_worker, exclusions

T64 = "_ZN5aiter44fmoe_stage1_bf16_pertokenFp8_g1u1_32x64_4tg_pf3E"
T64B = "_ZN5aiter41fmoe_stage1_bf16_pertokenFp8_g1u1_128x64_pf3E"
OK1 = "_ZN5aiter45fmoe_stage1_bf16_pertokenFp8_g1u1_128x128_pf2E"
OK2 = "_ZN5aiter44fmoe_stage1_bf16_pertokenFp8_g1u1_32x512_pf2E"
CK = "moe_ck2stages_gemm1_256x32x64x128_1x4_MulABScale_v1_Nswizzle0_Quant2_MulRoutedWeight1_silu_F8_F8_B16"


def test_only_tile_n_64_of_the_per_token_fp8_stage_1_family_matches():
    for name in (T64, T64B, "fmoe_stage1_bf16_pertokenFp8_g1u1_16x64_5tg_pf3"):
        assert exclusions.tile64(name), name
    for name in (OK1, OK2, CK, "", None):
        assert not exclusions.tile64(name), name
    # x640 and x264 are not tile_n 64; the blockscale family is tile_n 512
    assert not exclusions.tile64("fmoe_stage1_bf16_pertokenFp8_g1u1_16x640_pf3")
    assert not exclusions.tile64(
        "fmoe_stage1_bf16_pertokenFp8_blockscale_g1u1_16x512_pf3E"
    )


def test_the_stage_1_name_comes_from_the_pair_id_or_the_route():
    assert exclusions.stage1_of(f"fused_moe|{T64}|{CK}|bm32|ks0|nt0") == T64
    assert exclusions.stage1_of("fused_moe|1stage|asm_k|bm32|flat0|asm_1stage") == ""
    assert exclusions.stage1_of("fused_moe|graph|impl__x") == ""
    assert exclusions.stage1_of("dispatch", T64B) == T64B
    assert exclusions.stage1_of("dispatch") == ""


def test_a_stage_1_that_failed_accuracy_in_any_pair_taints_every_pair_with_it():
    entries = [
        ("p1", OK1, "ok"),
        ("p2", OK1, "fail"),  # the same stage 1 fails with another partner
        ("p3", OK1, "ok"),
        ("p4", OK2, "ok"),
        ("p5", OK2, "error"),  # an error is not an accuracy failure
        ("p6", T64, "ok"),
        ("dispatch", OK1, "ok"),
    ]
    got = exclusions.excluded(entries)
    assert set(got) == {"p1", "p2", "p3", "p6", "dispatch"}
    assert got["p6"] == exclusions.MOEBUG_064
    assert (
        got["p1"].startswith("stage-1 taint: ") and "failed accuracy in p2" in got["p1"]
    )
    assert "stage-1 taint" in exclusions.note(got) and "MOEBUG-064" in exclusions.note(
        got
    )
    assert exclusions.note({}) == ""


def cand(cid):
    return SimpleNamespace(
        id=cid,
        describe=lambda: {
            "source": "fused_moe",
            "entry": "e",
            "kernels": cid,
            "layout": "d",
        },
    )


def result(cid, us, outcome="ok", route=None):
    return {
        "ok": outcome == "ok",
        "outcome": outcome,
        "cand": cand(cid),
        "by_mode": {"warm": {"median_us": us}},
        "timing_mode": "graph",
        "route": route or {},
    }


def row_of(results, pre=None):
    fh = io.StringIO()
    S = SimpleNamespace(kind="x", D=5120, I=1536, E=41, topk=10)
    bar_worker._emit_summary(
        fh, SimpleNamespace(id="c1"), S, "warm", results, {"gpu": "g", "sku": ""}, [], "bar",
        pre_excluded=pre,
    )  # fmt: skip
    return next(
        json.loads(x) for x in fh.getvalue().splitlines() if '"event": "row"' in x
    )


def test_a_tile_64_candidate_never_sets_the_bar_and_the_row_says_so():
    fast = f"fused_moe|{T64}|{CK}|bm32|ks0|nt0"
    slow = f"fused_moe|{OK1}|{CK}|bm32|ks0|nt0"
    results = {
        "dispatch": result("dispatch", 90.0, route={"kernel1": CK}),
        fast: result(fast, 50.0),
        slow: result(slow, 70.0),
    }
    row = row_of(results)
    assert row["bar_us"] == 70.0 and row["bar_id"] == slow
    assert row["n_correct"] == 2
    assert "MOEBUG-064" in row["bar_exclusions"]


def test_dispatch_routed_to_a_tile_64_stage_1_is_not_a_bar_either():
    results = {
        "dispatch": result("dispatch", 40.0, route={"kernel1": T64B}),
        "p": result(f"fused_moe|{OK1}|{CK}|bm32|ks0|nt0", 60.0),
    }
    row = row_of(results)
    assert row["bar_id"].endswith("nt0") and row["bar_us"] == 60.0


def test_taint_removes_the_stage_1s_other_pairs_from_the_bar():
    good = f"fused_moe|{OK1}|{CK}|bm32|ks0|nt0"
    bad = f"fused_moe|{OK1}|{CK}_b|bm32|ks0|nt0"
    other = f"fused_moe|{OK2}|{CK}|bm64|ks0|nt0"
    results = {
        good: result(good, 30.0),
        bad: result(bad, 35.0, outcome="fail"),
        other: result(other, 80.0),
    }
    row = row_of(results)
    assert row["bar_id"] == other and row["bar_us"] == 80.0
    assert "taint" in row["bar_exclusions"]


def test_a_pre_excluded_note_reaches_the_row_and_the_column_exists():
    row = row_of(
        {"p": result(f"fused_moe|{OK1}|{CK}|bm32|ks0|nt0", 60.0)},
        pre="4 candidates not run, MOEBUG-064",
    )
    assert row["bar_exclusions"] == "4 candidates not run, MOEBUG-064"
    assert "bar_exclusions" in bar_runner.BAR_COLS


UNSAFE = "_ZN5aiter45fmoe_stage1_bf16_pertokenFp8_g1u1_32x256_2tg_pf2E"  # X08/X09 at 5120/1536/E41/top10
UNSAFE_OTHER_TILE = (
    "_ZN5aiter45fmoe_stage1_bf16_pertokenFp8_g1u1_32x256_2tg_pf3E"  # X05/X06/X10 passed
)


def test_the_unsafe_table_holds_every_failed_pair_of_the_verification_files():
    t = exclusions.unsafe_table()
    assert (5120, 1536, 41, 10, "32x256_2tg_pf2") in t and (
        5120,
        1536,
        40,
        8,
        "64x128_2tg_pf3",
    ) in t
    assert (5120, 1536, 41, 10, "32x128_3tg_pf2") in t
    assert (5120, 1536, 41, 10, "32x256_2tg_pf3") not in t  # a PASS row
    assert (4096, 384, 128, 8, "32x64_4tg_pf3") in t  # meta tuned row R24 (gfx942)
    assert exclusions.stage1_tile(UNSAFE) == "32x256_2tg_pf2"
    assert exclusions.stage1_tile("moe_ck2stages_gemm1_x") is None


def test_an_unsafe_pair_is_excluded_at_its_shape_only():
    shape = (5120, 1536, 41, 10)
    assert "unsafe pair" in exclusions.unsafe_reason(shape, UNSAFE)
    assert exclusions.unsafe_reason(shape, UNSAFE_OTHER_TILE) == ""
    assert exclusions.unsafe_reason((5120, 1536, 40, 8), UNSAFE) == ""  # another shape
    assert exclusions.unsafe_reason(None, UNSAFE) == ""
    got = exclusions.excluded(
        [("a", UNSAFE, "ok"), ("b", UNSAFE_OTHER_TILE, "ok")], shape
    )
    assert set(got) == {"a"} and got["a"].startswith("MOEBUG-064 unsafe pair")
    assert "unsafe pair at this shape" in exclusions.note(got)
    assert exclusions.excluded([("a", UNSAFE, "ok")], None) == {}


def test_an_unsafe_pair_never_sets_the_bar_at_its_shape():
    fast = f"fused_moe|{UNSAFE}|{CK}|bm32|ks0|nt0"
    slow = f"fused_moe|{UNSAFE_OTHER_TILE}|{CK}|bm32|ks0|nt0"
    row = row_of({fast: result(fast, 50.0), slow: result(slow, 70.0)})
    assert row["bar_id"] == slow and "unsafe pair" in row["bar_exclusions"]
