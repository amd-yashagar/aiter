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
    assert (
        exclusions.stage1_of("fused_moe|1stage|asm_k|bm32|flat0|asm_1stage") == "asm_k"
    )
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
        fh, SimpleNamespace(id="c1"), S, "warm", results, {"gpu": "g", "sku": "", "gfx": "gfx950"}, [], "bar",
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


UNSAFE = "_ZN5aiter45fmoe_stage1_bf16_pertokenFp8_g1u1_32x256_2tg_pf2E"  # X08 on gfx950
UNSAFE_OTHER_TILE = (
    "_ZN5aiter45fmoe_stage1_bf16_pertokenFp8_g1u1_32x256_2tg_pf3E"  # X05/X06/X10 passed
)


def test_the_unsafe_list_holds_every_failed_kernel_of_the_verification_files():
    t = exclusions.unsafe_table()
    for tile in (
        "64x128_2tg_pf3",
        "32x128_3tg_pf2",
        "32x256_2tg_pf2",
        "128x64_pf2",
        "16x64_5tg_pf3",
    ):
        assert ("gfx950", tile) in t, tile
    assert ("gfx942", "32x64_4tg_pf3") in t  # meta tuned rows R24
    assert ("gfx950", "32x256_2tg_pf3") not in t  # a PASS row
    assert "buffer probe" in t[("gfx950", "64x128_2tg_pf3")]
    assert exclusions.stage1_tile(UNSAFE) == "32x256_2tg_pf2"
    assert exclusions.stage1_tile("moe_ck2stages_gemm1_x") is None
    # another .co (doweight, blockscale) with the same tile is keyed by its whole name, not the listed tile
    assert (
        exclusions.stage1_tile(
            "_ZN5aiter51fmoe_stage1_bf16_pertokenFp8_doweight_g1u1_32x64_4tg_pf3E"
        )
        != "32x64_4tg_pf3"
    )
    assert (
        exclusions.stage1_tile(
            "fmoe_stage1_bf16_pertokenFp8_blockscale_g1u1_64x128_2tg_pf3E"
        )
        != "64x128_2tg_pf3"
    )


def test_an_unsafe_kernel_is_excluded_at_every_shape_on_its_arch_only():
    assert "unsafe kernel" in exclusions.unsafe_reason("gfx950", UNSAFE)
    assert exclusions.unsafe_reason("gfx950", UNSAFE_OTHER_TILE) == ""
    assert (
        exclusions.unsafe_reason("gfx942", UNSAFE) == ""
    )  # a separate .co, not on the list
    assert (
        exclusions.unsafe_reason("", UNSAFE) == ""
        and exclusions.unsafe_reason(None, UNSAFE) == ""
    )
    got = exclusions.excluded(
        [("a", UNSAFE, "ok"), ("b", UNSAFE_OTHER_TILE, "ok")], "gfx950"
    )
    assert set(got) == {"a"} and got["a"].startswith("MOEBUG-064 unsafe kernel")
    assert "unsafe kernel" in exclusions.note(got)
    assert exclusions.excluded([("a", UNSAFE, "ok")], None) == {}


def test_an_unsafe_kernel_never_sets_the_bar_on_its_arch():
    fast = f"fused_moe|{UNSAFE}|{CK}|bm32|ks0|nt0"
    slow = f"fused_moe|{UNSAFE_OTHER_TILE}|{CK}|bm32|ks0|nt0"
    row = row_of({fast: result(fast, 50.0), slow: result(slow, 70.0)})
    assert row["bar_id"] == slow and "unsafe kernel" in row["bar_exclusions"]


def test_the_list_is_extended_by_appending_verdicts_in_one_file(tmp_path):
    path = str(tmp_path / "unsafe.csv")
    rows = [
        {"gfx": "gfx950", "stage1_tile": "64x128_2tg_pf3", "source": "a.csv", "row_id": "X01", "evidence": "e"},
        {"gfx": "gfx950", "stage1_tile": "64x128_2tg_pf3", "source": "b.csv", "row_id": "Y1", "evidence": "dup"},
        {"gfx": "gfx942", "stage1_tile": "64x128_2tg_pf3E", "also_on": "gfx950", "source": "c.csv", "row_id": "Z", "evidence": "e"},
    ]  # fmt: skip
    assert (
        exclusions.append_verdicts(rows, path) == 2
    )  # the duplicate (arch, kernel) is not added; the other arch is its own entry
    assert exclusions.append_verdicts(rows[:1], path) == 0
    src = tmp_path / "verdicts.csv"
    src.write_text(
        "row_id,stage1_kernel,verdict,uniform_bad_reps,hot1_bad_reps\r\nX9,32x128_3tg_pf2E,FAIL,1/6,0/6\r\nX10,32x256_2tg_pf3,PASS,0/6,0/6\r\n"
    )
    got = exclusions.ingest(str(src), "gfx950", also_on="gfx951")
    assert [r["stage1_tile"] for r in got] == ["32x128_3tg_pf2"] and got[0][
        "also_on"
    ] == "gfx951"
    assert exclusions.append_verdicts(got, path) == 1


ONE = "_ZN5aiter47fmoe_bf16_pertokenFp8_g1u1_tkw1_silu_1tg_32x64E"  # P01, P03, P04, P05 on gfx950
ONE_OTHER = "_ZN5aiter47fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_32x256E"  # P06 passed


def test_the_one_stage_kernel_of_the_p_rows_is_on_the_list_by_name():
    assert (
        exclusions.kernel_key(ONE) == "fmoe_bf16_pertokenFp8_g1u1_tkw1_silu_1tg_32x64"
    )
    assert "unsafe kernel" in exclusions.unsafe_reason("gfx950", ONE)
    assert exclusions.unsafe_reason("gfx950", ONE_OTHER) == ""
    assert exclusions.unsafe_reason("gfx942", ONE) == ""
    # a 2-stage tile of the same text is another kernel; blockscale 1-stage is another family
    assert exclusions.kernel_key(
        "_ZN5aiter47fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x128E"
    ).startswith("fmoe_bf16_blockscale")
    cid = f"fused_moe|1stage|{ONE}|bm32|flat0|asm_1stage"
    assert exclusions.stage1_of(cid) == ONE
    got = exclusions.excluded([(cid, exclusions.stage1_of(cid), "ok")], "gfx950")
    assert "unsafe kernel" in got[cid]


def test_a_one_stage_kernel_never_sets_the_bar_on_its_arch():
    bad = f"fused_moe|1stage|{ONE}|bm32|flat0|asm_1stage"
    good = f"fused_moe|1stage|{ONE_OTHER}|bm32|flat0|asm_1stage"
    row = row_of({bad: result(bad, 20.0), good: result(good, 25.0)})
    assert row["bar_id"] == good and "unsafe kernel" in row["bar_exclusions"]


def test_one_stage_verdicts_ingest_with_the_family_prefix(tmp_path):
    src = tmp_path / "v.csv"
    src.write_text(
        "row_id,kind,stage1_kernel,verdict,uniform_bad_reps,hot1_bad_reps\r\n"
        "B1,shipped_gfx950_blockscale_1stage_asm_row,vs_silu_1tg_32x128,FAIL,1/6,0/6\r\n"
        "P1,shipped_gfx950_per_token_fp8_1stage_asm_row,tkw1_silu_1tg_32x64,FAIL,2/6,1/6\r\n"
        "X1,extra_shipped_gfx950_row,64x128_2tg_pf3,FAIL,3/6,6/6\r\n"
    )
    got = {r["row_id"]: r["stage1_tile"] for r in exclusions.ingest(str(src), "gfx950")}
    assert got == {
        "B1": "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x128",
        "P1": "fmoe_bf16_pertokenFp8_g1u1_tkw1_silu_1tg_32x64",
        "X1": "64x128_2tg_pf3",
    }


def test_taint_is_for_two_stage_pairs_only():
    ones = [
        "fused_moe|1stage|K|bm32|flat0|asm_1stage",
        "fused_moe|1stage|K|bm32|flat1|asm_1stage",
    ]
    got = exclusions.excluded([(ones[0], "K", "ok"), (ones[1], "K", "fail")], "gfx950")
    assert got == {}  # a failing flat variant does not take the other variant out


def test_a_pair_that_failed_accuracy_in_the_screen_taints_too():
    good = f"fused_moe|{OK1}|{CK}|bm32|ks0|nt0"
    screened_bad = (
        f"fused_moe|{OK1}|{CK}_s|bm32|ks0|nt0"  # failed while screened: in no result
    )
    other = f"fused_moe|{OK2}|{CK}|bm64|ks0|nt0"
    results = {good: result(good, 30.0), other: result(other, 80.0)}
    fh = io.StringIO()
    S = SimpleNamespace(kind="x", D=1, I=1, E=1, topk=1)
    bar_worker._emit_summary(
        fh, SimpleNamespace(id="c1"), S, "warm", results, {"gpu": "g", "sku": "", "gfx": "gfx942"}, [], "bar",
        failed=[screened_bad],
    )  # fmt: skip
    row = next(
        json.loads(x) for x in fh.getvalue().splitlines() if '"event": "row"' in x
    )
    assert row["bar_id"] == other and "taint" in row["bar_exclusions"]
