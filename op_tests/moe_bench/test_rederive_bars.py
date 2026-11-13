# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Re-deriving a bar file from its candidates file under the MOEBUG-064 / taint exclusions."""

import csv

from op_tests.moe_bench import rederive_bars

T64 = "_ZN5aiter44fmoe_stage1_bf16_pertokenFp8_g1u1_32x64_4tg_pf3E"
A1 = "_ZN5aiter45fmoe_stage1_bf16_pertokenFp8_g1u1_128x128_pf2E"
A2 = "_ZN5aiter44fmoe_stage1_bf16_pertokenFp8_g1u1_32x512_pf2E"
K2 = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeight_v1_Nswizzle0_Quant2_MulRoutedWeight0_F8_F8_B16"
BAR_HEAD = [
    "case_id",
    "family",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "mode",
    "status",
    "bar_id",
    "bar_us",
]
CAND_HEAD = ["case_id", "mode", "cand_id", "phase", "status", "median_us", "kernel1"]


def pair(s1, tag="a"):
    return f"fused_moe|{s1}|{K2}{tag}|bm32|ks0|nt0"


def write(path, head, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def shape(cid, tok="64"):
    return [cid, "fp8-token", tok, "7168", "256", "256", "8", "Silu"]


def test_exclusions_change_the_bar_and_the_report_names_the_route_and_the_why(tmp_path):
    bar_rows, cand_rows = [], []
    for case, cands in {
        "t1": [("dispatch", 90, "ok"), (pair(T64), 50, "ok"), (pair(A1), 70, "ok")],
        "t2": [
            ("dispatch", 90, "ok"),
            (pair(A1), 40, "ok"),
            (pair(A1, "b"), 45, "fail"),
            (pair(A2), 60, "ok"),
        ],
        "t3": [("dispatch", 90, "ok"), (pair(A2), 60, "ok")],
    }.items():
        best = min((c for c in cands if c[2] == "ok"), key=lambda c: c[1])
        bar_rows.append([*shape(case), "warm", "ok", best[0], str(best[1])])
        for cid, us, st in cands:
            cand_rows.append([case, "warm", cid, "confirm", st, str(us), ""])
    bar_rows.append([*shape("t4"), "warm", "ok", "dispatch", "10"])  # no candidate rows
    d = tmp_path / "set"
    d.mkdir()
    write(d / "bar-x.csv", BAR_HEAD, bar_rows)
    write(d / "candidates-x.csv", CAND_HEAD, cand_rows)
    (tmp_path / "tree" / "aiter" / "moe" / "routes").mkdir(parents=True)
    (tmp_path / "tree" / "aiter" / "moe" / "routes" / "mi300x.csv").write_text(
        "family,model_dim,inter_dim,experts,topk,act,doweight_stage1,m_min,m_max,kernel,config,verified,note\n"
        "fp8-token,7168,256,256,8,silu,0,32,128,k.route,,,\n"
    )
    (d / "bar-x.meta.json").write_text(
        '{"gfx": "gfx942", "harness_sha": "abcdef012345"}'
    )
    changes, unver, summary = rederive_bars.run(
        [str(tmp_path / "set")], [], [str(tmp_path / "tree")]
    )
    by = {c["case_id"]: c for c in changes}
    assert set(by) == {"t1", "t2"}  # t3 unchanged
    assert by["t1"]["old_id"] == pair(T64) and by["t1"]["new_id"] == pair(A1)
    assert (
        "MOEBUG-064" in by["t1"]["why_old_winner_excluded"]
        and by["t1"]["new_us"] == "70.0000"
    )
    assert by["t1"]["delta_pct"] == "40.00" and by["t1"]["derived_matches"] == "True"
    assert by["t1"]["shipped_routes"] == "mi300x.csv:k.route"
    # t2: a stage 1 failed accuracy with one partner: its other pair is not trusted
    assert (
        by["t2"]["new_id"] == pair(A2)
        and "stage-1 taint" in by["t2"]["why_old_winner_excluded"]
    )
    assert unver == [
        (
            "set/bar-x.csv" if False else str(d / "bar-x.csv"),
            "t4",
            "warm",
            "no candidate rows for the case",
        )
    ]
    assert summary[0]["changed"] == 2


def test_a_bar_file_without_a_candidates_file_is_unverifiable(tmp_path):
    write(
        tmp_path / "bar-y.csv", BAR_HEAD, [[*shape("c"), "warm", "ok", "dispatch", "5"]]
    )
    changes, unver, _ = rederive_bars.run([str(tmp_path)], [], [])
    assert changes == [] and unver[0][3] == "no candidates file"


def test_a_derived_bar_that_is_not_the_recorded_one_is_marked(tmp_path):
    write(
        tmp_path / "bar-z.csv", BAR_HEAD, [[*shape("c"), "warm", "ok", pair(A1), "999"]]
    )
    write(
        tmp_path / "candidates-z.csv", CAND_HEAD,
        [["c", "warm", pair(T64), "confirm", "ok", "20", ""], ["c", "warm", pair(A1), "confirm", "ok", "30", ""]],
    )  # fmt: skip
    changes, _, _ = rederive_bars.run([str(tmp_path)], [], [])
    assert (
        changes[0]["derived_matches"] == "False"
        and "differs from the recorded" in changes[0]["note"]
    )
