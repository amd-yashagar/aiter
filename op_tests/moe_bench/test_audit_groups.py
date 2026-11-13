# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The dropped-group audit over existing bar outputs."""

import csv

from op_tests.moe_bench import audit_groups


def _cand(case, s1, s2, group, status, phase="screen"):
    return {
        "case_id": case,
        "cand_id": f"fused_moe|{s1}|{s2}|{group}",
        "status": status,
        "phase": phase,
    }


def test_a_group_whose_only_tried_stage_1_never_ran_is_flagged_data_only():
    bars = [{"case_id": "c1", "mode": "warm", "status": "ok"}]
    cands = [
        # bm16: the first stage-1 kernel failed with both partners and nothing else was tried
        _cand("c1", "_ZN5aiter_asm", "ck2a", "bm16|ks0|nt0", "error"),
        _cand("c1", "_ZN5aiter_asm", "ck2b", "bm16|ks0|nt0", "error"),
        # bm32 ran
        _cand("c1", "ck1", "ck2a", "bm32|ks0|nt0", "ok"),
        _cand("c1", "ck1", "ck2a", "bm32|ks0|nt1", "fail"),
        _cand("c1", "ck3", "ck2a", "bm32|ks0|nt1", "ok"),
    ]
    flags, by_case = audit_groups.data_flags(bars, cands)
    assert [f[:2] for f in flags["c1"]] == [("base_failed?", "bm16/ks0/nt0")]
    assert ("bm32", "ks0", "nt1") in by_case["c1"]


def test_a_keyerror_case_is_flagged():
    bars = [
        {
            "case_id": "c2",
            "mode": "warm",
            "status": "error",
            "detail": "KeyError: ('a', 'b')",
        },
        {"case_id": "c3", "mode": "warm", "status": "error", "detail": "RuntimeError"},
    ]
    flags, _ = audit_groups.data_flags(bars, [])
    assert [f[0] for f in flags["c2"]] == ["keyerror"] and "c3" not in flags


def test_enumeration_confirms_a_dropped_group_and_finds_the_never_tried_ones():
    cands = [_cand("c1", "s1a", "s2a", "bm16|ks0|nt0", "error")]
    _, by_case = audit_groups.data_flags([], cands)
    enum = {
        "c1": {
            "bm16/ks0/nt0": {"pairs": 6, "s1": 3, "first_no_partner": False},
            "bm32/ks0/nt0": {"pairs": 4, "s1": 2, "first_no_partner": True},
        }
    }
    flags = audit_groups.enum_flags(enum, by_case)
    got = {(f[0], f[1]) for f in flags["c1"]}
    assert ("base_dropped", "bm16/ks0/nt0") in got  # 2 other stage-1 kernels untried
    assert ("never_tried", "bm32/ks0/nt0") in got
    assert ("first_no_partner", "bm32/ks0/nt0") in got
    # a group that has a single stage-1 kernel loses nothing when it fails
    enum["c1"]["bm16/ks0/nt0"]["s1"] = 1
    got = {(f[0], f[1]) for f in audit_groups.enum_flags(enum, by_case)["c1"]}
    assert ("base_dropped", "bm16/ks0/nt0") not in got


def test_a_flagged_case_is_routed_when_its_shape_has_a_route_row_on_that_arch():
    case = {
        "family": "fp8-block",
        "model_dim": "4096",
        "inter_dim": "256",
        "expert": "513",
        "topk": "11",
        "act_type": "Silu",
        "token": "4096",
    }
    routes = [
        {
            "family": "fp8-block",
            "model_dim": "4096",
            "inter_dim": "256",
            "experts": "513",
            "topk": "11",
            "act": "silu",
            "m_min": "4096",
            "m_max": "4096",
            "kernel": "k.prefill",
        }
    ]
    assert audit_groups.routed(case, routes) == ["k.prefill"]
    assert audit_groups.routed(dict(case, token="1024"), routes) == []


def test_the_replay_settles_the_soft_flag_of_a_group_it_enumerates():
    soft = ("base_failed?", "bm16/ks0/nt0", "3 rows, one stage-1 kernel")
    other = ("base_failed?", "bm64/ks0/nt0", "3 rows, one stage-1 kernel")
    flags = {"c1": [soft, other], "c2": [soft], "c3": [soft]}
    enum = {
        "c1": {
            "bm16/ks0/nt0": {"s1": 1, "pairs": 3}
        },  # one stage-1 kernel: nothing lost
        "c2": {"error": "replay failed"},
        "c3": {"bm32/ks0/nt0": {"s1": 4, "pairs": 12}},
    }
    audit_groups.settle_soft(flags, enum)
    assert flags["c1"] == [other]  # bm64 is not in today's enumeration: stays soft
    assert flags["c2"] == [soft]  # no replay: stays soft
    assert flags["c3"] == [soft]  # the group the flag names is not enumerated: stays
    enum["c3"]["bm16/ks0/nt0"] = {"s1": 1, "pairs": 3}
    audit_groups.settle_soft(flags, enum)
    assert "c3" not in flags


def test_a_row_written_under_another_column_set_is_a_hard_flag(tmp_path):
    from op_tests.moe_bench import bar_runner, schema

    hist = {len(h): h for h in bar_runner._history("bar")}
    h102, h107 = hist[102], hist[107]

    def row(h, **v):
        d = {c: "" for c in h}
        d.update(v)
        return [d[c] for c in h]

    prov = {"rocm": "7.2.53211", "torch": "2.13.0+rocm7.2", "flydsl": "0.5.0", "gfx": "gfx950",
            "cu_num": "256", "aiter_sha": "27faf0a57b4e", "aiter_dirty": "0",
            "harness_sha": "50c0d1b74f94", "date": "2026-11-12T18:00:00Z"}  # fmt: skip
    ok = row(h102, case_id="c1", mode="warm", status="ok", bar_us="1.5", **prov)
    cut = row(h107, case_id="c2", mode="warm", status="ok", bar_us="2.5", **prov)[:102]
    short = row(h102, case_id="c3", mode="warm", status="ok")[:50]
    bar = tmp_path / "bar-t.csv"
    with open(bar, "w", newline="") as f:
        csv.writer(f).writerows([h102, ok, cut, short])
    cands = tmp_path / "candidates-t.csv"
    cands.write_text("case_id,mode,cand_id,status\r\nc1,warm,dispatch,ok\r\n")
    got = audit_groups.unreadable(str(bar))
    assert [c for c, _ in got] == ["c2", "c3"]
    assert got[0][1].startswith("implausible values in") and "harness_sha" in got[0][1]
    assert got[1][1] == "50 fields, header has 102"
    res = audit_groups.audit(str(bar), str(cands), {})
    assert res["unreadable"] == 2
    assert {(r["case_id"], tuple(r["flags"])) for r in res["flagged"]} == {
        ("c2", ("unreadable_row",)),
        ("c3", ("unreadable_row",)),
    }
    # repaired: reads clean and the cut row's columns are where they belong
    bar_runner.repair_bar_csv(str(bar))
    assert audit_groups.unreadable(str(bar)) == []
    assert {r["case_id"] for r in schema.read_csv(str(bar), strict=True)} == {
        "c1",
        "c2",
    }
