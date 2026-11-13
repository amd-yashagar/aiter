# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The dropped-group audit over existing bar outputs."""

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
