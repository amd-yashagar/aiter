# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Marginal-effect screening of a block_m group that is not the full stage-1 x stage-2
product: a pair that is not in the group is skipped, nothing the bar should measure is lost."""

import types

from op_tests.moe_bench.candidates.fused_cfg import FusedCfgSource


def _c(s1, s2, group="pair-bm128-0-0-0"):
    return types.SimpleNamespace(
        id=f"{s1}+{s2}", kernels=f"{s1} + {s2}", group=group, meta={"s1": s1, "s2": s2}
    )


class Runner:
    """Time of a pair = f1[s1] + f2[s2]; a stage kernel in ``broken`` never runs."""

    def __init__(self, f1, f2, broken=()):
        self.f1, self.f2, self.broken = f1, f2, set(broken)
        self.measured = []

    def measure(self, cand, quick=True, check=True):
        self.measured.append(cand.id)
        s1, s2 = cand.meta["s1"], cand.meta["s2"]
        if s1 in self.broken or s2 in self.broken:
            return {"ok": False, "us": None}
        return {"ok": True, "us": self.f1[s1] + self.f2[s2]}


F1 = {"a": 10.0, "b": 12.0, "c": 20.0}
F2 = {"x": 5.0, "y": 7.0, "z": 9.0}


def _ids(ranked):
    return {c.id for _, c in ranked}


def test_a_pair_missing_from_the_group_is_skipped_not_raised_in_both_paths():
    # the first stage-1 kernel ("a") has no partner "y": the full product is not there
    group = [_c(a, b) for a in F1 for b in F2 if (a, b) != ("a", "y")]
    src = FusedCfgSource()
    ranked = src.rank(None, group, Runner(F1, F2))
    assert _ids(ranked) == {c.id for c in group}  # every candidate has a time
    predicted = {c.id: us for us, c in ranked}
    for c in group:
        assert predicted[c.id] == F1[c.meta["s1"]] + F2[c.meta["s2"]]
    keep = src.screen(types.SimpleNamespace(), group, Runner(F1, F2))
    assert keep and all(c in group for c in keep)  # no KeyError, screening completes


def test_the_bf16_case_the_first_stage_1_kernel_lacks_the_first_stage_2():
    """tier1-00072: (t128x128x128, ..._t128x128x128_atomic) was not in the group."""
    f1 = {"t128x128x128": 30.0, "t32x32x128": 11.0}
    f2 = {"t128x128x128_atomic": 6.0, "t32x128x128_atomic": 5.0}
    group = [
        _c("t128x128x128", "t32x128x128_atomic"),
        _c("t32x32x128", "t128x128x128_atomic"),
        _c("t32x32x128", "t32x128x128_atomic"),
    ]
    src = FusedCfgSource()
    ranked = src.rank(None, group, Runner(f1, f2))
    assert _ids(ranked) == {c.id for c in group}
    assert ranked[0][1].id == "t32x32x128+t32x128x128_atomic"
    assert (
        src.screen(types.SimpleNamespace(), group, Runner(f1, f2))[0].id
        == ranked[0][1].id
    )


def test_a_first_kernel_that_never_runs_does_not_drop_the_group():
    group = [_c(a, b) for a in F1 for b in F2]
    src = FusedCfgSource()
    ranked = src.rank(None, group, Runner(F1, F2, broken={"a"}))
    assert _ids(ranked) == {c.id for c in group if c.meta["s1"] != "a"}
    assert ranked[0][1].meta["s1"] == "b" and ranked[0][1].meta["s2"] == "x"
    keep = src.screen(types.SimpleNamespace(), group, Runner(F1, F2, broken={"a"}))
    assert keep and all(c.meta["s1"] != "a" for c in keep)


def test_pairs_the_sweep_cannot_reach_are_timed_directly_up_to_a_cap():
    group = [_c(a, b) for a in F1 for b in F2 if (a, b) != ("c", "x")]
    src = FusedCfgSource()
    runner = Runner(F1, F2)
    ranked = src.rank(None, group, runner)
    assert _ids(ranked) == {c.id for c in group}
    # ("c","y") and ("c","z") are in the group but "c" has no partner "x" (the base): timed
    # directly, once each
    assert runner.measured.count("c+y") >= 1 and runner.measured.count("c+z") >= 1
    src.UNRANKED_CAP = 0
    capped = FusedCfgSource()
    capped.UNRANKED_CAP = 1
    ranked = capped.rank(None, group, Runner(F1, F2))
    assert len(ranked) == len(group) - 1 and capped.unreached_skipped == 1
