# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Two-phase bar screening: which candidates get the full measurement."""

import pathlib
import types

from op_tests.moe_bench import bar_worker


class FakeMeasurer:
    def __init__(self, screen_us, wrong=()):
        self.screen_us, self.wrong = screen_us, set(wrong)
        self.calls = []

    def measure(self, cand, quick=False, modes=("warm",), check=True):
        self.calls.append((cand.id, quick, check))
        if not check:
            us = self.screen_us.get(cand.id)
            return {"ok": us is not None, "us": us}
        ok = cand.id not in self.wrong
        return {"ok": ok, "us": self.screen_us[cand.id] if ok else None}


def _cands(n):
    return [types.SimpleNamespace(id=f"c{i}") for i in range(n)]


def test_finalists_are_the_top_eight_plus_the_five_percent_band():
    cands = _cands(30)
    us = {c.id: 100.0 + i for i, c in enumerate(cands)}  # c0 100 .. c29 129
    for i in (8, 9):  # two more within 5% of the best (100 -> 105)
        us[cands[i].id] = 104.0 + i * 0.01
    for i in (10, 11, 12):  # clearly slower than the band
        us[cands[i].id] = 200.0 + i
    m = FakeMeasurer(us)
    results = {}
    bar_worker.two_phase(m, cands, (), ("warm", "cold"), results)
    screened = [c for c, q, chk in m.calls if q and not chk]
    full = [c for c, q, chk in m.calls if not q and chk]
    assert len(screened) == 30, "phase 1 times every candidate, without accuracy"
    fast = sorted(us, key=us.get)
    in_band = {c for c in us if us[c] <= 105.0}
    assert set(full) == set(fast[:8]) | in_band
    assert set(results) == set(full)


def test_the_set_widens_when_every_finalist_fails_accuracy():
    cands = _cands(30)
    us = {c.id: 100.0 + i for i, c in enumerate(cands)}
    fast = sorted(us, key=us.get)
    m = FakeMeasurer(us, wrong=fast[:8])  # the whole first set is wrong
    results = {}
    bar_worker.two_phase(m, cands, (), ("warm",), results)
    full = [c for c, q, chk in m.calls if not q and chk]
    assert full[:8] == fast[:8] and len(full) == 16, full
    assert any(r["ok"] for r in results.values())
    # widening stops at the first set with a correct candidate: nothing past 16 timed


def test_nothing_correct_ends_after_trying_every_screened_candidate():
    cands = _cands(11)
    us = {c.id: 100.0 + i for i, c in enumerate(cands)}
    m = FakeMeasurer(us, wrong=set(us))
    results = {}
    bar_worker.two_phase(m, cands, (), ("warm",), results)
    assert len(results) == 11 and not any(r["ok"] for r in results.values())


def test_failed_screens_and_bad_candidates_are_skipped():
    cands = _cands(12)
    us = {c.id: 100.0 + i for i, c in enumerate(cands) if i != 3}  # c3 errors
    m = FakeMeasurer(us)
    results = {}
    bar_worker.two_phase(m, cands, ("c0",), ("warm",), results)
    timed = {c for c, q, chk in m.calls}
    assert "c0" not in timed
    assert "c3" not in results
    assert "c1" in results


def test_forced_config_file_is_per_worker(tmp_path, monkeypatch):
    """Workers of one run share a tmpdir: each forces its row through its own file."""
    import os

    from op_tests.moe_bench.candidates import fused_cfg

    monkeypatch.setattr(fused_cfg, "reset_router", lambda *a, **k: None)
    fused_cfg.force_config({"kernelName1": "k"}, str(tmp_path))
    mine = os.environ["AITER_CONFIG_FMOE"]
    assert mine.endswith(f"force_fmoe_{os.getpid()}.csv") and os.path.exists(mine)
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    fused_cfg.force_config({"kernelName1": "other"}, str(tmp_path))
    other = os.environ["AITER_CONFIG_FMOE"]
    text = pathlib.Path(mine).read_text()
    assert other != mine and "kernelName1" in text
    assert text.splitlines()[1].count("k") >= 1
    os.environ.pop("AITER_CONFIG_FMOE", None)


class RankingSource:
    """A source that predicts pair times from their stages (FusedCfgSource.rank)."""

    def __init__(self, ranked):
        self.ranked, self.asked = ranked, 0

    def rank(self, ctx, cands, runner):
        self.asked += 1
        return self.ranked


def test_a_source_that_ranks_replaces_timing_every_candidate():
    cands = _cands(300)  # a product of stages: far too many to time one by one
    ranked = [(100.0 + i, c) for i, c in enumerate(cands[:20])]
    m = FakeMeasurer({c.id: us for us, c in ranked})
    src = RankingSource(ranked)
    results = {}
    bar_worker.two_phase(m, cands, (), ("warm",), results, src)
    assert src.asked == 1
    assert not [1 for _c, q, chk in m.calls if q and not chk], "no per-candidate timing"
    assert set(results) == {c.id for _, c in ranked[:8]}
