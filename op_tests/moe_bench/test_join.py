# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests for the joins and rules the harness gets wrong silently if they break."""

import csv

from op_tests.moe_bench import accuracy, bar_runner


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "mode", "status", "median_us"])
        w.writeheader()
        w.writerows(rows)


def test_dispatch_us_reaches_new_rows(tmp_path):
    p = tmp_path / "tier1.csv"
    _write(
        p,
        [
            {"case_id": "c1", "mode": "warm", "status": "ok", "median_us": "30.5"},
            {"case_id": "c1", "mode": "cold", "status": "ok", "median_us": "41.0"},
            {"case_id": "c2", "mode": "warm", "status": "unsupported", "median_us": ""},
        ],
    )
    status, us = bar_runner.dispatch_lookup(str(p))
    assert status == {"c1": "ok", "c2": "unsupported"}
    ev = bar_runner.fill_dispatch({"mode": "cold", "new_us": 40.0}, us, "c1")
    assert ev["dispatch_us"] == 41.0
    ev = bar_runner.fill_dispatch({"mode": "warm"}, us, "c2")
    assert "dispatch_us" not in ev
    ev = bar_runner.fill_dispatch({"mode": "warm", "dispatch_us": 7.0}, us, "c1")
    assert ev["dispatch_us"] == 7.0


def test_aiter_strict_rule():
    def res(allclose, logits, nan=0, sentinel=True):
        m = {"path_allclose": allclose, "path_logits_diff": logits}
        return {"per_seed": [m], "nan_inf": nan, "sentinel_ok": sentinel, "mean": m}

    # a large mismatch fraction alone does not fail (fp8 re-quantization flips elements)
    assert accuracy.rule(res(0.58, 1e-4)) is True
    # mismatch together with a logits diff above 0.01 fails
    assert accuracy.rule(res(0.001, 0.02)) is False
    assert accuracy.rule(res(0.0, 0.02)) is True
    assert accuracy.rule(res(0.0, 0.0, nan=1)) is False
    assert accuracy.rule(res(0.0, 0.0, sentinel=False)) is False
    # asm FLAT MXFP4 rows: logits_diff < 2e-3
    assert accuracy.rule(res(0.5, 1e-3), flat_mxfp4=True) is True
    assert accuracy.rule(res(0.5, 3e-3), flat_mxfp4=True) is False


def test_a16w4_false_layout_equals_std16():
    """MX worker finding (MXB-1 notes): for 1-byte and packed-fp4 weights
    shuffle_weight((16,16)) and shuffle_weight_a16w4(w, 16, False) give the same byte
    order, and e8m0_shuffle equals shuffle_scale_a16w4(False)."""
    import torch

    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility import fp4_utils

    g = torch.Generator().manual_seed(0)
    e, n, k = 4, 256, 512
    for dt, kk in ((torch.uint8, k // 2), (torch.int8, k)):
        w = torch.randint(0, 255, (e, n, kk), generator=g, dtype=torch.int64).to(dt)
        a = shuffle_weight(w.clone(), (16, 16))
        b = shuffle_weight_a16w4(w.clone(), 16, False)
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), dt
    s = torch.randint(100, 140, (e * n, k // 32), generator=g, dtype=torch.int64)
    s = s.to(torch.uint8)
    assert torch.equal(
        fp4_utils.e8m0_shuffle(s.clone()).view(torch.uint8),
        shuffle_scale_a16w4(s.clone(), e, False).view(torch.uint8),
    )


def test_shim_is_repointed_and_harness_checked(tmp_path):
    import os

    import pytest

    from op_tests.moe_bench import runner

    other = tmp_path / "other_harness"
    other.mkdir()
    shim = tmp_path / "shim"
    shim.mkdir()
    os.symlink(other, shim / "moe_bench_shim")  # left by another tree's run
    runner.make_shim(str(tmp_path))
    assert os.path.realpath(shim / "moe_bench_shim") == os.path.realpath(runner.HERE)
    runner.check_harness(runner.HERE, runner.HERE)
    with pytest.raises(RuntimeError):
        runner.check_harness(runner.HERE, str(other))
    with pytest.raises(RuntimeError):
        runner.check_harness(runner.HERE, "")
    tree = tmp_path / "tree"
    (tree / "op_tests" / "moe_bench").mkdir(parents=True)  # a stale copy of the harness
    assert runner.needs_shim(str(tree))
