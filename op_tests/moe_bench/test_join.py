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
