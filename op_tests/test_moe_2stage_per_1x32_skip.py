# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression: test_moe_2stage.py must not record per_1x32 cases as results on
an arch that cannot run them.

test_fmoe used to ``return`` (None) on a non-gfx950 arch for QuantType.per_1x32,
and the sweep recorded that as a result, so a4w4, a8w4, a16w4, a16wi4 and mxfp8
looked checked on gfx942. The sweep now skips those cases with a message."""

import logging
import runpy
import sys
from pathlib import Path

import pytest
import torch

import aiter
from aiter.jit.utils import chip_info

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="Needs a ROCm GPU",
)

SCRIPT = str(Path(__file__).with_name("test_moe_2stage.py"))
PER_1X32_QUANTS = ["4", "6", "7", "8", "9"]


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _run_sweep(monkeypatch, quant):
    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(
        sys,
        "argv",
        [SCRIPT, "-q", quant, "-t", "8", "-dim", "512,256", "-e", "8", "-k", "2"]
        + ["--no-flydsl-csv"],
    )
    handler = _Records()
    aiter.logger.addHandler(handler)
    try:
        runpy.run_path(SCRIPT, run_name="__main__")
    finally:
        aiter.logger.removeHandler(handler)
    return handler.messages


@pytest.mark.parametrize("quant", PER_1X32_QUANTS)
def test_per_1x32_cases_are_skipped_not_recorded_off_gfx950(monkeypatch, quant):
    messages = _run_sweep(monkeypatch, quant)
    skips = [m for m in messages if "gfx950-only" in m and "skip moe_2stage" in m]
    assert skips, "per_1x32 cases must be skipped with a message"
    summary = [m for m in messages if "moe_2stage: scanned" in m]
    assert summary and "recorded 0 results" in summary[-1], summary


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
