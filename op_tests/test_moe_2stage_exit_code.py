# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression: test_moe_2stage.py's legacy sweep must exit non-zero when a case
fails the accuracy check.

The sweep runs non-strict so one bad case does not stop the rest, but it used to
log "accuracy check failed (non-strict)" and still exit 0, which hid wrong
results from CI."""

import runpy
import sys
from pathlib import Path

import pytest
import torch

import aiter.fused_moe as fused_moe_module
from aiter import test_common

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="Needs a ROCm GPU",
)

SCRIPT = str(Path(__file__).with_name("test_moe_2stage.py"))
SWEEP_ARGS = [
    "-q",
    "0",
    "-t",
    "8",
    "-dim",
    "512,256",
    "-e",
    "8",
    "-k",
    "2",
    "--no-flydsl-csv",
]


def _called_from_sweep_loop():
    # Only the module-level case loop is non-strict; the targeted tests that run
    # before it assert strictly and must see the real output.
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_name == "test_fmoe":
            caller = frame.f_back
            while caller is not None and caller.f_code.co_name == "wrapper":
                caller = caller.f_back
            return caller is not None and caller.f_code.co_name == "<module>"
        frame = frame.f_back
    return False


def _run_sweep(monkeypatch, extra_args=()):
    real_perftest = test_common.run_perftest
    real_fused_moe = fused_moe_module.fused_moe

    def wrong_perftest(func, *args, **kwargs):
        out, us = real_perftest(func, *args, **kwargs)
        if func is real_fused_moe and _called_from_sweep_loop():
            out = torch.zeros_like(out)
        return out, us

    monkeypatch.setattr(test_common, "run_perftest", wrong_perftest)
    monkeypatch.setattr(sys, "argv", [SCRIPT, *SWEEP_ARGS, *extra_args])
    runpy.run_path(SCRIPT, run_name="__main__")


def test_legacy_sweep_exits_nonzero_on_accuracy_failure(monkeypatch):
    with pytest.raises(SystemExit) as excinfo:
        _run_sweep(monkeypatch)
    assert excinfo.value.code == 1


def test_legacy_sweep_can_allow_accuracy_failures(monkeypatch):
    _run_sweep(monkeypatch, ["--allow-accuracy-failures"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
