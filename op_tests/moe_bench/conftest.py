# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Test isolation: the suite never writes into a shared JIT dir's prebuild bookkeeping."""

import hashlib
import os

import pytest

STATE_FILES = (
    "prebuild_failed.json",
    "prebuild_missing.json",
    "prebuild_keymods.json",
    "prebuild_attempted.json",
)


def _shared_state():
    d = os.environ.get("AITER_JIT_DIR")
    if not d:
        return {}
    out = {}
    for f in STATE_FILES:
        try:
            with open(os.path.join(d, f), "rb") as fh:
                out[f] = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            out[f] = None
    return out


@pytest.fixture(autouse=True)
def _private_prebuild_state(tmp_path_factory, monkeypatch):
    """Every test (and every worker it spawns) keeps its prebuild records in its own dir."""
    from op_tests.moe_bench import prebuild

    prebuild._PLAN_CACHE.clear()
    monkeypatch.setenv(
        "MOE_BENCH_STATE_DIR", str(tmp_path_factory.mktemp("moe-bench-state"))
    )


def pytest_sessionstart(session):
    session.config._shared_state_before = _shared_state()


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_shared_state_before", {})
    changed = [f for f, h in _shared_state().items() if before.get(f) != h]
    if changed:
        print(
            "\nmoe_bench tests changed the shared JIT dir's prebuild files: "
            + ", ".join(changed)
        )
        session.exitstatus = 1
