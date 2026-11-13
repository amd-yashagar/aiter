# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Configs that cannot run on the arch are never tried (MOEBUG-061); a launch failure that
poisons the context is recorded with its error before the worker restarts."""

from op_tests.moe_bench import bar_runner, schema
from op_tests.moe_bench.candidates.fused_cfg import (
    flydsl_stage1_unsupported as unsupported,
)

# configs that need 128 KiB of LDS: t32x{16,32,64}x256_kw4, t64x{16,32}x128_kw4, t64x32x256_kw2
BIG = [
    "flydsl_moe1_abf16_wbf16_bf16_t32x16x256_bnt0_kw4",
    "flydsl_moe1_abf16_wbf16_bf16_t32x32x256_bnt0_kw4",
    "flydsl_moe1_abf16_wbf16_bf16_t32x64x256_bnt0_kw4",
    "flydsl_moe1_abf16_wbf16_bf16_t64x16x128_bnt0_kw4",
    "flydsl_moe1_abf16_wbf16_bf16_t64x32x128_bnt0_kw4",
    "flydsl_moe1_abf16_wbf16_bf16_t64x32x256_bnt0_kw2",
]
SMALL = "flydsl_moe1_abf16_wbf16_bf16_t16x16x128_bnt0_kw4"


def test_stage_1_configs_over_the_arch_lds_budget_are_unsupported_with_the_reason():
    for name in BIG:
        assert (
            unsupported(name, 3072, "gfx942")
            == "LDS 128 KiB > budget 64 KiB (MOEBUG-061)"
        ), name
        assert unsupported(name, 3072, "gfx950") is None  # 160 KiB budget
    assert unsupported(SMALL, 3072, "gfx942") is None
    assert unsupported(BIG[0], 3072, None) is None  # arch unknown: not guessed
    # stage 2 and non-a16 kernels have no LDS rule here
    assert (
        unsupported("flydsl_moe2_abf16_wbf16_bf16_t32x128x128_atomic", 3072, "gfx942")
        is None
    )


def test_every_enumerated_a16w16_config_over_64_kib_is_dropped_on_gfx942():
    from aiter.ops.flydsl import moe_kernels as mk

    names = mk.get_flydsl_stage1_kernels_a16w16("bf16")
    dropped = {n for n in names if unsupported(n, 3072, "gfx942")}
    assert set(BIG) - dropped == {b for b in BIG if b not in names}
    for n in names:
        p = mk.get_flydsl_kernel_params(n)
        need = mk._gemm1_lds_bytes(p["tile_m"], p["tile_n"], p["tile_k"], p["k_wave"])
        assert (n in dropped) == (need > 64 * 1024), n
        assert unsupported(n, 3072, "gfx950") is None


def _job():
    class Table:
        def __init__(self):
            self.rows = []

        def add(self, r):
            self.rows.append(r)

    row = {c: "" for c in schema.COLS}
    row.update(case_id="case1", kind="fused", token="8", modes="warm")
    case = schema.Case(row, {})
    job = bar_runner.BarGroupJob.__new__(bar_runner.BarGroupJob)
    job.cases, job.pending = [case], {"case1": case}
    job.bad, job.faults = {}, {}
    job.cand_table, job.meta = Table(), {}
    return job


def test_a_poisoning_fault_is_recorded_with_its_error_before_the_restart():
    job = _job()
    job.handle_other(
        {
            "event": "gpu_fault",
            "case_id": "case1",
            "cand_id": "k1",
            "detail": "hipErrorIllegalState",
        }
    )
    job.phase, job.inflight = "cand:k1", "case1"
    job._fail("crash", "worker exit 70")
    row = job.cand_table.rows[0]
    assert row["status"] == "crash"
    assert "worker exit 70" in row["detail"] and "hipErrorIllegalState" in row["detail"]
    assert job.bad["case1"] == {"k1"}  # the restarted worker skips it
    # a crash with no word from the worker keeps the plain detail
    job.phase = "cand:k2"
    job._fail("crash", "worker exit -11")
    assert job.cand_table.rows[1]["detail"] == "worker exit -11"
