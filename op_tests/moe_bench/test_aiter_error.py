# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A candidate aiter itself cannot run (a NameError in aiter's sources) is an ``aiter_error``."""


def test_a_nameerror_inside_aiter_is_an_aiter_error_not_a_harness_error():
    """k_batch > 1 stage 1: silu_and_mul_fq.py reads topk_ids_p, defined on one branch."""
    from op_tests.moe_bench import bar_worker

    def raise_from(filename):
        code = compile("def f():\n    return topk_ids_p[0]\nf()", filename, "exec")
        try:
            exec(code, {})  # noqa: S102
        except NameError as e:
            return e

    e = raise_from("/x/aiter/aiter/ops/flydsl/kernels/silu_and_mul_fq.py")
    detail = bar_worker.aiter_error_detail(e)
    assert "name 'topk_ids_p' is not defined" in detail
    assert "ops/flydsl/kernels/silu_and_mul_fq.py:2" in detail
    # the harness's own NameError (a bug of ours) is not hidden under the status
    assert (
        bar_worker.aiter_error_detail(raise_from("/x/aiter/op_tests/moe_bench/x.py"))
        is None
    )
    assert bar_worker.aiter_error_detail(raise_from("/somewhere/else.py")) is None
    assert bar_worker.aiter_error_detail(RuntimeError("x")) is None
