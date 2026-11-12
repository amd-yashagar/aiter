# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch-only coverage for Situv2 calls that no FlyDSL path can serve.

The CK 2-stage codegen has no Situv2 activation. A call that falls through to it
used to fail only after a long JIT build of the CK module; it must raise
NotImplementedError from dispatch instead.
"""

import functools

import pytest

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode

_SKIP = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="MI355X (gfx950) dispatch required"
)


def _dispatch(
    *,
    token=32,
    model_dim=256,
    inter_dim=128,
    expert=8,
    topk=2,
    dtype=dtypes.bf16,
    q_dtype_a=dtypes.fp8,
    gate_mode=GateMode.SEPARATED,
    gate_up_interleave=None,
):
    get_2stage_cfgs.cache_clear()
    return get_2stage_cfgs(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtype,
        q_dtype_a,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Situv2,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode=gate_mode,
        gate_up_interleave=gate_up_interleave,
    )


def _backend(fn):
    target = fn.func if isinstance(fn, functools.partial) else fn
    return getattr(target, "__name__", str(target))


@_SKIP
@pytest.mark.parametrize("dtype", [dtypes.bf16, dtypes.fp16])
@pytest.mark.parametrize(
    "shape",
    [
        {"token": 32, "model_dim": 256, "inter_dim": 128, "expert": 8, "topk": 2},
        {"token": 64, "model_dim": 512, "inter_dim": 256, "expert": 16, "topk": 4},
        {"token": 1024, "model_dim": 512, "inter_dim": 128, "expert": 8, "topk": 2},
    ],
)
def test_a8w4_situv2_separated_weights_raise_instead_of_ck_build(shape, dtype):
    with pytest.raises(NotImplementedError, match="Situv2 has no CK 2-stage"):
        _dispatch(dtype=dtype, **shape)


@_SKIP
def test_a4w4_situv2_without_flydsl_layout_raises():
    with pytest.raises(NotImplementedError, match="Situv2 has no CK 2-stage"):
        _dispatch(q_dtype_a=dtypes.fp4x2, gate_mode=GateMode.INTERLEAVE)


@_SKIP
@pytest.mark.parametrize("dtype", [dtypes.bf16, dtypes.fp16])
def test_a16w4_situv2_stays_on_flydsl(dtype):
    meta = _dispatch(dtype=dtype, q_dtype_a=dtypes.bf16)
    assert "flydsl" in _backend(meta.stage1)
    assert "flydsl" in _backend(meta.stage2)


@_SKIP
def test_a8w4_situv2_interleaved_weights_stay_on_flydsl():
    meta = _dispatch(
        model_dim=3584,
        inter_dim=384,
        expert=16,
        topk=4,
        gate_mode=GateMode.INTERLEAVE,
        gate_up_interleave=True,
    )
    assert "flydsl" in _backend(meta.stage1)
