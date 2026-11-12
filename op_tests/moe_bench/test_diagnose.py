# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""diagnose's variant machinery (the GPU call itself is exercised on a node)."""

import types

import pytest

torch = pytest.importorskip("torch")

from op_tests.moe_bench import diagnose


def test_swiglu_variants_change_exactly_what_they_name():
    import aiter.fused_moe as fm

    g = torch.tensor([[-3.0, 0.5, 20.0]], dtype=torch.float64, device="cpu")
    u = torch.tensor([[2.0, -0.5, 30.0]], dtype=torch.float64, device="cpu")
    base = fm.swiglu(g, u, limit=None)
    with diagnose.swiglu_variant():
        assert torch.equal(fm.swiglu(g, u, limit=None), base)  # default limit is 7.0
    with diagnose.swiglu_variant(mode="none"):
        assert not torch.equal(fm.swiglu(g, u, limit=None), base)
    with diagnose.swiglu_variant(plus1=False):
        no1 = fm.swiglu(g, u, limit=None)
    assert torch.allclose(no1, base * u.clamp(-7, 7) / (u.clamp(-7, 7) + 1))
    with diagnose.swiglu_variant(swap=True):
        assert not torch.equal(fm.swiglu(g, u, limit=None), base)
    assert fm.swiglu(g, u, limit=None).shape == base.shape  # the real function is back
    assert torch.equal(fm.swiglu(g, u, limit=None), base)


def test_interleave_position_tiles_16_gate_then_16_up():
    S = types.SimpleNamespace(I=64)
    pos = diagnose.interleave_position(S)
    assert pos[:16].tolist() == list(range(16))  # gate tile 0
    assert pos[64:80].tolist() == list(range(16, 32))  # up tile 0 follows it
    assert pos[16].item() == 32  # gate tile 1
    assert sorted(pos.tolist()) == list(range(128))
