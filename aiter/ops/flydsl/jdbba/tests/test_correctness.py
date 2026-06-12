# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for jdbba v3: every variant + the dispatcher, on jagged and
uniform regimes, against the torch eager reference.

Run: pytest aiter/ops/flydsl/jdbba/tests/test_correctness.py
(requires a ROCm GPU + flydsl).
"""

from __future__ import annotations

import pytest
import torch

from aiter.ops.flydsl.jdbba import (
    jagged_dense_bmm,
    list_variants,
    torch_reference,
)

# Correctness is regime/shape-structure dependent, not Mi-magnitude dependent, so
# the unit test uses a modest Mi to stay within memory; the real N=16384
# deployment shapes are correctness-gated in the bench harness sweep.
SHAPES = [
    (120, 256, 256, 2048),
    (120, 512, 512, 2048),
    (1024, 256, 256, 1024),
    (1024, 512, 512, 1024),
    (8, 256, 256, 4096),
]

_HAS_GPU = torch.cuda.is_available()


def _make_inputs(B, K_red, N_out, Mi, *, regime, seed=1234, device="cuda"):
    torch.manual_seed(0)
    if regime == "uniform":
        so = torch.arange(0, (B + 1) * Mi, Mi, dtype=torch.int32, device=device)
    else:
        g = torch.Generator().manual_seed(seed)
        u = torch.rand(B, generator=g)
        t = (Mi * (u**4)).floor().to(torch.int64)
        t[: max(1, B // 5)] = 0
        t[-1] = Mi
        if B > 1:
            t[-2] = int(0.9 * Mi)
        so = torch.zeros(B + 1, dtype=torch.int32)
        for i in range(B):
            so[i + 1] = so[i] + int(t[i])
        so = so.to(device)
    L = int(so[-1].item())
    jg = torch.randn(max(L, 1), K_red, dtype=torch.bfloat16, device=device)
    dn = torch.randn(B, K_red, N_out, dtype=torch.bfloat16, device=device)
    bs = torch.randn(B, N_out, dtype=torch.bfloat16, device=device)
    return jg, dn, bs, so


@pytest.mark.skipif(not _HAS_GPU, reason="requires ROCm GPU")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("regime", ["skew", "uniform"])
@pytest.mark.parametrize("variant", list(list_variants()))
def test_variant(variant, regime, shape):
    B, D, N, Mi = shape
    jg, dn, bs, so = _make_inputs(B, D, N, Mi, regime=regime)
    ref = torch_reference(jg, dn, bs, so)
    out = jagged_dense_bmm(jg, dn, bs, so, Mi, variant=variant,
                           uniform_seqlen=(regime == "uniform"))
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), out.float().flatten(), dim=0).item()
    del jg, dn, bs, so, ref, out
    torch.cuda.empty_cache()
    assert cos > 0.999, f"{variant} {regime} B{B}D{D}K{N}: cos={cos}"


@pytest.mark.skipif(not _HAS_GPU, reason="requires ROCm GPU")
@pytest.mark.parametrize("shape", SHAPES)
def test_dispatch(shape):
    B, D, N, Mi = shape
    jg, dn, bs, so = _make_inputs(B, D, N, Mi, regime="skew")
    ref = torch_reference(jg, dn, bs, so)
    out = jagged_dense_bmm(jg, dn, bs, so, Mi, uniform_seqlen=False)
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), out.float().flatten(), dim=0).item()
    assert cos > 0.999, f"dispatch B{B}D{D}K{N}: cos={cos}"


def _fp32_reference(jagged, dense, bias, seq_offsets):
    """Per-group matmul accumulated in fp32 (kept fp32, not cast to bf16)."""
    L = jagged.shape[0]
    n_out = dense.shape[2]
    out = torch.zeros((L, n_out), dtype=torch.float32, device=jagged.device)
    so = seq_offsets.to(torch.int64)
    for b in range(dense.shape[0]):
        s, e = int(so[b].item()), int(so[b + 1].item())
        if e > s:
            out[s:e] = jagged[s:e].float() @ dense[b].float() + bias[b].float()[None, :]
    return out


@pytest.mark.skipif(not _HAS_GPU, reason="requires ROCm GPU")
@pytest.mark.parametrize("variant", list(list_variants()))
def test_forward_tight_fp32(variant):
    """Tighter than cosine: bf16 output must match an fp32 reference within the
    op's own bf16 output-rounding band (mean-relative error), not just direction."""
    B, D, N, Mi = 64, 256, 256, 1024
    jg, dn, bs, so = _make_inputs(B, D, N, Mi, regime="skew")
    ref = _fp32_reference(jg, dn, bs, so)
    out = jagged_dense_bmm(jg, dn, bs, so, Mi, variant=variant, uniform_seqlen=False).float()
    denom = ref.abs().mean().clamp(min=1e-3)
    mean_rel = (out - ref).abs().mean() / denom
    # bf16 has ~3 decimal digits; mean-relative error from bf16 output rounding is
    # well under 1%. A real indexing/accumulation bug blows past this.
    assert mean_rel < 0.01, f"{variant}: mean_rel={mean_rel.item():.4f}"


@pytest.mark.skipif(not _HAS_GPU, reason="requires ROCm GPU")
def test_backward_matches_fp32_autograd():
    """Lock in the (correct) gradients of the autograd path against an
    autograd-through-fp32 reference. Guards d_jagged (kernel), d_dense / d_bias
    (reference path) from silent regressions."""
    from aiter.ops.flydsl import flydsl_jagged_dense_bmm_broadcast_add

    B, D, N, Mi = 16, 128, 128, 512
    jg, dn, bs, so = _make_inputs(B, D, N, Mi, regime="skew")

    def run(j, d, b, grad=False):
        j = j.clone().requires_grad_(grad)
        d = d.clone().requires_grad_(grad)
        b = b.clone().requires_grad_(grad)
        return j, d, b

    # op gradients
    jo, do, bo = run(jg, dn, bs, grad=True)
    out = flydsl_jagged_dense_bmm_broadcast_add(
        max_seq_len=Mi, seq_offsets=so, jagged=jo, dense=do, bias=bo)
    g = torch.randn_like(out)
    out.backward(g)

    # fp32 autograd reference
    jr = jg.float().clone().requires_grad_(True)
    dr = dn.float().clone().requires_grad_(True)
    br = bs.float().clone().requires_grad_(True)
    ref = _fp32_reference(jr, dr, br, so)
    ref.backward(g.float())

    for name, got, exp in (
        ("d_jagged", jo.grad, jr.grad),
        ("d_dense", do.grad, dr.grad),
        ("d_bias", bo.grad, br.grad),
    ):
        denom = exp.abs().mean().clamp(min=1e-3)
        mean_rel = (got.float() - exp).abs().mean() / denom
        assert mean_rel < 0.02, f"{name}: mean_rel={mean_rel.item():.4f}"
