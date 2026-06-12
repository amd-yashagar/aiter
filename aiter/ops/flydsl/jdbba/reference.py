# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Torch eager reference + correctness gate for jdbba v3 (no fbgemm dependency)."""

from __future__ import annotations

import torch


def torch_reference(jagged, dense, bias, seq_offsets):
    """Per-group ``jagged[s:e] @ dense[b] + bias[b]`` in fp32, cast to in dtype.

    jagged (L, K_red) ; dense (B, K_red, N_out) ; bias (B, N_out).
    """
    L = jagged.shape[0]
    n_out = dense.shape[2]
    out = torch.zeros((L, n_out), dtype=jagged.dtype, device=jagged.device)
    so = seq_offsets.to(torch.int64)
    for b in range(dense.shape[0]):
        s = int(so[b].item())
        e = int(so[b + 1].item())
        if e > s:
            out[s:e] = (
                jagged[s:e].float() @ dense[b].float() + bias[b].float()[None, :]
            ).to(jagged.dtype)
    return out


def cosine(a, b) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def check(got, ref, *, tol=0.999) -> tuple[bool, float]:
    cos = cosine(got, ref)
    return (cos > tol), cos
