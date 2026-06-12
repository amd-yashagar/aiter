# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public API for the FlyDSL grouped jagged x dense BMM with a fused bias add.

Computes, for a jagged batch of ``B`` variable-length sequences packed into a
single ``(L, D)`` tensor with ``B + 1`` row offsets::

    out[s:e] = jagged[s:e] @ dense[b] + bias[b]        # broadcast bias (B, K)
    out[s:e] = jagged[s:e] @ dense[b] + bias[s:e]      # elementwise bias (L, K)

This is a thin adapter over the ``jdbba`` variant framework (the single dispatch
+ launch + per-shape table system). The forward broadcast path runs through the
framework's per-shape variant selection (``jdbba.jagged_dense_bmm``); the
elementwise / explicit-tuning / custom-out paths and the d_jagged backward run
the ``lowlevel`` kernel directly via ``jdbba.launch.run_lowlevel`` (the variants
that those features require).

Usage::

    from aiter.ops.flydsl import flydsl_jagged_dense_bmm_broadcast_add
    out = flydsl_jagged_dense_bmm_broadcast_add(
        max_seq_len=max_seq_len, seq_offsets=seq_offsets,
        jagged=jagged, dense=dense, bias=bias,
    )
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from .jdbba import JdbbaConfig, resolve_config
from .jdbba import jagged_dense_bmm as _jdbba_forward
from .jdbba.launch import run_lowlevel
from .utils import is_flydsl_available

__all__ = [
    "flydsl_jagged_dense_bmm_broadcast_add",
    "flydsl_jagged_dense_bmm_add",
]

# Lowlevel tuning knobs exposed for forced-config benchmarking (None => dispatch).
_KNOBS = (
    "tile_m", "tile_n", "tile_k", "stages",
    "block_m_warps", "block_n_warps", "block_k_warps", "b_to_lds", "waves_per_eu",
)


def _lowlevel_cfg(B, reduction_k, output_n, max_seq_len, knobs) -> JdbbaConfig:
    cfg = resolve_config(
        n_groups=B, reduction_k=reduction_k, output_n=output_n,
        max_seq_len=max_seq_len, force_variant="lowlevel",
    )
    overrides = {k: v for k, v in knobs.items() if v is not None}
    return cfg.merged(**overrides) if overrides else cfg


def _bwd_dense_bias_ref(
    seq_offsets: Tensor, jagged: Tensor, d_out: Tensor, *, elementwise: bool,
) -> Tuple[Tensor, Tensor]:
    """Reference d_dense / d_bias (the forward is the tuned kernel; weight/bias
    grads use this reference). Numerically correct (fp32 accumulation). ``d_bias``
    is sync-free (``index_add``); ``d_dense`` does per-group eager matmuls reading
    the offsets with a single host transfer (``tolist``), not one sync per group.
    """
    B = int(seq_offsets.numel()) - 1
    d_in = jagged.shape[1]
    k_out = d_out.shape[1]
    device = d_out.device
    seq_lens = seq_offsets[1:] - seq_offsets[:-1]
    row_to_b = torch.repeat_interleave(torch.arange(B, device=device), seq_lens)

    if elementwise:
        d_bias = d_out
    else:
        acc = torch.zeros((B, k_out), dtype=torch.float32, device=device)
        acc.index_add_(0, row_to_b, d_out.float())
        d_bias = acc.to(d_out.dtype)

    d_dense = torch.zeros((B, d_in, k_out), dtype=d_out.dtype, device=device)
    offs = seq_offsets.tolist()
    for b in range(B):
        s, e = offs[b], offs[b + 1]
        if e > s:
            d_dense[b] = jagged[s:e].transpose(0, 1) @ d_out[s:e]
    return d_dense, d_bias


class _FlydslJaggedDenseBmmAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, max_seq_len, seq_offsets, jagged, dense, bias, elementwise):
        if jagged.dtype != torch.bfloat16:
            raise ValueError("only bf16 is supported")
        if not jagged.is_cuda:
            raise ValueError("tensors must be on CUDA/ROCm device")
        L, d_in = jagged.shape
        B, d_chk, k_out = dense.shape
        if d_chk != d_in:
            raise ValueError("jagged.shape[1] must match dense.shape[1] (D)")
        if seq_offsets.shape[0] != B + 1:
            raise ValueError("seq_offsets must have length B+1")
        if elementwise and bias.shape != (L, k_out):
            raise ValueError("elementwise bias must be (L, K)")
        if not elementwise and bias.shape != (B, k_out):
            raise ValueError("broadcast bias must be (B, K)")
        if not jagged.is_contiguous():
            jagged = jagged.contiguous()
        if not bias.is_contiguous():
            bias = bias.contiguous()

        if elementwise:
            out = torch.empty((L, k_out), dtype=jagged.dtype, device=jagged.device)
            run_lowlevel(
                out=out, jagged=jagged, dense=dense, bias=bias,
                seq_offsets=seq_offsets, max_seq_len=max_seq_len,
                cfg=_lowlevel_cfg(B, d_in, k_out, max_seq_len, {}),
                has_bias=True, elementwise=True, transpose_dense=True,
            )
        else:
            out = _jdbba_forward(
                jagged, dense, bias, seq_offsets, max_seq_len, uniform_seqlen=False,
            )
        ctx.save_for_backward(seq_offsets, jagged, dense)
        ctx.max_seq_len = max_seq_len
        ctx.B, ctx.K, ctx.D = B, k_out, d_in
        ctx.elementwise = elementwise
        return out

    @staticmethod
    def backward(ctx, d_out):
        seq_offsets, jagged, dense = ctx.saved_tensors
        d_jagged = torch.empty_like(jagged)
        # d_jagged = d_out @ dense^T per group: a transposed, bias-free launch.
        run_lowlevel(
            out=d_jagged, jagged=d_out, dense=dense, bias=None,
            seq_offsets=seq_offsets, max_seq_len=ctx.max_seq_len,
            cfg=resolve_config(
                n_groups=ctx.B, reduction_k=ctx.K, output_n=ctx.D,
                max_seq_len=ctx.max_seq_len, force_variant="lowlevel",
            ),
            has_bias=False, elementwise=False, transpose_dense=False,
            out_n=ctx.D, red_k=ctx.K,
        )
        d_dense, d_bias = _bwd_dense_bias_ref(
            seq_offsets, jagged, d_out, elementwise=ctx.elementwise
        )
        return None, None, d_jagged, d_dense, d_bias, None


def flydsl_jagged_dense_bmm_add(
    *,
    max_seq_len: int,
    seq_offsets: Tensor,
    jagged: Tensor,
    dense: Tensor,
    bias: Tensor,
    elementwise: bool = False,
    out: Optional[Tensor] = None,
    stream: Optional[torch.cuda.Stream] = None,
    **knobs,
) -> Tensor:
    """Grouped jagged x dense BMM plus a fused bias add (broadcast or elementwise).

    Default (broadcast, no grad, no ``out``) runs the framework's per-shape
    variant selection. ``elementwise=True``, an explicit ``out``, or explicit
    lowlevel tuning knobs (``tile_m`` …) force the lowlevel kernel. Tensors:
    ``jagged (L, D)``, ``dense (B, D, K)``, ``bias (B, K)`` or ``(L, K)``.
    """
    if not is_flydsl_available():
        raise RuntimeError("flydsl is not installed")
    bad = set(knobs) - set(_KNOBS)
    if bad:
        raise TypeError(f"unexpected keyword args: {sorted(bad)}")

    if out is None and (jagged.requires_grad or dense.requires_grad or bias.requires_grad):
        return _FlydslJaggedDenseBmmAddFunction.apply(
            max_seq_len, seq_offsets, jagged, dense, bias, elementwise
        )

    B, d_in, k_out = dense.shape
    forced = elementwise or out is not None or any(v is not None for v in knobs.values())
    if not forced:
        return _jdbba_forward(
            jagged, dense, bias, seq_offsets, max_seq_len,
            uniform_seqlen=False, stream=stream,
        )

    if out is None:
        out = torch.empty((jagged.shape[0], k_out), dtype=jagged.dtype, device=jagged.device)
    elif out.shape != (jagged.shape[0], k_out):
        raise ValueError("out shape mismatch")
    return run_lowlevel(
        out=out, jagged=jagged, dense=dense, bias=bias, seq_offsets=seq_offsets,
        max_seq_len=max_seq_len, cfg=_lowlevel_cfg(B, d_in, k_out, max_seq_len, knobs),
        has_bias=True, elementwise=elementwise, transpose_dense=True, stream=stream,
    )


def flydsl_jagged_dense_bmm_broadcast_add(
    *,
    max_seq_len: int,
    seq_offsets: Tensor,
    jagged: Tensor,
    dense: Tensor,
    bias: Tensor,
    out: Optional[Tensor] = None,
    stream: Optional[torch.cuda.Stream] = None,
    **knobs,
) -> Tensor:
    """Grouped jagged x dense BMM plus a per-batch broadcast bias ``(B, K)``."""
    return flydsl_jagged_dense_bmm_add(
        max_seq_len=max_seq_len, seq_offsets=seq_offsets, jagged=jagged,
        dense=dense, bias=bias, elementwise=False, out=out, stream=stream, **knobs,
    )
