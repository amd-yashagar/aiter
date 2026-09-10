# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Two-way KV split around the unchanged gfx942 production FMHA assembly.

This is deliberately narrow: it proves and exposes the ticket-shape scheduling
win without modifying the hand-written kernel. Two independent ASM launches
consume contiguous, 32-key-aligned KV slices on separate streams; the existing
fused logsumexp kernel combines their normalized outputs.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from aiter.jit.core import compile_ops

from .parallel import combine_split_kv


@lru_cache(maxsize=32)
def _stream_bundle(device_index: int, _current_stream_id: int):
    """Keep split launch streams/events alive; constructing them per call serializes."""
    device = torch.device("cuda", device_index)
    workers = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
    return workers, torch.cuda.Event(), (torch.cuda.Event(), torch.cuda.Event())


@compile_ops(
    "module_fmha_hd192_split_asm",
    fc_name="fmha_hd192_split2_producer",
    ffi_type="ctypes",
)
def _fmha_hd192_split2_producer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_parts: torch.Tensor,
    lse_parts: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_splits: int,
    softmax_scale: float,
) -> None: ...


def _validate_split2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    out: torch.Tensor | None,
) -> None:
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx942":
        raise ValueError("ASM split-2 requires gfx942")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q/k/v must be CUDA/HIP tensors")
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        raise ValueError("ASM split-2 supports bf16 q/k/v only")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must be packed THD tensors")
    if q.shape[-1] != 192 or k.shape[-1] != 192 or v.shape[-1] != 128:
        raise ValueError("ASM split-2 requires Dqk=192 and Dv=128")
    if q.shape[1] != k.shape[1] or k.shape[1] != v.shape[1]:
        raise ValueError("ASM split-2 currently requires Hq=Hkv")
    if cu_seqlens_q.numel() != 2 or cu_seqlens_k.numel() != 2:
        raise ValueError("ASM split-2 currently supports one packed sequence")
    if k.shape[0] != v.shape[0]:
        raise ValueError("K and V token counts must match")
    if int(max_seqlen_q) != q.shape[0] or int(max_seqlen_k) != k.shape[0]:
        raise ValueError("max sequence lengths must match the single packed sequence")
    if k.shape[0] < 64:
        raise ValueError("ASM split-2 requires at least two 32-key tiles")
    if out is not None and (
        out.device != q.device
        or out.dtype != q.dtype
        or tuple(out.shape) != (q.shape[0], q.shape[1], v.shape[2])
    ):
        raise ValueError("out must match [total_q, Hq, Dv], dtype, and device")


def flash_attn_varlen_gfx942_asm_split2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    *,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the production noncausal D192x128 ASM on two concurrent KV slices."""
    _validate_split2(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        out,
    )

    # Bypass flash_attn_varlen_func: its FlyDSL opt-in router must not replace
    # the production ASM producer used by this split wrapper.
    from aiter.ops.mha import fmha_v3_varlen_fwd

    scale = 1.0 / math.sqrt(192) if softmax_scale is None else float(softmax_scale)
    total_tiles = (k.shape[0] + 31) // 32
    first_tiles = (total_tiles + 1) // 2
    split = min(k.shape[0], first_tiles * 32)
    bounds = ((0, split), (split, k.shape[0]))
    cu_k_parts = [
        torch.tensor([0, hi - lo], dtype=torch.int32, device=q.device)
        for lo, hi in bounds
    ]

    current = torch.cuda.current_stream(q.device)
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    workers, ready, done = _stream_bundle(device_index, int(current.cuda_stream))
    current.record_event(ready)

    o_parts = torch.empty(
        2,
        q.shape[0],
        q.shape[1],
        v.shape[2],
        dtype=q.dtype,
        device=q.device,
    )
    lse_parts = []
    for index, ((lo, hi), worker) in enumerate(zip(bounds, workers, strict=True)):
        with torch.cuda.stream(worker):
            worker.wait_event(ready)
            o_part, lse_part, _, _ = fmha_v3_varlen_fwd(
                q,
                k[lo:hi],
                v[lo:hi],
                cu_seqlens_q,
                cu_k_parts[index],
                int(max_seqlen_q),
                hi - lo,
                0,
                0.0,
                scale,
                0.0,
                False,
                False,
                -1,
                -1,
                True,
                False,
                1,
                o_parts[index],
            )
            done[index].record(worker)
            assert o_part.data_ptr() == o_parts[index].data_ptr()
            lse_parts.append(lse_part)

    for event in done:
        current.wait_event(event)
    result, lse = combine_split_kv(
        o_parts,
        torch.stack(lse_parts, dim=0),
    )
    if out is not None:
        out.copy_(result)
        result = out
    if return_lse:
        return result, lse
    return result


def flash_attn_varlen_gfx942_asm_split2_unified(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    *,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
    kv_splits: int = 2,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Launch one split-aware ASM producer grid, then fused logsumexp combine."""
    _validate_split2(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        out,
    )
    scale = 1.0 / math.sqrt(192) if softmax_scale is None else float(softmax_scale)
    if kv_splits < 2 or kv_splits > 8 or k.shape[0] < kv_splits * 32:
        raise ValueError("kv_splits must be in [2, 8] with at least one tile per split")
    kv_tiles = (k.shape[0] + 31) // 32
    split_tiles = (kv_tiles + kv_splits - 1) // kv_splits
    if split_tiles * (kv_splits - 1) >= kv_tiles:
        raise ValueError("aligned split partition would contain an empty final split")
    out_parts = torch.empty(
        kv_splits,
        q.shape[0],
        q.shape[1],
        v.shape[2],
        dtype=q.dtype,
        device=q.device,
    )
    lse_parts = torch.empty(
        kv_splits,
        q.shape[1],
        q.shape[0],
        dtype=torch.float32,
        device=q.device,
    )
    _fmha_hd192_split2_producer(
        q,
        k,
        v,
        out_parts,
        lse_parts,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        int(kv_splits),
        scale,
    )
    result, lse = combine_split_kv(out_parts, lse_parts, out=out)
    if return_lse:
        return result, lse
    return result
