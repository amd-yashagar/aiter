# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL dense absorbed MLA prefill API.

Kernel: ``kernels.mla_prefill_dense``. Supports bf16 absorb attention with
compile-time ``qk_dim`` / ``v_dim`` (multiples of 16, ``v_dim <= qk_dim``).
"""

from __future__ import annotations

import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.mla_prefill_dense import (
    DEFAULT_BLOCK_M,
    DEFAULT_BLOCK_N,
    DEFAULT_NUM_WARPS,
    DEFAULT_WAVES_PER_EU,
    QK_HEAD_DIM,
    SUPPORTED_GFX,
    V_HEAD_DIM,
    _validate_geometry,
    choose_num_kv_splits,
    flydsl_mla_prefill_dense_fwd,
    prepare_mla_prefill_workspace as prepare_mla_prefill_dense_workspace,
    run_mla_prefill_prepared as run_mla_prefill_dense_prepared,
)
from .kernels.mla_prefill_mqa import (
    DEFAULT_NUM_HEADS as MQA_HEADS,
    DEFAULT_SEQ_TILE as MQA_SEQ_TILE,
    flydsl_mla_prefill_mqa_fwd,
    prepare_mla_prefill_mqa_workspace,
    run_mla_prefill_mqa_prepared,
)

__all__ = [
    "QK_HEAD_DIM",
    "V_HEAD_DIM",
    "choose_num_kv_splits",
    "flydsl_mla_prefill_fwd",
    "flydsl_mla_prefill_supported",
    "prepare_mla_prefill_workspace",
    "run_mla_prefill_prepared",
]


def flydsl_mla_prefill_supported(
    *,
    q_dtype: torch.dtype = torch.bfloat16,
    kv_dtype: torch.dtype = torch.bfloat16,
    qk_dim: int = QK_HEAD_DIM,
    v_dim: int = V_HEAD_DIM,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> bool:
    """Return True when this build can run the dense absorb prefill kernel."""
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        return False
    if q_dtype != torch.bfloat16 or kv_dtype != torch.bfloat16:
        return False
    try:
        _validate_geometry(qk_dim, v_dim, block_m, block_n, num_warps)
    except ValueError:
        return False
    return True


def flydsl_mla_prefill_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float | None = None,
    *,
    is_causal: bool = False,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    num_warps: int = DEFAULT_NUM_WARPS,
    waves_per_eu: int = DEFAULT_WAVES_PER_EU,
    num_kv_splits: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Dense absorb MLA prefill (bf16). MQA shared-KV path when H=12."""
    if (
        q.shape[1] == MQA_HEADS
        and q.shape[-1] == QK_HEAD_DIM
        and o.shape[-1] == V_HEAD_DIM
        and (MQA_SEQ_TILE * MQA_HEADS) % 16 == 0
    ):
        return flydsl_mla_prefill_mqa_fwd(
            q,
            kv_buffer,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            is_causal=is_causal,
            waves_per_eu=waves_per_eu,
            num_kv_splits=num_kv_splits,
            stream=stream,
        )
    if not flydsl_mla_prefill_supported(
        q_dtype=q.dtype,
        kv_dtype=kv_buffer.dtype,
        qk_dim=q.shape[-1],
        v_dim=o.shape[-1],
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
    ):
        raise RuntimeError(
            "flydsl_mla_prefill_fwd unsupported for "
            f"gfx={get_gfx()} q={q.dtype} kv={kv_buffer.dtype} "
            f"qk={q.shape[-1]} v={o.shape[-1]} "
            f"block_m={block_m} block_n={block_n} warps={num_warps}"
        )
    return flydsl_mla_prefill_dense_fwd(
        q,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        is_causal=is_causal,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        num_kv_splits=num_kv_splits,
        stream=stream,
    )


def prepare_mla_prefill_workspace(*args, **kwargs):
    q = args[0] if args else kwargs.get("q")
    if q is not None and q.ndim == 3 and q.shape[1] == MQA_HEADS:
        mqa_keys = {
            "is_causal",
            "seq_tile",
            "block_n",
            "waves_per_eu",
            "num_kv_splits",
            "k_double_buffer",
            "q_preload",
        }
        filtered = {k: v for k, v in kwargs.items() if k in mqa_keys}
        return prepare_mla_prefill_mqa_workspace(*args, **filtered)
    return prepare_mla_prefill_dense_workspace(*args, **kwargs)


def run_mla_prefill_prepared(ws, stream=None):
    if hasattr(ws, "seq_tile"):
        return run_mla_prefill_mqa_prepared(ws, stream=stream)
    return run_mla_prefill_dense_prepared(ws, stream=stream)
