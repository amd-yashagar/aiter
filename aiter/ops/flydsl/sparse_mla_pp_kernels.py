# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL absorbed sparse MLA 8-wave ping-pong host wrapper. bf16, H=16, D=512."""

from __future__ import annotations

import math
import struct

import flydsl.expr as fx
import torch

from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.sparse_mla_pp import (
    BLOCK_K,
    TOKENS_PER_WG,
    compile_sparse_mla_pp,
)
from aiter.ops.flydsl.kernels.sparse_mla_qblock import HEAD_DIM, NUM_HEADS
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg
from aiter.ops.flydsl.sparse_mla_qblock_kernels import (
    _index_span_supported,
    _require_contract,
    _require_gfx942,
    _uniform_topk,
)

__all__ = ["sparse_mla_pingpong_fwd_flydsl"]


def _checked_scale(softmax_scale: float | None) -> float:
    scale = 1.0 / math.sqrt(HEAD_DIM) if softmax_scale is None else float(softmax_scale)
    try:
        # Match the FP32 ABI and the kernel's scale * FP32(LOG2E), in that order.
        scale = struct.unpack("f", struct.pack("f", scale))[0]
        log2e = struct.unpack("f", struct.pack("f", LOG2E))[0]
        effective = struct.unpack("f", struct.pack("f", scale * log2e))[0]
    except OverflowError as exc:
        raise ValueError(
            "softmax_scale and scale * LOG2E must be finite and positive in FP32"
        ) from exc
    if not (
        math.isfinite(scale)
        and scale > 0
        and math.isfinite(effective)
        and effective > 0
    ):
        raise ValueError(
            "softmax_scale and scale * LOG2E must be finite and positive in FP32"
        )
    return scale


def sparse_mla_pingpong_fwd_flydsl(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_indices: torch.Tensor,
    out: torch.Tensor,
    kv_indptr: torch.Tensor | None = None,
    *,
    attn_sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    uniform_topk: int | None = None,
    verify_csr: bool = True,
):
    """Explicit experimental gfx942 backend; never selected automatically.

    BF16 contiguous q[T,16,512], kv[N,512], packed valid KV indices,
    positive T and topk divisible by 32. Four queries per 512-thread workgroup,
    64 KiB LDS; odd T is masked. Sinks affect only the denominator.
    No Q-block packing or block_q option: lists may differ between queries.
    All operands must share q's GPU. Scale and scale * LOG2E must remain
    finite and positive after FP32 rounding.

    Port of Zhuo Su's MIT-licensed wrapper at
    hellozhuo-amd/aiter@4d33e0e171f8c9e8a1e2388c0a3fd3ffa1e408a2.
    """
    _require_gfx942()
    n_tok, _n_kv = _require_contract(q, kv, out)
    if q.device.type != "cuda":
        raise ValueError("q must be on a GPU")
    for name, operand in (
        ("kv", kv),
        ("out", out),
        ("kv_indices", kv_indices),
        ("kv_indptr", kv_indptr),
        ("attn_sink", attn_sink),
    ):
        if operand is not None and operand.device != q.device:
            raise ValueError(f"{name} must be on the same GPU device as q")
    flat, topk = _uniform_topk(
        kv_indices, n_tok, kv_indptr, uniform_topk, verify_csr=verify_csr
    )
    if topk % (2 * BLOCK_K):
        raise ValueError("FlyDSL sparse MLA ping-pong requires topk divisible by 32")
    if not _index_span_supported(n_tok, topk):
        raise ValueError("FlyDSL sparse MLA index byte span must fit uint32")
    scale = _checked_scale(softmax_scale)
    has_sink = attn_sink is not None
    kernel = compile_sparse_mla_pp(has_sink)
    n_tiles = -(-n_tok // TOKENS_PER_WG)
    idx = flat.to(dtype=torch.int32).contiguous()
    if has_sink:
        attn_sink = attn_sink.reshape(-1).to(torch.float32).contiguous()
        if attn_sink.numel() != NUM_HEADS:
            raise ValueError(f"attn_sink must have {NUM_HEADS} elements")
    else:
        attn_sink = torch.empty(1, dtype=torch.float32, device=q.device)
    stream = torch.cuda.current_stream(q.device)
    _run_compiled(
        kernel,
        ptr_arg(q, fx.BFloat16),
        ptr_arg(kv, fx.BFloat16),
        ptr_arg(idx, fx.Int32),
        ptr_arg(out, fx.BFloat16),
        int(n_tok),
        int(topk),
        scale,
        ptr_arg(attn_sink, fx.Float32),
        int(n_tiles),
        fx.Stream(stream),
    )
    return out
