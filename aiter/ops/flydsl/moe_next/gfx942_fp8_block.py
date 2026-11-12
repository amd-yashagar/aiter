# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 blockscale (per_1x128) MoE kernels for aiter.moe.

Prefill path, same quantization points as aiter's per_1x128 path: sort,
per_1x128 activation quant, Gate/Up (blockscale fold, SiLU/GELU, bf16), per_1x128
quant of the intermediate, Down (blockscale fold, route weight) into per-sorted-row
bf16 rows, and an fp32 sum of each token's rows. Tile configs are data:
configs/<gpu>/fp8-block.json.
"""

import functools
import json
import os

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.moe import MoeKernel, register_kernel
from aiter.ops.flydsl.moe_common import GateMode

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
_ACTS = {ActivationType.Silu: "silu", ActivationType.Gelu: "gelu"}
_LDS_BYTES = 64 * 1024


@functools.cache
def _configs(gpu):
    path = os.path.join(_CONFIG_DIR, gpu, "fp8-block.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _tile(config, gpu, inter_dim):
    cfg = dict(_configs(gpu).get(config or "default", {}))
    block_m = int(cfg.get("block_m", 128))
    # Down keeps the tile's whole intermediate row block in LDS.
    while block_m > 32 and block_m * inter_dim > _LDS_BYTES:
        block_m //= 2
    down_m = min(block_m, int(cfg.get("down_block_m", block_m)))
    while down_m > 16 and down_m * inter_dim > 16384:
        down_m //= 2
    return (
        block_m,
        int(cfg.get("block_n", 128)),
        int(cfg.get("down_n_split", 1)),
        down_m,
    )


@functools.cache
def _gateup(N, K, topk, block_m, block_n, act, out_fp8, a_bf16):
    from aiter.ops.flydsl.kernels.moe_next.gfx942_fp8_block.gateup import (
        compile_gateup,
    )

    return compile_gateup(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        activation=act,
        OUT_FP8=out_fp8,
        A_BF16=a_bf16,
    )


@functools.cache
def _down(N, K, topk, block_m, n_split, down_m):
    from aiter.ops.flydsl.kernels.moe_next.gfx942_fp8_block.down import compile_down

    while n_split > 1 and (N // 64) % (2 * n_split):
        n_split //= 2
    return compile_down(
        N=N, K=K, TOPK=topk, BLOCK_M=down_m, N_SPLIT=n_split, SORT_BLOCK_M=block_m
    )


@functools.cache
def _sorted_sum(topk, model_dim):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import sorted_sum

    return sorted_sum(topk, model_dim, 0)


def _launch(fn, *args):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import _ptr
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    args = [_ptr(a) if isinstance(a, torch.Tensor) else a for a in args]
    _run_compiled(fn, *args, torch.cuda.current_stream())


@register_kernel
class Gfx942Fp8BlockPrefill(MoeKernel):
    id = "gfx942_fp8_block.prefill"
    family = "fp8-block"
    archs = ("gfx942",)
    routing = "tiles"
    epilogue = "reduce"

    def supports(self, spec, m, topk, call=None):
        if spec.quant != QuantType.per_1x128:
            return "needs per_1x128"
        if spec.w_dtype != dtypes.fp8 or not spec.g1u1 or not spec.w_shuffled:
            return "needs shuffled fp8 g1u1 weights"
        if spec.act not in _ACTS:
            return f"activation {spec.act}"
        if spec.doweight_stage1:
            return "doweight_stage1"
        if spec.gate_mode != GateMode.SEPARATED:
            return "gate/up must be separated"
        if (
            spec.has_bias
            or spec.swiglu_limit
            or spec.hidden_pad
            or spec.intermediate_pad
        ):
            return "bias, swiglu_limit or padding"
        if spec.out_dtype not in (None, torch.bfloat16):
            return "bf16 output only"
        D, I = spec.model_dim, spec.inter_dim
        if D % 256 or D % 128 or I % 128:
            return "model_dim % 256 and inter_dim % 128"
        # Down holds the tile's whole intermediate rows in registers (BLOCK_M x I / 256
        # VGPRs at BLOCK_M >= 32); larger inter_dim needs a K-chunked Down.
        if I > 512:
            return "inter_dim > 512 not implemented yet"
        if topk > 255:
            return "topk"
        if call is not None:
            if call.hidden_states.dtype != torch.bfloat16:
                return "bf16 hidden_states only"
            if call.expert_mask is not None or call.num_local_tokens is not None:
                return "EP and num_local_tokens not implemented yet"
            if call.a1_scale is not None or call.a2_scale is not None:
                return "prequantized activations"
            if call.extra:
                return f"unsupported {sorted(call.extra)}"
            if not call.hidden_states.is_contiguous():
                return "contiguous hidden_states only"
        return None

    def _gpu(self, weights):
        return getattr(weights, "gpu", None) or "mi300x"

    def __call__(self, call):
        from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting

        w = call.weights
        spec = call.spec
        x = call.hidden_states
        M, D, I, E, topk = (
            call.m,
            spec.model_dim,
            spec.inter_dim,
            spec.experts,
            call.topk,
        )
        block_m, block_n, n_split, down_m = _tile(call.config, self._gpu(w), I)
        act = _ACTS[spec.act]
        topk_ids = call.topk_ids.to(torch.int32)
        topk_weight = call.topk_weight.to(torch.float32)
        sorted_ids, sorted_w, sorted_eids, num_valid, _, _, loc_ids = moe_sorting(
            topk_ids,
            topk_weight,
            E,
            D,
            x.dtype,
            block_m,
            None,
            None,
            0,
            accumulate=False,
            output_aux=AUX_SORT_OPUS,
        )
        quant = aiter.get_hip_quant(QuantType.per_1x128)
        a_bf16 = bool(
            _configs(self._gpu(w))
            .get(call.config or "default", {})
            .get("inline_a_quant", False)
        )
        if a_bf16:
            xq, xs = x, torch.empty(1, dtype=torch.float32, device=x.device)
        else:
            xq, xs = quant(x, quant_dtype=dtypes.fp8)
        tasks = int(sorted_eids.shape[0])
        fused_quant = block_n == 256
        if fused_quant:
            aq = torch.empty((M * topk, I), dtype=dtypes.fp8, device=x.device)
            as_ = torch.empty(
                (M * topk, I // 128), dtype=torch.float32, device=x.device
            )
            out1, out1_scale = aq, as_
        else:
            out1 = torch.empty((M, topk, I), dtype=torch.bfloat16, device=x.device)
            out1_scale = xs
        _launch(
            _gateup(2 * I, D, topk, block_m, block_n, act, fused_quant, a_bf16),
            xq,
            w.w1,
            out1,
            sorted_ids,
            sorted_eids,
            num_valid,
            w.w1_scale,
            xs,
            out1_scale,
            M,
            tasks,
        )
        if not fused_quant:
            aq, as_ = quant(out1.view(M * topk, I), quant_dtype=dtypes.fp8)
        rows = torch.empty((tasks * block_m, D), dtype=torch.bfloat16, device=x.device)
        _launch(
            _down(D, I, topk, block_m, n_split, down_m),
            aq,
            w.w2,
            rows,
            sorted_ids,
            sorted_w,
            sorted_eids,
            num_valid,
            w.w2_scale,
            as_,
            M,
            tasks,
        )
        out = call.output
        if out is None:
            out = torch.empty((M, D), dtype=torch.bfloat16, device=x.device)
        _sorted_sum(topk, D)(loc_ids, rows, out, M)
        return out
