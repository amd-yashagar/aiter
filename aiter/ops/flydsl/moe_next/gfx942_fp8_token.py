# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 per-token (PTPC) and per-tensor MoE kernels for aiter.moe.

Prefill, same quantization points as aiter's 2-stage path: the core sort (with
the output zero-fill fused in), aiter's per-token / per-tensor activation quant,
Gate/Up (fp8 MFMA, scales and activation in the epilogue, bf16 intermediate),
aiter's quant of the intermediate, and Down with the scales and route weight in
the epilogue, added into the sort-zeroed output with coalesced packed bf16
atomics (config ``rows``: one bf16 row per route, summed per token in fp32).
Tile configs are data: configs/<gpu>/fp8-token.json.
"""

import functools
import json
import os
from typing import ClassVar

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.moe import MoeKernel, compile_only, register_kernel
from aiter.ops.flydsl.moe_common import GateMode

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
_ACTS = {ActivationType.Silu: "silu", ActivationType.Swiglu: "swiglu"}
_QUANTS = {QuantType.per_Token: "ptpc", QuantType.per_Tensor: "per_tensor"}
TILE_M = 64
_DOWN_LDS_A = 48 * 1024


@functools.cache
def _configs(gpu):
    path = os.path.join(_CONFIG_DIR, gpu, "fp8-token.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _tile(config, gpu, model_dim):
    cfg = dict(_configs(gpu).get("default", {}))
    cfg.update(_configs(gpu).get(config or "default", {}))
    block_n = int(cfg.get("gateup_block_n", 128))
    block_k = int(cfg.get("gateup_block_k", 256))
    while block_k > 64 and model_dim % (2 * block_k):
        block_k //= 2
    return block_n, block_k, cfg.get("combine", "atomic") == "atomic"


@functools.cache
def _gateup(N, K, topk, block_n, block_k, quant, act, swiglu_limit):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import compile_gemm

    return compile_gemm(
        N=N,
        K=K,
        weight_dtype="fp8",
        weight_quant_type=quant,
        TOPK=topk,
        BLOCK_TILE_SIZE_M=TILE_M,
        BLOCK_TILE_SIZE_N=block_n,
        tile_k=block_k,
        stage="gateup",
        alg="prefill_1x4",
        E=None,
        act_quant_type=quant,
        activation=act,
        swiglu_limit=swiglu_limit,
    )


@functools.cache
def _down(N, K, topk, per_tensor, atomic):
    from aiter.ops.flydsl.kernels.moe_next.gfx942_fp8_token.down import compile_down

    return compile_down(
        N=N, K=K, TOPK=topk, BLOCK_M=TILE_M, PER_TENSOR=per_tensor, ATOMIC=atomic
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


def _swiglu_limit(spec):
    if spec.act != ActivationType.Swiglu:
        return None
    return float(spec.swiglu_limit) if spec.swiglu_limit else 7.0


@register_kernel
class Gfx942Fp8TokenPrefill(MoeKernel):
    id = "gfx942_fp8_token.prefill"
    family = "fp8-token"
    archs = ("gfx942",)
    routing = "tiles"
    epilogue = "atomic"
    numerics: ClassVar[dict] = {
        "act_quant": "per_token_fp8",
        "inter": "bf16_then_per_token_fp8",
        "combine": "bf16_atomic",
    }

    def supports(self, spec, m, topk, call=None):
        if spec.quant not in _QUANTS:
            return "needs per_Token or per_Tensor"
        if spec.w_dtype != dtypes.fp8 or not spec.g1u1 or not spec.w_shuffled:
            return "needs shuffled fp8 g1u1 weights"
        if spec.act not in _ACTS:
            return f"activation {spec.act}"
        if spec.act == ActivationType.Silu and spec.swiglu_limit:
            return "clamped SiLU"
        if spec.doweight_stage1:
            return "doweight_stage1"
        if spec.gate_mode != GateMode.SEPARATED:
            return "gate/up must be separated"
        # aiter's per-token / per-tensor path drops bias and ignores padding.
        if spec.has_bias or spec.hidden_pad or spec.intermediate_pad:
            return "bias or padding"
        if spec.out_dtype not in (None, torch.bfloat16):
            return "bf16 output only"
        D, I = spec.model_dim, spec.inter_dim
        if D % 256 or I % 64:
            return "model_dim % 256 and inter_dim % 64"
        if TILE_M * I > _DOWN_LDS_A:
            return "inter_dim > 768 not implemented yet"
        if topk > 255:
            return "topk"
        if call is not None:
            if call.hidden_states.dtype != torch.bfloat16:
                return "bf16 hidden_states only"
            if call.out_dtype != torch.bfloat16:
                return "bf16 output only"
            if call.w1_scale is None or call.w2_scale is None:
                return "needs weight scales"
            if not (call.w1_scale.is_contiguous() and call.w2_scale.is_contiguous()):
                return "contiguous fp32 weight scales only"
            if call.w1_scale.dtype != torch.float32:
                return "fp32 weight scales only"
            if spec.quant == QuantType.per_Tensor and call.num_local_tokens is not None:
                # aiter's per-tensor quant asserts on num_rows.
                return "per-tensor with num_local_tokens"
            if spec.quant == QuantType.per_Token and call.a1_scale is not None:
                return "prequantized activations"
            if call.a2_scale is not None:
                return "static a2_scale"
            if call.output is not None and not call.output.is_contiguous():
                return "contiguous output only"
            if call.extra:
                return f"unsupported {sorted(call.extra)}"
            if not call.hidden_states.is_contiguous():
                return "contiguous hidden_states only"
        return None

    def _gpu(self, weights):
        return getattr(weights, "gpu", None) or "mi325x"

    def warmup(self, weights, m, topk, config):
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        spec = weights.spec
        D, I = spec.model_dim, spec.inter_dim
        block_n, block_k, atomic = _tile(config, self._gpu(weights), D)
        quant = _QUANTS[spec.quant]
        with compile_only():
            gu = _gateup(
                2 * I,
                D,
                topk,
                block_n,
                block_k,
                quant,
                _ACTS[spec.act],
                _swiglu_limit(spec),
            )
            types = (fx.Uint8, fx.Uint8, fx.BFloat16) + (fx.Int32,) * 2
            types += (fx.Int32,) * 2 + (fx.Float32,) * 2
            gu(*[flyc.from_c_void_p(t, 0) for t in types], m, 1, fx.Stream(None))
            dn = _down(D, I, topk, quant == "per_tensor", atomic)
            types = (fx.Uint8, fx.Uint8, fx.BFloat16, fx.Int32, fx.Float32)
            types += (fx.Int32, fx.Int32, fx.Float32, fx.Float32)
            dn(*[flyc.from_c_void_p(t, 0) for t in types], m, 1, fx.Stream(None))

    def __call__(self, call):
        from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting

        spec = call.spec
        x = call.hidden_states
        M, D, I, topk = call.m, spec.model_dim, spec.inter_dim, call.topk
        per_tensor = spec.quant == QuantType.per_Tensor
        quant = _QUANTS[spec.quant]
        block_n, block_k, atomic = _tile(call.config, self._gpu(call.weights), D)
        dev = x.device
        nlt = call.num_local_tokens

        out = call.output
        if out is None:
            out = torch.empty((M, D), dtype=torch.bfloat16, device=dev)
        sort = moe_sorting(
            call.topk_ids,
            call.topk_weight,
            call.global_experts,
            D,
            torch.bfloat16,
            TILE_M,
            call.expert_mask,
            nlt,
            0,
            accumulate=atomic,
            output_aux=None if atomic else AUX_SORT_OPUS,
            output=out if atomic else None,
        )
        sorted_ids, sorted_w, sorted_eids, num_valid = sort[:4]
        tasks = int(sorted_eids.shape[0])

        quant_fn = aiter.get_hip_quant(spec.quant)
        if per_tensor:
            xq, xs = quant_fn(x, scale=call.a1_scale, quant_dtype=dtypes.fp8)
        else:
            xq, xs = quant_fn(x, quant_dtype=dtypes.fp8, num_rows=nlt)
        xs = xs.to(torch.float32).contiguous()

        # The dynamic per-tensor quant of the intermediate reads every row,
        # including the route slots Gate/Up skips.
        alloc = torch.zeros if per_tensor else torch.empty
        inter = alloc((M, topk, I), dtype=torch.bfloat16, device=dev)
        _launch(
            _gateup(
                2 * I,
                D,
                topk,
                block_n,
                block_k,
                quant,
                _ACTS[spec.act],
                _swiglu_limit(spec),
            ),
            xq,
            call.w1,
            inter,
            sorted_ids,
            sorted_w,
            sorted_eids,
            num_valid,
            call.w1_scale,
            xs,
            M,
            tasks,
        )
        if per_tensor:
            aq, as_ = quant_fn(inter.view(M * topk, I), quant_dtype=dtypes.fp8)
        else:
            aq, as_ = quant_fn(
                inter.view(M * topk, I),
                quant_dtype=dtypes.fp8,
                num_rows=nlt,
                num_rows_factor=topk,
            )
        rows = out
        if not atomic:
            rows = torch.empty((tasks * TILE_M, D), dtype=torch.bfloat16, device=dev)
        _launch(
            _down(D, I, topk, per_tensor, atomic),
            aq,
            call.w2,
            rows,
            sorted_ids,
            sorted_w,
            sorted_eids,
            num_valid,
            call.w2_scale,
            as_.to(torch.float32).contiguous(),
            M,
            tasks,
        )
        if not atomic:
            _sorted_sum(topk, D)(sort[6], rows, out, M)
        return out
