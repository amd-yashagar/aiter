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
from typing import ClassVar

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.moe import MoeKernel, compile_only, register_kernel
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
    block_n = int(cfg.get("block_n", 256))
    # A bf16 intermediate would round more than the declared numerics allow.
    if block_n != 256:
        raise ValueError(f"fp8-block config {config!r}: block_n must be 256")
    return (
        block_m,
        block_n,
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
def _down(N, K, topk, block_m, n_split, down_m, rows_nt=False):
    from aiter.ops.flydsl.kernels.moe_next.gfx942_fp8_block.down import compile_down

    while n_split > 1 and (N // 64) % (2 * n_split):
        n_split //= 2
    return compile_down(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_M=down_m,
        N_SPLIT=n_split,
        SORT_BLOCK_M=block_m,
        ROWS_NT=rows_nt,
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
    # Each route's down output is rounded to bf16, then the topk rows are summed
    # in fp32 (sorted_sum); aiter's ASM 1-stage path bf16-atomics them instead.
    numerics: ClassVar[dict] = {
        "act_quant": "per_1x128_fp8",
        "inter": "fp32_direct",
        "combine": "bf16_rows_fp32_sum",
    }

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
            if call.out_dtype != torch.bfloat16:
                return "bf16 output only"
            if call.w1_scale is None or call.w2_scale is None:
                return "needs weight scales"
            if not (call.w1_scale.is_contiguous() and call.w2_scale.is_contiguous()):
                return "contiguous fp32 weight scales only"
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

    def warmup(self, weights, m, topk, config):
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        spec = weights.spec
        D, I = spec.model_dim, spec.inter_dim
        block_m, block_n, n_split, down_m = _tile(config, self._gpu(weights), I)
        a_bf16 = self._inline_a_quant(weights, config)
        with compile_only():
            gu = _gateup.__wrapped__(
                2 * I, D, topk, block_m, block_n, _ACTS[spec.act], True, a_bf16
            )
            out_t = fx.Uint8
            in_t = fx.BFloat16 if a_bf16 else fx.Uint8
            types = (
                in_t,
                fx.Uint8,
                out_t,
                fx.Int32,
                fx.Int32,
                fx.Int32,
                fx.Float32,
                fx.Float32,
                fx.Float32,
            )
            gu(*[flyc.from_c_void_p(t, 0) for t in types], m, 1, fx.Stream(None))
            rows_nt = self._cfg_flag(weights, config, "rows_nt")
            dn = _down.__wrapped__(D, I, topk, block_m, n_split, down_m, rows_nt)
            types = (
                fx.Uint8,
                fx.Uint8,
                fx.BFloat16,
                fx.Int32,
                fx.Float32,
                fx.Int32,
                fx.Int32,
                fx.Float32,
                fx.Float32,
            )
            dn(*[flyc.from_c_void_p(t, 0) for t in types], m, 1, fx.Stream(None))

    def _inline_a_quant(self, weights, config):
        return self._cfg_flag(weights, config, "inline_a_quant")

    def _cfg_flag(self, weights, config, key):
        return bool(self._cfg_value(weights, config, key) or False)

    def _cfg_value(self, weights, config, key):
        cfg = _configs(self._gpu(weights)).get(config or "default", {})
        return cfg.get(key)

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
        a_bf16 = self._inline_a_quant(w, call.config)
        if self._cfg_value(w, call.config, "sort") == "core":
            from aiter.ops.flydsl.moe_next.core.tile_map import route_to_tiles

            # One launch sorts the routes and applies the per_1x128 activation quant.
            tm = route_to_tiles(
                topk_ids,
                topk_weight,
                num_experts=E,
                tile_m=block_m,
                with_loc_ids=True,
                quant_input=None if a_bf16 else x,
                quant_mode=None if a_bf16 else "per_1x128_fp8",
            )
            sorted_ids, sorted_w = tm.sorted_token_ids, tm.sorted_weights
            sorted_eids, num_valid, loc_ids = (
                tm.sorted_expert_ids,
                tm.num_valid_ids,
                tm.loc_ids,
            )
            if a_bf16:
                xq, xs = x, torch.empty(1, dtype=torch.float32, device=x.device)
            else:
                xq, xs = tm.a_q, tm.a_scale
        else:
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
            if a_bf16:
                xq, xs = x, torch.empty(1, dtype=torch.float32, device=x.device)
            else:
                xq, xs = quant(x, quant_dtype=dtypes.fp8)
        tasks = int(sorted_eids.shape[0])
        aq = torch.empty((M * topk, I), dtype=dtypes.fp8, device=x.device)
        as_ = torch.empty((M * topk, I // 128), dtype=torch.float32, device=x.device)
        _launch(
            _gateup(2 * I, D, topk, block_m, block_n, act, True, a_bf16),
            xq,
            call.w1,
            aq,
            sorted_ids,
            sorted_eids,
            num_valid,
            call.w1_scale,
            xs,
            as_,
            M,
            tasks,
        )
        rows = torch.empty((tasks * block_m, D), dtype=torch.bfloat16, device=x.device)
        _launch(
            _down(
                D,
                I,
                topk,
                block_m,
                n_split,
                down_m,
                self._cfg_flag(w, call.config, "rows_nt"),
            ),
            aq,
            call.w2,
            rows,
            sorted_ids,
            sorted_w,
            sorted_eids,
            num_valid,
            call.w2_scale,
            as_,
            M,
            tasks,
        )
        out = call.output
        if out is None:
            out = torch.empty((M, D), dtype=torch.bfloat16, device=x.device)
        _sorted_sum(topk, D)(loc_ids, rows, out, M)
        return out
