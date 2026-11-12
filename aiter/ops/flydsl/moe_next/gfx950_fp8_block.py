# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 fp8 blockscale (per_1x128) MoE kernels for aiter.moe.

One call: 1x128 activation quant, sort, stage 1 (gate/up GEMM, activation and the
intermediate's 1x128 quant), stage 2 (down GEMM, route weight, atomic combine).
Weights stay in aiter's layout (shuffle_weight(16, 16) fp8, fp32 [E, N/128, K/128]
scales). Tile configs are data: configs/mi355x/fp8-block.json.
"""

import functools
import json
import os

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.moe import (  # noqa: F401  entry re-exported: importing this module registers its kernels
    CALL_KWARGS,
    PREPARE_KWARGS,
    MoeCall,
    MoeKernel,
    explain,
    forward,
    prepare,
    register_kernel,
)
from aiter.ops.flydsl.moe_common import GateMode

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
_ACTS = {ActivationType.Silu: "silu"}


@functools.cache
def _configs(gpu: str) -> dict:
    name = "mi355x" if gpu in ("mi355x", "mi350x") else gpu
    with open(os.path.join(_CONFIG_DIR, name, "fp8-block.json")) as f:
        return json.load(f)


def _tile_config(gpu, config, m, topk, experts):
    table = _configs(gpu)
    if config:
        return table["configs"][config]
    per_expert = max(1, -(-m * topk // experts))
    for row in table["default"]:
        if per_expert <= row["max_tokens_per_expert"]:
            return table["configs"][row["config"]]
    raise KeyError(per_expert)


class _Fp8BlockGfx950(MoeKernel):
    family = "fp8-block"
    archs = ("gfx950",)
    routing = "tiles"
    epilogue = "atomic"
    handles_kwargs = frozenset()

    def supports(self, spec, m, topk, call=None):
        if spec.quant != QuantType.per_1x128:
            return "quant is not per_1x128"
        if spec.w_dtype != dtypes.fp8:
            return f"weights are {spec.w_dtype}, not fp8"
        if not spec.g1u1:
            return "g1u0 not implemented"
        if spec.act not in _ACTS:
            return f"activation {spec.act} not implemented"
        if not spec.w_shuffled:
            return "weights not shuffle_weight(16, 16)"
        if spec.gate_mode != GateMode.SEPARATED:
            return "gate_mode is not SEPARATED"
        if spec.has_bias or spec.swiglu_limit is not None:
            return "bias / swiglu_limit not implemented"
        if spec.hidden_pad or spec.intermediate_pad:
            return "padding not implemented"
        if spec.model_dim % 256 or spec.inter_dim % 128:
            return "model_dim % 256 or inter_dim % 128"
        if spec.out_dtype not in (None, torch.bfloat16):
            return "output dtype is not bf16"
        if spec.global_experts != spec.experts:
            return "EP not implemented"
        if call is None:
            return None
        if call.hidden_states.dtype != torch.bfloat16:
            return "hidden_states is not bf16"
        if call.expert_mask is not None or call.num_local_tokens is not None:
            return "expert_mask / num_local_tokens not implemented"
        if call.a1_scale is not None or call.a2_scale is not None:
            return "prequantized activations not implemented"
        if call.extra:
            return f"{sorted(call.extra)} not implemented"
        if call.output is not None and call.output.dtype != torch.bfloat16:
            return "output= is not bf16"
        return None

    def prepare_weights(self, weights):
        s = weights.spec
        w1s = weights.w1_scale.to(torch.float32).contiguous().view(-1)
        w2s = weights.w2_scale.to(torch.float32).contiguous().view(-1)
        assert w1s.numel() == s.experts * (2 * s.inter_dim // 128) * (
            s.model_dim // 128
        )
        assert w2s.numel() == s.experts * (s.model_dim // 128) * (s.inter_dim // 128)
        return {"w1_scale": w1s, "w2_scale": w2s}

    def warmup(self, weights, m, topk, config):
        self._launchers(
            weights.spec,
            _tile_config(weights.gpu, config, m, topk, weights.spec.experts),
        )

    def _launchers(self, spec, cfg):
        from aiter.ops.flydsl.kernels.moe_next.gfx950_fp8_block.stage1 import (
            compile_stage1,
        )
        from aiter.ops.flydsl.kernels.moe_next.gfx950_fp8_block.stage2 import (
            compile_stage2,
        )

        s1 = compile_stage1(
            model_dim=spec.model_dim,
            inter_dim=spec.inter_dim,
            experts=spec.experts,
            tile_m=cfg["tile_m"],
            act=_ACTS[spec.act],
            doweight_stage1=spec.doweight_stage1,
        )
        s2 = compile_stage2(
            model_dim=spec.model_dim,
            inter_dim=spec.inter_dim,
            experts=spec.experts,
            tile_m=cfg["tile_m"],
            tile_n=cfg["tile_n2"],
            doweight_stage1=spec.doweight_stage1,
        )
        return s1, s2

    def __call__(self, call: MoeCall) -> torch.Tensor:
        import flydsl.expr as fx

        from aiter.fused_moe import moe_sorting
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
        from aiter.ops.quant import per_group_quant_hip

        spec, weights = call.spec, call.weights
        state = weights.state[self.id]
        cfg = _tile_config(weights.gpu, call.config, call.m, call.topk, spec.experts)
        tm = cfg["tile_m"]
        s1, s2 = self._launchers(spec, cfg)
        x = call.hidden_states
        M, D, I = call.m, spec.model_dim, spec.inter_dim
        xq, xs = per_group_quant_hip(x, quant_dtype=dtypes.fp8, group_size=128)
        sorted_ids, sorted_w, sorted_e, nvalid, out = moe_sorting(
            call.topk_ids,
            call.topk_weight,
            spec.experts,
            D,
            torch.bfloat16,
            block_size=tm,
            output=call.output,
        )
        rows = sorted_ids.numel()
        tiles = sorted_e.numel()
        inter = torch.empty((rows, I), dtype=dtypes.fp8, device=x.device)
        inter_s = torch.empty((rows, I // 128), dtype=torch.float32, device=x.device)
        stream = fx.Stream(torch.cuda.current_stream(x.device))
        _run_compiled(
            s1,
            inter.view(torch.int8),
            inter_s,
            xq.view(torch.int8),
            xs,
            weights.w1.view(torch.int8),
            state["w1_scale"],
            sorted_ids,
            sorted_e,
            sorted_w,
            nvalid,
            M,
            rows,
            tiles,
            stream,
        )
        _run_compiled(
            s2,
            out,
            inter.view(torch.int8),
            inter_s,
            weights.w2.view(torch.int8),
            state["w2_scale"],
            sorted_ids,
            sorted_e,
            sorted_w,
            nvalid,
            M,
            rows,
            tiles,
            stream,
        )
        return out


@register_kernel
class Fp8BlockGfx950Decode(_Fp8BlockGfx950):
    id = "gfx950_fp8_block.decode"


@register_kernel
class Fp8BlockGfx950Prefill(_Fp8BlockGfx950):
    id = "gfx950_fp8_block.prefill"
