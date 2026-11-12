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
from aiter.moe import MoeCall, MoeKernel, register_kernel
from aiter.ops.flydsl.moe_common import GateMode

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
_ACTS = {ActivationType.Silu: "silu"}
# Core's route_to_tiles is 1.2-1.6x slower on the device than Opus at mid and large M.
_CORE_SORT = os.getenv("AITER_FP8B_CORE_SORT", "0") == "1"


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
    # The *_af32 / *_rows configs combine in fp32, rounding less than bf16 atomics.
    numerics = {
        "act_quant": "per_1x128_fp8",
        "inter": "bf16_then_per_1x128_fp8",
        "combine": "bf16_atomic",
    }
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
        if spec.doweight_stage1:
            return "doweight_stage1 not verified against aiter's route-weight placement"
        if call is None:
            return None
        if call.hidden_states.dtype != torch.bfloat16:
            return "hidden_states is not bf16"
        if call.a1_scale is not None or call.a2_scale is not None:
            return "prequantized activations not implemented"
        if call.extra:
            return f"{sorted(call.extra)} not implemented"
        if call.out_dtype != torch.bfloat16:
            return "output dtype is not bf16"
        if (call.expert_mask is not None) != (call.global_experts != spec.experts):
            return "expert_mask does not match the local expert count"
        nb1 = spec.experts * (2 * spec.inter_dim // 128) * (spec.model_dim // 128)
        nb2 = spec.experts * (spec.model_dim // 128) * (spec.inter_dim // 128)
        for s, n in ((call.w1_scale, nb1), (call.w2_scale, nb2)):
            if (
                s is None
                or s.dtype != torch.float32
                or not s.is_contiguous()
                or s.numel() != n
            ):
                return "weight scales are not fp32 contiguous [E, N/128, K/128]"
        if not (call.w1.is_contiguous() and call.w2.is_contiguous()):
            return "weights are not contiguous"
        if call.output is not None and call.output.dtype != torch.bfloat16:
            return "output= is not bf16"
        return None

    def warmup(self, weights, m, topk, config):
        self._launchers(
            topk,
            weights.spec,
            _tile_config(weights.gpu, config, m, topk, weights.spec.experts),
        )

    def _launchers(self, topk, spec, cfg):
        from aiter.ops.flydsl.kernels.moe_next.gfx950_fp8_block.stage1 import (
            compile_stage1,
        )
        from aiter.ops.flydsl.kernels.moe_next.gfx950_fp8_block.stage1_8wave import (
            compile_stage1_8wave,
        )
        from aiter.ops.flydsl.kernels.moe_next.gfx950_fp8_block.stage2 import (
            compile_stage2,
        )

        if cfg.get("stage1") == "8wave":
            s1 = compile_stage1_8wave(
                model_dim=spec.model_dim,
                inter_dim=spec.inter_dim,
                experts=spec.experts,
                tile_m=cfg["tile_m"],
                act=_ACTS[spec.act],
            )
        else:
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
            topk=topk,
            tile_m=cfg["tile_m"],
            tile_n=cfg["tile_n2"],
            doweight_stage1=spec.doweight_stage1,
            combine=cfg.get("combine", "atomic_bf16"),
        )
        return s1, s2

    def __call__(self, call: MoeCall) -> torch.Tensor:
        import flydsl.expr as fx

        from aiter.fused_moe import moe_sorting
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
        from aiter.ops.flydsl.moe_next.core.intermediate import Layout
        from aiter.ops.flydsl.moe_next.core.tile_map import route_to_tiles
        from aiter.ops.quant import per_group_quant_hip

        spec, weights = call.spec, call.weights
        # The caller's own tensors, in aiter's layout (fp32 scales viewed, never copied).
        w1s = call.w1_scale.view(-1)
        w2s = call.w2_scale.view(-1)
        cfg = _tile_config(weights.gpu, call.config, call.m, call.topk, spec.experts)
        tm = cfg["tile_m"]
        s1, s2 = self._launchers(call.topk, spec, cfg)
        combine = cfg.get("combine", "atomic_bf16")
        x = call.hidden_states
        M, D, I = call.m, spec.model_dim, spec.inter_dim
        xq, xs = per_group_quant_hip(x, quant_dtype=dtypes.fp8, group_size=128)
        # atomic_bf16: the sort zeroes the output (or output=); atomic_f32: it zeroes
        # an f32 buffer; rows_f32 writes every route row, so nothing is zeroed.
        if combine == "atomic_bf16":
            buf = call.output
            if buf is None:
                buf = torch.empty((M, D), dtype=torch.bfloat16, device=x.device)
        elif combine == "atomic_f32":
            buf = torch.empty((M, D), dtype=torch.float32, device=x.device)
        else:
            # zeroed: masked, -1 and past-num_local_tokens routes write no row
            buf = torch.zeros((M * call.topk, D), dtype=torch.float32, device=x.device)
        zero = None if combine == "rows_f32" else buf
        # As fused_moe sorts: global expert count, EP mask (local expert ids out, masked
        # and -1 routes dropped, their rows zeroed), num_local_tokens.
        if _CORE_SORT:
            tmap = route_to_tiles(
                call.topk_ids.to(torch.int32),
                call.topk_weight.to(torch.float32),
                num_experts=call.global_experts,
                tile_m=tm,
                expert_mask=call.expert_mask,
                num_local_tokens=call.num_local_tokens,
                zero_output=zero,
            )
            sorted_ids, sorted_w = tmap.sorted_token_ids, tmap.sorted_weights
            sorted_e, nvalid = tmap.sorted_expert_ids, tmap.num_valid_ids
        else:
            sorted_ids, sorted_w, sorted_e, nvalid, _ = moe_sorting(
                call.topk_ids,
                call.topk_weight,
                call.global_experts,
                D,
                buf.dtype if zero is not None else torch.bfloat16,
                block_size=tm,
                expert_mask=call.expert_mask,
                num_local_tokens=call.num_local_tokens,
                output=zero,
                accumulate=zero is not None,
            )
        rows, tiles = sorted_ids.numel(), sorted_e.numel()
        inter = Layout.for_spec(spec, M, call.topk, tm).alloc(x.device)
        inter, inter_s = inter.values, inter.scales
        stream = fx.Stream(torch.cuda.current_stream(x.device))
        _run_compiled(
            s1,
            inter.view(torch.int8),
            inter_s,
            xq.view(torch.int8),
            xs,
            call.w1.view(torch.int8),
            w1s,
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
            buf,
            inter.view(torch.int8),
            inter_s,
            call.w2.view(torch.int8),
            w2s,
            sorted_ids,
            sorted_e,
            sorted_w,
            nvalid,
            M,
            rows,
            tiles,
            stream,
        )
        if combine == "atomic_bf16":
            return buf
        out = call.output
        if out is None:
            out = torch.empty((M, D), dtype=torch.bfloat16, device=x.device)
        if combine == "rows_f32":
            buf = buf.view(M, call.topk, D).sum(dim=1)
        out.copy_(buf)
        return out


@register_kernel
class Fp8BlockGfx950Decode(_Fp8BlockGfx950):
    id = "gfx950_fp8_block.decode"


@register_kernel
class Fp8BlockGfx950Prefill(_Fp8BlockGfx950):
    id = "gfx950_fp8_block.prefill"
