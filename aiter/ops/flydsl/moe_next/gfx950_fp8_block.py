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


_COUNTERS = {}
_UNUSED = {}
# Split-K target: about this many stage-1 workgroups in flight at decode, with at
# most _SPLITK_MAX splits (the range measured).
_SPLITK_WGS = 256
_SPLITK_MAX = 8


def _splits(cfg, spec, m, topk):
    """The config's split count; "auto" fills about _SPLITK_WGS workgroups from the
    live (tile, block) bound min(routes, experts) x I/128, with an even number of K
    tiles per split."""
    s = cfg.get("splits", 1)
    if s != "auto":
        return s
    kb = spec.model_dim // 128
    wgs = min(m * topk, spec.experts) * (spec.inter_dim // 128)
    best = 1
    for c in range(2, min(_SPLITK_MAX, _SPLITK_WGS // max(wgs, 1)) + 1):
        if kb % c == 0 and (kb // c) % 2 == 0:
            best = c
    return best


def _splitk_counters(device, n):
    """Split-K arrival counters for the current stream: zeroed once, and the last split
    of each (tile, block) puts its counter back to zero, so later calls (and graph
    replays) find them zero without a memset."""
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    cnt = _COUNTERS.get(key)
    if cnt is None or cnt.numel() < n:
        cnt = torch.zeros(max(n, 1024), dtype=torch.int32, device=device)
        _COUNTERS[key] = cnt
    return cnt


def _unused_i32(device):
    """Placeholder for the combine's absent expert_mask / num_local_tokens: an int32 vector,
    like the real ones, so every call traces to the cache key the host build produced."""
    t = _UNUSED.get(device)
    if t is None:
        t = _UNUSED[device] = torch.zeros(1, dtype=torch.int32, device=device)
    return t


def _atom_cache(cfg):
    """Stage 2's atomic cache-policy bits from the config's "atom_cache" names (sc0, nt, sc1)."""
    from aiter.ops.flydsl.kernels.moe_next.core import atomics as A

    bits = {"sc0": A.SC0, "nt": A.NT, "sc1": A.SC1}
    return sum(bits[b] for b in cfg.get("atom_cache", ()))


def _decode_stage1(cfg, spec):
    """Whether stage1_decode.py runs: it quantizes bf16 x itself (aiter's 1x128 HIP rule) in
    its K loop, so the call launches no separate activation quant."""
    # The decode stage 1 wins at inter_dim <= 256 (DSV3 / Qwen3.5 M=8: 84.4 vs 93.5 us
    # warm, 88.9 vs 113.9 cold) and loses to stage1.py at 1024-2048 (synthetic M=128
    # 103.8 vs 85.9, GLM-5.1 M=64 344 vs 293).
    return (
        cfg["tile_m"] == 16
        and cfg.get("stage1") not in ("general", "8wave")
        and spec.inter_dim <= 256
    )


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
    _combine = None  # set: overrides the config's "combine"

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
        spec = weights.spec
        cfg = _tile_config(weights.gpu, config, m, topk, spec.experts)
        self._launchers(topk, spec, cfg, _splits(cfg, spec, m, topk))

    def _launchers(self, topk, spec, cfg, splits):
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
            if _decode_stage1(cfg, spec):
                from aiter.ops.flydsl.kernels.moe_next.gfx950_fp8_block.stage1_decode import (
                    compile_stage1,
                )
            s1 = compile_stage1(
                model_dim=spec.model_dim,
                inter_dim=spec.inter_dim,
                experts=spec.experts,
                tile_m=cfg["tile_m"],
                act=_ACTS[spec.act],
                doweight_stage1=spec.doweight_stage1,
                splits=splits,
            )
        s2 = compile_stage2(
            model_dim=spec.model_dim,
            inter_dim=spec.inter_dim,
            experts=spec.experts,
            topk=topk,
            tile_m=cfg["tile_m"],
            tile_n=cfg["tile_n2"],
            doweight_stage1=spec.doweight_stage1,
            combine=self._combine or cfg.get("combine", "atomic_bf16"),
            rows_nt=bool(cfg.get("rows_nt", False)),
            atom_cache=_atom_cache(cfg),
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
        splits = _splits(cfg, spec, call.m, call.topk)
        s1, s2 = self._launchers(call.topk, spec, cfg, splits)
        combine = self._combine or cfg.get("combine", "atomic_bf16")
        x = call.hidden_states
        M, D, I = call.m, spec.model_dim, spec.inter_dim
        if _decode_stage1(cfg, spec):
            # bf16 x as bytes; stage 1 reads no activation scales (the slot takes any f32 tensor)
            xq, xs = x.contiguous().view(torch.int8), w1s
        else:
            xq, xs = per_group_quant_hip(x, quant_dtype=dtypes.fp8, group_size=128)
        # atomic_bf16: the sort zeroes the output (or output=); atomic_f32: it zeroes
        # an f32 buffer; rows_* write every route row, so nothing is zeroed.
        if combine == "atomic_bf16":
            buf = call.output
            if buf is None:
                buf = torch.empty((M, D), dtype=torch.bfloat16, device=x.device)
        elif combine == "atomic_f32":
            buf = torch.empty((M, D), dtype=torch.float32, device=x.device)
        elif combine == "rows_bf16":
            # rows the sort drops are never written: the combine skips them by topk_ids
            buf = torch.empty((M * call.topk, D), dtype=torch.bfloat16, device=x.device)
        else:
            # zeroed: masked, -1 and past-num_local_tokens routes write no row
            buf = torch.zeros((M * call.topk, D), dtype=torch.float32, device=x.device)
        zero = None if combine.startswith("rows") else buf
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
        rows = sorted_ids.numel()
        # The sort gives each expert with routes ceil(routes / tile_m) tiles, so at most
        # routes // tile_m + (experts with routes) are live: a grid bound from shapes
        # alone (graph-safe) that drops most of the dead tiles at decode.
        routes = M * call.topk
        tiles = min(sorted_e.numel(), routes // tm + min(spec.experts, routes) + 1)
        inter = Layout.for_spec(spec, M, call.topk, tm).alloc(x.device)
        inter, inter_s = inter.values, inter.scales
        stream = fx.Stream(torch.cuda.current_stream(x.device))
        splitk = ()
        if splits > 1:
            from aiter.ops.flydsl.kernels.moe_next.gfx950_fp8_block.stage1 import (
                splitk_workspace_floats,
            )

            ws = torch.empty(
                splitk_workspace_floats(tm, tiles, I, splits),
                dtype=torch.float32,
                device=x.device,
            )
            splitk = (ws, _splitk_counters(x.device, tiles * (I // 128)))
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
            *splitk,
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
        if combine == "rows_bf16":
            from aiter.ops.flydsl.kernels.moe_next.gfx950_fp8_block.combine import (
                compile_rows_reduce,
            )

            mask, local = call.expert_mask, call.num_local_tokens
            if mask is not None and mask.dtype != torch.int32:
                mask = mask.to(torch.int32)
            reduce = compile_rows_reduce(
                model_dim=D,
                topk=call.topk,
                has_mask=mask is not None,
                has_local=local is not None,
            )
            ids = call.topk_ids.to(torch.int32).contiguous()
            _run_compiled(
                reduce,
                out,
                buf,
                ids,
                mask if mask is not None else _unused_i32(x.device),
                local if local is not None else _unused_i32(x.device),
                M,
                stream,
            )
            return out
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


@register_kernel
class Fp8BlockGfx950PrefillRows(_Fp8BlockGfx950):
    """Stage 2 stores bf16 per-route rows; combine.py sums each token's topk rows in fp32
    and rounds once. Deterministic, unlike the atomic combine."""

    id = "gfx950_fp8_block.prefill_rows"
    epilogue = "reduce"
    numerics = {
        "act_quant": "per_1x128_fp8",
        "inter": "bf16_then_per_1x128_fp8",
        "combine": "bf16_rows_fp32_sum",
    }
    _combine = "rows_bf16"
