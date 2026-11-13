# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 per-token (PTPC) MoE for the mid-M band, for aiter.moe.

Between the family's sortless decode and its prefill: tens to about a thousand
tokens, where each active expert holds one to a few dozen rows and the call is
bound by streaming the active experts' weights. Core's sort gives one live tile
per expert (or a few under skew); Gate/Up and Down are driven by the live-tile
list and read each live expert's weights once. Same rounding points as aiter's
gfx942 whole-graph decode (bf16 activations against fp8 weights, a bf16
intermediate), with the combine either one bf16 row per route summed per token
in FP32 (``mid``) or packed bf16 atomics (``mid_atomic``).
Tile configs are data: configs/<gpu>/fp8-token-mid.json.
"""

import functools
import json
import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.moe import MoeKernel, compile_only, register_kernel
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.flydsl.moe_next.gfx942_fp8_token import (
    _ACTS,
    _launch,
    _swiglu_limit,
)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
_MAX_INTER = 512


@functools.cache
def _configs(gpu):
    path = os.path.join(_CONFIG_DIR, gpu, "fp8-token-mid.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@dataclass(frozen=True)
class _Tile:
    tile_m: int
    gu_waves: int
    ch: int
    ks: int
    pw: int
    pa: int
    w_cache: int
    nbd: int
    waves: int
    pwd: int
    coalesce: bool
    pad_rows: bool
    kc: bool


def _tile(config, gpu, model_dim, inter_dim):
    cfg = dict(_configs(gpu).get("default", {}))
    cfg.update(_configs(gpu).get(config or "default", {}))
    steps = model_dim // 64
    gw, ch = int(cfg.get("gateup_waves", 4)), int(cfg.get("gateup_ch", 2))
    while ch > 1 and steps % ch:
        ch //= 2
    while gw > 1 and inter_dim % (16 * gw):
        gw //= 2
    ks = int(cfg.get("gateup_ks", 1))
    while ks > 1 and (steps // ch) % ks:
        ks //= 2
    nbd, waves = int(cfg.get("down_nbd", 4)), int(cfg.get("down_waves", 4))
    while waves > 1 and model_dim % (16 * nbd * waves):
        waves //= 2
    while nbd > 1 and model_dim % (16 * nbd * waves):
        nbd //= 2
    pw, pa = int(cfg.get("gateup_pw", 1)), int(cfg.get("gateup_pa", 2))
    w_cache = int(cfg.get("w_cache", 2))
    pwd = int(cfg.get("down_pw", 2))
    coalesce = bool(cfg.get("down_coalesce", True))
    pad_rows = bool(cfg.get("sort_pad_rows", True))
    kc = bool(cfg.get("down_kc", False))
    return _Tile(
        int(cfg.get("tile_m", 64)),
        gw,
        ch,
        ks,
        pw,
        pa,
        w_cache,
        nbd,
        waves,
        pwd,
        coalesce,
        pad_rows,
        kc,
    )


@functools.cache
def _gateup(N, K, s_max, waves, ch, ks, pw, pa, w_cache, act, swiglu_limit, pt):
    from aiter.ops.flydsl.kernels.moe_next.gfx942_fp8_token_mid.gateup import (
        compile_gateup,
    )

    return compile_gateup(
        N=N,
        K=K,
        S_MAX=s_max,
        WAVES=waves,
        CH=ch,
        KS=ks,
        PW=pw,
        PA=pa,
        W_CACHE=w_cache,
        activation=act,
        swiglu_limit=swiglu_limit,
        PER_TENSOR=pt,
    )


@functools.cache
def _down(N, K, topk, s_max, nbd, waves, pwd, rows, w_cache, coalesce, pt, kc):
    from aiter.ops.flydsl.kernels.moe_next.gfx942_fp8_token_mid.down import (
        compile_down,
    )

    return compile_down(
        N=N,
        K=K,
        TOPK=topk,
        S_MAX=s_max,
        NBD=nbd,
        WAVES=waves,
        PWD=pwd,
        ROWS=rows,
        W_CACHE=w_cache,
        COALESCE=coalesce,
        PER_TENSOR=pt,
        KC=kc,
    )


@functools.cache
def _combine(N, topk, has_nlt):
    from aiter.ops.flydsl.kernels.moe_next.gfx942_fp8_token_mid.down import (
        compile_combine,
    )

    return compile_combine(N=N, TOPK=topk, HAS_NLT=has_nlt)


@register_kernel
class Gfx942Fp8TokenMid(MoeKernel):
    id = "gfx942_fp8_token.mid"
    family = "fp8-token"
    archs = ("gfx942",)
    routing = "tiles"
    epilogue = "reduce"
    numerics: ClassVar[dict] = {
        "act_quant": "none",
        "inter": "bf16",
        "combine": "bf16_rows_fp32_sum",
    }
    rows: ClassVar[bool] = True
    # The quant type this id serves; per-tensor takes one weight scale per expert.
    quant: ClassVar = QuantType.per_Token

    def supports(self, spec, m, topk, call=None):
        if spec.quant != self.quant:
            return f"needs {self.quant.name}"
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
        # aiter's per-token path drops bias and ignores padding.
        if spec.has_bias or spec.hidden_pad or spec.intermediate_pad:
            return "bias or padding"
        if spec.out_dtype not in (None, torch.bfloat16):
            return "bf16 output only"
        D, I = spec.model_dim, spec.inter_dim
        if D % 256 or I % 64:
            return "model_dim % 256 and inter_dim % 64"
        if I > _MAX_INTER:
            return f"inter_dim > {_MAX_INTER} not implemented yet"
        if topk > 255:
            return "topk"
        if call is not None:
            if call.hidden_states.dtype != torch.bfloat16:
                return "bf16 hidden_states only"
            if not call.hidden_states.is_contiguous():
                return "contiguous hidden_states only"
            if call.out_dtype != torch.bfloat16:
                return "bf16 output only"
            if call.w1_scale is None or call.w2_scale is None:
                return "needs weight scales"
            if not (call.w1_scale.is_contiguous() and call.w2_scale.is_contiguous()):
                return "contiguous fp32 weight scales only"
            if call.w1_scale.dtype != torch.float32:
                return "fp32 weight scales only"
            per_tensor = self.quant == QuantType.per_Tensor
            n1, n2 = (
                (spec.experts,) * 2
                if per_tensor
                else (spec.experts * 2 * I, spec.experts * D)
            )
            if call.w1_scale.numel() != n1 or call.w2_scale.numel() != n2:
                return (
                    "per-tensor weight scales"
                    if per_tensor
                    else "per-channel weight scales"
                )
            if call.a1_scale is not None or call.a2_scale is not None:
                return "prequantized activations"
            if call.output is not None and not call.output.is_contiguous():
                return "contiguous output only"
            if call.extra:
                return f"unsupported {sorted(call.extra)}"
        return None

    def _gpu(self, weights):
        return getattr(weights, "gpu", None) or "mi325x"

    def warmup(self, weights, m, topk, config):
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        spec = weights.spec
        D, I = spec.model_dim, spec.inter_dim
        t = _tile(config, self._gpu(weights), D, I)
        act, limit = _ACTS[spec.act], _swiglu_limit(spec)
        s_max = t.tile_m // 16
        with compile_only():
            gu = _gateup(
                2 * I,
                D,
                s_max,
                t.gu_waves,
                t.ch,
                t.ks,
                t.pw,
                t.pa,
                t.w_cache,
                act,
                limit,
                self.quant == QuantType.per_Tensor,
            )
            types = (fx.BFloat16, fx.Uint8, fx.BFloat16) + (fx.Int32,) * 3
            types += (fx.Float32,)
            gu(*[flyc.from_c_void_p(x, 0) for x in types], m, 1, fx.Stream(None))
            dn = _down(
                D,
                I,
                topk,
                s_max,
                t.nbd,
                t.waves,
                t.pwd,
                self.rows,
                t.w_cache,
                t.coalesce,
                self.quant == QuantType.per_Tensor,
                t.kc,
            )
            types = (fx.BFloat16, fx.Uint8, fx.BFloat16, fx.Int32, fx.Float32)
            types += (fx.Int32, fx.Int32, fx.Float32)
            dn(*[flyc.from_c_void_p(x, 0) for x in types], m, 1, fx.Stream(None))
            if self.rows:
                for has_nlt in (False, True):
                    cb = _combine(D, topk, has_nlt)
                    types = (fx.BFloat16, fx.Int32, fx.BFloat16, fx.Int32)
                    cb(*[flyc.from_c_void_p(x, 0) for x in types], m, fx.Stream(None))

    def __call__(self, call):
        from aiter.ops.flydsl.moe_next.core.tile_map import route_to_tiles

        spec = call.spec
        x = call.hidden_states
        M, D, I, topk = call.m, spec.model_dim, spec.inter_dim, call.topk
        dev = x.device
        t = _tile(call.config, self._gpu(call.weights), D, I)
        s_max = t.tile_m // 16
        nlt = call.num_local_tokens
        out = call.output
        if out is None:
            out = torch.empty((M, D), dtype=torch.bfloat16, device=dev)
        tm = route_to_tiles(
            call.topk_ids.to(torch.int32),
            call.topk_weight.to(torch.float32),
            num_experts=call.global_experts,
            tile_m=t.tile_m,
            expert_mask=call.expert_mask,
            num_local_tokens=nlt,
            zero_output=None if self.rows else out,
            with_loc_ids=self.rows,
            pad_rows=t.pad_rows,
        )
        tiles = tm.max_tiles
        inter = torch.empty((tm.rows, I), dtype=torch.bfloat16, device=dev)
        act, limit = _ACTS[spec.act], _swiglu_limit(spec)
        _launch(
            _gateup(
                2 * I,
                D,
                s_max,
                t.gu_waves,
                t.ch,
                t.ks,
                t.pw,
                t.pa,
                t.w_cache,
                act,
                limit,
                self.quant == QuantType.per_Tensor,
            ),
            x,
            call.w1,
            inter,
            tm.sorted_token_ids,
            tm.live_tiles,
            tm.num_live_tiles,
            call.w1_scale,
            M,
            tiles,
        )
        dst = out
        if self.rows:
            dst = torch.empty((M * topk, D), dtype=torch.bfloat16, device=dev)
        _launch(
            _down(
                D,
                I,
                topk,
                s_max,
                t.nbd,
                t.waves,
                t.pwd,
                self.rows,
                t.w_cache,
                t.coalesce,
                self.quant == QuantType.per_Tensor,
                t.kc,
            ),
            inter,
            call.w2,
            dst,
            tm.sorted_token_ids,
            tm.sorted_weights,
            tm.live_tiles,
            tm.num_live_tiles,
            call.w2_scale,
            M,
            tiles,
        )
        if self.rows:
            _launch(
                _combine(D, topk, nlt is not None),
                dst,
                tm.loc_ids,
                out,
                nlt if nlt is not None else tm.num_live_tiles,
                M,
            )
        return out


@register_kernel
class Gfx942Fp8TokenMidAtomic(Gfx942Fp8TokenMid):
    """Packed bf16 atomics into the sort-zeroed output, in place of rows + sum."""

    id = "gfx942_fp8_token.mid_atomic"
    epilogue = "atomic"
    numerics: ClassVar[dict] = {
        "act_quant": "none",
        "inter": "bf16",
        "combine": "bf16_atomic",
    }
    rows: ClassVar[bool] = False


@register_kernel
class Gfx942Fp8TensorMid(Gfx942Fp8TokenMid):
    """Per-tensor weights: the same kernels with one scale per expert. Activations
    stay unquantized bf16, so a dynamic per-tensor activation scale is never formed;
    a caller's static a1_scale is declined, as the family decode does."""

    id = "gfx942_fp8_tensor.mid"
    family = "fp8-tensor"
    quant: ClassVar = QuantType.per_Tensor


@register_kernel
class Gfx942Fp8TensorMidAtomic(Gfx942Fp8TokenMidAtomic):
    id = "gfx942_fp8_tensor.mid_atomic"
    family = "fp8-tensor"
    quant: ClassVar = QuantType.per_Tensor
