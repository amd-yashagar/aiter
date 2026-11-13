# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 bf16 (a16w16) MoE on core's tile map, for aiter.moe.

The mid-M structure of the fp8-token family without its quantization: core's
sort gives one live tile per expert (or a few under skew); Gate/Up and Down are
driven by the live-tile list and read each live expert's weights once per tile.
Rounding points are those of aiter's CK 2-stage path (bf16 activations, FP32
accumulation, the activation in FP32, a bf16 intermediate), with the combine
either one bf16 row per route summed per token in FP32 (``mid``, as aiter's
gfx942 whole-graph prefill) or packed bf16 atomics (``mid_atomic``, as CK).
Tile configs are data: configs/<gpu>/bf16-mid.json.
"""

import functools
import json
import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from aiter import ActivationType, QuantType
from aiter.moe import MoeKernel, compile_only, register_kernel
from aiter.ops.flydsl.moe_common import GateMode

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
_ACTS = {ActivationType.Silu: "silu", ActivationType.Gelu: "gelu"}
_LDS_BYTES = 64 * 1024


@functools.cache
def _configs(gpu):
    path = os.path.join(_CONFIG_DIR, gpu, "bf16-mid.json")
    if not os.path.exists(path):
        path = os.path.join(_CONFIG_DIR, "mi300x", "bf16-mid.json")
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
    nbd, waves = int(cfg.get("down_nbd", 2)), int(cfg.get("down_waves", 4))
    while waves > 1 and model_dim % (16 * nbd * waves):
        waves //= 2
    while nbd > 1 and model_dim % (16 * nbd * waves):
        nbd //= 2
    tile_m = int(cfg.get("tile_m", 32))
    # Down stages the tile's intermediate rows in LDS.
    while tile_m > 16 and tile_m * (inter_dim * 2 + 16) > _LDS_BYTES:
        tile_m //= 2
    return _Tile(
        tile_m,
        gw,
        ch,
        ks,
        int(cfg.get("gateup_pw", 2)),
        int(cfg.get("gateup_pa", 2)),
        int(cfg.get("w_cache", 0)),
        nbd,
        waves,
        int(cfg.get("down_pw", 2)),
        bool(cfg.get("down_coalesce", True)),
    )


def _launch(fn, *args):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import _ptr
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    args = [_ptr(a) if isinstance(a, torch.Tensor) else a for a in args]
    _run_compiled(fn, *args, torch.cuda.current_stream())


@functools.cache
def _gateup(I, K, s_max, waves, ch, ks, pw, pa, w_cache, act, g1u1, doweight):
    from aiter.ops.flydsl.kernels.moe_next.gfx942_bf16.gateup import compile_gateup

    return compile_gateup(
        I=I,
        K=K,
        S_MAX=s_max,
        WAVES=waves,
        CH=ch,
        KS=ks,
        PW=pw,
        PA=pa,
        W_CACHE=w_cache,
        activation=act,
        G1U1=g1u1,
        DOWEIGHT=doweight,
    )


@functools.cache
def _down(N, K, topk, s_max, nbd, waves, pwd, rows, doweight, w_cache, coalesce):
    from aiter.ops.flydsl.kernels.moe_next.gfx942_bf16.down import compile_down

    return compile_down(
        N=N,
        K=K,
        TOPK=topk,
        S_MAX=s_max,
        NBD=nbd,
        WAVES=waves,
        PWD=pwd,
        ROWS=rows,
        DOWEIGHT=doweight,
        W_CACHE=w_cache,
        COALESCE=coalesce,
    )


@functools.cache
def _combine(N, topk, has_nlt):
    from aiter.ops.flydsl.kernels.moe_next.gfx942_bf16.down import compile_combine

    return compile_combine(N=N, TOPK=topk, HAS_NLT=has_nlt)


@register_kernel
class Gfx942Bf16Mid(MoeKernel):
    id = "gfx942_bf16.mid"
    family = "a16w16"
    archs = ("gfx942",)
    routing = "tiles"
    epilogue = "reduce"
    numerics: ClassVar[dict] = {
        "act_quant": "none",
        "inter": "bf16",
        "combine": "bf16_rows_fp32_sum",
    }
    rows: ClassVar[bool] = True

    def supports(self, spec, m, topk, call=None):
        if spec.quant != QuantType.No:
            return "needs QuantType.No"
        if spec.w_dtype != torch.bfloat16 or not spec.w_shuffled:
            return "needs shuffled bf16 weights"
        if not spec.g1u1:
            return "g1u0 not implemented yet"
        if spec.act not in _ACTS:
            return f"activation {spec.act}"
        if spec.swiglu_limit:
            return "swiglu_limit"
        if spec.gate_mode != GateMode.SEPARATED:
            return "gate/up must be separated"
        if spec.has_bias or spec.hidden_pad or spec.intermediate_pad:
            return "bias or padding"
        if spec.out_dtype not in (None, torch.bfloat16):
            return "bf16 output only"
        D, I = spec.model_dim, spec.inter_dim
        if D % 256 or I % 64:
            return "model_dim % 256 and inter_dim % 64"
        if 16 * (I * 2 + 16) > _LDS_BYTES:
            return "inter_dim too large for one 16-row tile in LDS"
        if topk > 255:
            return "topk"
        if call is not None:
            if call.hidden_states.dtype != torch.bfloat16:
                return "bf16 hidden_states only"
            if not call.hidden_states.is_contiguous():
                return "contiguous hidden_states only"
            if call.out_dtype != torch.bfloat16:
                return "bf16 output only"
            if call.w1_scale is not None or call.w2_scale is not None:
                return "weight scales"
            if call.a1_scale is not None or call.a2_scale is not None:
                return "activation scales"
            if not (call.w1.is_contiguous() and call.w2.is_contiguous()):
                return "contiguous weights only"
            if call.output is not None and not call.output.is_contiguous():
                return "contiguous output only"
            if call.extra:
                return f"unsupported {sorted(call.extra)}"
        return None

    def _gpu(self, weights):
        return getattr(weights, "gpu", None) or "mi300x"

    def _kernels(self, spec, topk, t):
        D, I = spec.model_dim, spec.inter_dim
        s_max = t.tile_m // 16
        dw = bool(spec.doweight_stage1)
        gu = _gateup(
            I,
            D,
            s_max,
            t.gu_waves,
            t.ch,
            t.ks,
            t.pw,
            t.pa,
            t.w_cache,
            _ACTS[spec.act],
            True,
            dw,
        )
        dn = _down(
            D,
            I,
            topk,
            s_max,
            t.nbd,
            t.waves,
            t.pwd,
            self.rows,
            dw,
            t.w_cache,
            t.coalesce,
        )
        return gu, dn

    def warmup(self, weights, m, topk, config):
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        spec = weights.spec
        t = _tile(config, self._gpu(weights), spec.model_dim, spec.inter_dim)
        with compile_only():
            gu, dn = self._kernels(spec, topk, t)
            types = (fx.BFloat16, fx.BFloat16, fx.BFloat16, fx.Int32, fx.Float32)
            types += (fx.Int32, fx.Int32)
            for fn in (gu, dn):
                fn(*[flyc.from_c_void_p(x, 0) for x in types], m, 1, fx.Stream(None))
            if self.rows:
                for has_nlt in (False, True):
                    cb = _combine(spec.model_dim, topk, has_nlt)
                    types = (fx.BFloat16, fx.Int32, fx.BFloat16, fx.Int32)
                    cb(*[flyc.from_c_void_p(x, 0) for x in types], m, fx.Stream(None))

    def __call__(self, call):
        from aiter.ops.flydsl.moe_next.core.tile_map import route_to_tiles

        spec = call.spec
        x = call.hidden_states
        M, D, I, topk = call.m, spec.model_dim, spec.inter_dim, call.topk
        dev = x.device
        t = _tile(call.config, self._gpu(call.weights), D, I)
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
        )
        gu, dn = self._kernels(spec, topk, t)
        tiles = tm.max_tiles
        inter = torch.empty((tm.rows, I), dtype=torch.bfloat16, device=dev)
        _launch(
            gu,
            x,
            call.w1,
            inter,
            tm.sorted_token_ids,
            tm.sorted_weights,
            tm.live_tiles,
            tm.num_live_tiles,
            M,
            tiles,
        )
        dst = out
        if self.rows:
            dst = torch.empty((M * topk, D), dtype=torch.bfloat16, device=dev)
        _launch(
            dn,
            inter,
            call.w2,
            dst,
            tm.sorted_token_ids,
            tm.sorted_weights,
            tm.live_tiles,
            tm.num_live_tiles,
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
class Gfx942Bf16MidAtomic(Gfx942Bf16Mid):
    """Packed bf16 atomics into the sort-zeroed output, in place of rows + sum."""

    id = "gfx942_bf16.mid_atomic"
    epilogue = "atomic"
    numerics: ClassVar[dict] = {
        "act_quant": "none",
        "inter": "bf16",
        "combine": "bf16_atomic",
    }
    rows: ClassVar[bool] = False
