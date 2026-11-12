# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 MX MoE kernels for aiter.moe: a4w4, a8w4 and mxfp8 (per_1x32, e8m0).

One call: sort, per-token MX activation quant, stage 1 (gate/up GEMM, activation,
intermediate MX quant in stage 2's scale order), stage 2 (down GEMM, route weight,
combine). Weights stay in the caller's layout: fp4x2 or fp8 preshuffled with
shuffle_weight((16, 16)) / shuffle_weight_a16w4 (the same byte order), e8m0 scales
as e8m0_shuffle / shuffle_scale_a16w4; gate/up separated or 16-row interleaved
(gate_mode). The activation dtype is the one fused_moe itself resolves.
"""

import os

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.moe import MoeCall, MoeKernel, compile_only, register_kernel
from aiter.ops.flydsl.moe_common import GateMode

_ACTS = {
    ActivationType.Silu: "silu",
    ActivationType.Swiglu: "swiglu",
    ActivationType.Situv2: "situv2",
}
_DT = {dtypes.fp4x2: "fp4", dtypes.fp8: "fp8"}


def _pad(x, m):
    return (x + m - 1) // m * m


def activation_dtype(spec, weights, w1, m, hidden_dtype, has_a1_scale):
    """The activation dtype fused_moe quantizes to for this call."""
    from aiter.fused_moe import resolve_activation_dtype

    forced = weights.kwarg("quant_dtype_a")
    if forced is not None:
        return forced
    return resolve_activation_dtype(
        spec.quant,
        spec.w_dtype,
        activation=spec.act,
        gate_mode=spec.gate_mode,
        M=m,
        hidden_dtype=hidden_dtype,
        has_a1_scale=has_a1_scale,
        gate_up_interleave=getattr(w1, "gate_up_interleave", None),
    )


def parse_config(config):
    """'bm128_tn64_la4_nb7' -> {"bm": 128, "tn": 64, "la": 4, "nb": 7}; empty
    means defaults."""
    out = {}
    for tok in (config or "").split("_"):
        key = tok.rstrip("0123456789")
        keys = (
            "bm",
            "tn",
            "la",
            "nb",
            "ra",
            "nt",
            "ar",
            "dm",
            "iq",
            "xc",
            "nw",
            "sp",
        )
        if key in keys and tok[len(key) :]:
            out[key] = int(tok[len(key) :])
    return out


def tile_m_for(m, topk, experts, prefill):
    per_expert = -(-m * topk // experts)
    # Few large experts: one 64-row tile per expert streams its weights once and
    # keeps the MFMAs fed (4096/4096 E32: 310 -> 176 us at M=1024, 256 -> 184 at M=64).
    if experts <= 64 and m * topk >= 64 and per_expert <= 128:
        return 64
    if not prefill:
        return 16 if per_expert <= 16 else 32
    if per_expert > 64:
        return 128
    return 64 if per_expert > 32 else 32


class _MxGfx950(MoeKernel):
    family = "mx"
    archs = ("gfx950",)
    routing = "tiles"
    handles_kwargs = frozenset({"quant_dtype_a"})
    prefill = False

    def supports(self, spec, m, topk, call=None):
        if get_gfx_runtime() != "gfx950":
            return "not gfx950"
        if spec.quant != QuantType.per_1x32:
            return "quant is not per_1x32"
        if spec.w_dtype not in _DT:
            return f"weights are {spec.w_dtype}, not fp4x2/fp8"
        if not spec.g1u1 or not spec.w_shuffled:
            return "needs g1u1 preshuffled weights"
        if spec.act not in _ACTS:
            return f"activation {spec.act} not implemented"
        if spec.gate_mode not in (GateMode.SEPARATED, GateMode.INTERLEAVE):
            return f"gate_mode {spec.gate_mode} not implemented"
        if spec.doweight_stage1:
            return "doweight_stage1 not implemented"
        if spec.has_bias:
            return "bias not implemented"
        # intermediate_pad: the padded weight rows and columns are zero (as aiter's
        # tests and the frameworks build them), so computing the padded width
        # adds exact zeros; with a bias that would not hold.
        if spec.hidden_pad:
            return "hidden_pad not implemented"
        if spec.model_dim % 256 or spec.inter_dim % 32:
            return "model_dim % 256 or inter_dim % 32"
        if spec.out_dtype not in (None, torch.bfloat16):
            return "output dtype is not bf16"
        if spec.global_experts != spec.experts:
            return "EP not implemented"
        if call is None:
            return None
        x = call.hidden_states
        if x.dtype != torch.bfloat16 or x.stride(-1) != 1:
            return "hidden_states is not row-major bf16"
        if call.out_dtype not in (None, torch.bfloat16):
            return "output dtype is not bf16"
        a = activation_dtype(spec, call.weights, call.w1, m, x.dtype, False)
        if a not in _DT:
            return f"activation dtype {a} is not an MX family of this kernel"
        if spec.w_dtype == dtypes.fp8 and a != dtypes.fp8:
            return "mxfp8 needs fp8 activations"
        if call.expert_mask is not None or call.num_local_tokens is not None:
            return "expert_mask / num_local_tokens not implemented"
        if call.a1_scale is not None or call.a2_scale is not None:
            return "prequantized activations not implemented"
        if call.extra:
            return f"{sorted(call.extra)} not implemented"
        tag = getattr(call.w1, "gate_up_interleave", None)
        il = spec.gate_mode == GateMode.INTERLEAVE
        if tag is not None and bool(tag) != il:
            return "w1.gate_up_interleave disagrees with gate_mode"
        # Untagged weights follow aiter's convention (fused_moe
        # _flydsl_mxfp4_layout_is_compatible): fp4 activations come with separated
        # gate/up rows, fp8 activations with interleaved ones.
        if tag is None and il != (a == dtypes.fp8):
            return (
                f"untagged w1 with {_DT[a]} activations and gate_mode {spec.gate_mode}"
            )
        if not (call.w1.is_contiguous() and call.w2.is_contiguous()):
            return "weights not contiguous"
        E, D, I = spec.experts, spec.model_dim, spec.inter_dim
        if call.w1_scale.numel() < E * 2 * I * _pad(D // 32, 8):
            return "w1_scale is smaller than the e8m0_shuffle layout"
        if call.w2_scale.numel() < E * D * _pad(I // 32, 8):
            return "w2_scale is smaller than the e8m0_shuffle layout"
        return None

    def _kernels(self, spec, weights, m, topk, a_dtype, config):
        from aiter.ops.flydsl.kernels.moe_next.gfx950_mx.quant import (
            compile_token_quant,
        )
        from aiter.ops.flydsl.kernels.moe_next.gfx950_mx.stage1 import compile_stage1
        from aiter.ops.flydsl.kernels.moe_next.gfx950_mx.stage2 import (
            a_tile_resident,
            blocks_per_wg,
            compile_reduce,
            compile_stage2,
        )

        a = _DT[a_dtype]
        w = _DT[spec.w_dtype]
        cfg = parse_config(config)
        bm = cfg.get("bm") or tile_m_for(m, topk, spec.experts, self.prefill)
        tn = cfg.get("tn") or (
            128 if spec.inter_dim % 128 == 0 or spec.inter_dim % 64 else 64
        )
        # fp8 weights in prefill: two tiles ahead no longer fit 512 VGPRs (spills).
        la = cfg.get("la") or (
            (3 if self.prefill else 4)
            if spec.w_dtype != dtypes.fp8
            else (1 if self.prefill else 2)
        )
        routes = m * topk
        per_expert = -(-routes // spec.experts)
        live = min(spec.experts, routes) * -(-per_expert // bm)
        resident = a_tile_resident(spec.inter_dim, bm, a)
        nb = (cfg.get("nb") if resident else 1) or blocks_per_wg(
            spec.model_dim, live, resident=resident
        )
        reduce = self.epilogue == "reduce"
        xcd = bool(cfg.get("xc", 1 if self.prefill else 0))
        a_bytes = spec.model_dim if a == "fp8" else spec.model_dim // 2
        direct = (
            not self.prefill
            and bool(cfg.get("dm", 0 if spec.w_dtype == dtypes.fp8 else 1))
            and bm == 16
            and m * topk <= 256
            and (not cfg.get("iq", 0) or 16 * a_bytes <= 112 * 1024)
        )
        beta = weights.kwarg("beta", 1.0)
        lbeta = weights.kwarg("linear_beta", 1.0)
        q = compile_token_quant(model_dim=spec.model_dim, a_dtype=a)
        s1 = compile_stage1(
            model_dim=spec.model_dim,
            inter_dim=spec.inter_dim,
            experts=spec.experts,
            tile_m=bm,
            a_dtype=a,
            w_dtype=w,
            out_dtype=a,
            act=_ACTS[spec.act],
            interleave=spec.gate_mode == GateMode.INTERLEAVE,
            enable_bias=False,
            swiglu_limit=spec.swiglu_limit,
            situ_beta=float(beta),
            situ_linear_beta=float(lbeta),
            tile_n=tn,
            ahead=la,
            read_ahead=cfg.get("ra", 1),
            nt=bool(cfg.get("nt", 1)),
            a_resident=bool(cfg.get("ar", 0 if not self.prefill else 1)),
            direct_topk=topk if direct else 0,
            experts_total=spec.experts,
            inline_quant=bool(cfg.get("iq", 0)),
            xcd=xcd,
            waves=4 if direct else cfg.get("nw", 4),
            sched_pin=bool(cfg.get("sp", 0)),
        )
        s2 = compile_stage2(
            model_dim=spec.model_dim,
            inter_dim=spec.inter_dim,
            experts=spec.experts,
            tile_m=bm,
            a_dtype=a,
            w_dtype=w,
            doweight_stage1=False,
            reduce=reduce,
            topk=topk,
            blocks=1 if direct else nb,
            nt=bool(cfg.get("nt", 1)),
            direct=direct,
            xcd=xcd,
        )
        r = compile_reduce(model_dim=spec.model_dim, topk=topk) if reduce else None
        return bm, a, q, s1, s2, r, (direct, bool(cfg.get("iq", 0)), xcd)

    def warmup(self, weights, m, topk, config):
        import flydsl.expr as fx

        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        spec = weights.spec
        w1 = weights.w1
        a_dtype = activation_dtype(spec, weights, w1, m, torch.bfloat16, False)
        if a_dtype not in _DT:
            return
        _bm, _a, q, s1, s2, r, _direct = self._kernels(
            spec, weights, m, topk, a_dtype, config
        )
        st = fx.Stream(None)
        with compile_only():
            _run_compiled(q, 0, 0, 0, 1, spec.model_dim * 2, 1, st)
            _run_compiled(s1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, st)
            _run_compiled(s2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, st)
            if r is not None:
                _run_compiled(r, 0, 0, 1, 1, st)

    def __call__(self, call: MoeCall) -> torch.Tensor:
        import flydsl.expr as fx

        from aiter.fused_moe import moe_sorting
        from aiter.ops.flydsl.kernels.moe_next.gfx950_mx import common as cm
        from aiter.ops.flydsl.kernels.moe_next.gfx950_mx.quant import (
            token_quant_blocks,
        )
        from aiter.ops.flydsl.kernels.moe_next.gfx950_mx.stage2 import reduce_blocks
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
        from aiter.ops.flydsl.moe_next.core.tile_map import route_to_tiles

        spec, weights = call.spec, call.weights
        x = call.hidden_states
        M, D, I, E = call.m, spec.model_dim, spec.inter_dim, spec.experts
        a_dtype = activation_dtype(spec, weights, call.w1, M, x.dtype, False)
        bm, a, q, s1, s2, r, direct = self._kernels(
            spec, weights, M, call.topk, a_dtype, call.config
        )
        if direct[0]:
            return self._direct_call(call, spec, M, D, I, a, q, s1, s2, direct[1])
        dev = x.device
        out = call.output
        # Core's sort is faster at decode, Opus at large M (smci355, core be32165cd:
        # M=8 13.4 vs 16.4 us; DSV4-Pro M=4096 34.2 vs 30.7; Kimi-K3 63.7 vs 31.4).
        sort = os.environ.get("AITER_MOE_MX_SORT") or (
            "opus" if self.prefill else "core"
        )
        if sort == "core":
            if out is None:
                out = torch.empty((M, D), dtype=torch.bfloat16, device=dev)
            tm = route_to_tiles(
                call.topk_ids,
                call.topk_weight,
                num_experts=E,
                tile_m=bm,
                zero_output=None if r is not None else out,
            )
            sids, sw, se, nv = (
                tm.sorted_token_ids,
                tm.sorted_weights,
                tm.sorted_expert_ids,
                tm.num_valid_ids,
            )
        else:
            sids, sw, se, nv, out = moe_sorting(
                call.topk_ids,
                call.topk_weight,
                E,
                D,
                torch.bfloat16,
                block_size=bm,
                output=out,
            )
        rows, mblocks = sids.numel(), se.numel()
        if direct[2]:
            mblocks = (mblocks + 7) // 8 * 8
        stream = fx.Stream(torch.cuda.current_stream(dev))
        aq = torch.empty(
            (M, D // 2 if a == "fp4" else D), dtype=torch.uint8, device=dev
        )
        asc = torch.empty((M, cm.token_scale_cols(D)), dtype=torch.uint8, device=dev)
        _run_compiled(
            q,
            aq.data_ptr(),
            asc.data_ptr(),
            x.data_ptr(),
            M,
            x.stride(0) * x.element_size(),
            token_quant_blocks(M, D),
            stream,
        )
        inter = torch.empty(
            (rows, I // 2 if a == "fp4" else I), dtype=torch.uint8, device=dev
        )
        inter_s = torch.empty(
            (cm.inter_scale_bytes(rows, I),), dtype=torch.uint8, device=dev
        )
        _run_compiled(
            s1,
            inter.data_ptr(),
            inter_s.data_ptr(),
            aq.data_ptr(),
            asc.data_ptr(),
            call.w1.data_ptr(),
            call.w1_scale.data_ptr(),
            sids.data_ptr(),
            se.data_ptr(),
            nv.data_ptr(),
            0,
            M,
            mblocks,
            0,
            0,
            0,
            0,
            0,
            stream,
        )
        dst = out
        if r is not None:
            dst = torch.empty((M * call.topk, D), dtype=torch.bfloat16, device=dev)
        _run_compiled(
            s2,
            dst.data_ptr(),
            inter.data_ptr(),
            inter_s.data_ptr(),
            call.w2.data_ptr(),
            call.w2_scale.data_ptr(),
            sids.data_ptr(),
            se.data_ptr(),
            sw.data_ptr(),
            nv.data_ptr(),
            M,
            mblocks,
            stream,
        )
        if r is not None:
            _run_compiled(
                r, out.data_ptr(), dst.data_ptr(), M, reduce_blocks(M, D), stream
            )
        return out


def _direct_call(self, call, spec, M, D, I, a, q, s1, s2, inline):
    """Decode without a sort: stage 1 groups routes per expert itself, quantizes the
    token rows inline and zeroes the output; stage 2 reads its route rows."""
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.moe_next.gfx950_mx import common as cm
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    x = call.hidden_states
    dev = x.device
    R = M * call.topk
    rows = R * 16
    out = call.output
    if out is None:
        out = torch.empty((M, D), dtype=torch.bfloat16, device=dev)
    sid = torch.empty(rows, dtype=torch.int32, device=dev)
    sw = torch.empty(rows, dtype=torch.float32, device=dev)
    eid = torch.empty(R, dtype=torch.int32, device=dev)
    inter = torch.empty(
        (rows, I // 2 if a == "fp4" else I), dtype=torch.uint8, device=dev
    )
    inter_s = torch.empty(
        (cm.inter_scale_bytes(rows, I),), dtype=torch.uint8, device=dev
    )
    ids = call.topk_ids.contiguous()
    tw = call.topk_weight.contiguous()
    stream = fx.Stream(torch.cuda.current_stream(dev))
    aq = asc = None
    if not inline:
        from aiter.ops.flydsl.kernels.moe_next.gfx950_mx.quant import (
            token_quant_blocks,
        )

        aq = torch.empty(
            (M, D // 2 if a == "fp4" else D), dtype=torch.uint8, device=dev
        )
        asc = torch.empty((M, cm.token_scale_cols(D)), dtype=torch.uint8, device=dev)
        _run_compiled(
            q,
            aq.data_ptr(),
            asc.data_ptr(),
            x.data_ptr(),
            M,
            x.stride(0) * x.element_size(),
            token_quant_blocks(M, D),
            stream,
        )
    _run_compiled(
        s1,
        inter.data_ptr(),
        inter_s.data_ptr(),
        aq.data_ptr() if aq is not None else 0,
        asc.data_ptr() if asc is not None else 0,
        call.w1.data_ptr(),
        call.w1_scale.data_ptr(),
        sid.data_ptr(),
        eid.data_ptr(),
        sw.data_ptr(),
        0,
        M,
        R,
        ids.data_ptr(),
        tw.data_ptr(),
        x.data_ptr(),
        out.data_ptr(),
        x.stride(0) * x.element_size(),
        stream,
    )
    _run_compiled(
        s2,
        out.data_ptr(),
        inter.data_ptr(),
        inter_s.data_ptr(),
        call.w2.data_ptr(),
        call.w2_scale.data_ptr(),
        sid.data_ptr(),
        eid.data_ptr(),
        sw.data_ptr(),
        0,
        M,
        R,
        stream,
    )
    return out


_MxGfx950._direct_call = _direct_call

_ACT_TERM = {"fp4": "mx_fp4_round_up", "fp8": "mx_fp8_round_up"}
_FAMILIES = {
    # family: (activation dtype, weight dtype)
    "a4w4": (dtypes.fp4x2, dtypes.fp4x2),
    "a8w4": (dtypes.fp8, dtypes.fp4x2),
    "mxfp8": (dtypes.fp8, dtypes.fp8),
}


def _make(fam, prefill):
    a_dt, w_dt = _FAMILIES[fam]
    name = f"Mx{fam.capitalize()}Gfx950{'Prefill' if prefill else 'Decode'}"

    def supports(self, spec, m, topk, call=None):
        if spec.w_dtype != w_dt:
            return f"weights are {spec.w_dtype}, not {fam}"
        if call is not None:
            a = activation_dtype(
                spec, call.weights, call.w1, m, call.hidden_states.dtype, False
            )
            if a != a_dt:
                return f"activation dtype {a} is not {fam}"
        return _MxGfx950.supports(self, spec, m, topk, call)

    cls = type(
        name,
        (_MxGfx950,),
        {
            "id": f"gfx950_mx_{fam}.{'prefill' if prefill else 'decode'}",
            "family": fam,
            "epilogue": "reduce" if prefill else "atomic",
            "prefill": prefill,
            "numerics": {
                "act_quant": _ACT_TERM[_DT[a_dt]],
                "inter": "fp32_direct",
                "combine": "bf16_rows_fp32_sum" if prefill else "bf16_atomic",
            },
            "supports": supports,
        },
    )
    return register_kernel(cls)


for _fam in _FAMILIES:
    for _prefill in (False, True):
        _make(_fam, _prefill)
