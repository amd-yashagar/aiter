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


def tile_m_for(m, topk, experts, prefill):
    per_expert = -(-m * topk // experts)
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
            compile_reduce,
            compile_stage2,
        )

        a = _DT[a_dtype]
        w = _DT[spec.w_dtype]
        bm = int(config[2:]) if config.startswith("bm") else None
        bm = bm or tile_m_for(m, topk, spec.experts, self.prefill)
        reduce = self.epilogue == "reduce"
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
            round_bf16=False,
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
        )
        r = compile_reduce(model_dim=spec.model_dim, topk=topk) if reduce else None
        return bm, a, q, s1, s2, r

    def warmup(self, weights, m, topk, config):
        import flydsl.expr as fx

        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        spec = weights.spec
        w1 = weights.w1
        a_dtype = activation_dtype(spec, weights, w1, m, torch.bfloat16, False)
        if a_dtype not in _DT:
            return
        _bm, _a, q, s1, s2, r = self._kernels(spec, weights, m, topk, a_dtype, config)
        st = fx.Stream(None)
        with compile_only():
            _run_compiled(q, 0, 0, 0, 1, spec.model_dim * 2, 1, st)
            _run_compiled(s1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, st)
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

        spec, weights = call.spec, call.weights
        x = call.hidden_states
        M, D, I, E = call.m, spec.model_dim, spec.inter_dim, spec.experts
        a_dtype = activation_dtype(spec, weights, call.w1, M, x.dtype, False)
        bm, a, q, s1, s2, r = self._kernels(
            spec, weights, M, call.topk, a_dtype, call.config
        )
        dev = x.device
        sids, sw, se, nv, out = moe_sorting(
            call.topk_ids,
            call.topk_weight,
            E,
            D,
            torch.bfloat16,
            block_size=bm,
            output=call.output,
        )
        rows, mblocks = sids.numel(), se.numel()
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


@register_kernel
class MxGfx950Decode(_MxGfx950):
    id = "gfx950_mx.decode"
    epilogue = "atomic"
    prefill = False


@register_kernel
class MxGfx950Prefill(_MxGfx950):
    id = "gfx950_mx.prefill"
    epilogue = "reduce"
    prefill = True
