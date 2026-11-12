# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Case -> tensors: weights (quantized and shuffled once per group), routing,
activations, and the torch reference on the same quantized weights.

Weight and activation preparation follows ``op_tests/test_moe_2stage.py::test_fmoe``
(the quant, shuffle and bias branches), but runs on the GPU in expert chunks so the
big 256-896 expert groups fit. The reference is aiter's ``torch_moe_stage1/2``
called per expert chunk and token chunk, which gives the same result as one call.
"""

import math
from dataclasses import dataclass

import torch

from .schema import _flag_on, ep_local_experts

ROUTE_DISTS = ("uniform", "zipf", "hot1")
CHUNK_ELEMS = 1 << 28  # fp32 elements generated per expert chunk
REF_FP32_BYTES = 3 << 30  # cap on dequantized reference weights per call
REF_ACT_BYTES = 3 << 30  # cap on repeated fp32 activations per call


@dataclass
class Spec:
    case_id: str
    kind: str
    M: int
    D: int
    I: int
    E: int
    E_local: int
    topk: int
    act: object
    act_name: str
    dtype: torch.dtype
    aq: object
    wq: object
    qtype: object
    g1u1: bool
    doweight: bool
    gate_mode: str
    hidden_pad: int
    intermediate_pad: int
    swiglu_limit: object
    beta: object
    linear_beta: object
    bias: bool
    is_ep: bool
    route_dist: str
    empty_frac: float
    num_local_frac: float
    output_buf: bool
    extra: dict


def _dtype(aiter, name, per_token_i4=False):
    from aiter import dtypes

    if name in ("", "None"):
        return None
    if per_token_i4 and name == "i4x2":
        return torch.uint32
    return {
        "bf16": dtypes.bf16,
        "fp16": dtypes.fp16,
        "fp32": dtypes.fp32,
        "fp8": dtypes.fp8,
        "fp4x2": dtypes.fp4x2,
        "i8": dtypes.i8,
        "i4x2": dtypes.i4x2,
        "int4": torch.uint32,
    }[name]


def effective_gate_mode(aiter, qtype, aq, wq):
    """Mirror of test_moe_2stage._effective_gate_mode."""
    from aiter import dtypes
    from aiter.ops.flydsl.moe_common import GateMode

    if aq == dtypes.bf16 and wq == dtypes.fp4x2:
        return GateMode.SEPARATED.value
    if aq == dtypes.fp8 and wq == dtypes.fp4x2:
        return GateMode.INTERLEAVE.value
    if qtype == aiter.QuantType.per_1x32 and aq == dtypes.fp8 and wq == dtypes.fp8:
        return GateMode.INTERLEAVE.value
    return GateMode.SEPARATED.value


def resolve(case):
    """Case -> Spec with aiter enums and dtypes. Imports aiter lazily."""
    import aiter
    from aiter import dtypes

    r = case.row
    ex = case.extra
    qt_name = r["q_type"] or "No"
    qtype = {
        "No": aiter.QuantType.No,
        "per_Tensor": aiter.QuantType.per_Tensor,
        "per_Token": aiter.QuantType.per_Token,
        "per_1x32": aiter.QuantType.per_1x32,
        "per_1x128": aiter.QuantType.per_1x128,
        "per_128x128": aiter.QuantType.per_1x128,
    }[qt_name]
    act_name = r["act_type"] or "Silu"
    limit = None
    act_enum_name = "Silu" if act_name == "silu_clamped" else act_name
    act = getattr(aiter.ActivationType, act_enum_name)
    if act_name == "silu_clamped" or _flag_on(ex.get("swiglu_limit")):
        v = ex.get("swiglu_limit")
        try:
            limit = float(v) if v not in (None, "1", "") else None
        except ValueError:
            limit = None
        if limit is None:
            limit = 10.0 if r["q_dtype_w"] == "fp4x2" else 7.0
    pt_i4 = qt_name == "per_Token"
    aq = _dtype(aiter, r["q_dtype_a"] or r["dtype"])
    wq = _dtype(aiter, r["q_dtype_w"] or r["dtype"], per_token_i4=pt_i4)
    if r["q_type"] in ("", "No"):
        aq = wq = _dtype(aiter, r["dtype"])
    dtype = _dtype(aiter, r["dtype"])
    gate = ex.get("gate_mode")
    gate = gate.lower() if gate else effective_gate_mode(aiter, qtype, aq, wq)
    bias_flag = ex.get("bias", "")
    if bias_flag in ("", None):
        bias = (
            act_name == "Swiglu"
            and qtype == aiter.QuantType.per_1x32
            and wq == dtypes.fp4x2
            and aq in (dtypes.bf16, dtypes.fp8)
        )
    else:
        bias = bias_flag not in ("none", "None", "0")
    beta = linear_beta = None
    if act_name == "Situv2":
        from aiter.ops.flydsl.moe_common import (
            DEFAULT_SITUV2_BETA,
            DEFAULT_SITUV2_LINEAR_BETA,
        )

        beta, linear_beta = DEFAULT_SITUV2_BETA, DEFAULT_SITUV2_LINEAR_BETA
        if aq == dtypes.bf16 and wq == dtypes.fp4x2:
            bias = False
    if qtype == aiter.QuantType.per_1x32 and wq == dtypes.i4x2:
        bias = False
    E = int(r["expert"])
    is_ep = _flag_on(ex.get("ep_local")) or _flag_on(ex.get("expert_mask"))
    return Spec(
        case_id=case.id,
        kind=r["kind"],
        M=int(r["token"]),
        D=int(r["model_dim"] or 0),
        I=int(r["inter_dim"] or 0),
        E=E,
        E_local=ep_local_experts(case),
        topk=int(r["topk"]),
        act=act,
        act_name=act_name,
        dtype=dtype,
        aq=aq,
        wq=wq,
        qtype=qtype,
        g1u1=r["use_g1u1"] in ("1", "True", "true", "1.0"),
        doweight=r["doweight_stage1"] in ("1", "True", "true", "1.0"),
        gate_mode=gate,
        hidden_pad=int(float(ex.get("hidden_pad", 0) or 0)),
        intermediate_pad=int(float(ex.get("intermediate_pad", 0) or 0)),
        swiglu_limit=limit,
        beta=beta,
        linear_beta=linear_beta,
        bias=bias,
        is_ep=is_ep,
        route_dist=ex.get("route_dist", "uniform"),
        empty_frac=float(ex.get("empty_experts", 0) or 0),
        num_local_frac=float(ex.get("num_local_tokens", 0) or 0),
        output_buf=_flag_on(ex.get("output_buf")),
        extra=ex,
    )


def group_weight_bytes(S):
    """Rough bytes of one group's aiter weights plus reference weights."""
    from aiter import dtypes

    n1 = (2 if S.g1u1 else 1) * S.I * S.D
    n2 = S.D * S.I
    per = n1 + n2
    el = 1
    if S.wq in (dtypes.bf16, dtypes.fp16):
        el = 2
    elif S.wq in (dtypes.fp4x2, dtypes.i4x2) or S.wq == torch.uint32:
        el = 0.5
    return int(per * S.E_local * el)


def _seed(*parts):
    return int.from_bytes(
        __import__("hashlib").sha1("|".join(map(str, parts)).encode()).digest()[:4],
        "big",
    )


def _expert_chunk(S):
    per = (2 if S.g1u1 else 1) * S.I * S.D + S.D * S.I
    return max(1, min(S.E_local, CHUNK_ELEMS // max(per, 1)))


def _quant_block_128(aiter, weight, quant_dtype):
    """test_moe_2stage.weight_per_128x128_quant."""
    E, d1, d2 = weight.shape
    wb = weight.view(E, d1 // 128, 128, d2 // 128, 128).permute(0, 1, 3, 2, 4)
    wb = wb.contiguous().view(E, -1, 128 * 128)
    wq, ws = aiter.pertoken_quant(wb, quant_dtype=quant_dtype)
    wq = wq.view(E, d1 // 128, d2 // 128, 128, 128).permute(0, 1, 3, 2, 4)
    wq = wq.contiguous().view(E, d1, d2)
    return wq, ws.view(E, d1 // 128, d2 // 128)


def quantize_chunk(S, w1, w2):
    """Quantize one expert chunk. Returns (ref, aiter) tensor dicts.

    ref holds the unshuffled quantized weights and scales the torch reference
    reads; aiter holds what ``fused_moe`` is called with (shuffled).
    """
    import aiter
    from aiter import dtypes
    from aiter.int4_utils import convert_int8_to_uint32_int4, rearrange_4bit_elements
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant, per_1x32_i4_quant
    from aiter.ops.shuffle import (
        pack_int8_to_packed_int4,
        shuffle_scale_a16w4,
        shuffle_scale_for_int4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility import fp4_utils

    qT, AQ, WQ = S.qtype, S.aq, S.wq
    e = w1.shape[0]
    torch_quant = aiter.get_torch_quant(qT)
    is_mxfp8 = qT == aiter.QuantType.per_1x32 and AQ == dtypes.fp8 and WQ == dtypes.fp8
    is_block = qT == aiter.QuantType.per_1x128
    if qT == aiter.QuantType.per_Tensor:
        w1q, w1s = aiter.pertoken_quant(w1.view(e, -1), quant_dtype=WQ)
        w2q, w2s = aiter.pertoken_quant(w2.view(e, -1), quant_dtype=WQ)
        w1q, w2q = w1q.view(w1.shape), w2q.view(w2.shape)
    elif qT == aiter.QuantType.per_Token and WQ == torch.uint32:
        w1q, w1s = aiter.pertoken_quant(w1, quant_dtype=dtypes.i8, dtypeMax=7)
        w2q, w2s = aiter.pertoken_quant(w2, quant_dtype=dtypes.i8, dtypeMax=7)
    elif qT == aiter.QuantType.per_1x32 and WQ == dtypes.i4x2:
        w1q, w1s = per_1x32_i4_quant(w1)
        w2q, w2s = per_1x32_i4_quant(w2)
        w1q, w2q = w1q.view(dtypes.i4x2), w2q.view(dtypes.i4x2)
    elif is_mxfp8:
        w1q, w1s = per_1x32_f8_scale_f8_quant(
            w1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        w2q, w2s = per_1x32_f8_scale_f8_quant(
            w2, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    elif is_block:
        w1q, w1s = _quant_block_128(aiter, w1, WQ)
        w2q, w2s = _quant_block_128(aiter, w2, WQ)
    else:
        w1q, w1s = torch_quant(w1, quant_dtype=WQ)
        w2q, w2s = torch_quant(w2, quant_dtype=WQ)
    if qT == aiter.QuantType.per_1x32 and WQ not in (dtypes.i4x2, dtypes.fp8):
        w1q = w1q.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
        w2q = w2q.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)
    else:
        w1q, w2q = w1q.view(w1.shape), w2q.view(w2.shape)
    ref = {"w1": w1q, "w2": w2q, "w1_scale": w1s, "w2_scale": w2s}

    w1a, w2a, w1sa, w2sa = w1q, w2q, w1s, w2s
    if qT == aiter.QuantType.per_1x32 and WQ == dtypes.i4x2:
        w1a = pack_int8_to_packed_int4(shuffle_weight(w1a.view(dtypes.i8), (16, 16)))
        w1a = w1a.view(e, w1.shape[1], w1.shape[2] // 2).view(dtypes.i4x2)
        w2a = pack_int8_to_packed_int4(shuffle_weight(w2a.view(dtypes.i8), (16, 16)))
        w2a = w2a.view(e, w2.shape[1], w2.shape[2] // 2).view(dtypes.i4x2)
        w1sa = shuffle_scale_for_int4(w1s, group_size=32).view(-1).contiguous()
        w2sa = shuffle_scale_for_int4(w2s, group_size=32).view(-1).contiguous()
    elif WQ == torch.uint32:
        w1a = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(shuffle_weight(w1a, (16, 16), use_int4=True))
        )
        w2a = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(shuffle_weight(w2a, (16, 16), use_int4=True))
        )
        w1sa = fp4_utils.e8m0_shuffle(w1s)
        w2sa = fp4_utils.e8m0_shuffle(w2s)
    elif (
        qT == aiter.QuantType.per_1x32
        and AQ in (dtypes.bf16, dtypes.fp16, dtypes.fp8)
        and WQ == dtypes.fp4x2
    ):
        gu = AQ == dtypes.fp8
        w1a = shuffle_weight_a16w4(w1a, 16, gu)
        w1sa = shuffle_scale_a16w4(w1s, e, gu)
        w2a = shuffle_weight_a16w4(w2a, 16, False)
        w2sa = shuffle_scale_a16w4(w2s, e, False)
    elif is_mxfp8:
        w1a = shuffle_weight_a16w4(w1a, 16, True)
        w1sa = shuffle_scale_a16w4(w1s, e, True)
        w2a = shuffle_weight_a16w4(w2a, 16, False)
        w2sa = fp4_utils.e8m0_shuffle(w2s)
    else:
        w1a = shuffle_weight(w1a, layout=(16, 16))
        w2a = shuffle_weight(w2a, layout=(16, 16))
        w1sa = fp4_utils.e8m0_shuffle(w1s)
        w2sa = fp4_utils.e8m0_shuffle(w2s)
    attrs = {
        "w1": {"gate_up_interleave": getattr(w1a, "gate_up_interleave", None)},
        "w2": {"gate_up_interleave": getattr(w2a, "gate_up_interleave", None)},
    }
    return ref, {
        "w1": w1a,
        "w2": w2a,
        "w1_scale": w1sa,
        "w2_scale": w2sa,
        "_attrs": attrs,
    }


def _scale_view(t, e):
    """Return t as [e, rows_per_expert, ...] preserving trailing dims where possible."""
    if t is None:
        return None
    if t.shape[0] == e:
        return t
    return t.reshape(e, t.shape[0] // e, *t.shape[1:])


def _alloc_like_chunks(chunks, E):
    """Allocate full tensors from the first chunk's per-expert shapes."""
    out = {}
    e0 = chunks["_e"]
    for k, t in chunks.items():
        if k in ("_e", "_attrs"):
            continue
        if t is None:
            out[k] = None
            continue
        if t.shape[0] == e0:
            shape = (E, *t.shape[1:])
        else:
            assert t.shape[0] % e0 == 0, (k, t.shape, e0)
            shape = (t.shape[0] // e0 * E, *t.shape[1:])
        out[k] = torch.empty(shape, dtype=t.dtype, device=t.device)
    return out


def _store(full, chunk, e0, e):
    for k, t in chunk.items():
        if t is None or k in ("_e", "_attrs"):
            continue
        per = t.shape[0] // chunk["_e"]
        full[k][e0 * per : (e0 + e) * per].copy_(t)


def build_group(S, same_chunk=False):
    """Weights, scales and biases for one group.

    Returns (aiter_tensors, ref_tensors, bias). ``same_chunk`` fills every expert
    from one quantized chunk: used by the support probe, where only shapes,
    dtypes and layout flags matter.
    """
    dev = torch.device("cuda")
    E = S.E_local
    n1 = (2 if S.g1u1 else 1) * S.I
    ec = _expert_chunk(S)
    ref_full = aiter_full = attrs = None
    base = _seed("weights", S.D, S.I, E, S.g1u1, S.qtype, S.wq, S.aq)
    for ci, e0 in enumerate(range(0, E, ec)):
        e = min(ec, E - e0)
        if same_chunk and ci > 0:
            for full in (ref_full, aiter_full):
                for t in full.values():
                    if t is not None:
                        per = t.shape[0] // E
                        t[e0 * per : (e0 + e) * per].copy_(t[: e * per])
            continue
        g = torch.Generator(device=dev).manual_seed(base + ci)
        w1 = torch.randn((e, n1, S.D), dtype=S.dtype, device=dev, generator=g)
        w2 = torch.randn((e, S.D, S.I), dtype=S.dtype, device=dev, generator=g)
        if S.hidden_pad:
            w1[:, :, -S.hidden_pad :] = 0
            w2[:, -S.hidden_pad :, :] = 0
        if S.intermediate_pad:
            w1[:, -S.intermediate_pad :, :] = 0
            w1[:, S.I - S.intermediate_pad : S.I, :] = 0
            w2[:, :, -S.intermediate_pad :] = 0
        ref_c, aiter_c = quantize_chunk(S, w1, w2)
        del w1, w2
        ref_c["_e"] = aiter_c["_e"] = e
        if attrs is None:
            attrs = aiter_c["_attrs"]
        if ref_full is None:
            ref_full = _alloc_like_chunks(ref_c, E)
            aiter_full = _alloc_like_chunks(aiter_c, E)
        _store(ref_full, ref_c, e0, e)
        _store(aiter_full, aiter_c, e0, e)
        del ref_c, aiter_c
    for k in ("w1", "w2"):
        aiter_full[k].is_shuffled = True
        for name, val in (attrs or {}).get(k, {}).items():
            if val is not None:
                setattr(aiter_full[k], name, val)
    bias = None
    if S.bias:
        g = torch.Generator(device=dev).manual_seed(base + 7777)
        b1 = torch.clamp(
            torch.randn((E, n1), dtype=S.dtype, device=dev, generator=g), -1.0, 1.0
        )
        b2 = torch.clamp(
            torch.randn((E, S.D), dtype=S.dtype, device=dev, generator=g), -1.0, 1.0
        )
        bias = (b1, b2)
    return aiter_full, ref_full, bias


def make_routing(S, M, seed_tag=""):
    """Seeded router: softmax + topk on logits shaped by the case's distribution."""
    dev = torch.device("cuda")
    E, topk = S.E, S.topk
    g = torch.Generator(device=dev).manual_seed(_seed("route", S.case_id, M, seed_tag))
    logits = torch.randn((M, E), dtype=torch.float32, device=dev, generator=g)
    if S.route_dist == "zipf":
        perm = torch.randperm(E, generator=g, device=dev)
        rank = torch.empty(E, device=dev)
        rank[perm] = torch.arange(1, E + 1, device=dev, dtype=torch.float32)
        logits = logits + 3.0 * torch.log(1.0 / rank)
    elif S.route_dist == "hot1":
        logits[:, 0] += 12.0
    if S.empty_frac > 0:
        n_empty = min(E - topk, int(E * S.empty_frac))
        empty = torch.randperm(E, generator=g, device=dev)[:n_empty]
        logits[:, empty] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    w, ids = torch.topk(probs, topk, dim=-1)
    w = (w / w.sum(-1, keepdim=True)).to(torch.float32).contiguous()
    return w, ids.to(torch.int32).contiguous()


def make_expert_mask(S):
    mask = torch.zeros((S.E,), dtype=torch.int32, device="cuda")
    start = (S.E - S.E_local) // 2
    mask[start : start + S.E_local] = 1
    return mask


def make_inputs(S, M):
    g = torch.Generator(device="cuda").manual_seed(_seed("input", S.case_id, M))
    return torch.randn((M, S.D), dtype=S.dtype, device="cuda", generator=g)


def _slice_experts(t, e, e0, e1):
    if t is None:
        return None
    if t.shape[0] == e:
        return t[e0:e1]
    r = t.shape[0] // e
    return t[e0 * r : e1 * r]


def reference(
    S, ref_w, bias, hidden, topk_w, topk_ids, expert_mask, n_valid=None, hp=False
):
    """Torch reference on the unshuffled quantized weights, same flow as test_fmoe.

    ``hp`` is the high-precision variant for error measurement: the same quantized
    weights and activation quantization, fp32 accumulation, but the stage-1 output
    stays fp32 (no bf16 rounding of the intermediate) and the result is returned in
    fp32.
    """
    import aiter
    from aiter import dtypes
    from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant

    M = hidden.shape[0] if n_valid is None else n_valid
    hidden, topk_w, topk_ids = hidden[:M], topk_w[:M], topk_ids[:M]
    qT, AQ, WQ = S.qtype, S.aq, S.wq
    torch_quant = aiter.get_torch_quant(qT)
    is_mxfp8 = qT == aiter.QuantType.per_1x32 and AQ == dtypes.fp8 and WQ == dtypes.fp8
    E = S.E_local
    if expert_mask is not None:
        local = expert_mask.cumsum(0, dtype=torch.int32) - 1
        local[expert_mask == 0] = -1
        ids = local[topk_ids.long()].to(torch.int32)
    else:
        ids = topk_ids
    if qT == aiter.QuantType.per_1x128:
        a1q, a1s = aiter.pertoken_quant(hidden.view(M, -1, 128), quant_dtype=AQ)
        a1q, a1s = a1q.view(M, S.D), a1s.squeeze(-1)
    elif qT == aiter.QuantType.per_1x32 and AQ == dtypes.fp8 and WQ == dtypes.fp4x2:
        a1q, a1s = per_1x32_f8_scale_f8_quant(
            hidden, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    elif (
        (
            qT == aiter.QuantType.per_1x32
            and AQ in (dtypes.bf16, dtypes.fp16, dtypes.fp8)
            and WQ == dtypes.fp4x2
        )
        or is_mxfp8
        or (qT == aiter.QuantType.per_1x32 and WQ == dtypes.i4x2)
    ):
        a1q, a1s = hidden.to(dtypes.bf16), None
    else:
        a1q, a1s = torch_quant(hidden, quant_dtype=AQ)

    n1 = (2 if S.g1u1 else 1) * S.I
    w_el = max(1, ref_w["w1"].element_size())
    ec = max(1, min(E, REF_FP32_BYTES // (n1 * S.D * 4)))
    tc = max(1, REF_ACT_BYTES // (S.topk * max(S.D, n1) * 4))
    out1 = torch.zeros((M, S.topk, S.I), dtype=torch.float32, device=hidden.device)
    b1 = bias[0] if bias else None
    b2 = bias[1] if bias else None
    for t0 in range(0, M, tc):
        t1 = min(M, t0 + tc)
        a1s_c = (
            a1s[t0:t1]
            if (a1s is not None and a1s.dim() > 0 and a1s.shape[0] == M)
            else a1s
        )
        for e0 in range(0, E, ec):
            e1 = min(E, e0 + ec)
            sel = (ids[t0:t1] >= e0) & (ids[t0:t1] < e1)
            if not bool(sel.any()):
                continue
            ids_c = torch.where(sel, ids[t0:t1] - e0, torch.full_like(ids[t0:t1], -1))
            part = torch_moe_stage1(
                a1q[t0:t1],
                _slice_experts(ref_w["w1"], E, e0, e1),
                _slice_experts(ref_w["w2"], E, e0, e1),
                topk_w[t0:t1],
                ids_c,
                dtype=torch.float32,
                activation=S.act,
                quant_type=qT,
                a1_scale=a1s_c,
                w1_scale=_slice_experts(ref_w["w1_scale"], E, e0, e1),
                w1_bias=_slice_experts(b1, E, e0, e1) if b1 is not None else None,
                doweight=S.doweight,
                swiglu_limit=S.swiglu_limit,
                situ_beta=1.0 if S.beta is None else float(S.beta),
                situ_linear_beta=1.0 if S.linear_beta is None else float(S.linear_beta),
            )
            out1[t0:t1] = torch.where(sel[..., None], part, out1[t0:t1])
            del part
    if not hp:
        out1 = out1.to(S.dtype)

    if qT == aiter.QuantType.per_1x128:
        a2q, a2s = aiter.pertoken_quant(out1.view(M, -1, 128), quant_dtype=AQ)
        a2s = a2s.view(M, S.topk, -1)
    elif qT == aiter.QuantType.per_1x32 and AQ == dtypes.fp8 and WQ == dtypes.fp4x2:
        a2q, a2s = per_1x32_f8_scale_f8_quant(
            out1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    elif (
        (
            qT == aiter.QuantType.per_1x32
            and AQ in (dtypes.bf16, dtypes.fp16, dtypes.fp8)
            and WQ == dtypes.fp4x2
        )
        or is_mxfp8
        or (qT == aiter.QuantType.per_1x32 and WQ == dtypes.i4x2)
    ):
        a2q, a2s = out1, None
    else:
        a2q, a2s = torch_quant(out1, quant_dtype=AQ)
    a2q = a2q.view(M, S.topk, -1)
    del out1

    res = torch.zeros((M, S.D), dtype=torch.float32, device=hidden.device)
    ec2 = max(1, min(E, REF_FP32_BYTES // (S.D * S.I * 4)))
    tc2 = max(1, REF_ACT_BYTES // (S.topk * max(S.D, S.I) * 4))
    for t0 in range(0, M, tc2):
        t1 = min(M, t0 + tc2)
        if a2s is None:
            a2s_c = None
        elif a2s.dim() == 1 or a2s.shape[0] != M * S.topk and a2s.shape[0] != M:
            a2s_c = a2s
        else:
            a2s_c = a2s[t0:t1] if a2s.shape[0] == M else a2s[t0 * S.topk : t1 * S.topk]
        for e0 in range(0, E, ec2):
            e1 = min(E, e0 + ec2)
            sel = (ids[t0:t1] >= e0) & (ids[t0:t1] < e1)
            if not bool(sel.any()):
                continue
            ids_c = torch.where(sel, ids[t0:t1] - e0, torch.full_like(ids[t0:t1], -1))
            part = torch_moe_stage2(
                a2q[t0:t1],
                _slice_experts(ref_w["w1"], E, e0, e1),
                _slice_experts(ref_w["w2"], E, e0, e1),
                topk_w[t0:t1],
                ids_c,
                dtype=torch.float32,
                quant_type=qT,
                w2_scale=_slice_experts(ref_w["w2_scale"], E, e0, e1),
                a2_scale=a2s_c,
                w2_bias=_slice_experts(b2, E, e0, e1) if b2 is not None else None,
                doweight=not S.doweight,
            )
            res[t0:t1] += part
            del part
    del w_el
    return res if hp else res.to(S.dtype)


def fused_kwargs(S, aiter_w, bias, expert_mask):
    kw = {
        "w1_scale": aiter_w["w1_scale"],
        "w2_scale": aiter_w["w2_scale"],
        "quant_type": S.qtype,
        "activation": S.act,
        "doweight_stage1": S.doweight,
        "intermediate_pad": S.intermediate_pad,
        "hidden_pad": S.hidden_pad,
        "bias1": None if bias is None else bias[0].to(torch.float32),
        "bias2": None if bias is None else bias[1].to(torch.float32),
        "swiglu_limit": S.swiglu_limit,
        "beta": S.beta,
        "linear_beta": S.linear_beta,
        "gate_mode": S.gate_mode,
    }
    if expert_mask is not None:
        kw["expert_mask"] = expert_mask
    qa = act_dtype_override(S)
    if qa is not None:
        kw["quant_dtype_a"] = qa
    return kw


def act_dtype_override(S):
    """quant_dtype_a for MX rows labelled a4w4/a8w4.

    aiter resolves some of these rows to bf16 activations (Situv2 under the default env,
    INTERLEAVE/Swiglu a8w4 and separated Swiglu a4w4 below AITER_BF16_FP8_MOE_BOUND), which
    runs a16w4 instead of the labelled family. vLLM and SGLang set the activation dtype
    per call (SGLang moe_runner/aiter.py:318-327 ``quant_dtype_a=fp8``; vLLM
    rocm_aiter_moe.py:341-347 ``quant_dtype_a`` for DSv4.1 a4w4), so the harness does the
    same: the row's own activation dtype is passed explicitly (convention: per-call
    ``quant_dtype_a``, not the AITER_SITUV2_A4W4/A8W4 env). Rows labelled a16w4 are left
    alone.
    """
    from aiter import dtypes

    if (
        S.qtype.name == "per_1x32"
        and S.wq == dtypes.fp4x2
        and S.aq in (dtypes.fp4x2, dtypes.fp8)
    ):
        return S.aq
    return None


def flops(S, M):
    return 2.0 * M * S.topk * S.D * S.I * (3 if S.g1u1 else 2)


def weight_bytes_touched(S, aiter_w, topk_ids, expert_mask, M):
    """Bytes of w1+w2 (+scales) of the experts that received tokens."""
    ids = topk_ids[:M].long().flatten()
    if expert_mask is not None:
        local = expert_mask.cumsum(0, dtype=torch.int32) - 1
        local[expert_mask == 0] = -1
        ids = local[ids].long()
        ids = ids[ids >= 0]
    active = int(torch.unique(ids).numel())
    per = 0
    E = S.E_local
    for k in ("w1", "w2", "w1_scale", "w2_scale"):
        t = aiter_w.get(k)
        if t is not None:
            per += t.numel() * t.element_size() / E
    return active * per


def fingerprint(out, n_valid=None):
    """Compact output fingerprint: checksum of the bits plus a few sampled rows."""
    import hashlib

    o = out if n_valid is None else out[:n_valid]
    bits = o.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    M = o.shape[0]
    rows = sorted({0, M // 3, (2 * M) // 3, M - 1})
    sample = [
        [round(float(x), 4) for x in o[r, :8].float().cpu().tolist()] for r in rows
    ]
    f = o.float()
    return {
        "sha": hashlib.sha1(bits).hexdigest()[:16],
        "l2": float(f.norm()),
        "absmean": float(f.abs().mean()),
        "rows": rows,
        "sample": sample,
    }


def calc_diff(x, y):
    """test_moe_2stage.calc_diff: 1 - 2<x,y>/(|x|^2+|y|^2) in fp64."""
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if float(denom) == 0.0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def check_correctness(out, ref, sentinel_rows_ok=None):
    """Metrics and pass/fail with aiter's own rule for this path.

    Reference and tolerance: op_tests/test_moe_2stage.py::test_fmoe
    (checkAllclose rtol=atol=1e-2 with tol_err_ratio 0.05, logits_diff =
    calc_diff, fail when err != 0 and logits_diff > 0.01, or any NaN).
    """
    o, r = out.float(), ref.float()
    nan = int(torch.isnan(o).sum())
    inf = int(torch.isinf(o).sum())
    close = torch.isclose(r, o, rtol=1e-2, atol=1e-2)
    err = float((~close).float().mean()) if close.numel() else 0.0
    logits = calc_diff(r, o) if not (nan or inf) else float("inf")
    empty = (r.abs().amax(dim=-1) == 0) if r.numel() else None
    empty_ok = True
    if empty is not None and bool(empty.any()):
        empty_ok = bool((o[empty].abs().amax() <= 1e-2).item())
    passed = not nan and not inf and not (err != 0 and logits > 0.01) and empty_ok
    return {
        "err_ratio": err,
        "logits_diff": logits,
        "nan": nan,
        "inf": inf,
        "empty_rows_ok": empty_ok,
        "sentinel_ok": sentinel_rows_ok,
        "ref_pass": bool(passed),
        "ref_warn": bool(err >= 0.05 or logits > 1e-3),
    }


def free_vram_bytes():
    free, _ = torch.cuda.mem_get_info()
    return free


def span(x):
    return math.ceil(x)


ERR_FLOOR = 1e-2  # relative errors use max(|ref|, ERR_FLOOR * max|ref|) as denominator


def error_metrics(out, ref, rows_per_chunk=None):
    """Error of ``out`` against the high-precision reference, accumulated in fp64.

    max_abs, max_rel (denominator floored at 1% of max|ref|), mean_rel
    (sum|d| / sum|ref|), rms_rel (|d|_2 / |ref|_2) and err_cos (1 - cosine).
    """
    n = out.shape[0]
    cols = max(1, out.shape[-1])
    step = rows_per_chunk or max(1, (1 << 24) // cols)
    ref_absmax = float(ref.abs().max()) if ref.numel() else 0.0
    floor = max(ERR_FLOOR * ref_absmax, 1e-30)
    max_abs = max_rel = sum_d = sum_r = sq_d = sq_r = dot = sq_o = 0.0
    for i in range(0, n, step):
        o = out[i : i + step].double()
        r = ref[i : i + step].double()
        d = (o - r).abs()
        max_abs = max(max_abs, float(d.max()))
        max_rel = max(max_rel, float((d / r.abs().clamp_min(floor)).max()))
        sum_d += float(d.sum())
        sum_r += float(r.abs().sum())
        sq_d += float(d.pow(2).sum())
        sq_r += float(r.pow(2).sum())
        sq_o += float(o.pow(2).sum())
        dot += float((o * r).sum())
    tiny = 1e-300
    return {
        "err_max_abs": max_abs,
        "err_max_rel": max_rel,
        "err_mean_rel": sum_d / max(sum_r, tiny),
        "err_rms_rel": (sq_d**0.5) / max(sq_r**0.5, tiny),
        "err_cos": 1.0 - dot / max((sq_o**0.5) * (sq_r**0.5), tiny),
    }
