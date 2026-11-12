# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""prepare(), forward() and explain(): the MoE entry point."""

import os
from collections.abc import Iterable

import torch

import aiter.fused_moe as _aiter_fused_moe
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import GateMode

from .gpu import gpu_name
from .kernel import MoeCall
from .log import log_once
from .routes import resolve_table
from .spec import MoeSpec
from .weights import MoeWeights

# fused_moe keywords that belong to the layer: given once to prepare().
PREPARE_KWARGS = (
    "w1_scale",
    "w2_scale",
    "activation",
    "quant_type",
    "doweight_stage1",
    "dtype",
    "block_size_M",
    "moe_sorting_dispatch_policy",
    "hidden_pad",
    "intermediate_pad",
    "bias1",
    "bias2",
    "splitk",
    "swiglu_limit",
    "beta",
    "linear_beta",
    "gate_mode",
    "shared_w1",
    "shared_w2",
    "shared_w1_scale",
    "shared_w2_scale",
    "shared_expert_id",
    "quant_type_a",
    "quant_dtype_a",
    "quant_dtype_a2",
    "clamp_shared",
    "herd_topk",
    "herd_routed_scaling_factor",
)
# fused_moe keywords that change per call: given to forward().
CALL_KWARGS = (
    "expert_mask",
    "num_local_tokens",
    "output",
    "a1_scale",
    "a2_scale",
    "stage2_scatter",
)
# Prepare-level keywords only aiter's path implements (FHMoE shared experts,
# HERD routing, forced activation dtypes): set to a non-default value, they
# keep the whole layer on aiter's path. block_size_M, splitk and
# moe_sorting_dispatch_policy are tuning hints a new kernel may ignore.
AITER_ONLY_KWARGS = {
    "shared_w1": None,
    "shared_w2": None,
    "shared_w1_scale": None,
    "shared_w2_scale": None,
    "shared_expert_id": -1,
    "quant_type_a": None,
    "quant_dtype_a": None,
    "quant_dtype_a2": None,
    "clamp_shared": True,
    "herd_topk": 0,
    "herd_routed_scaling_factor": 1.0,
}


def disabled() -> bool:
    """``AITER_MOE_DISABLE=1`` is the kill switch: every call takes aiter's path."""
    return os.getenv("AITER_MOE_DISABLE", "0") == "1"


def strict() -> bool:
    """``AITER_MOE_STRICT=1`` raises kernel errors instead of falling back
    (tests, climbs and the gate, where a silent fallback would hide a bug)."""
    return os.getenv("AITER_MOE_STRICT", "0") == "1"


def prepare(
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    num_global_experts: int | None = None,
    routes=None,
    topk: int | None = None,
    warmup_m: Iterable[int] = (),
    **fused_moe_kwargs,
) -> MoeWeights:
    """Prepare one MoE layer, once, at weight load.

    ``w1``, ``w2`` and every keyword in ``PREPARE_KWARGS`` are exactly what the
    caller passes to ``fused_moe`` today (same layout, shuffle and
    ``is_shuffled`` tag); the fallback reuses them with no copy. Unset keywords
    keep fused_moe's defaults.

    ``num_global_experts`` is the size of ``expert_mask`` under expert
    parallelism. ``routes`` overrides the route table (tests and kernel
    bring-up; see ``RouteTable.override``). ``topk`` and ``warmup_m``
    optionally resolve and compile the routes for those token counts now, so
    errors surface before graph capture.
    """
    unknown = set(fused_moe_kwargs) - set(PREPARE_KWARGS)
    if unknown:
        raise TypeError(
            f"prepare() got keywords that are not prepare-level: {sorted(unknown)}"
            f"; call-level ones go to forward(): {CALL_KWARGS}"
        )
    kw = fused_moe_kwargs
    spec = MoeSpec.from_weights(
        w1,
        w2,
        activation=kw.get("activation", ActivationType.Silu),
        quant_type=kw.get("quant_type", QuantType.No),
        doweight_stage1=kw.get("doweight_stage1", False),
        dtype=kw.get("dtype"),
        gate_mode=kw.get("gate_mode") or GateMode.SEPARATED.value,
        has_bias=kw.get("bias1") is not None or kw.get("bias2") is not None,
        swiglu_limit=kw.get("swiglu_limit"),
        hidden_pad=kw.get("hidden_pad", 0),
        intermediate_pad=kw.get("intermediate_pad", 0),
        num_global_experts=num_global_experts,
    )
    gpu = gpu_name()
    weights = MoeWeights(
        spec=spec,
        w1=w1,
        w2=w2,
        fused_moe_kwargs=dict(kw),
        gpu=gpu,
        table=resolve_table(gpu, routes),
        topk=topk,
    )
    extra = sorted(
        k
        for k, d in AITER_ONLY_KWARGS.items()
        if k in kw and (kw[k] is not None if d is None else kw[k] != d)
    )
    if extra:
        weights.failed.add("*")
        log_once(f"aiter.moe: {extra} set; only aiter's path implements them")
    if topk is not None and not disabled():
        for m in warmup_m:
            _warmup(weights, int(m), int(topk))
    return weights


def _warmup(weights: MoeWeights, m: int, topk: int) -> None:
    try:
        res = weights.resolve(m, topk)
    except Exception as exc:
        if strict():
            raise
        log_once(f"aiter.moe: route resolution failed at M={m}: {exc!r}")
        return
    if res.kernel is None:
        return
    try:
        res.kernel.warmup(weights, m, topk, res.route.config)
    except Exception as exc:
        if strict():
            raise
        weights.mark_failed(res.kernel, exc, f"warmup M={m}")


def explain(
    weights: MoeWeights,
    M: int,
    *,
    topk: int | None = None,
    hidden_dtype: torch.dtype = dtypes.bf16,
    has_a1_scale: bool = False,
) -> dict:
    """Which path serves a call of ``M`` tokens: ``{route, family, gpu, reason}``.

    ``route`` is a kernel id or ``"aiter"``. ``topk`` defaults to the one given
    to prepare or seen by the last forward.
    """
    topk = topk if topk is not None else weights.topk
    out = {"route": "aiter", "family": "", "gpu": weights.gpu, "reason": ""}
    if disabled():
        out["reason"] = "AITER_MOE_DISABLE=1"
    elif "*" in weights.failed:
        out["reason"] = "prepare-level arguments only aiter implements"
    elif topk is None:
        out["reason"] = "topk unknown: pass topk= or call forward first"
    else:
        res = weights.resolve(M, topk, hidden_dtype, has_a1_scale)
        out.update(family=res.family, reason=res.reason)
        if res.kernel is not None:
            out["route"] = res.kernel.id
            out["config"] = res.route.config
    return out


def forward(
    hidden_states: torch.Tensor,
    weights: MoeWeights,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    expert_mask: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    **call_kwargs,
) -> torch.Tensor:
    """``fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, ...)`` with
    ``weights`` in place of ``w1, w2``: same semantics, same result.

    Raises only where fused_moe itself would. Graph-capture safe: the route is
    chosen on the host from shapes and dtypes; nothing here reads device memory.
    """
    unknown = set(call_kwargs) - set(CALL_KWARGS)
    if unknown:
        raise TypeError(
            f"forward() got keywords that are not call-level: {sorted(unknown)}"
            f"; prepare-level ones go to prepare(): {PREPARE_KWARGS}"
        )
    m = hidden_states.shape[0]
    topk = topk_ids.shape[1] if topk_ids.dim() == 2 else -1
    weights.topk = topk
    call = MoeCall(
        hidden_states=hidden_states,
        topk_weight=topk_weight,
        topk_ids=topk_ids,
        weights=weights,
        m=m,
        topk=topk,
        expert_mask=expert_mask,
        num_local_tokens=num_local_tokens,
        output=output,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        extra=call_kwargs,
    )
    if disabled() or "*" in weights.failed or topk < 1:
        return fallback(call)
    res = weights.resolve(m, topk, hidden_states.dtype, a1_scale is not None)
    kernel = res.kernel
    if kernel is None:
        return fallback(call)
    reason = kernel.supports(weights.spec, m, topk, call)
    if reason:
        log_once(f"aiter.moe: {kernel.id} declines a call at M={m}: {reason}")
        return fallback(call)
    call.config = res.route.config
    try:
        return kernel(call)
    except Exception as exc:
        if strict():
            raise
        weights.mark_failed(kernel, exc, f"M={m}")
        return fallback(call)


def fallback(call: MoeCall) -> torch.Tensor:
    """aiter's own path, called with the caller's own tensors and arguments."""
    kwargs = dict(call.weights.fused_moe_kwargs)
    for name in ("expert_mask", "num_local_tokens", "output", "a1_scale", "a2_scale"):
        value = getattr(call, name)
        if value is not None:
            kwargs[name] = value
    kwargs.update(call.extra)
    return _aiter_fused_moe.fused_moe(
        call.hidden_states,
        call.weights.w1,
        call.weights.w2,
        call.topk_weight,
        call.topk_ids,
        **kwargs,
    )
