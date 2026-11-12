# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Other candidate sources: pluggable slots, the ``new`` hook, and recorded skips.

Backends listed in the GM-123 bar definition that are not wired into this harness yet
report ``skip_reason`` so each case's candidates file says exactly which sources were
not measured and why; adding one means implementing ``enumerate`` and ``Candidate.call``
for it. Multi-GPU paths are skipped by definition.
"""

import importlib
import inspect
import os

from . import Candidate, Source, register


class _Skipped(Source):
    reason = ""

    def skip_reason(self, ctx):
        return self.reason


@register
class TritonMoe(_Skipped):
    name = "triton_moe"
    reason = (
        "not implemented: aiter/ops/triton/moe/* are grouped GEMMs (a16w4, a4w4, a8w4, "
        "a8w8 fnuz/MX, a8w8_blockscale, int8_smoothquant); an end-to-end wrapper "
        "(routing from topk ids, gemm1+swiglu, gemm2, reduce) is still to be written"
    )


@register
class SonicMoe(_Skipped):
    name = "sonicmoe"
    reason = "not implemented: SonicMoE entry (aiter/ops/triton/sonicmoe) not wired"


@register
class FlydslStandalone(_Skipped):
    name = "flydsl_standalone"
    reason = (
        "not implemented: standalone FlyDSL entries (flydsl_a16w4_moe) not wired; "
        "their kernels are already reached through the fused_moe forced configs"
    )


@register
class AsmMoeLegacy(_Skipped):
    name = "asm_moe_legacy"
    reason = (
        "not implemented: fused_moe_bf16_asm.asm_moe needs unshuffled per-token scales and "
        "its own sorting call; its kernels (fmoe_g1u1 1-stage) are covered by the "
        "fused_moe forced 1-stage candidates"
    )


@register
class MultiGpu(_Skipped):
    name = "multigpu"
    reason = "skipped: multi-GPU paths (MegaMoE, comm-fused MoE) are out of scope"


PREPARE_KWARGS = (
    "w1_scale",
    "w2_scale",
    "quant_type",
    "activation",
    "doweight_stage1",
    "bias1",
    "bias2",
    "gate_mode",
    "swiglu_limit",
    "beta",
    "linear_beta",
    "hidden_pad",
    "intermediate_pad",
)


def split_kwargs(kwargs, mod=None):
    """fused_moe kwargs -> (model-constant prepare kwargs, per-call kwargs).

    The split is the API module's own ``PREPARE_KWARGS`` (docs/aiter-moe-api.md section 2;
    ``aiter.moe.PREPARE_KWARGS``), with the list above as the fallback for toys.
    """
    names = set(getattr(mod, "PREPARE_KWARGS", None) or PREPARE_KWARGS)
    prep = {k: v for k, v in kwargs.items() if k in names}
    call = {k: v for k, v in kwargs.items() if k not in names}
    return prep, call


class NewCandidate(Candidate):
    """The new entry point, timed over the same region as every other candidate.

    ``MOE_BENCH_NEW_ENTRY=pkg.module`` names a module with the core API:

        weights = prepare(w1, w2, *, w1_scale, w2_scale, quant_type, activation, dtype,
                          doweight_stage1, bias1, bias2, gate_mode, swiglu_limit, ...,
                          num_global_experts=None, routes=None)
        out = forward(hidden_states, weights, topk_weight, topk_ids, *, expert_mask,
                      num_local_tokens, output, ...)
        explain(weights, M) -> {"route": kernel_id | "aiter", ...}

    ``prepare`` runs once per weight group, untimed, on the same aiter-layout tensors
    ``fused_moe`` gets in dispatch mode. ``MOE_BENCH_NEW_ROUTES`` (``--new-routes``)
    is passed as ``prepare(routes=...)``: a route table path or a kernel id. A case
    served by route ``aiter`` is reported with new_kernel_id ``aiter`` and never counts
    as won. A module whose ``prepare`` takes ``ref_weights`` uses the earlier provisional
    interface, and ``pkg.module:fn`` a function with fused_moe's own signature.
    """

    source = "new"
    entry = "MOE_BENCH_NEW_ENTRY"

    def __init__(self, mod, fn=None):
        kid = getattr(mod, "kernel_id", None) or getattr(fn or mod, "__name__", "new")
        super().__init__(f"new|{kid}", str(kid))
        self.mod, self.fn = mod, fn
        self.kernel_id = ""
        self.provisional = (
            fn is None and "ref_weights" in inspect.signature(mod.prepare).parameters
        )

    def prepare(self, ctx):
        if self.fn is not None:
            return
        S = ctx.S
        if self.provisional:
            if "new" not in ctx.group_cache:
                meta = {
                    "model_dim": S.D,
                    "inter_dim": S.I,
                    "experts": S.E_local,
                    "topk": S.topk,
                    "dtype": S.dtype,
                    "quant_type": S.qtype,
                    "activation": S.act,
                    "g1u1": S.g1u1,
                    "gate_mode": S.gate_mode,
                }
                ctx.group_cache["new"] = self.mod.prepare(
                    ctx.aw["w1"],
                    ctx.aw["w2"],
                    ctx.aw.get("w1_scale"),
                    ctx.aw.get("w2_scale"),
                    ref_weights=ctx.ref_weights,
                    meta=meta,
                )
            return
        if "new" not in ctx.group_cache:
            prep, _ = split_kwargs(ctx.kwargs, self.mod)
            kw = dict(prep, dtype=S.dtype, topk=S.topk)
            if S.is_ep:
                kw["num_global_experts"] = S.E
            routes = os.environ.get("MOE_BENCH_NEW_ROUTES")
            if routes:
                kw["routes"] = routes
                # measuring a named kernel: a decline must raise, not fall back
                os.environ.setdefault("AITER_MOE_STRICT", "1")
            ctx.group_cache["new"] = self.mod.prepare(ctx.aw["w1"], ctx.aw["w2"], **kw)
        explain = getattr(self.mod, "explain", None)
        if explain is not None:
            try:
                info = explain(ctx.group_cache["new"], S.M, topk=S.topk)
            except TypeError:
                info = explain(ctx.group_cache["new"], S.M)
            self.kernel_id = str(info.get("route", ""))
            self.kernels = self.kernel_id

    def call(self, ctx):
        kw = dict(ctx.kwargs)
        if self.fn is not None:

            def run():
                return self.fn(
                    ctx.hidden_static,
                    ctx.aw["w1"],
                    ctx.aw["w2"],
                    ctx.tw,
                    ctx.ti,
                    output=ctx.outbuf,
                    **kw,
                )

            return run
        handle = ctx.group_cache["new"]
        if self.provisional:

            def run():
                return self.mod.forward(
                    handle, ctx.hidden_static, ctx.tw, ctx.ti, output=ctx.outbuf, **kw
                )

            return run
        _, call_kw = split_kwargs(kw, self.mod)

        def run():
            return self.mod.forward(
                ctx.hidden_static, handle, ctx.tw, ctx.ti, output=ctx.outbuf, **call_kw
            )

        return run


@register
class NewEntry(Source):
    """The new entry point, plugged in with ``MOE_BENCH_NEW_ENTRY``."""

    name = "new"

    def skip_reason(self, ctx):
        if not os.environ.get("MOE_BENCH_NEW_ENTRY"):
            return "no new entry point configured (set MOE_BENCH_NEW_ENTRY=pkg.mod)"
        return None

    def enumerate(self, ctx):
        spec = os.environ["MOE_BENCH_NEW_ENTRY"]
        modname, _, fn = spec.partition(":")
        mod = importlib.import_module(modname)
        return [NewCandidate(mod, getattr(mod, fn) if fn else None)]
