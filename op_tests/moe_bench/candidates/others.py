# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Other candidate sources: pluggable slots, the ``new`` hook, and recorded skips.

Backends listed in the GM-123 bar definition that are not wired into this harness yet
report ``skip_reason`` so each case's candidates file says exactly which sources were
not measured and why; adding one means implementing ``enumerate`` and ``Candidate.call``
for it. Multi-GPU paths are skipped by definition.
"""

import importlib
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


class NewCandidate(Candidate):
    """The new entry point, timed over the same region as every other candidate.

    ``MOE_BENCH_NEW_ENTRY=pkg.module`` names a module with

        prepare(w1, w2, w1_scale, w2_scale, *, ref_weights, meta) -> handle
        forward(handle, hidden_states, topk_weight, topk_ids, **kwargs) -> output

    ``prepare`` runs once per weight group, untimed. ``forward`` is the timed call:
    ``kwargs`` are fused_moe's keyword arguments for the case (activation, quant_type,
    expert_mask, bias, swiglu_limit, output, ...). ``ref_weights`` are the unshuffled
    quantized weights and scales the reference uses. ``pkg.module:fn`` instead names a
    function with fused_moe's own signature. ``forward`` may raise NotImplementedError
    for cases it does not cover (recorded as unsupported).
    """

    source = "new"
    entry = "MOE_BENCH_NEW_ENTRY"

    def __init__(self, mod, fn=None):
        kid = getattr(mod, "kernel_id", None) or getattr(fn or mod, "__name__", "new")
        super().__init__(f"new|{kid}", str(kid))
        self.mod, self.fn = mod, fn

    def prepare(self, ctx):
        if self.fn is not None or "new" in ctx.group_cache:
            return
        S = ctx.S
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

        def run():
            return self.mod.forward(
                handle, ctx.hidden_static, ctx.tw, ctx.ti, output=ctx.outbuf, **kw
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
