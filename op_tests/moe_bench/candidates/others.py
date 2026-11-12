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
    source = "new"
    entry = "MOE_BENCH_NEW_ENTRY"

    def __init__(self, fn):
        super().__init__(
            f"new|{getattr(fn, 'kernel_id', fn.__name__)}",
            getattr(fn, "kernel_id", fn.__name__),
        )
        self.fn = fn

    def call(self, ctx):
        def run():
            return self.fn(
                ctx.hidden_static,
                ctx.aw["w1"],
                ctx.aw["w2"],
                ctx.tw,
                ctx.ti,
                output=ctx.outbuf,
                **ctx.kwargs,
            )

        return run


@register
class NewEntry(Source):
    """The new entry point. ``MOE_BENCH_NEW_ENTRY=pkg.module:function`` plugs it in.

    The function takes ``fused_moe``'s signature (hidden, w1, w2, topk_weight, topk_ids,
    keyword arguments), may carry a ``kernel_id`` attribute, and may raise
    ``NotImplementedError`` for cases it does not cover.
    """

    name = "new"

    def skip_reason(self, ctx):
        if not os.environ.get("MOE_BENCH_NEW_ENTRY"):
            return "no new entry point configured (set MOE_BENCH_NEW_ENTRY=pkg.mod:fn)"
        return None

    def enumerate(self, ctx):
        mod, _, fn = os.environ["MOE_BENCH_NEW_ENTRY"].partition(":")
        return [NewCandidate(getattr(importlib.import_module(mod), fn))]
