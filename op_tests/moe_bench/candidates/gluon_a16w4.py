# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Triton/Gluon a16w4 side API (``aiter.ops.triton.moe.moe_op_gemm_a16w4``) as a bar candidate.

It is not a ``fused_moe`` path: the whole call is its own top-k routing, gather, gemm1 with the
Swiglu epilogue, gemm2, scatter and combine, on its own weight copy. It is timed over the
region every candidate is timed over (bf16 hidden states in, final output out) with one
correction, and scored on the same accuracy check as the others.

Applies to a16w4 Swiglu cases on gfx950 only, E 128..512, D and I multiples of 256 (and
2*I a multiple of 32), no EP, no padding, SEPARATED gate/up. The gluon kernel needs the CDNA4
scale swizzle and K % 256 == 0 (otherwise aiter falls back to Triton on unswizzled scales),
and asserts on an expert map. Biases work in both GEMMs when the gate/up bias is interleaved
like the weights. E896 / topk16 never wins and is outside the range. It is enumerated at every
M and may lose (it wins from M >= 64).

Routing. ``routing()`` takes logits, ``fused_moe`` takes ids: the candidate crafts
``log(w)`` fp16 logits for the case's ids and weights (``-1e4`` elsewhere) and asserts the
ids ``routing()`` selects equal the case's, row by row, for every input the accuracy check
swaps in; a mismatch is a harness error, never a quiet pass.

Fairness. ``fused_moe`` is timed with its ids given; the side call includes its own top-k
(about 9.6 us warm, 13.3 us cold). The row's time is the whole call minus the top-k kernel
timed alone, in the same process, in the same mode (warm or cold, graph or eager) and
subtracted, so the candidate is not charged for work the others are handed. The raw time,
the top-k time and the tag stay on the row so a report shows it. The id-to-bitmatrix
prototype is not used.

Triton compiles a kernel the first time a shape runs. A leased (run-only) bar must not: the
candidate refuses a cache miss as a harness error (``harness-error``, the case's bar is
``bar_incomplete``), and ``python -m op_tests.moe_bench.triton_warm`` fills the persistent
``TRITON_CACHE_DIR`` in one short lease first.
"""

from . import Candidate, HarnessError, Source, register

NAME = "triton_gluon"
ENTRY = (
    "aiter.ops.triton.moe.moe_op_gemm_a16w4.moe_gemm_a16w4 (backend=gluon): routing, "
    "gather, gemm1+swiglu, gemm2, scatter, combine"
)
TAG = "gluon-whole-call: side API with its own routing, top-k kernel time subtracted"
KERNELS = "gluon:moe_gemm_a16w4 + gluon:moe_gemm_a16w4 (own routing and combine)"
CAND_ID = f"{NAME}|a16w4|whole-call"
ALPHA = 1.702  # aiter's swiglu() default, what every dispatch path uses
DEFAULT_LIMIT = 7.0  # swiglu() with no limit
E_RANGE = (128, 512)
GFX = "gfx950"
MASKED_LOGIT = -1e4
WEIGHT_SUM_TOL = 1e-3


class GluonHarnessError(HarnessError):
    """Routing ids differ, the weights do not fit, the top-k subtraction is meaningless."""


def is_a16w4_swiglu(S):
    """The family the gluon a16w4 API is a candidate for (before its own limits)."""
    from aiter import ActivationType, QuantType, dtypes

    return (
        S.kind == "fused"
        and S.qtype == QuantType.per_1x32
        and S.aq == dtypes.bf16
        and S.wq == dtypes.fp4x2
        and S.act == ActivationType.Swiglu
    )


def unsupported(S):
    """Why the gluon side API cannot compute this a16w4 Swiglu case, or None."""
    from aiter import dtypes
    from aiter.ops.flydsl.moe_common import GateMode

    why = []
    if S.M < 1:
        why.append("no tokens (M=0)")
    if str(S.gate_mode) != GateMode.SEPARATED.value:
        why.append(f"gate_mode {S.gate_mode} (the weights here are SEPARATED gate/up)")
    if S.dtype != dtypes.bf16:
        why.append(f"output dtype {S.dtype} (bf16 only)")
    if not E_RANGE[0] <= S.E <= E_RANGE[1]:
        why.append(f"E={S.E} outside {E_RANGE[0]}..{E_RANGE[1]}")
    if S.D % 256 or S.I % 256 or (2 * S.I) % 32:
        why.append(
            f"D={S.D}, I={S.I}: gluon needs K % 256 == 0 (D and I multiples of 256, "
            "N % 32); aiter would fall back to Triton on unswizzled scales"
        )
    if S.is_ep or S.E_local != S.E:
        why.append("EP: the gluon backend asserts on expert_map")
    if S.num_local_frac:
        why.append("num_local_tokens: not supported")
    if S.hidden_pad or S.intermediate_pad:
        why.append("hidden/intermediate padding: not supported")
    if S.doweight:
        why.append("doweight_stage1: the side API weights in stage 2")
    if S.beta is not None or S.linear_beta is not None:
        why.append("situ beta: not Swiglu")
    return "; ".join(why) or None


def swiglu_limit(S):
    return float(S.swiglu_limit) if S.swiglu_limit else DEFAULT_LIMIT


def _interleave_rows(t, inter):
    """[E, 2I, ...] gate rows then up rows -> gate/up alternating along dim 1."""
    out = t.new_empty(t.shape)
    out[:, 0::2] = t[:, :inter]
    out[:, 1::2] = t[:, inter:]
    return out


def relayout_weights(rw, S, bias=None, arch=GFX):
    """The case's quantized weights (aiter's unshuffled fp4x2 bytes and e8m0 scales, the
    ones ``fused_moe`` and the references use) in the gluon kernel's layout: gate/up rows
    interleaved, K-packed bytes viewed [E, K/2, N] column-major, scales swizzled for CDNA4.

    The same fp4 values and scales, so dispatch, the references and this path read
    identical weights; only the byte layout differs (both pack the even element in the low
    nibble), which makes the quantizer used irrelevant. Returns a dict (W1, W2, S1, S2,
    L1, L2, b1, b2).
    """
    import torch

    from aiter.ops.triton.utils.shuffle import shuffle_scale_moe

    E, D, inter = S.E, S.D, S.I

    def u8(t):
        return t.view(torch.uint8)

    w1 = _interleave_rows(u8(rw["w1"]).reshape(E, 2 * inter, D // 2), inter)
    s1 = _interleave_rows(u8(rw["w1_scale"]).reshape(E, 2 * inter, D // 32), inter)
    w2 = u8(rw["w2"]).reshape(E, D, inter // 2).contiguous()
    s2 = u8(rw["w2_scale"]).reshape(E, D, inter // 32).contiguous()
    out = {"W1": w1.transpose(1, 2), "W2": w2.transpose(1, 2)}
    out["S1"], out["L1"] = shuffle_scale_moe(
        s1.transpose(1, 2),
        arch=arch,
        preshuffle_factor=32,
        scale_kwidth=8,
        return_layout=True,
    )
    out["S2"], out["L2"] = shuffle_scale_moe(
        s2.transpose(1, 2),
        arch=arch,
        preshuffle_factor=32,
        scale_kwidth=8,
        return_layout=True,
    )
    out["b1"] = out["b2"] = None
    if bias is not None:
        b1 = bias[0].to(torch.float32)
        out["b1"] = _interleave_rows(b1.reshape(E, 2 * inter), inter).contiguous()
        out["b2"] = bias[1].to(torch.float32).reshape(E, D).contiguous()
    return out


def crafted_logits(tw, ti, n_experts, out=None):
    """fp16 logits whose top-k softmax is ``tw`` on the experts ``ti`` (-1e4 elsewhere)."""
    import torch

    if out is None:
        out = torch.empty(
            (tw.shape[0], n_experts), dtype=torch.float16, device=tw.device
        )
    out.fill_(MASKED_LOGIT)
    out.scatter_(1, ti.long(), torch.log(tw.float()).to(torch.float16))
    return out


def check_routed_ids(selected, ti):
    """Raise GluonHarnessError unless every row's selected experts equal ``ti``'s."""
    import torch

    a = torch.sort(selected.to(torch.int32), dim=1).values
    b = torch.sort(ti.to(torch.int32), dim=1).values
    if a.shape != b.shape or not bool((a == b).all()):
        bad = int((a != b).any(dim=1).sum()) if a.shape == b.shape else a.shape[0]
        raise GluonHarnessError(
            f"routing() selected other experts than the case's topk_ids for {bad} of "
            f"{b.shape[0]} tokens (crafted log(w) logits): the candidate is not scored"
        )


class GluonA16W4(Candidate):
    source = NAME
    entry = ENTRY
    group = "gluon"
    tag = TAG

    # bf16 activations, bf16 intermediate, per-token rows summed in fp32 by the combine
    @property
    def declared_numerics(self):
        return {"act_quant": "none", "inter": "bf16", "combine": "bf16_rows_fp32_sum"}

    def __init__(self):
        super().__init__(CAND_ID, KERNELS, layout="gluon_a16w4")
        self._armed = False
        self._k = 0
        self._tw = self._ti = None
        self._logits = None
        self._topk_timer = None
        self._w = None
        self.measure_host = False

    def _weights(self, ctx):
        key = ("gluon_a16w4", id(ctx.aw))
        if key not in ctx.group_cache:
            from .. import workload

            need = workload.group_weight_bytes(ctx.S) * 1.2
            free = workload.free_vram_bytes()
            if need > free:
                raise GluonHarnessError(
                    f"the gluon weight copy needs about {need / 2**30:.1f} GiB, "
                    f"{free / 2**30:.1f} GiB free"
                )
            if ctx.ref_weights is None:
                raise GluonHarnessError(
                    "the quantized reference weights are not in ctx"
                )
            ctx.group_cache[key] = relayout_weights(ctx.ref_weights, ctx.S, ctx.bias)
        return ctx.group_cache[key]

    def prepare(self, ctx):
        from .. import prebuild

        if prebuild.triton_run_only():
            if not prebuild.forbid_triton_compiles():
                raise GluonHarnessError(
                    "cannot guard Triton compiles in this triton build "
                    "(no ASTSource.make_ir): a leased run would compile on the GPU"
                )
            self._armed = True
        self._w = self._weights(ctx)
        self._k = ctx.ti.shape[1]
        self._tw = self._ti = None
        self._logits = None
        self._sync_inputs(ctx)

    def release(self, ctx):
        if self._armed:
            from .. import prebuild

            prebuild.restore_triton_compiles()
            self._armed = False
        self._topk_timer = None

    def _sync_inputs(self, ctx):
        """Refresh the crafted logits when the accuracy check swapped the case's routing
        (eager only: a captured graph replays on the buffers it was captured with)."""
        import torch

        if ctx.tw is self._tw and ctx.ti is self._ti:
            return
        tw = ctx.tw
        n = ctx.S.E
        if self._logits is None or self._logits.shape[0] != tw.shape[0]:
            self._logits = crafted_logits(tw, ctx.ti, n)
        else:
            crafted_logits(tw, ctx.ti, n, out=self._logits)
        err = float((tw.float().sum(dim=1) - 1.0).abs().max())
        if not err <= WEIGHT_SUM_TOL:
            raise GluonHarnessError(
                f"the case's routing weights do not sum to 1 (off by {err:.3g}): "
                "log(w) logits cannot reproduce them"
            )
        rd = self._route()[0]
        torch.cuda.synchronize()
        check_routed_ids(rd.topk_ids, ctx.ti)
        self._tw, self._ti = ctx.tw, ctx.ti

    def _route(self):
        from aiter.ops.triton.moe.moe_routing.routing import routing

        return routing(self._logits, self._k)

    def call(self, ctx):
        from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4

        w = self._w
        limit = swiglu_limit(ctx.S)
        out_dtype = ctx.S.dtype

        def run():
            self._sync_inputs(ctx)
            rd, gather, scatter = self._route()
            y1 = moe_gemm_a16w4(
                ctx.hidden_static,
                w["W1"],
                None,
                w["S1"],
                None,
                None,
                w["b1"],
                rd,
                gather,
                None,
                None,
                w["L1"],
                out_dtype,
                True,
                alpha=ALPHA,
                limit=limit,
                backend="gluon",
            )
            return moe_gemm_a16w4(
                y1,
                w["W2"],
                None,
                w["S2"],
                None,
                None,
                w["b2"],
                rd,
                None,
                scatter,
                rd.gate_scal,
                w["L2"],
                out_dtype,
                False,
                backend="gluon",
            )

        return run

    def topk_call(self):
        from aiter.ops.triton.moe.moe_routing.topk import topk

        logits, k = self._logits, self._k

        def run():
            return topk(logits, k, apply_softmax=True, HIST_BLOCK_M=32)

        return run

    def adjust_timing(self, st, cold, quick):
        """The whole call's stats minus the top-k kernel timed alone in this mode."""
        from ..timing import Timer

        if self._topk_timer is None:
            self._topk_timer = Timer(self.topk_call())
            self._topk_timer.prepare_graph()
        sub = self._topk_timer.time(cold=cold, quick=quick)["median_us"]
        return subtract_topk(st, sub)


def subtract_topk(st, topk_us):
    """Stats of the whole side call with the top-k kernel's time taken off, keeping the raw
    time and the subtrahend on the row."""
    net = st["median_us"] - topk_us
    if not net > 0:
        raise GluonHarnessError(
            f"the top-k kernel alone ({topk_us:.1f} us) is not shorter than the whole "
            f"call ({st['median_us']:.1f} us): the subtraction is meaningless"
        )
    out = dict(st)
    for k in ("median_us", "p10_us", "p90_us"):
        if k in st:
            out[k] = max(st[k] - topk_us, 0.0)
    out["raw_us"] = st["median_us"]
    out["topk_us"] = topk_us
    return out


@register
class TritonGluonA16W4(Source):
    name = NAME

    def _gfx(self, ctx):
        return (getattr(ctx, "dispatch_meta", None) or {}).get("gfx", "")

    def skip_reason(self, ctx):
        """Only a case this API could compute on this GPU and does not meet its limits is
        reported (the others are not in its family: no row, no noise)."""
        if self._gfx(ctx) != GFX or not is_a16w4_swiglu(ctx.S):
            return None
        why = unsupported(ctx.S)
        return f"gluon a16w4 side API not applicable: {why}" if why else None

    def enumerate(self, ctx):
        if self._gfx(ctx) != GFX or not is_a16w4_swiglu(ctx.S):
            return []
        if unsupported(ctx.S):
            return []
        return [GluonA16W4()]
