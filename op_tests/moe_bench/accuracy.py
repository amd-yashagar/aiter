# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy of every timed candidate against two fp64 references.

Source of truth: the "Accuracy" section of docs/aiter-moe-plan.md.

* ``ref_path`` reproduces the existing aiter path's quantization points: activation
  quant with aiter's own ``get_hip_quant`` (what ``fused_moe_2stages`` calls), stage-1
  output rounded to the output dtype and re-quantized with the same function, all other
  math in fp64 on the dequantized quantized weights. It measures implementation error.
* ``ref_exact`` keeps bf16 activations and dequantized weights with no activation or
  intermediate quantization, fp64 throughout. It exposes any extra or coarser
  quantization.

Both are computed by aiter's own ``torch_moe_stage1/2`` with their fp32 compute type
switched to fp64 for the call (``fp64_math``), on a deterministic sample of at most
256 rows, the same rows for every candidate; the case itself is never shrunk. For
per-tensor activation scales the intermediate scale needs the whole batch, so stage 1 runs
on every row to find it and stage 2 on the sample.

Inputs: 3 seeds per case, normal activations with about 1% outlier channels at 20-50x.
MX (per_1x32 with fp4/fp8 activations) path references are not implemented here; they
need validation of aiter's MX quant ops on gfx950 and raise NotImplementedError.
"""

import contextlib
import os
import re
import types

import torch

SEEDS = (0, 1, 2)
REPEATS = (
    2  # identical repeated calls per seed: some paths are not run-to-run deterministic
)
SAMPLE_ROWS = 256
OUTLIER_FRAC = 0.01
OUTLIER_SCALE = (20.0, 50.0)
FP64_CHUNK_BYTES = 3 << 30
METRICS = ("rel_l2", "max_norm", "cos", "allclose")
REFS = ("path", "exact")
GATE_METRICS = ("rel_l2", "max_norm")
# aiter's strict rule (op_tests/test_moe_2stage.py:667-675, test_fmoe with
# strict_accuracy): NaN fails; a nonzero checkAllclose mismatch fraction (rtol=atol=1e-2)
# fails only together with logits_diff above 0.01; asm FLAT MXFP4 rows use
# logits_diff < 2e-3 instead. The mismatch fraction itself is reported, not gated.
LOGITS_DIFF_LIMIT = 0.01
FLAT_MXFP4_LOGITS_LIMIT = 2e-3


def _seed(*parts):
    import hashlib

    return int.from_bytes(
        hashlib.sha1("|".join(map(str, parts)).encode()).digest()[:4], "big"
    )


def sample_rows(case_id, M, n=SAMPLE_ROWS):
    """Deterministic row sample (all rows when M <= n), the same for every candidate."""
    if M <= n:
        return torch.arange(M, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(_seed("rows", case_id, M))
    return torch.randperm(M, generator=g, device="cuda")[:n].sort().values


def acc_inputs(S, M, seed):
    """Activations of one seed (outlier channels) and routing that follows the case."""
    from . import workload

    g = torch.Generator(device="cuda").manual_seed(_seed("accin", S.case_id, M, seed))
    x = torch.randn((M, S.D), dtype=torch.float32, device="cuda", generator=g)
    n_out = max(1, round(OUTLIER_FRAC * S.D))
    ch = torch.randperm(S.D, generator=g, device="cuda")[:n_out]
    lo, hi = OUTLIER_SCALE
    scale = lo + (hi - lo) * torch.rand(n_out, generator=g, device="cuda")
    x[:, ch] *= scale
    tw, ti = workload.make_routing(S, M, seed_tag=f"acc{seed}")
    return x.to(S.dtype), tw, ti


@contextlib.contextmanager
def fp64_math():
    """Make aiter's torch reference compute in fp64 (its compute type is dtypes.fp32)."""
    import aiter.fused_moe as fm

    real = fm.dtypes

    class _Proxy(types.SimpleNamespace):
        def __getattr__(self, name):
            return torch.float64 if name == "fp32" else getattr(real, name)

    saved = {}
    for name in (
        "_mxfp4_per_1x32_weight_expert_to_f32",
        "_int4_per_1x32_weight_expert_to_f32",
    ):
        fn = getattr(fm, name)
        saved[name] = fn
        setattr(fm, name, lambda *a, _fn=fn, **k: _fn(*a, **k).double())
    fm.dtypes = _Proxy()
    try:
        yield
    finally:
        fm.dtypes = real
        for name, fn in saved.items():
            setattr(fm, name, fn)


def _slice_experts(t, e, e0, e1):
    if t is None:
        return None
    if t.shape[0] == e:
        return t[e0:e1]
    r = t.shape[0] // e
    return t[e0 * r : e1 * r]


def _path_quant(S, x, scale=None):
    """aiter's activation quant for this path (None for unquantized activations)."""
    import aiter

    qT = S.qtype
    if qT == aiter.QuantType.No:
        return x, None
    if qT == aiter.QuantType.per_1x32:
        if S.aq in (S.dtype, torch.bfloat16, torch.float16):
            return x, None  # a16w4 / a16wi4: bf16 activations
        raise NotImplementedError(
            "ref_path for MX activation quant (a4w4/a8w4/mxfp8) is not implemented"
        )
    if x.dtype == torch.float32:
        # the HIP quant kernels take bf16/fp16; fp32 aborts inside the op. fp32 here
        # is the fp32_direct intermediate (an in-kernel quantize of fp32 accumulators),
        # which aiter's torch quant reproduces.
        quant = aiter.get_torch_quant(qT)
        return (
            quant(x, scale=scale, quant_dtype=S.aq)
            if scale is not None
            else quant(x, quant_dtype=S.aq)
        )
    quant = aiter.get_hip_quant(qT)
    return quant(x, scale=scale, quant_dtype=S.aq)


def _ones_scale(S, rows, width):
    """Unit activation scales shaped like the real ones (ref_exact: weights only)."""
    import aiter

    qT = S.qtype
    if qT in (aiter.QuantType.per_Token,):
        return torch.ones((rows, 1), dtype=torch.float32, device="cuda")
    if qT == aiter.QuantType.per_Tensor:
        return torch.ones((1,), dtype=torch.float32, device="cuda")
    if qT == aiter.QuantType.per_1x128:
        return torch.ones((rows, width // 128), dtype=torch.float32, device="cuda")
    return None


def _stage1(S, ref_w, bias, a1q, a1s, tw, ids, E):
    from aiter.fused_moe import torch_moe_stage1

    n1 = (2 if S.g1u1 else 1) * S.I
    M = a1q.shape[0]
    ec = max(1, min(E, FP64_CHUNK_BYTES // (n1 * S.D * 8)))
    out1 = torch.zeros((M, S.topk, S.I), dtype=torch.float64, device="cuda")
    b1 = bias[0] if bias else None
    for e0 in range(0, E, ec):
        e1 = min(E, e0 + ec)
        sel = (ids >= e0) & (ids < e1)
        if not bool(sel.any()):
            continue
        ids_c = torch.where(sel, ids - e0, torch.full_like(ids, -1))
        part = torch_moe_stage1(
            a1q,
            _slice_experts(ref_w["w1"], E, e0, e1),
            _slice_experts(ref_w["w2"], E, e0, e1),
            tw,
            ids_c,
            dtype=torch.float64,
            activation=S.act,
            quant_type=S.qtype,
            a1_scale=a1s,
            w1_scale=_slice_experts(ref_w["w1_scale"], E, e0, e1),
            w1_bias=_slice_experts(b1, E, e0, e1) if b1 is not None else None,
            doweight=S.doweight,
            swiglu_limit=S.swiglu_limit,
            situ_beta=1.0 if S.beta is None else float(S.beta),
            situ_linear_beta=1.0 if S.linear_beta is None else float(S.linear_beta),
        )
        out1 = torch.where(sel[..., None], part, out1)
    return out1


def _stage2(S, ref_w, bias, a2q, a2s, tw, ids, E):
    from aiter.fused_moe import torch_moe_stage2

    M = a2q.shape[0]
    ec = max(1, min(E, FP64_CHUNK_BYTES // (S.D * S.I * 8)))
    res = torch.zeros((M, S.D), dtype=torch.float64, device="cuda")
    b2 = bias[1] if bias else None
    for e0 in range(0, E, ec):
        e1 = min(E, e0 + ec)
        sel = (ids >= e0) & (ids < e1)
        if not bool(sel.any()):
            continue
        ids_c = torch.where(sel, ids - e0, torch.full_like(ids, -1))
        res += torch_moe_stage2(
            a2q,
            _slice_experts(ref_w["w1"], E, e0, e1),
            _slice_experts(ref_w["w2"], E, e0, e1),
            tw,
            ids_c,
            dtype=torch.float64,
            quant_type=S.qtype,
            w2_scale=_slice_experts(ref_w["w2_scale"], E, e0, e1),
            a2_scale=a2s,
            w2_bias=_slice_experts(b2, E, e0, e1) if b2 is not None else None,
            doweight=not S.doweight,
        )
    return res


def _local_ids(ti, mask):
    if mask is None:
        return ti
    local = mask.cumsum(0, dtype=torch.int32) - 1
    local[mask == 0] = -1
    return local[ti.long()].to(torch.int32)


QUANTS = (
    "none",
    "per_token_fp8",
    "per_tensor_fp8",
    "per_1x128_fp8",
    "mx_fp4_round_up",
    "mx_fp4_even",
    "mx_fp8_round_up",
    "mx_fp8_even",
)
# Harness-only values, accepted from the route of an existing aiter path and never from a
# kernel's declaration: a 1-stage ASM kernel (and the MXMOE g1 epilogue) quantizes in-kernel
# with its own rule, so ref_path does not exist for it (ref_exact scores it).
IN_KERNEL = ("mx_fp4_in_kernel", "mx_fp8_in_kernel")
BARE_MX = ("mx_fp4", "mx_fp8")  # accepted for one release, with a warning
INTER = ("bf16", "fp32_direct")  # or bf16_then_<quant>
COMBINE = ("bf16_atomic", "fp32_atomic", "bf16_rows_fp32_sum", "fp32_reduce")
EXACT_ONLY_LIMIT = 0.5  # rel_l2 vs ref_exact above which a path with no ref_path fails


class NumericsError(ValueError):
    """A kernel's declared numerics are not in core's vocabulary or do not fit the case."""


class UnknownNumericsKey(NumericsError):
    """core declares a numerics key this harness does not know: every kernel that carries
    it would score against the wrong reference, so the run stops (once, at the first one)
    rather than reporting no correct candidate on every case."""

    def __init__(self, keys):
        self.keys = sorted(keys)
        super().__init__(
            f"harness {_sha('harness')} doesn't know numerics key "
            f"{', '.join(map(repr, self.keys))} from core {_sha('core')}; update the harness"
        )


def _sha(which):
    import subprocess

    try:
        if which == "harness":
            from . import runner

            return runner.harness_sha()
        import aiter

        root = os.path.dirname(os.path.dirname(aiter.__file__))
    except Exception:  # noqa: BLE001
        return "?"
    r = subprocess.run(
        ["git", "-C", root, "rev-parse", "--short=12", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip() or "?"


def quant_name(S):
    """The quant family the case implies: per_token_fp8, per_tensor_fp8, per_1x128_fp8,
    mx_fp4 or mx_fp8 (None: bf16 activations). MX families still need a scale rule."""
    import aiter

    q = S.qtype
    if q == aiter.QuantType.per_Token:
        return "per_token_fp8"
    if q == aiter.QuantType.per_1x128:
        return "per_1x128_fp8"
    if q == aiter.QuantType.per_Tensor:
        return "per_tensor_fp8"
    if q == aiter.QuantType.per_1x32:
        from aiter import dtypes

        if S.aq == dtypes.fp4x2:
            return "mx_fp4"
        if S.aq == dtypes.fp8:
            return "mx_fp8"
    return None


def split_quant(q):
    """'mx_fp4_even' -> ('mx_fp4', 'even'); any other value -> (value, None)."""
    for fam in ("mx_fp4", "mx_fp8"):
        if q.startswith(fam + "_"):
            return fam, q[len(fam) + 1 :]
    return q, None


def _check_quant(S, field, q, derived):
    """Validate one quant value of a declaration; returns it with bare MX resolved."""
    import warnings

    if q in BARE_MX:
        warnings.warn(
            f"numerics {field}={q!r} names no MX scale rule; assuming {q}_round_up "
            f"(aiter's default). Declare {q}_round_up or {q}_even; bare values will "
            "raise NumericsError in the next release",
            stacklevel=3,
        )
        q = q + "_round_up"
    if q not in QUANTS and not (derived and q in IN_KERNEL):
        raise NumericsError(f"unknown {field} {q!r}; core knows {QUANTS}")
    fam = split_quant(q)[0]
    qn = quant_name(S)
    if q != "none" and fam != qn:
        raise NumericsError(
            f"{field} {q!r} does not fit this case (quant type implies {qn!r})"
        )
    return q


def out_dtype_name(S):
    return "fp16" if S.dtype == torch.float16 else "bf16"


def normalize_numerics(S, declared, derived=False):
    """Validate a declared numerics dict against core's vocabulary and the case.

    Keys: ``act_quant`` and ``inter_quant`` (none, per_token_fp8, per_tensor_fp8,
    per_1x128_fp8, mx_fp4_round_up, mx_fp4_even, mx_fp8_round_up, mx_fp8_even),
    ``inter`` (bf16, fp32_direct, bf16_then_<quant>) and ``combine`` (bf16_atomic,
    fp32_atomic, bf16_rows_fp32_sum, fp32_reduce; "bf16" means the output dtype, which
    is recorded as ``out_dtype``). Anything else raises NumericsError: a guess would
    score the kernel against the wrong reference. ``combine`` is recorded as declared
    and validated, not modeled: the references sum in fp64. The intermediate's quant is
    the suffix of ``bf16_then_<quant>``, else ``inter_quant``, else ``act_quant``
    (the family's name when ``act_quant`` is none and the family is not MX).
    """
    if not isinstance(declared, dict):
        raise NumericsError(f"numerics must be a dict, got {declared!r}")
    allowed_keys = {
        "act_quant",
        "inter",
        "inter_quant",
        "combine",
        "source",
        "out_dtype",
    }
    extra = set(declared) - allowed_keys
    if extra:
        raise UnknownNumericsKey(extra)
    missing = [k for k in ("act_quant", "inter", "combine") if k not in declared]
    if missing:
        raise NumericsError(f"numerics missing {missing}: {declared!r}")
    act = _check_quant(S, "act_quant", declared["act_quant"], derived)
    inter, comb = declared["inter"], declared["combine"]
    qn = quant_name(S)
    if inter == "bf16":
        iq = "none"
    elif inter == "fp32_direct" or (
        isinstance(inter, str) and inter.startswith("bf16_then_")
    ):
        if inter == "fp32_direct":
            if "inter_quant" in declared:
                iq = _check_quant(S, "inter_quant", declared["inter_quant"], derived)
            elif act != "none":
                iq = act
            elif qn and not qn.startswith("mx_"):
                iq = qn
            elif qn:
                raise NumericsError(
                    "an MX intermediate needs its scale rule: declare inter_quant "
                    "(mx_fp4_round_up, mx_fp4_even, ...)"
                )
            else:
                iq = "none"
        else:
            iq = _check_quant(S, "inter", inter[len("bf16_then_") :], derived)
            if iq == "none":
                raise NumericsError(f"inter {inter!r} names no quant")
            inter = f"bf16_then_{iq}"
    else:
        raise NumericsError(
            f"unknown inter {inter!r}; core knows {INTER} and bf16_then_<quant>"
        )
    base = comb.removesuffix(" (inferred)") if isinstance(comb, str) else comb
    if base not in COMBINE and not (derived and comb == "n/a"):
        raise NumericsError(f"unknown combine {comb!r}; core knows {COMBINE}")
    if " (inferred)" in str(comb) and not derived:
        raise NumericsError(f"combine {comb!r} is a harness label, not a declaration")
    return {
        "act_quant": act,
        "inter": inter,
        "inter_quant": iq,
        "combine": comb,
        "out_dtype": out_dtype_name(S),
    }


NUMERICS_FIELDS = ("act_quant", "inter", "inter_quant", "combine", "out_dtype")


def numerics_key(n):
    return ";".join(f"{k}={n[k]}" for k in NUMERICS_FIELDS)


def key_fields(key):
    return dict(p.split("=", 1) for p in (key or "").split(";") if "=" in p)


def same_rounding(key_a, key_b):
    """Equal act_quant, inter and inter_quant (``combine`` is not modeled)."""
    if not key_a or not key_b:
        return False
    a, b = key_fields(key_a), key_fields(key_b)
    return all(a.get(k) == b.get(k) for k in ("act_quant", "inter", "inter_quant"))


def combine_of_route(route):
    """Epilogue of the aiter path that served a call.

    FlyDSL stage 2 names it (``_reduce``: per-route bf16 rows summed in fp32,
    ``_atomic``). aiter's CK, ck_tile and ASM 2-stage stage 2 and 1-stage ASM
    accumulate with bf16 atomics into the zero-filled output buffer: labelled
    "(inferred)". The gfx942 whole-graph path ends its prefill with ``sorted_sum``
    (bf16 rows, fp32 sum) and writes its batch-1 and decode paths with pairwise bf16
    atomics. "bf16" reads as the output dtype (see ``out_dtype``).
    """
    route = route or {}
    k2 = route.get("kernel2", "") or ""
    if route.get("stages") == "whole-graph":
        try:
            return (
                "bf16_rows_fp32_sum" if whole_graph_is_prefill(route) else "bf16_atomic"
            )
        except NumericsError:
            return "n/a"
    if route.get("stages") == "1-stage":
        return "bf16_atomic (inferred)"
    if route.get("stages") != "2-stage":
        return "n/a"
    if k2.startswith("flydsl_moe2"):
        if k2.endswith("_reduce"):
            return "bf16_rows_fp32_sum"
        if k2.endswith("_atomic"):
            return "bf16_atomic"
        return "n/a"
    if route.get("backend2") in ("ck", "ck_tile", "asm"):
        return "bf16_atomic (inferred)"
    return "n/a"


def whole_graph_is_prefill(route):
    """Whether a ``impl__flydsl_gfx942__<config>`` route runs the prefill sub-path, read
    from the config in its name. Raises when the name does not carry one: guessing puts
    the path on the wrong ref_path (the dell300x bar rejected every prefill config at
    ``ref_exact``'s 0.155 that way)."""
    from aiter.ops.flydsl.fused_moe_gfx942 import Config

    k1 = (route or {}).get("kernel1", "") or ""
    try:
        return bool(Config.from_string(k1.split("__", 2)[2]).use_prefill)
    except Exception as e:
        raise NumericsError(f"whole-graph route {k1!r} carries no config: {e}") from e


def in_kernel_variants(S, n):
    """The numerics to try for a path that quantizes MX in-kernel: {name: numerics}.

    Neither the scale rule nor the rounding before the quant of a 1-stage ASM kernel, or of
    the MXMOE g1 epilogue's intermediate, is named by the route, so ref_path is built under
    each combination (both quantizations for a path that quantizes its activations
    in-kernel, the intermediate alone where the activations go through the quant call)
    and the kernel is scored against the one it matches: ``round_up`` / ``even`` with the
    activation output quantized from fp32 (``fp32_direct``), and ``<rule>+bf16`` with it
    rounded to bf16 first (``bf16_then``). aiter's FlyDSL g1 epilogue is the latter: it
    reads 0.023 (logits_diff) from the fp32_direct reference on a4w4 SwiGLU and 6e-6 from
    the bf16_then one. Neither matching means another rule: reported, not guessed.
    """
    fam = quant_name(S)
    out = {}
    for rule in ("round_up", "even"):
        for rounding in ("fp32_direct", "bf16_then"):
            v = dict(n)
            if n["act_quant"].endswith("_in_kernel"):
                v["act_quant"] = f"{fam}_{rule}"
            if n["inter_quant"].endswith("_in_kernel"):
                v["inter_quant"] = f"{fam}_{rule}"
                if n["inter"] != "bf16":
                    v["inter"] = (
                        f"bf16_then_{fam}_{rule}"
                        if rounding == "bf16_then"
                        else "fp32_direct"
                    )
            out[rule if rounding == "fp32_direct" else f"{rule}+bf16"] = v
    return out


def has_in_kernel(n):
    return n["act_quant"].endswith("_in_kernel") or n["inter_quant"].endswith(
        "_in_kernel"
    )


def mx_rule_of_route(fam):
    """The scale rule of the quant call fused_moe makes for an MX 2-stage path.

    Read from the function the tree under test imports: the HIP quant (``aiter.ops.quant``)
    is ``round_up`` (kDefaultMxScaleRoundMode), the Triton fused sort quant
    (``aiter.ops.triton``) is ``even``. Anything else raises: a guess scores the path
    against the wrong reference.
    """
    import aiter.fused_moe as fm

    fn = getattr(
        fm,
        (
            "fused_dynamic_mxfp4_quant_moe_sort"
            if fam == "mx_fp4"
            else "fused_dynamic_mxfp8_quant_moe_sort"
        ),
    )
    mod = getattr(fn, "__module__", "") or ""
    if mod.startswith("aiter.ops.triton"):
        return "even"
    if mod.startswith("aiter.ops.quant"):
        return "round_up"
    raise NumericsError(f"cannot tell the MX scale rule of {mod}.{fn.__name__}")


def numerics_from_route(S, route):
    """Rounding points of the aiter path that served a call (the ref_path it is scored on).

    * CK 2-stage, ck_tile, ASM stage1 + CK, FlyDSL stage pairs: act quantized, stage-1
      output rounded to bf16 and then quantized (bf16_then_<quant>)
    * ASM 1-stage: act none for the a16/xbf16 kernels, else quantized; fp32_direct
    * FlyDSL whole-graph (impl__flydsl_gfx942): act none, bf16 intermediate unquantized
    * MXMOE and, for MX, flydsl_moe1: act quantized, intermediate rounding and scale
      rule matched against the output (see below)

    MX scale rule: the quant call the route makes (``mx_rule_of_route``); a 1-stage ASM
    kernel quantizes in-kernel (``in_kernel``), and so does the MXMOE g1 epilogue for the
    intermediate: ref_path is n/a there and ref_exact scores it.
    """
    route = route or {}
    qn = quant_name(S)
    full = route.get("full_impl", "") or ""
    k1 = route.get("kernel1", "") or ""
    one_stage = (
        route.get("stages") == "1-stage" or str(route.get("backend1", "")) == "asm"
    )
    mxmoe = k1.startswith("flydsl_mxmoe")
    mx = qn in ("mx_fp4", "mx_fp8")

    def q(rule_fam=None, in_kernel=False):
        if qn is None:
            return "none"
        if not mx:
            return qn
        if in_kernel:
            return f"{qn}_in_kernel"
        return f"{qn}_{mx_rule_of_route(qn)}"

    if "flydsl_gfx942" in full:
        # two numerics by sub-path (measured at 04df8dd46, MOEBUG-043): prefill quantizes
        # the activations per token and re-quantizes the intermediate to fp8 per token;
        # batch1 / decode keeps bf16 activations and has no intermediate requant
        if whole_graph_is_prefill(route):
            act = qn or "none"
            iq = qn or "none"
            inter = f"bf16_then_{iq}" if iq != "none" else "bf16"
        else:
            act, inter, iq = "none", "bf16", "none"
    elif one_stage and "flydsl_gfx942" not in full:
        a16 = bool(route.get("flat")) or "a16" in k1 or "xbf16" in k1
        act = "none" if a16 else q(in_kernel=True)
        inter, iq = "fp32_direct", q(in_kernel=True)
    elif mxmoe or (mx and k1.startswith("flydsl_moe1")):
        # MXMOE g1 quantizes the intermediate from fp32; flydsl_moe1 rounds it to bf16
        # first at some configs (a4w4 M=8) and quantizes from fp32 at others (M=64,
        # M=4096), measured on MI355X: the config picks it, not the kernel family, so the
        # rule and the rounding are matched against the output (in_kernel_variants)
        act, inter, iq = q(), "fp32_direct", q(in_kernel=True)
    else:
        act = q()
        iq = q()
        inter = f"bf16_then_{iq}" if iq != "none" else "bf16"
    return {
        "act_quant": act,
        "inter": inter,
        "inter_quant": iq,
        "combine": combine_of_route(route),
        "out_dtype": out_dtype_name(S),
    }


def _act_bits():
    from aiter import dtypes

    return {
        dtypes.bf16: 16,
        dtypes.fp16: 16,
        dtypes.fp8: 8,
        dtypes.i8: 8,
        dtypes.fp4x2: 4,
    }


def dispatch_rounds_more(S, M, gfx=None):
    """Why aiter's own dispatch rounds the activations more than the case asks for ("" if
    it does not).

    A case that passes bf16 activations to a mixed-precision weight format gets them
    quantized by ``fused_moe`` itself from a token count on: ``get_activation_dtype``
    switches SEPARATED Swiglu a16w4 to fp4x2 at ``GPTOSS_SWIGLU_MXFP4_BF16_BOUND`` (256)
    and INTERLEAVE Swiglu to fp8 at ``AITER_BF16_FP8_MOE_BOUND`` (gfx950). The call then
    runs a4w4 / a8w4 math, which is not what was asked for: dispatch is scored on the
    requested math (R2: activations and intermediate unquantized) and fails it."""
    import aiter.fused_moe as fm

    from . import workload

    bits = _act_bits()
    want = bits.get(S.aq)
    if want != 16 or S.kind != "fused" or workload.act_dtype_override(S) is not None:
        return ""
    if S.act.name != "Swiglu":
        # the Silu/Relu2 branches pick the dtype by their own rules (and later code can
        # keep bf16 for them): only the measured dtype settles those
        return ""
    try:
        got = fm.resolve_activation_dtype(
            S.qtype,
            S.wq,
            activation=S.act,
            gate_mode=S.gate_mode,
            M=M,
            hidden_dtype=S.dtype,
            gfx=gfx,
        )
    except Exception:  # noqa: BLE001  (a combination aiter has no activation rule for)
        return ""
    if got is None or bits.get(got, 16) >= want:
        return ""
    short = {16: "bf16", 8: "fp8", 4: "fp4x2"}
    return (
        f"activations requested {short[want]}, dispatched as {short[bits[got]]} "
        f"(M={M}, gate_mode={S.gate_mode}); the call runs a4w4/a8w4 math, scored on "
        "the requested math"
    )


# how fine an activation quantization is: a coarser one rounds the activations more than a
# finer one the case asks for (per-token scales in a per-1x128 blockscale case)
QUANT_COARSENESS = {
    "No": 0,
    "per_1x32": 1,
    "per_1x128": 2,
    "per_128x128": 2,
    "per_Token": 3,
    "per_Tensor": 4,
}
_CK_QUANT = {
    0: "No",
    1: "per_Tensor",
    2: "per_Token",
    3: "per_1x32",
    4: "per_1x128",
    5: "per_128x128",
}


def stage1_act_quant(kernel1):
    """The activation quantization a stage-1 kernel applies, from its name (None when the
    name does not say): ASM ``..._pertokenFp8_...`` takes per-token scales (also in its
    ``_blockscale`` variants, which are the weights' blocks), ``..._blockscaleFp8_...``
    per-128 block scales, and a CK gemm names its QuantType (``_Quant4_`` = per_1x128).
    """
    k = kernel1 or ""
    if re.search(r"pertoken(Fp8|Int8)", k):
        return "per_Token"
    if re.search(r"blockscale", k, re.IGNORECASE):
        return "per_1x128"
    m = re.search(r"_Quant(\d+)_", k)
    if m:
        return _CK_QUANT.get(int(m.group(1)))
    return None


def candidate_rounds_more(S, kernel1):
    """Why a candidate's stage 1 rounds the activations more than the case asks for, by
    quantization granularity ("" when it does not or the name does not say): a forced
    pair whose ASM stage 1 quantizes per token in a blockscale case is a coarser path than
    the requested numerics (the same kind of thing as a4w4 in an a16w4 case)."""
    got = stage1_act_quant(kernel1)
    want = S.qtype.name
    if got is None or got not in QUANT_COARSENESS or want not in QUANT_COARSENESS:
        return ""
    if QUANT_COARSENESS[got] <= QUANT_COARSENESS[want]:
        return ""
    return (
        f"stage-1 activations quantized {got}, the case asks for {want} "
        f"({str(kernel1)[:70]})"
    )


def measured_rounds_more(S, q_dtype_a, M):
    """The same judgement from the activation dtype aiter logged for a real call."""
    bits = _act_bits()
    want, got = bits.get(S.aq), bits.get(q_dtype_a)
    if want != 16 or got is None or got >= want:
        return ""
    short = {16: "bf16", 8: "fp8", 4: "fp4x2"}
    return (
        f"activations requested {short[want]}, dispatched as {short[got]} (M={M}, "
        f"gate_mode={S.gate_mode}); the call runs a4w4/a8w4 math, scored on the "
        "requested math"
    )


def requested_numerics(S, route=None):
    """The math the case asks for when it passes bf16 activations: nothing quantized, the
    intermediate unquantized (R2). The combine is the route's own."""
    return {
        "act_quant": "none",
        "inter": "bf16",
        "inter_quant": "none",
        "combine": combine_of_route(route),
        "out_dtype": out_dtype_name(S),
    }


def numerics_of(S, cand, route, ctx=None):
    """Rounding points to score a candidate on.

    A new kernel that served the call declares them (read from ``last_route()`` by the
    candidate); {"same_as": ...} means the dispatched aiter path's. A call that fell
    back to aiter (declined, or the off pass) is scored on the path that actually ran,
    taken from the route recorded during the call.
    """
    declared = getattr(cand, "declared_numerics", None)
    if declared is None:
        return numerics_from_route(S, route)
    if "same_as" in declared:
        base = getattr(ctx, "dispatch_route", None)
        if not base:
            raise NumericsError(
                "kernel declares same_as the dispatched path but the case's "
                "dispatch route is unknown"
            )
        if getattr(ctx, "dispatch_rounds_more", ""):
            # dispatch rounds more than requested: our kernels match the requested math
            return requested_numerics(S, base)
        return numerics_from_route(S, base)
    return normalize_numerics(S, declared)


def exact_ref(S, ref_w, bias, hidden, tw, ti, mask, rows):
    """ref_exact: bf16 activations, dequantized weights, no quantization, fp64."""
    import aiter

    E = S.E_local
    ids = _local_ids(ti, mask)
    r = len(rows)
    with fp64_math():
        a1s = _ones_scale(S, r, S.D)
        out1 = _stage1(S, ref_w, bias, hidden[rows], a1s, tw[rows], ids[rows], E)
        a2s = None
        if S.qtype in (aiter.QuantType.per_Token,):
            a2s = torch.ones((r, S.topk, 1), dtype=torch.float32, device="cuda")
        elif S.qtype == aiter.QuantType.per_Tensor:
            a2s = torch.ones((1,), dtype=torch.float32, device="cuda")
        elif S.qtype == aiter.QuantType.per_1x128:
            a2s = torch.ones(
                (r, S.topk, S.I // 128), dtype=torch.float32, device="cuda"
            )
        return _stage2(S, ref_w, bias, out1, a2s, tw[rows], ids[rows], E)


def path_ref(S, ref_w, bias, hidden, tw, ti, mask, rows, numerics=None):
    """ref_path at the given rounding points (see ``numerics_of``), fp64 elsewhere.

    ``numerics`` is a normalized dict (act_quant, inter, combine); None means the CK
    2-stage path. Raises NotImplementedError where aiter's activation quant is not
    reproduced (MX), NumericsError for values outside core's vocabulary.
    """
    import aiter

    n = numerics_from_route(S, {}) if numerics is None else numerics
    n = normalize_numerics(
        S, n, derived=n.get("combine") == "n/a" or "(inferred)" in str(n.get("combine"))
    )
    act_none = n["act_quant"] == "none"
    inter = n["inter"]
    iq = n["inter_quant"]
    if n["act_quant"].endswith("_in_kernel") or iq.endswith("_in_kernel"):
        raise NotImplementedError(
            "ref_path is n/a: the path quantizes MX activations in-kernel with its own "
            "rule (ref_exact scores it)"
        )
    act_fam, act_rule = split_quant(n["act_quant"])
    inter_fam, inter_rule = split_quant(iq)
    qn = quant_name(S)
    quant_inter = qn is not None and inter != "bf16" and iq != "none"
    E = S.E_local
    ids = _local_ids(ti, mask)
    h_rows, tw_rows, ids_rows = hidden[rows], tw[rows], ids[rows]
    r = len(rows)
    with fp64_math():
        if act_none:
            a1q, a1s = h_rows, _ones_scale(S, r, S.D)
        elif act_rule:
            a1q, a1s = _mx_deq(h_rows, act_fam, act_rule), None
        else:
            a1q_all, a1s_all = _path_quant(S, hidden)
            if a1s_all is None:
                a1q, a1s = h_rows, None
            elif a1s_all.numel() == 1:
                a1q, a1s = a1q_all[rows], a1s_all
            else:
                a1q, a1s = a1q_all[rows], a1s_all[rows]
            del a1q_all
        out1 = _stage1(S, ref_w, bias, a1q, a1s, tw_rows, ids_rows, E)
        flat = _inter_values(S, out1, inter).reshape(r * S.topk, S.I)
        if not quant_inter or S.qtype in (aiter.QuantType.No,):
            a2q = flat.reshape(r, S.topk, S.I)
            a2s = _ones_scale3(S, r)
        elif inter_rule:
            a2q = _mx_deq(flat, inter_fam, inter_rule).reshape(r, S.topk, S.I)
            a2s = None
        elif S.qtype == aiter.QuantType.per_Tensor:
            # the intermediate scale is a function of the whole batch
            a1q_f, a1s_f = (
                (hidden, _ones_scale(S, hidden.shape[0], S.D))
                if act_none
                else _path_quant(S, hidden)
            )
            full = _inter_values(
                S, _stage1(S, ref_w, bias, a1q_f, a1s_f, tw, ids, E), inter
            )
            _, a2s_t = _path_quant(S, full.reshape(-1, S.I))
            a2q, a2s_f = _path_quant(S, flat, scale=a2s_t)
            a2s = a2s_f if a2s_f is not None else a2s_t
            a2q = a2q.reshape(r, S.topk, S.I)
            del full
        else:
            a2q, a2s_f = _path_quant(S, flat)
            a2s = None if a2s_f is None else a2s_f.reshape(r, S.topk, -1)
            a2q = a2q.reshape(r, S.topk, S.I)
        return _stage2(S, ref_w, bias, a2q, a2s, tw_rows, ids_rows, E)


def _mx_deq(x, fam, rule):
    """Dequantized MX values of ``x`` per aiter's scale rule; a GEMM that consumes them
    with unit scales sees exactly what the quantized operand carries."""
    from . import mx_ref

    return mx_ref.mx_quant_ref(x, "fp4" if fam == "mx_fp4" else "fp8", rule)["deq"]


def _inter_values(S, out1, inter):
    """Stage-1 output as the kernel hands it on: fp32 for fp32_direct, else rounded to
    the output dtype. Never fp64: aiter's quant ops take at most fp32, and the kernels
    that quantize in-kernel see fp32 accumulators anyway."""
    return out1.float() if inter == "fp32_direct" else out1.to(S.dtype)


def _ones_scale3(S, r):
    import aiter

    if S.qtype == aiter.QuantType.per_Token:
        return torch.ones((r, S.topk, 1), dtype=torch.float32, device="cuda")
    if S.qtype == aiter.QuantType.per_Tensor:
        return torch.ones((1,), dtype=torch.float32, device="cuda")
    if S.qtype == aiter.QuantType.per_1x128:
        return torch.ones((r, S.topk, S.I // 128), dtype=torch.float32, device="cuda")
    return None


def references(S, ref_w, bias, hidden, tw, ti, mask, rows, numerics=None):
    """{"path": ..., "exact": ...} on the sampled rows ("path" absent for MX paths)."""
    out = {"exact": exact_ref(S, ref_w, bias, hidden, tw, ti, mask, rows)}
    try:
        out["path"] = path_ref(S, ref_w, bias, hidden, tw, ti, mask, rows, numerics)
    except NotImplementedError:
        pass
    return out


def row_metrics(y, ref):
    """rel_l2, max_norm, cosine distance, checkAllclose mismatch fraction, logits_diff."""
    y64, r64 = y.double(), ref.double()
    d = (y64 - r64).abs()
    ref_norm = r64.norm().clamp_min(1e-300)
    rms = r64.pow(2).mean(dim=-1).sqrt().clamp_min(1e-300)
    close = torch.isclose(r64, y64, rtol=1e-2, atol=1e-2)
    denom = (y64 * y64 + r64 * r64).sum().clamp_min(1e-300)
    return {
        "rel_l2": float(d.pow(2).sum().sqrt() / ref_norm),
        "max_norm": float((d / rms[:, None]).max()),
        "cos": float(
            1.0 - (y64 * r64).sum() / (y64.norm() * r64.norm()).clamp_min(1e-300)
        ),
        "allclose": float((~close).double().mean()),
        "logits_diff": float(1.0 - 2.0 * (y64 * r64).sum() / denom),
    }


def evaluate(
    S,
    ctx,
    call,
    refs_by_seed,
    rows,
    n_valid,
    seeds=SEEDS,
    sentinel=True,
    repeats=REPEATS,
    flat_mxfp4=False,
):
    """Run ``call`` on each seed's inputs and score it against both references.

    Returns {"per_seed": [...], "pass": bool|None, "nan_inf": int, "sentinel_ok": ...,
    "mean": {...}, "std": {...}}. ``ctx.hidden_static``, ``ctx.tw`` and ``ctx.ti`` are
    swapped for each seed and restored.
    """
    saved = (ctx.hidden_static.clone(), ctx.tw, ctx.ti)
    per_seed, nan_inf = [], 0
    nondeterministic = False
    try:
        for seed in seeds:
            hs, tw, ti = acc_inputs(S, S.M, seed)
            ctx.hidden_static.copy_(hs)
            ctx.tw, ctx.ti = tw, ti
            prev = None
            for rep in range(repeats):
                out = call()
                torch.cuda.synchronize()
                if prev is not None and not torch.equal(prev, out):
                    nondeterministic = True
                prev = out.clone() if repeats > 1 else None
                valid = out[:n_valid]  # rows past num_local_tokens are unspecified
                bad = int(torch.isnan(valid).sum()) + int(torch.isinf(valid).sum())
                nan_inf += bad
                y = out[:n_valid][rows[rows < n_valid]]
                m = {"nan_inf": bad, "seed": seed, "rep": rep}
                for ref_name, ref in refs_by_seed[seed].items():
                    keep = rows < n_valid
                    for k, v in row_metrics(y, ref[keep]).items():
                        m[f"{ref_name}_{k}"] = v
                per_seed.append(m)
    finally:
        ctx.hidden_static.copy_(saved[0])
        ctx.tw, ctx.ti = saved[1], saved[2]
    sentinel_ok = None
    try:
        if not sentinel:
            raise RuntimeError("sentinel check skipped")
        buf = torch.full((S.M, S.D), 12345.0, dtype=S.dtype, device="cuda")
        old, ctx.outbuf = ctx.outbuf, buf
        try:
            out_s = call()
            torch.cuda.synchronize()
        finally:
            ctx.outbuf = old
        sentinel_ok = not bool((out_s[:n_valid].float() == 12345.0).all(dim=-1).any())
    except Exception:  # noqa: BLE001  (output= unsupported by this candidate)
        sentinel_ok = None
    res = {
        "per_seed": per_seed,
        "nan_inf": nan_inf,
        "sentinel_ok": sentinel_ok,
        "nondeterministic": nondeterministic if repeats > 1 else None,
    }
    res.update(aggregate(per_seed))
    res["pass"] = rule(res, flat_mxfp4)
    return res


def aggregate(per_seed):
    mean, std = {}, {}
    keys = {k for m in per_seed for k in m if k not in ("seed", "rep")}
    for k in keys:
        vals = torch.tensor(
            [m[k] for m in per_seed if k in m], dtype=torch.float64, device="cpu"
        )
        mean[k] = float(vals.mean())
        std[k] = float(vals.std(unbiased=False)) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std}


def rule(res, flat_mxfp4=False):
    """aiter's strict test rule vs ref_path, plus NaN/Inf and the sentinel check.

    op_tests/test_moe_2stage.py:667-675 (test_fmoe, strict_accuracy): fail on NaN, or when
    the checkAllclose mismatch fraction is nonzero and logits_diff > 0.01; asm FLAT MXFP4
    rows additionally need logits_diff < 2e-3. None when ref_path is unavailable.
    """
    if res["nan_inf"] or res["sentinel_ok"] is False:
        return False
    if "path_allclose" not in res["mean"]:
        # ref_path is n/a (an in-kernel MX quant). Eligibility as a bar needs aiter's
        # strict logits_diff limit against ref_exact (a loose kernel must not set the
        # bar); the rel_l2 limit stays as a garbage guard. The verdict against the bar
        # is still on ref_exact in compare.
        e = res["mean"].get("exact_rel_l2")
        if e is None or any("exact_logits_diff" not in m for m in res["per_seed"]):
            return None
        if e > EXACT_ONLY_LIMIT:
            return False
        return all(m["exact_logits_diff"] <= LOGITS_DIFF_LIMIT for m in res["per_seed"])
    for m in res["per_seed"]:
        if m["path_allclose"] > 0 and m["path_logits_diff"] > LOGITS_DIFF_LIMIT:
            return False
        if flat_mxfp4 and m["path_logits_diff"] >= FLAT_MXFP4_LOGITS_LIMIT:
            return False
    return True


def match_rule(res, rules):
    """Pick the scale rule whose ref_path the kernel matches best (smallest mean
    rel_l2) and make its metrics the ``path_*`` ones, so aiter's strict rule is applied
    against it. Returns the rule, or None when no variant was scored."""
    scored = {
        r: res["mean"][f"path_{r}_rel_l2"]
        for r in rules
        if f"path_{r}_rel_l2" in res["mean"]
    }
    if not scored:
        return None
    best = min(scored, key=scored.get)
    for m in res["per_seed"]:
        for k in (*METRICS, "logits_diff"):
            if f"path_{best}_{k}" in m:
                m[f"path_{k}"] = m[f"path_{best}_{k}"]
    res.update(aggregate(res["per_seed"]))
    return best


def flat_columns(res, prefix=""):
    """Flat per-seed, mean and std columns of one evaluation."""
    out = {
        f"{prefix}acc_pass": res["pass"],
        f"{prefix}acc_nan_inf": res["nan_inf"],
        f"{prefix}nondeterministic": res["nondeterministic"],
    }
    for ref in REFS:
        for k in METRICS:
            name = f"{ref}_{k}"
            for m in res["per_seed"]:
                if name in m and m["rep"] == 0:
                    out[f"{prefix}{name}_s{m['seed']}"] = m[name]
            if name in res["mean"]:
                out[f"{prefix}{name}_mean"] = res["mean"][name]
                out[f"{prefix}{name}_std"] = res["std"][name]
    return out


def seed_columns():
    cols = ["acc_pass", "acc_nan_inf", "nondeterministic"]
    for ref in REFS:
        for k in METRICS:
            cols += [f"{ref}_{k}_s{i}" for i in range(len(SEEDS))]
            cols += [f"{ref}_{k}_mean", f"{ref}_{k}_std"]
    return cols


def gate_columns(who):
    """Mean/std of the gated metrics against both references for dispatch, bar or new."""
    cols = [f"{who}_acc_pass", f"{who}_nondeterministic"]
    for ref in REFS:
        cols.append(f"{who}_{ref}_allclose_mean")
        for k in GATE_METRICS:
            cols += [f"{who}_{ref}_{k}_mean", f"{who}_{ref}_{k}_std"]
    return cols
