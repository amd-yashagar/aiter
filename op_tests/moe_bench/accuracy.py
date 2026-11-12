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


DEFAULT_NUMERICS = {"act": "hip", "inter": "bf16"}


def numerics_key(n):
    n = dict(DEFAULT_NUMERICS, **(n or {}))
    return f"act={n['act']};inter={n['inter']}"


def numerics_of(S, cand, route):
    """Rounding points of the path that serves a candidate (the ref_path it is scored on).

    ``act``: ``hip`` = activations quantized with aiter's ``get_hip_quant``, ``none`` =
    bf16 activations into the GEMM. ``inter``: ``bf16`` = stage-1 output rounded to the
    output dtype, then re-quantized (CK 2-stage, torch_moe_stage1); ``fp32`` = quantized
    from the fp32 accumulators in-kernel (ASM 1-stage, MXMOE g1); ``none`` = bf16
    intermediate, no quantization (a16 paths, gfx942 FlyDSL whole-graph decode).
    New kernels declare their own ``numerics`` class attribute; aiter candidates are
    mapped from the route:

    * CK 2-stage, ck_tile, ASM stage1 + CK, FlyDSL stage pairs: act hip, inter bf16
    * ASM 1-stage: act none for the a16/xbf16 kernels, else hip; inter fp32
    * FlyDSL whole-graph (impl__flydsl_gfx942): act none, inter none
    * MXMOE: act hip, inter fp32
    """
    declared = getattr(cand, "numerics", None)
    if isinstance(declared, dict) and declared:
        return dict(DEFAULT_NUMERICS, **declared)
    route = route or {}
    full = route.get("full_impl", "") or ""
    k1 = route.get("kernel1", "") or ""
    if "flydsl_gfx942" in full:
        return {"act": "none", "inter": "none"}
    if route.get("stages") == "1-stage" or str(route.get("backend1", "")) == "asm":
        a16 = bool(route.get("flat")) or "a16" in k1 or "xbf16" in k1
        return {"act": "none" if a16 else "hip", "inter": "fp32"}
    if k1.startswith("flydsl_mxmoe"):
        return {"act": "hip", "inter": "fp32"}
    return dict(DEFAULT_NUMERICS)


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

    Raises NotImplementedError where aiter's activation quant is not reproduced (MX).
    """
    import aiter

    n = dict(DEFAULT_NUMERICS, **(numerics or {}))
    E = S.E_local
    ids = _local_ids(ti, mask)
    h_rows, tw_rows, ids_rows = hidden[rows], tw[rows], ids[rows]
    r = len(rows)
    with fp64_math():
        if n["act"] == "none":
            a1q, a1s = h_rows, _ones_scale(S, r, S.D)
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
        flat_src = out1 if n["inter"] == "fp32" else out1.to(S.dtype)
        flat = flat_src.reshape(r * S.topk, S.I)
        if n["inter"] == "none" or S.qtype in (aiter.QuantType.No,):
            a2q = flat.reshape(r, S.topk, S.I)
            a2s = _ones_scale3(S, r)
            if S.qtype == aiter.QuantType.No:
                a2s = None
        elif S.qtype == aiter.QuantType.per_Tensor:
            # the intermediate scale is a function of the whole batch
            a1q_f, a1s_f = (
                (hidden, _ones_scale(S, hidden.shape[0], S.D))
                if n["act"] == "none"
                else _path_quant(S, hidden)
            )
            full = _stage1(S, ref_w, bias, a1q_f, a1s_f, tw, ids, E)
            full = full if n["inter"] == "fp32" else full.to(S.dtype)
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
        vals = torch.tensor([m[k] for m in per_seed if k in m], dtype=torch.float64)
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
        return None
    for m in res["per_seed"]:
        if m["path_allclose"] > 0 and m["path_logits_diff"] > LOGITS_DIFF_LIMIT:
            return False
        if flat_mxfp4 and m["path_logits_diff"] >= FLAT_MXFP4_LOGITS_LIMIT:
            return False
    return True


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
