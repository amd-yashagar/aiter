# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch decision per case, without timing.

The route comes from aiter's own router: ``_fused_moe_impl`` is run on meta tensors
with a hook on the metadata it resolves (so aiter computes every argument of
``get_2stage_cfgs`` itself), and the tuned-row lookup is read from the tables
``get_2stage_cfgs`` loaded. Nothing is launched. ``run`` records the route the real
call used through ``RouteRecorder`` as well.
"""

import os
import re

import torch

ROUTE_COLS = [
    "route_key",
    "route_family",
    "tuned",
    "backend1",
    "kernel1",
    "backend2",
    "kernel2",
    "block_m",
    "ksplit",
    "stages",
    "flat",
    "output_aux",
    "fuse_quant",
    "full_impl",
    "sort_variant",
    "quant_variant",
    "route_error",
]

ENV_KEYS = (
    "AITER_CONFIG_FMOE",
    "AITER_BYPASS_TUNE_CONFIG",
    "AITER_ONLINE_TUNE",
    "AITER_KSPLIT",
    "AITER_USE_NT",
    "AITER_FLYDSL_FORCE",
    "AITER_MXMOE_FALLBACK",
    "AITER_MOE_SORT_BACKEND",
    "AITER_USE_CK_MOE_SORTING",
    "AITER_USE_FLYDSL_MOE_SORTING",
    "AITER_FLYDSL_FORCE_REDUCE",
    "AITER_FLYDSL_STAGE2_FP8",
    "AITER_BF16_FP8_MOE_BOUND",
    "AITER_XBFLOAT16",
    "GPTOSS_SWIGLU_MXFP4_BF16_BOUND",
    "CU_NUM",
    "GPU_ARCHS",
)


def backend_of(name):
    n = (name or "").strip()
    if not n:
        return ""
    if n.startswith("impl__flydsl_"):
        return "flydsl-graph"
    if n.startswith("flydsl_"):
        return "flydsl"
    if n.startswith("moe_ck2stages"):
        return "ck"
    if "cktile" in n:
        return "ck_tile"
    if n.startswith("opus"):
        return "opus"
    if n.startswith(("_ZN5aiter", "fmoe")):
        return "asm"
    return "other"


def _stage_name(stage):
    if stage is None:
        return ""
    kw = getattr(stage, "keywords", None) or {}
    for k in ("kernelName", "kernelName1", "kernelName2", "kernel_name"):
        if kw.get(k):
            return str(kw[k])
    func = getattr(stage, "func", stage)
    return getattr(func, "__name__", repr(func))


def describe_metadata(md):
    """Flatten a MOEMetadata into route columns."""
    k1 = _stage_name(md.stage1)
    k2 = _stage_name(md.stage2)
    full = ""
    if md.full_impl is not None:
        full = str(
            getattr(md.full_impl, "name", None)
            or getattr(getattr(md.full_impl, "impl", None), "__name__", None)
            or getattr(md.full_impl, "__name__", None)
            or repr(md.full_impl)
        )
    flat = int(md.flat) if md.flat is not None else 0
    if full:
        b1, b2, k1, k2 = backend_of(full) or "impl", "", full, ""
        stages = "whole-graph"
    else:
        b1, b2 = backend_of(k1), backend_of(k2)
        if not b1 and md.stage1 is not None:
            b1 = "python:" + _stage_name(md.stage1)
        if not b2 and md.stage2 is not None:
            b2 = "python:" + _stage_name(md.stage2)
        stages = "1-stage" if md.run_1stage else "2-stage"
    if flat:
        sort_variant = f"flat{flat}"
    elif md.output_aux:
        sort_variant = f"aux-{md.output_aux if md.output_aux is not True else 'fused'}"
    elif full:
        sort_variant = "in-graph"
    else:
        backend = os.environ.get("AITER_MOE_SORT_BACKEND", "auto").lower()
        if os.environ.get("AITER_USE_CK_MOE_SORTING") == "1":
            backend = "ck"
        elif os.environ.get("AITER_USE_FLYDSL_MOE_SORTING") == "1":
            backend = "flydsl"
        sort_variant = f"host-{backend}"
    quant_variant = ",".join(
        x
        for x in (
            f"fuse={md.fuse_quant}" if md.fuse_quant else "",
            "" if md.prequant else "noprequant",
            "skip-inter" if md.skip_inter_quant else "",
        )
        if x
    )
    return {
        "backend1": b1,
        "kernel1": k1,
        "backend2": b2,
        "kernel2": k2,
        "block_m": int(md.block_m) if md.block_m is not None else "",
        "ksplit": int(md.ksplit or 0),
        "stages": stages,
        "flat": flat,
        "output_aux": str(md.output_aux) if md.output_aux else "",
        "fuse_quant": md.fuse_quant or "",
        "full_impl": full,
        "sort_variant": sort_variant,
        "quant_variant": quant_variant,
    }


def _norm_kernel(name):
    return re.sub(r"\s+", "", name or "")


def route_key(desc, tuned):
    parts = [
        desc["stages"],
        f"{desc['backend1']}:{_norm_kernel(desc['kernel1'])}",
        f"{desc['backend2']}:{_norm_kernel(desc['kernel2'])}",
        f"bm{desc['block_m']}",
        f"ks{desc['ksplit']}",
        f"flat{desc['flat']}",
        f"aux{desc['output_aux']}",
        desc["sort_variant"],
        desc["quant_variant"],
        "tuned" if tuned else "heuristic",
    ]
    return "|".join(parts)


def route_family_of(desc):
    if desc["full_impl"]:
        return "whole-graph:" + desc["backend1"]
    b1, b2 = desc["backend1"], desc["backend2"]
    if desc["stages"] == "1-stage":
        return f"1stage-{b1}"
    return f"{b1}+{b2}"


def tuned_row_hit(fm, call_args, md_desc):
    """True when a tuned row matched the call and its kernels are what resolved."""
    cfg = getattr(fm, "cfg_2stages", None)
    if not cfg or not call_args:
        return False
    primary, fallback = cfg
    try:
        gfx = fm.get_gfx_runtime()
        cu = fm.get_cu_num()
    except Exception:  # noqa: BLE001
        return False
    a = call_args
    keys = (
        gfx,
        cu,
        a["token"],
        a["model_dim"],
        a["inter_dim"],
        a["expert"],
        a["topk"],
        str(a["activation"]),
        str(a["dtype"]),
        str(a["q_dtype_a"]),
        str(a["q_dtype_w"]),
        str(a["q_type"]),
        a["use_g1u1"],
        a["doweight_stage1"],
    )
    row = primary.get(keys)
    if row is None:
        keys_off = keys[:7] + (fm._ACT_TYPE_DISABLED_KEY,) + keys[8:]
        row = fallback.get(keys_off)
    if row is None:
        tiers = fm._PADDED_M_TIERS
        if a["token"] > tiers[0]:
            idx = tiers.index(a["token"]) if a["token"] in tiers else -1
            for t in reversed(tiers[:idx]):
                k2 = keys[:2] + (t,) + keys[3:]
                row = primary.get(k2) or fallback.get(
                    k2[:7] + (fm._ACT_TYPE_DISABLED_KEY,) + k2[8:]
                )
                if row is not None:
                    break
    if row is None:
        return False
    kn1 = str(row.get("kernelName1", "") or "").strip()
    kn2 = str(row.get("kernelName2", "") or "").strip()
    resolved = {md_desc["kernel1"], md_desc["kernel2"]}
    if md_desc["full_impl"]:
        return bool(kn1) and kn1 in md_desc["full_impl"]
    if kn1 and kn1 in resolved:
        return True
    if kn2 and kn2 in resolved:
        return True
    if bool(row.get("run_1stage", False)) and md_desc["stages"] == "1-stage":
        return bool(kn1)
    return False


class RouteRecorder:
    """Wrap ``aiter.fused_moe.get_2stage_cfgs`` to log every resolution."""

    def __init__(self):
        import aiter.fused_moe as fm

        self.fm = fm
        self.orig = fm.get_2stage_cfgs
        self.records = []
        self.active = False

    def __enter__(self):
        orig = self.orig
        rec = self
        names = [
            "token",
            "model_dim",
            "inter_dim",
            "expert",
            "topk",
            "dtype",
            "q_dtype_a",
            "q_dtype_w",
            "q_type",
            "use_g1u1",
            "activation",
            "doweight_stage1",
        ]

        def wrapper(*args, **kwargs):
            md = orig(*args, **kwargs)
            if rec.active:
                a = dict(zip(names, args))
                a.update(kwargs)
                rec.records.append((a, md))
            return md

        wrapper.cache_clear = getattr(orig, "cache_clear", lambda: None)
        self.fm.get_2stage_cfgs = wrapper
        self.active = True
        return self

    def __exit__(self, *exc):
        self.fm.get_2stage_cfgs = self.orig
        self.active = False

    def last(self):
        return self.records[-1] if self.records else (None, None)

    def clear(self):
        self.records.clear()


def _meta_tensors(S):
    """Meta-device stand-ins with the shapes and layout flags the router reads."""
    from aiter import dtypes

    n1 = (2 if S.g1u1 else 1) * S.I
    wd = S.wq
    d_last = S.D
    i_last = S.I
    if S.qtype.name == "per_1x32" and wd not in (dtypes.i4x2, dtypes.fp8):
        d_last //= 2
        i_last //= 2
    w1 = torch.empty((S.E_local, n1, d_last), dtype=wd, device="meta")
    w2 = torch.empty((S.E_local, S.D, i_last), dtype=wd, device="meta")
    w1.is_shuffled = w2.is_shuffled = True
    if S.qtype.name == "per_1x32" and S.wq == dtypes.fp4x2:
        w1.gate_up_interleave = S.aq == dtypes.fp8
        w2.gate_up_interleave = False
    hidden = torch.empty((S.M, S.D), dtype=S.dtype, device="meta")
    ids = torch.empty((S.M, S.topk), dtype=torch.int32, device="meta")
    wts = torch.empty((S.M, S.topk), dtype=torch.float32, device="meta")
    scale = torch.empty((1,), dtype=torch.float32, device="meta")
    return w1, w2, hidden, ids, wts, scale


class _DryStop(Exception):
    """Raised by the stubs that stand in for every launching function."""


_LAUNCHERS = (
    "_moe_sorting_impl",
    "_flydsl_moe_sorting",
    "moe_sorting",
    "_adaptive_moe_sort",
    "_fused_decode_sort_quant",
    "fused_moe_1stage",
    "fused_moe_2stages",
)


class _no_launch:
    """Replace every function that launches work, so only routing runs."""

    def __init__(self, fm):
        self.fm = fm
        self.saved = {}

    def __enter__(self):
        for name in _LAUNCHERS:
            if hasattr(self.fm, name):

                def stub(*a, **k):
                    raise _DryStop()

                stub.__name__ = name
                self.saved[name] = getattr(self.fm, name)
                setattr(self.fm, name, stub)
        return self

    def __exit__(self, *exc):
        for name, fn in self.saved.items():
            setattr(self.fm, name, fn)


def _stub_full_impl(md):
    """Keep the metadata but make a whole-graph impl a no-launch stub."""
    import dataclasses

    if md.full_impl is None:
        return md

    def stub(*a, **k):
        raise _DryStop()

    return dataclasses.replace(md, full_impl=stub)


def dry_route(case, S=None):
    """Route for one fused case: returns a dict of ROUTE_COLS (never raises)."""
    import aiter.fused_moe as fm

    from . import workload

    S = S or workload.resolve(case)
    out = {c: "" for c in ROUTE_COLS}
    w1, w2, hidden, ids, wts, scale = _meta_tensors(S)

    def meta(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    full_descs = []

    def transform(md):
        full_descs.append(md)
        return _stub_full_impl(md)

    # get_2stage_cfgs is lru-cached and 1-stage metadata binds fused_moe_1stage at
    # resolution time: entries made while the launchers are stubbed must not survive.
    fm.get_2stage_cfgs.cache_clear()
    with RouteRecorder() as rec, _no_launch(fm):
        try:
            fm._fused_moe_impl(
                hidden,
                w1,
                w2,
                wts,
                ids,
                expert_mask=meta((S.E,), torch.int32) if S.is_ep else None,
                activation=S.act.value,
                quant_type=S.qtype.value,
                doweight_stage1=S.doweight,
                w1_scale=scale,
                w2_scale=scale,
                hidden_pad=S.hidden_pad,
                intermediate_pad=S.intermediate_pad,
                bias1=meta((S.E_local, 1), torch.float32) if S.bias else None,
                bias2=meta((S.E_local, 1), torch.float32) if S.bias else None,
                swiglu_limit=S.swiglu_limit,
                beta=S.beta,
                linear_beta=S.linear_beta,
                gate_mode=S.gate_mode,
                num_local_tokens=meta((1,), torch.int32) if S.num_local_frac else None,
                quant_dtype_a=workload.act_dtype_override(S),
                _metadata_transform=transform,
            )
        except _DryStop:
            pass
        except Exception as e:  # noqa: BLE001
            if not full_descs:
                out["route_error"] = (
                    f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
                )
                out["route_key"] = "unsupported:" + out["route_error"][:120]
                return out
        args, _ = rec.last()
    fm.get_2stage_cfgs.cache_clear()
    if not full_descs:
        out["route_error"] = "no metadata resolved"
        out["route_key"] = "unsupported:no metadata"
        return out
    return route_from_record((args, full_descs[-1]), fm)


def route_from_record(rec, fm=None):
    """Route columns from a RouteRecorder record taken during a real call."""
    if fm is None:
        import aiter.fused_moe as fm
    args, md = rec
    out = {c: "" for c in ROUTE_COLS}
    if md is None:
        return out
    desc = describe_metadata(md)
    tuned = tuned_row_hit(fm, args, desc)
    out.update(desc)
    out["tuned"] = int(tuned)
    out["route_key"] = route_key(desc, tuned)
    out["route_family"] = route_family_of(desc)
    return out


def aux_route(case):
    """Non-GEMM cases: the dispatch is the op family plus the backend env."""
    out = {c: "" for c in ROUTE_COLS}
    if case.kind == "sorting":
        if case["q_type"] == "per_1x32":
            key = "mxfp4-sort-quant"
        else:
            key = (
                "moe_sorting:"
                + os.environ.get("AITER_MOE_SORT_BACKEND", "auto").lower()
            )
    else:
        key = f"topk:{case['dtype']}"
    out.update(route_key=f"{case.kind}|{key}", route_family=case.kind, tuned=0)
    return out
