# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regenerate the MoE case matrix and tier 3 from one aiter tree.

Tier 3 is every MoE CSV under ``aiter/configs`` (model_configs included) plus the
default sweeps of the MoE op_tests, folded to unique cases. The matrix logic is the
catalogue's; the tier rows follow ``schema.COLS``.
"""

import os
from collections import OrderedDict, defaultdict

from . import build
from .build import (
    load_tuned_tables,
    side_tables,
    table_record,
    untuned_tables,
)
from .tests_cases import gen_tests

ARCH_CU = {"gfx942": "304", "gfx950": "256"}


def nq(v):
    v = str(v)
    if v.startswith("float8_e4m3") or v == "fp8":
        return "fp8"
    if v in ("None", "bfloat16", "bf16", ""):
        return "bf16"
    return v


def nqt(v):
    return "per_1x128" if v == "per_128x128" else v


def sk(r):
    return (
        str(r["token"]),
        str(r["model_dim"]),
        str(r["inter_dim"]),
        str(r["expert"]),
        str(r["topk"]),
        r["act_type"],
        r["dtype"],
        nq(r["q_dtype_a"]),
        nq(r["q_dtype_w"]),
        nqt(r["q_type"]),
        r["use_g1u1"],
        r["doweight_stage1"],
        r["_ident"].get("hidden_pad", "0"),
        r["_ident"].get("intermediate_pad", "0"),
    )


def full_key(r):
    return (r["cu_num"], tuple(sorted(r["_ident"].items())), sk(r))


def merge_into(dst, src_name):
    if src_name not in dst["source"].split("|"):
        dst["source"] += "|" + src_name


def build_matrix(aiter_root):
    """Union of tuned rows, side tables, untuned lists and test sweeps."""
    build.configure(aiter_root)
    tuned = OrderedDict()
    for rel, row in load_tuned_tables():
        r = table_record(rel, row, tuned=True)
        k = full_key(r)
        if k in tuned:
            merge_into(tuned[k], rel)
        else:
            tuned[k] = r
    side = OrderedDict()
    for r in side_tables():
        k = full_key(r)
        if k in side:
            merge_into(side[k], r["source"])
        else:
            side[k] = r
    tuned_by_sk = defaultdict(list)
    for r in tuned.values():
        tuned_by_sk[sk(r)].append(r)

    unt = OrderedDict()
    for rel, row in untuned_tables():
        if not row.get("token"):
            continue
        r = table_record(rel, row, tuned=False)
        s = sk(r)
        if s in tuned_by_sk:
            for t in tuned_by_sk[s]:
                if "also_in_untuned_list" not in t["extra"]:
                    t["extra"] += (";" if t["extra"] else "") + "also_in_untuned_list=1"
            continue
        r["cu_num"] = ""
        r["extra"] = (r["extra"] + ";" if r["extra"] else "") + "untuned_only=1"
        k = ("", tuple(sorted(r["_ident"].items())), s)
        if k in unt:
            merge_into(unt[k], rel)
        else:
            unt[k] = r

    tests = OrderedDict()
    for r in gen_tests():
        s = sk(r)
        if s in tuned_by_sk:
            for t in tuned_by_sk[s]:
                if r["source"] not in t["source"].split("|"):
                    t["source"] += "|" + r["source"]
            continue
        k = ("", tuple(sorted(r["_ident"].items())), s + (r["extra"],))
        if k in tests:
            merge_into(tests[k], r["source"])
        else:
            tests[k] = r
    return (
        sorted(tuned.values(), key=_sort_key(0))
        + sorted(side.values(), key=_sort_key(1))
        + sorted(unt.values(), key=_sort_key(2))
        + sorted(tests.values(), key=_sort_key(3))
    )


def _tonum(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 10**9


def _sort_key(rank):
    return lambda r: (
        rank,
        _tonum(r["cu_num"]) if r["cu_num"] else 10**9,
        r["model"],
        _tonum(r["model_dim"]),
        _tonum(r["inter_dim"]),
        _tonum(r["expert"]),
        _tonum(r["topk"]),
        _tonum(r["token"]),
        r["q_type"],
        r["act_type"],
    )


def norm_dtype(s):
    s = (s or "").strip().replace("torch.", "").rstrip("?")
    s = s.removeprefix("mx")
    return {
        "float8_e4m3fn": "fp8",
        "float8_e4m3fnuz": "fp8",
        "fp8": "fp8",
        "float4_e2m1fn_x2": "fp4x2",
        "int4": "i4x2",
        "i4x2": "i4x2",
        "int8": "i8",
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
    }.get(s, s)


def norm_qtype(s):
    s = (s or "").replace("QuantType.", "").rstrip("?")
    return "per_1x128" if s == "per_128x128" else s


def family_of(qa, qw, qt):
    if qt == "No":
        return "a16w16"
    if qt == "per_Tensor":
        return "int8" if qw == "i8" else "fp8-tensor"
    if qt == "per_Token":
        if qw == "i8":
            return "int8"
        if qw == "i4x2":
            return "a8wi4"
        return "fp8-token"
    if qt == "per_1x128":
        return "fp8-block"
    if qt == "per_1x32":
        if qw == "i4x2":
            return "a16wi4"
        if qw == "fp8":
            return "mxfp8"
        if qw == "fp4x2":
            return {
                "fp4x2": "a4w4",
                "fp8": "a8w4",
                "bf16": "a16w4",
                "fp16": "a16w4",
            }.get(qa, "w4-auto")
    return "other"


def route_family(k, run_1stage):
    k = (k or "").strip()
    if not k or k in ("Null", "None"):
        return ""
    for prefix, name in (
        ("impl__flydsl_gfx942", "flydsl-gfx942-graph"),
        ("flydsl_mxmoe_g1", "flydsl-mxmoe-g1"),
        ("flydsl_mxmoe_g2", "flydsl-mxmoe-g2"),
        ("flydsl_moe2_layout", "flydsl-moe2-layout"),
    ):
        if k.startswith(prefix):
            return name
    if k.startswith(("flydsl_moe1_abf16", "flydsl_moe2_abf16")):
        return "flydsl-a16wmix"
    if k.startswith("flydsl_"):
        return "flydsl-mixed"
    if k.startswith("moe_ck2stages"):
        return "ck2stages"
    if "cktile" in k:
        return "ck_tile"
    if k.startswith("opus"):
        return "opus"
    if k.startswith(("_ZN5aiter", "fmoe")):
        return "asm-1stage" if run_1stage else "asm-stage1"
    if k.startswith(("grouped", "impl__")):
        return "other-impl"
    return "other"


def parse_extra(s):
    d = {}
    for kv in (s or "").split(";"):
        if "=" in kv:
            a, b = kv.split("=", 1)
            d[a.strip()] = b.strip()
    return d


def kind_of(row):
    base = os.path.basename(row["source"].split(":")[0])
    if base.startswith("test_moe_sorting"):
        return "sorting"
    if "topk" in base.lower():
        return "routing"
    if row["source"].startswith("triton_tests"):
        return "triton"
    if "grouped_fmoe" in base:
        return "fused-grouped"
    if row["q_type"] and row["model_dim"] and row["inter_dim"]:
        return "fused"
    return "other"


def as_int(s, default=0):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return default


def build_tier3(matrix_rows):
    """Fold the matrix into unique tier-3 case rows (schema.COLS)."""
    cases = OrderedDict()
    for r in matrix_rows:
        kind = kind_of(r)
        qa = norm_dtype(r["q_dtype_a"])
        qw = norm_dtype(r["q_dtype_w"])
        qt = norm_qtype(r["q_type"])
        act = (r["act_type"] or "").split("|")[0].rstrip("?")
        ex = parse_extra(r["extra"])
        flags = []
        for f in ("bias", "expert_mask", "num_shared", "gate_mode", "flat", "xbf16"):
            if f in ex and ex[f] not in ("", "0", "False", "None"):
                flags.append(f"{f}={ex[f]}")
        if act == "silu_clamped" or "swiglu_limit" in r["extra"]:
            flags.append("swiglu_limit")
        key = (
            kind,
            as_int(r["token"]),
            r["model_dim"],
            r["inter_dim"],
            r["expert"],
            r["topk"],
            act,
            norm_dtype(r["dtype"]),
            qa,
            qw,
            qt,
            r["use_g1u1"],
            r["doweight_stage1"],
            ";".join(sorted(flags)),
        )
        c = cases.setdefault(
            key,
            {
                "models": set(),
                "sources": set(),
                "route": {},
                "family": family_of(qa, qw, qt),
            },
        )
        if r["model"]:
            c["models"].add(r["model"].rstrip("?"))
        c["sources"].add(os.path.basename(r["source"].split(":")[0]))
        cu = r["cu_num"]
        gfx = ex.get("gfx") or {"304": "gfx942", "256": "gfx950"}.get(cu, "")
        run1 = ex.get("run_1stage", "0") in ("1", "True")
        for arch, acu in ARCH_CU.items():
            if cu == acu and gfx == arch and (r["kernel1"] or r["kernel2"]):
                rf1 = route_family(r["kernel1"], run1)
                rf2 = route_family(r["kernel2"], run1)
                c["route"][arch] = rf1 if (run1 or not rf2) else f"{rf1}+{rf2}"

    out = []
    for i, (key, c) in enumerate(cases.items()):
        kind, tok, md, idim, e, k, act, dt, qa, qw, qt, g1, dw, flags = key
        out.append(
            {
                "case_id": f"tier3-{i:05d}",
                "kind": kind,
                "family": c["family"] if kind.startswith("fused") else kind,
                "model": "|".join(sorted(c["models"]))[:120],
                "token": tok,
                "model_dim": md,
                "inter_dim": idim,
                "expert": e,
                "topk": k,
                "act_type": act,
                "dtype": dt,
                "q_dtype_a": qa,
                "q_dtype_w": qw,
                "q_type": qt,
                "use_g1u1": g1,
                "doweight_stage1": dw,
                "extra": flags,
                "route_gfx942": c["route"].get("gfx942", "heuristic"),
                "route_gfx950": c["route"].get("gfx950", "heuristic"),
                "modes": "warm,cold",
                "sources": "|".join(sorted(c["sources"]))[:160],
                "reason": "all-sources",
                "added_by": "build_tiers",
            }
        )
    return out
