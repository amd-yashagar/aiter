# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tier-file schema: columns, loading, validation, and case helpers.

Tier files are CSVs shared between nodes. Rows are only ever added; every added row
carries a ``reason`` and an ``added_by``. See the GM-123 plan for the tiers.
"""

import csv
import hashlib
import os
from dataclasses import dataclass, field

COLS = [
    "case_id",
    "kind",
    "family",
    "model",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
    "extra",
    "route_gfx942",
    "route_gfx950",
    "modes",
    "sources",
    "reason",
    "added_by",
]
REQUIRED = ["case_id", "kind", "family", "token", "expert", "topk", "modes"]
FUSED_REQUIRED = ["model_dim", "inter_dim", "act_type", "dtype", "q_type", "use_g1u1"]
KINDS = {"fused", "sorting", "routing", "fused-grouped", "triton", "other"}
RUNNABLE_KINDS = {"fused", "sorting", "routing"}
MODES = ("warm", "cold", "first")
QTYPES = {"No", "per_Tensor", "per_Token", "per_1x32", "per_1x128", "per_128x128"}
ACTS = {"Silu", "Swiglu", "Gelu", "Situv2", "silu_clamped", "Relu2", "GeluTanh", ""}
DTYPES = {"bf16", "fp16", "fp32", "fp8", "fp4x2", "i8", "i4x2", "int4", ""}
EXTRA_KEYS_USED = {
    "route_dist",
    "empty_experts",
    "ep_local",
    "num_local_tokens",
    "output_buf",
    "bias",
    "expert_mask",
    "num_shared",
    "gate_mode",
    "swiglu_limit",
    "flat",
    "xbf16",
    "hidden_pad",
    "intermediate_pad",
}


@dataclass
class Case:
    row: dict
    extra: dict = field(default_factory=dict)

    @property
    def id(self):
        return self.row["case_id"]

    def __getitem__(self, k):
        return self.row[k]

    def get(self, k, default=""):
        return self.row.get(k, default)

    @property
    def kind(self):
        return self.row["kind"]

    @property
    def token(self):
        return int(self.row["token"])

    @property
    def modes(self):
        return [m.strip() for m in self.row["modes"].split(",") if m.strip()]


def parse_extra(s):
    """``a=1;b;c=x`` -> {"a": "1", "b": "1", "c": "x"}."""
    out = {}
    for part in (s or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[part] = "1"
    return out


def _flag_on(v):
    return v not in (None, "", "0", "0.0", "False", "None", "none")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_tier(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cases = []
    for r in rows:
        r = {c: r.get(c, "") or "" for c in COLS}
        cases.append(Case(r, parse_extra(r["extra"])))
    return cases


def write_tier(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})


def validate(cases, name="tier", strict_provenance=True):
    """Return (errors, warnings). Errors break the schema; warnings are rows whose
    data aiter's own tables gave us that ``run`` reports as ``invalid``."""
    problems = []
    warnings = []
    seen = set()
    for c in cases:
        r, cid = c.row, c.row["case_id"]
        if cid in seen:
            problems.append(f"{name}: duplicate case_id {cid}")
        seen.add(cid)
        for k in REQUIRED:
            if r[k] == "" and (
                k in ("case_id", "kind", "modes") or r["kind"] in RUNNABLE_KINDS
            ):
                problems.append(f"{name}:{cid}: empty required column {k}")
        if r["kind"] not in KINDS:
            problems.append(f"{name}:{cid}: unknown kind {r['kind']!r}")
        if r["kind"] == "fused":
            for k in FUSED_REQUIRED:
                if r[k] == "" and k != "act_type":
                    problems.append(f"{name}:{cid}: fused row has empty {k}")
            if r["act_type"] == "":
                warnings.append(f"{name}:{cid}: empty act_type (Silu is used)")
        if r["q_type"] not in QTYPES and r["q_type"] != "":
            problems.append(f"{name}:{cid}: unknown q_type {r['q_type']!r}")
        if r["act_type"] not in ACTS:
            problems.append(f"{name}:{cid}: unknown act_type {r['act_type']!r}")
        for k in ("dtype", "q_dtype_a", "q_dtype_w"):
            if r[k] not in DTYPES:
                problems.append(f"{name}:{cid}: unknown {k} {r[k]!r}")
        for k in ("token", "expert", "topk", "model_dim", "inter_dim"):
            if r[k] != "":
                try:
                    v = int(r[k])
                except ValueError:
                    problems.append(f"{name}:{cid}: {k}={r[k]!r} is not an integer")
                    continue
                if r["kind"] in RUNNABLE_KINDS and (v < 0 or (k == "token" and v == 0)):
                    warnings.append(
                        f"{name}:{cid}: {k}={v} is not positive (run: invalid)"
                    )
        bad = [m for m in c.modes if m not in MODES]
        if bad:
            problems.append(f"{name}:{cid}: unknown modes {bad}")
        needs_provenance = strict_provenance and r["added_by"] != "build_tiers"
        if needs_provenance and (not r["reason"] or not r["added_by"]):
            problems.append(f"{name}:{cid}: added row needs reason and added_by")
    return problems, warnings


def case_signature(c):
    """Identity of what a case computes (ignores id, reason, route columns)."""
    r = c.row
    keys = [
        "kind",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]
    ex = ";".join(
        f"{k}={v}" for k, v in sorted(c.extra.items()) if k in EXTRA_KEYS_USED
    )
    return "|".join(str(r[k]) for k in keys) + "|" + ex


def short_hash(s, n=10):
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def ep_local_experts(c):
    """Local expert count for an EP case (global count otherwise)."""
    E = int(c["expert"])
    frac = c.extra.get("ep_local")
    if _flag_on(frac):
        return max(1, int(E * float(frac))) if float(frac) != 1.0 else E
    if _flag_on(c.extra.get("expert_mask")):
        return max(1, E // 2)
    return E


def weight_group_key(c):
    """Cases sharing a key share quantized, shuffled weights (and biases)."""
    r = c.row
    bias = c.extra.get("bias", "")
    has_bias = bias not in ("", "none", "None", "0")
    return (
        r["kind"],
        r["model_dim"],
        r["inter_dim"],
        ep_local_experts(c),
        r["use_g1u1"],
        r["dtype"],
        r["q_dtype_a"],
        r["q_dtype_w"],
        r["q_type"],
        r["act_type"] in ("Swiglu",) or has_bias,
        c.extra.get("gate_mode", ""),
        c.extra.get("hidden_pad", "0"),
        c.extra.get("intermediate_pad", "0"),
    )


def group_id(key):
    return short_hash(repr(key))


def tier_paths(tiers_dir):
    out = {}
    for name in sorted(os.listdir(tiers_dir)):
        if name.startswith("tier") and name.endswith(".csv"):
            out[name[:-4]] = os.path.join(tiers_dir, name)
    return out
