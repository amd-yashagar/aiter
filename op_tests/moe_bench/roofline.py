# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Hardware roofline of one MoE case on one GPU.

Reuse and sources (nothing here is a new hardware model):

* Hardware tables are ported from WaveScope (attlas's analyzer),
  ``WaveScope/src/analysis/rooflinePeaks.js``: ``SPEC_CACHE_BW`` (HBM, L2, vL1D and LDS
  GB/s per SKU), ``datasheetCompute`` (dense per-dtype MFMA peaks) and ``EMPIRICAL_PEAKS``
  (measured MI300X HBM, 4300 GB/s). Its FLOP and byte conventions follow
  rocprofiler-compute ``gfx942/0400_roofline.yaml`` (see ``analysis/roofline.js``).
  WaveScope's roofline code ingests hardware counters of a captured kernel; neither it nor
  attlas has an analytic predictor, so the per-case traffic model below is built here from
  those numbers.
* Package facts (256 MiB last-level Infinity Cache, 4 MiB L2 per XCD, 64 KiB LDS per CU on
  gfx942, 304/256 CUs) come from the vault note ``kn-reference-mi-arch-matrix``.
* Achievable HBM below the datasheet: WaveScope's measured MI300X figure; for gfx950 the
  vault's ``kn-hardware-mi350-achievable-hbm-bandwidth`` (64.6% of 8 TB/s, MI350X triad).
* Cache-defeated HBM as the honest ceiling for cold traffic and best-of-existing as the
  bar: ``kn-method-honest-kernel-baselines`` (T0 dispatch .. T2 best existing).
* Nothing in those sources gives a MALL (Infinity Cache) bandwidth. Warm traffic that fits in
  the 256 MiB MALL is therefore capped at the L2 figure, which MALL cannot exceed, so the
  bound stays a lower bound. Level choice by ratio is not hardware-measured at MALL.

Model (fused lower bound): time = max over levels of bytes_level / BW_level, with
compute = FLOPs / MFMA peak. ``roofline_us`` assumes perfect overlap of all levels and no
HBM round trip for the stage-1 to stage-2 intermediate (a fused kernel); ``roofline_staged_us``
keeps the two GEMMs as separate stages (sum of per-stage maxima), which is what today's
2-stage backends can at best reach; ``serial_us`` (sum of levels, no overlap) shows what
pipelining can recover. Tile sizes are model parameters (BM_MAX, BN, WAVE_TILE), not
hardware facts.
"""

import math
import re

MFMA_KEYS = ("MFMA_FP8", "MFMA_I8", "MFMA_BF16", "MFMA_FP16", "MFMA_F6F4")

# GB/s and GFLOP/s, WaveScope rooflinePeaks.js SPEC_CACHE_BW and datasheetCompute.
SPEC = {
    "MI300X": {
        "HBM": 4224,
        "L2": 26655,
        "vL1D": 34191,
        "LDS": 75657,
        "HBM_DATASHEET": 5325,
        "MFMA_BF16": 1307400,
        "MFMA_FP16": 1307400,
        "MFMA_FP8": 2614900,
        "MFMA_I8": 2614900,
        "MFMA_FP32": 163400,
    },
    "MI325X": {
        "HBM": 4641,
        "L2": 27882,
        "vL1D": 36843,
        "LDS": 76315,
        "HBM_DATASHEET": 6000,
        "MFMA_BF16": 1307400,
        "MFMA_FP16": 1307400,
        "MFMA_FP8": 2614900,
        "MFMA_I8": 2614900,
        "MFMA_FP32": 163400,
    },
    "MI355X": {
        "HBM": 6198,
        "L2": 34800,
        "vL1D": 38263,
        "LDS": 67368,
        "HBM_DATASHEET": 8000,
        "MFMA_BF16": 2516600,
        "MFMA_FP16": 2516600,
        "MFMA_FP8": 5033200,
        "MFMA_I8": 5033200,
        "MFMA_F6F4": 10000000,
        "MFMA_FP32": 157300,
    },
}
# WaveScope EMPIRICAL_PEAKS (MI300X measured); others scale the datasheet by a measured ratio.
HBM_ACHIEVABLE = {
    "MI300X": 4300.0,
    "MI325X": 4300.0 * 6000 / 5325,  # MI300X measured fraction applied to 6.0 TB/s
    "MI355X": 8000 * 0.646,  # kn-hardware-mi350-achievable-hbm-bandwidth (MI350X triad)
}
MALL_BYTES = 256 << 20  # kn-reference-mi-arch-matrix: 256 MiB on all three GPUs
CUS = {"MI300X": 304, "MI325X": 304, "MI355X": 256}
LDS_CAP = {"MI300X": 64 << 10, "MI325X": 64 << 10, "MI355X": 160 << 10}

BM_MAX = 128  # largest M tile a MoE GEMM here uses (CK/ASM/FlyDSL block_m)
BN = 256
WAVE_TILE = 64  # per-wave output tile edge for the LDS traffic estimate

BYTES = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp32": 4.0,
    "fp8": 1.0,
    "i8": 1.0,
    "fp4x2": 0.5,
    "i4x2": 0.5,
}


def sku_of(gpu_name):
    m = re.search(r"MI(\d{3})([AX]?)", gpu_name or "", re.IGNORECASE)
    if not m:
        return None
    key = f"MI{m.group(1)}{m.group(2).upper() or 'X'}"
    return key if key in SPEC else None


def mfma_key(aq, wq, gfx):
    """Which MFMA rate the GEMM inputs run at (weights decoded to the activation type)."""
    if aq == "fp4x2" and wq == "fp4x2" and gfx == "gfx950":
        return "MFMA_F6F4"
    if aq == "fp8":
        return "MFMA_FP8"
    if aq == "i8":
        return "MFMA_I8"
    return "MFMA_FP16" if aq == "fp16" else "MFMA_BF16"


def _weight_bytes_per_expert(D, I, g1u1, wq, qtype):
    n1 = (2 if g1u1 else 1) * I
    w = (n1 * D + D * I) * BYTES.get(wq, 2.0)
    scale = 0.0
    if qtype == "per_Token" or qtype == "per_Tensor":
        scale = (n1 + D) * 4.0
    elif qtype == "per_1x128":
        scale = ((n1 * D) + (D * I)) / (128 * 128) * 4.0
    elif qtype == "per_1x32":
        scale = ((n1 * D) + (D * I)) / 32.0
    return w + scale


def model(case, sku, gfx, mode, spec_hbm=False):
    """Roofline dict for one case on one GPU in ``mode`` (``warm`` or ``cold``).

    ``case`` needs: token, model_dim, inter_dim, expert, topk, use_g1u1, dtype,
    q_dtype_a, q_dtype_w, q_type (strings as in the tier files) and optionally
    ep_local (fraction of local experts).
    """
    peaks = SPEC[sku]
    M = int(float(case["token"]))
    D, I = int(float(case["model_dim"])), int(float(case["inter_dim"]))
    E, topk = int(float(case["expert"])), int(float(case["topk"]))
    g1u1 = str(case["use_g1u1"]) in ("1", "True", "1.0")
    aq = case.get("q_dtype_a") or case["dtype"]
    wq = case.get("q_dtype_w") or case["dtype"]
    qtype = case.get("q_type") or "No"
    ep = float(case.get("ep_local") or 1.0)
    el = max(1, int(E * ep))
    act_b = BYTES.get(aq, 2.0)
    out_b = BYTES.get(case["dtype"], 2.0)
    routes = M * topk * ep
    active = max(1.0, min(el, routes))
    n_e = routes / active

    flops1 = 2.0 * routes * D * I * (2 if g1u1 else 1)
    flops2 = 2.0 * routes * I * D
    key = mfma_key(aq, wq, gfx)
    compute_peak = peaks.get(key, peaks["MFMA_BF16"]) * 1e9
    w_per = _weight_bytes_per_expert(D, I, g1u1, wq, qtype)
    w_touched = active * w_per
    hbm_bw = (peaks["HBM_DATASHEET"] if spec_hbm else HBM_ACHIEVABLE[sku]) * 1e9
    l2_bw = peaks["L2"] * 1e9
    lds_bw = peaks["LDS"] * 1e9

    a1 = M * D * act_b
    out = M * D * out_b
    inter = routes * I * act_b
    n1 = (2 if g1u1 else 1) * I
    mtiles = math.ceil(n_e / BM_MAX)
    # L2 -> CU weight traffic: every expert's weights once per M tile of its tokens.
    w_l2 = active * mtiles * w_per
    act_l2_1 = routes * D * act_b * math.ceil(n1 / BN)
    act_l2_2 = routes * I * act_b * math.ceil(D / BN)
    lds1 = flops1 / 2.0 * (2.0 / WAVE_TILE) * act_b
    lds2 = flops2 / 2.0 * (2.0 / WAVE_TILE) * act_b

    warm_resident = mode == "warm" and w_touched <= MALL_BYTES
    hbm_bytes = a1 + out + (0.0 if warm_resident else w_touched)
    mall_bytes = w_touched if warm_resident else 0.0
    mall_bw = l2_bw  # MALL bandwidth is not documented; L2 is its ceiling

    def level_times(f, w_l2_part, act_l2_part, lds_part, hbm_part, mall_part):
        return {
            "compute": f / compute_peak,
            "LDS": lds_part / lds_bw,
            "L2": (w_l2_part + act_l2_part) / l2_bw,
            "MALL": mall_part / mall_bw,
            "HBM": hbm_part / hbm_bw,
        }

    half = 0.5
    w1_share = (n1 * D) / (n1 * D + D * I)
    t1 = level_times(
        flops1,
        w_l2 * w1_share,
        act_l2_1,
        lds1,
        (a1 + (0.0 if warm_resident else w_touched * w1_share)),
        mall_bytes * w1_share,
    )
    t2 = level_times(
        flops2,
        w_l2 * (1 - w1_share),
        act_l2_2,
        lds2,
        (out + (0.0 if warm_resident else w_touched * (1 - w1_share))),
        mall_bytes * (1 - w1_share),
    )
    # staged: the intermediate is written to and read back from HBM-backed memory
    t1["HBM"] += inter * half / hbm_bw
    t2["HBM"] += inter * half / hbm_bw
    fused = level_times(
        flops1 + flops2,
        w_l2,
        act_l2_1 + act_l2_2,
        lds1 + lds2,
        hbm_bytes,
        mall_bytes,
    )
    bound = max(fused, key=fused.get)
    roof = fused[bound]
    staged = max(t1.values()) + max(t2.values())
    serial = sum(fused.values())
    return {
        "roofline_us": roof * 1e6,
        "roofline_staged_us": staged * 1e6,
        "roofline_serial_us": serial * 1e6,
        "bound": bound,
        "mfma": key,
        "level_us": {k: v * 1e6 for k, v in fused.items()},
        "weights_touched_bytes": w_touched,
        "weights_in_mall": warm_resident,
        "flops": flops1 + flops2,
    }


def case_ep(case_row, extra):
    frac = extra.get("ep_local")
    if frac not in (None, "", "0"):
        return float(frac)
    if extra.get("expert_mask") not in (None, "", "0"):
        return 0.5
    return 1.0
