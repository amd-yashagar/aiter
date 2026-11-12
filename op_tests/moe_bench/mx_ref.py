# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MX (per-1x32 e8m0) activation quantization reference, one function per scale rule.

``round_up`` is ``ceil_pow2(amax / max_pos)`` (the HIP per_1x32 quant, aiter's default);
``even`` is ``(amax + round_bias) & exp_mask`` then floor (the Triton fused MoE sort
quant's). The e8m0 scale comes from aiter's own oracle
``aiter.utility.fp4_utils.f32_to_mx_e8m0_scale(mode=...)``; the element conversion is
fp4 e2m1 (round to nearest even, saturating, via aiter's torch helper) or fp8 e4m3
(the chip's flavor, clamped to its max). Tests compare both with aiter's ops bit for bit.
"""

import torch

RULES = ("round_up", "even")


def _mode(rule):
    from aiter.utility.mx_types import MxScaleRoundModeInt

    return {"round_up": MxScaleRoundModeInt.RoundUp, "even": MxScaleRoundModeInt.Even}[
        rule
    ]


def _fp8_flavor():
    from aiter.jit.utils.chip_info import get_gfx

    return "fnuz" if get_gfx() == "gfx942" else "fn"


def mx_quant_ref(x, fmt, rule):
    """Quantize ``x`` (rows, K) per 32 along K; returns a dict.

    ``deq``  fp32 (rows, K), the dequantized values (what a GEMM consumes)
    ``scale`` uint8 (rows, K/32), the e8m0 bytes
    ``q``    the quantized values: uint8 (rows, K/2) packed fp4, or the fp8 tensor
    """
    from aiter.utility import fp4_utils
    from aiter.utility.mx_types import MxDtypeInt

    if rule not in RULES:
        raise ValueError(f"unknown MX scale rule {rule!r}; known: {RULES}")
    if fmt not in ("fp4", "fp8"):
        raise ValueError(f"unknown MX format {fmt!r}")
    rows, k = x.shape
    if k % 32:
        raise ValueError(f"K={k} is not a multiple of 32")
    xb = x.float().reshape(rows, k // 32, 32)
    amax = xb.abs().amax(-1)
    if fmt == "fp4":
        dtype, max_pos = MxDtypeInt.FP4_E2M1, 6.0
    elif _fp8_flavor() == "fnuz":
        dtype, max_pos = MxDtypeInt.FP8_E4M3_FNUZ, 240.0
    else:
        dtype, max_pos = MxDtypeInt.FP8_E4M3, 448.0
    e8 = fp4_utils.f32_to_mx_e8m0_scale(amax, mode=_mode(rule), dtype=dtype)
    scale = fp4_utils.e8m0_to_f32(e8)
    y = xb / scale.unsqueeze(-1)
    if fmt == "fp4":
        codes = fp4_utils._f32_to_floatx_unpacked(y.reshape(rows, k), 2, 1)
        packed = fp4_utils.pack_uint4(codes)
        vals = fp4_utils.mxfp4_to_f32(packed.clone()).reshape(rows, k // 32, 32)
        q = packed
    else:
        from aiter import dtypes

        q = y.clamp(-max_pos, max_pos).reshape(rows, k).to(dtypes.fp8)
        vals = q.float().reshape(rows, k // 32, 32)
    deq = (vals * scale.unsqueeze(-1)).reshape(rows, k)
    return {"deq": deq, "scale": e8.view(torch.uint8), "q": q}
