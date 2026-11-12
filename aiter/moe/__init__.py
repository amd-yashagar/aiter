# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE entry point with per-GPU routing to new kernels and aiter's path as fallback.

    weights = aiter.moe.prepare(w1, w2, w1_scale=s1, w2_scale=s2,
                                quant_type=QuantType.per_1x128)
    out = aiter.moe.forward(hidden_states, weights, topk_weight, topk_ids)

``forward`` returns what ``aiter.fused_moe.fused_moe`` returns for the same
tensors. Environment: ``AITER_MOE_DISABLE=1`` sends every call to aiter's
path; ``AITER_MOE_ROUTES=<csv or kernel[:config]>`` replaces the route table
(tests); ``AITER_MOE_STRICT=1`` raises kernel errors instead of falling back
(tests); ``AITER_MOE_GPU=<name>`` overrides GPU detection.
"""

from .entry import CALL_KWARGS, PREPARE_KWARGS, explain, fallback, forward, prepare
from .kernel import MoeCall, MoeKernel
from .registry import get_kernel, kernels, register_kernel, unregister_kernel
from .spec import FAMILIES, MoeSpec
from .weights import MoeWeights

__all__ = [
    "CALL_KWARGS",
    "FAMILIES",
    "PREPARE_KWARGS",
    "MoeCall",
    "MoeKernel",
    "MoeSpec",
    "MoeWeights",
    "explain",
    "fallback",
    "forward",
    "get_kernel",
    "kernels",
    "prepare",
    "register_kernel",
    "unregister_kernel",
]
