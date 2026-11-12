# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routing behind aiter.fused_moe to new per-GPU MoE kernels (internal).

``aiter.fused_moe.fused_moe`` stays the entry. At the top of its body it asks
``router.try_route`` whether a new kernel serves the call; if not, today's path
runs unchanged. Environment:

- ``AITER_MOE_DISABLE=1``: no routing (checked once per process; call
  ``refresh()`` after changing AITER_MOE_* variables in-process).
- ``AITER_MOE_STRICT=1``: raise router and kernel errors (tests, climbs).
- ``AITER_MOE_ROUTES=<csv or kernel[:config]>``: replace the route table (tests).
- ``AITER_MOE_GPU=<name>``: override GPU detection.
- ``AITER_MOE_DEBUG=1``: log each route decision once per key.
"""

from .kernel import MoeCall, MoeKernel, compile_only
from .registry import get_kernel, kernels, numerics, register_kernel, unregister_kernel
from .router import last_route, refresh
from .spec import FAMILIES, MoeSpec
from .weights import MoeWeights

__all__ = [
    "FAMILIES",
    "MoeCall",
    "MoeKernel",
    "MoeSpec",
    "MoeWeights",
    "compile_only",
    "get_kernel",
    "kernels",
    "last_route",
    "numerics",
    "refresh",
    "register_kernel",
    "unregister_kernel",
]
