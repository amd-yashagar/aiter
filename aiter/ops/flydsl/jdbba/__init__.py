# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""jagged_dense_bmm (jdbba) v3 — modular multi-variant kernel framework.

Public surface::

    from aiter.ops.flydsl.jdbba import jagged_dense_bmm, JdbbaConfig, list_variants

    out = jagged_dense_bmm(jagged, dense, bias, seq_offsets, max_seq_len)
    # force a specific implementation type / instance:
    out = jagged_dense_bmm(..., variant="lowlevel")
    out = jagged_dense_bmm(..., config=JdbbaConfig(variant="highlevel", uniform_seqlen=True))

Variants are auto-registered on import (see ``variants.py``); the dispatcher
picks the per-shape best (variant + config) from an arch-keyed JSON table, with
a heuristic fallback. New kernel types register with one ``@register`` block.
"""

from __future__ import annotations

from .config import JdbbaConfig, shape_id
from .dispatch import jagged_dense_bmm, resolve_config
from .reference import torch_reference
from .variants import get_variant, list_variants

__all__ = [
    "jagged_dense_bmm",
    "resolve_config",
    "JdbbaConfig",
    "shape_id",
    "torch_reference",
    "list_variants",
    "get_variant",
]
