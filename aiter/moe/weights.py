# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Weights of one MoE layer, prepared once, with their per-M routes."""

import threading
from dataclasses import dataclass, field
from typing import Any

import torch

from aiter import dtypes

from .kernel import MoeKernel
from .log import log_once
from .registry import get_kernel
from .routes import Route, RouteTable
from .spec import MoeSpec


@dataclass(frozen=True)
class Resolved:
    """The route for one call key; ``kernel`` None means aiter's path."""

    kernel: MoeKernel | None
    route: Route
    family: str
    reason: str


@dataclass
class MoeWeights:
    """aiter-layout weights (what fused_moe takes) plus kernel state.

    ``fused_moe_kwargs`` holds the prepare-level fused_moe arguments exactly as
    the caller gave them; the fallback passes them back unchanged. ``state``
    maps a kernel id to what its ``prepare_weights`` returned; derived layouts
    live there, beside the aiter-layout tensors, never in place of them.
    """

    spec: MoeSpec
    w1: torch.Tensor
    w2: torch.Tensor
    fused_moe_kwargs: dict[str, Any]
    gpu: str
    table: RouteTable
    topk: int | None = None
    handle: int = 0
    uncovered: frozenset = frozenset()
    state: dict[str, Any] = field(default_factory=dict)
    failed: set = field(default_factory=set)
    _routes: dict = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def kwarg(self, name: str, default=None):
        """A prepare-level fused_moe argument as given (``default`` if unset)."""
        return self.fused_moe_kwargs.get(name, default)

    @property
    def w1_scale(self):
        return self.fused_moe_kwargs.get("w1_scale")

    @property
    def w2_scale(self):
        return self.fused_moe_kwargs.get("w2_scale")

    @property
    def bias1(self):
        return self.fused_moe_kwargs.get("bias1")

    @property
    def bias2(self):
        return self.fused_moe_kwargs.get("bias2")

    def resolve(
        self,
        m: int,
        topk: int,
        hidden_dtype: torch.dtype = dtypes.bf16,
        has_a1_scale: bool = False,
    ) -> Resolved:
        """Host-only, cached per (M, topk, hidden dtype, a1_scale given)."""
        key = (m, topk, hidden_dtype, has_a1_scale)
        hit = self._routes.get(key)
        if hit is None:
            with self._lock:
                hit = self._routes.get(key)
                if hit is None:
                    hit = self._resolve(*key)
                    self._routes[key] = hit
        return hit

    def _resolve(self, m, topk, hidden_dtype, has_a1_scale) -> Resolved:
        family = self.spec.family(m, hidden_dtype, has_a1_scale)
        route = self.table.lookup(self.spec, family, topk, m)
        if route.is_fallback:
            return Resolved(None, route, family, f"route {route.source}")
        kernel = get_kernel(route.kernel)
        if kernel is None:
            reason = f"unknown kernel {route.kernel!r} ({route.source})"
            log_once(f"aiter.moe: {reason}")
            return Resolved(None, route, family, reason)
        if kernel.id in self.failed:
            return Resolved(None, route, family, f"{kernel.id} failed earlier")
        missing = self.uncovered - kernel.handles_kwargs
        if missing:
            reason = f"{kernel.id} does not implement {sorted(missing)}"
            return Resolved(None, route, family, reason)
        reason = kernel.supports(self.spec, m, topk)
        if reason:
            log_once(f"aiter.moe: {kernel.id} declines {self.spec} at M={m}: {reason}")
            return Resolved(None, route, family, f"{kernel.id} declines: {reason}")
        if kernel.id not in self.state:
            self.state[kernel.id] = kernel.prepare_weights(self)
        return Resolved(kernel, route, family, f"route {route.source}")

    def mark_failed(self, kernel: MoeKernel, exc: BaseException, where: str) -> None:
        """Send every later call this weight set's ``kernel`` would serve to
        aiter's path."""
        with self._lock:
            self.failed.add(kernel.id)
            self.state.pop(kernel.id, None)
            self._routes.clear()
        log_once(
            f"aiter.moe: {kernel.id} failed at {where} for {self.spec}; "
            f"using aiter's path from now on: {exc!r}"
        )
