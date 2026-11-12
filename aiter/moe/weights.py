# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-weight-set routing state, built lazily on the first eager call."""

import threading
import weakref
from dataclasses import dataclass, field
from typing import Any

import torch

from .kernel import MoeKernel
from .log import log_once
from .registry import get_kernel
from .routes import Route, RouteTable
from .spec import MoeSpec


@dataclass(frozen=True)
class Resolved:
    """The route for one call key; ``kernel`` None means today's path."""

    kernel: MoeKernel | None
    route: Route
    family: str
    reason: str


@dataclass
class MoeWeights:
    """One weight set's state: weak references to the caller's aiter-layout
    tensors (never a copy, and never keeping them alive), the layer's
    fused_moe arguments, and kernel state. A kernel reads the tensors of the
    call it serves from ``MoeCall``.

    ``state`` maps a kernel id to what its ``prepare_weights`` returned: small
    state only (scale reorders, tile metadata, per-expert pointers, compiled
    handles). ``versions`` are the tensors' ``_version`` when it was built.
    """

    spec: MoeSpec
    w1_ref: weakref.ref
    w2_ref: weakref.ref
    fused_moe_kwargs: dict[str, Any]
    gpu: str
    table: RouteTable
    versions: tuple = ()
    uncovered: frozenset = frozenset()
    state: dict[str, Any] = field(default_factory=dict)
    failed: set = field(default_factory=set)
    compiled: set = field(default_factory=set)
    _routes: dict = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def kwarg(self, name: str, default=None):
        """A layer-level fused_moe argument as the caller gave it."""
        v = self.fused_moe_kwargs.get(name)
        return default if v is None else v

    @property
    def w1(self) -> torch.Tensor | None:
        return self.w1_ref()

    @property
    def w2(self) -> torch.Tensor | None:
        return self.w2_ref()

    def resolve(
        self, m, topk, hidden_dtype, out_dtype, a_dtype, has_a1_scale, build=True
    ) -> Resolved | None:
        """Host-only, cached per call key; lock-free on a hit. With
        ``build=False`` (capture, tracing) a miss returns None."""
        key = (m, topk, hidden_dtype, out_dtype, a_dtype, has_a1_scale)
        hit = self._routes.get(key)
        if hit is None and not build:
            return None
        if hit is None:
            with self._lock:
                hit = self._routes.get(key)
                if hit is None:
                    hit = self._resolve(m, topk, a_dtype)
                    self._routes[key] = hit
        return hit

    def _kernel_for(self, route: Route):
        """(kernel, reason): the route's kernel if it can serve this layer."""
        if route.is_fallback:
            return None, f"route {route.source}"
        kernel = get_kernel(route.kernel)
        if kernel is None:
            log_once(
                f"aiter.moe: route {route.source} names unknown kernel {route.kernel!r}"
            )
            return None, f"unknown kernel {route.kernel!r} ({route.source})"
        if kernel.id in self.failed:
            return None, f"{kernel.id} failed earlier"
        missing = self.uncovered - kernel.handles_kwargs
        if missing:
            return None, f"{kernel.id} does not implement {sorted(missing)}"
        return kernel, ""

    def _resolve(self, m, topk, a_dtype) -> Resolved:
        family = self.spec.family(a_dtype)
        route = self.table.lookup(self.spec, family, topk, m)
        kernel, reason = self._kernel_for(route)
        if kernel is None:
            return Resolved(None, route, family, reason)
        reason = kernel.supports(self.spec, m, topk)
        if reason:
            log_once(f"aiter.moe: {kernel.id} declines {self.spec} at M={m}: {reason}")
            return Resolved(None, route, family, f"{kernel.id} declines: {reason}")
        self._prepare(kernel)
        self._compile(kernel, route, m, topk)
        if kernel.id in self.failed:
            return Resolved(None, route, family, f"{kernel.id} failed to compile")
        return Resolved(kernel, route, family, f"route {route.source}")

    def _prepare(self, kernel):
        if kernel.id not in self.state:
            self.state[kernel.id] = kernel.prepare_weights(self)

    def _compile(self, kernel, route, m, topk):
        key = (kernel.id, route.config, route.source)
        if key in self.compiled:
            return
        try:
            kernel.warmup(self, m, topk, route.config)
            self.compiled.add(key)
        except Exception as exc:
            from .router import strict

            if strict():
                raise
            self.mark_failed(kernel, exc, f"compile M={m}")

    def compile_all(self, topk: int) -> None:
        """Compile every kernel and config this GPU's table maps for this
        shape and topk, across all M buckets, so a later capture finds them."""
        for row in self.table.rows:
            if not row.matches_shape(self.spec, topk):
                continue
            kernel, _ = self._kernel_for(row.route)
            if kernel is None or kernel.supports(self.spec, row.m_min, topk):
                continue
            with self._lock:
                self._prepare(kernel)
                self._compile(kernel, row.route, row.m_min, topk)

    def ready(self, kernel, route) -> bool:
        return (kernel.id, route.config, route.source) in self.compiled

    def mark_failed(self, kernel: MoeKernel, exc: BaseException, where: str) -> None:
        """Send every later call this weight set's ``kernel`` would serve to
        today's path. Never called during a capture."""
        with self._lock:
            self.failed.add(kernel.id)
            self.state.pop(kernel.id, None)
            self._routes.clear()
        log_once(
            f"aiter.moe: {kernel.id} failed at {where} for {self.spec}; "
            f"using today's path from now on: {exc!r}"
        )
