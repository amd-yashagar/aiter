# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-GPU route tables: (gpu, family, shape, M range) -> kernel id or ``aiter``.

One CSV per GPU in ``aiter/moe/routes/<gpu>.csv``. Columns:

    family, model_dim, inter_dim, experts, topk, act, doweight_stage1,
    m_min, m_max, kernel, config, verified, note

``*`` in a key column matches anything. ``experts`` is the local expert
count. ``m_min``/``m_max`` are inclusive true token counts. The first
matching row wins, so specific rows go before general ones. ``kernel`` is a
registered kernel id or ``aiter``; ``config`` is a name the kernel resolves
(its per-GPU tile config); ``verified`` names the gate run that admitted the
row. No matching row means ``aiter``.
"""

import csv
import functools
import os
from dataclasses import dataclass
from pathlib import Path

from aiter import logger

from .spec import MoeSpec

ROUTES_DIR = Path(__file__).resolve().parent / "routes"
FALLBACK = "aiter"
COLUMNS = (
    "family",
    "model_dim",
    "inter_dim",
    "experts",
    "topk",
    "act",
    "doweight_stage1",
    "m_min",
    "m_max",
    "kernel",
    "config",
    "verified",
    "note",
)
M_MAX = 1 << 31


@dataclass(frozen=True)
class Route:
    kernel: str
    config: str = ""
    source: str = ""

    @property
    def is_fallback(self) -> bool:
        return self.kernel == FALLBACK

    @classmethod
    def parse(cls, text: str, source: str) -> "Route":
        kernel, _, config = text.strip().partition(":")
        return cls(kernel, config, source)


FALLBACK_ROUTE = Route(FALLBACK, "", "no route")


@dataclass(frozen=True)
class RouteRow:
    family: str = "*"
    model_dim: str = "*"
    inter_dim: str = "*"
    experts: str = "*"
    topk: str = "*"
    act: str = "*"
    doweight_stage1: str = "*"
    m_min: int = 0
    m_max: int = M_MAX
    route: Route = FALLBACK_ROUTE

    def matches(self, spec: MoeSpec, family: str, topk: int, m: int) -> bool:
        return (
            self.m_min <= m <= self.m_max
            and (self.family == "*" or self.family == family)
            and self.matches_shape(spec, topk)
        )

    def matches_shape(self, spec: MoeSpec, topk: int) -> bool:
        """The row's shape and layer columns, ignoring M and family."""

        def eq(field, value):
            return field == "*" or field == str(value)

        return (
            eq(self.model_dim, spec.model_dim)
            and eq(self.inter_dim, spec.inter_dim)
            and eq(self.experts, spec.experts)
            and eq(self.topk, topk)
            and (self.act == "*" or self.act.lower() == spec.act.name.lower())
            and eq(self.doweight_stage1, int(spec.doweight_stage1))
        )


class RouteTable:
    def __init__(self, rows: list[RouteRow], source: str):
        self.rows = rows
        self.source = source

    @classmethod
    def load(cls, path: str | os.PathLike) -> "RouteTable":
        path = str(path)
        rows = []
        if not os.path.exists(path):
            return cls(rows, path)
        name = os.path.basename(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(
                line for line in f if line.strip() and not line.startswith("#")
            )
            missing = set(COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            for i, r in enumerate(reader, start=2):
                try:
                    keys = {c: r[c].strip() for c in COLUMNS[:7]}
                    rows.append(
                        RouteRow(
                            **keys,
                            m_min=int(r["m_min"]),
                            m_max=int(r["m_max"]),
                            route=Route(
                                r["kernel"].strip(),
                                (r["config"] or "").strip(),
                                f"{name}:{i}",
                            ),
                        )
                    )
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"{path}:{i}: bad route row: {exc}") from exc
        return cls(rows, path)

    @classmethod
    def override(cls, routes) -> "RouteTable":
        """A test-only table from ``AITER_MOE_ROUTES``.

        ``routes`` is a RouteTable; a CSV path (``*.csv`` or an existing
        file); ``"<kernel>[:<config>]"`` or
        ``"aiter"`` for every call; or a dict whose keys are a token count, an
        inclusive ``(m_min, m_max)`` pair or ``"*"``, and whose values are
        ``"<kernel>[:<config>]"`` or ``"aiter"``.
        """
        if isinstance(routes, RouteTable):
            return routes
        if isinstance(routes, (str, os.PathLike)):
            if str(routes).endswith(".csv") or os.path.isfile(routes):
                return cls.load(routes)
            return cls(
                [RouteRow(route=Route.parse(str(routes), "override"))], "override"
            )
        if isinstance(routes, dict):
            rows = []
            for key, value in routes.items():
                if key == "*":
                    lo, hi = 0, M_MAX
                elif isinstance(key, tuple):
                    lo, hi = key
                else:
                    lo = hi = int(key)
                rows.append(
                    RouteRow(
                        m_min=lo, m_max=hi, route=Route.parse(value, f"override {key}")
                    )
                )
            return cls(rows, "override")
        raise TypeError(
            f"routes must be a table, path, kernel id or dict, not {routes!r}"
        )

    def lookup(self, spec: MoeSpec, family: str, topk: int, m: int) -> Route:
        for row in self.rows:
            if row.matches(spec, family, topk, m):
                return row.route
        return FALLBACK_ROUTE


@functools.cache
def table_for(gpu: str) -> RouteTable:
    """The shipped route table for ``gpu``."""
    from .gpu import table_name

    name, measured = table_name(gpu)
    if not measured:
        logger.warning(
            "aiter.moe: no route table measured on %s; using %s's (unverified)",
            gpu,
            name,
        )
    return RouteTable.load(ROUTES_DIR / f"{name}.csv")


def resolve_table(gpu: str) -> RouteTable:
    """``AITER_MOE_ROUTES`` (tests and kernel bring-up), else the GPU's table."""
    routes = os.getenv("AITER_MOE_ROUTES")
    if routes:
        return RouteTable.override(routes)
    return table_for(gpu)
