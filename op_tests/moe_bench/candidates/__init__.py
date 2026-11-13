# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pluggable candidate sources for ``bar`` mode.

A *source* knows one backend: it lists the candidates that can compute a case on this
GPU and builds each candidate's timed closure. Every candidate is timed over the same
region: bf16 hidden states plus router outputs (topk ids and weights) to the final
output, including the backend's own sorting and activation quant, excluding one-time
weight preparation (layout, shuffle, quant), which runs untimed once per (weight group,
layout) through ``Ctx.layout``.

Add a backend by subclassing ``Source`` and calling ``register``; load extra modules
with ``--source-module pkg.mod`` (the module registers its sources on import).
"""

import importlib

REGISTRY = {}


class HarnessError(RuntimeError):
    """A candidate could not be run or checked for a reason that is the harness's, not the
    kernel's: its row is a ``harness-error`` and the case's bar says it is incomplete.
    """


class Ctx:
    """Everything a candidate needs for one case; built by the bar worker."""

    def __init__(
        self, case, S, aw, bias, mask, hidden, tw, ti, nlt, n_valid, kwargs, tmpdir
    ):
        self.case, self.S = case, S
        self.aw, self.bias, self.mask = aw, bias, mask
        self.hidden, self.tw, self.ti = hidden, tw, ti
        self.nlt, self.n_valid = nlt, n_valid
        self.kwargs = kwargs  # the dispatch call's fused_moe keyword arguments
        self.tmpdir = tmpdir
        self.layouts = {}  # layout name -> prepared weights (untimed, once per group)
        self.dispatch_args = None  # get_2stage_cfgs arguments of the dispatch call
        self.dispatch_md = None
        self.group_cache = {}  # shared by every case of the weight group
        self.ref_weights = None  # unshuffled quantized weights and scales (new mode)


class Candidate:
    """One way to compute a case.

    ``source`` is the backend family, ``entry`` the API called, ``kernels`` the kernel
    names or config, ``layout`` the weight layout it needs (prepared untimed).
    """

    source = ""
    entry = ""
    group = ""  # screening group, e.g. a block_m or "1stage"

    def __init__(self, cid, kernels, layout="default", meta=None):
        self.id = cid
        self.kernels = kernels
        self.layout = layout
        self.meta = meta or {}

    def describe(self):
        return {
            "source": self.source,
            "entry": self.entry,
            "kernels": self.kernels,
            "layout": self.layout,
        }

    def prepare(self, ctx):
        """Untimed setup (forcing a config, building a layout). May raise."""

    def call(self, ctx):
        """Return a zero-argument closure; its work is the timed region."""
        raise NotImplementedError

    def release(self, ctx):
        """Undo prepare() side effects."""


class Source:
    name = ""

    def skip_reason(self, ctx):
        """Why this backend does not apply to the case, or None."""
        return

    def enumerate(self, ctx):
        """All candidates for the case (cheap, no timing)."""
        return []

    def screen(self, ctx, cands, runner):
        """Narrow ``cands`` to those worth a full timing; ``runner`` measures one
        candidate quickly. Default: all of them."""
        return list(cands)


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def load_sources(names, modules=()):
    from . import fused_cfg, gluon_a16w4, others  # noqa: F401  (register built-ins)

    for m in modules:
        importlib.import_module(m)
    wanted = names or list(REGISTRY)
    return [REGISTRY[n]() for n in wanted if n in REGISTRY]
