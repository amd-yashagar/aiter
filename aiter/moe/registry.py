# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel registry. Family modules register themselves on import."""

import functools
import importlib
import os

from aiter import logger

from .kernel import MoeKernel, check_numerics

# Imported on the first lookup, so `import aiter.moe` does not pull in FlyDSL.
KERNEL_PACKAGES = ("aiter.ops.flydsl.moe_next",)

_KERNELS: dict[str, MoeKernel] = {}


def register_kernel(kernel):
    """Register a MoeKernel subclass or instance; usable as a class decorator."""
    inst = kernel() if isinstance(kernel, type) else kernel
    if not inst.id:
        raise ValueError(f"{kernel!r} has no id")
    check_numerics(inst.numerics)
    if inst.id in _KERNELS and type(_KERNELS[inst.id]) is not type(inst):
        raise ValueError(f"kernel id {inst.id!r} registered twice")
    _KERNELS[inst.id] = inst
    return kernel


def unregister_kernel(kernel_id: str) -> None:
    _KERNELS.pop(kernel_id, None)


@functools.cache
def _load_packages() -> None:
    for name in KERNEL_PACKAGES:
        try:
            importlib.import_module(name)
        except Exception:
            if os.getenv("AITER_MOE_STRICT", "0") == "1":
                raise
            logger.warning("aiter.moe: cannot import %s", name, exc_info=True)


def get_kernel(kernel_id: str) -> MoeKernel | None:
    _load_packages()
    return _KERNELS.get(kernel_id)


SAME_AS_AITER = {"same_as": "aiter dispatched path"}


def numerics(kernel_id: str) -> dict | None:
    """The kernel's declared rounding points (``MoeKernel.numerics``), with
    inter_quant defaulting to act_quant; a kernel that declares none is "same
    as aiter's dispatched path" (logged once)."""
    kernel = get_kernel(kernel_id)
    if kernel is None:
        return None
    if kernel.numerics is None:
        from .log import log_once

        log_once(
            f"aiter.moe: {kernel_id} declares no numerics; assuming {SAME_AS_AITER}"
        )
        return dict(SAME_AS_AITER)
    out = dict(kernel.numerics)
    if "act_quant" in out:
        out.setdefault("inter_quant", out["act_quant"])
    return out


def kernels() -> dict[str, MoeKernel]:
    _load_packages()
    return dict(_KERNELS)
