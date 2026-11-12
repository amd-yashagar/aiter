# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel registry. Family modules register themselves on import."""

import functools
import importlib
import os

from aiter import logger

from .kernel import MoeKernel

# Imported on the first lookup, so `import aiter.moe` does not pull in FlyDSL.
KERNEL_PACKAGES = ("aiter.ops.flydsl.moe_next",)

_KERNELS: dict[str, MoeKernel] = {}


def register_kernel(kernel):
    """Register a MoeKernel subclass or instance; usable as a class decorator."""
    inst = kernel() if isinstance(kernel, type) else kernel
    if not inst.id:
        raise ValueError(f"{kernel!r} has no id")
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


def kernels() -> dict[str, MoeKernel]:
    _load_packages()
    return dict(_KERNELS)
