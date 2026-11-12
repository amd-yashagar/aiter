# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of the aiter.moe kernels: launch wrappers, compile caches, tile
configs, and the MoeKernel classes. Importing this package imports every
``gfx*_*`` family module, which registers its kernels with aiter.moe; kernel
bodies are in kernels/moe_next/."""

import importlib
import os
import pkgutil

from aiter import logger

FAMILY_PREFIX = "gfx"


def _import_families() -> None:
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if not info.name.startswith(FAMILY_PREFIX):
            continue
        try:
            importlib.import_module(f"{__name__}.{info.name}")
        except Exception:
            if os.getenv("AITER_MOE_STRICT", "0") == "1":
                raise
            logger.warning(
                "aiter.moe: cannot import kernel family %s; its routes fall back",
                info.name,
                exc_info=True,
            )


_import_families()
