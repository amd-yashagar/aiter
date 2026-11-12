# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which GPU product this process runs on, for per-GPU route tables.

MI300X and MI325X are both gfx942 with 304 CUs, so the arch and the CU count
cannot tell them apart; the PCI chip id can.
"""

import functools
import os

from aiter import logger
from aiter.jit.utils.chip_info import _get_pci_chip_id, get_gfx_runtime

# PCI chip id -> route-table name.
CHIP_IDS = {
    0x74A1: "mi300x",
    0x74A5: "mi325x",
    0x75A0: "mi350x",
    0x75A3: "mi355x",
}

# A GPU without its own measured table reads another's; routes are then unverified.
TABLE_ALIASES = {"mi350x": "mi355x"}


@functools.cache
def gpu_name(device: int | None = None) -> str:
    """Route-table name of the GPU (``mi300x``, ``mi325x``, ``mi355x``, ...).

    ``AITER_MOE_GPU`` overrides the detection. An unknown product returns
    ``"<gfx>-<chip id>"``, which has no table, so every call takes aiter's path.
    """
    override = os.getenv("AITER_MOE_GPU")
    if override:
        return override.lower()
    gfx = get_gfx_runtime()
    try:
        chip = _get_pci_chip_id(device)
    except RuntimeError as exc:
        logger.warning("aiter.moe: cannot read the PCI chip id (%s)", exc)
        return f"{gfx}-unknown"
    return CHIP_IDS.get(chip, f"{gfx}-{chip:#06x}")


def table_name(gpu: str) -> tuple[str, bool]:
    """(table to read, whether it was measured on this GPU)."""
    alias = TABLE_ALIASES.get(gpu)
    return (alias, False) if alias else (gpu, True)
