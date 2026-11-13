# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_moe dispatch follows the live GPU, not GPU_ARCHS.

``get_gfx()`` returns the last entry of a multi-arch ``GPU_ARCHS`` ("gfx950;gfx942" gives
gfx942), whatever the process runs on. The 1-stage kernel table, the xbf16 check, the
FLAT guard, the asm top-k softmax gate and the SiTUv2/Swiglu a16w4 gate used it, so a
gfx950 process started with ``GPU_ARCHS="gfx950;gfx942"`` took gfx942's 1-stage entry
for int8 per-token (``run_1stage``) while its tuned-row lookup, which already used
``get_gfx_runtime()``, keyed on gfx950.
"""

import pytest

import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes

TOKEN, MODEL_DIM, INTER_DIM, EXPERTS, TOPK = 64, 2560, 256, 8, 2


def _metadata(monkeypatch, build, runtime):
    monkeypatch.setattr(fm, "get_gfx", lambda: build)
    monkeypatch.setattr(fm, "get_gfx_runtime", lambda: runtime)
    monkeypatch.setattr(fm, "get_cu_num", lambda: 256)
    fm.get_2stage_cfgs.cache_clear()
    try:
        return fm.get_2stage_cfgs(
            TOKEN,
            MODEL_DIM,
            INTER_DIM,
            EXPERTS,
            TOPK,
            dtypes.bf16,
            dtypes.i8,
            dtypes.i8,
            QuantType.per_Token,
            True,
            ActivationType.Silu,
            False,
            0,
            0,
            is_shuffled=True,
        )
    finally:
        fm.get_2stage_cfgs.cache_clear()


@pytest.mark.parametrize(
    "build,runtime,one_stage",
    [
        # int8 per-token has a 1-stage entry on gfx942 only.
        ("gfx942", "gfx950", False),
        ("gfx950", "gfx942", True),
        ("gfx950", "gfx950", False),
        ("gfx942", "gfx942", True),
    ],
)
def test_one_stage_table_follows_the_runtime_arch(
    monkeypatch, build, runtime, one_stage
):
    assert _metadata(monkeypatch, build, runtime).run_1stage is one_stage
