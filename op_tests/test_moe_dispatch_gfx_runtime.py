# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_moe dispatch follows the live GPU, not GPU_ARCHS (gfx950 and gfx942 sides).

``get_gfx()`` returns the last entry of a multi-arch ``GPU_ARCHS`` ("gfx950;gfx942" gives
gfx942), whatever the process runs on. The 1-stage kernel table, the xbf16 check, the
FLAT guard, the asm top-k softmax gate and the SiTUv2/Swiglu a16w4 gate used it, so a
gfx950 process started with ``GPU_ARCHS="gfx950;gfx942"`` took gfx942's 1-stage entry
for int8 per-token (``run_1stage``) while its tuned-row lookup, which already used
``get_gfx_runtime()``, keyed on gfx950.
"""

import inspect
import json
import os
import re
import subprocess
import sys

import pytest

import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime

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


# --- both directions, more routes, the other dispatch modules, and the real environment ---

CASES = [
    ("fp8 blockscale", "fp8", "fp8", "per_1x128", 64, "Gelu"),
    ("fp8 blockscale", "fp8", "fp8", "per_1x128", 4096, "Gelu"),
    ("int8 per_Token", "i8", "i8", "per_Token", 64, "Silu"),
    ("int8 per_Token", "i8", "i8", "per_Token", 512, "Gelu"),
    ("fp8 per_Token", "fp8", "fp8", "per_Token", 64, "Silu"),
    ("fp8 per_Tensor", "fp8", "fp8", "per_Tensor", 512, "Silu"),
    ("bf16", "bf16", "bf16", "No", 64, "Silu"),
]
SHAPE = (4096, 512, 64, 8)


def _route(case, gfx_for_build, gfx_runtime):
    _label, qa, qw, qt, tok, act = case
    saved = (fm.get_gfx, fm.get_gfx_runtime)
    fm.get_gfx = lambda: gfx_for_build
    fm.get_gfx_runtime = lambda: gfx_runtime
    fm.get_2stage_cfgs.cache_clear()
    try:
        m = fm.get_2stage_cfgs(
            tok,
            SHAPE[0],
            SHAPE[1],
            SHAPE[2],
            SHAPE[3],
            dtypes.bf16,
            getattr(dtypes, qa),
            getattr(dtypes, qw),
            getattr(QuantType, qt),
            True,
            getattr(ActivationType, act),
            False,
            0,
            0,
        )
        s1 = getattr(getattr(m.stage1, "func", m.stage1), "__name__", "full_impl")
        return f"{'1stage' if m.run_1stage else '2stage'}:{s1}:bm{m.block_m}"
    except (
        NotImplementedError,
        ValueError,
        RuntimeError,
        KeyError,
    ) as e:  # a refusal is a decision too
        return f"{type(e).__name__}"
    finally:
        fm.get_gfx, fm.get_gfx_runtime = saved
        fm.get_2stage_cfgs.cache_clear()


def _native():
    try:
        gfx = get_gfx_runtime()
    except (KeyError, OSError, RuntimeError):
        pytest.skip("no GPU architecture detectable")
    if gfx not in ("gfx942", "gfx950"):
        pytest.skip("CDNA only")
    return gfx


def _other(native):
    return "gfx950" if native == "gfx942" else "gfx942"


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-M{c[4]}-{c[5]}")
def test_default_route_ignores_the_build_arch(case):
    native = _native()
    assert _route(case, native, native) == _route(case, _other(native), native)


def test_runtime_modules_do_not_decide_with_get_gfx():
    from aiter.ops import quant, shuffle, topk

    for module in (fm, topk, quant, shuffle):
        source = inspect.getsource(module)
        calls = [
            line
            for line in source.splitlines()
            if re.search(r"\bget_gfx\(\)", line) and not line.lstrip().startswith("#")
        ]
        assert not calls, f"{module.__name__}: {calls}"


_PROBE = """
import json, sys
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
out = {}
for label, qa, qw, qt, tok, act in json.loads(sys.argv[1]):
    fm.get_2stage_cfgs.cache_clear()
    try:
        m = fm.get_2stage_cfgs(tok, 4096, 512, 64, 8, dtypes.bf16, getattr(dtypes, qa),
            getattr(dtypes, qw), getattr(QuantType, qt), True, getattr(ActivationType, act), False, 0, 0)
        s1 = getattr(getattr(m.stage1, "func", m.stage1), "__name__", "full_impl")
        out[f"{label}|{tok}|{act}"] = f"{'1stage' if m.run_1stage else '2stage'}:{s1}:bm{m.block_m}"
    except Exception as e:
        out[f"{label}|{tok}|{act}"] = type(e).__name__
print("PROBE" + json.dumps(out))
"""


def _probe(gpu_archs):
    env = dict(os.environ)
    env.pop("GPU_ARCHS", None)
    if gpu_archs:
        env["GPU_ARCHS"] = gpu_archs
    run = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(CASES)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    line = [x for x in run.stdout.splitlines() if x.startswith("PROBE")]
    assert line, run.stderr[-500:]
    return json.loads(line[0][5:])


def test_multi_arch_gpu_archs_environment_changes_no_route():
    native = _native()
    multi = f"{native};{_other(native)}"  # get_gfx() resolves to the other arch
    assert _probe(None) == _probe(multi)
