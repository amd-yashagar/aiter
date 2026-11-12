# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Bar mode measures aiter's own paths: the router hook is off and asserted off."""

import os
import subprocess
import sys
import types

import pytest

from op_tests.moe_bench import bar_worker


def _fake_moe(monkeypatch, last):
    import aiter

    moe = types.ModuleType("aiter.moe")
    moe.refreshed = 0

    def refresh():
        moe.refreshed += 1

    moe.refresh = refresh
    moe.last_route = lambda: last
    monkeypatch.setitem(sys.modules, "aiter.moe", moe)
    monkeypatch.setattr(aiter, "moe", moe, raising=False)
    return moe


def test_disable_routes_sets_the_env_and_refreshes_the_hook(monkeypatch):
    moe = _fake_moe(monkeypatch, ("aiter", "", "", "disabled", None))
    monkeypatch.delenv("AITER_MOE_DISABLE", raising=False)
    assert bar_worker.disable_routes() is True
    assert os.environ["AITER_MOE_DISABLE"] == "1" and moe.refreshed == 1
    monkeypatch.delenv("AITER_MOE_DISABLE")


def test_assert_no_route_raises_when_a_kernel_served_the_call(monkeypatch):
    _fake_moe(monkeypatch, ("aiter", "", "", "disabled", None))
    bar_worker.assert_no_route()  # nothing took the call
    _fake_moe(
        monkeypatch, ("gfx942_fp8_block.prefill", "fp8-block", "gfx942", "row", {})
    )
    with pytest.raises(RuntimeError, match="routes are not disabled"):
        bar_worker.assert_no_route()


def test_enumerate_names_a_routed_dispatch_instead_of_a_typeerror():
    from op_tests.moe_bench.candidates.fused_cfg import FusedCfgSource

    ctx = types.SimpleNamespace(dispatch_args=None, dispatch_meta={"gfx": "gfx942"})
    with pytest.raises(RuntimeError, match="AITER_MOE_DISABLE"):
        FusedCfgSource().enumerate(ctx)


DUMMY_KERNEL = """
import torch
from aiter.moe import MoeKernel, register_kernel


@register_kernel
class DummyRoute(MoeKernel):
    id = "dummy.route"
    family = "fp8-block"
    archs = ("gfx942", "gfx950")

    def supports(self, spec, m, topk, call=None):
        return None

    def __call__(self, call):
        return torch.zeros_like(call.hidden_states, dtype=call.out_dtype)
"""

# registers the dummy kernel's module as one more kernel package of aiter.moe, in every
# python process that inherits PYTHONPATH (the bar's workers included)
SITECUSTOMIZE = """
import importlib.machinery, sys


class _Finder:
    def find_spec(self, name, path, target=None):
        if name != "aiter.moe.registry":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        orig = spec.loader.exec_module

        def exec_module(mod):
            orig(mod)
            mod.KERNEL_PACKAGES = (*mod.KERNEL_PACKAGES, "dummy_route_kernel")

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())
"""


def _dummy_route_dir(tmp_path):
    d = tmp_path / "dummyroute"
    d.mkdir()
    (d / "dummy_route_kernel.py").write_text(DUMMY_KERNEL)
    (d / "sitecustomize.py").write_text(SITECUSTOMIZE)
    return str(d)


ROUTED_TREE = os.environ.get(
    "MOE_BENCH_ROUTED_TREE", "/workspaces/meta/wt/GM-123/aiter-core"
)


@pytest.mark.skipif(
    os.environ.get("MOE_BENCH_SLOW") != "1"
    or not os.path.isdir(os.path.join(ROUTED_TREE, "aiter", "moe")),
    reason="slow: MOE_BENCH_SLOW=1 and a tree with aiter.moe (MOE_BENCH_ROUTED_TREE)",
)
def test_bar_on_a_routed_tree_never_contains_a_moe_next_kernel(tmp_path):
    """A tree with a live route row (AITER_MOE_ROUTES) used to make 'dispatch' our routed
    kernel and enumeration raise a TypeError, so the bar collapsed to our own kernel."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tier = tmp_path / "t.csv"
    from op_tests.moe_bench import schema

    row = {c: "" for c in schema.COLS}
    row.update(
        case_id="t-routed", kind="fused", family="fp8-block", token="8", model_dim="2048",
        inter_dim="128", expert="256", topk="8", act_type="Silu", dtype="bf16",
        q_dtype_a="fp8", q_dtype_w="fp8", q_type="per_1x128", use_g1u1="1",
        doweight_stage1="0", modes="warm",
    )  # fmt: skip
    schema.write_tier(str(tier), [row])
    env = dict(
        os.environ,
        AITER_MOE_ROUTES="dummy.route",
        PYTHONPATH=here + ":" + _dummy_route_dir(tmp_path),
    )
    env.pop("AITER_MOE_DISABLE", None)  # the bar must set it itself
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable, "-m", "op_tests.moe_bench", "run", "--mode", "bar",
            "--aiter", ROUTED_TREE, "--cases", str(tier), "--out-dir", str(out),
            "--tier-name", "t", "--modes", "warm", "--launcher", "claim", "--wait", "1800",
            "--sources", "fused_moe", "--screen", "two-phase",
        ],
        env=env, cwd=here, check=True, timeout=3000,
    )  # fmt: skip
    bar = schema.read_csv(str(out / "bar-t.csv"))
    cands = schema.read_csv(str(out / "candidates-t.csv"))
    assert bar and all(r["routes_disabled"] == "1" for r in bar)
    text = " ".join(
        r.get("bar_kernels", "") + r.get("bar_id", "") + r.get("dispatch_kernel1", "")
        for r in bar
    )
    assert "dummy.route" not in text and "moe_next" not in text
    assert all("dummy.route" not in (r.get("kernels") or "") for r in cands)
    dispatch = [r for r in cands if r["cand_id"] == "dispatch"]
    assert dispatch and all(r["status"] != "error" for r in dispatch)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROUTED_TREE, "aiter", "moe")),
    reason="needs a tree with aiter.moe (MOE_BENCH_ROUTED_TREE)",
)
def test_the_routed_setup_really_routes_until_routes_are_disabled(tmp_path):
    """Non-vacuous: on the case of the bar test above, a live route row serves the call
    (and the route recorder sees no get_2stage_cfgs call, so dispatch_args is None and
    enumerate raised its TypeError); with routes disabled aiter's own path runs."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from op_tests.moe_bench import runner

    shim = runner.make_shim(str(tmp_path))
    script = """
import json, sys, torch
from moe_bench_shim import accuracy, schema, workload, routes, bar_worker
import aiter.fused_moe as fm
from aiter import moe
row = {c: "" for c in schema.COLS}
row.update(case_id="t-routed", kind="fused", family="fp8-block", token="8",
    model_dim="2048", inter_dim="128", expert="256", topk="8", act_type="Silu",
    dtype="bf16", q_dtype_a="fp8", q_dtype_w="fp8", q_type="per_1x128", use_g1u1="1",
    doweight_stage1="0", modes="warm")
S = workload.resolve(schema.Case(row, {}))
aw, rw, bias = workload.build_group(S)
hs, tw, ti = accuracy.acc_inputs(S, 8, 0)
kw = workload.fused_kwargs(S, aw, bias, None)
def call():
    with routes.RouteRecorder() as rec:
        fm.fused_moe(hs.clone(), aw["w1"], aw["w2"], tw, ti, **kw)
    torch.cuda.synchronize()
    return moe.last_route()[0], bool(rec.records)
moe.refresh()
before = call()
bar_worker.disable_routes()
after = call()
print("RESULT " + json.dumps({"before": before, "after": after}))
"""
    env = dict(
        os.environ,
        AITER_MOE_ROUTES="dummy.route",
        PYTHONPATH=":".join([ROUTED_TREE, shim, _dummy_route_dir(tmp_path)]),
    )
    env.pop("AITER_MOE_DISABLE", None)  # a bar run in this process may have set it
    gpu = [sys.executable, "-m", "moe_bench_shim.gpu_claim", "--wait", "1800", "--"]
    run = subprocess.run(
        [*gpu, sys.executable, "-c", script],
        env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=1800, check=False,
    )  # fmt: skip
    assert run.returncode == 0, run.stderr[-3000:]
    out = run.stdout
    import json

    res = json.loads(next(l for l in out.splitlines() if l.startswith("RESULT "))[7:])
    assert res["before"][0] == "dummy.route" and res["before"][1] is False
    # aiter's own path ran (the recorder saw get_2stage_cfgs). last_route() is not checked:
    # an inactive hook leaves the previous call's record in place.
    assert res["after"][1] is True
