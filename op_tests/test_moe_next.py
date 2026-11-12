# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter.moe routing behind fused_moe: router contract, lazy per-weight state,
graph capture and torch.compile, and parity with today's path."""

import argparse
import contextlib
import gc
import os
import sys
import tempfile
import threading
import time
from typing import ClassVar

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes, moe
from aiter.fused_moe import fused_moe, fused_topk
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.moe import router, routes
from aiter.ops.shuffle import shuffle_weight

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]


@contextlib.contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        moe.refresh()
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        moe.refresh()


class ProbeKernel(moe.MoeKernel):
    """Fills the output with 7; declines M > 64 and EP calls; fails at M == 13."""

    id = "test.probe"
    family = "a16w16"
    archs = ("gfx942", "gfx950")

    def __init__(self):
        self.prepared = 0
        self.compiled = 0
        self.calls = 0
        self.lock = threading.Lock()

    def supports(self, spec, m, topk, call=None):
        if m > 64:
            return "M > 64"
        if call is not None and call.expert_mask is not None:
            return "EP"
        return None

    def prepare_weights(self, weights):
        with self.lock:
            self.prepared += 1
        return {"marker": 7.0}

    def warmup(self, weights, m, topk, config):
        with self.lock:
            self.compiled += 1

    def __call__(self, call):
        self.calls += 1
        if call.m == 13:
            raise RuntimeError("probe failure")
        out = call.output
        if out is None:
            out = torch.empty(
                (call.m, call.spec.model_dim),
                dtype=call.out_dtype,
                device=call.hidden_states.device,
            )
        return out.fill_(call.weights.state[self.id]["marker"])


def _table(rows):
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, dir=os.environ.get("TMPDIR")
    ) as f:
        f.write(",".join(routes.COLUMNS) + "\n")
        f.writelines(r + "\n" for r in rows)
    return f.name


PROBE_ROWS = [
    "a16w16,512,256,8,2,silu,0,1,8,test.probe,cfgA,t,specific",
    "a16w16,*,*,*,*,*,*,1,128,test.probe,cfgB,t,general",
]


def _small_case(token=32, model_dim=512, inter_dim=256, E=8, topk=2, seed=0):
    torch.manual_seed(seed)
    dtype = dtypes.bf16
    hidden = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    gating = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(hidden, gating, topk, True)
    return hidden, w1, w2, topk_weights, topk_ids


def _is_probe(out):
    return bool((out == 7).all())


def test_router(probe):
    """Route rows, decline, error fallback, strict, kill switch, unknown kernel,
    not-yet-covered keywords, and last_route()."""
    hidden, w1, w2, tw, ti = _small_case(token=200)
    table = _table(PROBE_ROWS)
    with env(AITER_MOE_DISABLE="1"):
        ref = {m: fused_moe(hidden[:m], w1, w2, tw[:m], ti[:m]) for m in (13, 65, 129)}
    with env(AITER_MOE_ROUTES=table):
        out = fused_moe(hidden[:4], w1, w2, tw[:4], ti[:4])
        assert _is_probe(out) and moe.last_route()[0] == "test.probe"
        assert moe.last_route().numerics == {"same_as": "aiter dispatched path"}
        assert "routes" in moe.last_route()[3] or ".csv" in moe.last_route()[3]
        assert probe.prepared == 1 and probe.compiled >= 1, "lazy state built once"
        assert _is_probe(fused_moe(hidden[:9], w1, w2, tw[:9], ti[:9]))
        assert probe.prepared == 1
        out = fused_moe(hidden[:65], w1, w2, tw[:65], ti[:65])
        assert torch.equal(out, ref[65]) and moe.last_route()[0] == "aiter"
        assert "declines" in moe.last_route()[3]
        out = fused_moe(hidden[:129], w1, w2, tw[:129], ti[:129])
        assert torch.equal(out, ref[129]), "no row -> today's path"
        mask = torch.ones(8, dtype=torch.int32)
        fused_moe(hidden[:4], w1, w2, tw[:4], ti[:4], expert_mask=mask)
        assert moe.last_route()[0] == "aiter", "per-call decline"
        with env(AITER_MOE_STRICT="1", AITER_MOE_ROUTES=table):
            try:
                fused_moe(hidden[:13], w1, w2, tw[:13], ti[:13])
                raise AssertionError("strict must raise")
            except RuntimeError:
                pass
        out = fused_moe(hidden[:13], w1, w2, tw[:13], ti[:13])
        assert torch.equal(out, ref[13]), "kernel error -> today's path"
        n = probe.calls
        assert not _is_probe(fused_moe(hidden[:4], w1, w2, tw[:4], ti[:4]))
        assert probe.calls == n, "a failed kernel stays off for that weight set"
    with env(AITER_MOE_ROUTES=table):
        assert _is_probe(fused_moe(hidden[:4], w1, w2, tw[:4], ti[:4]))
        assert moe.last_route()[0] == "test.probe"
    with env(AITER_MOE_ROUTES=table, AITER_MOE_DISABLE="1"):
        for _ in range(2):  # the call that finds routing off, then the cached path
            assert not _is_probe(fused_moe(hidden[:4], w1, w2, tw[:4], ti[:4]))
            assert moe.last_route()[0] == "aiter", "inactive call records aiter"
            assert "inactive" in moe.last_route()[3]
        assert aiter.fused_moe._MOE_ROUTING is False, "kill switch"
    if not routes.table_for(router.gpu_name()).rows:  # this GPU has no rows
        with env(AITER_MOE_ROUTES=table):
            fused_moe(hidden[:4], w1, w2, tw[:4], ti[:4])
        with env(AITER_MOE_ROUTES=None):
            fused_moe(hidden[:4], w1, w2, tw[:4], ti[:4])
            assert moe.last_route()[0] == "aiter", "no route rows records aiter"
            assert "inactive" in moe.last_route()[3]
    with env(AITER_MOE_ROUTES="nope"):
        fused_moe(hidden[:4], w1, w2, tw[:4], ti[:4])
        assert moe.last_route()[0] == "aiter" and "unknown" in moe.last_route()[3]
    with env(AITER_MOE_ROUTES="test.probe"):
        fused_moe(hidden[:4], w1, w2, tw[:4], ti[:4], quant_dtype_a=dtypes.bf16)
        assert "quant_dtype_a" in moe.last_route()[3], "not yet covered"
        ProbeKernel.handles_kwargs = frozenset({"quant_dtype_a"})
        try:
            moe.refresh()
            out = fused_moe(
                hidden[:4], w1, w2, tw[:4], ti[:4], quant_dtype_a=dtypes.bf16
            )
            assert _is_probe(out), "handles_kwargs"
        finally:
            ProbeKernel.handles_kwargs = frozenset()

    class Bad(moe.MoeKernel):
        id = "test.bad_numerics"
        numerics: ClassVar[dict] = {"act_quant": "mx_fp4"}

    try:
        moe.register_kernel(Bad)
        raise AssertionError("bare mx_fp4 must raise")
    except ValueError:
        pass

    class Good(moe.MoeKernel):
        id = "test.good_numerics"
        numerics: ClassVar[dict] = {
            "act_quant": "mx_fp4_round_up",
            "inter": "bf16_then_mx_fp4_round_up",
            "combine": "bf16_rows_fp32_sum",
        }

    moe.register_kernel(Good)
    assert moe.numerics("test.good_numerics")["inter_quant"] == "mx_fp4_round_up"
    moe.unregister_kernel("test.good_numerics")
    return [
        {"check": "routes, decline, error, strict, kill switch, last_route, numerics"}
    ]


def test_family_import():
    """Each gfx*_* family module imports on its own; a broken one is logged."""
    import aiter.ops.flydsl.moe_next as pkg

    d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    with open(os.path.join(d, "gfx000_ok.py"), "w") as f:
        f.write(
            "from aiter import moe\n"
            "class K(moe.MoeKernel):\n"
            "    id = 'test.family_ok'\n"
            "    def supports(self, spec, m, topk, call=None):\n"
            "        return 'test only'\n"
            "moe.register_kernel(K)\n"
        )
    with open(os.path.join(d, "gfx000_broken.py"), "w") as f:
        f.write("raise RuntimeError('broken family')\n")
    old = list(pkg.__path__)
    pkg.__path__[:] = [d]
    try:
        pkg._import_families()
        assert moe.get_kernel("test.family_ok") is not None, "ok family registered"
        with env(AITER_MOE_STRICT="1"):
            try:
                pkg._import_families()
                raise AssertionError("strict must raise a family import error")
            except RuntimeError:
                pass
    finally:
        pkg.__path__[:] = old
        moe.unregister_kernel("test.family_ok")
        for name in ("gfx000_ok", "gfx000_broken"):
            sys.modules.pop(f"{pkg.__name__}.{name}", None)
    return [{"check": "family auto-import, isolated failures, strict raises"}]


def test_capture_and_compile(probe):
    """Routed calls under graph capture (warmup first) and torch.compile; a
    first call during capture takes today's path and builds nothing."""
    hidden, w1, w2, tw, ti = _small_case(token=8, seed=1)
    table = _table(PROBE_ROWS)
    with env(AITER_MOE_ROUTES=table):
        out = torch.empty_like(hidden)
        fused_moe(hidden, w1, w2, tw, ti, output=out)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fused_moe(hidden, w1, w2, tw, ti, output=out)
        out.fill_(-1.0)
        g.replay()
        torch.cuda.synchronize()
        assert _is_probe(out), "routed kernel replays from the graph"

        def layer(h, a, b):
            return fused_moe(h, w1, w2, a, b) * 1.0

        got = torch.compile(layer, fullgraph=True)(hidden, tw, ti)
        assert _is_probe(got), "routed under torch.compile fullgraph"

        _, v1, v2, _, _ = _small_case(token=8, seed=2)
        with env(AITER_MOE_DISABLE="1"):
            ref = fused_moe(hidden, v1, v2, tw, ti).clone()
        n_states = len(router._BY_KEY)
        out2 = torch.empty_like(hidden)
        g2 = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(g2, stream=s):
            fused_moe(hidden, v1, v2, tw, ti, output=out2)
        g2.replay()
        torch.cuda.synchronize()
        assert len(router._BY_KEY) == n_states, "nothing built during capture"
        assert torch.equal(out2, ref), "first call in capture takes today's path"
    return [{"check": "graph capture, torch.compile fullgraph, first call in capture"}]


def test_state_lifecycle(probe):
    """Rebuild on _version change, drop on free, two weight sets, threads."""
    hidden, w1, w2, tw, ti = _small_case(token=4, seed=3)
    table = _table(PROBE_ROWS)
    with env(AITER_MOE_ROUTES=table):
        p0 = probe.prepared
        fused_moe(hidden, w1, w2, tw, ti)
        assert probe.prepared == p0 + 1
        w1.add_(0)
        fused_moe(hidden, w1, w2, tw, ti)
        assert probe.prepared == p0 + 2, "_version change rebuilds"
        _, u1, u2, _, _ = _small_case(token=4, seed=4)
        fused_moe(hidden, u1, u2, tw, ti)
        assert probe.prepared == p0 + 3, "second weight set, own state"
        fused_moe(hidden, w1.view(w1.shape), w2, tw, ti)
        assert probe.prepared == p0 + 3, "a view of the same storage reuses state"
        n = len(router._BY_KEY)
        del u1, u2
        gc.collect()
        assert len(router._BY_KEY) == n - 1, "freeing the weights drops state"

        _, t1, t2, _, _ = _small_case(token=4, seed=5)
        p1 = probe.prepared
        errs = []

        def worker():
            try:
                for _ in range(20):
                    assert _is_probe(fused_moe(hidden, t1, t2, tw, ti))
            except Exception as exc:  # noqa: BLE001
                errs.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errs, errs
        assert probe.prepared == p1 + 1, "concurrent first calls build state once"
    return [{"check": "_version rebuild, free drops, two weight sets, views, threads"}]


def test_overhead():
    """Host time the hook adds per eager fused_moe call (graph replay runs no
    host code, so it adds nothing there)."""
    hidden, w1, w2, tw, ti = _small_case(token=4, seed=6)
    out = torch.empty_like(hidden)
    kw = {"output": out}

    def host_us(n=2000):
        for _ in range(50):
            fused_moe(hidden, w1, w2, tw, ti, **kw)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fused_moe(hidden, w1, w2, tw, ti, **kw)
        host = time.perf_counter() - t
        torch.cuda.synchronize()
        return host * 1e6 / n

    rows = []
    with env(AITER_MOE_DISABLE="1"):
        off = host_us()
    rows.append({"mode": "AITER_MOE_DISABLE=1", "host us/call": off, "added us": 0.0})
    with env(AITER_MOE_ROUTES=None, AITER_MOE_DISABLE=None):
        empty = host_us()
    rows.append(
        {
            "mode": "no table for this GPU",
            "host us/call": empty,
            "added us": empty - off,
        }
    )
    table = _table(["fp8-block,*,*,*,*,*,*,1,128,test.probe,,t,other family"])
    with env(AITER_MOE_ROUTES=table):
        miss = host_us()
    rows.append(
        {
            "mode": "table, no matching route",
            "host us/call": miss,
            "added us": miss - off,
        }
    )
    return rows


def _blockscale(w):
    e, n, k = w.shape
    blocks = w.view(e, n // 128, 128, k // 128, 128).permute(0, 1, 3, 2, 4).contiguous()
    q, scale = aiter.pertoken_quant(
        blocks.view(e, -1, 128 * 128), quant_dtype=dtypes.fp8
    )
    q = (
        q.view(e, n // 128, k // 128, 128, 128)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view_as(w)
    )
    return q, scale.view(e, n // 128, k // 128)


def _case_weights(family, model_dim, inter_dim, E):
    if family == "a16w16":
        w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtypes.bf16) / 10
        w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16) / 10
        return w1, w2, {}
    if family == "fp8-block":
        w1, s1 = _blockscale(
            torch.randn(E, 2 * inter_dim, model_dim, dtype=dtypes.bf16) * 0.5
        )
        w2, s2 = _blockscale(
            torch.randn(E, model_dim, inter_dim, dtype=dtypes.bf16) / inter_dim**0.5
        )
        w1 = shuffle_weight(w1, layout=(16, 16))
        w2 = shuffle_weight(w2, layout=(16, 16))
        return (
            w1,
            w2,
            {"quant_type": QuantType.per_128x128, "w1_scale": s1, "w2_scale": s2},
        )
    raise ValueError(family)


def test_fallback_parity(family, token, model_dim, inter_dim, E, topk, ep, outbuf):
    """Routing on but not taken gives today's result: bitwise where today's path
    is deterministic, else within the blockscale test's tolerance."""
    torch.manual_seed(token)
    w1, w2, kw = _case_weights(family, model_dim, inter_dim, E // 2 if ep else E)
    hidden = torch.randn(token, model_dim, dtype=dtypes.bf16)
    tw, ti = fused_topk(hidden, torch.randn(token, E, dtype=dtypes.bf16), topk, True)
    if ep:
        mask = torch.zeros(E, dtype=torch.int32)
        mask[: E // 2] = 1
        kw["expert_mask"] = mask
    args = (hidden, w1, w2, tw, ti)
    with env(AITER_MOE_DISABLE="1"):
        ref = fused_moe(*args, activation=ActivationType.Silu, **kw)
        stable = torch.equal(
            ref, fused_moe(*args, activation=ActivationType.Silu, **kw)
        )
    table = _table(["*,*,*,*,*,*,*,1,1000000,aiter,,t,everything to today's path"])
    with env(AITER_MOE_ROUTES=table):
        out = torch.full_like(hidden, -7.0) if outbuf else None
        got = fused_moe(*args, activation=ActivationType.Silu, output=out, **kw)
        assert moe.last_route()[0] == "aiter"
    if outbuf:
        assert got is out
    equal = torch.equal(got, ref)
    assert equal or not stable, f"{family} M={token}: differs from today's path"
    a, b = got.double(), ref.double()
    logits_diff = float(1 - 2 * (a * b).sum() / (a.square() + b.square()).sum())
    assert logits_diff < 1e-3, f"{family} M={token}: logits diff {logits_diff}"
    return {
        "gfx": get_gfx_runtime(),
        "gpu": moe.last_route()[2],
        "today stable": stable,
        "bit-identical": equal,
        "max abs diff": float((got.float() - ref.float()).abs().max()),
        "logits diff": logits_diff,
    }


PARITY_CASES = [
    # family, token, model_dim, inter_dim, E, topk, ep, outbuf
    ("a16w16", 32, 512, 256, 8, 2, False, False),
    ("a16w16", 32, 512, 256, 8, 2, False, True),
    ("a16w16", 33, 512, 256, 8, 2, True, False),
    ("fp8-block", 8, 7168, 256, 256, 8, False, False),
    ("fp8-block", 4096, 7168, 256, 256, 8, False, True),
    ("fp8-block", 257, 7168, 256, 256, 8, True, False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parity",
        nargs="*",
        type=int,
        default=list(range(len(PARITY_CASES))),
        help="indices into PARITY_CASES",
    )
    args = parser.parse_args()
    gfx = get_gfx_runtime()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.info("skipped: %s not in %s", gfx, SUPPORTED_GFX)
        return
    probe = ProbeKernel()
    moe.register_kernel(probe)
    try:
        checks = (
            test_router(probe)
            + test_family_import()
            + test_capture_and_compile(probe)
            + test_state_lifecycle(probe)
        )
    finally:
        moe.unregister_kernel(probe.id)
    aiter.logger.info(
        "contract checks:\n%s", pd.DataFrame(checks).to_markdown(index=False)
    )
    aiter.logger.info(
        "hook overhead:\n%s", pd.DataFrame(test_overhead()).to_markdown(index=False)
    )
    rows = []
    names = ("family", "token", "model_dim", "inter_dim", "E", "topk", "ep", "outbuf")
    for i in args.parity:
        row = dict(zip(names, PARITY_CASES[i]))
        row.update(test_fallback_parity(*PARITY_CASES[i]))
        rows.append(row)
    aiter.logger.info(
        "fallback parity:\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
