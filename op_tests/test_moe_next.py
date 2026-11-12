# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter.moe entry: routing, fallback and the fused_moe parity contract."""

import argparse
import contextlib
import os
import tempfile

import pandas as pd
import torch

import aiter
import aiter.fused_moe as aiter_fused_moe
from aiter import ActivationType, QuantType, dtypes, moe
from aiter.fused_moe import fused_moe, fused_topk
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.moe import routes
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
        routes.table_for.cache_clear()
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        routes.table_for.cache_clear()


@contextlib.contextmanager
def recorded_fused_moe():
    """Replace fused_moe with a recorder: router tests launch no MoE kernel."""
    calls = []
    real = aiter_fused_moe.fused_moe

    def fake(hidden, w1, w2, topk_weight, topk_ids, **kwargs):
        calls.append((hidden, w1, w2, topk_weight, topk_ids, kwargs))
        out = kwargs.get("output")
        return out if out is not None else hidden.new_full(hidden.shape, -1.0)

    aiter_fused_moe.fused_moe = fake
    try:
        yield calls
    finally:
        aiter_fused_moe.fused_moe = real


class ProbeKernel(moe.MoeKernel):
    """Writes a constant; declines M > 64; raises at M == 13."""

    id = "test.probe"
    family = "a16w16"
    archs = ("gfx942", "gfx950")

    def __init__(self):
        self.prepared = 0
        self.calls = 0

    def supports(self, spec, m, topk, call=None):
        if m > 64:
            return "M > 64"
        if call is not None and call.expert_mask is not None:
            return "EP"
        return None

    def prepare_weights(self, weights):
        self.prepared += 1
        return {"marker": 7.0}

    def __call__(self, call):
        self.calls += 1
        if call.m == 13:
            raise RuntimeError("probe failure")
        out = call.output
        if out is None:
            out = torch.empty_like(call.hidden_states)
        return out.fill_(call.weights.state[self.id]["marker"])


def _write_routes(path, rows):
    with open(path, "w") as f:
        f.write(",".join(routes.COLUMNS) + "\n")
        f.writelines(r + "\n" for r in rows)


def _small_case(token=32, model_dim=512, inter_dim=256, E=8, topk=2, seed=0):
    torch.manual_seed(seed)
    dtype = dtypes.bf16
    hidden = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    gating = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(hidden, gating, topk, True)
    return hidden, w1, w2, topk_weights, topk_ids


def test_router():
    """Route lookup, kill switch, force, decline, error fallback, strict."""
    probe = ProbeKernel()
    moe.register_kernel(probe)
    results = []
    try:
        hidden, w1, w2, tw, ti = _small_case(token=200)
        with tempfile.TemporaryDirectory() as d:
            table = os.path.join(d, "routes.csv")
            _write_routes(
                table,
                [
                    "a16w16,512,256,8,2,silu,0,1,8,test.probe,cfgA,t,specific",
                    "a16w16,*,*,*,*,*,*,1,128,test.probe,cfgB,t,general",
                    "fp8-block,*,*,*,*,*,*,1,128,test.probe,,t,other family",
                ],
            )
            with env(AITER_MOE_ROUTES=table):
                w = moe.prepare(w1, w2, topk=2, warmup_m=(4,))
                assert probe.prepared == 1, "prepare_weights runs once per weights"
                assert moe.explain(w, 4)["config"] == "cfgA", "first matching row wins"
                assert moe.explain(w, 9)["config"] == "cfgB"
                assert moe.explain(w, 9, topk=3)["config"] == "cfgB", "topk wildcard"
                assert moe.explain(w, 4, topk=3)["config"] == "cfgB", "topk key"
                assert moe.explain(w, 65)["route"] == "aiter", "declined -> aiter"
                assert "declines" in moe.explain(w, 65)["reason"]
                assert moe.explain(w, 129)["route"] == "aiter", "no row -> aiter"
                assert moe.explain(w, 4)["family"] == "a16w16"

                with recorded_fused_moe() as calls:
                    out = moe.forward(hidden[:4], w, tw[:4], ti[:4])
                    assert len(calls) == 0 and bool((out == 7).all()), "routed"
                    moe.forward(hidden[:65], w, tw[:65], ti[:65])
                    assert len(calls) == 1, "declined -> fallback"
                    mask = torch.ones(8, dtype=torch.int32)
                    moe.forward(hidden[:4], w, tw[:4], ti[:4], expert_mask=mask)
                    assert len(calls) == 2, "per-call decline -> fallback"
                    with env(AITER_MOE_DISABLE="1"):
                        moe.forward(hidden[:4], w, tw[:4], ti[:4])
                    assert len(calls) == 3, "kill switch"
                    with env(AITER_MOE_STRICT="1"):
                        try:
                            moe.forward(hidden[:13], w, tw[:13], ti[:13])
                            raise AssertionError("strict must raise")
                        except RuntimeError:
                            pass
                    assert len(calls) == 3
                    moe.forward(hidden[:13], w, tw[:13], ti[:13])
                    assert len(calls) == 4, "runtime error -> fallback"
                    assert probe.id in w.failed
                    n = probe.calls
                    moe.forward(hidden[:4], w, tw[:4], ti[:4])
                    assert probe.calls == n and len(calls) == 5, "failed stays off"
                results.append({"check": "route table, decline, error, strict"})

            with env(AITER_MOE_ROUTES="test.probe:X"):
                w = moe.prepare(w1, w2, topk=2)
                assert moe.explain(w, 100)["route"] == "aiter", "supports still holds"
                assert moe.explain(w, 3)["config"] == "X"
            w = moe.prepare(
                w1, w2, topk=2, routes={(1, 8): "test.probe:Y", "*": "aiter"}
            )
            assert moe.explain(w, 8)["config"] == "Y"
            assert moe.explain(w, 9)["route"] == "aiter"
            results.append({"check": "override by env, kernel id and dict"})

            w = moe.prepare(w1, w2, topk=2, routes="nope")
            assert moe.explain(w, 3)["route"] == "aiter", "unknown kernel -> aiter"
            results.append({"check": "unknown kernel"})

            w = moe.prepare(w1, w2, routes="test.probe", herd_topk=0, clamp_shared=True)
            assert (
                moe.explain(w, 3, topk=2)["route"] == "test.probe"
            ), "defaults are unset"
            w = moe.prepare(w1, w2, routes="test.probe", quant_dtype_a=dtypes.bf16)
            assert moe.explain(w, 3, topk=2)["route"] == "aiter", "not yet covered"
            assert "quant_dtype_a" in moe.explain(w, 3, topk=2)["reason"]
            ProbeKernel.handles_kwargs = frozenset({"quant_dtype_a"})
            try:
                w = moe.prepare(w1, w2, routes="test.probe", quant_dtype_a=dtypes.bf16)
                assert (
                    moe.explain(w, 3, topk=2)["route"] == "test.probe"
                ), "handles_kwargs"
            finally:
                ProbeKernel.handles_kwargs = frozenset()
            results.append({"check": "not-yet-covered prepare kwargs"})

            for bad in (
                lambda: moe.prepare(w1, w2, expert_mask=None),
                lambda: moe.forward(hidden, w, tw, ti, activation=1),
            ):
                try:
                    bad()
                    raise AssertionError("misplaced keyword must raise")
                except TypeError:
                    pass
            results.append({"check": "prepare/call keyword split"})
    finally:
        moe.unregister_kernel(probe.id)
    return results


def test_fallback_args():
    """The fallback hands fused_moe the caller's own tensors and arguments."""
    hidden, w1, w2, tw, ti = _small_case()
    b1 = torch.zeros(8, 512, dtype=dtypes.fp32)
    w = moe.prepare(w1, w2, routes="aiter", bias1=b1, splitk=0, swiglu_limit=7.0)
    out = torch.empty_like(hidden)
    mask = torch.ones(8, dtype=torch.int32)
    with recorded_fused_moe() as calls:
        ret = moe.forward(hidden, w, tw, ti, output=out, expert_mask=mask)
    (h, a, b, c, d, kw) = calls[0]
    assert ret is out
    assert h is hidden and a is w1 and b is w2 and c is tw and d is ti
    assert kw["bias1"] is b1 and kw["expert_mask"] is mask and kw["output"] is out
    assert kw["splitk"] == 0 and kw["swiglu_limit"] == 7.0
    assert set(kw) == {
        "bias1",
        "splitk",
        "swiglu_limit",
        "output",
        "expert_mask",
    }, "unset keywords keep fused_moe's defaults"
    return [{"check": "fallback forwards the same objects"}]


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
    """aiter-layout weights as fused_moe takes them, plus the prepare kwargs."""
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
            {"quant_type": QuantType.per_1x128, "w1_scale": s1, "w2_scale": s2},
        )
    raise ValueError(family)


def test_fallback_parity(family, token, model_dim, inter_dim, E, topk, ep, outbuf):
    """forward on fallback is bit-identical to fused_moe with the same tensors."""
    torch.manual_seed(token)
    # Under EP the weights hold the local experts (here the first half).
    w1, w2, kw = _case_weights(family, model_dim, inter_dim, E // 2 if ep else E)
    hidden = torch.randn(token, model_dim, dtype=dtypes.bf16)
    router = torch.randn(token, E, dtype=dtypes.bf16)
    tw, ti = fused_topk(hidden, router, topk, True)
    call_kw = {}
    if ep:
        mask = torch.zeros(E, dtype=torch.int32)
        mask[: E // 2] = 1
        call_kw["expert_mask"] = mask
    args = (hidden, w1, w2, tw, ti)
    ref = fused_moe(*args, activation=ActivationType.Silu, **kw, **call_kw)
    stable = torch.equal(
        ref, fused_moe(*args, activation=ActivationType.Silu, **kw, **call_kw)
    )
    w = moe.prepare(w1, w2, routes="aiter", activation=ActivationType.Silu, **kw)
    out = torch.full_like(hidden, -7.0) if outbuf else None
    got = moe.forward(hidden, w, tw, ti, output=out, **call_kw)
    if outbuf:
        assert got is out
    equal = torch.equal(got, ref)
    assert equal or not stable, f"{family} M={token}: fallback differs from fused_moe"
    # Where fused_moe is not run-to-run stable (atomic combine), parity is the
    # structural check in test_fallback_args plus this tolerance, the one
    # test_moe_2stage's blockscale test uses.
    a, b = got.double(), ref.double()
    logits_diff = float(1 - 2 * (a * b).sum() / (a.square() + b.square()).sum())
    assert logits_diff < 1e-3, f"{family} M={token}: logits diff {logits_diff}"
    return {
        "gfx": get_gfx_runtime(),
        "gpu": w.gpu,
        "fused_moe stable": stable,
        "bit-identical": equal,
        "max abs diff": float((got.float() - ref.float()).abs().max()),
        "logits diff": logits_diff,
    }


def test_graph_capture(token=8):
    """forward on fallback replays under a CUDA graph to the eager result."""
    hidden, w1, w2, tw, ti = _small_case(token=token)
    w = moe.prepare(w1, w2, routes="aiter")
    out = torch.empty_like(hidden)
    eager = moe.forward(hidden, w, tw, ti).clone()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        moe.forward(hidden, w, tw, ti, output=out)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        moe.forward(hidden, w, tw, ti, output=out)
    out.fill_(-7.0)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager), "graph replay differs from eager"
    return [{"check": f"graph capture M={token}"}]


def test_torch_compile(token=8):
    """forward under torch.compile (fullgraph), eager and under graph capture."""
    hidden, w1, w2, tw, ti = _small_case(token=token)
    w = moe.prepare(w1, w2, routes="aiter")
    eager = moe.forward(hidden, w, tw, ti)

    def layer(h, a, b):
        return moe.forward(h, w, a, b) * 1.0

    compiled = torch.compile(layer, fullgraph=True)
    got = compiled(hidden, tw, ti)
    assert torch.equal(got, eager), "torch.compile result differs from eager"
    out = torch.empty_like(hidden)

    def layer_out(h, a, b, o):
        return moe.forward(h, w, a, b, output=o)

    compiled_out = torch.compile(layer_out, fullgraph=True)
    assert torch.equal(compiled_out(hidden, tw, ti, out), eager)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        compiled_out(hidden, tw, ti, out)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        compiled_out(hidden, tw, ti, out)
    out.fill_(-7.0)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager), "compiled graph replay differs from eager"
    fake = moe.entry.moe_forward_fake(hidden, tw, ti, w.handle)
    assert fake.shape == eager.shape and fake.dtype == eager.dtype
    return [{"check": f"torch.compile fullgraph + graph capture M={token}"}]


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
    checks = (
        test_router()
        + test_fallback_args()
        + test_graph_capture()
        + test_torch_compile()
    )
    aiter.logger.info(
        "contract checks:\n%s", pd.DataFrame(checks).to_markdown(index=False)
    )
    rows = []
    for i in args.parity:
        case = PARITY_CASES[i]
        names = (
            "family",
            "token",
            "model_dim",
            "inter_dim",
            "E",
            "topk",
            "ep",
            "outbuf",
        )
        row = dict(zip(names, case))
        row.update(test_fallback_parity(*case))
        rows.append(row)
    aiter.logger.info(
        "fallback parity:\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
