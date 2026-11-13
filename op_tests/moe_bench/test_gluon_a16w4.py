# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gluon a16w4 side API as a bar candidate: where it applies, how its weights are laid
out, how its routing is checked, how its top-k is taken off the time, and that a Triton
compile in a leased run is a harness error."""

import io
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("aiter")

from op_tests.moe_bench import bar_worker, prebuild, schema, workload
from op_tests.moe_bench.candidates import HarnessError
from op_tests.moe_bench.candidates import gluon_a16w4 as g

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def case(**kw):
    row = {c: "" for c in schema.COLS}
    row.update(
        case_id="gl", kind="fused", family="a16w4", token="64", model_dim="3072",
        inter_dim="3072", expert="128", topk="4", act_type="Swiglu", dtype="bf16",
        q_dtype_a="bf16", q_dtype_w="fp4x2", q_type="per_1x32", use_g1u1="1",
        doweight_stage1="0", modes="warm", extra=kw.pop("extra", "gate_mode=SEPARATED"),
    )  # fmt: skip
    row.update({k: str(v) for k, v in kw.items()})
    return schema.Case(row, schema.parse_extra(row["extra"]))


def spec(**kw):
    return workload.resolve(case(**kw))


def test_applies_to_a16w4_swiglu_inside_its_limits():
    for kw in (
        {},
        {"expert": 256, "inter_dim": 1536},
        {"expert": 512, "model_dim": 2048, "inter_dim": 1024, "topk": 10},
        {"expert": 128, "token": 1},
    ):
        S = spec(**kw)
        assert g.is_a16w4_swiglu(S) and g.unsupported(S) is None, kw
    assert g.swiglu_limit(spec()) == 7.0


@pytest.mark.parametrize(
    "kw, why",
    [
        ({"expert": 896, "topk": 16, "model_dim": 3584, "inter_dim": 512}, "E=896"),
        ({"model_dim": 2880, "inter_dim": 2880}, "K % 256"),
        ({"inter_dim": 384}, "K % 256"),
        ({"extra": "gate_mode=SEPARATED;ep_local=1"}, "EP"),
        ({"extra": "gate_mode=SEPARATED;hidden_pad=128"}, "padding"),
        ({"expert": 64}, "E=64"),
        ({"doweight_stage1": "1"}, "doweight"),
        ({"extra": "gate_mode=INTERLEAVE"}, "gate_mode"),
        ({"token": 0}, "M=0"),
    ],
)
def test_the_limits_exclude_ep_non_256_k_and_e896(kw, why):
    S = spec(**kw)
    assert g.is_a16w4_swiglu(S)
    reason = g.unsupported(S)
    assert reason and why in reason, reason


def test_other_families_are_not_in_its_family():
    for kw in (
        {"act_type": "Silu"},
        {"q_dtype_a": "fp8"},
        {"q_type": "per_Token", "q_dtype_w": "fp8", "q_dtype_a": "fp8"},
    ):
        assert not g.is_a16w4_swiglu(spec(**kw)), kw


def ctx_for(S, gfx):
    return SimpleNamespace(S=S, dispatch_meta={"gfx": gfx})


def test_the_source_enumerates_one_candidate_on_gfx950_and_reports_a_missed_limit():
    src = g.TritonGluonA16W4()
    ok = ctx_for(spec(), "gfx950")
    assert src.skip_reason(ok) is None
    cands = src.enumerate(ok)
    assert [c.id for c in cands] == [g.CAND_ID] and cands[0].tag == g.TAG
    assert cands[0].describe()["source"] == "triton_gluon"
    # not its arch or family: no candidate and no skip row (nothing to explain)
    for c in (ctx_for(spec(), "gfx942"), ctx_for(spec(act_type="Silu"), "gfx950")):
        assert src.skip_reason(c) is None and src.enumerate(c) == []
    # its family and arch but a limit missed: no candidate, and the row says why
    bad = ctx_for(spec(model_dim=2880, inter_dim=2880), "gfx950")
    assert src.enumerate(bad) == []
    assert "K % 256" in src.skip_reason(bad)


def dequant(packed, scale):
    """[..., K/2] uint8 fp4x2 (even element in the low nibble) and [..., K/32] e8m0 -> f32."""
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    lut = torch.cat([lut, -lut])
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).flatten(-2)
    s = torch.pow(2.0, scale.to(torch.float32) - 127).repeat_interleave(32, dim=-1)
    return vals * s


def test_relayout_keeps_the_weights_and_interleaves_gate_and_up_with_the_biases():
    E, D, inter = 3, 256, 256
    S = SimpleNamespace(E=E, D=D, I=inter)
    gen = torch.Generator().manual_seed(0)
    rw = {
        "w1": torch.randint(0, 256, (E, 2 * inter, D // 2), dtype=torch.uint8, generator=gen),
        "w2": torch.randint(0, 256, (E, D, inter // 2), dtype=torch.uint8, generator=gen),
        "w1_scale": torch.randint(120, 130, (E * 2 * inter, D // 32), dtype=torch.uint8, generator=gen),
        "w2_scale": torch.randint(120, 130, (E * D, inter // 32), dtype=torch.uint8, generator=gen),
    }  # fmt: skip
    bias = (torch.randn(E, 2 * inter, generator=gen), torch.randn(E, D, generator=gen))
    out = g.relayout_weights(
        rw, S, bias, arch="gfx942"
    )  # an arch without the swizzle: scales stay as they are
    assert out["L1"] is None and out["L2"] is None
    w1 = dequant(rw["w1"], rw["w1_scale"].view(E, 2 * inter, -1))  # [E, 2I, D]
    got1 = dequant(
        out["W1"].transpose(1, 2), out["S1"].transpose(1, 2)
    )  # [E, N, D]: gate/up alternating
    assert torch.equal(got1[:, 0::2], w1[:, :inter])
    assert torch.equal(got1[:, 1::2], w1[:, inter:])
    w2 = dequant(rw["w2"], rw["w2_scale"].view(E, D, -1))
    assert torch.equal(
        dequant(out["W2"].transpose(1, 2), out["S2"].transpose(1, 2)), w2
    )
    # the kernel reads W column-major: stride(-2) == 1
    assert out["W1"].stride(-2) == 1 and out["W2"].stride(-2) == 1
    assert torch.equal(out["b1"][:, 0::2], bias[0][:, :inter])
    assert torch.equal(out["b1"][:, 1::2], bias[0][:, inter:])
    assert out["b1"].dtype == torch.float32 and torch.equal(out["b2"], bias[1])
    assert g.relayout_weights(rw, S, None, arch="gfx942")["b1"] is None


def test_crafted_logits_reproduce_the_ids_and_a_mismatch_is_a_harness_error():
    M, E, K = 24, 128, 4
    gen = torch.Generator().manual_seed(1)
    probs = torch.softmax(torch.randn(M, E, generator=gen), dim=-1)
    tw, ti = torch.topk(probs, K, dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True)).float()
    ti = ti.to(torch.int32)
    logits = g.crafted_logits(tw, ti, E)
    assert logits.dtype == torch.float16 and logits.shape == (M, E)
    picked = torch.topk(logits.float(), K, dim=1).indices
    g.check_routed_ids(picked, ti)  # order within a row is free
    g.check_routed_ids(picked.flip(1), ti)
    # softmax over the selected logits gives the case's weights back (fp16 rounding)
    sel = torch.softmax(torch.gather(logits.float(), 1, ti.long()), dim=1)
    assert torch.allclose(sel, tw, atol=1e-2)
    wrong = picked.clone()
    wrong[3, 0] = (wrong[3, 0] + 1) % E
    with pytest.raises(g.GluonHarnessError, match="1 of 24 tokens"):
        g.check_routed_ids(wrong, ti)
    assert issubclass(g.GluonHarnessError, HarnessError)


def test_the_topk_time_is_subtracted_and_both_stay_on_the_row(monkeypatch):
    st = {
        "median_us": 100.0,
        "p10_us": 90.0,
        "p90_us": 130.0,
        "std_us": 4.0,
        "iters": 50,
    }
    out = g.subtract_topk(st, 9.6)
    assert out["median_us"] == pytest.approx(90.4) and out["p10_us"] == pytest.approx(
        80.4
    )
    assert out["p90_us"] == pytest.approx(120.4) and out["std_us"] == 4.0
    assert out["raw_us"] == 100.0 and out["topk_us"] == 9.6
    assert st["median_us"] == 100.0  # the input is not modified
    with pytest.raises(g.GluonHarnessError, match="not shorter"):
        g.subtract_topk(st, 100.0)

    timed = []

    class FakeTimer:
        """Replay latency 4 us per graph, 2.5 us per top-k call, 3 us more for a cold first call."""

        def __init__(self, call, *a):
            self.call = call

        def prepare_graph(self):
            timed.append(("graph", self.call.copies))

        def time(self, cold=False, quick=False):
            timed.append(("cold" if cold else "warm", self.call.copies))
            us = 4.0 + 2.5 * self.call.copies
            return {"median_us": us + (3.0 if cold and self.call.copies == 1 else 0.0)}

    from op_tests.moe_bench import timing

    monkeypatch.setattr(timing, "Timer", FakeTimer)
    cand = g.GluonA16W4()
    monkeypatch.setattr(cand, "_logits", object(), raising=False)
    monkeypatch.setattr(
        g.GluonA16W4, "topk_call", lambda self, copies=1: _calls(copies)
    )
    warm = cand.adjust_timing({"median_us": 100.0}, False, True)
    cold = cand.adjust_timing({"median_us": 200.0}, True, False)
    # the in-call cost is the slope (2.5 us), not the standalone graph (6.5 us warm): the
    # replay latency is not the top-k's
    assert warm["topk_us"] == pytest.approx(2.5) and warm["median_us"] == pytest.approx(
        97.5
    )
    assert warm["topk_alone_us"] == pytest.approx(6.5)
    assert cold["topk_us"] == pytest.approx(5.5)  # slope plus the cold-cache penalty
    assert cold["topk_alone_us"] == pytest.approx(9.5) and cold[
        "median_us"
    ] == pytest.approx(194.5)
    assert timed.count(("graph", 1)) == 1 and timed.count(("graph", g.TOPK_COPIES)) == 1


def _calls(copies):
    def run():
        return None

    run.copies = copies
    return run


class Raises:
    id, source, entry, kernels, layout, group = "x", "triton_gluon", "e", "k", "d", "g"

    def __init__(self, exc):
        self.exc = exc

    def describe(self):
        return {"source": self.source, "entry": self.entry, "kernels": self.kernels, "layout": self.layout}  # fmt: skip

    def prepare(self, ctx):
        raise self.exc

    def release(self, ctx):
        pass


def events(fh):
    return [json.loads(x) for x in fh.getvalue().splitlines() if x.startswith("{")]


@pytest.mark.parametrize(
    "exc",
    [
        g.GluonHarnessError("routing() selected other experts"),
        prebuild.TritonNotCached("k"),
    ],
)
def test_a_harness_error_is_a_harness_error_row_and_marks_the_bar_incomplete(exc):
    fh = io.StringIO()
    m = bar_worker.Measurer(SimpleNamespace(), None, None, None, [], fh, "c1")
    res = m.measure(Raises(exc), modes=("warm",))
    assert res["outcome"] == "harness_error" and not res["ok"]
    ev = [e for e in events(fh) if e.get("event") == "cand"]
    assert len(ev) == 1 and ev[0]["status"] == "harness-error"
    assert str(exc)[:30] in ev[0]["detail"]
    _, incomplete = bar_worker._accounting({"fused_moe:*": 3}, m.stats, [], "bar")
    assert "harness-error" in incomplete and "triton_gluon" in incomplete


def test_the_bar_row_carries_the_tag_and_the_subtraction():
    cand = g.GluonA16W4()
    stats = g.subtract_topk({"median_us": 100.0}, 9.6)
    results = {
        "c": {
            "ok": True,
            "cand": cand,
            "by_mode": {"warm": stats},
            "timing_mode": "graph",
        }
    }
    fh = io.StringIO()
    S = SimpleNamespace(kind="x")
    bar_worker._emit_summary(
        fh,
        SimpleNamespace(id="c1"),
        S,
        "warm",
        results,
        {"gpu": "g", "sku": ""},
        [],
        "bar",
    )
    row = next(e for e in events(fh) if e.get("event") == "row")
    assert row["bar_source"] == "triton_gluon" and row["bar_tag"] == g.TAG
    assert row["bar_us"] == pytest.approx(90.4)
    assert row["bar_raw_us"] == 100.0 and row["bar_topk_us"] == 9.6


def test_the_new_columns_are_in_the_bar_and_candidate_tables():
    from op_tests.moe_bench import bar_runner

    for c in ("bar_tag", "bar_raw_us", "bar_topk_us"):
        assert c in bar_runner.BAR_COLS
    for c in ("tag", "raw_us", "topk_us"):
        assert c in bar_runner.CAND_COLS


def test_run_only_is_the_lease_condition_and_the_warm_lease_lifts_it(monkeypatch):
    for k in (
        prebuild.NO_BUILD_ENV,
        "FLYDSL_RUNTIME_RUN_ONLY",
        prebuild.TRITON_COMPILE_ENV,
    ):
        monkeypatch.delenv(k, raising=False)
    assert not prebuild.triton_run_only()
    monkeypatch.setenv(prebuild.NO_BUILD_ENV, "1")
    assert prebuild.triton_run_only()
    monkeypatch.setenv(prebuild.TRITON_COMPILE_ENV, "1")
    assert not prebuild.triton_run_only()


def test_the_cache_dir_is_persistent_next_to_the_state_unless_set(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    d = prebuild.use_triton_cache()
    assert d == str(prebuild.state_dir()) + "/triton-cache"
    import os

    assert os.environ["TRITON_CACHE_DIR"] == d and os.path.isdir(d)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "mine"))
    assert prebuild.triton_cache_dir() == str(tmp_path / "mine")


def _make_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def bump(x_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(x_ptr + i, tl.load(x_ptr + i, mask=m) + 1, mask=m)

    return bump


@needs_gpu
def test_a_triton_cache_miss_is_refused_in_a_lease_and_a_cached_kernel_runs(
    monkeypatch, tmp_path
):
    pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "tc"))
    x = torch.zeros(300, device="cuda")
    assert prebuild.forbid_triton_compiles()
    try:
        with pytest.raises(prebuild.TritonNotCached, match="triton_warm"):
            _make_kernel()[(2,)](x, 300, BLOCK=256)
    finally:
        prebuild.restore_triton_compiles()
    _make_kernel()[(2,)](x, 300, BLOCK=256)  # the warm lease: compiles into the cache
    torch.cuda.synchronize()
    assert prebuild.forbid_triton_compiles()
    try:
        _make_kernel()[(2,)](x, 300, BLOCK=256)  # a fresh JIT object: a disk-cache hit
        torch.cuda.synchronize()
    finally:
        prebuild.restore_triton_compiles()
    assert float(x[0]) == 2.0


@needs_gpu
def test_relayout_is_what_the_gluon_quantizer_makes_of_the_interleaved_weights():
    """The reference path of the whole-call measurement quantizes the interleaved, transposed
    weights with downcast_to_mxfp; the same values quantized row-major (what the harness's
    reference holds) and re-laid out must be byte-identical."""
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
    from aiter.ops.triton.utils.shuffle import shuffle_scale_moe
    from aiter.ops.triton.utils.types import str_to_torch_dtype

    E, D, inter = 2, 512, 256
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * inter, D, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(E, D, inter, dtype=torch.bfloat16, device="cuda")
    wdt = str_to_torch_dtype["mxfp4_e2m1"]
    q1, s1 = downcast_to_mxfp(w1, wdt, axis=2)  # row-major: the harness's layout
    q2, s2 = downcast_to_mxfp(w2, wdt, axis=2)
    rw = {
        "w1": q1.contiguous(), "w2": q2.contiguous(),
        "w1_scale": s1.contiguous().reshape(E * 2 * inter, -1),
        "w2_scale": s2.contiguous().reshape(E * D, -1),
    }  # fmt: skip
    S = SimpleNamespace(E=E, D=D, I=inter)
    mine = g.relayout_weights(rw, S, None, arch="gfx950")
    w1_il = torch.empty_like(w1)
    w1_il[:, 0::2], w1_il[:, 1::2] = w1[:, :inter], w1[:, inter:]
    gw1, gs1 = downcast_to_mxfp(w1_il.transpose(1, 2).contiguous(), wdt, axis=1)
    gw2, gs2 = downcast_to_mxfp(w2.transpose(1, 2).contiguous(), wdt, axis=1)
    ref_s1 = shuffle_scale_moe(gs1, arch="gfx950", preshuffle_factor=32, scale_kwidth=8)
    ref_s2 = shuffle_scale_moe(gs2, arch="gfx950", preshuffle_factor=32, scale_kwidth=8)
    assert torch.equal(mine["W1"], gw1) and torch.equal(mine["W2"], gw2)
    assert torch.equal(mine["S1"], ref_s1) and torch.equal(mine["S2"], ref_s2)
    assert (
        mine["W1"].stride() == gw1.stride() and mine["S1"].stride() == ref_s1.stride()
    )


def test_it_declares_numerics_the_accuracy_rules_accept():
    from op_tests.moe_bench import accuracy

    n = accuracy.numerics_of(spec(), g.GluonA16W4(), {})
    assert (
        n["act_quant"] == "none" and n["inter"] == "bf16" and n["inter_quant"] == "none"
    )
    assert n["combine"] == "bf16_rows_fp32_sum"


@needs_gpu
@pytest.mark.parametrize("M", [1, 16, 64, 256])
def test_routing_from_crafted_logits_selects_the_cases_ids_and_follows_a_swap(M):
    """aiter's own routing() on the crafted logits (Triton, any arch): same ids as the case's,
    re-checked when the accuracy check swaps ctx.tw/ti, and the top-k kernel times alone.
    """
    from op_tests.moe_bench import accuracy

    S = spec(token=M, model_dim=256, inter_dim=256)
    tw, ti = workload.make_routing(S, M)
    ctx = SimpleNamespace(S=S, tw=tw, ti=ti)
    cand = g.GluonA16W4()
    cand._k = ti.shape[1]
    cand._sync_inputs(ctx)
    rd = cand._route()[0]
    g.check_routed_ids(rd.topk_ids, ti)
    first = cand._tw
    tw2, ti2 = workload.make_routing(S, M, seed_tag="swap")
    ctx.tw, ctx.ti = tw2, ti2
    cand._sync_inputs(ctx)
    assert cand._tw is tw2 and cand._tw is not first
    g.check_routed_ids(cand._route()[0].topk_ids, ti2)
    bad = ti2.clone()
    bad[0, 0] = (bad[0, 0] + 1) % S.E if bad[0, 0] + 1 not in bad[0] else bad[0, 0]
    if not torch.equal(bad, ti2):
        with pytest.raises(g.GluonHarnessError):
            g.check_routed_ids(cand._route()[0].topk_ids, bad)
    for st in (True, False):
        out = cand.adjust_timing(
            {"median_us": 1e6, "p10_us": 1e6, "p90_us": 1e6}, st, True
        )
        assert 0 < out["topk_us"] < 1e4 and out["median_us"] == pytest.approx(
            1e6 - out["topk_us"]
        )
    assert accuracy is not None


@needs_gpu
def test_prepare_and_call_wire_the_side_api_arguments(monkeypatch):
    """prepare() builds the weights and checks the routing; the closure calls the side API
    twice with the case's hidden states, the gather/scatter of one routing(), the swiglu
    parameters and the weights' gammas (the GEMMs are stubbed: they need gfx950)."""
    import aiter.ops.triton.moe.moe_op_gemm_a16w4 as api

    S = spec(token=16, model_dim=256, inter_dim=256)
    E, D, inter, M = S.E, S.D, S.I, 16
    dev = "cuda"
    rw = {
        "w1": torch.randint(0, 256, (E, 2 * inter, D // 2), dtype=torch.uint8, device=dev),
        "w2": torch.randint(0, 256, (E, D, inter // 2), dtype=torch.uint8, device=dev),
        "w1_scale": torch.randint(120, 130, (E * 2 * inter, D // 32), dtype=torch.uint8, device=dev),
        "w2_scale": torch.randint(120, 130, (E * D, inter // 32), dtype=torch.uint8, device=dev),
    }  # fmt: skip
    bias = (torch.randn(E, 2 * inter, device=dev), torch.randn(E, D, device=dev))
    tw, ti = workload.make_routing(S, M)
    ctx = SimpleNamespace(
        S=S, aw=object(), ref_weights=rw, bias=bias, group_cache={}, tw=tw, ti=ti,
        hidden_static=workload.make_inputs(S, M), dispatch_meta={"gfx": "gfx950"},
    )  # fmt: skip
    calls = []

    def fake(
        x, w, xs, ws, xst, qst, b, rd, gather, scatter, gammas, layout, dt, swiglu, **kw
    ):
        calls.append({"x": x, "w": w, "b": b, "rd": rd, "gather": gather, "scatter": scatter, "gammas": gammas, "swiglu": swiglu, "kw": kw})  # fmt: skip
        rows = M * rd.n_expts_act
        return torch.zeros(rows, inter if swiglu else D, dtype=dt, device=dev)

    monkeypatch.setattr(api, "moe_gemm_a16w4", fake)
    monkeypatch.delenv(prebuild.NO_BUILD_ENV, raising=False)
    monkeypatch.delenv("FLYDSL_RUNTIME_RUN_ONLY", raising=False)
    cand = g.GluonA16W4()
    cand.prepare(ctx)
    out = cand.call(ctx)()
    assert out.shape[-1] == D
    first, second = calls
    assert first["x"] is ctx.hidden_static and first["swiglu"] is True
    assert first["gather"] is not None and first["scatter"] is None
    assert first["kw"] == {"alpha": 1.702, "limit": 7.0, "backend": "gluon"}
    assert first["b"].shape == (E, 2 * inter) and first["b"].dtype == torch.float32
    assert second["swiglu"] is False and second["kw"] == {"backend": "gluon"}
    assert second["scatter"] is not None and second["gammas"] is second["rd"].gate_scal
    assert second["b"].shape == (E, D)
    # the accuracy check swapping the routing re-crafts the logits for the new ids
    tw2, ti2 = workload.make_routing(S, M, seed_tag="x")
    ctx.tw, ctx.ti = tw2, ti2
    cand.call(ctx)()
    assert cand._tw is tw2
    ctx.tw = (
        torch.softmax(torch.randn(M, S.topk, device=dev), dim=1) * 2
    )  # not summing to 1
    with pytest.raises(g.GluonHarnessError, match="do not sum to 1"):
        cand.call(ctx)()
