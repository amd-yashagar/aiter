# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL whole-graph candidates must pass the router check (they read '<lambda>')."""

import pytest

from op_tests.moe_bench import bar_runner

# Qwen3.5-397B-A17B PTPC decode (tier1-00145): fp8-token, per_Token, M=8.
ROW = {
    "case_id": "t-wg-decode",
    "kind": "fused",
    "family": "fp8-token",
    "model": "Qwen3.5-397B-A17B",
    "token": "8",
    "model_dim": "4096",
    "inter_dim": "128",
    "expert": "513",
    "topk": "11",
    "act_type": "Silu",
    "dtype": "bf16",
    "q_dtype_a": "fp8",
    "q_dtype_w": "fp8",
    "q_type": "per_Token",
    "use_g1u1": "1",
    "doweight_stage1": "0",
    "modes": "warm",
}


def test_impl_tag_names_the_lambda():
    pytest.importorskip("torch")
    import aiter.fused_moe as fm
    from aiter.fused_moe_registry import (
        make_fused_moe_impl_kernel_name,
        register_fused_moe_impl,
    )
    from op_tests.moe_bench import routes

    register_fused_moe_impl("flydsl_test", lambda request, config: None)
    name = make_fused_moe_impl_kernel_name("flydsl_test", "16_16_16")
    routes.install_impl_tag()
    bound = fm.resolve_fused_moe_impl(name)
    assert bound.__name__ == "<lambda>" and bound.name == name

    class MD:
        stage1 = stage2 = None
        flat = 0
        output_aux = None
        full_impl = bound
        run_1stage = False
        block_m = 16
        ksplit = 0
        fuse_quant = None
        prequant = True
        skip_inter_quant = False

    d = routes.describe_metadata(MD)
    assert d["kernel1"] == name and d["stages"] == "whole-graph"
    assert d["backend1"] == "flydsl-graph"
    MD.full_impl = lambda r: None
    with pytest.raises(RuntimeError, match="no kernel name"):
        routes.describe_metadata(MD)


def test_whole_graph_candidate_admitted_for_fp8_token_decode():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from op_tests.moe_bench import candidates, routes, schema, workload
    from op_tests.moe_bench.bar_worker import DispatchCandidate, _gpu_identity
    from op_tests.moe_bench.candidates.fused_cfg import FusedCfgSource

    ident = _gpu_identity()
    if ident["gfx"] != "gfx942":
        pytest.skip("whole-graph candidates are enumerated on gfx942")
    row = {c: "" for c in schema.COLS}
    row.update(ROW)
    case = schema.Case(row, {})
    S = workload.resolve(case)
    aw, _rw, bias = workload.build_group(S)
    tw, ti = workload.make_routing(S, case.token)
    hidden = workload.make_inputs(S, case.token)
    kw = workload.fused_kwargs(S, aw, bias, None)
    import tempfile

    ctx = candidates.Ctx(
        case,
        S,
        aw,
        bias,
        None,
        hidden,
        tw,
        ti,
        None,
        case.token,
        kw,
        tempfile.mkdtemp(),
    )
    ctx.dispatch_meta = ident
    ctx.hidden_static = hidden.clone()
    ctx.outbuf = None
    with routes.RouteRecorder() as rec:
        DispatchCandidate().call(ctx)()
        torch.cuda.synchronize()
    ctx.dispatch_args = rec.last()[0]
    graph = [c for c in FusedCfgSource().enumerate(ctx) if c.group == "whole-graph"]
    assert graph, "no whole-graph candidates enumerated for an fp8-token decode case"
    admitted, rejected = [], []
    for c in graph:
        c.prepare(ctx)
        try:
            with routes.RouteRecorder() as rec:
                c.call(ctx)()
                torch.cuda.synchronize()
            reason = c.verify(routes.route_from_record(rec.last()))
        finally:
            c.release(ctx)
        (rejected if reason else admitted).append(reason or c.id)
    assert not any("<lambda>" in r for r in rejected), rejected[:3]
    assert admitted, f"every whole-graph candidate was discarded: {rejected[:3]}"
    assert bar_runner is not None


def test_source_discarded_everywhere_is_flagged(tmp_path):
    from op_tests.moe_bench.bar_worker import _accounting

    enum = {"fused_moe:*": 3, "fused_moe:flydsl-graph": 2, "fused_moe:asm": 1}
    stats = {
        "fused_moe:flydsl-graph": {"total": 2, "ok": 0, "discarded": 2, "unnamed": 2},
        "fused_moe:asm": {"total": 1, "ok": 1, "discarded": 0},
    }
    acct, why = _accounting(enum, stats, [], "bar")
    assert "fused_moe:flydsl-graph: 2 candidates resolved to an unnamed impl" in why
    assert "asm" not in why
    assert acct["fused_moe:flydsl-graph"]["enumerated"] == 2
    # aiter declining a candidate (a real other kernel resolved) is not a harness defect
    declined = {"fused_moe:flydsl-graph": {"total": 2, "ok": 0, "discarded": 2}}
    assert _accounting(enum, declined, [], "bar")[1] == ""

    import csv
    import json

    cols = ["source", "kernels", "status"]
    with open(tmp_path / "candidates-t.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(3):
            w.writerow(
                {
                    "source": "fused_moe",
                    "kernels": f"impl__flydsl_gfx942__{i}",
                    "status": "router-discarded",
                }
            )
        w.writerow(
            {"source": "fused_moe", "kernels": "moe_ck2stages_x", "status": "ok"}
        )
    (tmp_path / "bar-t.meta.json").write_text("{}")
    bad = bar_runner.check_discarded_sources(str(tmp_path), "t", "bar")
    assert bad == ["fused_moe:flydsl-graph"]
    meta = json.loads((tmp_path / "bar-t.meta.json").read_text())
    assert meta["router_discarded_sources"] == {"fused_moe:flydsl-graph": 3}


def test_relabel_finds_only_lambda_rows():
    from op_tests.moe_bench import relabel

    rows = [
        {"case_id": "a", "kernel1": "<lambda>", "full_impl": "<lambda>"},
        {"case_id": "b", "kernel1": "moe_ck2stages_x", "full_impl": ""},
        {"case_id": "c", "kernel1": "impl__flydsl_gfx942__16_16_16", "full_impl": ""},
    ]
    assert [r["case_id"] for r in relabel.affected_rows(rows)] == ["a"]
