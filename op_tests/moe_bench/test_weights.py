# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The weight builder at expert counts and inter dims where e8m0_shuffle pads across experts."""

import pytest

torch = pytest.importorskip("torch")

from op_tests.moe_bench import accuracy, schema, workload

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# (name, model_dim, inter_dim, experts, topk, extra): MOEBUG-040 shapes
CASES = [
    ("inter160_E512", 2560, 160, 512, 10, ""),  # tier1-00517, unpadded
    ("inter256_pad96_E513", 2560, 256, 513, 11, "intermediate_pad=96"),  # 00518/00519
    ("inter576_E384", 5120, 576, 384, 6, ""),  # tier1-00520
]


def _spec(model_dim, inter, experts, topk, extra):
    row = {c: "" for c in schema.COLS}
    row.update(
        {
            "case_id": f"t-w-{inter}-{experts}",
            "kind": "fused",
            "family": "a4w4",
            "token": "16",
            "model_dim": str(model_dim),
            "inter_dim": str(inter),
            "expert": str(experts),
            "topk": str(topk),
            "act_type": "Silu",
            "dtype": "bf16",
            "q_dtype_a": "fp4x2",
            "q_dtype_w": "fp4x2",
            "q_type": "per_1x32",
            "use_g1u1": "1",
            "doweight_stage1": "0",
            "extra": extra,
            "modes": "warm",
        }
    )
    return workload.resolve(schema.Case(row, schema.parse_extra(extra)))


@pytest.mark.parametrize("name,d,i,e,topk,extra", CASES, ids=[c[0] for c in CASES])
def test_scales_are_one_shuffle_over_all_experts(name, d, i, e, topk, extra):
    from aiter.utility import fp4_utils

    S = _spec(d, i, e, topk, extra)
    assert workload._expert_chunk(S) < S.E_local or S.E_local * 0 == 0
    aw, rw, _bias = workload.build_group(S)
    n1 = 2 * S.I
    for key, rows in (("w1_scale", S.E_local * n1), ("w2_scale", S.E_local * S.D)):
        got = aw[key]
        assert got.shape[0] == workload._pad_rows(rows), (key, tuple(got.shape))
        # the scale the references read is exactly E x rows x columns, unshuffled
        ref = rw[key]
        ref2d = ref.reshape(rows, -1)
        want = fp4_utils.e8m0_shuffle(
            ref2d.view(torch.uint8).view(torch.float8_e8m0fnu)
        )
        assert tuple(want.shape) == tuple(got.shape)
        assert torch.equal(
            got.view(torch.uint8)[:rows], want.view(torch.uint8)[:rows]
        ), f"{key}: chunked build differs from one shuffle over all experts"
        # padding rows past the real ones are zero
        assert not got.view(torch.uint8)[rows:].any()


@pytest.mark.parametrize("name,d,i,e,topk,extra", CASES, ids=[c[0] for c in CASES])
def test_reference_path_exists_for_the_case(name, d, i, e, topk, extra):
    """The padded layout (zero-weight padding) gets a ref_path like the unpadded model."""
    import aiter.fused_moe as fm

    S = _spec(d, i, e, topk, extra)
    _aw, rw, bias = workload.build_group(S)
    hs, tw, ti = accuracy.acc_inputs(S, 16, 0)
    rows = torch.arange(16, device="cuda")
    route = {"stages": "2-stage", "backend1": "ck", "backend2": "ck", "kernel1": "k"}
    n = accuracy.numerics_from_route(S, route)
    assert n["act_quant"].startswith("mx_fp4_") and fm is not None
    refs = accuracy.references(S, rw, bias, hs, tw, ti, None, rows, n)
    assert "path" in refs and "exact" in refs
    for r in refs.values():
        assert torch.isfinite(r).all() and r.abs().max() > 0


def test_padded_layout_reference_equals_the_unpadded_model():
    """inter 256 with intermediate_pad 96 is the inter-160 model plus zero-weight columns:
    slicing the padded reference weights down to the 160 real columns and scoring them as
    an inter-160 case gives the same ref_path."""
    S160 = _spec(256, 160, 8, 2, "")
    S256 = _spec(256, 256, 8, 2, "intermediate_pad=96")
    _aw, rw, bias = workload.build_group(S256)
    E, D, I1, I2 = S256.E_local, S256.D, S160.I, S256.I
    w1, w2 = rw["w1"], rw["w2"]
    assert not w1.view(torch.uint8)[:, I1:I2].any(), "padded gate rows are zero"
    assert not w1.view(torch.uint8)[
        :, I2 + I1 : 2 * I2
    ].any(), "padded up rows are zero"
    s1 = rw["w1_scale"].view(torch.uint8).reshape(E, 2 * I2, -1)
    s2 = rw["w2_scale"].view(torch.uint8).reshape(E, D, -1)
    cut = {
        "w1": torch.cat([w1[:, :I1], w1[:, I2 : I2 + I1]], 1).contiguous(),
        "w2": w2[:, :, : I1 // 2].contiguous(),
        "w1_scale": torch.cat([s1[:, :I1], s1[:, I2 : I2 + I1]], 1)
        .reshape(E * 2 * I1, -1)
        .contiguous()
        .view(rw["w1_scale"].dtype),
        "w2_scale": s2[:, :, : I1 // 32]
        .reshape(E * D, -1)
        .contiguous()
        .view(rw["w2_scale"].dtype),
    }
    hs, tw, ti = accuracy.acc_inputs(S256, 16, 0)
    rows = torch.arange(16, device="cuda")
    route = {"stages": "2-stage", "backend1": "ck", "backend2": "ck", "kernel1": "k"}
    padded = accuracy.path_ref(
        S256,
        rw,
        bias,
        hs,
        tw,
        ti,
        None,
        rows,
        accuracy.numerics_from_route(S256, route),
    )
    plain = accuracy.path_ref(
        S160,
        cut,
        bias,
        hs,
        tw,
        ti,
        None,
        rows,
        accuracy.numerics_from_route(S160, route),
    )
    assert torch.allclose(padded, plain, rtol=1e-10, atol=1e-12), (
        (padded - plain).abs().max().item()
    )
