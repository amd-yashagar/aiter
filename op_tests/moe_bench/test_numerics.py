# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerics a kernel declares reach the reference builder, in core's vocabulary."""

import sys
import types
from typing import NamedTuple

import pytest

from op_tests.moe_bench import accuracy
from op_tests.moe_bench.bar_worker import RoutedFusedMoe


class LastRoute(NamedTuple):
    kernel: str
    family: str
    gpu: str
    reason: str
    numerics: dict | None


class OldLastRoute(NamedTuple):
    kernel: str
    family: str
    gpu: str
    reason: str


def _S(qtype="per_1x128", aq="fp8"):
    import aiter
    from aiter import dtypes

    return types.SimpleNamespace(
        qtype=getattr(aiter.QuantType, qtype), aq=getattr(dtypes, aq)
    )


def _decl(**kw):
    d = {"act_quant": "per_1x128_fp8", "inter": "fp32_direct", "combine": "fp32_reduce"}
    d.update(kw)
    return d


def test_core_vocabulary_is_accepted_and_recorded():
    S = _S()
    n = accuracy.normalize_numerics(S, _decl())
    assert accuracy.numerics_key(n) == (
        "act_quant=per_1x128_fp8;inter=fp32_direct;combine=fp32_reduce"
    )
    n = accuracy.normalize_numerics(
        S, _decl(inter="bf16_then_per_1x128_fp8", combine="bf16_atomic")
    )
    assert n["inter"] == "bf16_then_per_1x128_fp8"
    n = accuracy.normalize_numerics(
        _S("per_Token"), _decl(act_quant="none", inter="bf16")
    )
    assert n["act_quant"] == "none"


@pytest.mark.parametrize(
    "bad",
    [
        {"act_quant": "hip"},
        {"inter": "fp32"},
        {"inter": "bf16_then_mx_fp4"},
        {"act_quant": "per_token_fp8"},  # case is per_1x128
        {"combine": "atomic"},
        {"extra_key": 1},
    ],
)
def test_unknown_or_misfit_values_fail_loudly(bad):
    with pytest.raises(accuracy.NumericsError):
        accuracy.normalize_numerics(_S(), _decl(**bad))


def test_missing_key_fails_loudly():
    with pytest.raises(accuracy.NumericsError):
        accuracy.normalize_numerics(_S(), {"act_quant": "none"})


def test_numerics_of_uses_declared_then_same_as_then_route():
    S = _S()
    cand = types.SimpleNamespace(declared_numerics=_decl())
    assert accuracy.numerics_of(S, cand, {})["inter"] == "fp32_direct"
    same = types.SimpleNamespace(declared_numerics={"same_as": "aiter dispatched path"})
    ctx = types.SimpleNamespace(dispatch_route={"stages": "1-stage"})
    n = accuracy.numerics_of(S, same, {}, ctx)
    assert n["inter"] == "fp32_direct" and n["act_quant"] == "per_1x128_fp8"
    with pytest.raises(accuracy.NumericsError):
        accuracy.numerics_of(S, same, {}, types.SimpleNamespace(dispatch_route=None))
    fell_back = types.SimpleNamespace(declared_numerics=None)
    n = accuracy.numerics_of(S, fell_back, {"kernel1": "some_ck_stage1"})
    assert n["inter"] == "bf16_then_per_1x128_fp8"


def test_probe_route_reads_last_route_numerics(monkeypatch):
    import aiter

    decl = _decl()
    lr = LastRoute("k1", "f", "g", "r", decl)
    fake = types.ModuleType("aiter.moe")
    fake.last_route = lambda: lr
    monkeypatch.setitem(sys.modules, "aiter.moe", fake)
    monkeypatch.setattr(aiter, "moe", fake, raising=False)
    cand = RoutedFusedMoe()
    cand.probe_route("eager")
    assert cand.declared_numerics == decl

    # a 4-field last_route (older core): fall back to aiter.moe.numerics(id)
    lr = OldLastRoute("k1", "f", "g", "r")
    fake.numerics = lambda kid: {
        "act_quant": "none",
        "inter": "bf16",
        "combine": "bf16_atomic",
    }
    cand.probe_route("eager")
    assert cand.declared_numerics["inter"] == "bf16"

    # declined: aiter's own path served the call, the route-derived numerics apply
    lr = LastRoute("aiter", "", "", "declined", None)
    cand.probe_route("eager")
    assert cand.declared_numerics is None


def test_fp32_direct_reference_builds_and_differs_from_bf16_round():
    """fp64 stage-1 output went straight into aiter's quant op on the fp32_direct path
    ("Unsupported dtype: torch.float64"); the reference must hand it fp32."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    import aiter
    from op_tests.moe_bench import schema, workload

    row = {c: "" for c in schema.COLS}
    row.update(
        case_id="t-fp32direct",
        kind="fused",
        family="fp8-block",
        token="64",
        model_dim="512",
        inter_dim="256",
        expert="8",
        topk="2",
        act_type="Silu",
        dtype="bf16",
        q_dtype_a="fp8",
        q_dtype_w="fp8",
        q_type="per_1x128",
        use_g1u1="1",
        doweight_stage1="0",
        modes="warm",
    )
    S = workload.resolve(schema.Case(row, {}))
    _aw, rw, bias = workload.build_group(S)
    hs, tw, ti = accuracy.acc_inputs(S, 64, 0)
    rows = torch.arange(32, device="cuda")
    refs = {}
    for inter in ("fp32_direct", "bf16_then_per_1x128_fp8"):
        n = accuracy.normalize_numerics(
            S, {"act_quant": "per_1x128_fp8", "inter": inter, "combine": "bf16_atomic"}
        )
        refs[inter] = accuracy.path_ref(S, rw, bias, hs, tw, ti, None, rows, n)
    assert all(
        r.dtype == torch.float64 and torch.isfinite(r).all() for r in refs.values()
    )
    assert not torch.equal(*refs.values())
    import aiter.fused_moe as fm

    assert fm.dtypes.fp32 == torch.float32  # fp64_math restored
    assert torch.get_default_dtype() == torch.float32
    assert aiter is not None
