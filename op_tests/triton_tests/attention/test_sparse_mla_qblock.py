# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL sparse MLA correctness for the GLM-5.3-Flash contract."""

from __future__ import annotations

import pytest
import torch

from aiter import dtypes
from aiter.ops.flydsl.sparse_mla_pp_kernels import sparse_mla_pingpong_fwd_flydsl
from aiter.ops.flydsl.sparse_mla_qblock_kernels import (
    sparse_mla_one_query_fwd_flydsl,
    sparse_mla_qblock_fwd_flydsl,
)
from aiter.ops.triton.attention.pa_decode_sparse import (
    _qblock_reuse_wins,
    pa_decode_sparse,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import assertAllclose

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx942",
    reason="FlyDSL sparse MLA is gfx942-only (64 KiB LDS)",
)

H, D = 16, 512


def _check_accuracy(ref, out, **kwargs):
    assertAllclose(ref, out, **kwargs)
    torch.testing.assert_close(ref, out, rtol=1e-2, atol=1e-2)


def _indices(pattern: str, n_tok: int, pool: int, topk: int):
    if pattern == "shared":
        return torch.arange(topk, dtype=torch.int32, device="cuda").repeat(n_tok)
    seed = torch.randint(0, pool, (n_tok, 1), dtype=torch.int32, device="cuda")
    k = torch.arange(topk, dtype=torch.int32, device="cuda")
    return ((k * 17 + seed) % pool).reshape(-1).contiguous()


def _indptr(n_tok, topk):
    return torch.arange(0, (n_tok + 1) * topk, topk, device="cuda", dtype=torch.int32)


def _torch_flat(q, kv, indices, topk, sink=None, scale=None):
    n_tok = q.shape[0]
    if scale is None:
        scale = q.shape[-1] ** -0.5
    gathered = kv[indices.view(n_tok, topk).long()].float()
    scores = torch.einsum("thd,tkd->thk", q.float(), gathered) * scale
    if sink is None:
        probs = torch.softmax(scores, dim=-1)
    else:
        sink_b = sink.float().view(1, -1, 1).expand(n_tok, q.shape[1], 1)
        probs = torch.softmax(torch.cat([scores, sink_b], dim=-1), dim=-1)[..., :topk]
    return torch.einsum("thk,tkd->thd", probs, gathered).to(q.dtype)


def _torch_csr(q, kv, indices, indptr, scale, sink):
    rows = []
    for t in range(q.shape[0]):
        gathered = kv[indices[indptr[t] : indptr[t + 1]].long()].float()
        scores = torch.einsum("hd,kd->hk", q[t].float(), gathered) * scale
        sink_b = sink.float().view(-1, 1)
        probs = torch.softmax(torch.cat([scores, sink_b], dim=-1), dim=-1)[:, :-1]
        rows.append(torch.einsum("hk,kd->hd", probs, gathered))
    return torch.stack(rows).to(q.dtype)


@pytest.mark.parametrize("n_tok", [7, 16])
@pytest.mark.parametrize("pattern", ["shared", "random"])
@pytest.mark.parametrize("kernel", ["one_q", "qblock", "pingpong"])
def test_flydsl_matches_torch(n_tok, pattern, kernel):
    torch.manual_seed(0)
    topk, pool = 2048, 8192
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices(pattern, n_tok, pool, topk)
    out = torch.empty_like(q)
    indptr = _indptr(n_tok, topk)
    if kernel == "one_q":
        sparse_mla_one_query_fwd_flydsl(q, kv, indices, out, kv_indptr=indptr)
    elif kernel == "pingpong":
        sparse_mla_pingpong_fwd_flydsl(q, kv, indices, out, kv_indptr=indptr)
    else:
        sparse_mla_qblock_fwd_flydsl(q, kv, indices, out, kv_indptr=indptr)
    ref = _torch_flat(q, kv, indices, topk)
    _check_accuracy(ref.to(dtypes.fp32), out.to(dtypes.fp32), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("kernel", ["one_q", "qblock", "pingpong"])
def test_flydsl_attn_sink(kernel):
    torch.manual_seed(3)
    n_tok, topk, pool = 8, 64, 8192
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices("shared", n_tok, pool, topk)
    sink = torch.randn(H, dtype=torch.float32, device="cuda")
    scale = 0.037
    out = torch.empty_like(q)
    common = {
        "kv_indptr": _indptr(n_tok, topk),
        "attn_sink": sink,
        "softmax_scale": scale,
    }
    if kernel == "one_q":
        sparse_mla_one_query_fwd_flydsl(q, kv, indices, out, **common)
    elif kernel == "pingpong":
        sparse_mla_pingpong_fwd_flydsl(q, kv, indices, out, **common)
    else:
        sparse_mla_qblock_fwd_flydsl(q, kv, indices, out, block_q=4, **common)
    ref = _torch_flat(q, kv, indices, topk, sink, scale)
    _check_accuracy(ref.to(dtypes.fp32), out.to(dtypes.fp32), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("pattern,qblock_reuse", [("shared", False), ("shared", True)])
def test_pa_decode_sparse_dispatch(pattern, qblock_reuse, monkeypatch):
    torch.manual_seed(2)
    n_tok, topk, pool = 16, 256, 8192
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices(pattern, n_tok, pool, topk)
    sink = torch.zeros(H, dtype=torch.float32, device="cuda")
    scale = D**-0.5
    calls = {"one_q": 0, "qblock": 0}
    import aiter.ops.flydsl.sparse_mla_qblock_kernels as _fly

    real_one = _fly.sparse_mla_one_query_fwd_flydsl
    real_qb = _fly.sparse_mla_qblock_fwd_flydsl

    def _wrap_one(*a, **k):
        calls["one_q"] += 1
        return real_one(*a, **k)

    def _wrap_qb(*a, **k):
        calls["qblock"] += 1
        return real_qb(*a, **k)

    monkeypatch.setattr(_fly, "sparse_mla_one_query_fwd_flydsl", _wrap_one)
    monkeypatch.setattr(_fly, "sparse_mla_qblock_fwd_flydsl", _wrap_qb)
    out = pa_decode_sparse(
        q,
        kv,
        indices,
        _indptr(n_tok, topk),
        sink,
        scale,
        has_invalid=False,
        kv_splits=1,
        qblock_reuse=qblock_reuse,
    )
    if qblock_reuse:
        assert calls["qblock"] == 1 and calls["one_q"] == 0
    else:
        assert calls["one_q"] == 1 and calls["qblock"] == 0
    ref = _torch_flat(q, kv, indices, topk, sink, scale)
    _check_accuracy(ref.to(dtypes.fp32), out.to(dtypes.fp32), rtol=2e-3, atol=2e-3)


def test_pa_decode_sparse_default_splits_skips_flydsl(monkeypatch):
    """Default kv_splits=None must keep Triton's auto split-K, not FlyDSL."""
    torch.manual_seed(6)
    n_tok, topk, pool = 8, 64, 1024
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices("shared", n_tok, pool, topk)
    sink = torch.zeros(H, dtype=torch.float32, device="cuda")
    import aiter.ops.flydsl.sparse_mla_qblock_kernels as _fly

    monkeypatch.setattr(
        _fly,
        "sparse_mla_one_query_fwd_flydsl",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("FlyDSL one_q")),
    )
    monkeypatch.setattr(
        _fly,
        "sparse_mla_qblock_fwd_flydsl",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("FlyDSL qblock")),
    )
    out = pa_decode_sparse(
        q, kv, indices, _indptr(n_tok, topk), sink, D**-0.5, has_invalid=False
    )
    ref = _torch_flat(q, kv, indices, topk, sink, D**-0.5)
    _check_accuracy(ref.to(dtypes.fp32), out.to(dtypes.fp32), rtol=2e-3, atol=2e-3)


def test_qblock_probe_requires_every_row():
    topk = 16
    shared = _indices("shared", 4, 64, topk)
    assert _qblock_reuse_wins(shared, 4, topk)
    rows = torch.stack(
        [(torch.arange(topk, device="cuda") + t * 8) % 64 for t in range(4)]
    ).to(torch.int32)
    rows[3] = rows[0]
    assert not _qblock_reuse_wins(rows.reshape(-1), 4, topk)


def test_unpacked_csr_falls_back():
    torch.manual_seed(4)
    n_tok, topk, pool = 2, 16, 64
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    rows = _indices("random", n_tok, pool, topk).view(n_tok, topk)
    indices = torch.cat(
        [rows.reshape(-1), torch.zeros(32, dtype=torch.int32, device="cuda")]
    )
    indptr = _indptr(n_tok, topk)
    sink = torch.zeros(H, dtype=torch.float32, device="cuda")
    scale = D**-0.5
    out = pa_decode_sparse(
        q, kv, indices, indptr, sink, scale, has_invalid=False, kv_splits=1
    )
    lying = torch.tensor([0, 8, 32], dtype=torch.int32, device="cuda")
    hinted = torch.arange(32, dtype=torch.int32, device="cuda")
    out_hint = pa_decode_sparse(
        q,
        kv,
        hinted,
        lying,
        sink,
        scale,
        has_invalid=False,
        kv_splits=1,
        uniform_topk=16,
    )
    _check_accuracy(
        _torch_flat(q, kv, rows.reshape(-1), topk, sink, scale).to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
    )
    _check_accuracy(
        _torch_csr(q, kv, hinted, lying, scale, sink).to(dtypes.fp32),
        out_hint.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
    )


def test_noncontiguous_q_falls_back():
    torch.manual_seed(5)
    n_tok, topk, pool = 4, 16, 64
    buf = torch.randn(n_tok, H, D + 8, dtype=torch.bfloat16, device="cuda")
    q = buf[:, :, :D]
    assert not q.is_contiguous()
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices("shared", n_tok, pool, topk)
    sink = torch.zeros(H, dtype=torch.float32, device="cuda")
    scale = D**-0.5
    out = pa_decode_sparse(
        q,
        kv,
        indices,
        _indptr(n_tok, topk),
        sink,
        scale,
        has_invalid=False,
        kv_splits=1,
    )
    ref = _torch_flat(q.contiguous(), kv, indices, topk, sink, scale)
    _check_accuracy(ref.to(dtypes.fp32), out.to(dtypes.fp32), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "argument", ["inv_rope_positions", "inv_rope_cos_sin_cache", "out_mxfp8"]
)
@pytest.mark.parametrize("block_h", [None, 16])
def test_gfx942_rejects_output_epilogue(argument, block_h):
    q = torch.ones(1, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.ones(16, D, dtype=torch.bfloat16, device="cuda")
    sink = torch.zeros(H, dtype=torch.float32, device="cuda")
    with pytest.raises((NotImplementedError, AssertionError), match="epilogue"):
        pa_decode_sparse(
            q,
            kv,
            _indices("shared", 1, 16, 16),
            _indptr(1, 16),
            sink,
            D**-0.5,
            has_invalid=False,
            kv_splits=1,
            block_h=block_h,
            **{argument: (q, q) if argument == "out_mxfp8" else q},
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize(
    "wrapper",
    [
        sparse_mla_one_query_fwd_flydsl,
        sparse_mla_qblock_fwd_flydsl,
        sparse_mla_pingpong_fwd_flydsl,
    ],
)
def test_flydsl_rejects_invalid_scale(scale, wrapper):
    q = torch.ones(1, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.ones(32, D, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="finite and positive"):
        wrapper(
            q,
            kv,
            _indices("shared", 1, 32, 32),
            torch.empty_like(q),
            softmax_scale=scale,
        )


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_pa_decode_sparse_nonpositive_scale(scale, monkeypatch):
    import aiter.ops.flydsl.sparse_mla_qblock_kernels as fly

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid scale reached FlyDSL")

    monkeypatch.setattr(fly, "sparse_mla_one_query_fwd_flydsl", forbidden)
    monkeypatch.setattr(fly, "sparse_mla_qblock_fwd_flydsl", forbidden)
    q = torch.ones(2, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.ones(16, D, dtype=torch.bfloat16, device="cuda")
    sink = torch.zeros(H, dtype=torch.float32, device="cuda")
    indices = _indices("shared", 2, 16, 16)
    out = pa_decode_sparse(
        q,
        kv,
        indices,
        _indptr(2, 16),
        sink,
        scale,
        has_invalid=False,
        kv_splits=1,
        qblock_reuse=True,
    )
    torch.testing.assert_close(
        out.float(),
        _torch_flat(q, kv, indices, 16, sink, scale).float(),
        rtol=1e-2,
        atol=1e-2,
    )


def test_flydsl_runtime_arch_helper(monkeypatch):
    import aiter.ops.flydsl.sparse_mla_qblock_kernels as fly
    from aiter.jit.utils import chip_info

    monkeypatch.setenv("GPU_ARCHS", "gfx950")
    chip_info.get_gfx_custom_op_core.cache_clear()
    chip_info.get_gfx.cache_clear()
    try:
        assert chip_info.get_gfx() == "gfx950"
        assert fly.get_gfx_runtime() == "gfx942"
        fly._require_gfx942()
    finally:
        chip_info.get_gfx_custom_op_core.cache_clear()
        chip_info.get_gfx.cache_clear()
    monkeypatch.setattr(fly, "get_gfx_runtime", lambda: "gfx950")
    with pytest.raises(RuntimeError, match="gfx942-only"):
        fly._require_gfx942()


def test_index_span_metadata():
    import aiter.ops.flydsl.sparse_mla_qblock_kernels as fly

    assert fly._index_span_supported(1, (1 << 30) - 1)
    assert not fly._index_span_supported(1, 1 << 30)
    assert not fly._index_span_supported(16384, 65552)
    assert fly._index_span_supported(4096, 65552)
    with pytest.raises(ValueError, match="index byte span"):
        fly._launch(None, None, None, None, 16384, 65552, 16384, 1, None, None)


def test_public_index_span_falls_back(monkeypatch):
    import importlib
    from types import SimpleNamespace

    pa = importlib.import_module("aiter.ops.triton.attention.pa_decode_sparse")
    q = SimpleNamespace(shape=(16384, H, D), is_contiguous=lambda: True)
    kv = SimpleNamespace(shape=(131072, D), ndim=2, is_contiguous=lambda: True)
    monkeypatch.setattr(pa, "_packed_uniform_topk", lambda *args: 65552)
    assert (
        pa._pa_decode_sparse_gfx942_flydsl(
            q, kv, None, None, D**-0.5, None, kv_splits=1
        )
        is None
    )


def test_public_packed_index_span_is_not_overconservative(monkeypatch):
    import importlib
    from types import SimpleNamespace

    import aiter.ops.flydsl.sparse_mla_qblock_kernels as fly

    pa = importlib.import_module("aiter.ops.triton.attention.pa_decode_sparse")
    q = SimpleNamespace(shape=(16384, H, D), dtype=torch.bfloat16)
    packed = (SimpleNamespace(shape=(4096, 65552)), 65552, True)
    sentinel = object()
    monkeypatch.setattr(fly, "suggest_block_q", lambda *args: 4)
    monkeypatch.setattr(fly, "pack_qblock_union", lambda *args: packed)
    monkeypatch.setattr(pa, "_check_out", lambda *args: None)

    def launch(*args, **kwargs):
        assert kwargs["packed"] is packed
        assert kwargs["block_q"] == 4
        return sentinel

    monkeypatch.setattr(fly, "sparse_mla_qblock_fwd_flydsl", launch)
    assert (
        pa._pa_decode_sparse_gfx942_qblock(
            q, None, None, None, D**-0.5, None, uniform_topk=65552
        )
        is sentinel
    )


def test_csr_terminal_offset_metadata():
    import importlib
    from types import SimpleNamespace

    import aiter.ops.flydsl.sparse_mla_qblock_kernels as fly

    pa = importlib.import_module("aiter.ops.triton.attention.pa_decode_sparse")
    indices = SimpleNamespace(numel=lambda: 65536 * 65536)
    indptr = SimpleNamespace(numel=lambda: 65537, dtype=torch.int32)
    assert pa._packed_uniform_topk(indices, indptr, 65536, 65536) is None
    with pytest.raises(ValueError, match="terminal offset"):
        fly._uniform_topk(indices, 65536, indptr, 65536)


@pytest.mark.parametrize("block_q", [2, 4])
def test_flydsl_odd_tail_sink(block_q):
    torch.manual_seed(1215)
    n_tok, topk = 7, 2048
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(8192, D, dtype=torch.bfloat16, device="cuda")
    sink = torch.randn(H, dtype=torch.float32, device="cuda")
    indices = _indices("shared", n_tok, 8192, topk)
    out = torch.empty_like(q)
    sparse_mla_qblock_fwd_flydsl(q, kv, indices, out, block_q=block_q, attn_sink=sink)
    ref = _torch_flat(q, kv, indices, topk, sink)
    torch.testing.assert_close(ref.float(), out.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "kernel,n_tok",
    [("one_q", 7), ("qblock", 7), ("pingpong", 7), ("pingpong", 4097)],
)
@pytest.mark.parametrize("case", ["sharp", "large_v", "duplicates_sink"])
def test_flydsl_finite_stress(kernel, n_tok, case):
    torch.manual_seed(1215)
    topk, pool = 2048, 131072
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices("shared", n_tok, pool, topk)
    sink = None
    if case == "sharp":
        q.mul_(16)
    elif case == "large_v":
        # A stale probability near 2**60 times V=2**100 overflows FP32.
        # The existing rescale threshold must keep this finite BF16 case safe.
        q.zero_()
        q[..., 0] = 1
        kv.zero_()
        kv[:, 1] = 2.0**100
        kv[16:topk, 0] = 944
    else:
        indices = (torch.arange(topk, dtype=torch.int32, device="cuda") % 256).repeat(
            n_tok
        )
        sink = torch.linspace(-2, 2, H, dtype=torch.float32, device="cuda")
    out = torch.empty_like(q)
    fn = {
        "one_q": sparse_mla_one_query_fwd_flydsl,
        "qblock": sparse_mla_qblock_fwd_flydsl,
        "pingpong": sparse_mla_pingpong_fwd_flydsl,
    }[kernel]
    fn(q, kv, indices, out, kv_indptr=_indptr(n_tok, topk), attn_sink=sink)
    ref = torch.cat(
        [
            _torch_flat(
                q[t : t + 64], kv, indices[t * topk : (t + 64) * topk], topk, sink
            )
            for t in range(0, n_tok, 64)
        ]
    )
    assert torch.isfinite(ref).all()
    assert torch.isfinite(out).all()
    _check_accuracy(ref.float(), out.float(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("topk", [16, 48])
def test_pingpong_rejects_unpaired_tiles(topk):
    q = torch.ones(1, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.ones(64, D, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="divisible by 32"):
        sparse_mla_pingpong_fwd_flydsl(
            q, kv, _indices("shared", 1, 64, topk), torch.empty_like(q)
        )


def test_pingpong_rejects_arch_and_index_span(monkeypatch):
    import aiter.ops.flydsl.sparse_mla_pp_kernels as pp
    import aiter.ops.flydsl.sparse_mla_qblock_kernels as common

    q = torch.ones(1, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.ones(32, D, dtype=torch.bfloat16, device="cuda")
    idx = _indices("shared", 1, 32, 32)
    monkeypatch.setattr(common, "get_gfx_runtime", lambda: "gfx950")
    with pytest.raises(RuntimeError, match="gfx942-only"):
        sparse_mla_pingpong_fwd_flydsl(q, kv, idx, torch.empty_like(q))
    monkeypatch.setattr(common, "get_gfx_runtime", lambda: "gfx942")
    # Metadata-only overflow: reject before dtype conversion or kernel dispatch.
    monkeypatch.setattr(pp, "_require_contract", lambda *args: (16384, 32))
    monkeypatch.setattr(pp, "_uniform_topk", lambda *args, **kwargs: (idx, 65536))
    with pytest.raises(ValueError, match="index byte span"):
        sparse_mla_pingpong_fwd_flydsl(q, kv, idx, torch.empty_like(q))


def test_pingpong_public_lazy_export():
    from aiter.ops.flydsl import sparse_mla_pingpong_fwd_flydsl as exported

    assert exported is sparse_mla_pingpong_fwd_flydsl


@pytest.mark.parametrize(
    "operand", ["q", "kv", "out", "kv_indices", "kv_indptr", "attn_sink"]
)
def test_pingpong_rejects_cpu_operands(operand, monkeypatch):
    import aiter.ops.flydsl.sparse_mla_pp_kernels as pp

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "device validation must precede CSR handling and compilation"
        )

    monkeypatch.setattr(pp, "_uniform_topk", forbidden)
    monkeypatch.setattr(pp, "compile_sparse_mla_pp", forbidden)
    monkeypatch.setattr(pp, "_run_compiled", forbidden)
    values = {}
    for name, shape, dtype in (
        ("q", (1, H, D), torch.bfloat16),
        ("kv", (32, D), torch.bfloat16),
        ("out", (1, H, D), torch.bfloat16),
        ("kv_indices", (32,), torch.int32),
        ("kv_indptr", (2,), torch.int32),
        ("attn_sink", (H,), torch.float32),
    ):
        values[name] = torch.empty(
            shape, dtype=dtype, device="cpu" if name == operand else "cuda"
        )
    with pytest.raises(ValueError, match="GPU"):
        sparse_mla_pingpong_fwd_flydsl(**values)


@pytest.mark.parametrize(
    "operand", ["kv", "out", "kv_indices", "kv_indptr", "attn_sink"]
)
def test_pingpong_rejects_cross_device_metadata(operand, monkeypatch):
    from types import SimpleNamespace

    import aiter.ops.flydsl.sparse_mla_pp_kernels as pp

    # Metadata only: do not touch or require a second physical GPU.
    monkeypatch.setattr(pp, "_require_contract", lambda *args: (1, 32))

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "device validation must precede CSR handling and compilation"
        )

    monkeypatch.setattr(pp, "_uniform_topk", forbidden)
    monkeypatch.setattr(pp, "compile_sparse_mla_pp", forbidden)
    monkeypatch.setattr(pp, "_run_compiled", forbidden)
    values = {
        name: SimpleNamespace(device=torch.device("cuda", int(name == operand)))
        for name in ("q", "kv", "out", "kv_indices", "kv_indptr", "attn_sink")
    }
    with pytest.raises(ValueError, match=f"{operand} must be on the same GPU"):
        sparse_mla_pingpong_fwd_flydsl(**values)


@pytest.mark.parametrize(
    "scale",
    [1e-50, 1e-46, 2.0**-150, 2.3586576387363357e38, 3e38, 3.4028234663852886e38, 1e39],
)
@pytest.mark.parametrize("has_sink", [False, True])
def test_pingpong_rejects_fp32_scale_edges(scale, has_sink, monkeypatch):
    import aiter.ops.flydsl.sparse_mla_pp_kernels as pp

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported scale reached compilation or launch")

    monkeypatch.setattr(pp, "compile_sparse_mla_pp", forbidden)
    monkeypatch.setattr(pp, "_run_compiled", forbidden)
    q = torch.zeros(1, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.ones(32, D, dtype=torch.bfloat16, device="cuda")
    sink = torch.zeros(H, dtype=torch.float32, device="cuda") if has_sink else None
    with pytest.raises(ValueError, match="finite and positive in FP32"):
        sparse_mla_pingpong_fwd_flydsl(
            q,
            kv,
            _indices("shared", 1, 32, 32),
            torch.empty_like(q),
            softmax_scale=scale,
            attn_sink=sink,
        )


@pytest.mark.parametrize("scale", [2.0**-149, 2.0**-126, 0.037, 2.3586574359122396e38])
@pytest.mark.parametrize("has_sink", [False, True])
def test_pingpong_representable_scale_edges(scale, has_sink, monkeypatch):
    import struct

    import aiter.ops.flydsl.sparse_mla_pp_kernels as pp

    checked = struct.unpack("f", struct.pack("f", scale))[0]
    assert pp._checked_scale(scale) == checked
    run = pp._run_compiled

    def capture(*args):
        assert args[7] == checked  # The validated FP32 scale is the launch argument.
        return run(*args)

    monkeypatch.setattr(pp, "_run_compiled", capture)
    q = torch.zeros(1, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.ones(32, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices("shared", 1, 32, 32)
    sink = (
        torch.linspace(-2, 2, H, dtype=torch.float32, device="cuda")
        if has_sink
        else None
    )
    out = sparse_mla_pingpong_fwd_flydsl(
        q, kv, indices, torch.empty_like(q), softmax_scale=scale, attn_sink=sink
    )
    assert torch.isfinite(out).all()
    ref = _torch_flat(q, kv, indices, 32, sink, checked)
    _check_accuracy(ref.float(), out.float(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("n_tok", [1, 3, 7])
@pytest.mark.parametrize("topk", [32, 64])
@pytest.mark.parametrize("has_sink", [False, True])
def test_pingpong_minimum_tiles_and_output_bounds(n_tok, topk, has_sink):
    torch.manual_seed(1215)
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(256, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices("random", n_tok, 256, topk)
    sink = (
        torch.linspace(-2, 2, H, dtype=torch.float32, device="cuda")
        if has_sink
        else None
    )
    guarded = torch.full((n_tok + 2, H, D), 42, dtype=q.dtype, device=q.device)
    out = guarded[1:-1]
    result = sparse_mla_pingpong_fwd_flydsl(
        q, kv, indices, out, attn_sink=sink, kv_indptr=_indptr(n_tok, topk)
    )
    assert result is out
    assert (guarded[0] == 42).all() and (guarded[-1] == 42).all()
    ref = _torch_flat(q, kv, indices, topk, sink)
    _check_accuracy(ref.float(), out.float(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("case", ["dtype", "heads", "dim", "out", "strides"])
def test_pingpong_rejects_geometry(case):
    q = torch.ones(1, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.ones(32, D, dtype=torch.bfloat16, device="cuda")
    if case == "dtype":
        q = q.float()
    elif case == "heads":
        q = q[:, :8]
    elif case == "dim":
        q = q[..., :256]
    elif case == "strides":
        q = q.expand(2, -1, -1)
    out = torch.empty_like(q)
    if case == "out":
        out = out.float()
    with pytest.raises((TypeError, ValueError)):
        sparse_mla_pingpong_fwd_flydsl(q, kv, _indices("shared", 1, 32, 32), out)


@pytest.mark.parametrize("capacity", ["tokens", "kv"])
def test_pingpong_rejects_offset_capacity(capacity):
    import aiter.ops.flydsl.sparse_mla_qblock_kernels as common

    n_tok = common._MAX_TOKENS if capacity == "tokens" else 1
    n_kv = common._MAX_KV_ROWS if capacity == "kv" else 32
    # Metadata-only tensors exercise overflow rejection without huge allocations.
    q = torch.empty(n_tok, H, D, dtype=torch.bfloat16, device="meta")
    kv = torch.empty(n_kv, D, dtype=torch.bfloat16, device="meta")
    indices = torch.empty(32, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="int32 offsets"):
        sparse_mla_pingpong_fwd_flydsl(q, kv, indices, torch.empty_like(q))


def test_flydsl_prefill_odd_tail_sink():
    torch.manual_seed(1215)
    n_tok, topk, pool = 4097, 2048, 131072
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices("random", n_tok, pool, topk)
    sink = torch.randn(H, dtype=torch.float32, device="cuda")
    out = torch.empty_like(q)
    sparse_mla_one_query_fwd_flydsl(q, kv, indices, out, attn_sink=sink)
    ref = torch.cat(
        [
            _torch_flat(
                q[t : t + 128], kv, indices[t * topk : (t + 128) * topk], topk, sink
            )
            for t in range(0, n_tok, 128)
        ]
    )
    assert torch.isfinite(out).all()
    _check_accuracy(ref.float(), out.float(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    "n_tok,topk,prefill",
    [(7, 2048, False), (4095, 2048, False), (4096, 2048, True), (4096, 64, False)],
)
def test_flydsl_prefill_selection(n_tok, topk, prefill, monkeypatch):
    import aiter.ops.flydsl.sparse_mla_qblock_kernels as fly

    class Selected(Exception):
        pass

    def compile_probe(block_q, has_sink, **kwargs):
        assert kwargs == {"prefill": prefill}
        raise Selected

    monkeypatch.setattr(fly, "compile_sparse_mla_qblock", compile_probe)
    with pytest.raises(Selected):
        fly._launch(None, None, None, None, n_tok, topk, n_tok, 1, None, None)
