# op_tests/test_topk_softmax.py
import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter.fused_moe import fused_topk
from aiter.ops.moe_op import topk_softmax

torch.set_default_device("cuda")


def topk_softmax_ref(gating_output, topk, need_renorm):
    """Pure PyTorch reference: full softmax + topk."""
    scores = torch.nn.functional.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = scores.topk(k=topk, dim=-1, largest=True, sorted=True)
    if need_renorm:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids.to(torch.int32)


def _run_topk_softmax(gating_output, topk, need_renorm):
    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32)
    topk_indices = torch.empty(num_tokens, topk, dtype=torch.int32)
    token_expert_indices = torch.empty(num_tokens, topk, dtype=torch.int32)
    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        need_renorm,
    )
    return topk_weights, topk_indices


@pytest.mark.parametrize("num_tokens", [1, 32, 512, 4096])
@pytest.mark.parametrize("num_experts", [8, 64, 128])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("need_renorm", [True, False])
def test_topk_softmax(num_tokens, num_experts, topk, need_renorm):
    if topk > num_experts:
        pytest.skip("topk > num_experts")

    torch.manual_seed(42)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32)

    weights_ref, ids_ref = topk_softmax_ref(gating_output, topk, need_renorm)
    topk_weights, topk_indices = _run_topk_softmax(gating_output, topk, need_renorm)

    torch.testing.assert_close(topk_weights, weights_ref, atol=1e-4, rtol=1e-4)

    for row in range(min(num_tokens, 16)):
        assert set(topk_indices[row].tolist()) == set(
            ids_ref[row].tolist()
        ), f"Row {row}: expert sets differ"


def test_topk_softmax_n_zero():
    pytest.skip(
        "topk_softmax ASM kernel does not support 0 tokens (invalid HIP launch)"
    )


def test_topk_softmax_bf16_input():
    """ASM topk_softmax expects fp32 logits; caller must cast bf16."""
    torch.manual_seed(42)
    gating_bf16 = torch.randn(32, 8, dtype=torch.bfloat16)
    gating_fp32 = gating_bf16.float()

    weights_ref, ids_ref = topk_softmax_ref(gating_fp32, 2, True)
    topk_weights, topk_indices = _run_topk_softmax(gating_fp32, 2, True)

    torch.testing.assert_close(topk_weights, weights_ref, atol=2e-3, rtol=2e-3)
    for row in range(16):
        assert set(topk_indices[row].tolist()) == set(
            ids_ref[row].tolist()
        ), f"Row {row}: expert sets differ"


@pytest.mark.parametrize(
    "num_experts,topk",
    [(128, 4), (128, 6), (128, 8), (256, 6), (256, 8), (384, 8), (640, 8), (257, 8)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gap", [20.0, 80.0, 100.0, 200.0])
@pytest.mark.parametrize("use_hip", [False, True], ids=["shifted-asm", "hip"])
def test_fused_topk_large_logit_gap(
    monkeypatch, num_experts, topk, dtype, gap, use_hip
):
    """One logit far above the rest: the softmax must not overflow.

    The asm top-k softmax kernel behind fused_topk computed exp() without
    subtracting the row max, so a gap of 88 or more gave NaN weights, the same id
    in every slot and the dominant expert dropped."""
    monkeypatch.setattr(fused_moe_module, "_USE_TOPK_SOFTMAX_HIP", use_hip)
    monkeypatch.setattr(fused_moe_module, "_USE_TOPK_SOFTMAX_SHIFT", True)
    num_tokens, dominant = 16, 3
    torch.manual_seed(0)
    hidden_states = torch.randn(num_tokens, 64, dtype=torch.bfloat16)
    gating_output = torch.randn(num_tokens, num_experts, dtype=dtype)
    gating_output[:, dominant] = gap

    topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk, True)

    assert torch.isfinite(topk_weights).all()
    assert (topk_ids[:, 0] == dominant).all()
    torch.testing.assert_close(
        topk_weights[:, 0], torch.ones(num_tokens), atol=1e-3, rtol=0
    )
    torch.testing.assert_close(
        topk_weights.sum(dim=-1), torch.ones(num_tokens), atol=1e-3, rtol=0
    )
    sorted_ids = topk_ids.sort(dim=-1).values
    assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all(), "duplicate expert ids"


def _by_weight(weights, ids):
    weights, order = weights.sort(dim=-1, descending=True, stable=True)
    return weights, ids.gather(-1, order)


@pytest.mark.parametrize(
    "num_experts,topk",
    [(128, 4), (128, 6), (128, 8), (256, 6), (256, 8), (384, 8), (640, 8)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gap", [0.0, 20.0, 60.0])
@pytest.mark.parametrize("num_tokens", [1, 16, 4096])
def test_fused_topk_matches_the_hip_kernel(
    monkeypatch, num_experts, topk, dtype, gap, num_tokens
):
    """The shifted asm path must pick the HIP kernel's ids exactly (the shift is in
    fp32, so near-tied logits keep their order) with the same weights."""
    monkeypatch.setattr(fused_moe_module, "_USE_TOPK_SOFTMAX_HIP", False)
    monkeypatch.setattr(fused_moe_module, "_USE_TOPK_SOFTMAX_SHIFT", True)
    torch.manual_seed(1)
    hidden_states = torch.randn(num_tokens, 64, dtype=torch.bfloat16)
    gating_output = torch.randn(num_tokens, num_experts, dtype=dtype)
    if gap:
        gating_output[:, 3] += gap

    weights, ids = fused_topk(hidden_states, gating_output, topk, True)

    ref_weights = torch.empty(num_tokens, topk, dtype=torch.float32)
    ref_ids = torch.empty(num_tokens, topk, dtype=torch.int32)
    topk_softmax(
        ref_weights,
        ref_ids,
        torch.empty(num_tokens, topk, dtype=torch.int32),
        gating_output,
        True,
    )
    weights, ids = _by_weight(weights, ids)
    ref_weights, ref_ids = _by_weight(ref_weights, ref_ids)
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "num_experts,topk",
    [(128, 8), (256, 6), (256, 8), (384, 8), (640, 8)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_topk_large_logit_gap_at_large_m(num_experts, topk, dtype):
    """Same as the gap test above at M = 4096, where fused_topk takes the HIP kernel
    for E <= 256 and the shifted asm kernel otherwise."""
    num_tokens, dominant = 4096, 3
    torch.manual_seed(0)
    hidden_states = torch.randn(num_tokens, 64, dtype=torch.bfloat16)
    gating_output = torch.randn(num_tokens, num_experts, dtype=dtype)
    gating_output[:, dominant] = 200.0

    topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk, True)

    assert torch.isfinite(topk_weights).all()
    assert (topk_ids[:, 0] == dominant).all()
    torch.testing.assert_close(
        topk_weights.sum(dim=-1), torch.ones(num_tokens), atol=1e-3, rtol=0
    )
    sorted_ids = topk_ids.sort(dim=-1).values
    assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all(), "duplicate expert ids"


@pytest.mark.parametrize(
    "num_experts,num_tokens,want",
    [
        (256, 16, "shifted"),
        (256, 1024, "shifted"),
        (256, 4096, "hip"),
        (128, 4096, "hip"),
        (384, 4096, "shifted"),
        (640, 32768, "shifted"),
    ],
)
def test_fused_topk_backend_choice(monkeypatch, num_experts, num_tokens, want):
    calls = []
    for name in ("topk_softmax", "topk_softmax_asm_shifted"):
        real = getattr(aiter, name)
        monkeypatch.setattr(
            aiter,
            name,
            lambda *a, _n=name, _r=real, **k: (calls.append(_n), _r(*a, **k))[1],
        )
    monkeypatch.setattr(fused_moe_module, "_USE_TOPK_SOFTMAX_HIP", False)
    monkeypatch.setattr(fused_moe_module, "_USE_TOPK_SOFTMAX_SHIFT", True)
    hidden_states = torch.randn(num_tokens, 64, dtype=torch.bfloat16)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16)

    fused_topk(hidden_states, gating_output, 8, True)

    assert calls == ["topk_softmax" if want == "hip" else "topk_softmax_asm_shifted"]


def test_fused_topk_concurrent_streams_do_not_share_scratch(monkeypatch):
    """Two streams call fused_topk with the same shape at the same time (two-batch
    overlap): each result must match the HIP kernel's."""
    monkeypatch.setattr(fused_moe_module, "_USE_TOPK_SOFTMAX_HIP", False)
    monkeypatch.setattr(fused_moe_module, "_USE_TOPK_SOFTMAX_SHIFT", True)
    num_tokens, num_experts, topk = 1024, 384, 8
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.manual_seed(3)
    gating = [
        torch.randn(num_tokens, num_experts, dtype=torch.bfloat16) * (1 + 4 * i)
        for i in range(2)
    ]
    hidden_states = torch.randn(num_tokens, 64, dtype=torch.bfloat16)
    refs = []
    for g in gating:
        ref_weights = torch.empty(num_tokens, topk, dtype=torch.float32)
        ref_ids = torch.empty(num_tokens, topk, dtype=torch.int32)
        topk_softmax(
            ref_weights,
            ref_ids,
            torch.empty(num_tokens, topk, dtype=torch.int32),
            g,
            True,
        )
        refs.append(_by_weight(ref_weights, ref_ids))
    torch.cuda.synchronize()

    for _ in range(100):
        outs = []
        for stream, g in zip(streams, gating):
            with torch.cuda.stream(stream):
                outs.append(fused_topk(hidden_states, g, topk, True))
        torch.cuda.synchronize()
        for (weights, ids), (ref_weights, ref_ids) in zip(outs, refs):
            weights, ids = _by_weight(weights, ids)
            assert torch.equal(ids, ref_ids)
            torch.testing.assert_close(weights, ref_weights, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("num_tokens", [1, 64])
def test_fused_topk_is_cuda_graph_safe(monkeypatch, num_tokens):
    monkeypatch.setattr(fused_moe_module, "_USE_TOPK_SOFTMAX_HIP", False)
    monkeypatch.setattr(fused_moe_module, "_USE_TOPK_SOFTMAX_SHIFT", True)
    num_experts, topk = 256, 8
    torch.manual_seed(5)
    hidden_states = torch.randn(num_tokens, 64, dtype=torch.bfloat16)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16)
    gating_output[:, 7] = 150.0
    fused_topk(hidden_states, gating_output, topk, True)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = fused_topk(hidden_states, gating_output, topk, True)
    gating_output.copy_(torch.randn(num_tokens, num_experts, dtype=torch.bfloat16))
    gating_output[:, 9] = 120.0
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(weights).all()
    assert (ids[:, 0] == 9).all()
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(num_tokens), atol=1e-3, rtol=0
    )


def test_topk_shift_scratch_is_per_stream_and_stable():
    gating_output = torch.randn(64, 256, dtype=torch.bfloat16)
    other = torch.cuda.Stream()
    here = fused_moe_module._topk_shift_buffer(gating_output)
    assert fused_moe_module._topk_shift_buffer(gating_output).data_ptr() == (
        here.data_ptr()
    )
    with torch.cuda.stream(other):
        there = fused_moe_module._topk_shift_buffer(gating_output)
        assert fused_moe_module._topk_shift_buffer(gating_output).data_ptr() == (
            there.data_ptr()
        )
    assert there.data_ptr() != here.data_ptr()
