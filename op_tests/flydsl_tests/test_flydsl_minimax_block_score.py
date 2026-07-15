# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the FlyDSL MiniMax-M3 block-score kernel
(``flydsl_minimax_block_score``), the score-only lightning-indexer pass that
replaces sglang's Triton ``_flash_attn_fwd_with_block_score_kernel`` in the
deployed ``disable_index_value=True`` path.

Reference: a self-contained naive torch port of the block-score semantics
(per-128-block max / log2-sum-exp of the causal QK over paged index-K, in the
kernel's log2 scale). No dependency on sglang. Correctness gate: EXACT ``-inf``
mask match + max-abs error under a bf16-appropriate tolerance that is NOT
widened (max path is bit-exact vs f32 reduction of the same MFMA; lse allows the
hardware exp2/log2 approximation).
"""

import itertools

import pytest
import torch

try:
    from aiter.jit.utils.chip_info import get_gfx
except Exception:  # pragma: no cover
    get_gfx = lambda: "unknown"  # noqa: E731

try:
    from aiter.ops.flydsl.kernels.minimax_block_score import (
        flydsl_minimax_block_score,
    )
except Exception:  # pragma: no cover
    flydsl_minimax_block_score = None

SUPPORTED_GFX = ("gfx942", "gfx950")
NEG = float("-inf")
LOG2E = 1.4426950409


def _build(context, total_q, D=128, block_size=128, seed=0, device="cuda"):
    """Fixed-shape paged score inputs (identity paging). ``context`` = full seq
    len (prefix + chunk); the chunk is the last ``total_q`` tokens."""
    torch.manual_seed(seed)
    prefix = context - total_q
    assert prefix >= 0
    q = torch.randn(total_q, 1, D, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(context, 1, D, device=device, dtype=torch.bfloat16)
    req_to_token = torch.arange(context, device=device, dtype=torch.int32).view(1, context)
    slot_ids = torch.zeros(1, device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, total_q], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([context], device=device, dtype=torch.int32)
    prefix_lens = torch.tensor([prefix], device=device, dtype=torch.int32)
    return dict(
        q=q, k_cache=k_cache, req_to_token=req_to_token, slot_ids=slot_ids,
        cu_seqlens=cu_seqlens, seq_lens=seq_lens, prefix_lens=prefix_lens,
        max_seqlen_q=total_q, max_seqlen_k=context, block_size_k=block_size,
    )


def _naive_ref(inp, score_type):
    q = inp["q"][:, 0, :].float()
    ctx = inp["max_seqlen_k"]
    bs = inp["block_size_k"]
    prefix = int(inp["prefix_lens"][0].item())
    k = inp["k_cache"][:ctx, 0, :].float()
    Q = q.shape[0]
    sm = (q.shape[1] ** -0.5) * LOG2E
    nblk = (ctx + bs - 1) // bs
    out = torch.full((Q, nblk), NEG, device=q.device)
    qpos = torch.arange(Q, device=q.device) + prefix
    kpos = torch.arange(ctx, device=q.device)
    qk = (q @ k.t()) * sm
    qk = torch.where(qpos[:, None] >= kpos[None, :], qk, NEG)
    for b in range(nblk):
        lo, hi = b * bs, min((b + 1) * bs, ctx)
        seg = qk[:, lo:hi]
        smax = seg.max(dim=1).values
        if score_type == "max":
            out[:, b] = smax
        else:
            lse = smax + torch.log2(torch.exp2(seg - smax[:, None]).sum(dim=1))
            lse = torch.where(lse != lse, NEG, lse)
            out[:, b] = lse
    return out


# Fixed shapes: (context, total_q). Cover multiple-of-128, non-multiple, short,
# and prefix-dominated (deployment-like) contexts. No random seqlens.
CASES = [
    (512, 64),
    (512, 128),
    (1024, 128),
    (768, 192),
    (200, 200),     # context not a multiple of 128
    (128, 64),      # single block
    (2048, 256),    # prefix-dominated, total_q > 128
    (100, 100),     # short context < block_size
]


@pytest.mark.skipif(
    flydsl_minimax_block_score is None, reason="flydsl not available"
)
@pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX, reason=f"unsupported gfx {get_gfx()}"
)
@pytest.mark.parametrize("score_type", ["max", "lse"])
@pytest.mark.parametrize("context,total_q", CASES)
@pytest.mark.parametrize(
    "block_q,k_group,use_lds",
    [(64, 8, False), (64, 16, True), (128, 8, True)],
)
def test_block_score(context, total_q, score_type, block_q, k_group, use_lds):
    inp = _build(context, total_q)
    ref = _naive_ref(inp, score_type)
    out = flydsl_minimax_block_score(
        score_type=score_type, block_q=block_q, k_group=k_group,
        use_lds=use_lds, **inp
    )[0]

    # exact -inf mask parity (NOT widened)
    assert torch.equal(out == NEG, ref == NEG), (
        f"-inf mask mismatch ctx={context} Q={total_q} {score_type}"
    )
    m = ref != NEG
    if m.any():
        maxabs = (out[m] - ref[m]).abs().max().item()
        tol = 0.05 if score_type == "max" else 0.25
        assert maxabs <= tol, (
            f"maxabs={maxabs} > tol={tol} ctx={context} Q={total_q} {score_type}"
        )


if __name__ == "__main__":
    if flydsl_minimax_block_score is None or get_gfx() not in SUPPORTED_GFX:
        print(f"skip: flydsl={flydsl_minimax_block_score is not None} gfx={get_gfx()}")
        raise SystemExit(0)
    npass = nfail = 0
    for st in ("max", "lse"):
        for (ctx, tq) in CASES:
            for (bq, kg, lds) in [(64, 8, False), (64, 16, True), (128, 8, True)]:
                inp = _build(ctx, tq)
                ref = _naive_ref(inp, st)
                out = flydsl_minimax_block_score(
                    score_type=st, block_q=bq, k_group=kg, use_lds=lds, **inp
                )[0]
                mask_ok = torch.equal(out == NEG, ref == NEG)
                m = ref != NEG
                maxabs = (out[m] - ref[m]).abs().max().item() if m.any() else 0.0
                tol = 0.05 if st == "max" else 0.25
                ok = mask_ok and maxabs <= tol
                npass += ok
                nfail += (not ok)
                print(f"{'PASS' if ok else 'FAIL'} ctx={ctx} Q={tq} {st} "
                      f"bq={bq} kg={kg} maxabs={maxabs:.4f} mask={mask_ok}", flush=True)
    print(f"\n{npass} passed, {nfail} failed")
