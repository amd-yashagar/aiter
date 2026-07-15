# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Honest A/B: FlyDSL block-score vs sglang's Triton
``_flash_attn_fwd_with_block_score_kernel`` (MiniMax-M3 sparse-prefill lightning
indexer, DEPLOYED score-only path: disable_index_value=True).

Both kernels are timed on the SAME inputs (built once, fed to both), the SAME
median/p5/p95 device-synced methodology, the SAME output dtype/layout, and BOTH
are correctness-gated against the SAME naive torch reference (bf16-appropriate
tolerance, NOT widened) before any timing is reported. All shapes are FIXED and
documented (no random sequence lengths): the deployment trace shape
q=[8192,1,128] bf16, index-K bf16 head_dim=128, page/pool 128, over fixed
context lengths 40k/80k/128k.

Reference (784 us/call in the MXFP4 TP-0 prefill trace) is the extracted Triton
kernel in ``triton_block_score.py`` (verified to reproduce the trace envelope).
"""

import argparse
import sys
import time

import torch

from triton_block_score import triton_block_score
from aiter.ops.flydsl.kernels.minimax_block_score import flydsl_minimax_block_score

dev = "cuda"


def build(context, total_q=8192, D=128, bs_k=128, seed=0):
    torch.manual_seed(seed)
    prefix = context - total_q
    assert prefix >= 0, "context must be >= total_q"
    q = torch.randn(total_q, 1, D, device=dev, dtype=torch.bfloat16)
    k_cache = torch.randn(context, 1, D, device=dev, dtype=torch.bfloat16)
    req_to_token = torch.arange(context, device=dev, dtype=torch.int32).view(1, context)
    slot_ids = torch.zeros(1, device=dev, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, total_q], device=dev, dtype=torch.int32)
    seq_lens = torch.tensor([context], device=dev, dtype=torch.int32)
    prefix_lens = torch.tensor([prefix], device=dev, dtype=torch.int32)
    return dict(
        q=q, k_cache=k_cache, req_to_token=req_to_token, slot_ids=slot_ids,
        cu_seqlens=cu_seqlens, seq_lens=seq_lens, prefix_lens=prefix_lens,
        max_seqlen_q=total_q, max_seqlen_k=context, block_size_k=bs_k,
    )


def naive_ref(inp, score_type):
    q = inp["q"][:, 0, :].float()
    ctx = inp["max_seqlen_k"]
    prefix = int(inp["prefix_lens"][0].item())
    k = inp["k_cache"][:ctx, 0, :].float()
    Q = q.shape[0]
    sm = (q.shape[1] ** -0.5) * 1.4426950409
    nblk = (ctx + 127) // 128
    out = torch.full((Q, nblk), float("-inf"), device=dev)
    # tile the query dim to bound memory for the [Q, ctx] score matrix
    qpos = torch.arange(Q, device=dev) + prefix
    kpos = torch.arange(ctx, device=dev)
    TILE = 1024
    for r0 in range(0, Q, TILE):
        r1 = min(r0 + TILE, Q)
        qk = (q[r0:r1] @ k.t()) * sm
        qk = torch.where(qpos[r0:r1, None] >= kpos[None, :], qk, float("-inf"))
        for b in range(nblk):
            lo, hi = b * 128, min((b + 1) * 128, ctx)
            seg = qk[:, lo:hi]
            smax = seg.max(dim=1).values
            if score_type == "max":
                out[r0:r1, b] = smax
            else:
                lse = smax + torch.log2(torch.exp2(seg - smax[:, None]).sum(dim=1))
                lse = torch.where(lse != lse, float("-inf"), lse)
                out[r0:r1, b] = lse
    return out


def med_time(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2], ts[max(0, len(ts)//20)], ts[min(len(ts)-1, len(ts)-1-len(ts)//20)]


def correctness(out, ref):
    m = ref != float("-inf")
    mask_ok = torch.equal(out == float("-inf"), ref == float("-inf"))
    if m.any():
        maxabs = (out[m] - ref[m]).abs().max().item()
        denom = ref[m].abs().clamp(min=1e-3)
        relmax = ((out[m] - ref[m]).abs() / denom).max().item()
    else:
        maxabs = relmax = 0.0
    return maxabs, relmax, mask_ok


def run(context, score_type, fly_cfgs, iters):
    inp = build(context)
    ref = naive_ref(inp, score_type)
    # tolerances (bf16 QK -> f32 score). max path is exact-ish; lse has exp/log approx.
    tol_abs = 0.05 if score_type == "max" else 0.25

    # ---- Triton reference ----
    sc_tri = torch.full((1, inp["q"].shape[0], (context + 127)//128),
                        float("-inf"), device=dev, dtype=torch.float32)
    def fn_tri():
        return triton_block_score(score=sc_tri, score_type=score_type, **inp)
    fn_tri(); torch.cuda.synchronize()
    tri_maxabs, tri_relmax, tri_mask = correctness(sc_tri[0], ref)
    tri_pass = tri_mask and tri_maxabs <= tol_abs
    tri_med = tri_p5 = tri_p95 = float("nan")
    if tri_pass:
        tri_med, tri_p5, tri_p95 = med_time(fn_tri, iters)

    print(f"\n== context={context} score_type={score_type} "
          f"(q=[8192,1,128] bf16, K bf16 D=128, page=128) ==", flush=True)
    print(f"  Triton : {tri_med:8.1f} us [{tri_p5:.1f}-{tri_p95:.1f}]  "
          f"maxabs={tri_maxabs:.4f} mask={tri_mask} pass={tri_pass}", flush=True)

    best = None
    for cfg in fly_cfgs:
        bq, kg, w, lds = cfg
        sc_fly = torch.full((1, inp["q"].shape[0], (context + 127)//128),
                            float("-inf"), device=dev, dtype=torch.float32)
        def fn_fly():
            return flydsl_minimax_block_score(
                score=sc_fly, score_type=score_type, block_q=bq, k_group=kg,
                waves_per_block=w, use_lds=bool(lds), **inp)
        try:
            fn_fly(); torch.cuda.synchronize()
        except Exception as e:
            print(f"  FlyDSL bq={bq} kg={kg}: COMPILE/RUN ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        f_maxabs, f_relmax, f_mask = correctness(sc_fly[0], ref)
        f_pass = f_mask and f_maxabs <= tol_abs
        f_med = f_p5 = f_p95 = float("nan")
        if f_pass:
            f_med, f_p5, f_p95 = med_time(fn_fly, iters)
        ratio = (f_med / tri_med) if (tri_med == tri_med and f_med == f_med) else float("nan")
        speedup = (tri_med / f_med) if (tri_med == tri_med and f_med == f_med) else float("nan")
        print(f"  FlyDSL bq={bq:3d} kg={kg:3d} w={w} lds={lds}: {f_med:8.1f} us [{f_p5:.1f}-{f_p95:.1f}]  "
              f"maxabs={f_maxabs:.4f} relmax={f_relmax:.3f} mask={f_mask} pass={f_pass}  "
              f"fly/tri={ratio:.2f}x speedup={speedup:.2f}x", flush=True)
        if f_pass and (best is None or f_med < best[4]):
            best = (bq, kg, w, lds, f_med, speedup)
    if best is not None:
        print(f"  BEST FlyDSL: bq={best[0]} kg={best[1]} w={best[2]} lds={best[3]} "
              f"{best[4]:.1f}us speedup={best[5]:.2f}x vs Triton", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="*", default=[40960, 81920, 131072])
    ap.add_argument("--score-type", nargs="*", default=["max"], choices=["max", "lse"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--fly-cfgs", type=str, default="64:16:1,64:16:2,64:16:4,64:32:4")
    args = ap.parse_args()
    def _parse(c):
        p = [int(x) for x in c.split(":")]
        return (p[0], p[1], p[2] if len(p) > 2 else 1, p[3] if len(p) > 3 else 0)
    fly_cfgs = [_parse(c) for c in args.fly_cfgs.split(",")]
    print(f"# device={torch.cuda.get_device_name(0)} fly_cfgs={fly_cfgs}", flush=True)
    for st in args.score_type:
        for ctx in args.contexts:
            run(ctx, st, fly_cfgs, args.iters)


if __name__ == "__main__":
    main()
