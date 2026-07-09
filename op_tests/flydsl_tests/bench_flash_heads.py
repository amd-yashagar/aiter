#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Device-self-time bench for the sparse MLA prefill kernel at DSv4 pro (H=128)
vs flash (H=8) head counts, on gfx942.

Metric: summed CUDA/HIP device self-time per iter (torch profiler), same as
bench_sparse_mla_prefill.py.  The H=8-vs-H=128 ratio at an IDENTICAL KV
workload (same tokens/topk/CSR) tells us whether the op is memory/load bound:
the KV gather + software V-transpose use all 8 warps regardless of head count,
so if H=8 time ~ H=128 time the head-MFMA (N axis) is hidden under memory
traffic (=> the naive head parametrization is near-optimal for flash).

Usage:
    python op_tests/flydsl_tests/bench_flash_heads.py --T 4096 --topk-main 512 --topk-extra 128
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_AITER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if os.path.isdir(os.path.join(_AITER_ROOT, "aiter")) and _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)

from op_tests.flydsl_tests.sparse_mla_prefill_ref import (  # noqa: E402
    NOPE_HEAD_DIM,
    PACKED_HEAD_DIM,
    ROPE_HEAD_DIM,
    default_scale,
    gen_kv,
    gen_q,
    gen_ragged_rows,
    identity_block_table,
    pack_fp8_ds_mla_cache,
    ragged_from_rows,
)


def _device_self_ms(fn, warmup: int, iters: int):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per_kernel = {}
    total_us = 0.0
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", 0.0) or getattr(e, "self_cuda_time_total", 0.0)
        if us <= 0:
            continue
        total_us += us
        per_kernel[e.key] = per_kernel.get(e.key, 0.0) + us
    inv = 1.0 / (iters * 1000.0)
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1])[:3]
    return total_us * inv, [(k.split("(")[0].split("<")[0][:36], v * inv) for k, v in top]


def build_2region(T, H, topk_main, topk_extra, main_tokens, extra_tokens, block_size, seed, device="cuda"):
    main_kv = gen_kv(main_tokens, seed=seed)
    extra_kv = gen_kv(extra_tokens, seed=seed + 1)
    main_cache = pack_fp8_ds_mla_cache(main_kv, block_size, is_extra=False)
    extra_cache = pack_fp8_ds_mla_cache(extra_kv, block_size, is_extra=True)
    main_rows = gen_ragged_rows(T, topk_main, main_tokens, seed=seed + 2)
    extra_rows = gen_ragged_rows(T, topk_extra, extra_tokens, seed=seed + 3)
    m_idx, m_iptr = ragged_from_rows(main_rows, torch.device(device))
    e_idx, e_iptr = ragged_from_rows(extra_rows, torch.device(device))
    q = gen_q(T, H, seed=seed + 4)
    out = torch.empty(T, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    return dict(
        q=q, out=out,
        main_cache=main_cache, m_idx=m_idx, m_iptr=m_iptr,
        main_bt=identity_block_table(main_tokens, block_size, torch.device(device)),
        extra_cache=extra_cache, e_idx=e_idx, e_iptr=e_iptr,
        extra_bt=identity_block_table(extra_tokens, block_size, torch.device(device)),
        block_size=block_size,
    )


def run_2region(inp):
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill_2region

    def fn():
        flydsl_sparse_mla_prefill_2region(
            inp["q"], inp["out"],
            inp["main_cache"], inp["m_idx"], inp["m_iptr"], inp["main_bt"],
            inp["extra_cache"], inp["e_idx"], inp["e_iptr"], inp["extra_bt"],
            block_size=inp["block_size"], main_is_fnuz=True, extra_is_fnuz=False,
        )
    return fn


def build_single(T, H, topk, num_tokens, block_size, seed, device="cuda"):
    kv = gen_kv(num_tokens, seed=seed)
    cache = pack_fp8_ds_mla_cache(kv, block_size, is_extra=False)
    rows = gen_ragged_rows(T, topk, num_tokens, seed=seed + 1)
    idx, iptr = ragged_from_rows(rows, torch.device(device))
    q = gen_q(T, H, seed=seed + 2)
    out = torch.empty(T, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=device)
    return dict(q=q, out=out, cache=cache, idx=idx, iptr=iptr,
                bt=identity_block_table(num_tokens, block_size, torch.device(device)),
                block_size=block_size)


def run_single(inp):
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    def fn():
        flydsl_sparse_mla_prefill(
            inp["q"], inp["cache"], inp["idx"], inp["iptr"], inp["out"],
            block_table=inp["bt"], block_size=inp["block_size"], packed=True,
        )
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--topk-main", type=int, default=512)
    ap.add_argument("--topk-extra", type=int, default=128)
    ap.add_argument("--topk-single", type=int, default=512)
    ap.add_argument("--main-tokens", type=int, default=65536)
    ap.add_argument("--extra-tokens", type=int, default=32768)
    ap.add_argument("--num-tokens", type=int, default=65536)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heads", type=int, nargs="+", default=[128, 8])
    args = ap.parse_args()

    arch = torch.cuda.get_device_properties(0).gcnArchName.lower().split(":")[0]
    print(f"bench_flash_heads gfx={arch} T={args.T} block_size={args.block_size} "
          f"warmup={args.warmup} iters={args.iters} metric=device_self_ms")
    print(f"=== 2-region (topk_main={args.topk_main}, topk_extra={args.topk_extra}) ===")
    base = None
    for H in args.heads:
        inp = build_2region(args.T, H, args.topk_main, args.topk_extra,
                             args.main_tokens, args.extra_tokens, args.block_size, args.seed + 100)
        fn = run_2region(inp)
        fn()
        ms, top = _device_self_ms(fn, args.warmup, args.iters)
        if base is None:
            base = ms
        print(f"  H={H:4d}  device_self={ms:8.4f} ms   ({ms/base:.2f}x vs H={args.heads[0]})")
        print(f"           top: {', '.join(f'{k}={v:.4f}' for k, v in top)}")

    print(f"=== single-region packed DSv4 (topk={args.topk_single}) ===")
    base = None
    for H in args.heads:
        inp = build_single(args.T, H, args.topk_single, args.num_tokens, args.block_size, args.seed + 200)
        fn = run_single(inp)
        fn()
        ms, top = _device_self_ms(fn, args.warmup, args.iters)
        if base is None:
            base = ms
        print(f"  H={H:4d}  device_self={ms:8.4f} ms   ({ms/base:.2f}x vs H={args.heads[0]})")
        print(f"           top: {', '.join(f'{k}={v:.4f}' for k, v in top)}")


if __name__ == "__main__":
    main()
