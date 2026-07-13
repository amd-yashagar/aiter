# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device self-time baseline for the DSv4 512-flat sparse MLA prefill path.

Measures the FlyDSL kernel's *device* self-time (CUDA events, kernel is the only
op on the stream) for:
  - flat512 single-region,
  - flat512 two-region (SWA + compressed),
  - packed fp8_ds_mla two-region (the 584B layout) at the SAME logical workload,

so the flat512 (512B/token) path can be compared against the packed (584B/token)
path on identical shapes.  Reports median and [min, max] ms over ``--iters``.

Run:
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/bench_flat512.py \
        --sq 2048 --topk-main 512 --topk-extra 128 --block-size 64
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

_AITER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)

from op_tests.flydsl_tests.sparse_mla_prefill_ref import (
    gen_kv,
    gen_kv_flat512,
    gen_q_flat512,
    gen_ragged_rows,
    identity_block_table,
    pack_flat512_cache,
    pack_fp8_ds_mla_cache,
    ragged_from_rows,
)

PACKED_HEAD_DIM = 512


def _time_kernel(fn, warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), min(times), max(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sq", type=int, default=2048, help="num query tokens (one CTA/token)")
    ap.add_argument("--heads", type=int, default=128, help="query heads (pro=128, flash=8)")
    ap.add_argument("--topk-main", type=int, default=512, help="SWA (region0) CSR len/query")
    ap.add_argument("--topk-extra", type=int, default=128, help="compressed (region1) CSR len/query")
    ap.add_argument("--main-tokens", type=int, default=65536)
    ap.add_argument("--extra-tokens", type=int, default=65536)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from aiter.ops.flydsl import (
        flydsl_sparse_mla_prefill_2region,
        flydsl_sparse_mla_prefill_flat512,
        flydsl_sparse_mla_prefill_flat512_2region,
    )

    dev = "cuda"
    T, H, bs = args.sq, args.heads, args.block_size
    print(
        f"flat512 bench  gfx942  sq={T}  H={H}  block_size={bs}  "
        f"topk_main={args.topk_main}  topk_extra={args.topk_extra}  "
        f"main_tokens={args.main_tokens}  extra_tokens={args.extra_tokens}  "
        f"warmup={args.warmup} iters={args.iters}  metric=gpu_kernel_ms (device self-time)"
    )

    q = gen_q_flat512(T, H, seed=args.seed + 4)
    out = torch.empty(T, H, PACKED_HEAD_DIM, dtype=torch.bfloat16, device=dev)

    # ---- flat512 caches (OCP fp8, 512B/token) ----
    main_kv = gen_kv_flat512(args.main_tokens, seed=args.seed)
    extra_kv = gen_kv_flat512(args.extra_tokens, seed=args.seed + 1)
    main_cache_f = pack_flat512_cache(main_kv, bs)
    extra_cache_f = pack_flat512_cache(extra_kv, bs)
    main_rows = gen_ragged_rows(T, args.topk_main, args.main_tokens, seed=args.seed + 2)
    extra_rows = gen_ragged_rows(T, args.topk_extra, args.extra_tokens, seed=args.seed + 3)
    m_idx, m_iptr = ragged_from_rows(main_rows, torch.device(dev))
    e_idx, e_iptr = ragged_from_rows(extra_rows, torch.device(dev))
    m_bt = identity_block_table(args.main_tokens, bs, torch.device(dev))
    e_bt = identity_block_table(args.extra_tokens, bs, torch.device(dev))

    def run_flat_single():
        flydsl_sparse_mla_prefill_flat512(
            q, main_cache_f, m_idx, m_iptr, out, block_table=m_bt, block_size=bs
        )

    def run_flat_2region():
        flydsl_sparse_mla_prefill_flat512_2region(
            q, out, main_cache_f, m_idx, m_iptr, m_bt,
            extra_cache_f, e_idx, e_iptr, e_bt, block_size=bs,
        )

    # ---- packed fp8_ds_mla caches (584B/token) at the SAME workload ----
    main_kv_p = gen_kv(args.main_tokens, seed=args.seed)          # [N, 512] (448 nope + 64 rope)
    extra_kv_p = gen_kv(args.extra_tokens, seed=args.seed + 1)
    main_cache_p = pack_fp8_ds_mla_cache(main_kv_p, bs, is_extra=False)  # fnuz
    extra_cache_p = pack_fp8_ds_mla_cache(extra_kv_p, bs, is_extra=True)  # OCP

    def run_packed_2region():
        flydsl_sparse_mla_prefill_2region(
            q, out, main_cache_p, m_idx, m_iptr, m_bt,
            extra_cache_p, e_idx, e_iptr, e_bt,
            block_size=bs, main_is_fnuz=True, extra_is_fnuz=False,
        )

    for name, fn in [
        ("flat512_single (region0 only)", run_flat_single),
        ("flat512_2region", run_flat_2region),
        ("packed_2region (fp8_ds_mla)", run_packed_2region),
    ]:
        med, lo, hi = _time_kernel(fn, args.warmup, args.iters)
        print(f"  {name:32s}  median={med:.4f} ms   [min={lo:.4f}, max={hi:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
