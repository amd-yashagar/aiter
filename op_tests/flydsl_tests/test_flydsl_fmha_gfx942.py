# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 packed-varlen FlyDSL FMHA vs ``aiter.flash_attn_varlen_func`` (ASM).

Does not replace production dispatch (opt in with ``AITER_FMHA_FLYDSL_GFX942=1``).

Usage:
    HIP_VISIBLE_DEVICES=0 python -m pytest op_tests/flydsl_tests/test_flydsl_fmha_gfx942.py -q
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/test_flydsl_fmha_gfx942.py --bench
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/test_flydsl_fmha_gfx942.py --once pingpong
"""

from __future__ import annotations

import argparse
import math
import sys

import pytest
import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl import (
    flash_attn_varlen_gfx942,
    flash_attn_varlen_gfx942_pingpong,
)
from aiter.ops.flydsl.kernels.fmha_gfx942.parallel import choose_kv_splits
from aiter.ops.mha import flash_attn_varlen_func
from aiter.test_common import checkAllclose, run_perftest

SUPPORTED_GFX = ("gfx942",)
TICKET_SQ, TICKET_SK, TICKET_H = 4096, 42700, 12
HEAD_DIM_QK, HEAD_DIM_V = 192, 128
ROOF_TFLOPS = 590.0
WARMUP, ITERS = 5, 21


def _is_gfx942() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return get_gfx() in SUPPORTED_GFX
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _is_gfx942(),
    reason="flash_attn_varlen_gfx942 is gfx942 only",
)


def _cos_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x64, y64 = x.double(), y.double()
    return 1 - 2 * (x64 * y64).sum().item() / max(
        (x64 * x64 + y64 * y64).sum().item(), 1e-12
    )


def _fwd_flops(sq: int, sk: int, h: int, d_qk: int, d_v: int, causal: bool) -> int:
    flop = h * (2 * sq * sk * d_qk + 2 * sq * sk * d_v)
    if causal:
        flop //= 2
    return flop


def _tflops(flop: int, us: float) -> float:
    if us <= 0:
        return float("inf")
    return flop / us / 1e6


def _make_packed(
    sq: int,
    sk: int,
    hq: int,
    d_qk: int,
    d_v: int,
    *,
    hk: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    cpu: bool = False,
):
    if hk is None:
        hk = hq
    device = "cpu" if cpu else "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(sq, hq, d_qk, dtype=dtype, device=device, generator=g)
    k = torch.randn(sk, hk, d_qk, dtype=dtype, device=device, generator=g)
    v = torch.randn(sk, hk, d_v, dtype=dtype, device=device, generator=g)
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, sk], dtype=torch.int32, device=device)
    if cpu:
        q, k, v, cu_q, cu_k = q.cuda(), k.cuda(), v.cuda(), cu_q.cuda(), cu_k.cuda()
    return q, k, v, cu_q, cu_k


def _run_asm(q, k, v, cu_q, cu_k, sq, sk, scale, causal, out=None):
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sq,
        sk,
        softmax_scale=scale,
        causal=causal,
        out=out,
    )


def _run_4wave(q, k, v, cu_q, cu_k, sq, sk, scale, causal, out=None, kv_splits=None):
    return flash_attn_varlen_gfx942(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sq,
        sk,
        softmax_scale=scale,
        causal=causal,
        out=out,
        kv_splits=kv_splits,
    )


def _run_pingpong(q, k, v, cu_q, cu_k, sq, sk, scale, out=None, kv_splits=None):
    return flash_attn_varlen_gfx942_pingpong(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sq,
        sk,
        softmax_scale=scale,
        causal=False,
        out=out,
        kv_splits=kv_splits,
    )


def _assert_close(out_ref, out_got, tag: str):
    cos = _cos_diff(out_ref, out_got)
    err_ratio = checkAllclose(
        out_ref.float(),
        out_got.float(),
        rtol=2e-2,
        atol=2e-2,
        printLog=False,
        msg=f"[{tag}] ",
    )
    assert cos < 1e-4, f"{tag} cos_diff={cos:.3e}"
    assert err_ratio < 0.05, f"{tag} checkAllclose ratio={err_ratio}"
    return cos


@pytest.mark.parametrize(
    "sq,sk,hq,hk,d_qk,d_v,causal",
    [
        (64, 256, 4, 4, 192, 128, False),
        (128, 256, 4, 4, 192, 128, True),
        (64, 128, 4, 4, 128, 128, False),
        (64, 128, 8, 2, 192, 128, False),
    ],
)
def test_fmha_gfx942_varlen_bf16(sq, sk, hq, hk, d_qk, d_v, causal):
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, hq, d_qk, d_v, hk=hk)
    scale = 1.0 / math.sqrt(d_qk)
    out_asm = _run_asm(q, k, v, cu_q, cu_k, sq, sk, scale, causal)
    out_fd = _run_4wave(q, k, v, cu_q, cu_k, sq, sk, scale, causal, kv_splits=1)
    _assert_close(
        out_asm, out_fd, f"Sq={sq} Sk={sk} H={hq}/{hk} D={d_qk}x{d_v} causal={causal}"
    )


def test_fmha_gfx942_varlen_f16():
    q, k, v, cu_q, cu_k = _make_packed(64, 128, 4, 128, 128, dtype=torch.float16)
    scale = 1.0 / math.sqrt(128)
    out_asm = _run_asm(q, k, v, cu_q, cu_k, 64, 128, scale, False)
    out_fd = _run_4wave(q, k, v, cu_q, cu_k, 64, 128, scale, False, kv_splits=1)
    _assert_close(out_asm, out_fd, "f16 D=128")


def test_fmha_gfx942_pingpong_boundary():
    sq, sk, h = 257, 511, 12
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, 192, 128)
    scale = 1.0 / math.sqrt(192)
    out_asm = _run_asm(q, k, v, cu_q, cu_k, sq, sk, scale, False)
    out_fd = _run_pingpong(q, k, v, cu_q, cu_k, sq, sk, scale, kv_splits=1)
    _assert_close(out_asm, out_fd, "pingpong Sq=257 Sk=511")


def test_choose_kv_splits_ticket_shape():
    assert (
        choose_kv_splits(
            max_seqlen_q=TICKET_SQ,
            max_seqlen_k=TICKET_SK,
            num_heads=TICKET_H,
            block_m=256,
            block_n=64,
            cu_count=304,
        )
        == 3
    )
    assert (
        choose_kv_splits(
            max_seqlen_q=TICKET_SQ,
            max_seqlen_k=TICKET_SK,
            num_heads=TICKET_H,
            block_m=128,
            block_n=64,
            cu_count=304,
        )
        == 3
    )


def test_fmha_gfx942_split_kv_matches_unsplit():
    q, k, v, cu_q, cu_k = _make_packed(64, 256, 4, 192, 128)
    scale = 1.0 / math.sqrt(192)
    out_s1 = _run_4wave(q, k, v, cu_q, cu_k, 64, 256, scale, False, kv_splits=1)
    out_s3 = _run_4wave(q, k, v, cu_q, cu_k, 64, 256, scale, False, kv_splits=3)
    _assert_close(out_s1, out_s3, "4-wave split-K S=3 vs S=1")


def test_fmha_gfx942_pingpong_split_kv_matches_unsplit():
    sq, sk, h = 128, 256, 12
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, 192, 128)
    scale = 1.0 / math.sqrt(192)
    out_s1 = _run_pingpong(q, k, v, cu_q, cu_k, sq, sk, scale, kv_splits=1)
    out_s3 = _run_pingpong(q, k, v, cu_q, cu_k, sq, sk, scale, kv_splits=3)
    _assert_close(out_s1, out_s3, "pingpong split-K S=3 vs S=1")


def _grid_info(sq: int, sk: int, h: int, impl: str, kv_splits: int | None):
    block_m = 256 if impl == "pingpong" else 128
    nq = (sq + block_m - 1) // block_m
    if kv_splits is None:
        kv_splits = choose_kv_splits(
            max_seqlen_q=sq,
            max_seqlen_k=sk,
            num_heads=h,
            block_m=block_m,
            block_n=64,
            cu_count=int(get_cu_num()),
        )
    grid = nq * h * kv_splits
    cu = int(get_cu_num())
    waves = (grid + cu - 1) // cu
    return kv_splits, grid, cu, waves


def _dispatch(impl, q, k, v, cu_q, cu_k, sq, sk, scale, causal, out=None, kv_splits=None):
    if impl == "asm":
        return _run_asm(q, k, v, cu_q, cu_k, sq, sk, scale, causal, out=out)
    if impl == "4wave":
        return _run_4wave(
            q, k, v, cu_q, cu_k, sq, sk, scale, causal, out=out, kv_splits=kv_splits
        )
    if impl == "pingpong":
        if causal:
            raise ValueError("pingpong is non-causal only")
        return _run_pingpong(
            q, k, v, cu_q, cu_k, sq, sk, scale, out=out, kv_splits=kv_splits
        )
    raise ValueError(f"unknown impl {impl!r}")


def run_once(impl: str, sq: int, sk: int, h: int, *, check: bool, kv_splits: int | None):
    scale = 1.0 / math.sqrt(HEAD_DIM_QK)
    q, k, v, cu_q, cu_k = _make_packed(
        sq, sk, h, HEAD_DIM_QK, HEAD_DIM_V, seed=0, cpu=True
    )
    torch.cuda.synchronize()
    out = _dispatch(impl, q, k, v, cu_q, cu_k, sq, sk, scale, False, kv_splits=kv_splits)
    torch.cuda.synchronize()
    if check and impl != "asm":
        ref = _run_asm(q, k, v, cu_q, cu_k, sq, sk, scale, False)
        _assert_close(ref, out, f"once {impl}")
        print(f"PASS: fmha_gfx942_once impl={impl} cos_diff={_cos_diff(ref, out):.3e}")
    else:
        print(f"dispatched {impl} Sq={sq} Sk={sk} H={h}")
    return 0


def run_bench(impls: list[str], sq: int, sk: int, h: int, kv_splits: int | None):
    scale = 1.0 / math.sqrt(HEAD_DIM_QK)
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, HEAD_DIM_QK, HEAD_DIM_V, seed=0)
    flop = _fwd_flops(sq, sk, h, HEAD_DIM_QK, HEAD_DIM_V, False)
    roof_us = flop / (ROOF_TFLOPS * 1e12) * 1e6
    print(
        f"gfx={get_gfx()} cu={get_cu_num()} Sq={sq} Sk={sk} H={h} "
        f"work={flop/1e12:.3f} TFLOP roof={ROOF_TFLOPS:.0f} TFLOPS ({roof_us:.0f} us)"
    )
    ref = _run_asm(q, k, v, cu_q, cu_k, sq, sk, scale, False)
    torch.cuda.synchronize()
    for impl in impls:
        out = _dispatch(
            impl, q, k, v, cu_q, cu_k, sq, sk, scale, False, kv_splits=kv_splits
        )
        torch.cuda.synchronize()
        if impl != "asm":
            _assert_close(ref, out, f"bench {impl}")
        splits, grid, cu, waves = (1, 0, int(get_cu_num()), 0)
        if impl != "asm":
            splits, grid, cu, waves = _grid_info(sq, sk, h, impl, kv_splits)
        _, us = run_perftest(
            lambda impl=impl: _dispatch(
                impl, q, k, v, cu_q, cu_k, sq, sk, scale, False, kv_splits=kv_splits
            ),
            num_iters=ITERS,
            num_warmup=WARMUP,
            use_cuda_event=True,
        )
        us = float(us)
        tflops = _tflops(flop, us)
        print(
            f"FMHA_IMPL={impl} FMHA_US={us:.3f} FMHA_TFLOPS={tflops:.1f} "
            f"FMHA_ROOF_PCT={100.0 * tflops / ROOF_TFLOPS:.1f} "
            f"KV_SPLITS={splits} GRID={grid} CU={cu} DISPATCH_WAVES={waves}"
        )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="gfx942 FlyDSL FMHA test / ticket bench / ATT once-launch"
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="CUDA-event A/B on the ticket shape after correctness",
    )
    parser.add_argument(
        "--once",
        choices=("4wave", "pingpong", "asm"),
        help="one dispatch for WaveScope ATT (CPU-generated inputs)",
    )
    parser.add_argument(
        "--impl",
        default="all",
        choices=("all", "4wave", "pingpong", "asm"),
    )
    parser.add_argument("--sq", type=int, default=TICKET_SQ)
    parser.add_argument("--sk", type=int, default=TICKET_SK)
    parser.add_argument("--heads", type=int, default=TICKET_H)
    parser.add_argument("--kv-splits", type=int, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not _is_gfx942():
        print(
            f"Skipping: need gfx942, got "
            f"{get_gfx() if torch.cuda.is_available() else 'no-cuda'}"
        )
        return 0

    if args.once:
        return run_once(
            args.once,
            args.sq,
            args.sk,
            args.heads,
            check=args.check,
            kv_splits=args.kv_splits,
        )
    if args.bench:
        impls = ["asm", "4wave", "pingpong"] if args.impl == "all" else [args.impl]
        return run_bench(impls, args.sq, args.sk, args.heads, args.kv_splits)

    parser.error("pass --bench or --once; pytest drives correctness tests")
    return 2


if __name__ == "__main__":
    sys.exit(main())
