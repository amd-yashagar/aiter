# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 packed-varlen FlyDSL FMHA vs ``aiter.flash_attn_varlen_func`` (ASM).

Does not replace production dispatch (opt in with ``AITER_FMHA_FLYDSL_GFX942=1``).
Calls the kernel directly via ``flash_attn_varlen_gfx942``.

MLA ticket shapes (SILOTIGER-957, chunked prefill Sq=4096, H=12) are the
``--bench`` cases. Smaller cases also cover D=128 self-attn, GQA, and f16.

Usage:
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/test_flydsl_fmha_gfx942.py
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/test_flydsl_fmha_gfx942.py --bench
"""

from __future__ import annotations

import argparse
import math
import sys

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import (
    flash_attn_varlen_gfx942,
    flash_attn_varlen_gfx942_pingpong,
)
from aiter.ops.mha import flash_attn_varlen_func
from aiter.test_common import checkAllclose, run_perftest

SUPPORTED_GFX = ("gfx942",)


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
):
    if hk is None:
        hk = hq
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    q = torch.randn(sq, hq, d_qk, dtype=dtype, device="cuda", generator=g)
    k = torch.randn(sk, hk, d_qk, dtype=dtype, device="cuda", generator=g)
    v = torch.randn(sk, hk, d_v, dtype=dtype, device="cuda", generator=g)
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, sk], dtype=torch.int32, device="cuda")
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


def _run_flydsl(q, k, v, cu_q, cu_k, sq, sk, scale, causal, out=None):
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
    )


def _assert_close(out_asm, out_fd, tag: str):
    cos = _cos_diff(out_asm, out_fd)
    err_ratio = checkAllclose(
        out_asm.float(),
        out_fd.float(),
        rtol=2e-2,
        atol=2e-2,
        printLog=False,
        msg=f"[{tag}] flydsl vs asm ",
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
    out_fd = _run_flydsl(q, k, v, cu_q, cu_k, sq, sk, scale, causal)
    _assert_close(out_asm, out_fd, f"Sq={sq} Sk={sk} H={hq}/{hk} D={d_qk}x{d_v} causal={causal}")


def test_fmha_gfx942_varlen_f16():
    q, k, v, cu_q, cu_k = _make_packed(
        64, 128, 4, 128, 128, dtype=torch.float16
    )
    scale = 1.0 / math.sqrt(128)
    out_asm = _run_asm(q, k, v, cu_q, cu_k, 64, 128, scale, False)
    out_fd = _run_flydsl(q, k, v, cu_q, cu_k, 64, 128, scale, False)
    _assert_close(out_asm, out_fd, "f16 D=128")


def test_fmha_gfx942_block_n_forwarded():
    q, k, v, cu_q, cu_k = _make_packed(64, 256, 2, 192, 128)
    scale = 1.0 / math.sqrt(192)
    out_default = flash_attn_varlen_gfx942(
        q, k, v, cu_q, cu_k, 64, 256, softmax_scale=scale, causal=False
    )
    out_bn = flash_attn_varlen_gfx942(
        q, k, v, cu_q, cu_k, 64, 256, softmax_scale=scale, causal=False, block_n=64
    )
    _assert_close(out_default, out_bn, "block_n forwarded")


@pytest.mark.parametrize("coupled_softmax", [False, True])
def test_fmha_gfx942_pingpong_boundary(coupled_softmax):
    sq, sk, h = 257, 511, 12
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, 192, 128)
    scale = 1.0 / math.sqrt(192)
    out_asm = _run_asm(q, k, v, cu_q, cu_k, sq, sk, scale, False)
    out_fd = flash_attn_varlen_gfx942_pingpong(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sq,
        sk,
        softmax_scale=scale,
        causal=False,
        coupled_softmax=coupled_softmax,
    )
    _assert_close(
        out_asm,
        out_fd,
        f"pingpong coupled={coupled_softmax} Sq=257 Sk=511",
    )


def run_case(sq: int, sk: int, h: int, causal: bool, *, bench: bool, seed: int = 0):
    d_qk, d_v = 192, 128
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, d_qk, d_v, seed=seed)
    scale = 1.0 / math.sqrt(d_qk)
    tag = f"Sq={sq} Sk={sk} H={h} causal={causal}"

    out_asm = _run_asm(q, k, v, cu_q, cu_k, sq, sk, scale, causal)
    torch.cuda.synchronize()
    out_fd = _run_flydsl(q, k, v, cu_q, cu_k, sq, sk, scale, causal)
    torch.cuda.synchronize()

    cos = _cos_diff(out_asm, out_fd)
    err = checkAllclose(
        out_asm.float(),
        out_fd.float(),
        rtol=2e-2,
        atol=2e-2,
        printLog=False,
        msg=f"[{tag}] flydsl vs asm ",
    )
    ret = {
        "Sq": sq,
        "Sk": sk,
        "H": h,
        "causal": causal,
        "cos": cos,
        "pass": bool(cos < 1e-4 and err < 0.05),
        "asm_us": None,
        "flydsl_us": None,
        "asm_tflops": None,
        "flydsl_tflops": None,
        "speedup": None,
    }
    print(f"  {tag}  cos_diff={cos:.3e}  pass={cos < 1e-4}")
    if not bench:
        return ret

    flop = _fwd_flops(sq, sk, h, d_qk, d_v, causal)
    _, us_asm = run_perftest(
        lambda: _run_asm(q, k, v, cu_q, cu_k, sq, sk, scale, causal)
    )
    _, us_fd = run_perftest(
        lambda: _run_flydsl(q, k, v, cu_q, cu_k, sq, sk, scale, causal)
    )
    ret["asm_us"] = float(us_asm)
    ret["flydsl_us"] = float(us_fd)
    ret["asm_tflops"] = _tflops(flop, us_asm)
    ret["flydsl_tflops"] = _tflops(flop, us_fd)
    ret["speedup"] = us_asm / us_fd if us_fd else float("inf")
    print(
        f"    asm={us_asm:.1f} us ({ret['asm_tflops']:.1f} TFLOPS)  "
        f"flydsl={us_fd:.1f} us ({ret['flydsl_tflops']:.1f} TFLOPS)  "
        f"speedup={ret['speedup']:.3f}x"
    )
    return ret


def main():
    parser = argparse.ArgumentParser(
        description="FlyDSL gfx942 varlen FMHA vs ASM flash_attn_varlen_func"
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="CUDA-event A/B on ticket shapes after correctness",
    )
    parser.add_argument("--heads", type=int, default=12)
    args = parser.parse_args()

    if not _is_gfx942():
        print(f"Skipping: need gfx942, got {get_gfx() if torch.cuda.is_available() else 'no-cuda'}")
        return 0

    cases = [
        (64, 256, args.heads, False),
        (128, 512, args.heads, False),
        (128, 256, args.heads, True),
        (256, 256, args.heads, True),
    ]
    if args.bench:
        cases.extend(
            [
                (4096, 8192, args.heads, False),
                (4096, 16384, args.heads, False),
                (4096, 131072, args.heads, False),
                (4096, 4096, args.heads, True),
            ]
        )

    rows = []
    failed = False
    for sq, sk, h, causal in cases:
        try:
            rows.append(run_case(sq, sk, h, causal, bench=args.bench))
        except Exception as exc:  # noqa: BLE001
            failed = True
            print(f"  FAIL Sq={sq} Sk={sk} causal={causal}: {exc}")
            rows.append({"Sq": sq, "Sk": sk, "H": h, "causal": causal, "pass": False})

    if failed or not all(r.get("pass") for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
