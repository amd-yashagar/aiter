# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile and dispatch the gfx942 FMHA ping-pong kernel exactly once.

The process disables FlyDSL's disk cache and emits line-table debug info so the
resulting dispatch is suitable for a WaveScope ATT profile.
"""

from __future__ import annotations

import argparse
import math
import os

# FlyDSL reads these options during import.
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
os.environ.setdefault("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "1")
os.environ.setdefault("FLYDSL_DEBUG_DUMP_ASM", "1")
os.environ.setdefault("FLYDSL_DUMP_IR", "1")

import torch

from aiter.ops.flydsl import flash_attn_varlen_gfx942_pingpong


def _cpu_bf16_randn(shape, generator):
    return torch.randn(shape, dtype=torch.bfloat16, generator=generator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exact",
        action="store_true",
        help="capture the exact ping-pong control instead of the coupled treatment",
    )
    parser.add_argument("--sq", type=int, default=4096)
    parser.add_argument("--sk", type=int, default=42700)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    h, dq, dv = 12, 192, 128
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    # CPU generation plus synchronous copies avoid setup compute dispatches.
    q = _cpu_bf16_randn((args.sq, h, dq), generator).cuda()
    k = _cpu_bf16_randn((args.sk, h, dq), generator).cuda()
    v = _cpu_bf16_randn((args.sk, h, dv), generator).cuda()
    cu_q = torch.tensor([0, args.sq], dtype=torch.int32).cuda()
    cu_k = torch.tensor([0, args.sk], dtype=torch.int32).cuda()
    out = torch.empty((args.sq, h, dv), dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()

    flash_attn_varlen_gfx942_pingpong(
        q,
        k,
        v,
        cu_q,
        cu_k,
        args.sq,
        args.sk,
        softmax_scale=1.0 / math.sqrt(dq),
        causal=False,
        out=out,
        coupled_softmax=not args.exact,
    )
    torch.cuda.synchronize()
    print(
        "dispatched fmha_pingpong_gfx942_kernel once; "
        f"coupled_softmax={not args.exact}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
