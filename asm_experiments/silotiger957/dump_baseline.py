#!/usr/bin/env python3
"""Compile the exact non-causal/LSE FlyDSL variant used by the HSACO bench."""

import math

import torch

from aiter.ops.flydsl import flash_attn_varlen_gfx942


def main() -> None:
    sq, sk, heads = 128, 256, 12
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(
        sq, heads, 192, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    k = torch.randn(
        sk, heads, 192, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    v = torch.randn(
        sk, heads, 128, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, sk], dtype=torch.int32, device="cuda")
    flash_attn_varlen_gfx942(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sq,
        sk,
        softmax_scale=1 / math.sqrt(192),
        causal=False,
        return_lse=True,
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
