#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rejected gfx942 ping-pong ISA treatment.

Replace each TRANS dest-hazard ``s_nop 0`` between ``v_exp`` and the VOP2
``v_add`` with one independent ``v_pk_mul_f32`` O-rescale, and delete the
post-softmax packed-mul block.

Measured 2026-09-08 on gfx942 Sq=4096 Sk=42700, 21 CUDA-event iters:
correctness PASS (cos vs ASM 2.63e-5) but +4.18% vs the rebuilt listing
(4439 vs 4260 us). Packed VOP3P in those slots fights the partner MFMA.
The nops are load-bearing for both the TRANS dest hazard and ping-pong.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def transform(text: str) -> str:
    lines = text.splitlines(keepends=True)
    pk: list[str] = []
    pk_range: tuple[int, int] | None = None
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("v_pk_mul_f32") and "v[64:65]" in lines[i]:
            start = i
            while (
                i < len(lines)
                and lines[i].lstrip().startswith("v_pk_mul_f32")
                and "v[64:65]" in lines[i]
            ):
                pk.append(lines[i])
                i += 1
            pk_range = (start, i)
            break
        i += 1
    if pk_range is None or len(pk) != 32:
        raise RuntimeError(f"expected 32 O-rescale pk_muls, found {len(pk)}")

    out: list[str] = []
    i = 0
    filled = 0
    while i < len(lines):
        if i == pk_range[0]:
            i = pk_range[1]
            continue
        if (
            i + 5 < len(lines)
            and "v_exp_f32" in lines[i]
            and lines[i + 1].strip() == ";;#ASMSTART"
            and lines[i + 2].strip() == ";;#ASMEND"
            and lines[i + 3].strip() == "s_nop 0"
            and lines[i + 4].strip() == ";;#ASMSTART"
            and "v_add_f32" in lines[i + 5]
            and filled < len(pk)
        ):
            out.extend(lines[i : i + 3])
            out.append(pk[filled])
            filled += 1
            out.extend(lines[i + 4 : i + 6])
            i += 6
            continue
        out.append(lines[i])
        i += 1
    if filled != 32:
        raise RuntimeError(f"filled {filled} nop slots, expected 32")
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    args.dst.write_text(transform(args.src.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
