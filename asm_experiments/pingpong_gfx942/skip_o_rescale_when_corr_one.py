#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gfx942 ping-pong ISA treatment: skip O-rescale when corr == 1.

Wrap the 32 ``v_pk_mul_f32`` O-rescales in ``s_and_saveexec`` so a tile
whose running max did not move (``exp2(0) == 1``) skips those VOP3P ops
on the VALU wave. Exact when ``m_new == m_running``; mixed lanes stay
predicated.

Hypothesis: after early KV tiles the max is stable, so most of 64 tiles
drop 32 packed muls from the VALU-critical overlap window and shrink
the 5.3× VALU:MFMA imbalance.

Measured 2026-09-08 on gfx942 Sq=4096 Sk=42700, 4 interleaved CUDA-event
rounds of 21 iters: correctness PASS (cos vs ASM 3.99e-5) but only
−0.49% vs the rebuilt listing (4276 vs 4297 us). Random ``randn`` scores
keep raising the row max, so ``corr == 1`` rarely holds wave-wide. Not a
2% win; do not keep as kernel source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PRE = """\
	v_cmp_neq_f32_e32 vcc, 1.0, v64
	s_and_saveexec_b64 s[36:37], vcc
"""
POST = """\
	s_or_b64 exec, exec, s[36:37]
"""


def transform(text: str) -> str:
    lines = text.splitlines(keepends=True)
    start = None
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("v_pk_mul_f32") and "v[64:65]" in lines[i]:
            start = i
            while (
                i < len(lines)
                and lines[i].lstrip().startswith("v_pk_mul_f32")
                and "v[64:65]" in lines[i]
            ):
                i += 1
            end = i
            if end - start != 32:
                raise RuntimeError(f"expected 32 O-rescale pk_muls, found {end - start}")
            return "".join(lines[:start] + [PRE] + lines[start:end] + [POST] + lines[end:])
        i += 1
    raise RuntimeError("O-rescale pk_mul block not found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    args.dst.write_text(transform(args.src.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
