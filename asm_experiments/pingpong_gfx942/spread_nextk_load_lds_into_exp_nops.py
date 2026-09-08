#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gfx942 ping-pong ISA treatment: spread next-K ``buffer_load … lds``.

Hoist next-K address setup into v248/v249 (VGPR 252, still 2 waves/SIMD),
replace 12 of the 32 TRANS dest-hazard ``s_nop 0`` slots with
``s_mov m0`` + voffset VALU + ``buffer_load_dword … lds``, and delete the
clustered post-softmax next-K DMA block. Close ``s_waitcnt vmcnt(12)`` stays.

Hypothesis: hide the ~533 cy VALU-close ``vmcnt(0)`` inside softmax while
the partner does QK 0–2. Nops still cover the ``v_exp`` dest hazard. No
packed VOP3P. Spreading 12 LDS writes is a different schedule than the
rejected VALU-head flood.

Measured 2026-09-08 on gfx942 Sq=4096 Sk=42700, 4 interleaved CUDA-event
rounds of 21 iters: correctness PASS (cos vs ASM 1.12e-5) but +3.56% vs
the rebuilt listing (4452 vs 4298 us). Same direction as clustered
VALU-head ``load_lds``: this group's K LDS writes fight the partner's QK
``ds_read`` on the same CU even when the 12 issues are spread across the
exp-nop slots. Do not keep this as kernel source.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

M0_SREGS = [25, 22, 23, 24, 26, 27, 28, 29, 30, 31, 33, 34]

# Evenly spaced among the 32 score-exp nop slots.
ISSUE_SLOTS = [i * 32 // 12 for i in range(12)]

PRELUDE = """\
	s_add_u32 s12, s12, 64
	s_addc_u32 s13, s13, 0
	v_add_u32_e32 v248, s12, v160
	v_mul_lo_u32 v248, v248, s15
	v_add_u32_e32 v248, s14, v248
	v_or_b32_e32 v249, v248, v143
"""

# v248 = K byte base; v249 = plane-0 (base | v143); v250 = current voffset.
OFFSET_INSTS = [
    "	v_mov_b32_e32 v250, v249\n",
    "	v_add_u32_e32 v250, 0x12000, v249\n",
    "	v_add_u32_e32 v250, 0x24000, v249\n",
    "	v_add_u32_e32 v250, 0x36000, v249\n",
    "	v_or_b32_e32 v250, 64, v249\n",
    "	v_add_u32_e32 v250, 0x12040, v249\n",
    "	v_add_u32_e32 v250, 0x24040, v249\n",
    "	v_add_u32_e32 v250, 0x36040, v249\n",
    "	v_add_u32_e32 v248, v248, v177\n	v_mov_b32_e32 v250, v248\n",
    "	v_add_u32_e32 v250, 0x12000, v248\n",
    "	v_add_u32_e32 v250, 0x24000, v248\n",
    "	v_add_u32_e32 v250, 0x36000, v248\n",
]


def _issue(index: int) -> str:
    return (
        f"	s_mov_b32 m0, s{M0_SREGS[index]}\n"
        f"{OFFSET_INSTS[index]}"
        "	buffer_load_dword v250, s[8:11], 0 offen lds\n"
    )


def transform(text: str) -> str:
    lines = text.splitlines(keepends=True)

    prelude_at: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == "v_mul_f32_e32 v184, 0xbdd53b95, v65":
            prelude_at = i
            break
    if prelude_at is None:
        raise RuntimeError("missing corr-scale v184 setup")

    nop_slots: list[int] = []
    i = 0
    while i < len(lines):
        if (
            i + 5 < len(lines)
            and "v_exp_f32" in lines[i]
            and lines[i + 1].strip() == ";;#ASMSTART"
            and lines[i + 2].strip() == ";;#ASMEND"
            and lines[i + 3].strip() == "s_nop 0"
            and lines[i + 4].strip() == ";;#ASMSTART"
            and "v_add_f32" in lines[i + 5]
        ):
            nop_slots.append(i + 3)
            i += 6
            continue
        i += 1
    if len(nop_slots) != 32:
        raise RuntimeError(f"expected 32 exp-nop-add slots, found {len(nop_slots)}")

    nextk_start = next(
        (
            i
            for i, line in enumerate(lines)
            if line.strip() == "s_add_u32 s12, s12, 64"
            and i > nop_slots[-1]
        ),
        None,
    )
    if nextk_start is None:
        raise RuntimeError("missing post-softmax next-K s12 increment")
    nextk_end = next(
        (
            i
            for i, line in enumerate(lines)
            if i > nextk_start and line.strip() == "s_waitcnt vmcnt(12)"
        ),
        None,
    )
    if nextk_end is None:
        raise RuntimeError("missing next-K vmcnt(12)")
    # Drop s_add / DMA / s_addc; keep the wait.
    if lines[nextk_end - 1].strip() != "s_addc_u32 s13, s13, 0":
        raise RuntimeError("expected s_addc immediately before vmcnt(12)")

    slot_set = set(ISSUE_SLOTS)
    issue_for_slot = {slot: n for n, slot in enumerate(ISSUE_SLOTS)}
    nop_replace = {nop_slots[slot]: issue_for_slot[slot] for slot in slot_set}

    out: list[str] = []
    for i, line in enumerate(lines):
        if i == prelude_at:
            out.append(line)
            out.append(PRELUDE)
            continue
        if i == nextk_start:
            continue
        if nextk_start < i < nextk_end:
            continue
        if i in nop_replace:
            out.append(_issue(nop_replace[i]))
            continue
        if ".amdhsa_next_free_vgpr 248" in line:
            out.append(line.replace("248", "252"))
            continue
        if ".amdhsa_accum_offset 248" in line:
            out.append(line.replace("248", "252"))
            continue
        if line.strip() == ".set fmha_pingpong_gfx942_kernel_0.num_vgpr, 248":
            out.append(line.replace("248", "252"))
            continue
        if re.match(r"\s+\.vgpr_count:\s+248\s*$", line):
            out.append(re.sub(r"248", "252", line))
            continue
        out.append(line)
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
