#!/usr/bin/env python3
"""Move independent non-causal mask setup into the three V-load wait windows."""

from __future__ import annotations

import argparse
from pathlib import Path


def _index(lines: list[str], instruction: str, start: int = 0) -> int:
    matches = [i for i in range(start, len(lines)) if lines[i] == instruction]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {instruction!r} at/after line {start + 1}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _next(lines: list[str], instruction: str, start: int) -> int:
    try:
        return lines.index(instruction, start)
    except ValueError as exc:
        raise RuntimeError(
            f"missing {instruction!r} at/after line {start + 1}"
        ) from exc


def transform(source: str, mode: str) -> str:
    lines = source.splitlines()

    addr_begin = _index(lines, "\tv_cmp_lt_i32_e32 vcc, v163, v103")
    addr_end = _index(lines, "\tv_add_u32_e32 v163, 55, v163", addr_begin)
    first_compare = lines[addr_begin]
    addr_setup = lines[addr_begin + 1 : addr_end + 1]
    del lines[addr_begin : addr_end + 1]

    # v153 still contains the fourth V-load result until the final v_perm.
    late_v153 = "\tv_add_u32_e32 v153, 32, v163"
    addr_setup.remove(late_v153)
    # The first mask compare must see the unadvanced base, not base + 55.
    late_v163 = "\tv_add_u32_e32 v163, 55, v163"
    addr_setup.remove(late_v163)
    delayed_mfma_sources = []
    if mode == "hazard-safe":
        for register in range(213, 220):
            prefix = f"\tv_add_u32_e32 v{register}, "
            instruction = next(line for line in addr_setup if line.startswith(prefix))
            addr_setup.remove(instruction)
            delayed_mfma_sources.append(instruction)

    mask_compares = []
    if mode == "full":
        cmp_begin = _index(lines, "\tv_cmp_lt_i32_e64 s[2:3], v213, v103")
        cmp_end = _index(
            lines, "\tv_cmp_lt_i32_e64 s[62:63], v163, v103", cmp_begin
        )
        mask_compares = lines[cmp_begin : cmp_end + 1]
        if len(mask_compares) != 30:
            raise RuntimeError(
                f"expected 30 mask compares, found {len(mask_compares)}"
            )
        del lines[cmp_begin : cmp_end + 1]

    address_anchor = _index(lines, "\tv_add_u32_e32 v163, s76, v136")
    lines[address_anchor + 1 : address_anchor + 1] = addr_setup

    wait4 = _next(lines, "\ts_waitcnt vmcnt(4)", address_anchor)
    wait2 = _next(lines, "\ts_waitcnt vmcnt(2)", wait4)
    if delayed_mfma_sources:
        lines[wait2:wait2] = delayed_mfma_sources
        wait2 += len(delayed_mfma_sources)
    if mask_compares:
        lines[wait2:wait2] = mask_compares[:15]

    wait0 = _next(lines, "\ts_waitcnt vmcnt(0)", wait2 + len(mask_compares[:15]))
    if mask_compares:
        lines[wait0:wait0] = mask_compares[15:]

    final_perm = _next(lines, "\tv_perm_b32 v151, v151, v153, s80", wait0)
    lines[final_perm + 1 : final_perm + 1] = [
        first_compare,
        late_v153,
        late_v163,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode",
        choices=("hazard-safe", "addresses", "full"),
        default="hazard-safe",
        help="full also extends scalar mask lifetimes and is kept for diagnosis",
    )
    args = parser.parse_args()
    args.output.write_text(transform(args.source.read_text(), args.mode))


if __name__ == "__main__":
    main()
