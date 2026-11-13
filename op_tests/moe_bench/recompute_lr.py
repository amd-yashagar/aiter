# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Recompute REVIEW rows and the least-rounding bar (lr) from existing bar outputs, no GPU.

A bar whose winner quantizes the activations coarser than the case asks for (per token in a
blockscale case: ``accuracy.candidate_rounds_more``) is a REVIEW row; its lr bar is the
fastest correct candidate that does not. The full bar is not changed.

    python -m op_tests.moe_bench.recompute_lr bar-x.csv candidates-x.csv [...]
"""

import csv
import sys

from . import accuracy


class _S:
    def __init__(self, q_type):
        self.qtype = type("Q", (), {"name": q_type})()


def _first_stage(kernels):
    return (kernels or "").split(" + ")[0].split(" (")[0].strip()


def recompute(bar_rows, cand_rows, mode="warm"):
    """[{case_id, token, bar_id, bar_us, bar_kernels, reason, lr_id, lr_us, lr_kernels,
    lr_phase}] for the cases whose bar winner rounds more than requested."""
    by_case = {}
    for r in cand_rows:
        if r.get("mode") == mode and r.get("status") == "ok" and r.get("median_us"):
            by_case.setdefault(r["case_id"], []).append(r)
    out = []
    for b in bar_rows:
        if b.get("mode") != mode or b.get("status") != "ok" or not b.get("bar_id"):
            continue
        S = _S(b["q_type"])
        reason = accuracy.candidate_rounds_more(S, _first_stage(b.get("bar_kernels")))
        if not reason:
            continue
        best = {}
        for r in by_case.get(b["case_id"], []):
            if accuracy.candidate_rounds_more(S, _first_stage(r.get("kernels"))):
                continue
            us = float(r["median_us"])
            key = r["cand_id"]
            # a confirmed time (full measurement) wins over a screen time of the same id
            if key not in best or (
                r["phase"] == "confirm" and best[key][2] != "confirm"
            ):
                best[key] = (us, r, r["phase"])
        lr = min(best.values(), key=lambda t: t[0]) if best else None
        out.append(
            {
                "case_id": b["case_id"],
                "token": b["token"],
                "bar_id": b["bar_id"],
                "bar_us": float(b["bar_us"]),
                "bar_kernels": b.get("bar_kernels", ""),
                "reason": reason,
                "lr_id": lr[1]["cand_id"] if lr else "",
                "lr_us": lr[0] if lr else None,
                "lr_kernels": (lr[1].get("kernels") or lr[1]["cand_id"]) if lr else "",
                "lr_phase": lr[2] if lr else "",
            }
        )
    return out


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    for bar, cand in zip(args[::2], args[1::2], strict=False):
        with open(bar, newline="") as f:
            bars = list(csv.DictReader(f))
        with open(cand, newline="") as f:
            cands = list(csv.DictReader(f))
        for r in recompute(bars, cands):
            lr = f"{r['lr_us']:.1f}" if r["lr_us"] else "none"
            print(
                f"{r['case_id']} M{r['token']}: bar {r['bar_us']:.1f} us "
                f"({_first_stage(r['bar_kernels'])[:48]}) -> lr {lr} us "
                f"({r['lr_phase']}) {r['lr_id'][-60:]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
