# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Real LDS of the compiled FlyDSL a16 stage-1 kernels, read from the code objects.

A FlyDSL artifact (``<cache>/launch_gemm1/<key>.pkl``) carries the lowered module; its
``gpu.binary`` attributes hold ``group_segment_fixed_size``, the kernel's total static LDS,
and the name of the kernel carries the shape (``h<K>``) and tile (``bm<M>_tn<N>[_kw<k>]``).
The tile_k is not in the name, so it is recovered from the size.

    python -m op_tests.moe_bench.lds_real <cache dir> [<cache dir> ...]
"""

import glob
import os
import pickle
import re
import sys

NAME = re.compile(
    r'gemm1_a16w4_port_a16w4_bf16_h(\d+)_i(\d+)_ne(\d+)_bm(\d+)_tn(\d+)([^"]*)"'
)


def model_bytes(bm, tn, tk, kw, h):
    """What gemm1.py allocates: the per-k-wave (double-buffered when K spans more than
    one tile) bf16 A tile, with the slice-K reduce scratch overlaid for kw > 1."""
    stages = 2 if (h // kw) // tk > 1 else 1
    a_lds = kw * stages * bm * tk * 2
    if kw == 1:
        return a_lds
    num_acc_n = (tn // (4 // kw)) // 16
    return max(a_lds, 4 * (num_acc_n * (bm // 16)) * 64 * 4 * 4)


def scan(cache_dirs):
    """[(chip, bm, tn, kw, h, group_segment_fixed_size)] over every gemm1 artifact."""
    out = []
    for root in cache_dirs:
        for f in glob.glob(os.path.join(root, "launch_gemm1*", "*.pkl")):
            try:
                with open(f, "rb") as fh:
                    ir = pickle.load(fh).ir
            except Exception:  # noqa: BLE001, S112
                continue
            m = NAME.search(ir)
            g = re.search(r"group_segment_fixed_size = (\d+)", ir)
            chip = re.search(r'chip = "(\w+)"', ir)
            if not (m and g and chip):
                continue
            h, _i, _ne, bm, tn, rest = m.groups()
            kw = re.search(r"_kw(\d)", rest)
            out.append(
                (
                    chip.group(1),
                    int(bm),
                    int(tn),
                    int(kw.group(1)) if kw else 1,
                    int(h),
                    int(g.group(1)),
                )
            )
    return out


def mismatches(rows):
    """Artifacts whose real size is no tile_k's modelled size."""
    bad = []
    for chip, bm, tn, kw, h, size in rows:
        if size not in {model_bytes(bm, tn, tk, kw, h) for tk in (64, 128, 256, 512)}:
            bad.append((chip, bm, tn, kw, h, size))
    return bad


def main(argv=None):
    rows = scan(argv if argv is not None else sys.argv[1:])
    bad = mismatches(rows)
    print(f"{len(rows)} code objects, {len(bad)} whose size the model does not give")
    for r in bad[:20]:
        print("  ", r)
    over = {}
    for chip, bm, tn, kw, h, size in rows:
        over.setdefault(chip, set())
        if size > 65536:
            over[chip].add((bm, tn, kw, size))
    for chip, s in over.items():
        print(chip, "kernels over 64 KiB:", len(s))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
