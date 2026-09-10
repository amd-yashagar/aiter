# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hardcoded-AGPR QK matching ``fwd_hd192x128`` (not LLVM ``=a``).

The ``=a`` constraint coalesced every pack onto ``a[0:1]`` and inserted
``v_accvgpr_write`` between MFMAs. Naming the source pairs in both the
instruction operands and ``{a[N:N+1]}`` constraints makes LLVM:

- ``buffer_load_dwordx2 a[0:47]`` for Q (parked for the whole kernel)
- ``ds_read_* a[48:63]`` for each K plane
- ``v_mfma ... a[K], a[Q], v[S]`` with S remaining VGPR for softmax

One inline-asm blob per N-half (four MFMAs) so the waitcnt pass cannot insert
``s_waitcnt`` between them. ``has_side_effects=False``: these MFMAs do not
touch memory; a True flag made LLVM drain ``vmcnt`` in the QK hot loop.
"""

from __future__ import annotations

from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import Int32
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

# Q: 24 packs x 2 AGPR = a[0:47]. K plane: 8 packs x 2 = a[48:63].
Q_AGPR0 = 0
K_AGPR0 = 48
N_Q_AGPR = 48
N_K_AGPR = 16


def _pack_v2i32(pack):
    return _raw(Vec(pack).bitcast(Int32))


def mfma_qk_chain(c, k_packs, q_packs, k_agpr0: int, q_agpr0: int):
    """Four waitcnt-free ``C += K_i * Q_i`` MFMAs in one asm blob.

    ``k_packs`` / ``q_packs`` are length-4 MFMA packs already allocated to
    ``a[k_agpr0 : k_agpr0+8]`` and ``a[q_agpr0 : q_agpr0+8]``.
    """
    c_raw = _raw(c)
    k_ops = [_pack_v2i32(p) for p in k_packs]
    q_ops = [_pack_v2i32(p) for p in q_packs]
    constraints = ["=v", "0"]
    for i in range(4):
        a = k_agpr0 + 2 * i
        constraints.append(f"{{a[{a}:{a + 1}]}}")
    for i in range(4):
        a = q_agpr0 + 2 * i
        constraints.append(f"{{a[{a}:{a + 1}]}}")
    # $0 dest / srcC, $1 tied, $2..$5 K, $6..$9 Q.
    lines = [
        f"v_mfma_f32_32x32x8_bf16 $0, ${2 + i}, ${6 + i}, $0" for i in range(4)
    ]
    return llvm.InlineAsmOp(
        c_raw.type,
        [c_raw, *k_ops, *q_ops],
        "\n".join(lines),
        ",".join(constraints),
        has_side_effects=False,
    ).res


def mfma_vgpr_chain(c, a_packs, b_packs):
    """Four waitcnt-free VGPR MFMAs in one asm blob (PV GEMM)."""
    c_raw = _raw(c)
    a_ops = [_pack_v2i32(p) for p in a_packs]
    b_ops = [_pack_v2i32(p) for p in b_packs]
    lines = [
        f"v_mfma_f32_32x32x8_bf16 $0, ${2 + i}, ${6 + i}, $0" for i in range(4)
    ]
    return llvm.InlineAsmOp(
        c_raw.type,
        [c_raw, *a_ops, *b_ops],
        "\n".join(lines),
        "=v,0,v,v,v,v,v,v,v,v",
        has_side_effects=False,
    ).res
