# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FMHA hardware exceptions (waits, barriers, VALU dest padding).

The ASM-parity kernel stays on named phases. This module owns the inline-asm
blobs LLVM would otherwise split or drain. ``fmha_asm_kernel.py`` must not
import ``llvm`` except through this file.
"""

from __future__ import annotations

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw


def _llvm_value(value):
    """Unwrap FlyDSL scalar/vector wrappers for LLVM pointer / asm ops."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def lds_ptr_from_i64(addr_i64):
    """``inttoptr`` to LDS ``!llvm.ptr<3>`` for ``buffer_load … lds``.

    The pointer type must be parsed in the *current* MLIR context; caching it
    across FlyDSL compiles reuses a type from a destroyed context.
    """
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), addr_i64)


def wait_vmcnt(count=0):
    """Opaque ``s_waitcnt vmcnt(N)``; LLVM cannot merge or drop it."""
    llvm.inline_asm(
        None,
        [],
        f"s_waitcnt vmcnt({count})",
        "~{memory}",
        has_side_effects=True,
    )


def wait_lgkmcnt(count=0):
    """Opaque ``s_waitcnt lgkmcnt(N)``."""
    llvm.inline_asm(
        None,
        [],
        f"s_waitcnt lgkmcnt({count})",
        "~{memory}",
        has_side_effects=True,
    )


def waitcnt_barrier(*, vmcnt=None, lgkmcnt=None):
    """One ``s_waitcnt`` + ``s_barrier`` blob so LLVM cannot insert a full drain.

    Matches production ASM ``fmha_fwd_hd192x128_bf16_1tg_4w_128x32_300``:
    publish LDS with lgkmcnt, leave newer VMEM in flight. Omit a counter
    (None) to not wait on it.
    """
    if vmcnt is not None and lgkmcnt is not None:
        wait = f"s_waitcnt vmcnt({vmcnt}) & lgkmcnt({lgkmcnt})"
    elif vmcnt is not None:
        wait = f"s_waitcnt vmcnt({vmcnt})"
    elif lgkmcnt is not None:
        wait = f"s_waitcnt lgkmcnt({lgkmcnt})"
    else:
        wait = None
    body = f"{wait}\ns_barrier" if wait is not None else "s_barrier"
    llvm.inline_asm(
        None,
        [],
        body,
        "~{memory}",
        has_side_effects=True,
    )


def lds_handoff_barrier():
    """Drain LDS traffic and publish it across waves without draining VMEM."""
    waitcnt_barrier(lgkmcnt=0)


def s_nop(count=1):
    """VALU dest-hazard padding (ASM ``s_nop`` after ``v_exp``)."""
    llvm.inline_asm(
        None,
        [],
        f"s_nop {int(count)}",
        "",
        has_side_effects=True,
    )


def pin_f32(value):
    """Keep an f32 on a VGPR across a scheduling boundary (empty ``=v,0``)."""
    return llvm.InlineAsmOp(
        T.f32,
        [_raw(value)],
        "",
        "=v,0",
        has_side_effects=False,
    ).res


def exp2_f32_with_dest_nops(value, nops=1):
    """``v_exp_f32`` plus ``s_nop`` dest padding in one blob.

    4-wave AGPR QK NaN'd when ``v_exp`` results were consumed without this
    hazard pad. Ping-pong survived on ``pin_f32`` alone; do not drop the nop
    on the 4-wave path.
    """
    nop_lines = "\n".join(f"s_nop {int(nops)}" for _ in range(1))
    return llvm.InlineAsmOp(
        T.f32,
        [_raw(value)],
        f"v_exp_f32 $0, $1\n{nop_lines}",
        "=v,v",
        has_side_effects=False,
    ).res


def ds_write2_b32(addr, value0, value1, offset0, offset1):
    """Paired LDS store matching the native V transpose publish."""
    llvm.inline_asm(
        None,
        [_llvm_value(addr), _llvm_value(value0), _llvm_value(value1)],
        (
            f"ds_write2_b32 $0, $1, $2 "
            f"offset0:{offset0} offset1:{offset1}"
        ),
        "v,v,v",
        has_side_effects=True,
    )
