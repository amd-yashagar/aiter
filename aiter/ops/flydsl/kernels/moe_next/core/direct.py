# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inline helpers for direct-mode kernels (``routing = "direct"``): no tile
map, the kernel groups its own routes and zero-fills its own output.

``group_routes`` puts up to 64 routes (M * topk <= DIRECT_MAX_ROUTES) in
aiter's sort order, in LDS: by expert, then by route index, with the sort's
rules (-1 and out-of-range ids dropped, experts outside ``expert_mask``
dropped and the rest renumbered to local ids, tokens at or past
``num_local_tokens`` dropped, a repeated expert within a token kept at its
last slot). It is ``route_to_tiles`` without the padding. ``zero_rows`` zeroes
the output rows below the token count, as aiter's sort does, for any element
type (16-byte stores; rows must be a multiple of 16 bytes).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .sort import (
    BLOCK,
    WAVE,
    _block_excl_scan2,
    _check,
    _it,
    _ld,
    _lds,
    _lds_st,
    _load_mask,
    _load_route,
    _zero_fill,
)

DIRECT_MAX_ROUTES = WAVE


def lds_words(num_experts):
    """Int32 LDS words ``group_routes`` needs: order, weight (as bits),
    group_expert, group_start (+1), meta (2), and with EP the mask and local
    tables and the scan scratch."""
    e_cap = -(-num_experts // BLOCK) * BLOCK
    return 4 * WAVE + 3 + 2 * e_cap + 2 * (BLOCK // WAVE) + 1


@flyc.jit
def group_routes(
    lds,
    ids,
    weights,
    mask,
    ltok,
    m_cap,
    num_experts,
    topk,
    has_mask,
    has_local,
):
    """Group the routes; every thread of the workgroup calls it (ballots in
    wave 0, the EP table over all threads). ``lds`` is an int32 LDS pointer of
    ``lds_words(num_experts)`` words. Afterwards (a barrier is included):

        order[r]        = token | slot << 24, r < n_valid, in sort order
        weight[r]       = that route's weight (fp32 bits)
        group_expert[k] = expert (local id under EP), k < n_groups
        group_start[k]  = first r of group k; group_start[n_groups] = n_valid
        meta            = (n_valid, n_groups)

    at word offsets ``offsets()``. Tensors: ``ids`` int32 and ``weights`` fp32
    [M, topk], ``mask`` int32 [E] (when ``has_mask``), ``ltok`` int32 [1]
    (when ``has_local``)."""
    o_order, o_w, o_ge, o_gs, o_meta, o_mask, o_local, o_scan = offsets(num_experts)
    E = num_experts
    e_cap = -(-E // BLOCK) * BLOCK
    key_bits = max(1, (E - 1).bit_length())
    tid = gpu.thread_idx.x
    c0, c1 = fx.Int32(0), fx.Int32(1)
    tokens = m_cap
    if has_local:
        t = _ld(_it(ltok), c0)
        tokens = (t < m_cap).select(t, m_cap)
    n = tokens * fx.Int32(topk)
    mask_lds = lds + fx.Int64(o_mask)
    local = lds + fx.Int64(o_local)
    if has_mask:
        _load_mask(mask_lds, _it(mask), tid, E, e_cap)
        gpu.barrier()
        ept = e_cap // BLOCK
        on = [
            (_lds(mask_lds, tid * fx.Int32(ept) + fx.Int32(k)) != c0).select(c1, c0)
            for k in range_constexpr(ept)
        ]
        acc = c0
        for v in on:
            acc = acc + v
        acc, _, _ = _block_excl_scan2(lds + fx.Int64(o_scan), tid, acc, c0)
        for k in range_constexpr(ept):
            _lds_st(local, acc, tid * fx.Int32(ept) + fx.Int32(k))
            acc = acc + on[k]
        gpu.barrier()
    e, inb, w, _ = _load_route(_it(ids), _it(weights), tid, n, True)
    e, valid, below, _ = _check(
        e, inb, mask_lds, E, has_mask, fx.Int32(-1), tid, topk, key_bits
    )
    # lanes with a smaller expert, from the expert-id bits, MSB first
    vmask = fx.Uint64(rocdl.ballot(T.i64, valid))
    eq = fx.Uint64(0xFFFFFFFFFFFFFFFF)
    less = fx.Uint64(0)
    for b in range_constexpr(key_bits - 1, -1, -1):
        bit = ((e >> fx.Int32(b)) & c1) != c0
        votes = fx.Uint64(rocdl.ballot(T.i64, bit))
        less = bit.select(less | (eq & ~votes), less)
        eq = eq & bit.select(votes, ~votes)
    less = less & vmask
    n_less = fx.Int32(fx.math.ctpop(fx.Uint32(less))) + fx.Int32(
        fx.math.ctpop(fx.Uint32(less >> fx.Uint64(32)))
    )
    rank = n_less + below
    first = valid & (below == c0)
    fmask = fx.Uint64(rocdl.ballot(T.i64, first))
    fl = fmask & less
    grp = fx.Int32(fx.math.ctpop(fx.Uint32(fl))) + fx.Int32(
        fx.math.ctpop(fx.Uint32(fl >> fx.Uint64(32)))
    )
    n_valid = fx.Int32(fx.math.ctpop(fx.Uint32(vmask))) + fx.Int32(
        fx.math.ctpop(fx.Uint32(vmask >> fx.Uint64(32)))
    )
    n_groups = fx.Int32(fx.math.ctpop(fx.Uint32(fmask))) + fx.Int32(
        fx.math.ctpop(fx.Uint32(fmask >> fx.Uint64(32)))
    )
    in_wave0 = tid < fx.Int32(WAVE)
    if valid & in_wave0:
        tok = tid // fx.Int32(topk)
        slot = tid - tok * fx.Int32(topk)
        _lds_st(lds + fx.Int64(o_order), tok | (slot << fx.Int32(24)), rank)
        _lds_st(lds + fx.Int64(o_w), w.bitcast(fx.Int32), rank)
    if first & in_wave0:
        ex = _lds(local, e) if has_mask else e
        _lds_st(lds + fx.Int64(o_ge), ex, grp)
        _lds_st(lds + fx.Int64(o_gs), rank, grp)
    if tid == c0:
        _lds_st(lds + fx.Int64(o_gs), n_valid, n_groups)
        _lds_st(lds + fx.Int64(o_meta), n_valid, c0)
        _lds_st(lds + fx.Int64(o_meta), n_groups, c1)
    gpu.barrier()


def offsets(num_experts):
    """Word offsets (order, weight, group_expert, group_start, meta, mask,
    local, scan) in the ``group_routes`` LDS block."""
    e_cap = -(-num_experts // BLOCK) * BLOCK
    o_order = 0
    o_w = o_order + WAVE
    o_ge = o_w + WAVE
    o_gs = o_ge + WAVE
    o_meta = o_gs + WAVE + 1
    o_mask = o_meta + 2
    o_local = o_mask + e_cap
    o_scan = o_local + e_cap
    return o_order, o_w, o_ge, o_gs, o_meta, o_mask, o_local, o_scan


def zero_rows(out, first, stride, tokens, m_cap, row_v4):
    """Zero output rows below min(tokens, m_cap): ``row_v4`` 16-byte vectors a
    row, grid-stride from vector ``first`` by ``stride`` (int32 tensor view)."""
    rows = (tokens < m_cap).select(tokens, m_cap)
    _zero_fill(_it(out), first, stride, rows * fx.Int32(row_v4))


__all__ = [
    "DIRECT_MAX_ROUTES",
    "group_routes",
    "lds_words",
    "offsets",
    "zero_rows",
]
