# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inline helpers for direct-mode kernels (``routing = "direct"``): no tile
map, the kernel groups its own routes and zero-fills its own output.

``group_routes`` puts up to 256 routes (M * topk <= DIRECT_MAX_ROUTES, one
per thread of a 256-thread workgroup) in aiter's sort order, in LDS: by expert, then by route index, with the sort's
rules (-1 and out-of-range ids dropped, experts outside ``expert_mask``
dropped and the rest renumbered to local ids, tokens at or past
``num_local_tokens`` dropped, a repeated expert within a token kept at its
last slot). It is ``route_to_tiles`` without the padding. ``zero_rows`` zeroes
the output rows below the token count, as aiter's sort does, for any element
type (16-byte stores; rows must be a multiple of 16 bytes).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

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

DIRECT_MAX_ROUTES = BLOCK
WAVES = BLOCK // WAVE


def offsets(num_experts):
    """Word offsets (order, weight, group_expert, group_start, meta, mask,
    count, start, scan) in the ``group_routes`` LDS block."""
    e_cap = -(-num_experts // BLOCK) * BLOCK
    o_order = 0
    o_w = o_order + BLOCK
    o_ge = o_w + BLOCK
    o_gs = o_ge + BLOCK
    o_meta = o_gs + BLOCK + 1
    o_mask = o_meta + 2
    o_cnt = o_mask + e_cap
    o_start = o_cnt + e_cap
    o_scan = o_start + e_cap
    return o_order, o_w, o_ge, o_gs, o_meta, o_mask, o_cnt, o_start, o_scan


def lds_words(num_experts):
    """Int32 LDS words ``group_routes`` needs (about 7 KB at E = 256)."""
    return offsets(num_experts)[-1] + 2 * WAVES + 1


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
    """Group up to 256 routes (one per thread of a 256-thread workgroup);
    every thread calls it. ``lds`` is an int32 LDS pointer of
    ``lds_words(num_experts)`` words. Afterwards (barrier included):

        order[r]        = token | slot << 24, r < n_valid, in sort order
        weight[r]       = that route's weight (fp32 bits)
        group_expert[k] = expert (local id under EP), k < n_groups
        group_start[k]  = first r of group k; group_start[n_groups] = n_valid
        meta            = (n_valid, n_groups)

    at word offsets ``offsets()``. Tensors: ``ids`` int32 and ``weights`` fp32
    [M, topk], ``mask`` int32 [E] (when ``has_mask``), ``ltok`` int32 [1]
    (when ``has_local``). Ranks within an expert follow route order: wave w
    adds its per-expert counts in phase w."""
    o_order, o_w, o_ge, o_gs, o_meta, o_mask, o_cnt, o_start, o_scan = offsets(
        num_experts
    )
    E = num_experts
    e_cap = -(-E // BLOCK) * BLOCK
    ept = e_cap // BLOCK
    key_bits = max(1, (E - 1).bit_length())
    tid = gpu.thread_idx.x
    wave = tid // fx.Int32(WAVE)
    c0, c1 = fx.Int32(0), fx.Int32(1)
    tokens = m_cap
    if has_local:
        t = _ld(_it(ltok), c0)
        tokens = (t < m_cap).select(t, m_cap)
    n = tokens * fx.Int32(topk)
    maskl = lds + fx.Int64(o_mask)
    cnt = lds + fx.Int64(o_cnt)
    start = lds + fx.Int64(o_start)
    scan = lds + fx.Int64(o_scan)
    if has_mask:
        # maskl[e] = local id + 1 for experts in the mask, else 0
        _load_mask(maskl, _it(mask), tid, E, e_cap)
        gpu.barrier()
        on = [
            (_lds(maskl, tid * fx.Int32(ept) + fx.Int32(k)) != c0).select(c1, c0)
            for k in range_constexpr(ept)
        ]
        acc = c0
        for v in on:
            acc = acc + v
        acc, _, _ = _block_excl_scan2(scan, tid, acc, c0)
        for k in range_constexpr(ept):
            loc = (on[k] != c0).select(acc + c1, c0)
            _lds_st(maskl, loc, tid * fx.Int32(ept) + fx.Int32(k))
            acc = acc + on[k]
    for k in range_constexpr(ept):
        _lds_st(cnt, c0, tid * fx.Int32(ept) + fx.Int32(k))
    gpu.barrier()
    e, inb, w, halo = _load_route(
        _it(ids), _it(weights), tid, n, True, topk, halo_end=n
    )
    if topk == 1:
        halo = fx.Int32(-1)
    e, valid, below, count = _check(
        e, inb, maskl, E, has_mask, halo, tid, topk, key_bits
    )
    first = valid & (below == c0)
    rank_in = c0
    for wv in range_constexpr(WAVES):
        mine = wave == fx.Int32(wv)
        base = _lds(cnt, e)
        rank_in = mine.select(base + below, rank_in)
        if mine & first:
            _lds_st(cnt, base + count, e)
        gpu.barrier()
    # experts -> start, and the group list, from one block scan
    cs = [_lds(cnt, tid * fx.Int32(ept) + fx.Int32(k)) for k in range_constexpr(ept)]
    a, b = c0, c0
    for c in cs:
        a = a + c
        b = b + (c > c0).select(c1, c0)
    acc, gacc, _ = _block_excl_scan2(scan, tid, a, b)
    for k in range_constexpr(ept):
        ex = tid * fx.Int32(ept) + fx.Int32(k)
        _lds_st(start, acc, ex)
        if cs[k] > c0:
            loc = (_lds(maskl, ex) - c1) if has_mask else ex
            _lds_st(lds + fx.Int64(o_ge), loc, gacc)
            _lds_st(lds + fx.Int64(o_gs), acc, gacc)
        acc = acc + cs[k]
        gacc = gacc + (cs[k] > c0).select(c1, c0)
    if tid == fx.Int32(BLOCK - 1):
        _lds_st(lds + fx.Int64(o_gs), acc, gacc)
        _lds_st(lds + fx.Int64(o_meta), acc, c0)
        _lds_st(lds + fx.Int64(o_meta), gacc, c1)
    gpu.barrier()
    if valid:
        rank = _lds(start, e) + rank_in
        tok = tid // fx.Int32(topk)
        slot = tid - tok * fx.Int32(topk)
        _lds_st(lds + fx.Int64(o_order), tok | (slot << fx.Int32(24)), rank)
        _lds_st(lds + fx.Int64(o_w), w.bitcast(fx.Int32), rank)
    gpu.barrier()


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
