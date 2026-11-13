# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inline helpers for direct-mode kernels (``routing = "direct"``): no tile
map, the kernel groups its own routes and zero-fills its own output.

``group_routes`` puts up to 256 routes (M * topk <= DIRECT_MAX_ROUTES, one
per thread of a 256-thread workgroup) in aiter's sort order, in LDS: by expert, then by route index, with the sort's
rules (-1 and out-of-range ids dropped, experts outside ``expert_mask``
dropped and the rest renumbered to local ids, tokens at or past
``num_local_tokens`` dropped, a repeated expert within a token kept at its
last slot, deterministically, which aiter's sorts do not promise). It is ``route_to_tiles`` without the padding. ``zero_rows`` zeroes
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
    _raw,
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


def slot_lds_words(tile_m):
    """Int32 LDS words ``route_slot`` needs: the tile list, per-wave counts
    (expert routes, EP experts below), the slot's validity and rank, a dump."""
    return tile_m + 2 * WAVES + 3


@flyc.jit
def route_slot(
    lds,
    ids,
    mask,
    ltok,
    m_cap,
    slot,
    num_experts,
    topk,
    tile_m,
    has_mask,
    has_local,
):
    """The light direct-mode entry: for route slot ``slot`` (a workgroup per
    slot), its expert, its rank among that expert's routes, whether it leads a
    tile (rank % tile_m == 0), and that tile's route list, without ordering
    the other experts. Same rules and order as the sort (and group_routes).
    Every thread of a 256-thread workgroup calls it, up to 256 routes; two
    barriers. Returns (expert, rank, leader), workgroup-uniform; ``expert`` is
    the local id under EP and -1 when the slot's route is dropped. Afterwards
    ``lds[0:tile_m]`` holds the tile's route indices (token * topk + k) in
    sort order, -1 past its routes (only meaningful when ``leader``)."""
    E = num_experts
    tid = gpu.thread_idx.x
    wave, lane = tid // fx.Int32(WAVE), tid % fx.Int32(WAVE)
    c0, c1 = fx.Int32(0), fx.Int32(1)
    o_cnt, o_low = tile_m, tile_m + WAVES
    o_valid, o_rank = tile_m + 2 * WAVES, tile_m + 2 * WAVES + 1
    dump = fx.Int32(tile_m + 2 * WAVES + 2)
    tokens = m_cap
    if has_local:
        t = _ld(_it(ltok), c0)
        tokens = (t < m_cap).select(t, m_cap)
    n = tokens * fx.Int32(topk)
    mask_it = _it(mask)
    ids_it = _it(ids)
    inb = tid < n
    u = inb.select(tid, c0)
    e = inb.select(_ld(ids_it, u), fx.Int32(-1))
    # the slot's expert, read once (uniform)
    e_r = _ld(ids_it, slot)
    # a repeated expert within a token keeps its last slot: drop this slot if
    # a later slot of the token has the same id (the loads hit the same lines)
    tok = u // fx.Int32(topk)
    k0 = u - tok * fx.Int32(topk)
    dup = tid < c0
    for k in range_constexpr(1, topk):
        sk = k0 + fx.Int32(k)
        ok = inb & (sk < fx.Int32(topk))
        other = _ld(ids_it, tok * fx.Int32(topk) + ok.select(sk, k0))
        dup = dup | (ok & (other == e))
    valid = inb & (e >= c0) & (e < fx.Int32(E)) & ~dup
    if has_mask:
        valid = valid & (_ld(mask_it, valid.select(e, c0)) != c0)
    match = valid & (e == e_r)
    peers = fx.Uint64(rocdl.ballot(T.i64, match))
    lo, hi = fx.Uint32(peers), fx.Uint32(peers >> fx.Uint64(32))
    below = fx.Int32(
        rocdl.mbcnt_hi(T.i32, _raw(hi), _raw(rocdl.mbcnt_lo(T.i32, _raw(lo), _raw(c0))))
    )
    count = fx.Int32(fx.math.ctpop(lo)) + fx.Int32(fx.math.ctpop(hi))
    first_lane = lane == c0
    _lds_st(lds, count, first_lane.select(fx.Int32(o_cnt) + wave, dump))
    if has_mask:
        # local id of e_r: experts below it in the mask, one or more per thread
        low = c0
        for k in range_constexpr(-(-E // BLOCK)):
            x = fx.Int32(k * BLOCK) + tid
            ok = x < e_r
            low = low + (ok & (_ld(mask_it, ok.select(x, c0)) != c0)).select(c1, c0)
        for sh in range_constexpr(6):
            low = low + low.shuffle_xor(fx.Int32(1 << sh), fx.Int32(WAVE))
        _lds_st(lds, low, first_lane.select(fx.Int32(o_low) + wave, dump))
    _lds_st(lds, valid.select(c1, c0), (tid == slot).select(fx.Int32(o_valid), dump))
    _lds_st(lds, fx.Int32(-1), (tid < fx.Int32(tile_m)).select(tid, dump))
    gpu.barrier()
    off = c0
    loc = c0
    for w in range_constexpr(WAVES):
        before = wave > fx.Int32(w)
        off = off + before.select(_lds(lds, fx.Int32(o_cnt + w)), c0)
        if has_mask:
            loc = loc + _lds(lds, fx.Int32(o_low + w))
    rank_u = off + below
    _lds_st(lds, rank_u, (tid == slot).select(fx.Int32(o_rank), dump))
    gpu.barrier()
    r_valid = _lds(lds, fx.Int32(o_valid)) != c0
    rank = _lds(lds, fx.Int32(o_rank))
    leader = r_valid & ((rank % fx.Int32(tile_m)) == c0)
    j = rank_u - rank
    put = match & (j >= c0) & (j < fx.Int32(tile_m))
    _lds_st(lds, tid, put.select(j, dump))
    gpu.barrier()
    expert = loc if has_mask else e_r
    return r_valid.select(expert, fx.Int32(-1)), rank, leader


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
    "route_slot",
    "slot_lds_words",
    "zero_rows",
]
