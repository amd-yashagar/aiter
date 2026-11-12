# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE routing sort that emits the tile map, with the output zero-fill fused in.

Output contract (the same arrays and values as aiter's Opus sort):
  sorted_token_ids[rows]  int32 (topk_slot << 24) | token, experts in id order,
                          each expert's routes in ascending token order and
                          padded to a multiple of tile_m with (topk << 24) | M
  sorted_weights[rows]    fp32 route weight, 0 for padding
  sorted_expert_ids[...]  int32 local expert id per tile
  num_valid_ids[2]        padded row total, token count
plus the live-tile list:
  live_tiles[t, 3]        local expert id, first row, real rows in the tile
  num_live_tiles[1]       number of live tiles (num_valid_ids[0] / tile_m)

Routes with an id outside [0, E) or a masked expert are dropped. Within one
256-route round, a route's rank among same-expert routes comes from wave
ballots over the expert id bits; ranks across waves and rounds follow route
order, which is token-major, so expert segments come out in token order.

Two launch shapes:
  oneshot   one launch; workgroup 0 sorts, the rest zero-fill the output.
  count +   per-chunk expert counts (and the zero-fill), then every chunk
  scatter   reads all counts, takes its prefix and scatters its routes.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

BLOCK = 256
WAVE = 64
WAVES = BLOCK // WAVE
OOB = 0x7FFFFFFF
STAGE_ROUNDS = 8
STAGE = STAGE_ROUNDS * BLOCK
COUNT_BATCH = 32


def _raw(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _it(t):
    return fx.get_iter(fx.rocdl.make_buffer_tensor(t, max_size=True))


def _ld(it, i):
    return fx.ptr_load(it + fx.Int64(i))


def _st(it, v, i):
    fx.ptr_store(v, it + fx.Int64(i))


def _lds(ptr, i):
    return fx.ptr_load(ptr + fx.Int64(i))


def _lds_st(ptr, v, i):
    fx.ptr_store(v, ptr + fx.Int64(i))


def _load_route(ids_it, w_it, i, end, with_weight):
    """(raw expert id, in range, weight) of route slot ``i``; loads only."""
    inb = i < end
    j = inb.select(i, fx.Int32(0))
    e = _ld(ids_it, j)
    w = _ld(w_it, j) if with_weight else None
    return e, inb, w


def _check(e, inb, mask_lds, num_experts, has_mask, ids_it=None, i=None, topk=1):
    """(expert or 0, valid): in range, a real expert, local under EP, and not
    repeated in a later slot of the same token (aiter's sorts keep only the
    last slot of a repeated expert)."""
    valid = inb & (e >= fx.Int32(0)) & (e < fx.Int32(num_experts))
    es = valid.select(e, fx.Int32(0))
    if has_mask:
        valid = valid & (_lds(mask_lds, es) != fx.Int32(0))
    if topk > 1:
        slot = i - i // fx.Int32(topk) * fx.Int32(topk)
        for j in range_constexpr(1, topk):
            later = slot + fx.Int32(j) < fx.Int32(topk)
            other = _ld(ids_it, (valid & later).select(i + fx.Int32(j), fx.Int32(OOB)))
            valid = valid & ~(later & (other == e))
    return valid.select(e, fx.Int32(0)), valid


def _load_mask(mask_lds, mask_it, tid, num_experts, e_cap):
    """mask[e] into LDS (1 where there is no mask)."""
    for k in range_constexpr(0, e_cap, BLOCK):
        e = fx.Int32(k) + tid
        live = e < fx.Int32(num_experts)
        _lds_st(
            mask_lds,
            live.select(_ld(mask_it, live.select(e, fx.Int32(0))), fx.Int32(0)),
            e,
        )


def _peer_rank(e, valid, key_bits):
    """(routes in lower lanes of this wave with the same expert, routes in the
    wave with that expert); only meaningful where ``valid``."""
    peers = fx.Uint64(rocdl.ballot(T.i64, valid))
    for b in range_constexpr(key_bits):
        bit = ((e >> fx.Int32(b)) & fx.Int32(1)) != fx.Int32(0)
        votes = fx.Uint64(rocdl.ballot(T.i64, bit))
        peers = peers & bit.select(votes, ~votes)
    lo = fx.Uint32(peers)
    hi = fx.Uint32(peers >> fx.Uint64(32))
    below = rocdl.mbcnt_hi(
        T.i32, _raw(hi), _raw(rocdl.mbcnt_lo(T.i32, _raw(lo), _raw(fx.Int32(0))))
    )
    count = fx.Int32(fx.math.ctpop(lo)) + fx.Int32(fx.math.ctpop(hi))
    return fx.Int32(below), count


def _zero_rows(tokens, m_cap):
    """Output rows the sort zeroes: those below the token count, as aiter's does."""
    return (tokens < m_cap).select(tokens, m_cap)


def _clear(ptr, n, tid):
    """Zero LDS ptr[0, n); n must be a multiple of BLOCK."""
    assert n % BLOCK == 0
    for k in range_constexpr(0, n, BLOCK):
        _lds_st(ptr, fx.Int32(0), fx.Int32(k) + tid)


@flyc.jit
def _zero_fill(z_it, first, stride, n_v4):
    """Grid-stride zero of ``n_v4`` 16-byte vectors."""
    zero = fx.full(4, 0, fx.Int32)
    iters = (n_v4 - first + stride - fx.Int32(1)) // stride
    for j in range(fx.Int32(0), iters, fx.Int32(1)):
        v = first + fx.Int32(j) * stride
        idx = (v < n_v4).select(v * fx.Int32(4), fx.Int32(OOB))
        fx.ptr_store(zero, z_it + fx.Int64(idx))


@flyc.jit
def _histogram(hist, mask_lds, tid, ids_it, begin, end, num_experts, topk, has_mask):
    """hist[e] += routes of expert e in [begin, end); loads are issued a stage
    (STAGE_ROUNDS rounds) at a time."""
    stages = (end - begin + fx.Int32(STAGE - 1)) // fx.Int32(STAGE)
    for st in range(fx.Int32(0), stages, fx.Int32(1)):
        i0 = begin + fx.Int32(st) * fx.Int32(STAGE) + tid
        raw = [
            _load_route(ids_it, None, i0 + fx.Int32(k * BLOCK), end, False)
            for k in range_constexpr(STAGE_ROUNDS)
        ]
        routes = [
            _check(
                raw[k][0],
                raw[k][1],
                mask_lds,
                num_experts,
                has_mask,
                ids_it,
                i0 + fx.Int32(k * BLOCK),
                topk,
            )
            for k in range_constexpr(STAGE_ROUNDS)
        ]
        for k in range_constexpr(STAGE_ROUNDS):
            e, valid = routes[k]
            if valid:
                fx.atomic_add(hist + fx.Int64(e), fx.Int32(1), syncscope="workgroup")
    gpu.barrier()


def _wave_incl_scan(v, lane):
    for d in range_constexpr(6):
        step = fx.Int32(1 << d)
        src = (lane >= step).select(lane - step, lane)
        other = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(src * fx.Int32(4)), _raw(v)))
        v = (lane >= step).select(v + other, v)
    return v


def _block_excl_scan2(scan, tid, a, b):
    """Exclusive block scans of two values per thread; returns
    (excl_a, excl_b, total_a). Two barriers."""
    lane, wave = tid % fx.Int32(WAVE), tid // fx.Int32(WAVE)
    ia, ib = _wave_incl_scan(a, lane), _wave_incl_scan(b, lane)
    last = lane == fx.Int32(WAVE - 1)
    dummy = fx.Int32(2 * WAVES)
    _lds_st(scan, ia, last.select(wave, dummy))
    _lds_st(scan, ib, last.select(wave + fx.Int32(WAVES), dummy))
    gpu.barrier()
    pa, pb, total = fx.Int32(0), fx.Int32(0), fx.Int32(0)
    for w in range_constexpr(WAVES):
        ta = _lds(scan, fx.Int32(w))
        tb = _lds(scan, fx.Int32(w + WAVES))
        before = wave > fx.Int32(w)
        pa = before.select(pa + ta, pa)
        pb = before.select(pb + tb, pb)
        total = total + ta
    gpu.barrier()
    return pa + ia - a, pb + ib - b, total


def _expert_scan(
    hist, start, local, scan, tid, mask_lds, num_experts, tile_m, ept, has_mask
):
    """hist (raw counts) -> start[e] (padded exclusive prefix) and local[e]
    (local expert id); returns the padded total."""
    cE, cT, c0, c1 = fx.Int32(num_experts), fx.Int32(tile_m), fx.Int32(0), fx.Int32(1)
    base = tid * fx.Int32(ept)
    vals = []
    for k in range_constexpr(ept):
        e = base + fx.Int32(k)
        live = e < cE
        cnt = live.select(_lds(hist, live.select(e, c0)), c0)
        if has_mask:
            on = live & (_lds(mask_lds, live.select(e, c0)) != c0)
        else:
            on = live
        vals.append((e, (cnt + cT - c1) // cT * cT, on.select(c1, c0)))
    acc, lacc = c0, c0
    for _, p, l1 in vals:
        acc, lacc = acc + p, lacc + l1
    acc, lacc, total = _block_excl_scan2(scan, tid, acc, lacc)
    for e, p, l1 in vals:
        _lds_st(start, acc, e)
        _lds_st(local, lacc, e)
        acc, lacc = acc + p, lacc + l1
    gpu.barrier()
    return total


@flyc.jit
def _write_meta(nv_it, nlive_it, tid, tokens, total, tile_m):
    """num_valid_ids and num_live_tiles (one thread)."""
    c0, c1 = fx.Int32(0), fx.Int32(1)
    if tid == c0:
        _st(nv_it, total, c0)
        _st(nv_it, tokens, c1)
        _st(nlive_it, total // fx.Int32(tile_m), c0)


def _expert_of(hist, start, row, num_experts, key_bits):
    """(expert, end of its real rows): the last expert whose padded start is at
    or below ``row``."""
    c1 = fx.Int32(1)
    lo, hi = fx.Int32(0), fx.Int32(num_experts - 1)
    for _ in range_constexpr(key_bits + 1):
        mid = (lo + hi + c1) >> c1
        ok = _lds(start, mid) <= row
        lo = ok.select(mid, lo)
        hi = ok.select(hi, mid - c1)
    return lo, _lds(start, lo) + _lds(hist, lo)


@flyc.jit
def _write_tile_list(
    hist,
    start,
    local,
    seid_it,
    live_it,
    total,
    first,
    stride,
    num_experts,
    tile_m,
    key_bits,
):
    """sorted_expert_ids and live_tiles, one tile per thread across the grid."""
    c0, c1, cT = fx.Int32(0), fx.Int32(1), fx.Int32(tile_m)
    n_tiles = total // cT
    iters = (n_tiles - first + stride - c1) // stride
    for j in range(c0, iters, c1):
        t = first + fx.Int32(j) * stride
        row = t * cT
        lo, real_end = _expert_of(hist, start, row, num_experts, key_bits)
        rows = real_end - row
        lid = _lds(local, lo)
        _st(seid_it, lid, t)
        _st(live_it, lid, t * fx.Int32(3))
        _st(live_it, row, t * fx.Int32(3) + c1)
        _st(live_it, (rows < cT).select(rows, cT), t * fx.Int32(3) + fx.Int32(2))


@flyc.jit
def _write_padding(
    hist, start, sids_it, sw_it, tid, tokens, num_experts, topk, tile_m, ept
):
    """Each expert's padding rows (its tail past the real rows, under tile_m):
    sentinel id and weight 0. A thread takes its experts; quads wholly in
    padding take one 16-byte store per array, the boundary quad scalar stores.
    At most tile_m / 4 quads per expert, whatever the routing skew."""
    c0, c1, c4 = fx.Int32(0), fx.Int32(1), fx.Int32(4)
    cT = fx.Int32(tile_m)
    sentinel = (fx.Int32(topk) << fx.Int32(24)) | tokens
    vsent = fx.full(4, 0, fx.Int32) + sentinel
    vzero = fx.full(4, 0.0, fx.Float32)
    for k in range_constexpr(ept):
        e = tid * fx.Int32(ept) + fx.Int32(k)
        live = e < fx.Int32(num_experts)
        es = live.select(e, c0)
        cnt = live.select(_lds(hist, es), c0)
        pad_beg = _lds(start, es) + cnt
        pad_end = pad_beg + (cnt + cT - c1) // cT * cT - cnt
        q0 = pad_beg // c4
        for q in range_constexpr(tile_m // 4 + 1):
            p = (q0 + fx.Int32(q)) * c4
            if (p >= pad_beg) & (p + c4 <= pad_end):
                fx.ptr_store(vsent, sids_it + fx.Int64(p))
                fx.ptr_store(vzero, sw_it + fx.Int64(p))
            if (p < pad_end) & ((p < pad_beg) | (p + c4 > pad_end)):
                for r in range_constexpr(4):
                    row = p + fx.Int32(r)
                    ok = (row >= pad_beg) & (row < pad_end)
                    _st(sids_it, sentinel, ok.select(row, fx.Int32(OOB)))
                    _st(sw_it, fx.Float32(0.0), ok.select(row, fx.Int32(OOB)))


@flyc.jit
def _rank_round(
    run, tbl, wave, k, e, valid, w, i, sids_it, sw_it, topk, key_bits, e_cap
):
    """Scatter one 256-route round; tbl half k % 2 is zero on entry and left zero."""
    c0, cK = fx.Int32(0), fx.Int32(topk)
    half = fx.Int32((k % 2) * WAVES * e_cap)
    below, cnt = _peer_rank(e, valid, key_bits)
    lead = valid & (below == c0)
    slot = half + wave * fx.Int32(e_cap) + e
    _lds_st(tbl, lead.select(cnt, c0), lead.select(slot, fx.Int32(2 * WAVES * e_cap)))
    gpu.barrier()
    pre = _lds(run, e)
    for q in range_constexpr(WAVES - 1):
        v = _lds(tbl, half + fx.Int32(q * e_cap) + e)
        pre = (wave > fx.Int32(q)).select(pre + v, pre)
    pos = valid.select(pre + below, fx.Int32(OOB))
    tok = i // cK
    _st(sids_it, ((i - tok * cK) << fx.Int32(24)) | tok, pos)
    _st(sw_it, w, pos)
    gpu.barrier()
    if lead:
        fx.atomic_add(run + fx.Int64(e), cnt, syncscope="workgroup")
        _lds_st(tbl, c0, slot)


@flyc.jit
def _rank_and_scatter(
    run,
    tbl,
    mask_lds,
    tid,
    ids_it,
    w_it,
    sids_it,
    sw_it,
    begin,
    end,
    num_experts,
    topk,
    has_mask,
    key_bits,
    e_cap,
):
    """Scatter routes [begin, end) in route order; run[e] holds each expert's
    next row. Rounds alternate tbl halves, so each needs two barriers; rounds
    past ``end`` are skipped (the test is uniform across the workgroup)."""
    wave = tid // fx.Int32(WAVE)
    stages = (end - begin + fx.Int32(STAGE - 1)) // fx.Int32(STAGE)
    for st in range(fx.Int32(0), stages, fx.Int32(1)):
        base = begin + fx.Int32(st) * fx.Int32(STAGE)
        raw = [
            _load_route(ids_it, w_it, base + fx.Int32(k * BLOCK) + tid, end, True)
            for k in range_constexpr(STAGE_ROUNDS)
        ]
        routes = [
            _check(
                raw[k][0],
                raw[k][1],
                mask_lds,
                num_experts,
                has_mask,
                ids_it,
                base + fx.Int32(k * BLOCK) + tid,
                topk,
            )
            for k in range_constexpr(STAGE_ROUNDS)
        ]
        for k in range_constexpr(STAGE_ROUNDS):
            if base + fx.Int32(k * BLOCK) < end:
                e, valid = routes[k]
                _rank_round(
                    run,
                    tbl,
                    wave,
                    k,
                    e,
                    valid,
                    raw[k][2],
                    base + fx.Int32(k * BLOCK) + tid,
                    sids_it,
                    sw_it,
                    topk,
                    key_bits,
                    e_cap,
                )
        gpu.barrier()


@flyc.jit
def _oneshot_body(
    hist,
    run,
    local,
    start,
    tbl,
    scan,
    mask_lds,
    tid,
    ids_it,
    w_it,
    mask_it,
    sids_it,
    sw_it,
    seid_it,
    nv_it,
    live_it,
    nlive_it,
    tokens,
    num_experts,
    topk,
    tile_m,
    has_mask,
    key_bits,
    ept,
    e_cap,
):
    """The whole sort in one workgroup, for at most STAGE routes: every route's
    id and weight is loaded once, up front, beside the mask."""
    n = tokens * fx.Int32(topk)
    wave = tid // fx.Int32(WAVE)
    raw = [
        _load_route(ids_it, w_it, fx.Int32(k * BLOCK) + tid, n, True)
        for k in range_constexpr(STAGE_ROUNDS)
    ]
    if has_mask:
        _load_mask(mask_lds, mask_it, tid, num_experts, e_cap)
    _clear(hist, e_cap, tid)
    _clear(tbl, 2 * WAVES * e_cap, tid)
    gpu.barrier()
    routes = [
        _check(
            raw[k][0],
            raw[k][1],
            mask_lds,
            num_experts,
            has_mask,
            ids_it,
            fx.Int32(k * BLOCK) + tid,
            topk,
        )
        for k in range_constexpr(STAGE_ROUNDS)
    ]
    for k in range_constexpr(STAGE_ROUNDS):
        if routes[k][1]:
            fx.atomic_add(
                hist + fx.Int64(routes[k][0]), fx.Int32(1), syncscope="workgroup"
            )
    gpu.barrier()
    total = _expert_scan(
        hist, start, local, scan, tid, mask_lds, num_experts, tile_m, ept, has_mask
    )
    _write_meta(nv_it, nlive_it, tid, tokens, total, tile_m)
    _write_tile_list(
        hist,
        start,
        local,
        seid_it,
        live_it,
        total,
        tid,
        fx.Int32(BLOCK),
        num_experts,
        tile_m,
        key_bits,
    )
    _write_padding(
        hist, start, sids_it, sw_it, tid, tokens, num_experts, topk, tile_m, ept
    )
    for k in range_constexpr(0, e_cap, BLOCK):
        e = fx.Int32(k) + tid
        _lds_st(run, _lds(start, e), e)
    gpu.barrier()
    for k in range_constexpr(STAGE_ROUNDS):
        if fx.Int32(k * BLOCK) < n:
            _rank_round(
                run,
                tbl,
                wave,
                k,
                routes[k][0],
                routes[k][1],
                raw[k][2],
                fx.Int32(k * BLOCK) + tid,
                sids_it,
                sw_it,
                topk,
                key_bits,
                e_cap,
            )


@flyc.jit
def _sum_counts(hist, run, c_it, tid, bid, n_chunks, num_experts, e_cap):
    """hist[e] = counts over all chunks, run[e] = counts over chunks before bid;
    COUNT_BATCH chunks' loads are issued together."""
    c0, cE = fx.Int32(0), fx.Int32(num_experts)
    batches = (n_chunks + fx.Int32(COUNT_BATCH - 1)) // fx.Int32(COUNT_BATCH)
    for bt in range(c0, batches, fx.Int32(1)):
        g0 = fx.Int32(bt) * fx.Int32(COUNT_BATCH)
        for k in range_constexpr(0, e_cap, BLOCK):
            e = fx.Int32(k) + tid
            live = e < cE
            es = live.select(e, c0)
            tot, pre = c0, c0
            vals = []
            for q in range_constexpr(COUNT_BATCH):
                g = g0 + fx.Int32(q)
                ok = live & (g < n_chunks)
                vals.append((g, ok.select(_ld(c_it, ok.select(g * cE + es, c0)), c0)))
            for g, c in vals:
                tot = tot + c
                pre = pre + (g < bid).select(c, c0)
            _lds_st(hist, _lds(hist, e) + tot, e)
            _lds_st(run, _lds(run, e) + pre, e)
    gpu.barrier()


@flyc.jit
def _store_counts(hist, c_it, tid, bid, num_experts, e_cap):
    for k in range_constexpr(0, e_cap, BLOCK):
        e = fx.Int32(k) + tid
        if e < fx.Int32(num_experts):
            _st(c_it, _lds(hist, e), bid * fx.Int32(num_experts) + e)


@flyc.jit
def _oneshot_big(
    hist,
    run,
    local,
    start,
    tbl,
    scan,
    mask_lds,
    tid,
    ids_it,
    w_it,
    mask_it,
    sids_it,
    sw_it,
    seid_it,
    nv_it,
    live_it,
    nlive_it,
    tokens,
    num_experts,
    topk,
    tile_m,
    has_mask,
    key_bits,
    ept,
    e_cap,
):
    """The whole sort in one workgroup for more than STAGE routes: the routes
    stream through in stages, once for the histogram and once to scatter."""
    n = tokens * fx.Int32(topk)
    if has_mask:
        _load_mask(mask_lds, mask_it, tid, num_experts, e_cap)
    _clear(hist, e_cap, tid)
    _clear(tbl, 2 * WAVES * e_cap, tid)
    gpu.barrier()
    _histogram(hist, mask_lds, tid, ids_it, fx.Int32(0), n, num_experts, topk, has_mask)
    total = _expert_scan(
        hist, start, local, scan, tid, mask_lds, num_experts, tile_m, ept, has_mask
    )
    _write_meta(nv_it, nlive_it, tid, tokens, total, tile_m)
    _write_tile_list(
        hist,
        start,
        local,
        seid_it,
        live_it,
        total,
        tid,
        fx.Int32(BLOCK),
        num_experts,
        tile_m,
        key_bits,
    )
    _write_padding(
        hist, start, sids_it, sw_it, tid, tokens, num_experts, topk, tile_m, ept
    )
    for k in range_constexpr(0, e_cap, BLOCK):
        e = fx.Int32(k) + tid
        _lds_st(run, _lds(start, e), e)
    gpu.barrier()
    _rank_and_scatter(
        run,
        tbl,
        mask_lds,
        tid,
        ids_it,
        w_it,
        sids_it,
        sw_it,
        fx.Int32(0),
        n,
        num_experts,
        topk,
        has_mask,
        key_bits,
        e_cap,
    )


@functools.lru_cache(maxsize=128)
def build(num_experts, topk, tile_m, has_mask, has_local, big=False):
    """Compile-time specialisation; returns (oneshot, count, scatter) launchers.
    ``big`` makes the one-shot stream more than STAGE routes."""
    E = num_experts
    key_bits = max(1, (E - 1).bit_length())
    ept = -(-E // BLOCK)  # experts per thread in the expert scan
    e_cap = ept * BLOCK

    @fx.struct
    class Lds:
        hist: fx.Array[fx.Int32, e_cap, 16]
        run: fx.Array[fx.Int32, e_cap, 16]
        local: fx.Array[fx.Int32, e_cap, 16]
        start: fx.Array[fx.Int32, e_cap, 16]
        tbl: fx.Array[fx.Int32, 2 * WAVES * e_cap + 1, 16]
        mask: fx.Array[fx.Int32, e_cap, 16]
        scan: fx.Array[fx.Int32, 2 * WAVES + 1, 16]

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def oneshot_kernel(
        ids: fx.Tensor,
        weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        sorted_eids: fx.Tensor,
        num_valid: fx.Tensor,
        live_tiles: fx.Tensor,
        num_live: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        zbuf: fx.Tensor,
        m_cap: fx.Int32,
        row_v4: fx.Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        c0, c1 = fx.Int32(0), fx.Int32(1)
        lds = fx.SharedAllocator().allocate(Lds).peek()
        hist, run, local, start = (
            lds.hist.ptr,
            lds.run.ptr,
            lds.local.ptr,
            lds.start.ptr,
        )
        tbl, scan = lds.tbl.ptr, lds.scan.ptr
        tokens = m_cap
        if has_local:
            tokens = _ld(_it(ltok), c0)
        if bid != c0:
            _zero_fill(
                _it(zbuf),
                (bid - c1) * fx.Int32(BLOCK) + tid,
                (gpu.grid_dim.x - c1) * fx.Int32(BLOCK),
                _zero_rows(tokens, m_cap) * row_v4,
            )
        if bid == c0:
            body = _oneshot_big if big else _oneshot_body
            body(
                hist,
                run,
                local,
                start,
                tbl,
                scan,
                lds.mask.ptr,
                tid,
                _it(ids),
                _it(weights),
                _it(mask),
                _it(sorted_ids),
                _it(sorted_w),
                _it(sorted_eids),
                _it(num_valid),
                _it(live_tiles),
                _it(num_live),
                tokens,
                E,
                topk,
                tile_m,
                has_mask,
                key_bits,
                ept,
                e_cap,
            )

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def count_kernel(
        ids: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        counts: fx.Tensor,
        zbuf: fx.Tensor,
        m_cap: fx.Int32,
        chunk: fx.Int32,
        n_chunks: fx.Int32,
        row_v4: fx.Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        c0 = fx.Int32(0)
        lds = fx.SharedAllocator().allocate(Lds).peek()
        hist = lds.hist.ptr
        tokens = m_cap
        if has_local:
            tokens = _ld(_it(ltok), c0)
        if bid < n_chunks:
            n = tokens * fx.Int32(topk)
            begin = bid * chunk
            end = begin + chunk
            end = (end < n).select(end, n)
            begin = (begin < end).select(begin, end)
            if has_mask:
                _load_mask(lds.mask.ptr, _it(mask), tid, E, e_cap)
            _clear(hist, e_cap, tid)
            gpu.barrier()
            _histogram(hist, lds.mask.ptr, tid, _it(ids), begin, end, E, topk, has_mask)
            _store_counts(hist, _it(counts), tid, bid, E, e_cap)
        _zero_fill(
            _it(zbuf),
            bid * fx.Int32(BLOCK) + tid,
            gpu.grid_dim.x * fx.Int32(BLOCK),
            _zero_rows(tokens, m_cap) * row_v4,
        )

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def scatter_kernel(
        ids: fx.Tensor,
        weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        sorted_eids: fx.Tensor,
        num_valid: fx.Tensor,
        live_tiles: fx.Tensor,
        num_live: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        counts: fx.Tensor,
        m_cap: fx.Int32,
        chunk: fx.Int32,
        n_chunks: fx.Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        c0 = fx.Int32(0)
        lds = fx.SharedAllocator().allocate(Lds).peek()
        hist, run, local, start = (
            lds.hist.ptr,
            lds.run.ptr,
            lds.local.ptr,
            lds.start.ptr,
        )
        tbl, scan = lds.tbl.ptr, lds.scan.ptr
        ids_it, w_it, mask_it = _it(ids), _it(weights), _it(mask)
        tokens = m_cap
        if has_local:
            tokens = _ld(_it(ltok), c0)
        n = tokens * fx.Int32(topk)
        mask_lds = lds.mask.ptr
        if has_mask:
            _load_mask(mask_lds, mask_it, tid, E, e_cap)
        _clear(hist, e_cap, tid)
        _clear(run, e_cap, tid)
        _clear(tbl, 2 * WAVES * e_cap, tid)
        gpu.barrier()
        _sum_counts(hist, run, _it(counts), tid, bid, n_chunks, E, e_cap)
        total = _expert_scan(
            hist, start, local, scan, tid, mask_lds, E, tile_m, ept, has_mask
        )
        if bid == c0:
            _write_meta(_it(num_valid), _it(num_live), tid, tokens, total, tile_m)
        _write_tile_list(
            hist,
            start,
            local,
            _it(sorted_eids),
            _it(live_tiles),
            total,
            bid * fx.Int32(BLOCK) + tid,
            n_chunks * fx.Int32(BLOCK),
            E,
            tile_m,
            key_bits,
        )
        if bid == c0:
            _write_padding(
                hist,
                start,
                _it(sorted_ids),
                _it(sorted_w),
                tid,
                tokens,
                E,
                topk,
                tile_m,
                ept,
            )
        for k in range_constexpr(0, e_cap, BLOCK):
            e = fx.Int32(k) + tid
            _lds_st(run, _lds(run, e) + _lds(start, e), e)
        gpu.barrier()
        begin = bid * chunk
        end = begin + chunk
        end = (end < n).select(end, n)
        begin = (begin < end).select(begin, end)
        _rank_and_scatter(
            run,
            tbl,
            mask_lds,
            tid,
            ids_it,
            w_it,
            _it(sorted_ids),
            _it(sorted_w),
            begin,
            end,
            E,
            topk,
            has_mask,
            key_bits,
            e_cap,
        )

    @flyc.jit
    def launch_oneshot(
        ids: fx.Tensor,
        weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        sorted_eids: fx.Tensor,
        num_valid: fx.Tensor,
        live_tiles: fx.Tensor,
        num_live: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        zbuf: fx.Tensor,
        m_cap: fx.Int32,
        row_v4: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = None,
    ):
        oneshot_kernel(
            ids,
            weights,
            sorted_ids,
            sorted_w,
            sorted_eids,
            num_valid,
            live_tiles,
            num_live,
            mask,
            ltok,
            zbuf,
            m_cap,
            row_v4,
        ).launch(grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    @flyc.jit
    def launch_count(
        ids: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        counts: fx.Tensor,
        zbuf: fx.Tensor,
        m_cap: fx.Int32,
        chunk: fx.Int32,
        n_chunks: fx.Int32,
        row_v4: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = None,
    ):
        count_kernel(
            ids, mask, ltok, counts, zbuf, m_cap, chunk, n_chunks, row_v4
        ).launch(grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    @flyc.jit
    def launch_scatter(
        ids: fx.Tensor,
        weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        sorted_eids: fx.Tensor,
        num_valid: fx.Tensor,
        live_tiles: fx.Tensor,
        num_live: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        counts: fx.Tensor,
        m_cap: fx.Int32,
        chunk: fx.Int32,
        n_chunks: fx.Int32,
        stream: fx.Stream = None,
    ):
        scatter_kernel(
            ids,
            weights,
            sorted_ids,
            sorted_w,
            sorted_eids,
            num_valid,
            live_tiles,
            num_live,
            mask,
            ltok,
            counts,
            m_cap,
            chunk,
            n_chunks,
        ).launch(grid=(n_chunks, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch_oneshot, launch_count, launch_scatter
