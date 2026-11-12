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
  oneshot   one launch, up to STAGE routes; workgroup 0 sorts, the rest
            zero-fill the output.
  count +   per-chunk expert counts, then every chunk reads all counts,
  scatter   takes its prefix and scatters its routes, while extra workgroups
            of the scatter launch zero-fill the output.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .quant import (
    MX_MODES,
    amax_partial_block,
    quant_1x128_block,
    quant_dynamic_tensor_block,
    quant_dynamic_tensor_single,
    quant_mx_block,
    quant_static_block,
    quant_token_block,
)

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


def _load_route(ids_it, w_it, i, end, with_weight, topk=1, halo_end=None):
    """(raw expert id, in range, weight, halo) of route slot ``i``; loads only.
    ``halo`` is the id of slot ``i + WAVE`` (-1 past ``halo_end``, the total
    route count, default ``end``), which the repeated-expert check reads for a
    token whose slots run into the next wave or chunk."""
    c0 = fx.Int32(0)
    inb = i < end
    e = _ld(ids_it, inb.select(i, c0))
    w = _ld(w_it, inb.select(i, c0)) if with_weight else None
    halo = None
    if topk > 1:
        hi = i + fx.Int32(WAVE)
        hin = hi < (end if halo_end is None else halo_end)
        halo = hin.select(_ld(ids_it, hin.select(hi, c0)), fx.Int32(-1))
    return e, inb, w, halo


def _lane_value(v, src):
    return fx.Int32(rocdl.ds_bpermute(T.i32, _raw(src * fx.Int32(4)), _raw(v)))


def _low_mask(n):
    """Lane bits [0, n) of a 64-lane mask, for 0 <= n <= 64."""
    full = n >= fx.Int32(WAVE)
    sh = full.select(fx.Int32(0), n)
    return full.select(
        fx.Uint64(0xFFFFFFFFFFFFFFFF), (fx.Uint64(1) << fx.Uint64(sh)) - fx.Uint64(1)
    )


def _check(e, inb, mask_lds, num_experts, has_mask, halo, i, topk, key_bits):
    """(expert or 0, valid, below, count) of a route slot.

    valid: in range, a real expert, local under EP, and not repeated in a later
    slot of the same token (aiter's sorts keep only the last slot of a repeated
    expert). below/count: valid routes of the same expert in lower lanes / in
    the wave. A token's slots are consecutive lanes, so a repeat inside the
    wave is a same-expert lane in (lane, lane + topk - 1 - slot]; a token that
    runs past lane 63 also reads the next wave's first lanes through ``halo``.
    Every lane of the wave must take part."""
    c0 = fx.Int32(0)
    base = inb & (e >= c0) & (e < fx.Int32(num_experts))
    es = base.select(e, c0)
    if has_mask:
        base = base & (_lds(mask_lds, es) != c0)
    peers = fx.Uint64(rocdl.ballot(T.i64, base))
    for b in range_constexpr(key_bits):
        bit = ((e >> fx.Int32(b)) & fx.Int32(1)) != c0
        votes = fx.Uint64(rocdl.ballot(T.i64, bit))
        peers = peers & bit.select(votes, ~votes)
    valid = base
    if topk > 1:
        lane = i % fx.Int32(WAVE)
        slot = i - i // fx.Int32(topk) * fx.Int32(topk)
        reach = lane + fx.Int32(topk - 1) - slot
        later = _low_mask(reach + fx.Int32(1)) & ~_low_mask(lane + fx.Int32(1))
        dup = (peers & later) != fx.Uint64(0)
        for x in range_constexpr(topk - 1):
            hv = fx.Int32(rocdl.readlane(T.i32, _raw(halo), _raw(fx.Int32(x))))
            dup = dup | ((reach >= fx.Int32(WAVE + x)) & (hv == e))
        valid = base & ~dup
        peers = peers & fx.Uint64(rocdl.ballot(T.i64, valid))
    lo = fx.Uint32(peers)
    hi = fx.Uint32(peers >> fx.Uint64(32))
    below = rocdl.mbcnt_hi(
        T.i32, _raw(hi), _raw(rocdl.mbcnt_lo(T.i32, _raw(lo), _raw(c0)))
    )
    count = fx.Int32(fx.math.ctpop(lo)) + fx.Int32(fx.math.ctpop(hi))
    return valid.select(e, c0), valid, fx.Int32(below), count


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
def _histogram(
    hist,
    mask_lds,
    tid,
    ids_it,
    begin,
    end,
    n_total,
    num_experts,
    topk,
    has_mask,
    key_bits,
):
    """hist[e] += routes of expert e in [begin, end); loads are issued a stage
    (STAGE_ROUNDS rounds) at a time."""
    stages = (end - begin + fx.Int32(STAGE - 1)) // fx.Int32(STAGE)
    for st in range(fx.Int32(0), stages, fx.Int32(1)):
        i0 = begin + fx.Int32(st) * fx.Int32(STAGE) + tid
        raw = [
            _load_route(
                ids_it, None, i0 + fx.Int32(k * BLOCK), end, False, topk, n_total
            )
            for k in range_constexpr(STAGE_ROUNDS)
        ]
        routes = [
            _check(
                raw[k][0],
                raw[k][1],
                mask_lds,
                num_experts,
                has_mask,
                raw[k][3],
                i0 + fx.Int32(k * BLOCK),
                topk,
                key_bits,
            )
            for k in range_constexpr(STAGE_ROUNDS)
        ]
        for k in range_constexpr(STAGE_ROUNDS):
            e, valid = routes[k][0], routes[k][1]
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
    run,
    tbl,
    wave,
    k,
    e,
    valid,
    below,
    cnt,
    w,
    i,
    sids_it,
    sw_it,
    topk,
    e_cap,
    loc_it,
    end,
    with_loc,
):
    """Scatter one 256-route round; tbl half k % 2 is zero on entry and left zero.
    With ``with_loc``, loc_ids[i] gets the route's sorted row (-1 if dropped)."""
    c0, cK = fx.Int32(0), fx.Int32(topk)
    half = fx.Int32((k % 2) * WAVES * e_cap)
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
    if with_loc:
        _st(
            loc_it,
            valid.select(pre + below, fx.Int32(-1)),
            (i < end).select(i, fx.Int32(OOB)),
        )
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
    n_total,
    num_experts,
    topk,
    has_mask,
    key_bits,
    e_cap,
    loc_it,
    with_loc,
):
    """Scatter routes [begin, end) in route order; run[e] holds each expert's
    next row. Rounds alternate tbl halves, so each needs two barriers; rounds
    past ``end`` are skipped (the test is uniform across the workgroup)."""
    wave = tid // fx.Int32(WAVE)
    stages = (end - begin + fx.Int32(STAGE - 1)) // fx.Int32(STAGE)
    for st in range(fx.Int32(0), stages, fx.Int32(1)):
        base = begin + fx.Int32(st) * fx.Int32(STAGE)
        raw = [
            _load_route(
                ids_it, w_it, base + fx.Int32(k * BLOCK) + tid, end, True, topk, n_total
            )
            for k in range_constexpr(STAGE_ROUNDS)
        ]
        routes = [
            _check(
                raw[k][0],
                raw[k][1],
                mask_lds,
                num_experts,
                has_mask,
                raw[k][3],
                base + fx.Int32(k * BLOCK) + tid,
                topk,
                key_bits,
            )
            for k in range_constexpr(STAGE_ROUNDS)
        ]
        for k in range_constexpr(STAGE_ROUNDS):
            if base + fx.Int32(k * BLOCK) < end:
                e, valid, below, cnt = routes[k]
                _rank_round(
                    run,
                    tbl,
                    wave,
                    k,
                    e,
                    valid,
                    below,
                    cnt,
                    raw[k][2],
                    base + fx.Int32(k * BLOCK) + tid,
                    sids_it,
                    sw_it,
                    topk,
                    e_cap,
                    loc_it,
                    end,
                    with_loc,
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
    rounds,
    loc_it,
    with_loc,
):
    """The whole sort in one workgroup, for at most ``rounds`` * BLOCK routes
    (``rounds`` <= STAGE_ROUNDS, from the static token count): every route's id
    and weight is loaded once, up front, beside the mask."""
    n = tokens * fx.Int32(topk)
    wave = tid // fx.Int32(WAVE)
    raw = [
        _load_route(ids_it, w_it, fx.Int32(k * BLOCK) + tid, n, True, topk)
        for k in range_constexpr(rounds)
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
            raw[k][3],
            fx.Int32(k * BLOCK) + tid,
            topk,
            key_bits,
        )
        for k in range_constexpr(rounds)
    ]
    for k in range_constexpr(rounds):
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
    for k in range_constexpr(rounds):
        if fx.Int32(k * BLOCK) < n:
            _rank_round(
                run,
                tbl,
                wave,
                k,
                routes[k][0],
                routes[k][1],
                routes[k][2],
                routes[k][3],
                raw[k][2],
                fx.Int32(k * BLOCK) + tid,
                sids_it,
                sw_it,
                topk,
                e_cap,
                loc_it,
                n,
                with_loc,
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


@functools.lru_cache(maxsize=128)
def build(
    num_experts,
    topk,
    tile_m,
    has_mask,
    has_local,
    rounds=STAGE_ROUNDS,
    with_loc=False,
    quant=None,
):
    """Compile-time specialisation; returns (oneshot, count, scatter) launchers.
    ``rounds`` bounds the one-shot's 256-route rounds (from M * topk);
    ``with_loc`` also writes loc_ids (each route's sorted row). ``quant`` =
    (mode, cols, fp8_max) adds workgroups that quantize the token rows (see
    quant.py): in the one-shot launch after workgroup 0, in the count launch
    after the chunk workgroups. Dynamic per-tensor (``per_tensor_fp8_dynamic``)
    writes partial amaxes there and quantizes in the scatter launch, after its
    chunk workgroups (after a one-shot launch: in quant.build_tensor_quant's);
    ``per_tensor_fp8_single`` does both passes in one one-shot workgroup."""
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
        red: fx.Array[fx.Float32, 8, 16]

    dyn_tensor = quant is not None and quant[0] == "per_tensor_fp8_dynamic"

    def run_quant(lds, qx, qq, qs, blk, tokens, m_cap, n_part=None):
        """The quant workgroups of the one-shot or count launch. Dynamic
        per-tensor: partial amaxes into qs here (the scatter launch
        quantizes), or both passes in one workgroup (``_single``)."""
        mode, cols, fp8_max = quant
        rows = (tokens < m_cap).select(tokens, m_cap)
        if mode == "per_tensor_fp8_dynamic":
            amax_partial_block(_it(qx), _it(qs), lds.red.ptr, blk, rows, cols, n_part)
        elif mode == "per_tensor_fp8_single":
            quant_dynamic_tensor_single(
                _it(qx), _it(qq), _it(qs), lds.red.ptr, rows, cols, fp8_max
            )
        elif mode == "per_1x128_fp8":
            quant_1x128_block(
                _it(qx), _it(qq), _it(qs), blk, rows * fx.Int32(cols // 128), fp8_max
            )
        elif mode in MX_MODES:
            quant_mx_block(_it(qx), _it(qq), _it(qs), blk, rows, cols, *MX_MODES[mode])
        elif mode == "per_tensor_fp8":
            quant_static_block(_it(qx), _it(qq), _it(qs), blk, rows, cols, fp8_max)
        else:
            quant_token_block(
                _it(qx), _it(qq), _it(qs), lds.red.ptr, blk, rows, cols, fp8_max
            )

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
        loc_ids: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        zbuf: fx.Tensor,
        qx: fx.Tensor,
        qq: fx.Tensor,
        qs: fx.Tensor,
        m_cap: fx.Int32,
        row_v4: fx.Int32,
        qw: fx.Int32,
    ):
        """Workgroup 0 sorts; 1..qw quantize; the rest zero-fill."""
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
            if quant is not None:  # noqa: SIM102 - trace-time test, then a device one
                if bid <= qw:
                    run_quant(lds, qx, qq, qs, bid - c1, tokens, m_cap, qw)
            if bid > qw:
                _zero_fill(
                    _it(zbuf),
                    (bid - c1 - qw) * fx.Int32(BLOCK) + tid,
                    (gpu.grid_dim.x - c1 - qw) * fx.Int32(BLOCK),
                    _zero_rows(tokens, m_cap) * row_v4,
                )
        if bid == c0:
            _oneshot_body(
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
                rounds,
                _it(loc_ids),
                with_loc,
            )

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def count_kernel(
        ids: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        counts: fx.Tensor,
        zbuf: fx.Tensor,
        qx: fx.Tensor,
        qq: fx.Tensor,
        qs: fx.Tensor,
        m_cap: fx.Int32,
        chunk: fx.Int32,
        n_chunks: fx.Int32,
        row_v4: fx.Int32,
        qw: fx.Int32,
    ):
        """Workgroups below n_chunks count a chunk each, the next qw quantize;
        with row_v4 > 0 every workgroup also takes a share of the zero-fill."""
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
            _histogram(
                hist,
                lds.mask.ptr,
                tid,
                _it(ids),
                begin,
                end,
                n,
                E,
                topk,
                has_mask,
                key_bits,
            )
            _store_counts(hist, _it(counts), tid, bid, E, e_cap)
        if quant is not None:  # noqa: SIM102 - trace-time test, then a device one
            if (bid >= n_chunks) & (bid < n_chunks + qw):
                run_quant(lds, qx, qq, qs, bid - n_chunks, tokens, m_cap, qw)
        if row_v4 > fx.Int32(0):
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
        loc_ids: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        counts: fx.Tensor,
        zbuf: fx.Tensor,
        qx: fx.Tensor,
        qq: fx.Tensor,
        qs: fx.Tensor,
        qp: fx.Tensor,
        m_cap: fx.Int32,
        chunk: fx.Int32,
        n_chunks: fx.Int32,
        row_v4: fx.Int32,
        qw: fx.Int32,
        n_part: fx.Int32,
    ):
        """Workgroups below n_chunks scatter a chunk each; the next qw
        quantize (dynamic per-tensor, from the count launch's n_part partial
        amaxes in qp); the rest zero-fill the output, overlapping the scatter."""
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        tokens0 = m_cap
        if has_local:
            tokens0 = _ld(_it(ltok), fx.Int32(0))
        if dyn_tensor:  # noqa: SIM102 - trace-time test, then a device one
            if (bid >= n_chunks) & (bid < n_chunks + qw):
                _, cols, fp8_max = quant
                rows = (tokens0 < m_cap).select(tokens0, m_cap)
                quant_dynamic_tensor_block(
                    _it(qx),
                    _it(qq),
                    _it(qs),
                    _it(qp),
                    lds.red.ptr,
                    bid - n_chunks,
                    rows,
                    cols,
                    n_part,
                    fp8_max,
                )
        if bid >= n_chunks + qw:
            _zero_fill(
                _it(zbuf),
                (bid - n_chunks - qw) * fx.Int32(BLOCK) + tid,
                (gpu.grid_dim.x - n_chunks - qw) * fx.Int32(BLOCK),
                _zero_rows(tokens0, m_cap) * row_v4,
            )
        if bid < n_chunks:
            c0 = fx.Int32(0)
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
                n,
                E,
                topk,
                has_mask,
                key_bits,
                e_cap,
                _it(loc_ids),
                with_loc,
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
        loc_ids: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        zbuf: fx.Tensor,
        qx: fx.Tensor,
        qq: fx.Tensor,
        qs: fx.Tensor,
        m_cap: fx.Int32,
        row_v4: fx.Int32,
        qw: fx.Int32,
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
            loc_ids,
            mask,
            ltok,
            zbuf,
            qx,
            qq,
            qs,
            m_cap,
            row_v4,
            qw,
        ).launch(grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    @flyc.jit
    def launch_count(
        ids: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        counts: fx.Tensor,
        zbuf: fx.Tensor,
        qx: fx.Tensor,
        qq: fx.Tensor,
        qs: fx.Tensor,
        m_cap: fx.Int32,
        chunk: fx.Int32,
        n_chunks: fx.Int32,
        row_v4: fx.Int32,
        qw: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = None,
    ):
        count_kernel(
            ids,
            mask,
            ltok,
            counts,
            zbuf,
            qx,
            qq,
            qs,
            m_cap,
            chunk,
            n_chunks,
            row_v4,
            qw,
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
        loc_ids: fx.Tensor,
        mask: fx.Tensor,
        ltok: fx.Tensor,
        counts: fx.Tensor,
        zbuf: fx.Tensor,
        qx: fx.Tensor,
        qq: fx.Tensor,
        qs: fx.Tensor,
        qp: fx.Tensor,
        m_cap: fx.Int32,
        chunk: fx.Int32,
        n_chunks: fx.Int32,
        row_v4: fx.Int32,
        qw: fx.Int32,
        n_part: fx.Int32,
        grid: fx.Int32,
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
            loc_ids,
            mask,
            ltok,
            counts,
            zbuf,
            qx,
            qq,
            qs,
            qp,
            m_cap,
            chunk,
            n_chunks,
            row_v4,
            qw,
            n_part,
        ).launch(grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch_oneshot, launch_count, launch_scatter
