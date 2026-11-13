# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 per-token / per-tensor MoE decode, sortless, unquantized activations.

Same rounding points as aiter's gfx942 FlyDSL whole-graph decode: bf16 hidden
states against fp8 weights converted to fp32 (no activation quant), a bf16
intermediate (no requant), and FP32 sums.

Gate/Up: one workgroup per (route, 16 gate channels with their 16 up channels).
Each lane owns one channel row of aiter's shuffle_weight(w1, (16, 16)) layout
and streams whole 16-byte K blocks of it; the four lanes of a row and the four
waves split K, and the epilogue applies w1_scale, the activation and writes the
route's bf16 intermediate row.

Down: one workgroup per (token, 64 output rows); each wave owns 16 rows and
walks the token's routes, adding route_weight * w2_scale * (w2 . inter) in FP32;
one bf16 store per output, no atomics.
"""

import collections

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import common as fxh
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import _f32_to_bf16


def _rsrc(ptr):
    """Buffer descriptor at a typed pointer; offsets stay within one expert or row."""
    return fx.rocdl.get_buffer_rsrc(
        fx.get_iter(
            fx.rocdl.make_buffer_tensor(
                fx.make_view(ptr, fx.make_layout(1, 1)), max_size=True
            )
        )
    )


NT_W = 2  # gfx942 cpol nt bit; Python int literal (kb buffer-cache-bits-literal)


def _load_dwords(rsrc, byte_off, cache_modifier=0):
    """Four dwords (16 bytes) at a byte offset."""
    v = buffer_ops.buffer_load(
        rsrc, byte_off >> 2, vec_width=4, dtype=T.i32, cache_modifier=cache_modifier
    )
    return Vec(v)


def _fp8x16_to_f32(words):
    out = []
    for d in range_constexpr(4):
        w = _raw(words[d])
        lo = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, w, word_sel=False))
        hi = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, w, word_sel=True))
        out += [lo[0], lo[1], hi[0], hi[1]]
    return out


def _bf16x8_to_f32(words):
    out = []
    for d in range_constexpr(4):
        w = fx.Int32(words[d])
        out.append((w << 16).bitcast(fx.Float32))
        out.append((w & fx.Int32(-65536)).bitcast(fx.Float32))
    return out


def _sum16(v):
    s = v[0]
    for i in range_constexpr(1, 16):
        s = s + v[i]
    return s


def _lane_reduce(x):
    """Sum over the four lanes sharing a channel row (lane % 16)."""
    x = x + x.shuffle_xor(fx.Int32(16), fx.Int32(64))
    return x + x.shuffle_xor(fx.Int32(32), fx.Int32(64))


def compile_decode_gateup(
    *,
    N,
    K,
    TOPK,
    activation="silu",
    PER_TENSOR=False,
    swiglu_limit=None,
    CHUNK=8,
    WAVES=4,
    GLOBAL_E=None,
    HAS_MASK=False,
    HAS_NLT=False,
    R=None,
    DUP=2,
    CH2=None,
    SB=True,
    CM_U=2,
    CM_S=0,
    ORDER=0,
):
    """N = 2 * inter_dim, K = model_dim; grid (I / 16, routes). CHUNK K steps of
    weight loads are issued before any of them is used; WAVES waves split K.

    Each route's local expert id (the count of expert_mask ones below its global
    id under EP; -1 for a -1 id, a masked expert or a token at or past
    num_local_tokens) is written to local_ids by the first channel group, and
    Down reads it from there."""
    assert activation in ("silu", "swiglu")
    I = N // 2
    assert I % 16 == 0
    KB = K // 16
    assert WAVES in (4, 8) and KB % (4 * WAVES) == 0
    assert not HAS_MASK or GLOBAL_E, "EP needs the global expert count"
    NTHR = 64 * WAVES
    STEPS = KB // (4 * WAVES)
    CH = max(c for c in range(1, min(CHUNK, STEPS) + 1) if STEPS % c == 0)
    X_DW = K // 2
    X_ROUNDS = -(-X_DW // (4 * NTHR))
    limit = float(swiglu_limit) if swiglu_limit else 7.0
    # Dedup: routes sharing an expert are paired (ranks 2k, 2k+1 in route order);
    # the even-rank route's blocks compute both from one weight read.
    if R is None:
        DUP = 1
    NDUP = DUP
    R_ROUNDS = -(-R // NTHR) if R else 0
    CHP = CH if CH2 is None else max(
        c for c in range(1, min(CH2, STEPS) + 1) if STEPS % c == 0
    )

    def _act(g, u):
        if const_expr(activation == "swiglu"):
            neg = fx.Float32(-limit)
            g = -((-g).maximumf(neg))
            u = (-((-u).maximumf(neg))).maximumf(neg)
            t = rocdl.exp2(T.f32, _raw(g * (-1.702 * 1.4426950408889634)))
            return (g * rocdl.rcp(T.f32, _raw(1.0 + t))) * (u + 1.0)
        return (g * rocdl.rcp(T.f32, _raw(1.0 + fx.exp(-g)))) * u

    @fx.struct
    class SharedStorage:
        red: fx.Array[fx.Float32, 32 * WAVES * NDUP, 16]
        cnt: fx.Array[fx.Int32, WAVES, 16]
        dsum: fx.Array[fx.Int32, WAVES, 16]
        dmin: fx.Array[fx.Int32, WAVES, 16]
        x: fx.Array[fx.Int32, K // 2, 16]
        x2: fx.Array[fx.Int32, (NDUP - 1) * (K // 2) + 4, 16]

    @flyc.kernel
    def decode_gateup(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_local_ids: fx.Pointer,
        p_mask: fx.Pointer,
        p_nlt: fx.Pointer,
        M: fx.Int32,
        E: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        if const_expr(ORDER == 0):
            gi = gpu.block_idx.x
            r = gpu.block_idx.y
        else:
            gi = gpu.block_idx.y
            r = gpu.block_idx.x
        tok = r // TOPK
        lane = tid % 64
        wave = tid // 64
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        e_g = fx.recast_iter(fx.Int32, fxh._as_ptr(p_topk_ids))[r]
        valid = e_g >= 0
        if const_expr(HAS_NLT):
            valid = valid & (tok < fx.recast_iter(fx.Int32, fxh._as_ptr(p_nlt))[0])
        e = e_g
        if const_expr(HAS_MASK):
            mask = fx.recast_iter(fx.Int32, fxh._as_ptr(p_mask))
            e_c = valid.select(e_g, fx.Int32(0))
            valid = valid & (mask[e_c] != 0)
            below = fx.Int32(0)
            for j in range_constexpr(-(-GLOBAL_E // NTHR)):
                i = fx.Int32(j * NTHR) + tid
                ic = (i < GLOBAL_E).select(i, fx.Int32(0))
                below = below + ((i < e_c) & (mask[ic] != 0)).select(
                    fx.Int32(1), fx.Int32(0)
                )
            for sh in range_constexpr(6):
                below = below + below.shuffle_xor(fx.Int32(1 << sh), fx.Int32(64))
            cnt = fx.make_view(lds.cnt.ptr, fx.make_layout(WAVES, 1))
            if lane == 0:
                cnt[wave] = below
            gpu.barrier()
            e = cnt[0]
            for w in range_constexpr(1, WAVES):
                e = e + cnt[w]
        if (gi == 0) & (tid == 0):
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_local_ids))[r] = valid.select(
                e, fx.Int32(-1)
            )
        if valid:
            x_rsrc = _rsrc(fxh._as_ptr(p_input, fx.BFloat16) + fx.Int64(tok) * K)
            x_lds0 = fx.make_view(lds.x.ptr, fx.make_layout(K // 2, 1))
            for i in range_constexpr(X_ROUNDS):
                at = fx.Int32(i * 4 * NTHR) + tid * 4
                if at < X_DW:
                    xw = _load_dwords(x_rsrc, at * 4)
                    for d in range_constexpr(4):
                        x_lds0[at + d] = fx.Int32(xw[d])
        if const_expr(NDUP == 1):
            # the staged x must be visible to every wave before the K loop; with
            # NDUP == 2 the rank barrier below orders it (kernel.py's DUP = 1 path,
            # reached only at M * TOPK > 1024 there, had no barrier: an LDS race)
            gpu.barrier()
        ids_all = fx.recast_iter(fx.Int32, fxh._as_ptr(p_topk_ids))
        run1 = valid
        run2 = valid & (tid < 0)
        partner = fx.Int32(0)
        if const_expr(NDUP == 2):
            BIG = 1 << 30
            below = fx.Int32(0)
            pmin = fx.Int32(BIG)
            for j in range_constexpr(R_ROUNDS):
                t = fx.Int32(j * NTHR) + tid
                inr = t < R
                tc = inr.select(t, fx.Int32(0))
                m = inr & (ids_all[tc] == e_g)
                if const_expr(HAS_NLT):
                    m = m & ((tc // TOPK) < fx.recast_iter(fx.Int32, fxh._as_ptr(p_nlt))[0])
                below = below + (m & (t < r)).select(fx.Int32(1), fx.Int32(0))
                pmin = (m & (t > r) & (t < pmin)).select(t, pmin)
            for sh in range_constexpr(6):
                below = below + below.shuffle_xor(fx.Int32(1 << sh), fx.Int32(64))
                o = pmin.shuffle_xor(fx.Int32(1 << sh), fx.Int32(64))
                pmin = (o < pmin).select(o, pmin)
            dsum = fx.make_view(lds.dsum.ptr, fx.make_layout(WAVES, 1))
            dmin = fx.make_view(lds.dmin.ptr, fx.make_layout(WAVES, 1))
            if lane == 0:
                dsum[wave] = below
                dmin[wave] = pmin
            gpu.barrier()
            rank = dsum[0]
            partner = dmin[0]
            for w in range_constexpr(1, WAVES):
                rank = rank + dsum[w]
                o = dmin[w]
                partner = (o < partner).select(o, partner)
            # any other route with this expert -> cached loads (L2 sharing on the
            # same XCD: block id % 8 == gi % 8 for every route when I/16 % 8 == 0)
            uniq = (rank == 0) & (partner >= BIG)
            run2 = valid & (rank + (partner < BIG).select(fx.Int32(1), fx.Int32(0)) > 0)
            run1 = valid & uniq
        for nt_i in range_constexpr(NDUP):
            NT = 1
            CM = CM_U if const_expr(nt_i == 0) else CM_S
            go = run1 if const_expr(nt_i == 0) else run2
            if go:
                c = lane % 16
                q = lane // 16
                rows = [r] if const_expr(NT == 1) else [r, partner]
                w_rsrc = _rsrc(fxh._as_ptr(p_weight, fx.Uint8) + fx.Int64(e) * (N * K))
                g_row = gi * (16 * K) + c * 16
                u_row = (I // 16 + gi) * (16 * K) + c * 16
                x_lds = fx.make_view(lds.x.ptr, fx.make_layout(K // 2, 1))
                x2_lds = fx.make_view(lds.x2.ptr, fx.make_layout((NDUP - 1) * (K // 2) + 4, 1))
                CHN = CH if const_expr(NT == 1) else CHP
                ag0 = Vec.filled(16, 0.0, fx.Float32)
                au0 = Vec.filled(16, 0.0, fx.Float32)
                ag1 = Vec.filled(16, 0.0, fx.Float32)
                au1 = Vec.filled(16, 0.0, fx.Float32)
                for ch in range_constexpr(STEPS // CHN):
                    wgs, wus = [], []
                    for s in range_constexpr(CHN):
                        kb = fx.Int32((ch * CHN + s) * 4 * WAVES) + wave * 4 + q
                        wgs.append(_load_dwords(w_rsrc, g_row + kb * 256, CM))
                        wus.append(_load_dwords(w_rsrc, u_row + kb * 256, CM))
                    for s in range_constexpr(CHN):
                        kb = fx.Int32((ch * CHN + s) * 4 * WAVES) + wave * 4 + q
                        wgf = Vec.from_elements(_fp8x16_to_f32(wgs[s]), fx.Float32)
                        wuf = Vec.from_elements(_fp8x16_to_f32(wus[s]), fx.Float32)
                        xs = [x_lds[kb * 8 + d] for d in range_constexpr(8)]
                        xv = Vec.from_elements(
                            _bf16x8_to_f32(xs[:4]) + _bf16x8_to_f32(xs[4:]), fx.Float32
                        )
                        ag0 = fx.fma(wgf, xv, ag0)
                        au0 = fx.fma(wuf, xv, au0)
                        if const_expr(NT == 2):
                            ys = [x2_lds[kb * 8 + d] for d in range_constexpr(8)]
                            yv = Vec.from_elements(
                                _bf16x8_to_f32(ys[:4]) + _bf16x8_to_f32(ys[4:]), fx.Float32
                            )
                            ag1 = fx.fma(wgf, yv, ag1)
                            au1 = fx.fma(wuf, yv, au1)
                        if const_expr(NT > 1 and SB):
                            rocdl.sched_barrier(0)
                red = fx.make_view(lds.red.ptr, fx.make_layout(32 * WAVES * NDUP, 1))
                for t in range_constexpr(NT):
                    sg = _lane_reduce(_sum16(ag0 if const_expr(t == 0) else ag1))
                    su = _lane_reduce(_sum16(au0 if const_expr(t == 0) else au1))
                    if q == 0:
                        red[wave * 16 + c + t * 32 * WAVES] = sg
                        red[wave * 16 + c + (t * 32 * WAVES + 16 * WAVES)] = su
                gpu.barrier()
                if tid < 16 * NT:
                    cc = tid % 16
                    tk = tid // 16
                    base = tk * (32 * WAVES) + cc
                    g = red[base]
                    u = red[base + 16 * WAVES]
                    for w in range_constexpr(1, WAVES):
                        g = g + red[base + w * 16]
                        u = u + red[base + (16 * WAVES + w * 16)]
                    ws = fx.recast_iter(fx.Float32, fxh._as_ptr(p_w_scale))
                    chn = gi * 16 + cc
                    if const_expr(PER_TENSOR):
                        g = g * ws[e]
                        u = u * ws[e]
                    else:
                        g = g * ws[e * N + chn]
                        u = u * ws[e * N + I + chn]
                    y = _act(g, u)
                    yn = y.shuffle_xor(fx.Int32(1), fx.Int32(64))
                    pair = _f32_to_bf16(Vec.from_elements([y, yn], fx.Float32))
                    word = pair.bitcast(fx.Int32)[0]
                    orow = r if const_expr(NT == 1) else (tk == 0).select(r, partner)
                    if tid % 2 == 0:
                        o_rsrc = _rsrc(
                            fxh._as_ptr(p_output, fx.BFloat16) + fx.Int64(orow) * I
                        )
                        buffer_ops.buffer_store(word, o_rsrc, chn * 2, offset_is_bytes=True)

    @flyc.jit
    def launch(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_local_ids: fx.Pointer,
        p_mask: fx.Pointer,
        p_nlt: fx.Pointer,
        M: fx.Int32,
        E: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        decode_gateup(
            p_input,
            p_weight,
            p_output,
            p_topk_ids,
            p_w_scale,
            p_local_ids,
            p_mask,
            p_nlt,
            M,
            E,
        ).launch(
            grid=(I // 16, M * TOPK, 1) if const_expr(ORDER == 0) else (M * TOPK, I // 16, 1),
            block=(NTHR, 1, 1),
            stream=stream,
        )

    return launch


def compile_decode_down(
    *, N, K, TOPK, PER_TENSOR=False, ROUTE_LOADS=24, WAVES=4, CM_W=NT_W, CM_A=0, ORDER=0
):
    """N = model_dim, K = inter_dim; grid (N / 64, M); expert ids are Gate/Up's
    local_ids (a -1 route adds nothing). Up to ROUTE_LOADS loads per
    lane (whole routes) are issued before any is used; a dropped route (-1) loads
    expert 0 and is selected away."""
    assert N % (16 * WAVES) == 0 and K % 64 == 0
    ROWS = 16 * WAVES
    KB = K // 16
    STEPS = KB // 4
    RCH = max(1, min(TOPK, ROUTE_LOADS // (3 * STEPS)))

    @flyc.kernel
    def decode_down(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_topk_weight: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        if const_expr(ORDER == 0):
            nb = gpu.block_idx.x
            tok = gpu.block_idx.y
        else:
            nb = gpu.block_idx.y
            tok = gpu.block_idx.x
        lane = tid % 64
        wave = tid // 64
        c = lane % 16
        q = lane // 16
        n = nb * ROWS + wave * 16 + c
        ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_topk_ids))
        rws = fx.recast_iter(fx.Float32, fxh._as_ptr(p_topk_weight))
        ws = fx.recast_iter(fx.Float32, fxh._as_ptr(p_w_scale))
        row_off = (n // 16) * (16 * K) + (n % 16) * 16
        acc = fx.Float32(0.0)
        for j0 in range_constexpr(0, TOPK, RCH):
            js = list(range(j0, min(TOPK, j0 + RCH)))
            loaded = []
            for j in js:
                r = tok * TOPK + j
                e_raw = ids[r]
                e = (e_raw >= 0).select(e_raw, fx.Int32(0))
                w_rsrc = _rsrc(fxh._as_ptr(p_weight, fx.Uint8) + fx.Int64(e) * (N * K))
                a_rsrc = _rsrc(fxh._as_ptr(p_input, fx.BFloat16) + fx.Int64(r) * K)
                lw, la = [], []
                for s in range_constexpr(STEPS):
                    kb = fx.Int32(s * 4) + q
                    lw.append(_load_dwords(w_rsrc, row_off + kb * 256, CM_W))
                    la.append(_load_dwords(a_rsrc, kb * 32, CM_A))
                    la.append(_load_dwords(a_rsrc, kb * 32 + 16, CM_A))
                if const_expr(PER_TENSOR):
                    sc = ws[e]
                else:
                    sc = ws[e * N + n]
                loaded.append((lw, la, rws[r] * sc, e_raw >= 0))
            for lw, la, fac, ok in loaded:
                part = Vec.filled(16, 0.0, fx.Float32)
                for s in range_constexpr(STEPS):
                    av = Vec.from_elements(
                        _bf16x8_to_f32(la[2 * s]) + _bf16x8_to_f32(la[2 * s + 1]),
                        fx.Float32,
                    )
                    part = fx.fma(
                        Vec.from_elements(_fp8x16_to_f32(lw[s]), fx.Float32), av, part
                    )
                # A dropped route's intermediate row is never written, so it is
                # selected away rather than scaled by zero (it may hold NaN).
                acc = acc + ok.select(fac * _lane_reduce(_sum16(part)), fx.Float32(0.0))
        accn = acc.shuffle_xor(fx.Int32(1), fx.Int32(64))
        pair = _f32_to_bf16(Vec.from_elements([acc, accn], fx.Float32))
        word = pair.bitcast(fx.Int32)[0]
        if (q == 0) & (c % 2 == 0):
            o_rsrc = _rsrc(fxh._as_ptr(p_output, fx.BFloat16) + fx.Int64(tok) * N)
            buffer_ops.buffer_store(word, o_rsrc, n * 2, offset_is_bytes=True)

    @flyc.jit
    def launch(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_topk_weight: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        decode_down(
            p_input, p_weight, p_output, p_topk_ids, p_topk_weight, p_w_scale, M
        ).launch(
            grid=(N // ROWS, M, 1) if const_expr(ORDER == 0) else (M, N // ROWS, 1),
            block=(64 * WAVES, 1, 1),
            stream=stream,
        )

    return launch
