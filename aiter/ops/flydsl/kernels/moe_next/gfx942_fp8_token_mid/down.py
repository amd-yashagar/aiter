# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 per-token MoE Down for the mid-M band, and its per-token combine.

Down: grid (model_dim / (16 * NBD * WAVES), live-tile bound). Each wave walks NBD
blocks of 16 output channels of one live tile's expert over the whole K
(inter_dim), so each live expert's W2 is read once per call. B is the tile's
bf16 intermediate rows (Gate/Up's [sorted rows, inter_dim], contiguous per
tile), staged once per workgroup in LDS; the epilogue scales
by w2_scale and the route weight and either writes one bf16 row per route at
[token, slot] (ROWS) or adds into out[token] with packed bf16 atomics (the sort
zero-fills the output).

Combine (ROWS): out[t] = sum over the token's routes of rows[t, slot] in FP32, in
slot order, rounded to bf16 once; dropped routes (loc_ids -1) add nothing and
tokens at or past num_local_tokens are left untouched.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import common as fxh

from .common import (
    bf16_pair,
    bf16x4_words,
    fp8_dword_to_bf16x4,
    lds_load_x4,
    lds_store_x2,
    lds_store_x4,
    load_s1,
    load_x4,
    mfma,
    rsrc,
)


def compile_down(
    *,
    N,
    K,
    TOPK,
    S_MAX,
    NBD=4,
    WAVES=4,
    PWD=2,
    ROWS=True,
    W_CACHE=0,
    COALESCE=True,
    PER_TENSOR=False,
    KC=False,
):
    """N = model_dim, K = inter_dim; PER_TENSOR takes one W2 scale per expert. Each
    wave walks NBD channel blocks with W2
    prefetched PWD blocks ahead; the tile's intermediate rows are staged once
    per workgroup in LDS. COALESCE transposes each wave's NBD x 16 output
    channels through LDS so every lane writes (or adds) 16 contiguous bytes of
    one row.

    KC walks K in 64-wide steps through a double-buffered LDS chunk of the tile's
    rows (16-byte segments swizzled by row % 8), every block of the wave advancing
    per step, so only one step of the tile is live and 128-row tiles fit; each
    output's MFMA order is unchanged, so the result is bit-identical."""
    assert K % 64 == 0 and N % (16 * NBD * WAVES) == 0
    KSTEPS = K // 64
    NTHR = 64 * WAVES
    ROW_DW = K // 2 + 4  # LDS row stride in dwords, padded by 16 bytes
    SEGS = K // 8  # 16-byte segments per row
    LDS_DW = S_MAX * 16 * ROW_DW
    OOB = 0x7FFFFFF0
    TR_DW = NBD * 8 + 4  # transpose row stride in dwords
    TR_WAVE = S_MAX * 16 * TR_DW
    # The transpose reuses the intermediate tile's LDS once every wave is done.
    LDS_DW = max(LDS_DW, WAVES * TR_WAVE) if COALESCE else LDS_DW
    CH_DW = 32  # one KC chunk row: 64 bf16
    BUF_DW = S_MAX * 16 * CH_DW
    TRG = S_MAX  # subtiles per transpose group
    if KC:
        # Transpose groups of TRG subtiles so the transpose fits the chunk buffers.
        TRG = max(1, min(S_MAX, 2 * BUF_DW // (WAVES * 16 * TR_DW)))
        TR_WAVE = TRG * 16 * TR_DW
        LDS_DW = max(2 * BUF_DW, WAVES * TR_WAVE) if COALESCE else 2 * BUF_DW

    @fx.struct
    class SharedStorage:
        a: fx.Array[fx.Int32, LDS_DW, 16]

    @flyc.kernel
    def mid_down(
        p_inter: fx.Pointer,
        p_w: fx.Pointer,
        p_out: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_w: fx.Pointer,
        p_live: fx.Pointer,
        p_nlive: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        bx = gpu.block_idx.x
        t = gpu.block_idx.y
        lane = tid % 64
        wave = tid // 64
        c = lane % 16
        q = lane // 16
        nlive = load_s1(p_nlive, fx.Int32(0))
        if t < nlive:
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            a_lds = lds.a.ptr
            o_lds = a_lds
            o_base = wave * TR_WAVE
            e = load_s1(p_live, t * 3)
            r0 = load_s1(p_live, t * 3 + 1)
            real = load_s1(p_live, t * 3 + 2)
            n_sub = (real + 15) // 16
            nb0 = (bx * WAVES + wave) * NBD
            w_rs = rsrc(fxh._as_ptr(p_w, fx.Uint8) + fx.Int64(e) * (N * K), N * K)
            a_rs = rsrc(fxh._as_ptr(p_inter, fx.BFloat16))
            s_rs = rsrc(fxh._as_ptr(p_w_scale, fx.Float32))
            ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids))
            rws = fx.recast_iter(fx.Float32, fxh._as_ptr(p_sorted_w))
            o_rs = rsrc(fxh._as_ptr(p_out, fx.BFloat16))
            out_bf16 = fxh._as_ptr(p_out, fx.BFloat16)
            w_lane = c * 16 + q * 256

            def load_w(b):
                live = b < NBD
                base = (nb0 + b) * (16 * K) + w_lane
                return [
                    load_x4(w_rs, live.select(base + st * 1024, OOB), W_CACHE)
                    for st in range_constexpr(KSTEPS)
                ]

            def store_direct(s, ch, y, toks, slots, oks):
                if oks[s]:
                    if const_expr(ROWS):
                        buffer_ops.buffer_store(
                            bf16x4_words(y),
                            o_rs,
                            ((toks[s] * TOPK + slots[s]) * N + ch) * 2,
                            offset_is_bytes=True,
                        )
                    else:
                        yb = fxh._f32_to_bf16(Vec.from_elements(y, fx.Float32))
                        fxh.atomic_add_bf16(out_bf16 + (fx.Int64(toks[s]) * N + ch), yb)

            def kc_body(SV, flush, toks, slots, facs, oks, scs):
                n_seg = SV * 16 * 8
                RND = -(-n_seg // NTHR)
                g_off, l_off = [], []
                for i in range_constexpr(RND):
                    idx = tid + i * NTHR
                    row = (idx < n_seg).select(idx // 8, fx.Int32(0))
                    seg = idx % 8
                    g_off.append(((r0 + row) * K) * 2 + seg * 16)
                    l_off.append(row * CH_DW + (seg ^ (row % 8)) * 4)

                def a_load(st):
                    return [
                        load_x4(a_rs, g_off[i] + st * 128) for i in range_constexpr(RND)
                    ]

                def a_store(buf, regs):
                    for i in range_constexpr(RND):
                        if const_expr((i + 1) * NTHR <= n_seg):
                            lds_store_x4(a_lds, buf * BUF_DW + l_off[i], regs[i])
                        else:
                            if tid + i * NTHR < n_seg:
                                lds_store_x4(a_lds, buf * BUF_DW + l_off[i], regs[i])

                def w_step(st):
                    return [
                        load_x4(
                            w_rs, (nb0 + b) * (16 * K) + w_lane + st * 1024, W_CACHE
                        )
                        for b in range_constexpr(NBD)
                    ]

                a_store(0, a_load(0))
                wq = [w_step(p) for p in range_constexpr(min(PWD, KSTEPS))]
                gpu.barrier()
                acc = [
                    [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(SV)]
                    for _ in range_constexpr(NBD)
                ]
                sw = c % 8
                for st in range_constexpr(KSTEPS):
                    if const_expr(st + 1 < KSTEPS):
                        nxt = a_load(st + 1)
                    if const_expr(st + PWD < KSTEPS):
                        wq.append(w_step(st + PWD))
                    wb = [
                        [fp8_dword_to_bf16x4(wq[st][b][j]) for j in range_constexpr(4)]
                        for b in range_constexpr(NBD)
                    ]
                    base = (st % 2) * BUF_DW
                    for s in range_constexpr(SV):
                        rr = (s * 16 + c) * CH_DW + base
                        lo = lds_load_x4(a_lds, rr + ((2 * q) ^ sw) * 4)
                        hi = lds_load_x4(a_lds, rr + ((2 * q + 1) ^ sw) * 4)
                        bw = [lo[d] for d in range_constexpr(4)] + [
                            hi[d] for d in range_constexpr(4)
                        ]
                        for j in range_constexpr(4):
                            for b in range_constexpr(NBD):
                                acc[b][s] = mfma(wb[b][j], bf16_pair(bw, j), acc[b][s])
                    if const_expr(st + 1 < KSTEPS):
                        a_store((st + 1) % 2, nxt)
                    gpu.barrier()
                words = [[None] * SV for _ in range_constexpr(NBD)]
                for b in range_constexpr(NBD):
                    ch = (nb0 + b) * 16 + q * 4
                    for s in range_constexpr(SV):
                        y = [
                            acc[b][s][i] * scs[b][i] * facs[s]
                            for i in range_constexpr(4)
                        ]
                        if const_expr(COALESCE):
                            words[b][s] = bf16x4_words(y)
                        else:
                            store_direct(s, ch, y, toks, slots, oks)
                if const_expr(COALESCE):
                    for g0 in range_constexpr(0, SV, TRG):
                        gn = min(TRG, SV - g0)
                        if const_expr(g0 > 0):
                            gpu.barrier()
                        for b in range_constexpr(NBD):
                            for s in range_constexpr(g0, g0 + gn):
                                lds_store_x2(
                                    o_lds,
                                    o_base
                                    + ((s - g0) * 16 + c) * TR_DW
                                    + b * 8
                                    + q * 2,
                                    words[b][s],
                                )
                        flush(gn, g0 * 16)

            for SV in range_constexpr(1, S_MAX + 1):
                if n_sub == SV:
                    # The tile's rows are contiguous in the intermediate.
                    n_seg = SV * 16 * SEGS
                    for i in range_constexpr(0 if KC else -(-n_seg // NTHR)):
                        idx = tid + i * NTHR
                        row = idx // SEGS
                        seg = idx % SEGS
                        v = load_x4(a_rs, ((r0 + row) * K) * 2 + seg * 16)
                        if const_expr((i + 1) * NTHR <= n_seg):
                            lds_store_x4(a_lds, row * ROW_DW + seg * 4, v)
                        else:
                            if idx < n_seg:
                                lds_store_x4(a_lds, row * ROW_DW + seg * 4, v)
                    toks, slots, facs, oks = [], [], [], []
                    for s in range_constexpr(SV):
                        r = r0 + s * 16 + c
                        sid = fx.Int32(ids[r]).bitcast(fx.Uint32)
                        toks.append(fx.Int32(sid & 0xFFFFFF))
                        slots.append(fx.Int32(sid >> 24))
                        facs.append(rws[r])
                        oks.append(s * 16 + c < real)
                    if const_expr(PER_TENSOR):
                        w_e = fx.Float32(buffer_ops.buffer_load(s_rs, e, vec_width=1))
                        scs = [[w_e] * 4] * NBD
                    else:
                        scs = [
                            Vec(
                                buffer_ops.buffer_load(
                                    s_rs, e * N + (nb0 + b) * 16 + q * 4, vec_width=4
                                )
                            )
                            for b in range_constexpr(NBD)
                        ]
                    w_sets = []
                    for p in range_constexpr(0 if KC else PWD):
                        w_sets += load_w(fx.Int32(p))
                    if const_expr(not KC):
                        gpu.barrier()

                    def block(
                        b,
                        wv,
                        SV=SV,
                        toks=toks,
                        slots=slots,
                        facs=facs,
                        oks=oks,
                        scs=scs,
                    ):
                        acc = [Vec.filled(4, 0.0, fx.Float32)] * SV
                        outs = []
                        for st in range_constexpr(KSTEPS):
                            bw = []
                            for s in range_constexpr(SV):
                                off = (s * 16 + c) * ROW_DW + st * 32 + q * 8
                                lo = lds_load_x4(a_lds, off)
                                hi = lds_load_x4(a_lds, off + 4)
                                bw.append(
                                    [lo[d] for d in range_constexpr(4)]
                                    + [hi[d] for d in range_constexpr(4)]
                                )
                            for j in range_constexpr(4):
                                wb = fp8_dword_to_bf16x4(wv[st][j])
                                for s in range_constexpr(SV):
                                    acc[s] = mfma(wb, bf16_pair(bw[s], j), acc[s])
                        ch = (nb0 + b) * 16 + q * 4
                        sc = scs[b]
                        for s in range_constexpr(SV):
                            y = [
                                acc[s][i] * sc[i] * facs[s] for i in range_constexpr(4)
                            ]
                            if const_expr(COALESCE):
                                outs.append(bf16x4_words(y))
                            else:
                                if oks[s]:
                                    if const_expr(ROWS):
                                        buffer_ops.buffer_store(
                                            bf16x4_words(y),
                                            o_rs,
                                            ((toks[s] * TOPK + slots[s]) * N + ch) * 2,
                                            offset_is_bytes=True,
                                        )
                                    else:
                                        yb = fxh._f32_to_bf16(
                                            Vec.from_elements(y, fx.Float32)
                                        )
                                        fxh.atomic_add_bf16(
                                            out_bf16 + (fx.Int64(toks[s]) * N + ch), yb
                                        )
                        return outs

                    def flush(SV=SV, row0=0):
                        if const_expr(ROWS):
                            segs = 2 * NBD  # 16-byte segments per row
                        else:
                            segs = 8 * NBD  # dwords per row: one atomic per lane
                        total = SV * 16 * segs
                        for i in range_constexpr(-(-total // 64)):
                            idx = lane + i * 64
                            row = idx // segs
                            seg = idx % segs
                            row_c = (idx < total).select(row, fx.Int32(0))
                            sid = fx.Int32(ids[r0 + row0 + row_c]).bitcast(fx.Uint32)
                            tok = fx.Int32(sid & 0xFFFFFF)
                            slot = fx.Int32(sid >> 24)
                            ok = (idx < total) & (row0 + row_c < real)
                            if const_expr(ROWS):
                                v = lds_load_x4(o_lds, o_base + row_c * TR_DW + seg * 4)
                                ch0 = nb0 * 16 + seg * 8
                                if ok:
                                    buffer_ops.buffer_store(
                                        v,
                                        o_rs,
                                        ((tok * TOPK + slot) * N + ch0) * 2,
                                        offset_is_bytes=True,
                                    )
                            else:
                                v = fx.recast_iter(fx.Int32, o_lds)[
                                    o_base + row_c * TR_DW + seg
                                ]
                                ch0 = nb0 * 16 + seg * 2
                                if ok:
                                    fxh.atomic_add_bf16(
                                        out_bf16 + (fx.Int64(tok) * N + ch0),
                                        Vec.from_elements([v], fx.Int32).bitcast(
                                            fx.BFloat16
                                        ),
                                    )

                    if const_expr(KC):
                        kc_body(SV, flush, toks, slots, facs, oks, scs)
                    tiles_out = []
                    for b in range_constexpr(0 if KC else NBD):
                        if const_expr(b + PWD < NBD):
                            w_sets += load_w(fx.Int32(b + PWD))
                        tiles_out.append(
                            block(b, w_sets[b * KSTEPS : (b + 1) * KSTEPS])
                        )
                    if const_expr(COALESCE and not KC):
                        gpu.barrier()
                        for b in range_constexpr(NBD):
                            for s in range_constexpr(SV):
                                lds_store_x2(
                                    o_lds,
                                    o_base + (s * 16 + c) * TR_DW + b * 8 + q * 2,
                                    tiles_out[b][s],
                                )
                        flush()

    @flyc.jit
    def launch(
        p_inter: fx.Pointer,
        p_w: fx.Pointer,
        p_out: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_w: fx.Pointer,
        p_live: fx.Pointer,
        p_nlive: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
        tiles: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        mid_down(
            p_inter, p_w, p_out, p_sorted_ids, p_sorted_w, p_live, p_nlive, p_w_scale, M
        ).launch(
            grid=(N // (16 * NBD * WAVES), tiles, 1),
            block=(NTHR, 1, 1),
            stream=stream,
        )

    return launch


def compile_combine(*, N, TOPK, HAS_NLT=False):
    """N = model_dim; grid (ceil(N / 2048), M), 256 threads x 8 columns."""
    assert N % 8 == 0
    COLS = 256 * 8

    @flyc.kernel
    def mid_combine(
        p_rows: fx.Pointer,
        p_loc: fx.Pointer,
        p_out: fx.Pointer,
        p_nlt: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        bx = gpu.block_idx.x
        tok = gpu.block_idx.y
        col = bx * COLS + tid * 8
        n_tok = M
        if const_expr(HAS_NLT):
            n_tok = load_s1(p_nlt, fx.Int32(0))
        if (tok < n_tok) & (col < N):
            r_rs = rsrc(fxh._as_ptr(p_rows, fx.BFloat16))
            o_rs = rsrc(fxh._as_ptr(p_out, fx.BFloat16))
            acc = [fx.Float32(0.0)] * 8
            for k in range_constexpr(TOPK):
                loc = load_s1(p_loc, tok * TOPK + k)
                w = load_x4(r_rs, ((tok * TOPK + k) * N + col) * 2)
                ok = loc >= 0
                for d in range_constexpr(4):
                    v = fx.Int32(w[d])
                    lo = (v << 16).bitcast(fx.Float32)
                    hi = (v & fx.Int32(-65536)).bitcast(fx.Float32)
                    acc[2 * d] = acc[2 * d] + ok.select(lo, fx.Float32(0.0))
                    acc[2 * d + 1] = acc[2 * d + 1] + ok.select(hi, fx.Float32(0.0))
            words = []
            for d in range_constexpr(4):
                pair = fxh._f32_to_bf16(
                    Vec.from_elements([acc[2 * d], acc[2 * d + 1]], fx.Float32)
                )
                words.append(pair.bitcast(fx.Int32)[0])
            buffer_ops.buffer_store(
                Vec.from_elements(words, fx.Int32),
                o_rs,
                (tok * N + col) * 2,
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch(
        p_rows: fx.Pointer,
        p_loc: fx.Pointer,
        p_out: fx.Pointer,
        p_nlt: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        mid_combine(p_rows, p_loc, p_out, p_nlt, M).launch(
            grid=(-(-N // COLS), M, 1), block=(256, 1, 1), stream=stream
        )

    return launch
