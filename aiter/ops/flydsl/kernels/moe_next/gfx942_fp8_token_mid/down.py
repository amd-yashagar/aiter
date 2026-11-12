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
    lds_store_x4,
    load_s1,
    load_x4,
    mfma,
    rsrc,
)


def compile_down(*, N, K, TOPK, S_MAX, NBD=4, WAVES=4, PWD=2, ROWS=True, W_CACHE=0):
    """N = model_dim, K = inter_dim. Each wave walks NBD channel blocks with W2
    prefetched PWD blocks ahead; the tile's intermediate rows are staged once
    per workgroup in LDS."""
    assert K % 64 == 0 and N % (16 * NBD * WAVES) == 0
    KSTEPS = K // 64
    NTHR = 64 * WAVES
    ROW_DW = K // 2 + 4  # LDS row stride in dwords, padded by 16 bytes
    SEGS = K // 8  # 16-byte segments per row
    LDS_DW = S_MAX * 16 * ROW_DW
    OOB = 0x7FFFFFF0

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

            for SV in range_constexpr(1, S_MAX + 1):
                if n_sub == SV:
                    # The tile's rows are contiguous in the intermediate.
                    n_seg = SV * 16 * SEGS
                    for i in range_constexpr(-(-n_seg // NTHR)):
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
                    w_sets = []
                    for p in range_constexpr(PWD):
                        w_sets += load_w(fx.Int32(p))
                    gpu.barrier()

                    def block(b, wv, SV=SV, toks=toks, slots=slots, facs=facs, oks=oks):
                        acc = [Vec.filled(4, 0.0, fx.Float32)] * SV
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
                        sc = Vec(buffer_ops.buffer_load(s_rs, e * N + ch, vec_width=4))
                        for s in range_constexpr(SV):
                            y = [
                                acc[s][i] * sc[i] * facs[s] for i in range_constexpr(4)
                            ]
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

                    init = w_sets
                    results = init
                    for iv, state in range(0, NBD, 1, init=init):
                        w_c = [Vec(v) for v in state]
                        b = fx.Int32(iv)
                        w_n = load_w(b + PWD)
                        block(b, w_c[:KSTEPS])
                        results = yield w_c[KSTEPS:] + w_n
                    _ = results

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
