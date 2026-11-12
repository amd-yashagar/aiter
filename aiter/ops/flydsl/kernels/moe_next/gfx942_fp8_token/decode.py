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


def _load_dwords(rsrc, byte_off):
    """Four dwords (16 bytes) at a byte offset."""
    v = buffer_ops.buffer_load(rsrc, byte_off >> 2, vec_width=4, dtype=T.i32)
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
    assert (K // 2) % (4 * 64 * WAVES) == 0, "the x row is staged 16 bytes per lane"
    NTHR = 64 * WAVES
    STEPS = KB // (4 * WAVES)
    CH = min(CHUNK, STEPS)
    assert STEPS % CH == 0
    limit = float(swiglu_limit) if swiglu_limit else 7.0

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
        red: fx.Array[fx.Float32, 32 * WAVES, 16]
        cnt: fx.Array[fx.Int32, WAVES, 16]
        x: fx.Array[fx.Int32, K // 2, 16]

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
        gi = gpu.block_idx.x
        r = gpu.block_idx.y
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
            c = lane % 16
            q = lane // 16
            w_rsrc = _rsrc(fxh._as_ptr(p_weight, fx.Uint8) + fx.Int64(e) * (N * K))
            x_rsrc = _rsrc(fxh._as_ptr(p_input, fx.BFloat16) + fx.Int64(tok) * K)
            g_row = gi * (16 * K) + c * 16
            u_row = (I // 16 + gi) * (16 * K) + c * 16
            x_lds = fx.make_view(lds.x.ptr, fx.make_layout(K // 2, 1))
            for i in range_constexpr(K // 2 // (4 * NTHR)):
                xw = _load_dwords(x_rsrc, (fx.Int32(i * 4 * NTHR) + tid * 4) * 4)
                for d in range_constexpr(4):
                    x_lds[fx.Int32(i * 4 * NTHR + d) + tid * 4] = fx.Int32(xw[d])
            gpu.barrier()
            acc_g = Vec.filled(16, 0.0, fx.Float32)
            acc_u = Vec.filled(16, 0.0, fx.Float32)
            for ch in range_constexpr(STEPS // CH):
                wgs, wus = [], []
                for s in range_constexpr(CH):
                    kb = fx.Int32((ch * CH + s) * 4 * WAVES) + wave * 4 + q
                    wgs.append(_load_dwords(w_rsrc, g_row + kb * 256))
                    wus.append(_load_dwords(w_rsrc, u_row + kb * 256))
                for s in range_constexpr(CH):
                    kb = fx.Int32((ch * CH + s) * 4 * WAVES) + wave * 4 + q
                    xs = [x_lds[kb * 8 + d] for d in range_constexpr(8)]
                    xv = Vec.from_elements(
                        _bf16x8_to_f32(xs[:4]) + _bf16x8_to_f32(xs[4:]), fx.Float32
                    )
                    acc_g = fx.fma(
                        Vec.from_elements(_fp8x16_to_f32(wgs[s]), fx.Float32), xv, acc_g
                    )
                    acc_u = fx.fma(
                        Vec.from_elements(_fp8x16_to_f32(wus[s]), fx.Float32), xv, acc_u
                    )
            sg = _lane_reduce(_sum16(acc_g))
            su = _lane_reduce(_sum16(acc_u))
            red = fx.make_view(lds.red.ptr, fx.make_layout(128, 1))
            if q == 0:
                red[wave * 16 + c] = sg
                red[16 * WAVES + wave * 16 + c] = su
            gpu.barrier()
            if tid < 16:
                g = red[tid]
                u = red[16 * WAVES + tid]
                for w in range_constexpr(1, WAVES):
                    g = g + red[w * 16 + tid]
                    u = u + red[16 * WAVES + w * 16 + tid]
                ws = fx.recast_iter(fx.Float32, fxh._as_ptr(p_w_scale))
                ch = gi * 16 + tid
                if const_expr(PER_TENSOR):
                    g = g * ws[e]
                    u = u * ws[e]
                else:
                    g = g * ws[e * N + ch]
                    u = u * ws[e * N + I + ch]
                y = _act(g, u)
                yn = y.shuffle_xor(fx.Int32(1), fx.Int32(64))
                pair = _f32_to_bf16(Vec.from_elements([y, yn], fx.Float32))
                word = pair.bitcast(fx.Int32)[0]
                if tid % 2 == 0:
                    o_rsrc = _rsrc(fxh._as_ptr(p_output, fx.BFloat16) + fx.Int64(r) * I)
                    buffer_ops.buffer_store(word, o_rsrc, ch * 2, offset_is_bytes=True)

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
        ).launch(grid=(I // 16, M * TOPK, 1), block=(NTHR, 1, 1), stream=stream)

    return launch


def compile_decode_down(*, N, K, TOPK, PER_TENSOR=False, ROUTE_LOADS=24):
    """N = model_dim, K = inter_dim; grid (N / 64, M); expert ids are Gate/Up's
    local_ids (a -1 route adds nothing). Up to ROUTE_LOADS loads per
    lane (whole routes) are issued before any is used; a dropped route (-1) loads
    expert 0 and is selected away."""
    assert N % 64 == 0 and K % 64 == 0
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
        nb = gpu.block_idx.x
        tok = gpu.block_idx.y
        lane = tid % 64
        wave = tid // 64
        c = lane % 16
        q = lane // 16
        n = nb * 64 + wave * 16 + c
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
                    lw.append(_load_dwords(w_rsrc, row_off + kb * 256))
                    la.append(_load_dwords(a_rsrc, kb * 32))
                    la.append(_load_dwords(a_rsrc, kb * 32 + 16))
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
        ).launch(grid=(N // 64, M, 1), block=(256, 1, 1), stream=stream)

    return launch
