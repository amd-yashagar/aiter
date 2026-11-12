# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 blockscale (per_1x128) MoE for decode, routed directly (no sort).

Stage 1 runs one workgroup per (route, gate/up column tile): it reads the route's
bf16 token row, applies aiter's per_1x128 activation quant inline, and streams the
expert's W1 rows (aiter's shuffle_weight(w1, (16, 16)) layout) with the four waves
splitting K. Each 128-wide K block is one MFMA partial folded into the FP32
accumulator with sa[k] * sb[n_block, k]. The activation output stays FP32, so
stage 2 quantizes the intermediate from FP32 as aiter's ASM 1-stage path does.

MFMA operands: lane l holds row (or token column) l % 16 and the 16 K values of
16-wide chunk base + l // 16 of each 64-wide step; both operands use the same
K permutation, so the dot products are unchanged. Only column 0 of the B operand
matters (one token); every column holds the same token.
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

SCALE_BLOCK = 128
WAVES = 4
FP8_MAX = 240.0


def _rsrc(ptr, dtype, n):
    view = fx.make_view(fx.recast_iter(dtype, fxh._as_ptr(ptr)), fx.make_layout(n, 1))
    return fx.rocdl.get_buffer_rsrc(
        fx.get_iter(fx.rocdl.make_buffer_tensor(view, max_size=False))
    )


def _sload(rsrc, off):
    v = buffer_ops.buffer_load(
        rsrc,
        rocdl.readfirstlane(T.i32, _raw(fx.Int32(off))),
        vec_width=1,
        is_scalar=True,
    )
    return fx.Int32(v)


def _quant_words(xs, inv):
    """aiter's per_1x128 quant of 4k FP32 values: med3(x * inv, -240, 240) -> fp8."""
    words = []
    for w in range_constexpr(len(xs) // 4):
        q = [
            (xs[w * 4 + j] * inv)
            .maximumf(fx.Float32(-FP8_MAX))
            .minimumf(fx.Float32(FP8_MAX))
            for j in range_constexpr(4)
        ]
        pk = rocdl.cvt_pk_fp8_f32(
            T.i32, _raw(q[0]), _raw(q[1]), _raw(fx.Int32(0)), False
        )
        pk = rocdl.cvt_pk_fp8_f32(T.i32, _raw(q[2]), _raw(q[3]), pk, True)
        words.append(fx.Int32(pk))
    return words


def _i64_pair(words):
    v = Vec.from_elements(words, fx.Int32).bitcast(fx.Int64)
    return v[0], v[1]


def _mfma(a, b, acc):
    return fx.rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [a, b, acc, 0, 0, 0])


def compile_decode_gateup(*, N, K, TOPK, GN=32, PF=2, W_CACHE=0, activation="silu"):
    """N = 2 * inter_dim (gate rows then up rows), K = model_dim. Output: FP32
    act(gate) * up, [M * TOPK, N / 2], row = token * TOPK + slot. The hidden
    states and W1 are passed as Int32 pointers (16-byte loads of bf16 / fp8).
    W_CACHE is the buffer-load cache policy of the W1 stream (each weight is read
    once per route)."""
    assert activation in ("silu", "gelu")
    I = N // 2
    KB = K // SCALE_BLOCK
    NB = N // SCALE_BLOCK
    assert K % SCALE_BLOCK == 0 and KB % WAVES == 0
    assert GN % 16 == 0 and SCALE_BLOCK % GN == 0 and I % GN == 0
    KBW = KB // WAVES
    RB = GN // 16
    RED = WAVES * 2 * RB * 16

    @fx.struct
    class SharedStorage:
        red: fx.Array[fx.Float32, RED, 16]

    def _act(g, u):
        if const_expr(activation == "gelu"):
            return g * 0.5 * (1.0 + fx.erf(g * 0.7071067811865476)) * u
        return (g * rocdl.rcp(T.f32, _raw(1.0 + fx.exp(-g)))) * u

    @flyc.kernel
    def decode_gateup(
        p_x: fx.Pointer,
        p_w: fx.Pointer,
        p_ws: fx.Pointer,
        p_ids: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        g = lane // 16
        n0 = lane % 16
        nt = gpu.block_idx.x
        r = gpu.block_idx.y
        tok = r // TOPK

        ids_rsrc = _rsrc(p_ids, fx.Int32, M * TOPK)
        e = _sload(ids_rsrc, r)
        x_rsrc = _rsrc(p_x, fx.Int32, M * (K // 2))
        w_view = fx.make_view(
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_w)) + fx.Int64(e) * (N * K // 4),
            fx.make_layout(N * K // 4, 1),
        )
        w_rsrc = fx.rocdl.get_buffer_rsrc(
            fx.get_iter(fx.rocdl.make_buffer_tensor(w_view, max_size=False))
        )
        ws_rsrc = _rsrc(p_ws, fx.Int32, 1 << 28)

        n1_base = [(h * I) // 16 + nt * RB for h in range_constexpr(2)]
        nb = [(h * I) // SCALE_BLOCK + (nt * GN) // SCALE_BLOCK for h in range(2)]
        acc = [[Vec.filled(4, 0.0, fx.Float32) for _ in range(RB)] for _ in range(2)]

        def load_kb(kb):
            """Issue every global load of K block kb (activations, weights, scales)."""
            kblk = wave * KBW + kb
            k1_0 = kblk * (SCALE_BLOCK // 16)
            xv = []
            for c in range_constexpr(2):
                for half in range_constexpr(2):
                    xv.append(
                        buffer_ops.buffer_load(
                            x_rsrc,
                            (tok * K + (k1_0 + c * 4 + g) * 16 + half * 8) // 2,
                            vec_width=4,
                            dtype=fx.Int32,
                        )
                    )
            wv = [
                [
                    [
                        buffer_ops.buffer_load(
                            w_rsrc,
                            (
                                (n1_base[h] + rb) * 16 * K
                                + (k1_0 + c * 4 + g) * 256
                                + n0 * 16
                            )
                            // 4,
                            vec_width=4,
                            dtype=fx.Int32,
                            cache_modifier=W_CACHE,
                        )
                        for c in range_constexpr(2)
                    ]
                    for rb in range_constexpr(RB)
                ]
                for h in range_constexpr(2)
            ]
            sb = [
                fx.Int32(_sload(ws_rsrc, (e * NB + nb[h]) * KB + kblk)).bitcast(
                    fx.Float32
                )
                for h in range_constexpr(2)
            ]
            return xv, wv, sb

        pending = [load_kb(kb) for kb in range_constexpr(min(PF, KBW))]
        for kb in range_constexpr(KBW):
            if const_expr(kb + PF < KBW):
                pending.append(load_kb(kb + PF))
            xv, wv, sb = pending.pop(0)
            xs = []
            for v in xv:
                bf = Vec(v).bitcast(fx.BFloat16)
                xs += [bf[j].to(fx.Float32) for j in range_constexpr(8)]
            amax = fx.Float32(1e-10)
            for x in xs:
                amax = amax.maximumf(fx.absf(x))
            for sh in range_constexpr(2):
                amax = amax.maximumf(amax.shuffle_xor(fx.Int32(16 << sh), fx.Int32(64)))
            sa = amax * fx.Float32(1.0 / FP8_MAX)
            inv = fx.Float32(1.0) / sa
            a_ops = [
                _i64_pair(_quant_words(xs[c * 16 : c * 16 + 16], inv))
                for c in range_constexpr(2)
            ]
            for h in range_constexpr(2):
                s = Vec.filled(4, sa * sb[h], fx.Float32)
                for rb in range_constexpr(RB):
                    part = Vec.filled(4, 0.0, fx.Float32)
                    for c in range_constexpr(2):
                        v = wv[h][rb][c]
                        w0, w1 = _i64_pair([Vec(v)[j] for j in range_constexpr(4)])
                        part = _mfma(w0, a_ops[c][0], part)
                        part = _mfma(w1, a_ops[c][1], part)
                    acc[h][rb] = fx.fma(Vec(part), s, acc[h][rb])

        lds = fx.SharedAllocator().allocate(SharedStorage)
        red = fx.make_view(lds.red.peek().ptr, fx.make_layout(RED, 1))
        if n0 == 0:
            for h in range_constexpr(2):
                for rb in range_constexpr(RB):
                    for i in range_constexpr(4):
                        red[((wave * 2 + h) * RB + rb) * 16 + g * 4 + i] = Vec(
                            acc[h][rb]
                        )[i]
        gpu.barrier()
        if tid < GN:
            rb = tid // 16
            row = tid % 16
            gs = fx.Float32(red[(0 * RB + rb) * 16 + row])
            us = fx.Float32(red[(1 * RB + rb) * 16 + row])
            for w in range_constexpr(1, WAVES):
                gs = gs + fx.Float32(red[((w * 2 + 0) * RB + rb) * 16 + row])
                us = us + fx.Float32(red[((w * 2 + 1) * RB + rb) * 16 + row])
            out_rsrc = _rsrc(p_out, fx.Float32, M * TOPK * I)
            buffer_ops.buffer_store(_act(gs, us), out_rsrc, r * I + nt * GN + tid)

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_w: fx.Pointer,
        p_ws: fx.Pointer,
        p_ids: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        decode_gateup(p_x, p_w, p_ws, p_ids, p_out, M).launch(
            grid=(I // GN, M * TOPK, 1), block=(WAVES * 64, 1, 1), stream=stream
        )

    return launch


def compile_decode_down(*, N, K, TOPK, RB=2, PF=2):
    """N = model_dim, K = inter_dim. One workgroup per (token, 64 * RB output
    channels); each wave owns 16 * RB channels and walks the token's TOPK routes:
    the FP32 intermediate row of the route is quantized per 128 columns with aiter's
    per_1x128 math, multiplied by the expert's W2 rows (shuffle_weight(w2, (16, 16))),
    and added with its route weight to an FP32 sum that is rounded to bf16 once.
    W2 is passed as an Int32 pointer; the output is bf16 [M, N]."""
    KB = K // SCALE_BLOCK
    NB = N // SCALE_BLOCK
    assert K % SCALE_BLOCK == 0 and SCALE_BLOCK % (16 * RB) == 0
    BN = WAVES * 16 * RB
    assert N % BN == 0
    STEPS = TOPK * KB

    @flyc.kernel
    def decode_down(
        p_a: fx.Pointer,
        p_w: fx.Pointer,
        p_ws: fx.Pointer,
        p_ids: fx.Pointer,
        p_tw: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        g = lane // 16
        n0 = lane % 16
        nt = gpu.block_idx.x
        tok = gpu.block_idx.y

        ids_rsrc = _rsrc(p_ids, fx.Int32, M * TOPK)
        tw_rsrc = _rsrc(p_tw, fx.Int32, M * TOPK)
        a_rsrc = _rsrc(p_a, fx.Int32, M * TOPK * K)
        ws_rsrc = _rsrc(p_ws, fx.Int32, 1 << 28)
        n1_base = (nt * BN + wave * 16 * RB) // 16
        nb = (nt * BN + wave * 16 * RB) // SCALE_BLOCK
        experts = [_sload(ids_rsrc, tok * TOPK + j) for j in range_constexpr(TOPK)]
        w_rsrcs = [
            fx.rocdl.get_buffer_rsrc(
                fx.get_iter(
                    fx.rocdl.make_buffer_tensor(
                        fx.make_view(
                            fx.recast_iter(fx.Int32, fxh._as_ptr(p_w))
                            + fx.Int64(e) * (N * K // 4),
                            fx.make_layout(N * K // 4, 1),
                        ),
                        max_size=False,
                    )
                )
            )
            for e in experts
        ]

        def load_step(st):
            j, kb = st // KB, st % KB
            r = tok * TOPK + j
            e = experts[j]
            k1_0 = kb * (SCALE_BLOCK // 16)
            av = [
                buffer_ops.buffer_load(
                    a_rsrc,
                    r * K + (k1_0 + c * 4 + g) * 16 + q * 4,
                    vec_width=4,
                    dtype=fx.Int32,
                )
                for c in range_constexpr(2)
                for q in range_constexpr(4)
            ]
            wv = [
                [
                    buffer_ops.buffer_load(
                        w_rsrcs[j],
                        ((n1_base + rb) * 16 * K + (k1_0 + c * 4 + g) * 256 + n0 * 16)
                        // 4,
                        vec_width=4,
                        dtype=fx.Int32,
                    )
                    for c in range_constexpr(2)
                ]
                for rb in range_constexpr(RB)
            ]
            sb = fx.Int32(_sload(ws_rsrc, (e * NB + nb) * KB + kb)).bitcast(fx.Float32)
            return av, wv, sb

        out = [Vec.filled(4, 0.0, fx.Float32) for _ in range(RB)]
        route = [Vec.filled(4, 0.0, fx.Float32) for _ in range(RB)]
        pending = [load_step(st) for st in range_constexpr(min(PF, STEPS))]
        for st in range_constexpr(STEPS):
            if const_expr(st + PF < STEPS):
                pending.append(load_step(st + PF))
            av, wv, sb = pending.pop(0)
            xs = []
            for v in av:
                f = Vec(v).bitcast(fx.Float32)
                xs += [f[i] for i in range_constexpr(4)]
            amax = fx.Float32(1e-10)
            for x in xs:
                amax = amax.maximumf(fx.absf(x))
            for sh in range_constexpr(2):
                amax = amax.maximumf(amax.shuffle_xor(fx.Int32(16 << sh), fx.Int32(64)))
            sa = amax * fx.Float32(1.0 / FP8_MAX)
            inv = fx.Float32(1.0) / sa
            a_ops = [
                _i64_pair(_quant_words(xs[c * 16 : c * 16 + 16], inv))
                for c in range_constexpr(2)
            ]
            s = Vec.filled(4, sa * sb, fx.Float32)
            for rb in range_constexpr(RB):
                part = Vec.filled(4, 0.0, fx.Float32)
                for c in range_constexpr(2):
                    w0, w1 = _i64_pair([Vec(wv[rb][c])[i] for i in range_constexpr(4)])
                    part = _mfma(w0, a_ops[c][0], part)
                    part = _mfma(w1, a_ops[c][1], part)
                route[rb] = fx.fma(Vec(part), s, route[rb])
            if const_expr(st % KB == KB - 1):
                j = st // KB
                wt = fx.Int32(_sload(tw_rsrc, tok * TOPK + j)).bitcast(fx.Float32)
                wvec = Vec.filled(4, wt, fx.Float32)
                for rb in range_constexpr(RB):
                    out[rb] = fx.fma(route[rb], wvec, out[rb])
                    route[rb] = Vec.filled(4, 0.0, fx.Float32)

        if n0 == 0:
            out_rsrc = _rsrc(p_out, fx.Int32, M * N // 2)
            for rb in range_constexpr(RB):
                bf = fxh._f32_to_bf16(Vec(out[rb]))
                words = Vec(bf).bitcast(fx.Int32)
                ch = nt * BN + wave * 16 * RB + rb * 16 + g * 4
                buffer_ops.buffer_store(
                    Vec.from_elements([words[0], words[1]], fx.Int32),
                    out_rsrc,
                    (tok * N + ch) // 2,
                )

    @flyc.jit
    def launch(
        p_a: fx.Pointer,
        p_w: fx.Pointer,
        p_ws: fx.Pointer,
        p_ids: fx.Pointer,
        p_tw: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        decode_down(p_a, p_w, p_ws, p_ids, p_tw, p_out, M).launch(
            grid=(N // BN, M, 1), block=(WAVES * 64, 1, 1), stream=stream
        )

    return launch
