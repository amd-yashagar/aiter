# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 per-token MoE Gate/Up for the mid-M band.

Grid (inter_dim / (16 * WAVES), live-tile bound). A workgroup takes one live
tile ``(local expert, first sorted row, real rows)`` of core's tile map; each of
its WAVES waves owns one block of 16 gate channels with their 16 up channels,
so each live expert's W1 is read once per call. Lanes stream 16-byte chunks of
W1 straight to VGPRs one K chunk (CH steps of 64) ahead of the MFMAs. The
tile's tokens are the MFMA's B operand: unquantized bf16 hidden-state rows,
gathered by sorted token id with coalesced row loads into one LDS chunk the
waves share (sentinel rows read 0 through the descriptor bound), up to S_MAX
subtiles of 16 with one specialized body per subtile count. The epilogue
applies w1_scale and the activation in FP32 and writes the bf16 intermediate
at the tile's sorted rows ([sorted rows, inter_dim]).
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
    make_act,
    mfma,
    rsrc,
)


def compile_gateup(
    *,
    N,
    K,
    S_MAX,
    WAVES=4,
    CH=2,
    PW=2,
    PA=2,
    activation="silu",
    swiglu_limit=None,
    W_CACHE=0,
):
    """N = 2 * inter_dim (gate rows then up rows), K = model_dim. Weights are
    prefetched PW chunks ahead and token rows PA chunks ahead; prefetches past
    the last chunk read out of range (no traffic)."""
    assert PW >= 1 and PA >= 1
    assert activation in ("silu", "swiglu")
    I = N // 2
    assert I % (16 * WAVES) == 0
    NSTEP = K // 64
    assert K % 64 == 0 and NSTEP % CH == 0
    NCH = NSTEP // CH
    NTHR = 64 * WAVES
    SEGS = CH * 8  # 16-byte segments per row and chunk
    OOB = 0x7FFFFFF0
    ROW_DW = SEGS * 4 + 4  # LDS row stride in dwords, padded by 16 bytes
    LDS_DW = S_MAX * 16 * ROW_DW
    act = make_act(activation, swiglu_limit)

    @fx.struct
    class SharedStorage:
        a: fx.Array[fx.Int32, LDS_DW, 16]

    @flyc.kernel
    def mid_gateup(
        p_x: fx.Pointer,
        p_w: fx.Pointer,
        p_out: fx.Pointer,
        p_sorted_ids: fx.Pointer,
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
            blk = bx * WAVES + wave
            w_rs = rsrc(fxh._as_ptr(p_w, fx.Uint8) + fx.Int64(e) * (N * K), N * K)
            x_rs = rsrc(fxh._as_ptr(p_x, fx.BFloat16), fx.Int64(M) * (K * 2))
            ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids))
            g_base = blk * (16 * K) + c * 16 + q * 256
            u_base = g_base + I * K
            o_rs = rsrc(fxh._as_ptr(p_out, fx.BFloat16))

            for SV in range_constexpr(1, S_MAX + 1):
                if n_sub == SV:
                    # Thread tid stages segments tid + i * NTHR of the chunk's
                    # SV * 16 rows x SEGS segments.
                    n_seg = SV * 16 * SEGS
                    ROUNDS = -(-n_seg // NTHR)
                    g_off, l_off = [], []
                    for i in range_constexpr(ROUNDS):
                        idx = tid + i * NTHR
                        row = idx // SEGS
                        seg = idx % SEGS
                        row_c = (idx < n_seg).select(row, fx.Int32(0))
                        tok = ids[r0 + row_c] & 0xFFFFFF
                        # rows past the chunk (idx >= n_seg) read past M: 0
                        tok = (idx < n_seg).select(tok, M)
                        g_off.append(tok * (K * 2) + seg * 16)
                        l_off.append(row * ROW_DW + seg * 4)

                    def load_a(n, g_off=g_off, ROUNDS=ROUNDS):
                        live = n < NCH
                        return [
                            load_x4(
                                x_rs,
                                live.select(g_off[i] + n * (CH * 128), OOB),
                            )
                            for i in range_constexpr(ROUNDS)
                        ]

                    def store_a(regs, l_off=l_off, ROUNDS=ROUNDS, n_seg=n_seg):
                        for i in range_constexpr(ROUNDS):
                            if const_expr((i + 1) * NTHR <= n_seg):
                                lds_store_x4(a_lds, l_off[i], regs[i])
                            else:
                                if tid + i * NTHR < n_seg:
                                    lds_store_x4(a_lds, l_off[i], regs[i])

                    def load_w(n):
                        live = n < NCH
                        out = []
                        for st in range_constexpr(CH):
                            g = (n * CH + st) * 1024
                            gb = live.select(g_base + g, OOB)
                            ub = live.select(u_base + g, OOB)
                            out.append(load_x4(w_rs, gb, W_CACHE))
                            out.append(load_x4(w_rs, ub, W_CACHE))
                        return out

                    def compute(wv, acc, SV=SV):
                        acc = list(acc)
                        for st in range_constexpr(CH):
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
                                wg = fp8_dword_to_bf16x4(wv[2 * st][j])
                                wu = fp8_dword_to_bf16x4(wv[2 * st + 1][j])
                                for s in range_constexpr(SV):
                                    b = bf16_pair(bw[s], j)
                                    acc[s] = mfma(wg, b, acc[s])
                                    acc[SV + s] = mfma(wu, b, acc[SV + s])
                        return acc

                    zero = Vec.filled(4, 0.0, fx.Float32)
                    nw, na = 2 * CH, ROUNDS
                    store_a(load_a(fx.Int32(0)))
                    a_sets = []
                    for p in range_constexpr(1, PA):
                        a_sets += load_a(fx.Int32(p))
                    w_sets = []
                    for p in range_constexpr(PW):
                        w_sets += load_w(fx.Int32(p))
                    gpu.barrier()
                    init = [zero] * (2 * SV) + w_sets + a_sets
                    results = init
                    for iv, state in range(0, NCH, 1, init=init):
                        acc_c = [Vec(v) for v in state[: 2 * SV]]
                        w_c = [Vec(v) for v in state[2 * SV : 2 * SV + PW * nw]]
                        a_c = [Vec(v) for v in state[2 * SV + PW * nw :]]
                        n = fx.Int32(iv)
                        a_n = load_a(n + PA)
                        w_n = load_w(n + PW)
                        acc_n = compute(w_c[:nw], acc_c)
                        a_all = a_c + a_n
                        gpu.barrier()
                        store_a(a_all[:na])
                        gpu.barrier()
                        results = yield acc_n + w_c[nw:] + w_n + a_all[na:]
                    acc = [Vec(v) for v in results[: 2 * SV]]

                    ch = blk * 16 + q * 4
                    s_rs = rsrc(fxh._as_ptr(p_w_scale, fx.Float32))
                    sg = Vec(buffer_ops.buffer_load(s_rs, e * N + ch, vec_width=4))
                    su = Vec(buffer_ops.buffer_load(s_rs, e * N + I + ch, vec_width=4))
                    for s in range_constexpr(SV):
                        y = [
                            act(acc[s][i] * sg[i], acc[SV + s][i] * su[i])
                            for i in range_constexpr(4)
                        ]
                        words = bf16x4_words(y)
                        row = r0 + s * 16 + c
                        if s * 16 + c < real:
                            buffer_ops.buffer_store(
                                words, o_rs, (row * I + ch) * 2, offset_is_bytes=True
                            )

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_w: fx.Pointer,
        p_out: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_live: fx.Pointer,
        p_nlive: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
        tiles: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        mid_gateup(p_x, p_w, p_out, p_sorted_ids, p_live, p_nlive, p_w_scale, M).launch(
            grid=(I // (16 * WAVES), tiles, 1), block=(NTHR, 1, 1), stream=stream
        )

    return launch
