# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 bf16 MoE Gate/Up on core's tile map.

Grid (inter_dim / (16 * WAVES), live-tile bound). A workgroup takes one live
tile ``(local expert, first sorted row, real rows)`` of core's tile map; each of
its WAVES waves owns one block of 16 gate channels (and, gated, its 16 up
channels), so each live expert's W1 is read once per tile. Lanes stream 16-byte
chunks of W1 straight to VGPRs PW chunks (of CH K steps of 64) ahead of the
MFMAs. The tile's tokens are the MFMA's B operand: bf16 hidden-state rows,
gathered by sorted token id with coalesced row loads into one LDS chunk the
waves share (rows past the tile read 0 through the descriptor bound), up to
S_MAX subtiles of 16 with one specialized body per subtile count. The epilogue
runs in FP32 as aiter's CK stage 1 does (route weight on gate and up when
DOWEIGHT, then act(gate) * up) and writes the bf16 intermediate at the tile's
sorted rows ([sorted rows, inter_dim]), only for the tile's real rows.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import common as fxh

from .common import (
    OOB,
    bf16x4_words,
    half,
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
    I,
    K,
    S_MAX,
    WAVES=4,
    CH=2,
    PW=2,
    PA=2,
    KS=1,
    activation="silu",
    G1U1=True,
    DOWEIGHT=False,
    W_CACHE=0,
):
    """I = inter_dim, K = model_dim; W1 is [E, (2 if G1U1 else 1) * I, K] (gate
    rows, then up rows). Weights are prefetched PW chunks ahead and token rows
    PA chunks ahead; prefetches past the last chunk read out of range (no
    traffic). KS groups of WAVES waves split K, each staging its own token
    chunk, and fold their FP32 partials in LDS in group order."""
    assert PW >= 1 and PA >= 1
    assert activation in ("silu", "gelu")
    assert I % (16 * WAVES) == 0
    N = (2 if G1U1 else 1) * I
    NSTEP = K // 64
    assert K % 64 == 0 and NSTEP % CH == 0
    assert (NSTEP // CH) % KS == 0
    NCH = NSTEP // CH // KS  # chunks per K group
    NTG = 64 * WAVES  # threads per K group
    NTHR = NTG * KS
    SEGS = CH * 8  # 16-byte segments per token row and chunk
    ROW_DW = SEGS * 4 + 4  # LDS row stride in dwords, padded by 16 bytes
    BUF_DW = S_MAX * 16 * ROW_DW
    NACC = (2 if G1U1 else 1) * S_MAX
    ACC_DW = NACC * 4 * 64  # one wave's partials
    LDS_DW = max(KS * BUF_DW, WAVES * (KS - 1) * ACC_DW)
    NW = (4 if G1U1 else 2) * CH  # 16-byte weight loads per lane and chunk
    act = make_act(activation)

    @fx.struct
    class SharedStorage:
        a: fx.Array[fx.Int32, LDS_DW, 16]

    @flyc.kernel
    def bf16_gateup(
        p_x: fx.Pointer,
        p_w: fx.Pointer,
        p_out: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_w: fx.Pointer,
        p_live: fx.Pointer,
        p_nlive: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        bx = gpu.block_idx.x
        t = gpu.block_idx.y
        lane = tid % 64
        kg = tid // NTG
        cw = (tid // 64) % WAVES
        tg = tid % NTG
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
            blk = bx * WAVES + cw
            buf = kg * BUF_DW
            k0 = kg * NCH  # this group's first chunk
            w_rs = rsrc(
                fxh._as_ptr(p_w, fx.Uint8) + fx.Int64(e) * (N * K * 2), N * K * 2
            )
            x_rs = rsrc(fxh._as_ptr(p_x, fx.BFloat16), fx.Int64(M) * (K * 2))
            ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids))
            rws = fx.recast_iter(fx.Float32, fxh._as_ptr(p_sorted_w))
            g_base = blk * (32 * K) + c * 16 + q * 256
            u_base = g_base + I * K * 2
            o_rs = rsrc(fxh._as_ptr(p_out, fx.BFloat16))

            for SV in range_constexpr(1, S_MAX + 1):
                if n_sub == SV:
                    # Group thread tg stages segments tg + i * NTG of the chunk's
                    # SV * 16 rows x SEGS segments.
                    n_seg = SV * 16 * SEGS
                    ROUNDS = -(-n_seg // NTG)
                    g_off, l_off = [], []
                    for i in range_constexpr(ROUNDS):
                        idx = tg + i * NTG
                        row = idx // SEGS
                        seg = idx % SEGS
                        row_c = (idx < n_seg).select(row, fx.Int32(0))
                        tok = ids[r0 + row_c] & 0xFFFFFF
                        # rows past the chunk (idx >= n_seg) read past M: 0
                        tok = (idx < n_seg).select(tok, M)
                        g_off.append(tok * (K * 2) + seg * 16 + k0 * (CH * 128))
                        l_off.append(buf + row * ROW_DW + seg * 4)

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
                            if const_expr((i + 1) * NTG <= n_seg):
                                lds_store_x4(a_lds, l_off[i], regs[i])
                            else:
                                if tg + i * NTG < n_seg:
                                    lds_store_x4(a_lds, l_off[i], regs[i])

                    def load_w(n):
                        live = n < NCH
                        out = []
                        for st in range_constexpr(CH):
                            g = ((k0 + n) * CH + st) * 2048
                            for h in range_constexpr(2):
                                gb = live.select(g_base + g + h * 1024, OOB)
                                out.append(load_x4(w_rs, gb, W_CACHE))
                                if const_expr(G1U1):
                                    ub = live.select(u_base + g + h * 1024, OOB)
                                    out.append(load_x4(w_rs, ub, W_CACHE))
                        return out

                    def compute(wv, acc, SV=SV):
                        acc = list(acc)
                        per = 2 if G1U1 else 1
                        for st in range_constexpr(CH):
                            bw = []
                            for s in range_constexpr(SV):
                                off = buf + (s * 16 + c) * ROW_DW + st * 32 + q * 4
                                bw.append(
                                    [
                                        lds_load_x4(a_lds, off + h * 16)
                                        for h in range_constexpr(2)
                                    ]
                                )
                            for h in range_constexpr(2):
                                wg = wv[(st * 2 + h) * per]
                                for i2 in range_constexpr(2):
                                    ag = half(wg, i2)
                                    if const_expr(G1U1):
                                        au = half(wv[(st * 2 + h) * per + 1], i2)
                                    for s in range_constexpr(SV):
                                        b = half(bw[s][h], i2)
                                        acc[s] = mfma(ag, b, acc[s])
                                        if const_expr(G1U1):
                                            acc[SV + s] = mfma(au, b, acc[SV + s])
                        return acc

                    na_acc = (2 if G1U1 else 1) * SV
                    zero = Vec.filled(4, 0.0, fx.Float32)
                    na = ROUNDS
                    store_a(load_a(fx.Int32(0)))
                    a_sets = []
                    for p in range_constexpr(1, PA):
                        a_sets += load_a(fx.Int32(p))
                    w_sets = []
                    for p in range_constexpr(PW):
                        w_sets += load_w(fx.Int32(p))
                    gpu.barrier()
                    init = [zero] * na_acc + w_sets + a_sets
                    results = init
                    for iv, state in range(0, NCH, 1, init=init):
                        acc_c = [Vec(v) for v in state[:na_acc]]
                        w_c = [Vec(v) for v in state[na_acc : na_acc + PW * NW]]
                        a_c = [Vec(v) for v in state[na_acc + PW * NW :]]
                        n = fx.Int32(iv)
                        a_n = load_a(n + PA)
                        w_n = load_w(n + PW)
                        acc_n = compute(w_c[:NW], acc_c)
                        a_all = a_c + a_n
                        gpu.barrier()
                        store_a(a_all[:na])
                        gpu.barrier()
                        results = yield acc_n + w_c[NW:] + w_n + a_all[na:]
                    acc = [Vec(v) for v in results[:na_acc]]

                    def epilogue(acc, SV=SV):
                        ch = blk * 16 + q * 4
                        for s in range_constexpr(SV):
                            row = r0 + s * 16 + c
                            if const_expr(DOWEIGHT):
                                wt = rws[row]
                                gv = [acc[s][i] * wt for i in range_constexpr(4)]
                            else:
                                gv = [acc[s][i] for i in range_constexpr(4)]
                            if const_expr(G1U1):
                                if const_expr(DOWEIGHT):
                                    uv = [
                                        acc[SV + s][i] * wt for i in range_constexpr(4)
                                    ]
                                else:
                                    uv = [acc[SV + s][i] for i in range_constexpr(4)]
                                y = [act(gv[i]) * uv[i] for i in range_constexpr(4)]
                            else:
                                y = [act(gv[i]) for i in range_constexpr(4)]
                            words = bf16x4_words(y)
                            if s * 16 + c < real:
                                buffer_ops.buffer_store(
                                    words,
                                    o_rs,
                                    (row * I + ch) * 2,
                                    offset_is_bytes=True,
                                )

                    if const_expr(KS > 1):
                        red = fx.make_view(a_lds, fx.make_layout(LDS_DW, 1))
                        red_w = na_acc * 4 * 64
                        gpu.barrier()
                        if kg > 0:
                            base = ((kg - 1) * WAVES + cw) * red_w
                            for a in range_constexpr(na_acc):
                                for i in range_constexpr(4):
                                    red[base + (a * 4 + i) * 64 + lane] = acc[a][
                                        i
                                    ].bitcast(fx.Int32)
                        gpu.barrier()
                        if kg == 0:
                            fin = []
                            for a in range_constexpr(na_acc):
                                vals = [acc[a][i] for i in range_constexpr(4)]
                                for g in range_constexpr(1, KS):
                                    base = ((g - 1) * WAVES + cw) * red_w
                                    vals = [
                                        vals[i]
                                        + fx.Int32(
                                            red[base + (a * 4 + i) * 64 + lane]
                                        ).bitcast(fx.Float32)
                                        for i in range_constexpr(4)
                                    ]
                                fin.append(Vec.from_elements(vals, fx.Float32))
                            epilogue(fin)
                    else:
                        epilogue(acc)

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_w: fx.Pointer,
        p_out: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_w: fx.Pointer,
        p_live: fx.Pointer,
        p_nlive: fx.Pointer,
        M: fx.Int32,
        tiles: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        bf16_gateup(
            p_x, p_w, p_out, p_sorted_ids, p_sorted_w, p_live, p_nlive, M
        ).launch(grid=(I // (16 * WAVES), tiles, 1), block=(NTHR, 1, 1), stream=stream)

    return launch
