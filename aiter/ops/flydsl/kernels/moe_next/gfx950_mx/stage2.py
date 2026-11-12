# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 MX MoE stage 2: down GEMM, route weight and combine.

One workgroup (4 waves) owns a sorted tile of BM rows and NBW blocks of 256 output
channels (64 per wave per block). The intermediate tile (stage 1's output: contiguous
sorted rows, scales in the mfma_native order) is staged in LDS once and reused for
every channel block; W2 streams straight to VGPRs from aiter's preshuffled layout,
one block ahead. W2 is MFMA operand A (channels on M) and the intermediate operand
B, so each lane ends with four consecutive channels of one route row.

Epilogue: route weight in fp32 (unless doweight_stage1), bf16, then either 8-byte
stores of each route row into out[token * topk + slot] (reduce=True, for a separate
fp32 reduce) or pk-bf16 buffer atomics into out[token] (zeroed by the sort).
Padding rows are skipped.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import BFloat16, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.mxfp4_gemm_common import lds_typed_ptr, lds_vec_load

from . import common as cm
from .stage1 import _scale_atoms


def blocks_per_wg(model_dim, m_tiles, cu_num=256):
    """Channel blocks one workgroup loops over: the largest of 7, 4, 2, 1 that
    divides model_dim / 256 and still leaves two workgroups per CU."""
    nb = model_dim // 256
    for c in (7, 4, 2, 1):
        if nb % c == 0 and (nb // c) * m_tiles >= 2 * cu_num:
            return c
    return 1


def _build_stage2(
    *,
    model_dim,
    inter_dim,
    experts,
    tile_m,
    a_dtype,
    w_dtype,
    doweight_stage1,
    reduce,
    topk,
    blocks=1,
    nt=False,
    direct=False,
    xcd=False,
):
    D = int(model_dim)
    INTER = int(inter_dim)
    BM = int(tile_m)
    BN = 256
    NBW = int(blocks)
    assert D % (BN * NBW) == 0, f"model_dim={D} must be a multiple of {BN * NBW}"
    assert INTER % 128 == 0, f"inter_dim={INTER} must be a multiple of 128"
    assert BM in (16, 32, 64, 128)
    KS = INTER // 128
    MT = BM // 16
    a_f8 = a_dtype == "fp8"
    w_f8 = w_dtype == "fp8"
    A_ROW = INTER if a_f8 else INTER // 2
    TRB = 128 if a_f8 else 64
    NCH = TRB // 16
    rows_per_pass = 256 // NCH
    a_passes = max(1, BM // rows_per_pass)
    W_ROW = INTER if w_f8 else INTER // 2
    W_TILE_BYTES = 16 * W_ROW
    W_KSTEP = 2048 if w_f8 else 1024
    W_NDW = 8 if w_f8 else 4
    A_NDW = 8 if a_f8 else 4
    GP8 = cm.pad_up(INTER // 32, 8)
    a_lds_bytes = BM * TRB
    # The whole intermediate tile stays in LDS when it fits; else a 2-slot K ring
    # (then one channel block per workgroup).
    PRE = KS * a_lds_bytes <= 96 * 1024
    assert PRE or NBW == 1, "channel-block persistence needs the whole K in LDS"
    lds_bytes = (KS if PRE else 2) * a_lds_bytes
    # Prefetch the next block's W2 while this one computes, when registers allow.
    W_AHEAD = 4 * KS * W_NDW <= 64
    XCD = bool(xcd) and not direct

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int8, lds_bytes, 16]

    @flyc.kernel
    def mx_stage2(
        arg_out: fx.Int64,
        arg_a: fx.Int64,
        arg_as: fx.Int64,
        arg_w: fx.Int64,
        arg_ws: fx.Int64,
        arg_stids: fx.Int64,
        arg_eids: fx.Int64,
        arg_sw: fx.Int64,
        arg_nvalid: fx.Int64,
        i32_tokens: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, cm._raw(tid // fx.Int32(64))))
        ngrp = fx.Int32(gpu.block_id("x"))
        mblk = fx.Int32(gpu.block_id("y"))
        if const_expr(XCD):
            ngrp, mblk = cm.xcd_tile(ngrp, mblk, D // (BN * NBW))
        m_row = mblk * fx.Int32(BM)
        eids = cm.global_ptr(arg_eids, T.i32)
        if const_expr(direct):
            # Direct mode: one tile per route slot; non-leading slots hold -1.
            e = fx.Int32(rocdl.readfirstlane(T.i32, cm._raw(eids[mblk])))
            cond = e >= fx.Int32(0)
        else:
            cond = m_row < cm.global_ptr(arg_nvalid, T.i32)[0]
            e = fx.Int32(
                rocdl.readfirstlane(
                    T.i32, cm._raw(eids[cond.select(mblk, fx.Int32(0))])
                )
            )
        if cond:
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            lds_base = fx.Int32(fx.ptrtoint(lds.buf.ptr))
            lane16 = lane % fx.Int32(16)
            lgrp = lane // fx.Int32(16)
            a_row_in_pass = tid // fx.Int32(NCH)
            a_chunk = tid % fx.Int32(NCH)
            stage_rows = [
                (a_row_in_pass + fx.Int32(p * rows_per_pass)) % fx.Int32(BM)
                for p in range_constexpr(a_passes)
            ]

            def load_a(ks):
                return [
                    cm.global_vec_load(
                        arg_a,
                        fx.Int64(m_row + r) * fx.Int64(A_ROW)
                        + fx.Int64(ks * TRB)
                        + fx.Int64(a_chunk * fx.Int32(16)),
                        4,
                    )
                    for r in stage_rows
                ]

            def store_a(regs, slot):
                for p in range_constexpr(a_passes):
                    r = stage_rows[p]
                    sw = (a_chunk ^ (r & fx.Int32(NCH - 1))) * fx.Int32(16)
                    off = fx.Int32(slot * a_lds_bytes) + r * fx.Int32(TRB) + sw
                    fx.ptr_store(
                        regs[p],
                        lds_typed_ptr(lds_base, T.i32, align=16, byte_offset=off),
                    )

            def read_a(slot, j):
                row = fx.Int32(j * 16) + lane16
                base = fx.Int32(slot * a_lds_bytes) + row * fx.Int32(TRB)
                rmask = row & fx.Int32(NCH - 1)
                lo = Vec(
                    lds_vec_load(
                        lds_base,
                        base + ((lgrp ^ rmask) * fx.Int32(16)),
                        Vec.make_type(4, fx.Int32),
                        fx.Int32,
                        align=16,
                    )
                )
                if const_expr(a_f8):
                    hi = Vec(
                        lds_vec_load(
                            lds_base,
                            base + (((lgrp + fx.Int32(4)) ^ rmask) * fx.Int32(16)),
                            Vec.make_type(4, fx.Int32),
                            fx.Int32,
                            align=16,
                        )
                    )
                    lo = lo.shuffle(hi, list(range(8)))
                f = fx.make_rmem_tensor(A_NDW, fx.Int32)
                f.store(lo)
                return f

            def w_tile0(b):
                return (ngrp * fx.Int32(NBW) + fx.Int32(b)) * fx.Int32(
                    BN // 16
                ) + wave * fx.Int32(4)

            def load_w(b, ks):
                p0 = w_tile0(b)
                frags = []
                for t in range_constexpr(4):
                    base = (e * fx.Int32(D // 16) + p0 + fx.Int32(t)) * fx.Int32(
                        W_TILE_BYTES // 4
                    )
                    ldw = lane * fx.Int32(4) + fx.Int32(ks * W_KSTEP // 4)
                    v = cm.buffer_load16(arg_w, base, ldw, nt)
                    if const_expr(w_f8):
                        v = v.shuffle(
                            cm.buffer_load16(arg_w, base, ldw + fx.Int32(256), nt),
                            list(range(8)),
                        )
                    f = fx.make_rmem_tensor(W_NDW, fx.Int32)
                    f.store(v)
                    frags.append(f)
                return frags

            def load_w_scales(b, kp):
                p0 = w_tile0(b)
                return [
                    cm.global_vec_load(
                        arg_ws,
                        fx.Int64(
                            e * fx.Int32(D // 32) + p0 // fx.Int32(2) + fx.Int32(q)
                        )
                        * fx.Int64(GP8 * 32)
                        + fx.Int64(kp * 256)
                        + fx.Int64(lane * fx.Int32(4)),
                        1,
                    )
                    for q in range_constexpr(2)
                ]

            def load_block(b):
                return (
                    [load_w(b, ks) for ks in range(KS)],
                    [load_w_scales(b, kp) for kp in range((KS + 1) // 2)],
                )

            # Intermediate scales: one dword per 16-row tile and K pair, its row
            # parity shifted down so the MFMA opsel stays static.
            def load_a_scales(kp):
                out = []
                for j in range_constexpr(MT):
                    r0 = m_row + fx.Int32(j * 16)
                    v = cm.global_vec_load(
                        arg_as,
                        fx.Int64(r0 // fx.Int32(32)) * fx.Int64(GP8 * 32)
                        + fx.Int64(kp * 256)
                        + fx.Int64(lane * fx.Int32(4)),
                        1,
                    )
                    out.append(
                        v.shrui(((r0 // fx.Int32(16)) & fx.Int32(1)) * fx.Int32(8))
                    )
                return out

            # Route rows of this lane (one per token tile): token, slot, weight.
            stids = cm.global_ptr(arg_stids, T.i32)
            swp = cm.global_ptr(arg_sw, T.f32)
            row_tok, row_slot, row_w = [], [], []
            for j in range_constexpr(MT):
                packed = stids[m_row + fx.Int32(j * 16) + lane16]
                row_tok.append(packed & fx.Int32(0xFFFFFF))
                row_slot.append(packed.shrui(fx.Int32(24)))
                if const_expr(not doweight_stage1):
                    row_w.append(swp[m_row + fx.Int32(j * 16) + lane16])

            atoms = _scale_atoms(w_dtype, a_dtype)
            zero4 = fx.full(4, 0.0, fx.Float32)
            acc = [
                [fx.make_rmem_tensor(4, fx.Float32) for _ in range_constexpr(4)]
                for _ in range_constexpr(MT)
            ]
            out_view = fx.Tensor(
                fx.make_view(
                    cm.global_ptr(arg_out, T.bf16, 4),
                    fx.make_layout((1, 1), (1, 1)),
                )
            )
            out_buf = fx.rocdl.make_buffer_tensor(out_view, max_size=True)
            atomic = fx.make_copy_atom(fx.rocdl.BufferAtomicPkAdd(BFloat16), BFloat16)

            def mfma_step(slot, ks, wf, ws, as_):
                for j in range_constexpr(MT):
                    bf = read_a(slot, j)
                    for t in range_constexpr(4):
                        fx.gemm(
                            atoms[((ks % 2) * 2 + t % 2, (ks % 2) * 2)],
                            acc[j][t],
                            wf[t],
                            bf,
                            acc[j][t],
                            scale_a=ws[t // 2],
                            scale_b=as_[j],
                        )

            @flyc.jit
            def store_if_valid(tok, slot_id, ch, pk):
                if tok < i32_tokens:
                    if const_expr(reduce):
                        cm.global_vec_store(
                            pk,
                            arg_out,
                            (
                                fx.Int64(tok * fx.Int32(topk) + slot_id) * fx.Int64(D)
                                + fx.Int64(ch)
                            )
                            * fx.Int64(2),
                            2,
                        )
                    else:
                        for h in range_constexpr(2):
                            frag = fx.make_rmem_tensor(2, BFloat16)
                            frag.store(
                                Vec.from_elements([pk[h]], fx.Int32).bitcast(BFloat16)
                            )
                            fx.copy(
                                atomic,
                                frag,
                                out_buf[None, tok * fx.Int32(D) + ch + fx.Int32(2 * h)],
                            )

            def epilogue(b):
                ch0 = w_tile0(b) * fx.Int32(16) + lgrp * fx.Int32(4)
                for j in range_constexpr(MT):
                    for t in range_constexpr(4):
                        v = Vec(acc[j][t].load())
                        vals = [v[i] for i in range(4)]
                        if const_expr(not doweight_stage1):
                            vals = [x * row_w[j] for x in vals]
                        pk = Vec.from_elements(vals, fx.Float32).to(BFloat16)
                        store_if_valid(
                            row_tok[j],
                            row_slot[j],
                            ch0 + fx.Int32(t * 16),
                            pk.bitcast(fx.Int32),
                        )

            if const_expr(PRE):
                a_all = [load_a(ks) for ks in range(KS)]
                as_all = [load_a_scales(kp) for kp in range((KS + 1) // 2)]
                cur = load_block(0)
                for ks in range_constexpr(KS):
                    store_a(a_all[ks], ks)
                gpu.barrier()
                for b in range_constexpr(NBW):
                    nxt = None
                    if const_expr(W_AHEAD and b + 1 < NBW):
                        nxt = load_block(b + 1)
                    for j in range_constexpr(MT):
                        for t in range_constexpr(4):
                            acc[j][t].store(zero4)
                    wfs, wss = cur
                    for ks in range_constexpr(KS):
                        mfma_step(ks, ks, wfs[ks], wss[ks // 2], as_all[ks // 2])
                    epilogue(b)
                    if const_expr(b + 1 < NBW):
                        cur = nxt if const_expr(W_AHEAD) else load_block(b + 1)
            else:
                for j in range_constexpr(MT):
                    for t in range_constexpr(4):
                        acc[j][t].store(zero4)
                a_regs = load_a(0)
                w_cur = load_w(0, 0)
                ws = load_w_scales(0, 0)
                as_ = load_a_scales(0)
                store_a(a_regs, 0)
                gpu.barrier()
                for ks in range_constexpr(KS):
                    slot = ks % 2
                    if const_expr(ks + 1 < KS):
                        a_regs = load_a(ks + 1)
                        w_nxt = load_w(0, ks + 1)
                    if const_expr(ks % 2 == 1 and ks + 1 < KS):
                        ws_nxt = load_w_scales(0, (ks + 1) // 2)
                        as_nxt = load_a_scales((ks + 1) // 2)
                    mfma_step(slot, ks, w_cur, ws, as_)
                    if const_expr(ks + 1 < KS):
                        store_a(a_regs, 1 - slot)
                        w_cur = w_nxt
                        if const_expr(ks % 2 == 1):
                            ws = ws_nxt
                            as_ = as_nxt
                    gpu.barrier()
                epilogue(0)

    @flyc.jit
    def launch_mx_stage2(
        arg_out: fx.Int64,
        arg_a: fx.Int64,
        arg_as: fx.Int64,
        arg_w: fx.Int64,
        arg_ws: fx.Int64,
        arg_stids: fx.Int64,
        arg_eids: fx.Int64,
        arg_sw: fx.Int64,
        arg_nvalid: fx.Int64,
        i32_tokens: fx.Int32,
        i32_m_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        mx_stage2(
            arg_out,
            arg_a,
            arg_as,
            arg_w,
            arg_ws,
            arg_stids,
            arg_eids,
            arg_sw,
            arg_nvalid,
            i32_tokens,
        ).launch(
            grid=(D // (BN * NBW), fx.Int64(i32_m_blocks), 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_mx_stage2


@functools.lru_cache(maxsize=256)
def compile_stage2(**kw):
    return _build_stage2(**kw)


def _build_reduce(*, model_dim, topk):
    """out[t] = bf16(sum over slots of rows[t * topk + k]) in fp32, 8 columns per thread."""
    D = int(model_dim)
    TK = int(topk)
    assert D % 8 == 0
    CPT = D // 8

    @flyc.kernel
    def mx_route_reduce(arg_out: fx.Int64, arg_rows: fx.Int64, i32_tokens: fx.Int32):
        gid = fx.block_idx.x * fx.Int32(256) + fx.thread_idx.x
        t = gid // fx.Int32(CPT)
        c = gid - t * fx.Int32(CPT)
        if t < i32_tokens:
            acc = [fx.Float32(0.0) for _ in range(8)]
            for k in range_constexpr(TK):
                v = cm.global_vec_load(
                    arg_rows,
                    (
                        fx.Int64(t * fx.Int32(TK) + fx.Int32(k)) * fx.Int64(D)
                        + fx.Int64(c * fx.Int32(8))
                    )
                    * fx.Int64(2),
                    4,
                )
                b = v.bitcast(BFloat16)
                for i in range_constexpr(8):
                    acc[i] = acc[i] + fx.Float32(b[i])
            pk = Vec.from_elements(acc, fx.Float32).to(BFloat16).bitcast(fx.Int32)
            cm.global_vec_store(
                pk,
                arg_out,
                (fx.Int64(t) * fx.Int64(D) + fx.Int64(c * fx.Int32(8))) * fx.Int64(2),
                4,
            )

    @flyc.jit
    def launch_mx_route_reduce(
        arg_out: fx.Int64,
        arg_rows: fx.Int64,
        i32_tokens: fx.Int32,
        i32_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        mx_route_reduce(arg_out, arg_rows, i32_tokens).launch(
            grid=(fx.Int64(i32_blocks), 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch_mx_route_reduce


@functools.lru_cache(maxsize=64)
def compile_reduce(*, model_dim, topk):
    return _build_reduce(model_dim=model_dim, topk=topk)


def reduce_blocks(tokens, model_dim):
    return (tokens * (model_dim // 8) + 255) // 256
