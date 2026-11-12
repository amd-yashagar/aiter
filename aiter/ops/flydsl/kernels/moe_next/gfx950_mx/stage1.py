# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 MX MoE stage 1: gate/up GEMM, activation and the intermediate's MX quant.

One workgroup (4 waves) owns one sorted tile of BM rows (one expert) and BNH
intermediate columns: wave w computes gate and up for columns w*32 .. w*32+31, so the
activation is local. Tokens are MFMA operand A (gathered token-major rows staged in
LDS, XOR-swizzled 16 B chunks), weights operand B (straight to VGPRs from aiter's
preshuffled layout, one K tile ahead). Scaled MFMA 16x16x128 with e8m0 scales:
token scales in the lane-native order (common.py), weight scales as aiter's
e8m0_shuffle (separated or 16-row gate/up interleaved rows).

Epilogue: bias, activation in fp32 (aiter's FlyDSL helpers), then the group-32 MX
quant of the intermediate straight from fp32 (or from bf16 with round_bf16), values
by sorted row ([rows, I/2] fp4 or [rows, I] fp8) and scales in the mfma_native order.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float4E2M1FN, Float8E4M3FN, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
    _activation_mul_batch,
    _inline_dpp_quad_amax,
    lds_typed_ptr,
    lds_vec_load,
)

from . import common as cm

_ELEM = {"fp4": Float4E2M1FN, "fp8": Float8E4M3FN}


def _scale_atoms(a_dtype, w_dtype):
    return {
        (oa, ob): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, _ELEM[a_dtype], _ELEM[w_dtype], opsel_a=oa, opsel_b=ob
            )
        )
        for oa in range(4)
        for ob in range(4)
    }


def _build_stage1(
    *,
    model_dim,
    inter_dim,
    experts,
    tile_m,
    a_dtype,
    w_dtype,
    out_dtype,
    act,
    interleave,
    enable_bias,
    swiglu_limit,
    situ_beta,
    situ_linear_beta,
    round_bf16,
    tile_n=128,
    ahead=2,
    read_ahead=1,
    nt=False,
    a_resident=True,
    direct_topk=0,
    experts_total=0,
    inline_quant=True,
    xcd=False,
):
    K = int(model_dim)
    INTER = int(inter_dim)
    BM = int(tile_m)
    BNH = int(tile_n)
    assert BNH in (64, 128)
    WCOLS = BNH // 4
    NT = WCOLS // 8  # weight tiles per wave: NT // 2 gate tiles, then their up tiles
    LA = int(ahead)
    ALA = min(LA, 2)  # token tiles: two register sets (tiles kt+1, kt+2) at most
    STATIC_HALF = interleave or (WCOLS // 16) % 2 == 0
    assert K % 256 == 0, f"model_dim={K} must be a multiple of 256"
    assert INTER % 32 == 0, f"inter_dim={INTER} must be a multiple of 32"
    # Column blocks past inter_dim (inter_dim % tile_n != 0): their weight tiles
    # load the last real tile and their columns are not stored.
    RAGGED = INTER % BNH != 0
    assert BM in (16, 32, 64, 128)
    KT = K // 256
    MT = BM // 16
    a_f8 = a_dtype == "fp8"
    a_scale_cap = 254 if a_f8 else 0xFF
    w_f8 = w_dtype == "fp8"
    o_f8 = out_dtype == "fp8"
    A_ROW = K if a_f8 else K // 2
    TILE_RB = 256 if a_f8 else 128
    NCH = TILE_RB // 16
    W_ROW = K if w_f8 else K // 2
    W_TILE_BYTES = 16 * W_ROW
    W_KSTEP = 2048 if w_f8 else 1024
    W_NDW = 8 if w_f8 else 4
    A_NDW = 8 if a_f8 else 4
    GP = cm.token_scale_cols(K)
    GQ = GP // 4
    GP8 = cm.pad_up(K // 32, 8)
    N_TILES = 2 * INTER // 16
    OUT_ROW = INTER if o_f8 else INTER // 2
    OGP8 = cm.pad_up(INTER // 32, 8)
    out_max = cm.max_pos(out_dtype)
    rows_per_pass = 256 // NCH
    a_passes = max(1, BM // rows_per_pass)
    a_lds_bytes = BM * TILE_RB
    acc_stride = BNH * 4 + 16
    acc_bytes = BM * acc_stride
    # Small tiles keep every K tile of the token rows in LDS: one barrier, then the
    # loop streams weights only, so nothing forces their loads to drain early.
    A_RESIDENT = a_resident and KT * a_lds_bytes <= 112 * 1024
    # Direct mode (no sort): the tile of route slot r holds the routes of r's expert
    # from r's rank on; token rows are quantized from bf16 into LDS by this kernel.
    DIRECT = int(direct_topk) > 0
    TOPK = int(direct_topk)
    INLINE = DIRECT and inline_quant
    XCD = bool(xcd) and not DIRECT
    assert not INLINE or (A_RESIDENT and BM == 16), "inline quant: BM 16, resident A"
    assert not DIRECT or BM == 16, "direct mode needs BM 16"
    NX = -(-INTER // BNH)
    ZCPW = -(-K // (8 * max(TOPK, 1) * NX))  # zero-fill 16 B chunks per workgroup
    G = K // 32
    region0 = max((KT if A_RESIDENT else 2) * a_lds_bytes, acc_bytes)
    sc_off = region0
    rl_off = sc_off + (BM * GP if INLINE else 0)
    a_dump = rl_off + BM * 4 + 32  # 16 B dump for idle lanes of the inline quant
    lds_bytes = rl_off + (BM * 4 + 48 if DIRECT else 0)
    limit = float("inf") if swiglu_limit is None else float(swiglu_limit)

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int8, lds_bytes, 16]

    def phys_tile(ct_gate, up):
        if const_expr(interleave):
            return ct_gate * fx.Int32(2) + fx.Int32(up)
        return ct_gate + fx.Int32(up * (INTER // 16))

    @flyc.jit
    def _zero16_if(cond, arg, off_i64):
        if cond:
            z = fx.Int32(0)
            cm.global_vec_store(
                fx.Vector.from_elements([z, z, z, z], fx.Int32), arg, off_i64, 4
            )

    @flyc.jit
    def _rowmap_if(cond, ur, row, tokens, arg_tw, arg_sid_out, arg_sw_out):
        if cond:
            live = ur >= fx.Int32(0)
            urc = live.select(ur, fx.Int32(0))
            tok = urc // fx.Int32(TOPK)
            slot = urc - tok * fx.Int32(TOPK)
            packed = live.select(tok | (slot << fx.Int32(24)), tokens)
            cm.global_ptr(arg_sid_out, T.i32)[row] = packed
            twv = cm.global_ptr(arg_tw, T.f32)[urc]
            cm.global_ptr(arg_sw_out, T.f32)[row] = live.select(twv, fx.Float32(0.0))

    @flyc.jit
    def _eid_if(cond, idx, val, arg_eid_out):
        if cond:
            cm.global_ptr(arg_eid_out, T.i32)[idx] = val

    def _lds_i32(lds_base, byte_off):
        return lds_typed_ptr(lds_base, T.i32, align=4, byte_offset=byte_off)

    def _lds_ld(lds_base, byte_off):
        return fx.ptr_load(_lds_i32(lds_base, byte_off), result_type=fx.Int32)

    def _direct_prologue(
        lds_base,
        tid,
        nblk,
        mblk,
        tokens,
        arg_tids,
        arg_tw,
        arg_zero,
        arg_sid_out,
        arg_eid_out,
        arg_sw_out,
    ):
        """Zero this workgroup's slice of the output, then find whether route slot
        r = mblk leads a tile of its expert (its rank among that expert's routes is
        a multiple of BM) and list the tile's routes in LDS. Repeated experts in one
        token keep the last slot, as aiter's sort does. Stores that only some lanes
        make go to a dump word, so no lane-dependent branch is needed."""
        c0 = fx.Int32(0)
        routes = tokens * fx.Int32(TOPK)
        zc = (mblk * fx.Int32(NX) + nblk) * fx.Int32(ZCPW) + tid
        zlive = (tid < fx.Int32(ZCPW)) & (zc < tokens * fx.Int32(K // 8))
        _zero16_if(zlive, arg_zero, fx.Int64(zc) * fx.Int64(16))
        tids = cm.global_ptr(arg_tids, T.i32)
        u = tid
        inb = u < routes
        uc = inb.select(u, c0)
        id_u = inb.select(tids[uc], fx.Int32(-1))
        t_u = uc // fx.Int32(TOPK)
        s_u = uc - t_u * fx.Int32(TOPK)
        dup = u < c0
        for k in range_constexpr(1, TOPK):
            sk = s_u + fx.Int32(k)
            ok = inb & (sk < fx.Int32(TOPK))
            idk = tids[t_u * fx.Int32(TOPK) + ok.select(sk, s_u)]
            dup = dup | (ok & (idk == id_u))
        valid_u = inb & (id_u >= c0) & (id_u < fx.Int32(experts_total)) & ~dup
        e_r = fx.Int32(rocdl.readfirstlane(T.i32, cm._raw(tids[mblk])))
        match = valid_u & (id_u == e_r)
        peers = fx.Uint64(rocdl.ballot(T.i64, cm._raw(match)))
        lo = fx.Uint32(peers)
        hi = fx.Uint32(peers >> fx.Uint64(32))
        below = fx.Int32(
            rocdl.mbcnt_hi(
                T.i32,
                cm._raw(hi),
                cm._raw(rocdl.mbcnt_lo(T.i32, cm._raw(lo), cm._raw(c0))),
            )
        )
        cnt = fx.Int32(fx.math.ctpop(lo)) + fx.Int32(fx.math.ctpop(hi))
        w = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        cnt_off = fx.Int32(rl_off + BM * 4)
        dump = cnt_off + fx.Int32(24)
        rl = fx.Int32(rl_off)
        fx.ptr_store(
            cnt,
            _lds_i32(lds_base, (lane == c0).select(cnt_off + w * fx.Int32(4), dump)),
        )
        fx.ptr_store(
            fx.Int32(-1),
            _lds_i32(
                lds_base, (tid < fx.Int32(BM)).select(rl + tid * fx.Int32(4), dump)
            ),
        )
        fx.ptr_store(
            valid_u.select(fx.Int32(1), c0),
            _lds_i32(lds_base, (tid == mblk).select(cnt_off + fx.Int32(16), dump)),
        )
        gpu.barrier()
        off = c0
        for i in range_constexpr(4):
            ci = _lds_ld(lds_base, cnt_off + fx.Int32(4 * i))
            off = off + (w > fx.Int32(i)).select(ci, c0)
        rank_u = off + below
        fx.ptr_store(
            rank_u,
            _lds_i32(lds_base, (tid == mblk).select(cnt_off + fx.Int32(20), dump)),
        )
        gpu.barrier()
        r_valid = _lds_ld(lds_base, cnt_off + fx.Int32(16))
        my_rank = _lds_ld(lds_base, cnt_off + fx.Int32(20))
        leader = (r_valid != c0) & ((my_rank & fx.Int32(BM - 1)) == c0)
        slot_i = rank_u - my_rank
        put = match & (slot_i >= c0) & (slot_i < fx.Int32(BM))
        fx.ptr_store(u, _lds_i32(lds_base, put.select(rl + slot_i * fx.Int32(4), dump)))
        gpu.barrier()
        # Route rows for stage 2, written once, by the first channel block.
        ur = _lds_ld(lds_base, rl + (tid & fx.Int32(BM - 1)) * fx.Int32(4))
        _rowmap_if(
            (nblk == c0) & (tid < fx.Int32(BM)),
            ur,
            mblk * fx.Int32(BM) + tid,
            tokens,
            arg_tw,
            arg_sid_out,
            arg_sw_out,
        )
        _eid_if(
            (nblk == c0) & (tid == c0),
            mblk,
            leader.select(e_r, fx.Int32(-1)),
            arg_eid_out,
        )
        return e_r, leader

    def _inline_quant(lds_base, tid, arg_hidden, hstride):
        """bf16 token rows of the tile -> MX values in the resident LDS A tiles and
        e8m0 scales in LDS (lane-native order), with aiter's RoundUp rule."""
        maxpos = cm.max_pos(a_dtype)
        for q in range_constexpr((BM * G + 255) // 256):
            p = tid + fx.Int32(q * 256)
            live_p = p < fx.Int32(BM * G)
            pc = live_p.select(p, fx.Int32(0))
            i = pc // fx.Int32(G)
            g = pc - i * fx.Int32(G)
            ur = fx.ptr_load(
                _lds_i32(lds_base, fx.Int32(rl_off) + i * fx.Int32(4)),
                result_type=fx.Int32,
            )
            urc = (ur >= fx.Int32(0)).select(ur, fx.Int32(0))
            t = urc // fx.Int32(TOPK)
            src = t * hstride + g * fx.Int32(64)
            dw = []
            for k in range_constexpr(4):
                v = cm.global_vec_load(arg_hidden, src + fx.Int32(16 * k), 4)
                dw.extend([v[j] for j in range_constexpr(4)])
            amax = cm.bf16_group_amax(dw)
            e8 = cm.e8m0_ceil(amax, maxpos, a_scale_cap)
            qs = (e8 << fx.Int32(23)).bitcast(fx.Float32)
            kt = g // fx.Int32(8)
            gi = g & fx.Int32(7)
            rmask = i & fx.Int32(NCH - 1)
            base = kt * fx.Int32(a_lds_bytes) + i * fx.Int32(TILE_RB)
            if const_expr(a_f8):
                outs = []
                for k in range_constexpr(4):
                    outs.extend(cm.quant_bf16_fp8(dw[4 * k : 4 * k + 4], qs))
                for h in range_constexpr(2):
                    c = gi * fx.Int32(2) + fx.Int32(h)
                    dst = live_p.select(
                        base + ((c ^ rmask) * fx.Int32(16)), fx.Int32(a_dump)
                    )
                    fx.ptr_store(
                        fx.Vector.from_elements(outs[4 * h : 4 * h + 4], fx.Int32),
                        lds_typed_ptr(lds_base, T.i32, align=16, byte_offset=dst),
                    )
            else:
                outs = [cm.quant_bf16_fp4(dw[4 * k : 4 * k + 4], qs) for k in range(4)]
                dst = live_p.select(
                    base + ((gi ^ rmask) * fx.Int32(16)), fx.Int32(a_dump)
                )
                fx.ptr_store(
                    fx.Vector.from_elements(outs, fx.Int32),
                    lds_typed_ptr(lds_base, T.i32, align=16, byte_offset=dst),
                )
            soff = (
                fx.Int32(sc_off)
                + i * fx.Int32(GP)
                + (g & fx.Int32(3)) * fx.Int32(GQ)
                + (g >> fx.Int32(2))
            )
            sdst = live_p.select(soff, fx.Int32(sc_off + BM * GP + BM * 4 + 28))
            fx.ptr_store(
                fx.Int8(e8), lds_typed_ptr(lds_base, T.i8, align=1, byte_offset=sdst)
            )

    @flyc.kernel
    def mx_stage1(
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        arg_aq: fx.Int64,
        arg_as: fx.Int64,
        arg_w: fx.Int64,
        arg_ws: fx.Int64,
        arg_stids: fx.Int64,
        arg_eids: fx.Int64,
        arg_nvalid: fx.Int64,
        arg_bias: fx.Int64,
        i32_tokens: fx.Int32,
        arg_tids: fx.Int64,
        arg_tw: fx.Int64,
        arg_hidden: fx.Int64,
        arg_zero: fx.Int64,
        i32_hstride: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, cm._raw(tid // fx.Int32(64))))
        nblk = fx.Int32(gpu.block_id("x"))
        mblk = fx.Int32(gpu.block_id("y"))
        if const_expr(XCD):
            nblk, mblk = cm.xcd_tile(nblk, mblk, NX)
        m_row = mblk * fx.Int32(BM)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base = fx.Int32(fx.ptrtoint(lds.buf.ptr))
        if const_expr(DIRECT):
            e, cond = _direct_prologue(
                lds_base,
                tid,
                nblk,
                mblk,
                i32_tokens,
                arg_tids,
                arg_tw,
                arg_zero,
                arg_stids,
                arg_eids,
                arg_nvalid,
            )
        else:
            nvalid = cm.global_ptr(arg_nvalid, T.i32)[0]
            cond = m_row < nvalid
            e = fx.Int32(
                rocdl.readfirstlane(
                    T.i32,
                    cm._raw(
                        cm.global_ptr(arg_eids, T.i32)[cond.select(mblk, fx.Int32(0))]
                    ),
                )
            )
        if cond:
            stids = cm.global_ptr(arg_stids, T.i32)
            lane16 = lane % fx.Int32(16)
            lgrp = lane // fx.Int32(16)

            def safe_tok(packed):
                t = packed & fx.Int32(0xFFFFFF)
                return (t < i32_tokens).select(t, fx.Int32(0))

            # Token ids of the MFMA A rows (one per token tile) and of the rows this
            # thread stages into LDS.
            def row_token(r):
                # Direct mode: the tile's route list is in LDS (-1 = padding).
                if const_expr(DIRECT):
                    u = fx.ptr_load(
                        lds_typed_ptr(
                            lds_base,
                            T.i32,
                            align=4,
                            byte_offset=fx.Int32(rl_off) + r * fx.Int32(4),
                        ),
                        result_type=fx.Int32,
                    )
                    return (u >= fx.Int32(0)).select(u, fx.Int32(0)) // fx.Int32(TOPK)
                return safe_tok(stids[m_row + r])

            mfma_tok = (
                []
                if INLINE
                else [row_token(fx.Int32(j * 16) + lane16) for j in range_constexpr(MT)]
            )
            a_row_in_pass = tid // fx.Int32(NCH)
            a_chunk = tid % fx.Int32(NCH)
            # With fewer than 256 row chunks per tile, the extra threads repeat rows
            # (identical data to identical LDS addresses).
            stage_rows = [
                (a_row_in_pass + fx.Int32(p * rows_per_pass)) % fx.Int32(BM)
                for p in range_constexpr(a_passes)
            ]
            stage_tok = [] if INLINE else [row_token(r) for r in stage_rows]

            def load_a_tile(kt):
                regs = []
                for p in range_constexpr(a_passes):
                    off = (
                        fx.Int64(stage_tok[p]) * fx.Int64(A_ROW)
                        + fx.Int64(kt * TILE_RB)
                        + fx.Int64(a_chunk * fx.Int32(16))
                    )
                    regs.append(cm.global_vec_load(arg_aq, off, 4))
                return regs

            def store_a_tile(regs, slot):
                for p in range_constexpr(a_passes):
                    r = stage_rows[p]
                    sw = (a_chunk ^ (r & fx.Int32(NCH - 1))) * fx.Int32(16)
                    off = fx.Int32(slot * a_lds_bytes) + r * fx.Int32(TILE_RB) + sw
                    ptr = lds_typed_ptr(lds_base, T.i32, align=16, byte_offset=off)
                    fx.ptr_store(regs[p], ptr)

            def read_a_frag(slot, j, ks):
                row = fx.Int32(j * 16) + lane16
                base = fx.Int32(slot * a_lds_bytes) + row * fx.Int32(TILE_RB)
                rmask = row & fx.Int32(NCH - 1)
                if const_expr(a_f8):
                    c0 = fx.Int32(ks * 8) + lgrp
                    lo = Vec(
                        lds_vec_load(
                            lds_base,
                            base + ((c0 ^ rmask) * fx.Int32(16)),
                            Vec.make_type(4, fx.Int32),
                            fx.Int32,
                            align=16,
                        )
                    )
                    hi = Vec(
                        lds_vec_load(
                            lds_base,
                            base + (((c0 + fx.Int32(4)) ^ rmask) * fx.Int32(16)),
                            Vec.make_type(4, fx.Int32),
                            fx.Int32,
                            align=16,
                        )
                    )
                    v = lo.shuffle(hi, list(range(8)))
                else:
                    c0 = fx.Int32(ks * 4) + lgrp
                    v = Vec(
                        lds_vec_load(
                            lds_base,
                            base + ((c0 ^ rmask) * fx.Int32(16)),
                            Vec.make_type(4, fx.Int32),
                            fx.Int32,
                            align=16,
                        )
                    )
                f = fx.make_rmem_tensor(A_NDW, fx.Int32)
                f.store(v)
                return f

            # Weight tiles of this wave: gate tiles then the matching up tiles. Each
            # tile's e8m0 dword holds [k step parity][16-row tile parity]; the row
            # parity is shifted down at load so the MFMA opsel stays static.
            ct0 = nblk * fx.Int32(BNH // 16) + wave * fx.Int32(WCOLS // 16)
            e_tile0 = e * fx.Int32(N_TILES)

            def clamp_ct(ct):
                if const_expr(RAGGED):
                    return fx.min(ct, fx.Int32(INTER // 16 - 1))
                return ct

            tiles = [
                phys_tile(clamp_ct(ct0 + fx.Int32(t % (NT // 2))), t // (NT // 2))
                for t in range(NT)
            ]
            w_base_dw = [(e_tile0 + t) * fx.Int32(W_TILE_BYTES // 4) for t in tiles]
            if const_expr(STATIC_HALF):
                # One dword per 32-row pair; the tile's half is a static opsel.
                if const_expr(interleave):
                    pair_of = [t % (NT // 2) for t in range(NT)]
                    half_of = [t // (NT // 2) for t in range(NT)]
                else:
                    pair_of = [t // (NT // 2) for t in range(NT)]
                    half_of = [t % (NT // 2) for t in range(NT)]
                lead = [pair_of.index(q) for q in range(max(pair_of) + 1)]
                sc_row = [(e_tile0 + tiles[t]) // fx.Int32(2) for t in lead]
                sc_shift = [None] * len(lead)
            else:
                pair_of = list(range(NT))
                half_of = [0] * NT
                sc_row = [(e_tile0 + t) // fx.Int32(2) for t in tiles]
                sc_shift = [(t & fx.Int32(1)) * fx.Int32(8) for t in tiles]

            def load_w(kt):
                frags = []
                for t in range_constexpr(NT):
                    per_ks = []
                    for ks in range_constexpr(2):
                        ldw = lane * fx.Int32(4) + fx.Int32(
                            (2 * kt + ks) * W_KSTEP // 4
                        )
                        v = cm.buffer_load16(arg_w, w_base_dw[t], ldw, nt)
                        if const_expr(w_f8):
                            hi = cm.buffer_load16(
                                arg_w, w_base_dw[t], ldw + fx.Int32(256), nt
                            )
                            v = v.shuffle(hi, list(range(8)))
                        f = fx.make_rmem_tensor(W_NDW, fx.Int32)
                        f.store(v)
                        per_ks.append(f)
                    frags.append(per_ks)
                scales = []
                for q in range_constexpr(len(sc_row)):
                    off = (
                        fx.Int64(sc_row[q]) * fx.Int64(GP8 * 32)
                        + fx.Int64(kt * 256)
                        + fx.Int64(lane * fx.Int32(4))
                    )
                    v = cm.global_vec_load(arg_ws, off, 1)
                    if const_expr(sc_shift[q] is not None):
                        v = v.shrui(sc_shift[q])
                    scales.append(v)
                return frags, scales

            def load_a_scales(kt):
                if const_expr(INLINE):
                    return [
                        fx.ptr_load(
                            lds_typed_ptr(
                                lds_base,
                                T.i32,
                                align=4,
                                byte_offset=fx.Int32(sc_off)
                                + (fx.Int32(j * 16) + lane16) * fx.Int32(GP)
                                + lgrp * fx.Int32(GQ)
                                + fx.Int32(4 * (kt // 2)),
                            ),
                            result_type=fx.Int32,
                        )
                        for j in range_constexpr(MT)
                    ]
                return [
                    cm.global_vec_load(
                        arg_as,
                        fx.Int64(mfma_tok[j]) * fx.Int64(GP)
                        + fx.Int64(lgrp * fx.Int32(GQ))
                        + fx.Int64(4 * (kt // 2)),
                        1,
                    )
                    for j in range_constexpr(MT)
                ]

            atoms = _scale_atoms(a_dtype, w_dtype)
            zero4 = fx.full(4, 0.0, fx.Float32)
            acc = [
                [fx.make_rmem_tensor(4, fx.Float32) for _ in range_constexpr(NT)]
                for _ in range_constexpr(MT)
            ]
            for j in range_constexpr(MT):
                for t in range_constexpr(NT):
                    acc[j][t].store(zero4)

            # Software pipeline: weights are issued LA tiles ahead, token tiles ALA;
            # tile kt+1's registers go to LDS at the end of tile kt.
            if const_expr(INLINE):
                _inline_quant(lds_base, tid, arg_hidden, i32_hstride)
                a_pend = {}
            elif const_expr(A_RESIDENT):
                for i in range_constexpr(KT):
                    store_a_tile(load_a_tile(i), i)
                a_pend = {}
            else:
                a_pend = {i: load_a_tile(i) for i in range(min(ALA, KT))}
            w_pend = {i: load_w(i) for i in range(min(LA, KT))}
            if const_expr(not INLINE):
                s_pend = {
                    p: load_a_scales(2 * p) for p in range((min(LA, KT) + 1) // 2)
                }
            if const_expr(not A_RESIDENT):
                store_a_tile(a_pend.pop(0), 0)
            gpu.barrier()
            if const_expr(INLINE):
                s_pend = {
                    p: load_a_scales(2 * p) for p in range((min(LA, KT) + 1) // 2)
                }
            for kt in range_constexpr(KT):
                slot = kt if A_RESIDENT else kt % 2
                nk = kt + LA
                if const_expr(not A_RESIDENT and kt + ALA < KT):
                    a_pend[kt + ALA] = load_a_tile(kt + ALA)
                if const_expr(nk < KT):
                    w_pend[nk] = load_w(nk)
                    if const_expr(nk % 2 == 0 and nk // 2 not in s_pend):
                        s_pend[nk // 2] = load_a_scales(nk)
                frags, wsc = w_pend.pop(kt)
                a_sc = s_pend[kt // 2]
                # With read_ahead, LDS operand reads run one token tile ahead of
                # their MFMAs.
                order = [(ks, j) for ks in range(2) for j in range(MT)]
                af_next = read_a_frag(slot, 0, 0) if read_ahead else None
                for n in range_constexpr(len(order)):
                    ks, j = order[n]
                    oa = (2 * kt + ks) % 4
                    if const_expr(read_ahead):
                        af = af_next
                        if const_expr(n + 1 < len(order)):
                            nks, nj = order[n + 1]
                            af_next = read_a_frag(slot, nj, nks)
                    else:
                        af = read_a_frag(slot, j, ks)
                    for t in range_constexpr(NT):
                        fx.gemm(
                            atoms[(oa, ks * 2 + half_of[t])],
                            acc[j][t],
                            af,
                            frags[t][ks],
                            acc[j][t],
                            scale_a=a_sc[j],
                            scale_b=wsc[pair_of[t]],
                        )
                if const_expr(not A_RESIDENT):
                    if const_expr(kt + 1 < KT):
                        store_a_tile(a_pend.pop(kt + 1), 1 - slot)
                    gpu.barrier()
            if const_expr(A_RESIDENT):
                gpu.barrier()

            # Epilogue 1: bias + activation in registers, f32 results to LDS.
            col_w = wave * fx.Int32(WCOLS)
            for j in range_constexpr(MT):
                for h in range_constexpr(NT // 2):
                    g = Vec(acc[j][h].load())
                    u = Vec(acc[j][h + NT // 2].load())
                    gv = [g[i] for i in range_constexpr(4)]
                    uv = [u[i] for i in range_constexpr(4)]
                    col = col_w + fx.Int32(h * 16) + lane16
                    if const_expr(enable_bias):
                        gcol = nblk * fx.Int32(BNH) + col
                        bp = cm.global_ptr(arg_bias, T.f32)
                        bg = bp[e * fx.Int32(2 * INTER) + gcol]
                        bu = bp[e * fx.Int32(2 * INTER) + fx.Int32(INTER) + gcol]
                        gv = [x + bg for x in gv]
                        uv = [x + bu for x in uv]
                    res = _activation_mul_batch(
                        gv,
                        uv,
                        act=act,
                        situ_beta=situ_beta,
                        situ_linear_beta=situ_linear_beta,
                        swiglu_limit=limit,
                    )
                    for i in range_constexpr(4):
                        row = fx.Int32(j * 16) + lgrp * fx.Int32(4) + fx.Int32(i)
                        off = row * fx.Int32(acc_stride) + col * fx.Int32(4)
                        fx.ptr_store(
                            res[i],
                            lds_typed_ptr(lds_base, T.f32, align=4, byte_offset=off),
                        )
            gpu.barrier()

            # Epilogue 2: group-32 MX quant of 8 columns per thread, quads reduce amax.
            CPR = BNH // 8
            RPP = 256 // CPR
            prow = tid // fx.Int32(CPR)
            chunk = tid % fx.Int32(CPR)
            sp = cm.global_ptr(arg_out_scale, fx.Int8.ir_type, 1)
            for p in range_constexpr((BM + RPP - 1) // RPP):
                r = prow + fx.Int32(p * RPP)
                off = (r % fx.Int32(BM)) * fx.Int32(acc_stride) + chunk * fx.Int32(32)
                v0 = Vec(
                    lds_vec_load(
                        lds_base,
                        off,
                        Vec.make_type(4, fx.Float32),
                        fx.Float32,
                        align=16,
                    )
                )
                v1 = Vec(
                    lds_vec_load(
                        lds_base,
                        off + fx.Int32(16),
                        Vec.make_type(4, fx.Float32),
                        fx.Float32,
                        align=16,
                    )
                )
                vals = [v0[i] for i in range_constexpr(4)] + [
                    v1[i] for i in range_constexpr(4)
                ]
                if const_expr(round_bf16):
                    vals = [x.to(fx.BFloat16).to(fx.Float32) for x in vals]
                lmax = fx.absf(vals[0])
                for i in range_constexpr(1, 8):
                    lmax = lmax.maximumf(fx.absf(vals[i]))
                lmax = _inline_dpp_quad_amax(lmax.bitcast(fx.Int32)).bitcast(fx.Float32)
                e8 = cm.e8m0_ceil(lmax, out_max)
                qs = (e8 << fx.Int32(23)).bitcast(fx.Float32)
                srow = m_row + r
                gcol = nblk * fx.Int32(BNH) + chunk * fx.Int32(8)
                if (r < fx.Int32(BM)) & (gcol < fx.Int32(INTER)):
                    if const_expr(o_f8):
                        dws = cm.quant_f32_fp8(vals, qs)
                        cm.global_vec_store(
                            Vec.from_elements(dws, fx.Int32),
                            arg_out,
                            fx.Int64(srow) * fx.Int64(OUT_ROW) + fx.Int64(gcol),
                            2,
                        )
                    else:
                        dw = cm.quant_f32_fp4(vals, qs)
                        cm.global_vec_store(
                            dw,
                            arg_out,
                            fx.Int64(srow) * fx.Int64(OUT_ROW)
                            + fx.Int64(gcol // fx.Int32(2)),
                            1,
                        )
                    if (chunk & fx.Int32(3)) == fx.Int32(0):
                        grp = gcol // fx.Int32(32)
                        soff = (
                            (srow // fx.Int32(32)) * fx.Int32(OGP8 * 32)
                            + (grp // fx.Int32(8)) * fx.Int32(256)
                            + (grp & fx.Int32(3)) * fx.Int32(64)
                            + (srow & fx.Int32(15)) * fx.Int32(4)
                            + ((grp & fx.Int32(7)) // fx.Int32(4)) * fx.Int32(2)
                            + ((srow & fx.Int32(31)) // fx.Int32(16))
                        )
                        sp[soff] = fx.Int8(e8)

    @flyc.jit
    def launch_mx_stage1(
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        arg_aq: fx.Int64,
        arg_as: fx.Int64,
        arg_w: fx.Int64,
        arg_ws: fx.Int64,
        arg_stids: fx.Int64,
        arg_eids: fx.Int64,
        arg_nvalid: fx.Int64,
        arg_bias: fx.Int64,
        i32_tokens: fx.Int32,
        i32_m_blocks: fx.Int32,
        arg_tids: fx.Int64,
        arg_tw: fx.Int64,
        arg_hidden: fx.Int64,
        arg_zero: fx.Int64,
        i32_hstride: fx.Int32,
        stream: fx.Stream,
    ):
        mx_stage1(
            arg_out,
            arg_out_scale,
            arg_aq,
            arg_as,
            arg_w,
            arg_ws,
            arg_stids,
            arg_eids,
            arg_nvalid,
            arg_bias,
            i32_tokens,
            arg_tids,
            arg_tw,
            arg_hidden,
            arg_zero,
            i32_hstride,
        ).launch(
            grid=(NX, fx.Int64(i32_m_blocks), 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_mx_stage1


@functools.lru_cache(maxsize=256)
def compile_stage1(**kw):
    return _build_stage1(**kw)
