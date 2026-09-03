# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MQA absorbed MLA prefill (FlyDSL) — shared latent KV, all heads in one CTA.

This is the SILOTIGER-957 schedule, not a one-head-per-CTA MHA fork.

  Flattened M = SEQ_TILE * H (default 8 tokens x 12 heads = 96)
  Grid ``(num_seq_tiles, 1, num_kv_splits)`` — H is inside the CTA
  CTA = 6 waves (M / 16); each wave owns 16 flattened (token, head) rows
  One K tile in LDS is reused across all heads (the absorb/MQA point)
  GEMM1 ``S = K @ Q^T``, online softmax, GEMM2 ``O += V @ P`` from K layout
  Shared-KV MQA: grid ``(seq_tiles, 1, splits)``, all heads in the CTA.

Contract (gfx942, Kimi TP=8):
  Q  : bf16 [total_q, H, 576]
  KV : bf16 rows of 576 (paged/dense); V = leading 512
  O  : bf16 [total_q, H, 512]

NOTE: Do NOT use ``from __future__ import annotations``.
"""

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops, vector

MFMA = 16
WARP_SIZE = 64
VEC = 4
DEFAULT_NUM_HEADS = 12
DEFAULT_SEQ_TILE = 8  # 8 * 12 = 96 = 6 * 16
DEFAULT_NUM_WARPS = (DEFAULT_SEQ_TILE * DEFAULT_NUM_HEADS) // MFMA  # 6
DEFAULT_BLOCK_N = MFMA  # 16; BN=32 was slower on gfx942 MQA
DEFAULT_WAVES_PER_EU = 2
DEFAULT_QK_DIM = 576
DEFAULT_V_DIM = 512
_LOG2E = math.log2(math.e)
_NEG_INF = -3.4028234663852886e38
SUPPORTED_GFX = ("gfx942", "gfx950")

_SPLIT_WG_PER_CU = 2
_MAX_KV_SPLITS = 64


def _validate_mqa(qk_dim, v_dim, num_heads, seq_tile, block_n, k_double_buffer):
    if qk_dim % MFMA or v_dim % MFMA or v_dim > qk_dim:
        raise ValueError(f"qk={qk_dim} v={v_dim} must be MFMA multiples, v<=qk")
    m = int(seq_tile) * int(num_heads)
    if m % MFMA:
        raise ValueError(f"SEQ_TILE*H={m} must be a multiple of {MFMA}")
    if block_n % MFMA or block_n <= 0:
        raise ValueError(f"block_n={block_n}")
    k_bufs = 2 if k_double_buffer else 1
    lds = k_bufs * block_n * qk_dim * 2
    if lds > 64 * 1024:
        raise ValueError(f"LDS {lds} B > 64 KiB")


def choose_num_kv_splits_mqa(
    *,
    num_q_tiles: int,
    max_kv_len: int,
    block_n: int = DEFAULT_BLOCK_N,
    num_cu=None,
    max_splits: int = _MAX_KV_SPLITS,
) -> int:
    """Splits from seq-tile count only (H lives inside the CTA)."""
    base = max(int(num_q_tiles), 1)
    kv = int(max_kv_len)
    bn = int(block_n)
    if kv < 1024:
        return 1
    if num_cu is None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        num_cu = int(getattr(props, "multi_processor_count", 304) or 304)
    if base >= int(num_cu):
        return 1
    target = max(int(num_cu) * _SPLIT_WG_PER_CU, base)
    splits = (target + base - 1) // base
    by_chunk = max(1, kv // 512)
    max_by_tiles = max(1, (kv + bn - 1) // bn)
    return max(1, min(int(splits), int(by_chunk), int(max_by_tiles), int(max_splits)))


def _bf16x4_from_vec(vec):
    return vector.bitcast(T.i16x4, fx.Vector(vec))


def _mfma(a, b, c):
    return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a, b, c, 0, 0, 0])


@lru_cache(maxsize=32)
def compile_mla_prefill_mqa(
    *,
    qk_dim: int = DEFAULT_QK_DIM,
    v_dim: int = DEFAULT_V_DIM,
    num_heads: int = DEFAULT_NUM_HEADS,
    seq_tile: int = DEFAULT_SEQ_TILE,
    block_n: int = DEFAULT_BLOCK_N,
    is_causal: bool = False,
    waves_per_eu: int = DEFAULT_WAVES_PER_EU,
    k_double_buffer: bool = False,
    q_preload: bool = True,
):
    _validate_mqa(qk_dim, v_dim, num_heads, seq_tile, block_n, k_double_buffer)
    QK_DIM, V_DIM = int(qk_dim), int(v_dim)
    H = int(num_heads)
    SQT = int(seq_tile)
    BN = int(block_n)
    M = SQT * H
    NW = M // MFMA
    NT = NW * WARP_SIZE
    CAUSAL = bool(is_causal)
    K_DBUF = bool(k_double_buffer)
    Q_PRELOAD = bool(q_preload)
    NUM_QK_K = QK_DIM // MFMA
    NUM_V_K = V_DIM // MFMA
    NUM_N_SUB = BN // MFMA
    LDS_K = BN * QK_DIM
    LDS_K_TOTAL = LDS_K * (2 if K_DBUF else 1)
    USE_VEC_K = LDS_K % (NT * VEC) == 0
    K_NVEC = LDS_K // (NT * VEC) if USE_VEC_K else 0
    K_PER = LDS_K // NT

    attrs = {"rocdl.waves_per_eu": int(waves_per_eu)} if waves_per_eu >= 1 else {}
    hints = {"waves_per_eu": int(waves_per_eu)} if waves_per_eu >= 1 else {}

    @fx.struct
    class SharedStorage:
        k: fx.Array[fx.BFloat16, LDS_K_TOTAL, 16]

    @flyc.kernel(known_block_size=[NT, 1, 1])
    def kn(
        q: fx.Tensor,
        kv: fx.Tensor,
        o: fx.Tensor,
        partial_o: fx.Tensor,
        partial_ml: fx.Tensor,
        tile_q_start: fx.Tensor,
        tile_batch: fx.Tensor,
        qo_indptr: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_indices: fx.Tensor,
        page_size: fx.Int32,
        num_tiles: fx.Int32,
        num_splits: fx.Int32,
        sm_scale: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        tile = fx.Int32(gpu.block_id("x"))
        split = fx.Int32(gpu.block_id("z"))
        warp = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)

        q_rsrc = buffer_ops.create_buffer_resource(q, max_size=True)
        kv_rsrc = buffer_ops.create_buffer_resource(kv, max_size=True)
        o_rsrc = buffer_ops.create_buffer_resource(o, max_size=True)
        po_rsrc = buffer_ops.create_buffer_resource(partial_o, max_size=True)
        tqs = fx.rocdl.make_buffer_tensor(tile_q_start)
        tb = fx.rocdl.make_buffer_tensor(tile_batch)
        qo = fx.rocdl.make_buffer_tensor(qo_indptr)
        kip = fx.rocdl.make_buffer_tensor(kv_indptr)
        kix = fx.rocdl.make_buffer_tensor(kv_indices)
        pml_buf = fx.rocdl.make_buffer_tensor(partial_ml)
        sm = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_k = sm.k

        ab_row = lane % fx.Int32(MFMA)
        ab_col0 = (lane // fx.Int32(MFMA)) * fx.Int32(VEC)
        c_row0, c_col = ab_col0, ab_row

        q0 = fx.Int32(tqs[tile])
        b = fx.Int32(tb[tile])
        q_seq0 = fx.Int32(qo[b])
        q_seq1 = fx.Int32(qo[b + 1])
        q_len = q_seq1 - q_seq0
        kv0 = fx.Int32(kip[b])
        kv1 = fx.Int32(kip[b + 1])
        kv_len = kv1 - kv0

        chunk = (kv_len + num_splits - fx.Int32(1)) // num_splits
        n_begin = split * chunk
        n_end_raw = n_begin + chunk
        n_end = (n_end_raw < kv_len).select(n_end_raw, kv_len)
        split_len_raw = n_end - n_begin
        split_len = (split_len_raw > fx.Int32(0)).select(split_len_raw, fx.Int32(0))

        # Flattened M: token-major then head. Wave owns 16 consecutive M rows.
        m_load = warp * fx.Int32(MFMA) + ab_row
        seq_load = q0 + m_load // fx.Int32(H)
        head_load = m_load % fx.Int32(H)
        q_base = (seq_load * fx.Int32(H) + head_load) * fx.Int32(QK_DIM)

        m_c = warp * fx.Int32(MFMA) + c_col
        seq_c = q0 + m_c // fx.Int32(H)
        head_c = m_c % fx.Int32(H)

        log2e = fx.Float32(_LOG2E)
        neg_inf = fx.Float32(_NEG_INF)
        z = fx.Float32(0.0)
        one = fx.Float32(1.0)
        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        bf0 = fx.BFloat16(0.0)
        lds_k_elems = fx.Int32(LDS_K)
        qk_i = fx.Int32(QK_DIM)

        def fetch_k(n0i):
            vecs = []
            oks = []
            if const_expr(USE_VEC_K):
                flat_first = tid * fx.Int32(K_NVEC * VEC)
                row = flat_first // qk_i
                col0 = flat_first % qk_i
                jj = n0i + row
                ok = jj < n_end
                safe_jj = ok.select(jj, fx.Int32(0))
                page = fx.Int32(kix[kv0 + safe_jj // page_size])
                phys = page * page_size + (safe_jj % page_size)
                phys = ok.select(phys, fx.Int32(0))
                for e in range_constexpr(K_NVEC):
                    col = col0 + fx.Int32(e * VEC)
                    vecs.append(
                        fx.Vector(
                            buffer_ops.buffer_load(
                                kv_rsrc, phys * qk_i + col, vec_width=VEC, dtype=T.bf16
                            )
                        )
                    )
                    oks.append(ok)
            else:
                for e in range_constexpr(K_PER):
                    flat = tid + fx.Int32(e) * fx.Int32(NT)
                    row = flat // qk_i
                    col = flat % qk_i
                    jj = n0i + row
                    ok = jj < n_end
                    safe_jj = ok.select(jj, fx.Int32(0))
                    page = fx.Int32(kix[kv0 + safe_jj // page_size])
                    phys = page * page_size + (safe_jj % page_size)
                    g = ok.select(phys * qk_i + col, fx.Int32(0))
                    vecs.append(
                        buffer_ops.buffer_load(
                            kv_rsrc, g, vec_width=1, dtype=T.bf16
                        )
                    )
                    oks.append(ok)
            return vecs, oks

        def commit_k(base, vecs, oks):
            if const_expr(USE_VEC_K):
                for e in range_constexpr(K_NVEC):
                    flat0 = (tid * fx.Int32(K_NVEC) + fx.Int32(e)) * fx.Int32(VEC)
                    row = flat0 // qk_i
                    col = flat0 % qk_i
                    vec, ok = vecs[e], oks[e]
                    idx = base + row * qk_i + col
                    lds_k[idx + 0] = ok.select(fx.BFloat16(vec[0]), bf0)
                    lds_k[idx + 1] = ok.select(fx.BFloat16(vec[1]), bf0)
                    lds_k[idx + 2] = ok.select(fx.BFloat16(vec[2]), bf0)
                    lds_k[idx + 3] = ok.select(fx.BFloat16(vec[3]), bf0)
            else:
                for e in range_constexpr(K_PER):
                    flat = tid + fx.Int32(e) * fx.Int32(NT)
                    lds_k[base + flat] = oks[e].select(fx.BFloat16(vecs[e]), bf0)

        def load_k_tile(base, n0i):
            vecs, oks = fetch_k(n0i)
            commit_k(base, vecs, oks)

        def k_a_frag(k_base_n, ki, row0):
            row = row0 + ab_row
            col = fx.Int32(ki * MFMA) + ab_col0
            idx = k_base_n + row * qk_i + col
            return vector.bitcast(
                T.i16x4,
                fx.Vector.from_elements(
                    [
                        fx.BFloat16(lds_k[idx + 0]),
                        fx.BFloat16(lds_k[idx + 1]),
                        fx.BFloat16(lds_k[idx + 2]),
                        fx.BFloat16(lds_k[idx + 3]),
                    ],
                    dtype=fx.BFloat16,
                ),
            )

        def q_b_frag(ki, q_frags):
            if const_expr(Q_PRELOAD):
                return q_frags[ki]
            return _bf16x4_from_vec(
                buffer_ops.buffer_load(
                    q_rsrc,
                    q_base + fx.Int32(ki * MFMA) + ab_col0,
                    vec_width=VEC,
                    dtype=T.bf16,
                )
            )

        def v_a_frag(k_base_n, di, row0):
            vals = []
            for v in range_constexpr(4):
                d = fx.Int32(di * MFMA) + ab_row
                kv_r = row0 + ab_col0 + fx.Int32(v)
                idx = k_base_n + kv_r * qk_i + d
                vals.append(fx.BFloat16(lds_k[idx]))
            return vector.bitcast(
                T.i16x4, fx.Vector.from_elements(vals, dtype=fx.BFloat16)
            )

        def softmax_o(n0i, k_base_n, row0, s, m_cur, l_cur, o_cur):
            scores = []
            for v in range_constexpr(4):
                sc = fx.Float32(s[v]) * sm_scale
                q_pos = seq_c - q_seq0
                kv_col = n0i + row0 + c_row0 + fx.Int32(v)
                ok = kv_col < n_end
                if const_expr(CAUSAL):
                    lim = kv_len - q_len + q_pos + fx.Int32(1)
                    ok = ok & (kv_col < lim)
                scores.append(ok.select(sc, neg_inf))

            m_tile = scores[0]
            for v in range_constexpr(3):
                m_tile = m_tile.maximumf(scores[v + 1])
            for xor_m in (16, 32):
                m_tile = m_tile.maximumf(
                    m_tile.shuffle_xor(fx.Int32(xor_m), fx.Int32(64))
                )
            m_new = m_cur.maximumf(m_tile)
            empty = m_cur <= fx.Float32(_NEG_INF * 0.5)
            alpha = empty.select(z, fmath.exp2((m_cur - m_new) * log2e))

            p_f = []
            p_sum = z
            for v in range_constexpr(4):
                pv = fmath.exp2((scores[v] - m_new) * log2e)
                pv = (scores[v] > fx.Float32(_NEG_INF * 0.5)).select(pv, z)
                p_f.append(pv)
                p_sum = p_sum + pv
            for xor_m in (16, 32):
                p_sum = p_sum + p_sum.shuffle_xor(fx.Int32(xor_m), fx.Int32(64))
            l_new = l_cur * alpha + p_sum
            p_frag = vector.bitcast(
                T.i16x4,
                fx.Vector.from_elements(
                    [p_f[v].to(fx.BFloat16) for v in range_constexpr(4)],
                    dtype=fx.BFloat16,
                ),
            )
            o_new = []
            for di in range_constexpr(NUM_V_K):
                scaled = fx.Vector.from_elements(
                    [fx.Float32(o_cur[di][v]) * alpha for v in range_constexpr(4)],
                    dtype=fx.Float32,
                )
                o_new.append(_mfma(v_a_frag(k_base_n, di, row0), p_frag, scaled))
            return m_new, l_new, o_new

        m_i = neg_inf
        l_i = z
        o_acc = [zero4 for _ in range_constexpr(NUM_V_K)]
        q_frags = []
        if const_expr(Q_PRELOAD):
            for ki in range_constexpr(NUM_QK_K):
                q_frags.append(
                    _bf16x4_from_vec(
                        buffer_ops.buffer_load(
                            q_rsrc,
                            q_base + fx.Int32(ki * MFMA) + ab_col0,
                            vec_width=VEC,
                            dtype=T.bf16,
                        )
                    )
                )

        if const_expr(K_DBUF):
            load_k_tile(fx.Int32(0), n_begin)
            fx.gpu.barrier()
            init = [m_i, l_i, fx.Int32(0)] + o_acc
            final = init
            for n0, st in range(0, split_len, BN, init=init):
                m_cur = fx.Float32(st[0])
                l_cur = fx.Float32(st[1])
                phase = fx.Int32(st[2])
                o_cur = [fx.Vector(st[3 + di]) for di in range_constexpr(NUM_V_K)]
                n0i = n_begin + fx.Int32(n0)
                cur_base = phase * lds_k_elems
                nxt_base = (fx.Int32(1) - phase) * lds_k_elems
                nxt_vecs, nxt_oks = fetch_k(n0i + fx.Int32(BN))
                m_new, l_new, o_new = m_cur, l_cur, o_cur
                for n_sub in range_constexpr(NUM_N_SUB):
                    row0 = fx.Int32(n_sub * MFMA)
                    s = zero4
                    for ki in range_constexpr(NUM_QK_K):
                        s = _mfma(
                            k_a_frag(cur_base, ki, row0), q_b_frag(ki, q_frags), s
                        )
                    m_new, l_new, o_new = softmax_o(
                        n0i, cur_base, row0, s, m_new, l_new, o_new
                    )
                commit_k(nxt_base, nxt_vecs, nxt_oks)
                fx.gpu.barrier()
                final = yield [m_new, l_new, fx.Int32(1) - phase] + o_new
            final_m = fx.Float32(final[0])
            final_l = fx.Float32(final[1])
            final_o = [fx.Vector(final[3 + di]) for di in range_constexpr(NUM_V_K)]
        else:
            init = [m_i, l_i] + o_acc
            final = init
            for n0, st in range(0, split_len, BN, init=init):
                m_cur = fx.Float32(st[0])
                l_cur = fx.Float32(st[1])
                o_cur = [fx.Vector(st[2 + di]) for di in range_constexpr(NUM_V_K)]
                n0i = n_begin + fx.Int32(n0)
                load_k_tile(fx.Int32(0), n0i)
                fx.gpu.barrier()
                m_new, l_new, o_new = m_cur, l_cur, o_cur
                for n_sub in range_constexpr(NUM_N_SUB):
                    row0 = fx.Int32(n_sub * MFMA)
                    s = zero4
                    for ki in range_constexpr(NUM_QK_K):
                        s = _mfma(
                            k_a_frag(fx.Int32(0), ki, row0), q_b_frag(ki, q_frags), s
                        )
                    m_new, l_new, o_new = softmax_o(
                        n0i, fx.Int32(0), row0, s, m_new, l_new, o_new
                    )
                fx.gpu.barrier()
                final = yield [m_new, l_new] + o_new
            final_m = fx.Float32(final[0])
            final_l = fx.Float32(final[1])
            final_o = [fx.Vector(final[2 + di]) for di in range_constexpr(NUM_V_K)]

        use_split = num_splits > fx.Int32(1)
        if use_split:
            base_ml = ((tile * num_splits + split) * fx.Int32(M) + m_c) * fx.Int32(2)
            pml_buf[base_ml + 0] = final_m
            pml_buf[base_ml + 1] = final_l
            base_o = ((tile * num_splits + split) * fx.Int32(M) + m_c) * fx.Int32(V_DIM)
            for di in range_constexpr(NUM_V_K):
                buffer_ops.buffer_store(
                    final_o[di], po_rsrc, base_o + fx.Int32(di * MFMA) + c_row0
                )
        else:
            inv = (final_l > z).select(one / final_l, z)
            o_row = (seq_c * fx.Int32(H) + head_c) * fx.Int32(V_DIM)
            for di in range_constexpr(NUM_V_K):
                bf = fx.Vector.from_elements(
                    [
                        (fx.Float32(final_o[di][v]) * inv).to(fx.BFloat16)
                        for v in range_constexpr(4)
                    ],
                    dtype=fx.BFloat16,
                )
                buffer_ops.buffer_store(
                    bf, o_rsrc, o_row + fx.Int32(di * MFMA) + c_row0
                )

    stream0 = fx.Stream(None)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        kv: fx.Tensor,
        o: fx.Tensor,
        partial_o: fx.Tensor,
        partial_ml: fx.Tensor,
        tile_q_start: fx.Tensor,
        tile_batch: fx.Tensor,
        qo_indptr: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_indices: fx.Tensor,
        page_size: fx.Int32,
        num_tiles: fx.Int32,
        num_splits: fx.Int32,
        sm_scale: fx.Float32,
        stream: fx.Stream = stream0,
    ):
        kn(
            q,
            kv,
            o,
            partial_o,
            partial_ml,
            tile_q_start,
            tile_batch,
            qo_indptr,
            kv_indptr,
            kv_indices,
            page_size,
            num_tiles,
            num_splits,
            sm_scale,
            value_attrs=attrs,
        ).launch(
            grid=(num_tiles, 1, num_splits),
            block=(NT, 1, 1),
            stream=stream,
        )

    launch.compile_hints = dict(hints)
    return launch


def _pad_seq(q, o, qo_indptr, seq_tile):
    device = q.device
    batch = int(qo_indptr.numel()) - 1
    lens = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)
    if bool((lens % seq_tile == 0).all().item()):
        return q, o, qo_indptr, [int(x) for x in lens.tolist()]
    nhead, qk, vdim = q.shape[1], q.shape[2], o.shape[2]
    qp, op, ind, real = [], [], [0], []
    for b in range(batch):
        s, e = int(qo_indptr[b]), int(qo_indptr[b + 1])
        slen = e - s
        real.append(slen)
        pad = (seq_tile - slen % seq_tile) % seq_tile
        qp.append(q[s:e])
        op.append(o[s:e])
        if pad:
            qp.append(torch.zeros(pad, nhead, qk, dtype=q.dtype, device=device))
            op.append(torch.zeros(pad, nhead, vdim, dtype=o.dtype, device=device))
        ind.append(ind[-1] + slen + pad)
    return (
        torch.cat(qp, 0),
        torch.cat(op, 0),
        torch.tensor(ind, dtype=torch.int32, device=device),
        real,
    )


def _build_seq_tiles(qo_indptr, seq_tile):
    device = qo_indptr.device
    batch = int(qo_indptr.numel()) - 1
    if batch == 1:
        s = int(qo_indptr[0].item())
        e = int(qo_indptr[1].item())
        assert (e - s) % seq_tile == 0
        if e == s:
            z = torch.zeros(1, dtype=torch.int32, device=device)
            return z, z, 0
        starts = torch.arange(s, e, seq_tile, dtype=torch.int32, device=device)
        return starts, torch.zeros_like(starts), int(starts.numel())
    starts, batches = [], []
    for b in range(batch):
        s, e = int(qo_indptr[b]), int(qo_indptr[b + 1])
        assert (e - s) % seq_tile == 0
        for q0 in range(s, e, seq_tile):
            starts.append(q0)
            batches.append(b)
    if not starts:
        z = torch.zeros(1, dtype=torch.int32, device=device)
        return z, z, 0
    return (
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(batches, dtype=torch.int32, device=device),
        len(starts),
    )


def _combine_mqa(partial_o, partial_ml, o_pad, seq_tile, num_heads):
    m = partial_ml[..., 0]
    l = partial_ml[..., 1]
    m_max = m.amax(dim=1, keepdim=True)
    valid = m > (_NEG_INF * 0.5)
    alpha = torch.where(valid, torch.exp(m - m_max), torch.zeros_like(m))
    o_sum = (partial_o * alpha.unsqueeze(-1)).sum(dim=1)
    l_sum = (l * alpha).sum(dim=1)
    inv = torch.where(l_sum > 0, 1.0 / l_sum, torch.zeros_like(l_sum))
    merged = (o_sum * inv.unsqueeze(-1)).to(o_pad.dtype)
    o_pad.copy_(merged.reshape(-1, seq_tile, num_heads, merged.shape[-1]).reshape(
        -1, num_heads, merged.shape[-1]
    ))


class MqaPrefillWorkspace:
    __slots__ = (
        "q_pad",
        "o_pad",
        "o_out",
        "kv_flat",
        "qo_pad",
        "kv_indptr",
        "kv_indices",
        "tile_q_start",
        "tile_batch",
        "partial_o",
        "partial_ml",
        "real_lens",
        "page_size",
        "num_heads",
        "seq_tile",
        "ntiles",
        "num_kv_splits",
        "sm_scale",
        "launch",
        "needs_unpad",
    )


def prepare_mla_prefill_mqa_workspace(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale=None,
    *,
    is_causal=False,
    seq_tile=DEFAULT_SEQ_TILE,
    block_n=DEFAULT_BLOCK_N,
    waves_per_eu=DEFAULT_WAVES_PER_EU,
    num_kv_splits=None,
    k_double_buffer=False,
    q_preload=True,
):
    if q.dtype != torch.bfloat16 or kv_buffer.dtype != torch.bfloat16:
        raise TypeError("bf16 required")
    total_q, num_heads, qk_dim = q.shape
    v_dim = o.shape[-1]
    if o.shape[:2] != (total_q, num_heads):
        raise ValueError("o/q shape mismatch")
    _validate_mqa(
        qk_dim, v_dim, num_heads, seq_tile, block_n, k_double_buffer
    )

    if kv_buffer.ndim == 4:
        P, page_size, nk, kd = kv_buffer.shape
        if nk != 1 or kd != qk_dim:
            raise ValueError(f"bad kv {tuple(kv_buffer.shape)}")
        kv_flat = kv_buffer.reshape(P * page_size, qk_dim).contiguous()
    elif kv_buffer.ndim == 3:
        page_size = 1
        kv_flat = kv_buffer.reshape(-1, qk_dim).contiguous()
    elif kv_buffer.ndim == 2:
        page_size = 1
        kv_flat = kv_buffer.contiguous()
    else:
        raise ValueError(f"kv ndim {kv_buffer.ndim}")

    if sm_scale is None:
        sm_scale = qk_dim**-0.5

    q = q.contiguous()
    o = o.contiguous()
    q_pad, o_pad, qo_pad, real = _pad_seq(
        q, o, qo_indptr.contiguous(), seq_tile
    )
    tqs, tba, ntiles = _build_seq_tiles(qo_pad, seq_tile)
    ws = MqaPrefillWorkspace()
    ws.o_out = o
    ws.real_lens = real
    ws.seq_tile = int(seq_tile)
    ws.needs_unpad = q_pad.data_ptr() != q.data_ptr() or any(
        (seq_tile - sl % seq_tile) % seq_tile for sl in real
    )
    if ntiles == 0:
        ws.ntiles = 0
        return ws

    max_kv = int((kv_indptr[1:] - kv_indptr[:-1]).max().item())
    if num_kv_splits is None:
        num_kv_splits = choose_num_kv_splits_mqa(
            num_q_tiles=ntiles, max_kv_len=max_kv, block_n=block_n
        )
    num_kv_splits = max(1, int(num_kv_splits))
    m = seq_tile * num_heads

    ws.q_pad = q_pad
    ws.o_pad = o_pad
    ws.kv_flat = kv_flat
    ws.qo_pad = qo_pad
    ws.kv_indptr = kv_indptr.contiguous()
    ws.kv_indices = kv_indices.contiguous()
    ws.tile_q_start = tqs
    ws.tile_batch = tba
    ws.page_size = int(page_size)
    ws.num_heads = int(num_heads)
    ws.ntiles = int(ntiles)
    ws.num_kv_splits = num_kv_splits
    ws.sm_scale = float(sm_scale)
    ws.launch = compile_mla_prefill_mqa(
        qk_dim=qk_dim,
        v_dim=v_dim,
        num_heads=num_heads,
        seq_tile=seq_tile,
        block_n=block_n,
        is_causal=is_causal,
        waves_per_eu=waves_per_eu,
        k_double_buffer=k_double_buffer,
        q_preload=q_preload,
    )
    if num_kv_splits > 1:
        ws.partial_o = torch.empty(
            ntiles, num_kv_splits, m, v_dim, dtype=torch.float32, device=q.device
        )
        ws.partial_ml = torch.empty(
            ntiles, num_kv_splits, m, 2, dtype=torch.float32, device=q.device
        )
    else:
        ws.partial_o = torch.empty(1, dtype=torch.float32, device=q.device)
        ws.partial_ml = torch.empty(1, dtype=torch.float32, device=q.device)
    return ws


def run_mla_prefill_mqa_prepared(ws, stream=None):
    if ws.ntiles == 0:
        return ws.o_out
    if stream is None:
        stream = torch.cuda.current_stream()
    ws.launch(
        ws.q_pad.view(-1),
        ws.kv_flat.view(-1),
        ws.o_pad.view(-1),
        ws.partial_o.view(-1),
        ws.partial_ml.view(-1),
        ws.tile_q_start,
        ws.tile_batch,
        ws.qo_pad,
        ws.kv_indptr,
        ws.kv_indices,
        ws.page_size,
        ws.ntiles,
        ws.num_kv_splits,
        ws.sm_scale,
        fx.Stream(stream),
    )
    if ws.num_kv_splits > 1:
        _combine_mqa(
            ws.partial_o, ws.partial_ml, ws.o_pad, ws.seq_tile, ws.num_heads
        )
    if ws.needs_unpad:
        dst = src = 0
        o = ws.o_out
        st = ws.seq_tile
        for slen in ws.real_lens:
            o[dst : dst + slen].copy_(ws.o_pad[src : src + slen])
            dst += slen
            src += slen + (st - slen % st) % st
    elif ws.o_pad.data_ptr() != ws.o_out.data_ptr():
        ws.o_out.copy_(ws.o_pad[: ws.o_out.shape[0]])
    return ws.o_out


def flydsl_mla_prefill_mqa_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale=None,
    *,
    is_causal=False,
    seq_tile=DEFAULT_SEQ_TILE,
    block_n=DEFAULT_BLOCK_N,
    waves_per_eu=DEFAULT_WAVES_PER_EU,
    num_kv_splits=None,
    k_double_buffer=False,
    stream=None,
):
    ws = prepare_mla_prefill_mqa_workspace(
        q,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        is_causal=is_causal,
        seq_tile=seq_tile,
        block_n=block_n,
        waves_per_eu=waves_per_eu,
        num_kv_splits=num_kv_splits,
        k_double_buffer=k_double_buffer,
    )
    return run_mla_prefill_mqa_prepared(ws, stream=stream)
