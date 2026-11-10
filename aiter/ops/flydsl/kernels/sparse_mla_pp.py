# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL absorbed sparse MLA, one full-D query per wave.

Derived from hellozhuo-amd/aiter, authored by hellozhuo-amd (Zhuo Su),
commit 4d33e0e171f8c9e8a1e2388c0a3fd3ffa1e408a2 (MIT).

The PP backend keeps the per-lane row gather and V^T/P^T MFMA mapping, but
assigns each token's full D=512 to one wave instead of a D-half wave pair.
Four waves serve four independent tokens, with private 16 KiB LDS regions.
There is no partial-score exchange, duplicate softmax, or workgroup barrier.
A single gather stage keeps full-D operands in registers without spilling
with waves_per_eu=1; each iteration performs PV(t-1), QK(t), then the V phase.
The host wrapper and its explicit experimental backend name are unchanged.
"""

from __future__ import annotations

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.sparse_mla_qblock import (
    HEAD_DIM,
    NEG_INF,
    NUM_HEADS,
    RESCALE_TAU,
    VEC,
    WAVE,
    _colmajor,
    _elem,
    _exp2,
    _ext,
    _halves,
    _head_max,
    _mfma_ssa,
    _py,
    _rescale_all,
    _rowmajor,
    _to_bf16,
    _zero4,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

NWAVE = 4
TOKENS_PER_WG = 4
BLOCK_THREADS = NWAVE * WAVE
BLOCK_K = 16


@lru_cache(maxsize=8)
def compile_sparse_mla_pp(has_sink=False, setprio=True):
    """Launcher; grid = ceil(T / TOKENS_PER_WG) workgroups."""
    HAS_SINK = bool(has_sink)
    SETPRIO = bool(setprio)
    module_name = f"sparse_mla_pp_fulld_sink{int(HAS_SINK)}_prio{int(SETPRIO)}"
    BIG = 1 << 24
    CHUNKS = HEAD_DIM // VEC
    REGION = BLOCK_K * HEAD_DIM  # elements of one wave's private LDS region

    def _lds_load(view, n):
        r = fx.make_rmem_tensor(fx.make_layout(n, 1), fx.BFloat16)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy(16 * n), fx.BFloat16), view, r)
        return r.load()

    def _lds_store(vec, view, n):
        r = fx.make_rmem_tensor(fx.make_layout(n, 1), fx.BFloat16)
        r.store(vec)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy(16 * n), fx.BFloat16), r, view)

    def _swz(layout):
        # the 16-B chunk (offset bits 3..5) XORed with row & 7 (row pitch 512: bits 9..11)
        return fx.make_composed_layout(fx.static(fx.SwizzleType.get(3, 3, 6)), layout)

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def sparse_mla_pp_kernel(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        idx_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        n_tok: fx.Int32,
        n_unique: fx.Int32,
        scale: fx.Float32,
        sink_ptr: fx.Pointer,
    ):
        # ---- static tilings (full D per wave)
        n_d_tiles = _ext(fx.zipped_divide(fx.make_layout(HEAD_DIM, 1), (16,)), 1)
        qk_tiles = fx.zipped_divide(fx.make_layout(n_d_tiles, 1), (2,))
        pv_tiles = fx.zipped_divide(fx.make_layout(n_d_tiles, 1), (4,))
        QK_PAIRS = _ext(qk_tiles, 1)
        PV_GROUPS = _ext(pv_tiles, 1)
        HALVES = _ext(qk_tiles, 0)
        H4 = _ext(pv_tiles, 0)
        halves = fx.zipped_divide(fx.make_layout(VEC, 1), (H4,))
        acc_l = _colmajor(H4, PV_GROUPS)
        N_ACC = _py(fx.size(acc_l))

        tid = fx.Int32(gpu.thread_id("x"))
        tc = fx.idx2crd(tid, _colmajor(WAVE, NWAVE))
        lane = fx.get_(tc, 0).unpack()
        warp = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.get_(tc, 1).unpack()))
        # wave -> token slot
        slot = warp
        lanes = _colmajor(16, 4)
        lc = fx.idx2crd(lane, lanes)
        lane_row = fx.get_(lc, 0).unpack()
        lane_g = fx.get_(lc, 1).unpack()
        head_xor = [_py(fx.crd2idx((0, b), lanes)) for b in (1, 2)]
        block = fx.Int32(gpu.block_id("x"))
        t_idx = fx.get_scalar(fx.crd2idx((slot, block), _colmajor(TOKENS_PER_WG, BIG)))
        n_tok_i = fx.Int32(n_tok)
        live = t_idx < n_tok_i
        q_row = live.select(t_idx, fx.Int32(0))

        # ---- global views (elements)
        q_g = fx.make_view(
            fx.rocdl.make_buffer_ptr(q_ptr),
            _rowmajor(BIG, NUM_HEADS, QK_PAIRS, 4, VEC),
        )
        out_g = fx.make_view(
            fx.rocdl.make_buffer_ptr(out_ptr),
            _rowmajor(BIG, NUM_HEADS, PV_GROUPS, 4, 4, H4),
        )
        n_u = fx.Int32(n_unique)
        idx_tiles = fx.zipped_divide(_rowmajor(BIG, n_u), (1, BLOCK_K))
        n_kt = fx.get_scalar(fx.size(fx.get_(fx.get_shape(idx_tiles), [1, 1])))
        idx_lane = fx.make_view(
            fx.rocdl.make_buffer_ptr(idx_ptr), _rowmajor(BLOCK_K, 1)
        )[lane_row, None]
        idx_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        kv_g = fx.make_view(
            fx.rocdl.make_buffer_ptr(kv_ptr), _rowmajor(BIG, CHUNKS, VEC)
        )
        # lane (r, g), pair p: 16-B chunk (p, g) of row idx[r]
        g_chunks = _rowmajor(QK_PAIRS, 4)

        # ---- LDS: 4 private regions [16 rows][512 d], swizzled
        @fx.struct
        class SharedStorage:
            kv: fx.Array[fx.BFloat16, NWAVE * REGION, 16]

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        regions = _colmajor(REGION, NWAVE)
        own_base = fx.get_scalar(fx.crd2idx((0, warp), regions))
        own_ptr = fx.add_offset(storage.kv.ptr, own_base)
        tile_l = fx.make_layout((BLOCK_K, HEAD_DIM // VEC, VEC), (HEAD_DIM, VEC, 1))
        own_tile = own_ptr.view(_swz(tile_l))
        rows = fx.make_layout(BLOCK_K, HEAD_DIM)
        vt_tok = fx.flat_divide(rows, (4,))
        vt_rows = _swz(
            fx.make_layout((fx.get_shape(vt_tok), 16, 4), (fx.get_stride(vt_tok), 4, 1))
        )

        qk_scale = fx.Float32(scale) * fx.Float32(LOG2E)
        neg_inf = fx.Float32(arith.constant(NEG_INF, type=T.f32))
        zero_f = fx.Float32(arith.constant(0.0, type=T.f32))
        one_f = fx.Float32(arith.constant(1.0, type=T.f32))
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))

        q_frags = [
            _halves(q_g[q_row, lane_row, pr, lane_g, None].load(), halves)
            for pr in range_constexpr(QK_PAIRS)
        ]

        def _load_idx(kt):
            kt = (kt < n_kt).select(kt, fx.Int32(0))
            off = fx.get_scalar(fx.crd2idx(((0, 0), (q_row, kt)), idx_tiles))
            atom = fx.atom_set_value(idx_atom, "soffset", off)
            r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            fx.copy(atom, idx_lane, r)
            return fx.Int32(fx.Vector(r.load())[0])

        def _chunk(pr):
            return fx.get_scalar(fx.crd2idx((pr, lane_g), g_chunks))

        def _gather(idx_v):
            return [
                kv_g[idx_v, _chunk(pr), None].load() for pr in range_constexpr(QK_PAIRS)
            ]

        def _store_tile(frags):
            lane_chunks = _rowmajor(QK_PAIRS, 4)
            for pr in range_constexpr(QK_PAIRS):
                c = fx.get_scalar(fx.crd2idx((pr, lane_g), lane_chunks))
                _lds_store(fx.Vector(frags[pr]), own_tile[lane_row, c, None], VEC)

        def _read_vt():
            # V^T A operands of tiles (h, g): row r is d = 64 g + 4 r + h
            ops = []
            for g in range_constexpr(PV_GROUPS):
                vt_op = fx.add_offset(own_ptr, g * 64).view(vt_rows)
                rows_g = [
                    fx.Vector(_lds_load(vt_op[(e, lane_g), lane_row, None], 4))
                    for e in range_constexpr(4)
                ]
                ops += [
                    fx.Vector.from_elements(
                        [rows_g[e][h] for e in range_constexpr(4)], fx.BFloat16
                    )
                    for h in range_constexpr(H4)
                ]
            return ops

        def _pv(acc, vt, p_b):
            acc = list(acc)
            for g in range_constexpr(PV_GROUPS):
                for h in range_constexpr(H4):
                    t = _py(fx.crd2idx((h, g), acc_l))
                    acc[t] = _mfma_ssa(mma, vt[t], p_b, acc[t])
            return acc

        def _qk(frags):
            s_accs = [_zero4() for _ in range_constexpr(HALVES)]
            for pr in range_constexpr(QK_PAIRS):
                k = _halves(frags[pr], halves)
                for h in range_constexpr(HALVES):
                    s_accs[h] = _mfma_ssa(mma, k[h], q_frags[pr][h], s_accs[h])
            s_acc = s_accs[0]
            for a in s_accs[1:]:
                s_acc = fx.Vector.from_elements(
                    [_elem(s_acc, i) + _elem(a, i) for i in range_constexpr(4)],
                    fx.Float32,
                )
            return s_acc

        def _m_phase(acc, vt, p_b, frags):
            if const_expr(SETPRIO):
                fx.rocdl.s_setprio(1)
            acc = _pv(acc, vt, p_b)
            s_acc = _qk(frags)
            if const_expr(SETPRIO):
                fx.rocdl.s_setprio(0)
            return acc, s_acc

        def _softmax(s_acc, m_i, l_i, acc):
            scores = [live.select(_elem(s_acc, i), neg_inf) for i in range_constexpr(4)]
            m_w = scores[0]
            for i in range_constexpr(1, 4):
                m_w = (m_w > scores[i]).select(m_w, scores[i])
            moved = fx.Int64(
                fx.rocdl.ballot(
                    fx.Int64.ir_type, (m_w - m_i) * qk_scale > fx.Float32(RESCALE_TAU)
                )
            )
            m_n = m_i
            alpha = one_f
            if moved != fx.Int64(0):
                m_h = _head_max(m_w, head_xor)
                m_n = (m_h > m_i).select(m_h, m_i)
                alpha = (m_n == neg_inf).select(zero_f, _exp2((m_i - m_n) * qk_scale))
                acc = _rescale_all(acc, alpha)
            empty = m_n == neg_inf
            p_t = []
            for i in range_constexpr(4):
                p = empty.select(zero_f, _exp2((scores[i] - m_n) * qk_scale))
                p_t.append(live.select(p, zero_f))
            l_i = l_i * alpha + ((p_t[0] + p_t[1]) + (p_t[2] + p_t[3]))
            p_b = fx.Vector.from_elements([_to_bf16(p) for p in p_t], fx.BFloat16)
            return m_n, l_i, acc, p_b

        def _v_phase(t, s_acc, m_i, l_i, acc, frags, idx_s, prefetch):
            _store_tile(frags)
            # prefetch is static: a runtime skip would merge a gather-free path into the
            # loop's waitcnt state and drain the next tile's gather one phase early.
            new_frags = frags
            if const_expr(prefetch):
                new_frags = _gather(idx_s)
                idx_s = _load_idx(t + fx.Int32(2))
            m_i, l_i, acc, p_b = _softmax(s_acc, m_i, l_i, acc)
            vt = _read_vt()
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            return m_i, l_i, acc, p_b, vt, new_frags, idx_s

        # Full-D ownership: no peer exchange, one softmax and one gather stage.
        m_i = neg_inf
        l_i = zero_f
        acc = [_zero4() for _ in range_constexpr(N_ACC)]
        vt = [
            fx.Vector.filled(4, 0.0, fx.Float32).to(fx.BFloat16)
            for _ in range_constexpr(N_ACC)
        ]
        p_b = fx.Vector.filled(4, 0.0, fx.Float32).to(fx.BFloat16)
        bufs = _gather(_load_idx(fx.Int32(0)))
        idx_next = _load_idx(fx.Int32(1))
        loop_init = [m_i, l_i, p_b, idx_next] + acc + vt + bufs

        def _unflat(state):
            m, l, p, idx = state[:4]
            rest = state[4:]
            return (
                fx.Float32(m), fx.Float32(l), fx.Vector(p), fx.Int32(idx),
                [fx.Vector(a) for a in rest[:N_ACC]],
                [fx.Vector(a) for a in rest[N_ACC : 2 * N_ACC]],
                [fx.Vector(a) for a in rest[2 * N_ACC :]],
            )

        for kt, state in range(
            fx.Int32(0), n_kt - fx.Int32(1), fx.Int32(1), init=loop_init
        ):
            m_i, l_i, p_b, idx_next, acc, vt, bufs = _unflat(state)
            acc, s_acc = _m_phase(acc, vt, p_b, bufs)
            m_i, l_i, acc, p_b, vt, bufs, idx_next = _v_phase(
                fx.Int32(kt), s_acc, m_i, l_i, acc, bufs, idx_next, True
            )
            results = yield [m_i, l_i, p_b, idx_next] + acc + vt + bufs
        m_i, l_i, p_b, idx_next, acc, vt, bufs = _unflat(list(results))
        acc, s_acc = _m_phase(acc, vt, p_b, bufs)
        m_i, l_i, acc, p_b, vt, bufs, idx_next = _v_phase(
            n_kt - fx.Int32(1), s_acc, m_i, l_i, acc, bufs, idx_next, False
        )
        acc = _pv(acc, vt, p_b)

        for sh in head_xor:
            l_i = l_i + l_i.shuffle_xor(sh, WAVE)
        tiny = fx.Float32(1.0e-30)
        head = lane_row
        denom = l_i
        alpha_q = fx.Float32(1.0)
        if const_expr(HAS_SINK):
            sink = fx.make_view(
                fx.rocdl.make_buffer_ptr(sink_ptr), fx.make_layout(NUM_HEADS, 1)
            )[head]
            m_scaled = m_i * fx.Float32(scale)
            m_final = (sink > m_scaled).select(sink, m_scaled)
            alpha_q = _exp2((m_scaled - m_final) * fx.Float32(LOG2E))
            sink_term = _exp2((sink - m_final) * fx.Float32(LOG2E))
            denom = denom * alpha_q + sink_term
        denom = (denom > tiny).select(denom, tiny)
        inv = alpha_q / denom
        if live:
            for g in range_constexpr(PV_GROUPS):
                for i in range_constexpr(4):
                    vals = [
                        _to_bf16(_elem(acc[_py(fx.crd2idx((h, g), acc_l))], i) * inv)
                        for h in range_constexpr(H4)
                    ]
                    out_g[t_idx, head, g, lane_g, i, None].store(
                        fx.Vector.from_elements(vals, fx.BFloat16)
                    )

    @flyc.jit
    def launch_sparse_mla_pp(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        idx_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        n_tok: fx.Int32,
        n_unique: fx.Int32,
        scale: fx.Float32,
        sink_ptr: fx.Pointer,
        n_tiles: fx.Int32,
        stream: fx.Stream,
    ):
        sparse_mla_pp_kernel(
            q_ptr, kv_ptr, idx_ptr, out_ptr, n_tok, n_unique, scale, sink_ptr
        ).launch(
            grid=(fx.Int64(n_tiles), 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_sparse_mla_pp.compile_hints = {
        "waves_per_eu": 1,
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_sparse_mla_pp
