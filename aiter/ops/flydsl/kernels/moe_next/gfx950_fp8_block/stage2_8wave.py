# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Portions Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 fp8 blockscale (per_1x128) MoE prefill stage 2 on eight MFMA waves.

Same contract as stage2.py with combine atomic_bf16 (down GEMM, route weight, pk-bf16
atomics into out[token], padding rows skipped), on the pipeline of stage1_8wave.py.
A workgroup takes the 128 sorted rows of one tile against BLOCKS_PER_WG consecutive
128-channel blocks of W2. The tile's A ([128, I] fp8, I = 256) is DMA'd into LDS once
and stays there; each step streams one channel block of the shuffle_weight(16, 16) W2
(both K tiles, 32 KiB) into an LDS ping-pong buffer by raw buffer->LDS DMA. Per 128-K
group g: P = W2[:, g] . A[:, g]^T into a fresh partial, folded one MFMA later as
acc += P * (inter_scale[row, g] * w2_scale[block, g]); then out += bf16(acc * w[row]).

Waves 0-7 run the MFMAs on a (4, 2) wave grid, 4-7 one barrier interval behind 0-3.
They write each finished 64-row slice, weighted and rounded to bf16, to an LDS staging
tile; waves 8-11 read it and issue the atomics. The atomics stay off the MFMA waves,
whose DMA waits count every outstanding vector memory operation. The store waves keep
in step through the same s_barrier sequence.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

_NO_PACKED_F32 = {"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]}

BLOCK_M = 128  # sorted rows per tile
BLOCK_N = 128  # output channels per step
BLOCK_K = 128
M_SLICES = 2
PHASE_M_REP = BLOCK_M // 2 // 16 // M_SLICES
N_REP = BLOCK_N // 64
MMA_WAVES = 8
STORE_WAVES = 4
THREADS = 64 * (MMA_WAVES + STORE_WAVES)
# Padded A LDS (stage1_8wave.py): 16 bytes after each eight rows, 32 more after sixteen.
A_GROUP = 8 * BLOCK_K + 16
LDS_TILE = (BLOCK_M // 16) * (2 * A_GROUP + 32)
A_CHUNK = 4 * (2 * A_GROUP + 32)
B_TILE = BLOCK_N * BLOCK_K
B_TILE_K_BYTES = 16 * BLOCK_K  # one K tile of a 16-row shuffle_weight group
# Staging row: 128 bf16 channels + 16 bytes, so the 16 rows of an MFMA store spread
# over all banks. Dword 16 * wave_n + 4 * grp + 2 * n0 + h holds the channel pair
# n0 * 32 + wave_n * 8 + grp * 2 + h of the C fragment: one lane's four dwords of a row.
C_ROW_DWORDS = BLOCK_N // 2 + 4
STAGES = 2
NUM_XCDS = 8
DMA_PER_STEP = 4  # per MFMA wave: 2 K-tile-0 + 2 K-tile-1 W2 buffer->LDS loads
SLICE_ROWS = BLOCK_M // M_SLICES
ROWS_PER_STORE_WAVE = SLICE_ROWS // STORE_WAVES


def _lds_byte_ptr(ptr, byte_offset):
    return fx.to_llvm_ptr(
        fx.add_offset(fx.recast_iter(fx.Uint8, ptr), fx.make_int_tuple(byte_offset))
    )


def _view_i32(t):
    return fx.make_view(
        fx.recast_iter(fx.Int32, fx.get_iter(t)), fx.make_layout(1 << 30, 1)
    )


def _rsrc(t, num_bytes):
    return rocdl.get_buffer_rsrc(
        fx.get_iter(
            rocdl.make_buffer_tensor(t, max_size=False, num_records_bytes=num_bytes)
        )
    )


def _build_stage2_8wave(
    *, model_dim, inter_dim, experts, tile_m, blocks_per_wg, doweight_stage1
):
    elem_t = fx.Float8E4M3FN
    K = int(inter_dim)
    KB = K // BLOCK_K
    N = int(model_dim)
    NB = N // BLOCK_N
    NPW = int(blocks_per_wg)
    NCH = NB // NPW
    assert int(tile_m) == BLOCK_M, f"tile_m={tile_m}: the 8-wave stage 2 is 128 rows"
    assert KB == 2, f"inter_dim={K}: the 8-wave stage 2 holds two K tiles of A"
    assert N % BLOCK_N == 0, f"model_dim={N} must be a multiple of 128"
    assert NB % NPW == 0, f"blocks_per_wg={NPW} must divide {NB}"
    assert experts * N * K < 2**31

    fields = {f"a{k}": fx.Array[elem_t, LDS_TILE, 16] for k in range(KB)}
    # One LDS object per buffer, so DMA into one does not alias reads of another.
    for s in range(STAGES):
        for k in range(KB):
            fields[f"b{k}{s}"] = fx.Array[elem_t, B_TILE, 16]
    fields["c"] = fx.Array[fx.Int32, BLOCK_M * C_ROW_DWORDS, 16]
    LDS = fx.struct(type("LDS", (), {"__annotations__": fields}))

    def _mma(
        lds,
        arg_inter,
        arg_inter_scale,
        arg_w,
        arg_w_scale,
        arg_expert_ids,
        arg_sorted_weights,
        rows,
        nb0,
        e_idx,
    ):
        tid = gpu.thread_idx.x
        wave_id = tid // 64
        lane_id = tid % 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        grp = lane_id // 16
        row0 = e_idx * fx.Int32(BLOCK_M)
        expert_id = fx.Int32(
            rocdl.readfirstlane(T.i32, _view_i32(arg_expert_ids)[e_idx].ir_value())
        )

        # DMA lane (row, 16-byte K chunk) of stage1_8wave.py; LDS row L = c * 64 +
        # tid // 8 holds tile row (L % 8) * 16 + (L % 64) // 8 + c * 8.
        lane_row = tid // 8
        a_voff = [
            (row0 + (lane_row % 8) * 16 + lane_row // 8 + c * 8) * fx.Int32(K)
            + (tid % 8) * 16
            for c in range_constexpr(2)
        ]
        a_rsrc = _rsrc(arg_inter, fx.Int64(rows) * fx.Int64(K))
        w_iter = fx.add_offset(
            fx.recast_iter(fx.Uint8, fx.get_iter(arg_w)),
            fx.Int64(expert_id) * fx.Int64(N * K),
        )
        w_rsrc = rocdl.get_buffer_rsrc(
            rocdl.make_buffer_ptr(w_iter, num_records_bytes=fx.Int32(N * K))
        )
        sw_rsrc = _rsrc(arg_w_scale, experts * NB * KB * 4)
        sw_base = expert_id * fx.Int32(NB * KB)

        _rd = fx.make_layout(
            ((2, BLOCK_M // 16, 8), (32, BLOCK_K // 32)),
            ((A_GROUP, 2 * A_GROUP + 32, BLOCK_K), (1, 32)),
        )
        _b_rd = fx.make_layout(
            ((16, BLOCK_N // 16), (16, BLOCK_K // 16)),
            ((16, BLOCK_K * 16), (1, 256)),
        )
        a_bufs = [getattr(lds, f"a{k}").ptr for k in range(KB)]
        b_bufs = [
            [getattr(lds, f"b{k}{s}").ptr for s in range(STAGES)] for k in range(KB)
        ]
        ldsA_rd = [fx.make_view(p, _rd) for p in a_bufs]
        ldsB_rd = [[fx.make_view(p, _b_rd) for p in bb] for bb in b_bufs]

        mma_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, elem_t))
        mma_atom = fx.atom_set_value(mma_atom, "scale_a", fx.Int32(0))
        mma_atom = fx.atom_set_value(mma_atom, "scale_b", fx.Int32(0))
        k_perm = fx.make_layout((32, 4), (1, 32))
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((4, 2, 1), (1, 4, 0)), (None, None, k_perm)
        )
        lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_t)
        copy_a = fx.make_tiled_copy_B(lds_copy_atom, tiled_mma).get_slice(tid)
        copy_b = fx.make_tiled_copy_A(lds_copy_atom, tiled_mma).get_slice(tid)
        s2r_B = [[copy_b.partition_S(v) for v in vv] for vv in ldsB_rd]
        thr_mma = tiled_mma.thr_slice(tid)

        def _a_m_slice_view(src, m_slice):
            return fx.flat_divide(src, (BLOCK_M // M_SLICES, BLOCK_K))[
                None, None, m_slice, 0
            ]

        frag_A = thr_mma.make_fragment_B(_a_m_slice_view(ldsA_rd[0], 0))
        frag_B = [thr_mma.make_fragment_A(ldsB_rd[k][0]) for k in range(KB)]
        dest_A = copy_a.retile(frag_A)
        dest_B = [copy_b.retile(f) for f in frag_B]

        c_fake = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.recast_iter(fx.Float32, fx.get_iter(arg_inter_scale)),
                fx.make_layout((BLOCK_N, BLOCK_M), (BLOCK_M, 1)),
            ),
            max_size=False,
        )
        c_fake = fx.flat_divide(c_fake, (BLOCK_N, BLOCK_M))[None, None, 0, 0]
        frag_C = thr_mma.make_fragment_C(c_fake)  # [4, N_REP, 4]
        frag_P = thr_mma.make_fragment_C(
            fx.flat_divide(c_fake, (BLOCK_N, BLOCK_M // M_SLICES))[None, None, 0, 0]
        )
        m_reps = M_SLICES * PHASE_M_REP

        # Row mr * 32 + wave_m * 16 + lane % 16 of the C fragment: its two A scales
        # (one per K tile) and its route weight, for the whole workgroup.
        lane_rows = [
            row0 + fx.Int32(mr * 32) + wave_m * fx.Int32(16) + lane_id % 16
            for mr in range_constexpr(m_reps)
        ]
        sa_rsrc = _rsrc(arg_inter_scale, fx.Int64(rows) * fx.Int64(KB * 4))
        sa_pairs = [
            Vec(
                buffer_ops.buffer_load(
                    sa_rsrc, (r * fx.Int32(KB)).ir_value(), vec_width=2
                )
            )
            for r in lane_rows
        ]
        tw_rsrc = _rsrc(arg_sorted_weights, fx.Int64(rows) * fx.Int64(4))
        if const_expr(not doweight_stage1):
            tws = [
                fx.Float32(buffer_ops.buffer_load(tw_rsrc, r.ir_value(), vec_width=1))
                for r in lane_rows
            ]

        def sa_vec(kt, m_slice):
            return Vec.from_elements(
                [
                    sa_pairs[m_slice * PHASE_M_REP + m0][kt]
                    for m0 in range_constexpr(PHASE_M_REP)
                ],
                fx.Float32,
            )

        def load_sb(nb):
            # s_buffer_load through inline asm: the intrinsic's lowering adds waits
            a0 = (sw_base + nb * fx.Int32(KB)) * fx.Int32(4)
            a1 = a0 + fx.Int32(4)
            res = _llvm.inline_asm(
                ir.Type.parse("!llvm.struct<(f32, f32)>"),
                [sw_rsrc, a0.ir_value(), a1.ir_value()],
                "s_buffer_load_dword $0, $2, $3\ns_buffer_load_dword $1, $2, $4",
                "=&s,=&s,s,s,s,~{memory}",
                has_side_effects=True,
            )
            return (
                fx.Float32(_llvm.extractvalue(T.f32, res, [0])),
                fx.Float32(_llvm.extractvalue(T.f32, res, [1])),
            )

        def lds_rd_A(kt, m_slice):
            src = copy_a.partition_S(_a_m_slice_view(ldsA_rd[kt], m_slice))
            fx.copy(lds_copy_atom, src, dest_A)

        def lds_rd_B(kt, buf):
            fx.copy(lds_copy_atom, s2r_B[kt][buf], dest_B[kt], pred=None)

        a_wave_off = fx.Int32(
            rocdl.readfirstlane(
                T.i32,
                (
                    (wave_id % 2) * A_GROUP + wave_id // 2 * (2 * A_GROUP + 32)
                ).ir_value(),
            )
        )
        b_wave_off = fx.Int32(rocdl.readfirstlane(T.i32, (wave_id * 1024).ir_value()))
        # [N/16, K/16, 16 n0, 16 k0]: each wave DMAs 1 KiB of a 16-row group
        b_voff = (wave_id // 2) * (16 * K) + (wave_id % 2) * 1024 + lane_id * 16
        b_src = [
            [b_voff + (c * 64 * K + kt * B_TILE_K_BYTES) for c in range_constexpr(2)]
            for kt in range_constexpr(KB)
        ]
        zero = fx.Int32(0)

        def dma_A():
            for kt in range_constexpr(KB):
                for c in range_constexpr(2):
                    rocdl.raw_ptr_buffer_load_lds(
                        a_rsrc,
                        _lds_byte_ptr(a_bufs[kt], a_wave_off + c * A_CHUNK),
                        fx.Int32(16),
                        a_voff[c],
                        fx.Int32(kt * BLOCK_K),
                        zero,
                        zero,
                    )

        def dma_B(kt, buf, nb):
            soff = nb * fx.Int32(BLOCK_N * K)
            for c in range_constexpr(2):
                rocdl.raw_ptr_buffer_load_lds(
                    w_rsrc,
                    _lds_byte_ptr(b_bufs[kt][buf], b_wave_off + c * 8192),
                    fx.Int32(16),
                    b_src[kt][c],
                    soff,
                    zero,
                    zero,
                )

        def do_gemm(frag_Bk, dq_sa, dq_sb, dq_slice):
            # Retire the pending partial (scales dq_*, M slice dq_slice) into frag_C
            # under the next MFMA, which overwrites it from a zero accumulator.
            for m0 in range_constexpr(PHASE_M_REP):
                dq = Vec(dq_sa)[m0] * dq_sb
                rocdl.sched_barrier(0)
                for n0 in range_constexpr(N_REP):
                    sub_C = frag_C[None, n0, dq_slice * PHASE_M_REP + m0]
                    sub_P = frag_P[None, n0, m0]
                    partial = Vec(sub_P.load())
                    accum = Vec(sub_C.load())
                    sub_C.store(
                        Vec.from_elements(
                            [fx.fma(partial[e], dq, accum[e]) for e in range(4)],
                            fx.Float32,
                        )
                    )
                    rocdl.sched_barrier(0)
                    sub_P.store(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32),
                            [
                                Vec(frag_Bk[None, n0, 0].load()).bitcast(fx.Int32),
                                Vec(frag_A[None, m0, 0].load()).bitcast(fx.Int32),
                                Vec.filled(4, 0.0, fx.Float32),
                                0,
                                0,
                                0,
                                fx.Int32(0),
                                0,
                                fx.Int32(0),
                            ],
                        )
                    )
                    rocdl.sched_barrier(0)

        def stage_slice(m_slice):
            # out = bf16(acc * w[row]) into the staging tile, then a fresh accumulator.
            # One 16-byte store per row: a merged ds_write2 loses the LDS alias info
            # and waits for every DMA in flight.
            for m0 in range_constexpr(PHASE_M_REP):
                mr = m_slice * PHASE_M_REP + m0
                row = fx.Int32(mr * 32) + wave_m * fx.Int32(16) + lane_id % 16
                words = []
                for n0 in range_constexpr(N_REP):
                    sub_C = frag_C[None, n0, mr]
                    acc = Vec(sub_C.load())
                    if const_expr(not doweight_stage1):
                        acc = acc * tws[mr]
                    pk = acc.to(fx.BFloat16).bitcast(fx.Int32)
                    words += [pk[0], pk[1]]
                    sub_C.store(Vec.filled(4, 0.0, fx.Float32))
                d = (
                    row * fx.Int32(C_ROW_DWORDS)
                    + wave_n * fx.Int32(16)
                    + grp * fx.Int32(4)
                )
                _llvm.StoreOp(
                    Vec.from_elements(words, fx.Int32).ir_value(),
                    _lds_byte_ptr(lds.c.ptr, d * fx.Int32(4)),
                    alignment=16,
                )

        def begin_compute_phase():
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        def end_compute_phase():
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        # Per step j (channel block nb, buffer t = j % 2), one barrier interval each:
        #   I0: LDS A(k0, s0); SB(nb)               P0: k0 s0, retire k1 s1 (nb - 1)
        #   I1: LDS B(k1), A(k1, s0); stage s1 (nb - 1); DMA B(k0, nb + 2) -> t
        #                                           P1: k1 s0, retire k0 s0
        #   I2: wait B(k0, nb + 1); LDS A(k0, s1); DMA B(k1, nb + 2) -> t
        #                                           P2: k0 s1, retire k1 s0
        #   I3: wait B(k1, nb + 1); LDS B(k0, nb + 1), A(k1, s1); stage s0 (nb)
        #                                           P3: k1 s1, retire k0 s1
        # Waves 4-7 trail by one interval, so a buffer is rewritten two intervals after
        # its last read and read two intervals after its wait; the store waves read a
        # staged slice after both halves have written it. Prefetches past the last
        # block reload it, so every step issues and waits the same.
        last = nb0 + fx.Int32(NPW - 1)

        def _clamp(nb):
            return (nb < last).select(nb, last)

        rocdl.sched_barrier(0)
        dma_A()
        for s in range_constexpr(STAGES):
            nb = _clamp(nb0 + fx.Int32(s))
            for kt in range_constexpr(KB):
                dma_B(kt, s, nb)
        rocdl.sched_barrier(0)
        if wave_id >= 4:
            rocdl.s_barrier()
        frag_C.fill(0)
        frag_P.fill(0)
        rocdl.s_waitcnt(vmcnt=DMA_PER_STEP * (STAGES - 1), lgkmcnt=0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)
        lds_rd_B(0, 0)

        def step(nb, t, fifo_sa, fifo_sb):
            nn = _clamp(nb + fx.Int32(STAGES))

            lds_rd_A(0, 0)
            sb0, sb1 = load_sb(nb)
            begin_compute_phase()
            do_gemm(frag_B[0], fifo_sa, fifo_sb, 1)
            end_compute_phase()

            lds_rd_B(1, t)
            lds_rd_A(1, 0)
            stage_slice(1)
            dma_B(0, t, nn)
            begin_compute_phase()
            do_gemm(frag_B[1], sa_vec(0, 0), sb0, 0)
            end_compute_phase()

            rocdl.s_waitcnt(vmcnt=DMA_PER_STEP * (STAGES - 1))
            lds_rd_A(0, 1)
            dma_B(1, t, nn)
            begin_compute_phase()
            do_gemm(frag_B[0], sa_vec(1, 0), sb1, 0)
            end_compute_phase()

            rocdl.s_waitcnt(vmcnt=DMA_PER_STEP * (STAGES - 1))
            lds_rd_B(0, (t + 1) % STAGES)
            lds_rd_A(1, 1)
            stage_slice(0)
            begin_compute_phase()
            do_gemm(frag_B[1], sa_vec(0, 1), sb0, 1)
            end_compute_phase()
            return sa_vec(1, 1), sb1

        fifo_sa = Vec.filled(PHASE_M_REP, 0.0, fx.Float32)
        fifo_sb = fx.Float32(0)
        n_main = NPW - NPW % STAGES
        for j in range(0, n_main, STAGES):
            for u in range_constexpr(STAGES):
                fifo_sa, fifo_sb = step(
                    nb0 + fx.Int32(j) + fx.Int32(u), u, fifo_sa, fifo_sb
                )
        for u in range_constexpr(NPW % STAGES):
            fifo_sa, fifo_sb = step(nb0 + fx.Int32(n_main + u), u, fifo_sa, fifo_sb)

        for m0 in range_constexpr(PHASE_M_REP):
            dq = Vec(fifo_sa)[m0] * fifo_sb
            for n0 in range_constexpr(N_REP):
                cs = frag_C[None, n0, (M_SLICES - 1) * PHASE_M_REP + m0]
                p = Vec(frag_P[None, n0, m0].load())
                c = Vec(cs.load())
                cs.store(
                    Vec.from_elements(
                        [fx.fma(p[e], dq, c[e]) for e in range(4)], fx.Float32
                    )
                )
        stage_slice(1)
        rocdl.s_waitcnt(lgkmcnt=0)
        if wave_id < 4:
            rocdl.s_barrier()
        rocdl.s_barrier()
        # the clamped tail prefetches still target LDS
        rocdl.s_waitcnt(vmcnt=0)

    _mma = ASTRewriter.transform(_mma)

    def _store(lds, arg_out, arg_sorted_token_ids, M, nb0, e_idx):
        tid = gpu.thread_idx.x
        sw = tid // 64 - fx.Int32(MMA_WAVES)
        lane_id = tid % 64
        row0 = e_idx * fx.Int32(BLOCK_M)
        sid_view = _view_i32(arg_sorted_token_ids)
        out_bytes = M * fx.Int32(N * 2)
        out_rsrc = _rsrc(arg_out, out_bytes)
        c_view = fx.make_view(lds.c.ptr, fx.make_layout(BLOCK_M * C_ROW_DWORDS, 1))

        # This wave stores rows s * 64 + sw * 16 + i of slice s; a padding row
        # (token >= M) points past the descriptor, which drops its atomics.
        pair = (
            lane_id // 2 % 2 * 32
            + lane_id // 16 * 8
            + lane_id // 4 % 4 * 2
            + lane_id % 2
        )
        row_off = []
        for s in range_constexpr(M_SLICES):
            for i in range_constexpr(ROWS_PER_STORE_WAVE):
                r = fx.Int32(s * SLICE_ROWS + i) + sw * fx.Int32(ROWS_PER_STORE_WAVE)
                tok = sid_view[row0 + r] & 0xFFFFFF
                row_off.append(
                    (tok < M).select(tok * fx.Int32(N * 2), out_bytes)
                    + pair * fx.Int32(4)
                )
        z = fx.Int32(0)

        def flush(s, nb):
            vals = []
            for i in range_constexpr(ROWS_PER_STORE_WAVE):
                r = fx.Int32(s * SLICE_ROWS + i) + sw * fx.Int32(ROWS_PER_STORE_WAVE)
                vals.append(c_view[r * fx.Int32(C_ROW_DWORDS) + lane_id])
            rocdl.s_waitcnt(lgkmcnt=0)
            ch = nb * fx.Int32(BLOCK_N * 2)
            for i in range_constexpr(ROWS_PER_STORE_WAVE):
                v = Vec.from_elements([vals[i]], fx.Int32).bitcast(fx.BFloat16)
                rocdl.raw_ptr_buffer_atomic_fadd(
                    v, out_rsrc, row_off[s * ROWS_PER_STORE_WAVE + i], ch, z
                )

        def barriers(n):
            for _ in range_constexpr(n):
                rocdl.s_barrier()

        # Barrier for barrier with waves 0-3: two before the first step, eight per step
        # and two after the last. Slice 0 of block j is staged by the end of step j,
        # slice 1 four barriers into step j + 1 (or by the last barrier).
        barriers(2 + 8)
        for j in range(1, NPW):
            flush(0, nb0 + fx.Int32(j) - fx.Int32(1))
            barriers(4)
            flush(1, nb0 + fx.Int32(j) - fx.Int32(1))
            barriers(4)
        flush(0, nb0 + fx.Int32(NPW - 1))
        barriers(2)
        flush(1, nb0 + fx.Int32(NPW - 1))

    _store = ASTRewriter.transform(_store)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def fp8b_stage2_8wave(
        arg_out: fx.Tensor,
        arg_inter: fx.Tensor,
        arg_inter_scale: fx.Tensor,
        arg_w: fx.Tensor,
        arg_w_scale: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_rows: fx.Int32,
    ):
        num_valid = fx.Int32(
            rocdl.readfirstlane(T.i32, _view_i32(arg_num_valid_ids)[0].ir_value())
        )
        # Workgroup pid runs on XCD pid % 8. Renumber the live ones so each XCD takes
        # a contiguous run of (tile, channel chunk) pairs: a tile's chunks share its A
        # rows, and an expert's consecutive tiles its W2, in one L2.
        pid = gpu.block_idx.x + gpu.block_idx.y * fx.Int32(NCH)
        live = (num_valid + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M) * fx.Int32(NCH)
        if pid < live:
            per_xcd = (live + fx.Int32(NUM_XCDS - 1)) // fx.Int32(NUM_XCDS)
            tall = live % fx.Int32(NUM_XCDS)
            tall = (tall == fx.Int32(0)).select(fx.Int32(NUM_XCDS), tall)
            xcd = pid % fx.Int32(NUM_XCDS)
            local = pid // fx.Int32(NUM_XCDS)
            v = (xcd < tall).select(
                xcd * per_xcd + local,
                tall * per_xcd + (xcd - tall) * (per_xcd - fx.Int32(1)) + local,
            )
            nb0 = v % fx.Int32(NCH) * fx.Int32(NPW)
            e_idx = v // fx.Int32(NCH)
            lds = fx.SharedAllocator().allocate(LDS).peek()
            if gpu.thread_idx.x < fx.Int32(64 * MMA_WAVES):
                _mma(
                    lds,
                    arg_inter,
                    arg_inter_scale,
                    arg_w,
                    arg_w_scale,
                    arg_expert_ids,
                    arg_sorted_weights,
                    i32_rows,
                    nb0,
                    e_idx,
                )
            else:
                _store(lds, arg_out, arg_sorted_token_ids, i32_tokens, nb0, e_idx)

    @flyc.jit
    def launch_fp8b_stage2_8wave(
        arg_out: fx.Tensor,
        arg_inter: fx.Tensor,
        arg_inter_scale: fx.Tensor,
        arg_w: fx.Tensor,
        arg_w_scale: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_rows: fx.Int32,
        i32_tiles: fx.Int32,
        stream: fx.Stream,
    ):
        fp8b_stage2_8wave(
            arg_out,
            arg_inter,
            arg_inter_scale,
            arg_w,
            arg_w_scale,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            i32_tokens,
            i32_rows,
            value_attrs=_NO_PACKED_F32,
        ).launch(
            grid=(NCH, fx.Int64(i32_tiles), 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch_fp8b_stage2_8wave


def default_blocks_per_wg(model_dim):
    nb = int(model_dim) // BLOCK_N
    for npw in (4, 2, 1):
        if nb % npw == 0:
            return npw
    return 1


@functools.lru_cache(maxsize=256)
def compile_stage2_8wave(
    *,
    model_dim,
    inter_dim,
    experts,
    topk,
    tile_m=128,
    doweight_stage1=False,
    combine="atomic_bf16",
    blocks_per_wg=None,
):
    assert (
        combine == "atomic_bf16"
    ), f"combine={combine}: the 8-wave stage 2 is bf16 atomics"
    del topk  # the atomics combine into out[token]
    return _build_stage2_8wave(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        tile_m=tile_m,
        blocks_per_wg=blocks_per_wg or default_blocks_per_wg(model_dim),
        doweight_stage1=doweight_stage1,
    )
