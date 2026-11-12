# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Portions Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 fp8 blockscale (per_1x128) MoE prefill stage 1 on eight waves.

Same contract as stage1.py (gate/up GEMM, activation, 1x128 fp8 quant of the
intermediate, one 128-channel block per workgroup), on the pipeline of
kernels/gemm_a8w8_blockscale_8wave.py with preshuffle_b: raw buffer->LDS DMA of
the gathered A rows and the shuffle_weight(16, 16) gate and up slabs into padded
LDS ping-pong buffers; the tile's A scales gathered once into an LDS table
(token x K block); B scales by scalar loads; a (4, 2)
wave grid of 16x16x128 MFMAs, each into a fresh partial folded one MFMA later
as acc += P * (a_scale[token, g] * w1_scale[block, g]).

The workgroup tile is the 128 sorted rows of one sort block against 256 weight
rows: B_l = the 128 gate rows of channel block blk_n, B_r = the matching 128 up
rows. Each wave holds 64 rows x 32 gate + 32 up channels, as two 32-row M slices.
Waves 4-7 run one barrier interval behind waves 0-3, so one wave of each SIMD
issues MFMAs while the other reads LDS and issues DMA.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

from . import helpers as fxh

# keep fp32 folds as VOP2: v_pk_fma_f32 does not issue under an MFMA
_NO_PACKED_F32 = {"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]}

CHANNELS = 128
BLOCK_M = 128  # sorted rows per tile
BLOCK_N = 128  # gate (and up) rows per tile
BLOCK_K = 128
M_SLICES = 2
PHASE_M_REP = BLOCK_M // 2 // 16 // M_SLICES
N_REP = BLOCK_N // 64
# Padded LDS: 16 bytes after each eight rows, 32 more after each sixteen.
A_GROUP = 8 * BLOCK_K + 16
LDS_TILE = (BLOCK_M // 16) * (2 * A_GROUP + 32)
A_CHUNK = 4 * (2 * A_GROUP + 32)  # LDS distance of 64 rows
B_TILE_K_BYTES = 16 * BLOCK_K  # one K tile of a 16-row shuffle_weight group
STAGES = 2  # 3 stages spill in the tail and run slower
NUM_XCDS = 8
DMA_PER_TILE = 6  # per wave: 2 B_l + 2 B_r + 2 A buffer->LDS loads


def _lds_byte_ptr(ptr, byte_offset):
    return fx.to_llvm_ptr(
        fx.add_offset(fx.recast_iter(fx.Uint8, ptr), fx.make_int_tuple(byte_offset))
    )


def _view_i32(t):
    return fx.make_view(
        fx.recast_iter(fx.Int32, fx.get_iter(t)), fx.make_layout(1 << 30, 1)
    )


def _build_stage1_8wave(*, model_dim, inter_dim, experts, tile_m, act):
    elem_t = fx.Float8E4M3FN
    K = int(model_dim)
    KB = K // BLOCK_K
    I = int(inter_dim)
    IB = I // CHANNELS
    assert int(tile_m) == BLOCK_M, f"tile_m={tile_m}: the 8-wave stage 1 is 128 rows"
    assert K % 256 == 0, f"model_dim={K} must be a multiple of 256"
    assert I % CHANNELS == 0, f"inter_dim={I} must be a multiple of 128"
    assert act == "silu", act
    # descriptor-relative B offsets span one expert's gate/up slab
    assert 2 * I * K < 2**31
    # The tile's A scales [128 rows, KB] stay in LDS for the whole K loop, rows padded
    # so that 16 consecutive rows of one K group fall in distinct banks.
    SA_STRIDE = KB + 2 * (1 - KB // 2 % 2)
    SA_PAIRS = KB // 2  # one 8-byte load per lane, 16 lanes per row
    SA_PASSES = (SA_PAIRS + 15) // 16

    # One LDS object per buffer: DMA into one stage must not alias reads of another,
    # or the backend waits on vmcnt before every LDS read.
    fields = {}
    for s in range(STAGES):
        for name in ("a", "bl", "br"):
            fields[f"{name}{s}"] = fx.Array[elem_t, LDS_TILE, 16]
    # two spare floats take the idle lanes' stores
    fields["sa"] = fx.Array[fx.Float32, BLOCK_M * SA_STRIDE + 2, 16]
    fields["amax"] = fx.Array[fx.Float32, 4 * BLOCK_M, 16]
    LDS = fx.struct(type("LDS", (), {"__annotations__": fields}))

    def _body(
        arg_inter,
        arg_inter_scale,
        arg_x,
        arg_x_scale,
        arg_w,
        arg_w_scale,
        arg_sorted_token_ids,
        arg_expert_ids,
        M,
        rows,
        blk_n,
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

        lds = fx.SharedAllocator().allocate(LDS).peek()

        # Gathered rows: sorted id = token | slot << 24; padding rows (token >= M)
        # read past the descriptor's end, which returns zeros.
        sid_view = _view_i32(arg_sorted_token_ids)

        def _token(r):
            tok = sid_view[row0 + r] & 0xFFFFFF
            return (tok < M).select(tok, M)

        # DMA lane (row, 16-byte K chunk) of the dense kernel; LDS row L = c * 64 +
        # tid // 8 holds tile row (L % 8) * 16 + (L % 64) // 8 + c * 8.
        lane_row = tid // 8
        a_voff = [
            _token((lane_row % 8) * 16 + lane_row // 8 + c * 8) * fx.Int32(K)
            + (tid % 8) * 16
            for c in range_constexpr(2)
        ]
        sa_rows = [r * 32 + tid // 16 for r in range_constexpr(BLOCK_M // 32)]
        sa_toks = [_token(r) for r in sa_rows]

        x_iter = fx.get_iter(arg_x)
        xs_iter = fx.get_iter(arg_x_scale)
        x_bytes = M * fx.Int32(K)
        xs_bytes = M * fx.Int32(KB * 4)
        # gate rows blk_n * 128 .. + 128 of the expert's [2I, K] slab; up rows I later
        w_iter = fx.add_offset(
            fx.recast_iter(fx.Uint8, fx.get_iter(arg_w)),
            fx.Int64(expert_id * fx.Int32(2 * IB) + blk_n) * fx.Int64(CHANNELS * K),
        )
        w_bytes = (fx.Int32(2 * IB) - blk_n) * fx.Int32(CHANNELS * K)

        def dma_resource(src_iter, num_bytes, byte_offset):
            base = fx.add_offset(
                fx.recast_iter(fx.Uint8, src_iter), fx.Int64(byte_offset)
            )
            remaining = num_bytes - byte_offset
            remaining = (remaining > 0).select(remaining, fx.Int32(0))
            return rocdl.get_buffer_rsrc(
                rocdl.make_buffer_ptr(base, num_records_bytes=remaining)
            )

        sw_rsrc = rocdl.get_buffer_rsrc(
            fx.get_iter(
                rocdl.make_buffer_tensor(
                    arg_w_scale, num_records_bytes=experts * 2 * IB * KB * 4
                )
            )
        )
        sw_gate = (expert_id * fx.Int32(2 * IB) + blk_n) * fx.Int32(KB)
        sw_up = sw_gate + fx.Int32(IB * KB)

        # LDS row L starts at (L % 8) * 128 + (L // 8 % 2) * A_GROUP + (L // 16) *
        # (2 * A_GROUP + 32); _rd indexes by tile row, undoing the DMA's row order.
        _rd = fx.make_layout(
            ((2, BLOCK_M // 16, 8), (32, BLOCK_K // 32)),
            ((A_GROUP, 2 * A_GROUP + 32, BLOCK_K), (1, 32)),
        )
        # LDS B [8 n1, 4 k2, 2 k1, 16 n0, 16 k0], the shuffle_weight(16, 16) order
        _b_rd = fx.make_layout(
            ((16, BLOCK_N // 16), (16, BLOCK_K // 16)),
            ((16, BLOCK_K * 16), (1, 256)),
        )
        a_bufs = [getattr(lds, f"a{s}").ptr for s in range(STAGES)]
        bl_bufs = [getattr(lds, f"bl{s}").ptr for s in range(STAGES)]
        br_bufs = [getattr(lds, f"br{s}").ptr for s in range(STAGES)]
        sa_view = fx.make_view(lds.sa.ptr, fx.make_layout(BLOCK_M * SA_STRIDE + 2, 1))
        ldsA_rd = [fx.make_view(p, _rd) for p in a_bufs]
        ldsBl_rd = [fx.make_view(p, _b_rd) for p in bl_bufs]
        ldsBr_rd = [fx.make_view(p, _b_rd) for p in br_bufs]

        # B-first: weights are MFMA operand A (4 waves on channels), rows operand B
        # (2 waves). Both operands take 32 consecutive K per lane.
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
        s2r_Bl = [copy_b.partition_S(v) for v in ldsBl_rd]
        s2r_Br = [copy_b.partition_S(v) for v in ldsBr_rd]
        thr_mma = tiled_mma.thr_slice(tid)

        def _a_m_slice_view(src, m_slice):
            return fx.flat_divide(src, (BLOCK_M // M_SLICES, BLOCK_K))[
                None, None, m_slice, 0
            ]

        frag_A = thr_mma.make_fragment_B(_a_m_slice_view(ldsA_rd[0], 0))
        frag_Bl = thr_mma.make_fragment_A(ldsBl_rd[0])
        frag_Br = thr_mma.make_fragment_A(ldsBr_rd[0])
        dest_A = copy_a.retile(frag_A)
        dest_Bl = copy_b.retile(frag_Bl)
        dest_Br = copy_b.retile(frag_Br)

        c_fake = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.recast_iter(fx.Float32, xs_iter),
                fx.make_layout((BLOCK_N, BLOCK_M), (BLOCK_M, 1)),
            ),
            max_size=False,
        )
        c_fake = fx.flat_divide(c_fake, (BLOCK_N, BLOCK_M))[None, None, 0, 0]
        frag_Cg = thr_mma.make_fragment_C(c_fake)  # [4, N_REP, 4]
        frag_Cu = thr_mma.make_fragment_C(c_fake)
        frag_P = thr_mma.make_fragment_C(
            fx.flat_divide(c_fake, (BLOCK_N, BLOCK_M // M_SLICES))[None, None, 0, 0]
        )

        sa_row_base = (wave_m * 16 + lane_id % 16) * fx.Int32(SA_STRIDE)

        def lds_rd_sa(ki, m_slice):
            col = sa_row_base + ki
            return Vec.from_elements(
                [
                    sa_view[
                        col + fx.Int32((m_slice * PHASE_M_REP + m0) * 32 * SA_STRIDE)
                    ]
                    for m0 in range_constexpr(PHASE_M_REP)
                ],
                fx.Float32,
            )

        xs_rsrc = rocdl.get_buffer_rsrc(
            fx.get_iter(
                rocdl.make_buffer_tensor(
                    arg_x_scale, max_size=False, num_records_bytes=xs_bytes
                )
            )
        )

        def load_sa_table():
            vals = []
            for r in range_constexpr(BLOCK_M // 32):
                for c in range_constexpr(SA_PASSES):
                    pair = fx.Int32(c * 16) + tid % 16
                    vals.append(
                        buffer_ops.buffer_load(
                            xs_rsrc,
                            (sa_toks[r] * fx.Int32(KB) + pair * 2).ir_value(),
                            vec_width=2,
                            mask=(pair < fx.Int32(SA_PAIRS)).ir_value(),
                        )
                    )
            return vals

        def store_sa_table(vals):
            for r in range_constexpr(BLOCK_M // 32):
                for c in range_constexpr(SA_PASSES):
                    pair = fx.Int32(c * 16) + tid % 16
                    v = Vec(vals[r * SA_PASSES + c])
                    idx = (pair < fx.Int32(SA_PAIRS)).select(
                        sa_rows[r] * fx.Int32(SA_STRIDE) + pair * 2,
                        fx.Int32(BLOCK_M * SA_STRIDE),
                    )
                    sa_view[idx] = v[0]
                    sa_view[idx + 1] = v[1]

        def load_sb(ki):
            # s_buffer_load through inline asm: the intrinsic's lowering adds waits
            ag = (sw_gate + ki) * fx.Int32(4)
            au = (sw_up + ki) * fx.Int32(4)
            res = _llvm.inline_asm(
                ir.Type.parse("!llvm.struct<(f32, f32)>"),
                [sw_rsrc, ag.ir_value(), au.ir_value()],
                "s_buffer_load_dword $0, $2, $3\ns_buffer_load_dword $1, $2, $4",
                "=&s,=&s,s,s,s,~{memory}",
                has_side_effects=True,
            )
            return (
                fx.Float32(_llvm.extractvalue(T.f32, res, [0])),
                fx.Float32(_llvm.extractvalue(T.f32, res, [1])),
            )

        def lds_rd_A(buf, m_slice):
            src = copy_a.partition_S(_a_m_slice_view(ldsA_rd[buf], m_slice))
            fx.copy(lds_copy_atom, src, dest_A)

        def lds_rd_Bl(buf):
            fx.copy(lds_copy_atom, s2r_Bl[buf], dest_Bl, pred=None)

        def lds_rd_Br(buf):
            fx.copy(lds_copy_atom, s2r_Br[buf], dest_Br, pred=None)

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
            [b_voff + (q * I + c * 64) * K for c in range_constexpr(2)]
            for q in range_constexpr(2)
        ]
        zero = fx.Int32(0)

        def dma_A(buf, ki):
            rsrc = dma_resource(x_iter, x_bytes, ki * fx.Int32(BLOCK_K))
            for c in range_constexpr(2):
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc,
                    _lds_byte_ptr(a_bufs[buf], a_wave_off + c * A_CHUNK),
                    fx.Int32(16),
                    a_voff[c],
                    zero,
                    zero,
                    zero,
                )

        def dma_B(q, buf, ki):
            rsrc = dma_resource(w_iter, w_bytes, ki * fx.Int32(B_TILE_K_BYTES))
            dst = (bl_bufs, br_bufs)[q][buf]
            for c in range_constexpr(2):
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc,
                    _lds_byte_ptr(dst, b_wave_off + c * 8192),
                    fx.Int32(16),
                    b_src[q][c],
                    zero,
                    zero,
                    zero,
                )

        def do_gemm(frag_C, frag_B, dq_sa, dq_sb, dq_slice):
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
                                Vec(frag_B[None, n0, 0].load()).bitcast(fx.Int32),
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

        def begin_compute_phase():
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        def end_compute_phase():
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        # Per K tile k (buffer t = k % STAGES, S = STAGES), one barrier interval each:
        #   I0: LDS A[s0], SA[s0]; SB(k)          P0: gate s0, retire up s1 (k-1)
        #   I1: LDS B_r; DMA B_l(k+S) -> t        P1: up s0, retire gate s0
        #   I2: wait B_l(k+1); LDS A[s1], SA[s1]; DMA B_r(k+S) -> t
        #                                         P2: gate s1, retire up s0
        #   I3: wait A, B_r (k+1); LDS B_l(k+1); DMA A(k+S) -> t
        #                                         P3: up s1, retire gate s1
        # SA, the whole tile's A scales, is loaded into LDS once before the loop.
        # Waves 4-7 trail by one interval, so a buffer is rewritten two intervals
        # after its last read and read two intervals after its wait. Prefetches past
        # the last tile reload it, so every tile issues and waits the same.
        last = fx.Int32(KB - 1)

        def _clamp(kt):
            return (kt < fx.Int32(KB)).select(kt, last)

        sa_vals = load_sa_table()
        rocdl.sched_barrier(0)
        for kt in range_constexpr(STAGES):
            kc = _clamp(fx.Int32(kt))
            dma_B(0, kt, kc)
            dma_B(1, kt, kc)
            dma_A(kt, kc)
        rocdl.sched_barrier(0)
        store_sa_table(sa_vals)
        if wave_id >= 4:
            rocdl.s_barrier()
        frag_Cg.fill(0)
        frag_Cu.fill(0)
        frag_P.fill(0)
        rocdl.s_waitcnt(vmcnt=DMA_PER_TILE * (STAGES - 1), lgkmcnt=0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)
        lds_rd_Bl(0)

        def k_tile(ki, t, fifo_sa1, fifo_sbu):
            kn = _clamp(ki + fx.Int32(STAGES))

            lds_rd_A(t, 0)
            sa0 = lds_rd_sa(ki, 0)
            sbg, sbu = load_sb(ki)
            begin_compute_phase()
            do_gemm(frag_Cu, frag_Bl, fifo_sa1, fifo_sbu, 1)
            end_compute_phase()

            lds_rd_Br(t)
            dma_B(0, t, kn)
            begin_compute_phase()
            do_gemm(frag_Cg, frag_Br, sa0, sbg, 0)
            end_compute_phase()

            # B_r(k+1), A(k+1), the tiles between and B_l(k+S) may still be in flight
            rocdl.s_waitcnt(vmcnt=DMA_PER_TILE * (STAGES - 1))
            lds_rd_A(t, 1)
            sa1 = lds_rd_sa(ki, 1)
            dma_B(1, t, kn)
            begin_compute_phase()
            do_gemm(frag_Cu, frag_Bl, sa0, sbu, 0)
            end_compute_phase()

            rocdl.s_waitcnt(vmcnt=DMA_PER_TILE * (STAGES - 2) + 4)
            lds_rd_Bl((t + 1) % STAGES)
            dma_A(t, kn)
            begin_compute_phase()
            do_gemm(frag_Cg, frag_Br, sa1, sbg, 1)
            end_compute_phase()
            return sa1, sbu

        fifo_sa1 = Vec.filled(PHASE_M_REP, 0.0, fx.Float32)
        fifo_sbu = fx.Float32(0)
        k_main = KB - KB % STAGES
        for kidx in range(0, k_main, STAGES):
            for u in range_constexpr(STAGES):
                fifo_sa1, fifo_sbu = k_tile(
                    fx.Int32(kidx) + fx.Int32(u), u, fifo_sa1, fifo_sbu
                )
        for u in range_constexpr(KB % STAGES):
            fifo_sa1, fifo_sbu = k_tile(fx.Int32(k_main + u), u, fifo_sa1, fifo_sbu)

        for m0 in range_constexpr(PHASE_M_REP):
            dq = Vec(fifo_sa1)[m0] * fifo_sbu
            for n0 in range_constexpr(N_REP):
                cs = frag_Cu[None, n0, (M_SLICES - 1) * PHASE_M_REP + m0]
                p = Vec(frag_P[None, n0, m0].load())
                c = Vec(cs.load())
                cs.store(
                    Vec.from_elements(
                        [fx.fma(p[e], dq, c[e]) for e in range(4)], fx.Float32
                    )
                )
        if wave_id < 4:
            rocdl.s_barrier()
        # the clamped tail prefetches still target LDS
        rocdl.s_waitcnt(vmcnt=0)

        # h = bf16(silu(gate) * up), silu(g) = g * rcp(1 + exp(-g)), in frag_Cg.
        m_reps = fxh.reps(frag_Cg, 2)
        for mr in range_constexpr(m_reps):
            for n0 in range_constexpr(N_REP):
                g = frag_Cg[None, n0, mr].load()
                up = frag_Cu[None, n0, mr].load()
                out = []
                for v in range_constexpr(4):
                    gv = g[v]
                    e = fx.math.exp(-gv)
                    a = gv * rocdl.rcp(T.f32, (fx.Float32(1.0) + e).ir_value())
                    out.append(a * up[v])
                frag_Cg[None, n0, mr].store(
                    Vec.from_elements(out, fx.Float32).to(fx.BFloat16).to(fx.Float32)
                )

        # Row r = mr * 32 + wave_m * 16 + lane % 16 holds channels n0 * 64 + wave_n * 16
        # + grp * 4 + v: amax over v, n0, the four lane groups and the four wave_n.
        amax_view = fx.make_view(lds.amax.ptr, fx.make_layout(4 * BLOCK_M, 1))
        col = lane_id % 16
        for mr in range_constexpr(m_reps):
            mx = fx.Float32(fxh.AMAX_FLOOR)
            for n0 in range_constexpr(N_REP):
                h = frag_Cg[None, n0, mr].load()
                for v in range_constexpr(4):
                    mx = fx.max(mx, fx.math.absf(h[v]))
            mx = fx.max(mx, mx.shuffle_xor(16, 64))
            mx = fx.max(mx, mx.shuffle_xor(32, 64))
            if grp == fx.Int32(0):
                amax_view[
                    wave_n * fx.Int32(BLOCK_M)
                    + fx.Int32(mr * 32)
                    + wave_m * fx.Int32(16)
                    + col
                ] = mx
        gpu.barrier()

        out_rsrc = rocdl.get_buffer_rsrc(
            fx.get_iter(
                rocdl.make_buffer_tensor(
                    arg_inter,
                    max_size=False,
                    num_records_bytes=fx.Int64(rows) * fx.Int64(I),
                )
            )
        )
        scale_rsrc = rocdl.get_buffer_rsrc(
            fx.get_iter(
                rocdl.make_buffer_tensor(
                    arg_inter_scale,
                    max_size=False,
                    num_records_bytes=fx.Int64(rows) * fx.Int64(IB * 4),
                )
            )
        )
        for mr in range_constexpr(m_reps):
            r = fx.Int32(mr * 32) + wave_m * fx.Int32(16) + col
            amax = fx.max(
                fx.max(amax_view[r], amax_view[fx.Int32(BLOCK_M) + r]),
                fx.max(
                    amax_view[fx.Int32(2 * BLOCK_M) + r],
                    amax_view[fx.Int32(3 * BLOCK_M) + r],
                ),
            )
            # As aiter's per-group HIP quant: scale = absMax * (1 / max), q = x * (1 / scale).
            scale = amax * fx.Float32(fxh.FP8_MAX_RCP)
            inv_scale = fx.Float32(1.0) / scale
            row = row0 + r
            if (wave_n == fx.Int32(0)) & (grp == fx.Int32(0)):
                rocdl.raw_ptr_buffer_store(
                    scale.bitcast(fx.Int32).ir_value(),
                    scale_rsrc,
                    ((row * fx.Int32(IB) + blk_n) * fx.Int32(4)).ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=ir.IntegerAttr.get(T.i32, 0),
                )
            for n0 in range_constexpr(N_REP):
                h = frag_Cg[None, n0, mr].load()
                q = [h[v] * inv_scale for v in range(4)]
                lo = rocdl.cvt_pk_fp8_f32(
                    res=T.i32, src_a=q[0], src_b=q[1], old=fx.Int32(0), word_sel=False
                )
                pk = rocdl.cvt_pk_fp8_f32(
                    res=T.i32, src_a=q[2], src_b=q[3], old=lo, word_sel=True
                )
                ch = (
                    blk_n * fx.Int32(CHANNELS)
                    + fx.Int32(n0 * 64)
                    + wave_n * fx.Int32(16)
                    + grp * fx.Int32(4)
                )
                rocdl.raw_ptr_buffer_store(
                    pk,
                    out_rsrc,
                    (row * fx.Int32(I) + ch).ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=ir.IntegerAttr.get(T.i32, 0),
                )

    _body = ASTRewriter.transform(_body)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def fp8b_stage1_8wave(
        arg_inter: fx.Tensor,
        arg_inter_scale: fx.Tensor,
        arg_x: fx.Tensor,
        arg_x_scale: fx.Tensor,
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
        # a contiguous run of (tile, block) pairs: the IB blocks of a tile share its A
        # rows, and an expert's consecutive tiles its weights, in one L2.
        pid = gpu.block_idx.x + gpu.block_idx.y * fx.Int32(IB)
        live = (num_valid + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M) * fx.Int32(IB)
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
            _body(
                arg_inter,
                arg_inter_scale,
                arg_x,
                arg_x_scale,
                arg_w,
                arg_w_scale,
                arg_sorted_token_ids,
                arg_expert_ids,
                i32_tokens,
                i32_rows,
                v % fx.Int32(IB),
                v // fx.Int32(IB),
            )

    @flyc.jit
    def launch_fp8b_stage1_8wave(
        arg_inter: fx.Tensor,
        arg_inter_scale: fx.Tensor,
        arg_x: fx.Tensor,
        arg_x_scale: fx.Tensor,
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
        fp8b_stage1_8wave(
            arg_inter,
            arg_inter_scale,
            arg_x,
            arg_x_scale,
            arg_w,
            arg_w_scale,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            i32_tokens,
            i32_rows,
            value_attrs=_NO_PACKED_F32,
        ).launch(grid=(IB, fx.Int64(i32_tiles), 1), block=(512, 1, 1), stream=stream)

    return launch_fp8b_stage1_8wave


@functools.lru_cache(maxsize=256)
def compile_stage1_8wave(*, model_dim, inter_dim, experts, tile_m=128, act="silu"):
    return _build_stage1_8wave(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        tile_m=tile_m,
        act=act,
    )
