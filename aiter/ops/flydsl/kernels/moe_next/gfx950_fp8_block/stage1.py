# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Portions Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 fp8 blockscale (per_1x128) MoE stage 1: gate/up GEMM, activation and the
1x128 fp8 quantization of the intermediate, in one kernel.

Built on FlyDSL's kernels/moe/moe_gemm_2stage/gemm1.py (upstream 19418894): B-first
MFMA with the shuffle_weight(16, 16) W1 loaded straight to VGPRs, A rows gathered by
sorted token id through an LDS ping-pong.

Per sorted tile of tile_m rows and one 128-channel block of the intermediate:
  for each 128-wide K group g:  P = A[:, g] . W1[:, g]^T (fresh fp32 partial)
                                acc += P * (a_scale[token, g] * w1_scale[block, g])
  gate, up = acc_gate, acc_up   (x route weight first when doweight_stage1, as the
                                 reference multiplies act_input before the activation)
  h = act(gate) * up            (SiLU as CK computes it: x * rcp(1 + exp(-x)))
  h = bf16(h)                   (the reference rounds stage 1 to bf16 before quant)
  scale = max(amax(|h|), 1e-10) * (1/448) over the 128 channels; q = fp8(h * (1/scale)),
                                (aiter's per-group HIP quant, which the dispatched path uses)
Writes q to inter[row, channels] and scale to inter_scale[row, block], row = sorted
position (core's per_1x128 intermediate layout: values [rows, I], scales [rows, I/128]).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops

from . import helpers as fxh

# keep fp32 folds as VOP2: v_pk_fma_f32 does not issue under an MFMA (lag-scale-fold-behind-mfma)
_NO_PACKED_F32 = {"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]}

CHANNELS = 128  # intermediate channels per workgroup = one quant group
# sc0 | sc1: split-K partials cross XCDs, which do not share L2; these accesses write
# through and read past it.
CPOL_COHERENT = 0x1 | 0x10


def _build_stage1(
    *, model_dim, inter_dim, experts, tile_m, act, doweight_stage1, wide_k, splits
):
    elem_t = fx.Float8E4M3FN
    K = int(model_dim)
    KB = K // 128
    N_e = 2 * int(inter_dim)
    I = int(inter_dim)
    IB = I // CHANNELS
    BM = int(tile_m)
    TILE_K = 128
    SPLITS = int(splits)
    KS = KB // SPLITS  # K tiles per split
    assert K % TILE_K == 0 and KB % 2 == 0, f"model_dim={K} must be a multiple of 256"
    assert I % CHANNELS == 0, f"inter_dim={I} must be a multiple of 128"
    assert BM in (16, 32, 64, 128), f"tile_m={BM}"
    assert act in ("silu", "gelu")
    assert KB % SPLITS == 0 and KS % 2 == 0, f"{KB} K tiles do not split {SPLITS} ways"
    assert SPLITS == 1 or wide_k, "split-K is built on the wide-K path"
    S, NTL = SPLITS, KS  # split count; K tiles each split's workgroup walks

    # The A gather copies 256 // (TILE_K / 16) = 32 rows per pass: a tile_m 16 buffer
    # must still hold 32 rows, or the copy runs into the other ping-pong buffer.
    _gather_rows = 256 // (TILE_K // 16)
    a_lds_size = max(BM, _gather_rows) * TILE_K

    # only this split's NTL K groups are staged (padded to whole 256-thread passes)
    _scl_pad = -(-(NTL * 2 * BM) // 256) * 256

    @fx.struct
    class GemmBuffers:
        a_ping: fx.Array[elem_t, a_lds_size, 16]
        a_pong: fx.Array[elem_t, a_lds_size, 16]
        # a_scale[token, kb] * w1_scale[gate|up block, kb] for every row and K group,
        # [kb][gate, up][row]: staged once so the K loop issues no scale buffer_loads.
        scl: fx.Array[fx.Float32, _scl_pad, 16]

    @fx.union
    class SharedStorage:
        sorted_lds: fx.Array[fx.Int32, 256, 16]
        gemm: GemmBuffers
        amax: fx.Array[fx.Float32, 4 * 128, 16]

    _val_per_thr = 16
    _swz_params = (3, 4, 3)
    NT_B = 3  # W1 load cache policy (0 cached, 2 nt)
    DIRECT_A = bool(wide_k)  # A tile straight to MFMA operand VGPRs, no LDS/barrier in loop
    PF = 2  # register prefetch ring depth (tiles); KB must be a multiple of lcm(PF, 2)
    assert NTL % PF == 0 and PF % 2 == 0
    _thrs_k = TILE_K // _val_per_thr
    _thrs_m = 256 // _thrs_k
    _m_per_wave = _thrs_m // 4

    def _gemm(
        blk_n, arg_p_input, arg_p_weight, lds, M, a_scale_t, w_scale_t, expert_id, rot, sp,
        sid_v, sorted_ids
    ):
        tid = gpu.thread_idx.x
        mma_atom, tiled_mma = fxh.make_1x4_tiled_mma(elem_t, wide_k)

        a_tensor = fx.rocdl.make_buffer_tensor(
            arg_p_input, max_size=False, num_records_bytes=fx.Int64(M) * fx.Int64(K)
        )
        b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)
        a_size_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((BM, K), (K, 1))),
            max_size=False,
        )
        a_tile = fx.flat_divide(a_size_buf, fx.make_tile(BM, TILE_K))[
            None, None, 0, None
        ]
        buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_t)
        # W1 is a read-once stream per (expert, block): non-temporal loads (kb: nt-stores)
        b_cp_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(NT_B), elem_t)
        g2r_tv_layout = fx.make_layout(
            ((_thrs_k, _thrs_m), (1, _val_per_thr)),
            ((_thrs_m * _val_per_thr, 1), (1, _thrs_m)),
        )
        a_mem_cp_g2r = fx.make_tiled_copy(
            buf_cp_atom_r, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
        )
        tiled_copy_sortid_a = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
            fx.make_layout(((_thrs_k, _m_per_wave, 4), 1), ((0, 1, _m_per_wave), 0)),
            fx.make_tile(_thrs_m),
        )
        if const_expr(not DIRECT_A):
            a_index_frag = fxh.read_sorted_index(
                tiled_copy_sortid_a, tid, lds.sorted_lds, BM
            )
            a_idx = fxh.TensorWithIndex(
                a_tensor, BM, TILE_K, a_index_frag, a_mem_cp_g2r, tid
            )
        a_mem_thr = a_mem_cp_g2r.get_slice(tid).partition_S(a_tile)
        # Stage every K group's combined scale in LDS (read by the loop after the
        # barrier that follows the first A write).
        lane16 = tid % fx.Int32(16)
        scl_view = fx.make_view(lds.gemm.scl.ptr, fx.make_layout(_scl_pad, 1))
        sid_view = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BM, 1))
        sw_gate_base = (expert_id * fx.Int32(2 * IB) + blk_n) * fx.Int32(KB)
        sw_up_base = (expert_id * fx.Int32(2 * IB) + fx.Int32(IB) + blk_n) * fx.Int32(
            KB
        )
        # Scale loads issue here; their LDS stores wait until after the first tile
        # loads are issued, so the prologue pays one memory latency, not two.
        def _issue_scale_loads():
            scl_vals = []
            for i in range_constexpr(_scl_pad // 256):
                idx = tid + fx.Int32(i * 256)
                row = idx % fx.Int32(BM)
                gsel = (idx // fx.Int32(BM)) % fx.Int32(2)
                kbl = idx // fx.Int32(2 * BM)
                if const_expr((i + 1) * 256 > NTL * 2 * BM):
                    kbl = (kbl < fx.Int32(NTL)).select(kbl, fx.Int32(NTL - 1))
                kbi = sp * fx.Int32(NTL) + kbl
                if const_expr(DIRECT_A):
                    tok = sid_v & fx.Int32(0xFFFFFF)  # row == tid % BM (BM divides 256)
                else:
                    tok = sid_view[row] & fx.Int32(0xFFFFFF)
                sa = a_scale_t[tok * fx.Int32(KB) + kbi]
                sw = w_scale_t[(gsel == fx.Int32(0)).select(sw_gate_base, sw_up_base) + kbi]
                scl_vals.append((idx, sa, sw))
            return scl_vals

        if const_expr(not DIRECT_A):
            scl_vals = _issue_scale_loads()
        if const_expr(not DIRECT_A):
            gpu.barrier()

        swz = fx.SwizzleType.get(*_swz_params)
        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_t)
        a_lds_w_bufs, a_lds_r_bufs, a_frag_bufs, a_frag_retile_bufs = [], [], [], []
        for _buf_ptr in (lds.gemm.a_ping.ptr, lds.gemm.a_pong.ptr):
            a_lds = fx.make_view(
                _buf_ptr,
                fx.make_composed_layout(
                    fx.static(swz), fx.make_ordered_layout((BM, TILE_K), order=(1, 0))
                ),
            )
            a_r2s = fx.make_tiled_copy(
                uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
            ).get_slice(tid)
            a_lds_w_bufs.append(a_r2s.partition_D(a_lds))
            a_lds_r_bufs.append(
                fx.make_tiled_copy_B(uni_cp_atom, tiled_mma)
                .get_slice(tid)
                .partition_S(a_lds)
            )
            a_frag = tiled_mma.make_fragment_B(a_lds)
            a_frag_bufs.append(a_frag)
            a_frag_retile_bufs.append(
                fx.make_tiled_copy_B(uni_cp_atom, tiled_mma)
                .get_slice(tid)
                .retile(a_frag)
            )
        a_cp_frag_bufs = [
            fx.make_fragment_like(a_mem_thr[None, None, None, 0]) for _ in range(PF)
        ]
        a_cp_frag_retile_bufs = [
            fx.make_tiled_copy(
                uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
            )
            .get_slice(tid)
            .retile(f)
            for f in a_cp_frag_bufs
        ]

        bl_tile = fx.flat_divide(b_tensor, fx.make_tile(CHANNELS, TILE_K))[
            None, None, blk_n * 2 + 0, None
        ]
        br_tile = fx.flat_divide(b_tensor, fx.make_tile(CHANNELS, TILE_K))[
            None, None, blk_n * 2 + 1, None
        ]
        b_g2r = fx.make_tiled_copy_A(buf_cp_atom_r, tiled_mma).get_slice(tid)
        bl_g2r = b_g2r.partition_S(bl_tile)
        br_g2r = b_g2r.partition_S(br_tile)
        bl_frag_bufs = [
            tiled_mma.make_fragment_A(bl_tile[None, None, 0]) for _ in range(PF)
        ]
        br_frag_bufs = [
            tiled_mma.make_fragment_A(br_tile[None, None, 0]) for _ in range(PF)
        ]
        bl_ret_bufs = [b_g2r.retile(f) for f in bl_frag_bufs]
        br_ret_bufs = [b_g2r.retile(f) for f in br_frag_bufs]
        _b_loads_per_tile = 2 * (
            fx.size(fx.get_shape(bl_frag_bufs[0])).to_py_value() // _val_per_thr
        )
        _a_loads_per_tile = (
            fx.size(fx.get_shape(a_cp_frag_bufs[0])).to_py_value() // _val_per_thr
        )
        if const_expr(DIRECT_A):
            _a_loads_per_tile = 2 * (BM // 16)
        _loads_per_tile = _a_loads_per_tile + _b_loads_per_tile

        c_fake_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(arg_p_input), fx.make_layout((CHANNELS, BM), (BM, 1))
            ),
            max_size=False,
        )
        c_fake = fx.flat_divide(c_fake_buf, fx.make_tile(CHANNELS, BM))[
            None, None, 0, 0
        ]
        c_gate = tiled_mma.make_fragment_C(c_fake)
        c_up = tiled_mma.make_fragment_C(c_fake)
        p_gate = p_up = None
        if const_expr(not wide_k):
            p_gate = tiled_mma.make_fragment_C(c_fake)
            p_up = tiled_mma.make_fragment_C(c_fake)
        c_gate.fill(0)
        c_up.fill(0)
        m_reps = fxh.reps(c_gate, 1)
        n_reps = fxh.reps(c_gate, 2)

        k_iters = fxh.reps(a_lds_r_bufs[0], 2)
        num_tiles = KB

        def _phys(kt):
            # rotated K start per workgroup (launch r1 k4): breaks lockstep HBM channel camping
            r = fx.Int32(kt) + rot
            r = (r < fx.Int32(NTL)).select(r, r - fx.Int32(NTL))
            return sp * fx.Int32(NTL) + r

        # Direct A: each lane loads its 32 B of the MFMA B operand (token column lane%16,
        # K (lane//16)*32..+32) straight from global; the 4 waves share the lines in L2,
        # so the K loop has no LDS write and no barrier (kb operand-direct-to-vgpr).
        _NA = BM // 16
        a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(a_tensor))
        lane64 = tid % fx.Int32(64)
        a_lane_off = []
        for n in range_constexpr(_NA):
            if const_expr(BM == 16):
                tokn = sid_v & fx.Int32(0xFFFFFF)  # tid % 16 == lane % 16
            else:
                tokn = sorted_ids[fx.Int32(n * 16) + lane64 % fx.Int32(16)] & fx.Int32(0xFFFFFF)
            a_lane_off.append(tokn * fx.Int32(K) + (lane64 // fx.Int32(16)) * fx.Int32(32))
        a_ring = [[fx.make_rmem_tensor(8, fx.Int32) for _ in range(_NA)] for _ in range(PF)]

        def _load_a_direct(kb, s):
            for n in range_constexpr(_NA):
                off = a_lane_off[n] + kb * fx.Int32(TILE_K)
                lo = Vec(rocdl.raw_ptr_buffer_load(
                    T.vec(4, T.i32), a_rsrc, off.ir_value(), fx.Int32(0).ir_value(), aux=0
                ))
                hi = Vec(rocdl.raw_ptr_buffer_load(
                    T.vec(4, T.i32), a_rsrc, (off + fx.Int32(16)).ir_value(),
                    fx.Int32(0).ir_value(), aux=0
                ))
                a_ring[s][n].store(
                    Vec.from_elements([lo[i] for i in range(4)] + [hi[i] for i in range(4)], fx.Int32)
                )

        def _load_gmem(kt, s):
            kb = _phys(kt)
            if const_expr(DIRECT_A):
                _load_a_direct(kb, s)
            else:
                a_idx.copy(buf_cp_atom_r, kb, a_cp_frag_bufs[s])
            fx.copy(b_cp_atom, bl_g2r[None, None, None, kb], bl_ret_bufs[s])
            fx.copy(b_cp_atom, br_g2r[None, None, None, kb], br_ret_bufs[s])

        def _scales(kt):
            base = (_phys(kt) - sp * fx.Int32(NTL)) * fx.Int32(2 * BM) + lane16
            fg = [scl_view[base + fx.Int32(n * 16)] for n in range_constexpr(n_reps)]
            fu = [
                scl_view[base + fx.Int32(BM + n * 16)] for n in range_constexpr(n_reps)
            ]
            return fg, fu

        def _write_a_lds(s, lb):
            fx.copy(uni_cp_atom, a_cp_frag_retile_bufs[s], a_lds_w_bufs[lb])

        def _read_a_lds(s):
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_cp_atom,
                    a_lds_r_bufs[s][None, None, ki],
                    a_frag_retile_bufs[s][None, None, ki],
                )

        def _fold(acc, m, n, p, f):
            c = acc[None, m, n].load()
            acc[None, m, n].store(
                Vec.from_elements([fx.fma(p[v], f, c[v]) for v in range(4)], fx.Float32)
            )

        def _mfma_fold(lb, s, fg, fu):
            if const_expr(not wide_k):
                p_gate.fill(0)
                p_up.fill(0)
                fx.gemm(mma_atom, p_gate, bl_frag_bufs[s], a_frag_bufs[lb], p_gate)
                fx.gemm(mma_atom, p_up, br_frag_bufs[s], a_frag_bufs[lb], p_up)
                for n in range_constexpr(n_reps):
                    for m in range_constexpr(m_reps):
                        _fold(c_gate, m, n, p_gate[None, m, n].load(), fg[n])
                        _fold(c_up, m, n, p_up[None, m, n].load(), fu[n])
                return
            # One 16x16x128 MFMA per (operand, m, n) into a fresh 4-register partial
            # (inline zero accumulator), folded into the fp32 accumulator one MFMA later
            # so the fold's VALU issues under the next MFMA (lag-scale-fold-behind-mfma).
            pending = None
            for n in range_constexpr(n_reps):
                if const_expr(DIRECT_A):
                    tok = Vec(a_ring[s][n].load())
                else:
                    # this token rep's A straight from the current LDS buffer: only one rep live
                    fx.copy(
                        uni_cp_atom,
                        a_lds_r_bufs[lb][None, n, 0],
                        a_frag_retile_bufs[lb][None, n, 0],
                    )
                    tok = Vec(a_frag_bufs[lb][None, n, 0].load()).bitcast(fx.Int32)
                for m in range_constexpr(m_reps):
                    for acc, wf, sw in (
                        (c_gate, bl_frag_bufs[s], fg[n]),
                        (c_up, br_frag_bufs[s], fu[n]),
                    ):
                        p = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32),
                            [
                                Vec(wf[None, m, 0].load()).bitcast(fx.Int32),
                                tok,
                                Vec.filled(4, 0.0, fx.Float32),
                                0,
                                0,
                                0,
                                fx.Int32(0),
                                0,
                                fx.Int32(0),
                            ],
                        )
                        if const_expr(pending is not None):
                            _fold(*pending)
                        pending = (acc, m, n, Vec(p), sw)
            _fold(*pending)

        # Ring of PF register sets for the gathered A rows and the W1 tile: tile kt+PF-1
        # is issued while tile kt's MFMAs run, and each wait leaves the PF-2 newest tiles
        # in flight. A goes through an LDS ping-pong one tile ahead of its MFMAs.
        last = fx.Int32(NTL - 1)
        if const_expr(DIRECT_A):
            # W1 needs only expert_id: tile 0's W1 issues first, then the scale and A
            # loads that wait on the sorted id
            kb0 = _phys(0)
            fx.copy(b_cp_atom, bl_g2r[None, None, None, kb0], bl_ret_bufs[0])
            fx.copy(b_cp_atom, br_g2r[None, None, None, kb0], br_ret_bufs[0])
            rocdl.sched_barrier(0)
            scl_vals = _issue_scale_loads()
            _load_a_direct(kb0, 0)
            rocdl.sched_barrier(0)
            for t in range_constexpr(1, PF - 1):
                _load_gmem(min(t, NTL - 1), t)
                rocdl.sched_barrier(0)
        else:
            for t in range_constexpr(PF - 1):
                _load_gmem(min(t, NTL - 1), t)
                rocdl.sched_barrier(0)  # keep tiles in issue order: the vmcnt counts them
        for idx, sa, sw in scl_vals:
            scl_view[idx] = sa * sw
        if const_expr(DIRECT_A):
            gpu.barrier()  # the loop reads scales other threads stored above
            for ktp in range(0, NTL, PF):
                for u in range_constexpr(PF):
                    kt = fx.Int32(ktp) + fx.Int32(u)
                    kn = kt + fx.Int32(PF - 1)
                    kn = (kn < fx.Int32(NTL)).select(kn, last)
                    _load_gmem(kn, (u + PF - 1) % PF)
                    rocdl.s_waitcnt(fxh.encode_waitcnt(vmcnt=(PF - 1) * _loads_per_tile))
                    fg, fu = _scales(kt)
                    _mfma_fold(0, u, fg, fu)
            return c_gate, c_up
        # Rolled K loop, PF tiles per trip so the ring slots and LDS buffers stay static.
        # Prefetches past the last tile are clamped to it (loaded and ignored), so every
        # trip issues the same loads and waits on the same vmcnt. Tile kt's A is written
        # to LDS buffer kt%2 in its own step (one barrier per step: buffer kt%2 was last
        # read two steps ago), so the wait keeps PF-1 tiles in flight.
        for ktp in range(0, NTL, PF):
            for u in range_constexpr(PF):
                kt = fx.Int32(ktp) + fx.Int32(u)
                kn = kt + fx.Int32(PF - 1)
                kn = (kn < fx.Int32(NTL)).select(kn, last)
                _load_gmem(kn, (u + PF - 1) % PF)
                rocdl.s_waitcnt(fxh.encode_waitcnt(vmcnt=(PF - 1) * _loads_per_tile))
                _write_a_lds(u, u % 2)
                gpu.barrier()
                # after the barrier: step 0 reads scales other threads stored above
                fg, fu = _scales(kt)
                if const_expr(not wide_k):
                    _read_a_lds(u % 2)
                _mfma_fold(u % 2, u, fg, fu)
        return c_gate, c_up

    _gemm = ASTRewriter.transform(_gemm)

    def _activate(c_gate, c_up, tw_frag):
        """bf16(act(gate) * up) as fp32, in place in c_gate."""
        m_reps = fxh.reps(c_gate, 1)
        n_reps = fxh.reps(c_gate, 2)
        for n in range_constexpr(n_reps):
            for m in range_constexpr(m_reps):
                g = c_gate[None, m, n].load()
                u = c_up[None, m, n].load()
                out = []
                for v in range_constexpr(4):
                    gv = g[v]
                    uv = u[v]
                    if const_expr(doweight_stage1):
                        gv = gv * tw_frag[0, n]
                        uv = uv * tw_frag[0, n]
                    if const_expr(act == "silu"):
                        e = fx.math.exp(-gv)
                        a = gv * rocdl.rcp(T.f32, (fx.Float32(1.0) + e).ir_value())
                    else:
                        a = (
                            fx.Float32(0.5)
                            * gv
                            * (
                                fx.Float32(1.0)
                                + fx.math.erf(gv * fx.Float32(0.7071067811865476))
                            )
                        )
                    out.append(a * uv)
                h = Vec.from_elements(out, fx.Float32).to(fx.BFloat16).to(fx.Float32)
                c_gate[None, m, n].store(h)

    _activate = ASTRewriter.transform(_activate)

    def _quant_store(h_frag, epi, tid, blk_n, row0, out_rsrc, scale_rsrc):
        """Per sorted row: amax over the 128 channels (lanes l, l^16, l^32, l^48 and four
        waves), scale = amax / 448, q = fp8(h / scale), packed four channels per dword.
        """
        m_reps = fxh.reps(h_frag, 1)
        n_reps = fxh.reps(h_frag, 2)
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)
        col = lane % fx.Int32(16)
        grp = lane // fx.Int32(16)
        amax_view = fx.make_view(epi.amax.ptr, fx.make_layout(4 * 128, 1))
        for n in range_constexpr(n_reps):
            mx = fx.Float32(fxh.AMAX_FLOOR)
            for m in range_constexpr(m_reps):
                h = h_frag[None, m, n].load()
                for v in range_constexpr(4):
                    mx = fx.max(mx, fx.math.absf(h[v]))
            mx = fx.max(mx, mx.shuffle_xor(16, 64))
            mx = fx.max(mx, mx.shuffle_xor(32, 64))
            if grp == fx.Int32(0):
                amax_view[wave * fx.Int32(128) + fx.Int32(n * 16) + col] = mx
        gpu.barrier()
        for n in range_constexpr(n_reps):
            r = fx.Int32(n * 16) + col
            amax = fx.max(
                fx.max(amax_view[r], amax_view[fx.Int32(128) + r]),
                fx.max(amax_view[fx.Int32(256) + r], amax_view[fx.Int32(384) + r]),
            )
            # As aiter's per-group HIP quant (quant_kernels.cu: absMax floored at
            # kAmaxFloor, scale = absMax * (1 / max), q = x * (1 / scale)).
            scale = amax * fx.Float32(fxh.FP8_MAX_RCP)
            inv_scale = fx.Float32(1.0) / scale
            row = row0 + r
            if (wave == fx.Int32(0)) & (grp == fx.Int32(0)):
                rocdl.raw_ptr_buffer_store(
                    scale.bitcast(fx.Int32).ir_value(),
                    scale_rsrc,
                    ((row * fx.Int32(IB) + blk_n) * fx.Int32(4)).ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=ir.IntegerAttr.get(T.i32, 0),
                )
            for m in range_constexpr(m_reps):
                h = h_frag[None, m, n].load()
                q = [h[v] * inv_scale for v in range(4)]
                lo = rocdl.cvt_pk_fp8_f32(
                    res=T.i32, src_a=q[0], src_b=q[1], old=fx.Int32(0), word_sel=False
                )
                pk = rocdl.cvt_pk_fp8_f32(
                    res=T.i32, src_a=q[2], src_b=q[3], old=lo, word_sel=True
                )
                ch = (
                    blk_n * fx.Int32(CHANNELS)
                    + fx.Int32(m * 64)
                    + wave * fx.Int32(16)
                    + grp * fx.Int32(4)
                )
                rocdl.raw_ptr_buffer_store(
                    pk,
                    out_rsrc,
                    (row * fx.Int32(I) + ch).ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=ir.IntegerAttr.get(T.i32, 0),
                )

    _quant_store = ASTRewriter.transform(_quant_store)

    @flyc.kernel
    def fp8b_stage1(
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
        arg_ws: fx.Tensor,
        arg_cnt: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_rows: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y
        split = gpu.block_idx.z
        M = i32_tokens
        in_ptr = fx.recast_iter(elem_t, fx.get_iter(arg_x))
        arg_p_input = fx.make_view(
            in_ptr, fx.make_layout((M, fx.Int32(K)), (fx.Int32(K), 1))
        )
        # num_valid, expert id and this tile's sorted ids load in parallel (all in
        # bounds for every launched tile), not as a chain of dependent round trips
        num_valid = fxh_view_i32(arg_num_valid_ids)[0]
        expert_id = fxh_view_i32(arg_expert_ids)[e_idx]
        row0 = e_idx * fx.Int32(BM)
        sorted_ids = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.recast_iter(fx.Int32, fx.get_iter(arg_sorted_token_ids) + row0),
                fx.make_layout(BM, 1),
            ),
            max_size=False,
        )
        sid_v = sorted_ids[tid % fx.Int32(BM)]
        # expert_id in the condition: its load issues with num_valid's, not after the branch
        if (e_idx * fx.Int32(BM) < num_valid) & (expert_id != fx.Int32(-0x7FFFFFF)):
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.gemm = lds.gemm.peek()
            w_ptr = fx.recast_iter(elem_t, fx.get_iter(arg_w))
            arg_p_weight = fxh.make_gateup_weight_view(
                w_ptr, expert_id, CHANNELS, N_e, K
            )
            # Rows the gather layouts read beyond BM decode to the sentinel M (OOB).
            if const_expr(not DIRECT_A):
                sentinel = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(256, 1))
                sentinel[tid] = (tid < fx.Int32(BM)).select(sid_v, M)
                gpu.barrier()
            a_scale_t = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Float32, fx.get_iter(arg_x_scale)),
                    fx.make_layout(M * fx.Int32(KB), 1),
                ),
                max_size=False,
                num_records_bytes=fx.Int64(M) * fx.Int64(KB * 4),
            )
            w_scale_t = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Float32, fx.get_iter(arg_w_scale)),
                    fx.make_layout(fx.Int32(experts * 2 * IB * KB), 1),
                ),
                max_size=False,
            )
            tw_frag = None
            if const_expr(doweight_stage1):
                tw_frag = fxh.load_sorted_weight_frag(arg_sorted_weights, row0, BM, tid)
            c_gate, c_up = _gemm(
                blk_n,
                arg_p_input,
                arg_p_weight,
                lds,
                M,
                a_scale_t,
                w_scale_t,
                expert_id,
                # rotated K start per (tile, block) (it breaks lockstep HBM channel camping)
                (e_idx * fx.Int32(IB) + blk_n) % fx.Int32(NTL),
                split,
                sid_v,
                sorted_ids,
            )
            if const_expr(SPLITS == 1):
                _epilogue(c_gate, c_up, tw_frag, lds, tid, blk_n, row0, arg_inter,
                          arg_inter_scale, i32_rows)
            else:
                # Each split stores its fp32 partials; the last to arrive at the
                # (tile, block) counter sums all of them in split order (the same
                # sum whichever split is last) and runs the epilogue.
                ws_rsrc = fx.rocdl.get_buffer_rsrc(
                    fx.get_iter(fx.rocdl.make_buffer_tensor(arg_ws, max_size=False))
                )
                pair = e_idx * fx.Int32(IB) + blk_n
                # The coherent stores, waited on, are the release: an agent-scope
                # fence would write back the whole L2 (splitk_epilogue.py's protocol).
                _store_partials(c_gate, c_up, ws_rsrc, pair * fx.Int32(SPLITS) + split, tid)
                rocdl.s_waitcnt(fxh.encode_waitcnt(vmcnt=0))
                gpu.barrier()
                flag = fx.make_view(lds.amax.peek().ptr, fx.make_layout(4 * 128, 1))
                cnt_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_cnt))) + fx.Int64(pair) * fx.Int64(4)
                if tid == fx.Int32(0):
                    old = fx.Int32(comm_ops.atomic_add_agent(cnt_addr, fx.Int32(1)))
                    flag[0] = old.to(fx.Float32)
                gpu.barrier()
                is_last = flag[0] == fx.Float32(SPLITS - 1)
                gpu.barrier()
                if is_last:
                    _sum_partials(c_gate, c_up, ws_rsrc, pair * fx.Int32(SPLITS), tid)
                    if tid == fx.Int32(0):
                        comm_ops.atomic_add_agent(cnt_addr, fx.Int32(-SPLITS))
                    _epilogue(c_gate, c_up, tw_frag, lds, tid, blk_n, row0, arg_inter,
                              arg_inter_scale, i32_rows)

    def _partial_slots(c_gate, c_up):
        return [(acc, m, n) for acc in (c_gate, c_up)
                for n in range_constexpr(fxh.reps(c_gate, 2))
                for m in range_constexpr(fxh.reps(c_gate, 1))]

    def _store_partials(c_gate, c_up, ws_rsrc, slot, tid):
        """This thread's accumulators, vec4 j at ((slot * J + j) * 256 + tid) * 4."""
        slots = _partial_slots(c_gate, c_up)
        for j, (acc, m, n) in enumerate(slots):
            off = ((slot * fx.Int32(len(slots)) + fx.Int32(j)) * fx.Int32(256) + tid) * fx.Int32(4)
            buffer_ops.buffer_store(
                acc[None, m, n].load(), ws_rsrc, off, cache_modifier=CPOL_COHERENT
            )

    def _sum_partials(c_gate, c_up, ws_rsrc, slot0, tid):
        slots = _partial_slots(c_gate, c_up)
        for j, (acc, m, n) in enumerate(slots):
            total = None
            for s in range_constexpr(SPLITS):
                off = (((slot0 + fx.Int32(s)) * fx.Int32(len(slots)) + fx.Int32(j)) * fx.Int32(256)
                       + tid) * fx.Int32(4)
                p = Vec(
                    buffer_ops.buffer_load(
                        ws_rsrc, off, vec_width=4, cache_modifier=CPOL_COHERENT
                    )
                )
                total = p if total is None else total + p
            acc[None, m, n].store(total)

    def _epilogue(c_gate, c_up, tw_frag, lds, tid, blk_n, row0, arg_inter, arg_inter_scale, i32_rows):
        _activate(c_gate, c_up, tw_frag)
        out_rsrc = fx.rocdl.get_buffer_rsrc(
            fx.get_iter(
                fx.rocdl.make_buffer_tensor(
                    arg_inter,
                    max_size=False,
                    num_records_bytes=fx.Int64(i32_rows) * fx.Int64(I),
                )
            )
        )
        scale_rsrc = fx.rocdl.get_buffer_rsrc(
            fx.get_iter(
                fx.rocdl.make_buffer_tensor(
                    arg_inter_scale,
                    max_size=False,
                    num_records_bytes=fx.Int64(i32_rows) * fx.Int64(IB * 4),
                )
            )
        )
        lds.amax = lds.amax.peek()
        gpu.barrier()
        _quant_store(c_gate, lds, tid, blk_n, row0, out_rsrc, scale_rsrc)

    _store_partials = ASTRewriter.transform(_store_partials)
    _sum_partials = ASTRewriter.transform(_sum_partials)
    _epilogue = ASTRewriter.transform(_epilogue)

    def fxh_view_i32(t):
        return fx.make_view(
            fx.recast_iter(fx.Int32, fx.get_iter(t)), fx.make_layout(1 << 20, 1)
        )

    @flyc.jit
    def launch_fp8b_stage1(
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
        fp8b_stage1(
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
            arg_x,  # no split-K: workspace and counters unused
            arg_x,
            i32_tokens,
            i32_rows,
            value_attrs=_NO_PACKED_F32,
        ).launch(grid=(IB, fx.Int64(i32_tiles), 1), block=(256, 1, 1), stream=stream)

    @flyc.jit
    def launch_fp8b_stage1_splitk(
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
        arg_ws: fx.Tensor,
        arg_cnt: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_rows: fx.Int32,
        i32_tiles: fx.Int32,
        stream: fx.Stream,
    ):
        fp8b_stage1(
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
            arg_ws,
            arg_cnt,
            i32_tokens,
            i32_rows,
            value_attrs=_NO_PACKED_F32,
        ).launch(
            grid=(IB, fx.Int64(i32_tiles), SPLITS), block=(256, 1, 1), stream=stream
        )

    return launch_fp8b_stage1 if SPLITS == 1 else launch_fp8b_stage1_splitk


def splitk_workspace_floats(tile_m, tiles, inter_dim, splits):
    """fp32 elements of the split-K workspace: per (tile, block) pair and split, the
    gate and up accumulators of one tile (2 x tile_m x 128 values)."""
    return tiles * (inter_dim // CHANNELS) * splits * 2 * tile_m * CHANNELS


@functools.lru_cache(maxsize=256)
def compile_stage1(
    *,
    model_dim,
    inter_dim,
    experts,
    tile_m,
    act="silu",
    doweight_stage1=False,
    wide_k=True,
    splits=1,
):
    """splits > 1: split-K over hidden into that many workgroups per (tile, block); the
    launcher then also takes the fp32 workspace (splitk_workspace_floats) and one
    int32 counter per (tile, block) pair, zero before the call and zero after it."""
    return _build_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        tile_m=tile_m,
        act=act,
        doweight_stage1=doweight_stage1,
        wide_k=wide_k,
        splits=splits,
    )
