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

from . import helpers as fxh

# keep fp32 folds as VOP2: v_pk_fma_f32 does not issue under an MFMA (lag-scale-fold-behind-mfma)
_NO_PACKED_F32 = {"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]}

CHANNELS = 128  # intermediate channels per workgroup = one quant group


def _build_stage1(
    *, model_dim, inter_dim, experts, tile_m, act, doweight_stage1, wide_k
):
    elem_t = fx.Float8E4M3FN
    K = int(model_dim)
    KB = K // 128
    N_e = 2 * int(inter_dim)
    I = int(inter_dim)
    IB = I // CHANNELS
    BM = int(tile_m)
    TILE_K = 128
    assert K % TILE_K == 0 and KB % 2 == 0, f"model_dim={K} must be a multiple of 256"
    assert I % CHANNELS == 0, f"inter_dim={I} must be a multiple of 128"
    assert BM in (16, 32, 64, 128), f"tile_m={BM}"
    assert act in ("silu", "gelu")

    # The A gather copies 256 // (TILE_K / 16) = 32 rows per pass: a tile_m 16 buffer
    # must still hold 32 rows, or the copy runs into the other ping-pong buffer.
    _gather_rows = 256 // (TILE_K // 16)
    a_lds_size = max(BM, _gather_rows) * TILE_K

    @fx.struct
    class GemmBuffers:
        a_ping: fx.Array[elem_t, a_lds_size, 16]
        a_pong: fx.Array[elem_t, a_lds_size, 16]

    @fx.union
    class SharedStorage:
        sorted_lds: fx.Array[fx.Int32, 256, 16]
        gemm: GemmBuffers
        amax: fx.Array[fx.Float32, 4 * 128, 16]

    _val_per_thr = 16
    _swz_params = (3, 4, 3)
    _thrs_k = TILE_K // _val_per_thr
    _thrs_m = 256 // _thrs_k
    _m_per_wave = _thrs_m // 4

    def _gemm(
        blk_n, arg_p_input, arg_p_weight, lds, M, a_scale_t, w_scale_t, expert_id
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
        a_index_frag = fxh.read_sorted_index(
            tiled_copy_sortid_a, tid, lds.sorted_lds, BM
        )
        a_idx = fxh.TensorWithIndex(
            a_tensor, BM, TILE_K, a_index_frag, a_mem_cp_g2r, tid
        )
        a_mem_thr = a_mem_cp_g2r.get_slice(tid).partition_S(a_tile)
        tok_frag = fxh.token_rep_frag(lds.sorted_lds, BM, tid)
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
            fx.make_fragment_like(a_mem_thr[None, None, None, 0]) for _ in range(2)
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
            tiled_mma.make_fragment_A(bl_tile[None, None, 0]) for _ in range(2)
        ]
        br_frag_bufs = [
            tiled_mma.make_fragment_A(br_tile[None, None, 0]) for _ in range(2)
        ]
        bl_ret_bufs = [b_g2r.retile(f) for f in bl_frag_bufs]
        br_ret_bufs = [b_g2r.retile(f) for f in br_frag_bufs]
        _b_loads_per_tile = 2 * (
            fx.size(fx.get_shape(bl_frag_bufs[0])).to_py_value() // _val_per_thr
        )

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
        # w1_scale [E, 2I/128, KB]: gate block blk_n, up block IB + blk_n.
        sw_gate_base = (expert_id * fx.Int32(2 * IB) + blk_n) * fx.Int32(KB)
        sw_up_base = (expert_id * fx.Int32(2 * IB) + fx.Int32(IB) + blk_n) * fx.Int32(
            KB
        )
        tok_rows = [
            (tok_frag[0, n] & 0xFFFFFF) * fx.Int32(KB) for n in range_constexpr(n_reps)
        ]

        def _load_gmem(kt, s):
            kb = fx.Int32(kt)
            a_idx.copy(buf_cp_atom_r, kb, a_cp_frag_bufs[s])
            fx.copy(buf_cp_atom_r, bl_g2r[None, None, None, kb], bl_ret_bufs[s])
            fx.copy(buf_cp_atom_r, br_g2r[None, None, None, kb], br_ret_bufs[s])

        def _scales(kt):
            kb = fx.Int32(kt)
            sg = w_scale_t[sw_gate_base + kb]
            su = w_scale_t[sw_up_base + kb]
            sa = [a_scale_t[tok_rows[n] + kb] for n in range_constexpr(n_reps)]
            return sa, sg, su

        def _write_a_lds(s):
            fx.copy(uni_cp_atom, a_cp_frag_retile_bufs[s], a_lds_w_bufs[s])

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

        def _mfma_fold(s, sa, sg, su):
            if const_expr(not wide_k):
                p_gate.fill(0)
                p_up.fill(0)
                fx.gemm(mma_atom, p_gate, bl_frag_bufs[s], a_frag_bufs[s], p_gate)
                fx.gemm(mma_atom, p_up, br_frag_bufs[s], a_frag_bufs[s], p_up)
                for n in range_constexpr(n_reps):
                    for m in range_constexpr(m_reps):
                        _fold(c_gate, m, n, p_gate[None, m, n].load(), sa[n] * sg)
                        _fold(c_up, m, n, p_up[None, m, n].load(), sa[n] * su)
                return
            # One 16x16x128 MFMA per (operand, m, n) into a fresh 4-register partial
            # (inline zero accumulator), folded into the fp32 accumulator one MFMA later
            # so the fold's VALU issues under the next MFMA (lag-scale-fold-behind-mfma).
            pending = None
            for n in range_constexpr(n_reps):
                # this token rep's A straight from the current LDS buffer: only one rep live
                fx.copy(
                    uni_cp_atom,
                    a_lds_r_bufs[s][None, n, 0],
                    a_frag_retile_bufs[s][None, n, 0],
                )
                tok = Vec(a_frag_bufs[s][None, n, 0].load()).bitcast(fx.Int32)
                for m in range_constexpr(m_reps):
                    for acc, wf, sw in (
                        (c_gate, bl_frag_bufs[s], sg),
                        (c_up, br_frag_bufs[s], su),
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
                        pending = (acc, m, n, Vec(p), sa[n] * sw)
            _fold(*pending)

        _load_gmem(0, 0)
        rocdl.s_waitcnt(fxh.encode_waitcnt(vmcnt=_b_loads_per_tile))
        _write_a_lds(0)
        gpu.barrier()
        if const_expr(not wide_k):
            _read_a_lds(0)
        # Rolled K loop, two tiles per trip so the ping-pong buffers stay static. The
        # prefetch past the last tile is clamped to it (loaded and ignored), so every
        # trip issues the same loads and waits on the same vmcnt.
        last = fx.Int32(num_tiles - 1)
        for kt2 in range(0, num_tiles, 2):
            for u in range_constexpr(2):
                kt = fx.Int32(kt2) + fx.Int32(u)
                sa, sg, su = _scales(kt)
                kn = kt + fx.Int32(1)
                kn = (kn < fx.Int32(num_tiles)).select(kn, last)
                _load_gmem(kn, 1 - u)
                rocdl.s_waitcnt(fxh.encode_waitcnt(vmcnt=_b_loads_per_tile))
                _write_a_lds(1 - u)
                _mfma_fold(u, sa, sg, su)
                gpu.barrier()
                if const_expr(not wide_k):
                    _read_a_lds(1 - u)
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
        i32_tokens: fx.Int32,
        i32_rows: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y
        M = i32_tokens
        in_ptr = fx.recast_iter(elem_t, fx.get_iter(arg_x))
        arg_p_input = fx.make_view(
            in_ptr, fx.make_layout((M, fx.Int32(K)), (fx.Int32(K), 1))
        )
        num_valid = fxh_view_i32(arg_num_valid_ids)[0]
        if e_idx * fx.Int32(BM) < num_valid:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.gemm = lds.gemm.peek()
            row0 = e_idx * fx.Int32(BM)
            sorted_ids = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Int32, fx.get_iter(arg_sorted_token_ids) + row0),
                    fx.make_layout(BM, 1),
                ),
                max_size=False,
            )
            expert_id = fxh_view_i32(arg_expert_ids)[e_idx]
            w_ptr = fx.recast_iter(elem_t, fx.get_iter(arg_w))
            arg_p_weight = fxh.make_gateup_weight_view(
                w_ptr, expert_id, CHANNELS, N_e, K
            )
            # Rows the gather layouts read beyond BM decode to the sentinel M (OOB).
            sentinel = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(256, 1))
            sentinel[tid] = M
            gpu.barrier()
            if tid < fx.Int32(BM):
                lds_view = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BM, 1))
                lds_view[tid] = sorted_ids[tid]
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
            )
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
            i32_tokens,
            i32_rows,
            value_attrs=_NO_PACKED_F32,
        ).launch(grid=(IB, fx.Int64(i32_tiles), 1), block=(256, 1, 1), stream=stream)

    return launch_fp8b_stage1


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
):
    return _build_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        tile_m=tile_m,
        act=act,
        doweight_stage1=doweight_stage1,
        wide_k=wide_k,
    )
