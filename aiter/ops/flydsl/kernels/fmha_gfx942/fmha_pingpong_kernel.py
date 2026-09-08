# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opt-in gfx942 packed-varlen FMHA ping-pong experiment.

Two four-wave groups use group-private K/V LDS and a two-rendezvous phase
skew copied from the FP8 ping-pong skeleton: one cohort streams PV+QK MFMA
while the partner runs softmax VALU. Mid-plane K barriers are removed; K
planes 3..5 DMA into already-consumed 3-slot buffers and become visible at
one rendezvous. The exact softmax is the matched control.
``coupled_softmax=True`` selects the declared mixed-exp plus BF16
MFMA-denominator treatment. This module is never used by production dispatch.
"""

from __future__ import annotations

import math as host_math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.fmha_gfx942.config import (
    BLOCK_SIZE,
    D_CHUNK,
    K_DMA_ISSUE_STRIDE_BYTES,
    K_DMA_WARP_STRIDE_BYTES,
    K_PLANE_ELEMS,
    K_STAGE_COUNT,
    K_SUB_N,
    MFMA_K,
    SUPPORTED_DTYPES,
    SUPPORTED_GFX,
    WARP_SIZE,
    WAVES_PER_EU as DEFAULT_WAVES_PER_EU,
    k_lds_elems,
    validate_fmha_gfx942_tiles,
    v_lds_elems,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

_LOG2E = host_math.log2(host_math.e)
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3

_DTYPE_TO_STR = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
}


def _llvm_value(value):
    """Unwrap FlyDSL scalar/vector wrappers for LLVM pointer load ops."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (
        (n & _VMCNT_LO_MASK)
        | _LGKMCNT_EXPCNT_BASE
        | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    )
    rocdl.s_waitcnt(val)


def _lds_handoff_barrier():
    """Drain LDS traffic and publish it across waves without draining VMEM."""
    llvm.inline_asm(
        None,
        [],
        "s_waitcnt lgkmcnt(0)\ns_barrier",
        "~{memory}",
        has_side_effects=True,
    )


def build_fmha_pingpong_gfx942_module(
    num_heads,
    head_dim_qk,
    head_dim_v,
    causal,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=DEFAULT_WAVES_PER_EU,
    num_kv_heads=None,
    return_lse=False,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    dualwave_coupled_softmax=False,
    rotate_pv_accumulators=False,
    vop2_o_rescale=False,
):
    """Build the fixed-shape packed-THD ping-pong launcher for gfx942."""
    gpu_arch = get_hip_arch()
    arch_base = str(gpu_arch).split(":")[0]
    if not any(arch_base.startswith(g) for g in SUPPORTED_GFX):
        raise ValueError(f"fmha_varlen_gfx942 requires gfx942, got {gpu_arch!r}")
    if dtype_str not in SUPPORTED_DTYPES:
        raise ValueError(
            f"fmha_varlen_gfx942 supports {SUPPORTED_DTYPES}, got {dtype_str!r}"
        )
    if num_kv_heads is None:
        num_kv_heads = num_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    validate_fmha_gfx942_tiles(
        head_dim_qk, head_dim_v, block_m=256, block_n=64
    )

    DUALWAVE_SWP = True
    DUALWAVE_COUPLED_SOFTMAX = bool(dualwave_coupled_softmax)
    ROTATE_PV_ACCUMULATORS = bool(rotate_pv_accumulators)
    VOP2_O_RESCALE = bool(vop2_o_rescale)
    if (
        dtype_str != "bf16"
        or int(num_heads) != 12
        or int(num_kv_heads) != 12
        or int(head_dim_qk) != 192
        or int(head_dim_v) != 128
        or bool(causal)
    ):
        raise ValueError(
            "gfx942 ping-pong requires non-causal packed bf16 H=Hkv=12 "
            "QK=192 V=128 BLOCK_N=64"
        )

    HEAD_DIM_QK = int(head_dim_qk)
    HEAD_DIM_V = int(head_dim_v)
    BLOCK_M = 256
    BLOCK_N = 64
    RETURN_LSE = bool(return_lse)
    CAUSAL = bool(causal)
    NUM_HEADS_Q = int(num_heads)
    NUM_HEADS_KV = int(num_kv_heads)
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    flat_work_group_size = 512
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    WAVES_PER_GROUP = 4

    REDUCE_MODE = "xor"

    K_STEP_QK = MFMA_K
    K_STEPS_QK = HEAD_DIM_QK // K_STEP_QK
    D_CHUNKS = HEAD_DIM_V // D_CHUNK
    PV_K_STEP = MFMA_K
    PV_K_STEPS = K_SUB_N // PV_K_STEP

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(HEAD_DIM_QK)

    STRIDE_TOKEN_Q = NUM_HEADS_Q * HEAD_DIM_QK
    STRIDE_TOKEN_K = NUM_HEADS_KV * HEAD_DIM_QK
    STRIDE_TOKEN_V = NUM_HEADS_KV * HEAD_DIM_V
    STRIDE_TOKEN_O = NUM_HEADS_Q * HEAD_DIM_V

    K_PLANES = HEAD_DIM_QK // 32
    K_BUFFER_COUNT = 3
    # Two workgroup rendezvous per KV tile, matching the FP8 ping-pong
    # skeleton: one publishes K planes 3..5 after QK 0..2, one closes the
    # phase after QK 3..5 / V+K prefetch. Mid-plane barriers are removed
    # because those DMAs land in already-consumed 3-slot buffers.
    PHASE_BARRIERS = 2
    V_D64_TILES = HEAD_DIM_V // 64
    V_KGROUP_STRIDE = (HEAD_DIM_V // 8) * 72
    V_HALF_TILE_SIZE = 4 * V_KGROUP_STRIDE

    LDS_K_TILE_SIZE = k_lds_elems(HEAD_DIM_QK)
    LDS_K_GROUP_SIZE = K_PLANE_ELEMS * K_BUFFER_COUNT
    LDS_V_TILE_SIZE = v_lds_elems(HEAD_DIM_V, BLOCK_N)

    k_allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name="fmha_varlen_gfx942_k_smem",
    )
    LDS_GROUPS = 2
    k_allocator.ptr = LDS_K_GROUP_SIZE * 2 * LDS_GROUPS
    v_allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name="fmha_varlen_gfx942_v_smem",
    )
    v_allocator.ptr = LDS_V_TILE_SIZE * 2 * LDS_GROUPS

    @flyc.kernel(known_block_size=[flat_work_group_size, 1, 1])
    def fmha_pingpong_gfx942_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        cu_seqlens_k: fx.Tensor,
        max_seqlen_q: fx.Int32,
        total_q: fx.Int32,
    ):
        elem_dtype = fx.BFloat16 if dtype_str == "bf16" else fx.Float16
        elem_type = elem_dtype.ir_type
        compute_type = fx.Float32.ir_type

        # All FP operations use aggressive fast-math (no NaN/Inf checks, reassociation).
        # The unsafe_fp_math/fast_fp_math builder params control LLVM-level attributes only.
        fm_fast = fx.arith.FastMathFlags.fast
        v4f16_type = Vec.make_type(4, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v4f16_type
        MFMA_LANE_K = 4

        def _mfma(mfma_fn, a, b, c):
            return mfma_fn(v16f32_type, [a, b, c])

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fadd_vop2(a, b):
            # Non-packed VOP2: v_pk_add_f32 contends with the partner MFMA.
            return llvm.InlineAsmOp(
                T.f32,
                [_raw(a), _raw(b)],
                "v_add_f32_e32 $0, $1, $2",
                "=v,v,v",
                has_side_effects=False,
            ).res

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul_vop2(a, b):
            return llvm.InlineAsmOp(
                T.f32,
                [_raw(a), _raw(b)],
                "v_mul_f32_e32 $0, $1, $2",
                "=v,v,v",
                has_side_effects=False,
            ).res

        def _pin_f32(v):
            return llvm.InlineAsmOp(
                T.f32,
                [_raw(v)],
                "",
                "=v,0",
                has_side_effects=False,
            ).res

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        def mfma_acc(a, b, c):
            if const_expr(dtype_str == "bf16"):
                a = Vec(a).bitcast(fx.Int16)
                b = Vec(b).bitcast(fx.Int16)
                return _mfma(rocdl.mfma_f32_32x32x8bf16_1k, a, b, c)
            return _mfma(rocdl.mfma_f32_32x32x8f16, a, b, c)

        max_sq = fx.Index(max_seqlen_q)
        total_q_v = fx.Index(total_q)

        # ---- LDS view ----
        lds_k = SmemPtr(
            k_allocator.get_base(),
            0,
            elem_type,
            shape=(LDS_K_GROUP_SIZE * LDS_GROUPS,),
        ).get()
        lds_v = SmemPtr(
            v_allocator.get_base(),
            0,
            elem_type,
            shape=(LDS_V_TILE_SIZE * LDS_GROUPS,),
        ).get()

        # ---- Thread / block indices ----
        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)

        # ---- Wave decomposition ----
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32  # 0/1
        if const_expr(DUALWAVE_SWP):
            wave_group = wave_id // fx.Index(WAVES_PER_GROUP)
            wave_in_group = wave_id % fx.Index(WAVES_PER_GROUP)
        else:
            # Compile-time constants preserve the original four-wave path.
            wave_group = fx.Index(0)
            wave_in_group = wave_id

        wave_q_offset = wave_id * ROWS_PER_WAVE

        q_head_idx = block_id % NUM_HEADS_Q
        batch_q_tile_id = block_id // NUM_HEADS_Q
        num_q_tiles = (max_sq + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M
        kv_head_idx = q_head_idx if GQA_GROUP_SIZE == 1 else q_head_idx // GQA_GROUP_SIZE

        cuq_rsrc = buffer_ops.create_buffer_resource(cu_seqlens_q, max_size=True)
        cuk_rsrc = buffer_ops.create_buffer_resource(cu_seqlens_k, max_size=True)
        q0_i32 = fx.Int32(buffer_ops.buffer_load(cuq_rsrc, fx.Int32(batch_idx), vec_width=1, dtype=fx.Int32))
        q1_i32 = fx.Int32(buffer_ops.buffer_load(cuq_rsrc, fx.Int32(batch_idx) + fx.Int32(1), vec_width=1, dtype=fx.Int32))
        k0_i32 = fx.Int32(buffer_ops.buffer_load(cuk_rsrc, fx.Int32(batch_idx), vec_width=1, dtype=fx.Int32))
        k1_i32 = fx.Int32(buffer_ops.buffer_load(cuk_rsrc, fx.Int32(batch_idx) + fx.Int32(1), vec_width=1, dtype=fx.Int32))
        q0 = fx.Index(q0_i32)
        k0 = fx.Index(k0_i32)
        q_len = fx.Index(q1_i32 - q0_i32)
        k_len = fx.Index(k1_i32 - k0_i32)
        seq_len_v = k_len
        q_len_i32 = fx.Int32(q_len)
        k_len_i32 = fx.Int32(k_len)

        def global_idx_q(token_idx, col):
            return token_idx * STRIDE_TOKEN_Q + q_head_idx * HEAD_DIM_QK + col

        def global_idx_v(token_idx, col):
            return token_idx * STRIDE_TOKEN_V + kv_head_idx * HEAD_DIM_V + col

        def global_idx_o(token_idx, col):
            return token_idx * STRIDE_TOKEN_O + q_head_idx * HEAD_DIM_V + col

        def _bitcast_i32(value):
            return fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))

        def bf16_trunc_pack_v4(f32_vals):
            """Pack four f32 to bf16 with ASM's two v_perm_b32 sequence."""
            sel = 0x07060302
            u = [_bitcast_i32(v) for v in f32_vals]
            packed = [
                fx.Int32(rocdl.perm_b32(u[1], u[0], sel)),
                fx.Int32(rocdl.perm_b32(u[3], u[2], sel)),
            ]
            return (
                Vec.from_elements(packed, fx.Int32)
                .bitcast(elem_dtype)
                .ir_value()
            )

        def k_buf_base(buf_id):
            group_base = wave_group * fx.Index(LDS_K_GROUP_SIZE)
            if const_expr(isinstance(buf_id, int)):
                return group_base + fx.Index(buf_id * K_PLANE_ELEMS)
            return group_base + buf_id * fx.Index(K_PLANE_ELEMS)

        def v_buf_base(buf_id):
            return (
                wave_group * fx.Index(LDS_V_TILE_SIZE)
                + fx.Index(buf_id * LDS_V_TILE_SIZE)
            )

        def _v_lds_elem_offset(n, d):
            """Padded native V layout, generalized from D=64 to HEAD_DIM_V."""
            n_half = n // fx.Index(32)
            k = n % fx.Index(32)
            return (
                n_half * fx.Index(V_HALF_TILE_SIZE)
                + (k // fx.Index(8)) * fx.Index(V_KGROUP_STRIDE)
                + (d // fx.Index(8)) * fx.Index(72)
                + (d % fx.Index(8)) * fx.Index(8)
                + (k % fx.Index(8))
            )

        # K buffer resource for DMA-to-LDS (byte offen). Batch base is k0.
        # k0/k_len are wave-uniform but live in VGPRs after the cu_seqlens load;
        # readfirstlane so MakeBufferRsrc stays SGPR-uniform (no waterfall on
        # every buffer_load_lds — that alone dominated the DMA path in ISA).
        k0_u = fx.Index(rocdl.readfirstlane(T.i32, fx.Int32(k0)))
        k_len_u = fx.Index(rocdl.readfirstlane(T.i32, fx.Int32(k_len)))
        _k_nrec_bytes = _raw(k_len_u * fx.Index(STRIDE_TOKEN_K * 2))
        _k_batch_byte_off = _raw(k0_u * fx.Index(STRIDE_TOKEN_K * 2))
        k_rsrc = buffer_ops.create_buffer_resource(
            K,
            max_size=False,
            num_records_bytes=_k_nrec_bytes,
            base_byte_offset=_k_batch_byte_off,
        )
        _v_nrec_bytes = _raw(k_len_u * fx.Index(STRIDE_TOKEN_V * 2))
        _v_batch_byte_off = _raw(k0_u * fx.Index(STRIDE_TOKEN_V * 2))
        v_rsrc = buffer_ops.create_buffer_resource(
            V,
            max_size=False,
            num_records_bytes=_v_nrec_bytes,
            base_byte_offset=_v_batch_byte_off,
        )
        k_lds_base_i64 = arith.index_cast(
            T.i64, memref_dialect.extract_aligned_pointer_as_index(lds_k)
        )
        k_lds_base_i64 = rocdl.readfirstlane(T.i64, k_lds_base_i64)
        v_lds_base_i32 = arith.index_cast(
            T.i32, memref_dialect.extract_aligned_pointer_as_index(lds_v)
        )
        _lds_ptr3_ty = ir.Type.parse("!llvm.ptr<3>")
        # wave_id is uniform within a wave; pin to SGPR for m0 addressing.
        dma_wave_id = wave_in_group if DUALWAVE_SWP else wave_id
        wave_id_u = fx.Index(rocdl.readfirstlane(T.i32, fx.Int32(dma_wave_id)))

        def _swizzle_a(x):
            """Runtime SwizzleA: swap bits 2 and 3 of the N-column index."""
            b2 = (x >> fx.Index(2)) & fx.Index(1)
            b3 = (x >> fx.Index(3)) & fx.Index(1)
            # Clear bits 2–3 (0xC). Use a positive mask — Python ~0xC is -13.
            return (x & fx.Index(0xFFFFFFF3)) | (b2 << fx.Index(3)) | (b3 << fx.Index(2))

        def _lds_elem_offset(j, d):
            """ASM K LDS bf16 offset for row ``j``, headdim ``d``."""
            return (
                (j % fx.Index(4)) * fx.Index(136)
                + ((j // fx.Index(4)) % fx.Index(4)) * fx.Index(32)
                + (j // fx.Index(16)) * fx.Index(544)
                + (d % fx.Index(32))
                + (d // fx.Index(32)) * fx.Index(K_PLANE_ELEMS)
            )

        def coop_dma_k_plane(tile_start, plane, buf_id):
            """DMA one 64x32 K plane into a ping-pong LDS stage."""
            lane_id = lane
            warp_id_i = wave_id_u
            d_in_chunk = (lane_id & fx.Index(15)) * fx.Index(2)
            n_base = (lane_id >> fx.Index(4)) * fx.Index(4) + warp_id_i
            stride_bytes = fx.Int32(STRIDE_TOKEN_K * 2)
            # tile_start may be past kv_upper on the last prefetch; clamp via
            # buffer OOB (num_records) — invalid rows read as 0.
            tile_start_u = fx.Index(rocdl.readfirstlane(T.i32, fx.Int32(tile_start)))
            kv_byte_base = fx.Int32(tile_start_u * fx.Index(STRIDE_TOKEN_K * 2))
            kv_head_byte = fx.Int32(kv_head_idx * HEAD_DIM_QK * 2)
            buf_byte = fx.Int32(k_buf_base(buf_id) * fx.Index(2))
            n_base_i32 = fx.Int32(n_base)
            d_in_chunk_i32 = fx.Int32(d_in_chunk)

            k_col_offset = plane * 32
            m0_base = (
                buf_byte
                + fx.Int32(warp_id_i * fx.Index(K_DMA_WARP_STRIDE_BYTES))
            )
            voffset = (
                kv_byte_base
                + n_base_i32 * stride_bytes
                + kv_head_byte
                + fx.Int32(k_col_offset * 2)
                + d_in_chunk_i32 * fx.Int32(2)
            )
            for issue in range_constexpr(4):
                m0_bytes = m0_base + fx.Int32(
                    issue * K_DMA_ISSUE_STRIDE_BYTES
                )
                lds_addr = rocdl.readfirstlane(
                    T.i64, k_lds_base_i64 + fx.Int64(m0_bytes)
                )
                lds_ptr = llvm.inttoptr(_lds_ptr3_ty, lds_addr)
                rocdl.raw_ptr_buffer_load_lds(
                    k_rsrc,
                    lds_ptr,
                    fx.Int32(4),
                    voffset,
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32(0),
                )
                voffset = voffset + fx.Int32(16) * stride_bytes

        def coop_load_k(tile_start, plane=0, buf_id=0):
            coop_dma_k_plane(tile_start, plane, buf_id)

        def _ds_write2_b32(addr, value0, value1, offset0, offset1):
            llvm.inline_asm(
                None,
                [_llvm_value(addr), _llvm_value(value0), _llvm_value(value1)],
                (
                    f"ds_write2_b32 $0, $1, $2 "
                    f"offset0:{offset0} offset1:{offset1}"
                ),
                "v,v,v",
                has_side_effects=True,
            )

        def coop_load_v_global(tile_start, n_half):
            vecs = []
            # Native mapping: each wave owns 8 rows. Each lane loads two
            # adjacent rows x four contiguous D values (two b64 loads).
            n_hdim = lane >> fx.Index(2)
            k_within = lane & fx.Index(3)
            row_in_wave = dma_wave_id * fx.Index(8) + k_within * fx.Index(2)
            row0 = tile_start + fx.Index(n_half * 32) + row_in_wave
            row1 = row0 + fx.Index(1)
            for d64 in range_constexpr(V_D64_TILES):
                col = fx.Index(d64 * 64) + n_hdim * fx.Index(4)
                vecs.append(
                    (
                        buffer_ops.buffer_load(
                            v_rsrc,
                            global_idx_v(row0, col),
                            vec_width=4,
                            dtype=elem_dtype,
                        ),
                        buffer_ops.buffer_load(
                            v_rsrc,
                            global_idx_v(row1, col),
                            vec_width=4,
                            dtype=elem_dtype,
                        ),
                    )
                )
            return vecs

        def coop_store_v_lds(vecs, n_half, buf_id=0):
            v_base = v_buf_base(buf_id)
            n_hdim = lane >> fx.Index(2)
            k_within = lane & fx.Index(3)
            k_in_half = dma_wave_id * fx.Index(8) + k_within * fx.Index(2)
            k_pos = fx.Index(n_half * 32) + k_in_half
            for d64 in range_constexpr(V_D64_TILES):
                load0, load1 = vecs[d64]
                in0 = Vec(load0).bitcast(fx.Int32)
                in1 = Vec(load1).bitcast(fx.Int32)
                out0 = fx.Int32(
                    rocdl.perm_b32(in0[0], in1[0], 0x01000504)
                )
                out1 = fx.Int32(
                    rocdl.perm_b32(in0[0], in1[0], 0x03020706)
                )
                out2 = fx.Int32(
                    rocdl.perm_b32(in0[1], in1[1], 0x01000504)
                )
                out3 = fx.Int32(
                    rocdl.perm_b32(in0[1], in1[1], 0x03020706)
                )
                d_pos = fx.Index(d64 * 64) + n_hdim * fx.Index(4)
                addr = fx.Int32(
                    v_lds_base_i32
                    + fx.Int32(
                        (
                            v_base + _v_lds_elem_offset(k_pos, d_pos)
                        )
                        * fx.Index(2)
                    )
                )
                # Consecutive D values are 8 bf16 elements (16 bytes)
                # apart in the padded transpose.
                _ds_write2_b32(addr, out0, out1, 0, 4)
                _ds_write2_b32(addr, out2, out3, 8, 12)

        _q_nrec_bytes = _raw(q_len * fx.Index(STRIDE_TOKEN_Q * 2))
        _o_nrec_bytes = _raw(q_len * fx.Index(STRIDE_TOKEN_O * 2))
        _q_batch_byte_off = _raw(q0 * fx.Index(STRIDE_TOKEN_Q * 2))
        _o_batch_byte_off = _raw(q0 * fx.Index(STRIDE_TOKEN_O * 2))
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_q_nrec_bytes, base_byte_offset=_q_batch_byte_off
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=_o_nrec_bytes, base_byte_offset=_o_batch_byte_off
        )
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)

        # ---- Preload Q^T B-operand packs once (register-resident) ----
        # B operand: j = lane_mod_32, k-subblock = lane_div_32*MFMA_LANE_K. Q is
        # num_records-bounded (q_rsrc) so OOB rows read 0 -- no q_in_bounds select.
        # Reload-at-use was measured slower on 4096x8192 (2.32 ms vs 1.56 ms)
        # and did not raise occupancy.
        q_row = q_start + wave_q_offset + lane_mod_32
        q_row_i32 = fx.Int32(q_row)
        causal_bound_i32 = q_row_i32 + k_len_i32 - q_len_i32
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            g_idx = global_idx_q(q_row, q_col)
            q_b_packs.append(buffer_ops.buffer_load(q_rsrc, g_idx, vec_width=MFMA_LANE_K, dtype=elem_dtype))

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        c_sch_score_scale = fx.Float32(
            sm_scale * _LOG2E * float(1 << 23)
        )
        c_sch_pattern_bias = fx.Float32(
            float(127 * (1 << 23) - 486411)
        )
        c_sch_scale = fx.Float32(float(1 << 23))

        def approx_exp2_score(score, neg_scaled_max):
            pattern_bias = fmath.fma(
                neg_scaled_max,
                c_sch_scale,
                c_sch_pattern_bias,
                fastmath=fm_fast,
            )
            pattern = fmath.fma(
                score,
                c_sch_score_scale,
                pattern_bias,
                fastmath=fm_fast,
            )
            pattern = _fmax(pattern, c_zero_f)
            bits = arith.fptoui(T.i32, _raw(pattern))
            return fx.Float32(ArithValue(bits).bitcast(compute_type))

        l_ones_pack = Vec.filled(4, 1.0, elem_dtype).ir_value()

        def denominator_mfma_half(p_vals):
            """Reduce 32 probabilities with two independent 2-MFMA chains."""
            p_packs = []
            for pks in range_constexpr(PV_K_STEPS):
                p_base = pks * 4
                p_packs.append(
                    bf16_trunc_pack_v4(p_vals[p_base : p_base + 4])
                )

            acc0 = c_zero_v16f32
            for pks in range_constexpr(2):
                acc0 = mfma_acc(l_ones_pack, p_packs[pks], acc0)
            sum0 = Vec(acc0)[0]

            acc1 = c_zero_v16f32
            for pks in range_constexpr(2):
                acc1 = mfma_acc(l_ones_pack, p_packs[pks + 2], acc1)
            sum1 = Vec(acc1)[0]
            return p_packs, _fadd(sum0, sum1)

        width_i32 = fx.Int32(WARP_SIZE)
        shuf_32_i32 = fx.Int32(32)
        c4_i32 = fx.Int32(4)
        lane_i32 = fx.Int32(lane)
        lane_xor_32_i32 = lane_i32 ^ shuf_32_i32
        lane_xor_32_byte = lane_xor_32_i32 * c4_i32

        def reduction_peer(v_f32):
            if const_expr(REDUCE_MODE == "ds_bpermute"):
                v_i32 = fx.Int32(ArithValue(v_f32).bitcast(fx.Int32.ir_type))
                peer_i32 = rocdl.ds_bpermute(fx.Int32.ir_type, lane_xor_32_byte, v_i32)
                return fx.Float32(ArithValue(peer_i32).bitcast(compute_type))
            return fx.Float32(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        def _phase_barrier():
            """One compiler-fenced, LDS-visible full-workgroup rendezvous."""
            rocdl.sched_barrier(0)
            _lds_handoff_barrier()
            rocdl.sched_barrier(0)

        wave_group_i32 = rocdl.readfirstlane(T.i32, fx.Int32(wave_group))

        def _wait_one_full_phase(wait_group):
            """Conditionally hold one wave group for one complete phase."""
            barriers = "\n".join("s_barrier" for _ in range(PHASE_BARRIERS))
            llvm.inline_asm(
                None,
                [wave_group_i32],
                (
                    f"s_cmp_eq_u32 $0, {wait_group}\n"
                    "s_cbranch_scc0 1f\n"
                    f"{barriers}\n"
                    "1:"
                ),
                "s,~{memory}",
                has_side_effects=True,
            )

        def _phase_pad(count):
            if const_expr(DUALWAVE_SWP):
                for _ in range_constexpr(count):
                    _phase_barrier()

        v_slot = 0
        v_base = v_buf_base(v_slot)
        _steps = (
            [
                (dc, pks)
                for pks in range(PV_K_STEPS)
                for dc in range(D_CHUNKS)
            ]
            if ROTATE_PV_ACCUMULATORS
            else [
                (dc, pks)
                for dc in range(D_CHUNKS)
                for pks in range(PV_K_STEPS)
            ]
        )

        def _read_v_pack(step_idx, n_half):
            dc, pks = _steps[step_idx]
            d_pos = fx.Index(dc * D_CHUNK) + lane_mod_32
            k_base_v = (
                fx.Index((pks // 2) * 16)
                + lane_div_32 * fx.Index(8)
                + fx.Index((pks % 2) * 4)
                + fx.Index(n_half * K_SUB_N)
            )
            v_idx = v_base + _v_lds_elem_offset(k_base_v, d_pos)
            return Vec.load(v4f16_type, lds_v, [v_idx])

        def _apply_pv(p_packs_lo, p_packs_hi, o_accs, corr_vec):
            if const_expr(CAUSAL):
                for n_half in range_constexpr(2):
                    p_packs = p_packs_lo if n_half == 0 else p_packs_hi
                    v_cur = _read_v_pack(0, n_half)
                    for si in range_constexpr(len(_steps)):
                        dc, pks = _steps[si]
                        if const_expr(si + 1 < len(_steps)):
                            v_nxt = _read_v_pack(si + 1, n_half)
                        o_accs[dc] = mfma_acc(
                            v_cur, p_packs[pks], o_accs[dc]
                        )
                        if const_expr(
                            not DUALWAVE_SWP
                            and n_half == 0
                            and D_CHUNKS <= PV_K_STEPS
                            and dc == 0
                            and pks < D_CHUNKS - 1
                        ):
                            o_accs[pks + 1] = Vec(o_accs[pks + 1]) * corr_vec
                        if const_expr(si + 1 < len(_steps)):
                            v_cur = v_nxt

                    for _ in range_constexpr(4):
                        rocdl.sched_vmem(1)
                        rocdl.sched_dsrd(2)
                        rocdl.sched_mfma(4)
                    rocdl.sched_barrier(0)
            else:
                v_lo_cur = _read_v_pack(0, 0)
                v_hi_cur = _read_v_pack(0, 1)
                for si in range_constexpr(len(_steps)):
                    dc, pks = _steps[si]
                    if const_expr(si + 1 < len(_steps)):
                        v_lo_nxt = _read_v_pack(si + 1, 0)
                        v_hi_nxt = _read_v_pack(si + 1, 1)
                    o_accs[dc] = mfma_acc(
                        v_lo_cur, p_packs_lo[pks], o_accs[dc]
                    )
                    o_accs[dc] = mfma_acc(
                        v_hi_cur, p_packs_hi[pks], o_accs[dc]
                    )
                    if const_expr(
                        not DUALWAVE_SWP
                        and D_CHUNKS <= PV_K_STEPS
                        and dc == 0
                        and pks < D_CHUNKS - 1
                    ):
                        o_accs[pks + 1] = Vec(o_accs[pks + 1]) * corr_vec
                    if const_expr(si + 1 < len(_steps)):
                        v_lo_cur = v_lo_nxt
                        v_hi_cur = v_hi_nxt

                for pv_group in range_constexpr(8):
                    if const_expr(pv_group % 2 == 0):
                        rocdl.sched_vmem(1)
                    rocdl.sched_dsrd(2)
                    rocdl.sched_mfma(4)
                rocdl.sched_barrier(0)
            return o_accs

        # ---- KV loop upper bound ----
        _q_end = q_start + BLOCK_M
        _has_q = q_start < q_len
        if const_expr(CAUSAL):
            _causal_end = _q_end + (seq_len_v - q_len)
            _kv_cap = fx.Index(ArithValue(_causal_end < seq_len_v).select(_causal_end, seq_len_v))
        else:
            _kv_cap = seq_len_v
        kv_upper = fx.Index(ArithValue(_has_q).select(_kv_cap, fx.Index(0)))

        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        if const_expr(DUALWAVE_SWP):
            zero_p_pack = Vec.filled(4, 0.0, elem_dtype).ir_value()
            for _ in range_constexpr(PV_K_STEPS * 2):
                init_args.append(zero_p_pack)

        loop_results = init_args
        _kv_loop_start = fx.Index(0)
        _kv_loop_step = fx.Index(BLOCK_N)
        if const_expr(DUALWAVE_SWP):
            # The first delayed PV is algebraically zero. Initialize its private
            # V tile as well, avoiding any dependence on undefined LDS bits.
            zero_v4 = Vec.filled(4, 0.0, elem_dtype)
            group_tid = wave_in_group * fx.Index(WARP_SIZE) + lane
            for pass_id in range_constexpr(LDS_V_TILE_SIZE // (WAVES_PER_GROUP * WARP_SIZE * 4)):
                zero_idx = (
                    v_base
                    + group_tid * fx.Index(4)
                    + fx.Index(pass_id * WAVES_PER_GROUP * WARP_SIZE * 4)
                )
                Vec.store(zero_v4, lds_v, [zero_idx])
        # Prologue: K tile 0, D-plane 0 into ping stage 0.
        if kv_upper > fx.Index(0):
            coop_load_k(_kv_loop_start, plane=0, buf_id=0)
            if const_expr(DUALWAVE_SWP):
                coop_load_k(_kv_loop_start, plane=1, buf_id=1)
                coop_load_k(_kv_loop_start, plane=2, buf_id=2)
            _waitcnt_vm_n(0)
            gpu.barrier()
        if const_expr(DUALWAVE_SWP):
            # Group 0 executes a complete matrix phase before group 1 starts.
            # Two barriers match the in-loop PV+QK0-2 / QK3-5 split.
            rocdl.sched_barrier(0)
            _wait_one_full_phase(1)
            rocdl.sched_barrier(0)
        for kv_block_start, inner_iter_args in range(
            _kv_loop_start, kv_upper, _kv_loop_step, init=init_args
        ):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            if const_expr(DUALWAVE_SWP):
                p_arg = 2 + D_CHUNKS
                p_packs_lo_prev = [
                    inner_iter_args[p_arg + i]
                    for i in range_constexpr(PV_K_STEPS)
                ]
                p_packs_hi_prev = [
                    inner_iter_args[p_arg + PV_K_STEPS + i]
                    for i in range_constexpr(PV_K_STEPS)
                ]
                rocdl.s_setprio(1)
                rocdl.sched_barrier(0)
                o_accs = _apply_pv(
                    p_packs_lo_prev,
                    p_packs_hi_prev,
                    o_accs,
                    Vec.from_elements([fx.Float32(1.0)], fx.Float32).broadcast_to(16),
                )

            for kv_sub in range_constexpr(1):
                kv_start = kv_block_start

                # ==== GEMM1: two-stage 32-D K pipeline ====
                seqk0 = _swizzle_a(lane_mod_32)
                seqk1 = fx.Index(K_SUB_N) + seqk0

                s_acc_lo = c_zero_v16f32
                s_acc_hi = c_zero_v16f32
                for plane in range_constexpr(K_PLANES):
                    stage = plane % K_BUFFER_COUNT
                    k_base = k_buf_base(stage)
                    next_plane = plane + (2 if DUALWAVE_SWP else 1)

                    # 4-wave path: DMA the next plane while this one feeds MFMA.
                    # Dual-wave path: do not overwrite a live 3-slot buffer;
                    # planes 3..5 are issued after each of 0..2 is consumed.
                    if const_expr(not DUALWAVE_SWP and next_plane < K_PLANES):
                        coop_dma_k_plane(
                            kv_start,
                            next_plane,
                            next_plane % K_BUFFER_COUNT,
                        )
                        if const_expr(
                            not CAUSAL and plane + 2 == K_PLANES
                        ):
                            # Start context V while plane 4 feeds MFMA. K plane
                            # 5 is older in the VMEM queue, so vmcnt(4) below
                            # can expose K without draining these V requests.
                            _v_vecs_prefetch = coop_load_v_global(
                                kv_start, n_half=0
                            )
                    elif const_expr(not DUALWAVE_SWP):
                        if const_expr(CAUSAL):
                            _v_vecs_prefetch = coop_load_v_global(
                                kv_start, n_half=0
                            )
                        else:
                            _v_vecs_hi = coop_load_v_global(
                                kv_start, n_half=1
                            )

                    k_packs_lo = []
                    k_packs_hi = []
                    for ks_local in range_constexpr(4):
                        d = (
                            fx.Index(ks_local * K_STEP_QK)
                            + lane_div_32 * MFMA_LANE_K
                        )
                        k_packs_lo.append(
                            Vec.load(
                                mfma_pack_type,
                                lds_k,
                                [k_base + _lds_elem_offset(seqk0, d)],
                            )
                        )
                        k_packs_hi.append(
                            Vec.load(
                                mfma_pack_type,
                                lds_k,
                                [k_base + _lds_elem_offset(seqk1, d)],
                            )
                        )

                    for ks_local in range_constexpr(4):
                        qks = plane * 4 + ks_local
                        s_acc_lo = mfma_acc(
                            k_packs_lo[ks_local],
                            q_b_packs[qks],
                            s_acc_lo,
                        )
                        s_acc_hi = mfma_acc(
                            k_packs_hi[ks_local],
                            q_b_packs[qks],
                            s_acc_hi,
                        )

                    # Six selected LDS instructions and eight MFMAs per plane.
                    for _ in range_constexpr(2):
                        rocdl.sched_vmem(
                            4
                            if (
                                not DUALWAVE_SWP
                                and not CAUSAL
                                and plane + 2 == K_PLANES
                            )
                            else 2
                        )
                        rocdl.sched_dsrd(3)
                        rocdl.sched_mfma(4)
                    rocdl.sched_barrier(0)

                    if const_expr(DUALWAVE_SWP):
                        # Buffer plane%3 is now free. Overlap K(plane+3) DMA
                        # with later MFMAs; one barrier publishes all three.
                        dma_plane = plane + K_BUFFER_COUNT
                        if const_expr(dma_plane < K_PLANES):
                            coop_dma_k_plane(
                                kv_start,
                                dma_plane,
                                dma_plane % K_BUFFER_COUNT,
                            )
                        if const_expr(plane + 1 == K_BUFFER_COUNT):
                            _waitcnt_vm_n(0)
                            _phase_barrier()
                    elif const_expr(next_plane < K_PLANES):
                        _waitcnt_vm_n(
                            4
                            if (not CAUSAL and plane + 2 == K_PLANES)
                            else 0
                        )
                        gpu.barrier()

                if const_expr(DUALWAVE_SWP):
                    # Close the two-rendezvous matrix phase.
                    _phase_barrier()
                    rocdl.s_setprio(0)

                # ==== Online softmax over 64 KV positions ====
                if const_expr(DUALWAVE_SWP):
                    # V and next-tile K staging belong to the VALU phase.
                    # Their VMEM latency is hidden by exact softmax below.
                    _v_vecs_prefetch = coop_load_v_global(
                        kv_start, n_half=0
                    )
                    _v_vecs_hi = coop_load_v_global(kv_start, n_half=1)
                s_raw_lo = []
                s_raw_hi = []
                for r in range_constexpr(16):
                    s_raw_lo.append(Vec(s_acc_lo)[r])
                    s_raw_hi.append(Vec(s_acc_hi)[r])

                if const_expr(CAUSAL):
                    # SwizzleA S_acc columns (native): n = kv_start + k_sub*8 +
                    # ((r//8)*16 + r%8); hi half adds +K_SUB_N.
                    # Unroll into scalars: dynamic `if tile_needs_mask` cannot
                    # mutate a Python list (FlyDSL stateful-if requires SSA).
                    kv_start_i32 = fx.Int32(kv_start)
                    lane_div_32_i32 = fx.Int32(lane_div_32)
                    max_kv_col_i32 = kv_start_i32 + fx.Int32(BLOCK_N - 1)
                    tile_needs_mask = max_kv_col_i32 > (
                        fx.Int32(q_start) + k_len_i32 - q_len_i32
                    )
                    s_raw_lo_0 = s_raw_lo[0]
                    s_raw_lo_1 = s_raw_lo[1]
                    s_raw_lo_2 = s_raw_lo[2]
                    s_raw_lo_3 = s_raw_lo[3]
                    s_raw_lo_4 = s_raw_lo[4]
                    s_raw_lo_5 = s_raw_lo[5]
                    s_raw_lo_6 = s_raw_lo[6]
                    s_raw_lo_7 = s_raw_lo[7]
                    s_raw_lo_8 = s_raw_lo[8]
                    s_raw_lo_9 = s_raw_lo[9]
                    s_raw_lo_10 = s_raw_lo[10]
                    s_raw_lo_11 = s_raw_lo[11]
                    s_raw_lo_12 = s_raw_lo[12]
                    s_raw_lo_13 = s_raw_lo[13]
                    s_raw_lo_14 = s_raw_lo[14]
                    s_raw_lo_15 = s_raw_lo[15]
                    s_raw_hi_0 = s_raw_hi[0]
                    s_raw_hi_1 = s_raw_hi[1]
                    s_raw_hi_2 = s_raw_hi[2]
                    s_raw_hi_3 = s_raw_hi[3]
                    s_raw_hi_4 = s_raw_hi[4]
                    s_raw_hi_5 = s_raw_hi[5]
                    s_raw_hi_6 = s_raw_hi[6]
                    s_raw_hi_7 = s_raw_hi[7]
                    s_raw_hi_8 = s_raw_hi[8]
                    s_raw_hi_9 = s_raw_hi[9]
                    s_raw_hi_10 = s_raw_hi[10]
                    s_raw_hi_11 = s_raw_hi[11]
                    s_raw_hi_12 = s_raw_hi[12]
                    s_raw_hi_13 = s_raw_hi[13]
                    s_raw_hi_14 = s_raw_hi[14]
                    s_raw_hi_15 = s_raw_hi[15]

                    if tile_needs_mask:
                        lane_off_i32 = lane_div_32_i32 * fx.Int32(8)
                        # Register offs {0..7, 16..23} (native softmax_mask).
                        kv_col_lo_0 = kv_start_i32 + lane_off_i32 + fx.Int32(0)
                        s_raw_lo_0 = ArithValue(kv_col_lo_0 > causal_bound_i32).select(c_neg_inf, s_raw_lo_0)
                        s_raw_hi_0 = ArithValue(kv_col_lo_0 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_0
                        )
                        kv_col_lo_1 = kv_start_i32 + lane_off_i32 + fx.Int32(1)
                        s_raw_lo_1 = ArithValue(kv_col_lo_1 > causal_bound_i32).select(c_neg_inf, s_raw_lo_1)
                        s_raw_hi_1 = ArithValue(kv_col_lo_1 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_1
                        )
                        kv_col_lo_2 = kv_start_i32 + lane_off_i32 + fx.Int32(2)
                        s_raw_lo_2 = ArithValue(kv_col_lo_2 > causal_bound_i32).select(c_neg_inf, s_raw_lo_2)
                        s_raw_hi_2 = ArithValue(kv_col_lo_2 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_2
                        )
                        kv_col_lo_3 = kv_start_i32 + lane_off_i32 + fx.Int32(3)
                        s_raw_lo_3 = ArithValue(kv_col_lo_3 > causal_bound_i32).select(c_neg_inf, s_raw_lo_3)
                        s_raw_hi_3 = ArithValue(kv_col_lo_3 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_3
                        )
                        kv_col_lo_4 = kv_start_i32 + lane_off_i32 + fx.Int32(4)
                        s_raw_lo_4 = ArithValue(kv_col_lo_4 > causal_bound_i32).select(c_neg_inf, s_raw_lo_4)
                        s_raw_hi_4 = ArithValue(kv_col_lo_4 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_4
                        )
                        kv_col_lo_5 = kv_start_i32 + lane_off_i32 + fx.Int32(5)
                        s_raw_lo_5 = ArithValue(kv_col_lo_5 > causal_bound_i32).select(c_neg_inf, s_raw_lo_5)
                        s_raw_hi_5 = ArithValue(kv_col_lo_5 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_5
                        )
                        kv_col_lo_6 = kv_start_i32 + lane_off_i32 + fx.Int32(6)
                        s_raw_lo_6 = ArithValue(kv_col_lo_6 > causal_bound_i32).select(c_neg_inf, s_raw_lo_6)
                        s_raw_hi_6 = ArithValue(kv_col_lo_6 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_6
                        )
                        kv_col_lo_7 = kv_start_i32 + lane_off_i32 + fx.Int32(7)
                        s_raw_lo_7 = ArithValue(kv_col_lo_7 > causal_bound_i32).select(c_neg_inf, s_raw_lo_7)
                        s_raw_hi_7 = ArithValue(kv_col_lo_7 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_7
                        )
                        kv_col_lo_8 = kv_start_i32 + lane_off_i32 + fx.Int32(16)
                        s_raw_lo_8 = ArithValue(kv_col_lo_8 > causal_bound_i32).select(c_neg_inf, s_raw_lo_8)
                        s_raw_hi_8 = ArithValue(kv_col_lo_8 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_8
                        )
                        kv_col_lo_9 = kv_start_i32 + lane_off_i32 + fx.Int32(17)
                        s_raw_lo_9 = ArithValue(kv_col_lo_9 > causal_bound_i32).select(c_neg_inf, s_raw_lo_9)
                        s_raw_hi_9 = ArithValue(kv_col_lo_9 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_9
                        )
                        kv_col_lo_10 = kv_start_i32 + lane_off_i32 + fx.Int32(18)
                        s_raw_lo_10 = ArithValue(kv_col_lo_10 > causal_bound_i32).select(c_neg_inf, s_raw_lo_10)
                        s_raw_hi_10 = ArithValue(kv_col_lo_10 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_10
                        )
                        kv_col_lo_11 = kv_start_i32 + lane_off_i32 + fx.Int32(19)
                        s_raw_lo_11 = ArithValue(kv_col_lo_11 > causal_bound_i32).select(c_neg_inf, s_raw_lo_11)
                        s_raw_hi_11 = ArithValue(kv_col_lo_11 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_11
                        )
                        kv_col_lo_12 = kv_start_i32 + lane_off_i32 + fx.Int32(20)
                        s_raw_lo_12 = ArithValue(kv_col_lo_12 > causal_bound_i32).select(c_neg_inf, s_raw_lo_12)
                        s_raw_hi_12 = ArithValue(kv_col_lo_12 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_12
                        )
                        kv_col_lo_13 = kv_start_i32 + lane_off_i32 + fx.Int32(21)
                        s_raw_lo_13 = ArithValue(kv_col_lo_13 > causal_bound_i32).select(c_neg_inf, s_raw_lo_13)
                        s_raw_hi_13 = ArithValue(kv_col_lo_13 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_13
                        )
                        kv_col_lo_14 = kv_start_i32 + lane_off_i32 + fx.Int32(22)
                        s_raw_lo_14 = ArithValue(kv_col_lo_14 > causal_bound_i32).select(c_neg_inf, s_raw_lo_14)
                        s_raw_hi_14 = ArithValue(kv_col_lo_14 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_14
                        )
                        kv_col_lo_15 = kv_start_i32 + lane_off_i32 + fx.Int32(23)
                        s_raw_lo_15 = ArithValue(kv_col_lo_15 > causal_bound_i32).select(c_neg_inf, s_raw_lo_15)
                        s_raw_hi_15 = ArithValue(kv_col_lo_15 + fx.Int32(K_SUB_N) > causal_bound_i32).select(
                            c_neg_inf, s_raw_hi_15
                        )

                    s_raw_lo = [
                        s_raw_lo_0,
                        s_raw_lo_1,
                        s_raw_lo_2,
                        s_raw_lo_3,
                        s_raw_lo_4,
                        s_raw_lo_5,
                        s_raw_lo_6,
                        s_raw_lo_7,
                        s_raw_lo_8,
                        s_raw_lo_9,
                        s_raw_lo_10,
                        s_raw_lo_11,
                        s_raw_lo_12,
                        s_raw_lo_13,
                        s_raw_lo_14,
                        s_raw_lo_15,
                    ]
                    s_raw_hi = [
                        s_raw_hi_0,
                        s_raw_hi_1,
                        s_raw_hi_2,
                        s_raw_hi_3,
                        s_raw_hi_4,
                        s_raw_hi_5,
                        s_raw_hi_6,
                        s_raw_hi_7,
                        s_raw_hi_8,
                        s_raw_hi_9,
                        s_raw_hi_10,
                        s_raw_hi_11,
                        s_raw_hi_12,
                        s_raw_hi_13,
                        s_raw_hi_14,
                        s_raw_hi_15,
                    ]
                else:
                    # Non-causal KV padding mask (same SwizzleA column map).
                    kv_start_i32 = fx.Int32(kv_start)
                    lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(8)
                    seq_len_i32 = fx.Int32(seq_len_v)
                    for r in range_constexpr(16):
                        _off = (r // 8) * 16 + (r % 8)
                        kv_col = kv_start_i32 + lane_off_i32 + fx.Int32(_off)
                        s_raw_lo[r] = ArithValue(kv_col >= seq_len_i32).select(
                            c_neg_inf, s_raw_lo[r]
                        )
                        s_raw_hi[r] = ArithValue(
                            kv_col + fx.Int32(K_SUB_N) >= seq_len_i32
                        ).select(c_neg_inf, s_raw_hi[r])

                local_max = s_raw_lo[0]
                for r in range_constexpr(15):
                    local_max = _fmax(local_max, s_raw_lo[r + 1])
                for r in range_constexpr(16):
                    local_max = _fmax(local_max, s_raw_hi[r])
                peer_max = reduction_peer(local_max)
                row_max = _fmax(local_max, peer_max)
                m_new_raw = _fmax(m_running, row_max)

                diff_m_raw = _fsub(m_running, m_new_raw)
                diff_m_scaled = _fmul(diff_m_raw, c_sm_scale_log2e)
                corr = fx.Float32(rocdl.exp2(T.f32, _raw(diff_m_scaled)))

                scaled_max = _fmul(c_sm_scale_log2e, m_new_raw)
                neg_scaled_max = _fsub(c_zero_f, scaled_max)

                p_vals_lo = []
                p_vals_hi = []
                local_sum = c_zero_f
                for r in range_constexpr(16):
                    diff_lo = fmath.fma(
                        s_raw_lo[r],
                        c_sm_scale_log2e,
                        neg_scaled_max,
                        fastmath=fm_fast,
                    )
                    p_lo = fx.Float32(rocdl.exp2(T.f32, _raw(diff_lo)))
                    if const_expr(DUALWAVE_SWP):
                        p_lo = fx.Float32(_pin_f32(p_lo))
                    p_vals_lo.append(p_lo)
                    if const_expr(DUALWAVE_SWP):
                        local_sum = _fadd_vop2(local_sum, p_lo)
                    else:
                        local_sum = _fadd(local_sum, p_lo)
                if const_expr(DUALWAVE_COUPLED_SOFTMAX):
                    p_packs_lo, denominator_lo = denominator_mfma_half(
                        p_vals_lo
                    )
                for r in range_constexpr(16):
                    if const_expr(
                        DUALWAVE_COUPLED_SOFTMAX and r >= 4
                    ):
                        p_hi = approx_exp2_score(
                            s_raw_hi[r], neg_scaled_max
                        )
                    else:
                        diff_hi = fmath.fma(
                            s_raw_hi[r],
                            c_sm_scale_log2e,
                            neg_scaled_max,
                            fastmath=fm_fast,
                        )
                        p_hi = fx.Float32(
                            rocdl.exp2(T.f32, _raw(diff_hi))
                        )
                    if const_expr(DUALWAVE_SWP):
                        p_hi = fx.Float32(_pin_f32(p_hi))
                    p_vals_hi.append(p_hi)
                    if const_expr(DUALWAVE_SWP):
                        local_sum = _fadd_vop2(local_sum, p_hi)
                    else:
                        local_sum = _fadd(local_sum, p_hi)
                if const_expr(DUALWAVE_COUPLED_SOFTMAX):
                    p_packs_hi, denominator_hi = denominator_mfma_half(
                        p_vals_hi
                    )

                if const_expr(DUALWAVE_COUPLED_SOFTMAX):
                    tile_sum = _fadd(denominator_lo, denominator_hi)
                else:
                    peer_sum = reduction_peer(local_sum)
                    tile_sum = _fadd(local_sum, peer_sum)
                l_corr = _fmul(corr, l_running)
                l_new = _fadd(l_corr, tile_sum)

                corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(16)
                if const_expr(not DUALWAVE_SWP and D_CHUNKS <= PV_K_STEPS):
                    o_accs[0] = _fmul(Vec(o_accs[0]), corr_vec)
                else:
                    for dc in range_constexpr(D_CHUNKS):
                        if const_expr(VOP2_O_RESCALE):
                            o_accs[dc] = Vec.from_elements(
                                [
                                    _fmul_vop2(Vec(o_accs[dc])[i], corr)
                                    for i in range_constexpr(16)
                                ],
                                fx.Float32,
                            )
                        else:
                            o_accs[dc] = _fmul(Vec(o_accs[dc]), corr_vec)

                # Match the matrix-side visibility barrier after QK planes 0..2.
                _phase_pad(1)

                # Context has eight outstanding V loads (four per half).
                # Retire only half 0 here; keep half 1 in flight while the
                # first LDS stores issue. Causal has only loaded half 0.
                _waitcnt_vm_n(0 if CAUSAL else 4)
                coop_store_v_lds(
                    _v_vecs_prefetch, n_half=0, buf_id=v_slot
                )
                if const_expr(CAUSAL):
                    # Short causal winner: overlap V1 with PV0.
                    _v_vecs_hi = coop_load_v_global(
                        kv_start, n_half=1
                    )
                else:
                    # Long-context winner: both V halves are already resident;
                    # prefetch next K and keep one combined PV phase.
                    coop_dma_k_plane(
                        kv_start + _kv_loop_step, plane=0, buf_id=0
                    )
                    if const_expr(DUALWAVE_SWP):
                        coop_dma_k_plane(
                            kv_start + _kv_loop_step, plane=1, buf_id=1
                        )
                        coop_dma_k_plane(
                            kv_start + _kv_loop_step, plane=2, buf_id=2
                        )
                    # The four older V-half-1 requests are complete while the
                    # newly issued K DMAs remain outstanding.
                    _waitcnt_vm_n(12 if DUALWAVE_SWP else 4)
                    coop_store_v_lds(
                        _v_vecs_hi, n_half=1, buf_id=v_slot
                    )
                if const_expr(not DUALWAVE_SWP):
                    _lds_handoff_barrier()

                if const_expr(dtype_str == "bf16"):
                    if const_expr(not DUALWAVE_COUPLED_SOFTMAX):
                        p_packs_lo = []
                        p_packs_hi = []
                        for pks in range_constexpr(PV_K_STEPS):
                            p_base = pks * 4
                            p_packs_lo.append(
                                bf16_trunc_pack_v4(
                                    p_vals_lo[p_base : p_base + 4]
                                )
                            )
                            p_packs_hi.append(
                                bf16_trunc_pack_v4(
                                    p_vals_hi[p_base : p_base + 4]
                                )
                            )
                else:
                    p_f16_lo = [
                        fx.Float32(p_vals_lo[r]).to(elem_dtype)
                        for r in range_constexpr(16)
                    ]
                    p_f16_hi = [
                        fx.Float32(p_vals_hi[r]).to(elem_dtype)
                        for r in range_constexpr(16)
                    ]
                    p_packs_lo = []
                    p_packs_hi = []
                    for pks in range_constexpr(PV_K_STEPS):
                        p_base = pks * 4
                        p_packs_lo.append(
                            Vec.from_elements(
                                p_f16_lo[p_base : p_base + 4], elem_dtype
                            ).ir_value()
                        )
                        p_packs_hi.append(
                            Vec.from_elements(
                                p_f16_hi[p_base : p_base + 4], elem_dtype
                            ).ir_value()
                        )

                _steps = [
                    (dc, pks)
                    for dc in range(D_CHUNKS)
                    for pks in range(PV_K_STEPS)
                ]

                def _read_v_pack(step_idx, n_half):
                    dc, pks = _steps[step_idx]
                    d_pos = fx.Index(dc * D_CHUNK) + lane_mod_32
                    k_base_v = (
                        fx.Index((pks // 2) * 16)
                        + lane_div_32 * fx.Index(8)
                        + fx.Index((pks % 2) * 4)
                        + fx.Index(n_half * K_SUB_N)
                    )
                    v_idx = v_base + _v_lds_elem_offset(k_base_v, d_pos)
                    return Vec.load(v4f16_type, lds_v, [v_idx])

                if const_expr(not DUALWAVE_SWP and CAUSAL):
                    for n_half in range_constexpr(2):
                        if const_expr(n_half == 1):
                            _waitcnt_vm_n(0)
                            coop_dma_k_plane(
                                kv_start + _kv_loop_step,
                                plane=0,
                                buf_id=0,
                            )
                            coop_store_v_lds(
                                _v_vecs_hi, n_half=1, buf_id=v_slot
                            )
                            _lds_handoff_barrier()

                        p_packs = (
                            p_packs_lo if n_half == 0 else p_packs_hi
                        )
                        v_cur = _read_v_pack(0, n_half)
                        for si in range_constexpr(len(_steps)):
                            dc, pks = _steps[si]
                            if const_expr(si + 1 < len(_steps)):
                                v_nxt = _read_v_pack(si + 1, n_half)
                            o_accs[dc] = mfma_acc(
                                v_cur, p_packs[pks], o_accs[dc]
                            )
                            if const_expr(
                                n_half == 0
                                and D_CHUNKS <= PV_K_STEPS
                                and dc == 0
                                and pks < D_CHUNKS - 1
                            ):
                                o_accs[pks + 1] = (
                                    Vec(o_accs[pks + 1]) * corr_vec
                                )
                            if const_expr(si + 1 < len(_steps)):
                                v_cur = v_nxt

                        for _ in range_constexpr(4):
                            rocdl.sched_vmem(1)
                            rocdl.sched_dsrd(2)
                            rocdl.sched_mfma(4)
                        rocdl.sched_barrier(0)
                elif const_expr(not DUALWAVE_SWP):
                    v_lo_cur = _read_v_pack(0, 0)
                    v_hi_cur = _read_v_pack(0, 1)
                    for si in range_constexpr(len(_steps)):
                        dc, pks = _steps[si]
                        if const_expr(si + 1 < len(_steps)):
                            v_lo_nxt = _read_v_pack(si + 1, 0)
                            v_hi_nxt = _read_v_pack(si + 1, 1)
                        o_accs[dc] = mfma_acc(
                            v_lo_cur, p_packs_lo[pks], o_accs[dc]
                        )
                        o_accs[dc] = mfma_acc(
                            v_hi_cur, p_packs_hi[pks], o_accs[dc]
                        )
                        if const_expr(
                            D_CHUNKS <= PV_K_STEPS
                            and dc == 0
                            and pks < D_CHUNKS - 1
                        ):
                            o_accs[pks + 1] = (
                                Vec(o_accs[pks + 1]) * corr_vec
                            )
                        if const_expr(si + 1 < len(_steps)):
                            v_lo_cur = v_lo_nxt
                            v_hi_cur = v_hi_nxt

                    for pv_group in range_constexpr(8):
                        if const_expr(pv_group % 2 == 0):
                            rocdl.sched_vmem(1)
                        rocdl.sched_dsrd(2)
                        rocdl.sched_mfma(4)
                    rocdl.sched_barrier(0)

                m_running = m_new_raw
                l_running = l_new
                _waitcnt_vm_n(0)
                if const_expr(DUALWAVE_SWP):
                    # Second VALU rendezvous: publishes V(t) and K(t+1)
                    # planes 0..2 while pairing with the partner's QK 3..5 close.
                    _phase_barrier()
                else:
                    gpu.barrier()  # K(t+1) visible for the next GEMM1

            _yield_args = [m_running, l_running] + o_accs
            if const_expr(DUALWAVE_SWP):
                _yield_args += p_packs_lo + p_packs_hi
            loop_results = yield _yield_args

        if const_expr(DUALWAVE_SWP):
            # Drain P(last)*V(last) as one final matrix phase. Group 0 then
            # supplies a complete phase of barriers while group 1 drains.
            p_arg = 2 + D_CHUNKS
            p_packs_lo_last = [
                loop_results[p_arg + i]
                for i in range_constexpr(PV_K_STEPS)
            ]
            p_packs_hi_last = [
                loop_results[p_arg + PV_K_STEPS + i]
                for i in range_constexpr(PV_K_STEPS)
            ]
            rocdl.s_setprio(1)
            rocdl.sched_barrier(0)
            o_last = [
                loop_results[2 + i] for i in range_constexpr(D_CHUNKS)
            ]
            o_last = _apply_pv(
                p_packs_lo_last,
                p_packs_hi_last,
                o_last,
                Vec.from_elements([fx.Float32(1.0)], fx.Float32).broadcast_to(16),
            )
            for _ in range_constexpr(PHASE_BARRIERS):
                _phase_barrier()
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            _wait_one_full_phase(0)
            rocdl.sched_barrier(0)
            loop_results = (
                [loop_results[0], loop_results[1]]
                + o_last
                + list(loop_results[p_arg:])
            )

        # ---- Normalize and store O (128-bit buffer_store_dwordx4) ----
        # gfx950: pack 4 f32 -> 2 bf16 dwords (cvt_pk_bf16_f32), permlane32_swap fuses
        # each lane's 4 cols with its half-wave partner's -> 8 cols/store. O is
        # num_records-bounded (o_rsrc) -> partial-q-tile OOB rows drop.
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        inv_l = rocdl.rcp(T.f32, l_final)
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)
        v_o = [Vec(o_finals[dc]) * inv_l_vec for dc in range_constexpr(D_CHUNKS)]

        for dc in range_constexpr(D_CHUNKS):
            for grp in range_constexpr(4):
                r0 = grp * 4
                o_f16 = [fx.Float32(Vec(v_o[dc])[r0 + i]).to(elem_dtype) for i in range_constexpr(4)]
                pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
                o2 = Vec.from_elements([_raw(pack[0]), _raw(pack[1])], fx.Int32)
                d_col = fx.Index(dc * D_CHUNK) + lane_div_32 * fx.Index(4) + fx.Index(grp * 8)
                o_global = global_idx_o(q_row, d_col)
                buffer_ops.buffer_store(o2, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)

        if const_expr(RETURN_LSE):
            if q_row < q_len:
                log_l = fmath.log(_raw(l_final), fastmath=fm_fast)
                lse_val = _fadd(_fmul(loop_results[0], fx.Float32(sm_scale)), log_l)
                packed_q = q0 + q_row
                lse_idx = q_head_idx * total_q_v + packed_q
                buffer_ops.buffer_store(fx.Float32(lse_val), lse_rsrc, lse_idx)


    @flyc.jit
    def launch_fmha_pingpong_gfx942(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        cu_seqlens_q: fx.Tensor,
        cu_seqlens_k: fx.Tensor,
        batch_size: fx.Int32,
        max_seqlen_q: fx.Int32,
        total_q: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        k_allocator.finalized = False
        v_allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            k_allocator.finalize()
            v_allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(max_seqlen_q)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS_Q

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        fmha_pingpong_gfx942_kernel(
            Q,
            K,
            V,
            O,
            LSE,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            total_q,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": (
                    f"{int(flat_work_group_size)},{int(flat_work_group_size)}"
                ),
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(flat_work_group_size, 1, 1),
            stream=stream,
        )

    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    launch_fmha_pingpong_gfx942.compile_hints = dict(_fmha_compile_hints)
    return launch_fmha_pingpong_gfx942


@lru_cache(maxsize=32)
def _get_pingpong_launcher(
    num_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_v: int,
    causal: bool,
    dtype_str: str,
    return_lse: bool,
    waves_per_eu: int,
    sm_scale: float,
    dualwave_coupled_softmax: bool,
    rotate_pv_accumulators: bool,
    vop2_o_rescale: bool,
):
    return build_fmha_pingpong_gfx942_module(
        num_heads=num_heads,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        causal=causal,
        dtype_str=dtype_str,
        sm_scale=sm_scale,
        waves_per_eu=waves_per_eu,
        num_kv_heads=num_kv_heads,
        return_lse=return_lse,
        dualwave_coupled_softmax=dualwave_coupled_softmax,
        rotate_pv_accumulators=rotate_pv_accumulators,
        vop2_o_rescale=vop2_o_rescale,
    )


def flash_attn_varlen_gfx942_pingpong(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    *,
    waves_per_eu: int = DEFAULT_WAVES_PER_EU,
    coupled_softmax: bool = False,
    rotate_pv_accumulators: bool = False,
    vop2_o_rescale: bool = False,
    stream: torch.cuda.Stream | None = None,
):
    """Run the opt-in gfx942 packed-THD ping-pong experiment.

    ``max_seqlen_k`` is accepted to match ``flash_attn_varlen_func``; the kernel
    reads per-batch KV length from ``cu_seqlens_k``.

    ``coupled_softmax=False`` is the exact ping-pong control. ``True`` enables
    the inseparable mixed-exp plus BF16 MFMA-denominator treatment.
    """
    del max_seqlen_k
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q/k/v must be packed THD [total, H, D]")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError(
            "flash_attn_varlen_gfx942_pingpong requires CUDA/HIP tensors"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    dtype_str = _DTYPE_TO_STR.get(q.dtype)
    if dtype_str != "bf16":
        raise ValueError(
            "flash_attn_varlen_gfx942_pingpong supports bf16 only, "
            f"got {q.dtype!r}"
        )
    total_q, hq, dq = q.shape
    _total_k, hk, dk = k.shape
    dv = v.shape[-1]
    if dk != dq:
        raise ValueError(f"K head dim must match Q, got Q={dq} K={dk}")
    if v.shape[0] != _total_k or v.shape[1] != hk:
        raise ValueError(
            f"V must be [total_k, Hkv, Dv], got {tuple(v.shape)} vs K {tuple(k.shape)}"
        )
    if hq % hk != 0:
        raise ValueError(f"num_heads ({hq}) must be divisible by num_kv_heads ({hk})")
    if (
        hq != 12
        or hk != 12
        or dq != 192
        or dv != 128
        or causal
    ):
        raise ValueError(
            "gfx942 ping-pong requires non-causal packed bf16 H=Hkv=12 "
            "QK=192 V=128"
        )
    validate_fmha_gfx942_tiles(dq, dv, block_m=256, block_n=64)
    if softmax_scale is None:
        softmax_scale = 1.0 / host_math.sqrt(dq)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_seqlens_k = cu_seqlens_k.to(torch.int32).contiguous()
    if out is None:
        out = torch.empty(total_q, hq, dv, device=q.device, dtype=q.dtype)
    else:
        out = out.contiguous()
        if tuple(out.shape) != (total_q, hq, dv):
            raise ValueError(
                f"out shape {tuple(out.shape)} != {(total_q, hq, dv)}"
            )
    if return_lse:
        lse = torch.empty(hq, total_q, device=q.device, dtype=torch.float32)
    else:
        lse = torch.empty(1, device=q.device, dtype=torch.float32)

    batch = int(cu_seqlens_q.numel() - 1)
    exe = _get_pingpong_launcher(
        num_heads=int(hq),
        num_kv_heads=int(hk),
        head_dim_qk=int(dq),
        head_dim_v=int(dv),
        causal=bool(causal),
        dtype_str=dtype_str,
        return_lse=bool(return_lse),
        waves_per_eu=int(waves_per_eu),
        sm_scale=float(softmax_scale),
        dualwave_coupled_softmax=bool(coupled_softmax),
        rotate_pv_accumulators=bool(rotate_pv_accumulators),
        vop2_o_rescale=bool(vop2_o_rescale),
    )
    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    if launch_stream.device != q.device:
        raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
    with torch.cuda.device(q.device.index):
        _run_compiled(
            exe,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            batch,
            int(max_seqlen_q),
            int(total_q),
            launch_stream,
        )
    if return_lse:
        return out, lse
    return out
