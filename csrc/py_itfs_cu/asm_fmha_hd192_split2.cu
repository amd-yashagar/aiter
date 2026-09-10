// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "aiter_hip_common.h"
#include "mha_fwd.h"
#include <cstring>

namespace {

constexpr const char* kKernelName =
    "_ZN5aiter40fmha_fwd_hd192x128_bf16_rtna_group_splitE";
constexpr const char* kCoName =
    "fmha_hd192_split/fwd_hd192x128_bf16_rtna_group_split.co";

} // namespace

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmha_hd192_split2_producer,
    (aiter_tensor_t* q,
     aiter_tensor_t* k,
     aiter_tensor_t* v,
     aiter_tensor_t* out_parts,
     aiter_tensor_t* lse_parts,
     aiter_tensor_t* cu_seqlens_q,
     aiter_tensor_t* cu_seqlens_k,
     int max_seqlen_q,
     int max_seqlen_k,
     int num_splits,
     float softmax_scale,
     hipStream_t stream),
    (q,
     k,
     v,
     out_parts,
     lse_parts,
     cu_seqlens_q,
     cu_seqlens_k,
     max_seqlen_q,
     max_seqlen_k,
     num_splits,
     softmax_scale,
     stream))
{
    AITER_CHECK(q && k && v && out_parts && lse_parts && cu_seqlens_q && cu_seqlens_k,
                "fmha_hd192_split2_producer: null tensor argument");
    HipDeviceGuard device_guard{q->device_id};
    AITER_CHECK(get_gpu_arch() == "gfx942",
                "fmha_hd192_split2_producer requires gfx942");
    AITER_CHECK(q->dtype() == AITER_DTYPE_bf16 && k->dtype() == AITER_DTYPE_bf16 &&
                    v->dtype() == AITER_DTYPE_bf16 &&
                    out_parts->dtype() == AITER_DTYPE_bf16,
                "fmha_hd192_split2_producer requires bf16 q/k/v/out");
    AITER_CHECK(lse_parts->dtype() == AITER_DTYPE_fp32,
                "fmha_hd192_split2_producer requires fp32 LSE");
    AITER_CHECK(cu_seqlens_q->dtype() == AITER_DTYPE_i32 &&
                    cu_seqlens_k->dtype() == AITER_DTYPE_i32,
                "fmha_hd192_split2_producer requires int32 cumulative lengths");
    AITER_CHECK(q->dim() == 3 && k->dim() == 3 && v->dim() == 3,
                "q/k/v must be packed [token, head, dim]");
    AITER_CHECK(q->size(2) == 192 && k->size(2) == 192 && v->size(2) == 128,
                "fmha_hd192_split2_producer requires Dqk=192, Dv=128");
    AITER_CHECK(q->size(1) == k->size(1) && k->size(1) == v->size(1),
                "fmha_hd192_split2_producer currently requires Hq=Hkv");
    AITER_CHECK(k->size(0) == v->size(0), "K/V token counts must match");
    AITER_CHECK(cu_seqlens_q->size(0) == 2 && cu_seqlens_k->size(0) == 2,
                "fmha_hd192_split2_producer currently supports batch=1");
    AITER_CHECK(max_seqlen_q == q->size(0) && max_seqlen_k == k->size(0),
                "max sequence lengths must match the single packed sequence");
    AITER_CHECK(max_seqlen_k >= 64,
                "fmha_hd192_split2_producer requires at least two KV tiles");
    AITER_CHECK(num_splits >= 2 && num_splits <= 8 &&
                    max_seqlen_k >= num_splits * 32,
                "num_splits must be in [2, 8] with at least one tile per split");
    AITER_CHECK(q->stride(2) == 1 && k->stride(2) == 1 && v->stride(2) == 1,
                "q/k/v require a contiguous head dimension");
    AITER_CHECK(out_parts->dim() == 4 && out_parts->size(0) == num_splits &&
                    out_parts->size(1) == q->size(0) &&
                    out_parts->size(2) == q->size(1) && out_parts->size(3) == 128,
                "out_parts must be [num_splits, total_q, H, 128]");
    AITER_CHECK(lse_parts->dim() == 3 && lse_parts->size(0) == num_splits &&
                    lse_parts->size(1) == q->size(1) &&
                    lse_parts->size(2) == q->size(0),
                "lse_parts must be [num_splits, H, total_q]");

    aiter::fmha_fwd_v3_args args;
    std::memset(&args, 0, sizeof(args));
    args.ptr_o            = out_parts->data_ptr();
    args.ptr_q            = q->data_ptr();
    args.ptr_k            = k->data_ptr();
    args.ptr_v            = v->data_ptr();
    args.ptr_lse          = lse_parts->data_ptr();
    args.scalar           = softmax_scale;
    args.s_seq_len        = max_seqlen_q;
    args.s_Seqs           = q->stride(0) * 2;
    args.s_Ts             = 128 * q->stride(0) * 2;
    args.s_Hs             = q->stride(1) * 2;
    args.s_Bs             = 0;
    args.s_gqa            = 1;
    args.s_k_Seqs         = k->stride(0) * 2;
    args.s_k_Hs           = k->stride(1) * 2;
    args.s_k_Bs           = 0;
    args.s_opt            = 0;
    args.s_lse            = 1;
    args.s_kv_seq_len     = max_seqlen_k;
    args.s_qk_head_dim    = 192;
    args.s_v_head_dim     = 128;
    args.s_q_head_num     = q->size(1);
    args.s_v_Seqs         = v->stride(0) * 2;
    args.s_v_Hs           = v->stride(1) * 2;
    // Batch=1 specialization: the kernel repurposes s_v_Bs as aligned KV extent.
    const int kv_tiles     = (max_seqlen_k + 31) / 32;
    const int split_tiles  = (kv_tiles + num_splits - 1) / num_splits;
    AITER_CHECK(split_tiles * (num_splits - 1) < kv_tiles,
                "aligned split partition would contain an empty final split");
    args.s_v_Bs            = split_tiles * 32;
    args.s_o_Seqs         = out_parts->stride(1) * 2;
    args.s_o_Hs           = out_parts->stride(2) * 2;
    // The split producer repurposes s_o_Bs as its split-major byte stride.
    args.s_o_Bs           = out_parts->stride(0) * 2;
    args.ptr_qseq         = cu_seqlens_q->data_ptr();
    args.ptr_kseq         = cu_seqlens_k->data_ptr();
    args.s_lse_Hs         = lse_parts->stride(1) * 4;
    args.ptr_qseq_padding = cu_seqlens_q->data_ptr();
    args.ptr_kseq_padding = cu_seqlens_k->data_ptr();

    size_t arg_size = sizeof(args);
    static AiterAsmKernel kernel{kKernelName, kCoName};
    const int q_tiles = (max_seqlen_q + 127) / 128;
    kernel.launch_kernel({&args,
                          &arg_size,
                          static_cast<int>(q->size(1)),
                          num_splits,
                          q_tiles,
                          256,
                          1,
                          1,
                          stream});
}
