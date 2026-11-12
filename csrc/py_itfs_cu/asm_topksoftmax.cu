// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "asm_topksoftmax_configs.hpp"
#include <memory>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_T;
    p2 _p0;
    void* ptr_W;
    p2 _p1;
    void* ptr_A;
    p2 _p2;
    unsigned int batch;
    p3 _p4;
    unsigned int expert;
    p3 _p5;
    unsigned int topk;
    p3 _p6;
    unsigned int renormalize;
    p3 _p7;
    unsigned int out_stride;
    p3 _p8;
};

std::pair<std::string, int> get_heuristic_kernel_topksoftmax(std::string arch_id,
    std::string dtype,
    int max_subm,
    int num_experts,
    int topk,
    CFG* cfgs)
{
    int subm = -1;
    std::string kernelName = "";
    for(const auto& el : *cfgs)
    {
        if (el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if (cfg.dtype != dtype || cfg.subm > max_subm || cfg.num_experts != num_experts || cfg.topk != topk)
            continue;
        
        if (cfg.subm > subm)
        {
            kernelName = el.first;
            subm = cfg.subm;
        }
    }

    AITER_CHECK(!kernelName.empty(),
    __func__,
    ": cannot get heuristic kernel!"
    " arch_id:", arch_id,
    " dtype:", dtype,
    " max_subm:", max_subm,
    " num_experts:", num_experts,
    " topk:", topk);
    return {kernelName, subm};
}

AITER_C_ITFS
void topk_softmax_asm(aiter_tensor_t* topk_weights,         // [num_tokens, topk]
                      aiter_tensor_t* topk_indices,         // [num_tokens, topk]
                      aiter_tensor_t* token_expert_indices, // [num_tokens, topk]
                      aiter_tensor_t* gating_output,        // [num_tokens, num_experts]
                      int need_renorm, hipStream_t stream)
{
    std::string arch_id = get_gpu_arch();
    const uint num_experts = gating_output->size(-1);
    const uint num_tokens  = gating_output->numel() / num_experts;
    const uint topk        = topk_weights->size(-1);
    const uint out_stride  = topk_weights->stride(0);
    const uint MAX_SUBM    = num_tokens < 10000 ? 4 : 12;
    std::string dtype;
    if(gating_output->dtype() == AITER_DTYPE_fp32)
        dtype = "fp32";
    else if(gating_output->dtype() == AITER_DTYPE_bf16)
        dtype = "bf16";
    else
        AITER_CHECK(false, __func__, ": unsupport gating_output dtype:", AiterDtype_to_str(gating_output->dtype()));

    // args.out_stride below multiplies by a hard-coded 4 bytes per element, and
    // the kernel writes 32-bit ids and weights. A wider caller buffer is
    // therefore partially written rather than rejected.
    AITER_CHECK(topk_weights->dtype() == AITER_DTYPE_fp32,
                __func__,
                ": topk_weights must be float32, got ",
                AiterDtype_to_str(topk_weights->dtype()));
    AITER_CHECK(topk_indices->dtype() == AITER_DTYPE_i32,
                __func__,
                ": topk_indices must be int32, got ",
                AiterDtype_to_str(topk_indices->dtype()));
    // token_expert_indices is accepted but never forwarded into KernelArgs
    // below (only ptr_T/ptr_W/ptr_A are) -- this kernel neither reads nor
    // writes it, so no dtype contract applies to it here. The existing
    // test_asm case discards this buffer explicitly ("Not used. Will be
    // used in the future.").

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_T      = topk_indices->ptr;
    args.ptr_W      = topk_weights->ptr;
    args.ptr_A      = gating_output->ptr;

    args.batch       = num_tokens;
    args.expert      = num_experts;
    args.topk        = topk;
    args.renormalize = need_renorm ? 1 : 0;
    args.out_stride  = out_stride * 4;

    CFG* config_map = &cfg_topksoftmax;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = nullptr;
    auto [kernelName, subm] = get_heuristic_kernel_topksoftmax(arch_id, dtype, MAX_SUBM, num_experts, topk, config_map);

    auto it = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();

        impl_ptr =
            &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
    }
    else
        AITER_CHECK(false, __func__, " not find kernel " + kernelName);


    const HipDeviceGuard device_guard(gating_output->device_id);

    uint gdx = (num_tokens + subm - 1) / subm;
    AITER_CHECK(gdx >> 31 == 0, "num_tokens too large: ", num_tokens);
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             static_cast<int>(gdx), // gdx
                             1,                     // gdy
                             1,                     // gdz
                             256,                   // bdx: 4 wv64
                             1,                     // bdy
                             1,                     // bdz
                             stream});
}

// The asm top-k softmax kernels evaluate exp() without subtracting the row max and
// overflow for a logit 88 or more above the rest. Softmax is shift invariant, so the
// caller feeds them logits shifted by the row max, as fp32 (a bf16 shift would round
// near-tied logits and could change the top-k ids).
namespace {
constexpr int SHIFT_THREADS = 256;

__device__ inline float shift_load(const float* p, int i) { return p[i]; }
__device__ inline float shift_load(const uint16_t* p, int i)
{
    return __uint_as_float(static_cast<uint32_t>(p[i]) << 16);
}

template <typename T>
__global__ void row_max_shift_kernel(const T* __restrict__ x,
                                     float* __restrict__ out,
                                     int num_experts,
                                     long x_stride)
{
    const T* row_in = x + blockIdx.x * x_stride;
    float* row_out  = out + static_cast<long>(blockIdx.x) * num_experts;
    float m         = -INFINITY;
    for(int i = threadIdx.x; i < num_experts; i += SHIFT_THREADS)
        m = fmaxf(m, shift_load(row_in, i));
    for(int off = warpSize / 2; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor(m, off));
    __shared__ float wave_max[SHIFT_THREADS / 32];
    const int wave = threadIdx.x / warpSize;
    if(threadIdx.x % warpSize == 0)
        wave_max[wave] = m;
    __syncthreads();
    m = -INFINITY;
    for(int w = 0; w < SHIFT_THREADS / warpSize; ++w)
        m = fmaxf(m, wave_max[w]);
    if(m == -INFINITY)
        m = 0.f;
    // exp(x - 80) = 1.8e-35 is still a normal fp32; below the normal range the
    // kernel's weights flush to zero and it repeats the first expert id. Weights that
    // small renormalize to 0, so clamping does not change the result.
    for(int i = threadIdx.x; i < num_experts; i += SHIFT_THREADS)
    {
        const float d = shift_load(row_in, i) - m;
        row_out[i]    = d < -80.f ? -80.f : d;
    }
}
} // namespace

static void
topk_row_max_shift(aiter_tensor_t* gating_output, // [num_tokens, num_experts], bf16 or fp32
                   aiter_tensor_t* shifted,       // [num_tokens, num_experts], fp32
                   hipStream_t stream)
{
    const int num_experts = gating_output->size(-1);
    const long num_tokens = gating_output->numel() / num_experts;
    AITER_CHECK(shifted->dtype() == AITER_DTYPE_fp32,
                __func__,
                ": shifted must be float32, got ",
                AiterDtype_to_str(shifted->dtype()));
    AITER_CHECK(shifted->numel() == gating_output->numel(),
                __func__,
                ": shifted must have the shape of gating_output");
    AITER_CHECK(gating_output->stride(-1) == 1, __func__, ": gating_output must be row contiguous");
    if(num_tokens == 0)
        return;
    const HipDeviceGuard device_guard(gating_output->device_id);
    const long x_stride = gating_output->stride(0);
    if(gating_output->dtype() == AITER_DTYPE_fp32)
        row_max_shift_kernel<float><<<num_tokens, SHIFT_THREADS, 0, stream>>>(
            static_cast<const float*>(gating_output->ptr),
            static_cast<float*>(shifted->ptr),
            num_experts,
            x_stride);
    else if(gating_output->dtype() == AITER_DTYPE_bf16)
        row_max_shift_kernel<uint16_t><<<num_tokens, SHIFT_THREADS, 0, stream>>>(
            static_cast<const uint16_t*>(gating_output->ptr),
            static_cast<float*>(shifted->ptr),
            num_experts,
            x_stride);
    else
        AITER_CHECK(false,
                    __func__,
                    ": unsupport gating_output dtype:",
                    AiterDtype_to_str(gating_output->dtype()));
}

// topk_softmax_asm on logits shifted by their row max. `shifted` is a caller-allocated
// fp32 buffer of the gating_output shape, written by the shift kernel and read by the
// asm kernel on the same stream.
AITER_C_ITFS
void topk_softmax_asm_shifted(aiter_tensor_t* topk_weights,
                              aiter_tensor_t* topk_indices,
                              aiter_tensor_t* token_expert_indices,
                              aiter_tensor_t* gating_output,
                              aiter_tensor_t* shifted,
                              int need_renorm,
                              hipStream_t stream)
{
    topk_row_max_shift(gating_output, shifted, stream);
    topk_softmax_asm(
        topk_weights, topk_indices, token_expert_indices, shifted, need_renorm, stream);
}
