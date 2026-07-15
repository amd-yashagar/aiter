# SPDX-License-Identifier: MIT
# Standalone extraction of sglang's MiniMax-M3 sparse-prefill
# `_flash_attn_fwd_with_block_score_kernel` (the per-128-block relevance-score
# kernel of the lightning indexer). Copied VERBATIM from
#   sglang/python/sglang/srt/layers/attention/minimax_sparse_ops/prefill/
#     flash_with_topk_idx.py
# so the Triton `@triton.jit` source lives in a real .py file (triton needs real
# source, not exec'd strings). We drop the `from ..common.utils import ...`
# (the score kernel itself imports nothing from utils) and provide a *direct*
# launcher that mirrors the wrapper's launch for the deployed score-only path
# (disable_index_value=True: v_cache=None, sink=None, o=None), so the Triton
# kernel is timed exactly as in production but isolated from the topk kernel.
#
# Original copyright: Copyright 2025 XunhaoLai. All rights reserved.

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "BLOCK_SIZE_VD": lambda args: triton.next_power_of_2(args["v_head_dim"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=3
        ),
    ],
    key=[
        "qk_head_dim",
        "v_head_dim",
        "block_size",
        "use_gumbel_topk",
        "SCORE_TYPE",
        "DISABLE_INDEX_VALUE",
    ],
)
@triton.jit
def _flash_attn_fwd_with_block_score_kernel(
    q_ptr,  # Q: n x h x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    sink_ptr,  # Sink: h x d
    o_ptr,  # O: n x h x d
    score_ptr,  # Score: h x n x max_seqblock
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    # seqlens
    cu_seqlens,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_heads,
    gqa_group_size,
    qk_head_dim,
    v_head_dim,
    block_size: tl.constexpr,
    # sm_scale
    sm_scale,
    # gumbel topk
    use_gumbel_topk: tl.constexpr,
    gumbel_seed,
    # stride
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_sink_h,
    stride_sink_d,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_s_h,
    stride_s_q,
    stride_s_k,
    stride_r2t_b,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    # has sink
    HAS_SINK: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    DISABLE_INDEX_VALUE: tl.constexpr,
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    sm_scale_log2e = sm_scale * 1.4426950409
    tl.static_assert(BLOCK_SIZE_K >= block_size)
    BLOCKS_PER_K_BLOCK: tl.constexpr = BLOCK_SIZE_K // block_size
    # get batch id and head id
    pid_q, pid_bh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    pid_kh = pid_h // gqa_group_size
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (
        tl.load(slot_ids + pid_b).to(tl.int64) + max_slots
    ) % max_slots  # safety against negative
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return
    block_num = (seq_len + block_size - 1) // block_size
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, qk_head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KD),
        order=(1, 0),
    )
    s_ptrs = tl.make_block_ptr(
        base=score_ptr + seq_start * stride_s_q + pid_h * stride_s_h,
        shape=(q_len, block_num),
        strides=(stride_s_q, stride_s_k),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    if HAS_SINK:
        off_d = tl.arange(0, BLOCK_SIZE_KD)
        sink = tl.load(
            sink_ptr + pid_h * stride_sink_h + off_d * stride_sink_d,
            mask=off_d < qk_head_dim,
            other=0,
        )
    # init statistics
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_kd = tl.arange(0, BLOCK_SIZE_KD)
    off_vd = tl.arange(0, BLOCK_SIZE_VD)
    off_bpk = tl.arange(0, BLOCKS_PER_K_BLOCK)
    kd_mask = off_kd < qk_head_dim
    vd_mask = off_vd < v_head_dim
    if HAS_SINK:
        m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
        lse_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
        qsink = tl.sum(q * sink[None, :], axis=1) * sm_scale_log2e  # (BLOCK_SIZE_Q,)
        m_i += qsink
        lse_i += qsink
    else:
        m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_VD), 0, dtype=tl.float32)
    # attention
    diag_start = (prefix_len + pid_q * BLOCK_SIZE_Q) // BLOCK_SIZE_K * BLOCK_SIZE_K
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        # paged load K via req_to_token: pos -> slot -> k_cache
        pos = i + off_k
        pos_mask = pos < seq_len
        slots = tl.load(
            req_to_token_ptr + sid * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)
        slots = (slots + max_slots) % max_slots  # safety against negative
        # k shape: [BLOCK_SIZE_KD, BLOCK_SIZE_K] (transposed for tl.dot)
        k = tl.load(
            k_cache_ptr
            + slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_kd[:, None] * stride_k_d,
            mask=kd_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        # compute qk
        qk = tl.dot(q, k) * sm_scale_log2e
        if i >= diag_start:
            qk = tl.where(off_q[:, None] >= (i + off_k)[None, :], qk, float("-inf"))
        # K boundary mask: positions beyond seq_len contribute -inf
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        # save score
        score = tl.reshape(
            qk, (BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK, block_size), can_reorder=False
        )
        sub_max = tl.max(score, axis=2)
        if SCORE_TYPE == "max":
            score = sub_max
        else:  # "lse"
            score = sub_max + tl.log2(
                tl.sum(tl.exp2(score - sub_max[:, :, None]), axis=2)
            )
            score = tl.where(score != score, float("-inf"), score)
        if use_gumbel_topk:
            local_seed = (pid_h | (pid_b << 7) | (gumbel_seed << 19)).to(tl.int32)
            noise_offset = (off_q << 13)[:, None] | (off_bpk + i // block_size)[None, :]
            noise = tl.rand(local_seed, offset=noise_offset)
            noise = tl.clamp(noise, min=1e-9, max=1 - 1e-9)
            noise = -tl.log(-tl.log(noise)) * 1.4426950409
            score += noise
        tl.store(s_ptrs, score.to(score_ptr.dtype.element_ty), boundary_check=(0, 1))
        if not DISABLE_INDEX_VALUE:
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            acc_o_scale = tl.exp2(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]
            v = tl.load(
                v_cache_ptr
                + slots[:, None] * stride_v_s
                + pid_kh * stride_v_h
                + off_vd[None, :] * stride_v_d,
                mask=pos_mask[:, None] & vd_mask[None, :],
                other=0.0,
            )
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        s_ptrs = tl.advance(s_ptrs, (0, BLOCKS_PER_K_BLOCK))
    if not DISABLE_INDEX_VALUE:
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + seq_start * stride_o_n + pid_h * stride_o_h,
            shape=(q_len, v_head_dim),
            strides=(stride_o_n, stride_o_d),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_VD),
            order=(1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


@torch.no_grad()
def triton_block_score(
    q: torch.Tensor,  # [total_q, num_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, qk_head_dim] (paged)
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len] int32
    slot_ids: torch.Tensor,  # [batch] int32
    cu_seqlens: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_k: int,
    sm_scale: Optional[float] = None,
    score_type: str = "max",
    score: Optional[torch.Tensor] = None,
):
    """Direct launcher for the DEPLOYED score-only path (disable_index_value=True):
    v_cache=None, sink=None, o=None. Mirrors flash_prefill_with_topk_index's launch
    args exactly, but skips the downstream _topk_index_kernel so the timed region is
    just the block-score kernel (the trace's 784 us/call op)."""
    assert q.dtype in (torch.bfloat16, torch.float16)
    assert k_cache.dtype == q.dtype
    assert cu_seqlens.dtype == torch.int32
    total_q, num_heads, qk_head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    v_head_dim = qk_head_dim  # placeholder; V never loaded
    gqa_group_size = num_heads // num_kv_heads
    batch_size = cu_seqlens.shape[0] - 1
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    max_seqblock_k = triton.cdiv(max_seqlen_k, block_size_k)
    if score is None:
        score = torch.full(
            (num_heads, total_q, max_seqblock_k),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        score.fill_(float("-inf"))

    def grid(META):
        return (
            triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]),
            batch_size * num_heads,
        )

    _flash_attn_fwd_with_block_score_kernel[grid](
        q,
        k_cache,
        None,  # v_cache
        None,  # sink
        None,  # o
        score,
        req_to_token,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        slot_ids,
        max_slots,
        num_heads,
        gqa_group_size,
        qk_head_dim,
        v_head_dim,
        block_size_k,
        sm_scale,
        False,  # use_gumbel_topk
        1,  # gumbel_seed
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        0,
        0,
        0,  # v strides
        0,
        0,  # sink strides
        0,
        0,
        0,  # o strides
        score.stride(0),
        score.stride(1),
        score.stride(2),
        req_to_token.stride(0),
        SCORE_TYPE=score_type,
        DISABLE_INDEX_VALUE=True,
    )
    return score
