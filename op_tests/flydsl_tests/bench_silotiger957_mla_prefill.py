# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SILOTIGER-957 apples-to-apples MLA prefill bench (gfx942).

What this measures (and what it does not)
----------------------------------------
Ticket https://amd.atlassian.net/browse/SILOTIGER-957 times a **per-MLA-layer
attention block** on 8x MI325X vs 8x H200 (Kineto, ISL 128K / OSL 1K /
concurrency 4, ``max_num_batched_tokens=4096``).

On AMD that block is **not** one 128K FMHA. Rank-0 Kineto (ticket) is:

    context FMHA (non-causal, LSE)  1.54 calls x 3145.0 us = 4835.4 us
    new-token FMHA (causal)         1.00 call  x  150.8 us =  150.8 us
    kv_b_proj decompress + LSE merge                     ~  389   us
    ---------------------------------------------------------------
    layer total                                          5375.5 us

H200 FLASHMLA is published only as the **layer total** (2825.6 us at 128K,
376.8 us at 8K). The two backends decompose work differently; the ticket
explicitly does not compare per-kernel.

This harness therefore has three matched columns:

1. **Isolated FMHA** — ticket reproduction: packed THD, QK=192 V=128 bf16,
   H=12, ``return_lse=True``. Default context shape is ``Sq=4096, Skv=42700``
   (ticket snippet), plus the Skv sweep they asked for.
2. **E2E chunk** — ``kv_b_proj`` (512 -> 12*(128+128)) + K concat (nope||rope)
   + the same FMHA. This is one context chunk of
   ``AiterFlashAttnPrefillBackend.run_prefill_context_chunk``.
3. **Layer reconstruction** — ``1.54 * context_e2e + 1.0 * causal_fmha``,
   compared to the published H200 / MI325X layer bars. Not a vLLM process.

This host is gfx942. Ticket HW is 8x MI325X (they list 304 CU). Use ratios
and the published bars; do not treat a kernel-only CUDA-event number as a
Kineto layer win.

Usage::

    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/bench_silotiger957_mla_prefill.py
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/bench_silotiger957_mla_prefill.py --quick
"""

from __future__ import annotations

import argparse
import math
import sys

import pandas as pd
import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl import flash_attn_varlen_gfx942
from aiter.ops.mha import flash_attn_varlen_func
from aiter.test_common import checkAllclose, run_perftest

SUPPORTED_GFX = ("gfx942",)

HEAD_DIM_QK = 192
HEAD_DIM_V = 128
QK_NOPE = 128
QK_ROPE = 64
KV_LORA = 512
HEADS = 12
SQ = 4096

# Ticket isolated reproduction (ISL 128K chunked prefill).
TICKET_SKV = 42700
TICKET_CTX_CALLS = 1.54
TICKET_LAYER_MI325X_US = {8000: 438.0, 128000: 5375.5}
TICKET_LAYER_H200_US = {8000: 376.8, 128000: 2825.6}
TICKET_CTX_FMHA_US = 3145.0
TICKET_CAUSAL_FMHA_US = 150.8
TICKET_DECOMP_US = 389.0

WARMUP = 5
ITERS = 21


def _cos_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x64, y64 = x.double(), y.double()
    return 1 - 2 * (x64 * y64).sum().item() / max(
        (x64 * x64 + y64 * y64).sum().item(), 1e-12
    )


def _tflops(sq: int, sk: int, h: int, causal: bool, us: float) -> float:
    flop = h * (2 * sq * sk * HEAD_DIM_QK + 2 * sq * sk * HEAD_DIM_V)
    if causal:
        flop //= 2
    if us <= 0:
        return float("inf")
    return flop / us / 1e6


def _cu_seqlens(n: int) -> torch.Tensor:
    return torch.tensor([0, n], dtype=torch.int32, device="cuda")


def _time(fn) -> float:
    _, us = run_perftest(
        fn, num_iters=ITERS, num_warmup=WARMUP, use_cuda_event=True
    )
    return float(us)


def _fmha_asm(q, k, v, cu_q, cu_k, sq, sk, scale, causal, *, lse: bool, out=None):
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sq,
        sk,
        softmax_scale=scale,
        causal=causal,
        return_lse=lse,
        out=out,
    )


def _fmha_flydsl(q, k, v, cu_q, cu_k, sq, sk, scale, causal, *, lse: bool, out=None):
    return flash_attn_varlen_gfx942(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sq,
        sk,
        softmax_scale=scale,
        causal=causal,
        return_lse=lse,
        out=out,
    )


def _make_qkv(sq: int, sk: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(sq, HEADS, HEAD_DIM_QK, dtype=torch.bfloat16, device="cuda", generator=g)
    k = torch.randn(sk, HEADS, HEAD_DIM_QK, dtype=torch.bfloat16, device="cuda", generator=g)
    v = torch.randn(sk, HEADS, HEAD_DIM_V, dtype=torch.bfloat16, device="cuda", generator=g)
    return q, k, v, _cu_seqlens(sq), _cu_seqlens(sk)


def _make_e2e(sq: int, sk: int, seed: int = 0):
    """Latent cache row + kv_b_proj weights for one context chunk."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(sq, HEADS, HEAD_DIM_QK, dtype=torch.bfloat16, device="cuda", generator=g)
    kv_c = torch.randn(sk, KV_LORA, dtype=torch.bfloat16, device="cuda", generator=g)
    k_pe = torch.randn(sk, 1, QK_ROPE, dtype=torch.bfloat16, device="cuda", generator=g)
    weight = torch.randn(
        HEADS * (QK_NOPE + HEAD_DIM_V),
        KV_LORA,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    return q, kv_c, k_pe, weight, _cu_seqlens(sq), _cu_seqlens(sk)


def _decompress(kv_c, k_pe, weight):
    kv_nope = F.linear(kv_c, weight).view(-1, HEADS, QK_NOPE + HEAD_DIM_V)
    k_nope, v = kv_nope.split([QK_NOPE, HEAD_DIM_V], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, HEADS, -1)], dim=-1)
    return k, v


def _run_isolated(sq: int, sk: int, causal: bool, lse: bool, *, bench: bool):
    scale = 1.0 / math.sqrt(HEAD_DIM_QK)
    q, k, v, cu_q, cu_k = _make_qkv(sq, sk)
    tag = f"fmha Sq={sq} Sk={sk} H={HEADS} causal={causal} lse={lse}"

    out_asm = _fmha_asm(q, k, v, cu_q, cu_k, sq, sk, scale, causal, lse=lse)
    out_fd = _fmha_flydsl(q, k, v, cu_q, cu_k, sq, sk, scale, causal, lse=lse)
    if lse:
        out_asm, lse_asm = out_asm
        out_fd, lse_fd = out_fd
        _ = lse_asm, lse_fd
    cos = _cos_diff(out_asm, out_fd)
    err = checkAllclose(
        out_asm.float(),
        out_fd.float(),
        rtol=2e-2,
        atol=2e-2,
        printLog=False,
        msg=f"[{tag}] ",
    )
    row = {
        "kind": "fmha",
        "Sq": sq,
        "Sk": sk,
        "causal": causal,
        "lse": lse,
        "cos": cos,
        "pass": bool(cos < 1e-4 and err < 0.05),
        "asm_us": None,
        "flydsl_us": None,
        "asm_tflops": None,
        "flydsl_tflops": None,
        "flydsl_vs_asm": None,
        "ticket_ctx_fmha_us": TICKET_CTX_FMHA_US if (not causal and sk == TICKET_SKV) else None,
    }
    print(f"  {tag}  cos={cos:.3e} pass={row['pass']}")
    if not bench:
        return row

    o_asm = torch.empty(sq, HEADS, HEAD_DIM_V, dtype=q.dtype, device=q.device)
    o_fd = torch.empty_like(o_asm)

    def _asm():
        return _fmha_asm(q, k, v, cu_q, cu_k, sq, sk, scale, causal, lse=lse, out=o_asm)

    def _fd():
        return _fmha_flydsl(q, k, v, cu_q, cu_k, sq, sk, scale, causal, lse=lse, out=o_fd)

    row["asm_us"] = _time(_asm)
    row["flydsl_us"] = _time(_fd)
    row["asm_tflops"] = _tflops(sq, sk, HEADS, causal, row["asm_us"])
    row["flydsl_tflops"] = _tflops(sq, sk, HEADS, causal, row["flydsl_us"])
    row["flydsl_vs_asm"] = row["asm_us"] / row["flydsl_us"]
    print(
        f"    asm={row['asm_us']:.1f} us ({row['asm_tflops']:.1f} TFLOPS)  "
        f"flydsl={row['flydsl_us']:.1f} us ({row['flydsl_tflops']:.1f} TFLOPS)  "
        f"flydsl/asm={row['flydsl_vs_asm']:.3f}x"
    )
    return row


def _run_e2e(sq: int, sk: int, *, bench: bool):
    scale = 1.0 / math.sqrt(HEAD_DIM_QK)
    q, kv_c, k_pe, weight, cu_q, cu_k = _make_e2e(sq, sk)
    tag = f"e2e  Sq={sq} Sk={sk} decomp+concat+fmha lse=True"

    def _one(fmha):
        k, v = _decompress(kv_c, k_pe, weight)
        return fmha(q, k, v, cu_q, cu_k, sq, sk, scale, False, lse=True)

    out_asm, _ = _one(_fmha_asm)
    out_fd, _ = _one(_fmha_flydsl)
    cos = _cos_diff(out_asm, out_fd)
    err = checkAllclose(
        out_asm.float(),
        out_fd.float(),
        rtol=2e-2,
        atol=2e-2,
        printLog=False,
        msg=f"[{tag}] ",
    )
    row = {
        "kind": "e2e",
        "Sq": sq,
        "Sk": sk,
        "causal": False,
        "lse": True,
        "cos": cos,
        "pass": bool(cos < 1e-4 and err < 0.05),
        "asm_us": None,
        "flydsl_us": None,
        "decomp_us": None,
        "asm_fmha_us": None,
        "flydsl_fmha_us": None,
        "flydsl_vs_asm": None,
    }
    print(f"  {tag}  cos={cos:.3e} pass={row['pass']}")
    if not bench:
        return row

    k_buf = torch.empty(sk, HEADS, HEAD_DIM_QK, dtype=q.dtype, device=q.device)
    v_buf = torch.empty(sk, HEADS, HEAD_DIM_V, dtype=q.dtype, device=q.device)
    o_asm = torch.empty(sq, HEADS, HEAD_DIM_V, dtype=q.dtype, device=q.device)
    o_fd = torch.empty_like(o_asm)

    def _decomp_only():
        return _decompress(kv_c, k_pe, weight)

    def _asm_e2e():
        k, v = _decompress(kv_c, k_pe, weight)
        return _fmha_asm(q, k, v, cu_q, cu_k, sq, sk, scale, False, lse=True, out=o_asm)

    def _fd_e2e():
        k, v = _decompress(kv_c, k_pe, weight)
        return _fmha_flydsl(q, k, v, cu_q, cu_k, sq, sk, scale, False, lse=True, out=o_fd)

    def _asm_fmha_only():
        return _fmha_asm(
            q, k_buf, v_buf, cu_q, cu_k, sq, sk, scale, False, lse=True, out=o_asm
        )

    def _fd_fmha_only():
        return _fmha_flydsl(
            q, k_buf, v_buf, cu_q, cu_k, sq, sk, scale, False, lse=True, out=o_fd
        )

    row["decomp_us"] = _time(_decomp_only)
    k_fill, v_fill = _decompress(kv_c, k_pe, weight)
    k_buf.copy_(k_fill)
    v_buf.copy_(v_fill)
    torch.cuda.synchronize()
    row["asm_fmha_us"] = _time(_asm_fmha_only)
    row["flydsl_fmha_us"] = _time(_fd_fmha_only)
    row["asm_us"] = _time(_asm_e2e)
    row["flydsl_us"] = _time(_fd_e2e)
    row["flydsl_vs_asm"] = row["asm_us"] / row["flydsl_us"]
    print(
        f"    decomp={row['decomp_us']:.1f} us  "
        f"asm_e2e={row['asm_us']:.1f} us (fmha {row['asm_fmha_us']:.1f})  "
        f"flydsl_e2e={row['flydsl_us']:.1f} us (fmha {row['flydsl_fmha_us']:.1f})  "
        f"flydsl/asm={row['flydsl_vs_asm']:.3f}x"
    )
    return row


def _print_layer(ctx_row, causal_row, decomp_us: float):
    ctx_asm = ctx_row.get("asm_us") or ctx_row.get("asm_fmha_us")
    ctx_fd = ctx_row.get("flydsl_us") or ctx_row.get("flydsl_fmha_us")
    if ctx_asm is None or causal_row.get("asm_us") is None:
        return
    asm_layer = TICKET_CTX_CALLS * ctx_asm + causal_row["asm_us"] + decomp_us
    fd_layer = TICKET_CTX_CALLS * ctx_fd + causal_row["flydsl_us"] + decomp_us
    h200 = TICKET_LAYER_H200_US[128000]
    mi = TICKET_LAYER_MI325X_US[128000]
    print("\nLayer reconstruction (ISL 128K call mix from the ticket):")
    print(f"  1.54*context + 1*causal + decomp")
    print(f"  this-host ASM    {asm_layer:8.1f} us   vs ticket MI325X {mi:.1f}   vs H200 {h200:.1f}")
    print(f"  this-host FlyDSL {fd_layer:8.1f} us   H200/ASM={h200 / asm_layer:.3f}  H200/FlyDSL={h200 / fd_layer:.3f}")
    print(
        "  A layer win vs H200 requires this reconstruction <= "
        f"{h200:.1f} us on comparable HW. This box is not H200."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Ticket 42700 + causal 4096 only (no Skv sweep, no 128K)",
    )
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()
    bench = not args.no_bench

    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        print(f"Skipping: need gfx942, got {gfx}")
        return 0

    props = torch.cuda.get_device_properties(0)
    print(
        f"host gfx={gfx} name={props.name!r} CU={get_cu_num()}  "
        f"ticket bars are MI325X Kineto vs H200 FLASHMLA layer totals"
    )
    print(
        f"warmup={WARMUP} iters={ITERS} cuda_event  "
        f"H=12 QK=192 V=128 bf16 return_lse=True"
    )

    rows = []
    failed = False

    def _take(row):
        rows.append(row)
        if not row["pass"]:
            nonlocal failed
            failed = True

    # Tiny compile/correctness first.
    _take(_run_isolated(64, 256, False, True, bench=False))

    skv_list = [TICKET_SKV] if args.quick else [8192, 16384, 32768, TICKET_SKV, 65536, 131072]
    print("\n=== Isolated context FMHA (ticket kernel) ===")
    ctx_ticket = None
    for sk in skv_list:
        row = _run_isolated(SQ, sk, False, True, bench=bench)
        _take(row)
        if sk == TICKET_SKV:
            ctx_ticket = row

    print("\n=== Isolated new-token causal FMHA ===")
    causal_row = _run_isolated(SQ, SQ, True, True, bench=bench)
    _take(causal_row)

    print("\n=== E2E one context chunk (kv_b_proj + concat + FMHA) ===")
    e2e_sk = TICKET_SKV if args.quick else TICKET_SKV
    e2e_row = _run_e2e(SQ, e2e_sk, bench=bench)
    _take(e2e_row)

    if bench and ctx_ticket is not None:
        decomp_us = e2e_row.get("decomp_us") or TICKET_DECOMP_US
        _print_layer(ctx_ticket, causal_row, decomp_us)

    df = pd.DataFrame(rows)
    cols = [
        c
        for c in (
            "kind",
            "Sq",
            "Sk",
            "causal",
            "lse",
            "cos",
            "pass",
            "decomp_us",
            "asm_us",
            "flydsl_us",
            "asm_fmha_us",
            "flydsl_fmha_us",
            "asm_tflops",
            "flydsl_tflops",
            "flydsl_vs_asm",
            "ticket_ctx_fmha_us",
        )
        if c in df.columns
    ]
    print("\n" + df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
