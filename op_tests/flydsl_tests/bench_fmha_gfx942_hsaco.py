#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Correctness-gated A/B bench for a reassembled gfx942 FlyDSL FMHA HSACO.

The code object must be the non-causal, return-LSE, bf16 H=12/QK=192/V=128
variant emitted by ``FLYDSL_DUMP_IR=1``.  This launcher intentionally mirrors
FlyDSL's dynamic-layout C ABI instead of introducing a second kernel ABI.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import statistics
from pathlib import Path

import torch
from hip import hip
from hip._hip_helpers import HipModuleLaunchKernel_extra

from aiter.ops.flydsl import flash_attn_varlen_gfx942
from aiter.ops.mha import flash_attn_varlen_func
from aiter.test_common import run_perftest


KERNEL = "fmha_varlen_gfx942_kernel_0"
HEADS = 12
HEAD_DIM_QK = 192
HEAD_DIM_V = 128
BLOCK_M = 128


class _Layout3(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("shape", ctypes.c_int32 * 3), ("stride", ctypes.c_int64 * 2)]


class _Layout2(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("shape", ctypes.c_int32 * 2), ("stride", ctypes.c_int64 * 1)]


class _Layout1(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("shape", ctypes.c_int32 * 1)]


def _hip_check(result):
    error = result[0] if isinstance(result, tuple) else result
    if error != hip.hipError_t.hipSuccess:
        raise RuntimeError(hip.hipGetErrorString(error))
    return result


def _layout(tensor: torch.Tensor):
    unit_dim = tensor.stride().index(1)
    shape = tuple(tensor.shape)
    strides = tuple(
        tensor.stride(dim) for dim in range(tensor.ndim) if dim != unit_dim
    )
    layout_type = {1: _Layout1, 2: _Layout2, 3: _Layout3}[tensor.ndim]
    return layout_type(shape, strides) if strides else layout_type(shape)


class HsacoFmha:
    """Loaded custom code object with a reusable, prepacked HIP launch buffer."""

    def __init__(self, path: Path):
        code = path.read_bytes()
        _, self.module = _hip_check(hip.hipModuleLoadData(code))
        _, self.function = _hip_check(
            hip.hipModuleGetFunction(self.module, KERNEL.encode())
        )

    def prepare(self, q, k, v, out, lse, cu_q, cu_k, max_seqlen_q):
        args = []
        for tensor in (q, k, v, out, lse, cu_q, cu_k):
            args.extend((ctypes.c_void_p(tensor.data_ptr()), _layout(tensor)))
        args.extend(
            (
                ctypes.c_int32(max_seqlen_q),
                ctypes.c_int32(q.shape[0]),
            )
        )
        extra = HipModuleLaunchKernel_extra(tuple(args))
        grid_x = (
            (cu_q.numel() - 1)
            * ((max_seqlen_q + BLOCK_M - 1) // BLOCK_M)
            * HEADS
        )

        def launch():
            _hip_check(
                hip.hipModuleLaunchKernel(
                    self.function,
                    grid_x,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    0,
                    None,
                    extra,
                )
            )

        # Keep ctypes entries alive for the lifetime of the prepared launch.
        launch._abi_args = args
        launch._abi_extra = extra
        return launch

    def close(self):
        if self.module is not None:
            _hip_check(hip.hipModuleUnload(self.module))
            self.module = None


def _make_qkv(sq: int, sk: int):
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(
        sq, HEADS, HEAD_DIM_QK, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    k = torch.randn(
        sk, HEADS, HEAD_DIM_QK, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    v = torch.randn(
        sk, HEADS, HEAD_DIM_V, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, sk], dtype=torch.int32, device="cuda")
    return q, k, v, cu_q, cu_k


def _time(fn, warmup: int, iters: int) -> float:
    _, microseconds = run_perftest(
        fn,
        num_iters=iters,
        num_warmup=warmup,
        use_cuda_event=True,
    )
    return float(microseconds)


def _cos_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x64, y64 = x.double(), y.double()
    return 1 - 2 * (x64 * y64).sum().item() / max(
        (x64 * x64 + y64 * y64).sum().item(), 1e-12
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--co", type=Path, required=True)
    parser.add_argument(
        "--reference-co",
        type=Path,
        help="optional second HSACO for direct interleaved assembly A/B",
    )
    parser.add_argument("--sq", type=int, default=4096)
    parser.add_argument("--sk", type=int, default=42700)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=21)
    args = parser.parse_args()

    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx942":
        raise RuntimeError(f"requires gfx942, got {arch}")

    q, k, v, cu_q, cu_k = _make_qkv(args.sq, args.sk)
    scale = 1 / math.sqrt(HEAD_DIM_QK)
    custom_o = torch.empty(
        args.sq, HEADS, HEAD_DIM_V, dtype=q.dtype, device=q.device
    )
    custom_lse = torch.empty(HEADS, args.sq, dtype=torch.float32, device=q.device)
    flydsl_o = torch.empty_like(custom_o)
    asm_o = torch.empty_like(custom_o)

    module = HsacoFmha(args.co)
    custom = module.prepare(
        q, k, v, custom_o, custom_lse, cu_q, cu_k, args.sq
    )
    reference_module = None
    reference = None
    reference_o = None
    reference_lse = None
    if args.reference_co is not None:
        reference_o = torch.empty_like(custom_o)
        reference_lse = torch.empty_like(custom_lse)
        reference_module = HsacoFmha(args.reference_co)
        reference = reference_module.prepare(
            q, k, v, reference_o, reference_lse, cu_q, cu_k, args.sq
        )

    def flydsl():
        return flash_attn_varlen_gfx942(
            q,
            k,
            v,
            cu_q,
            cu_k,
            args.sq,
            args.sk,
            softmax_scale=scale,
            causal=False,
            return_lse=True,
            out=flydsl_o,
        )

    def production_asm():
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            args.sq,
            args.sk,
            softmax_scale=scale,
            causal=False,
            return_lse=True,
            out=asm_o,
        )

    ref_o, ref_lse = flydsl()
    custom()
    if reference is not None:
        reference()
    torch.cuda.synchronize()
    output_equal = torch.equal(ref_o, custom_o)
    lse_equal = torch.equal(ref_lse, custom_lse)
    print(
        f"correctness output_equal={output_equal} lse_equal={lse_equal} "
        f"output_cos_diff={_cos_diff(ref_o, custom_o):.3e} "
        f"output_max_abs={(ref_o.float() - custom_o.float()).abs().max().item():.3e} "
        f"lse_max_abs={(ref_lse - custom_lse).abs().max().item():.3e}"
    )
    if reference is not None:
        reference_equal = torch.equal(custom_o, reference_o) and torch.equal(
            custom_lse, reference_lse
        )
        print(f"candidate_reference_equal={reference_equal}")
        output_equal = output_equal and reference_equal
    if not (output_equal and lse_equal):
        module.close()
        if reference_module is not None:
            reference_module.close()
        return 2

    methods = {
        "flydsl": flydsl,
        "candidate": custom,
        "production_asm": production_asm,
    }
    if reference is not None:
        methods["reference_co"] = reference
    samples = {name: [] for name in methods}
    for round_index in range(args.rounds):
        names = list(methods)
        if round_index % 2:
            names.reverse()
        for name in names:
            samples[name].append(_time(methods[name], args.warmup, args.iters))

    for name, values in samples.items():
        print(
            f"{name:14s} median={statistics.median(values):8.1f} us "
            f"min={min(values):8.1f} max={max(values):8.1f} "
            f"samples={','.join(f'{value:.1f}' for value in values)}"
        )
    flydsl_median = statistics.median(samples["flydsl"])
    candidate_median = statistics.median(samples["candidate"])
    asm_median = statistics.median(samples["production_asm"])
    print(
        f"candidate/flydsl={candidate_median / flydsl_median:.4f} "
        f"candidate/production_asm={candidate_median / asm_median:.4f}"
    )
    if reference is not None:
        reference_median = statistics.median(samples["reference_co"])
        print(f"candidate/reference_co={candidate_median / reference_median:.4f}")
    module.close()
    if reference_module is not None:
        reference_module.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
