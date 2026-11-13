# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL gfx9_fp8 ``_f32_to_bf16``: bit-exact RNE (NaN to 0x7FFF) and MFMA-safe.

The helper used inline asm to round, and LLVM does not pad MFMA destinations that are
read inside inline asm. Fed an accumulator straight from an MFMA it read stale values
(64 of 256 wrong in a one-MFMA kernel on gfx942 and gfx950). It is now plain integer
RNE. This checks the conversion against torch's cast over every edge class (NaN, inf,
zeros, denormals, ties, max finite, every high half) and the MFMA-fed case against an
integer RNE written in the test."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import pytest
import torch
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import common as fxh
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import _ptr
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


def rsrc(ptr):
    view = fx.make_view(ptr, fx.make_layout(1, 1))
    return fx.rocdl.get_buffer_rsrc(
        fx.get_iter(fx.rocdl.make_buffer_tensor(view, max_size=True))
    )


@flyc.kernel
def conv_kernel(p_in: fx.Pointer, p_vec: fx.Pointer, p_sc: fx.Pointer):
    idx = gpu.block_idx.x * 64 + gpu.thread_idx.x
    in_rs = rsrc(fxh._as_ptr(p_in, fx.Float32))
    vec_rs = rsrc(fxh._as_ptr(p_vec, fx.BFloat16))
    sc_rs = rsrc(fxh._as_ptr(p_sc, fx.BFloat16))
    raw = Vec(buffer_ops.buffer_load(in_rs, idx * 2, vec_width=2, dtype=T.i32))
    pair = raw.bitcast(fx.Float32)
    both = fxh._f32_to_bf16(Vec.from_elements([pair[0], pair[1]], fx.Float32))
    buffer_ops.buffer_store(
        both.bitcast(fx.Int32)[0], vec_rs, idx * 4, offset_is_bytes=True
    )
    one = fxh._f32_to_bf16(pair[0])
    buffer_ops.buffer_store(one.bitcast(fx.Int16), sc_rs, idx * 2, offset_is_bytes=True)


@flyc.jit
def conv_launch(
    p_in: fx.Pointer,
    p_vec: fx.Pointer,
    p_sc: fx.Pointer,
    nblocks: fx.Int32,
    stream: fx.Stream,
):
    CompilationContext.get_current()
    conv_kernel(p_in, p_vec, p_sc).launch(
        grid=(nblocks, 1, 1), block=(64, 1, 1), stream=stream
    )


def int_rne(v):
    bits = v.bitcast(fx.Uint32)
    return (bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))) >> 16


@flyc.kernel
def mfma_kernel(p_a: fx.Pointer, p_b: fx.Pointer, p_asm: fx.Pointer, p_int: fx.Pointer):
    lane = gpu.thread_idx.x
    c, q = lane % 16, lane // 16
    a_rs = rsrc(fxh._as_ptr(p_a, fx.BFloat16))
    b_rs = rsrc(fxh._as_ptr(p_b, fx.BFloat16))
    a = Vec(
        buffer_ops.buffer_load(a_rs, (c * 16 + q * 4) >> 1, vec_width=2, dtype=T.i32)
    )
    b = Vec(
        buffer_ops.buffer_load(b_rs, (c * 16 + q * 4) >> 1, vec_width=2, dtype=T.i32)
    )
    acc = Vec.filled(4, 0.0, fx.Float32)
    acc = Vec(
        rocdl.mfma_f32_16x16x16bf16_1k(
            T.f32x4, [a.bitcast(fx.Int16), b.bitcast(fx.Int16), acc, 0, 0, 0]
        )
    )
    asm_rs = rsrc(fxh._as_ptr(p_asm, fx.BFloat16))
    int_rs = rsrc(fxh._as_ptr(p_int, fx.BFloat16))
    for h in range_constexpr(2):
        pair = fxh._f32_to_bf16(
            Vec.from_elements([acc[2 * h], acc[2 * h + 1]], fx.Float32)
        )
        w_int = int_rne(acc[2 * h]) | (int_rne(acc[2 * h + 1]) << 16)
        off = (c * 16 + q * 4 + 2 * h) * 2
        buffer_ops.buffer_store(
            pair.bitcast(fx.Int32)[0], asm_rs, off, offset_is_bytes=True
        )
        buffer_ops.buffer_store(
            w_int.bitcast(fx.Int32), int_rs, off, offset_is_bytes=True
        )


@flyc.jit
def mfma_launch(
    p_a: fx.Pointer,
    p_b: fx.Pointer,
    p_asm: fx.Pointer,
    p_int: fx.Pointer,
    stream: fx.Stream,
):
    CompilationContext.get_current()
    mfma_kernel(p_a, p_b, p_asm, p_int).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream
    )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or get_gfx() not in ("gfx942", "gfx950"),
    reason="Needs gfx942 or gfx950",
)


def _edge_classes():
    rng = np.random.default_rng(0)
    cls = {}
    cls["nan"] = np.concatenate(
        [
            np.array(
                [0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF]
                + [0xFFFFFFFF, 0x7FBFFFFF, 0x7F8FFFFF, 0xFF8FFFFF],
                dtype=np.uint32,
            ),
            (rng.integers(1, 1 << 23, 4096, dtype=np.uint32) | np.uint32(0x7F800000))
            | (rng.integers(0, 2, 4096, dtype=np.uint32) << 31),
        ]
    )
    cls["inf_zero"] = np.array([0x7F800000, 0xFF800000, 0, 0x80000000], dtype=np.uint32)
    cls["denormal"] = np.concatenate(
        [
            np.array(
                [1, 0x7FFF, 0x8000, 0x8001, 0xFFFF, 0x7FFFFF, 0x400000, 0x807FFFFF]
                + [0x80000001, 0x80008000],
                dtype=np.uint32,
            ),
            rng.integers(1, 1 << 23, 4096, dtype=np.uint32)
            | (rng.integers(0, 2, 4096, dtype=np.uint32) << 31),
        ]
    )
    hi = np.arange(1 << 16, dtype=np.uint32) << 16
    lo = np.array([0, 0x7FFF, 0x8000, 0x8001, 0xFFFF], dtype=np.uint32)
    cls["ties_and_every_high_half"] = (hi[:, None] | lo[None, :]).reshape(-1)
    cls["max_finite"] = np.array(
        [0x7F7FFFFF, 0xFF7FFFFF, 0x7F7F8000, 0x7F7F7FFF, 0x7F7F8001], dtype=np.uint32
    )
    cls["random"] = rng.integers(0, 1 << 32, 1 << 18, dtype=np.uint64).astype(np.uint32)
    return cls


@pytest.fixture(scope="module")
def converted():
    cls = _edge_classes()
    names = list(cls)
    bits = np.concatenate([cls[n] for n in names])
    n = len(bits)
    padded = np.concatenate([bits, np.zeros((-n) % 128, dtype=np.uint32)])
    x = torch.from_numpy(padded.view(np.float32).copy()).cuda()
    out_vec = torch.zeros(len(padded), dtype=torch.bfloat16, device="cuda")
    out_scalar = torch.zeros(len(padded) // 2, dtype=torch.bfloat16, device="cuda")
    _run_compiled(
        conv_launch,
        _ptr(x),
        _ptr(out_vec),
        _ptr(out_scalar),
        len(padded) // 128,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    f = torch.from_numpy(bits.view(np.float32).copy())
    # torch's cast is RNE; ck_tile canonicalises every NaN to 0x7FFF.
    gold = f.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16).copy()
    gold[np.isnan(bits.view(np.float32))] = 0x7FFF
    vec = out_vec.cpu().view(torch.int16).numpy().view(np.uint16)[:n]
    scalar = np.zeros(n, dtype=np.uint16)
    scalar[0::2] = (
        out_scalar.cpu().view(torch.int16).numpy().view(np.uint16)[: len(scalar[0::2])]
    )
    return names, {nm: len(cls[nm]) for nm in names}, gold, vec, scalar


@pytest.mark.parametrize("path", ["vector", "scalar"])
@pytest.mark.parametrize(
    "klass",
    ["nan", "inf_zero", "denormal", "ties_and_every_high_half", "max_finite", "random"],
)
def test_conversion_matches_ck_tile_rne(converted, path, klass):
    names, sizes, gold, vec, scalar = converted
    start = sum(sizes[n] for n in names[: names.index(klass)])
    idx = np.arange(start, start + sizes[klass])
    if path == "scalar":
        idx = idx[idx % 2 == 0]
        got = scalar
    else:
        got = vec
    bad = np.nonzero(got[idx] != gold[idx])[0]
    assert len(bad) == 0, (
        f"{len(bad)} of {len(idx)} differ, first: input "
        f"{hex(int(_edge_classes()[klass][idx[bad[0]] - start]))} got "
        f"{hex(int(got[idx[bad[0]]]))} want {hex(int(gold[idx[bad[0]]]))}"
    )


@pytest.mark.parametrize("seed", range(5))
def test_conversion_of_mfma_accumulators_is_exact(seed):
    torch.manual_seed(seed)
    a = torch.randn(16, 16, device="cuda").to(torch.bfloat16)
    b = torch.randn(16, 16, device="cuda").to(torch.bfloat16)
    helper = torch.zeros(16, 16, dtype=torch.bfloat16, device="cuda")
    reference = torch.zeros(16, 16, dtype=torch.bfloat16, device="cuda")
    _run_compiled(
        mfma_launch,
        _ptr(a),
        _ptr(b),
        _ptr(helper),
        _ptr(reference),
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    assert int((helper != reference).sum()) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
