# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Activation quant for the routed tokens, bit-exact with aiter's own quant ops.

fp8 per row (per_token_fp8: a row is a token, as aiter's
dynamic_per_token_scaled_quant; per_1x128_fp8: a row is 128 columns of a
token, as aiter's dynamic_per_group_scaled_quant):
    amax  = max |x| over the row (fp32, from 0 per token, from kAmaxFloor =
            1e-10 per group)
    scale = amax * float(1 / fp8_max)                       -> scales (fp32)
    q     = cvt_pk_fp8_f32(med3(x * rcp(scale), -fp8_max, fp8_max))
fp8_max is the arch's (240 for gfx942 e4m3fnuz, 448 for gfx950 e4m3fn).

MX (mx_fp4_round_up, mx_fp8_round_up; gfx950): a group is 32 columns of a
token, scale = RoundUp e8m0 of amax / max_pos (6 for fp4, 448 for fp8), values
by the hardware scaled conversion, as aiter's per_1x32_mx_quant_hip; scales
lane-native (quant_mx_block).

per_tensor_fp8 (static, a given scale): as aiter's static_per_tensor_quant,
q = cvt_pk_fp8_f32(med3(x * rcp(scale), -fp8_max, fp8_max)). Dynamic (no
scale, aiter's dynamic_per_tensor_quant) takes two passes: partial amaxes
(amax_partial_block), then the scale and the quant (quant_dynamic_tensor_block).

The helpers take a workgroup-relative block index, so a quant pass can run as
standalone kernels or in extra workgroups of the sort launch.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

BLOCK = 256
WAVE = 64
GROUP_LANES = 16  # lanes per 128-column group (8 bf16 each)
OOB = 0x7FFFFFFF
AMAX_FLOOR = 1e-10  # aiter::kAmaxFloor (csrc/include/quant_amax_floor.h)


def _raw(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _it(t):
    return fx.get_iter(fx.rocdl.make_buffer_tensor(t, max_size=True))


def _bf16x8_f32(v):
    """8 bf16 (4 dwords) -> 8 f32 (exact)."""
    out = []
    for d in v:
        out.append((d << fx.Int32(16)).bitcast(fx.Float32))
        out.append((d & fx.Int32(-65536)).bitcast(fx.Float32))
    return out


def _absmax(vals, acc):
    for x in vals:
        a = (x.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)).bitcast(fx.Float32)
        acc = (a > acc).select(a, acc)
    return acc


def _fp8x4(vals, inv, fp8_max):
    """4 f32 -> one dword of 4 fp8, as aiter's fp32_to_fp8_scaled_x4."""
    hi, lo = _raw(fx.Float32(fp8_max)), _raw(fx.Float32(-fp8_max))
    clamped = [fx.Float32(rocdl.fmed3(T.f32, _raw(x * inv), lo, hi)) for x in vals]
    pk = rocdl.cvt_pk_fp8_f32(
        T.i32, _raw(clamped[0]), _raw(clamped[1]), _raw(fx.Int32(0)), 0
    )
    pk = rocdl.cvt_pk_fp8_f32(T.i32, _raw(clamped[2]), _raw(clamped[3]), pk, 1)
    return fx.Int32(pk)


def _load8(x_it, elem):
    """8 bf16 at element offset ``elem`` (16-byte aligned) as 4 dwords; ``x_it``
    iterates the bf16 input viewed as int32."""
    v = fx.Vector(
        fx.ptr_load(
            x_it + fx.Int64(elem >> fx.Int32(1)),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    return [v[j] for j in range(4)]


def _fshuffle_xor(x, off, width):
    return (
        x.bitcast(fx.Int32)
        .shuffle_xor(fx.Int32(off), fx.Int32(width))
        .bitcast(fx.Float32)
    )


@flyc.jit
def quant_1x128_block(x_it, q_it, s_it, blk, n_groups, fp8_max):
    """16 groups of 128 columns per workgroup, from group ``blk * 16``; 16
    lanes per group, 8 columns per lane. Groups at or past n_groups are skipped
    (rows past the token count included)."""
    tid = gpu.thread_idx.x
    lane16 = tid % fx.Int32(GROUP_LANES)
    grp = blk * fx.Int32(BLOCK // GROUP_LANES) + tid // fx.Int32(GROUP_LANES)
    live = grp < n_groups
    col = grp * fx.Int32(128) + lane16 * fx.Int32(8)
    x = _bf16x8_f32(_load8(x_it, live.select(col, fx.Int32(0))))
    amax = _absmax(x, fx.Float32(AMAX_FLOOR))
    for sh in range_constexpr(4):
        other = _fshuffle_xor(amax, 1 << sh, GROUP_LANES)
        amax = (other > amax).select(other, amax)
    scale = amax * fx.Float32(1.0 / fp8_max)
    inv = fx.Float32(rocdl.rcp(T.f32, _raw(scale)))
    d0 = _fp8x4(x[0:4], inv, fp8_max)
    d1 = _fp8x4(x[4:8], inv, fp8_max)
    q = live.select(col >> fx.Int32(2), fx.Int32(OOB))
    fx.ptr_store(d0, q_it + fx.Int64(q))
    fx.ptr_store(d1, q_it + fx.Int64(live.select(q + fx.Int32(1), fx.Int32(OOB))))
    s = (live & (lane16 == fx.Int32(0))).select(grp, fx.Int32(OOB))
    fx.ptr_store(scale, s_it + fx.Int64(s))


@flyc.jit
def quant_token_block(x_it, q_it, s_it, red, row, n_rows, cols, fp8_max):
    """One token row per workgroup: 8 columns per thread per pass, max over the
    row through wave shuffles and ``red`` (WAVES floats of LDS)."""
    tid = gpu.thread_idx.x
    live = row < n_rows
    passes = -(-cols // (BLOCK * 8))
    vals, cols_at = [], []
    amax = fx.Float32(0.0)
    for p in range_constexpr(passes):
        c = fx.Int32(p * BLOCK * 8) + tid * fx.Int32(8)
        ok = live & (c < fx.Int32(cols))
        elem = row * fx.Int32(cols) + c
        x = _bf16x8_f32(_load8(x_it, ok.select(elem, fx.Int32(0))))
        x = [ok.select(v, fx.Float32(0.0)) for v in x]
        amax = _absmax(x, amax)
        vals.append(x)
        cols_at.append((ok, elem))
    amax = _block_max(amax, red)
    scale = amax * fx.Float32(1.0 / fp8_max)
    inv = fx.Float32(rocdl.rcp(T.f32, _raw(scale)))
    for p in range_constexpr(passes):
        ok, elem = cols_at[p]
        d0 = _fp8x4(vals[p][0:4], inv, fp8_max)
        d1 = _fp8x4(vals[p][4:8], inv, fp8_max)
        q = ok.select(elem >> fx.Int32(2), fx.Int32(OOB))
        fx.ptr_store(d0, q_it + fx.Int64(q))
        fx.ptr_store(d1, q_it + fx.Int64(ok.select(q + fx.Int32(1), fx.Int32(OOB))))
    s = (live & (tid == fx.Int32(0))).select(row, fx.Int32(OOB))
    fx.ptr_store(scale, s_it + fx.Int64(s))


def _block_max(amax, red):
    """Max of a per-thread f32 over the workgroup, through wave shuffles and
    ``red`` (WAVES floats of LDS); every thread gets it."""
    tid = gpu.thread_idx.x
    lane, wave = tid % fx.Int32(WAVE), tid // fx.Int32(WAVE)
    for sh in range_constexpr(6):
        other = _fshuffle_xor(amax, 1 << sh, WAVE)
        amax = (other > amax).select(other, amax)
    gpu.barrier()
    fx.ptr_store(amax, red + fx.Int64((lane == fx.Int32(0)).select(wave, fx.Int32(4))))
    gpu.barrier()
    amax = fx.ptr_load(red)
    for w in range_constexpr(1, BLOCK // WAVE):
        other = fx.ptr_load(red + fx.Int64(w))
        amax = (other > amax).select(other, amax)
    return amax


@flyc.jit
def _quant_tensor_elems(x_it, q_it, blk, n_rows, cols, inv, fp8_max):
    elem = (blk * fx.Int32(BLOCK) + gpu.thread_idx.x) * fx.Int32(8)
    live = elem < n_rows * fx.Int32(cols)
    x = _bf16x8_f32(_load8(x_it, live.select(elem, fx.Int32(0))))
    d0 = _fp8x4(x[0:4], inv, fp8_max)
    d1 = _fp8x4(x[4:8], inv, fp8_max)
    if live:
        q = elem >> fx.Int32(2)
        fx.ptr_store(d0, q_it + fx.Int64(q))
        fx.ptr_store(d1, q_it + fx.Int64(q + fx.Int32(1)))


@flyc.jit
def quant_static_block(x_it, q_it, s_it, blk, n_rows, cols, fp8_max):
    """Static per-tensor fp8 (aiter's static_per_tensor_quant): q =
    cvt(med3(x * rcp(scale[0]))), 8 columns per thread, 2048 per workgroup
    from element blk * 2048, over the first n_rows token rows."""
    inv = fx.Float32(rocdl.rcp(T.f32, _raw(fx.ptr_load(s_it))))
    _quant_tensor_elems(x_it, q_it, blk, n_rows, cols, inv, fp8_max)


# Dynamic per-tensor in one one-shot workgroup up to this M x K (register-
# resident; measured on MI300X: <= 5.0 us against 5.9-6.3 for one-shot + a
# quant launch; a 64K build is slower at every size).
SINGLE_WG_ELEMS = 1 << 15


def _block_max_bits(bits, red):
    """Signed int max of per-thread float bits over the workgroup (aiter's
    atomicMaxFloat for values >= 0); every thread gets it, as an f32."""
    tid = gpu.thread_idx.x
    lane, wave = tid % fx.Int32(WAVE), tid // fx.Int32(WAVE)
    for sh in range_constexpr(6):
        other = bits.shuffle_xor(fx.Int32(1 << sh), fx.Int32(WAVE))
        bits = (other > bits).select(other, bits)
    gpu.barrier()
    fx.ptr_store(
        bits.bitcast(fx.Float32),
        red + fx.Int64((lane == fx.Int32(0)).select(wave, fx.Int32(4))),
    )
    gpu.barrier()
    bits = fx.ptr_load(red).bitcast(fx.Int32)
    for w in range_constexpr(1, BLOCK // WAVE):
        other = fx.ptr_load(red + fx.Int64(w)).bitcast(fx.Int32)
        bits = (other > bits).select(other, bits)
    return bits.bitcast(fx.Float32)


@flyc.jit
def _amax_strided(x_it, red, first, n, step):
    """max |x| (from 0, NaN skipped, as aiter's per-row fmaxf) over the
    2048-element blocks first, first + step, ... of the first n elements."""
    tid = gpu.thread_idx.x
    blk_elems = fx.Int32(BLOCK * 8)
    n_blocks = (n + blk_elems - fx.Int32(1)) // blk_elems
    left = n_blocks - first
    iters = (left > fx.Int32(0)).select(
        (left + step - fx.Int32(1)) // step, fx.Int32(0)
    )
    amax = fx.Float32(0.0)
    for j in range(fx.Int32(0), iters, fx.Int32(1)):
        elem = (first + fx.Int32(j) * step) * blk_elems + tid * fx.Int32(8)
        ok = elem < n
        x = _bf16x8_f32(_load8(x_it, ok.select(elem, fx.Int32(0))))
        amax = _absmax([ok.select(v, fx.Float32(0.0)) for v in x], amax)
    return _block_max_bits(amax.bitcast(fx.Int32), red)


@flyc.jit
def amax_partial_block(x_it, p_it, red, blk, n_rows, cols, n_part):
    """Dynamic per-tensor, pass 1: workgroup ``blk`` of ``n_part`` writes the
    max |x| of its blocks of the first n_rows rows to p_it[blk] (int32 view of
    the float bits). Every slot is written every call: no initialised state."""
    amax = _amax_strided(x_it, red, blk, n_rows * fx.Int32(cols), n_part)
    if gpu.thread_idx.x == fx.Int32(0):
        fx.ptr_store(amax.bitcast(fx.Int32), p_it + fx.Int64(blk))


@flyc.jit
def quant_dynamic_tensor_block(
    x_it, q_it, s_it, p_it, red, blk, n_rows, cols, n_part, fp8_max
):
    """Dynamic per-tensor, pass 2: scale = max(partials) * float(1 / fp8_max),
    combined on the float bits as aiter's atomicMaxFloat. It equals aiter's max
    over rows of row_amax * float(1 / fp8_max): the product rounds
    monotonically. Workgroup 0 writes it to s_it[0]; then quantize as the
    static mode."""
    tid = gpu.thread_idx.x
    v = fx.ptr_load(p_it + fx.Int64((tid < n_part).select(tid, fx.Int32(0))))
    amax = _block_max_bits((tid < n_part).select(v, fx.Int32(0)), red)
    scale = amax * fx.Float32(1.0 / fp8_max)
    if (blk == fx.Int32(0)) & (tid == fx.Int32(0)):
        fx.ptr_store(scale, s_it)
    inv = fx.Float32(rocdl.rcp(T.f32, _raw(scale)))
    _quant_tensor_elems(x_it, q_it, blk, n_rows, cols, inv, fp8_max)


@flyc.jit
def quant_dynamic_tensor_single(x_it, q_it, s_it, red, n_rows, cols, fp8_max):
    """Dynamic per-tensor in one workgroup, M x K <= SINGLE_WG_ELEMS: one
    register-resident load, the amax, then the quant from registers."""
    tid = gpu.thread_idx.x
    n = n_rows * fx.Int32(cols)
    vals, at = [], []
    amax = fx.Float32(0.0)
    for b in range_constexpr(SINGLE_WG_ELEMS // (BLOCK * 8)):
        elem = fx.Int32(b * BLOCK * 8) + tid * fx.Int32(8)
        ok = elem < n
        x = _bf16x8_f32(_load8(x_it, ok.select(elem, fx.Int32(0))))
        amax = _absmax([ok.select(v, fx.Float32(0.0)) for v in x], amax)
        vals.append(x)
        at.append((ok, elem))
    amax = _block_max_bits(amax.bitcast(fx.Int32), red)
    scale = amax * fx.Float32(1.0 / fp8_max)
    if tid == fx.Int32(0):
        fx.ptr_store(scale, s_it)
    inv = fx.Float32(rocdl.rcp(T.f32, _raw(scale)))
    for b in range_constexpr(SINGLE_WG_ELEMS // (BLOCK * 8)):
        ok, elem = at[b]
        d0 = _fp8x4(vals[b][0:4], inv, fp8_max)
        d1 = _fp8x4(vals[b][4:8], inv, fp8_max)
        if ok:
            q = elem >> fx.Int32(2)
            fx.ptr_store(d0, q_it + fx.Int64(q))
            fx.ptr_store(d1, q_it + fx.Int64(q + fx.Int32(1)))


@functools.lru_cache(maxsize=64)
def build_tensor_quant(cols, fp8_max, has_local):
    """Dynamic per-tensor pass 2 as its own launch, after a one-shot sort
    whose extra workgroups wrote the partial amaxes."""

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, 8, 16]

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def tensor_quant_kernel(
        x: fx.Tensor,
        q: fx.Tensor,
        s: fx.Tensor,
        p: fx.Tensor,
        ltok: fx.Tensor,
        m_cap: fx.Int32,
        n_part: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(Lds).peek()
        rows = m_cap
        if has_local:
            t = fx.ptr_load(_it(ltok) + fx.Int64(0))
            rows = (t < m_cap).select(t, m_cap)
        quant_dynamic_tensor_block(
            _it(x),
            _it(q),
            _it(s),
            _it(p),
            lds.red.ptr,
            gpu.block_idx.x,
            rows,
            cols,
            n_part,
            fp8_max,
        )

    @flyc.jit
    def launch(
        x: fx.Tensor,
        q: fx.Tensor,
        s: fx.Tensor,
        p: fx.Tensor,
        ltok: fx.Tensor,
        m_cap: fx.Int32,
        n_part: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = None,
    ):
        tensor_quant_kernel(x, q, s, p, ltok, m_cap, n_part).launch(
            grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    return launch


def tensor_partials(m, cols):
    """Partial-amax workgroups of a dynamic per-tensor pass (at most BLOCK)."""
    return max(1, min(BLOCK, -(-(m * cols) // (BLOCK * 8))))


def _e8m0_round_up(amax, max_pos):
    """aiter's MxScaleRoundMode RoundUp: the biased exponent of
    ceil_pow2(amax / max_pos); 0xFF for inf and nan."""
    u = (amax * fx.Float32(1.0 / max_pos)).bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)
    e = (u + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23))
    return (fx.Uint32(e) < fx.Uint32(0xFF)).select(e, fx.Int32(0xFF))


def _bf16x2(d):
    return _raw(fx.Vector.from_elements([d], fx.Int32).bitcast(fx.BFloat16))


def _mx_fp4(dwords, qs):
    """8 bf16 (4 dwords) -> one dword of 8 fp4, x / qs rounded to nearest even."""
    pk = _raw(fx.Int32(0))
    for j, d in enumerate(dwords):
        pk = rocdl.cvt_scalef32_pk_fp4_bf16(T.i32, pk, _bf16x2(d), _raw(qs), j)
    return fx.Int32(pk)


def _mx_fp8(dwords, qs):
    """8 bf16 (4 dwords) -> two dwords of 8 fp8 e4m3fn, x / qs (RNE)."""
    i16x2 = fx.Vector.make_type(2, fx.Int16)
    out = []
    for p in range(2):
        v = _raw(fx.full(2, 0, fx.Int16))
        for h in range(2):
            v = rocdl.cvt_scalef32_pk_fp8_bf16(
                i16x2, v, _bf16x2(dwords[2 * p + h]), _raw(qs), h
            )
        out.append(fx.Vector(v).bitcast(fx.Int32)[0])
    return out


def mx_scale_cols(cols):
    """Bytes per token of the lane-native MX scales: the group count padded to 16."""
    return -(-(cols // 32) // 16) * 16


def mx_offsets(u, cols, fp4, c):
    """Unit ``u`` of an MX pass -> (token, group, first bf16 element, first
    output dword, scale byte). Plain arithmetic, so the kernel (``c`` =
    fx.Int32) and host checks (``c`` = int, or tensors) share it."""
    gp = mx_scale_cols(cols)
    t = u // c(gp)
    g = u - t * c(gp)
    src = t * c(cols) + g * c(32)
    q = t * c(cols // 8) + g * c(4) if fp4 else t * c(cols // 4) + g * c(8)
    s = t * c(gp) + (g & c(3)) * c(gp // 4) + (g >> c(2))
    return t, g, src, q, s


@flyc.jit
def quant_mx_block(x_it, q_it, s_it, blk, n_rows, cols, fp4, index_debug):
    """MX quant, one (token, 32-column group) per thread, from unit blk * 256;
    units run over n_rows x mx_scale_cols(cols), and the padding groups get a
    0 scale. As aiter's per_1x32_mx_quant_hip: amax from kAmaxFloor, NaN
    skipped, RoundUp e8m0 (fp8: capped at 254 = 2^127), values by the hardware scaled conversion. Scales
    lane-native: token t, group g at byte t*Gp + (g%4)*(Gp/4) + g/4. ``q_it``
    iterates the output as int32, ``s_it`` the scales as int8. gfx950 only,
    except with ``index_debug``: each output dword is then t*4096 + g*16 + j
    in place of the converted values (any arch), to check the addressing.
    Stores are branched, not sent to an out-of-range offset: the buffer
    descriptor spans 2^32 bytes, so no byte offset of an int8 store is out of
    range."""
    u = blk * fx.Int32(BLOCK) + gpu.thread_idx.x
    t, g, src, q, s = mx_offsets(u, cols, fp4, fx.Int32)
    live = t < n_rows
    val = live & (g < fx.Int32(cols // 32))
    src = val.select(src, fx.Int32(0))
    dw = []
    for i in range_constexpr(4):
        dw.extend(_load8(x_it, src + fx.Int32(8 * i)))
    amax = _absmax(_bf16x8_f32(dw[0:8]) + _bf16x8_f32(dw[8:16]), fx.Float32(AMAX_FLOOR))
    e = _e8m0_round_up(amax, 6.0 if fp4 else 448.0)
    if const_expr(not fp4):
        e = (e < fx.Int32(254)).select(e, fx.Int32(254))  # aiter's mxfp8_cap_scale
    qs = (e << fx.Int32(23)).bitcast(fx.Float32)
    if const_expr(index_debug):
        tag = t * fx.Int32(4096) + g * fx.Int32(16)
        out = [tag + fx.Int32(j) for j in range_constexpr(4 if fp4 else 8)]
    elif const_expr(fp4):
        out = [_mx_fp4(dw[4 * i : 4 * i + 4], qs) for i in range_constexpr(4)]
    else:
        out = []
        for i in range_constexpr(4):
            out.extend(_mx_fp8(dw[4 * i : 4 * i + 4], qs))
    if val:
        for h in range_constexpr(len(out) // 4):
            fx.ptr_store(
                fx.Vector.from_elements(out[4 * h : 4 * h + 4], fx.Int32),
                q_it + fx.Int64(q + fx.Int32(4 * h)),
            )
    if live:
        fx.ptr_store(fx.Int8(val.select(e, fx.Int32(0))), s_it + fx.Int64(s))


# mode -> (fp4, index_debug)
MX_MODES = {
    "mx_fp4_round_up": (True, False),
    "mx_fp8_round_up": (False, False),
    "mx_fp4_index_debug": (True, True),
    "mx_fp8_index_debug": (False, True),
}


@functools.lru_cache(maxsize=64)
def build_standalone(mode, cols, fp8_max, has_local):
    """A standalone launcher for one mode (tests, and calls without a sort)."""

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, 8, 16]

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def quant_kernel(
        x: fx.Tensor, q: fx.Tensor, s: fx.Tensor, ltok: fx.Tensor, m_cap: fx.Int32
    ):
        bid = gpu.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        rows = m_cap
        if has_local:
            t = fx.ptr_load(_it(ltok) + fx.Int64(0))
            rows = (t < m_cap).select(t, m_cap)
        if mode == "per_1x128_fp8":
            quant_1x128_block(
                _it(x), _it(q), _it(s), bid, rows * fx.Int32(cols // 128), fp8_max
            )
        elif mode in MX_MODES:
            quant_mx_block(_it(x), _it(q), _it(s), bid, rows, cols, *MX_MODES[mode])
        elif mode == "per_tensor_fp8":
            quant_static_block(_it(x), _it(q), _it(s), bid, rows, cols, fp8_max)
        else:
            quant_token_block(
                _it(x), _it(q), _it(s), lds.red.ptr, bid, rows, cols, fp8_max
            )

    @flyc.jit
    def launch(
        x: fx.Tensor,
        q: fx.Tensor,
        s: fx.Tensor,
        ltok: fx.Tensor,
        m_cap: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = None,
    ):
        quant_kernel(x, q, s, ltok, m_cap).launch(
            grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    return launch


def blocks(mode, m, cols):
    """Workgroups a quant pass needs for ``m`` token rows."""
    if mode == "per_1x128_fp8":
        return -(-(m * (cols // 128)) // (BLOCK // GROUP_LANES))
    if mode in MX_MODES:
        return -(-(m * mx_scale_cols(cols)) // BLOCK)
    if mode == "per_tensor_fp8":
        return -(-(m * cols) // (BLOCK * 8))
    return m
