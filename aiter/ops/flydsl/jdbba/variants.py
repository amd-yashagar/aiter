# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Variant registry for jagged_dense_bmm (jdbba) v3.

A *variant* is one implementation "type" of the op (a distinct kernel + launch
strategy). Variants are registered by name and share a common tensor contract,
so the dispatcher / autotuner can select among them per shape without any
knowledge of their internals. Adding a new specialized kernel (split-K, fused,
persistent, ...) is a single ``@register`` block — no changes elsewhere.

Provenance: the ``lowlevel`` kernel is ``kernels/jagged_dense_bmm.py``; the
``highlevel`` kernel (``kernels/jagged_dense_bmm_gen.py``) originates from
anguyenh's ``anguyenh/flydsl-jdbba`` branch (Apache-2.0) and is integrated here
as a joint contribution.

Common tensor contract (genrec convention; every variant accepts these):
    jagged       (L, K_red)            bf16   packed rows of all groups
    dense        (B, K_red, N_out)     bf16   per-group weight (D, K) in bench naming
    bias         (B, N_out)            bf16   per-group broadcast bias
    seq_offsets  (B + 1,)              int    prefix-sum row offsets
    max_seq_len  int                          per-group row envelope (Mi)
  returns:
    out          (L, N_out)            bf16

Each variant exposes:
    name           : str
    relevant_keys  : tuple[str, ...]   config knobs it actually consumes
    supports(cfg, *, n_groups, reduction_k, output_n, max_seq_len) -> bool
    run(jagged, dense, bias, seq_offsets, max_seq_len, cfg, *, uniform_seqlen,
        stream) -> out
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

_REGISTRY: Dict[str, "Variant"] = {}


class Variant:
    def __init__(self, name, run, supports, relevant_keys):
        self.name = name
        self.run = run
        self.supports = supports
        self.relevant_keys = tuple(relevant_keys)


def register(name, *, supports, relevant_keys):
    def deco(run_fn: Callable) -> Callable:
        _REGISTRY[name] = Variant(name, run_fn, supports, relevant_keys)
        return run_fn

    return deco


def get_variant(name: str) -> Variant:
    if name not in _REGISTRY:
        raise KeyError(f"unknown jdbba variant {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_variants() -> tuple:
    return tuple(sorted(_REGISTRY))


# --------------------------------------------------------------------------- #
# Variant: lowlevel  (the feature-rich hand-written MLIR kernel)
#   kernels/jagged_dense_bmm.py — split-K/slice-K capable, B-to-LDS option,
#   manual instruction scheduling, configurable warp layout. Wins large-B /
#   small-D shapes (e.g. B1024D256) via the W2x4 + b_to_lds + tile_k=32 config.
# --------------------------------------------------------------------------- #

_LOWLEVEL_KEYS = (
    "tile_m", "tile_n", "tile_k", "stages",
    "block_m_warps", "block_n_warps", "block_k_warps",
    "b_to_lds", "waves_per_eu", "xcd_c",
)


def _lowlevel_supports(cfg, *, n_groups, reduction_k, output_n, max_seq_len):
    tn = cfg.tile_n or 128
    tk = cfg.tile_k or (32 if reduction_k <= 256 else 32)
    if output_n % tn != 0 or reduction_k % tk != 0:
        return False
    if reduction_k // tk < cfg.stages:
        return False
    if tn % (cfg.block_n_warps * 16) != 0:
        return False
    tm = cfg.tile_m or 128
    if tm % (cfg.block_m_warps * 16) != 0:
        return False
    # LDS budget (gfx942 64KB): A double-buffer (+ optional B) vs C tile.
    as_b = cfg.stages * tm * tk * 2
    bs_b = cfg.stages * tn * tk * 2 if cfg.b_to_lds else 0
    c_b = tm * tn * 2
    return max(as_b + bs_b, c_b) <= 65536


def _lowlevel_run(jagged, dense, bias, seq_offsets, max_seq_len, cfg, *,
                  uniform_seqlen, stream):
    from .launch import run_lowlevel

    B, k_red, n_out = dense.shape
    L = jagged.shape[0]
    if not jagged.is_contiguous():
        jagged = jagged.contiguous()
    out = torch.empty((L, n_out), dtype=jagged.dtype, device=jagged.device)
    return run_lowlevel(
        out=out, jagged=jagged, dense=dense, bias=bias, seq_offsets=seq_offsets,
        max_seq_len=max_seq_len, cfg=cfg, has_bias=True, elementwise=False,
        transpose_dense=True, stream=stream,
    )


register("lowlevel", supports=_lowlevel_supports, relevant_keys=_LOWLEVEL_KEYS)(
    _lowlevel_run
)


# --------------------------------------------------------------------------- #
# Variant: highlevel  (the clean CuTe-style layout-API kernel)
#   kernels/jagged_dense_bmm_gen.py — TiledMma/TiledCopy, B register prefetch
#   (b_stages), LDS C-shuffle epilogue, 2-phase XCD remap. Wins D=512 and small-B
#   shapes. Simpler/cleaner; the preferred home for future high-level work.
# --------------------------------------------------------------------------- #

_HIGHLEVEL_KEYS = (
    "tile_m", "tile_n", "xcd_c", "xcd_w",
    "use_mfma_k32", "waves_per_eu", "uniform_seqlen",
)


def _highlevel_supports(cfg, *, n_groups, reduction_k, output_n, max_seq_len):
    tn = cfg.tile_n or 128
    if output_n % tn != 0:
        return False
    # gen derives block_k internally (128 if K<=256 on gfx950 else 64/32); just
    # require K leaves >=2 K-tiles for the double-buffer at the smallest block_k.
    return reduction_k >= 64 and reduction_k % 32 == 0


def _highlevel_run(jagged, dense, bias, seq_offsets, max_seq_len, cfg, *,
                   uniform_seqlen, stream):
    import flydsl.compiler as flyc

    from .launch import dense_b_panel
    from ..kernels.jagged_dense_bmm_gen import BLOCK_M as GEN_BLOCK_M
    from ..kernels.jagged_dense_bmm_gen import jagged_dense_bmm as gen_kernel

    B, k_red, n_out = dense.shape
    L = jagged.shape[0]
    # config may pin the regime (a tuned knob: forcing remap-on can win even on
    # jagged data); else use the caller's runtime regime hint.
    uni = cfg.uniform_seqlen if cfg.uniform_seqlen is not None else uniform_seqlen

    # Tall (B*N, K) dense is the kernel's operand layout; cache the transpose per
    # live dense tensor (data_ptr + weakref) so static weights transpose once
    # instead of paying a multi-hundred-MB copy on every call.
    dense_tall = dense_b_panel(dense, batch_size=B, n_dim=n_out, k_dim=k_red,
                               transpose_dense=True)
    bias_flat = bias.reshape(B * n_out).contiguous()
    # empty (not zeros): the kernel writes every row in [0, L); only the padding
    # tail [L, L+BLOCK_M) is undefined and is sliced off below. zeros here would
    # memset a multi-GB buffer on every call (a benchmarking + runtime artifact).
    out = torch.empty(L + GEN_BLOCK_M, n_out, dtype=jagged.dtype, device=jagged.device)
    if seq_offsets.dtype != torch.int32 or not seq_offsets.is_contiguous():
        seq_offsets = seq_offsets.to(torch.int32).contiguous()

    tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    kw = {}
    if stream is not None:
        kw["stream"] = stream
    gen_kernel(
        tC, tA, dense_tall, bias_flat, seq_offsets, B, max_seq_len,
        xcd_c=(cfg.xcd_c or None),
        xcd_w=(cfg.xcd_w or None),
        use_mfma_k32=cfg.use_mfma_k32,
        uniform_seqlen=uni,
        block_m=cfg.tile_m,
        block_n=cfg.tile_n,
        waves_per_eu=int(cfg.waves_per_eu or 0),
        **kw,
    )
    return out[:L]


register("highlevel", supports=_highlevel_supports, relevant_keys=_HIGHLEVEL_KEYS)(
    _highlevel_run
)
