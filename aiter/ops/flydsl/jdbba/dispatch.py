# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-shape, arch-keyed dispatch + public entry for jdbba v3.

Resolution order for a shape: explicit ``config`` arg > JSON table exact match >
D-bucketed heuristic (which also picks the variant). The table is
``arch-keyed-v1``: ``{schema, by_arch: {<arch>: {winners, fallback}}}`` so gfx942
and gfx950 keep separate winners (gfx950 has the 16x16x32 atom; gfx942 does not).

A table entry is a config dict that includes a ``variant`` key, so the dispatcher
selects BOTH the implementation type and its knobs per shape. Override the table
file via ``FLYDSL_JDBBA_DISPATCH_JSON`` and the detected arch via
``FLYDSL_JDBBA_ARCH``.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Optional

import torch

from .config import JdbbaConfig, d_bucket, shape_id
from .variants import get_variant


@functools.lru_cache(maxsize=1)
def _detect_arch() -> Optional[str]:
    env = os.environ.get("FLYDSL_JDBBA_ARCH")
    if env:
        return env
    try:
        from flydsl.runtime.device import get_rocm_arch

        return get_rocm_arch()
    except Exception:
        return None


def _table_path() -> Path:
    env = os.environ.get("FLYDSL_JDBBA_DISPATCH_JSON")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "dispatch_table.json"


@functools.lru_cache(maxsize=4)
def _load_table(arch: Optional[str]) -> dict:
    path = _table_path()
    if not path.is_file():
        return {"winners": {}, "fallback": {}}
    data = json.loads(path.read_text())
    by_arch = data.get("by_arch")
    section = {}
    if isinstance(by_arch, dict):
        if arch and arch in by_arch:
            section = by_arch[arch]
        elif len(by_arch) == 1:
            section = next(iter(by_arch.values()))
    else:
        section = data  # flat / override schema
    return {
        "winners": dict(section.get("winners") or {}),
        "fallback": dict(section.get("fallback") or {}),
    }


def _pick_tile_n(output_n: int) -> int:
    for cand in (256, 128, 64):
        if output_n % cand == 0:
            return cand
    raise ValueError(f"output dim N={output_n} must be divisible by 64/128/256")


def _pick_tile_k(reduction_k: int) -> int:
    if reduction_k % 64 == 0 and reduction_k // 64 >= 2:
        return 64
    if reduction_k % 32 == 0:
        return 32
    raise ValueError(f"reduction dim K={reduction_k} must be divisible by 32/64")


def _lds_fits(*, tile_m, tile_n, tile_k, stages, b_to_lds) -> bool:
    as_b = stages * tile_m * tile_k * 2
    bs_b = stages * tile_n * tile_k * 2 if b_to_lds else 0
    c_b = tile_m * tile_n * 2
    return max(as_b + bs_b, c_b) <= 65536


# Strong lowlevel default for shapes absent from the tuned table (proven across
# the broad catalog): 128x128x32 with the B operand staged through LDS; small
# (D,K) <= 256 prefer 64x64. Degrades tile/LDS to stay valid + fit LDS.
def _heuristic(*, reduction_k: int, output_n: int, n_groups: int, table: dict) -> JdbbaConfig:
    """Robust per-shape fallback (lowlevel) for shapes not in ``winners``.

    A JSON ``fallback`` (global / by_d_bucket) may override the base config; else
    a 128x128x32 b_to_lds default is shrunk to divide the dims and fit LDS.
    """
    fb = table.get("fallback") or {}
    base = dict(fb.get("global") or {})
    by_bucket = fb.get("by_d_bucket") or {}
    if isinstance(by_bucket.get(d_bucket(reduction_k)), dict):
        b = by_bucket[d_bucket(reduction_k)]
        base.update(b.get("config") or b)

    stages = int(base.get("stages", 2))
    bmw = int(base.get("block_m_warps", 2))
    bnw = int(base.get("block_n_warps", 2))

    tile_n = 0
    for cand in (int(base.get("tile_n", 128)), 128, 64):
        if output_n % cand == 0 and cand % (bnw * 16) == 0:
            tile_n = cand
            break
    if tile_n == 0:
        bnw, tile_n = 1, _pick_tile_n(output_n)

    tile_k = 0
    for cand in (int(base.get("tile_k", 32)), 32, 64):
        if reduction_k % cand == 0 and reduction_k // cand >= stages:
            tile_k = cand
            break
    if tile_k == 0:
        tile_k = _pick_tile_k(reduction_k)

    tile_m = int(base.get("tile_m", 128))
    if reduction_k <= 256 and output_n <= 256:
        tile_m = min(tile_m, 64)
    if tile_m % (bmw * 16) != 0:
        tile_m = bmw * 16

    b_to_lds = bool(base.get("b_to_lds", True))
    if not _lds_fits(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, stages=stages, b_to_lds=b_to_lds):
        b_to_lds = False
        while tile_m > bmw * 16 and not _lds_fits(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, stages=stages, b_to_lds=b_to_lds
        ):
            tile_m //= 2

    return JdbbaConfig(
        variant=str(base.get("variant", "lowlevel")),
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, stages=stages,
        block_m_warps=bmw, block_n_warps=bnw, block_k_warps=1,
        b_to_lds=b_to_lds, waves_per_eu=int(base.get("waves_per_eu", 0)),
        xcd_c=int(base.get("xcd_c", 0)),
    )


def resolve_config(
    *,
    n_groups: int,
    reduction_k: int,
    output_n: int,
    max_seq_len: int,
    config: Optional[JdbbaConfig] = None,
    force_variant: Optional[str] = None,
) -> JdbbaConfig:
    """Per-shape config: explicit ``config`` > table exact-match > heuristic.

    ``force_variant`` (e.g. ``"lowlevel"``) accepts the table winner only if it
    is that variant, else falls back to the heuristic — used by paths that can
    only run a specific variant (elementwise bias, the d_jagged backward).
    """
    if config is not None:
        return config
    arch = _detect_arch()
    table = _load_table(arch)
    sid = shape_id(
        n_groups=n_groups, reduction_k=reduction_k,
        output_n=output_n, max_seq_len=max_seq_len,
    )
    entry = table["winners"].get(sid)
    if entry is not None:
        cfg = JdbbaConfig.from_dict(entry)
        if (force_variant is None or cfg.variant == force_variant) and get_variant(
            cfg.variant
        ).supports(
            cfg, n_groups=n_groups, reduction_k=reduction_k,
            output_n=output_n, max_seq_len=max_seq_len,
        ):
            return cfg
    return _heuristic(
        reduction_k=reduction_k, output_n=output_n, n_groups=n_groups, table=table
    )


def jagged_dense_bmm(
    jagged,
    dense,
    bias,
    seq_offsets,
    max_seq_len: int,
    *,
    config: Optional[JdbbaConfig] = None,
    variant: Optional[str] = None,
    uniform_seqlen: bool = True,
    stream=None,
):
    """Public v3 entry. Resolves (or accepts) a per-shape variant+config and runs it.

    Tensors follow the genrec convention: jagged ``(L, K_red)``, dense
    ``(B, K_red, N_out)``, bias ``(B, N_out)``, seq_offsets ``(B+1,)``. Returns
    ``(L, N_out)``. Pass ``config`` to force a full instance, or ``variant`` to
    force a variant while resolving its config from the table/heuristic.
    """
    if jagged.dtype != torch.bfloat16:
        raise ValueError("jdbba v3 currently supports bf16 only")
    B, reduction_k, output_n = dense.shape

    cfg = resolve_config(
        n_groups=B, reduction_k=reduction_k, output_n=output_n,
        max_seq_len=max_seq_len, config=config,
    )
    if variant is not None and config is None:
        cfg = cfg.merged(variant=variant)

    v = get_variant(cfg.variant)
    return v.run(
        jagged, dense, bias, seq_offsets, max_seq_len, cfg,
        uniform_seqlen=uniform_seqlen, stream=stream,
    )
