# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bench-and-select autotuner for jdbba v3.

For each (shape, regime) it enumerates candidate instances across ALL registered
variants (shape-filtered by ``variant.supports``), correctness-gates each against
the torch reference, times the survivors with ``triton.testing.do_bench``, and
records the per-shape winner. Results checkpoint to JSON and resume, and the
``winners`` block is drop-in mergeable into ``dispatch_table.json``.

Selection is pure "run benchmark, pick fastest" — no auto-heuristic — exactly the
explicit, testable selection the design calls for. New variants are picked up
automatically (they self-register on import).

Usage::

    HIP_VISIBLE_DEVICES=0 python -m aiter.ops.flydsl.jdbba.autotune \\
        --shapes deployment_all --regime both --out runs/jdbba_v3_tune.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch
import triton.testing

from .config import JdbbaConfig, shape_id
from .reference import torch_reference
from .variants import get_variant, list_variants

# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #
SHAPE_GRIDS = {
    "deployment_all": [
        (120, 256, 256, 16384),
        (120, 512, 512, 16384),
        (1024, 256, 256, 16384),
        (1024, 512, 512, 16384),
    ],
    "deployment_smoke": [(8, 256, 256, 16384)],
}


def _make_inputs(B, K_red, N_out, Mi, *, regime, seed=1234, device="cuda"):
    torch.manual_seed(0)
    if regime == "uniform":
        so = torch.arange(0, (B + 1) * Mi, Mi, dtype=torch.int32, device=device)
    elif regime == "deploy":
        # Match the deployment harness (recsys_harness/common.gen_jagged_offsets):
        # uniformly-random per-group lengths scaled by sparsity. seed/sparsity
        # match bench_jdbba.BenchSpec defaults so tune == validation workload.
        sparsity = 0.95
        g = torch.Generator(device=device).manual_seed(1001)
        lengths = torch.randint(1, Mi + 1, size=(B,), device=device, generator=g)
        lengths = (lengths.float() * sparsity).clamp(min=1.0).to(torch.int64)
        so = torch.zeros(B + 1, dtype=torch.int64, device=device)
        so[1:] = torch.cumsum(lengths, dim=0)
        so = so.to(torch.int32)
    else:
        g = torch.Generator().manual_seed(seed)
        u = torch.rand(B, generator=g)
        t = (Mi * (u**4)).floor().to(torch.int64)
        t[: max(1, B // 5)] = 0
        t[-1] = Mi
        if B > 1:
            t[-2] = int(0.9 * Mi)
        so = torch.zeros(B + 1, dtype=torch.int32)
        for i in range(B):
            so[i + 1] = so[i] + int(t[i])
        so = so.to(device)
    L = int(so[-1].item())
    jg = torch.randn(max(L, 1), K_red, dtype=torch.bfloat16, device=device)
    dn = torch.randn(B, K_red, N_out, dtype=torch.bfloat16, device=device)
    bs = torch.randn(B, N_out, dtype=torch.bfloat16, device=device)
    return jg, dn, bs, so, L


# --------------------------------------------------------------------------- #
# Candidate config grids per variant
# --------------------------------------------------------------------------- #
def _lowlevel_candidates(B, K_red, N_out, Mi):
    tile_ms = (128, 64)
    tile_ns = (128, 256) if N_out % 256 == 0 else (128,)
    # include tile_k=128 (the big-tile D512 winner uses tn=256/tk=128/W1x16).
    tile_ks = (32, 64, 128)
    # (block_m_warps, block_n_warps): include full-block N layouts (the axis the
    # earlier 11.6k sweep missed) for the big-N shapes.
    warp_layouts = ((2, 2), (1, 4), (2, 4), (1, 8), (1, 16))
    b_to_lds_opts = (True, False)
    wpe_opts = (0, 1, 2, 4)
    xcd_opts = (0, 2) if B % 16 == 0 else (0,)
    for tm, tn, tk, (bmw, bnw), b2l, wpe, xcd in itertools.product(
        tile_ms, tile_ns, tile_ks, warp_layouts, b_to_lds_opts, wpe_opts, xcd_opts
    ):
        yield JdbbaConfig(
            variant="lowlevel", tile_m=tm, tile_n=tn, tile_k=tk, stages=2,
            block_m_warps=bmw, block_n_warps=bnw, block_k_warps=1,
            b_to_lds=b2l, waves_per_eu=wpe, xcd_c=xcd,
        )


def _highlevel_candidates(B, K_red, N_out, Mi):
    tile_ns = (None, 128, 256) if N_out % 256 == 0 else (None, 128)
    wpe_opts = (0, 1, 2)
    # Remap-ON (uniform_seqlen=True): sweep the XCD chunk/window — the lever that
    # won D512 on jagged data. 0 => kernel default (K>256 -> chunk 120). Explicit
    # chunks trade L2-reuse vs chiplet load-imbalance; gen's Mi=7680 tune liked
    # 32 (B120D512) / 240 (B1024D512), never swept for jagged N=16384.
    xcd_c_on = (0, 8, 16, 32, 60, 120, 240)
    xcd_w_on = (8, 16)
    for tn, wpe in itertools.product(tile_ns, wpe_opts):
        # remap-ON family (the D512 winner) with the chunk/window sweep
        for xc, xw in itertools.product(xcd_c_on, xcd_w_on):
            yield JdbbaConfig(
                variant="highlevel", tile_n=tn, uniform_seqlen=True,
                waves_per_eu=wpe, xcd_c=xc, xcd_w=(xw if xc else None),
            )
        # remap-OFF family (the small-D / skew winner)
        yield JdbbaConfig(
            variant="highlevel", tile_n=tn, uniform_seqlen=False,
            waves_per_eu=wpe, xcd_c=0,
        )


_CANDIDATE_GENS = {
    "lowlevel": _lowlevel_candidates,
    "highlevel": _highlevel_candidates,
}


def _candidates(B, K_red, N_out, Mi, variants):
    for vname in variants:
        gen = _CANDIDATE_GENS.get(vname)
        if gen is None:
            continue
        v = get_variant(vname)
        for cfg in gen(B, K_red, N_out, Mi):
            if v.supports(cfg, n_groups=B, reduction_k=K_red,
                          output_n=N_out, max_seq_len=Mi):
                yield cfg


# --------------------------------------------------------------------------- #
# Tuning loop
# --------------------------------------------------------------------------- #
def tune_shape(B, K_red, N_out, Mi, *, regime, variants, warmup, rep,
               tol=0.999, verbose=True):
    jg, dn, bs, so, L = _make_inputs(B, K_red, N_out, Mi, regime=regime)
    ref = torch_reference(jg, dn, bs, so)
    uni = regime == "uniform"

    results = []
    best = None
    for cfg in _candidates(B, K_red, N_out, Mi, variants):
        v = get_variant(cfg.variant)
        try:
            out = v.run(jg, dn, bs, so, Mi, cfg, uniform_seqlen=uni, stream=None)
            torch.cuda.synchronize()
            cos = torch.nn.functional.cosine_similarity(
                ref.float().flatten(), out.float().flatten(), dim=0).item()
            if cos <= tol:
                continue
            ms = triton.testing.do_bench(
                lambda: v.run(jg, dn, bs, so, Mi, cfg, uniform_seqlen=uni, stream=None),
                warmup=warmup, rep=rep,
            )
        except AssertionError:
            # Coarse supports() let an invalid warp/tile combo through; the
            # kernel's own asserts reject it. Expected — skip quietly.
            continue
        except Exception as e:  # noqa: BLE001 - record + skip bad instances
            if verbose:
                print(f"    skip {cfg.variant} ({type(e).__name__}: {str(e)[:70]})")
            continue
        rec = {"ms": ms, "cos": cos, "config": cfg.to_dict()}
        results.append(rec)
        if best is None or ms < best["ms"]:
            best = rec
            if verbose:
                print(f"    * {ms:8.4f}ms cos={cos:.4f} {cfg.variant} "
                      f"tn={cfg.tile_n} tk={cfg.tile_k} "
                      f"W{cfg.block_m_warps}x{cfg.block_n_warps} "
                      f"b2l={cfg.b_to_lds} wpe={cfg.waves_per_eu} "
                      f"xcd={cfg.xcd_c} uni={cfg.uniform_seqlen}")
    results.sort(key=lambda r: r["ms"])
    return {"L": L, "n_candidates": len(results), "best": best, "all": results[:10]}


def main(argv=None):
    p = argparse.ArgumentParser(description="jdbba v3 autotuner")
    p.add_argument("--shapes", default="deployment_all",
                   help="grid name in SHAPE_GRIDS, or 'B,D,N,Mi' custom")
    p.add_argument("--regime", default="both",
                   choices=["uniform", "skew", "deploy", "both"])
    p.add_argument("--variants", nargs="+", default=list(list_variants()))
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--rep", type=int, default=40)
    p.add_argument("--out", default="runs/jdbba_v3_tune.json")
    args = p.parse_args(argv)

    if args.shapes in SHAPE_GRIDS:
        shapes = SHAPE_GRIDS[args.shapes]
    else:
        shapes = [tuple(int(x) for x in args.shapes.split(","))]
    regimes = ["uniform", "skew"] if args.regime == "both" else [args.regime]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if out_path.is_file():
        state = json.loads(out_path.read_text())

    print(f"=== jdbba v3 autotune: variants={args.variants} "
          f"shapes={len(shapes)} regimes={regimes} ===")
    t0 = time.time()
    for (B, D, N, Mi) in shapes:
        for regime in regimes:
            sid = shape_id(n_groups=B, reduction_k=D, output_n=N, max_seq_len=Mi)
            key = f"{sid}::{regime}"
            if key in state and state[key].get("best"):
                print(f"[skip cached] {key}")
                continue
            print(f"[tune] {key}")
            res = tune_shape(B, D, N, Mi, regime=regime, variants=args.variants,
                             warmup=args.warmup, rep=args.rep)
            state[key] = res
            out_path.write_text(json.dumps(state, indent=2))
            if res["best"]:
                b = res["best"]
                print(f"  WINNER {key}: {b['ms']:.4f}ms  {b['config']['variant']}  "
                      f"({res['n_candidates']} valid)")
    print(f"=== done in {time.time() - t0:.1f}s -> {out_path} ===")

    # Emit a dispatch-table 'winners' block. Prefer the 'deploy' regime (matches
    # the deployment harness distribution); fall back to 'skew' if absent.
    have_deploy = any(k.endswith("::deploy") and v.get("best")
                      for k, v in state.items())
    pick_regime = "deploy" if have_deploy else "skew"
    winners = {}
    for key, res in state.items():
        sid, _, regime = key.partition("::")
        if regime == pick_regime and res.get("best"):
            cfg = {k: v for k, v in res["best"]["config"].items() if v is not None}
            winners[sid] = cfg
    wpath = out_path.with_suffix(".winners.json")
    wpath.write_text(json.dumps({"winners": winners}, indent=2))
    print(f"=== winners (skew) -> {wpath} ===")


if __name__ == "__main__":
    main()
