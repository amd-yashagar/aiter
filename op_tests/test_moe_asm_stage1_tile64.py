# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression: ASM per-token fp8 MoE kernels that return wrong or non-finite results (MOEBUG-064).

On gfx942 the tile_n 64 stage-1 kernels (16x64_5tg_pf3, 32x64_4tg_pf2/pf3, 64x64_2tg_pf3, 64x64_3tg_pf2) and, at
larger token counts or other kernels, 128x64, 32x128_3tg_pf2 and 32x256_2tg_pf2 give garbage through
``fused_moe`` (a shipped tuned row for Gelu 4096/384/E128 at M 16 and 32 did, and so does any forced row for Hunyuan3
per-tensor 4096/192/E192 at M >= 4096). On gfx950 the same tile_n 64 kernels, 64x128_2tg_pf3, 32x128_3tg_pf2,
32x256_2tg_pf2 and the 1-stage tkw1_silu_1tg_32x64 kernel do too. Nothing outside the kernel is at fault (zeroed or
poisoned outputs and padding make no difference), so dispatch and the tuner refuse these kernels at the token
counts where they failed, and a tuned row naming one falls back to the default heuristic."""

import csv
import glob
import importlib.util
import os
import sys

import pytest
import torch

import aiter
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import asm_stage1_refusal, fused_moe, fused_topk
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = fm._ASM_BAD_MIN_TOKEN
BAD = TABLE["gfx942"]


def _kname(tile):
    if tile.startswith("1stage:"):
        name = f"fmoe_bf16_pertokenFp8_g1u1_{tile.split(':', 1)[1]}"
    else:
        name = f"fmoe_stage1_bf16_pertokenFp8_g1u1_{tile}"
    return f"_ZN5aiter{len(name)}{name}E"


def _on_gfx942():
    return (
        torch.cuda.is_available()
        and bool(torch.version.hip)
        and get_gfx_runtime() == "gfx942"
    )


needs_gfx942 = pytest.mark.skipif(not _on_gfx942(), reason="Needs a gfx942 GPU")


def _gemm2(tile, version, quant):
    return (
        f"moe_ck2stages_gemm2_{tile}_MulABScaleExpertWeight_{version}_Nswizzle0_Quant{quant}"
        "_MulRoutedWeight1_F8_F8_B16"
    )


S2_PERTOKEN = _gemm2("256x32x64x128_1x4", "v1", 2)
S2_PERTENSOR = {
    16: _gemm2("64x16x64x64_1x1", "v1", 1),
    32: _gemm2("256x32x64x64_1x4", "v1", 1),
    64: _gemm2("256x64x64x64_1x4", "v1", 1),
}
ROW_COLS = [
    "gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk", "act_type", "dtype",
    "q_dtype_a", "q_dtype_w", "q_type", "use_g1u1", "doweight_stage1", "block_m", "ksplit", "us1",
    "kernelName1", "err1", "us2", "kernelName2", "err2", "us", "run_1stage", "xbf16", "flat",
    "tflops", "bw",
]  # fmt: skip


# --- no GPU needed -------------------------------------------------------------------------------


ALL_ENTRIES = [
    (gfx, tile, tok)
    for gfx, d in sorted(TABLE.items())
    for tile, tok in sorted(d.items())
]


@pytest.mark.parametrize("gfx, tile, min_token", ALL_ENTRIES)
def test_refusal_table(gfx, tile, min_token):
    assert asm_stage1_refusal(_kname(tile), min_token, gfx) is not None
    assert asm_stage1_refusal(_kname(tile), 1 << 20, gfx) is not None
    if min_token > 1:
        assert asm_stage1_refusal(_kname(tile), min_token - 1, gfx) is None
    other = "gfx950" if gfx == "gfx942" else "gfx942"
    if tile not in TABLE[other]:
        assert asm_stage1_refusal(_kname(tile), 1 << 20, other) is None
    assert asm_stage1_refusal(_kname(tile), 1 << 20, "gfx90a") is None


def test_refusal_ignores_other_kernels():
    assert asm_stage1_refusal(_kname("128x128_pf3"), 1 << 20, "gfx942") is None
    assert asm_stage1_refusal(_kname("32x256_2tg_pf3"), 1 << 20, "gfx942") is None
    name = "fmoe_stage1_bf16_pertokenFp8_doweight_g1u1_32x64_4tg_pf3"
    assert asm_stage1_refusal(f"_ZN5aiter{len(name)}{name}E", 1 << 20, "gfx942") is None
    assert (
        asm_stage1_refusal("moe_ck2stages_gemm1_256x32x64x128", 1 << 20, "gfx942")
        is None
    )
    assert asm_stage1_refusal(float("nan"), 1 << 20, "gfx942") is None


_GFX = {"256": "gfx950", "80": "gfx942", "228": "gfx942", "304": "gfx942"}
# shipped rows the table still discards at run time: verified clean in the tested runs, but their kernel is refused
# at that token count (gfx942 glm5 ptpc 16x64_5tg_pf3 at M 32 and 256; gfx950 1-stage tkw1 32x64 at M 2).
KEPT_BUT_REFUSED = {
    ("a8w8_blockscale_tuned_fmoe_ds_v3.csv", "2", "5120", "1024", "128", "gfx950"),
    ("glm5_ptpc_tuned_fmoe.csv", "32", "6144", "256", "257", "gfx942"),
    ("glm5_ptpc_tuned_fmoe.csv", "256", "6144", "256", "257", "gfx942"),
}
DROPPED = [  # (file, token, model_dim, inter_dim, expert, kernel tile) rows removed because they failed (MOEBUG-064)
    ("tuned_fmoe.csv", "16", "4096", "384", "128", "32x64_4tg_pf3"),
    ("tuned_fmoe.csv", "32", "4096", "384", "128", "32x64_4tg_pf3"),
    ("tuned_fmoe.csv", "16", "7168", "256", "256", "32x64_4tg_pf3"),
    ("glm47_fp8_tuned_fmoe.csv", "256", "5120", "1536", "40", "64x128_2tg_pf3"),
    ("glm47_fp8_tuned_fmoe.csv", "8", "5120", "1536", "41", "32x128_3tg_pf2"),
    ("glm47_fp8_tuned_fmoe.csv", "16", "5120", "1536", "41", "32x256_2tg_pf2"),
    ("glm47_fp8_tuned_fmoe.csv", "32", "5120", "1536", "41", "32x256_2tg_pf2"),
    (
        "a8w8_blockscale_tuned_fmoe_ds_v3.csv",
        "1",
        "5120",
        "1024",
        "128",
        "1stage:tkw1_silu_1tg_32x64",
    ),
    (
        "a8w8_blockscale_tuned_fmoe_ds_v3.csv",
        "4",
        "5120",
        "1024",
        "128",
        "1stage:tkw1_silu_1tg_32x64",
    ),
    (
        "a8w8_blockscale_tuned_fmoe_ds_v3.csv",
        "8",
        "5120",
        "1024",
        "128",
        "1stage:tkw1_silu_1tg_32x64",
    ),
    (
        "a8w8_blockscale_tuned_fmoe_ds_v3.csv",
        "16",
        "5120",
        "1024",
        "128",
        "1stage:tkw1_silu_1tg_32x64",
    ),
]


def _shipped_rows():
    for path in sorted(
        glob.glob(
            os.path.join(ROOT, "aiter", "configs", "**", "*fmoe*.csv"), recursive=True
        )
    ):
        with open(path) as f:
            for r in csv.DictReader(f):
                if "kernelName1" in r and "cu_num" in r:
                    yield (
                        os.path.basename(path),
                        r,
                        r.get("gfx") or _GFX.get(r["cu_num"]),
                    )


def test_failing_shipped_rows_are_gone():
    present = {
        (
            name,
            r["token"],
            r["model_dim"],
            r["inter_dim"],
            r["expert"],
            r["kernelName1"],
        )
        for name, r, _gfx in _shipped_rows()
    }
    for name, token, d, i, e, tile in DROPPED:
        assert (name, token, d, i, e, _kname(tile)) not in present, (
            name,
            token,
            d,
            i,
            e,
            tile,
        )


def test_shipped_rows_refused_at_run_time_are_the_known_clean_ones():
    refused = {
        (name, r["token"], r["model_dim"], r["inter_dim"], r["expert"], gfx)
        for name, r, gfx in _shipped_rows()
        if asm_stage1_refusal(r["kernelName1"], int(r["token"]), gfx) is not None
    }
    assert refused == KEPT_BUT_REFUSED, sorted(refused ^ KEPT_BUT_REFUSED)


def _tuner():
    path = os.path.join(ROOT, "csrc", "ck_gemm_moe_2stages_codegen", "gemm_moe_tune.py")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("gemm_moe_tune_t64", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gemm_moe_tune_t64"] = mod
    spec.loader.exec_module(mod)
    key = ["gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk", "act_type", "dtype", "q_dtype_a",
           "q_dtype_w", "q_type", "use_g1u1", "doweight_stage1"]  # fmt: skip
    res = ["block_m", "ksplit", "us1", "kernelName1", "err1", "us2", "kernelName2", "err2", "us", "run_1stage",
           "xbf16", "flat", "tflops", "bw"]  # fmt: skip
    argv, sys.argv = sys.argv, ["gemm_moe_tune"]
    try:
        return mod.FmoeTuner("fmoeTuner", key, res + ["nt"], "fmoe tuner")
    finally:
        sys.argv = argv


def _enumerated(tuner, gfx, token, inter_dim, qtype):
    info = (gfx, 304 if gfx == "gfx942" else 256, token, 4096, inter_dim, 128, 8, ActivationType.Gelu,
            dtypes.bf16, dtypes.fp8, dtypes.fp8, qtype, True, False)  # fmt: skip
    return {t[0][2] for t in tuner.gen_2stages_asm1_task(info, [16, 32, 64, 128])}


@pytest.mark.parametrize("gfx", ["gfx942", "gfx950"])
@pytest.mark.parametrize("qtype", [QuantType.per_Token, QuantType.per_Tensor])
def test_tuner_does_not_offer_refused_kernels(gfx, qtype):
    tuner = _tuner()
    for token in (8, 16, 32, 256, 512, 4096, 8192):
        offered = _enumerated(tuner, gfx, token, 384, qtype)
        for tile, min_token in TABLE[gfx].items():
            if token >= min_token and not tile.startswith("1stage:"):
                assert _kname(tile) not in offered, (tile, token)
    # below its threshold a kernel is still offered
    assert _kname("32x64_4tg_pf3") in _enumerated(tuner, gfx, 8, 384, qtype)


# --- gfx942 GPU ----------------------------------------------------------------------------------


@needs_gfx942
@pytest.mark.parametrize("tile, min_token", sorted(BAD.items()))
def test_launcher_refuses_known_bad_kernel(tile, min_token):
    out = torch.empty(min_token, 2, 64, dtype=dtypes.bf16, device="cuda")
    ids = torch.zeros(4, dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError, match="MOEBUG-064"):
        fm.asm_stage1(
            None, None, None, ids, ids, ids, out, 2, 32, _kname(tile), 0,
            ActivationType.Silu, QuantType.per_Token,
        )  # fmt: skip


def _force_row(monkeypatch, tmp_path, row):
    path = tmp_path / "force_fmoe.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_COLS)
        w.writeheader()
        w.writerow({c: row.get(c, "") for c in ROW_COLS})
    monkeypatch.setenv("AITER_CONFIG_FMOE", str(path))
    AITER_CONFIGS.get_config_file.cache_clear()
    fm.cfg_2stages = None
    fm.get_2stage_cfgs.cache_clear()


def _reset():
    AITER_CONFIGS.get_config_file.cache_clear()
    fm.cfg_2stages = None
    fm.get_2stage_cfgs.cache_clear()


def _row(M, D, I, E, topk, act, qtype, k1, bm, k2):
    return {
        "gfx": "gfx942", "cu_num": 304, "token": fm.get_padded_M(M), "model_dim": D, "inter_dim": I,
        "expert": E, "topk": topk, "act_type": str(act), "dtype": str(dtypes.bf16),
        "q_dtype_a": str(dtypes.fp8), "q_dtype_w": str(dtypes.fp8), "q_type": str(qtype), "use_g1u1": 1,
        "doweight_stage1": 0, "block_m": bm, "ksplit": 0, "us1": 0, "kernelName1": k1, "err1": "0%", "us2": 0,
        "kernelName2": k2, "err2": "0%", "us": 0, "run_1stage": 0, "xbf16": 0, "flat": 0, "tflops": 0, "bw": 0,
    }  # fmt: skip


def _inputs(M, D, I, E, topk, qtype, seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed + M)
    x = torch.randn(M, D, dtype=dtypes.bf16, device="cuda", generator=g)
    score = torch.randn(M, E, dtype=dtypes.bf16, device="cuda", generator=g)
    tw, ti = fused_topk(x, score, topk, True)
    w1 = torch.randn(E, 2 * I, D, dtype=dtypes.bf16, device="cuda", generator=g)
    w2 = torch.randn(E, D, I, dtype=dtypes.bf16, device="cuda", generator=g)
    if qtype == QuantType.per_Tensor:
        w1q, w1s = aiter.pertoken_quant(w1.view(E, -1), quant_dtype=dtypes.fp8)
        w2q, w2s = aiter.pertoken_quant(w2.view(E, -1), quant_dtype=dtypes.fp8)
        w1q, w2q = w1q.view(w1.shape), w2q.view(w2.shape)
    else:
        w1q, w1s = aiter.pertoken_quant(w1, quant_dtype=dtypes.fp8)
        w2q, w2s = aiter.pertoken_quant(w2, quant_dtype=dtypes.fp8)
    return x, tw, ti, w1q, w1s, w2q, w2s


def _routed_kernels(x, tw, ti, w1q, w1s, w2q, w2s, qtype, act):
    cb = []
    fm.kernel_bench_callable = cb
    try:
        out = fused_moe(
            x, shuffle_weight(w1q, layout=(16, 16)), shuffle_weight(w2q, layout=(16, 16)), tw, ti,
            w1_scale=w1s, w2_scale=w2s, quant_type=qtype, activation=act, doweight_stage1=False,
        )  # fmt: skip
    finally:
        fm.kernel_bench_callable = None
    torch.cuda.synchronize()
    stages = dict(cb)
    return (
        out,
        stages["stage1"].keywords.get("kernelName"),
        stages["stage2"].keywords.get("kernelName"),
    )


def _per_token_reference(x, tw, ti, w1q, w1s, w2q, w2s, I, E, topk, act):
    M, D = x.shape
    a1q, a1s = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
    xf = a1q.float() * a1s.float()
    o1 = torch.zeros(M, topk, I, dtype=torch.float32, device="cuda")
    torch_act = aiter.get_torch_act(act)
    for e in range(E):
        r, j = (ti == e).nonzero(as_tuple=True)
        if r.numel():
            h = xf[r] @ (w1q[e].float() * w1s[e].float()).T
            o1[r, j] = torch_act(h[:, :I]) * h[:, I:]
    a2q, a2s = aiter.pertoken_quant(
        o1.to(dtypes.bf16).view(-1, I), quant_dtype=dtypes.fp8
    )
    a2f = (a2q.float() * a2s.float()).view(M, topk, I)
    out = torch.zeros(M, D, dtype=torch.float32, device="cuda")
    for e in range(E):
        r, j = (ti == e).nonzero(as_tuple=True)
        if r.numel():
            out.index_add_(
                0,
                r,
                (a2f[r, j] @ (w2q[e].float() * w2s[e].float()).T)
                * tw[r, j].float().unsqueeze(1),
            )
    return out.to(dtypes.bf16)


@needs_gfx942
@pytest.mark.parametrize("M", [16, 32])
def test_shipped_row_shape_r24(monkeypatch, tmp_path, M):
    """Gelu, per-token fp8, D4096 I384 E128 top8, the shipped 32x64_4tg_pf3E rows at M 16 and 32 (MOEBUG-064 R24 / R32):
    the forced row must not run the ASM kernel, and the result must be right."""
    D, I, E, topk = 4096, 384, 128, 8
    k1 = _kname("32x64_4tg_pf3")
    _force_row(
        monkeypatch,
        tmp_path,
        _row(
            M,
            D,
            I,
            E,
            topk,
            ActivationType.Gelu,
            QuantType.per_Token,
            k1,
            32,
            S2_PERTOKEN,
        ),
    )
    try:
        data = _inputs(M, D, I, E, topk, QuantType.per_Token)
        for _ in range(3):
            out, got1, _got2 = _routed_kernels(
                *data, QuantType.per_Token, ActivationType.Gelu
            )
            assert got1 != k1, "the refused ASM stage-1 kernel was dispatched"
            ref = _per_token_reference(*data, I, E, topk, ActivationType.Gelu)
            assert torch.isfinite(out.float()).all()
            err = float(
                torch.linalg.vector_norm(out.float() - ref.float())
                / torch.linalg.vector_norm(ref.float())
            )
            assert err < 0.05, err
    finally:
        _reset()


@needs_gfx942
@pytest.mark.parametrize(
    "tile, M, bm",
    [("16x64_5tg_pf3", 4096, 16), ("32x64_4tg_pf2", 4096, 32), ("32x64_4tg_pf3", 4096, 32),
     ("64x64_2tg_pf3", 4096, 64), ("64x64_3tg_pf2", 8192, 64)],
)  # fmt: skip
def test_hunyuan3_forced_pairs_fall_back(monkeypatch, tmp_path, tile, M, bm):
    """Hunyuan3 per-tensor fp8, D4096 I192 E192 top8: a forced row naming one of the failing ASM stage-1 kernels
    must resolve to the default heuristic instead (accuracy of that heuristic at I192 is covered by
    test_moe_ck_fp8_small_inter_dim.py, the CK BK1 fix)."""
    D, I, E, topk = 4096, 192, 192, 8
    k1 = _kname(tile)
    _force_row(
        monkeypatch,
        tmp_path,
        _row(
            M,
            D,
            I,
            E,
            topk,
            ActivationType.Silu,
            QuantType.per_Tensor,
            k1,
            bm,
            S2_PERTENSOR[bm],
        ),
    )
    try:
        data = _inputs(M, D, I, E, topk, QuantType.per_Tensor)
        _out, got1, _got2 = _routed_kernels(
            *data, QuantType.per_Tensor, ActivationType.Silu
        )
        assert got1 != k1, "the refused ASM stage-1 kernel was dispatched"
    finally:
        _reset()


@needs_gfx942
def test_small_token_row_still_uses_asm(monkeypatch, tmp_path):
    """Below the refusal threshold the kernel is still used (32x64_4tg_pf3E passes at M <= 8) and correct."""
    D, I, E, topk, M = 4096, 384, 128, 8, 8
    k1 = _kname("32x64_4tg_pf3")
    _force_row(
        monkeypatch,
        tmp_path,
        _row(
            M,
            D,
            I,
            E,
            topk,
            ActivationType.Gelu,
            QuantType.per_Token,
            k1,
            32,
            S2_PERTOKEN,
        ),
    )
    try:
        data = _inputs(M, D, I, E, topk, QuantType.per_Token)
        out, got1, _got2 = _routed_kernels(
            *data, QuantType.per_Token, ActivationType.Gelu
        )
        assert got1 == k1
        ref = _per_token_reference(*data, I, E, topk, ActivationType.Gelu)
        err = float(
            torch.linalg.vector_norm(out.float() - ref.float())
            / torch.linalg.vector_norm(ref.float())
        )
        assert err < 0.05, err
    finally:
        _reset()
