# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A path that quantizes the activations coarser than the case asks for (per token in a
per-1x128 blockscale case) rounds more than requested, like a4w4 in an a16w4 case: the bar
keeps it, the row is REVIEW and the least-rounding bar (lr_*) is the best path that doesn't."""

import io
import json
import types

from op_tests.moe_bench import accuracy, bar_worker, schema, workload

ASM_PERTOKEN = (
    "_ZN5aiter59" + "fmoe_stage1_bf16_pertokenFp8_" + "blockscale_g1u1_64x128_2tg_pf3E"
)
ASM_BLOCK = "_ZN5aiter49fmoe_bf16_blockscaleFp8_g1u1_novs_silu_1tg_32x256E"
CK_GEMM1 = (
    "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleExpertWeightA8W8blkscale_v1_"
    "Nswizzle0_Quant4_MulRoutedWeight0_silu_F8_F8_B16"
)
CK_TENSOR = "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleExpertWeight_v1_Nswizzle0_Quant1_F8_F8_B16"


def _case(q_type="per_1x128", family="fp8-block"):
    row = {c: "" for c in schema.COLS}
    row.update(
        case_id="c1", kind="fused", family=family, token="1024", model_dim="7168",
        inter_dim="256", expert="257", topk="9", act_type="Silu", dtype="bf16",
        q_dtype_a="fp8", q_dtype_w="fp8", q_type=q_type, use_g1u1="1",
        doweight_stage1="0", modes="warm",
    )  # fmt: skip
    return schema.Case(row, {})


def test_the_stage_1_activation_quantization_is_read_off_the_kernel_name():
    assert accuracy.stage1_act_quant(ASM_PERTOKEN) == "per_Token"
    assert accuracy.stage1_act_quant(ASM_BLOCK) == "per_1x128"
    assert accuracy.stage1_act_quant(CK_GEMM1) == "per_1x128"  # Quant4
    assert accuracy.stage1_act_quant(CK_TENSOR) == "per_Tensor"  # Quant1
    assert accuracy.stage1_act_quant("flydsl_moe1_afp8_wfp8_bf16_t32x32x256") is None
    assert (
        accuracy.stage1_act_quant("_ZN5aiter38fmoe_bf16_pertokenMXfp4_g1u1_vs_gelu")
        is None
    )
    assert accuracy.stage1_act_quant("") is None


def test_per_token_activations_in_a_blockscale_case_round_more_than_requested():
    S = workload.resolve(_case("per_1x128"))
    why = accuracy.candidate_rounds_more(S, ASM_PERTOKEN)
    assert "per_Token" in why and "per_1x128" in why
    assert (
        accuracy.candidate_rounds_more(S, CK_GEMM1) == ""
    )  # the requested granularity
    assert accuracy.candidate_rounds_more(S, ASM_BLOCK) == ""
    assert accuracy.candidate_rounds_more(S, "flydsl_moe1_x") == ""  # name does not say
    # a per-token case is not rounded more by a per-token path, but by a per-tensor one
    ptpc = workload.resolve(_case("per_Token", "fp8-token"))
    assert accuracy.candidate_rounds_more(ptpc, ASM_PERTOKEN) == ""
    assert accuracy.candidate_rounds_more(ptpc, CK_TENSOR)


def _res(cid, us, kernels, **extra):
    cand = types.SimpleNamespace(
        id=cid,
        describe=lambda: {"source": "fused_moe", "entry": "e", "kernels": kernels},
        kernels=kernels,
        confirmed="",
    )
    return dict(
        {"ok": True, "cand": cand, "by_mode": {"warm": {"median_us": us}}, "acc": None},
        **extra,
    )


def _summary(results):
    fh = io.StringIO()
    S = workload.resolve(_case())
    ident = {"gpu": "MI355X", "sku": None, "gfx": "gfx950"}
    bar_worker._emit_summary(fh, _case(), S, "warm", results, ident, [], "bar")
    return json.loads(fh.getvalue().splitlines()[-1])


def test_a_bar_won_by_a_coarser_path_is_review_and_the_lr_bar_is_the_best_that_is_not():
    results = {
        "dispatch": _res("dispatch", 359.9, ""),
        "fast": _res(
            "fused_moe|asm|ck",
            325.5,
            ASM_PERTOKEN + " + gemm2",
            coarse_quant="stage-1 activations quantized per_Token ...",
        ),
        "ck": _res("fused_moe|ck|ck", 331.0, CK_GEMM1 + " + gemm2"),
        "slow": _res("fused_moe|ck|ck2", 340.0, CK_GEMM1 + " + gemm2b"),
    }
    row = _summary(results)
    assert row["bar_id"] == "fused_moe|asm|ck" and row["bar_us"] == 325.5  # unchanged
    assert row["review"] == "rounds-more" and "per_Token" in row["bar_rounds_more"]
    assert row["lr_id"] == "fused_moe|ck|ck" and row["lr_us"] == 331.0
    assert row["lr_status"] == "ok" and row["lr_kernels"].startswith("moe_ck2stages")


def test_a_bar_won_by_a_path_that_does_not_round_more_has_no_review_or_lr():
    row = _summary({"a": _res("a", 300.0, CK_GEMM1), "b": _res("b", 310.0, CK_GEMM1)})
    assert row["bar_id"] == "a"
    assert "review" not in row and "lr_us" not in row and "bar_rounds_more" not in row


def test_when_every_correct_path_rounds_more_the_lr_bar_says_none():
    row = _summary({"a": _res("a", 300.0, ASM_PERTOKEN, coarse_quant="per token")})
    assert row["review"] == "rounds-more" and row["lr_status"] == "none"
    assert "lr_us" not in row


def test_recompute_lr_finds_review_rows_in_existing_outputs_without_changing_the_bar():
    from op_tests.moe_bench import recompute_lr

    def bar(case, mode, kernels, us):
        return {
            "case_id": case, "mode": mode, "status": "ok", "token": "1024",
            "q_type": "per_1x128", "bar_id": f"fused_moe|{kernels}", "bar_us": str(us),
            "bar_kernels": kernels,
        }  # fmt: skip

    def cand(case, cid, kernels, us, phase="confirm", mode="warm"):
        return {
            "case_id": case, "cand_id": cid, "kernels": kernels, "median_us": str(us),
            "phase": phase, "mode": mode, "status": "ok",
        }  # fmt: skip

    bars = [
        bar("a", "warm", ASM_PERTOKEN + " + gemm2 (block_m=64, ksplit=0)", 325.5),
        bar("b", "warm", CK_GEMM1 + " + gemm2", 300.0),  # not flagged
    ]
    cands = [
        cand("a", "fused_moe|asm", ASM_PERTOKEN + " + gemm2", 325.5),
        cand("a", "fused_moe|ck", CK_GEMM1 + " + gemm2", 346.0),
        cand("a", "fused_moe|ck", CK_GEMM1 + " + gemm2", 340.0, phase="screen"),
        cand("a", "dispatch", "", 360.0),
        cand("b", "fused_moe|ck", CK_GEMM1 + " + gemm2", 300.0),
    ]
    rows = recompute_lr.recompute(bars, cands)
    assert [r["case_id"] for r in rows] == ["a"]
    r = rows[0]
    assert r["bar_us"] == 325.5  # the full bar is as measured
    assert r["lr_id"] == "fused_moe|ck" and r["lr_us"] == 346.0  # confirmed over screen
    assert r["lr_phase"] == "confirm"
