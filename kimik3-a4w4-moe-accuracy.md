# Kimi-K3 a4w4 MoE accuracy regression in aiter: diagnosis handover

Status: diagnosis only. No code, CSV change, branch, or PR was produced. The local CSV edit was reverted.
gfx950-only kernels; node k08-325 is gfx942 and could not run them.

## Symptom

Kimi-K3 serving on gfx950 lost accuracy after updating aiter to include
[fe4814c](https://github.com/ROCm/aiter/commit/fe4814c64b1f7bfe62742659eaf641b13d4f348d)
([PR #4526](https://github.com/ROCm/aiter/pull/4526)), which retuned
`aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv`.
Logs (in `Mehdi/` on the node, not in git): old aiter run (fine), new aiter docker run (bad),
and a new aiter run with modifications that mostly fixed it. The low-score runs used a 4k limit of 50.

MoE shape: gfx950, cu 256, model_dim 3584, inter_dim 384, expert 896, topk 16,
SiTUv2, a4w4 per_1x32.

## Finding

The regressing component is the decode buckets (tokens 1-64), which now dispatch to the
MXMOE BM16 inline-sort low-M path introduced by
[PR #5300 / 36a076b](https://github.com/ROCm/aiter/commit/36a076b871fee3db3e413291b2fbe14ea7863d5a).
fe4814c only extended that path: it renamed kernels, retuned tokens 1/2 (hpf, BN128), and added 8192.
Token 8192 is not the cause.

Suspect kernels (decode rows):
- GEMM1 `flydsl_mxmoe_g1_a4w4_16x{128,256}x256_f16in[_hpf]_nt_situv2[_xcd4]` does inline activation quant, gate/up, SiTUv2, FP4 requant, and native BM16 scale layout in one kernel.
- GEMM2 `flydsl_moe2_layout_afp4_wfp4_bf16_t16x{128,256}x128_atomic[_nt]_sbm16`.
- Fused sort and zero `module_moe_mxfp4_aux`.

The old good path was `flydsl_moe1_afp4_wfp4_bf16_t32...` (BM32) with `..._sbm32` GEMM2.
Which of the three new pieces is numerically wrong is not proven yet; that needs gfx950.
Upstream main (781bf751b, 2026-09-23) still ships these MXMOE rows.

Why CI missed it: `op_tests/test_moe_2stage.py` fails only when `err != 0 and logits_diff > 0.01`,
and the MXMOE err in the CSV is end-to-end while moe1 err is per stage, so they are not comparable.

## Fix and next steps

1. Mitigation: restore the decode rows (tokens 1-64; optionally 8192) to the pre-#5300 rows from
   [a43fe2589](https://github.com/ROCm/aiter/commit/a43fe2589). Make sure the deployed config is updated:
   the dist-packages copy and the merged `/tmp/aiter_configs/tuned_fmoe.csv` are regenerated.
2. Bisect on gfx950:
   `python3 op_tests/test_moe_2stage.py -q 4 -a situv2 --no-legacy --csv-filter flydsl_mxmoe_g1_a4w4_16x flydsl_moe1_afp4_wfp4_bf16_t32`
   and compare GEMM1 alone (FP4 requant/scale layout), then GEMM2 sbm16, then the aux sort.
3. Tighten the CI gate so a nonzero logits diff on the a4w4 SiTUv2 decode buckets fails.
4. File upstream against #5300. Any accuracy gap left after the revert is separate
   (vLLM upgrade, other changes, or run noise).
