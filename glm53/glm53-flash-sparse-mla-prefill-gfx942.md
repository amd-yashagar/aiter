# GLM-5.3-Flash sparse-MLA prefill, gfx942

**Handover (2026-10-29).** Branch `ws/glm53-sparse-mla` is on `origin` at `778ab353d` and matches the local tip. No pull request: GitHub has no PR whose head is `ws/glm53-sparse-mla`, and that SHA is not mentioned on a PR. The commit is the first FlyDSL kernel. This chat’s rewrite is uncommitted in `aiter/ops/flydsl/kernels/sparse_mla_qblock.py` only (host, dispatch, and tests were not edited). Do not commit or open a PR unless asked.

That dirty file is the kernel to keep: one-deep coalesced gather, DPP softmax, P in registers, index address clamped before the load, `waves_per_eu=2`, `MFMA_CHUNK=8` for one-query and `2` for q-block. Last medians on this MI325, `T=16384 topk=2048 pool=131072`, kernel time only: random one-query **7229 µs** (152 TFLOPS), shared q-block **3898 µs** (282 TFLOPS). The table below is the earlier committed-era bench. Two-deep prefetch was tried and reverted: numerics passed, but LLVM drained the second tile with `vmcnt(0)` before the LDS store (205 → 249 VGPRs, same overlap). ASM in `poc_kl` is paged decode (D=576), not this gather. Next lever for the fallback (one-query) is inline-asm gathers so a partial `vmcnt` cannot be widened. Official pytest has not been run.

DSA prefill on MI325X is ~121 ms/chunk (11 layers × ~11 ms). Reported H200 is 75 ms/chunk. Fix the prefill kernel; do not treat decode as the answer.

**Model** (`zai-org/GLM-5.3-Flash`): 11 DSA + 34 KDA. Absorbed NoPE MLA, `kv_lora_rank=512`, `qk_rope_head_dim=0`, 64 heads, `index_topk=2048`. TP4 → local `H=16`.

**What ran** (trace `GLM5.3Flash/dp0_pp0_tp0_dcp0_ep0_rank0.1790148934656179187.pt.trace.json.gz`, gfx942 304 CU / 64 KiB LDS):

`_pa_decode_sparse_BLOCK_M_16_BLOCK_K_32_HEAD_SIZE_512_NUM_SPLITS_1_MAIN_FMT_BF16_ROPE_SEPARATE_0`

Gluon kernel, Python under `_gluon_kernels/gfx950/` (that suffix is not in the HIP symbol). Enabled on gfx942 by `gfx942-sparse-mla` / `ced68ed2f` on #4919: arch list, `BLOCK_K=32` (64 KiB LDS), `num_warps` uncoupled from `BLOCK_K`, async off. Grid `(C, NUM_SPLITS, cdiv(H, BLOCK_M))` — `BLOCK_M` tiles heads, one query per CTA.

2123 calls, 2112 full-chunk ≥9 ms: mean 10.27 / p50 **10.36** / p10 9.90 / p90 10.50 / max 10.72 ms. Decode is a different `.kd` (`NUM_SPLITS=8`, 286 × 30 µs). Indexer `fp8_mqa_logits_H32_D128_*_flydsl` ~1.0 ms is separate.

A different image (`vllm-rocm-nightly:385dce36-glm53flash-tuned`, gluon not enabled) ran `_sparse_attn_prefill_ragged_kernel` at 11.0 ms — same 121 ms/chunk accounting, different binary. `main` still asserts gfx950.

**Contract** (per GPU, TP4, bf16 pool, CSR topk=2048): `q,out [16384,16,512]`, `kv [pages,512]`, optional sink `[16]` fp32. MQA, K≡V (one load per selected row). Ref: fp32 gather+softmax; keep `atol=rtol=1e-2`.

**Roofline** (MI325X 6.0 TB/s, 1307 TFLOPS bf16): `4·H·D·T·topk` = 1.10 TFLOP, 34.4 GB if every query gathers 2048 rows → 32 FLOP/byte, memory-bound, **5.7 ms/layer floor with no reuse** (63 ms/chunk). Trace ~10.4 ms is 3.3 TB/s (55% HBM). H200 75 ms/chunk = 6.8 ms/layer; moving 34.4 GB at that time would exceed H200's 4.8 TB/s, so H200 is winning on fewer unique bytes. The 378 GB / 134 MB diagnosis does not match this specialization (one 512-wide load, `BLOCK_M=16` covers H=16). Tile tweaks on one-query-per-CTA cannot beat 5.7 ms; split-K is the decode lever.

**Microbench** this MI325X, `T=16384 H=16 D=512 topk=2048`, pool 131072. `import triton` before `torch`. max|delta| ≈ 4e-3. Matched harness times pack+kernel when the candidate packs. Fresh after rebasing to `f4be7f1c5`:

| kernel | shared | random |
|---|---:|---:|
| serving Gluon (original trace, real top-k) | **10.36 p50** | |
| gfx942 Gluon enablement, matched synthetic | 9.36 | 9.98 |
| Triton `one_q` | 18.33 | 18.36 |
| Triton `qblock` (pack e2e) | 4.63 | 14.74 |
| FlyDSL `one_q` (MFMA, two waves) | **8.54** | **9.12** |
| FlyDSL `qblock` (MFMA, BQ4 exact reuse) | **4.48** | 12.37 pack+fallback |

The production-style `pa_decode_sparse` entry with explicit `uniform_topk` and reuse hints measures **9.07–9.08 ms normal** and **4.55 ms exact reuse**. No-reuse random reaches 3.82 TB/s, about 64% of the 6.0 TB/s logical HBM roof; it is a reference-beating result, not the hardware floor. Q-block full-gather bandwidth accounting exceeds HBM because each BQ4 workgroup stages one shared KV list for four queries.

**Integration status:** the exact gfx942 FlyDSL path is wired in AITER's `pa_decode_sparse`, with current-main generic Triton fallback outside `bf16/H16/D512/uniform/no-invalid`. Current vLLM at pinned `c961121` calls `mla_decode_fwd`, not `pa_decode_sparse`; production integration is therefore still missing. Its real ragged/invalid prefix must be measured before enabling the fast-valid specialization. Active AITER PR #5721 enables the Gluon `sparse_mla_fwd` path on gfx942 but does not add this FlyDSL path or Q-block reuse.

**Work:** Stay absorbed. Triton Q-block: `aiter/ops/triton/attention/sparse_mla_qblock.py`. FlyDSL: `aiter/ops/flydsl/kernels/sparse_mla_qblock.py` + `sparse_mla_qblock_kernels.py`. Matched harness `op_tests/test_sparse_mla_qblock.py`; pytest `op_tests/triton_tests/attention/test_sparse_mla_qblock.py`. The exact-reuse test includes an adversarial equal-bookends/different-middle case. Local ticket — do not ship in the PR. Out of scope: KDA/GEMM/MoE, decode, gfx950.

```
PYTHONPATH=/workspaces/quant-k08-325-WS/workstreams/aiter python3 op_tests/test_sparse_mla_qblock.py
PYTHONPATH=/workspaces/quant-k08-325-WS/workstreams/aiter python3 -m pytest op_tests/triton_tests/attention/test_sparse_mla_qblock.py -q
```

Pin `f4be7f1c5`. vLLM `c961121`. Model: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json
