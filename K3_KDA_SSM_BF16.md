# Kimi-K3 FlyDSL causal conv1d (prefill)

Branch: `k3-kda-ssm-bf16`  
Pair with: `amd-mvarjoka/sglang` `k3-kda-ssm-bf16`  
Target: gfx950. One commit on `ROCm/aiter` main.

## What changed

`aiter/ops/flydsl/causal_conv1d_flydsl.py` only:

- Index activations as `feat * sx0 + tok * sx1` (previously ignored `sx1`).
- Type weight/bias buffers from the **weight tensor dtype** (`fp32` / `bf16` / `fp16`), not from the activation dtype.

Kimi-K3 prefill passes a transposed `[dim, T]` activation and **FP32** conv weights. Without this, FlyDSL was wrong-width and ignored the token stride.

## Why SGLang does not dispatch it yet

On gfx950 K3 shapes (T=1024, dim=4608) this kernel measured ~4.6× **slower** than Triton `causal_conv1d_fn`. SGLang serving still uses Triton for prefill conv. The SGLang adapter `kda_prefill_conv_aiter_hip.py` exists to correctness-gate this fix; it is not on the serving path.

Fused **decode** KDA + bf16 SSM lives in the SGLang branch, not here.

## How to use / test

```bash
pytest op_tests/test_causal_conv1d_prefill_split_qkv.py -v
```

From the SGLang worktree, with this AITER on `PYTHONPATH`:

```bash
pytest test/registered/kernels/ops/kimi_k3/flydsl_ops/test_kimi_k3_kda_prefill_conv.py -v
```

Expect bit-exact (or policy-tight) match vs Triton on contiguous + transposed layouts, including FP32 weights.

## Expected

Correctness only. Do not treat this commit as a prefill speed win. Do not enable FlyDSL prefill conv in SGLang serving until a measured gfx950 win vs Triton on K3 shapes.
