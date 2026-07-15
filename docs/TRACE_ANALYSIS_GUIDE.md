---
# Trace Analysis Guide: Extracting Kernel Ticket Information from GPU Traces
**Audience:** trace-generating colleagues who need to hand off actionable kernel
optimization briefs to the kernel dev team.
**Also usable as a Claude skill** — see the frontmatter version at
`.cursor/skills/ or ~/.claude/commands/ (see skill README)`.
---

## Purpose

The kernel dev team needs structured information extracted from your GPU traces
before they can open optimization tickets. This guide tells you exactly what to
collect and how to format it, so handoff takes minutes rather than days.

**What you produce:** one *ticket brief* per hot operation (ops consuming >2% of
total compute time). Each brief contains five things the kernel devs need.

---

## Section 1 — Prerequisites and tools

### What you need

- An **eager-mode trace** (`.json.gz`) — see below
- **TraceLens** installed in your Python venv
- Python packages: `openpyxl`, `pandas`

### Trace mode: always use eager mode

Record **two** traces for every profiling run:

| Mode | `enforce_eager` | Purpose |
|---|---|---|
| **Eager** | `True` | **Ticket generation** — shows every kernel individually with full shapes and parameters |
| Compiled | `False` (default) | E2E wall-clock comparison between compiled runs only |

The compiled trace hides kernel launches inside CUDA graph replay events —
individual shapes and kernel names are not accessible. The eager trace is the only
format that contains what the kernel dev team needs. It runs slower per step but
that does not affect the analysis.

**If you only have a compiled trace:** note this in your submission. The kernel dev
team may ask for an eager re-run.

### Finding and running TraceLens

Check if TraceLens is already installed:
```bash
TraceLens_generate_perf_report_pytorch_inference --help
```

If not on PATH, look for it in `../TraceLens`, `~/TraceLens`, or install from
the local checkout:
```bash
pip install -e /path/to/TraceLens --no-deps
```

Generate the report:
```bash
# 1. Find available GPU arch platforms:
python -c "from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms; print(list_platforms())"
# → e.g. ['MI300X', 'MI325X']

# 2. Run TraceLens (pick the closest platform to your GPU):
TraceLens_generate_perf_report_pytorch_inference \
    --profile_json_path  <eager_trace.json.gz> \
    --output_xlsx_path   <report.xlsx> \
    --gpu_arch_platform  <MI325X|MI300X|...> \
    --enable_kernel_summary \
    --topk_ops 80
```

**Platform note (as of TraceLens v0.1.0.dev):** MI355 / gfx950 is not yet in the
arch JSON list. Use `MI325X` for MI355 — kernel names and timing are unaffected;
only the roofline peak numbers will be slightly conservative (MI355 is faster on
FP4/FP8 than MI325X). Adding `MI355X.json` to
`TraceLens/Agent/Analysis/utils/arch/` is a one-file fix.

**Important:** share the `.xlsx` unencrypted. DRM-protected Excel files (produced
by some corporate tools) cannot be opened programmatically. If in doubt, share the
raw trace and let the kernel dev team regenerate the report.

---

## Section 2 — The five things to extract per hot op

Open the TraceLens Excel report. For each op in `ops_summary` that takes **>2% of
total kernel time**, extract all five items below.

### Item 1: Performance fraction

**Source:** `ops_summary` sheet

| Column | What to record |
|---|---|
| `name` | op name (e.g. `aiter::fused_moe_`) |
| `Percentage (%)` | % of total kernel time |
| `total_direct_kernel_time_ms` | absolute ms |
| `Count` | number of calls |

Report **both** percentage and absolute ms:
- Percentages are stable across runs with different trace window lengths
- Absolute ms lets the kernel dev verify their microbenchmark reproduces the
  same latency seen in the trace

Also note the `gpu_timeline` sheet for the overall GPU utilisation breakdown
(computation_time, idle_time, exposed_comm_time) — this gives context.

### Item 2: GPU kernel name(s)

**Source:** `unified_perf_summary` sheet → `trunc_kernel_details` column,
**or** `kernel_summary` sheet → kernel name column,
**or** (if the op is absent from `unified_perf_summary`) → `ops_unique_args` sheet → `kernel_details__summarize_kernel_stats` column

> **TraceLens quirk:** Some ops are treated as aggregation parents by TraceLens
> and do **not** appear as rows in `unified_perf_summary`. The most common example
> is `aiter::fused_moe_` — it shows 0 rows in `unified_perf_summary` but has full
> shape and kernel detail in `ops_unique_args`. Always check both sheets.

- Record the **exact mangled C++ kernel name**, e.g.
  `mfma_moe1_silu_mul_afp4_wfp4_bf16_t64x128x256_pm1_async_v32`
  or `_ZN7ck_tile6kentry...gfx950...FmhaFwdGroupModeKargs...`
- Include it verbatim — the name encodes compile-time tile sizes, dtype,
  pipeline mode, and the kernel dev can decode it
- If one op dispatches **multiple GPU kernels** (e.g. MoE = sorting + stage1 GEMM
  + stage2 GEMM + reduction), list all of them with individual `mean_us`

### Item 3: Input shapes, dtypes, and scalar parameters

**Source:** `unified_perf_summary` sheet, these four columns:

| Column | What it contains |
|---|---|
| `Input Dims` | tensor shapes, e.g. `((32768,4096), (128,3072,2048), ...)` |
| `Input type` | dtypes, e.g. `('c10::BFloat16', 'c10::Float4_e2m1fn_x2', ...)` |
| `Input Strides` | strides per tensor — critical for detecting packed layouts |
| `Concrete Inputs` | scalar args in positional order, e.g. topk, num_experts, softmax_scale |

**Copy these columns verbatim.** The gotchas section below explains how to
interpret non-obvious values (packed FP4 dims, inferred head_dim, etc.).

### Item 4: Per-call timing across all shape variants

**Source:** `unified_perf_summary` sheet

| Column | What to record |
|---|---|
| `Kernel Time (µs)_mean` | mean per call |
| `Kernel Time (µs)_sum` | total across all calls |
| `Kernel Time (µs)_min` / `_max` | flag high variance |
| `operation_count` | call count |

If the same op appears with different input shapes (e.g. prefill attention has
7 distinct (tokens, seqlen) groups), list each group as a table row:

```
| total_tokens | num_seqs | max_seqlen | mean_us |
|---|---|---|---|
| 86736 | 8 | 61504 | 32508 |
| 130424 | 14 | 26320 | 17663 |
| ... |
```

### Item 5: Roofline position

Compute from the shapes and timing. The kernel dev uses this to gauge how much
headroom exists and which direction to push.

**MI355 hardware peaks (gfx950 / CDNA4):**

| dtype | Peak compute | HBM3E BW | Ridge point |
|---|---|---|---|
| FP4 MFMA (E2M1) | ~5 200 TFLOPS/s | 8.8 TB/s | ~591 FLOP/Byte |
| FP8 MFMA (E4M3) | ~2 600 TFLOPS/s | 8.8 TB/s | ~295 FLOP/Byte |
| BF16 MFMA | ~1 300 TFLOPS/s | 8.8 TB/s | ~148 FLOP/Byte |

**FLOPS formulas by operation type:**

```
Dense GEMM (linear layer):
  FLOPS = 2 × M × N × K

MoE FFN, 2-stage (compute per stage separately — they often differ in efficiency):
  M_eff = M_tokens × topk          (routed token-expert pairs, not M_tokens)
  Stage 1 FLOPS = 2 × M_eff × N_up × K_up
  Stage 2 FLOPS = 2 × M_eff × N_down × K_down
  FP4 weights: K_phys = K_logical/2 (packed); scale bytes = N × K_logical/32
  Add scale tensor bytes to Bytes calculation

Prefill attention (full sequence, non-causal):
  FLOPS ≈ 4 × S_q × S_k × H × D_padded
  ("4" approximates QK matmul + softmax + PV matmul + output)
  If head_dim is padded (e.g. 72→128), use D_padded for kernel FLOPS
  and D_logical for "useful" FLOPS — report both and note the padding waste

Decode attention (paged KV cache):
  Memory-bound at short context; compute-bound at long context
  Report as memory-bound unless you know the average context length
  KV cache read dominates Bytes; context length may be unknown from trace alone

Quantisation kernel (e.g. dynamic_per_group_scaled_quant):
  Bytes-only: Bytes_in + Bytes_out_packed + Bytes_scales
  Always memory-bound (AI << any ridge point)
```

**Compute and report:**
```
AI (FLOP/Byte) = FLOPS / Bytes
Bound          = "compute" if AI > Ridge, else "memory"
Achieved       = FLOPS / (mean_us × 1e-6)  [TFLOPS/s]
Efficiency     = Achieved / Peak × 100      [%]
Headroom       = 100 - Efficiency           [%]
```

---

## Section 3 — Output format: the ticket brief

Produce one brief per hot op using this template:

```markdown
## Op: <aiter_op_name / vLLM_op_name>

**% of total compute:** X%  (Y.Y ms total, Z calls)

**GPU kernel(s):**
- `<exact_kernel_name>` — mean Aµs, calls N
- `<second_kernel_if_any>` — mean Bµs, calls N

**Roofline:**
FLOPS = X TFLOPs, Bytes ≈ Y GB, AI ≈ Z FLOP/Byte → BOUND
Achieved: W TFLOPS/s → E% of FP4/FP8/BF16 peak  (headroom: H%)

**Input shapes and dtypes (from trace):**
```
Input Dims:     ((shape1), (shape2), ...)
Input type:     ('dtype1', 'dtype2', ...)
Input Strides:  ((stride1), ...)
Concrete Inputs (scalars, positional): ('val1', 'val2', ...)
```

**Key parameters decoded:**
- M = X tokens, N = Y, K = Z  (logical, not packed)
- num_experts = N, topk = K, quant_type = "..."
- softmax_scale = 0.XXXXX  (→ head_dim_logical = round(1/scale²))

**Shape variants:** (if multiple)
| key | mean_us | calls |
|---|---|---|
| M=32768 | 1843 | 470 |

**Is there a tuned config for this shape?**
```bash
grep ",<expert>,<topk>," aiter/configs/model_configs/*<dtype>*fmoe*.csv | grep ",<model_dim>,"
```
→ Result: present / ABSENT (action needed if absent)

**Notes / open questions:**
- block_size not visible in trace — read from vLLM config `--block-size`
- <any other gotcha specific to this op>
```

---

## Section 4 — MoE-specific: tuning guide and untuned CSV

If any MoE op is in the hotlist, also provide the **aiter tuning input**. The
kernel dev team can run the tuner themselves if the analyst delivers this.

### Check for an existing tuned config first

```bash
# Run on the aiter machine, from the aiter repo root
grep ",<num_experts>,<topk>," aiter/configs/model_configs/*fp4*fmoe*.csv \
    | grep ",<model_dim>,"
```

If a matching row exists → tuning may already be done; mention the row and the
current `us` value so the dev can compare.

If absent → build the **untuned CSV** for the tuner.

### Build the untuned MoE CSV

The CSV has these columns (from `aiter/docs/MOE_TUNING_AGENT_GUIDE.md`):

```
token, model_dim, inter_dim, expert, topk, act_type,
dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1
```

Fill from trace + model config:

| CSV column | Source |
|---|---|
| `token` | M_tokens values from trace (one row per distinct M: 256, 512, 1024, 2048, 4096, 8192, 16384, 32768) |
| `model_dim` | hidden dimension (e.g. 4096); from model `config.json` |
| `inter_dim` | MoE FFN intermediate width per expert (e.g. 3072); from model config |
| `expert` | num_experts (from `Concrete Inputs` in trace) |
| `topk` | top-k (from `Concrete Inputs`) |
| `act_type` | typically `ActivationType.Silu` |
| `dtype` | activation dtype (e.g. `torch.bfloat16`) |
| `q_dtype_a` | activation quant dtype (same as dtype for BF16, or `torch.float8_...`) |
| `q_dtype_w` | weight quant dtype (e.g. `torch.float4_e2m1fn_x2` for FP4) |
| `q_type` | `QuantType.per_1x32` for MX-FP4 block scales |
| `use_g1u1` | always `1` (gate-up fused; only supported config) |
| `doweight_stage1` | `0` (default) |

**Example (from this model, Qwen3-VL w4a4):**
```csv
token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1
256,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
512,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
1024,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
2048,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
4096,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
8192,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
16384,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
32768,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
```

Save as `<model>_untuned_fmoe.csv` and attach to the ticket. The kernel dev team
runs `gemm_moe_tune.py` and produces a tuned CSV. For detailed instructions, tuner
flags, and escalation: see `aiter/docs/MOE_TUNING_AGENT_GUIDE.md`.

**Escalation Slack channel:** `#tiger-aiter-kernel-support`

---

## Section 5 — Gotchas and checklist

Work through this checklist before submitting your analysis.

### Gotchas

**1. Eager trace is required — compiled trace is not sufficient.**
The compiled trace hides kernels inside CUDA graph replay events. Individual
shapes and kernel names are not accessible from it. The eager trace is the only
format that contains what the kernel dev team needs.

**2. head_dim may be padded — infer the logical value from softmax_scale.**
For attention kernels, the tensor shapes in `Input Dims` may show the *padded*
head_dim (e.g. 128) while the model actually uses a smaller *logical* head_dim
(e.g. 72). To find the logical value:
```
D_logical = round(1 / softmax_scale²)
```
For example: softmax_scale = 0.11785 → D_logical = round(1/0.01389) = 72.
This matters for FLOP counting, understanding padding waste, and knowing what
head_dim a new kernel must support.

**3. FP4 tensors: physical K ≠ logical K.**
Relevant for ops using `Float4_e2m1fn_x2` weights (MoE GEMMs, quantized linears).
- Physical K in the trace: `K_phys` (what appears in `Input Dims`)
- Logical K (what the kernel dev needs): `K_logical = K_phys × 2`
- Scale shapes: `(N, K_logical/32)` — one E8M0 scale per 32 weight elements
Always report both physical (trace) and logical (for FLOP counting).

**4. Paged KV-cache block_size: not visible for decode attention directly, but
often visible in nearby ops.**
For decode attention (`kernel_unified_attention_2d`, `aiter::paged_attention`),
the KV cache tensors appear as empty `()` args — the kernel accesses cache via
internal vLLM state, not explicit tensor arguments. However, `block_size` and
`page_size` ARE visible in the `Concrete Inputs` of the op that *writes* to the
KV cache. Look for nearby ops like `aiter::fused_qk_norm_mrope_3d_cache_pts_quant_shuffle`
— in the Qwen3-VL trace its `Concrete Inputs` contains `block_size=64` (pos 24)
and `page_size=16` (pos 25). Always check KV-cache-writing ops before falling back
to the vLLM serving config (`--block-size`, default 16).

**5. `dynamic_per_group_scaled_quant` is a separate top-level op for w4a4 MoE.**
For FP4-quantized MoE, this quantisation kernel fires as a standalone top-level
call *between* MoE stage 1 and stage 2 — it does NOT appear as a child of
`fused_moe_`. Check `ops_summary`: if it has its own row, it has its own shapes
to report and its own latency to include in the MoE analysis. The dominant shape
is `(M_tokens × topk, K_down)` BF16 input (e.g. `(262144, 1536)`).

**6. For MoE: report M_tokens AND M_inter separately.**
`fused_moe_` is called with M_tokens (e.g. 32768). But the inter-stage
quantisation and stage 2 GEMM operate on M_inter = M_tokens × topk (e.g.
32768 × 8 = 262144) because all routed token-expert pairs are materialised
as separate rows. Both dimensions appear in the trace in different rows — include
both so microbenchmark authors use the right M for each kernel stage.

**7. Unencrypted xlsx only.** DRM-protected Excel files cannot be opened
programmatically. Share the raw trace or regenerate with TraceLens.

### Required metadata — include in every submission

```
Point of contact:  <name of person who ran the trace>
Slack channel:     <#team-channel> (default: #tiger-aiter-kernel-support)
Deadline:          <date, if results feed a benchmark / MLPerf submission / release>
Hardware:
  GPU SKU:         e.g. MI355X (gfx950)
  Number of GPUs:  e.g. 8
  ROCm version:    cat /opt/rocm/.info/version
  vLLM version:    pip show vllm | grep Version
  aiter commit:    git -C /path/to/aiter rev-parse HEAD
vLLM serving config:
  --block-size:    <value, default 16>
  --max-num-seqs:  <value>
  Quantisation:    <method, e.g. w4a4 MX-FP4>
```

### Checklist before submitting

- [ ] Eager trace used (not compiled-only)
- [ ] TraceLens report generated and unencrypted
- [ ] All ops > 2% of total compute time have a ticket brief
- [ ] Each brief has all five items: %, kernel name, shapes, timing, roofline
- [ ] head_dim verified via softmax_scale (if attention op present)
- [ ] FP4 physical vs logical K both reported (if FP4 op present)
- [ ] block_size noted from serving config (if decode attention present)
- [ ] `dynamic_per_group_scaled_quant` checked as separate op (if w4a4 MoE present)
- [ ] M_tokens AND M_inter both reported (if MoE present)
- [ ] Tuned CSV existence checked; untuned CSV attached if absent (if MoE present)
- [ ] Required metadata block filled in (contact, deadline, hardware, serving config)

---

## Quick reference: most useful TraceLens sheets and columns

| What you need | Sheet | Column(s) |
|---|---|---|
| Op % of compute, call count | `ops_summary` | `Percentage (%)`, `Count`, `total_direct_kernel_time_ms` |
| Per-call kernel timing | `unified_perf_summary` | `Kernel Time (µs)_mean/sum/min/max` |
| Tensor shapes and dtypes | `unified_perf_summary` | `Input Dims`, `Input type` |
| Strides (layout detection) | `unified_perf_summary` | `Input Strides` |
| Scalar params (topk, scale) | `unified_perf_summary` | `Concrete Inputs` |
| Exact GPU kernel names | `unified_perf_summary` | `trunc_kernel_details` |
| Top GPU kernels ranked | `kernel_summary` | kernel name, total duration, count, mean |
| GPU utilisation overview | `gpu_timeline` | computation_time, idle_time, busy_time |
