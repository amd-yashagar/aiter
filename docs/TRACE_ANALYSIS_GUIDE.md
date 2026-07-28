# Trace Analysis Guide: Extracting Kernel Ticket Information from GPU Traces

**Audience:** anyone who generates GPU traces and needs to hand off an actionable
kernel optimization request to the kernel team (AITER, Triton, Composable Kernel,
ASM, hipBLASLt).

## Purpose

Kernel requests often arrive as "low performance compared to competitor, please
fix". Without shapes, data types, profiling data, and model context the kernel team
cannot triage, scope, or start work. This guide tells you exactly what to collect
from your trace and what context to attach, so a request is actionable on arrival.

**What you produce:** one *ticket brief* per hot operation. Each brief has:

- **four items extracted from the trace** (Section 2) — performance fraction, GPU
  kernel names, shapes/dtypes/scalars, per-call timing
- **request context** (Section 3) — operation type, model context, data-type
  details, performance target, fusion opportunities, and a reproducer

**Which ops to include:** every op above ~2% of total kernel time. Also include a
group of several small related kernels when together they are a meaningful share
(>5%) of end-to-end time — the kernel team prioritizes by end-to-end impact, so
give both the per-op share and the end-to-end share.

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

**MI355X / gfx950 platform support:**

TraceLens v0.1.0.dev does not yet include an MI355X arch spec. Install it with the
one-liner below; once present, TraceLens reports correct TFLOPS/s and efficiency %
for gfx950 in the `GEMM` sheet automatically — no manual roofline computation
needed for GEMM ops.

```bash
cat > "$(python -c "import TraceLens, os; \
    print(os.path.join(os.path.dirname(TraceLens.__file__), \
    'Agent/Analysis/utils/arch/MI355X.json'))")" << 'EOF'
{
    "name": "MI355X",
    "mem_bw_gbps": 6198,
    "memory_gb": 288,
    "max_achievable_tflops": {
        "matrix_fp16": 1236,
        "matrix_bf16": 1236,
        "matrix_fp32": 157,
        "matrix_fp64": 79,
        "matrix_fp8": 2470,
        "matrix_fp4": 4940,
        "matrix_int8": 2470,
        "vector_fp16": 157,
        "vector_bf16": 157,
        "vector_fp32": 157,
        "vector_fp64": 79
    }
}
EOF
```

These are empirically measured peaks for MI355X (CDNA4) — slightly below the
official AMD datasheet values and therefore more realistic for kernel benchmarking.
Official spec: BF16 matrix = 1300 TFLOPS/s, FP8 = 2600 TFLOPS/s,
FP4 = 5200 TFLOPS/s, HBM3E BW = 8.8 TB/s.

Remove this step once MI355X is merged into TraceLens upstream.

**Important:** share the `.xlsx` unencrypted. DRM-protected Excel files (produced
by some corporate tools) cannot be opened programmatically. If in doubt, share the
raw trace and let the kernel dev team regenerate the report.

---

## Section 2 — The four things to extract per hot op

Open the TraceLens Excel report. For each op in `ops_summary` that takes **>2% of
total kernel time**, extract all four items below.

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

**Decode ops are the exception:** they often show empty `()` args even in an eager
trace, so the shapes have to be logged at the call site and extracted from the log
instead — see gotcha 5, and attach the log file with your submission.

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

If shapes are dynamic and vary between runs or inference steps, give the most
common shapes plus the observed min/max range — a kernel tuned for one shape can
regress on another.

---

## Section 3 — Output format: the ticket brief

### Classify the operation first

State the category so the request routes to the right owner:

`GEMM` (standard / batched / grouped / stream-K / blockscale) · `Attention`
(prefill / decode / paged / quantized) · `MoE` (gating, fused 1-stage or 2-stage
GEMM) · `Normalization` (RMSNorm, LayerNorm, GroupNorm) · `Activation /
elementwise` (SiLU, GELU, fused bias+activation, residual add) · `Quantization`
(FP8, blockscale, AWQ/GPTQ dequant) · `Embedding` (RoPE, token/position) ·
`Reduction` (softmax, TopK, AllReduce) · `Custom / fused`

For a **custom or fused** op, also describe the maths — a reference implementation
(PyTorch, NumPy, or pseudocode) is the fastest way to convey it.

### Brief template

Produce one brief per hot op:

````markdown
## Op: <aiter_op_name / vLLM_op_name>

**Operation type:** <category above; describe the maths if custom/fused>

**% of total compute:** X% (Y.Y ms total, Z calls) — and W% of end-to-end latency

**GPU kernel(s):**
- `<exact_kernel_name>` — mean Aµs, calls N
- `<second_kernel_if_any>` — mean Bµs, calls N

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

**Data types:**
- input A / input B: <e.g. bf16 / fp4_e2m1 packed>
- output: <e.g. bf16>    accumulation: <e.g. fp32>
- scales: <dtype, e.g. e8m0> · granularity: <per tensor / per channel / per block>
  · block size: <e.g. 32> · zero point: <if any>

**Model context:**
- layer / block: <e.g. gate+up projection in the MLP; QKV projection>
- phase: <prefill / decode / both>
- inside a captured graph (CUDA/hipGraph): <yes / no>
- model architecture: <e.g. Qwen3-VL MoE>
- framework: <vLLM x.y / SGLang x.y / custom>

**Performance target:**
- current: <mean µs per call, total ms>
- target: <concrete goal — e.g. match <reference kernel> at N µs, or ≥90% of <baseline>>
- baseline / comparison: <competitor number on comparable hardware, or reference kernel>

**Shape variants:** (if multiple)
| key | mean_us | calls |
|---|---|---|
| M=32768 | 1843 | 470 |

**Fusion opportunities:**
- <e.g. bias add / activation / quantization / residual immediately after this op>

**Reproducer:** <attached microbenchmark, or existing benchmark script + exact
command — see Section 4>

**Is there a tuned config for this shape?** (MoE)
```bash
grep ",<expert>,<topk>," aiter/configs/model_configs/*tuned_fmoe*.csv \
    | grep -v untuned | grep ",<model_dim>,"
```
→ Result: present / ABSENT (action needed if absent)

**Notes / open questions:**
- <e.g. block_size not visible in trace — read from vLLM config `--block-size`>
````

---

## Section 4 — Reproducer, environment, and deeper profiling

A trace shows *where* time goes; a reproducer lets the kernel team iterate on it.
Requests that arrive with one start days sooner.

### Reproducer

- A **standalone Python script** that builds the inputs and calls the op in
  isolation, runnable with minimal dependencies.
- If an **existing benchmark already reproduces it** (for example a script in
  AITER), say so and give the exact command — that is usually enough, no new script
  needed.
- For kernels **not authored by the kernel team**, include the kernel source.
- For AITER / vLLM / SGLang kernels, name the **specific function** and its version
  (git commit, tag, or release).

### Environment

Pin the environment so numbers are comparable:

- Docker image or container tag
- ROCm version — `cat /opt/rocm/.info/version`
- Framework version — `pip show vllm | grep Version`
- aiter commit — `git -C /path/to/aiter rev-parse HEAD`
- Triton commit, if a Triton kernel is involved

### Deeper profiling (when the PyTorch trace is not enough)

For hardware-level questions — scheduling gaps, memory-bandwidth limits,
MFMA/VALU utilisation — collect a rocprofv3 kernel trace alongside the PyTorch
trace:

```bash
rocprofv3 --stats --kernel-trace -- python <your_script.py>
```

This writes per-kernel CSVs with start/end timestamps, kernel names, and counter
data. Attach the CSVs, not screenshots.

### Model code

Link the model implementation in the framework (or in `transformers`). It lets the
kernel team judge fusion opportunities and how a new kernel would integrate.

---

## Section 5 — MoE-specific: tuning guide and untuned CSV

If any MoE op is in the hotlist, also provide the **aiter tuning input**. The
kernel dev team can run the tuner themselves if the analyst delivers this.

### Check for an existing tuned config first

```bash
# Run on the aiter machine, from the aiter repo root
grep ",<num_experts>,<topk>," aiter/configs/model_configs/*tuned_fmoe*.csv \
    | grep -v untuned | grep ",<model_dim>,"
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
| `inter_dim` | MoE FFN intermediate width per expert (e.g. 3072); `moe_intermediate_size` in HF `config.json` — the field name differs |
| `expert` | num_experts (from `Concrete Inputs` in trace) |
| `topk` | top-k (from `Concrete Inputs`) |
| `act_type` | typically `ActivationType.Silu` |
| `dtype` | activation dtype (e.g. `torch.bfloat16`) |
| `q_dtype_a` | activation quant dtype (same as dtype for BF16, `torch.float8_...` for FP8, or `torch.float4_e2m1fn_x2` for FP4 activations / w4a4) |
| `q_dtype_w` | weight quant dtype (e.g. `torch.float4_e2m1fn_x2` for FP4) |
| `q_type` | `QuantType.per_1x32` for MX-FP4 block scales |
| `use_g1u1` | always `1` (gate-up fused; only supported config) |
| `doweight_stage1` | `0` (default) |

**Example — Qwen3-VL w4a4:**
```csv
token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1
256,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
512,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
1024,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
2048,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
4096,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
8192,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
16384,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
32768,4096,3072,128,8,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
```

Save as `<model>_untuned_fmoe.csv` and attach to the ticket. The kernel dev team
runs `gemm_moe_tune.py` and produces a tuned CSV. For detailed instructions, tuner
flags, and escalation: see `aiter/docs/MOE_TUNING_AGENT_GUIDE.md`.

**Escalation Slack channel:** `#tiger-aiter-kernel-support`

---

## Section 6 — Gotchas and checklist

Work through this checklist before submitting your analysis.

### Gotchas

**1. Eager trace is required — a compiled trace is not sufficient.**
See Section 1: a compiled trace hides kernels inside graph-replay events, so
individual shapes and kernel names are not recoverable from it.

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

**5. Decode ops may carry no usable shapes at all — log them and attach the log.**
Even in an eager trace, decode-path ops often expose empty `()` tensor arguments,
because the kernel reads the KV cache and its metadata from internal framework
state rather than from explicit arguments (same root cause as gotcha 4). When
`Input Dims` is empty or clearly incomplete for a decode op, the trace alone cannot
tell the kernel team what the kernel was called with. Do this instead:
- add a log line at the call site, or enable the framework's debug logging, so that
  each call prints its tensor shapes, dtypes, and scalar arguments;
- re-run the same workload and extract the shapes from the log;
- report those shapes in the brief **and attach the log file**, so the kernel team
  can verify the values and see the call sequence.
Mark clearly in the brief which shapes came from logs rather than from the trace.

**6. `dynamic_per_group_scaled_quant` is a separate top-level op for w4a4 MoE.**
For FP4-quantized MoE, this quantisation kernel fires as a standalone top-level
call *between* MoE stage 1 and stage 2 — it does NOT appear as a child of
`fused_moe_`. Check `ops_summary`: if it has its own row, it has its own shapes
to report and its own latency to include in the MoE analysis. The dominant shape
is `(M_tokens × topk, K_down)` BF16 input (e.g. `(262144, 1536)`).

**7. For MoE: report M_tokens AND M_inter separately.**
`fused_moe_` is called with M_tokens (e.g. 32768). But the inter-stage
quantisation and stage 2 GEMM operate on M_inter = M_tokens × topk (e.g.
32768 × 8 = 262144) because all routed token-expert pairs are materialised
as separate rows. Both dimensions appear in the trace in different rows — include
both so microbenchmark authors use the right M for each kernel stage.

**8. Unencrypted xlsx only.** DRM-protected Excel files cannot be opened
programmatically. Share the raw trace or regenerate with TraceLens.

### Required metadata — include in every submission

```
Point of contact:   <name of person who ran the trace — someone reachable for questions>
Slack channel:      <#request-specific channel> (default: #tiger-aiter-kernel-support)
Deadline:           <date, and what it gates — benchmark / MLPerf submission / release>
Definition of done: <implemented + measured on target HW | merged to public repo>
Hardware:
  GPU SKU:          e.g. MI355X (gfx950)
  Number of GPUs:   e.g. 8
Environment:
  Docker image:     <image or container tag>
  ROCm version:     cat /opt/rocm/.info/version
  vLLM version:     pip show vllm | grep Version
  aiter commit:     git -C /path/to/aiter rev-parse HEAD
Model:
  Architecture:     <e.g. Qwen3-VL MoE>
  Framework:        <vLLM / SGLang / custom, with version>
  Model code:       <link to the implementation in the framework or transformers>
Serving config:
  --block-size:     <value, default 16>
  --max-num-seqs:   <value>
  Quantisation:     <method, e.g. w4a4 MX-FP4>
```

A deadline is only actionable with a definition of done: "implemented and measured
on the target hardware" and "merged into the public repo" are different milestones,
so state which one the date gates.

### Checklist before submitting

- [ ] Eager trace used (not compiled-only)
- [ ] TraceLens report generated and unencrypted
- [ ] All ops > 2% of total compute time have a ticket brief
- [ ] Each brief has all four trace items: %, kernel name, shapes, timing
- [ ] For GEMM ops: copy `TFLOPS/s_mean` and `Compute Spec` from the TraceLens `GEMM` sheet into the brief (no manual calculation needed once MI355X.json is installed)
- [ ] For attention and MoE ops: include kernel time; compute efficiency manually or leave it to the kernel team
- [ ] Operation type classified (maths or reference implementation if custom/fused)
- [ ] Data types complete: inputs, output, accumulation, scale dtype + granularity
      + block size
- [ ] Model context filled in (layer/block, prefill vs decode, graph capture,
      architecture, framework)
- [ ] Performance target and baseline stated — not just "make it faster"
- [ ] Fusion opportunities noted (bias, activation, quantization, residual)
- [ ] Reproducer attached, or an existing benchmark script + exact command given
- [ ] Environment pinned (docker image, ROCm, framework version, aiter commit)
- [ ] head_dim verified via softmax_scale (if attention op present)
- [ ] FP4 physical vs logical K both reported (if FP4 op present)
- [ ] block_size noted from serving config (if decode attention present)
- [ ] Decode shapes logged, extracted, and the log file attached (if the trace shows
      empty args for decode ops)
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
