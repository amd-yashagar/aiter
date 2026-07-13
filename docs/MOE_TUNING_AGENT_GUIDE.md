# MoE kernel tuning — agent guide

The **easy, agent-led path** to tune AITER fused MoE (fmoe) kernels for your shapes. The
[README](../csrc/ck_gemm_moe_2stages_codegen/README.md) has the raw instructions; this guide
just walks you down one simple path and quietly collects the handoff data you'd need if you
ever get stuck and want the tuning team to help.

One goal: **tune MoE kernels for your shapes.** It pivots on a single question — *do you have
the untuned config CSV?* If yes, you tune. If no, we build it together from your model config
and logs, then you tune. That's it. All paths are relative to the **aiter repo root** (the
directory with `setup.py`).

> **Support:** the AITER MoE / kernel tuning point of contact **and** the place to ask for
> help are the same — on Slack, **`tiger-aiter-kernel-support`**. Stuck or unsure? Ask there.

## How to use this guide

1. Get this file from the **AITER tuning support branch** and drop it at
   `aiter/docs/MOE_TUNING_AGENT_GUIDE.md` in your aiter checkout (any path works if your
   `@`-mention matches).
2. In Cursor (or any agent editor), `@MOE_TUNING_AGENT_GUIDE.md` and say **"Start MoE tuning"**.
   The agent reads this file and follows the **Agent instructions** below — you don't copy
   anything into chat.

*Optional Cursor rule* — create `.cursor/rules/moe-tuning.mdc` at your aiter repo root:

```yaml
---
description: MoE (fmoe) kernel tuning — read agent guide when user asks
alwaysApply: false
---
When the user asks to tune fused MoE (fmoe) kernels, read
aiter/docs/MOE_TUNING_AGENT_GUIDE.md and follow the tuning path in order.
```

## Agent instructions

```
You are helping a user tune AITER fused MoE (fmoe) kernels via gemm_moe_tune.py.

ALWAYS ask the short must-ask questions below FIRST — even if an untuned CSV
already exists in the repo. The shipped `aiter/configs/untuned_fmoe.csv` is NOT
necessarily the user's shapes; never silently assume it or auto-proceed with it.
Then follow ONE linear path (no decision tree): if they have (or point you to) an
untuned CSV, tune it; if not, help build it from their model config + serving
logs (Step 1a), then tune.

Must-ask questions (keep it short — ask these, then proceed; don't re-ask what a
chosen CSV already contains in its rows):
1. What model / shapes do you want to tune? Do you have your OWN untuned config
   CSV — if so, what's its path? (If a repo `untuned_fmoe.csv` already exists,
   ASK whether to use it or their own shapes — never assume it's theirs.)
2. If no CSV yet: gather the fields to build one — walk them through the Step 1a
   mapping.
3. Which GPU / gfx arch are you tuning on?
4. Full set of token sizes, or a quick single-row check (`--last`) first?
5. Any flags to add (see Useful flags — more candidates `-k`, profile all `-o2`,
   compare pre/post `--compare`)?

Inform, don't drive:
- PRESENT the tune command (with its flag annotations and the useful-flags list)
  and ask the USER to run it. Run it yourself ONLY if the user explicitly asks.
- Walk the user through the rest of the path so they stay in control: output-path
  options (default vs model_configs), the estimate-improvement step before
  re-serving, deploy, and trace-confirm. Don't skip these.
- Let the user know the support channel exists: on Slack,
  **`tiger-aiter-kernel-support`** is both the point of contact and where to ask
  for help — mention it early and again if they head toward escalation.
- Make the user aware of the handoff items and COLLECT the applicable ones into
  the progress record so an escalation package is ready — especially when heading
  toward escalation: a point of contact (at least one reachable person); the Slack
  channel (`tiger-aiter-kernel-support`); if a benchmark is involved, its script +
  exact command line + params to reproduce on AMD hardware; a link/reference to a
  spec MD file (don't paste the whole spec); and optionally a reference impl
  (Triton/other) or a paper. DO ask for a POC/contact, and — if a benchmark is
  involved — for the benchmark script + command.
- Keep the progress record (below) updated after each answer/command so an
  escalation package is ready.

Rules:
- Run from the aiter repo root after `python3 setup.py develop`, on a free GPU.
- First tune pass JIT-builds kernels (several minutes) — always warn, and recommend
  `--last` first to tune ONE row and confirm progress before the full run.
- Only G1U1 (gate-up fused) configs are tunable. NEVER widen --errRatio to force a pass.
- Encouraging tone; the user stays in control — present commands, don't overload them.
```

## The tuning path

### Step 1 — The one question: do you have an untuned config CSV?

The untuned CSV is just the list of shapes you want tuned. Its columns are:

```
token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1
```

Real example (one model shape, three token sizes — from
`aiter/configs/model_configs/qwen3_235b_bf16_untuned_fmoe.csv`):

```
token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1
1,4096,384,128,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.bfloat16,QuantType.No,1,0
128,4096,384,128,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.bfloat16,QuantType.No,1,0
2048,4096,384,128,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.bfloat16,QuantType.No,1,0
```

Shipped CSVs use `torch.*` dtype names (`torch.bfloat16`, `torch.float8_e4m3fnuz`); `dtypes.*`
names (`dtypes.bf16`, `dtypes.fp8`) also parse. Save your rows to
`aiter/configs/untuned_fmoe.csv` (or `aiter/configs/model_configs/<model>_untuned_fmoe.csv`).

**Have your own CSV (or confirmed the existing repo one holds your shapes)? → Step 2.**
If a repo `untuned_fmoe.csv` exists but you haven't confirmed it's yours, ask first —
don't assume it. No CSV yet → Step 1a.

### Step 1a — Build the CSV from what you have (only if no CSV)

You almost certainly have the pieces already. The left column below is the literal
untuned-CSV header (the same header shown in Step 1); map each one to what you already have:

| CSV column | Where it comes from |
|---|---|
| `model_dim` | model config `hidden_size` |
| `inter_dim` | MoE intermediate size (per-expert FFN width) |
| `expert` | number of experts your model routes over |
| `topk` | experts activated per token |
| `act_type` | activation, usually `ActivationType.Silu` |
| `dtype` | compute dtype, usually `torch.bfloat16` |
| `q_dtype_a` / `q_dtype_w` | activation / weight quant dtype (e.g. `torch.float8_e4m3fnuz`, `torch.float4_e2m1fn_x2`), else same as `dtype` |
| `q_type` | `QuantType.No`, `.per_Token`, `.per_1x128` (blockscale), or `.per_1x32` (MXFP4, gfx950 only) |
| `use_g1u1` | `1` (only G1U1 is supported) |
| `doweight_stage1` | `0` (default; `1` applies routing weights in stage 1) |
| `token` | the batch/token sizes you serve or want tuned — e.g. `1, 8, 64, 512, 2048` |

- **model config** (HF `config.json` / model card) → `model_dim`, `inter_dim`, `expert`, `topk`.
- **quant / serving config** → `dtype`, `q_dtype_a`, `q_dtype_w`, `q_type`, `act_type`.
- **serving logs / traces** → which token counts actually run; make one row per token size.

If unsure, copy the closest row from `aiter/configs/model_configs/` and edit the dims to
match your model. Write your rows to `aiter/configs/untuned_fmoe.csv` and continue.

### Step 2 — Run the tune

From the aiter repo root (after `python3 setup.py develop`, on a free GPU):

```bash
# Recommended first: tune only ONE row to confirm the JIT build works and see progress.
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i aiter/configs/untuned_fmoe.csv -o aiter/configs/tuned_fmoe.csv --last

# Then the full run over all rows:
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i aiter/configs/untuned_fmoe.csv -o aiter/configs/tuned_fmoe.csv
```

Flags: `-i` = input untuned CSV (the shapes to tune), `-o` = output tuned CSV (winners land
here), `--last` = tune only the last row first to sanity-check the build.

**Output path — both are optional.** `-i` and `-o` have defaults, so you can run with neither:
omitting `-o` writes to `aiter/configs/tuned_fmoe.csv` and omitting `-i` reads
`aiter/configs/untuned_fmoe.csv` (the exact paths the explicit example uses; if the
`AITER_CONFIG_FMOE` env var is set, that path becomes the `-o` default instead). **Alternative:**
save your winners as `aiter/configs/model_configs/<model>_tuned_fmoe.csv` (matching the shipped
naming, e.g. `qwen3_235b_bf16_tuned_fmoe.csv`, `dsv3_fp4_tuned_fmoe.csv`) so they sit beside the
other model tunings and are auto-merged at runtime (see Step 5).

**Heads up:** the first run JIT-builds the MoE kernels — several minutes on a free GPU, *not*
hung. `--last` first is the easiest way to confirm it's working before the full file.

### Step 3 — Check the result

Winners land in your `-o` file. Its literal header (from the shipped `aiter/configs/tuned_fmoe.csv`)
is the input columns followed by the tuning results:

```
cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw,_tag
```

Freshly tuned files prepend a `gfx` column (and may add `xbf16` / `flat`); older shipped files
lead with `cu_num` as above. The columns to read: `gfx` / `cu_num` (arch + compute-unit count —
kernels are picked per GPU SKU), `us` / `us1` / `us2` (measured stage timings, µs), `run_1stage`
(`1` = fused 1-stage asm, `0` = 2-stage CK), `kernelName1` / `kernelName2` (the winning stage-1 /
stage-2 kernels), and `err1` / `err2` (accuracy vs reference). A valid, low-`err` winner means
that shape is tuned.

**Estimate the gain before re-serving (optional but cheap).** `us` is the tuned kernel's
per-call time (µs) for that shape — `us1` + `us2` across the two stages. From your *current*
traces/logs, note the MoE per-call time you're getting today, then compare it against the tuned
row's `us` for the matching shape. If the tuned `us` isn't clearly lower, re-serving won't speed
that shape up — which is itself useful signal to send us. Match by shape and by GPU: tuned rows
are keyed on `gfx` + `cu_num`, so a row only applies on the same GPU SKU it was tuned on.

### Step 4 — Validate (brief)

```bash
# Edit the test instance, then run it (AITER_REBUILD=1 if kernels were built before this tune):
AITER_REBUILD=1 python3 op_tests/test_moe.py      # or: op_tests/test_moe_2stage.py
```

### Step 5 — Serve with the new tunings & confirm

Deploy the tuned CSV one of two ways (they don't combine — pick one):

```bash
# (a) Env var: point serving straight at your file.
export AITER_CONFIG_FMOE=/path/to/tuned_fmoe.csv
#     A single file is used as-is. To also keep the shipped defaults, pass an
#     os.pathsep (":") list; aiter merges them into /tmp/aiter_configs/tuned_fmoe.csv:
export AITER_CONFIG_FMOE=aiter/configs/tuned_fmoe.csv:/path/to/your_tuned_fmoe.csv

# (b) model_configs file: drop it in and leave AITER_CONFIG_FMOE UNSET.
cp your_winners.csv aiter/configs/model_configs/<model>_tuned_fmoe.csv
```

With `AITER_CONFIG_FMOE` unset (b), aiter globs `aiter/configs/model_configs/*tuned_fmoe*.csv`
(files with `untuned` in the name are ignored) and merges them with the shipped
`aiter/configs/tuned_fmoe.csv` into `/tmp/aiter_configs/tuned_fmoe.csv`. Setting the env var (a)
uses only what the env var points at — the `model_configs/` glob is skipped in that case.

Restart / rebuild so JIT picks up the tuned kernels (`AITER_REBUILD=1` for local tests), then
serve and **confirm the tuned kernel is the one selected**. On each MoE call aiter looks up the
row by shape key and dispatches its `kernelName1` / `kernelName2` (or the 1-stage asm path when
`run_1stage=1`), and logs an INFO line:

```
[fused_moe] using 2stage (kernelName1='...', kernelName2='...') for (gfx, cu_num, token, ...)
```

That log line is the reliable confirmation: check it names your tuned row's kernels and the
right 1-stage-vs-2-stage choice. In an rocprof / rocprofv3 trace the launched kernel symbol
should correspond to that name — kernel symbols can be decorated, so match on the kernel-name
substring and the stage count rather than expecting a byte-for-byte string.

### Useful flags (short list — full list in the [README](../csrc/ck_gemm_moe_2stages_codegen/README.md))

| Flag | Purpose |
|---|---|
| `--last` | Tune only the last row — do this first to sanity-check the build |
| `--all` | Retune all shapes (rather than skipping already-tuned ones) |
| `--mp N` | Use N GPUs in parallel (default: all visible) |
| `-k` / `--splitK` | Enable split-K candidates |
| `--errRatio F` | Max error ratio for a valid kernel (fmoe default **0.5**; `--help` misleadingly prints 0.05) |
| `-o2` / `--profile_file` | Save **all** candidates, not just winners |
| `--compare` / `--update_improved` | Benchmark pre/post; only update tuned CSV if improved |
| `--run_config [TUNED_CSV]` | Benchmark only (no tuning) — check a shape you already tuned |
| `-v` | Verbose logging |

## Progress record (agent keeps this updated)

```yaml
# moe_tuning_session — hand this over verbatim if you escalate
request: ""              # what the user wants tuned, in one line
status: ""               # in_progress | tuned | stuck
point_of_contact: ""     # at least one person reachable for clarification
slack_channel: ""        # request/support channel (default: tiger-aiter-kernel-support)
untuned_csv: ""          # path + the token/model_dim/... rows
tuned_csv: ""            # -o output path, if produced
commands_run: []
problem: ""              # what's going wrong, if anything
benchmark:               # if a benchmark is provided
  script: ""             # path to the benchmark script
  command: ""            # exact command line used
  params: ""             # params / notes to reproduce (at least on AMD HW)
spec_md_ref: ""          # link/reference to an MD spec file (don't paste the full spec)
reference_impl_or_paper: "" # optional: Triton/other ref impl, or a paper describing the algorithm
env:
  aiter_commit: ""       # git rev-parse HEAD
  rocm_version: ""       # cat /opt/rocm/.info/version
  gfx: ""                # get_gfx() or rocm-smi
  framework: ""          # vllm | sglang | op_tests | custom
serving_logs: ""         # path, if any
trace_files: ""          # rocprof / profiler output, if any
```

Version / env commands (run from the aiter repo root):

```bash
pip show aiter | grep -i version
git rev-parse HEAD
cat /opt/rocm/.info/version 2>/dev/null || rocm-smi | head
python -c "from aiter.jit.utils.chip_info import get_gfx; print(get_gfx())"
```

## If you get stuck → what to send us

Don't spin your wheels. Reach out on Slack **`tiger-aiter-kernel-support`** (the point of
contact and support channel) and package these:

- [ ] A **point of contact** — at least one person reachable for clarification
- [ ] The **Slack channel** for the request (`tiger-aiter-kernel-support`)
- [ ] The **untuned CSV** (the shapes you're tuning)
- [ ] The **tuned CSV** (`-o` output), if any was produced
- [ ] **Trace files** (rocprof / profiler), if the issue is performance
- [ ] **Serving logs** around the slow / failing MoE call
- [ ] A short **description of the problem** (what you expected vs. saw)
- [ ] **Benchmark reproduction** (if a benchmark is provided): the script, the exact
      command line, and the params to reproduce it — at least on AMD hardware
- [ ] **Spec reference**: a link to an MD file describing it (e.g. in a git repo) rather
      than the full spec pasted in
- [ ] *Optional:* a **reference implementation** (Triton or other) or a **paper** describing
      the algorithm
- [ ] The **progress record** block above, filled in
- [ ] **Versions**: aiter commit, ROCm version, GPU / gfx arch

## FAQ

| Symptom | Likely cause / fix |
|---|---|
| First run seems to hang | It's JIT-building kernels (minutes). Not hung — run `--last` first to confirm progress. |
| "No valid kernel" / all rejected | `--errRatio` too tight or an unsupported quant path. Check quant type; don't widen tolerance to force a pass. |
| A shape was skipped | It's already in the tuned CSV. Use `--all` to retune it. |
| Missing config at serving time | Shape isn't in the merged CSV — add it to the untuned CSV, tune, and set `AITER_CONFIG_FMOE`. |

*Facts aligned with `csrc/ck_gemm_moe_2stages_codegen/README.md`.*
