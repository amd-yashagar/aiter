# SILOTIGER-957 gfx942 FMHA assembly experiments

Contract: non-causal packed-THD FMHA, bf16, H=12, QK=192, V=128,
`return_lse=True`, gfx942. The matched ticket shape is `Sq=4096, Sk=42700`.
Correctness requires byte-identical output and LSE against the current FlyDSL
kernel; timing uses 8 interleaved rounds of 21 CUDA-event iterations.

## Repro

Dump the current FlyDSL compilation:

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
FLYDSL_DUMP_IR=1 \
FLYDSL_DUMP_DIR=$PWD/asm_experiments/silotiger957 \
python3 asm_experiments/silotiger957/dump_baseline.py
```

Reassemble the emitted source:

```bash
/opt/rocm/llvm/bin/clang -x assembler \
  -target amdgcn-amd-amdhsa -mcpu=gfx942 \
  -o asm_experiments/silotiger957/fmha_varlen_gfx942_kernel_0/rebuilt.co \
  asm_experiments/silotiger957/fmha_varlen_gfx942_kernel_0/21_final_isa.s
```

Build the isolated V-wait treatment:

```bash
python3 asm_experiments/silotiger957/make_v_wait_overlap.py \
  asm_experiments/silotiger957/fmha_varlen_gfx942_kernel_0/21_final_isa.s \
  asm_experiments/silotiger957/fmha_varlen_gfx942_kernel_0/v_wait_hazard_safe.s
/opt/rocm/llvm/bin/clang -x assembler \
  -target amdgcn-amd-amdhsa -mcpu=gfx942 \
  -o asm_experiments/silotiger957/fmha_varlen_gfx942_kernel_0/v_wait_hazard_safe.co \
  asm_experiments/silotiger957/fmha_varlen_gfx942_kernel_0/v_wait_hazard_safe.s
```

Run a direct code-object A/B:

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. \
python3 op_tests/flydsl_tests/bench_fmha_gfx942_hsaco.py \
  --co asm_experiments/silotiger957/fmha_varlen_gfx942_kernel_0/v_wait_hazard_safe.co \
  --reference-co asm_experiments/silotiger957/fmha_varlen_gfx942_kernel_0/rebuilt.co
```

## Results

- Reassembled baseline: output and LSE byte-identical to FlyDSL.
- Baseline round trip: rebuilt/FlyDSL median `0.9972`; production ASM
  `3811.7 us`, rebuilt `4407.8 us`.
- V-wait address-overlap treatment: output and LSE byte-identical on both
  `Sq=256, Sk=300` (partial KV tile) and the ticket shape.
- Direct candidate/reference A/B: `4411.1 / 4410.5 us = 1.0001`; reject.
- Moving the mask compares as well failed correctness. Moving the mask base
  advance before its compare also failed correctness. These variants remain
  selectable in `make_v_wait_overlap.py` for diagnosis, not deployment.

### Structural FlyDSL treatments

- Q-operand AGPR pinning used one multi-output LLVM inline-asm anchor so all 48
  Q registers remained in `a[0:47]`; all 48 GEMM1 MFMAs consumed AGPRs directly
  with no `v_accvgpr_read/write` copies. Output and LSE were byte-identical on
  `Sq=256, Sk=300` and `Sq=4096, Sk=42700`. The target A/B was
  `4410.2 / 4406.5 us = 1.0008`; reject and revert.
- Two-KV-tile source unrolling reproduced the expanded static loop shape:
  160 MFMAs versus 80 in baseline, with 232 total registers. It did not
  reproduce production's low-wait schedule: 109 static waits versus 31 in
  production. Output and LSE remained byte-identical, but the target A/B was
  `5075.5 / 4405.1 us = 1.1522`; reject and revert.
- Full-QK residency staged all six K planes in 45 KiB LDS and pinned 48 Q plus
  96 K registers in AGPRs. This produced direct AGPR MFMA inputs with no
  `v_accvgpr_read` copies, 344 combined registers, and 40 waits, but occupancy
  fell to one wave/SIMD. It was byte-identical and measured
  `7596.6 / 4409.8 us = 1.7227`; reject and revert.
- Coupling full-QK residency with two-tile unrolling emitted 160 MFMAs, 144
  AGPRs, 45 KiB LDS, and 66 waits. It remained byte-identical but measured
  `7111.7 / 4405.2 us = 1.6144`; reject and revert. Matching production's
  resource counts and static expansion does not make LLVM reproduce its
  instruction-level overlap.
- Per-plane K-only AGPR residency reused 16 AGPRs for each eight-MFMA burst
  while leaving Q in VGPRs. It preserved the 27 KiB LDS footprint and reduced
  static waits from 61 to 45, with direct `a[...]` K operands and no
  `v_accvgpr_read/write` copies. It did not reduce the six LDS reads per plane:
  output and LSE were byte-identical, but the target A/B was
  `4528.6 / 4415.0 us = 1.0257`; reject and revert.
- True two-group temporal specialization used eight waves at `BLOCK_M=256`,
  two group-local V buffers (45 KiB LDS), 242 VGPRs, and conditional barrier
  staggering so one group executed the per-plane K-memory cluster while the
  other executed its MFMA cluster. Output and LSE were byte-identical. The
  unstaggered eight-wave control measured `5287.3 / 4400.4 us = 1.2015`; the
  phase-staggered treatment measured `5526.4 / 4407.6 us = 1.2538`. Halving
  the ticket grid from 384 to 192 CTAs loses more CU/XCD parallelism than two
  waves/SIMD recover, and the phase shift adds a further regression; reject
  and revert.
- A BF16 ones-column denominator reduction replaced 32 FP32 additions with
  eight `v_mfma_f32_32x32x8_bf16` instructions. Static MFMAs rose from 80 to
  88, FP32 adds fell from 33 to 1, and VGPRs fell from 242 to 222. Accuracy
  passed (`7.93e-6` output cosine error, zero tolerance failures, `0.00288`
  maximum LSE error), but the eight dependent MFMAs serialized the denominator:
  `5714.3 / 4419.5 us = 1.2930`; reject and revert.
- A wave-uniform last-tile guard dynamically skipped all 32 noncausal padding
  masks on 667 of 668 target KV tiles. Output and LSE were byte-identical and
  VGPRs fell from 242 to 232, but the changed loop CFG/schedule measured
  `5004.4 / 4421.0 us = 1.1320`; reject and revert. The ATT mask attribution
  was hidden rather than additive elapsed time. Reconsider only as part of a
  separately generated asymmetric pipeline, not as a presumed prerequisite.
- The first correctness-gated anti-phase implementation used one eight-wave
  workgroup with two four-wave cohorts, double-buffered V, and a one-phase
  barrier skew. It passed the ticket gate (`1.325e-5` cosine difference,
  zero tolerance failures) with 238 VGPRs, no spills, and 46,080 B LDS, but
  measured `6196.1 / 4409.5 us = 1.4052` versus FlyDSL and `3766.3 us` for
  production ASM. Its ISA has 42 static barriers, 107 waits, and no
  `s_setprio`, versus 8 barriers and 61 waits in the baseline. This rejects
  commit `4ff1b8187`'s six-rendezvous-per-phase schedule, not asymmetric
  wave specialization itself; the synchronization tax overwhelms overlap.
- A lower-synchronization asymmetric treatment used group-private K/V LDS,
  three-plane K prefetch, delayed PV, four full-phase rendezvous, and
  `s_setprio` phase swaps. It passed boundary and target correctness
  (`1.11e-5` target cosine difference) with 242 VGPRs, no spills, and
  64,512 B LDS. Target latency was `4678.3 / 4424.7 us = 1.0573`, with
  production ASM at `3804.3 us`. Commit `72bf70d66` is the retained
  experimental base for coupled mixed-exp plus MFMA-denominator testing;
  its exact-softmax result alone does not reject the coupled treatment.
- The coupled asymmetric treatment kept 12 Schraudolph high-half
  exponentials and replaced the denominator with four independent
  two-MFMA chains. It passed boundary and target correctness (`6.88e-5`
  target cosine difference), but measured `4726.6 / 4675.1 us = 1.0110`
  against exact ping-pong, with the four-wave baseline at `4412.2 us`.
  Its ISA removed 12 exponentials but added eight MFMAs and eight VGPRs
  (250 versus 242), so matrix-pipe contention outweighed the shorter VALU
  sequence. Reject the coupled numerical mode for performance; preserve
  exact ping-pong in dedicated commit `c091bd08a`.
- Replacing six of the eight steady-state full-workgroup barriers with
  cohort-local LDS atomic counters met its synchronization target without
  deadlock: two full barriers per tile, three `ds_add_u32`, six polling
  `ds_read_b32`, and three `s_sleep`. Correctness passed, but resources rose
  from 242/38 to 250/40 VGPR/SGPR and target latency measured
  `4828.6 / 4668.9 us = 1.0342` against exact ping-pong. Reject and restore;
  LDS polling costs more than the removed cross-cohort barrier stalls.
- Redistributing delayed PV across the four matrix rendezvous achieved the
  intended `16/24/24/16` MFMA bursts with 236 VGPRs and unchanged arithmetic,
  barriers, and LDS. Correctness passed, but target latency measured
  `5244.5 / 4665.6 us = 1.1241` against exact ping-pong. Equal MFMA counts
  did not equalize elapsed phases: moving PV exposed V-LDS reads and MFMA
  work behind later rendezvous on the critical path. Reject and restore.

The production ASM is not a locally rescheduled version of the FlyDSL kernel:
it uses a 512-entry combined VGPR/AGPR allocation and 64 KiB LDS versus
FlyDSL's 242 registers and 27 KiB LDS, and its static body contains 216 MFMA
instructions and 31 waits versus 80 MFMA and 61 waits. The next viable
treatment would need explicit instruction-level control over full K/V operand
residency and next-tile DMA/MFMA interleaving. FlyDSL source-level AGPR anchors
can express register class and lifetime, but the current LLVM scheduler does
not reproduce production's overlap. Neither local scheduling, AGPR pinning,
full-K staging, nor loop unrolling closes the remaining ~15% gap.
