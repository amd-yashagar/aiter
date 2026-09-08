# gfx942 FMHA ping-pong ISA ledger

Contract: non-causal packed-THD FMHA, bf16, H=12, QK=192, V=128,
eight-wave ping-pong, `return_lse=True`, gfx942. Ticket shape
`Sq=4096, Sk=42700`. Cosine vs production ASM must stay `< 1e-4`.
Timing is 4–8 interleaved rounds of 21 CUDA-event iterations.

The kept kernel is `aiter/ops/flydsl/kernels/fmha_gfx942/fmha_pingpong_kernel.py`.
Do not apply the scripts below to that source; they document rejected
listings. Rebuild the control with:

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. \
python3 op_tests/flydsl_tests/bench_fmha_gfx942_pingpong_hsaco.py \
  --co candidate.co --reference-co rebuilt.co --gate-asm-cosine
```

## Rejected treatments (2026-09-08)

| Treatment | Script | vs rebuilt listing |
|---|---|---|
| Weave O-rescale `v_pk_mul` into exp-hazard `s_nop`s | `weave_o_rescale_into_exp_nops.py` | PASS cosine, **+4.18%** |
| Spread next-K `buffer_load … lds` across those nops | `spread_nextk_load_lds_into_exp_nops.py` | PASS cosine, **+3.56%** |
| Skip O-rescale when `corr == 1` | `skip_o_rescale_when_corr_one.py` | PASS cosine, **−0.49%** (not a 2% win) |
| Drain next-K `vmcnt(0)` after the close barrier | FlyDSL edit, reverted | PASS cosine, within noise of 4253 µs |

Next-K DMA during softmax writes this group's K LDS while the partner
`ds_read`s on the same CU. Filling TRANS-hazard nops with packed VALU
fights the partner MFMA. Random `randn` scores keep raising the row max,
so a `corr == 1` skip almost never fires wave-wide.
