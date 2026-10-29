# Handover — DSv4 Pro+Flash sparse MLA prefill (SILOTIGER-613)

**Date:** 2026-10-29 (session work originally ~2026-08-19)  
**Ticket:** [SILOTIGER-613](https://amd.atlassian.net/browse/SILOTIGER-613)  
**Verdict:** **Not upstreamed.** No PR opened from this work. Critical fixes are **local uncommitted** dirty files only.

## Where things live

| What | Location | Remote? |
|---|---|---|
| Pro tip (Sami) | `workstreams/dsv4-pro-sami` @ `36b763bc8` | `origin/samremes/gfx942-sparse-mla-prefill` |
| Flash tip (Yash) | `yashagar/samremes/gfx942-sparse-mla-prefill` @ `071616d8e` | fork only (num_q_heads + T≥32k) |
| Unified + dual-path fixes | `workstreams/dsv4-flash-yashagar` branch `ws/dsv4-unified-delivery` @ `ce0c52f0b` + **4 dirty files** | **local only — never pushed** |
| `origin/main` | — | **does not** contain `sparse_mla_prefill*` |

`ws/dsv4-unified-delivery` = Flash tip rebased onto `origin/main` (of that era) + uncommitted dual-path patch.

## Uncommitted files (must not lose)

```
aiter/ops/flydsl/kernels/sparse_mla_prefill.py      # dual-path address: Pro i32 packed + Flash i64 flat512
aiter/ops/flydsl/sparse_mla_prefill_kernels.py      # flat512 resolve, block strides, per_tensor scales
aiter/ops/flydsl/kernels/buffer_ops.py              # always emit aux IntegerAttr
op_tests/flydsl_tests/test_sparse_mla_prefill.py    # flat512 padded / oracle / empty-r1 tests
```

Diffstat ~+509/−72. Full suite was green on gfx942 (Pro H=128, Flash H=8, GLM, padded phys×stride overflow).

## Design to preserve (Pro + Flash coexistence)

1. **Pro packed `fp8_ds_mla` (H=128):** global buffer rsrc + **i32** token voffsets (hot path / perf).
2. **Flash `dsv4_flat512` (H=8):** i64 `phys * host_block_stride` → `create_buffer_resource_from_addr` per token (fixes padded-pool int32 overflow / GSM8K-style faults). Do **not** flatten padded pools to 1D shapes that exceed int32 memref dims.
3. **GLM flat576:** same Pro-style packed i32 path (`_packed_token_base_i32`); avoid i64+i32 `arith.addi`.
4. Sink: mask heads `>= NUM_Q_HEADS` for Flash H=8.

## Not done

- Commit / push / PR
- E2E GSM8K / serving perf remeasure (unit tests only)
- Merge into `origin/main`

## Related (do not confuse)

- `workstreams/aiter` on `ws/glm53-sparse-mla` has `sparse_mla_qblock.py` — **GLM-5.3** path, not this DSv4 dual-path delivery.
- Featherless overlay / enablement docs under `/tmp/featherless-docs/featherless-harness/` (vLLM wiring), separate from aiter commit.

## Next agent checklist

1. `cd workstreams/dsv4-flash-yashagar && git status` — confirm 4 dirty files still present.
2. Rebase/refresh onto current `origin/main` if needed; re-run suite (stub aiter via package shells; broken `triton` on py3.14 may need rename).
3. Commit only if user asks; push branch + open PR against ROCm/aiter when ready.
