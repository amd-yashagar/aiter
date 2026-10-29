# Node quant-k08-325 work inventory (wrap-up 2026-10-29)

Node: 8x gfx942 (MI325X). Workspace `/workspaces/quant-k08-325-WS`.
Everything that existed only on this node was pushed as `wip/k08-325/*` branches to the
fork `git@github.com:amd-yashagar/aiter.git`, or saved as git bundles where push access was missing.
No local branch, worktree, or file was changed by the wrap-up.

## Already upstream (nothing to do)

| Work | Where |
|---|---|
| gfx942 FMHA hd192 split asm | PR [ROCm/aiter#5592](https://github.com/ROCm/aiter/pull/5592), branch `feat/gfx942-fmha-hd192-split-asm` @ `d7c3e42ca` |
| MLA PFL bf16 CSV causal flag | PR [ROCm/aiter#5211](https://github.com/ROCm/aiter/pull/5211), branch `fix/mla-pfl-bf16-csv-causal-flag` @ `629a87752` |
| GLM5.3 sparse MLA (pushed part) | `origin/ws/glm53-sparse-mla` @ `778ab353d`, no PR, Jira SILOTIGER-1215 |
| DSv4 Pro sparse MLA prefill (Sami) | `origin/samremes/gfx942-sparse-mla-prefill` @ `36b763bc8` |
| DSv4 Flash tip | `yashagar/samremes/gfx942-sparse-mla-prefill` @ `071616d8e` (fork) |
| FlyDSL, WaveScope, PyISA, flylens, ruka-dashboard, flydsl-lab, TransferBench, amd-gpu-agent-kit | clean and synced with their remotes |

## Preserved during wrap-up (fork `amd-yashagar/aiter`)

| Branch | Content |
|---|---|
| `wip/k08-325/glm53-sparse-mla-qblock-dpp` | `778ab353d` + uncommitted `sparse_mla_qblock.py` (DPP softmax, P in registers, `ds_bpermute`, next-16-KV prefetch; random one-query ~7229 us, shared q-block ~3898 us). Not re-validated. |
| `wip/k08-325/glm53-sparse-mla-prerebase-stash` | The only aiter stash: "pre-rebase gfx942 FlyDSL sparse MLA 2026-10-05" (`ops/flydsl/__init__.py`, `triton/attention/pa_decode_sparse.py`, `sparse_mla.py`). Likely superseded by `778ab353d`; kept for safety. |
| `wip/k08-325/dsv4-unified-delivery` | Local branch `ws/dsv4-unified-delivery` (13 commits never pushed, tip `ce0c52f0b`) + 4 uncommitted Pro+Flash coexistence files + its `HANDOVER.md` (SILOTIGER-613). See `dsv4/HANDOVER.md` here. |
| `feat/flydsl-gfx942-fmha-hd192x128` | Fast-forwarded by 1 local commit `234c3664a` "WIP: preserve gfx942 FMHA experiments" (16 files). |
| `wip/k08-325/node-inventory` | This document and the handover notes (orphan, docs only). |

## Saved as bundles (no push access from this node)

Saved to the personal agent store `.../files/k08-325-bundles/`. Restore with
`git fetch <bundle> <branch>:<branch>` inside a clone of the repo.

| Repo | Why not pushed | Bundle content |
|---|---|---|
| `AMD-SiloASM/poc_kl` | SSH key gets "repository not found" | `feat/gfx942-fmha-hd192-split-asm`, 2 unpushed commits (`84b6033` hd192 split-K FMHA shader, `9835e16` split-kv naming refactor); full-history bundle |
| `AMD-ROCm-Internal/quilt` | no push rights | `wip/k08-325-snapshot` = `origin/main` `3e86678` + uncommitted gfx942 FMHA hd192 example (`examples/fmha/`, `tests/test_fmha_hd192.py`) and changes to `arch/kernel/mfma/convert` (~+2114/-24) |

## Deliberately not preserved

- `gpucore.*` GPU core dumps (workstreams/aiter ~2.6 GB, quilt ~570 MB).
- `rocm-systems`: only submodule pointer drift and a local `install/` build output.
- `amd-gpu-agent-kit/.cursor/rules`: generated graphify rule.
- Local review checkouts of other people's PRs: `pr-4641-review` (Aakif Nawaz, MoE SwiGLU FP8), `pr-5024` (Hui Shi, CK_TILE MHA tuning), worktrees `/tmp/aiter-pr-4641`, and `/tmp/aiter-flydsl-sparse-donor` (already on origin user branches).
- Non-git data in the workspace: `GLM5.3Flash/` (140 MB prefill traces), `Mehdi/` (15 MB Kimi-K3 serving logs), `.wavescope/` (438 MB ATT captures), `DSv4/deepseek-v4-pro-uplift-plan-tp8.html`. Copy them off manually if needed.

## Handovers

- `kimik3-a4w4-moe-accuracy.md`: diagnosis only, no code.
- `glm53/HANDOVER.md`, `glm53/glm53-flash-sparse-mla-prefill-gfx942.md`: GLM5.3 notes (were untracked `docs/issues/`).
- `dsv4/HANDOVER.md`: DSv4 Pro+Flash.
