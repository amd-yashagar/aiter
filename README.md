# Node quant-k08-325 handover: start here

Everything from node quant-k08-325 (8x MI325X, gfx942, ROCm 7.14, Python 3.14, Ubuntu 26.04),
packed on 2026-10-29 when the node was wrapped up. This folder is self-contained: code that only
existed on the node is in `git-bundles/` and `patches/`, so nothing depends on the node surviving.

## For the next agent

1. Read `docs/NODE_INVENTORY.md`. It lists every workstream, where its code lives, and its state.
2. Read the handover for the workstream you are continuing:
   - Kimi-K3 a4w4 MoE accuracy (gfx950): `docs/kimik3-a4w4-moe-accuracy.md`
   - GLM-5.3 Flash sparse MLA on gfx942 (SILOTIGER-1215): `docs/glm53/`
   - DSv4 Pro+Flash sparse MLA prefill (SILOTIGER-613): `docs/dsv4/HANDOVER.md`
   - gfx942 FMHA hd192 split-K (PR #5592) and the MLA PFL CSV fix (PR #5211): their PRs, plus `chat-history/`
   - SILOTIGER-857 hipBLASLt/FlyDSL GEMM: `data/silotiger_857/` logs and `chat-history/` (source code was lost; see the inventory)
3. To get context from an earlier chat, use `chat-history/INDEX.md` to choose the transcript, then `rg` inside it.
   Treat transcripts and old handovers as claims to re-verify, not as facts.

## Layout

| Path | Content |
|---|---|
| `docs/` | Node inventory and the per-workstream handovers |
| `git-bundles/` | All unpushed git work, restorable offline (see below) |
| `patches/` | The same uncommitted work as plain `git diff` patches and tarballs of untracked files, for reading |
| `chat-history/` | All Cursor agent transcripts from this node (JSONL, with subagents) and `INDEX.md` |
| `data/` | Kimi-K3 serving logs, GLM5.3Flash prefill traces, DSv4 uplift plan, SILOTIGER-857 logs/scripts, WaveScope profile config |
| `tmp-scratch/` | Loose `/tmp` work: bench scripts, sparse-MLA prototypes, `.co`/`.s` FMHA experiments, ISA dumps, MHA tuning CSVs, `featherless-docs/` (vLLM wiring docs/overlays), merged aiter configs |
| `env/` | `pip-freeze.txt`, `system.txt`, `repos.psv` (every repo with remote, branch, HEAD) |

## Restoring the code

The fork `git@github.com:amd-yashagar/aiter.git` already holds the aiter branches (`wip/k08-325/*`,
`feat/flydsl-gfx942-fmha-hd192x128`). The bundles are an offline copy of the same branches.

```bash
# aiter: own branches, delta on top of public ROCm/aiter main
git clone git@github.com:ROCm/aiter.git && cd aiter
git fetch /path/git-bundles/aiter-own-branches.bundle 'refs/heads/*:refs/heads/*' 'refs/k08/*:refs/heads/*'

# poc_kl (no access from the node; full-history bundle)
git clone /path/git-bundles/poc_kl-feat-gfx942-fmha-hd192-split-asm.bundle poc_kl -b feat/gfx942-fmha-hd192-split-asm

# quilt (no push rights from the node; full-history bundle)
git clone /path/git-bundles/quilt-wip-k08-325-snapshot.bundle quilt -b wip/k08-325-snapshot
```

When you have access, push `poc_kl` and `quilt` from those clones to their real remotes
(`AMD-SiloASM/poc_kl`, `AMD-ROCm-Internal/quilt`).

## Environment notes

- aiter was installed editable from `/workspaces/quant-k08-325-WS/aiter` (`amd-aiter 0.1.22.dev19`), FlyDSL 0.3.2, triton 3.8.0.
- No vLLM or SGLang source was modified on the node. `/tmp/vllm-sparse/repo` was a clean upstream vLLM clone at `c961121`.
  The vLLM wiring for DSv4/GLM lives in `tmp-scratch/featherless-docs/featherless-harness/`.
- The Cursor kit used on the node is `amd-gpu-agent-kit` (fork `amd-yashagar/amd-gpu-agent-kit`, in sync).
