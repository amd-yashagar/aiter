# Handover: gfx942 sparse MLA prefill

Local note. Do not commit this file or `docs/issues/`.

## On the remote

Branch `ws/glm53-sparse-mla` on `git@github.com:ROCm/aiter.git`, even with origin at `778ab353d` (2026-10-05). No pull request was opened. Jira comment on SILOTIGER-1215 points at that commit: https://amd.atlassian.net/browse/SILOTIGER-1215

That commit is the FlyDSL path in `pa_decode_sparse` for packed bf16 rows, H=16, D=512. One-query and exact Q-block reuse. Numbers recorded with the commit: one-query 8.13 ms shared / 8.78 ms random, exact reuse 4.60 ms, T=16384, topk=2048, MI325X.

## Not on the remote

`aiter/ops/flydsl/kernels/sparse_mla_qblock.py` is modified and uncommitted. The working tree is a later kernel (DPP softmax, P kept in registers, `ds_bpermute`, prefetch of the next 16 KV rows). It is not `778ab353d`. Do not describe it as the pushed kernel until it is committed.

Untracked and out of any commit: `docs/issues/`, and `gpucore.*.gpu` dumps.
