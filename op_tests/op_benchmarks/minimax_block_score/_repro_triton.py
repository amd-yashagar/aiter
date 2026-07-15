import sys, time
import torch
from triton_block_score import triton_block_score

torch.manual_seed(0)
dev = "cuda"


def build(context, total_q=8192, D=128, bs_k=128):
    prefix = context - total_q
    assert prefix >= 0
    q = torch.randn(total_q, 1, D, device=dev, dtype=torch.bfloat16)
    max_slots = context
    k_cache = torch.randn(max_slots, 1, D, device=dev, dtype=torch.bfloat16)
    req_to_token = torch.arange(context, device=dev, dtype=torch.int32).view(1, context)
    slot_ids = torch.zeros(1, device=dev, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, total_q], device=dev, dtype=torch.int32)
    seq_lens = torch.tensor([context], device=dev, dtype=torch.int32)
    prefix_lens = torch.tensor([prefix], device=dev, dtype=torch.int32)
    return dict(
        q=q, k_cache=k_cache, req_to_token=req_to_token, slot_ids=slot_ids,
        cu_seqlens=cu_seqlens, seq_lens=seq_lens, prefix_lens=prefix_lens,
        max_seqlen_q=total_q, max_seqlen_k=context, block_size_k=bs_k,
    )


def naive_ref(inp, score_type="max"):
    q = inp["q"][:, 0, :].float()  # [Q, D]
    ctx = inp["max_seqlen_k"]
    prefix = int(inp["prefix_lens"][0].item())
    k = inp["k_cache"][:ctx, 0, :].float()  # identity paging
    Q = q.shape[0]
    sm = (q.shape[1] ** -0.5)
    log2e = 1.4426950409
    nblk = (ctx + 127) // 128
    out = torch.full((Q, nblk), float("-inf"), device=dev)
    qk = (q @ k.t()) * sm * log2e  # [Q, ctx]  (log2 domain, matches kernel)
    qpos = torch.arange(Q, device=dev) + prefix
    kpos = torch.arange(ctx, device=dev)
    causal = qpos[:, None] >= kpos[None, :]
    qk = torch.where(causal, qk, float("-inf"))
    for b in range(nblk):
        lo, hi = b * 128, min((b + 1) * 128, ctx)
        seg = qk[:, lo:hi]
        smax = seg.max(dim=1).values
        if score_type == "max":
            out[:, b] = smax
        else:
            lse = smax + torch.log2(torch.exp2(seg - smax[:, None]).sum(dim=1))
            lse = torch.where(lse != lse, float("-inf"), lse)
            out[:, b] = lse
    return out


def time_kernel(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts)//2], ts[len(ts)//20], ts[-max(1,len(ts)//20)]


for context in (40960, 81920, 131072):
    inp = build(context)
    score = torch.full((1, 8192, (context + 127)//128), float("-inf"), device=dev, dtype=torch.float32)
    def fn():
        return triton_block_score(score=score, score_type="max", **inp)
    fn()  # warm+autotune
    torch.cuda.synchronize()
    # correctness vs naive
    ref = naive_ref(inp, "max")
    got = score[0]
    m = ref != float("-inf")
    maxabs = (got[m] - ref[m]).abs().max().item()
    med, p5, p95 = time_kernel(fn)
    print(f"ctx={context:7d}  med={med:8.1f}us  p5={p5:8.1f}  p95={p95:8.1f}  maxabs={maxabs:.4f}", flush=True)
