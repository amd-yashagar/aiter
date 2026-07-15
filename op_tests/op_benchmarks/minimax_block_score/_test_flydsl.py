import sys
import torch
from triton_block_score import triton_block_score
from aiter.ops.flydsl.kernels.minimax_block_score import flydsl_minimax_block_score

torch.manual_seed(0)
dev = "cuda"


def build(context, total_q, D=128, bs_k=128):
    prefix = context - total_q
    assert prefix >= 0
    q = torch.randn(total_q, 1, D, device=dev, dtype=torch.bfloat16)
    k_cache = torch.randn(context, 1, D, device=dev, dtype=torch.bfloat16)
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
    q = inp["q"][:, 0, :].float()
    ctx = inp["max_seqlen_k"]
    prefix = int(inp["prefix_lens"][0].item())
    k = inp["k_cache"][:ctx, 0, :].float()
    Q = q.shape[0]
    sm = (q.shape[1] ** -0.5) * 1.4426950409
    nblk = (ctx + 127) // 128
    out = torch.full((Q, nblk), float("-inf"), device=dev)
    qk = (q @ k.t()) * sm
    qpos = torch.arange(Q, device=dev) + prefix
    kpos = torch.arange(ctx, device=dev)
    qk = torch.where(qpos[:, None] >= kpos[None, :], qk, float("-inf"))
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
    # entirely-future blocks stay -inf; but rows where every key masked -> -inf already
    return out


def check(context, total_q, score_type, bq=64, kg=8):
    inp = build(context, total_q)
    ref = naive_ref(inp, score_type)
    # triton
    tri = triton_block_score(score_type=score_type, **inp)[0]
    # flydsl
    fly = flydsl_minimax_block_score(score_type=score_type, block_q=bq, k_group=kg, **inp)[0]
    m = ref != float("-inf")
    tri_maxabs = (tri[m] - ref[m]).abs().max().item()
    fly_maxabs = (fly[m] - ref[m]).abs().max().item()
    # mask parity
    fly_mask_ok = torch.equal(fly == float("-inf"), ref == float("-inf"))
    tri_mask_ok = torch.equal(tri == float("-inf"), ref == float("-inf"))
    denom = ref[m].abs().clamp(min=1e-3)
    fly_relmax = ((fly[m]-ref[m]).abs()/denom).max().item()
    print(f"ctx={context} Q={total_q} {score_type}: "
          f"tri_maxabs={tri_maxabs:.4f}(mask={tri_mask_ok}) "
          f"fly_maxabs={fly_maxabs:.4f} fly_relmax={fly_relmax:.4f}(mask={fly_mask_ok})",
          flush=True)
    return fly_maxabs, fly_mask_ok


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "max"
    check(512, 64, st)
    check(512, 128, st)
    check(1024, 128, st)
    check(768, 192, st)
