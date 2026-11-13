"""Which top-k number is the top-k's cost inside the gluon call's graph?

Three candidates, all with the harness's Timer (one HIP event pair around one graph replay):
  alone     the top-k kernel as a graph of its own (what the harness subtracted: carries the
            per-replay launch latency that the whole call pays once, whatever it contains)
  slope     (graph of N top-k calls - graph of 1) / (N - 1): the kernel plus its inter-kernel
            gap, no launch latency
  removed   routing() minus the same routing() fed the top-k outputs of a precomputed call
            (sort kernels only): the direct difference the subtraction wants

    python -m op_tests.moe_bench.validation.topk_cost_study   (any arch: the kernels are Triton)
"""

import json
import sys

import torch


def main(argv=None):
    from aiter.ops.triton.moe.moe_routing import routing as R
    from aiter.ops.triton.moe.moe_routing.topk import topk
    from op_tests.moe_bench.timing import Timer

    E, K, N = 128, 4, 9
    out = []
    for M in (1, 16, 64, 256, 4096):
        gen = torch.Generator(device="cuda").manual_seed(M)
        probs = torch.softmax(torch.randn(M, E, device="cuda", generator=gen), dim=-1)
        tw, ti = torch.topk(probs, K, dim=-1)
        tw = tw / tw.sum(-1, keepdim=True)
        lg = torch.full((M, E), -1e4, dtype=torch.float16, device="cuda")
        lg.scatter_(1, ti, torch.log(tw).to(torch.float16))

        def tk(copies=1, lg=lg):
            def run():
                for _ in range(copies):
                    r = topk(lg, K, apply_softmax=True, HIST_BLOCK_M=32)
                return r

            return run

        def whole(lg=lg):
            return R.routing(lg, K)

        scal, indx, bm = topk(lg, K, apply_softmax=True, HIST_BLOCK_M=32)
        direct = M <= 256  # reusing one bitmatrix for the sort alone faults at large M
        block_m = max(16, min(1 << (max(1, M * K // E) - 1).bit_length(), 128))
        hb = 32 if M > 16 else 1 << (M - 1).bit_length()
        sort_fn = R.sort_tokens_fused if M <= 16 else R.sort_tokens

        def sort_only(
            scal=scal, indx=indx, bm=bm, sort_fn=sort_fn, block_m=block_m, hb=hb
        ):
            return sort_fn(scal, indx, E, bm, block_m, hb)

        def t(fn, cold=False, **kw):
            tm = Timer(fn)
            tm.prepare_graph()
            return tm.time(cold=cold, **kw)["median_us"], tm.mode

        warm1, mode = t(tk(1))
        warmn, _ = t(tk(N))
        cold1, _ = t(tk(1), cold=True)
        if direct:
            w_whole, _ = t(whole)
            w_sort, _ = t(sort_only)
            c_whole, _ = t(whole, cold=True)
            c_sort, _ = t(sort_only, cold=True)
        row = {
            "M": M,
            "mode": mode,
            "warm_alone": round(warm1, 2),
            "warm_slope": round((warmn - warm1) / (N - 1), 2),
            "warm_removed": round(w_whole - w_sort, 2) if direct else None,
            "cold_alone": round(cold1, 2),
            "cold_slope_plus_penalty": round(
                (warmn - warm1) / (N - 1) + max(cold1 - warm1, 0), 2
            ),
            "cold_removed": round(c_whole - c_sort, 2) if direct else None,
        }
        print(json.dumps(row), flush=True)
        out.append(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
