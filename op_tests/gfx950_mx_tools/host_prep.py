"""Host-compile every gfx950_mx kernel a lease job needs (run under hostcc*.sh, no GPU).

  t_prep.py CASE:KERNEL:CFG [CASE:KERNEL:CFG ...]

Fills FLYDSL_RUNTIME_CACHE_DIR so the lease runs with FLYDSL_RUNTIME_RUN_ONLY=1.
PREP_ROUTES=<routes csv or kernel id> also compiles what aiter.moe's compile_all warms:
every matching route row's kernel at the row's m_min (a bare kernel id is one row, m_min 0).
PREP_ALSO_M=m,... adds more M values for the job's own kernel.
KERNEL is the id suffix after gfx950_mx_ (a8w4.decode, mxfp8.prefill, ...).
"""

import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bar_profile as bp  # noqa: E402
from moe_bench import workload  # noqa: E402


class FakeW1:
    def __init__(self, interleave):
        self.gate_up_interleave = interleave


class FakeWeights:
    def __init__(self, spec, kw, interleave):
        self.spec = spec
        self.fused_moe_kwargs = kw
        self.w1 = FakeW1(interleave)

    def kwarg(self, name, default=None):
        v = self.fused_moe_kwargs.get(name)
        return default if v is None else v


def dt(name):
    from aiter import dtypes

    return {
        "fp8": dtypes.fp8,  # without a GPU aiter may pick fnuz; the kernels key on "fp8"
        "fp4x2": dtypes.fp4x2,
        "bf16": torch.bfloat16,
    }[name]


def prep(item):
    import aiter.moe as am
    from aiter.moe.spec import MoeSpec
    from aiter.ops.flydsl.moe_common import GateMode

    import aiter.ops.flydsl.moe_next.gfx950_mx  # noqa: F401  (registers the kernels)

    cid, kid, cfg = item.split(":", 2)
    c = bp.load_case(cid)
    S = workload.resolve(c)
    aq, wq = dt(c.row["q_dtype_a"]), dt(c.row["q_dtype_w"])
    gm = GateMode(S.gate_mode)
    spec = MoeSpec(
        experts=S.E_local,
        global_experts=S.E,
        model_dim=S.D,
        inter_dim=S.I,
        act=S.act,
        quant=S.qtype,
        w_dtype=wq,
        out_dtype=None,
        g1u1=True,
        doweight_stage1=bool(S.doweight),
        gate_mode=gm,
        w_shuffled=True,
        has_bias=bool(S.bias),
        swiglu_limit=S.swiglu_limit,
        hidden_pad=int(getattr(S, "hidden_pad", 0) or 0),
        intermediate_pad=int(getattr(S, "intermediate_pad", 0) or 0),
    )
    kw = {"quant_dtype_a": aq}
    if S.beta is not None:
        kw["beta"] = S.beta
    if S.linear_beta is not None:
        kw["linear_beta"] = S.linear_beta
    # the harness tags only the fp8-activation shuffles (gate/up interleaved)
    from aiter import dtypes

    interleave = True if aq == dtypes.fp8 else None
    k = am.registry.get_kernel(f"gfx950_mx_{kid}")
    from aiter.ops.flydsl.moe_next.gfx950_mx import activation_dtype

    fw = FakeWeights(spec, kw, interleave)
    if activation_dtype(spec, fw, fw.w1, S.M, torch.bfloat16, False) not in (
        dtypes.fp8,
        dtypes.fp4x2,
    ):
        raise RuntimeError("not an MX activation dtype")
    t0 = time.time()
    # the decode kernel's tiles path sorts with core's route_to_tiles (FlyDSL too);
    # its build key is dtypes, ranks and constexprs, so CPU tensors of the shape do
    if not k.prefill:
        bm, _, _, _, _, r, direct = k._kernels(spec, fw, S.M, S.topk, aq, cfg)
        if not direct[0]:
            from aiter.moe import compile_only
            from aiter.ops.flydsl.moe_next.core.tile_map import route_to_tiles

            ti = torch.zeros((S.M, S.topk), dtype=torch.int32)
            tw = torch.zeros((S.M, S.topk), dtype=torch.float32)
            out = torch.empty((S.M, S.D), dtype=torch.bfloat16)
            cs, torch.cuda.current_stream = torch.cuda.current_stream, lambda *a: None
            with compile_only():
                route_to_tiles(
                    ti, tw, num_experts=S.E_local, tile_m=bm,
                    zero_output=None if r is not None else out,
                )
            torch.cuda.current_stream = cs
    extra = [int(v) for v in os.environ.get("PREP_ALSO_M", "").split(",") if v]
    for m in [S.M, *extra]:
        k.warmup(fw, m, S.topk, cfg)
    # aiter.moe's compile_all warms every route row of the family at its m_min,
    # with that row's kernel and config (not only the kernel serving this M)
    routes = os.environ.get("PREP_ROUTES")
    if routes:
        from aiter.moe.routes import RouteTable

        fam = c.row["family"]
        for row in RouteTable.override(routes).rows:
            if row.family not in ("*", fam) or not row.matches_shape(spec, S.topk):
                continue
            rk = am.registry.get_kernel(row.route.kernel)
            if rk is None or not row.route.kernel.startswith("gfx950_mx"):
                continue
            rk.warmup(fw, row.m_min, S.topk, row.route.config)
    print(f"prep {item}: {time.time() - t0:.1f}s", flush=True)


def main():
    bad = 0
    for item in sys.argv[1:]:
        try:
            prep(item)
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"prep {item}: FAILED {exc!r}"[:400], flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
