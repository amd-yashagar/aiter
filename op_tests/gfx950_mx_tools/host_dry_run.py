"""Host-build (no GPU) every FlyDSL kernel a diagnose run reaches: the same inputs as
op_tests.moe_bench.diagnose, replayed through fused_moe inside aiter's dry_run
(FakeTensors, COMPILE_ONLY=1), so the cache keys are the runtime's own.

  PYTHONPATH=<tree> t_dry.py CASES.csv ID [ID ...]
"""
import sys


def main():
    import torch

    from aiter.aot.flydsl.dry_run import dry_run
    from aiter.aot.flydsl.moe import _without_device_queries
    from op_tests.moe_bench import accuracy, schema, workload

    cases = {c.id: c for c in schema.load_tier(sys.argv[1])}
    # Values never matter for the build keys: CUDA generators (which need a device)
    # become dropped arguments.
    real_gen = torch.Generator
    torch.Generator = lambda device=None: real_gen()
    for name in ("randn", "rand", "randint", "randperm", "normal"):
        real = getattr(torch, name)

        def wrap(*a, _real=real, **k):
            k.pop("generator", None)
            return _real(*a, **k)

        setattr(torch, name, wrap)
    bad = 0
    for cid in sys.argv[2:]:
        case = cases[cid]
        try:
            with dry_run("gfx950", 256), _without_device_queries():
                import aiter.fused_moe as fm

                S = workload.resolve(case)
                M = min(case.token, 16)
                aw, rw, bias = workload.build_group(S)
                hs, tw, ti = accuracy.acc_inputs(S, M, 0)
                mask = workload.make_expert_mask(S) if S.is_ep else None
                kw = workload.fused_kwargs(S, aw, bias, mask)
                # fused_moe is a custom op whose fake runs under dry_run; the host
                # code that picks and compiles kernels is _fused_moe_impl
                import inspect

                params = inspect.signature(fm._fused_moe_impl).parameters
                args = {
                    k: (v.value if k in ("activation", "quant_type", "quant_type_a", "gate_mode") and hasattr(v, "value") else v)
                    for k, v in kw.items()
                    if k in params
                }
                dropped = sorted(set(kw) - set(params))
                if dropped:
                    print(f"dry {cid}: kwargs not in _fused_moe_impl: {dropped}", flush=True)
                fm._fused_moe_impl(hs.clone(), aw["w1"], aw["w2"], tw, ti, **args)
            print(f"dry {cid}: ok", flush=True)
        except Exception as exc:  # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"dry {cid}: FAILED {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
