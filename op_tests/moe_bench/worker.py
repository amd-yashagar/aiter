# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Subprocess worker: runs one weight group and appends JSON lines to a file.

Lines: {"event": "case_start", "case_id": ...} before each case and
{"event": "row", ...} per (case, mode) result. The parent kills this process on a
timeout or reads its exit code after a crash; the line marker says which case it
was in.
"""

import argparse
import functools
import json
import os
import sys
import time
import traceback
from pathlib import Path

UNSUPPORTED_FAMILIES_NON_GFX950 = {"a4w4", "a8w4", "mxfp8"}
PROBE_FAMILIES = {"a4w4", "a8w4", "mxfp8", "a16w4", "a16wi4", "a8wi4"}
SENTINEL = 12345.0


def emit(fh, obj):
    fh.write(json.dumps(obj) + "\n")
    fh.flush()


def _err(e):
    first = str(e).strip().splitlines()[0][:300] if str(e).strip() else ""
    return f"{type(e).__name__}: {first}"


def _is_unsupported(e, family, gfx):
    if gfx != "gfx950" and family in UNSUPPORTED_FAMILIES_NON_GFX950:
        return True
    return isinstance(e, NotImplementedError)


def _num_local(S, M, device="cuda"):
    import torch

    if not S.num_local_frac:
        return None, M
    n = max(1, int(M * S.num_local_frac))
    return torch.tensor([n], dtype=torch.int32, device=device), n


def run_fused_group(cases, modes_req, fh, timeout_info):
    import torch

    import aiter
    import aiter.fused_moe as fm
    from aiter.jit.utils.chip_info import get_gfx

    from . import routes, workload
    from .timing import Timer

    gfx = get_gfx()
    specs = {c.id: workload.resolve(c) for c in cases}
    live = []
    for c in cases:
        S = specs[c.id]
        r = routes.dry_route(c, S)
        if r.get("route_error"):
            for m in [m for m in c.modes if m in modes_req] or ["warm"]:
                emit(
                    fh,
                    {
                        "event": "row",
                        "case_id": c.id,
                        "mode": m,
                        "status": "unsupported",
                        "detail": r["route_error"],
                        **r,
                    },
                )
        else:
            live.append(c)
    if not live:
        return
    S0 = specs[live[0].id]
    family = live[0]["family"]
    need = workload.group_weight_bytes(S0) * 2.2
    free = workload.free_vram_bytes()
    if need > free * 0.92:
        for c in live:
            for m in [m for m in c.modes if m in modes_req] or ["warm"]:
                emit(
                    fh,
                    {
                        "event": "row",
                        "case_id": c.id,
                        "mode": m,
                        "status": "oom",
                        "detail": f"group needs about {need/2**30:.1f} GiB, "
                        f"{free/2**30:.1f} GiB free",
                    },
                )
        return

    if gfx != "gfx950" and family in PROBE_FAMILIES:
        emit(fh, {"event": "case_start", "case_id": live[0].id, "phase": "probe"})
        probe_err = None
        try:
            aw, _rw, bias = workload.build_group(S0, same_chunk=True)
            M = live[0].token
            hidden = workload.make_inputs(S0, M)
            tw, ti = workload.make_routing(S0, M)
            mask = workload.make_expert_mask(S0) if S0.is_ep else None
            kw = workload.fused_kwargs(S0, aw, bias, mask)
            aiter.fused_moe.fused_moe(hidden, aw["w1"], aw["w2"], tw, ti, **kw)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            probe_err = e
        finally:
            aw = _rw = None
            torch.cuda.empty_cache()
        if probe_err is not None and _is_unsupported(probe_err, family, gfx):
            for c in live:
                for m in [m for m in c.modes if m in modes_req] or ["warm"]:
                    emit(
                        fh,
                        {
                            "event": "row",
                            "case_id": c.id,
                            "mode": m,
                            "status": "unsupported",
                            "detail": _err(probe_err),
                            **routes.dry_route(c, specs[c.id]),
                        },
                    )
            return

    emit(fh, {"event": "case_start", "case_id": live[0].id, "phase": "weights"})
    t0 = time.time()
    aw, rw, bias = workload.build_group(S0)
    torch.cuda.synchronize()
    build_s = time.time() - t0
    emit(fh, {"event": "info", "phase": "weights", "seconds": build_s})

    inputs = {}
    refs = {}
    for c in live:
        S = specs[c.id]
        M = c.token
        emit(fh, {"event": "case_start", "case_id": c.id, "phase": "reference"})
        tw, ti = workload.make_routing(S, M)
        mask = workload.make_expert_mask(S) if S.is_ep else None
        hidden = workload.make_inputs(S, M)
        _, n_valid = _num_local(S, M)
        t1 = time.time()
        try:
            ref = workload.reference(S, rw, bias, hidden, tw, ti, mask, n_valid)
            refs[c.id] = (ref, time.time() - t1, None)
        except torch.OutOfMemoryError as e:
            refs[c.id] = (None, 0.0, f"oom: {_err(e)}")
        except Exception as e:  # noqa: BLE001
            refs[c.id] = (None, 0.0, f"reference error: {_err(e)}")
        inputs[c.id] = None
        torch.cuda.empty_cache()
    rw = None
    torch.cuda.empty_cache()

    for c in live:
        S = specs[c.id]
        M = c.token
        modes = [m for m in c.modes if m in modes_req]
        if not modes:
            continue
        emit(fh, {"event": "case_start", "case_id": c.id, "phase": "run"})
        tw, ti = workload.make_routing(S, M)
        mask = workload.make_expert_mask(S) if S.is_ep else None
        hidden = workload.make_inputs(S, M)
        nlt, n_valid = _num_local(S, M)
        kw = workload.fused_kwargs(S, aw, bias, mask)
        if nlt is not None:
            kw["num_local_tokens"] = nlt
        common = {"event": "row", "case_id": c.id}
        ref, ref_s, ref_err = refs[c.id]

        def call(h=hidden, kw=kw, tw=tw, ti=ti, outbuf=None):
            return fm.fused_moe(h, aw["w1"], aw["w2"], tw, ti, output=outbuf, **kw)

        with routes.RouteRecorder() as rec:
            try:
                out = call()
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                status = "unsupported" if _is_unsupported(e, family, gfx) else "error"
                for m in modes:
                    emit(
                        fh,
                        dict(
                            common,
                            mode=m,
                            status=status,
                            detail=_err(e),
                            tb=traceback.format_exc()[-1200:],
                            **routes.dry_route(c, S),
                        ),
                    )
                continue
        route = routes.route_from_record(rec.last())
        sentinel_ok = None
        sentinel_note = ""
        try:
            buf = torch.full((M, S.D), SENTINEL, dtype=S.dtype, device="cuda")
            out_s = call(outbuf=buf)
            torch.cuda.synchronize()
            rows = out_s[:n_valid]
            sentinel_ok = not bool((rows.float() == SENTINEL).all(dim=-1).any())
        except Exception as e:  # noqa: BLE001
            sentinel_note = f"output= call: {_err(e)}"
        corr = {}
        fp = workload.fingerprint(out, n_valid)
        if ref is not None:
            corr = workload.check_correctness(out[:n_valid], ref, sentinel_ok)
        else:
            corr = {"ref_pass": None, "sentinel_ok": sentinel_ok}
            sentinel_note = (sentinel_note + " " + (ref_err or "")).strip()
        corr["sentinel_ok"] = sentinel_ok
        flop = workload.flops(S, n_valid)
        gb = workload.weight_bytes_touched(S, aw, ti, mask, n_valid)
        outbuf = (
            torch.empty((M, S.D), dtype=S.dtype, device="cuda")
            if S.output_buf
            else None
        )
        hidden_static = hidden.clone()
        pool = []
        per = hidden.numel() * hidden.element_size()
        k = int(max(1, min(4, (2 << 30) // max(per, 1))))
        for j in range(k):
            pool.append(workload.make_inputs(S, M) if j else hidden.clone())
        timer = Timer(
            functools.partial(call, h=hidden_static, outbuf=outbuf), hidden_static, pool
        )
        try:
            timer.prepare_graph()
            status = "ok" if corr.get("ref_pass") is not False else "fail"
            detail = sentinel_note
            if (
                status == "fail"
                and gfx != "gfx950"
                and family in UNSUPPORTED_FAMILIES_NON_GFX950
            ):
                status = "unsupported"
                detail = (
                    f"ran but wrong result on {gfx} (err_ratio={corr['err_ratio']:.4f}, "
                    f"logits_diff={corr['logits_diff']:.4g}); aiter's own test_moe_2stage "
                    "skips per_1x32 off gfx950"
                )
            for m in modes:
                st = timer.time(cold=(m == "cold"))
                us = st["median_us"]
                emit(
                    fh,
                    dict(
                        common,
                        mode=m,
                        status=status,
                        detail=detail,
                        timing_mode=timer.mode,
                        capture_error=timer.capture_error,
                        tflops=flop / (us * 1e-6) / 1e12,
                        gbs=gb / (us * 1e-6) / 1e9,
                        ref_seconds=ref_s,
                        weight_build_seconds=build_s,
                        fp=fp,
                        **st,
                        **corr,
                        **route,
                    ),
                )
        except Exception as e:  # noqa: BLE001
            for m in modes:
                emit(
                    fh,
                    dict(
                        common,
                        mode=m,
                        status="error",
                        detail=_err(e),
                        tb=traceback.format_exc()[-1200:],
                        **corr,
                        **route,
                    ),
                )
        torch.cuda.empty_cache()


def _aux_closure(case):
    import torch

    import aiter.fused_moe as fm

    from . import workload

    r = case.row
    E, topk, M = int(r["expert"]), int(r["topk"]), case.token
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        r["dtype"], torch.bfloat16
    )
    g = torch.Generator(device="cuda").manual_seed(workload._seed("aux", case.id))
    if case.kind == "routing":
        gating = torch.randn((M, E), dtype=dt, device="cuda", generator=g)
        hidden = torch.empty((M, 8), dtype=dt, device="cuda")
        return lambda: fm.fused_topk(hidden, gating, topk, True)
    logits = torch.randn((M, E), dtype=torch.float32, device="cuda", generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), topk, dim=-1)
    ids = ids.to(torch.int32).contiguous()
    w = (w / w.sum(-1, keepdim=True)).contiguous()
    D = int(r["model_dim"] or 4096)
    if r["q_type"] == "per_1x32":
        hidden = torch.randn((M, D), dtype=torch.bfloat16, device="cuda", generator=g)
        return lambda: fm._fused_decode_sort_quant(
            hidden,
            ids,
            w,
            model_dim=D,
            global_experts=E,
            block_m=32,
            moebuf_dtype=torch.bfloat16,
            accumulate=True,
        )
    mask = None
    if case.extra.get("expert_mask") not in (None, "", "0"):
        mask = torch.zeros((E,), dtype=torch.int32, device="cuda")
        mask[: max(1, E // 2)] = 1
    return lambda: fm.moe_sorting(ids, w, E, D, dt, 32, mask)


def run_aux_case(case, modes_req, fh):
    import torch

    from . import routes
    from .timing import Timer

    route = routes.aux_route(case)
    common = {"event": "row", "case_id": case.id}
    modes = [m for m in case.modes if m in modes_req]
    if not modes:
        return
    emit(fh, {"event": "case_start", "case_id": case.id, "phase": "run"})
    try:
        call = _aux_closure(case)
        call()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        for m in modes:
            emit(
                fh, dict(common, mode=m, status="unsupported", detail=_err(e), **route)
            )
        return
    timer = Timer(call)
    timer.prepare_graph()
    for m in modes:
        st = timer.time(cold=(m == "cold"))
        emit(
            fh,
            dict(
                common,
                mode=m,
                status="ok",
                timing_mode=timer.mode,
                capture_error=timer.capture_error,
                **st,
                **route,
            ),
        )


def run_first_case(case, fh, variant):
    """First call in a fresh process: JIT module load and FlyDSL compile included.

    ``variant`` is ``warm`` (caches as they are) or ``empty`` (FlyDSL cache off,
    set by the parent through the environment). Weights are filled from one
    quantized chunk since only the call's own cost matters here.
    """
    import torch

    import aiter.fused_moe as fm

    from . import routes, workload

    S = workload.resolve(case)
    common = {"event": "row", "case_id": case.id, "mode": f"first-{variant}"}
    emit(fh, {"event": "case_start", "case_id": case.id, "phase": "first"})
    try:
        aw, _rw, bias = workload.build_group(S, same_chunk=True)
        tw, ti = workload.make_routing(S, case.token)
        mask = workload.make_expert_mask(S) if S.is_ep else None
        hidden = workload.make_inputs(S, case.token)
        kw = workload.fused_kwargs(S, aw, bias, mask)
        torch.cuda.synchronize()
        with routes.RouteRecorder() as rec:
            t0 = time.perf_counter()
            fm.fused_moe(hidden, aw["w1"], aw["w2"], tw, ti, **kw)
            torch.cuda.synchronize()
            first = time.perf_counter() - t0
            t1 = time.perf_counter()
            fm.fused_moe(hidden, aw["w1"], aw["w2"], tw, ti, **kw)
            torch.cuda.synchronize()
            second = time.perf_counter() - t1
        emit(
            fh,
            dict(
                common,
                status="ok",
                first_call_s=first,
                second_call_s=second,
                median_us=first * 1e6,
                **routes.route_from_record(rec.last()),
            ),
        )
    except Exception as e:  # noqa: BLE001
        emit(fh, dict(common, status="error", detail=_err(e)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--group", required=True, help="JSON file: {cases: [...], modes: [...]}"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.group).read_text())
    from .schema import Case, parse_extra

    cases = [Case(r, parse_extra(r.get("extra", ""))) for r in spec["cases"]]
    modes = spec["modes"]
    with open(a.out, "a") as fh:
        emit(fh, {"event": "worker_start", "pid": os.getpid(), "cases": len(cases)})
        if spec.get("first"):
            for c in cases:
                run_first_case(c, fh, spec["first"])
        elif cases[0].kind == "fused":
            run_fused_group(cases, modes, fh, None)
        else:
            for c in cases:
                run_aux_case(c, modes, fh)
        emit(fh, {"event": "worker_done"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
