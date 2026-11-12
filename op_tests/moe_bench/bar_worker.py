# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Subprocess worker for ``bar`` and ``new`` mode: one weight group, JSON lines out.

Per case it measures the dispatch (``fused_moe`` as dispatched), then every candidate
the registered sources enumerate (screened, then the finalists confirmed), all over the
same timed region and against the same reference. The bar is the fastest correct one.

Events: ``case_start`` (phase ``cand:<id>`` while a candidate runs, so a crash is
attributed to it), ``cand`` (one per candidate and mode) and ``row`` (per case and mode).
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from .worker import _err, _num_local, emit

SKIP_STATUS = ("unsupported", "invalid", "oom")


class DispatchCandidate:
    id = "dispatch"
    source = "dispatch"
    entry = "aiter.fused_moe.fused_moe"
    kernels = ""
    layout = "default"
    group = "dispatch"

    def describe(self):
        return {
            "source": self.source,
            "entry": self.entry,
            "kernels": self.kernels,
            "layout": self.layout,
        }

    def prepare(self, ctx):
        pass

    def call(self, ctx):
        import aiter.fused_moe as fm

        def run():
            return fm.fused_moe(
                ctx.hidden_static,
                ctx.aw["w1"],
                ctx.aw["w2"],
                ctx.tw,
                ctx.ti,
                output=ctx.outbuf,
                **ctx.kwargs,
            )

        return run

    def release(self, ctx):
        pass


class Measurer:
    """Correctness plus timing of one candidate over the shared timed region."""

    def __init__(self, ctx, ref, ref_err, pool, fh, case_id):
        self.ctx, self.ref, self.ref_err, self.pool = ctx, ref, ref_err, pool
        self.fh, self.case_id = fh, case_id
        self.records = []  # confirmed results for the bar

    def measure(self, cand, quick=False, modes=("warm",)):
        """Returns {"ok", "us", ...}; emits one ``cand`` event per mode."""
        import torch

        from . import routes, workload
        from .timing import Timer

        ctx = self.ctx
        phase = "screen" if quick else "confirm"
        emit(
            self.fh,
            {
                "event": "case_start",
                "case_id": self.case_id,
                "phase": f"cand:{cand.id}",
            },
        )
        base = dict(
            cand.describe(),
            event="cand",
            case_id=self.case_id,
            cand_id=cand.id,
            phase=phase,
        )
        res = {"ok": False, "us": None, "cand": cand}
        try:
            cand.prepare(ctx)
            call = cand.call(ctx)
            with routes.RouteRecorder() as rec:
                out = call()
                torch.cuda.synchronize()
            route = routes.route_from_record(rec.last()) if rec.records else {}
            if getattr(cand, "id", "") == "dispatch":
                ctx.dispatch_args = rec.last()[0]
                res["route"] = route
            reason = cand.verify(route) if hasattr(cand, "verify") else None
            if reason:
                self._emit(base, modes, status="router-discarded", detail=reason)
                return res
            n = ctx.n_valid
            if self.ref is None:
                self._emit(
                    base, modes, status="no-reference", detail=self.ref_err or ""
                )
                return res
            corr = workload.check_correctness(out[:n], self.ref, None)
            res["corr"] = corr
            if not corr["ref_pass"]:
                self._emit(base, modes, status="fail", detail="correctness", **corr)
                return res
            timer = Timer(call, ctx.hidden_static, self.pool)
            timer.prepare_graph()
            res["timing_mode"] = timer.mode
            for m in modes:
                st = timer.time(cold=(m == "cold"), quick=quick)
                res.setdefault("by_mode", {})[m] = st
                self._emit(
                    base,
                    [m],
                    status="ok",
                    timing_mode=timer.mode,
                    **st,
                    **corr,
                    new_kernel_id=getattr(cand, "kernel_id", ""),
                    kernel1=route.get("kernel1", ""),
                    kernel2=route.get("kernel2", ""),
                    route_key=route.get("route_key", ""),
                )
            res["ok"] = True
            res["us"] = res["by_mode"][modes[0]]["median_us"]
            return res
        except Exception as e:  # noqa: BLE001
            status = "unsupported" if isinstance(e, NotImplementedError) else "error"
            self._emit(
                base,
                modes,
                status=status,
                detail=_err(e),
                tb=traceback.format_exc()[-800:],
            )
            return res
        finally:
            try:
                cand.release(ctx)
            except Exception:  # noqa: BLE001
                emit(self.fh, {"event": "log", "msg": f"release failed for {cand.id}"})

    def _emit(self, base, modes, **kw):
        for m in modes:
            emit(self.fh, dict(base, mode=m, **kw))


def _gpu_identity():
    import torch

    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    from . import roofline

    name = torch.cuda.get_device_properties(0).name
    return {
        "gpu": name,
        "gfx": get_gfx(),
        "cu_num": get_cu_num(),
        "sku": roofline.sku_of(name),
    }


def _roof_case(case, S):
    from . import roofline

    d = dict(case.row)
    d["ep_local"] = roofline.case_ep(case.row, case.extra)
    return d


def run_bar_group(cases, spec, fh):
    import torch

    from . import candidates, routes, workload

    modes_req = spec["modes"]
    mode_kind = spec.get("kind", "bar")
    dispatch_status = spec.get("dispatch_status", {})
    bad = spec.get("bad_cands", {})
    main_failures = spec.get("main_failures", {})
    ident = _gpu_identity()
    specs = {c.id: workload.resolve(c) for c in cases}
    live = []
    for c in cases:
        want = [m for m in c.modes if m in modes_req] or ["warm"]
        st = dispatch_status.get(c.id)
        if st in SKIP_STATUS:
            for m in want:
                emit(
                    fh,
                    {
                        "event": "row",
                        "case_id": c.id,
                        "mode": m,
                        "status": st,
                        "detail": f"dispatch result is {st}; bar not attempted",
                    },
                )
            continue
        r = routes.dry_route(c, specs[c.id])
        if r.get("route_error") and mode_kind == "bar":
            for m in want:
                emit(
                    fh,
                    {
                        "event": "row",
                        "case_id": c.id,
                        "mode": m,
                        "status": "unsupported",
                        "detail": r["route_error"],
                    },
                )
            continue
        live.append(c)
    if not live:
        return
    S0 = specs[live[0].id]
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
                        "detail": f"group needs about {need/2**30:.1f} GiB, {free/2**30:.1f} GiB free",
                    },
                )
        return

    emit(fh, {"event": "case_start", "case_id": live[0].id, "phase": "weights"})
    aw, rw, bias = workload.build_group(S0)
    torch.cuda.synchronize()
    refs = {}
    for c in live:
        S = specs[c.id]
        emit(fh, {"event": "case_start", "case_id": c.id, "phase": "reference"})
        tw, ti = workload.make_routing(S, c.token)
        mask = workload.make_expert_mask(S) if S.is_ep else None
        hidden = workload.make_inputs(S, c.token)
        _, n_valid = _num_local(S, c.token)
        try:
            refs[c.id] = (
                workload.reference(S, rw, bias, hidden, tw, ti, mask, n_valid),
                None,
            )
        except Exception as e:  # noqa: BLE001
            refs[c.id] = (None, f"reference error: {_err(e)}")
        torch.cuda.empty_cache()
    ref_weights = rw if mode_kind == "new" else None
    rw = None
    torch.cuda.empty_cache()
    group_cache = {}

    sources = candidates.load_sources(
        spec.get("sources"), spec.get("source_modules", ())
    )
    tmpdir = spec["tmpdir"]
    for c in live:
        S = specs[c.id]
        modes = [m for m in c.modes if m in modes_req]
        if not modes:
            continue
        M = c.token
        tw, ti = workload.make_routing(S, M)
        mask = workload.make_expert_mask(S) if S.is_ep else None
        hidden = workload.make_inputs(S, M)
        nlt, n_valid = _num_local(S, M)
        kw = workload.fused_kwargs(S, aw, bias, mask)
        if nlt is not None:
            kw["num_local_tokens"] = nlt
        ctx = candidates.Ctx(
            c, S, aw, bias, mask, hidden, tw, ti, nlt, n_valid, kw, tmpdir
        )
        ctx.dispatch_meta = ident
        ctx.group_cache = group_cache
        ctx.ref_weights = ref_weights
        ctx.hidden_static = hidden.clone()
        ctx.outbuf = (
            torch.empty((M, S.D), dtype=S.dtype, device="cuda")
            if S.output_buf
            else None
        )
        per = hidden.numel() * hidden.element_size()
        k = int(max(1, min(4, (2 << 30) // max(per, 1))))
        pool = [workload.make_inputs(S, M) if j else hidden.clone() for j in range(k)]
        ref, ref_err = refs[c.id]
        meas = Measurer(ctx, ref, ref_err, pool, fh, c.id)
        results = {}
        skipped = []
        emit(fh, {"event": "case_start", "case_id": c.id, "phase": "run"})
        if mode_kind == "bar":
            d = meas.measure(DispatchCandidate(), quick=False, modes=modes)
            results["dispatch"] = d
        for src in sources:
            if (mode_kind == "new") != (src.name == "new"):
                continue
            reason = src.skip_reason(ctx)
            if reason:
                skipped.append((src.name, reason))
                emit(
                    fh,
                    {
                        "event": "cand",
                        "case_id": c.id,
                        "cand_id": f"{src.name}|skipped",
                        "source": src.name,
                        "entry": "",
                        "kernels": "",
                        "phase": "enumerate",
                        "mode": modes[0],
                        "status": "skipped",
                        "detail": reason,
                    },
                )
                continue
            if mode_kind == "new" and src.name != "new":
                continue
            if mode_kind == "bar" and src.name == "new":
                continue
            try:
                cands = [x for x in src.enumerate(ctx) if x.id not in bad.get(c.id, ())]
            except Exception as e:  # noqa: BLE001
                emit(
                    fh,
                    {
                        "event": "cand",
                        "case_id": c.id,
                        "cand_id": f"{src.name}|enumerate",
                        "source": src.name,
                        "phase": "enumerate",
                        "mode": modes[0],
                        "status": "error",
                        "detail": _err(e),
                    },
                )
                continue
            for msg in getattr(ctx, "enum_errors", []):
                emit(
                    fh,
                    {
                        "event": "cand",
                        "case_id": c.id,
                        "cand_id": f"{src.name}|generator",
                        "source": src.name,
                        "phase": "enumerate",
                        "mode": modes[0],
                        "status": "note",
                        "detail": msg,
                    },
                )
            emit(
                fh,
                {"event": "log", "msg": f"{c.id} {src.name}: {len(cands)} candidates"},
            )
            if mode_kind == "bar":
                keep = src.screen(ctx, cands, _Screener(meas, bad.get(c.id, ())))
            else:
                keep = cands
            for cand in keep:
                results[cand.id] = meas.measure(cand, quick=False, modes=modes)
        for m in modes:
            _emit_summary(
                fh, c, S, m, results, ident, skipped, mode_kind, main_failures.get(c.id)
            )
        from .candidates.fused_cfg import unforce

        unforce()
        torch.cuda.empty_cache()


class _Screener:
    def __init__(self, meas, bad):
        self.meas = meas
        self.bad = set(bad)

    def measure(self, cand, quick=True):
        if cand.id in self.bad:
            return {"ok": False, "us": None}
        return self.meas.measure(cand, quick=quick, modes=("warm",))


def _emit_summary(fh, c, S, mode, results, ident, skipped, kind, main_failure=None):
    from . import roofline

    ok = {
        cid: r
        for cid, r in results.items()
        if r.get("ok") and mode in r.get("by_mode", {})
    }
    row = {
        "event": "row",
        "case_id": c.id,
        "mode": mode,
        "n_candidates": len(results),
        "n_correct": len(ok),
        "skipped_sources": "; ".join(n for n, _ in skipped),
        "gpu": ident["gpu"],
        "main_failure": main_failure or "",
    }
    disp = results.get("dispatch")
    if disp is not None:
        row["dispatch_status"] = "ok" if disp.get("ok") else "fail-or-error"
        if disp.get("ok") and mode in disp["by_mode"]:
            row["dispatch_us"] = disp["by_mode"][mode]["median_us"]
        for k in ("route_key", "kernel1", "kernel2", "route_family"):
            row["dispatch_" + k] = disp.get("route", {}).get(k, "")
    if kind == "new":
        if ok:
            r = next(iter(ok.values()))
            row.update(
                new_us=r["by_mode"][mode]["median_us"],
                new_id=r["cand"].id,
                new_kernel_id=getattr(r["cand"], "kernel_id", ""),
                status="ok",
            )
        else:
            row.update(status="no-correct-candidate")
    elif ok:
        best = min(ok.values(), key=lambda r: r["by_mode"][mode]["median_us"])
        d = best["cand"].describe()
        row.update(
            status="ok",
            bar_us=best["by_mode"][mode]["median_us"],
            bar_id=best["cand"].id,
            bar_source=d["source"],
            bar_entry=d["entry"],
            bar_kernels=d["kernels"],
            bar_timing_mode=best.get("timing_mode", ""),
        )
    else:
        # a case no existing backend computes correctly has no bar
        row["status"] = "no-bar"
    if ident["sku"] and S.kind == "fused":
        rf = roofline.model(_roof_case(c, S), ident["sku"], ident["gfx"], mode)
        row.update(
            roofline_us=rf["roofline_us"],
            roofline_staged_us=rf["roofline_staged_us"],
            roofline_serial_us=rf["roofline_serial_us"],
            bound=rf["bound"],
            mfma=rf["mfma"],
        )
        for k in ("bar_us", "dispatch_us", "new_us"):
            if row.get(k):
                row[k.replace("_us", "_roofline_eff")] = rf["roofline_us"] / row[k]
    emit(fh, row)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.group).read_text())
    from .schema import Case, parse_extra

    cases = [Case(r, parse_extra(r.get("extra", ""))) for r in spec["cases"]]
    with open(a.out, "a") as fh:
        emit(fh, {"event": "worker_start", "pid": os.getpid(), "cases": len(cases)})
        run_bar_group(cases, spec, fh)
        emit(fh, {"event": "worker_done"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
