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


class Bf16BoundCandidate(DispatchCandidate):
    """The least-rounding aiter path where dispatch rounds more than requested: the same
    ``fused_moe`` call with the activation-dtype switch raised out of reach
    (``GPTOSS_SWIGLU_MXFP4_BF16_BOUND`` for SEPARATED Swiglu, ``AITER_BF16_FP8_MOE_BOUND``
    for INTERLEAVE), so aiter keeps the bf16 activations (CK-Tile bf16) it was given."""

    id = "dispatch_bf16bound"
    source = "aiter_bf16_bound"
    entry = "aiter.fused_moe.fused_moe (activation-dtype bound raised)"
    group = "dispatch_bf16bound"
    HUGE = 1 << 62

    def __init__(self):
        self.confirmed = ""
        self._saved = None

    def prepare(self, ctx):
        import aiter.fused_moe as fm

        self._saved = (
            fm._SWIGLU_MXFP4_BF16_BOUND,
            os.environ.get("AITER_BF16_FP8_MOE_BOUND"),
        )
        fm._SWIGLU_MXFP4_BF16_BOUND = self.HUGE
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = str(self.HUGE)

    def release(self, ctx):
        import aiter.fused_moe as fm

        if self._saved is None:
            return
        fm._SWIGLU_MXFP4_BF16_BOUND, env = self._saved
        if env is None:
            os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
        else:
            os.environ["AITER_BF16_FP8_MOE_BOUND"] = env
        self._saved = None

    def confirm(self, args, route, S):
        """Reason the override did not take effect (None when the call ran with the
        activations the case passed). ``args`` are the get_2stage_cfgs arguments that
        aiter logged for the call."""
        if not args:
            return (
                "no get_2stage_cfgs call recorded: cannot confirm the activation dtype"
            )
        if args.get("q_dtype_a") != S.aq:
            return (
                f"activation dtype is still {args.get('q_dtype_a')} (case passes "
                f"{S.aq}): the bound override did not take effect"
            )
        self.confirmed = (
            f"q_dtype_a={str(args['q_dtype_a']).split('.')[-1]} "
            f"kernel1={route.get('kernel1', '')} kernel2={route.get('kernel2', '')}"
        )
        self.kernels = f"{route.get('kernel1', '')} + {route.get('kernel2', '')}"
        return None


def _is_gpu_fault(msg):
    return any(
        k in msg
        for k in ("AcceleratorError", "HIP error", "invalid StreamId", "hipError")
    )


def disable_routes():
    """AITER_MOE_DISABLE=1 for this process: bar mode measures aiter's own paths. On a tree
    with live route rows the dispatched call would otherwise be our routed kernel (and the
    route recorder sees no get_2stage_cfgs call at all). Returns whether the tree has
    ``aiter.moe`` (and so a hook that had to be told)."""
    os.environ["AITER_MOE_DISABLE"] = "1"
    try:
        from aiter import moe
    except Exception:  # noqa: BLE001  (a tree without aiter.moe has no hook)
        return False
    if hasattr(moe, "refresh"):
        moe.refresh()
    return True


def assert_no_route():
    """After an aiter-side call: ``aiter.moe.last_route()`` must report no route taken."""
    try:
        from aiter import moe

        last = moe.last_route()
    except Exception:  # noqa: BLE001
        return
    kid, reason = _parse_route(last)
    if kid not in ("", "aiter"):
        raise RuntimeError(
            f"aiter.moe served an aiter-side measurement with {kid!r} ({reason}); "
            "routes are not disabled (AITER_MOE_DISABLE)"
        )
    # a core that records every call while routing is off (router._INACTIVE) must say so;
    # an older one keeps the previous call's record, so only the kernel id can be judged
    if hasattr(getattr(moe, "router", None), "_INACTIVE") and not reason.startswith(
        "routing inactive"
    ):
        raise RuntimeError(
            f"aiter.moe last_route() says {reason!r}, not 'routing inactive': the hook "
            "looked at this call (AITER_MOE_DISABLE not in effect)"
        )


def mixed_branch_note(c):
    """MOEBUG-054: aiter's flydsl/opus 2-stage branch wraps a CK-Tile or ASM stage 1 in
    ck_moe_stage1, so such a pair resolves to a CK module whatever the stage-1 name says.
    """
    m = getattr(c, "meta", None) or {}
    s1, s2 = m.get("s1") or "", m.get("s2") or ""
    if s1.startswith(("cktile_", "_ZN")) and s2.startswith(("flydsl_", "opus_")):
        return (
            " (MOEBUG-054: aiter resolves this stage-1 name through ck_moe_stage1 "
            "in the flydsl/opus branch)"
        )
    return ""


def drop_build_failed(fh, case_id, source, mode, cands):
    """Candidates whose module failed to build on a host prebuild never reach the lease:
    each is reported ``build_failed`` with the module's recorded error, not timed."""
    from . import prebuild

    excluded = prebuild.excluded_keys(case_id) if source == "fused_moe" else {}
    if not excluded:
        return cands
    keep = []
    for x in cands:
        hit = excluded.get(prebuild.cand_key(x))
        if hit is None:
            keep.append(x)
            continue
        md, err = hit
        note = mixed_branch_note(x)
        emit(
            fh,
            {
                "event": "cand",
                "case_id": case_id,
                "cand_id": x.id,
                "source": source,
                "kernels": x.kernels or "",
                "phase": "enumerate",
                "mode": mode,
                "status": "build_failed",
                "detail": f"build_failed: {md}: {err[-200:]}{note}",
            },
        )
    emit(
        fh,
        {
            "event": "log",
            "msg": f"{case_id} {source}: {len(cands) - len(keep)} candidates "
            "excluded (their module failed to build on the host prebuild)",
        },
    )
    return keep


def drop_aot_uncovered(fh, case_id, source, mode, cands):
    """In a run-only lease: candidates whose FlyDSL artifacts the host build's cover does
    not contain are reported ``aot-missing`` (a harness error that makes the bar
    incomplete) instead of failing one by one inside the lease. Returns (kept, n)."""
    if source != "fused_moe" or not (
        os.environ.get("FLYDSL_RUNTIME_RUN_ONLY") == "1"
        or os.environ.get("MOE_BENCH_NO_BUILD") == "1"
    ):
        return cands, 0
    from . import prebuild
    from .flydsl_prebuild import elements, is_flydsl

    entry = prebuild.read_manifest().get(case_id)
    if not entry or "covered" not in entry:
        return cands, 0
    covered = set(entry["covered"])
    keep, n = [], 0
    for x in cands:
        if is_flydsl(x) and any(e not in covered for e in elements(x)):
            n += 1
            emit(
                fh,
                {
                    "event": "cand",
                    "case_id": case_id,
                    "cand_id": x.id,
                    "source": source,
                    "kernels": x.kernels or "",
                    "phase": "enumerate",
                    "mode": mode,
                    "status": "aot-missing",
                    "detail": "no usable AOT cache: the host build's cover lacks "
                    "this candidate's FlyDSL artifact",
                },
            )
        else:
            keep.append(x)
    return keep, n


def aiter_error_detail(e):
    """Error text of a NameError / UnboundLocalError raised inside aiter's own sources (not
    the harness's), else None: a candidate aiter cannot run (e.g. ``topk_ids_p`` undefined in
    silu_and_mul_fq.py for k_batch > 1 stage 1). It is neither timeable nor a harness error:
    it gets its own status, ``aiter_error``, and stays out of the bar."""
    if not isinstance(e, (NameError, UnboundLocalError)):
        return None
    tb, last = e.__traceback__, None
    while tb is not None:
        last, tb = tb, tb.tb_next
    if last is None:
        return None
    fn = last.tb_frame.f_code.co_filename.replace("\\", "/")
    if "/aiter/" not in fn or "/moe_bench/" in fn:
        return None
    return f"{_err(e)} at {fn.rsplit('/aiter/', 1)[-1]}:{last.tb_lineno}"


def not_prebuilt_row(md_name):
    """(status, detail, outcome) of a candidate that reached a module not built beforehand.

    A leased run never compiles: the candidate is not timed. A module that failed to build
    on the CPU is reported with its recorded error (``build_failed``); any other is logged
    as ``not prebuilt: <module>`` and recorded for the next prebuild."""
    from . import prebuild

    failed = prebuild.known_failed().get(md_name)
    if failed:
        return (
            "build_failed",
            f"build_failed: {md_name}: {failed[-200:]}",
            "build_failed",
        )
    prebuild.record_missing(md_name)
    return "not-prebuilt", f"not prebuilt: {md_name}", "notprebuilt"


def _declared_numerics(moe, last_route, kid):
    """Numerics of the kernel that served the last call: ``last_route().numerics``, else
    ``aiter.moe.numerics(id)``; None when aiter's own path served it."""
    if kid in ("", "aiter"):
        return None
    declared = getattr(last_route, "numerics", None)
    if declared is None and hasattr(moe, "numerics"):
        declared = moe.numerics(kid)
    return declared


def _parse_route(v):
    """aiter.moe.last_route() -> (kernel id or "aiter", reason). Tuple, dict or str."""
    if v is None:
        return "", ""
    if isinstance(v, dict):
        return str(v.get("route") or v.get("kernel") or ""), str(v.get("reason", ""))
    if isinstance(v, (tuple, list)):
        return str(v[0]), str(v[3]) if len(v) > 3 else ""
    return str(v), ""


class RoutedFusedMoe(DispatchCandidate):
    """``aiter.fused_moe.fused_moe`` of the tree under test, with routing on (new) or
    switched off by AITER_MOE_DISABLE (off). The router inside fused_moe sends covered
    cases to the new kernels; ``aiter.moe.last_route()`` says what ran."""

    source = "new"
    measure_host = True

    def __init__(self, off=False):
        self.id = "off|fused_moe" if off else "new|fused_moe"
        self.entry = "aiter.fused_moe.fused_moe" + (
            " (AITER_MOE_DISABLE=1)" if off else " (routing on)"
        )
        self.kernel_id = ""
        self.reason = ""
        self.route_changed = False
        self.declared_numerics = None  # None: aiter's own path served the call

    def probe_route(self, stage):
        """Record the route of the last call: after the eager call and after capture.
        The capture's wins: a bucket skipped during capture shows up as ``aiter``."""
        try:
            from aiter import moe

            lr = moe.last_route()
            kid, reason = _parse_route(lr)
            declared = _declared_numerics(moe, lr, kid)
        except Exception:  # noqa: BLE001  (tree without aiter.moe)
            return
        self.declared_numerics = declared
        if stage == "capture" and self.kernel_id and kid != self.kernel_id:
            self.route_changed = True
        self.kernel_id, self.reason = kid, reason


class Measurer:
    """Correctness plus timing of one candidate over the shared timed region."""

    def __init__(self, ctx, acc_refs, acc_err, rows, pool, fh, case_id, ref_ctx=None):
        self.ctx, self.acc_refs, self.acc_err = ctx, acc_refs, acc_err
        self.rows, self.pool = rows, pool
        self.ref_ctx = ref_ctx or {}
        self._path_cache = {}
        self.stats = (
            {}
        )  # "source:backend" -> {"total", "ok", "fail", "compile", "error"}
        self.fh, self.case_id = fh, case_id
        self.records = []  # confirmed results for the bar
        self.routes_disabled = False  # bar mode: assert no route served a call

    def measure(self, cand, quick=False, modes=("warm",), check=True):
        """Returns {"ok", "us", "outcome", ...}; emits one ``cand`` event per mode.

        ``check=False`` is the timing-only screen of two-phase bar mode: the router check
        and a short warm timing, no accuracy (finalists get the full measurement)."""
        res = self._measure(cand, quick, modes, check)
        self._tally(cand, res)
        return res

    def _tally(self, cand, res):
        from . import routes

        kernels = getattr(cand, "kernels", "") or ""
        key = f"{cand.source}:{routes.backend_of(kernels.split(' ')[0]) or 'dispatch'}"
        t = self.stats.setdefault(
            key, {"total": 0, "ok": 0, "fail": 0, "compile": 0, "error": 0}
        )
        t["total"] += 1
        t["unnamed"] = t.get("unnamed", 0) + bool(res.get("unnamed"))
        t[res.get("outcome") or ("ok" if res.get("ok") else "error")] = (
            t.get(res.get("outcome") or ("ok" if res.get("ok") else "error"), 0) + 1
        )

    def _measure(self, cand, quick=False, modes=("warm",), check=True):
        import torch

        from aiter import dtypes

        from . import accuracy, routes
        from . import prebuild as prebuild_mod
        from .timing import Timer

        ctx = self.ctx
        prebuild_mod.CURRENT["candidate"] = f"{self.case_id}|{cand.id}"
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
                call()
                torch.cuda.synchronize()
            if self.routes_disabled:
                assert_no_route()
            if hasattr(cand, "probe_route"):
                cand.probe_route("eager")
            route = routes.route_from_record(rec.last()) if rec.records else {}
            if getattr(cand, "id", "") == "dispatch":
                ctx.dispatch_args = rec.last()[0]
                res["route"] = route
                # the dtype aiter logged settles it; the static rule only seeds the flag
                # the routed kernels' ``same_as`` reads
                res["rounds_more"] = accuracy.measured_rounds_more(
                    ctx.S, ctx.dispatch_args.get("q_dtype_a"), ctx.case.token
                )
                if res["rounds_more"] != getattr(ctx, "dispatch_rounds_more", ""):
                    emit(
                        self.fh,
                        {
                            "event": "log",
                            "msg": f"{self.case_id}: static rule said "
                            f"{bool(ctx.dispatch_rounds_more)}, measured "
                            f"{bool(res['rounds_more'])}; using the measured one",
                        },
                    )
                    ctx.dispatch_rounds_more = res["rounds_more"]
            reason = cand.verify(route) if hasattr(cand, "verify") else None
            if reason is None and hasattr(cand, "confirm"):
                reason = cand.confirm(
                    rec.last()[0] if rec.records else None, route, ctx.S
                )
                if reason is None:
                    emit(
                        self.fh,
                        {
                            "event": "log",
                            "msg": f"{self.case_id} {cand.id} ran: {cand.confirmed}",
                        },
                    )
            if reason:
                res["outcome"] = "discarded"
                # a route that resolved to some other real kernel is aiter declining the
                # candidate (not applicable here); an unnamed impl is a harness defect
                res["unnamed"] = "'<lambda>'" in reason
                self._emit(base, modes, status="router-discarded", detail=reason)
                return res
            if not check:
                return self._time_only(cand, base, call, route, res)
            if self.acc_refs is None:
                res["outcome"] = "noref"
                self._emit(
                    base,
                    modes,
                    status="no-reference",
                    detail=self.acc_err or "",
                    new_kernel_id=getattr(cand, "kernel_id", ""),
                )
                return res
            if res.get("rounds_more"):
                num = accuracy.requested_numerics(ctx.S, route)
            else:
                num = accuracy.numerics_of(ctx.S, cand, route, ctx)
            variants = (
                accuracy.in_kernel_variants(ctx.S, num)
                if accuracy.has_in_kernel(num)
                else None
            )
            nkey = accuracy.numerics_key(num)
            base["numerics"] = nkey
            res["numerics"] = nkey
            try:
                refs = (
                    self.refs_for_variants(variants) if variants else self.refs_for(num)
                )
            except NotImplementedError:
                raise
            except Exception as e:  # noqa: BLE001
                if _is_gpu_fault(_err(e)):
                    emit(
                        self.fh,
                        {
                            "event": "log",
                            "msg": f"gpu fault building the reference: {_err(e)}",
                        },
                    )
                    os._exit(70)
                res["outcome"] = "noref"
                self._emit(
                    base,
                    modes,
                    status="no-reference",
                    detail=f"reference error (not the candidate): {_err(e)}",
                    new_kernel_id=getattr(cand, "kernel_id", ""),
                )
                return res
            acc = accuracy.evaluate(
                ctx.S,
                ctx,
                call,
                refs,
                self.rows,
                ctx.n_valid,
                seeds=accuracy.SEEDS[:1] if quick else accuracy.SEEDS,
                sentinel=not quick,
                repeats=1 if quick else accuracy.REPEATS,
                flat_mxfp4=bool(route.get("flat"))
                and ctx.S.qtype.name == "per_1x32"
                and ctx.S.wq == dtypes.fp4x2,
            )
            if variants:
                rule = accuracy.match_rule(acc, tuple(variants))
                if rule:
                    nkey = (
                        accuracy.numerics_key(variants[rule]) + ";rule_source=matched"
                    )
                    base["numerics"] = res["numerics"] = nkey
                    acc["matched_rule"] = rule
                    acc["pass"] = accuracy.rule(
                        acc,
                        flat_mxfp4=bool(route.get("flat"))
                        and ctx.S.qtype.name == "per_1x32"
                        and ctx.S.wq == dtypes.fp4x2,
                    )
            res["acc"] = acc
            cols = accuracy.flat_columns(acc)
            if acc["pass"] is None:
                self._emit(
                    base,
                    modes,
                    status="no-ref-path",
                    detail="ref_path unavailable for this path",
                    **cols,
                )
                return res
            if not acc["pass"]:
                res["outcome"] = "fail"
                detail = "accuracy rule"
                if res.get("rounds_more"):
                    detail = (
                        f"dispatch rounds more than requested: {res['rounds_more']}"
                    )
                self._emit(base, modes, status="fail", detail=detail, **cols)
                return res
            timer = Timer(call, ctx.hidden_static, self.pool)
            timer.prepare_graph()
            if hasattr(cand, "probe_route"):
                cand.probe_route("capture")
            res["timing_mode"] = timer.mode
            if getattr(cand, "measure_host", False):
                res["host_us"] = _host_us(call)
            for m in modes:
                st = timer.time(cold=(m == "cold"), quick=quick)
                res.setdefault("by_mode", {})[m] = st
                self._emit(
                    base,
                    [m],
                    status="ok",
                    timing_mode=timer.mode,
                    **st,
                    **cols,
                    new_kernel_id=getattr(cand, "kernel_id", ""),
                    reason=getattr(cand, "reason", ""),
                    host_us=res.get("host_us", ""),
                    kernel1=route.get("kernel1", ""),
                    kernel2=route.get("kernel2", ""),
                    route_key=route.get("route_key", ""),
                )
            res["ok"] = True
            res["outcome"] = "ok"
            res["us"] = res["by_mode"][modes[0]]["median_us"]
            return res
        except accuracy.UnknownNumericsKey as e:
            emit(self.fh, {"event": "harness_stale", "msg": str(e)})
            os._exit(75)
        except Exception as e:  # noqa: BLE001
            from . import prebuild

            md = prebuild.not_prebuilt_module(e)
            if md:
                status, detail, res["outcome"] = not_prebuilt_row(md)
                self._emit(base, modes, status=status, detail=detail)
                return res
            detail = aiter_error_detail(e)
            if detail:
                res["outcome"] = "aiter_error"
                self._emit(base, modes, status="aiter_error", detail=detail)
                return res
            if "no usable AOT cache" in str(e):
                # a FlyDSL kernel the host build did not cover: a harness error, never a
                # quiet drop (the bar is incomplete and says so)
                res["outcome"] = "aotmiss"
                self._emit(base, modes, status="aot-missing", detail=_err(e))
                return res
            status = "unsupported" if isinstance(e, NotImplementedError) else "error"
            msg = _err(e)
            if _is_gpu_fault(msg):
                # The HIP context may be poisoned (every later call fails "invalid
                # StreamId"): exit so the parent marks this candidate crashed and starts
                # a fresh worker for the rest.
                emit(self.fh, {"event": "log", "msg": f"gpu fault in {cand.id}: {msg}"})
                # the runner records this candidate with the error before it restarts us
                emit(
                    self.fh,
                    {
                        "event": "gpu_fault",
                        "case_id": self.case_id,
                        "cand_id": cand.id,
                        "detail": msg,
                    },
                )
                os._exit(70)
            res["outcome"] = (
                "compile"
                if any(k in msg for k in ("DSLCompileError", "lld", "ompile"))
                else "error"
            )
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

    def refs_for(self, numerics):
        """{seed: {"exact", "path"}} with ref_path at the candidate's rounding points."""
        from . import accuracy

        key = accuracy.numerics_key(numerics)
        if key not in self._path_cache:
            rc = self.ref_ctx
            by_seed = {}
            for seed in accuracy.SEEDS:
                hs, tw, ti = accuracy.acc_inputs(rc["S"], rc["M"], seed)
                d = {"exact": self.acc_refs[seed]}
                try:
                    d["path"] = accuracy.path_ref(
                        rc["S"],
                        rc["rw"],
                        rc["bias"],
                        hs,
                        tw,
                        ti,
                        rc["mask"],
                        self.rows,
                        numerics,
                    )
                except NotImplementedError:
                    pass
                by_seed[seed] = d
                del hs
            self._path_cache[key] = by_seed
        return self._path_cache[key]

    def _time_only(self, cand, base, call, route, res):
        from .timing import Timer

        timer = Timer(call, self.ctx.hidden_static, self.pool)
        timer.prepare_graph()
        if hasattr(cand, "probe_route"):
            cand.probe_route("capture")
        res["timing_mode"] = timer.mode
        st = timer.time(cold=False, quick=True)
        res["by_mode"] = {"warm": st}
        self._emit(
            base,
            ["warm"],
            status="ok",
            detail="timing-only screen (accuracy checked for finalists)",
            timing_mode=timer.mode,
            **st,
            new_kernel_id=getattr(cand, "kernel_id", ""),
            kernel1=route.get("kernel1", ""),
            kernel2=route.get("kernel2", ""),
            route_key=route.get("route_key", ""),
        )
        res["ok"] = True
        res["outcome"] = "ok"
        res["unchecked"] = True
        res["us"] = st["median_us"]
        return res

    def refs_for_variants(self, variants):
        """{seed: {"exact", "path_<rule>"...}}: one ref_path per candidate scale rule."""
        out = {}
        for rule, n in variants.items():
            for seed, d in self.refs_for(n).items():
                slot = out.setdefault(seed, {"exact": d["exact"]})
                if "path" in d:
                    slot[f"path_{rule}"] = d["path"]
        return out

    def _emit(self, base, modes, **kw):
        for m in modes:
            emit(self.fh, dict(base, mode=m, **kw))


def _host_us(call, n=50):
    """Host time of an eager call: CPU time from entry to return, GPU drained between calls."""
    import statistics
    import time

    import torch

    for _ in range(5):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        call()
        ts.append((time.perf_counter() - t0) * 1e6)
        torch.cuda.synchronize()
    return statistics.median(ts)


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

    from . import accuracy, candidates, routes, workload

    modes_req = spec["modes"]
    mode_kind = spec.get("kind", "bar")
    if mode_kind == "bar":
        disable_routes()
    dispatch_status = spec.get("dispatch_status", {})
    bad = spec.get("bad_cands", {})
    main_failures = spec.get("main_failures", {})
    ident = _gpu_identity()
    specs = {c.id: workload.resolve(c) for c in cases}
    live = []
    dry_routes = {}
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
        dry_routes[c.id] = r
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
        mask = workload.make_expert_mask(S) if S.is_ep else None
        _, n_valid = _num_local(S, c.token)
        rows = accuracy.sample_rows(c.id, n_valid)
        try:
            by_seed = {}
            for seed in accuracy.SEEDS:
                hs, tw, ti = accuracy.acc_inputs(S, c.token, seed)
                by_seed[seed] = accuracy.exact_ref(S, rw, bias, hs, tw, ti, mask, rows)
                del hs
            refs[c.id] = (by_seed, None, rows)
        except Exception as e:  # noqa: BLE001
            refs[c.id] = (None, f"reference error: {_err(e)}", rows)
        torch.cuda.empty_cache()
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
        ctx.dispatch_route = dry_routes.get(c.id)
        ctx.dispatch_rounds_more = accuracy.dispatch_rounds_more(
            S, c.token, ident["gfx"]
        )
        if ctx.dispatch_rounds_more:
            emit(
                fh,
                {
                    "event": "log",
                    "msg": f"{c.id}: dispatch rounds more: {ctx.dispatch_rounds_more}",
                },
            )
        ctx.group_cache = group_cache
        ctx.hidden_static = hidden.clone()
        ctx.outbuf = (
            torch.empty((M, S.D), dtype=S.dtype, device="cuda")
            if S.output_buf
            else None
        )
        per = hidden.numel() * hidden.element_size()
        k = int(max(1, min(4, (2 << 30) // max(per, 1))))
        pool = [workload.make_inputs(S, M) if j else hidden.clone() for j in range(k)]
        acc_refs, acc_err, rows = refs[c.id]
        meas = Measurer(
            ctx,
            acc_refs,
            acc_err,
            rows,
            pool,
            fh,
            c.id,
            ref_ctx={"S": S, "M": c.token, "rw": rw, "bias": bias, "mask": mask},
        )
        meas.routes_disabled = mode_kind == "bar"
        results = {}
        skipped = []
        enum_counts, enum_errors = {}, []
        emit(fh, {"event": "case_start", "case_id": c.id, "phase": "run"})
        if mode_kind == "bar":
            if "dispatch" in bad.get(c.id, ()):
                # the dispatch call killed an earlier worker: do not run it again
                results["dispatch"] = {"ok": False, "crashed": True}
            else:
                d = meas.measure(DispatchCandidate(), quick=False, modes=modes)
                results["dispatch"] = d
            if ctx.dispatch_rounds_more:
                lr = Bf16BoundCandidate()
                results[lr.id] = meas.measure(lr, quick=False, modes=modes)
        elif mode_kind in ("new", "off"):
            cand = RoutedFusedMoe(off=mode_kind == "off")
            results[cand.id] = meas.measure(cand, quick=False, modes=modes)
        for src in sources:
            if mode_kind in ("new", "off") or src.name == "new":
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
            cands = drop_build_failed(fh, c.id, src.name, modes[0], cands)
            cands, n_aot = drop_aot_uncovered(fh, c.id, src.name, modes[0], cands)
            if n_aot:
                enum_errors.append(
                    f"{n_aot} {src.name} candidates not in the host build's FlyDSL cover"
                )
            for name1, bm, npairs, why in (
                getattr(ctx, "unsupported", None) or ()
                if src.name == "fused_moe"
                else ()
            ):
                emit(
                    fh,
                    {
                        "event": "cand",
                        "case_id": c.id,
                        "cand_id": f"{src.name}|{name1}|bm{bm}",
                        "source": src.name,
                        "kernels": name1,
                        "phase": "enumerate",
                        "mode": modes[0],
                        "status": "unsupported",
                        "detail": f"{why} ({npairs} pairs not built or timed)",
                    },
                )
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
            for x in cands:
                b = routes.backend_of((x.kernels or "").split(" ")[0]) or "dispatch"
                k = f"{src.name}:{b}"
                enum_counts[k] = enum_counts.get(k, 0) + 1
            enum_counts.setdefault(f"{src.name}:*", len(cands))
            enum_errors += getattr(ctx, "enum_errors", [])
            emit(
                fh,
                {"event": "log", "msg": f"{c.id} {src.name}: {len(cands)} candidates"},
            )
            if mode_kind == "bar" and spec.get("screen") == "two-phase":
                two_phase(meas, cands, bad.get(c.id, ()), modes, results, src)
                continue
            if mode_kind == "bar":
                keep = src.screen(ctx, cands, _Screener(meas, bad.get(c.id, ())))
            else:
                keep = cands
            for cand in keep:
                results[cand.id] = meas.measure(cand, quick=False, modes=modes)
        acct, incomplete = _accounting(enum_counts, meas.stats, enum_errors, mode_kind)
        for m in modes:
            _emit_summary(
                fh,
                c,
                S,
                m,
                results,
                ident,
                skipped,
                mode_kind,
                main_failures.get(c.id),
                (acct, incomplete),
            )
        from .candidates.fused_cfg import unforce

        unforce()
        torch.cuda.empty_cache()


FINALISTS_TOP = 8
CROSS_TOP = 4  # top stage-1 x top stage-2 kernels per block_m group, always finalists
FINALISTS_BAND = 0.05  # also anything within 5% of the best screened time


def cross_products(src, top=CROSS_TOP):
    """Pairs of the ``top`` fastest stage-1 and ``top`` fastest stage-2 kernels of each
    block_m group, from the stage times ``src.rank`` just measured."""
    out, seen = [], set()
    for t1, t2, find in (getattr(src, "last_stage", None) or {}).values():
        k1s = sorted(t1, key=t1.get)[:top]
        k2s = sorted(t2, key=t2.get)[:top]
        for k1 in k1s:
            for k2 in k2s:
                c = find.get((k1, k2))
                if c is not None and c.id not in seen:
                    seen.add(c.id)
                    out.append(c)
    return out


def two_phase(meas, cands, bad, modes, results, src=None):
    """Two-phase bar screening of one source's candidates.

    Phase 1 ranks the candidates by a short warm timing without the accuracy check: every
    candidate is timed, or, where the source can predict a pair's time from its stages
    (``src.rank``), the stages are. Phase 2 runs the full measurement (warm and cold
    timing, accuracy) for the finalists: the fastest ``FINALISTS_TOP`` plus anything within
    ``FINALISTS_BAND`` of the best screened time. If every finalist fails, the set widens by
    the next ``FINALISTS_TOP`` ranked candidates until one passes or none is left. Results
    go into ``results``.
    """
    bad = set(bad)
    cands = [c for c in cands if c.id not in bad]
    if src is not None and hasattr(src, "rank"):
        screened = src.rank(None, cands, _Screener(meas, bad))
    else:
        screened = []
        for cand in cands:
            r = meas.measure(cand, quick=True, modes=("warm",), check=False)
            if r["ok"]:
                screened.append((r["us"], cand))
        screened.sort(key=lambda t: t[0])
    if not screened:
        return
    best = screened[0][0]
    n = max(
        FINALISTS_TOP, sum(1 for us, _ in screened if us <= best * (1 + FINALISTS_BAND))
    )
    # the guard: the cross product of the top stage-1 and stage-2 kernels of every block_m
    # group is measured in full too, so a pair whose stages are each fast but whose
    # predicted sum was off is never missed (the bar is never slower than aiter's best
    # real pair for want of timing it)
    for cand in cross_products(src):
        if cand.id not in results and cand.id not in bad:
            results[cand.id] = meas.measure(cand, quick=False, modes=modes)
    done = 0
    while done < len(screened):
        for _, cand in screened[done:n]:
            if cand.id not in results:
                results[cand.id] = meas.measure(cand, quick=False, modes=modes)
        done = n
        if any(results[c.id].get("ok") for _, c in screened[:done]):
            return
        n = min(len(screened), n + FINALISTS_TOP)


class _Screener:
    def __init__(self, meas, bad):
        self.meas = meas
        self.bad = set(bad)

    def measure(self, cand, quick=True, check=True):
        if cand.id in self.bad:
            return {"ok": False, "us": None}
        return self.meas.measure(cand, quick=quick, modes=("warm",), check=check)


def _gate(who, acc, numerics=""):
    """Gated accuracy columns (mean/std vs both references) of one candidate."""
    from . import accuracy

    if not acc:
        return {}
    out = {
        f"{who}_numerics": numerics,
        f"{who}_acc_pass": acc["pass"],
        f"{who}_nondeterministic": acc["nondeterministic"],
    }
    for ref in accuracy.REFS:
        for k in accuracy.GATE_METRICS:
            name = f"{ref}_{k}"
            if name in acc["mean"]:
                out[f"{who}_{name}_mean"] = acc["mean"][name]
                out[f"{who}_{name}_std"] = acc["std"][name]
        if f"{ref}_allclose" in acc["mean"]:
            out[f"{who}_{ref}_allclose_mean"] = acc["mean"][f"{ref}_allclose"]
    return out


def _accounting(enum_counts, stats, enum_errors, kind):
    """Per source and backend: candidates enumerated, tried, passed, failed to compile.

    A case whose fused_moe source gave nothing, whose generators raised, or whose
    backend had every tried candidate fail to compile is ``bar_incomplete``: its bar
    cannot be trusted and stays out of "won" verdicts until it is rerun.
    """
    acct, reasons = {}, []
    for k in sorted(set(enum_counts) | set(stats)):
        t = dict(stats.get(k, {}))
        acct[k] = {"enumerated": enum_counts.get(k, 0), **t}
    if kind != "bar":
        return acct, ""
    if enum_counts.get("fused_moe:*", 0) == 0:
        reasons.append("fused_moe enumerated zero candidates")
    if enum_errors:
        reasons.append("generator errors: " + "; ".join(enum_errors)[:300])
    for k, t in acct.items():
        total = t.get("total", 0)
        if t.get("unnamed", 0) > 0 and t.get("ok", 0) == 0:
            reasons.append(
                f"{k}: {t['unnamed']} candidates resolved to an unnamed impl "
                "('<lambda>'), the router check cannot tell what ran"
            )
        if t.get("aotmiss", 0) > 0:
            reasons.append(
                f"{k}: {t['aotmiss']} candidates hit no usable AOT cache (FlyDSL not "
                "prebuilt): a harness error, the bar misses them"
            )
        if (
            t.get("enumerated", 0) > 0
            and total > 0
            and t.get("ok", 0) == 0
            and t.get("compile", 0) == total
        ):
            reasons.append(f"{k}: all {total} tried candidates failed to compile")
    return acct, "; ".join(reasons)


def _emit_summary(
    fh, c, S, mode, results, ident, skipped, kind, main_failure=None, accounting=None
):
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
        "cand_accounting": json.dumps(accounting[0]) if accounting else "",
        "bar_incomplete": accounting[1] if accounting else "",
    }
    disp = results.get("dispatch")
    if disp is not None:
        row["dispatch_status"] = "ok" if disp.get("ok") else "fail-or-error"
        if disp.get("ok") and mode in disp["by_mode"]:
            row["dispatch_us"] = disp["by_mode"][mode]["median_us"]
        for k in ("route_key", "kernel1", "kernel2", "route_family"):
            row["dispatch_" + k] = disp.get("route", {}).get(k, "")
        row.update(_gate("dispatch", disp.get("acc"), disp.get("numerics", "")))
        if disp.get("rounds_more"):
            row["dispatch_rounds_more"] = disp["rounds_more"]
            if not disp.get("ok"):
                row["dispatch_status"] = "rounds-more"
    lr = results.get("dispatch_bf16bound")
    if lr is not None:
        row["lr_status"] = "ok" if lr.get("ok") else lr.get("outcome", "fail-or-error")
        if lr.get("ok") and mode in lr["by_mode"]:
            row["lr_us"] = lr["by_mode"][mode]["median_us"]
            row["lr_id"] = lr["cand"].id
            row["lr_kernels"] = lr["cand"].kernels
            row["lr_confirmed"] = lr["cand"].confirmed
        row.update(_gate("lr", lr.get("acc"), lr.get("numerics", "")))
    if kind in ("new", "off"):
        who = kind
        if ok:
            r = next(iter(ok.values()))
            row.update(
                {
                    f"{who}_us": r["by_mode"][mode]["median_us"],
                    f"{who}_id": r["cand"].id,
                    f"{who}_kernel_id": getattr(r["cand"], "kernel_id", ""),
                    f"{who}_host_us": r.get("host_us", ""),
                    f"{who}_reason": getattr(r["cand"], "reason", ""),
                    "route_changed": int(getattr(r["cand"], "route_changed", False)),
                    "status": "ok",
                }
            )
            row.update(_gate(who, r.get("acc"), r.get("numerics", "")))
        elif any(r.get("outcome") == "noref" for r in results.values()):
            # a reference that could not be built is its own state, never a pass; a
            # routed row (served by a kernel, not aiter) fails the gate
            kid = next(
                getattr(r["cand"], "kernel_id", "")
                for r in results.values()
                if r.get("outcome") == "noref"
            )
            row.update(
                status="no-reference",
                **{f"{who}_kernel_id": kid},
                gate_fail="1" if kid not in ("", "aiter") else "",
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
        row.update(_gate("bar", best.get("acc"), best.get("numerics", "")))
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
    from . import prebuild

    prebuild.forbid_builds()
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.group).read_text())
    from .schema import Case, parse_extra

    cases = [Case(r, parse_extra(r.get("extra", ""))) for r in spec["cases"]]
    with open(a.out, "a") as fh:
        emit(
            fh,
            {
                "event": "worker_start",
                "pid": os.getpid(),
                "cases": len(cases),
                "harness": os.path.realpath(os.path.dirname(__file__)),
            },
        )
        run_bar_group(cases, spec, fh)
        emit(fh, {"event": "worker_done"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
