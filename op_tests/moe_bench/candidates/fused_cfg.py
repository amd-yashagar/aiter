# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every ``fused_moe`` kernel candidate, forced through a one-row ``AITER_CONFIG_FMOE``.

Candidates come from aiter's own tuner generators in
``csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`` (``FmoeTuner.gen_*_task``: ASM
1-stage, ASM stage1 + CK, CK 2-stage instances, ck_tile, FlyDSL stage pairs, Opus) plus
the gfx942 FlyDSL whole-graph configs (``get_tune_space``). Stage candidates are paired by
the tuner's own merge key (block_m, nt, v2, a16w16). Nothing is hand-listed.

Screening: the tuner's stage timers run isolated stages; here a stage is screened by its
marginal end-to-end effect instead. Each stage-1 candidate is timed with a fixed working
stage-2 of the same block_m, then each stage-2 candidate with the best stage-1; the sum
of marginals ranks the pairs, and the top few are confirmed end to end by the worker.
"""

import csv
import importlib.util
import os
import re
import sys

from . import Candidate, Source, register

BLOCK_MS = [16, 32, 64, 128]
CONFIRM_PAIRS = 4
CONFIRM_SINGLES = 3
_TUNER = {}
CSV_COLS = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
    "block_m",
    "ksplit",
    "us1",
    "kernelName1",
    "err1",
    "us2",
    "kernelName2",
    "err2",
    "us",
    "run_1stage",
    "xbf16",
    "flat",
    "tflops",
    "bw",
    "_tag",
]


def tuner():
    """The aiter FmoeTuner, imported from the tree under test (generators only)."""
    if "t" in _TUNER:
        return _TUNER["t"]
    import aiter

    root = os.path.dirname(os.path.dirname(os.path.abspath(aiter.__file__)))
    path = os.path.join(root, "csrc", "ck_gemm_moe_2stages_codegen", "gemm_moe_tune.py")
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("gemm_moe_tune", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gemm_moe_tune"] = mod
    spec.loader.exec_module(mod)
    key = [
        "gfx",
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]
    result = [
        "block_m",
        "ksplit",
        "us1",
        "kernelName1",
        "err1",
        "us2",
        "kernelName2",
        "err2",
        "us",
        "run_1stage",
        "xbf16",
        "flat",
        "tflops",
        "bw",
    ]
    argv, sys.argv = sys.argv, ["gemm_moe_tune"]
    try:
        _TUNER["t"] = mod.FmoeTuner("fmoeTuner", key, result + ["nt"], "fmoe tuner")
    finally:
        sys.argv = argv
    return _TUNER["t"]


def force_config(row, tmpdir):
    """Make aiter's router read exactly this one tuned row."""
    import aiter.fused_moe as fm
    from aiter.jit.core import AITER_CONFIGS

    path = os.path.join(tmpdir, "force_fmoe.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerow({c: row.get(c, "") for c in CSV_COLS})
    os.environ["AITER_CONFIG_FMOE"] = path
    reset_router(fm, AITER_CONFIGS)


def reset_router(fm=None, configs=None):
    if fm is None:
        import aiter.fused_moe as fm
    if configs is None:
        from aiter.jit.core import AITER_CONFIGS as configs
    configs.get_config_file.cache_clear()
    fm.cfg_2stages = None
    fm.get_2stage_cfgs.cache_clear()


def unforce():
    os.environ.pop("AITER_CONFIG_FMOE", None)
    reset_router()


def _a16w16(name):
    return str(name).startswith(
        ("flydsl_moe1_abf16_wbf16_", "flydsl_moe2_abf16_wbf16_")
    )


class ForcedConfig(Candidate):
    source = "fused_moe"
    entry = "aiter.fused_moe.fused_moe(AITER_CONFIG_FMOE one-row)"

    def __init__(self, cid, kernels, row, group, meta=None):
        super().__init__(cid, kernels, "default", meta)
        self.row = row
        self.group = group

    def prepare(self, ctx):
        row = dict(self.row)
        a = ctx.dispatch_args
        row.update(
            gfx=ctx.dispatch_meta["gfx"],
            cu_num=ctx.dispatch_meta["cu_num"],
            token=a["token"],
            model_dim=a["model_dim"],
            inter_dim=a["inter_dim"],
            expert=a["expert"],
            topk=a["topk"],
            act_type=str(a["activation"]),
            dtype=str(a["dtype"]),
            q_dtype_a=str(a["q_dtype_a"]),
            q_dtype_w=str(a["q_dtype_w"]),
            q_type=str(a["q_type"]),
            use_g1u1=int(a["use_g1u1"]),
            doweight_stage1=int(a["doweight_stage1"]),
        )
        force_config(row, ctx.tmpdir)

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
        unforce()

    def verify(self, route):
        """None when the router used this candidate, else why not."""
        want1 = self.row["kernelName1"]
        want2 = self.row.get("kernelName2") or ""
        got1 = route.get("kernel1", "") or route.get("full_impl", "")
        got2 = route.get("kernel2", "") or ""
        if want1 and want1 not in (got1, route.get("full_impl", "")):
            return f"router resolved {got1!r}, not {want1!r}"
        if want2 and want2 != got2:
            return f"router resolved stage2 {got2!r}, not {want2!r}"
        return None


@register
class FusedCfgSource(Source):
    name = "fused_moe"

    def skip_reason(self, ctx):
        if ctx.S.kind != "fused":
            return "not a fused MoE case"
        return None

    def enumerate(self, ctx):
        a = ctx.dispatch_args
        info = (
            ctx.dispatch_meta["gfx"],
            ctx.dispatch_meta["cu_num"],
            a["token"],
            a["model_dim"],
            a["inter_dim"],
            a["expert"],
            a["topk"],
            a["activation"],
            a["dtype"],
            a["q_dtype_a"],
            a["q_dtype_w"],
            a["q_type"],
            bool(a["use_g1u1"]),
            bool(a["doweight_stage1"]),
        )
        t = tuner()
        s1, s2, ones = {}, {}, []
        errors = []
        for g in (
            "gen_2stages_asm1_task",
            "gen_2stages_task",
            "gen_flydsl_2stages_task",
            "gen_flydsl_v2_2stages_task",
            "gen_flydsl_i4_2stages_task",
            "gen_opus_2stages_task",
        ):
            try:
                tasks = getattr(t, g)(info, BLOCK_MS)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{g}: {type(e).__name__}: {str(e)[:100]}")
                continue
            for task in tasks:
                _, stage, kname, bm, *extra = task[0]
                flat = int(extra[0]) if len(extra) > 0 else 0
                v2 = int(extra[1]) if len(extra) > 1 else 0
                nt = int(bool(extra[2])) if len(extra) > 2 else 0
                rec = (kname, int(bm), flat, v2, nt)
                dst = s1 if stage == "stage1" else s2 if stage == "stage2" else None
                if dst is not None and rec not in dst.setdefault(
                    (int(bm), nt, v2, _a16w16(kname)), []
                ):
                    dst[(int(bm), nt, v2, _a16w16(kname))].append(rec)
        try:
            for task in t.gen_1stage_asm_task(info):
                _, stage, kname, bm, *extra = task[0]
                ones.append((stage, kname, int(bm), int(extra[0]) if extra else 0))
        except Exception as e:  # noqa: BLE001
            errors.append(f"gen_1stage_asm_task: {type(e).__name__}: {str(e)[:100]}")
        cands = []
        for key, lst1 in sorted(s1.items()):
            lst2 = s2.get(key)
            if not lst2:
                continue
            for k1, bm, flat, v2, nt in lst1:
                m = re.search(r"_sk(\d+)$", k1)
                ks = int(m.group(1)) if m else 0
                name1 = re.sub(r"_sk\d+$", "", k1)
                for k2, *_ in lst2:
                    row = {
                        "block_m": bm,
                        "ksplit": ks,
                        "kernelName1": name1,
                        "kernelName2": k2,
                        "run_1stage": 0,
                        "xbf16": 0,
                        "flat": 0,
                        "us1": 0,
                        "us2": 0,
                        "us": 0,
                        "err1": "0%",
                        "err2": "0%",
                        "tflops": 0,
                        "bw": 0,
                    }
                    c = ForcedConfig(
                        f"fused_moe|{name1}|{k2}|bm{bm}|ks{ks}|nt{nt}",
                        f"{name1} + {k2} (block_m={bm}, ksplit={ks})",
                        row,
                        f"pair-bm{bm}-{key[1]}-{key[2]}-{int(key[3])}",
                        {"s1": k1, "s2": k2, "bm": bm},
                    )
                    cands.append(c)
        for stage, kname, bm, flat in ones:
            row = {
                "block_m": bm,
                "ksplit": 0,
                "kernelName1": kname,
                "kernelName2": "",
                "run_1stage": 1,
                "xbf16": int(stage == "asm_1stage_xbf16"),
                "flat": flat,
                "us1": 0,
                "us2": 0,
                "us": 0,
                "err1": "0%",
                "err2": "0%",
                "tflops": 0,
                "bw": 0,
            }
            cands.append(
                ForcedConfig(
                    f"fused_moe|1stage|{kname}|bm{bm}|flat{flat}|{stage}",
                    f"{kname} (1-stage {stage}, block_m={bm}, flat={flat})",
                    row,
                    "1stage",
                )
            )
        cands += self._whole_graph(ctx, info)
        ctx.enum_errors = errors
        return cands

    def _whole_graph(self, ctx, info):
        from aiter import QuantType

        out = []
        if ctx.dispatch_meta["gfx"] != "gfx942" or info[13]:
            return out
        try:
            from aiter.fused_moe_registry import make_fused_moe_impl_kernel_name
            from aiter.ops.flydsl.fused_moe_gfx942 import (
                Config,
                _Problem,
                get_tune_space,
            )
        except Exception:  # noqa: BLE001
            return out
        for cs in get_tune_space():
            config = Config.from_string(cs)
            prob = _Problem(
                batch=ctx.S.M,
                experts=info[5],
                gateup_dim=info[4] * 2,
                hidden_dim=info[3],
                model_dim=info[3],
                inter_dim=info[4],
                topk=info[6],
                quant_type="no" if info[11] == QuantType.No else "",
            )
            if config.unsupported_reason(prob) is not None:
                continue
            k1 = make_fused_moe_impl_kernel_name("flydsl_gfx942", cs)
            row = {
                "block_m": config.BLOCK_M,
                "ksplit": 0,
                "kernelName1": k1,
                "kernelName2": "",
                "run_1stage": 0,
                "xbf16": 0,
                "flat": 0,
                "us1": 0,
                "us2": 0,
                "us": 0,
                "err1": "0%",
                "err2": "0%",
                "tflops": 0,
                "bw": 0,
            }
            out.append(ForcedConfig(f"fused_moe|graph|{cs}", k1, row, "whole-graph"))
        return out

    def screen(self, ctx, cands, runner):
        """Marginal-effect screening of pairs; singles are timed directly."""
        pairs = [c for c in cands if c.group.startswith("pair-")]
        singles = [c for c in cands if not c.group.startswith("pair-")]
        keep = []
        by_bm = {}
        for c in pairs:
            by_bm.setdefault(c.group, []).append(c)
        stage_marg = {}
        best_overall = []
        for lst in by_bm.values():
            s1_names = sorted({c.meta["s1"] for c in lst})
            s2_names = sorted({c.meta["s2"] for c in lst})
            find = {(c.meta["s1"], c.meta["s2"]): c for c in lst}
            base = None
            for k2 in s2_names:
                res = runner.measure(find[(s1_names[0], k2)], quick=True)
                if res["ok"]:
                    base = (s1_names[0], k2, res["us"])
                    break
            if base is None:
                continue
            t1 = {}
            for k1 in s1_names:
                c = find.get((k1, base[1]))
                if c is None:
                    continue
                res = runner.measure(c, quick=True)
                if res["ok"]:
                    t1[k1] = res["us"]
            if not t1:
                continue
            best1 = min(t1, key=t1.get)
            t2 = {}
            for k2 in s2_names:
                c = find.get((best1, k2))
                if c is None:
                    continue
                res = runner.measure(c, quick=True)
                if res["ok"]:
                    t2[k2] = res["us"]
            if not t2:
                continue
            ref = t2.get(base[1], base[2])
            for (k1, k2), c in find.items():
                if k1 in t1 and k2 in t2:
                    stage_marg[c.id] = t1[k1] + t2[k2] - ref
                    best_overall.append(c)
        best_overall.sort(key=lambda c: stage_marg[c.id])
        keep += best_overall[:CONFIRM_PAIRS]
        timed = []
        for c in singles:
            res = runner.measure(c, quick=True)
            if res["ok"]:
                timed.append((res["us"], c))
        timed.sort(key=lambda x: x[0])
        for kind in ("1stage", "whole-graph"):
            keep += [c for _, c in timed if c.group == kind][:CONFIRM_SINGLES]
        ctx.screen_us = stage_marg
        return keep
