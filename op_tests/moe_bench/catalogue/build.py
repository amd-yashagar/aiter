import glob
import os
from collections import OrderedDict

import pandas as pd

from .backend import backend as backend_of

R = ""
OP = ""


def configure(aiter_root):
    """Point every table and test lookup at one aiter tree."""
    global R, OP
    R = os.path.join(aiter_root, "aiter", "configs") + os.sep
    OP = os.path.join(aiter_root, "op_tests") + os.sep


COLS = [
    "source",
    "model",
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
    "extra",
    "kernel1",
    "kernel2",
    "backend1",
    "backend2",
]
LEGACY_GFX = {"256": "gfx950", "80": "gfx942", "228": "gfx942", "304": "gfx942"}
CU_NAME = {
    ("gfx942", "304"): "MI300X/MI325X",
    ("gfx942", "80"): "MI308X",
    ("gfx942", "228"): "MI300A",
    ("gfx950", "256"): "MI355X",
}


def strip(v):
    v = str(v).strip()
    for p in ("torch.", "ActivationType.", "QuantType.", "GateMode."):
        v = v.removeprefix(p)
    return v


MODEL_BY_FILE = [
    ("glm5_3_flash", "GLM-5.3-Flash"),
    ("glm53flash", "GLM-5.3-Flash"),
    ("glm5_3", "GLM-5.3"),
    ("glm53", "GLM-5.3"),
    ("glm5_1", "GLM-5.1"),
    ("glm5", "GLM-5.x"),
    ("glm47", "GLM-4.7"),
    ("minimax-m2_5", "MiniMax-M2.5"),
    ("minimax_m25", "MiniMax-M2.5"),
    ("minimax_m3", "MiniMax-M3"),
    ("qwen3_8_flash_next", "Qwen3.8-Flash-Next"),
    ("qwen38_flash_next", "Qwen3.8-Flash-Next"),
    ("qwen3_8_flash", "Qwen3.8-Flash"),
    ("qwen3_8_2400b", "Qwen3.8-2400B"),
    ("qwen3next_80b", "Qwen3-Next-80B-A3B"),
    ("qwen3_vl", "Qwen3-VL"),
    ("qwen3_235b", "Qwen3-235B-A22B"),
    ("qwen3_5_397b", "Qwen3.5-397B-A17B"),
    ("qwen3_5_35b", "Qwen3.5-35B-A3B"),
    ("qwen3_6_35b", "Qwen3.6-35B-A3B"),
    ("kimik2", "Kimi-K2"),
    ("kimik3", "Kimi-K3"),
    ("k2_horizon", "K2-Horizon-375B"),
    ("dsv41", "DeepSeek-V4.1(hidden5120)"),
    ("dsv4_", "DeepSeek-V4"),
    ("dsv3", "DeepSeek-V3/R1"),
    ("gptoss", "gpt-oss"),
    ("hunyuan3", "Hunyuan3"),
    ("mimo_v2_5_pro", "MiMo-V2.5-Pro"),
    ("mistral_large4", "Mistral-Large-4"),
    ("muse-spark", "Muse-Spark-Open"),
    ("beam_", "Beam(?)"),
]
# (model_dim, routed experts, routed topk) -> model, used for generic tables and tests ('?' = inferred)
SIG = {
    (7168, 256, 8): "DeepSeek-V3/R1",
    (7168, 384, 8): "Kimi-K2",
    (7168, 384, 6): "DeepSeek-V4-Pro",
    (4096, 256, 6): "DeepSeek-V4-Flash",
    (5120, 384, 6): "DeepSeek-V4.1(hidden5120)",
    (4096, 128, 8): "Qwen3-235B-A22B",
    (2048, 128, 8): "Qwen3-30B-A3B",
    (5120, 128, 1): "Llama4-Maverick",
    (5120, 16, 1): "Llama4-Scout",
    (2048, 512, 10): "Qwen3-Next-80B-A3B",
    (4096, 512, 10): "Qwen3.5-397B-A17B",
    (2048, 256, 8): "Qwen3.5-35B-A3B",
    (6144, 256, 8): "GLM-5.x",
    (4096, 288, 8): "GLM-5.3-Flash",
    (5120, 160, 8): "GLM-4.7",
    (3072, 256, 8): "MiniMax-M2.5",
    (6144, 128, 4): "MiniMax-M3/Mistral-Large-4",
    (3072, 128, 4): "gpt-oss-120b(padded 2880->3072)",
    (3584, 896, 16): "Kimi-K3",
    (2560, 512, 10): "Qwen3.8-Flash",
    (8192, 512, 10): "Qwen3.8-2400B",
    (6144, 8, 2): "Mixtral-8x22B|Grok-1",
    (4096, 192, 8): "Hunyuan3?",
    (6144, 192, 8): "K2-Horizon-375B",
    (6144, 384, 8): "MiMo-V2.5-Pro?",
    (5120, 32, 8): "Beam(?)",
    (2304, 8, 2): "unknown(2304/1536/8/2 gelu)",
}


def sig_model(md, E, k, shared_guess=True):
    try:
        md, E, k = int(md), int(E), int(k)
    except (TypeError, ValueError):
        return ""
    if k < 0:  # gfx1250 grouped tables use topk=-1 as a wildcard
        for (smd, sE, sk_), name in SIG.items():
            for ep_div in (1, 2, 4, 8, 16):
                for sh in (0, 1):
                    if smd == md and sE == (E - sh) * ep_div:
                        return name + "(EP?)"
        return ""
    for sh in (0, 1, 2):
        m = SIG.get((md, E - sh, k - sh))
        if m:
            return m
    for ep_div in (2, 4, 8, 16):  # EP shards: local experts * ep == global experts
        for sh in (0, 1):
            for ks in (0, 1, 2):
                m = SIG.get((md, (E - sh) * ep_div, k - ks))
                if m:
                    return m + "(EP?)"
    return ""


def model_for_file(rel):
    base = os.path.basename(rel)
    if (
        rel in ("tuned_fmoe.csv", "untuned_fmoe.csv")
        or "ds_v3" in base
        or "swiglu_limit" in base
        or base.startswith("dsv4_")
    ):
        return None
    for key, name in MODEL_BY_FILE:
        if key in base:
            return name
    return None


def rec(**kw):
    r = {c: "" for c in COLS}
    ident = kw.pop("ident", {})
    r.update(kw)
    r["_ident"] = ident
    return r


def b01(v):
    s = str(v).strip().lower()
    return "1" if s in ("1", "true", "1.0") else "0"


def norm_q(v):
    s = strip(v)
    return "None" if s in ("None", "none", "") else s


# ---------------------------------------------------------------- tables
def load_tuned_tables():
    out = []
    files = [
        R + "tuned_fmoe.csv",
        R + "tuned_swiglu_limit_fmoe.csv",
        R + "tuned_fhmoe.csv",
        R + "tuned_grouped_fmoe.csv",
    ]
    files += sorted(glob.glob(R + "model_configs/*tuned*fmoe*.csv"))
    files = [f for f in files if "untuned" not in os.path.basename(f)]
    for f in files:
        rel = os.path.relpath(f, R)
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        for _, row in d.iterrows():
            out.append((rel, row.to_dict()))
    return out


def table_record(rel, row, tuned=True):
    cu = row.get("cu_num", "")
    gfx = row.get("gfx", "")
    if gfx in ("", "0", "nan"):
        gfx = LEGACY_GFX.get(cu, "?") if cu else ""
    kn1 = row.get("kernelName1", "")
    kn2 = row.get("kernelName2", "")
    model = model_for_file(rel)
    if model is None:
        model = sig_model(row["model_dim"], row["expert"], row["topk"])
        model = (model + "?") if model else "?unknown"
    ident = {}
    extra = OrderedDict()
    if gfx:
        extra["gfx"] = gfx
        ident["gfx"] = gfx
    for k in (
        "shared_expert_id",
        "hidden_pad",
        "intermediate_pad",
        "gate_mode",
        "ep_fused",
    ):
        if row.get(k, "") not in ("", "nan"):
            v = strip(row[k])
            if k in ("hidden_pad", "intermediate_pad") and v == "0":
                continue
            extra[k] = v
            ident[k] = v
    if (
        row.get("shared_expert_id", "") not in ("", "nan")
        and int(float(row["shared_expert_id"])) >= 0
    ):
        extra["shared"] = 1
    tag = row.get("_tag", "")
    if tag not in ("", "nan"):
        extra["tag"] = tag
        ident["tag"] = tag
    elif rel.startswith("tuned_swiglu") or "swiglu_limit_fmoe" in rel:
        pass
    if "swiglu_limit_fmoe" in rel:
        extra["table"] = "swiglu_limit"
        ident["table"] = "swiglu_limit"
    if "grouped" in rel:
        extra["table"] = "grouped_gfx1250"
        ident["table"] = "grouped"
    if tuned:
        for k in ("block_m", "ksplit", "run_1stage", "xbf16", "flat", "nt"):
            v = row.get(k, "")
            if v not in ("", "nan", "0") or (k == "run_1stage" and v == "0"):
                if k in ("ksplit", "xbf16", "flat", "nt") and v in ("0", ""):
                    continue
                extra[k] = v
        if row.get("us", "") not in ("", "nan"):
            extra["us"] = row["us"]
        if row.get("comm_fused_configs", "") not in ("", "nan"):
            extra["comm_fused_configs"] = row["comm_fused_configs"]
        if "grouped" in rel:
            extra["tile"] = "{}x{}x{}/{}x{}x{}".format(
                row.get("tile_m"),
                row.get("tile_n"),
                row.get("tile_k"),
                row.get("tile_m2"),
                row.get("tile_n2"),
                row.get("tile_k2"),
            )
        # replay marker used by test_moe_2stage._iter_csv_cases
        if not tag.strip() or tag in ("nan",):
            fly = "flydsl_" in kn1 or "flydsl_" in kn2
            opus = kn1.startswith("opus_moe1_")
            asm_mx = (
                kn1.startswith("_ZN5aiter") or row.get("flat") == "POST_REDUCE"
            ) and strip(row["q_type"]) == "per_1x32"
            if (fly or opus or asm_mx) and "grouped" not in rel and "fhmoe" not in rel:
                extra["t2s_csv_replay"] = 1
    r = rec(
        source=rel,
        model=model,
        cu_num=cu,
        token=row["token"],
        model_dim=row["model_dim"],
        inter_dim=row["inter_dim"],
        expert=row["expert"],
        topk=row["topk"],
        act_type=strip(row["act_type"]),
        dtype=strip(row["dtype"]),
        q_dtype_a=norm_q(row["q_dtype_a"]),
        q_dtype_w=norm_q(row["q_dtype_w"]),
        q_type=strip(row["q_type"]),
        use_g1u1=b01(row["use_g1u1"]),
        doweight_stage1=b01(row["doweight_stage1"]),
        extra=";".join(f"{k}={v}" for k, v in extra.items()),
        kernel1=kn1 if tuned else "",
        kernel2=kn2 if tuned else "",
        backend1=backend_of(kn1) if tuned else "",
        backend2=(
            (backend_of(kn2) or ("none(1stage-or-fused)" if kn1 else ""))
            if tuned
            else ""
        ),
        ident=ident,
    )
    if r["backend1"] == "asm" and r["kernel1"] and not r["kernel1"].startswith("_ZN"):
        pass
    return r


def side_tables():
    out = []
    # tuned_flydsl_moe_a16w4.csv
    d = pd.read_csv(R + "tuned_flydsl_moe_a16w4.csv", dtype=str)
    for _, r in d.iterrows():
        tiles = f"bm{r.block_m}/s1n{r.stage1_n}k{r.stage1_k}/s2n{r.stage2_n}k{r.stage2_k}/reuse{r.reuse_lds}/kw{r.stage1_k_wave}/pf{r.stage1_prefetch}"
        out.append(
            rec(
                source="tuned_flydsl_moe_a16w4.csv",
                model="DeepSeek-V4.1(hidden5120) TP8?",
                cu_num=r.cu_num,
                token=r.token,
                model_dim=r.hidden_dim,
                inter_dim=r.inter_dim,
                expert=r.experts,
                topk=r.topk,
                act_type="silu_clamped",
                dtype="bfloat16",
                q_dtype_a="bfloat16",
                q_dtype_w="float4_e2m1fn_x2",
                q_type="per_1x32",
                use_g1u1="1",
                doweight_stage1="0",
                extra=f"gfx={r.gfx};logical_inter={r.logical_inter};swiglu_limit={r.swiglu_limit};w_layout={r.w_layout};activation_bf16=1;table=flydsl_a16w4",
                kernel1="flydsl_a16w4_gemm1(" + tiles + ")",
                kernel2="flydsl_a16w4_gemm2(" + tiles + ")",
                backend1="flydsl",
                backend2="flydsl",
                ident={
                    "gfx": r.gfx,
                    "table": "a16w4",
                    "swiglu_limit": r.swiglu_limit,
                    "logical_inter": r.logical_inter,
                },
            )
        )
    d = pd.read_csv(R + "mega_moe_tp_tuned.csv", dtype=str)
    for _, r in d.iterrows():
        mdl = {"6144/768/129/5": "MiniMax-M3 TP4"}.get(
            f"{r.model_dim}/{r.inter_dim}/{r.experts}/{r.topk}", "?"
        )
        out.append(
            rec(
                source="mega_moe_tp_tuned.csv",
                model=mdl,
                cu_num=r.cu_num,
                token=r.tot,
                model_dim=r.model_dim,
                inter_dim=r.inter_dim,
                expert=r.experts,
                topk=r.topk,
                act_type="",
                dtype="bfloat16",
                q_dtype_a="float4_e2m1fn_x2",
                q_dtype_w="float4_e2m1fn_x2",
                q_type="per_1x32",
                use_g1u1="1",
                doweight_stage1="0",
                extra=f"gfx={r.gfx};tp={r.tp};comm_mode={r.comm_mode};lb={r.lb};mt={r.mt};npp={r.npp};q={r.q};pf={r.pf};lp={r.lp};ll={r.ll};llr={r.llr};us={r.us};table=mega_moe_tp",
                kernel1="mega_moe_tp(single fused kernel)",
                kernel2="",
                backend1="flydsl",
                backend2="",
                ident={
                    "gfx": r.gfx,
                    "table": "mega_tp",
                    "tp": r.tp,
                    "comm_mode": r.comm_mode,
                    "lb": r.lb,
                },
            )
        )
    d = pd.read_csv(R + "mega_moe_tp_bigm_tuned.csv", dtype=str)
    for _, r in d.iterrows():
        mdl = {
            "6144/768/129/5": "MiniMax-M3 TP4",
            "4096/512/257/6": "?(4096/512/E257/top6)",
            "6144/512/257/9": "GLM-5.x TP4?",
        }.get(f"{r.model_dim}/{r.inter_dim}/{r.experts}/{r.topk}", "?")
        out.append(
            rec(
                source="mega_moe_tp_bigm_tuned.csv",
                model=mdl,
                cu_num=r.cu_num,
                token=r.tokens,
                model_dim=r.model_dim,
                inter_dim=r.inter_dim,
                expert=r.experts,
                topk=r.topk,
                act_type="Swiglu|Silu",
                dtype="bfloat16",
                q_dtype_a="float4_e2m1fn_x2",
                q_dtype_w="float4_e2m1fn_x2",
                q_type="per_1x32",
                use_g1u1="1",
                doweight_stage1="0",
                extra=f"gfx={r.gfx};tp={r.tp};chunks={r.chunks};us={r.us};q_rel={r.q_rel};table=mega_moe_tp_bigm;opts={r.opts}",
                kernel1="mega_moe_tp_bigM(chunked)",
                kernel2="",
                backend1="flydsl",
                backend2="",
                ident={"gfx": r.gfx, "table": "mega_bigm", "tp": r.tp},
            )
        )
    return out


def untuned_tables():
    out = []
    files = [
        R + "untuned_fmoe.csv",
        R + "untuned_swiglu_limit_fmoe.csv",
        R + "untuned_fhmoe.csv",
        R + "untuned_grouped_fmoe.csv",
    ]
    files += sorted(glob.glob(R + "model_configs/*untuned*fmoe*.csv"))
    for f in files:
        rel = os.path.relpath(f, R)
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        for _, row in d.iterrows():
            row = row.to_dict()
            out.append((rel, row))
    return out
