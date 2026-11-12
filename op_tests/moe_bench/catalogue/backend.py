def backend(k):
    k = (k or "").strip()
    if k == "":
        return ""
    if k == "Null":
        return "none(1stage)"
    if k.startswith("fmoe_"):
        return "asm"
    if k.startswith("_ZN5aiter"):
        return "asm"
    if k.startswith("moe_ck2stages"):
        return "ck"
    if k.startswith("cktile_") or k.endswith("_cktile"):
        return "ck_tile"
    if k.startswith("opus_"):
        return "opus"
    if k.startswith("impl__flydsl_"):
        return "flydsl"
    if k.startswith("flydsl_"):
        return "flydsl"
    if k.startswith("grouped_gemm"):
        return "flydsl"
    if "triton" in k:
        return "triton"
    return "unknown"
