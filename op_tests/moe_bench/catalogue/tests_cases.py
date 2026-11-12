import ast
import itertools
from pathlib import Path

from . import build
from .build import rec, sig_model

BF = "bfloat16"


def mk(
    source,
    token,
    md,
    idim,
    E,
    topk,
    act="Silu",
    dtype=BF,
    qa="bfloat16",
    qw="bfloat16",
    qt="No",
    g1u1="1",
    dw1="0",
    extra="",
    model=None,
    k1="",
    k2="",
    ident=None,
):
    if model is None:
        model = sig_model(md, E, topk) if md not in ("", None) else ""
        model = (model + "?") if model else ""
    return rec(
        source=source,
        model=model,
        cu_num="",
        token=str(token),
        model_dim=str(md),
        inter_dim=str(idim),
        expert=str(E),
        topk=str(topk),
        act_type=act,
        dtype=dtype,
        q_dtype_a=qa,
        q_dtype_w=qw,
        q_type=qt,
        use_g1u1=g1u1,
        doweight_stage1=dw1,
        extra=extra,
        kernel1=k1,
        kernel2=k2,
        backend1="",
        backend2="",
        ident=ident or {},
    )


FP8, FP4, I8, I4, I4X2 = "fp8", "float4_e2m1fn_x2", "int8", "int4", "i4x2"
# quant combos of test_moe_2stage.l_quant: (q_type, a dtype, w dtype)
L_QUANT = [
    ("No", "None", "None"),
    ("per_Tensor", FP8, FP8),
    ("per_Token", FP8, FP8),
    ("per_Token", FP8, I4),
    ("per_1x32", FP4, FP4),
    ("per_128x128", FP8, FP8),
    ("per_1x32", BF, FP4),
    ("per_1x32", FP8, FP4),
    ("per_1x32", BF, I4X2),
    ("per_1x32", FP8, FP8),
]
BF = "bfloat16"
QUANT_NAMES = {
    0: "a16w16",
    1: "a8w8-pertensor",
    2: "a8w8-pertoken",
    3: "a8w4-int4",
    4: "a4w4-mxfp4",
    5: "a8w8-128x128",
    6: "a16w4-mxfp4",
    7: "a8w4-mxfp4",
    8: "a16wi4",
    9: "mxfp8",
}


def gen_tests():
    out = []
    add = out.append

    # ------------------------------------------------------------ test_moe_2stage.py legacy sweep
    s = "test_moe_2stage.py:legacy_sweep"
    tokens = [1, 3, 5, 16, 32, 64, 128, 256, 1024, 4096, 8192, 163840]
    for qi, (qt, qa, qw) in enumerate(L_QUANT):
        qa_ = BF if qa == "None" and qt == "No" else qa
        qw_ = BF if qw == "None" and qt == "No" else qw
        extra = [
            "op=fused_moe",
            f"combo={QUANT_NAMES[qi]}",
            "preshuffle=1",
            "cli_default=-dim 7168,256 -e 257 -k 9",
        ]
        act, hp, ip, ident = "Silu", 0, 0, {}
        bias = False
        if (qt, qa, qw) == ("per_1x32", BF, FP4):
            act, hp, ip = "Swiglu", 192, 128
            bias = True
            extra.append("gate_mode=SEPARATED")
        elif (qt, qa, qw) == ("per_1x32", FP8, FP4):
            hp, ip = 192, 128
            bias = True
            extra.append("gate_mode=INTERLEAVE")
        elif (qt, qa, qw) == ("per_1x32", FP8, FP8):
            extra.append("gate_mode=INTERLEAVE")
        if qt == "per_1x32":
            extra.append("gfx950_only=1")
        if hp:
            extra += [f"hidden_pad={hp}", f"intermediate_pad={ip}"]
            ident = {"hidden_pad": str(hp), "intermediate_pad": str(ip)}
        extra.append("bias=s1+s2" if bias else "bias=none")
        for t in tokens:
            add(
                mk(
                    s,
                    t,
                    7168,
                    256,
                    257,
                    9,
                    act=act,
                    qa=qa_,
                    qw=qw_,
                    qt=qt,
                    extra=";".join(extra),
                    ident=ident,
                    model="DeepSeek-V3/R1 (default shape; shared expert)",
                )
            )
    s = "test_moe_2stage.py:blockscale_swiglu_limit"
    for t, lim in [
        (1, "1.0"),
        (1, "7.0"),
        (1, "None"),
        (32, "1.0"),
        (32, "7.0"),
        (32, "None"),
        (1024, "1.0"),
        (1024, "7.0"),
    ]:
        add(
            mk(
                s,
                t,
                256,
                256,
                8,
                2,
                qa=FP8,
                qw=FP8,
                qt="per_128x128",
                extra=f"op=fused_moe;swiglu_limit={lim};gfx942|gfx950",
                model="synthetic",
            )
        )
    s = "test_moe_2stage.py:presorted_moe_hook"
    for t in (1, 4, 512):
        add(
            mk(
                s,
                t,
                3584,
                384,
                896,
                16,
                act="Situv2",
                qa=FP4,
                qw=FP4,
                qt="per_1x32",
                extra="op=fused_moe;presorted_BM16;gate_mode=SEPARATED;gfx950_only=1;timed:moe_sorting|presorted|presorted+output=",
            )
        )
    s = "test_moe_2stage.py:bm16_tiled_scale_boundary"
    for t in (32, 33, 64):
        add(
            mk(
                s,
                t,
                7168,
                512,
                385,
                7,
                qa=FP8,
                qw=FP4,
                qt="per_1x32",
                extra="op=fused_moe;gate_mode=INTERLEAVE;gfx950_only=1;AITER_BF16_FP8_MOE_BOUND=0",
                model="DeepSeek-V4-Pro? TP4 +shared",
            )
        )
    s = "test_moe_2stage.py:swiglu_mxfp4_nonsplit_nobias_dispatch"
    for gu in ("GUGU", "GGUU"):
        for t in (128, 255, 256, 257, 512):
            add(
                mk(
                    s,
                    t,
                    6144,
                    1536,
                    129,
                    5,
                    act="Swiglu",
                    qa=(FP8 if t >= 256 else BF),
                    qw=FP4,
                    qt="per_1x32",
                    extra=f"op=get_2stage_cfgs(dispatch only);layout={gu};gate_mode=INTERLEAVE;gfx950_only=1",
                    model="MiniMax-M3?",
                )
            )
    s = "test_moe_2stage.py:swiglu_mxfp4_untagged_layout_dispatch"
    for t in (1, 8, 32, 128):
        add(
            mk(
                s,
                t,
                6144,
                512,
                129,
                5,
                act="Swiglu",
                qa=BF,
                qw=FP4,
                qt="per_1x32",
                extra="op=get_2stage_cfgs(dispatch only);swiglu_limit=7.0;gate_mode=SEPARATED;gfx950_only=1",
                model="MiniMax-M3?",
            )
        )
    add(
        mk(
            "test_moe_2stage.py:output_buffer_contract",
            32,
            512,
            256,
            8,
            2,
            extra="op=fused_moe;output=buffer;a16w16;no-quant",
            model="synthetic",
        )
    )
    add(
        mk(
            "test_moe_2stage.py:swiglu_oai_flat_dispatch",
            "1,2,4,8,16",
            6144,
            768,
            129,
            5,
            act="Swiglu",
            qa=BF,
            qw=FP4,
            qt="per_1x32",
            extra="op=get_2stage_cfgs(dispatch only);gfx950 256CU only;asm FLAT 16x32/16x128 vs flydsl",
            model="MiniMax-M3 TP4",
        )
    )

    # ------------------------------------------------------------ test_moe.py (legacy asm_moe)
    s = "test_moe.py"
    combos = [
        ("test_fmoe_16_bit", "No", "0", "No", BF, BF, 32, 5),
        ("g1u1_no_quant", "No", "1", "No", BF, BF, 32, 5),
        ("g1u1_int8quant", "int8quant", "1", "per_Token", I8, I8, 32, 5),
        ("g1u1_fp8quant", "fp8quant", "1", "per_Token", FP8, FP8, 32, 5),
        ("g1u0_int8smoothquant", "int8smoothquant", "0", "per_Token", I8, I8, 32, 5),
        ("g1u1_int8smoothquant", "int8smoothquant", "1", "per_Token", I8, I8, 32, 5),
        ("g1u1_fp8smoothquant", "fp8smoothquant", "1", "per_Token", FP8, FP8, 32, 5),
        ("g1u1_int4", "wint4afp8smoothquant", "1", "per_Token", FP8, I4, 8, 3),
    ]
    for name, qn, g1u1, qt, qa, qw, E, k in combos:
        ex = f"op=asm_moe(legacy fused_moe_bf16_asm);test={name};quant_algo={qn};ref=torch_moe"
        if name in ("g1u0_int8smoothquant", "g1u1_int4"):
            ex += ";gfx942_only=1"
        if qn.endswith("smoothquant"):
            ex += ";smooth_scale=1"
        add(
            mk(
                s,
                128,
                4096,
                1024,
                E,
                k,
                qa=qa,
                qw=qw,
                qt=qt,
                g1u1=g1u1,
                extra=ex,
                model="synthetic",
            )
        )

    # ------------------------------------------------------------ test_moe_tkw1.py
    for t in (1, 128, 256):
        for E in (16, 128):
            add(
                mk(
                    "test_moe_tkw1.py",
                    t,
                    5120,
                    1024,
                    E,
                    1,
                    qa=FP8,
                    qw=FP8,
                    qt="per_Token",
                    dw1="1",
                    extra="op=asm_moe_tkw1(legacy);topk_weight_on_stage1;ref=torch_moe_tkw1",
                    model="Llama4-Maverick" if E == 128 else "Llama4-Scout",
                )
            )

    # ------------------------------------------------------------ test_moe_blockscale.py
    for t in (1, 2, 5, 16, 32, 163840):
        add(
            mk(
                "test_moe_blockscale.py",
                t,
                7168,
                256,
                256,
                8,
                qa=FP8,
                qw=FP8,
                qt="per_1x128",
                extra="op=fmoe_fp8_blockscale_g1u1(asm direct, 128x128);ref=torch_moe_blockscale",
                model="DeepSeek-V3/R1 TP8",
            )
        )

    # ------------------------------------------------------------ test_moe_dp_share_expert.py
    for t in (1, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 1024, 8192):
        add(
            mk(
                "test_moe_dp_share_expert.py",
                t,
                5120,
                1536,
                8,
                8,
                qa=I8,
                qw=I8,
                qt="per_Token",
                extra="op=fused_moe_dp_share_expert;share_expert_only(E=topk=8);dp_size=8;smoothquant=optional(-s)",
                model="synthetic(shared experts)",
            )
        )

    # ------------------------------------------------------------ test_moe_ep.py
    s = "test_moe_ep.py"
    for name, E, sh, qt, qa, qw, g1u1 in [
        ("g1u1_no_quant", 32, 2, "No", BF, BF, "1"),
        ("g1u1_int8quant", 32, 2, "per_Token", I8, I8, "1"),
        ("g1u1_fp8quant", 32, 2, "per_Token", FP8, FP8, "1"),
        ("g1u0_int8smoothquant", 32, 2, "per_Token", I8, I8, "0"),
        ("g1u1_int8smoothquant", 32, 0, "per_Token", I8, I8, "1"),
        ("g1u1_fp8smoothquant", 32, 2, "per_Token", FP8, FP8, "1"),
    ]:
        add(
            mk(
                s,
                128,
                4096,
                1024,
                E,
                5,
                qa=qa,
                qw=qw,
                qt=qt,
                g1u1=g1u1,
                extra=f"op=fused_moe(expert_mask EP);test={name};ep=8;shared_E={sh};ref=torch_moe(expert_mask);rtol=0.01,atol=10"
                + (";gfx942_only=1" if name == "g1u0_int8smoothquant" else ""),
                model="synthetic EP",
            )
        )
    for name, qa in (("g1u1_a8w4_mxfp4", FP8), ("g1u1_a4w4_mxfp4", FP4)):
        add(
            mk(
                s,
                128,
                4096,
                1024,
                32,
                5,
                qa=qa,
                qw=FP4,
                qt="per_1x32",
                extra=f"op=fused_moe(expert_mask EP);test={name};ep=8;shared_E=0;ep_mode=real|fake;gfx950_only=1;ref=torch_moe_stage1/2 MXFP4;atol=rtol=5e-2",
                model="synthetic EP",
            )
        )

    # ------------------------------------------------------------ test_moe_cktile_stage1_splitk.py
    s = "test_moe_cktile_stage1_splitk.py"
    for md, (E, k, t) in itertools.product(
        [256, 512, 768, 1024, 1280, 6144], [(4, 2, 4), (128, 4, 16)]
    ):
        add(
            mk(
                s,
                t,
                md,
                256,
                E,
                k,
                act="Swiglu",
                qa=BF,
                qw=FP4,
                qt="per_1x32",
                extra="op=fused_moe(cktile split-k);gate_mode=SEPARATED;gfx950_only=1;rel_l2<MAX_REL_L2",
                model="synthetic",
            )
        )
    for hp in (192, 256):
        add(
            mk(
                s,
                4,
                1024,
                256,
                4,
                2,
                act="Swiglu",
                qa=BF,
                qw=FP4,
                qt="per_1x32",
                extra=f"op=fused_moe(cktile split-k);hidden_pad={hp};gfx950_only=1",
                model="synthetic",
                ident={"hidden_pad": str(hp)},
            )
        )
    for hp in (0, 192):
        add(
            mk(
                s,
                16,
                3072,
                256,
                32,
                4,
                act="Swiglu",
                qa=BF,
                qw=FP4,
                qt="per_1x32",
                extra=f"op=fused_moe(cktile split-k);hidden_pad={hp};bias=s1+s2;gfx950_only=1",
                model="gpt-oss-like(3072,topk4)",
                ident={"hidden_pad": str(hp)},
            )
        )
    for md, sk in [(512, 4), (768, 2), (1024, 4)]:
        add(
            mk(
                s,
                1,
                md,
                256,
                128,
                4,
                act="Silu",
                qa=BF,
                qw=FP4,
                qt="per_1x32",
                extra=f"op=fused_moe(cktile split-k);split_k={sk};gfx950_only=1",
                model="synthetic",
            )
        )

    # ------------------------------------------------------------ test_moe_a16w16_default_dispatch.py
    s = "test_moe_a16w16_default_dispatch.py"
    for idim, bm, dt in itertools.product(
        [192, 256, 320, 384, 448], [32, 64, 128, 256], [BF, "float16"]
    ):
        add(
            mk(
                s,
                33,
                256,
                idim,
                8,
                2,
                dtype=dt,
                qa=dt,
                qw=dt,
                qt="No",
                extra=f"op=fused_moe(AITER_BYPASS_TUNE_CONFIG=1);block_m={bm};ref=fp32 per-expert loop;rtol=0.02,atol=0.002;CK a16w16 alignment",
                model="synthetic",
            )
        )
    for t in (4, 512, 8192):
        add(
            mk(
                s,
                t,
                2560,
                320,
                512,
                10,
                qt="No",
                extra="op=fused_moe(AITER_BYPASS_TUNE_CONFIG=1);gfx942_only=1;MTP TP2/EP1",
                model="Qwen3.8-Flash-Next",
            )
        )
    for idim, bm, qt in itertools.product(
        [384, 512], [32, 64], ["per_Tensor", "per_Token"]
    ):
        add(
            mk(
                s,
                16,
                896,
                idim,
                8,
                2,
                qa=FP8,
                qw=FP8,
                qt=qt,
                extra=f"op=fused_moe;block_m={bm};gfx950_only=1;stage2 K-alignment;rtol=0.05,atol=0.02",
                model="synthetic",
            )
        )
    for idim, bm in itertools.product([384, 512], [32, 64]):
        add(
            mk(
                s,
                16,
                896,
                idim,
                8,
                2,
                qa=I8,
                qw=I8,
                qt="per_Tensor",
                extra=f"op=fused_moe;block_m={bm};gfx950_only=1;int8 stage2 K-alignment",
                model="synthetic",
            )
        )

    # ------------------------------------------------------------ test_moe_glm5_fp4_ep_dispatch.py
    for m, tier in [
        (5, 8),
        (12, 16),
        (24, 32),
        (36, 64),
        (64, 64),
        (8193, 16384),
        (16385, 32768),
        (131072, 131072),
    ]:
        add(
            mk(
                "test_moe_glm5_fp4_ep_dispatch.py",
                tier,
                6144,
                2048,
                65,
                9,
                qa=FP4,
                qw=FP4,
                qt="per_1x32",
                extra=f"op=get_2stage_cfgs(dispatch only);ep=4;is_ep=1;gfx950 256CU(mocked);runtime_m={m}",
                model="GLM-5.x EP4",
            )
        )

    # ------------------------------------------------------------ mxfp4 / mxfp8 dispatch tests
    for idim, want in [
        (128, 1),
        (192, 1),
        (256, 1),
        (320, 1),
        (384, 1),
        (512, 1),
        (640, 1),
        (768, 1),
    ]:
        add(
            mk(
                "test_moe_mxfp4_inter_dim_dispatch.py",
                32,
                6144,
                idim,
                32,
                4,
                act="Swiglu",
                qa=BF,
                qw=FP4,
                qt="per_1x32",
                extra="op=get_2stage_cfgs(dispatch only);gate_mode=SEPARATED;gfx942|gfx950;expects flydsl for 128/384/640, cktile for 192/256/320/512/768",
                model="synthetic",
            )
        )
    for t in (1, 8, 112):
        add(
            mk(
                "test_moe_mxfp8_passthrough.py",
                t,
                2048,
                512,
                8,
                2,
                act="Swiglu",
                qa=FP8,
                qw=FP4,
                qt="per_1x32",
                extra="op=fused_moe(MXFP8 activation passthrough);gate_mode=INTERLEAVE;AITER_BF16_FP8_MOE_BOUND=0;gfx950_only=1;bit-exact vs internal quant",
                model="synthetic",
            )
        )
    for unr in ("padding_rows", "ep_non_local_experts"):
        add(
            mk(
                "test_moe_per_tensor_a2_init.py",
                32,
                1024,
                512,
                8,
                2,
                qa=FP8,
                qw=FP8,
                qt="per_Tensor",
                extra=f"op=fused_moe;unrouted={unr};stale a2 memory",
                model="synthetic",
            )
        )

    # ------------------------------------------------------------ routing / sorting tests (op != fused GEMM)
    s = "test_moe_sorting.py"
    for m, (E, k), pad, em, dp in itertools.product(
        [1, 7, 31, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        [(32, 5), (256, 8), (40, 6), (385, 7)],
        [0, 1000],
        [1, 0],
        [0, 1],
    ):
        add(
            mk(
                s,
                m,
                4096,
                4096,
                E,
                k,
                qa="",
                qw="",
                qt="",
                g1u1="",
                extra=f"op=moe_sorting;padding_extra={pad};expert_mask={em};dispatch_policy={dp};backends=opus|ck|flydsl(dp0);ref=torch loop;atol=0(num_post_pad)",
                model="synthetic",
            )
        )
    s = "test_moe_sorting_mxfp4.py"
    for dim, (E, k), m in itertools.product(
        [4096, 7168],
        [(32, 5), (256, 8), (512, 8)],
        [1, 64, 128, 256, 1024, 2050, 4200, 10000, 163840],
    ):
        add(
            mk(
                s,
                m,
                dim,
                "",
                E,
                k,
                qa="",
                qw=FP4,
                qt="per_1x32",
                g1u1="",
                extra="op=mxfp4_moe_sort+quant(stage1) hip|triton|split;block_m=32;exact byte compare",
                model="synthetic",
            )
        )
    for dim, (E, k), m in itertools.product(
        [256, 2048],
        [(32, 5), (256, 8), (512, 8)],
        [1, 64, 128, 256, 1024, 2050, 4200, 10000, 163840],
    ):
        add(
            mk(
                s,
                m,
                "",
                dim,
                E,
                k,
                qa="",
                qw=FP4,
                qt="per_1x32",
                g1u1="",
                extra="op=mxfp4_moe_sort+quant(stage2 inter_dim=dim2);block_m=32;exact byte compare",
                model="synthetic",
            )
        )
    for (E, k), m in itertools.product(
        [(513, 11), (512, 10)], [1, 2, 3, 4, 8, 16, 17, 32, 64, 100, 127, 128]
    ):
        add(
            mk(
                s,
                m,
                4096,
                "",
                E,
                k,
                qa=FP4,
                qw=FP4,
                qt="per_1x32",
                g1u1="",
                extra="op=mxfp4_moe_sort_quant_fwd(fused sort+quant);block_m 16/32/64?;torch.testing atol=0",
                model="Qwen3.5-397B-A17B (shared)",
            )
        )
    s = "test_moeTopkSoftmax.py"
    for dt, E, t in itertools.product(
        ["float32", BF],
        [128, 256],
        [
            1,
            2,
            5,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            10000,
            16384,
            65536,
            163840,
        ],
    ):
        add(
            mk(
                s,
                t,
                "",
                "",
                E,
                8,
                qa="",
                qw="",
                qt="",
                g1u1="",
                dtype=dt,
                extra="op=topk_softmax(hip|asm);renormalize=1;strided gating;ref=sorted ids+weights checkAllclose",
                model="synthetic",
            )
        )
    for t in [
        1,
        2,
        5,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        10000,
        16384,
        65536,
        163840,
    ]:
        add(
            mk(
                s,
                t,
                "",
                "",
                256,
                8,
                qa="",
                qw="",
                qt="",
                g1u1="",
                extra="op=biased_grouped_topk(group=8,topk_group=4,sigmoid+bias);ref=torch golden",
                model="DeepSeek-V3/R1",
            )
        )
        add(
            mk(
                s,
                t,
                "",
                "",
                384,
                8,
                qa="",
                qw="",
                qt="",
                g1u1="",
                extra="op=biased_grouped_topk(group=1);scale_factor=2.827",
                model="Kimi-K2.5",
            )
        )
        add(
            mk(
                s,
                t,
                "",
                "",
                896,
                16,
                qa="",
                qw="",
                qt="",
                g1u1="",
                extra="op=biased_grouped_topk(group=1);row-strided slice of fused front buffer",
                model="Kimi-K3",
            )
        )
        for sf in ("softmax", "sigmoid"):
            add(
                mk(
                    s,
                    t,
                    "",
                    "",
                    256,
                    8,
                    qa="",
                    qw="",
                    qt="",
                    g1u1="",
                    extra=f"op=grouped_topk(scoring={sf};group=8;topk_group=4;scale=1.5)",
                    model="DeepSeek-V3/R1",
                )
            )
    s = "test_topk_softmax.py"
    for t, E, k, rn in itertools.product(
        [1, 32, 512, 4096], [8, 64, 128], [1, 2, 8], [1, 0]
    ):
        if k > E:
            continue
        add(
            mk(
                s,
                t,
                "",
                "",
                E,
                k,
                qa="",
                qw="",
                qt="",
                g1u1="",
                dtype="float32",
                extra=f"op=topk_softmax(asm);need_renorm={rn};torch.testing atol=rtol=1e-4;pytest",
                model="synthetic",
            )
        )
    s = "test_topk_softmax_shared_gate.py"
    for t, nsh, sc, rn in itertools.product(
        [1, 4, 16, 64, 256, 1024, 4096, 16384], [1, 2], [1.0, 0.5], [0, 1]
    ):
        add(
            mk(
                s,
                t,
                4096,
                "",
                512,
                10,
                qa="",
                qw="",
                qt="",
                g1u1="",
                extra=f"op=topk_softmax+shared_gate(sigmoid);num_shared={nsh};scale={sc};renorm={rn};dtype=bf16",
                model="Qwen3-Next/3.5 (512 routed, top10 +shared)?",
            )
        )
    s = "test_moe_topk_gating.py"
    nexp, ntok, nk = (
        [64, 128, 256, 384],
        [16384, 4096, 1024, 256, 64, 1],
        [1, 2, 4, 6, 8],
    )
    for E, t, k, dt in itertools.product(nexp, ntok, nk, ["float16", BF]):
        add(
            mk(
                s,
                t,
                "",
                "",
                E,
                k,
                qa="",
                qw="",
                qt="",
                g1u1="",
                dtype=dt,
                extra="op=topk_sigmoid;section=sigmoid;|score-idx err|<=0.01",
                model="synthetic",
            )
        )
    for E, t, k, dt in itertools.product(nexp, ntok, nk, ["float16", BF, "float32"]):
        add(
            mk(
                s,
                t,
                "",
                "",
                E,
                k,
                qa="",
                qw="",
                qt="",
                g1u1="",
                dtype=dt,
                extra="op=topk_softplus(sqrtsoftplus);section=softplus;err<=0.01;weight tol 1e-4",
                model="synthetic",
            )
        )
    for E, t, k, dt, rn in itertools.product(
        [64, 128, 256, 384, 512, 640], ntok, nk, ["float16", BF, "float32"], [0, 1]
    ):
        add(
            mk(
                s,
                t,
                "",
                "",
                E,
                k,
                qa="",
                qw="",
                qt="",
                g1u1="",
                dtype=dt,
                extra=f"op=topk_softmax(fused topk_gating vs vllm);section=softmax;renormalize={rn};weight tol 1e-4",
                model="synthetic",
            )
        )
    for t in (1, 4, 16, 64, 256, 1024, 2048, 4096):
        add(
            mk(
                s,
                t,
                "",
                "",
                512,
                10,
                qa="",
                qw="",
                qt="",
                g1u1="",
                dtype=BF,
                extra="op=topk_gating+shared(sigmoid);section=shared;num_shared=1;renorm=1",
                model="Qwen3-Next/3.5 (512+1)?",
            )
        )
    # ------------------------------------------------------------ route rows / scatter reduce / local ids
    for t, k, E, dist in itertools.product(
        [1, 64, 205, 2517], [5], [129], ["uniform", "shared", "hot", "dropped"]
    ):
        add(
            mk(
                "test_moe_route_rows.py",
                t,
                "",
                "",
                E,
                k,
                qa="",
                qw="",
                qt="",
                g1u1="",
                extra=f"op=route_rows;distribution={dist};exact int compare;MiniMax-M3 TP shape",
                model="MiniMax-M3",
            )
        )
    for M in (1, 64):
        add(
            mk(
                "test_moe_scatter_reduce.py",
                M,
                6144,
                "",
                129,
                5,
                qa=BF,
                qw="",
                qt="",
                g1u1="",
                extra="op=scatter_reduce(bf16|mxfp4);mb=16|128",
                model="MiniMax-M3",
            )
        )

    # ------------------------------------------------------------ FlyDSL a16w4 / gfx942 tests
    s = "test_flydsl_moe_a16wfp4.py"
    for (md, idim), t, (b, lb) in itertools.product(
        [(3584, 512), (3584, 384)], [1, 16, 128], [(1.0, 1.0), (4.0, 25.0)]
    ):
        add(
            mk(
                s,
                t,
                md,
                idim,
                896,
                16,
                act="Situv2",
                qa=BF,
                qw=FP4,
                qt="per_1x32",
                extra=f"op=fused_moe(a16w4 SiTUv2);gate_mode=SEPARATED;beta={b};linear_beta={lb};gfx942|gfx950;ref=bf16 SiTUv2 torch;cos/logits_diff gate",
                model="Kimi-K3",
            )
        )
    for t in (1, 8, 32):
        add(
            mk(
                s,
                t,
                5120,
                384,
                384,
                6,
                act="silu_clamped",
                qa=BF,
                qw=FP4,
                qt="per_1x32",
                extra="op=fused_moe(a16w4 clamped silu);logical_inter=288;swiglu_limit=10.0;tuned_flydsl_moe_a16w4.csv lookup (gfx942 304CU);rtol=atol=0.03",
                model="DeepSeek-V4.1(hidden5120) TP8?",
            )
        )
    for t, E, ab in itertools.product([1, 8, 33, 128], [8, 384], [0, 1]):
        add(
            mk(
                s,
                t,
                5120,
                384,
                E,
                6,
                act="silu_clamped",
                qa=BF,
                qw=FP4,
                qt="per_1x32",
                extra=f"op=flydsl_a16w4_gemm1 only (main() default sweep);activation_bf16={ab};swiglu_limit=10.0;tile_m32 n128 k256",
                model="DeepSeek-V4.1(hidden5120)?",
            )
        )
    s = "test_flydsl_moe_gfx942.py"
    ec = [
        (4096, 512, 256, 2048, 8, "ptpc", "silu", "random"),
        (4096, 512, 256, 2048, 8, "ptpc", "silu", "hot"),
        (16384, 1024, 384, 1, 1, "per_tensor", "silu", "hot"),
        (2048, 2560, 320, 513, 11, "no", "silu", "padding"),
        (2048, 2560, 320, 513, 11, "ptpc", "silu", "padding"),
        (2048, 2560, 320, 513, 11, "per_tensor", "silu", "pad_rows"),
        (2, 2560, 320, 513, 11, "ptpc", "silu", "pad_rows"),
    ]
    for pad in (0, 32, 64):
        ec.append((16384, 2048, 512, 257, 9, "ptpc", "silu", "random"))
    seen = set()
    qmap = {
        "no": ("No", BF, BF),
        "per_tensor": ("per_Tensor", FP8, FP8),
        "ptpc": ("per_Token", FP8, FP8),
    }
    for t, md, idim, E, k, q, act, routing in ec:
        qt, qa, qw = qmap[q]
        key = (t, md, idim, E, k, q, routing)
        if key in seen:
            continue
        seen.add(key)
        add(
            mk(
                s,
                t,
                md,
                idim,
                E,
                k,
                act=act.capitalize(),
                qa=qa,
                qw=qw,
                qt=qt,
                extra=f"op=impl__flydsl_gfx942 whole-MoE(EDGE_CASES);routing={routing};gfx942;cu 304&80 checked;err limits: elem 1.0 RMS, token 0.2, logits 1e-3",
                model="Qwen3.8-Flash-Next" if md == 2560 else "synthetic",
            )
        )
    for t, (md, k), pad in itertools.product(
        (65, 2048), ((2048, 9), (2560, 11), (4096, 8), (6144, 4)), (0, 32, 64, 128)
    ):
        add(
            mk(
                s,
                t,
                md,
                "",
                "",
                k,
                qa="",
                qw="",
                qt="",
                g1u1="",
                extra=f"op=sorted_sum(padded reduction);padding={pad};atol=0",
                model="synthetic",
            )
        )

    # ------------------------------------------------------------ flydsl_tests/test_flydsl_moe.py
    s = "flydsl_tests/test_flydsl_moe.py"
    for idim in (256, 384, 640):
        add(
            mk(
                s,
                16,
                512,
                idim,
                8,
                2,
                act="Swiglu",
                qa=FP8,
                qw=FP4,
                qt="per_1x32",
                extra="op=flydsl stage2 (a8w4 GUI) + e2e;gate_mode=INTERLEAVE;gfx950_only=1;atol=1.0,rtol=0.05,max_err_ratio=0.05",
                model="synthetic(DSV4 i640)",
            )
        )
    for t, md, idim, E, k, bm, tm, tn, tk, gm, od, seed, b, lb in [
        (16, 256, 128, 8, 2, 32, 32, 256, 256, "separated", "bf16", 1, 2.0, 1.5),
        (64, 512, 256, 16, 4, 32, 32, 256, 256, "separated", "bf16", 2, 2.0, 1.5),
        (16, 256, 128, 8, 2, 64, 64, 128, 256, "separated", "bf16", 3, 2.0, 1.5),
        (32, 256, 128, 8, 2, 32, 32, 128, 256, "separated", "f16", 4, 2.0, 1.5),
        (16, 256, 128, 8, 2, 32, 32, 256, 256, "separated", "bf16", 5, 1.0, 1.0),
        (16, 256, 128, 8, 2, 32, 32, 256, 256, "interleave", "bf16", 6, 2.0, 1.5),
        (64, 512, 640, 16, 4, 32, 32, 256, 256, "interleave", "bf16", 7, 2.0, 1.5),
    ]:
        add(
            mk(
                s,
                t,
                md,
                idim,
                E,
                k,
                act="Situv2",
                dtype="float16" if od == "f16" else BF,
                qa=FP8,
                qw=FP4,
                qt="per_1x32",
                extra=f"op=flydsl stage1 SiTUv2 vec4;gate_mode={gm.upper()};tile={tm}x{tn}x{tk};beta={b};linear_beta={lb};gfx950_only=1",
                model="synthetic",
            )
        )
    for idim, gm in itertools.product(
        [128, 256, 384, 512, 640], ["separated", "interleave"]
    ):
        add(
            mk(
                s,
                1024,
                512,
                idim,
                8,
                2,
                act="Situv2",
                qa=FP8,
                qw=FP4,
                qt="per_1x32",
                extra=f"op=flydsl e2e a8w4 SiTUv2;gate_mode={gm.upper()};beta=2.0;linear_beta=1.5;gfx950_only=1",
                model="synthetic",
            )
        )

    # ------------------------------------------------------------ test_fhmoe.py (heterogeneous MoE)
    s = "test_fhmoe.py"
    for t in (4, 1, 8, 16, 24, 32, 40, 64):
        add(
            mk(
                s,
                t,
                256,
                128,
                9,
                3,
                act="Silu",
                qa=FP8,
                qw=FP4,
                qt="per_1x32",
                extra="op=fused_moe(shared expert FHMoE);default profile routed_topk=2+shared;shared_expert_id=8;gate_mode=INTERLEAVE;swiglu_limit=10;rel_l2<=4e-2,cos<=1e-3;full sweep only with AITER_HETERO_MOE_FULL_SWEEP=1 (default M=[4]);gfx950",
                model="synthetic",
            )
        )
    for t in (4, 1, 8, 16, 24, 32, 40, 64):
        add(
            mk(
                s,
                t,
                6144,
                256,
                257,
                9,
                act="Silu",
                qa=FP8,
                qw=FP4,
                qt="per_1x32",
                extra="op=fused_moe(shared expert FHMoE);AITER_HETERO_MOE_HY4=1 profile;shared_expert_id=256;gate_mode=INTERLEAVE;rel_l2<=4e-2",
                model="HY4 (hidden6144,E257)",
            )
        )
    for t in (4, 1, 8, 16, 24, 32, 40, 64):
        add(
            mk(
                s,
                t,
                7168,
                384,
                385,
                7,
                act="Silu",
                qa=FP8,
                qw=FP4,
                qt="per_1x32",
                extra="op=fused_moe(shared expert FHMoE);AITER_HETERO_MOE_DSV4=1 profile;shared_expert_id=384;gate_mode=INTERLEAVE;rel_l2<=4e-2 (reduce test 5e-2)",
                model="DeepSeek-V4-Pro",
            )
        )
    for t in (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        38836,
    ):
        add(
            mk(
                s,
                t,
                6144,
                256,
                257,
                9,
                act="Silu",
                qa=FP8,
                qw=FP8,
                qt="per_1x32",
                extra="op=fused_moe(FHMoE MXFP8 routed + FP8 shared);production decode+prefill buckets;shared_expert_id=256;gate_mode=INTERLEAVE;sampled fp32 ref rel_l2<=4e-2,cos<=1e-3;kernel-name fragments asserted per bucket",
                model="HY4 MXFP8",
            )
        )
    add(
        mk(
            s,
            4,
            256,
            128,
            9,
            3,
            act="Silu",
            qa=FP4,
            qw=FP4,
            qt="per_1x32",
            extra="op=fused_moe(FHMoE a4w4 routed + fp8 shared);gate_mode=SEPARATED;rel_l2<=5e-2",
            model="synthetic",
        )
    )

    # ------------------------------------------------------------ multi-GPU
    for model, (md, idim, E, k, act) in {
        "glm5": (6144, 2048, 257, 9, "Silu"),
        "dsv3": (7168, 2048, 256, 8, "Silu"),
        "dsv4": (7168, 3072, 384, 6, "Silu"),
        "kimi3": (3584, 3072, 896, 16, "Situv2"),
        "m3": (6144, 3072, 129, 5, "Swiglu"),
    }.items():
        default = model in ("m3", "glm5")
        toks = [8, 64, 256] if default else []
        for t in toks:
            for mode in ("ag_rs", "ar"):
                add(
                    mk(
                        "multigpu_tests/test_mega_moe_tp.py",
                        t,
                        md,
                        idim,
                        E,
                        k,
                        act=act,
                        qa=FP4,
                        qw=FP4,
                        qt="per_1x32",
                        extra=f"op=MegaMoE TP (tp=4..8 inter sharded);comm_mode={mode};routes=balanced|random|hot;seeds=0,1;global tokens;rel_l2<=0.10;gfx950;torchrun;default models m3,glm5",
                        model={"m3": "MiniMax-M3", "glm5": "GLM-5.x"}[model],
                    )
                )
    add(
        mk(
            "multigpu_tests/test_mega_moe_tp.py",
            "2048,4096(opt)",
            6144,
            3072,
            129,
            5,
            act="Swiglu",
            qa=FP4,
            qw=FP4,
            qt="per_1x32",
            extra="op=MegaMoE TP big-M cases via --tokens (not default)",
            model="MiniMax-M3",
        )
    )
    for net, (md, idim, E, k, sl) in {
        "glm52": (6144, 2048, 256, 8, 0.0),
        "v4_pro": (7168, 3072, 384, 6, 10.0),
        "v41_flash": (5120, 2304, 384, 6, 10.0),
        "r1_v3": (7168, 2048, 256, 8, 0.0),
        "kimi_k3_route": (3584, 3072, 896, 16, 0.0),
    }.items():
        add(
            mk(
                "multigpu_tests/test_mega_moe_v2.py",
                128,
                md,
                idim,
                E,
                k,
                act="Situv2" if net.startswith("kimi") else "Silu",
                qa=FP8,
                qw=FP4,
                qt="per_1x32",
                extra=f"op=MegaMoEV2 (EP, mori);network={net};quant=a8w4 default|a4w4;bs_list default 128;swiglu_limit={sl};rtol 0.10 (a8w4)/0.25 (a4w4);iters=30;gfx950 multi-GPU",
                model={
                    "glm52": "GLM-5.2",
                    "v4_pro": "DeepSeek-V4-Pro",
                    "v41_flash": "DeepSeek-V4.1-Flash",
                    "r1_v3": "DeepSeek-V3/R1",
                    "kimi_k3_route": "Kimi-K3",
                }[net],
            )
        )
    for t in (7, 14, 28, 56, 70, 84, 126, 168, 224, 280, 336, 448):
        add(
            mk(
                "multigpu_tests/test_mega_moe_small_m.py",
                t,
                7168,
                3072,
                384,
                6,
                qa=FP8,
                qw=FP4,
                qt="per_1x32",
                extra="op=MegaMoEV2 forward_small_m;networks from test_mega_moe_v2 NETWORKS;mtpr=8192,small_m_mtprs=256,512;rtol=0.08;multi-GPU",
                model="DeepSeek-V4-Pro?",
            )
        )
    for t in (8192,):
        add(
            mk(
                "multigpu_tests/bench_mega_moe_v2.py",
                t,
                7168,
                3072,
                384,
                6,
                qa=FP8,
                qw=FP4,
                qt="per_1x32",
                extra="op=Mori EP vs MegaMoEV2 bench;--tokens 8192 default;iters=10",
                model="DeepSeek-V4-Pro",
            )
        )
    add(
        mk(
            "multigpu_tests/bench_mega_moe.py",
            128,
            7168,
            3072,
            384,
            6,
            qa=FP4,
            qw=FP4,
            qt="per_1x32",
            extra="op=MegaMoEGfx1250 61 layers chained;token_per_rank=128;gfx1250;quant a4w4_mxfp4 default",
            model="DeepSeek-V4-Pro",
        )
    )
    for t in (128, 512, 1024, 2048, 4096):
        add(
            mk(
                "multigpu_tests/test_wide_ep_moe.py",
                t,
                3584,
                3072,
                896,
                16,
                act="Situv2",
                qa=FP4,
                qw=FP4,
                qt="per_1x32",
                extra="op=TestWideEpMoe EP16 2-node;model-config=kimi;relL2<0.15;accuracy up to bs 512",
                model="Kimi-K3",
            )
        )
        add(
            mk(
                "multigpu_tests/test_wide_ep_moe.py",
                t,
                7168,
                3072,
                384,
                6,
                act="Silu",
                qa=FP8,
                qw=FP4,
                qt="per_1x32",
                extra="op=TestWideEpMoe EP16 2-node;model-config=dsv4 (a8w4,interleave,swiglu_limit=10);aiter strict a8w4",
                model="DeepSeek-V4-Pro",
            )
        )
    for t in (64, 1024, 4096, 32768):
        for route in ("uniform", "skew"):
            add(
                mk(
                    "multigpu_tests/test_comm_fused_moe.py",
                    t,
                    7168,
                    384,
                    384,
                    6,
                    qa=FP8,
                    qw=FP4,
                    qt="per_1x32",
                    extra=f"op=comm-fused FlyDSL MoE stage2 (TP8);route={route};modes eager|graph;max_abs 1.0,max_rel_l2 0.05;gfx950 8GPU",
                    model="DeepSeek-V4-Pro TP8",
                )
            )
    for qt in ("per_Token", "per_1x128"):
        add(
            mk(
                "multigpu_tests/test_dispatch_combine.py",
                128,
                6144,
                1024,
                16,
                2,
                qa=FP8,
                qw=FP8,
                qt=qt,
                extra="op=dispatch+combine (mori intranode);world=8",
                model="synthetic",
            )
        )
        add(
            mk(
                "multigpu_tests/test_mori_all2all.py",
                128,
                6144,
                1024,
                16,
                2,
                qa=FP8,
                qw=FP8,
                qt=qt,
                extra="op=mori all2all+fused_moe;world=8",
                model="synthetic",
            )
        )

    # ------------------------------------------------------------ triton tests / benchmarks
    out += triton_cases()
    # ------------------------------------------------------------ triton model_shapes.json
    import json

    ms = json.loads(
        Path(
            build.OP + "op_benchmarks/triton/model_benchmarking_tool/model_shapes.json"
        ).read_text()
    )
    for mdl, ks in ms.items():
        for kern, shapes in ks.items():
            if not kern.startswith("moe_op"):
                continue
            for sh in shapes:
                add(
                    mk(
                        "op_benchmarks/triton/model_benchmarking_tool/model_shapes.json",
                        "M sweep",
                        sh["Dim1"],
                        "",
                        sh["E"],
                        sh["TopK"],
                        act="Swiglu?",
                        qa="",
                        qw="",
                        qt="",
                        extra=f"op=triton {kern};dim2={sh['Dim2']} (=2*inter,TP-sharded);model_shapes.json",
                        model=mdl,
                    )
                )
    return out


def triton_cases():
    out = []
    base = build.OP + "triton_tests/moe/"
    spec = {
        "test_moe_gemm_a16w4.py": ("a16w4", BF, FP4, "per_1x32"),
        "test_moe_gemm_a4w4.py": ("a4w4", FP4, FP4, "per_1x32"),
        "test_moe_gemm_a8w4.py": ("a8w4", FP8, FP4, "per_1x32"),
        "test_moe_gemm_a8w8.py": ("a8w8", FP8, FP8, "per_1x32?"),
        "test_moe_gemm_a8w8_blockscale.py": ("a8w8_blockscale", FP8, FP8, "per_1x128"),
        "test_moe_gemm_int8_smoothquant.py": ("int8_smoothquant", I8, I8, "per_Token"),
    }
    for fn, (tag, qa, qw, qt) in spec.items():
        tree = ast.parse(Path(base + fn).read_text())
        fields = None
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef) and n.name == "Case":
                fields = [
                    (s.target.id, ast.unparse(s.value) if s.value is not None else None)
                    for s in n.body
                    if isinstance(s, ast.AnnAssign)
                ]
        cases = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "Case":
                vals = {}
                for name, dflt in fields:
                    vals[name] = ast.literal_eval(dflt) if dflt is not None else None
                for (name, _), a in zip(fields, n.args):
                    vals[name] = ast.literal_eval(a)
                for kw in n.keywords:
                    vals[kw.arg] = ast.literal_eval(kw.value)
                cases.append(vals)
        # dedupe identical case tuples
        uniq = []
        for c in cases:
            if c not in uniq:
                uniq.append(c)
        for c in uniq:
            qa_, qw_ = qa, qw
            if "act_dtype_str" in c:
                qa_ = c["act_dtype_str"]
            if "weight_dtype_str" in c:
                qw_ = c["weight_dtype_str"]
            ex = [
                f"op=triton moe_gemm_{tag} (single grouped GEMM, not whole MoE)",
                f"gemm_n={c['n']}",
                f"gemm_k={c['k']}",
            ]
            for kx, v in c.items():
                if kx not in (
                    "m",
                    "n",
                    "k",
                    "n_expts_tot",
                    "n_expts_act",
                    "act_dtype_str",
                    "weight_dtype_str",
                ):
                    ex.append(f"{kx}={v}")
            ex.append(
                "sweeps: do_gather/do_scatter x has_y_gammas x apply_swiglu x backend (see md)"
            )
            ex.append("tol maxtol/rmstol default 2e-2/4e-3")
            out.append(
                mk(
                    f"triton_tests/moe/{fn}",
                    c["m"],
                    "",
                    "",
                    c["n_expts_tot"],
                    c["n_expts_act"],
                    act="Swiglu|none",
                    qa=qa_,
                    qw=qw_,
                    qt=qt,
                    g1u1="",
                    extra=";".join(ex),
                    model="",
                )
            )
    return out
