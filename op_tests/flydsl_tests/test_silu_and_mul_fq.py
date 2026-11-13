import pytest
import torch

from aiter import dtypes
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.ops.flydsl.moe_kernels import _get_compiled_silu_fused, _run_compiled
from aiter.ops.quant import per_1x32_f4_quant
from aiter.utility.fp4_utils import moe_mxfp4_sort


def _swiglu_ref(x: torch.Tensor, inter_dim: int) -> torch.Tensor:
    gate, up = x.float().split([inter_dim, inter_dim], dim=-1)
    gate = gate.clamp(max=7.0)
    up = up.clamp(min=-7.0, max=7.0)
    return gate * torch.sigmoid(1.702 * gate) * (up + 1.0)


def _situv2_ref(
    x: torch.Tensor, inter_dim: int, beta: float, linear_beta: float
) -> torch.Tensor:
    gate, up = x.float().split([inter_dim, inter_dim], dim=-1)
    return (
        beta
        * torch.tanh(gate / beta)
        * torch.sigmoid(gate)
        * linear_beta
        * torch.tanh(up / linear_beta)
    )


@pytest.mark.parametrize("beta, linear_beta", [(1.0, 1.0), (4.0, 25.0)])
def test_flydsl_situv2_fused_uses_runtime_betas(beta: float, linear_beta: float):
    torch.manual_seed(0)
    device = torch.device("cuda")
    token_num, topk, inter_dim = 4, 2, 256
    rows = token_num * topk
    x = torch.randn(rows, inter_dim * 2, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, inter_dim, dtype=torch.bfloat16, device=device)
    row_ids = torch.arange(rows, dtype=torch.int32, device=device)
    sorted_ids = (row_ids // topk) | ((row_ids % topk) << 24)
    num_valid_ids = torch.tensor([rows], dtype=torch.int32, device=device)

    kernel = _get_compiled_silu_fused(
        inter_dim,
        topk,
        quant_mode="none",
        gui_layout=False,
        act="situv2",
        enable_bias=False,
    )
    _run_compiled(
        kernel,
        (
            ptr_arg(x),
            ptr_arg(out.view(torch.uint8)),
            ptr_arg(torch.empty(0, dtype=torch.uint8, device=device)),
            ptr_arg(sorted_ids),
            ptr_arg(num_valid_ids),
            ptr_arg(sorted_ids),
            ptr_arg(torch.empty(0, dtype=torch.float32, device=device)),
            token_num,
            rows,
            beta,
            1.0 / beta,
            linear_beta,
            1.0 / linear_beta,
            float("inf"),
            torch.cuda.current_stream(),
        ),
    )

    torch.testing.assert_close(
        out.float(),
        _situv2_ref(x, inter_dim, beta, linear_beta),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("token_num, topk, inter_dim", [(4, 2, 256)])
def test_flydsl_swiglu_fused_fp4_quant_matches_reference(
    token_num: int, topk: int, inter_dim: int
):
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("MXFP4 is not available in this torch build")

    torch.manual_seed(0)
    device = torch.device("cuda")
    rows = token_num * topk

    gate = torch.linspace(-8.0, 8.0, rows * inter_dim, device=device).reshape(
        rows, inter_dim
    )
    up = torch.linspace(-2.0, 2.0, rows * inter_dim, device=device).reshape(
        rows, inter_dim
    )
    x = torch.cat([gate, up], dim=-1).to(torch.bfloat16)

    token_ids = torch.arange(token_num, device=device).repeat_interleave(topk)
    slot_ids = torch.arange(topk, device=device).repeat(token_num)
    order = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4], device=device)
    sorted_ids = ((slot_ids[order] << 24) | token_ids[order]).to(torch.int32)
    num_valid_ids = torch.tensor([rows], dtype=torch.int32, device=device)

    out = torch.empty(
        (token_num, topk, inter_dim // 2), dtype=dtypes.fp4x2, device=device
    )
    scale_cols = inter_dim // 32
    padded_rows = (sorted_ids.numel() + 255) // 256 * 256
    padded_cols = (scale_cols + 7) // 8 * 8
    out_scale_sorted = torch.empty(
        padded_rows * padded_cols, dtype=torch.uint8, device=device
    )

    kernel = _get_compiled_silu_fused(
        inter_dim,
        topk,
        quant_mode="fp4",
        gui_layout=False,
        act="swiglu",
        enable_bias=False,
    )
    _run_compiled(
        kernel,
        (
            ptr_arg(x),
            ptr_arg(out.view(-1).view(torch.uint8)),
            ptr_arg(out_scale_sorted),
            ptr_arg(sorted_ids),
            ptr_arg(num_valid_ids),
            ptr_arg(sorted_ids),
            ptr_arg(torch.empty(0, dtype=torch.float32, device=device)),
            token_num,
            sorted_ids.numel(),
            1.0,
            1.0,
            1.0,
            1.0,
            7.0,
            torch.cuda.current_stream(),
        ),
    )

    ref_act = _swiglu_ref(x, inter_dim)
    ref_out, ref_scale = per_1x32_f4_quant(ref_act, quant_dtype=dtypes.fp4x2)
    ref_scale_sorted = moe_mxfp4_sort(
        ref_scale.view(token_num, topk, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
    )

    ref_out = ref_out.view(token_num, topk, -1)
    torch.testing.assert_close(out.view(torch.uint8), ref_out.view(torch.uint8))
    ref_scale_bytes = ref_scale_sorted.view(torch.uint8).view(-1)
    scale_positions = []
    n32_sort = scale_cols * 32
    for row in range(rows):
        for col in range(scale_cols):
            scale_positions.append(
                (row >> 5) * n32_sort
                + (col >> 3) * 256
                + (col & 3) * 64
                + (row & 15) * 4
                + ((col >> 2) & 1) * 2
                + ((row >> 4) & 1)
            )
    scale_positions = torch.tensor(scale_positions, dtype=torch.long, device=device)
    torch.testing.assert_close(
        out_scale_sorted[scale_positions],
        ref_scale_bytes[scale_positions],
    )


def _splitk_post_inputs(token_num, topk, inter_dim, num_experts, device):
    rows = token_num * topk
    topk_ids = torch.randint(0, num_experts, (rows,), dtype=torch.int32, device=device)
    row_ids = torch.arange(rows, dtype=torch.int32, device=device)
    sorted_ids = (row_ids // topk) | ((row_ids % topk) << 24)
    num_valid_ids = torch.tensor([rows], dtype=torch.int32, device=device)
    return rows, topk_ids, sorted_ids, num_valid_ids


def _to_interleaved(gate, up):
    # [gate_0:16, up_0:16, gate_16:32, ...] per row
    rows, n = gate.shape
    return torch.cat(
        [gate.view(rows, n // 16, 16), up.view(rows, n // 16, 16)], dim=-1
    ).view(rows, 2 * n)


@pytest.mark.parametrize("gui_layout", [False, True])
def test_flydsl_splitk_bias_fused_matches_reference(gui_layout: bool):
    # enable_bias=True (split-K stage 1 with a bias) raised
    # NameError: topk_ids_p, so it never compiled.
    torch.manual_seed(0)
    device = torch.device("cuda")
    token_num, topk, inter_dim, num_experts = 8, 2, 256, 4
    rows, topk_ids, sorted_ids, num_valid_ids = _splitk_post_inputs(
        token_num, topk, inter_dim, num_experts, device
    )
    gate = torch.randn(rows, inter_dim, device=device).to(torch.bfloat16)
    up = torch.randn(rows, inter_dim, device=device).to(torch.bfloat16)
    x = (
        _to_interleaved(gate, up)
        if gui_layout
        else torch.cat([gate, up], dim=-1).contiguous()
    )
    bias = torch.randn(num_experts, inter_dim * 2, dtype=torch.float32, device=device)
    out = torch.empty(rows, inter_dim, dtype=torch.bfloat16, device=device)

    kernel = _get_compiled_silu_fused(
        inter_dim,
        topk,
        quant_mode="none",
        gui_layout=gui_layout,
        act="swiglu",
        enable_bias=True,
    )
    _run_compiled(
        kernel,
        (
            ptr_arg(x),
            ptr_arg(out.view(torch.uint8)),
            ptr_arg(torch.empty(0, dtype=torch.uint8, device=device)),
            ptr_arg(sorted_ids),
            ptr_arg(num_valid_ids),
            ptr_arg(topk_ids),
            ptr_arg(bias.view(-1)),
            token_num,
            rows,
            1.0,
            1.0,
            1.0,
            1.0,
            7.0,
            torch.cuda.current_stream(),
        ),
    )

    row_bias = bias[topk_ids.long()]
    biased = torch.cat(
        [gate.float() + row_bias[:, :inter_dim], up.float() + row_bias[:, inter_dim:]],
        dim=-1,
    )
    torch.testing.assert_close(
        out.float(), _swiglu_ref(biased, inter_dim), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize("gui_layout", [False, True])
@pytest.mark.parametrize("quant_mode", ["none", "fp4", "fp8"])
def test_flydsl_splitk_zero_bias_matches_no_bias(quant_mode: str, gui_layout: bool):
    if quant_mode == "fp4" and not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("MXFP4 is not available in this torch build")
    torch.manual_seed(0)
    device = torch.device("cuda")
    token_num, topk, inter_dim, num_experts = 8, 2, 256, 4
    rows, topk_ids, sorted_ids, num_valid_ids = _splitk_post_inputs(
        token_num, topk, inter_dim, num_experts, device
    )
    x = torch.randn(rows, inter_dim * 2, device=device).to(torch.bfloat16)
    zero_bias = torch.zeros(num_experts * inter_dim * 2, device=device)
    scale_bytes = ((rows + 255) // 256 * 256) * ((inter_dim // 32 + 7) // 8 * 8)
    out_bytes = {
        "none": rows * inter_dim * 2,
        "fp8": rows * inter_dim,
        "fp4": rows * inter_dim // 2,
    }[quant_mode]

    results = {}
    for enable_bias in (False, True):
        out = torch.zeros(out_bytes, dtype=torch.uint8, device=device)
        scale = torch.zeros(scale_bytes, dtype=torch.uint8, device=device)
        kernel = _get_compiled_silu_fused(
            inter_dim,
            topk,
            quant_mode=quant_mode,
            gui_layout=gui_layout,
            act="swiglu",
            enable_bias=enable_bias,
        )
        _run_compiled(
            kernel,
            (
                ptr_arg(x),
                ptr_arg(out),
                ptr_arg(scale),
                ptr_arg(sorted_ids),
                ptr_arg(num_valid_ids),
                ptr_arg(topk_ids if enable_bias else sorted_ids),
                ptr_arg(
                    zero_bias
                    if enable_bias
                    else torch.empty(0, dtype=torch.float32, device=device)
                ),
                token_num,
                rows,
                1.0,
                1.0,
                1.0,
                1.0,
                7.0,
                torch.cuda.current_stream(),
            ),
        )
        torch.cuda.synchronize()
        results[enable_bias] = (out.clone(), scale.clone())

    torch.testing.assert_close(results[True][0], results[False][0], rtol=0, atol=0)
    torch.testing.assert_close(results[True][1], results[False][1], rtol=0, atol=0)
