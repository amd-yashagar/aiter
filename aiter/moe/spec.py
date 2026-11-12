# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What a MoE layer computes, independent of the token count."""

from dataclasses import dataclass

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import get_inter_dim, resolve_activation_dtype
from aiter.ops.flydsl.moe_common import GateMode

# per_128x128 weights are served as per_1x128, as in fused_moe's quant_remap.
_QUANT_REMAP = {QuantType.per_128x128: QuantType.per_1x128}

FAMILIES = (
    "a16w16",
    "fp8-tensor",
    "fp8-token",
    "fp8-block",
    "int8",
    "a8wi4",
    "a16wi4",
    "a16w4",
    "a4w4",
    "a8w4",
    "mxfp8",
)


@dataclass(frozen=True)
class MoeSpec:
    """Static description of one MoE layer (topk comes from each call).

    ``experts`` is the local expert count (``w1.shape[0]``);
    ``global_experts`` is the size of ``expert_mask`` under expert
    parallelism, else equal to ``experts``. ``inter_dim`` is the logical
    intermediate size (int4 packing undone). ``out_dtype`` None means the
    output takes the dtype of ``hidden_states``, as in ``fused_moe``.
    """

    experts: int
    global_experts: int
    model_dim: int
    inter_dim: int
    act: ActivationType
    quant: QuantType
    w_dtype: torch.dtype
    out_dtype: torch.dtype | None
    g1u1: bool
    doweight_stage1: bool
    gate_mode: GateMode
    w_shuffled: bool
    has_bias: bool = False
    swiglu_limit: float | None = None
    hidden_pad: int = 0
    intermediate_pad: int = 0

    @classmethod
    def from_weights(
        cls,
        w1: torch.Tensor,
        w2: torch.Tensor,
        *,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
        doweight_stage1: bool = False,
        dtype: torch.dtype | None = None,
        gate_mode=GateMode.SEPARATED,
        has_bias: bool = False,
        swiglu_limit: float | None = None,
        hidden_pad: int = 0,
        intermediate_pad: int = 0,
        num_global_experts: int | None = None,
    ) -> "MoeSpec":
        experts, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
        quant = QuantType(int(quant_type))
        return cls(
            experts=experts,
            global_experts=int(num_global_experts or experts),
            model_dim=model_dim,
            inter_dim=inter_dim,
            act=ActivationType(int(activation)),
            quant=_QUANT_REMAP.get(quant, quant),
            w_dtype=w1.dtype,
            out_dtype=dtype,
            g1u1=inter_dim != w1.shape[1],
            doweight_stage1=bool(doweight_stage1),
            gate_mode=GateMode(gate_mode or GateMode.SEPARATED),
            w_shuffled=bool(
                getattr(w1, "is_shuffled", False) or getattr(w2, "is_shuffled", False)
            ),
            has_bias=bool(has_bias),
            swiglu_limit=swiglu_limit,
            hidden_pad=int(hidden_pad),
            intermediate_pad=int(intermediate_pad),
        )

    def a_dtype(
        self, m: int, hidden_dtype=dtypes.bf16, has_a1_scale: bool = False
    ) -> torch.dtype:
        """The activation dtype fused_moe quantizes to at token count ``m``."""
        return resolve_activation_dtype(
            self.quant,
            self.w_dtype,
            activation=self.act,
            gate_mode=self.gate_mode,
            M=m,
            hidden_dtype=hidden_dtype,
            has_a1_scale=has_a1_scale,
        )

    def family(
        self, m: int, hidden_dtype=dtypes.bf16, has_a1_scale: bool = False
    ) -> str:
        """The quant family at token count ``m``; names match the tier files."""
        q, w = self.quant, self.w_dtype
        if q == QuantType.No:
            return "a16w16"
        if q == QuantType.per_Tensor:
            return "fp8-tensor" if w == dtypes.fp8 else "int8"
        if q == QuantType.per_Token:
            if w == dtypes.i8:
                return "int8"
            if w in (dtypes.i4x2, torch.uint32):
                return "a8wi4"
            return "fp8-token"
        if q == QuantType.per_1x128:
            return "fp8-block"
        if q == QuantType.per_1x32:
            if w == dtypes.i4x2:
                return "a16wi4"
            if w == dtypes.fp8:
                return "mxfp8"
            a = self.a_dtype(m, hidden_dtype, has_a1_scale)
            if a in (dtypes.bf16, dtypes.fp16):
                return "a16w4"
            if a == dtypes.fp8:
                return "a8w4"
            return "a4w4"
        return f"unknown-{q.name}"
