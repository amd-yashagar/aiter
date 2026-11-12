# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The interface every family kernel implements."""

import contextlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from .spec import MoeSpec

if TYPE_CHECKING:
    from .weights import MoeWeights


@dataclass
class MoeCall:
    """One forward call, as the router hands it to a kernel.

    ``m`` is ``hidden_states.shape[0]`` and ``topk`` is ``topk_ids.shape[1]``;
    ``out_dtype`` is the output dtype fused_moe resolved; ``global_experts`` is
    the size of ``expert_mask`` under EP, else the local expert count.
    ``config`` is the route row's config name. When ``output`` is given the
    kernel writes into it and returns it. ``w1``, ``w2``, their scales and the
    biases are the call's own tensors, in the caller's layout. Other
    layer-level arguments (swiglu_limit, betas, pads) are on ``weights``.
    """

    hidden_states: torch.Tensor
    topk_weight: torch.Tensor
    topk_ids: torch.Tensor
    weights: "MoeWeights"
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor | None
    w2_scale: torch.Tensor | None
    bias1: torch.Tensor | None
    bias2: torch.Tensor | None
    m: int
    topk: int
    out_dtype: torch.dtype
    global_experts: int
    config: str = ""
    expert_mask: torch.Tensor | None = None
    num_local_tokens: torch.Tensor | None = None
    output: torch.Tensor | None = None
    a1_scale: torch.Tensor | None = None
    a2_scale: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> MoeSpec:
        return self.weights.spec


@contextlib.contextmanager
def compile_only():
    """Compile FlyDSL kernels without launching them (``COMPILE_ONLY=1``): under
    FlyDSL 0.5.0 a ``@flyc.jit`` call can otherwise launch once on a visible GPU."""
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("COMPILE_ONLY", None)
        else:
            os.environ["COMPILE_ONLY"] = old


class MoeKernel:
    """Base class for a family kernel.

    Subclasses set ``id`` (unique, e.g. ``"gfx942_fp8_block.decode"``),
    ``family`` and ``archs``, and implement ``supports`` and ``__call__``.
    One instance serves every shape; per-shape state lives in the
    ``MoeWeights`` slot the kernel's ``prepare_weights`` fills.
    """

    id: ClassVar[str] = ""
    family: ClassVar[str] = ""
    archs: ClassVar[tuple[str, ...]] = ()
    # "tiles": the kernel sorts with core.tile_map.route_to_tiles; "direct": it
    # takes raw topk_ids/topk_weight and owns EP masks, -1 ids,
    # num_local_tokens and the zero-fill itself (its supports() bounds M*topk).
    routing: ClassVar[str] = "tiles"
    # "atomic": the sort zero-fills the output; "reduce": every row is written once.
    epilogue: ClassVar[str] = "atomic"
    # Rounding points, e.g. {"act_quant": "per_1x128_fp8", "inter":
    # "bf16_then_per_1x128_fp8", "combine": "bf16_atomic"}; None means "same as
    # aiter's dispatched path". act_quant: none, per_token_fp8, per_1x128_fp8,
    # mx_fp4, mx_fp8; inter: bf16, fp32_direct, bf16_then_<quant>; combine:
    # bf16_atomic, fp32_atomic, bf16_rows_fp32_sum, fp32_reduce. A kernel may round less than, never more than,
    # the least-rounding aiter path for its family and arch.
    numerics: ClassVar[dict | None] = None
    # Not-yet-covered layer keywords (router.NOT_YET_COVERED) this kernel
    # implements; any other one set on a layer makes it decline.
    handles_kwargs: ClassVar[frozenset] = frozenset()

    def supports(
        self, spec: MoeSpec, m: int, topk: int, call: MoeCall | None = None
    ) -> str | None:
        """None if this kernel computes the case with fused_moe's semantics,
        else a short reason. ``call`` is None when routes are resolved (at
        prepare or the first call of a token count); then only the static
        case is judged. With ``call`` it judges the per-call arguments
        (EP mask, num_local_tokens, output, a1_scale, extra). Host only: must
        not read device memory."""
        raise NotImplementedError

    def prepare_weights(self, weights: "MoeWeights") -> Any:
        """Run once per weight set, on its first eager call. Returns this
        kernel's small state for those weights (scale reorders, tile metadata,
        per-expert pointers); the router stores it in ``weights.state[self.id]``.
        Never a weight copy, and never a change to the caller's tensors, which
        today's path still reads."""
        return None

    def warmup(self, weights: "MoeWeights", m: int, topk: int, config: str) -> None:
        """Compile everything a call at token count ``m`` will launch, without
        launching anything (run compiles under ``kernel.compile_only()``). It
        runs on the first eager call of a weight set, for every M bucket the
        route table maps, so a later graph capture finds the kernels ready."""

    def __call__(self, call: MoeCall) -> torch.Tensor:
        """Compute the layer; must be graph-capture safe (no host sync)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id}>"
