# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The interface every family kernel implements."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from .spec import MoeSpec

if TYPE_CHECKING:
    from .weights import MoeWeights


@dataclass
class MoeCall:
    """One forward call, as the router hands it to a kernel.

    ``m`` is ``hidden_states.shape[0]`` and ``topk`` is ``topk_ids.shape[1]``.
    ``config`` is the route row's config name. When ``output`` is given the
    kernel writes into it and returns it. ``extra`` holds call-level
    fused_moe arguments a kernel must decline unless it implements them
    (``stage2_scatter``). Prepare-level arguments (scales, bias,
    swiglu_limit, betas, pads) are on ``weights``.
    """

    hidden_states: torch.Tensor
    topk_weight: torch.Tensor
    topk_ids: torch.Tensor
    weights: "MoeWeights"
    m: int
    topk: int
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

    @property
    def out_dtype(self) -> torch.dtype:
        return self.spec.out_dtype or self.hidden_states.dtype


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
        """Run once per weight set, at load. Returns this kernel's state for
        those weights (derived layouts, scale reshapes, compiled handles); the
        router stores it in ``weights.state[self.id]``. Must not modify the
        aiter-layout tensors, which the fallback still uses."""
        return None

    def warmup(self, weights: "MoeWeights", m: int, topk: int, config: str) -> None:
        """Compile everything a call at token count ``m`` will launch, so a
        compile error surfaces at prepare time rather than under graph
        capture."""

    def __call__(self, call: MoeCall) -> torch.Tensor:
        """Compute the layer; must be graph-capture safe (no host sync)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id}>"
