# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The hook behind aiter.fused_moe: route a call to a new kernel or return None.

``try_route`` is called at the top of ``fused_moe._fused_moe_impl``, after its
argument checks. It returns the output when a kernel served the call, and None
for "run today's path". It never raises (unless ``AITER_MOE_STRICT=1``).

Per-weight state (``MoeWeights``) is built lazily, on the first eager call for
a weight set, and holds only small state and compiled kernels; never a weight
copy. It is found through an identity cache (one dict lookup on ``id()``s,
verified by weak reference) backed by a key of each weight tensor's
``(data_ptr, shape, stride, dtype, device)`` plus the layer's fused_moe
arguments. A finalizer on each weight's owning tensor drops the state when the
weights are freed; a changed ``_version`` rebuilds it on the next eager call.
Inside a CUDA graph capture or a torch.compile trace nothing is built,
compiled or marked failed: a call without ready state takes today's path.
"""

import os
import threading
import weakref
from typing import NamedTuple

import torch

from . import routes as _routes
from .gpu import gpu_name
from .kernel import MoeCall
from .log import log_once
from .spec import MoeSpec
from .weights import MoeWeights

# fused_moe arguments that describe the layer; the hook passes their values as
# one tuple in this order. They key the per-weight state.
LAYER_KWARGS = (
    "activation",
    "quant_type",
    "doweight_stage1",
    "gate_mode",
    "swiglu_limit",
    "beta",
    "linear_beta",
    "hidden_pad",
    "intermediate_pad",
    "block_size_M",
    "moe_sorting_dispatch_policy",
    "quant_type_a",
    "quant_dtype_a",
    "quant_dtype_a2",
    "herd_topk",
)

# Re-entrant: a weight finalizer (_drop) can fire inside a locked region.
_LOCK = threading.RLock()
_BY_ID: dict = {}  # identity key -> (weakrefs of the call's tensors, MoeWeights)
_BY_KEY: dict = {}  # storage key -> MoeWeights
_LAST = threading.local()


def strict() -> bool:
    return os.getenv("AITER_MOE_STRICT", "0") == "1"


_DEBUG = os.getenv("AITER_MOE_DEBUG", "0") == "1"


class LastRoute(NamedTuple):
    kernel: str
    family: str
    gpu: str
    reason: str
    numerics: dict | None


def last_route() -> LastRoute:
    """(kernel id or "aiter", family, gpu, reason, numerics) of the last
    fused_moe call on this thread; numerics is None for today's path."""
    return getattr(_LAST, "route", LastRoute("aiter", "", "", "no call yet", None))


_INACTIVE = LastRoute(
    "aiter",
    "",
    "",
    "routing inactive (AITER_MOE_DISABLE=1, or no route rows and no AITER_MOE_ROUTES)",
    None,
)


def record_inactive() -> None:
    """A fused_moe call the hook did not route at all: today's path."""
    _LAST.route = _INACTIVE


def _record(kernel, family, gpu, reason, key):
    num = None
    if kernel != "aiter":
        from .registry import numerics

        num = numerics(kernel)
    _LAST.route = LastRoute(kernel, family, gpu, reason, num)
    if _DEBUG:
        log_once(f"aiter.moe route {key}: {kernel} ({family}, {gpu}): {reason}")


def active() -> bool:
    """Whether the hook should run at all: off under AITER_MOE_DISABLE=1, and
    when this GPU's table is empty and AITER_MOE_ROUTES is unset."""
    if os.getenv("AITER_MOE_DISABLE", "0") == "1":
        return False
    if os.getenv("AITER_MOE_ROUTES"):
        return True
    return bool(_routes.table_for(gpu_name()).rows)


def refresh() -> None:
    """Re-read the environment and route tables (tests, and after changing
    AITER_MOE_* variables in-process)."""
    import aiter.fused_moe as fm

    global _DEBUG
    _DEBUG = os.getenv("AITER_MOE_DEBUG", "0") == "1"
    _routes.table_for.cache_clear()
    gpu_name.cache_clear()
    with _LOCK:
        _BY_ID.clear()
        _BY_KEY.clear()
    fm._MOE_ROUTING = None


def _owner(t):
    """The tensor whose lifetime bounds ``t``'s storage (the base of a view)."""
    return t._base if t._base is not None else t


def _storage_key(t):
    if t is None:
        return None
    return (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype, t.device)


def _versions(tensors):
    return tuple(t._version if t is not None else -1 for t in tensors)


def _drop(key):
    with _LOCK:
        w = _BY_KEY.pop(key, None)
        if w is not None:
            for k in [k for k, (_, _, v) in _BY_ID.items() if v is w]:
                del _BY_ID[k]


def _fast_vals(vals):
    """The layer values with the two pybind enums (activation, quant_type) as
    ints: comparing pybind enums costs far more than comparing ints."""
    return (vals[0].value, vals[1].value, *vals[2:])


def _state(w1, w2, tensors, vals, make_kw, global_e, topk, capturing):
    """MoeWeights for this weight set, building it on an eager call. The fast
    path is one dict lookup on ints, then layer, weakref and _version checks."""
    ident = (*map(id, tensors), global_e)
    hit = _BY_ID.get(ident)
    fast = _fast_vals(vals)
    if hit is not None:
        live, layer_vals, w = hit
        present = [t for t in tensors if t is not None]
        if layer_vals == fast and all(
            r() is t and t._version == v for (r, v), t in zip(live, present)
        ):
            return w
    if capturing:
        return None
    kw = make_kw()
    skey = (*map(_storage_key, tensors), *fast, global_e)
    with _LOCK:
        w = _BY_KEY.get(skey)
        if w is not None and w.versions != _versions(tensors):
            _BY_KEY.pop(skey)
            w = None
        if w is None:
            w = _build(w1, w2, tensors, kw, global_e, topk)
            _BY_KEY[skey] = w
            for t in {id(_owner(t)): _owner(t) for t in (w1, w2)}.values():
                weakref.finalize(t, _drop, skey)
        _BY_ID[ident] = (
            tuple((weakref.ref(t), t._version) for t in tensors if t is not None),
            fast,
            w,
        )
    return w


def _build(w1, w2, tensors, kw, global_e, topk):
    spec = MoeSpec.from_weights(
        w1,
        w2,
        activation=kw["activation"],
        quant_type=kw["quant_type"],
        doweight_stage1=kw["doweight_stage1"],
        dtype=kw.get("dtype"),
        gate_mode=kw["gate_mode"],
        has_bias=kw.get("bias1") is not None or kw.get("bias2") is not None,
        swiglu_limit=kw.get("swiglu_limit"),
        hidden_pad=kw.get("hidden_pad", 0),
        intermediate_pad=kw.get("intermediate_pad", 0),
        num_global_experts=global_e,
    )
    gpu = gpu_name()
    table = _routes.resolve_table(gpu)
    tensor_kwargs = ("w1_scale", "w2_scale", "bias1", "bias2")
    w = MoeWeights(
        spec=spec,
        w1_ref=weakref.ref(w1),
        w2_ref=weakref.ref(w2),
        fused_moe_kwargs={k: v for k, v in kw.items() if k not in tensor_kwargs},
        gpu=gpu,
        table=table,
        versions=_versions(tensors),
    )
    w.uncovered = _uncovered(kw)
    w.compile_all(topk)
    return w


# Layer arguments no new kernel covers yet; a kernel lists the ones it
# implements in handles_kwargs. FHMoE shared experts never reach this hook
# (fused_moe sends them to aiter.fhmoe first) and become their own family; the
# forced activation dtypes are served by the families they apply to; HERD is
# gfx1250-only and stays on today's path on gfx942 and gfx950.
NOT_YET_COVERED = {
    "quant_type_a": None,
    "quant_dtype_a": None,
    "quant_dtype_a2": None,
    "herd_topk": 0,
}


def _uncovered(kw):
    return frozenset(
        k
        for k, d in NOT_YET_COVERED.items()
        if kw.get(k) is not None and (d is None or kw.get(k) != d)
    )


def try_route(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    expert_mask,
    num_local_tokens,
    output,
    a1_scale,
    a2_scale,
    w1_scale,
    w2_scale,
    bias1,
    bias2,
    out_dtype,
    q_dtype_a,
    has_private,
    stage2_scatter,
    layer_vals,
):
    """Serve a fused_moe call with a new kernel, or return None."""
    m, topk = topk_ids.shape
    if has_private or stage2_scatter is not None:
        _record("aiter", "", "", "private or stage2_scatter arguments", None)
        return None
    capturing = (
        torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()
    )
    global_e = expert_mask.numel() if expert_mask is not None else w1.shape[0]
    tensors = (w1, w2, w1_scale, w2_scale, bias1, bias2)

    def make_kw():
        kw = dict(zip(LAYER_KWARGS, layer_vals))
        kw.update(w1_scale=w1_scale, w2_scale=w2_scale, bias1=bias1, bias2=bias2)
        return kw

    try:
        w = _state(w1, w2, tensors, layer_vals, make_kw, global_e, topk, capturing)
        if w is None:
            log_once(
                f"aiter.moe: no routing state for a weight set during graph "
                f"capture or tracing (M={m}); this call takes today's path"
            )
            _record("aiter", "", "", "no state during capture", None)
            return None
        res = w.resolve(
            m,
            topk,
            hidden_states.dtype,
            out_dtype,
            q_dtype_a,
            a1_scale is not None,
            build=not capturing,
        )
        if res is None:
            log_once(
                f"aiter.moe: route for M={m} not resolved before graph capture or "
                f"tracing; this call takes today's path"
            )
            _record("aiter", "", w.gpu, "route not resolved during capture", (m, topk))
            return None
        kernel = res.kernel
        if kernel is None:
            _record("aiter", res.family, w.gpu, res.reason, (m, topk))
            return None
        if capturing and not w.ready(kernel, res.route):
            log_once(
                f"aiter.moe: {kernel.id} not compiled for {w.spec} M={m} during "
                f"graph capture; this call takes today's path"
            )
            _record(
                "aiter", res.family, w.gpu, "kernel not ready during capture", (m, topk)
            )
            return None
        call = MoeCall(
            hidden_states=hidden_states,
            topk_weight=topk_weight,
            topk_ids=topk_ids,
            weights=w,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            bias1=bias1,
            bias2=bias2,
            m=m,
            topk=topk,
            config=res.route.config,
            expert_mask=expert_mask,
            num_local_tokens=num_local_tokens,
            output=output,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            out_dtype=out_dtype,
            global_experts=global_e,
        )
        reason = kernel.supports(w.spec, m, topk, call)
        if reason:
            log_once(f"aiter.moe: {kernel.id} declines a call at M={m}: {reason}")
            _record("aiter", res.family, w.gpu, f"declined: {reason}", (m, topk))
            return None
    except Exception as exc:
        if strict():
            raise
        log_once(f"aiter.moe: router error, using today's path: {exc!r}")
        _record("aiter", "", "", f"router error {exc!r}", None)
        return None
    try:
        out = kernel(call)
    except Exception as exc:
        if strict():
            raise
        if not capturing:
            w.mark_failed(kernel, exc, f"M={m}")
        else:
            log_once(f"aiter.moe: {kernel.id} failed during capture: {exc!r}")
        _record("aiter", res.family, w.gpu, f"kernel error {exc!r}", (m, topk))
        return None
    _record(kernel.id, res.family, w.gpu, f"route {res.route.source}", (m, topk))
    return out
