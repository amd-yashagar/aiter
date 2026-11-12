# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Integer handles for MoeWeights, so the custom op takes tensors and scalars only.

The registry holds weak references: a layer's weights live as long as the
caller keeps its MoeWeights, and a handle never keeps them alive.
"""

import itertools
import threading
import weakref

_LOCK = threading.Lock()
_NEXT = itertools.count(1)
_WEIGHTS: "weakref.WeakValueDictionary[int, object]" = weakref.WeakValueDictionary()


def register(weights) -> int:
    with _LOCK:
        handle = next(_NEXT)
        _WEIGHTS[handle] = weights
    return handle


def lookup(handle: int):
    weights = _WEIGHTS.get(handle)
    if weights is None:
        raise KeyError(f"aiter.moe: no live MoeWeights for handle {handle}")
    return weights
