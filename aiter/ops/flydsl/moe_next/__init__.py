# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of the aiter.moe kernels: launch wrappers, compile caches, tile
configs, and the MoeKernel classes. Importing this package registers every
family kernel with aiter.moe; kernel bodies are in kernels/moe_next/."""
