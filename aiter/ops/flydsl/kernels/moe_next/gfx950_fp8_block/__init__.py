# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 fp8 blockscale (per_1x128) MoE stage kernels: stage1 (gate/up, activation,
1x128 quant of the intermediate) and stage2 (down, route weight, atomic combine)."""
