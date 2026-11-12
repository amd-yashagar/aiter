# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only: the fmoe tuner offers an ASM candidate only for the activation it computes.

The 2-stage ASM stage-1 kernels (``fmoe_stage1_*``) fuse SiLU in their epilogue and name no
activation, yet the tuner added them for every activation, so a Swiglu or Gelu shape was
tuned against kernels whose output is wrong (rel_l2 about 1750 on the fp8 per-token 128x64
kernel). Kernel symbols that name an activation serve only that one.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from aiter import ActivationType, QuantType, dtypes

_ASM_DIR = str(REPO_ROOT / "hsa" / "gfx942")


def _tune_module():
    from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

    return gemm_moe_tune


class TestKernelActivationMatch(unittest.TestCase):
    def test_symbols_that_name_an_activation_serve_only_that_one(self):
        match = _tune_module()._asm_kernel_matches_activation
        silu = "_ZN5aiter49fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x32E"
        swiglu = "_ZN5aiter56fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_swiglu_oai_1"
        self.assertTrue(match(silu, ActivationType.Silu))
        self.assertFalse(match(silu, ActivationType.Swiglu))
        self.assertTrue(match(swiglu, ActivationType.Swiglu))
        self.assertFalse(match(swiglu, ActivationType.Silu))
        self.assertFalse(match(silu, ActivationType.Gelu))

    def test_unnamed_stage1_kernels_are_silu_only(self):
        match = _tune_module()._asm_kernel_matches_activation
        name = "_ZN5aiter44fmoe_stage1_bf16_pertokenFp8_g1u1_128x64_pf2E"
        self.assertTrue(match(name, ActivationType.Silu, fused_stage1=True))
        for act in (ActivationType.Swiglu, ActivationType.Gelu, ActivationType.Situv2):
            self.assertFalse(match(name, act, fused_stage1=True))


class TestStage1Candidates(unittest.TestCase):
    def _tasks(self, act):
        tune = _tune_module()
        tuner = tune.FmoeTuner.__new__(tune.FmoeTuner)
        key = (
            "gfx942",
            304,
            16,
            7168,
            256,
            8,
            2,
            act,
            dtypes.bf16,
            dtypes.fp8,
            dtypes.fp8,
            QuantType.per_Token,
            True,
            False,
        )
        with mock.patch.object(tune, "get_asm_dir", return_value=_ASM_DIR):
            return tuner.gen_2stages_asm1_task(key, [16, 32, 64, 128])

    @staticmethod
    def _stage1_kernels(tasks):
        return [t[0][2] for t in tasks if t[0][1] == "stage1"]

    def test_silu_keeps_the_asm_stage1_candidates(self):
        self.assertTrue(self._stage1_kernels(self._tasks(ActivationType.Silu)))

    def test_swiglu_and_gelu_get_no_silu_only_stage1_candidate(self):
        for act in (ActivationType.Swiglu, ActivationType.Gelu):
            with self.subTest(act=act):
                self.assertEqual(self._stage1_kernels(self._tasks(act)), [])

    def test_the_128x64_kernel_is_among_the_silu_candidates_only(self):
        silu = self._stage1_kernels(self._tasks(ActivationType.Silu))
        self.assertTrue(any("128x64" in k for k in silu))
        swiglu = self._stage1_kernels(self._tasks(ActivationType.Swiglu))
        self.assertFalse(any("128x64" in k for k in swiglu))


if __name__ == "__main__":
    unittest.main()
