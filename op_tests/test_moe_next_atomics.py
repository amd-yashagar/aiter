# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cache-policy bits of core's buffer atomics reach the ISA (moe_next/core/atomics.py).

Host-only: each arch compiles in a subprocess with COMPILE_ONLY, no GPU
visible, and the final ISA dumped, then the atomics' modifiers are checked.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

ARCHS = {
    "gfx942": "f16",
    "gfx950": "bf16",
}  # packed type each arch's buffer atomic takes


def _compile(arch, pk):
    """Subprocess side: compile the probe kernels for ``arch``."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import torch
    from flydsl.expr import rocdl

    from aiter.ops.flydsl.kernels.moe_next.core import atomics as A

    pk_t = fx.BFloat16 if pk == "bf16" else fx.Float16

    def rsrc(t):
        return fx.rocdl.get_buffer_rsrc(
            fx.get_iter(fx.rocdl.make_buffer_tensor(t, max_size=True))
        )

    @flyc.kernel(known_block_size=[64, 1, 1])
    def atomic_core_kernel(out: fx.Tensor, outf: fx.Tensor):
        tid = fx.gpu.thread_idx.x
        v = fx.full(2, 1.0, pk_t)
        A.buffer_atomic_add_pk(rsrc(out), tid * fx.Int32(2), v, 2, A.NT | A.SC1)
        A.buffer_atomic_add_f32(rsrc(outf), tid, fx.full(1, 1.0, fx.Float32), A.SC1)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def atomic_raw_kernel(out: fx.Tensor, outf: fx.Tensor):
        tid = fx.gpu.thread_idx.x
        v = fx.full(2, 1.0, pk_t)
        bits = fx.Int32(A.NT | A.SC1)
        rocdl.raw_ptr_buffer_atomic_fadd(
            v, rsrc(out), tid * fx.Int32(4), fx.Int32(0), bits
        )

    @flyc.jit
    def launch(out: fx.Tensor, outf: fx.Tensor, stream: fx.Stream = None):
        atomic_core_kernel(out, outf).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )
        atomic_raw_kernel(out, outf).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    launch(torch.zeros(128, dtype=torch.int16), torch.zeros(64, dtype=torch.float32))


def _isa(arch, pk):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        env = dict(
            os.environ,
            COMPILE_ONLY="1",
            ARCH=arch,
            FLYDSL_GPU_ARCH=arch,
            HIP_VISIBLE_DEVICES="",
            CUDA_VISIBLE_DEVICES="",
            FLYDSL_DUMP_IR="1",
            FLYDSL_DUMP_DIR=d,
        )
        subprocess.run(
            [sys.executable, __file__, "--compile", arch, pk],
            env=env,
            check=True,
            capture_output=True,
        )
        text = []
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith("final_isa.s"):
                    with open(os.path.join(root, f)) as fh:
                        text.append(fh.read())
        return "\n".join(text)


def _kernel(isa, name):
    m = re.search(rf"^{name}_\d+:\n(.*?)\n\s*s_endpgm", isa, re.DOTALL | re.MULTILINE)
    assert m, f"no {name} in the ISA"
    return [ln.strip() for ln in m.group(1).splitlines() if "buffer_atomic" in ln]


def test_atomic_cache_bits():
    """Core's helpers carry NT | SC1 (packed) and SC1 (f32) into the ISA on
    gfx942 and gfx950. The same bits passed to FlyDSL as an fx.Int32 are
    dropped (kept as a record of the FlyDSL behaviour the helpers avoid), and
    the helpers refuse anything but a Python int."""
    from aiter.ops.flydsl.kernels.moe_next.core import atomics as A

    rows = []
    for arch, pk in ARCHS.items():
        isa = _isa(arch, pk)
        core = _kernel(isa, "atomic_core_kernel")
        raw = _kernel(isa, "atomic_raw_kernel")
        pk_ops = [ln for ln in core if f"buffer_atomic_pk_add_{pk}" in ln]
        f32_ops = [ln for ln in core if "buffer_atomic_add_f32" in ln]
        assert pk_ops and all(re.search(r"\bnt\b.*\bsc1\b", ln) for ln in pk_ops), core
        assert f32_ops and all(re.search(r"\bsc1\b", ln) for ln in f32_ops), core
        assert not any(re.search(r"\b(nt|sc0|sc1)\b", ln) for ln in raw), raw
        rows.append(f"{arch}: {pk_ops[0]} | {f32_ops[0]} | fx.Int32 aux: {raw[0]}")
    for bad in (True, 2.0, None):
        try:
            A.cache_bits(bad)
        except TypeError:
            continue
        raise AssertionError(f"cache_bits accepted {bad!r}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--compile", nargs=2, metavar=("ARCH", "PK"))
    a = p.parse_args()
    if a.compile:
        _compile(*a.compile)
    else:
        for r in test_atomic_cache_bits():
            print(r)
