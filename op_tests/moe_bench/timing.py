# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""HIP-event timing of a whole-call closure, as serving runs it (graph replay).

``run_perftest`` is not used: it sums profiler kernel times and drops buffer
rotation when the weights are large. Here the timed interval is the HIP events
around one graph replay of the entire call, so launch gaps count. Where graph
capture fails the same closure is timed eagerly and the row says so.
"""

import math
import statistics

import torch

FLUSH_BYTES = 512 << 20  # L2 plus MALL is 256 MB on MI300X, MI325X and MI355X
MIN_WARM_ITERS = 10
MIN_WARM_MS = 50.0
MAX_WARM_ITERS = 200
MIN_ITERS = 50
MIN_MS = 200.0
MAX_ITERS = 5000
EVENT_CHUNK = 100  # thousands of queued events and replays segfault torch.cuda.Event.record on gfx950


class Timer:
    """Holds the replay closure, the graph (or eager fallback) and cold-mode state."""

    def __init__(self, call, hidden_static=None, hidden_pool=None):
        self.call = call
        self.hidden_static = hidden_static
        self.hidden_pool = hidden_pool or []
        self.graph = None
        self.mode = "eager"
        self.capture_error = ""
        self.scratch = None
        self.out = None

    def prepare_graph(self, warmup_calls=3):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup_calls):
                self.out = self.call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                self.out = self.call()
            g.replay()
            torch.cuda.synchronize()
            self.graph = g
            self.mode = "graph"
        except Exception as e:  # noqa: BLE001
            self.graph = None
            self.mode = "eager"
            self.capture_error = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
            torch.cuda.synchronize()
        return self.mode

    def replay(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self.out = self.call()

    def _prepare_cold(self, i):
        if self.scratch is None:
            self.scratch = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device="cuda")
        self.scratch.fill_(i & 0xFF)
        if self.hidden_pool and self.hidden_static is not None:
            self.hidden_static.copy_(self.hidden_pool[i % len(self.hidden_pool)])

    def _measure(self, n, cold):
        out = []
        chunk = EVENT_CHUNK
        for lo in range(0, n, chunk):
            m = min(chunk, n - lo)
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(m)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(m)]
            for i in range(m):
                if cold:
                    self._prepare_cold(lo + i)
                starts[i].record()
                self.replay()
                ends[i].record()
            torch.cuda.synchronize()
            out += [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
        return out

    def time(self, cold=False, quick=False):
        """Warm up until stable, then time. Returns a dict of stats in microseconds.

        ``quick`` is for screening candidates: fewer iterations, same events and graph.
        """
        if quick:
            self._measure(3, cold)
            ts = sorted(self._measure(15, cold))
            n = max(15, min(300, math.ceil(20_000.0 / max(ts[len(ts) // 2], 1e-3))))
            ts = sorted(self._measure(n, cold))
            return {
                "median_us": statistics.median(ts),
                "p10_us": ts[int(0.1 * (len(ts) - 1))],
                "p90_us": ts[int(0.9 * (len(ts) - 1))],
                "std_us": statistics.pstdev(ts) if len(ts) > 1 else 0.0,
                "iters": n,
                "warm_iters": 18,
            }
        warm_total_ms = 0.0
        iters = 0
        prev = None
        while iters < MAX_WARM_ITERS:
            block = self._measure(5, cold)
            iters += 5
            warm_total_ms += sum(block) / 1000.0
            med = statistics.median(block)
            stable = prev is not None and abs(med - prev) <= 0.05 * max(prev, 1e-9)
            prev = med
            if iters >= MIN_WARM_ITERS and warm_total_ms >= MIN_WARM_MS and stable:
                break
        n = max(MIN_ITERS, min(MAX_ITERS, math.ceil(MIN_MS * 1000.0 / max(prev, 1e-3))))
        ts = sorted(self._measure(n, cold))
        q = lambda p: ts[min(len(ts) - 1, int(p * (len(ts) - 1) + 0.5))]
        return {
            "median_us": statistics.median(ts),
            "p10_us": q(0.10),
            "p90_us": q(0.90),
            "std_us": statistics.pstdev(ts) if len(ts) > 1 else 0.0,
            "iters": n,
            "warm_iters": iters,
        }
