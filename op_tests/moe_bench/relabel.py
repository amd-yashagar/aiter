# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU-free relabel of dispatch rows whose route read ``<lambda>``.

Before ``routes.install_impl_tag`` a whole-graph impl (``impl__flydsl_*``) had no name
and the route columns of its dispatch rows said ``<lambda>``. The timings are right; the
labels are not. This re-resolves each affected case's dispatch route on meta tensors and
writes ``dispatch-labels.csv`` beside the baseline. The baseline file is not touched.

    python -m op_tests.moe_bench.relabel --baseline tier1.csv --cases tier1.csv
"""

import argparse
import csv
import os
import sys

LABEL_COLS = ("route_key", "route_family", "kernel1", "kernel2", "full_impl")


def affected_rows(rows):
    return [
        r
        for r in rows
        if any("<lambda>" in (r.get(c) or "") for c in LABEL_COLS)
        or (r.get("full_impl") or "") == "<lambda>"
    ]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="dispatch CSV to inspect")
    ap.add_argument("--cases", required=True, help="tier CSV with the case rows")
    ap.add_argument("--out", help="default: dispatch-labels.csv beside the baseline")
    a = ap.parse_args(argv)
    with open(a.baseline, newline="") as f:
        rows = list(csv.DictReader(f))
    bad = affected_rows(rows)
    print(f"{len(bad)} of {len(rows)} rows read <lambda>", flush=True)
    if not bad:
        return 0
    from . import routes, runner, workload

    cases = {c.id: c for c in runner.select_cases([a.cases])}
    out = a.out or os.path.join(
        os.path.dirname(os.path.abspath(a.baseline)), "dispatch-labels.csv"
    )
    done = {}
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["case_id", "mode", "old_route_key", "old_kernel1"]
            + [f"new_{c}" for c in LABEL_COLS]
        )
        for r in bad:
            cid = r["case_id"]
            if cid not in done:
                done[cid] = routes.dry_route(cases[cid], workload.resolve(cases[cid]))
            n = done[cid]
            w.writerow(
                [cid, r.get("mode", ""), r.get("route_key", ""), r.get("kernel1", "")]
                + [n.get(c, "") for c in LABEL_COLS]
            )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
