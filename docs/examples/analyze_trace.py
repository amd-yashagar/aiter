"""
analyze_trace.py — extract kernel ticket facts from a raw Kineto/roctracer trace.

Works with any op, including ops that TraceLens does not recognize.

Usage:
    python analyze_trace.py <eager_trace.json.gz> [options]

Options:
    --threshold FLOAT   Only report ops >= this % of total GPU kernel time (default: 2.0)
    --op NAME           Show full detail for a specific op name (can repeat)
    --json PATH         Write structured JSON output to PATH (for downstream tools)

Memory tip: eager traces are large (200-400 MB gzip, 15-20 M events).  The
parser skips python_function events to stay lean, but still expects 1-2 GB RAM
free during load.
"""

import argparse
import collections
import gzip
import json
import sys

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_trace(path: str) -> list:
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path) as fh:
        d = json.load(fh)
    # support both {"traceEvents": [...]} and bare [...]
    return d["traceEvents"] if isinstance(d, dict) else d


def build_attribution(events: list):
    """Return (op_stats, per_op_kernels, per_op_shapes).

    op_stats:        op_name -> {"total_us": float, "calls": int}
    per_op_kernels:  op_name -> {kernel_name -> {"total_us", "calls", "min_us", "max_us"}}
    per_op_shapes:   op_name -> list of {dims, types, strides, concrete, count, total_us,
                                          min_us, max_us}
    """
    # cpu_op indexed by External id
    cpu_by_extid: dict[int, dict] = {}
    # cuda_runtime: correlation -> External id
    rt_corr_to_ext: dict[int, int] = {}

    for e in events:
        cat = e.get("cat")
        if cat == "cpu_op":
            ext = e["args"].get("External id")
            if ext is not None:
                cpu_by_extid[ext] = e
        elif cat == "cuda_runtime":
            corr = e["args"].get("correlation")
            ext = e["args"].get("External id")
            if corr is not None and ext is not None:
                rt_corr_to_ext[corr] = ext

    op_stats: dict[str, dict] = collections.defaultdict(lambda: {"total_us": 0.0, "calls": 0})
    per_op_kernels: dict[str, dict] = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: {"total_us": 0.0, "calls": 0, "min_us": float("inf"), "max_us": 0.0}
        )
    )
    # shape variants: keyed by (op_name, frozen dims/types/strides/concrete)
    shape_accum: dict = {}  # (op_name, shape_key) -> dict

    for e in events:
        if e.get("cat") != "kernel":
            continue
        dur = e.get("dur", 0.0)
        kname = e.get("name", "")
        corr = e["args"].get("correlation")
        ext = rt_corr_to_ext.get(corr)
        if ext is None:
            ext = e["args"].get("External id")
        cpu = cpu_by_extid.get(ext) if ext is not None else None
        if cpu is None:
            continue

        op_name = cpu["name"]
        op_stats[op_name]["total_us"] += dur
        op_stats[op_name]["calls"] += 1

        kd = per_op_kernels[op_name][kname]
        kd["total_us"] += dur
        kd["calls"] += 1
        kd["min_us"] = min(kd["min_us"], dur)
        kd["max_us"] = max(kd["max_us"], dur)

        # shape variants (keyed on the cpu_op's args — same shape group shares same cpu_op)
        args = cpu["args"]
        dims = json.dumps(args.get("Input Dims", []), separators=(",", ":"))
        types = json.dumps(args.get("Input type", []), separators=(",", ":"))
        strides = json.dumps(args.get("Input Strides", []), separators=(",", ":"))
        concrete = json.dumps(args.get("Concrete Inputs", []), separators=(",", ":"))
        sk = (op_name, dims, types, strides, concrete)
        if sk not in shape_accum:
            shape_accum[sk] = {
                "dims": args.get("Input Dims", []),
                "types": args.get("Input type", []),
                "strides": args.get("Input Strides", []),
                "concrete": args.get("Concrete Inputs", []),
                "count": 0,
                "total_us": 0.0,
                "min_us": float("inf"),
                "max_us": 0.0,
            }
        sv = shape_accum[sk]
        sv["count"] += 1
        sv["total_us"] += dur
        sv["min_us"] = min(sv["min_us"], dur)
        sv["max_us"] = max(sv["max_us"], dur)

    per_op_shapes: dict[str, list] = collections.defaultdict(list)
    for (op_name, *_), sv in shape_accum.items():
        per_op_shapes[op_name].append(sv)

    return op_stats, per_op_kernels, per_op_shapes


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt(us: float) -> str:
    if us >= 1000:
        return f"{us/1000:.2f} ms"
    return f"{us:.1f} µs"


def print_ranking(op_stats: dict, threshold: float):
    total = sum(s["total_us"] for s in op_stats.values())
    ranked = sorted(op_stats.items(), key=lambda kv: kv[1]["total_us"], reverse=True)
    print(f"\n{'Op':<60} {'%':>6}  {'total':>10}  {'calls':>7}")
    print("-" * 90)
    for name, s in ranked:
        pct = 100 * s["total_us"] / total if total else 0
        if pct < threshold:
            continue
        print(f"{name:<60} {pct:>5.1f}%  {_fmt(s['total_us']):>10}  {s['calls']:>7}")
    print(f"\n  Total attributed GPU kernel time: {_fmt(total)}")


def print_op_detail(
    op_name: str,
    op_stats: dict,
    per_op_kernels: dict,
    per_op_shapes: dict,
    total_us: float,
):
    s = op_stats.get(op_name)
    if s is None:
        print(f"\n[!] Op '{op_name}' not found in trace.")
        return
    pct = 100 * s["total_us"] / total_us if total_us else 0
    print(f"\n{'=' * 90}")
    print(f"Op: {op_name}")
    print(f"    {pct:.1f}% of total GPU kernel time  |  {_fmt(s['total_us'])} total  |  {s['calls']} calls")

    print(f"\n  GPU kernels dispatched:")
    kernels = sorted(
        per_op_kernels[op_name].items(), key=lambda kv: kv[1]["total_us"], reverse=True
    )
    for kname, kd in kernels:
        mean = kd["total_us"] / kd["calls"] if kd["calls"] else 0
        print(f"    [{kd['calls']:>5}×] mean {_fmt(mean):>10}  total {_fmt(kd['total_us']):>10}  "
              f"min {_fmt(kd['min_us']):>10}  max {_fmt(kd['max_us']):>10}")
        # wrap long kernel names
        for chunk in [kname[i:i+80] for i in range(0, len(kname), 80)]:
            print(f"           {chunk}")

    shapes = sorted(per_op_shapes[op_name], key=lambda x: x["total_us"], reverse=True)
    print(f"\n  Shape variants ({len(shapes)} distinct call group(s)):")
    for i, sv in enumerate(shapes, 1):
        mean = sv["total_us"] / sv["count"] if sv["count"] else 0
        print(f"\n  Variant {i}  ({sv['count']} calls,  mean {_fmt(mean)},  "
              f"min {_fmt(sv['min_us'])},  max {_fmt(sv['max_us'])})")
        print(f"    Input Dims:      {sv['dims']}")
        print(f"    Input type:      {sv['types']}")
        print(f"    Input Strides:   {sv['strides']}")
        print(f"    Concrete Inputs: {sv['concrete']}")


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_json_output(op_stats, per_op_kernels, per_op_shapes, threshold):
    total = sum(s["total_us"] for s in op_stats.values())
    out = {"total_gpu_kernel_us": total, "ops": []}
    for name, s in sorted(op_stats.items(), key=lambda kv: kv[1]["total_us"], reverse=True):
        pct = 100 * s["total_us"] / total if total else 0
        if pct < threshold:
            continue
        kernels = [
            {
                "name": kn,
                "calls": kd["calls"],
                "total_us": kd["total_us"],
                "mean_us": kd["total_us"] / kd["calls"] if kd["calls"] else 0,
                "min_us": kd["min_us"],
                "max_us": kd["max_us"],
            }
            for kn, kd in sorted(
                per_op_kernels[name].items(), key=lambda kv: kv[1]["total_us"], reverse=True
            )
        ]
        shapes = [
            {
                "dims": sv["dims"],
                "types": sv["types"],
                "strides": sv["strides"],
                "concrete": sv["concrete"],
                "calls": sv["count"],
                "total_us": sv["total_us"],
                "mean_us": sv["total_us"] / sv["count"] if sv["count"] else 0,
                "min_us": sv["min_us"] if sv["min_us"] != float("inf") else None,
                "max_us": sv["max_us"],
            }
            for sv in sorted(per_op_shapes[name], key=lambda x: x["total_us"], reverse=True)
        ]
        out["ops"].append({
            "name": name,
            "pct": round(pct, 2),
            "total_us": s["total_us"],
            "calls": s["calls"],
            "kernels": kernels,
            "shapes": shapes,
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("trace", help="Path to eager-mode .json.gz trace")
    p.add_argument(
        "--threshold", type=float, default=2.0,
        help="Report ops >= this %% of total GPU kernel time (default: 2.0)",
    )
    p.add_argument(
        "--op", action="append", dest="ops", default=[],
        help="Show full detail for this op name (can repeat)",
    )
    p.add_argument("--json", dest="json_out", help="Write JSON output to this file")
    args = p.parse_args()

    print(f"Loading trace: {args.trace}", file=sys.stderr)
    events = load_trace(args.trace)
    print(f"  {len(events):,} events loaded", file=sys.stderr)

    print("Attributing GPU kernel time ...", file=sys.stderr)
    op_stats, per_op_kernels, per_op_shapes = build_attribution(events)

    print_ranking(op_stats, args.threshold)

    total = sum(s["total_us"] for s in op_stats.values())
    for op_name in args.ops:
        print_op_detail(op_name, op_stats, per_op_kernels, per_op_shapes, total)

    if args.json_out:
        out = build_json_output(op_stats, per_op_kernels, per_op_shapes, args.threshold)
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nJSON written to {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
