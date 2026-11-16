#!/usr/bin/env python3
"""Per-component decode step time per case and image, and the ROCm 10.1 gap.

<dir> holds summarize.py's <case>.<image>.trace.json and, when present,
itl.py's <case>.itl.json. Each image's per-batch-size step, idle and kernel
times are pooled over its runs, then weighted by the batch-size histogram of
both images' steps pooled, so component deltas sum to the traced gap.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from itl import BASE, NEW

# Kernel name pattern to component; a kernel takes its first match.
COMPONENTS = [
    (
        "communication",
        r"cross_device_reduce|nccl|rccl|msccl|_dcp_a2a_|all_?reduce|all_?gather",
    ),
    ("MLA attention", r"mla"),
    ("KDA", r"kda|causal_conv1d|rmsnorm_gated"),
    ("MoE", r"moe|topk|gemm[12]_a4w4|situv2_and_mul"),
    ("GEMM", r"gemm|gemv|flatmm"),
]
UNMATCHED = "unmatched kernels"


def component(kernel):
    for name, pattern in COMPONENTS:
        if re.search(pattern, kernel, re.IGNORECASE):
            return name
    return UNMATCHED


def pool(runs):
    """{bs: (steps, {row: us per step})} over runs, step-weighted."""
    pooled = defaultdict(lambda: [0.0, defaultdict(float)])
    for run in runs:
        for bs, s in run["batch_sizes"].items():
            p = pooled[bs]
            p[0] += s["steps"]
            p[1]["step"] += s["steps"] * s["step_us"]
            p[1]["idle"] += s["steps"] * s["idle_us"]
            for kernel, us in s["kernels"].items():
                p[1][component(kernel)] += s["steps"] * us
    return {
        bs: (n, {k: v / n for k, v in rows.items()}) for bs, (n, rows) in pooled.items()
    }


def weighted(images):
    """{image: {row: us per step}} under the pooled batch-size weights."""
    sizes = set.intersection(*(set(p) for p in images.values()))
    counts = {bs: sum(p[bs][0] for p in images.values()) for bs in sizes}
    total = sum(counts.values())
    out = {}
    for image, p in images.items():
        rows = defaultdict(float)
        for bs in sizes:
            for row, us in p[bs][1].items():
                rows[row] += counts[bs] / total * us
        out[image] = rows
    return out


def a1_gaps(itl_json):
    images = json.loads(itl_json.read_text())
    lines = []
    for key, what in (
        ("mean_itl_ms", "whole hour"),
        ("window_mean_itl_ms", "900-930 s"),
    ):
        new, base = (
            statistics.mean(r[key] for r in images[i].values()) for i in (NEW, BASE)
        )
        lines.append(
            f"A1 gap, {what}: {new - base:+.3f} ms ({100 * (new - base) / base:+.1f}%)"
        )
    return lines


def report(case, summaries, itl_json):
    for image in (NEW, BASE):
        if image not in summaries:
            raise SystemExit(f"{case}: no {image} trace summary")
    images = sorted(summaries)
    rows = weighted({i: pool(summaries[i]["runs"]) for i in images})
    lines = [
        case,
        f"{'ms per step':<20}" + "".join(f"{i:>22}" for i in images) + f"{'delta':>10}",
    ]
    for row in [c for c, _ in COMPONENTS] + [UNMATCHED, "idle", "step"]:
        figures = [rows[i][row] / 1000 for i in images]
        delta = (rows[NEW][row] - rows[BASE][row]) / 1000
        lines.append(
            f"{row:<20}" + "".join(f"{f:>22.3f}" for f in figures) + f"{delta:>+10.3f}"
        )
    gap = rows[NEW]["step"] - rows[BASE]["step"]
    lines.append(
        f"traced gap: {gap / 1000:+.3f} ms ({100 * gap / rows[BASE]['step']:+.1f}%)"
    )
    if itl_json.exists():
        lines += a1_gaps(itl_json)
    other = sum(rows[NEW][r] - rows[BASE][r] for r in (UNMATCHED, "idle"))
    share = f"{100 * (gap - other) / gap:.0f}%" if gap else "n/a, the gap is 0"
    lines.append(f"explained share: {share}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir")
    args = parser.parse_args()
    cases = defaultdict(dict)
    for path in sorted(Path(args.dir).glob("*.trace.json")):
        summary = json.loads(path.read_text())
        cases[summary["case"]][summary["image"]] = summary
    if not cases:
        raise SystemExit(f"{args.dir}: no *.trace.json")
    for case, summaries in sorted(cases.items()):
        print(report(case, summaries, Path(args.dir) / f"{case}.itl.json"))
        print()


if __name__ == "__main__":
    main()
