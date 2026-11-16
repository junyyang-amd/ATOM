#!/usr/bin/env python3
"""Per-image mean ITL and the ROCm 10.1 gap from a directory of aiperf runs.

<out-dir> holds one <image-tag>-r<k>/ per run, each with the CI's
benchmark_results/: the pd-atom-*.json results and aiperf's
aiperf-*/profile_export_aiperf_timeslices.csv.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

NEW = "ROCm_10.1.0"
BASE = "nightly_202611021458"
# Seconds into the profiling phase, counted from slice 0's start: the trace
# window the traced runs capture.
WINDOW_S = (900, 930)


class MissingResults(Exception):
    pass


def only(run, pattern):
    found = sorted((run / "benchmark_results").glob(pattern))
    if len(found) != 1:
        raise MissingResults(f"{run}: {len(found)} files match {pattern}, want 1")
    return found[0]


def window_mean(timeslices_csv):
    """Unweighted mean of per-slice ITL avg over slices starting in WINDOW_S.

    aiperf writes no slice for a second without data, so a slice's index is
    not its second: the window is found by Start_NS.
    """
    start0 = None
    values = []
    with open(timeslices_csv, newline="") as f:
        for row in csv.DictReader(f):
            start = int(row["Start_NS"])
            if start0 is None:
                start0 = start
            if row["Metric"] != "Inter Token Latency" or row["Stat"] != "avg":
                continue
            if WINDOW_S[0] <= (start - start0) / 1e9 < WINDOW_S[1]:
                values.append(float(row["Value"]))
    if not values:
        raise MissingResults(f"{timeslices_csv}: no ITL slice in {WINDOW_S} s")
    return statistics.mean(values)


def run_figures(run):
    results = json.loads(only(run, "pd-atom-*.json").read_text())
    return {
        "mean_itl_ms": results["mean_itl_ms"],
        "median_itl_ms": results["median_itl_ms"],
        "window_mean_itl_ms": window_mean(
            only(run, "aiperf-*/profile_export_aiperf_timeslices.csv")
        ),
    }


def collect(out_dir):
    """{image: {run: figures}} for every <image>-r<k>/ under out_dir."""
    images = {}
    for run in sorted(Path(out_dir).iterdir()):
        image, sep, k = run.name.rpartition("-r")
        if run.is_dir() and sep and k.isdigit():
            images.setdefault(image, {})[f"r{k}"] = run_figures(run)
    for image in (NEW, BASE):
        if image not in images:
            raise MissingResults(f"{out_dir}: no {image}-r<k>/ run")
    return images


def report(images):
    lines = []
    means = {}
    for image, runs in images.items():
        mean = [r["mean_itl_ms"] for r in runs.values()]
        median = [r["median_itl_ms"] for r in runs.values()]
        means[image] = statistics.mean(mean)
        lines.append(
            f"{image}: mean ITL {means[image]:.2f} ms "
            f"(min-max {min(mean):.2f}-{max(mean):.2f}, {len(mean)} runs), "
            f"median ITL {statistics.mean(median):.2f} ms"
        )
    gap = means[NEW] - means[BASE]
    lines.append(
        f"gap {NEW} minus {BASE}: {gap:+.2f} ms ({100 * gap / means[BASE]:+.1f}%)"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("--json", help="write per image and run figures here")
    args = parser.parse_args()
    try:
        images = collect(args.out_dir)
    except MissingResults as e:
        raise SystemExit(f"MissingResults: {e}")
    print(report(images))
    if args.json:
        Path(args.json).write_text(json.dumps(images, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
