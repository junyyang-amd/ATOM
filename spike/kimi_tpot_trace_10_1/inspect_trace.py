"""Per rank of one kineto capture: kernel events, kernel launches, the
gpu_user_annotation names with counts, and the share of pure-decode iteration
time that kernel intervals cover.

An iteration runs from the start of one `decode[...]` step with no `p=`
field and the given batch size to the start of the next step of any kind
(`decode`, `prefill`, ...): the target forward, the draft and its KV write.

Usage: python3 inspect_trace.py <trace-dir> [--bs N]
"""

import argparse
import collections
import glob
import gzip
import itertools
import json
import os
import re


def iterations(ev, bs):
    windows = sorted(
        (e["ts"], e["dur"], e["name"])
        for e in ev
        if e.get("cat") == "gpu_user_annotation"
        and re.match(r"(dummy_|eager_)*(decode|prefill)\[", e["name"])
    )
    steps = []  # a step's window repeats on each stream it runs: keep the first
    for w in windows:
        if not steps or w[0] >= steps[-1][0] + steps[-1][1]:
            steps.append(w)
    out = []
    for (ts, _, name), (nts, _, _) in itertools.pairwise(steps):
        m = re.match(r"decode\[bs=(\d+)", name)
        if m and " p=" not in name and (bs is None or int(m.group(1)) == bs):
            out.append((ts, nts))
    return out


def covered(kernels, lo, hi):
    """Length of the union of kernel intervals clipped to [lo, hi)."""
    total, end = 0.0, lo
    for s, e in kernels:
        if e <= end:
            continue
        if s >= hi:
            break
        s = max(s, end)
        e = min(e, hi)
        if e > s:
            total += e - s
            end = e
    return total


def inspect(path, bs):
    with gzip.open(path) as f:
        d = json.load(f)
    ev = d["traceEvents"]
    cats = collections.Counter(e.get("cat") for e in ev)
    launches = collections.Counter(
        e["name"]
        for e in ev
        if e.get("cat") == "cuda_runtime"
        and ("LaunchKernel" in e["name"] or e["name"] == "hipGraphLaunch")
    )
    ann = collections.Counter(
        re.split(r"\[| tokens=", e["name"])[0]
        for e in ev
        if e.get("cat") == "gpu_user_annotation"
    )
    kernels = sorted(
        (e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") == "kernel"
    )
    its = iterations(ev, bs)
    span = sum(h - lo for lo, h in its)
    cov = sum(covered(kernels, lo, h) for lo, h in its)
    return {
        "hip": d.get("hip_runtime_version"),
        "roctracer": d.get("roctracer_version"),
        "kernel_events": cats["kernel"],
        "launches": dict(launches.most_common()),
        "annotations": dict(ann.most_common()),
        "iterations": len(its),
        "iter_mean_us": round(span / len(its), 1) if its else None,
        "kernel_coverage": round(cov / span, 4) if span else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--bs", type=int, help="batch size of pure-decode steps")
    a = ap.parse_args()
    files = sorted(
        glob.glob(os.path.join(a.trace_dir, "**/*.pt.trace.json.gz"), recursive=True)
    )
    for path in files:
        rank = os.path.basename(os.path.dirname(path))
        print(rank, json.dumps(inspect(path, a.bs)), flush=True)


if __name__ == "__main__":
    main()
