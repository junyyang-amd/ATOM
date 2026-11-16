#!/usr/bin/env python3
"""Per-step summary of a case's traced decode runs for one image.

Each <run-trace-dir> holds one kineto trace per decode rank
(*.pt.trace.json[.gz], at any depth). A step runs from the start of a
pure-decode window (decode[...] without p=) to the start of the next step
window; a window repeated on a second stream counts once. Each instant of a
step is shared equally among the kernels running then; an instant with none is
idle, so per batch size the kernel times plus idle equal the step time.

Writes <out>/<case>.<image>.trace.json: per run and real batch size the step
count and the step time, idle and per-kernel time per step, averaged over the
run's ranks.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_trace_summary import load_trace_events, parse_label  # noqa: E402

STEP_WINDOWS = {
    "decode",
    "prefill",
    "eager_decode",
    "dummy_decode",
    "dummy_eager_decode",
    "dummy_prefill",
}
GPU_WORK = {"kernel", "gpu_memcpy", "gpu_memset"}


def steps(events):
    """[(start, end, real batch size)] of the pure-decode steps."""
    windows = sorted(
        (e["ts"], e["dur"], parse_label(e["name"]))
        for e in events
        if e.get("cat") == "gpu_user_annotation"
        and parse_label(e["name"])["type"] in STEP_WINDOWS
    )
    kept = []  # a window repeated on another stream starts inside the first
    for w in windows:
        if not kept or w[0] >= kept[-1][0] + kept[-1][1]:
            kept.append(w)
    return [
        (ts, next_ts, int(label["bs"].split("/")[0]))
        for (ts, _, label), (next_ts, _, _) in zip(kept, kept[1:])
        if label["type"] == "decode" and "p" not in label
    ]


def rank_totals(events):
    """{bs: [steps, step_us, idle_us, Counter(kernel: us)]} summed over steps."""
    points = []
    for e in events:
        if e.get("cat") in GPU_WORK:
            points.append((e["ts"], 1, e["name"]))
            points.append((e["ts"] + e["dur"], -1, e["name"]))
    totals = defaultdict(lambda: [0, 0.0, 0.0, Counter()])
    for start, end, bs in steps(events):
        totals[bs][0] += 1
        totals[bs][1] += end - start
        points.append((start, 2, bs))
        points.append((end, -2, None))
    # At one instant a step's end (-2) comes before the next step's start (2).
    points.sort(key=lambda p: p[:2])
    running = Counter()
    n_running = 0
    step_bs = None
    prev = None
    for t, kind, payload in points:
        if step_bs is not None and t > prev:
            total = totals[step_bs]
            dt = t - prev
            if n_running == 0:
                total[2] += dt
            else:
                for name, n in running.items():
                    total[3][name] += dt * n / n_running
        prev = t
        if kind == 1:
            running[payload] += 1
            n_running += 1
        elif kind == -1:
            running[payload] -= 1
            n_running -= 1
            if not running[payload]:
                del running[payload]
        else:
            step_bs = payload
    return totals


def summarize_run(run_dir):
    traces = sorted(Path(run_dir).rglob("*.pt.trace.json*"))
    if not traces:
        raise SystemExit(f"{run_dir}: no *.pt.trace.json[.gz]")
    pooled = defaultdict(lambda: [0, 0.0, 0.0, Counter()])
    for trace in traces:
        for bs, (n, step_us, idle_us, kernels) in rank_totals(
            load_trace_events(str(trace))
        ).items():
            p = pooled[bs]
            p[0] += n
            p[1] += step_us
            p[2] += idle_us
            p[3].update(kernels)
    batch_sizes = {}
    for bs, (n, step_us, idle_us, kernels) in sorted(pooled.items()):
        batch_sizes[str(bs)] = {
            "steps": n / len(traces),
            "step_us": round(step_us / n, 3),
            "idle_us": round(idle_us / n, 3),
            "kernels": {
                k: round(us / n, 3) for k, us in sorted(kernels.items()) if us > 0
            },
        }
    return {"dir": str(run_dir), "ranks": len(traces), "batch_sizes": batch_sizes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("image")
    parser.add_argument("run_trace_dirs", nargs="+")
    parser.add_argument("--out", default="docs/kimi_k3_rocm10_tpot")
    args = parser.parse_args()
    summary = {
        "case": args.case,
        "image": args.image,
        "runs": [summarize_run(d) for d in args.run_trace_dirs],
    }
    out = Path(args.out) / f"{args.case}.{args.image}.trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(out)


if __name__ == "__main__":
    main()
