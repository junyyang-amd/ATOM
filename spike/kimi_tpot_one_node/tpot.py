"""Table of the one-node A/B: per launch and ISL, the benchmark's median TPOT
and the steady TPOT, then per image the mean and min-max over launches.

The steady TPOT counts only the window in which all 48 requests decode: from
the last request's first token to the first request's last token. The
scheduler runs prefill first, so a request that finished prefill early waits
for the others' prefills, which the benchmark's TPOT books as decode time.

Usage: python3 tpot.py OUT_DIR   (reads OUT_DIR/<image-tag>-r<k>/isl<N>.json)
"""

import json
import re
import statistics
import sys
from pathlib import Path


def steady_tpot_ms(res):
    """Median over requests of window time / tokens emitted in the window."""
    times = []
    for ttft, itl in zip(res["ttfts"], res["itls"]):
        t, ts = ttft, [ttft]
        for gap in itl:
            t += gap
            ts.append(t)
        times.append(ts)
    t0 = max(ts[0] for ts in times)
    t1 = min(ts[-1] for ts in times)
    if t1 <= t0:
        return float("nan")
    per_req = []
    for ts, out in zip(times, res["output_lens"]):
        chunks = sum(t0 < t <= t1 for t in ts)
        tokens_per_chunk = (out - 1) / (len(ts) - 1)
        per_req.append((t1 - t0) / (chunks * tokens_per_chunk))
    return statistics.median(per_req) * 1000


def main(out):
    rows = {}  # (isl, tag) -> [(launch, median, steady)]
    for path in sorted(Path(out).glob("*-r*/isl*.json")):
        tag, launch = re.fullmatch(r"(.+)-r(\d+)", path.parent.name).groups()
        isl = int(path.stem[3:])
        res = json.loads(path.read_text())
        assert res["completed"] == 48, f"{path}: {res['completed']} completed"
        rows.setdefault((isl, tag), []).append(
            (int(launch), res["median_tpot_ms"], steady_tpot_ms(res))
        )
    print("| image | ISL | launch | median TPOT ms | steady TPOT ms |")
    print("|---|---|---|---|---|")
    for (isl, tag), runs in sorted(rows.items()):
        for launch, med, steady in sorted(runs):
            print(f"| {tag} | {isl} | {launch} | {med:.2f} | {steady:.2f} |")
    print()
    print("| image | ISL | n | median TPOT mean [min-max] | steady TPOT mean [min-max] |")
    print("|---|---|---|---|---|")
    stats = {}
    for (isl, tag), runs in sorted(rows.items()):
        cols = []
        for k in (1, 2):
            v = [r[k] for r in runs]
            stats[isl, tag, k] = (statistics.mean(v), min(v), max(v))
            cols.append("%.2f [%.2f-%.2f]" % stats[isl, tag, k])
        print(f"| {tag} | {isl} | {len(runs)} | {cols[0]} | {cols[1]} |")
    print()
    for isl in sorted({i for i, _ in rows}):
        tags = sorted(t for i, t in rows if i == isl)
        if len(tags) != 2:
            continue
        new, old = (t for t in tags if "10.1" in t), (t for t in tags if "10.1" not in t)
        new, old = next(new), next(old)
        for k, label in ((1, "median"), (2, "steady")):
            n, o = stats[isl, new, k], stats[isl, old, k]
            beyond = n[1] > o[2]
            print(
                f"ISL {isl} {label} TPOT: {new} {n[0]:.2f} vs {old} {o[0]:.2f} ms, "
                f"gap {100 * (n[0] / o[0] - 1):+.1f}%, "
                f"{'beyond' if beyond else 'not beyond'} both ranges"
            )


if __name__ == "__main__":
    main(sys.argv[1])
