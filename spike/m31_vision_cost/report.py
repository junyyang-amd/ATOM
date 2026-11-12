"""Join load.py outputs with the wrapper's timing lines into the result table."""

import glob
import json
import statistics as st
import sys

out = sys.argv[1]
timing = [
    json.loads(line)
    for path in glob.glob(f"{out}/timing/*.jsonl")
    for line in open(path)
]
encodes = [t for t in timing if t["kind"] == "encode"]
prefills = [t for t in timing if t["kind"] == "prefill"]


def window(records, run):
    return [t for t in records if run["start"] <= t["start"] <= run["end"]]


def span(xs):
    return f"{st.median(xs):.1f} [{min(xs):.1f}, {max(xs):.1f}]"


def overlap(lo, hi):
    return sum(max(0.0, min(hi, e["end"]) - max(lo, e["start"])) for e in encodes)


def describe(tag):
    runs = sorted(glob.glob(f"{out}/{tag}*.json"))
    for path in runs:
        run = json.load(open(path))
        enc, pre = window(encodes, run), window(prefills, run)
        reqs = run["requests"]
        ttft = [r["ttft_ms"] for r in reqs]
        share = [overlap(r["send"], r["first"]) * 1e3 / r["ttft_ms"] for r in reqs]
        enc_ms = sum(e["end"] - e["start"] for e in enc) * 1e3
        pre_ms = sum(t["end"] - t["start"] for t in pre) * 1e3
        images = sum(e["images"] for e in enc)
        print(
            f"| {path.rsplit('/', 1)[1][:-5]} | {len(reqs)} | {images} "
            f"| {span([e['rows'] / e['images'] for e in enc])} "
            f"| {span([e['bytes_bf16'] / e['images'] / 2**20 for e in enc])} "
            f"| {span([e['rows'] for e in enc])} "
            f"| {span([e['bytes_bf16'] / 2**20 for e in enc])} "
            f"| {span([(e['end'] - e['start']) * 1e3 / e['images'] for e in enc])} "
            f"| {st.median(ttft):.0f} | {100 * enc_ms / pre_ms:.1f}% "
            f"| {100 * st.median(share):.1f}% |"
        )


print(
    "| run | requests | images | rows/image | MB/image | rows/request "
    "| MB/request | encode ms/image | median TTFT ms | encode/prefill time "
    "| median encode share of TTFT |"
)
print("|" + "---|" * 11)
for tag in sys.argv[2:]:
    describe(tag)
