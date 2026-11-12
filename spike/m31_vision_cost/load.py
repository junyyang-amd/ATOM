"""Stream chat requests with K MMStar images each at fixed concurrency; record TTFT."""

import argparse
import base64
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq

MMSTAR = (
    "/share_nfs/junyan_tmp_r120/hf/hub/datasets--Lin-Chen--MMStar/snapshots/"
    "bc98d668301da7b14f648724866e57302778ab27/mmstar.parquet"
)

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://127.0.0.1:8610")
p.add_argument("--images", type=int, default=4)
p.add_argument("--concurrency", type=int, default=32)
p.add_argument("--requests", type=int, default=128)
p.add_argument("--offset", type=int, default=0)
p.add_argument("--max-tokens", type=int, default=64)
p.add_argument("--out", required=True)
args = p.parse_args()

column = pq.read_table(MMSTAR, columns=["image"]).column("image").to_pylist()


def data_url(i):
    raw = column[i % len(column)]
    mime = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def one(n):
    first = args.offset + n * args.images
    content = [
        {"type": "image_url", "image_url": {"url": data_url(first + j)}}
        for j in range(args.images)
    ]
    content.append({"type": "text", "text": "Describe each image in one sentence."})
    body = {
        "model": "minimax-m3.1",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "chat_template_kwargs": {"thinking_mode": "disabled"},
    }
    req = urllib.request.Request(
        args.base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    send = time.time()
    ttft_at = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            if ttft_at is None and line.startswith(b"data: {"):
                delta = json.loads(line[6:])["choices"][0].get("delta", {})
                if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning")):
                    ttft_at = time.time()
    return {"send": send, "first": ttft_at, "ttft_ms": (ttft_at - send) * 1e3}


start = time.time()
with ThreadPoolExecutor(args.concurrency) as pool:
    results = list(pool.map(one, range(args.requests)))
json.dump(
    {"args": vars(args), "start": start, "end": time.time(), "requests": results},
    open(args.out, "w"),
)
