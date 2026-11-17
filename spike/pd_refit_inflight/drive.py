"""One refit run against a live 1P1D pair, its fresh-pair reference, and the table.

drive.py run --order p|d|both --new <ckpt> --out run.json
    64 streamed greedy completions (max_tokens 256, ignore_eos) through the
    router; once --pause-after have their first token, pause P and D, update both to
    <ckpt>, resume, in that order ("both": each step on the two at once).
drive.py reference run*.json --out ref.json
    against a fresh pair: per completed request, the greedy continuation of
    its prompt plus the text it had before the resume (none for one done
    before the pause). Taken on a new-checkpoint pair and an old one.
drive.py table new_ref.json old_ref.json run*.json
"""

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROUTER = "http://127.0.0.1:18100"
SIDES = {"p": "http://127.0.0.1:18110", "d": "http://127.0.0.1:18120"}
N, MAX_TOKENS, HANG_S, AGREE = 64, 256, 180, 64


def prompts() -> list[str]:
    """64 fixed prompts of 200 to 3000 words, so prefills overlap decodes."""
    rng = random.Random(92)
    words = ["the", "of", "and", "river", "stone", "light", "market", "winter"]
    words += ["engine", "garden", "letter"]
    return [
        f"Story {i}: " + " ".join(rng.choices(words, k=rng.randint(200, 3000)))
        for i in range(N)
    ]


def model_id() -> str:
    return requests.get(f"{ROUTER}/v1/models", timeout=10).json()["data"][0]["id"]


class Stream(threading.Thread):
    def __init__(self, idx, prompt, model, resumed: threading.Event):
        super().__init__(daemon=True)
        self.idx, self.prompt, self.model, self.resumed = idx, prompt, model, resumed
        self.pre, self.post, self.first = "", "", None
        self.finish, self.error, self.done = None, None, False

    def run(self):
        body = {
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
        }
        try:
            with requests.post(
                f"{ROUTER}/v1/completions", json=body, stream=True, timeout=600
            ) as r:
                if r.status_code != 200:
                    self.error = f"HTTP {r.status_code}: {r.text[:300]}"
                    return
                for line in r.iter_lines():
                    if not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    if "error" in chunk:
                        self.error = str(chunk["error"])[:300]
                        return
                    for c in chunk.get("choices", []):
                        if c.get("text"):
                            if self.first is None:
                                self.first = time.monotonic()
                            if self.resumed.is_set():
                                self.post += c["text"]
                            else:
                                self.pre += c["text"]
                        self.finish = c.get("finish_reason") or self.finish
            self.done = self.error is None
        except Exception as exc:  # noqa: BLE001 - a dropped stream is a failure
            self.error = f"{type(exc).__name__}: {exc}"[:300]


def step(name: str, order: str, **body) -> dict:
    """POST /spike/<name> on P and D in *order*; each side's answer."""

    def call(side):
        try:
            return requests.post(
                f"{SIDES[side]}/spike/{name}", json=body or None, timeout=900
            ).json()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if order == "both":
        with ThreadPoolExecutor(2) as pool:
            return dict(zip("pd", pool.map(call, "pd")))
    first = order
    second = "d" if order == "p" else "p"
    return {first: call(first), second: call(second)}


def state_of(s: Stream) -> str:
    if s.done:
        return "done"
    if s.error:
        return "failed"
    return "decoding" if s.first is not None else "no_token"


def cmd_run(args):
    model = model_id()
    resumed = threading.Event()
    streams = [Stream(i, p, model, resumed) for i, p in enumerate(prompts())]
    t0 = time.monotonic()
    for s in streams:
        s.start()
    while sum(s.first is not None for s in streams) < args.pause_after:
        if time.monotonic() - t0 > 300:
            sys.exit("no first tokens within 300 s")
        time.sleep(0.005)
    pause = step("pause", args.order)
    at_pause = [state_of(s) for s in streams]
    time.sleep(0.5)  # two metric pushes
    engines = {
        side: requests.get(f"{u}/spike/state").json() for side, u in SIDES.items()
    }
    update = step("update", args.order, path=args.new)
    time.sleep(1.0)  # chunks already sent before the pause drain to the client
    resumed.set()
    resume = step("resume", args.order)
    deadline = time.monotonic() + HANG_S
    for s in streams:
        s.join(max(0.0, deadline - time.monotonic()))
    reqs = [
        {
            "idx": s.idx,
            "at_pause": st,
            "outcome": "completed" if s.done else "failed" if s.error else "hung",
            "error": s.error,
            "finish": s.finish,
            "pre": s.pre,
            "post": s.post,
        }
        for s, st in zip(streams, at_pause)
    ]
    out = {"order": args.order, "pause": pause, "engines": engines}
    out |= {"update": update, "resume": resume, "requests": reqs}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(summary_line(out, None))


def cmd_reference(args):
    model, ps = model_id(), prompts()
    jobs = {}
    for path in args.runs:
        for r in json.loads(Path(path).read_text())["requests"]:
            if r["outcome"] == "completed":
                jobs[(r["idx"], "" if r["at_pause"] == "done" else r["pre"])] = None

    def ask(key):
        idx, pre = key
        body = {
            "model": model,
            "prompt": ps[idx] + pre,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "ignore_eos": True,
        }
        r = requests.post(f"{ROUTER}/v1/completions", json=body, timeout=600)
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    with ThreadPoolExecutor(32) as pool:
        texts = list(pool.map(ask, list(jobs)))
    ref = [{"idx": k[0], "pre": k[1], "text": t} for k, t in zip(jobs, texts)]
    Path(args.out).write_text(json.dumps(ref, indent=1))
    print(f"{len(ref)} reference continuations")


def agrees(text: str, ref: str) -> bool:
    """Whether *text* starts as *ref* does. Only the first AGREE characters
    count: two greedy runs on the same weights can part later on batch
    numerics, while Qwen3-1.7B and its Base part within a few tokens."""
    n = min(AGREE, len(text))
    return text[:n] == ref[:n]


def weights_seen(r, refs) -> str | None:
    """Which fresh pair's greedy continuation the text after the resume is:
    "new", "old", "both" or "neither"; None for a request with none."""
    if r["outcome"] != "completed" or refs is None:
        return None
    if r["at_pause"] == "done":  # all of it before the pause
        key, text = (r["idx"], ""), r["pre"]
    else:
        key, text = (r["idx"], r["pre"]), r["post"]
    if not text:
        return None
    hits = [side for side, ref in refs.items() if agrees(text, ref[key])]
    return {0: "neither", 2: "both"}.get(len(hits)) or hits[0]


def summary_line(run, refs) -> str:
    upd = " ".join(
        f"{side.upper()}={'applied' if a['ok'] else 'refused'}"
        f"({a['s'] if a.get('s') is not None else '-'}s)"
        for side, a in run["update"].items()
    )
    counts = {}
    for r in run["requests"]:
        key = (r["at_pause"], r["outcome"])
        counts[key] = counts.get(key, 0) + 1
        seen = weights_seen(r, refs)
        if seen is not None:
            counts[(r["at_pause"], f"{seen}_weights")] = (
                counts.get((r["at_pause"], f"{seen}_weights"), 0) + 1
            )
    body = " ".join(f"{a}/{b}={n}" for (a, b), n in sorted(counts.items()))
    return f"order={run['order']} update: {upd} | {body}"


def cmd_table(args):
    refs = None
    if args.new_ref != "-":
        refs = {
            side: {
                (r["idx"], r["pre"]): r["text"]
                for r in json.loads(Path(path).read_text())
            }
            for side, path in (("new", args.new_ref), ("old", args.old_ref))
        }
    for path in args.runs:
        run = json.loads(Path(path).read_text())
        print(path.rsplit("/", 1)[-1], summary_line(run, refs))
        for side, a in run["update"].items():
            if not a["ok"]:
                print(f"  {side.upper()} update: {a['error']}")
        for side in ("pause", "resume"):
            bad = {k: v["error"] for k, v in run[side].items() if not v["ok"]}
            if bad:
                print(f"  {side} errors: {bad}")
        errs = {r["error"] for r in run["requests"] if r["error"]}
        for e in sorted(errs)[:3]:
            print(f"  request error: {e}")
        print(f"  engines at pause: {json.dumps(run['engines'])}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--order", choices=("p", "d", "both"), required=True)
    r.add_argument("--new", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--pause-after", type=int, default=8)
    f = sub.add_parser("reference")
    f.add_argument("runs", nargs="+")
    f.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("new_ref", help="the fresh new-weights pair's json, or -")
    t.add_argument("old_ref", help="the fresh old-weights pair's json, or -")
    t.add_argument("runs", nargs="+")
    args = ap.parse_args()
    {"run": cmd_run, "reference": cmd_reference, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
