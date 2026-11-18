"""Refit from disk on a 1P1D ATOM pair with completions in flight.

P and D ``--enable-rl`` ATOM workers on OLD, KV over mooncake ``tcp``,
behind ATOMesh's PD router. IN_FLIGHT streamed completions through ATOMesh;
once PAUSE_AFTER of them have a token, pause -> ``update_weights_from_disk``
(NEW, a new ``weight_version``) -> continue as fan-outs with
``selector=engine=atom``. Then P's and D's ``/model_info``, and the fixed
prompts one at a time through ATOMesh, compared teacher-forced with a fresh
1P1D on NEW (``Harness.equals_fresh``). PASS when some completions were in
flight at the pause, every one completes, P and D report the new version and
the outputs equal fresh. The JSON holds the completions' states at the pause
and the count of D's ``recompute stale remote KV:`` log lines.
"""

from __future__ import annotations

import json
import random
import threading
import time
import urllib.request

from accept import PROMPTS, SAMPLING, HarnessError, Verdict, http, token_bytes
from checks.refit_disk import FRESH_COMPARE, NEW, OLD, SELECTOR, cycle_calls, refit

ISSUE = 111
VERSION = "pd-v2"
IN_FLIGHT, PAUSE_AFTER, LOAD_TOKENS = 64, 8, 256
FINISH_TIMEOUT_S = 600
STALE_LINE = "recompute stale remote KV:"


def load_prompts() -> list[str]:
    """IN_FLIGHT fixed prompts of 200 to 3000 words, so prefills overlap decodes."""
    rng = random.Random(ISSUE)
    words = "the of and river stone light market winter engine garden letter"
    return [
        f"Story {i}: " + " ".join(rng.choices(words.split(), k=rng.randint(200, 3000)))
        for i in range(IN_FLIGHT)
    ]


def launch(h, checkpoint: str) -> tuple[str, str, str]:
    """P and D on ``checkpoint`` behind ATOMesh: P's, D's and ATOMesh's URLs."""
    p, d = h.start_pd_pair(checkpoint)
    return p, d, h.start_mesh([p, d], pd=True, model_path=checkpoint)


def stream(url: str, body: dict):
    """The chunks of one streamed completion."""
    data = json.dumps(body | {"stream": True}).encode()
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url + "/v1/completions", data, headers)
    with urllib.request.urlopen(req, timeout=FINISH_TIMEOUT_S) as resp:
        for line in resp:
            line = line.strip()
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                yield json.loads(line[6:])


class Completion(threading.Thread):
    """One streamed completion; ``state()`` is how far it has got."""

    def __init__(self, url: str, body: dict):
        super().__init__(daemon=True)
        self.url, self.body = url, body
        self.tokens, self.finish, self.error = 0, None, None

    def run(self):
        try:
            for chunk in stream(self.url, self.body):
                if "error" in chunk:
                    raise HarnessError(chunk["error"])
                for choice in chunk.get("choices", []):
                    self.tokens += bool(choice.get("text"))
                    self.finish = choice.get("finish_reason") or self.finish
            if self.finish is None:
                raise HarnessError("stream ended without a finish_reason")
        except Exception as e:  # noqa: BLE001 - any failure is the request's
            self.error = f"{type(e).__name__}: {e}"[:300]

    def state(self) -> str:
        if self.error:
            return "failed"
        if self.finish:
            return "completed"
        return "decoding" if self.tokens else "waiting"


def refit_in_flight(h, mesh: str, model: str, new: str) -> tuple:
    """The in-flight completions' states at the pause and at the end, their
    distinct errors, and the refit's seconds."""
    load = {
        "model": model,
        "max_tokens": LOAD_TOKENS,
        "temperature": 0,
        "ignore_eos": True,
    }
    running = [Completion(mesh, load | {"prompt": p}) for p in load_prompts()]
    for c in running:
        c.start()
    deadline = time.monotonic() + FINISH_TIMEOUT_S
    while sum(c.tokens > 0 for c in running) < PAUSE_AFTER:
        if time.monotonic() > deadline or all(c.error for c in running):
            break
        time.sleep(0.01)
    at_pause = [c.state() for c in running]
    refit_s = refit(h, SELECTOR, new, VERSION)
    for c in running:
        c.join(max(0.0, deadline - time.monotonic()))
    at_end = [c.state() if c.finish or c.error else "hung" for c in running]
    errors = sorted({c.error for c in running if c.error})
    return at_pause, at_end, errors, refit_s


def generate(h, mesh: str, checkpoint: str, model: str) -> list[dict]:
    """PROMPTS one at a time through ATOMesh's chat route on their token ids,
    greedy: ``{"output_ids"}``, read back from the logprobs' bytes."""
    spelling, hidden = token_bytes(checkpoint)
    ids_of = {raw: i for i, raw in spelling.items() if i not in hidden}
    out = []
    for prompt in PROMPTS:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "-"}],
            "prompt_token_ids": h.prompt_ids(mesh, model, prompt),
            "max_tokens": SAMPLING["max_new_tokens"],
            "temperature": SAMPLING["temperature"],
            "logprobs": True,
        }
        status, answer = http("POST", mesh + "/v1/chat/completions", body)
        if status != 200:
            raise HarnessError(f"chat on {mesh} answered {status}: {answer}")
        entries = answer["choices"][0]["logprobs"]["content"]
        out.append({"output_ids": [ids_of[bytes(e["bytes"])] for e in entries]})
    return out


def plan(h) -> list[str]:
    old = h.model_dir(OLD)
    p, d, mesh = launch(h, old)
    calls = [
        f"POST /v1/rl/engine/{path}?selector={SELECTOR} {body}"
        for path, body in cycle_calls(h.model_dir(NEW), VERSION)
    ]
    steps = [
        *h.steps,
        (
            f"POST {mesh}/v1/completions x{IN_FLIGHT} at once, streamed, "
            f"max_tokens {LOAD_TOKENS}; once {PAUSE_AFTER} have a token:"
        ),
        *calls,
        f"GET {p}/model_info, GET {d}/model_info",
        f"POST {mesh}/v1/chat/completions x{len(PROMPTS)}, one at a time",
    ]
    h.steps.clear()
    launch(h, h.model_dir(NEW))
    return [*steps, *h.steps, FRESH_COMPARE]


def run(h) -> Verdict:
    old, new = h.model_dir(OLD), h.model_dir(NEW)
    p, d, mesh = launch(h, old)
    at_pause, at_end, errors, refit_s = refit_in_flight(h, mesh, old, new)
    versions = {url: http("GET", url + "/model_info")[1] for url in (p, d)}
    outputs = generate(h, mesh, new, old)
    stale = h.logs(f"rl-{ISSUE}-atom-{d.rsplit(':', 1)[1]}").count(STALE_LINE)
    fresh = launch(h, new)[2]
    compared = h.equals_fresh(fresh, new, PROMPTS, outputs)
    return judge(at_pause, at_end, errors, versions, compared, refit_s, stale)


def judge(at_pause, at_end, errors, versions, compared, refit_s, stale) -> Verdict:
    """The verdict on the in-flight completions' states at the pause and at
    the end (and their errors), P's and D's ``/model_info``, and the
    ``equals_fresh`` comparison."""
    tally = {
        when: {s: states.count(s) for s in sorted(set(states))}
        for when, states in (("at_pause", at_pause), ("at_end", at_end))
    }
    failed = []
    if not {"waiting", "decoding"} & set(at_pause):
        failed.append("no completion in flight at the pause")
    if tally["at_end"] != {"completed": IN_FLIGHT}:
        failed.append(f"in flight ended {tally['at_end']}, errors {errors[:3]}")
    for url, info in versions.items():
        if not isinstance(info, dict) or info.get("weight_version") != VERSION:
            failed.append(f"{url} model_info {info}")
    if not compared["equal"]:
        failed.append(f"differs from fresh at {compared['first']}")
    data = {
        "in_flight": tally | {"errors": errors},
        "refit_s": round(refit_s, 2),
        "model_info": versions,
    }
    data |= {"fresh": compared, "stale_recomputes": stale}
    return Verdict(not failed, "; ".join(failed), data)
