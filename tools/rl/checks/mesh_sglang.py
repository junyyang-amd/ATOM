"""ATOMesh's /v1/rl routes over two SGLang workers.

Lists the workers, reads each one and its ``model_info`` through the
per-worker proxy, runs pause -> ``update_weights_from_disk`` -> continue as
fan-outs, reads ``model_info`` again, and probes SMG's error answers. PASS
when every route answered as SMG's protocol says and both workers report the
new ``weight_version``.
"""

from __future__ import annotations

import urllib.parse

from accept import Verdict

ISSUE = 106
OLD, NEW = "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B"
SELECTOR = "engine=sglang"
VERSION = "mesh-sglang-v2"
# error code: (method, path, status); `{id}` is the first listed worker's id.
PROBES = {
    "selector_required": ("GET", "/v1/rl/engine/model_info", 400),
    "invalid_selector": ("GET", "/v1/rl/engine/model_info?selector=%3Dsglang", 400),
    "no_workers_match": ("GET", "/v1/rl/engine/model_info?selector=engine%3Dvllm", 400),
    "worker_not_found": ("GET", "/v1/rl/workers/no-such-worker", 404),
    "invalid_engine_path": ("GET", "/v1/rl/workers/{id}/engine/model%20info", 400),
}
CYCLE = ("pause_generation", "update_weights_from_disk", "continue_generation")


def _launch(h) -> str:
    """Two SGLang workers on OLD behind ATOMesh; the NEW checkpoint's path."""
    h.start_mesh(h.start_sglang_workers(2, h.model_dir(OLD)), backend="sglang")
    return h.model_dir(NEW)


def _update_body(new: str) -> dict:
    return {"model_path": new, "weight_version": VERSION, "flush_cache": True}


def plan(h) -> list[str]:
    new = _launch(h)
    bodies = {"update_weights_from_disk": _update_body(new)}
    cycle = [
        f"POST /v1/rl/engine/{p}?selector={SELECTOR} {bodies.get(p, {})}" for p in CYCLE
    ]
    probes = [f"{m} {p} -> {s} {code}" for code, (m, p, s) in PROBES.items()]
    return [
        *h.steps,
        "GET /v1/rl/workers; GET /v1/rl/workers/{id} for each",
        "GET /v1/rl/workers/{id}/engine/model_info for each",
        *cycle,
        "GET /v1/rl/workers/{id}/engine/model_info for each",
        *probes,
    ]


def run(h) -> Verdict:
    new = _launch(h)
    obs = {"workers": h.call_mesh("GET", "/v1/rl/workers")}
    listed = obs["workers"][1]
    ids = (
        [w["id"] for w in listed.get("workers", [])] if isinstance(listed, dict) else []
    )
    quoted = {i: urllib.parse.quote(i, safe="") for i in ids}
    obs["worker"] = {i: h.call_mesh("GET", f"/v1/rl/workers/{quoted[i]}") for i in ids}
    obs["model_info_before"] = {i: h.rl_worker_call(i, "model_info") for i in ids}
    bodies = {"update_weights_from_disk": _update_body(new)}
    for path in CYCLE:
        obs[path] = h.rl_fanout(path, bodies.get(path, {}), SELECTOR)
    obs["model_info_after"] = {i: h.rl_worker_call(i, "model_info") for i in ids}
    first = quoted[ids[0]] if ids else "none"
    for code, (method, path, _) in PROBES.items():
        obs[code] = h.call_mesh(method, path.format(id=first))
    return judge(obs)


def _version(outcome) -> str | None:
    """``weight_version`` of a proxied ``model_info`` (an RlCallOutcome)."""
    status, body = outcome
    inner = body.get("body") if isinstance(body, dict) else None
    return (
        inner.get("weight_version")
        if status == 200 and isinstance(inner, dict)
        else None
    )


def _fanout_ok(answer) -> bool:
    """200, both workers called, none failed, each engine answered 200."""
    status, body = answer
    if status != 200 or not isinstance(body, dict):
        return False
    results = body.get("results", {}).values()
    return (
        body.get("total") == body.get("succeeded") == 2
        and body.get("failed") == []
        and all(r.get("status") == 200 for r in results)
    )


def judge(obs: dict) -> Verdict:
    """The verdict on ``obs``: each value an HTTP (status, body) answer, or a
    ``{worker id: answer}`` map for the per-worker reads."""
    routes: dict[str, dict] = {}

    def record(route: str, ok: bool, answer) -> None:
        routes[route] = {"ok": ok, "status": answer[0], "body": answer[1]}

    status, body = obs["workers"]
    workers = body.get("workers", []) if isinstance(body, dict) else []
    record(
        "GET /v1/rl/workers",
        status == 200
        and body.get("protocol_version") == 1
        and body.get("total") == len(workers) == 2
        and all(w.get("engine") == "sglang" for w in workers),
        obs["workers"],
    )
    for w in workers:
        answer = obs["worker"].get(w["id"], (None, None))
        record(f"GET /v1/rl/workers/{w['id']}", answer == (200, w), answer)
    for wid, answer in obs["model_info_before"].items():
        got = _version(answer)
        record(f"model_info {wid} before", got not in (None, VERSION), answer)
    for wid, answer in obs["model_info_after"].items():
        record(f"model_info {wid} after", _version(answer) == VERSION, answer)
    for path in CYCLE:
        record(f"POST /v1/rl/engine/{path}", _fanout_ok(obs[path]), obs[path])
    for code, (_, _, want) in PROBES.items():
        status, body = obs[code]
        error = body.get("error") if isinstance(body, dict) else None
        record(f"probe {code}", status == want and error == code, obs[code])
    failed = [route for route, r in routes.items() if not r["ok"]]
    reason = "; ".join(failed)
    return Verdict(not failed, reason, {"routes": routes})
