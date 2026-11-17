"""Refit from disk on two ATOM workers through ATOMesh.

Two ``--enable-rl`` ATOM workers on OLD behind ATOMesh; pause ->
``update_weights_from_disk`` (NEW, a new ``weight_version``) -> continue as
fan-outs with ``selector=engine=atom``; then the fixed prompts on each worker's
``/generate``, and the same prompts on a fresh worker started on NEW. PASS when
every output reports the new version and has the fresh worker's token ids. The
JSON holds the refit's wall time, pause sent to continue answered.
"""

from __future__ import annotations

import time

from accept import PROMPTS, HarnessError, Verdict

ISSUE = 107
OLD, NEW = "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B"
SELECTOR = "engine=atom"
VERSION = "refit-disk-v2"
CYCLE = ("pause_generation", "update_weights_from_disk", "continue_generation")


def launch(h) -> tuple[list[str], str]:
    """Two ATOM workers on OLD behind ATOMesh: their URLs and NEW's path."""
    workers = h.start_atom_workers(2, h.model_dir(OLD))
    h.start_mesh(workers, backend="atom")
    return workers, h.model_dir(NEW)


def cycle_calls(checkpoint: str, version: str) -> list[tuple]:
    """The fan-outs of one refit: ``(path, body)`` in order."""
    update = {"model_path": checkpoint, "weight_version": version}
    return [(p, update if p == "update_weights_from_disk" else {}) for p in CYCLE]


def refit(h, selector: str, checkpoint: str, version: str) -> float:
    """Pause, load ``checkpoint`` as ``version`` and continue the workers
    ``selector`` names, each a fan-out through ATOMesh; the seconds from
    sending the pause to the continue's answer. A call that any worker
    refuses raises."""
    started = time.monotonic()
    for path, body in cycle_calls(checkpoint, version):
        status, answer = h.rl_fanout(path, body, selector)
        if not _fanout_ok(status, answer):
            raise HarnessError(f"{path} fan-out answered {status}: {answer}")
    return time.monotonic() - started


def _fanout_ok(status, answer) -> bool:
    """200, at least one worker called, none failed, each engine succeeded."""
    if status != 200 or not isinstance(answer, dict):
        return False
    results = list(answer.get("results", {}).values())
    return (
        answer.get("total") == answer.get("succeeded") == len(results) > 0
        and answer.get("failed") == []
        and all(
            r.get("status") == 200 and (r.get("body") or {}).get("success")
            for r in results
        )
    )


def first_mismatch(prompts, outputs, fresh) -> str | None:
    """The first prompt whose output token ids differ from the fresh ones."""
    for i, (out, want) in enumerate(zip(outputs, fresh, strict=True)):
        if out["output_ids"] != want["output_ids"]:
            return f"prompt {i} {prompts[i]!r} differs from fresh"
    return None


def plan(h) -> list[str]:
    workers, new = launch(h)
    calls = [
        f"POST /v1/rl/engine/{p}?selector={SELECTOR} {b}"
        for p, b in cycle_calls(new, VERSION)
    ]
    generate = [
        f"POST {url}/generate x{len(PROMPTS)}, one at a time" for url in workers
    ]
    h.start_atom_workers(1, new)
    return [*h.steps[:-1], *calls, *generate, h.steps[-1], "POST <fresh>/generate"]


def run(h) -> Verdict:
    workers, new = launch(h)
    refit_s = refit(h, SELECTOR, new, VERSION)
    outputs = {url: h.generate(url, PROMPTS) for url in workers}
    fresh = h.generate(h.start_atom_workers(1, new)[0], PROMPTS)
    return judge(outputs, fresh, refit_s)


def judge(outputs: dict, fresh: list[dict], refit_s: float) -> Verdict:
    """The verdict on each worker's ``/generate`` answers against ``fresh``."""
    workers, failed = {}, []
    for url, outs in outputs.items():
        versions = sorted({o["meta_info"]["weight_version"] for o in outs})
        mismatch = first_mismatch(PROMPTS, outs, fresh)
        workers[url] = {"weight_versions": versions, "mismatch": mismatch}
        if versions != [VERSION]:
            failed.append(f"{url} weight_version {versions}")
        if mismatch:
            failed.append(f"{url} {mismatch}")
    data = {"refit_s": round(refit_s, 2), "workers": workers}
    return Verdict(not failed, "; ".join(failed), data)
