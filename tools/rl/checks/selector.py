"""A selector refits only the worker it names.

Two ``--enable-rl`` ATOM workers on OLD behind ATOMesh. An
``update_weights_from_disk`` fan-out whose selector matches no worker, then
pause -> ``update_weights_from_disk`` (NEW) -> continue with
``selector=url=<worker 1>``; then the fixed prompts on each worker's
``/generate``, compared teacher-forced with a fresh worker on the checkpoint
it should hold (``Harness.equals_fresh``). PASS when the miss is refused with
400 ``no_workers_match``, worker 1 reports the new version and equals fresh on
NEW, and worker 2 reports one version, not the new one, and equals fresh on
OLD.
"""

from __future__ import annotations

from accept import PROMPTS, Verdict
from checks.refit_disk import FRESH_COMPARE, OLD, cycle_calls, launch, refit

ISSUE = 108
VERSION = "selector-v2"
MISS = "url=http://no-such-worker"


def plan(h) -> list[str]:
    workers, new = launch(h)
    calls = [
        f"POST /v1/rl/engine/{p}?selector=url={workers[0]} {b}"
        for p, b in cycle_calls(new, VERSION)
    ]
    generate = [
        f"POST {url}/generate x{len(PROMPTS)}, one at a time" for url in workers
    ]
    h.start_atom_workers(1, new)
    h.start_atom_workers(1, h.model_dir(OLD))
    return [
        *h.steps[:-2],
        f"POST /v1/rl/engine/update_weights_from_disk?selector={MISS} -> 400 no_workers_match",
        *calls,
        *generate,
        *h.steps[-2:],
        FRESH_COMPARE,
    ]


def run(h) -> Verdict:
    workers, new = launch(h)
    old = h.model_dir(OLD)
    update = {"model_path": new, "weight_version": VERSION}
    miss = h.rl_fanout("update_weights_from_disk", update, MISS)
    refit_s = refit(h, f"url={workers[0]}", new, VERSION)
    outputs = {url: h.generate(url, PROMPTS) for url in workers}
    fresh_new = h.start_atom_workers(1, new)[0]
    fresh_old = h.start_atom_workers(1, old)[0]
    compared = {
        workers[0]: h.equals_fresh(fresh_new, new, PROMPTS, outputs[workers[0]]),
        workers[1]: h.equals_fresh(fresh_old, old, PROMPTS, outputs[workers[1]]),
    }
    return judge(miss, outputs, compared, refit_s)


def judge(miss, outputs: dict, compared: dict, refit_s: float) -> Verdict:
    """The verdict on the selector that matched nothing (``miss``, its HTTP
    answer), on each worker's ``/generate`` answers (worker 1 first) and on
    their ``equals_fresh`` comparison."""
    status, body = miss
    error = body.get("error") if isinstance(body, dict) else None
    failed = []
    if (status, error) != (400, "no_workers_match"):
        failed.append(f"selector {MISS} answered {status} {error}")
    workers = {}
    for n, (url, outs) in enumerate(outputs.items()):
        versions = sorted({o["meta_info"]["weight_version"] for o in outs})
        workers[url] = {"weight_versions": versions, "fresh": compared[url]}
        refitted = versions == [VERSION]
        if (n == 0) != refitted or len(versions) != 1:
            failed.append(f"{url} weight_version {versions}")
        if not compared[url]["equal"]:
            failed.append(f"{url} differs from fresh at {compared[url]['first']}")
    data = {"miss": {"status": status, "body": body}, "refit_s": round(refit_s, 2)}
    return Verdict(not failed, "; ".join(failed), data | {"workers": workers})
