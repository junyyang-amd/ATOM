"""Twenty refits under rollout load on two ATOM workers through ATOMesh.

Two ``--enable-rl`` ATOM workers on OLD behind ATOMesh. Each round runs
pause -> ``update_weights_from_disk`` -> continue as fan-outs, alternating NEW
and OLD with a new ``weight_version``, then ROLLOUTS concurrent ``/generate``
requests through ATOMesh, then reads ``GET /v1/rl/workers``. PASS when no
request failed and every listing holds the ids and urls listed before the
first round. The JSON holds each round's failures, refit time and the
``weight_version`` its rollouts reported.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from accept import PROMPTS, HarnessError, Verdict
from checks.refit_disk import NEW, OLD, SELECTOR, launch, refit

ISSUE = 109
ROUNDS = 20
ROLLOUTS = 64


def checkpoint(r: int) -> str:
    """The repo round ``r`` (from 1) loads: NEW, OLD, NEW, ..."""
    return NEW if r % 2 else OLD


def version(r: int) -> str:
    return f"stability-{r}"


def rollout_prompts() -> list[str]:
    return [PROMPTS[i % len(PROMPTS)] for i in range(ROLLOUTS)]


def plan(h) -> list[str]:
    launch(h)
    rounds = [
        f"round {r}: POST /v1/rl/engine/{{pause_generation, update_weights_from_disk"
        f" {checkpoint(r)} as {version(r)}, continue_generation}}?selector={SELECTOR}"
        f"; POST <mesh>/generate x{ROLLOUTS}, concurrent; GET /v1/rl/workers"
        for r in range(1, ROUNDS + 1)
    ]
    return [*h.steps, "GET /v1/rl/workers", *rounds]


def listed(h) -> list[list[str]]:
    """The ``[id, url]`` of each worker ``GET /v1/rl/workers`` lists, sorted."""
    status, body = h.call_mesh("GET", "/v1/rl/workers")
    if status != 200 or not isinstance(body, dict):
        raise HarnessError(f"GET /v1/rl/workers answered {status}: {body}")
    return sorted([w.get("id"), w.get("url")] for w in body.get("workers", []))


def rollouts(h) -> tuple[list[str], list[str]]:
    """ROLLOUTS concurrent ``/generate`` through ATOMesh: each answer's
    ``weight_version`` and each failure's message."""

    def one(prompt):
        try:
            return h.generate(h.mesh_url, [prompt])[0]["meta_info"]["weight_version"]
        except Exception as e:  # noqa: BLE001 - every failure is counted
            return e

    with ThreadPoolExecutor(ROLLOUTS) as pool:
        answers = list(pool.map(one, rollout_prompts()))
    errors = [f"{type(a).__name__}: {a}" for a in answers if isinstance(a, Exception)]
    versions = [a for a in answers if not isinstance(a, Exception)]
    return versions, errors


def run(h) -> Verdict:
    _, new = launch(h)
    paths = {NEW: new, OLD: h.model_dir(OLD)}
    before = listed(h)
    rounds = []
    for r in range(1, ROUNDS + 1):
        refit_s = refit(h, SELECTOR, paths[checkpoint(r)], version(r))
        versions, errors = rollouts(h)
        rounds.append(
            {
                "round": r,
                "checkpoint": checkpoint(r),
                "refit_s": round(refit_s, 2),
                "failed": len(errors),
                "first_error": errors[0] if errors else None,
                "weight_versions": sorted(set(versions)),
                "workers": listed(h),
            }
        )
    return judge(before, rounds)


def judge(before: list, rounds: list[dict]) -> Verdict:
    """The verdict on the listing ``before`` the first round and on each
    round's record."""
    failed = []
    for rd in rounds:
        if rd["failed"]:
            failed.append(
                f"round {rd['round']}: {rd['failed']} of {ROLLOUTS} requests failed,"
                f" first {rd['first_error']}"
            )
        if rd["workers"] != before:
            failed.append(f"round {rd['round']}: workers {rd['workers']}")
    data = {"workers": before, "failed_per_round": [rd["failed"] for rd in rounds]}
    data["rounds"] = rounds
    return Verdict(not failed, "; ".join(failed), data)
