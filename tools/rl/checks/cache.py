"""No prefix cache from the old weights survives a refit from disk.

On each of two ATOM workers behind ATOMesh, ``/generate`` of PROBE (a long
shared prefix and a tail) twice, then a refit to NEW, then PROBE once more,
compared teacher-forced with a fresh worker started on NEW. PASS when, on each
worker, the second send before the refit reports cached prompt tokens (the
probe can see a hit), the send after it reports none, and its output equals
fresh.
"""

from __future__ import annotations

from accept import Verdict
from checks.refit_disk import FRESH_COMPARE, SELECTOR, cycle_calls, launch, refit

ISSUE = 107
VERSION = "cache-v2"
# Many KV blocks long, so a cache hit covers whole blocks.
PREFIX = " ".join(
    f"Station {i} on the coastal line opens at {6 + i % 4} in the morning, "
    f"serves {i * 37 % 500} passengers a day and has {i % 3 + 1} platforms."
    for i in range(12)
)
PROBE = PREFIX + " Which station serves the most passengers?"


def plan(h) -> list[str]:
    workers, new = launch(h)
    calls = [
        f"POST /v1/rl/engine/{p}?selector={SELECTOR} {b}"
        for p, b in cycle_calls(new, VERSION)
    ]
    h.start_atom_workers(1, new)
    return [
        *h.steps[:-1],
        *[f"POST {url}/generate PROBE x2" for url in workers],
        *calls,
        *[f"POST {url}/generate PROBE" for url in workers],
        h.steps[-1],
        FRESH_COMPARE,
    ]


def run(h) -> Verdict:
    workers, new = launch(h)
    before = {url: h.generate(url, [PROBE, PROBE]) for url in workers}
    refit(h, SELECTOR, new, VERSION)
    after = {url: h.generate(url, [PROBE])[0] for url in workers}
    fresh = h.start_atom_workers(1, new)[0]
    compared = {
        url: h.equals_fresh(fresh, new, [PROBE], [out]) for url, out in after.items()
    }
    return judge(before, after, compared)


def judge(before: dict, after: dict, compared: dict) -> Verdict:
    """The verdict on each worker's PROBE answers: ``before`` the two sent
    before the refit, ``after`` the one sent after it, ``compared`` its
    ``equals_fresh`` comparison."""
    workers, failed = {}, []
    for url, (_, again) in before.items():
        hit = again["meta_info"]["cached_tokens"]
        cached = after[url]["meta_info"]["cached_tokens"]
        workers[url] = {"cached_before": hit, "cached_after": cached}
        workers[url]["fresh"] = compared[url]
        if not hit > 0:
            failed.append(f"{url} cached_tokens {hit} on the repeat before the refit")
        if cached != 0:
            failed.append(f"{url} cached_tokens {cached} after the refit")
        if not compared[url]["equal"]:
            failed.append(f"{url} differs from fresh at {compared[url]['first']}")
    return Verdict(not failed, "; ".join(failed), {"workers": workers})
