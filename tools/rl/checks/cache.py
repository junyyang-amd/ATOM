"""No prefix cache from the old weights survives a refit from disk.

On each of two ATOM workers behind ATOMesh, ``/generate`` of PROBE (a long
shared prefix and a tail) twice, then a refit to NEW, then PROBE once more;
PROBE also on a fresh worker started on NEW. PASS when, on each worker, the
second send before the refit reports cached prompt tokens (the probe can see
a hit), the send after it reports none, and its token ids equal the fresh
worker's.
"""

from __future__ import annotations

from accept import Verdict
from checks.refit_disk import SELECTOR, cycle_calls, first_mismatch, launch, refit

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
        "POST <fresh>/generate PROBE",
    ]


def run(h) -> Verdict:
    workers, new = launch(h)
    before = {url: h.generate(url, [PROBE, PROBE]) for url in workers}
    refit(h, SELECTOR, new, VERSION)
    after = {url: h.generate(url, [PROBE])[0] for url in workers}
    fresh = h.generate(h.start_atom_workers(1, new)[0], [PROBE])[0]
    return judge(before, after, fresh)


def judge(before: dict, after: dict, fresh: dict) -> Verdict:
    """The verdict on each worker's PROBE answers: ``before`` the two sent
    before the refit, ``after`` the one sent after it."""
    workers, failed = {}, []
    for url, (_, again) in before.items():
        hit = again["meta_info"]["cached_tokens"]
        cached = after[url]["meta_info"]["cached_tokens"]
        mismatch = first_mismatch([PROBE], [after[url]], [fresh])
        workers[url] = {"cached_before": hit, "cached_after": cached}
        workers[url]["mismatch"] = mismatch
        if not hit > 0:
            failed.append(f"{url} cached_tokens {hit} on the repeat before the refit")
        if cached != 0:
            failed.append(f"{url} cached_tokens {cached} after the refit")
        if mismatch:
            failed.append(f"{url} {mismatch}")
    return Verdict(not failed, "; ".join(failed), {"workers": workers})
