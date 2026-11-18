"""SMG's own refit client drives ATOMesh over ATOM workers.

Two ``--enable-rl`` ATOM workers on OLD behind ATOMesh; then SMG's
``examples/rl/refit_from_disk.py``, unmodified from the SMG checkout
``--smg`` names, with ``--selector engine=atom``: pause ->
``update_weights_from_disk`` (NEW) -> continue, then one ``/generate``
through ATOMesh that must report the new ``weight_version``. PASS when the
script exits 0. A checkout at another commit, or with local changes, is
refused.
"""

from __future__ import annotations

import shlex

from accept import HarnessError, Verdict
from checks.refit_disk import launch

ISSUE = 108
SMG_COMMIT = "6c7bda118c5c6c02e65fe104cc8d93a0f906c965"
VERSION = "smg-client-v2"


def script(h, new: str) -> list[str]:
    """The command running SMG's client against ATOMesh."""
    return [
        "env",
        f"PYTHONPATH={h.smg}/bindings/python/src",
        "python3",
        f"{h.smg}/examples/rl/refit_from_disk.py",
        "--smg",
        h.mesh_url,
        "--model-path",
        new,
        "--weight-version",
        VERSION,
        "--selector",
        "engine=atom",
    ]


def check_commit(h) -> None:
    """Refuse an SMG checkout that is not the pinned commit, unmodified."""
    head = h.sh(["git", "-C", str(h.smg), "rev-parse", "HEAD"]).strip()
    changed = h.sh(["git", "-C", str(h.smg), "status", "--porcelain", "-uno"]).strip()
    if head != SMG_COMMIT or changed:
        raise HarnessError(
            f"{h.smg} is at {head}{' with local changes' if changed else ''}; "
            f"the check runs SMG {SMG_COMMIT} unmodified"
        )


def plan(h) -> list[str]:
    _, new = launch(h)
    pinned = f"{h.smg} must be at {SMG_COMMIT}, unmodified"
    return [pinned, *h.steps, shlex.join(script(h, new))]


def run(h) -> Verdict:
    check_commit(h)
    _, new = launch(h)
    return Verdict(True, data={"script": h.sh(script(h, new)).splitlines()})
