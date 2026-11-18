"""The refit time of each kind, from the results refit_disk and refit_trainer
wrote: ``refit_s`` of ``refit_disk.json`` and of each row of
``refit_trainer.json``, with the commit each was measured at. PASS when each
holds a time.
"""

from __future__ import annotations

import json

import accept
from accept import Verdict

ISSUE = 110
SOURCES = ("refit_disk", "refit_trainer")


def plan(h) -> list[str]:
    return [f"read {accept.OUT_DIR / f'{check}.json'} refit_s" for check in SOURCES]


def run(h) -> Verdict:
    kinds, missing = {}, []
    for check in SOURCES:
        path = accept.OUT_DIR / f"{check}.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        data = record.get("data", {})
        if "rows" in data:
            refit_s = {row: got.get("refit_s") for row, got in data["rows"].items()}
            times = list(refit_s.values())
        else:
            refit_s = data.get("refit_s")
            times = [refit_s]
        if not times or None in times:
            missing.append(check)
        kinds[check] = {"commit": record.get("commit"), "passed": record.get("passed")}
        kinds[check]["refit_s"] = refit_s
    reason = "no refit time in " + ", ".join(missing) if missing else ""
    return Verdict(not missing, reason, kinds)
