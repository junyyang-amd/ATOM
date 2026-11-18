"""Refit from a trainer over RCCL through ATOMesh, at TP1 and at TP2.

Each row: one ``--enable-rl`` ATOM worker on OLD at the row's TP behind
ATOMesh, and SMG's ``examples/rl/refit_from_trainer.py``, run unmodified from
the checkout ``--smg`` names (at SMG_COMMIT) with ``--selector engine=atom``,
as the trainer on the next GPU. It loads NEW, lays out ranks from ``GET
/v1/rl/workers`` ``tp_size``, pauses, has each worker join the weight-update
group, broadcasts NEW in chunks (the last carries the version and
``flush_cache``), destroys the group, continues and checks each worker's
``model_info`` version. The trainer must exit within TRAINER_DEADLINE_S, so a
refused update fails the check instead of leaving the broadcast waiting. Then
the fixed prompts on the worker's ``/generate``, compared teacher-forced with
a fresh worker on NEW at the same TP (``Harness.equals_fresh``).

PASS when, in every row, the trainer exits 0 and the outputs report the new
version and equal fresh. Each row records its refit time: from the trainer's
line just before its pause (one ``GET /v1/rl/workers`` before the pause is
sent) to its line after the continue answered, by docker's log timestamps.
"""

from __future__ import annotations

import datetime
import time

from accept import NODE_IP, PROMPTS, HarnessError, Verdict

ISSUE = 110
SMG_COMMIT = "6c7bda118c5c6c02e65fe104cc8d93a0f906c965"
SELECTOR = "engine=atom"
VERSION = "refit-trainer-v2"
# name: (tp, OLD, NEW, the trainer's GPU). Workers take GPUs from 0 upward, so
# the trainer's is the one after its row's worker; the fresh worker follows it.
ROWS = {
    "tp1": (1, "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B", 1),
    "tp2": (2, "Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B", 5),
}
# The script's --timeout: each control call, the group rendezvous and the
# group's collective timeout.
TRAINER_TIMEOUT_S = 300
TRAINER_DEADLINE_S = 1800
# What the script prints just before its pause and after the continue answered.
BEFORE_PAUSE = "parameter(s) to broadcast"
AFTER_CONTINUE = "continue_generation ... resumed"
FRESH_COMPARE = "POST <fresh>/v1/chat/completions per output token, teacher-forced"


def smg_problem(h) -> str | None:
    """Why the SMG checkout is not the pinned, unmodified one, or None."""
    head = h.sh(["git", "-C", h.smg, "rev-parse", "HEAD"], check=False).strip()
    if head != SMG_COMMIT:
        return f"{h.smg} is at {head or 'no commit'}, not {SMG_COMMIT}"
    if h.sh(["git", "-C", h.smg, "status", "--porcelain"], check=False).strip():
        return f"{h.smg} has local changes"
    return None


def trainer_argv(smg: str, mesh_url: str, checkpoint: str, port: int) -> list[str]:
    return [
        "env",
        f"PYTHONPATH={smg}/bindings/python/src",
        "PYTHONUNBUFFERED=1",  # the log timestamps time the refit
        "python3",
        f"{smg}/examples/rl/refit_from_trainer.py",
        "--smg",
        mesh_url,
        "--model-path",
        checkpoint,
        "--weight-version",
        VERSION,
        "--selector",
        SELECTOR,
        "--master-address",
        NODE_IP,
        "--master-port",
        str(port),
        "--timeout",
        str(TRAINER_TIMEOUT_S),
    ]


def launch_worker(h, tp: int, old: str) -> str:
    """One ATOM worker on ``old`` at ``tp`` behind ATOMesh: its URL."""
    worker = h.start_atom_workers(1, h.model_dir(old), tp=tp)[0]
    h.start_mesh([worker], backend="atom")
    return worker


def start_trainer(h, tp: int, new: str, gpu: int) -> str:
    argv = trainer_argv(h.smg, h.mesh_url, h.model_dir(new), h.take_port())
    # The row's worker holds the tp GPUs just before the trainer's.
    workers = list(range(gpu - tp, gpu))
    return h.start_trainer([gpu], argv, role=f"trainer-tp{tp}", peers=workers)


def plan(h) -> list[str]:
    steps = [f"git -C {h.smg} rev-parse HEAD: {SMG_COMMIT}, no local changes"]
    for tp, old, new, gpu in ROWS.values():
        h.steps.clear()
        worker = launch_worker(h, tp, old)
        steps += [*h.steps, f"GET /v1/rl/workers: tp_size {tp}"]
        h.steps.clear()
        name = start_trainer(h, tp, new, gpu)
        steps += [
            *h.steps,
            f"wait for {name} to exit 0, at most {TRAINER_DEADLINE_S} s",
        ]
        steps.append(f"POST {worker}/generate x{len(PROMPTS)}, one at a time")
        h.steps.clear()
        h.start_atom_workers(1, h.model_dir(new), tp=tp)
        steps += [*h.steps, FRESH_COMPARE]
    return steps


def run(h) -> Verdict:
    problem = smg_problem(h)
    if problem:
        return Verdict(False, problem)
    rows = {}
    for row, (tp, old, new, gpu) in ROWS.items():
        worker = launch_worker(h, tp, old)
        _, listed = h.call_mesh("GET", "/v1/rl/workers")
        tp_sizes = [w.get("tp_size") for w in listed["workers"]]
        if tp_sizes != [tp]:
            raise HarnessError(f"{row}: /v1/rl/workers tp_size {tp_sizes}")
        name = start_trainer(h, tp, new, gpu)
        code = wait_exit(h, name)
        if code:
            raise HarnessError(f"{row}: {name} exited {code}: {h.logs(name)[-1500:]}")
        log = h.sh(["docker", "logs", "--timestamps", name])
        outputs = h.generate(worker, PROMPTS)
        fresh = h.start_atom_workers(1, h.model_dir(new), tp=tp)[0]
        rows[row] = {
            "tp_size": tp,
            "refit_s": refit_seconds(log),
            "weight_versions": sorted(
                {o["meta_info"]["weight_version"] for o in outputs}
            ),
            "fresh": h.equals_fresh(fresh, h.model_dir(new), PROMPTS, outputs),
        }
    return judge(rows)


def wait_exit(h, name: str) -> int:
    """The container's exit code; HarnessError when it still runs after
    TRAINER_DEADLINE_S."""
    deadline = time.monotonic() + TRAINER_DEADLINE_S
    while True:
        state = ["docker", "inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}"]
        running, code = h.sh([*state, name]).split()
        if running != "true":
            return int(code)
        if time.monotonic() > deadline:
            raise HarnessError(
                f"{name} still running after {TRAINER_DEADLINE_S} s: "
                f"{h.logs(name)[-1500:]}"
            )
        time.sleep(5)


def refit_seconds(log: str) -> float:
    """Seconds from the BEFORE_PAUSE line to the AFTER_CONTINUE line of a
    ``docker logs --timestamps`` output."""
    seen = {}
    for line in log.splitlines():
        stamp, _, text = line.partition(" ")
        for mark in (BEFORE_PAUSE, AFTER_CONTINUE):
            if mark in text:
                seen[mark] = _seconds(stamp)
    if len(seen) < 2:
        raise HarnessError(f"trainer log lacks {BEFORE_PAUSE!r} or {AFTER_CONTINUE!r}")
    return round(seen[AFTER_CONTINUE] - seen[BEFORE_PAUSE], 2)


def _seconds(stamp: str) -> float:
    """POSIX seconds of docker's ``2026-11-18T10:00:00.123456789Z``."""
    whole, _, fraction = stamp.rstrip("Z").partition(".")
    when = datetime.datetime.fromisoformat(whole).replace(tzinfo=datetime.timezone.utc)
    return when.timestamp() + float(f"0.{fraction or 0}")


def judge(rows: dict) -> Verdict:
    """The verdict on each row's versions and its ``equals_fresh`` comparison."""
    failed = []
    for row, got in rows.items():
        if got["weight_versions"] != [VERSION]:
            failed.append(f"{row} weight_version {got['weight_versions']}")
        if not got["fresh"]["equal"]:
            failed.append(f"{row} differs from fresh at {got['fresh']['first']}")
    return Verdict(not failed, "; ".join(failed), {"rows": rows})
