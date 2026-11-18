# SPDX-License-Identifier: MIT
"""tools/rl/accept.py's refit_trainer and timing checks: the refit_trainer
``--dry-run`` plan through accept.py's loader, its verdicts on a fake trainer
and workers, and the timing verdict on canned results."""

import json
import shlex
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "rl"))
import accept
from checks import refit_trainer

REPOS = ("Qwen3-1.7B-Base", "Qwen3-1.7B", "Qwen3-8B-Base", "Qwen3-8B")


@pytest.fixture
def models(tmp_path, monkeypatch):
    """An HF cache holding the checkpoints, so nothing is downloaded."""
    for repo in REPOS:
        (tmp_path / "hf" / "hub" / f"models--Qwen--{repo}" / "snapshots" / "s").mkdir(
            parents=True
        )
    monkeypatch.setattr(accept, "HF_HOME", str(tmp_path / "hf"))
    return tmp_path / "hf" / "hub"


def test_dry_run_plans_each_row_worker_trainer_then_fresh(models, monkeypatch, capsys):
    monkeypatch.setattr(accept.subprocess, "run", pytest.fail)
    assert accept.main(["refit_trainer", "--dry-run"]) == 0
    steps = capsys.readouterr().out.splitlines()
    runs = []
    for step in steps:
        if step.startswith("docker run"):
            argv = shlex.split(step)
            gpus = [a for a in argv if a.startswith("HIP_VISIBLE_DEVICES=")]
            runs.append(
                (argv[argv.index("--name") + 1], gpus[0][20:] if gpus else None)
            )
    assert runs == [
        ("rl-110-atom-18200", "0"),
        ("rl-110-mesh-build", None),
        ("rl-110-mesh-18202", None),
        ("rl-110-trainer-tp1", "1,0"),
        ("rl-110-atom-18205", "2"),
        ("rl-110-atom-18207", "3,4"),
        ("rl-110-mesh-18209", None),
        ("rl-110-trainer-tp2", "5,3,4"),
        ("rl-110-atom-18212", "6,7"),
    ]
    trainer = shlex.split(next(s for s in steps if "rl-110-trainer-tp2" in s))
    script = trainer[trainer.index("python3") :]
    assert script[:2] == [
        "python3",
        f"{accept.SMG_DIR}/examples/rl/refit_from_trainer.py",
    ]
    args = dict(zip(script[2::2], script[3::2]))
    assert args["--smg"] == "http://127.0.0.1:18209"
    assert args["--selector"] == "engine=atom"
    assert args["--model-path"] == str(models / "models--Qwen--Qwen3-8B/snapshots/s")
    assert args["--master-port"] == "18211"
    assert "tp_size 2" in steps[steps.index(next(s for s in steps if "18209" in s)) + 1]
    assert steps[-1] == refit_trainer.FRESH_COMPARE


def fake_trainer(exit_code=0, stale=(), drift=None, running=False, tp_size=None):
    """A Harness whose trainer refits each row's worker to the checkpoint it
    loads unless the row is in ``stale``; ``drift`` names a row whose outputs
    differ from fresh, ``running`` a trainer that never exits."""

    class FakeTrainer(accept.Harness):
        def __init__(self, issue, dry_run=False):
            super().__init__(issue, dry_run)
            self.held, self.rows = {}, []

        def model_dir(self, repo):
            return "/ckpt/" + repo

        def remove_leftovers(self):
            pass

        def start_atom_workers(self, n, model, tp=1, extra=()):
            url = f"http://w{len(self.held)}"
            self.held[url] = (model, "default")
            self.rows.append((url, tp))
            return [url]

        def start_mesh(self, urls, backend="atom", pd=False, model_path=None):
            self.mesh_url = "http://mesh"
            return self.mesh_url

        def call_mesh(self, method, path, body=None):
            tp = tp_size or self.rows[-1][1]
            return 200, {"workers": [{"tp_size": tp}], "total": 1}

        def start_trainer(self, gpus, argv, role="trainer", peers=()):
            row = f"tp{self.rows[-1][1]}"
            if exit_code == 0 and row not in stale:
                version = argv[argv.index("--weight-version") + 1]
                self.held[self.rows[-1][0]] = (
                    argv[argv.index("--model-path") + 1],
                    version,
                )
            return f"rl-110-{role}"

        def sh(self, argv, check=True):
            if argv[:3] == ["git", "-C", accept.SMG_DIR]:
                return refit_trainer.SMG_COMMIT if "rev-parse" in argv else ""
            if argv[:2] == ["docker", "inspect"]:
                return "true 0" if running else f"false {exit_code}"
            assert argv[:3] == ["docker", "logs", "--timestamps"]
            return (
                "2026-11-18T10:00:00.250000000Z loading /ckpt ...\n"
                "2026-11-18T10:00:59.500000000Z 311 parameter(s) to broadcast, 64 per call\n"
                "2026-11-18T10:01:02.000000000Z continue_generation ... resumed\n"
            )

        def logs(self, name):
            return "init_weights_update_group failed"

        def generate(self, url, prompts):
            meta = {"weight_version": self.held[url][1]}
            return [{"output_ids": [1], "meta_info": meta} for _ in prompts]

        def equals_fresh(self, url, checkpoint, prompts, outputs):
            row = f"tp{self.rows[-1][1]}"
            assert self.held[url][0] == checkpoint
            first = "prompt 3 token 7" if row == drift else None
            return {"equal": first is None, "first": first, "gap": 0.0, "margin": 0.5}

    return FakeTrainer


@pytest.fixture
def on_node(tmp_path, monkeypatch):
    monkeypatch.setattr(accept.socket, "gethostname", lambda: accept.NODE_HOST)
    monkeypatch.setattr(accept, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(refit_trainer.time, "sleep", lambda s: None)
    return tmp_path / "out"


class NoContainers(accept.Harness):
    def remove_leftovers(self):
        pass


def _run(monkeypatch, capsys, check, harness=NoContainers):
    monkeypatch.setattr(accept, "Harness", harness)
    code = accept.main([check])
    return code, capsys.readouterr().out.strip()


def test_refit_trainer_passes_with_a_row_per_tp(on_node, monkeypatch, capsys):
    assert _run(monkeypatch, capsys, "refit_trainer", fake_trainer()) == (
        0,
        "PASS refit_trainer",
    )
    rows = json.loads((on_node / "refit_trainer.json").read_text())["data"]["rows"]
    assert rows["tp2"] == {
        "tp_size": 2,
        "refit_s": 2.5,
        "weight_versions": [refit_trainer.VERSION],
        "fresh": {"equal": True, "first": None, "gap": 0.0, "margin": 0.5},
    }
    assert set(rows) == {"tp1", "tp2"}


@pytest.mark.parametrize(
    "harness,reason",
    [
        (fake_trainer(stale=("tp2",)), "tp2 weight_version ['default']"),
        (fake_trainer(drift="tp1"), "tp1 differs from fresh at prompt 3 token 7"),
        (fake_trainer(exit_code=1), "HarnessError: tp1: rl-110-trainer-tp1 exited 1"),
        (fake_trainer(running=True), "HarnessError: rl-110-trainer-tp1 still running"),
        (fake_trainer(tp_size=1), "HarnessError: tp2: /v1/rl/workers tp_size [1]"),
    ],
)
def test_refit_trainer_fails_naming_the_row(
    on_node, monkeypatch, capsys, harness, reason
):
    monkeypatch.setattr(refit_trainer, "TRAINER_DEADLINE_S", 0)
    code, out = _run(monkeypatch, capsys, "refit_trainer", harness)
    assert code == 1 and out.startswith(f"FAIL refit_trainer: {reason}")


def test_refit_trainer_refuses_another_smg_commit(on_node, monkeypatch, capsys):
    class Moved(fake_trainer()):
        def sh(self, argv, check=True):
            return "0" * 40 if "rev-parse" in argv else super().sh(argv, check)

    code, out = _run(monkeypatch, capsys, "refit_trainer", Moved)
    assert code == 1 and out.startswith(
        f"FAIL refit_trainer: {accept.SMG_DIR} is at 0000"
    )


def _record(out: Path, check: str, data: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    record = {"check": check, "commit": "c0ffee", "passed": True, "data": data}
    (out / f"{check}.json").write_text(json.dumps(record))


def test_timing_passes_with_each_refit_time(on_node, monkeypatch, capsys):
    _record(on_node, "refit_disk", {"refit_s": 2.66, "workers": {}})
    rows = {"tp1": {"refit_s": 1.5}, "tp2": {"refit_s": 6.25}}
    _record(on_node, "refit_trainer", {"rows": rows})
    assert _run(monkeypatch, capsys, "timing") == (0, "PASS timing")
    data = json.loads((on_node / "timing.json").read_text())["data"]
    assert data["refit_disk"] == {"commit": "c0ffee", "passed": True, "refit_s": 2.66}
    assert data["refit_trainer"]["refit_s"] == {"tp1": 1.5, "tp2": 6.25}


@pytest.mark.parametrize(
    "trainer_data,reason",
    [
        (None, "no refit time in refit_trainer"),
        ({}, "no refit time in refit_trainer"),
        (
            {"rows": {"tp1": {"refit_s": 1.5}, "tp2": {}}},
            "no refit time in refit_trainer",
        ),
    ],
)
def test_timing_fails_without_a_trainer_time(
    on_node, monkeypatch, capsys, trainer_data, reason
):
    _record(on_node, "refit_disk", {"refit_s": 2.66})
    if trainer_data is not None:
        _record(on_node, "refit_trainer", trainer_data)
    assert _run(monkeypatch, capsys, "timing") == (1, f"FAIL timing: {reason}")
