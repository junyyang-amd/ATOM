# SPDX-License-Identifier: MIT
"""tools/rl/accept.py's selector and smg_client checks: their ``--dry-run``
plan through accept.py's loader, and their verdicts on canned answers."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from test_rl_accept_refit_disk import _container, _run, fake_atom, models, on_node

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "rl"))
import accept
from checks import smg_client

__all__ = ["models", "on_node"]  # fixtures, imported for this module's tests


def test_selector_plans_a_miss_then_a_refit_of_worker_1_and_two_fresh_workers(
    models, monkeypatch, capsys
):
    monkeypatch.setattr(accept.subprocess, "run", pytest.fail)
    assert accept.main(["selector", "--dry-run"]) == 0
    steps = capsys.readouterr().out.splitlines()
    runs = [_container(s) for s in steps if s.startswith("docker run")]
    assert runs[-2:] == [("rl-108-atom-18206", "2"), ("rl-108-atom-18208", "3")]
    assert "Qwen3-1.7B/" in steps[-3] and "Qwen3-1.7B-Base/" in steps[-2]
    calls = [s.split(" {")[0] for s in steps if s.startswith("POST /v1/rl/")]
    assert calls[0].endswith(
        "?selector=url=http://no-such-worker -> 400 no_workers_match"
    )
    worker_1 = f"selector=url=http://{accept.NODE_IP}:18200"
    assert all(c.endswith(worker_1) for c in calls[1:]) and len(calls) == 4


def fake_select(ignore_selector=False, miss=(400, {"error": "no_workers_match"})):
    """fake_atom's workers behind an ATOMesh that applies a fan-out to the
    workers whose URL the selector names (to every worker with
    ``ignore_selector``) and answers ``miss`` when it names none."""

    class FakeSelect(fake_atom()):
        def rl_fanout(self, path, body, selector):
            named = [u for u in self.workers if selector == f"url={u}"]
            if not named:
                return miss
            if path == "update_weights_from_disk":
                for url in self.workers if ignore_selector else named:
                    self.held[url] = (body["model_path"], body["weight_version"])
            ok = {u: {"status": 200, "body": {"success": True}} for u in named}
            return 200, {"results": ok, "failed": [], "total": 1, "succeeded": 1}

        def equals_fresh(self, url, checkpoint, prompts, outputs):
            assert self.held[url][0] == checkpoint
            same = all(
                o["output_ids"][0] == hash((checkpoint, p))
                for p, o in zip(prompts, outputs)
            )
            return {
                "equal": same,
                "first": None if same else "prompt 0",
                "gap": 0.0,
                "margin": 0.5,
            }

    return FakeSelect


@pytest.mark.parametrize(
    "harness,reason",
    [
        (fake_select(), None),
        (
            fake_select(ignore_selector=True),
            "http://w1 weight_version ['selector-v2']; http://w1 differs from fresh at prompt 0",
        ),
        (
            fake_select(miss=(200, {})),
            "selector url=http://no-such-worker answered 200 None",
        ),
    ],
)
def test_selector_passes_only_when_worker_2_keeps_the_old_weights(
    on_node, monkeypatch, capsys, harness, reason
):
    code, out = _run(monkeypatch, capsys, "selector", harness)
    assert (code, out) == (
        (0, "PASS selector") if reason is None else (1, f"FAIL selector: {reason}")
    )


def test_smg_client_plans_smgs_script_against_atomesh(models, monkeypatch, capsys):
    monkeypatch.setattr(accept.subprocess, "run", pytest.fail)
    assert accept.main(["smg_client", "--dry-run", "--smg", "/smg"]) == 0
    steps = capsys.readouterr().out.splitlines()
    assert steps[0] == f"/smg must be at {smg_client.SMG_COMMIT}, unmodified"
    assert shlex.split(steps[-1])[:4] == [
        "env",
        "PYTHONPATH=/smg/bindings/python/src",
        "python3",
        "/smg/examples/rl/refit_from_disk.py",
    ]
    assert steps[-1].endswith(
        f"--weight-version {smg_client.VERSION} --selector engine=atom"
    )
    assert f"--smg http://127.0.0.1:18204 --model-path {accept.HF_HOME}" in steps[-1]


def fake_smg(script_fails=False):
    class FakeSmg(fake_atom()):
        def sh(self, argv, check=True):
            if argv[0] == "git":
                return super().sh(argv, check)
            if script_fails:
                raise accept.HarnessError(
                    "MISMATCH: engine did not report the new version"
                )
            assert argv[-2:] == ["--selector", "engine=atom"]
            return "OK\n"

    return FakeSmg


def _smg_checkout(path, commit_msg="c"):
    git = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        [*git, "commit", "-q", "--allow-empty", "-m", commit_msg], check=True
    )
    return subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.mark.parametrize("script_fails", [False, True])
def test_smg_client_passes_on_the_scripts_exit_status(
    on_node, tmp_path, monkeypatch, capsys, script_fails
):
    monkeypatch.setattr(smg_client, "SMG_COMMIT", _smg_checkout(tmp_path / "smg"))
    monkeypatch.setattr(accept, "Harness", fake_smg(script_fails))
    code = accept.main(["smg_client", "--smg", str(tmp_path / "smg")])
    out = capsys.readouterr().out.strip()
    if script_fails:
        assert (code, out) == (
            1,
            "FAIL smg_client: HarnessError: MISMATCH: engine did not report the new version",
        )
    else:
        assert (code, out) == (0, "PASS smg_client")
        assert json.loads((on_node / "smg_client.json").read_text())["data"] == {
            "script": ["OK"]
        }


def test_smg_client_refuses_another_smg_commit(on_node, tmp_path, monkeypatch, capsys):
    head = _smg_checkout(tmp_path / "smg")
    monkeypatch.setattr(accept, "Harness", fake_smg())
    code = accept.main(["smg_client", "--smg", str(tmp_path / "smg")])
    out = capsys.readouterr().out.strip()
    assert code == 1
    assert out == (
        f"FAIL smg_client: HarnessError: {tmp_path / 'smg'} is at {head}; "
        f"the check runs SMG {smg_client.SMG_COMMIT} unmodified"
    )


# g48's GPUs: amd-smi's index (PCI bus order) and KFD node, from `amd-smi list`.
G48_NODES = {0: 3, 1: 5, 2: 4, 3: 2, 4: 7, 5: 9, 6: 8, 7: 6}


@pytest.mark.parametrize("hip,busy", [(0, True), (1, False), (3, False)])
def test_a_busy_gpu_is_found_by_its_hip_index(monkeypatch, hip, busy):
    """amd-smi's GPU 3 is HIP's GPU 0: HIP numbers GPUs in KFD node order."""
    listed = [{"gpu": g, "node_id": n} for g, n in G48_NODES.items()]
    rows = [
        {"gpu": g, "vram_used": {"value": 250000 if g == 3 else 283}} for g in G48_NODES
    ]
    h = accept.Harness(108)
    monkeypatch.setattr(
        h, "sh", lambda argv: json.dumps(listed if "list" in argv else rows)
    )
    if busy:
        with pytest.raises(accept.HarnessError, match=rf"GPU \[{hip}\] busy"):
            h.refuse_busy([hip])
    else:
        h.refuse_busy([hip])
