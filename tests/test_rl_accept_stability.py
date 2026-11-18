# SPDX-License-Identifier: MIT
"""tools/rl/accept.py's stability check: its ``--dry-run`` plan through
accept.py's loader, and its verdict over a fake ATOMesh."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "rl"))
import accept
from checks import stability


def test_dry_run_plans_two_workers_then_twenty_alternating_refits(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(accept, "HF_HOME", str(tmp_path))
    monkeypatch.setattr(accept.subprocess, "run", pytest.fail)
    assert accept.main(["stability", "--dry-run"]) == 0
    steps = capsys.readouterr().out.splitlines()
    names = [s.split("--name ")[1].split()[0] for s in steps if "--name " in s]
    assert names[-4:] == [
        "rl-109-atom-18200",
        "rl-109-atom-18202",
        "rl-109-mesh-build",
        "rl-109-mesh-18204",
    ]
    rounds = [s for s in steps if s.startswith("round ")]
    assert len(rounds) == 20 and "Qwen3-1.7B as stability-1," in rounds[0]
    assert "Qwen3-1.7B-Base as stability-2," in rounds[1]


def fake_mesh(fail_round=None, new_id_round=None):
    """A Harness over a fake ATOMesh on two workers. In ``fail_round`` one
    rollout gets a 502; from ``new_id_round`` on, a worker is listed anew."""

    class FakeMesh(accept.Harness):
        loaded, rounds = None, 0

        def model_dir(self, repo):
            return "/ckpt/" + repo

        def remove_leftovers(self):
            pass

        def start_atom_workers(self, n, model, tp=1, extra=()):
            return ["http://w0", "http://w1"]

        def start_mesh(self, urls, backend="atom", pd=False, model_path=None):
            self.mesh_url = "http://mesh"

        def rl_fanout(self, path, body, selector):
            if path == "update_weights_from_disk":
                self.loaded, self.rounds = body, self.rounds + 1
            ok = {"status": 200, "body": {"success": True}}
            return 200, {
                "results": {"w0": ok},
                "failed": [],
                "total": 1,
                "succeeded": 1,
            }

        def call_mesh(self, method, path, body=None):
            renamed = new_id_round is not None and self.rounds >= new_id_round
            ids = ["b", "x" if renamed else "a"]
            workers = [{"id": i, "url": f"http://w{i}"} for i in ids]
            return 200, {"workers": workers}

        def generate(self, url, prompts):
            if self.rounds == fail_round and prompts[0] == accept.PROMPTS[3]:
                raise accept.HarnessError(f"/generate on {url} answered 502")
            return [{"meta_info": {"weight_version": self.loaded["weight_version"]}}]

    return FakeMesh


@pytest.mark.parametrize(
    "harness,out",
    [
        (fake_mesh(), "PASS stability"),
        (
            fake_mesh(fail_round=3),
            "FAIL stability: round 3: 2 of 64 requests failed, first "
            + "HarnessError: /generate on http://mesh answered 502",
        ),
        (
            fake_mesh(new_id_round=20),
            "FAIL stability: round 20: workers [['b', 'http://wb'], ['x', 'http://wx']]",
        ),
    ],
    ids=["pass", "failed_request", "worker_relisted"],
)
def test_stability_verdict(tmp_path, monkeypatch, capsys, harness, out):
    monkeypatch.setattr(accept.socket, "gethostname", lambda: accept.NODE_HOST)
    monkeypatch.setattr(accept, "OUT_DIR", tmp_path)
    monkeypatch.setattr(accept, "Harness", harness)
    assert accept.main(["stability"]) == (0 if out.startswith("PASS") else 1)
    assert capsys.readouterr().out.strip() == out
    data = json.loads((tmp_path / "stability.json").read_text())["data"]
    assert len(data["failed_per_round"]) == 20
    assert [r["weight_versions"] for r in data["rounds"][:2]] == [
        ["stability-1"],
        ["stability-2"],
    ]
    assert data["rounds"][1]["checkpoint"] == stability.OLD
