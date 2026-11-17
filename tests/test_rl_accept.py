# SPDX-License-Identifier: MIT
"""tools/rl/accept.py: the ATOM worker command lines it starts, ``--dry-run``,
and the verdict, exit status and JSON of ``mesh_sglang`` on canned answers."""

import json
import shlex
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "rl"))
import accept
from checks import mesh_sglang

SERVER = "exec python3 -m atom.entrypoints.openai_server "


@pytest.fixture
def models(tmp_path, monkeypatch):
    """An HF cache holding the checkpoints, so nothing is downloaded."""
    for repo in ("Qwen3-1.7B-Base", "Qwen3-1.7B"):
        (tmp_path / "hf" / "hub" / f"models--Qwen--{repo}" / "snapshots" / "s").mkdir(
            parents=True
        )
    monkeypatch.setattr(accept, "HF_HOME", str(tmp_path / "hf"))


def _server_argv(step: str) -> list[str]:
    """The openai_server argv inside one dry-run ``docker run ... bash -c``."""
    script = shlex.split(step)[-1]
    return shlex.split(script.split(SERVER, 1)[1])


def _parse(argv):
    return EngineArgs.add_cli_args(FlexibleArgumentParser()).parse_known_args(argv)


def test_atom_workers_start_as_rl_servers_on_their_own_ports():
    h = accept.Harness(106, dry_run=True)
    h.start_atom_workers(2, "/ckpt", tp=2)
    parsed = [_parse(_server_argv(step)) for step in h.steps]
    for (ns, unknown), port in zip(parsed, (18200, 18202)):
        assert (ns.model, ns.enable_rl, ns.tensor_parallel_size) == ("/ckpt", True, 2)
        assert ns.port == port + 1
        assert unknown == ["--host", "0.0.0.0", "--server-port", str(port)]
    assert [s.split("HIP_VISIBLE_DEVICES=")[1].split()[0] for s in h.steps] == [
        "0,1",
        "2,3",
    ]


def test_pd_pair_moves_kv_over_mooncake_tcp():
    h = accept.Harness(111, dry_run=True)
    h.start_pd_pair("/ckpt")
    configs = [
        json.loads(_parse(_server_argv(s))[0].kv_transfer_config) for s in h.steps
    ]
    assert [c["kv_role"] for c in configs] == ["kv_producer", "kv_consumer"]
    assert {c["protocol"] for c in configs} == {"tcp"}
    assert configs[0]["handshake_port"] != configs[1]["handshake_port"]


def test_dry_run_prints_each_container_and_runs_nothing(models, monkeypatch, capsys):
    def refuse(argv, *args, **kwargs):
        raise AssertionError(f"dry run ran {argv}")

    monkeypatch.setattr(accept.subprocess, "run", refuse)
    assert accept.main(["mesh_sglang", "--dry-run"]) == 0
    runs = [
        s for s in capsys.readouterr().out.splitlines() if s.startswith("docker run")
    ]
    assert [shlex.split(s)[shlex.split(s).index("--name") + 1] for s in runs] == [
        "rl-106-sglang-18200",
        "rl-106-sglang-18201",
        "rl-106-mesh-build",
        "rl-106-mesh-18202",
    ]
    assert " ".join(accept.LONG_ROLLOUT) in runs[-1]


def fake_mesh(stale=(), selector_required=True):
    """A Harness whose ATOMesh answers over two SGLang workers as SMG's protocol
    says, except that workers in ``stale`` ignore an update and, without
    ``selector_required``, a fan-out with no selector goes to every worker."""

    class FakeMesh(accept.Harness):
        def __init__(self, issue, dry_run=False):
            super().__init__(issue, dry_run)
            self.versions = {"w0": "default", "w1": "default"}

        def model_dir(self, repo):
            return "/ckpt/" + repo

        def start_sglang_workers(self, n, model):
            return [f"http://w{i}" for i in range(n)]

        def start_mesh(self, urls, backend="atom", pd=False, model_path=None):
            self.mesh_url = "http://mesh"
            return self.mesh_url

        def entry(self, wid):
            return {"id": wid, "url": f"http://{wid}", "engine": "sglang"}

        def call_mesh(self, method, path, body=None):
            url = urlsplit(path)
            parts = url.path.strip("/").split("/")[2:]
            if parts == ["workers"]:
                rows = [self.entry(w) for w in self.versions]
                return 200, {"protocol_version": 1, "workers": rows, "total": 2}
            if parts[0] == "workers":
                if parts[1] not in self.versions:
                    return 404, {"error": "worker_not_found"}
                if len(parts) == 2:
                    return 200, self.entry(parts[1])
                if "%20" in parts[3]:
                    return 400, {"error": "invalid_engine_path"}
                version = self.versions[parts[1]]
                return 200, {"status": 200, "body": {"weight_version": version}}
            selector = parse_qs(url.query).get("selector", [""])[0]
            if not selector and selector_required:
                return 400, {"error": "selector_required"}
            if selector.startswith("="):
                return 400, {"error": "invalid_selector"}
            if selector == "engine=vllm":
                return 400, {"error": "no_workers_match"}
            for wid in self.versions:
                if parts[1] == "update_weights_from_disk" and wid not in stale:
                    self.versions[wid] = body["weight_version"]
            results = {w: {"status": 200} for w in self.versions}
            return 200, {"results": results, "failed": [], "total": 2, "succeeded": 2}

    return FakeMesh


@pytest.fixture
def on_node(tmp_path, monkeypatch):
    monkeypatch.setattr(accept.socket, "gethostname", lambda: accept.NODE_HOST)
    monkeypatch.setattr(accept, "OUT_DIR", tmp_path / "out")
    return tmp_path / "out" / "mesh_sglang.json"


def _run(monkeypatch, capsys, harness):
    monkeypatch.setattr(accept, "Harness", harness)
    code = accept.main(["mesh_sglang"])
    return code, capsys.readouterr().out.strip()


def test_mesh_sglang_passes_when_every_route_answers_as_smg(
    on_node, monkeypatch, capsys
):
    code, out = _run(monkeypatch, capsys, fake_mesh())
    assert (code, out) == (0, "PASS mesh_sglang")
    record = json.loads(on_node.read_text())
    assert record["passed"] and record["check"] == "mesh_sglang"
    routes = record["data"]["routes"]
    assert all(r["ok"] for r in routes.values())
    assert routes["model_info w1 after"]["body"]["body"] == {
        "weight_version": mesh_sglang.VERSION
    }
    assert {f"probe {code}" for code in mesh_sglang.PROBES} <= routes.keys()


@pytest.mark.parametrize(
    "harness,reason",
    [
        (fake_mesh(stale=("w1",)), "model_info w1 after"),
        (fake_mesh(selector_required=False), "probe selector_required"),
    ],
)
def test_mesh_sglang_fails_naming_the_route(
    on_node, monkeypatch, capsys, harness, reason
):
    code, out = _run(monkeypatch, capsys, harness)
    assert code == 1 and out == f"FAIL mesh_sglang: {reason}"
    record = json.loads(on_node.read_text())
    assert not record["passed"] and record["reason"] == reason


def test_a_harness_error_is_a_fail(on_node, monkeypatch, capsys):
    class Busy(fake_mesh()):
        def start_sglang_workers(self, n, model):
            raise accept.HarnessError("GPU [0] busy")

    code, out = _run(monkeypatch, capsys, Busy)
    assert code == 1 and out == "FAIL mesh_sglang: HarnessError: GPU [0] busy"
    assert not json.loads(on_node.read_text())["passed"]
