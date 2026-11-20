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


def test_containers_running_the_tree_mount_it_at_its_own_path():
    """Mounting /share_nfs alone leaves a tree elsewhere unmounted: docker then
    creates the -w directory empty and the image's own atom runs."""
    h = accept.Harness(106, dry_run=True)
    h.start_atom_workers(1, "/ckpt")
    h.start_trainer([1], ["python3", "train.py"])
    h.build_mesh()
    tree = f"{accept.TREE}:{accept.TREE}"
    for step in h.steps:
        argv = shlex.split(step)
        assert tree in [argv[i + 1] for i, a in enumerate(argv) if a == "-v"], step


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


def fake_mesh(stale=(), no_selector="selector_required"):
    """A Harness whose ATOMesh answers over two SGLang workers as SMG's protocol
    says, except that workers in ``stale`` ignore an update. A fan-out with no
    selector is refused with 400 and the error code ``no_selector``, or goes to
    every worker when that is None."""

    class FakeMesh(accept.Harness):
        def __init__(self, issue, dry_run=False):
            super().__init__(issue, dry_run)
            self.versions = {"w0": "default", "w1": "default"}

        def model_dir(self, repo):
            return "/ckpt/" + repo

        def remove_leftovers(self):
            pass

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
            if not selector and no_selector:
                return 400, {"error": no_selector}
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
        (fake_mesh(no_selector=None), "probe selector_required"),
        (fake_mesh(no_selector="invalid_selector"), "probe selector_required"),
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


def test_a_run_first_removes_its_issues_leftover_containers(
    on_node, monkeypatch, capsys
):
    """An earlier run whose ssh went away never tore down; its containers hold
    GPUs, so the next run removes them before taking any."""
    real_run, calls = accept.subprocess.run, []

    def run(argv, *args, **kwargs):
        if argv[0] != "docker":
            return real_run(argv, *args, **kwargs)
        calls.append(argv)
        names = "rl-106-sglang-18200\nrl-1060-x\nrl-107-sglang-18200\n"
        out = names if argv[1] == "ps" else ""
        return accept.subprocess.CompletedProcess(argv, 0, out, "")

    class Leftovers(fake_mesh()):
        remove_leftovers = accept.Harness.remove_leftovers

    monkeypatch.setattr(accept.subprocess, "run", run)
    code, out = _run(monkeypatch, capsys, Leftovers)
    assert (code, out) == (0, "PASS mesh_sglang")
    assert ["docker", "rm", "-f", "rl-106-sglang-18200"] in calls
    assert not [c for c in calls if c[1] == "rm" and "rl-107-sglang-18200" in c]


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """A fresh server over a three-token byte-level vocabulary plus EOS:
    prompt "p" is ids [7, 8], and ``tops`` maps the ids after the prompt to
    the chat route's top logprobs there (``[]``: its top token is hidden).
    The chat requests are recorded in ``asked``."""
    vocab = {"a": 0, "Ġb": 1, "c": 2}  # Ġ is the byte-level space
    eos = {"id": 3, "content": "<|im_end|>", "special": True}
    tokenizer = {"model": {"vocab": vocab}, "added_tokens": [eos]}
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer))
    tops, asked = {}, []

    def http(method, url, body=None, timeout=None):
        if url.endswith("/v1/completions"):
            return 200, {"prompt_token_ids": [7, 8]}
        asked.append(body["prompt_token_ids"])
        top = [
            {"bytes": list(raw), "logprob": lp}
            for raw, lp in tops[tuple(body["prompt_token_ids"][2:])]
        ]
        entries = [{"top_logprobs": top}] if top else []
        return 200, {"choices": [{"logprobs": {"content": entries}}]}

    monkeypatch.setattr(accept, "http", http)
    return str(tmp_path), tops, asked


@pytest.mark.parametrize(
    "behind,equal,why",
    [
        (0.0, True, None),
        (0.75, True, None),  # a step above two fresh servers' largest gap
        (accept.FRESH_MARGIN, True, None),
        (accept.FRESH_MARGIN + 0.25, False, f"{accept.FRESH_MARGIN + 0.25:.3f} below"),
        (3.75, False, "3.750 below"),  # Qwen3-1.7B-Base against fresh Qwen3-1.7B
    ],
)
def test_equals_fresh_holds_each_token_within_the_margin_of_the_fresh_top(
    fresh, behind, equal, why
):
    checkpoint, tops, asked = fresh
    tops[()] = [(b"a", -0.1), (b" b", -0.1 - behind)]
    tops[(1,)] = [(b"c", -0.2)]
    got = accept.Harness(107).equals_fresh(
        "u", checkpoint, ["p"], [{"output_ids": [1, 2]}]
    )
    assert got["equal"] is equal and got["margin"] == accept.FRESH_MARGIN
    assert got["first"] == (f"prompt 0 'p' token 0 (1): {why}" if why else None)
    assert got["gap"] == pytest.approx(behind if equal else 0.0)
    assert asked == ([[7, 8], [7, 8, 1]] if equal else [[7, 8]])


def test_equals_fresh_fails_a_token_outside_the_fresh_top(fresh):
    checkpoint, tops, _ = fresh
    tops[()] = [(b"a", -0.1)]
    got = accept.Harness(107).equals_fresh(
        "u", checkpoint, ["p"], [{"output_ids": [2]}]
    )
    assert got["first"] == "prompt 0 'p' token 0 (2): not in its top"


@pytest.mark.parametrize("token,equal", [(3, True), (0, False)])
def test_equals_fresh_takes_a_hidden_fresh_top_as_a_special_token(fresh, token, equal):
    checkpoint, tops, _ = fresh
    tops[()] = []
    outputs = [{"output_ids": [token]}]
    assert (
        accept.Harness(107).equals_fresh("u", checkpoint, ["p"], outputs)["equal"]
        is equal
    )


def test_equals_fresh_asks_a_prefix_once_per_server(fresh):
    checkpoint, tops, asked = fresh
    tops[()] = [(b"a", -0.1)]
    h, outputs = accept.Harness(107), [{"output_ids": [0]}]
    assert all(h.equals_fresh("u", checkpoint, ["p"], outputs)["equal"] for _ in "xy")
    assert asked == [[7, 8]]


class _Answer:
    """What ``urlopen`` returns: a 200 with a JSON body."""

    status = 200

    def __init__(self, body):
        self._raw = json.dumps(body).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_atomesh_runs_with_the_rl_bearer_and_every_mesh_call_carries_it(monkeypatch):
    sent = []

    def urlopen(req, timeout):
        sent.append((req.full_url, req.get_header("Authorization")))
        return _Answer({"total": 1})

    monkeypatch.setattr(accept.urllib.request, "urlopen", urlopen)
    h = accept.Harness(106)
    ran = []
    monkeypatch.setattr(h, "sh", lambda argv, check=True: ran.append(argv) or "")
    monkeypatch.setattr(accept.socket.socket, "connect_ex", lambda self, addr: 1)
    mesh = h.start_mesh(["http://w0"])
    h.call_mesh("GET", "/v1/rl/workers")
    launch = next(argv for argv in ran if "launch" in argv)
    key = launch.index("--rl-api-key")
    assert launch[key + 1] == accept.RL_API_KEY
    bearer = f"Bearer {accept.RL_API_KEY}"
    assert sent == [(mesh + "/v1/rl/workers", bearer)] * 2
