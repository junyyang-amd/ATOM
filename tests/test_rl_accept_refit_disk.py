# SPDX-License-Identifier: MIT
"""tools/rl/accept.py's refit_disk and cache checks: their ``--dry-run`` plan
through accept.py's loader, and their verdicts on canned worker answers."""

import json
import shlex
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "rl"))
import accept
from checks import cache, refit_disk


@pytest.fixture
def models(tmp_path, monkeypatch):
    """An HF cache holding the checkpoints, so nothing is downloaded."""
    for repo in ("Qwen3-1.7B-Base", "Qwen3-1.7B"):
        (tmp_path / "hf" / "hub" / f"models--Qwen--{repo}" / "snapshots" / "s").mkdir(
            parents=True
        )
    monkeypatch.setattr(accept, "HF_HOME", str(tmp_path / "hf"))


def _container(step: str) -> tuple[str, str | None]:
    """(name, HIP_VISIBLE_DEVICES) of a ``docker run`` step."""
    argv = shlex.split(step)
    gpus = [a.split("=", 1)[1] for a in argv if a.startswith("HIP_VISIBLE_DEVICES=")]
    return argv[argv.index("--name") + 1], (gpus or [None])[0]


CYCLE = [f"POST /v1/rl/engine/{p}" for p in refit_disk.CYCLE]


@pytest.mark.parametrize(
    "check,calls",
    [
        (
            "refit_disk",
            [
                *CYCLE,
                "18200/generate x32, one at a time",
                "18202/generate x32, one at a time",
            ],
        ),
        (
            "cache",
            ["18200/generate PROBE x2", "18202/generate PROBE x2", *CYCLE]
            + ["18200/generate PROBE", "18202/generate PROBE"],
        ),
    ],
)
def test_dry_run_plans_two_workers_a_refit_then_a_fresh_worker(
    models, monkeypatch, capsys, check, calls
):
    monkeypatch.setattr(accept.subprocess, "run", pytest.fail)
    assert accept.main([check, "--dry-run"]) == 0
    steps = capsys.readouterr().out.splitlines()
    runs = [_container(s) for s in steps if s.startswith("docker run")]
    assert runs == [
        ("rl-107-atom-18200", "0"),
        ("rl-107-atom-18202", "1"),
        ("rl-107-mesh-build", None),
        ("rl-107-mesh-18204", None),
        ("rl-107-atom-18206", "2"),
    ]
    assert "--backend atom" in steps[3] and "rl-107-atom-18206" in steps[-2]
    between = steps[4:-2]
    assert [
        s.split(":")[-1] if "/generate" in s else s.split("?")[0] for s in between
    ] == calls
    assert all("?selector=engine=atom " in s for s in between if "/v1/rl/" in s)
    assert steps[-1].startswith("POST <fresh>/generate")


def fake_atom(stale=(), drift=None, keep_cache=False):
    """A Harness over fake ATOM workers that each answer ``/generate`` by the
    checkpoint they hold. Workers in ``stale`` ignore a refit, ``drift`` is a
    prompt index the refitted workers answer differently (-1: any prompt not
    in PROMPTS), and with
    ``keep_cache`` a refit leaves the prefix cache in place."""

    class FakeAtom(accept.Harness):
        def __init__(self, issue, dry_run=False):
            super().__init__(issue, dry_run)
            self.held, self.seen, self.fanouts = {}, {}, []

        def model_dir(self, repo):
            return "/ckpt/" + repo

        def remove_leftovers(self):
            pass

        def start_atom_workers(self, n, model, tp=1, extra=()):
            urls = [f"http://w{len(self.held) + i}" for i in range(n)]
            self.held |= {u: (model, "default") for u in urls}
            self.seen |= {u: set() for u in urls}
            return urls

        def start_mesh(self, urls, backend="atom", pd=False, model_path=None):
            self.workers, self.mesh_url = list(urls), "http://mesh"
            return self.mesh_url

        def rl_fanout(self, path, body, selector):
            assert selector == "engine=atom"
            self.fanouts.append(path)
            results = {}
            for url in self.workers:
                if path == "update_weights_from_disk" and url not in stale:
                    self.held[url] = (body["model_path"], body["weight_version"])
                    self.seen[url] = self.seen[url] if keep_cache else set()
                results[url] = {"status": 200, "body": {"success": True}}
            return 200, {"results": results, "failed": [], "total": 2, "succeeded": 2}

        def generate(self, url, prompts):
            model, version = self.held[url]
            out = []
            for prompt in prompts:
                i = accept.PROMPTS.index(prompt) if prompt in accept.PROMPTS else -1
                ids = [hash((model, prompt)), int(i == drift and version != "default")]
                cached = 16 if prompt in self.seen[url] else 0
                self.seen[url].add(prompt)
                meta = {"weight_version": version, "cached_tokens": cached}
                out.append({"output_ids": ids, "meta_info": meta})
            return out

    return FakeAtom


@pytest.fixture
def on_node(tmp_path, monkeypatch):
    monkeypatch.setattr(accept.socket, "gethostname", lambda: accept.NODE_HOST)
    monkeypatch.setattr(accept, "OUT_DIR", tmp_path / "out")
    return tmp_path / "out"


def _run(monkeypatch, capsys, check, harness):
    monkeypatch.setattr(accept, "Harness", harness)
    code = accept.main([check])
    return code, capsys.readouterr().out.strip()


def test_refit_disk_passes_and_records_the_refit_time(on_node, monkeypatch, capsys):
    assert _run(monkeypatch, capsys, "refit_disk", fake_atom()) == (
        0,
        "PASS refit_disk",
    )
    record = json.loads((on_node / "refit_disk.json").read_text())
    assert record["passed"] and record["data"]["refit_s"] >= 0
    assert record["data"]["workers"]["http://w1"] == {
        "weight_versions": [refit_disk.VERSION],
        "mismatch": None,
    }


@pytest.mark.parametrize(
    "harness,reason",
    [
        (fake_atom(stale=("http://w1",)), "http://w1 weight_version ['default']"),
        (
            fake_atom(drift=5),
            f"http://w0 prompt 5 {accept.PROMPTS[5]!r} differs from fresh",
        ),
    ],
)
def test_refit_disk_fails_naming_the_worker_and_first_differing_prompt(
    on_node, monkeypatch, capsys, harness, reason
):
    code, out = _run(monkeypatch, capsys, "refit_disk", harness)
    assert code == 1 and out.startswith(f"FAIL refit_disk: {reason}")


def test_refit_disk_fails_on_a_refused_fanout(on_node, monkeypatch, capsys):
    class Refused(fake_atom()):
        def rl_fanout(self, path, body, selector):
            status, answer = super().rl_fanout(path, body, selector)
            if path == "update_weights_from_disk":
                answer["results"]["http://w0"]["body"]["success"] = False
            return status, answer

    code, out = _run(monkeypatch, capsys, "refit_disk", Refused)
    assert code == 1
    assert out.startswith("FAIL refit_disk: HarnessError: update_weights_from_disk")


def test_cache_passes_when_the_refit_drops_the_prefix(on_node, monkeypatch, capsys):
    assert _run(monkeypatch, capsys, "cache", fake_atom()) == (0, "PASS cache")
    workers = json.loads((on_node / "cache.json").read_text())["data"]["workers"]
    assert workers["http://w0"] == {
        "cached_before": 16,
        "cached_after": 0,
        "mismatch": None,
    }


def test_cache_fails_when_the_prefix_survives_the_refit(on_node, monkeypatch, capsys):
    code, out = _run(monkeypatch, capsys, "cache", fake_atom(keep_cache=True))
    after = [f"http://w{i} cached_tokens 16 after the refit" for i in (0, 1)]
    assert (code, out) == (1, "FAIL cache: " + "; ".join(after))


def test_cache_names_the_probe_briefly_when_it_differs_from_fresh(
    on_node, monkeypatch, capsys
):
    code, out = _run(monkeypatch, capsys, "cache", fake_atom(drift=-1))
    differs = f"prompt 0 {cache.PROBE[:40]!r} differs from fresh"
    assert (code, out) == (1, f"FAIL cache: http://w0 {differs}; http://w1 {differs}")
