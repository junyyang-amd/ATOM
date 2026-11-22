# SPDX-License-Identifier: MIT
"""tools/rl/accept.py's pd check: its ``--dry-run`` plan through accept.py's
loader, and its verdict on a fake 1P1D pair refitted with completions in
flight."""

import json
import sys
import threading
from pathlib import Path

import pytest
from test_rl_accept import fresh
from test_rl_accept_refit_disk import _container, _run, fake_atom, models, on_node

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "rl"))
import accept
from checks import pd, refit_disk

__all__ = ["fresh", "models", "on_node"]  # fixtures, imported for this module's tests


def test_pd_plans_a_refit_under_load_then_a_fresh_pair(models, monkeypatch, capsys):
    monkeypatch.setattr(accept.subprocess, "run", pytest.fail)
    assert accept.main(["pd", "--dry-run"]) == 0
    steps = capsys.readouterr().out.splitlines()
    runs = [_container(s) for s in steps if s.startswith("docker run")]
    assert runs == [
        ("rl-111-atom-18201", "0"),
        ("rl-111-atom-18204", "1"),
        ("rl-111-mesh-build", None),
        ("rl-111-mesh-18206", None),
        ("rl-111-atom-18209", "2"),
        ("rl-111-atom-18212", "3"),
        ("rl-111-mesh-18214", None),
    ]
    assert "Qwen3-1.7B-Base/" in steps[0] and "Qwen3-1.7B/" in steps[-3]
    assert all("--pd-disaggregation" in steps[i] for i in (3, -2))
    calls = [s.split("?")[0] for s in steps[5:8]]
    assert calls == [f"POST /v1/rl/engine/{p}" for p in refit_disk.CYCLE]
    assert steps[4].startswith("POST http://127.0.0.1:18206/v1/completions x64")


def fake_pair(stale=(), failing=None):
    """fake_atom's workers as a P/D pair whose streamed completions each send
    a token, then finish once the pair continues (prompt ``failing`` breaks)."""
    resumed = threading.Event()

    class FakePair(fake_atom(stale)):
        def start_pd_pair(self, model, extra=()):
            FakePair.last = self
            urls = [f"http://w:{len(self.held) + i}" for i in (0, 1)]
            self.held |= {u: (model, "default") for u in urls}
            return urls

        def rl_fanout(self, path, body, selector):
            assert path != "pause_generation" or not resumed.is_set()
            if path == "continue_generation":
                resumed.set()
            return super().rl_fanout(path, body, selector)

        def logs(self, name):
            assert name == "rl-111-atom-1"
            return pd.STALE_LINE * 3

        def prompt_ids(self, url, model, prompt):
            return [1]

        def equals_fresh(self, url, checkpoint, prompts, outputs):
            assert checkpoint == "/ckpt/" + refit_disk.NEW
            assert outputs == [{"output_ids": [5]}] * len(accept.PROMPTS)
            return {"equal": True, "first": None, "gap": 0.0, "margin": 0.5}

    def stream(url, body):
        yield {"choices": [{"text": "a"}]}
        resumed.wait(5)
        if failing is not None and body["prompt"] == pd.load_prompts()[failing]:
            raise OSError("reset")
        yield {"choices": [{"text": "b", "finish_reason": "length"}]}

    def http(method, url, body=None):
        if url.endswith("/model_info"):
            return 200, {"weight_version": FakePair.last.held[url[:-11]][1]}
        return 200, {"choices": [{"logprobs": {"content": [{"bytes": [120]}]}}]}

    return FakePair, stream, http


@pytest.mark.parametrize(
    "stale,failing,reason",
    [
        ((), None, None),
        (("http://w:1",), None, "http://w:1 model_info {'weight_version': 'default'}"),
        (
            (),
            3,
            "in flight ended {'completed': 63, 'failed': 1}, errors ['OSError: reset']",
        ),
    ],
)
def test_pd_passes_when_every_completion_in_flight_completes_on_v2(
    on_node, monkeypatch, capsys, stale, failing, reason
):
    harness, stream, http = fake_pair(stale, failing)
    monkeypatch.setattr(pd, "stream", stream)
    monkeypatch.setattr(pd, "http", http)
    monkeypatch.setattr(pd, "token_bytes", lambda checkpoint: ({5: b"x"}, set()))
    code, out = _run(monkeypatch, capsys, "pd", harness)
    assert (code, out) == (
        (0, "PASS pd") if reason is None else (1, f"FAIL pd: {reason}")
    )
    data = json.loads((on_node / "pd.json").read_text())["data"]
    assert data["in_flight"]["at_pause"] == {"decoding": 64}
    assert data["stale_recomputes"] == 3


def test_pd_fails_a_completion_that_finishes_with_no_token(
    on_node, monkeypatch, capsys
):
    """D aborts a request whose handoff failed; it reaches the client as a
    finish with no token."""
    harness, stream, http = fake_pair()
    aborted = pd.load_prompts()[3]

    def emptied(url, body):
        for chunk in stream(url, body):
            if body["prompt"] == aborted:
                chunk = {"choices": [c | {"text": ""} for c in chunk["choices"]]}
            yield chunk

    monkeypatch.setattr(pd, "stream", emptied)
    monkeypatch.setattr(pd, "http", http)
    monkeypatch.setattr(pd, "token_bytes", lambda checkpoint: ({5: b"x"}, set()))
    code, out = _run(monkeypatch, capsys, "pd", harness)
    assert (code, out) == (
        1,
        (
            "FAIL pd: in flight ended {'completed': 63, 'failed': 1}, errors "
            "[\"HarnessError: finished 'length' with no token\"]"
        ),
    )


def test_equals_fresh_names_the_checkpoint_as_the_model(fresh, monkeypatch):
    checkpoint, tops, _ = fresh
    tops[()] = [(b"a", -0.1)]
    named, answer = [], accept.http

    def http(method, url, body=None, timeout=None):
        named.append(body["model"])
        return answer(method, url, body)

    monkeypatch.setattr(accept, "http", http)
    accept.Harness(111).equals_fresh("u", checkpoint, ["p"], [{"output_ids": [0]}])
    assert named == [checkpoint, checkpoint]
