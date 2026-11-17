# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``/generate`` and ``/model_info`` under ``--enable-rl``: SGLang's shapes,
served by the completion path and the engine's ``weight_version``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

try:
    from fastapi.testclient import TestClient

    from atom.entrypoints.openai import api_server
except ImportError as exc:  # pragma: no cover - environment-dependent skip
    skip_if_dependency_missing(exc, "api_server import unavailable")

from atom.model_engine.request import RequestOutput


class _Engine:
    """The completion path's view of the engine: preprocess hands the
    sequence its outputs, the core manager answers utility commands."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.prompts = []
        self.sampling = []
        self.utility = []
        self.requests = {}
        self.config = SimpleNamespace(model="/ckpt/base", max_model_len=None)
        self.io_processor = self
        self.core_mgr = self

    def preprocess(self, prompt, sampling_params, stream_callback=None, **_kwargs):
        self.prompts.append(prompt)
        self.sampling.append(sampling_params)
        seq = SimpleNamespace(
            id=7, num_prompt_tokens=5, max_tokens=sampling_params.max_tokens
        )
        self.requests[seq.id] = seq
        for output in self.outputs:
            stream_callback(output)
        return seq

    def add_request(self, _seqs):
        pass

    def abort_request(self, _seq_id):
        pass

    def broadcast_utility_command_sync(self, cmd, timeout=300.0, **kwargs):
        self.utility.append(cmd)
        return [{"cmd": cmd, "result": "v2"}, {"cmd": cmd, "result": "v2"}]


def _out(tokens, finished=False, reason=None, version="v2", cached=0):
    return RequestOutput(
        request_id=7,
        output_tokens=tokens,
        finished=finished,
        finish_reason=reason,
        num_cached_tokens=cached,
        weight_version=version,
    )


@pytest.fixture
def serve(monkeypatch):
    """Register the RL routes for *enable_rl* over a fake engine; the app's
    routes are put back afterwards."""
    routes = list(api_server.app.router.routes)
    engine_routes = set(api_server._ENGINE_ROUTES)
    monkeypatch.setattr(api_server, "delivered_text", lambda ids: f"text{list(ids)}")
    monkeypatch.setattr(api_server, "record_nonstream_first_token", lambda: None)

    def start(outputs=(), enable_rl=True):
        fake = _Engine(list(outputs))
        monkeypatch.setattr(api_server, "engine", fake)
        api_server.register_rl_routes(SimpleNamespace(enable_rl=enable_rl))
        return TestClient(api_server.app), fake

    yield start
    api_server.app.router.routes[:] = routes
    api_server._ENGINE_ROUTES.clear()
    api_server._ENGINE_ROUTES.update(engine_routes)
    api_server.app.openapi_schema = None


def test_generate_answers_sglang_meta_info(serve):
    client, fake = serve(
        [_out([11, 12], version="default", cached=3), _out([13], True, "eos")]
    )
    reply = client.post("/generate", json={"text": "1+1=", "sampling_params": {}})
    assert reply.status_code == 200
    body = reply.json()
    assert body["text"] == "text[11, 12, 13]"
    assert body["output_ids"] == [11, 12, 13]
    meta = body["meta_info"]
    assert meta["id"].startswith("generate-")
    assert {k: v for k, v in meta.items() if k != "id"} == {
        "finish_reason": {"type": "stop"},
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "cached_tokens": 3,
        # The last output's: a request preempted by a refit ends on the new one.
        "weight_version": "v2",
    }
    assert fake.prompts == ["1+1="]


def test_generate_maps_sampling_params_to_the_completion_path(serve):
    client, fake = serve([_out([11], True, "max_tokens")])
    reply = client.post(
        "/generate",
        json={
            "input_ids": [1, 2, 3],
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 20,
                "stop": "\n",
                "ignore_eos": True,
            },
        },
    )
    assert reply.status_code == 200
    assert reply.json()["meta_info"]["finish_reason"] == {"type": "length", "length": 1}
    assert fake.prompts == [[1, 2, 3]]
    (params,) = fake.sampling
    assert (params.max_tokens, params.temperature, params.top_p, params.top_k) == (
        1,
        0.5,
        0.9,
        20,
    )
    assert params.stop_strings == ["\n"] and params.ignore_eos


def test_generate_defaults_are_sglangs(serve):
    client, fake = serve([_out([11], True, "eos")])
    assert client.post("/generate", json={"text": "hi"}).status_code == 200
    (params,) = fake.sampling
    assert (params.max_tokens, params.temperature, params.top_p, params.top_k) == (
        128,
        1.0,
        1.0,
        -1,
    )
    assert not params.ignore_eos


def test_an_output_with_no_version_still_renders(serve):
    """A rejection built outside the scheduler carries no version."""
    client, _ = serve([_out([], True, "unschedulable: too long", version=None)])
    reply = client.post("/generate", json={"text": "hi"})
    assert reply.status_code == 200
    meta = reply.json()["meta_info"]
    assert meta["weight_version"] is None
    assert meta["finish_reason"] == {"type": "unschedulable: too long"}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"text": "hi", "input_ids": [1]},
        {"input_ids": []},
        {"text": "hi", "stream": True},
        {"text": "hi", "sampling_params": {"max_new_tokens": 0}},
    ],
)
def test_generate_refuses_what_it_cannot_serve(serve, body):
    client, fake = serve([_out([11], True, "eos")])
    assert client.post("/generate", json=body).status_code == 400
    assert fake.prompts == []


def test_model_info_answers_dp_rank_0s_version(serve):
    client, fake = serve()
    reply = client.get("/model_info")
    assert reply.status_code == 200
    assert reply.json() == {"model_path": "/ckpt/base", "weight_version": "v2"}
    assert fake.utility == ["get_weight_version"]


def test_a_frontend_worker_relays_model_info_to_the_hub(serve):
    serve()
    assert "/model_info" in api_server._ENGINE_ROUTES
    assert "/generate" not in api_server._ENGINE_ROUTES


def test_without_enable_rl_generate_and_model_info_do_not_exist(serve):
    client, fake = serve([_out([11], True, "eos")], enable_rl=False)
    assert client.post("/generate", json={"text": "hi"}).status_code == 404
    assert client.get("/model_info").status_code == 404
    assert fake.prompts == [] and fake.utility == []
