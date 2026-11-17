# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The RL control routes ``--enable-rl`` adds to the API server: each is one
synchronous utility command, answered as SGLang answers it."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

try:
    from fastapi.testclient import TestClient

    from atom.entrypoints.openai import api_server
except ImportError as exc:  # pragma: no cover - environment-dependent skip
    skip_if_dependency_missing(exc, "api_server import unavailable")

from atom.model_engine.arg_utils import EngineArgs


class _CoreManager:
    """Records each synchronous broadcast; raises *error* when one is set."""

    def __init__(self, error: Exception | None = None):
        self.calls = []
        self.error = error

    def broadcast_utility_command_sync(self, cmd, timeout=300.0, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.error is not None:
            raise self.error
        return [{"cmd": cmd}]


@pytest.fixture
def serve(monkeypatch):
    """Register the RL routes for *enable_rl* on the module's app, served by a
    recording core manager; the app's routes are put back afterwards."""
    routes = list(api_server.app.router.routes)
    engine_routes = set(api_server._ENGINE_ROUTES)

    def start(enable_rl: bool, error: Exception | None = None):
        core_mgr = _CoreManager(error)
        monkeypatch.setattr(api_server, "engine", SimpleNamespace(core_mgr=core_mgr))
        api_server.register_rl_routes(SimpleNamespace(enable_rl=enable_rl))
        return TestClient(api_server.app), core_mgr

    yield start
    api_server.app.router.routes[:] = routes
    api_server._ENGINE_ROUTES.clear()
    api_server._ENGINE_ROUTES.update(engine_routes)
    api_server.app.openapi_schema = None


@pytest.mark.parametrize(
    "path,body,cmd,kwargs",
    [
        ("/pause_generation", {}, "pause_generation", {}),
        ("/pause_generation", {"mode": "keep"}, "pause_generation", {"mode": "keep"}),
        ("/continue_generation", {}, "resume_generation", {}),
        ("/flush_cache", None, "clear_kv_cache", {}),
        (
            "/update_weights_from_disk",
            {
                "model_path": "/ckpt/v2",
                "weight_version": "v2",
                "flush_cache": True,
                "load_format": "auto",
            },
            "update_weights_from_disk",
            {"model_path": "/ckpt/v2", "weight_version": "v2", "flush_cache": True},
        ),
    ],
)
def test_each_route_sends_its_utility_command(serve, path, body, cmd, kwargs):
    client, core_mgr = serve(enable_rl=True)
    reply = client.post(path, json=body) if body is not None else client.post(path)
    assert reply.status_code == 200
    assert reply.json() == {"success": True, "message": f"{cmd} succeeded"}
    assert core_mgr.calls == [(cmd, kwargs)]


def test_a_command_the_engine_refuses_answers_400(serve):
    client, core_mgr = serve(
        enable_rl=True,
        error=RuntimeError("unsupported pause mode 'abort'; only 'keep'"),
    )
    reply = client.post("/pause_generation", json={"mode": "abort"})
    assert reply.status_code == 400
    assert reply.json()["success"] is False
    assert "only 'keep'" in reply.json()["message"]
    assert core_mgr.calls == [("pause_generation", {"mode": "abort"})]


@pytest.mark.parametrize(
    "body", [{}, {"weight_version": "v2"}, {"model_path": ""}, [1], "x"]
)
def test_an_update_without_a_model_path_is_refused_before_the_engine(serve, body):
    client, core_mgr = serve(enable_rl=True)
    reply = client.post("/update_weights_from_disk", json=body)
    assert reply.status_code == 400 and reply.json()["success"] is False
    assert core_mgr.calls == []


def test_a_body_that_is_not_json_is_refused(serve):
    client, core_mgr = serve(enable_rl=True)
    reply = client.post("/pause_generation", content=b"{not json")
    assert reply.status_code == 400 and reply.json()["success"] is False
    assert core_mgr.calls == []


def test_without_enable_rl_the_routes_do_not_exist(serve):
    client, core_mgr = serve(enable_rl=False)
    for path in api_server._RL_ROUTES:
        assert client.post(path, json={}).status_code == 404
    assert not api_server._ENGINE_ROUTES & set(api_server._RL_ROUTES)
    assert core_mgr.calls == []


def test_a_frontend_worker_relays_the_routes_to_the_hub(serve):
    """Only the process that owns the engine can send utility commands."""
    serve(enable_rl=True)
    assert set(api_server._RL_ROUTES) <= api_server._ENGINE_ROUTES


@pytest.mark.parametrize("argv,enabled", [(["--enable-rl"], True), ([], False)])
def test_the_flag_reaches_the_engine_config(argv, enabled):
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    args = parser.parse_args(argv)
    assert args.enable_rl is enabled
    assert EngineArgs.from_cli_args(args)._get_engine_kwargs()["enable_rl"] is enabled
