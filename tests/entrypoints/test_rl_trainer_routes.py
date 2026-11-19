# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The trainer weight-update group routes ``--enable-rl`` adds: joining and
leaving the group run the runners' weight-group methods through
``collective_rpc`` under SGLang's field names, an update is one synchronous
utility command, and a refused update is answered 400 once its tensors are
received."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

try:
    from fastapi.testclient import TestClient

    from atom.entrypoints.openai import api_server
except ImportError as exc:  # pragma: no cover - environment-dependent skip
    skip_if_dependency_missing(exc, "api_server import unavailable")

from test_kv_change import _Runners, _running, _wired

from atom.model_engine.collective_rpc import RpcResult
from atom.model_engine.engine_utility import EngineUtilityHandler

ROUTES = (
    "/init_weights_update_group",
    "/update_weights_from_distributed",
    "/destroy_weights_update_group",
)
INIT = {
    "master_address": "10.0.0.1",
    "master_port": 29500,
    "rank_offset": 1,
    "world_size": 3,
    "group_name": "refit",
    "backend": "nccl",
}
UPDATE = {
    "names": ["proj.weight"],
    "dtypes": ["bfloat16"],
    "shapes": [[2, 3]],
    "group_name": "refit",
    "weight_version": "v2",
    "flush_cache": True,
}


class _CoreManager:
    """Records each synchronous broadcast and collective call; the ranks in
    *failing* fail the latter."""

    def __init__(self, failing=()):
        self.calls = []
        self.failing = set(failing)

    def broadcast_utility_command_sync(self, cmd, timeout=300.0, **kwargs):
        self.calls.append((cmd, kwargs))
        return [{"cmd": cmd}]

    def collective_rpc(
        self, method, args=(), kwargs=None, barrier=False, timeout=300.0
    ):
        self.calls.append((method, kwargs))
        return [
            RpcResult(
                "r", rank, error="RuntimeError: no" if rank in self.failing else None
            )
            for rank in range(2)
        ]


@pytest.fixture
def serve(monkeypatch):
    """Register the RL routes for *enable_rl* on the module's app, served by
    *core_mgr*; the app's routes are put back afterwards."""
    routes = list(api_server.app.router.routes)
    engine_routes = set(api_server._ENGINE_ROUTES)

    def start(enable_rl: bool, core_mgr=None):
        core_mgr = core_mgr or _CoreManager()
        monkeypatch.setattr(api_server, "engine", SimpleNamespace(core_mgr=core_mgr))
        api_server.register_rl_routes(SimpleNamespace(enable_rl=enable_rl))
        return TestClient(api_server.app), core_mgr

    yield start
    api_server.app.router.routes[:] = routes
    api_server._ENGINE_ROUTES.clear()
    api_server._ENGINE_ROUTES.update(engine_routes)
    api_server.app.openapi_schema = None


@pytest.mark.parametrize(
    "path,body,sent",
    [
        (
            "/init_weights_update_group",
            INIT,
            (
                "init_rdma_weight_group",
                {
                    "master_addr": "10.0.0.1",
                    "master_port": 29500,
                    "base_rank": 1,
                    "world_size": 3,
                    "group_name": "refit",
                },
            ),
        ),
        (
            "/update_weights_from_distributed",
            UPDATE,
            ("update_weights_from_distributed", UPDATE),
        ),
        (
            "/destroy_weights_update_group",
            {"group_name": "refit"},
            ("destroy_rdma_weight_group", {"group_name": "refit"}),
        ),
    ],
)
def test_each_route_sends_its_command_under_its_names(serve, path, body, sent):
    client, core_mgr = serve(enable_rl=True)
    reply = client.post(path, json={**body, "unknown": 1})
    assert reply.status_code == 200
    assert reply.json() == {"success": True, "message": f"{path[1:]} succeeded"}
    assert core_mgr.calls == [sent]


def test_a_group_over_another_backend_is_refused(serve):
    client, core_mgr = serve(enable_rl=True)
    reply = client.post("/init_weights_update_group", json={**INIT, "backend": "gloo"})
    assert reply.status_code == 400 and "'gloo'" in reply.json()["message"]
    assert core_mgr.calls == []


def test_a_join_a_rank_failed_answers_400_with_that_rank(serve):
    client, _ = serve(enable_rl=True, core_mgr=_CoreManager(failing={1}))
    reply = client.post("/init_weights_update_group", json=INIT)
    assert reply.status_code == 400
    assert reply.json() == {"success": False, "message": "TP rank 1: RuntimeError: no"}


def _refused_by_the_fence(monkeypatch, sched):
    """Nothing running can be preempted, so the weight change is refused."""
    monkeypatch.setattr(sched, "_is_preemptable", lambda seq: False)


def _refused_by_pd(monkeypatch, sched):
    """A P/D engine updates only while paused, and this one is not."""
    monkeypatch.setattr(EngineUtilityHandler, "_pd_disaggregated", lambda self: True)


@pytest.mark.parametrize(
    "refuse, reason",
    [(_refused_by_the_fence, "cannot preempt"), (_refused_by_pd, "only while paused")],
)
def test_a_refused_update_is_received_then_answered_400(
    serve, monkeypatch, refuse, reason
):
    sched, _ = _running(1)
    refuse(monkeypatch, sched)
    runners = _Runners()
    mgr, (engine,) = _wired([sched], [runners])
    client, _ = serve(enable_rl=True, core_mgr=mgr)

    reply = client.post("/update_weights_from_distributed", json=UPDATE)

    assert reply.status_code == 400 and reply.json()["success"] is False
    assert reason in reply.json()["message"]
    (method, payload), (discard, _) = runners.calls
    assert discard == "discard_failed_weight_sync"
    assert method == "update_weights_from_distributed"
    assert payload.args == (["proj.weight"], ["bfloat16"], [[2, 3]], "refit")
    assert reason in payload.kwargs["refusal"], "received to be discarded"
    assert sched.weight_version != "v2"
    assert not engine._weight_sync_fenced, "nothing was written"


def test_without_enable_rl_the_routes_do_not_exist(serve):
    client, core_mgr = serve(enable_rl=False)
    for path in ROUTES:
        assert client.post(path, json={}).status_code == 404
    assert core_mgr.calls == []


def test_a_frontend_worker_relays_the_routes_to_the_hub(serve):
    serve(enable_rl=True)
    assert set(ROUTES) <= api_server._ENGINE_ROUTES
