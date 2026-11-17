# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Refit from a trainer: each TP rank joins the trainer's weight-update group,
receives every tensor by broadcast into one ``update_weights`` call, and
leaves the group. ``torch.distributed`` is stood in for."""

import pytest
import test_kv_change
import torch
from aiter_stub import atom_without_aiter
from test_collective_rpc_transport import AsyncIOProcManager, _mgr
from test_kv_change import _running, _wired
from torch import nn

from atom.model_engine.collective_rpc import RpcPayload, RpcResult
from atom.model_engine.engine_utility import kv_change

with atom_without_aiter() as _load:
    ext = _load("atom.rollout.model_runner_ext")

GROUP = "refit"


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4, 3)
        self.proj = nn.Linear(3, 2, bias=False)


class _Dist:
    """What the runner calls in ``torch.distributed``, recorded."""

    def __init__(self, monkeypatch, sent=None):
        self.rendezvous = []
        self.helper = []
        self.destroyed = []
        self.received = []
        self.sent = sent or {}
        monkeypatch.setattr(ext, "rendezvous", self._rendezvous)
        monkeypatch.setattr(ext, "_new_process_group_helper", self._helper)
        monkeypatch.setattr(ext, "PrefixStore", lambda prefix, store: (prefix, store))
        monkeypatch.setattr(ext.dist, "broadcast", self._broadcast)
        monkeypatch.setattr(ext.dist, "destroy_process_group", self.destroyed.append)

    def _rendezvous(self, url, rank, world_size, timeout):
        self.rendezvous.append((url, rank, world_size))
        store = type("Store", (), {"set_timeout": lambda self, t: None})()
        yield store, rank, world_size

    def _helper(self, world_size, rank, ranks, backend, store, group_name, **kw):
        group = object()
        self.helper.append((world_size, rank, str(backend), store[0], group_name))
        return group, None

    def _broadcast(self, tensor, src, group):
        """Rank 0 of the group sends the next tensor of ``sent``, in order."""
        assert src == 0
        name = list(self.sent)[len(self.received)]
        self.received.append((name, tensor.dtype, tuple(tensor.shape), group))
        tensor.copy_(self.sent[name])


def _runner(rank=0):
    runner = object.__new__(ext.RLHFModelRunner)
    runner.model = _Model()
    runner.device = torch.device("cpu")
    runner.label = "test"
    runner.rank = rank
    runner.world_size = 1
    runner.kv_cache = None
    runner._kv_cache_backup = None
    runner._weights_update_groups = {}
    return runner


def _join(runner, rank_offset=1, world_size=3):
    return runner.init_weights_update_group(
        "10.0.0.1", 29500, rank_offset, world_size, GROUP, "nccl", 60.0
    )


# ── the group ───────────────────────────────────────────────────────────────


def test_rank_layout_and_group_name_reach_the_helper(monkeypatch):
    """SMG's layout: the trainer is rank 0, an engine's TP rank k joins as
    rank_offset + k, under the group's own store prefix."""
    d = _Dist(monkeypatch)

    assert _join(_runner(rank=1), rank_offset=3, world_size=5) == 4

    assert d.rendezvous == [("tcp://10.0.0.1:29500", 4, 5)]
    assert d.helper == [(5, 4, "nccl", GROUP, GROUP)]


def test_destroy_releases_the_group_and_a_second_init_works(monkeypatch):
    d = _Dist(monkeypatch)
    runner = _runner()
    _join(runner)
    first = runner._weights_update_groups[GROUP]
    with pytest.raises(ValueError, match="already exists"):
        _join(runner)

    runner.destroy_weights_update_group(GROUP)
    assert d.destroyed == [first] and runner._weights_update_groups == {}
    _join(runner)

    assert len(d.helper) == 2
    assert runner._weights_update_groups[GROUP] is not first


def test_destroying_a_group_never_joined_is_refused(monkeypatch):
    d = _Dist(monkeypatch)
    with pytest.raises(ValueError, match="no weight-update group"):
        _runner().destroy_weights_update_group(GROUP)
    assert d.destroyed == []


# ── the update ──────────────────────────────────────────────────────────────


def test_the_update_receives_in_order_into_one_update_weights(monkeypatch):
    sent = {
        "proj.weight": torch.full((2, 3), 0.5),
        "embed.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
    }
    d = _Dist(monkeypatch, sent)
    runner = _runner()
    _join(runner)
    group = runner._weights_update_groups[GROUP]
    calls = []
    update_weights = runner.update_weights

    def counted(named_tensors, *args, **kwargs):
        calls.append(1)
        return update_weights(named_tensors, *args, **kwargs)

    monkeypatch.setattr(runner, "update_weights", counted)

    updated = runner.update_weights_from_distributed(
        list(sent), ["float32", "float32"], [[2, 3], [4, 3]], GROUP
    )

    assert updated == 2 and calls == [1]
    assert d.received == [
        ("proj.weight", torch.float32, (2, 3), group),
        ("embed.weight", torch.float32, (4, 3), group),
    ]
    params = dict(runner.model.named_parameters())
    for name, tensor in sent.items():
        assert torch.equal(params[name].detach(), tensor), name


@pytest.mark.parametrize(
    "names, dtypes, shapes, group, match",
    [
        (["proj.weight"], ["float32"], [[2, 3]], "other", "no weight-update group"),
        (["proj.weight"], [], [[2, 3]], GROUP, "differ in length"),
        (["proj.weight"], ["float33"], [[2, 3]], GROUP, "not torch dtypes"),
    ],
)
def test_a_malformed_update_receives_nothing(
    monkeypatch, names, dtypes, shapes, group, match
):
    d = _Dist(monkeypatch)
    runner = _runner()
    _join(runner)
    with pytest.raises(ValueError, match=match):
        runner.update_weights_from_distributed(names, dtypes, shapes, group)
    assert d.received == []


# ── the utility commands ────────────────────────────────────────────────────


def test_the_update_is_a_weight_update():
    assert kv_change("update_weights_from_distributed", {"sync_id": None}) == "weights"
    assert kv_change("init_weights_update_group", {}) is None
    assert kv_change("destroy_weights_update_group", {}) is None


def _engine_with_runners(monkeypatch):
    sched, _ = _running(1)
    mgr, (engine,) = _wired([sched])
    sent = []

    def utility_rpc(self, method, payload, timeout=300.0):
        sent.append((method, payload.args, engine._weight_sync_fenced))
        return [
            RpcResult(payload.request_id, 0, value=2),
            RpcResult(payload.request_id, 1, value=2),
        ]

    monkeypatch.setattr(test_kv_change._Runners, "utility_rpc", utility_rpc)
    return mgr, engine, sched, sent


def test_init_and_destroy_reach_every_rank_unfenced(monkeypatch):
    mgr, engine, _, sent = _engine_with_runners(monkeypatch)
    mgr.broadcast_utility_command_sync(
        "init_weights_update_group",
        master_address="10.0.0.1",
        master_port=29500,
        rank_offset=1,
        world_size=3,
        group_name=GROUP,
        backend="nccl",
        timeout=5,
    )
    mgr.broadcast_utility_command_sync(
        "destroy_weights_update_group", group_name=GROUP, timeout=5
    )

    (init, init_args, init_fenced), destroy = sent
    assert init == "init_weights_update_group"
    assert init_args[:6] == ("10.0.0.1", 29500, 1, 3, GROUP, "nccl")
    assert 0 < init_args[6] <= 5, "the group shares the call's deadline"
    assert destroy == ("destroy_weights_update_group", (GROUP,), False)
    assert not init_fenced and not engine._weight_sync_fenced


def test_the_update_runs_through_the_fence_and_commits_its_version(monkeypatch):
    mgr, engine, sched, sent = _engine_with_runners(monkeypatch)
    mgr.broadcast_utility_command_sync(
        "update_weights_from_distributed",
        names=["proj.weight"],
        dtypes=["bfloat16"],
        shapes=[[2, 3]],
        group_name=GROUP,
        weight_version="v2",
        timeout=5,
    )

    assert sent == [
        (
            "update_weights_from_distributed",
            (["proj.weight"], ["bfloat16"], [[2, 3]], GROUP),
            True,
        )
    ]
    assert not engine._weight_sync_fenced, "FINISH committed it"
    assert sched.weight_version == "v2"


@pytest.mark.parametrize(
    "method",
    [
        "init_weights_update_group",
        "update_weights_from_distributed",
        "destroy_weights_update_group",
    ],
)
def test_collective_rpc_refuses_it(method):
    """Called directly, the update would skip the fence, preemption and
    prefix drop; reserved, the name can never be opened to it."""
    assert method in AsyncIOProcManager._RESERVED_RPC_NAMES
    mgr = _mgr(1)
    with pytest.raises(ValueError, match="not open to collective_rpc"):
        mgr.collective_rpc(method, RpcPayload("x"), timeout=5)
    assert mgr.rpc_broadcast_mq.sent == []
