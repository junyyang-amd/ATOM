# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Refit from a trainer: each TP rank receives a chunk's tensors on the
trainer's weight group, then writes them in one ``update_weights`` call, a
sync of its own. A refused chunk is still received. The broadcast is stood
in for."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from test_collective_rpc_transport import AsyncIOProcManager, _mgr
from test_kv_change import _running, _wired
from test_rollout_update_from_disk import OneRank, _runner, ext, spy_require_full

from atom.model_engine.collective_rpc import RpcPayload, WeightReloadRefused
from atom.model_engine.engine_utility import kv_change

GROUP = "refit"
PROJ = torch.full((2, 3), 0.5)
EMBED = torch.arange(12, dtype=torch.float32).reshape(4, 3)
BOTH = {"proj.weight": PROJ, "embed.weight": EMBED}


class _Trainer:
    """Rank 0 of the weight group *runner* joined: broadcasts the tensors of
    ``sent``, in order, into each receive."""

    def __init__(self, monkeypatch, runner, sent=(), fail=False):
        self.sent = list(sent)
        self.received = []
        self.fail = fail
        self.group = object()
        runner._rdma_groups = {
            GROUP: SimpleNamespace(group=self.group, timeout=timedelta(seconds=5))
        }
        monkeypatch.setattr(ext, "_receive", self._receive)

    def _receive(self, tensor, group, timeout):
        assert group is self.group and timeout == timedelta(seconds=5)
        if self.fail:
            raise RuntimeError("the trainer is gone")
        tensor.copy_(self.sent[len(self.received)])
        self.received.append((tuple(tensor.shape), tensor.dtype))


def _chunk(tensors):
    return {
        "names": list(tensors),
        "dtypes": ["float32"] * len(tensors),
        "shapes": [list(t.shape) for t in tensors.values()],
        "group_name": GROUP,
    }


def _send(mgr, tensors, **kwargs):
    return mgr.broadcast_utility_command_sync(
        "update_weights_from_distributed", **_chunk(tensors), timeout=5, **kwargs
    )


def _engine_over(monkeypatch, sent, unavailable=None):
    """An engine over one real runner, the trainer to send *sent*."""
    sched, _ = _running(1)
    runner = _runner()
    trainer = _Trainer(monkeypatch, runner, sent)
    mgr, (engine,) = _wired([sched], [OneRank(runner, unavailable)])
    return runner, trainer, mgr, engine


# ── through the engine ──────────────────────────────────────────────────────


def test_a_chunk_is_received_whole_then_written_and_committed(monkeypatch):
    runner, trainer, mgr, engine = _engine_over(monkeypatch, BOTH.values())
    seen = spy_require_full(monkeypatch, runner)

    _send(mgr, BOTH, weight_version="v2")

    assert trainer.received == [((2, 3), torch.float32), ((4, 3), torch.float32)]
    assert seen == [False], "one update_weights call"
    assert torch.equal(runner.model.proj.weight.detach(), PROJ)
    assert torch.equal(runner.model.embed.weight.detach(), EMBED)
    assert not engine._weight_sync_fenced, "FINISH committed it"
    assert engine.scheduler.weight_version == "v2"


def test_a_chunk_while_the_weights_are_not_whole_must_rewrite_them_all(monkeypatch):
    """A chunk leaving a parameter as it was fails after writing, and the
    engine stays fenced; one rewriting every parameter takes the fence down."""
    runner, _, mgr, engine = _engine_over(monkeypatch, [PROJ, *BOTH.values()])
    seen = spy_require_full(monkeypatch, runner)
    engine._rl_weights_inconsistent = engine._weight_sync_fenced = True

    with pytest.raises(RuntimeError, match="not whole"):
        _send(mgr, {"proj.weight": PROJ})
    assert engine._weight_sync_fenced and engine._rl_weights_inconsistent

    _send(mgr, BOTH)

    assert seen == [True, True]
    assert not engine._weight_sync_fenced and not engine._rl_weights_inconsistent


def _refused_by_the_engine(monkeypatch, mgr, engine):
    """Nothing running can be preempted, so the weight change is refused."""
    monkeypatch.setattr(engine.scheduler, "_is_preemptable", lambda seq: False)
    return "refused"


def _refused_as_unreloadable(monkeypatch, mgr, engine):
    """The runners said at startup that a reload cannot rewrite this model,
    which CoreManager hears too."""
    mgr._record_ready_payload({"weight_syncs_unavailable": UNRELOADABLE})
    return UNRELOADABLE["weight_reload"]


UNRELOADABLE = {"weight_reload": "the model attends through MLA"}


@pytest.mark.parametrize(
    "refuse, unavailable",
    [(_refused_by_the_engine, {}), (_refused_as_unreloadable, UNRELOADABLE)],
)
def test_a_refused_chunk_is_still_received_and_written_nowhere(
    monkeypatch, refuse, unavailable
):
    """The trainer is already broadcasting it: refused without receiving, it
    would wait in its broadcast until the group timed out."""
    runner, trainer, mgr, engine = _engine_over(monkeypatch, BOTH.values(), unavailable)
    before = runner.model.proj.weight.detach().clone()
    reason = refuse(monkeypatch, mgr, engine)

    with pytest.raises(RuntimeError, match=reason):
        _send(mgr, BOTH, weight_version="v2")

    assert len(trainer.received) == 2, "every tensor named was received"
    assert torch.equal(runner.model.proj.weight.detach(), before)
    assert not engine._weight_sync_fenced and not engine._rl_weights_inconsistent
    assert engine.scheduler.weight_version != "v2"


# ── the runner ──────────────────────────────────────────────────────────────


def test_a_broadcast_that_fails_leaves_the_group_and_writes_nothing(monkeypatch):
    runner = _runner()
    _Trainer(monkeypatch, runner, fail=True)
    left = []
    monkeypatch.setattr(runner, "destroy_rdma_weight_group", left.append)
    before = runner.model.proj.weight.detach().clone()

    with pytest.raises(WeightReloadRefused, match="the trainer is gone"):
        runner.update_weights_from_distributed(
            ["proj.weight"], ["float32"], [[2, 3]], GROUP
        )

    assert left == [GROUP]
    assert torch.equal(runner.model.proj.weight.detach(), before)


@pytest.mark.parametrize(
    "names, dtypes, shapes, group, match",
    [
        (["proj.weight"], ["float32"], [[2, 3]], "other", "no weight-update group"),
        (["proj.weight"], [], [[2, 3]], GROUP, "differ in length"),
        (["proj.weight"], ["float33"], [[2, 3]], GROUP, "not torch dtypes"),
    ],
)
def test_a_malformed_chunk_receives_nothing(
    monkeypatch, names, dtypes, shapes, group, match
):
    runner = _runner()
    trainer = _Trainer(monkeypatch, runner)
    with pytest.raises(WeightReloadRefused, match=match):
        runner.update_weights_from_distributed(names, dtypes, shapes, group)
    assert trainer.received == []


# ── the utility command ─────────────────────────────────────────────────────


def test_the_update_is_a_weight_update():
    assert kv_change("update_weights_from_distributed", {"sync_id": None}) == "weights"


def test_collective_rpc_refuses_the_update():
    """Called directly, the update would skip the fence, preemption and
    prefix drop; reserved, the name can never be opened to it."""
    assert "update_weights_from_distributed" in AsyncIOProcManager._RESERVED_RPC_NAMES
    mgr = _mgr(1)
    with pytest.raises(ValueError, match="not open to collective_rpc"):
        mgr.collective_rpc(
            "update_weights_from_distributed", RpcPayload("x"), timeout=5
        )
    assert mgr.rpc_broadcast_mq.sent == []
