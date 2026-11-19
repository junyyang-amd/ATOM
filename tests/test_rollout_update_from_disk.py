# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Refit from disk: ``update_weights_from_disk`` is a weight update like any
other -- fenced, preempting, dropping the prefix index and committed by
FINISH -- that each TP rank reads from a safetensors checkpoint."""

import pytest
import safetensors.torch
import test_kv_change
import torch
from aiter_stub import atom_without_aiter
from conftest import atom_config_double
from test_collective_rpc_transport import _mgr
from test_kv_change import _count_index_drops, _running, _wired
from torch import nn

from atom.config import Config
from atom.model_engine.collective_rpc import RpcPayload, RpcResult
from atom.model_engine.engine_utility import kv_change
from atom.utils.selector import Family

with atom_without_aiter() as _load:
    RLHFModelRunner = _load("atom.rollout.model_runner_ext").RLHFModelRunner

_BASE_RUNNER = "atom.model_engine.model_runner.ModelRunner"
_RLHF_RUNNER = "atom.rollout.model_runner_ext.RLHFModelRunner"


# ── which runner serves ─────────────────────────────────────────────────────


def _selected(runner_qualname=_BASE_RUNNER, enable_rl=False, enable_rapidserve=False):
    config = Config.__new__(Config)
    config.runner_qualname = runner_qualname
    config.enable_rl = enable_rl
    config.enable_rapidserve = enable_rapidserve
    config._select_runner()
    return config.runner_qualname


def test_enable_rl_serves_the_rlhf_runner():
    assert _selected(enable_rl=True) == _RLHF_RUNNER


def test_without_enable_rl_the_runner_is_model_runner():
    assert _selected() == _BASE_RUNNER


def test_a_runner_the_user_named_is_kept():
    assert _selected("my.Runner", enable_rl=True) == "my.Runner"


def test_rapidserve_still_selects_its_runner():
    assert _selected(enable_rapidserve=True) == (
        "atom.model_engine.model_runner.RapidServeModelRunner"
    )


# ── the utility command ─────────────────────────────────────────────────────


def test_it_commits_new_weights():
    assert kv_change("update_weights_from_disk", {"sync_id": None}) == "weights"


def test_it_runs_through_the_fence_and_finish(monkeypatch):
    sched, seqs = _running(2)
    drops = _count_index_drops(monkeypatch, sched)
    mgr, (engine,) = _wired([sched])
    sent = []

    def utility_rpc(self, method, payload, timeout=300.0):
        sent.append((method, payload.args, engine._weight_sync_fenced))
        return [RpcResult(payload.request_id, 0, value=3)]

    monkeypatch.setattr(test_kv_change._Runners, "utility_rpc", utility_rpc)

    replies = mgr.broadcast_utility_command_sync(
        "update_weights_from_disk", model_path="/ckpt/v2", timeout=5
    )

    assert [(r["cmd"], r["result"]) for r in replies] == [
        ("update_weights_from_disk", 3)
    ]
    assert sent == [("update_weights_from_disk", ("/ckpt/v2",), True)]
    assert not sched.running and list(sched.waiting)[:2] == seqs
    assert drops == [1]
    assert not engine._weight_sync_fenced, "FINISH committed it"


def test_a_rank_that_fails_to_load_leaves_the_engine_fenced(monkeypatch):
    sched, _ = _running(1)
    mgr, (engine,) = _wired([sched])

    def utility_rpc(self, method, payload, timeout=300.0):
        if method != "update_weights_from_disk":
            return [RpcResult(payload.request_id, 0, value=True)]
        return [RpcResult(payload.request_id, 0, error="FileNotFoundError: /nope")]

    monkeypatch.setattr(test_kv_change._Runners, "utility_rpc", utility_rpc)

    with pytest.raises(RuntimeError, match="/nope"):
        mgr.broadcast_utility_command_sync(
            "update_weights_from_disk", model_path="/nope", timeout=5
        )
    assert engine._weight_sync_fenced and engine._rl_weights_inconsistent


def test_collective_rpc_refuses_it():
    """Called directly it would skip the fence, preemption and prefix drop."""
    mgr = _mgr(1)
    with pytest.raises(ValueError, match="not open to collective_rpc"):
        mgr.collective_rpc("update_weights_from_disk", RpcPayload("x"), timeout=5)
    assert mgr.rpc_broadcast_mq.sent == []


# ── the runner reads the checkpoint ─────────────────────────────────────────


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4, 3)
        self.proj = nn.Linear(3, 2, bias=False)


def _runner():
    runner = object.__new__(RLHFModelRunner)
    runner._init_weight_updater()
    runner.config = atom_config_double()
    runner.attn_family = Family.MHA
    runner.model = _Model()
    runner.device = torch.device("cpu")
    runner.label = "test"
    runner.rank = 0
    runner.world_size = 1
    runner.kv_cache = None
    runner._kv_cache_backup = None
    return runner


def test_the_runner_applies_a_safetensors_checkpoint(tmp_path):
    runner = _runner()
    new = {
        "embed.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "proj.weight": torch.full((2, 3), 0.5),
    }
    safetensors.torch.save_file(new, str(tmp_path / "model.safetensors"))

    assert runner.update_weights_from_disk(str(tmp_path)) == 2

    for name, param in runner.model.named_parameters():
        assert torch.equal(param.detach(), new[name]), name
