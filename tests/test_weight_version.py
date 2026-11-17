# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The engine's ``weight_version``: the scheduler holds it, a committed weight
update that names one replaces it, and every output carries it."""

import argparse
import pickle
from unittest import mock

import pytest
import test_kv_change
from conftest import MockConfig
from test_kv_change import _running, _wired

from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.collective_rpc import FINISH_WEIGHT_SYNC_CMD, RpcResult
from atom.model_engine.request import RequestOutput
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.sampling_params import SamplingParams
from tests.logprob_helpers import token_logprob


def _versions_seen(mgr, sched):
    """Record, for each command delivered to the engine, the scheduler's
    version just before and just after it ran."""
    seen = []
    send = mgr._send_control

    def recording_send(dp_rank, payload, copy=False):
        cmd = pickle.loads(payload)[1]["cmd"]
        before = sched.weight_version
        send(dp_rank, payload, copy)
        seen.append((cmd, before, sched.weight_version))

    mgr._send_control = recording_send
    return seen


def test_an_update_naming_a_version_changes_it_only_at_its_finish():
    sched, _ = _running(1)
    mgr, _ = _wired([sched])
    seen = _versions_seen(mgr, sched)

    mgr.broadcast_utility_command_sync(
        "update_weights_from_disk",
        model_path="/ckpt/v2",
        weight_version="v2",
        timeout=5,
    )

    assert seen == [
        ("update_weights_from_disk", "default", "default"),
        (FINISH_WEIGHT_SYNC_CMD, "default", "v2"),
    ]


def test_an_update_naming_no_version_keeps_the_last_one():
    sched, _ = _running(1)
    sched.weight_version = "v2"
    mgr, _ = _wired([sched])

    mgr.broadcast_utility_command_sync("update_weights", named_tensors=[], timeout=5)

    assert sched.weight_version == "v2"


def test_an_update_one_engine_fails_changes_no_engine(monkeypatch):
    """Without a FINISH nothing commits, so the engine that applied the
    update still reports the weights it serves."""
    scheds = [_running(1)[0], _running(1)[0]]
    mgr, engines = _wired(scheds)
    calls = []

    def utility_rpc(self, method, payload, timeout=300.0):
        calls.append(method)
        if method == "update_weights_from_disk" and calls.count(method) == 2:
            return [RpcResult(payload.request_id, 0, error="OSError: disk")]
        return [RpcResult(payload.request_id, 0, value=1)]

    monkeypatch.setattr(test_kv_change._Runners, "utility_rpc", utility_rpc)

    with pytest.raises(RuntimeError, match="disk"):
        mgr.broadcast_utility_command_sync(
            "update_weights_from_disk",
            model_path="/ckpt/v2",
            weight_version="v2",
            timeout=5,
        )
    assert [s.weight_version for s in scheds] == ["default", "default"]
    assert all(e._weight_sync_fenced for e in engines)


def test_get_weight_version_answers_the_scheduler_version():
    sched, _ = _running(0)
    sched.weight_version = "v7"
    mgr, _ = _wired([sched])
    replies = mgr.broadcast_utility_command_sync("get_weight_version", timeout=5)
    assert [r["result"] for r in replies] == ["v7"]


def test_postprocess_stamps_every_output(seq_factory):
    sched = Scheduler(MockConfig(num_kvcache_blocks=64, weight_version="v3"))
    bad = seq_factory(
        [1, 2, 3], sampling_params=SamplingParams(max_tokens=8, logprobs=True)
    )
    good = seq_factory(
        [1, 2, 4], sampling_params=SamplingParams(max_tokens=8, logprobs=True)
    )
    sched.add(bad)
    sched.add(good)
    sched.schedule()
    queue = mock.Mock()
    sched.postprocess(
        list(sched.running),
        ScheduledBatchOutput(
            req_ids=[bad.id, good.id],
            token_ids=[(10,), (11,)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
            # No logprob for `bad`: it leaves as a logprob_error, one of the
            # outputs built on a path of its own.
            logprobs={bad.id: [], good.id: [token_logprob(-0.5, 1)]},
        ),
        stream_output_queue=queue,
    )
    outputs = dict(queue.put_nowait.call_args.args[0])
    assert outputs[bad.id].finish_reason == "logprob_error"
    assert outputs[good.id].output_tokens == [11]
    assert {o.weight_version for o in outputs.values()} == {"v3"}


def test_an_output_built_outside_the_scheduler_has_no_version():
    """A rejection (FrontendHub, CoreManager) never saw a scheduler."""
    output = RequestOutput(request_id=1, output_tokens=[], finished=True)
    assert output.weight_version is None


@pytest.mark.parametrize(
    "argv,version", [(["--weight-version", "v1"], "v1"), ([], "default")]
)
def test_the_flag_reaches_the_engine_config(argv, version):
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    args = parser.parse_args(argv)
    kwargs = EngineArgs.from_cli_args(args)._get_engine_kwargs()
    assert kwargs["weight_version"] == version
