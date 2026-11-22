# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A refit under P/D: P stamps each handoff with its ``weight_version``, and D
decodes on the received KV only when the stamp is its own version; otherwise
it frees the KV and prefills the request itself. A P/D engine updates only
while paused, to a new version, and resumes only on a versioned one."""

import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import test_pd_admission
from conftest import MockConfig
from test_kv_change import _engine, _run, _Runners, _running, _wired
from test_pd_admission import receive_success

from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.model_engine.engine_utility import (
    PAUSE_GENERATION_CMD,
    RESUME_GENERATION_CMD,
)
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import SequenceStatus
from atom.sampling_params import SamplingParams

UPDATE = "update_weights_from_disk"

# A Mooncake P/D consumer scheduler, as the admission tests build it.
build = test_pd_admission.build
mooncake_module = test_pd_admission.mooncake_module


def _stamped(request, stamp):
    seq = request()
    if stamp is not None:
        seq.kv_transfer_params["weight_version"] = stamp
    return seq


def _parked(sched, request, stamp):
    """A request D has reserved blocks for and parked on its receive."""
    seq = _stamped(request, stamp)
    sched.add(seq)
    sched.schedule()
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    return seq


def _count_releases(monkeypatch, sched):
    """The PD admission releases that returned a credit."""
    released = []
    release = sched.pd_admission.release

    def counting(seq):
        if str(seq.id) in sched.pd_admission.pending:
            released.append(seq.id)
        release(seq)

    monkeypatch.setattr(sched.pd_admission, "release", counting)
    return released


# ── D: the stamp decides between first decode and a local prefill ──────────


@pytest.mark.parametrize("stamp, served", [("v1", "v1"), (None, "v2")])
def test_a_matching_or_absent_stamp_decodes_on_the_received_kv(build, stamp, served):
    sched, request = build()
    sched.weight_version = served
    seq = _parked(sched, request, stamp)

    sched._update_from_kv_xfer_finished(receive_success(seq))
    batch, _ = sched.schedule()

    assert batch.req_ids == [seq.id] and batch.total_seqs_num_prefill == 0
    assert list(seq.output_tokens) == [42], "P's first token, injected"
    assert sched._num_parked_remote_kv == 0


def test_a_stale_stamp_frees_the_kv_and_prefills_on_d(build, monkeypatch, caplog):
    sched, request = build()
    sched.weight_version = "v1"
    seq = _parked(sched, request, "v1")
    seq.num_cached_tokens = 4  # a prefix hit D claimed before its refit
    sched.weight_version = "v2"  # D's refit
    released = _count_releases(monkeypatch, sched)
    freed = []
    deallocate = sched.block_manager.deallocate
    monkeypatch.setattr(
        sched.block_manager,
        "deallocate",
        lambda s: (freed.append((s.id, list(s.block_table))), deallocate(s)),
    )
    finished = []
    request_finished = sched.kv_connector.request_finished
    monkeypatch.setattr(
        sched.kv_connector,
        "request_finished",
        lambda s: (finished.append(s.id), request_finished(s)),
    )

    sched._update_from_kv_xfer_finished(receive_success(seq))
    with caplog.at_level(logging.INFO, logger="atom"):
        batch, _ = sched.schedule()

    assert freed == [(seq.id, [0, 1])], "the whole table, once"
    assert batch.req_ids == [seq.id] and batch.total_seqs_num_prefill == 1
    assert list(seq.output_tokens) == [], "D samples the first token itself"
    assert seq.num_cached_tokens == 0
    assert sched._num_parked_remote_kv == 0
    assert released == [seq.id] and not sched.pd_admission.pending
    assert not batch.connector_meta_output.reqs_to_recv, "nothing received again"
    (log,) = [
        r
        for r in caplog.records
        if r.getMessage().startswith("recompute stale remote KV:")
    ]
    assert log.levelno == logging.INFO
    assert f"request {seq.id} " in log.getMessage()
    assert "'v1'" in log.getMessage() and "'v2'" in log.getMessage()

    # Its finish ends the connector's and the admission's bookkeeping, and
    # neither is released a second time.
    assert sched.abort_request(seq.id)
    assert sched.postprocess(
        [], ScheduledBatchOutput([], [], None, None, None), batch=batch
    ) == [seq]
    assert finished == [seq.id]
    assert released == [seq.id]


def test_a_stale_stamp_waits_for_its_transfer_to_finish(build):
    sched, request = build()
    seq = _parked(sched, request, "v0")

    batch, _ = sched.schedule()

    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert seq.block_table and sched._num_parked_remote_kv == 1
    assert seq.id not in batch.req_ids


def test_a_receive_failing_across_a_refit_prefills_on_d(build, monkeypatch):
    """P's send times out across D's refit and the failure arrives before
    D's version changes; D resumes on new weights and prefills the request
    rather than abort it."""
    sched, request = build()
    sched.weight_version = "v1"
    seq = _parked(sched, request, "v1")
    released = _count_releases(monkeypatch, sched)

    sched._update_from_kv_xfer_finished(KVConnectorOutput(failed_recving={seq.id}))
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS, "not aborted"
    sched.weight_version = "v2"  # D's refit commits
    batch, _ = sched.schedule()

    assert batch.req_ids == [seq.id] and batch.total_seqs_num_prefill == 1
    assert seq.num_cached_tokens == 0 and list(seq.output_tokens) == []
    assert not sched.take_rejected() and not sched.failed_recving_kv_req_ids
    assert sched._num_parked_remote_kv == 0
    assert released == [seq.id] and not sched.pd_admission.pending


def test_a_failed_receive_on_the_current_version_is_aborted(build, monkeypatch):
    sched, request = build()
    sched.weight_version = "v1"
    seq = _parked(sched, request, "v1")
    free = sched.block_manager.kv.num_free + len(seq.block_table)
    released = _count_releases(monkeypatch, sched)

    sched._update_from_kv_xfer_finished(KVConnectorOutput(failed_recving={seq.id}))
    batch, _ = sched.schedule()

    assert seq.id not in batch.req_ids
    assert sched.take_rejected() == [seq] and seq.leave_reason == "aborted"
    assert seq not in sched.waiting and not sched.failed_recving_kv_req_ids
    assert sched._num_parked_remote_kv == 0
    assert sched.block_manager.kv.num_free == free
    assert released == [seq.id] and not sched.pd_admission.pending


# ── P: the stamp ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("enable_rl", [True, False])
def test_p_stamps_its_handoff_only_with_enable_rl(
    monkeypatch, mooncake_module, seq_factory, enable_rl
):
    config = MockConfig(
        num_kvcache_blocks=64,
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
        kv_transfer_config={"kv_connector": "mooncake", "kv_role": "kv_producer"},
        weight_version="v3",
        enable_rl=enable_rl,
    )
    connector = mooncake_module.MooncakeConnectorScheduler(config)
    monkeypatch.setattr(
        "atom.utils.forward_context.get_kvconnector", lambda *args: connector
    )
    sched = Scheduler(config)
    seq = seq_factory(list(range(8)), sampling_params=SamplingParams(max_tokens=1))
    seq.kv_transfer_params = {"do_remote_decode": True, "transfer_id": "t"}
    sched.add(seq)
    sched.schedule()
    queue = mock.Mock()

    sched.postprocess(
        list(sched.running),
        ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(5,)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        stream_output_queue=queue,
    )

    (output,) = dict(queue.put_nowait.call_args.args[0]).values()
    handoff = output.kv_transfer_params_output
    assert handoff["first_token_id"] == 5
    assert handoff.get("weight_version") == ("v3" if enable_rl else None)


# ── D's refit with requests parked ─────────────────────────────────────────


def test_a_weight_change_preempts_with_parked_requests(build):
    sched, request = build(options={"enabled": True, "max_pending_kv_bytes": 4096})
    running = _parked(sched, request, None)
    sched._update_from_kv_xfer_finished(receive_success(running))
    sched.schedule()
    parked = _parked(sched, request, "v1")
    blocks = list(parked.block_table)
    engine = _engine(sched)
    engine._generation_paused = True

    reply, ran = _run(engine, "clear_kv_cache", {"timeout": 0.05})
    assert "wait for remote KV" in reply["error"], "a KV change still refuses"

    reply, ran = _run(engine, UPDATE, {"sync_id": None, "weight_version": "v2"})

    assert "error" not in reply and ran == [UPDATE]
    assert not sched.running and running.status == SequenceStatus.WAITING
    assert parked.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert list(parked.block_table) == blocks and sched._num_parked_remote_kv == 1


def test_a_pd_weight_change_does_not_wait_for_transfers(build):
    sched, _ = build()
    engine = _engine(sched)
    engine._generation_paused = True
    engine.has_pending_kv_work = lambda: True
    engine._advance_idle_kv_transfer = lambda dispatch_new=True: None

    reply, ran = _run(engine, "clear_kv_cache", {"timeout": 0.05})
    assert "still in flight" in reply["error"], "a KV change still waits"

    reply, ran = _run(engine, UPDATE, {"sync_id": None, "weight_version": "v2"})
    assert "error" not in reply and ran == [UPDATE]


def test_an_offload_engine_still_waits_and_refuses_parked_requests():
    sched, seqs = _running(1)
    sched.kv_connector = SimpleNamespace(is_offload=True, drop_cached_kv=lambda: True)
    engine = _engine(sched)
    sched._num_parked_remote_kv = 1

    reply, ran = _run(engine, UPDATE, {"sync_id": None, "timeout": 0.05})
    assert "wait for remote KV" in reply["error"] and ran == []

    sched._num_parked_remote_kv = 0
    engine.has_pending_kv_work = lambda: True
    engine._advance_idle_kv_transfer = lambda dispatch_new=True: None
    reply, ran = _run(engine, UPDATE, {"sync_id": None, "timeout": 0.05})
    assert "still in flight" in reply["error"] and ran == []
    assert list(sched.running) == seqs


# ── the P/D update and resume refusals ─────────────────────────────────────


def test_an_unpaused_pd_update_is_refused_before_loading(build):
    sched, _ = build()

    reply, ran = _run(_engine(sched), UPDATE, {"sync_id": None, "weight_version": "v2"})

    assert reply["error"].startswith("refused") and "paused" in reply["error"]
    assert ran == []


def test_a_pd_update_to_the_version_it_serves_is_refused(build):
    sched, _ = build()
    sched.weight_version = "v2"
    engine = _engine(sched)
    engine._generation_paused = True

    reply, ran = _run(engine, UPDATE, {"sync_id": None, "weight_version": "v2"})

    assert reply["error"].startswith("refused") and "'v2'" in reply["error"]
    assert ran == []


def test_an_engine_without_pd_takes_an_unpaused_same_version_update():
    sched, _ = _running(1)
    sched.weight_version = "v2"

    reply, ran = _run(_engine(sched), UPDATE, {"sync_id": None, "weight_version": "v2"})

    assert "error" not in reply and ran == [UPDATE]


def test_a_versionless_pd_update_holds_the_pause_until_a_version(build):
    sched, _ = build()
    mgr, (engine,) = _wired([sched])
    mgr.broadcast_utility_command_sync(PAUSE_GENERATION_CMD, timeout=5)

    mgr.broadcast_utility_command_sync(UPDATE, model_path="/ckpt/a", timeout=5)
    with pytest.raises(RuntimeError, match="weight_version"):
        mgr.broadcast_utility_command_sync(RESUME_GENERATION_CMD, timeout=5)
    assert engine._generation_paused

    mgr.broadcast_utility_command_sync(
        UPDATE, model_path="/ckpt/b", weight_version="v2", timeout=5
    )
    mgr.broadcast_utility_command_sync(RESUME_GENERATION_CMD, timeout=5)

    assert sched.weight_version == "v2" and not engine._generation_paused


def test_a_versionless_pd_update_refused_before_writing_leaves_resume_allowed(build):
    """Every rank refused it before writing, so the weights are still the
    version the engine serves."""
    sched, _ = build()
    runners = _Runners(unasked=True, refusal="cannot reload", refusing={0})
    mgr, (engine,) = _wired([sched], [runners])
    mgr.broadcast_utility_command_sync(PAUSE_GENERATION_CMD, timeout=5)

    with pytest.raises(RuntimeError, match="cannot reload"):
        mgr.broadcast_utility_command_sync(UPDATE, model_path="/ckpt/a", timeout=5)
    mgr.broadcast_utility_command_sync(RESUME_GENERATION_CMD, timeout=5)

    assert not engine._generation_paused


def test_a_versionless_update_leaves_an_engine_without_pd_resumable():
    sched, _ = _running(0)
    mgr, (engine,) = _wired([sched])
    mgr.broadcast_utility_command_sync(PAUSE_GENERATION_CMD, timeout=5)
    mgr.broadcast_utility_command_sync(UPDATE, model_path="/ckpt/a", timeout=5)

    mgr.broadcast_utility_command_sync(RESUME_GENERATION_CMD, timeout=5)

    assert not engine._generation_paused
