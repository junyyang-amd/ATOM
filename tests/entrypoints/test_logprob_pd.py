# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Logprobs under P/D transfer: P samples the first token and publishes its
record with the handoff; D injects that token with its record ahead of the
tokens it decodes, so the request reads as one a single server served."""

import json
from types import SimpleNamespace
from unittest import mock

import pytest
import test_pd_admission
from conftest import MockConfig
from test_pd_admission import receive_success

from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.sampling_params import SamplingParams
from tests.logprob_helpers import token_logprob

build = test_pd_admission.build
mooncake_module = test_pd_admission.mooncake_module

P_RECORD = token_logprob(-0.25, 1, [(5, -0.25), (6, -1.5)])
D_RECORD = token_logprob(-0.75, 2, [(9, -0.5), (10, -0.75)])


def _step(sched, seq, token, record):
    queue = mock.Mock()
    sched.postprocess(
        list(sched.running),
        ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(token,)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
            logprobs={seq.id: [record]} if record is not None else None,
        ),
        stream_output_queue=queue,
    )
    return dict(queue.put_nowait.call_args.args[0])[seq.id]


def _producer_handoff(monkeypatch, mooncake_module, seq_factory, record):
    config = MockConfig(
        num_kvcache_blocks=64,
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
        kv_transfer_config={"kv_connector": "mooncake", "kv_role": "kv_producer"},
    )
    connector = mooncake_module.MooncakeConnectorScheduler(config)
    monkeypatch.setattr(
        "atom.utils.forward_context.get_kvconnector", lambda *args: connector
    )
    sched = Scheduler(config)
    params = SamplingParams(
        max_tokens=1, logprobs=record is not None, top_logprobs=2 * bool(record)
    )
    seq = seq_factory(list(range(8)), sampling_params=params)
    seq.kv_transfer_params = {"do_remote_decode": True, "transfer_id": "t"}
    sched.add(seq)
    sched.schedule()
    output = _step(sched, seq, 5, record)
    # The router relays the handoff through P's JSON response.
    return output, json.loads(json.dumps(output.kv_transfer_params_output))


def test_p_publishes_its_first_token_record(monkeypatch, mooncake_module, seq_factory):
    output, handoff = _producer_handoff(
        monkeypatch, mooncake_module, seq_factory, P_RECORD
    )
    assert output.logprobs == [P_RECORD]
    assert handoff["first_token_id"] == 5
    assert handoff["first_token_logprob"] == list(P_RECORD)


def test_p_without_logprobs_publishes_no_record(
    monkeypatch, mooncake_module, seq_factory
):
    _, handoff = _producer_handoff(monkeypatch, mooncake_module, seq_factory, None)
    assert "first_token_logprob" not in handoff


def _consumer(build, handoff):
    sched, request = build()
    seq = request()
    seq.kv_transfer_params.update(handoff)
    seq.return_logprobs, seq.top_logprobs = True, 2
    sched.add(seq)
    sched.schedule()
    sched._update_from_kv_xfer_finished(receive_success(seq))
    batch, _ = sched.schedule()
    assert batch.req_ids == [seq.id] and batch.total_seqs_num_prefill == 0
    return sched, seq


def test_d_answers_ps_record_then_its_own(
    build, monkeypatch, mooncake_module, seq_factory
):
    _, handoff = _producer_handoff(monkeypatch, mooncake_module, seq_factory, P_RECORD)
    handoff = {k: handoff[k] for k in ("first_token_id", "first_token_logprob")}
    sched, seq = _consumer(build, handoff)

    first = _step(sched, seq, 10, D_RECORD)
    second = _step(sched, seq, 11, D_RECORD)

    assert first.finish_reason is None
    assert first.output_tokens == [5, 10]
    assert first.logprobs == [P_RECORD, D_RECORD]
    assert second.output_tokens == [11] and second.logprobs == [D_RECORD]
    assert list(seq.logprobs) == [-0.25, -0.75, -0.75]


@pytest.mark.parametrize(
    "record",
    [{}, {"first_token_logprob": 5}, {"first_token_logprob": []}],
    ids=["missing", "scalar", "narrower"],
)
def test_d_without_ps_record_ends_only_that_request(build, record):
    sched, seq = _consumer(build, {"first_token_id": 5, **record})

    output = _step(sched, seq, 10, D_RECORD)

    assert output.finish_reason == "logprob_error"


@pytest.mark.parametrize(
    "config, kv_transfer, refused",
    [
        ({"kv_transfer_config": {"kv_role": "kv_producer"}}, None, None),
        ({"kv_transfer_config": {}}, {"do_remote_decode": True}, None),
        ({"kv_transfer_config": {}}, {"do_remote_prefill": True}, None),
        ({"kv_transfer_config": {}}, {"chunked_transfer": True}, "chunked P/D"),
        ({"enable_rapidserve": True}, None, "RapidServe"),
    ],
)
def test_a_pd_engine_admits_logprobs(config, kv_transfer, refused):
    params = SamplingParams(logprobs=True, top_logprobs=2)

    def validate(**kwargs):
        params.validate_logprob_options(
            SimpleNamespace(**config), kv_transfer=kv_transfer, **kwargs
        )

    if refused:
        with pytest.raises(ValueError, match=refused):
            validate()
        return
    validate()
    with pytest.raises(ValueError, match="text input only"):
        validate(multimodal=True)
