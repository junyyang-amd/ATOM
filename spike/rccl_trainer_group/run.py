"""Spike: does a second RCCL group with an external trainer work beside ATOM's TP group?

    HIP_VISIBLE_DEVICES=4,5,6 PYTHONPATH=$PWD python -m spike.rccl_trainer_group.run

ATOM runs at TP2 on devices 0-1 with `SpikeRunner` (`runner.py`); this process
is the trainer on device 2. Each round: pause, every TP rank joins a world-3
group with the trainer at rank 0 (SMG's rank layout, `rank_offset` 1), the
trainer broadcasts every parameter of a checkpoint in chunks of `--chunk`
while the ranks receive and apply them through `update_weights`, the KV pool
is cleared, the group destroyed, generation resumed and one greedy request
served. Rounds alternate the checkpoint away from the one ATOM started on, so
every round changes the weights. Then a fresh engine on each broadcast
checkpoint serves the same request, and each round's output is compared with
it token for token. Throwaway; never lands.
"""

import argparse
import json
import socket
import sys
import threading
import time
import traceback

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from atom import SamplingParams
from atom.rollout.async_engine import AsyncLLMEngine
from spike.rccl_trainer_group.runner import new_group

RUNNER = "spike.rccl_trainer_group.runner.SpikeRunner"
PROMPT = "The three most important inventions of the twentieth century are"
GROUP = "spike_refit"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_engine(model, port, args):
    return AsyncLLMEngine(
        model,
        tensor_parallel_size=2,
        runner_qualname=RUNNER,
        port=port,
        cudagraph_capture_sizes=[1, 2, 4],
        max_model_len=4096,
    )


def serve_one(engine, max_tokens):
    out = engine.generate(
        [PROMPT],
        SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
    )
    return out[0]["token_ids"], out[0]["text"]


def spike_rpc(engine, op, rpc_timeout, **kwargs):
    replies = engine.collective_rpc(
        "configure_hidden_states",
        kwargs={
            "aux_layer_ids": [],
            "mooncake_config": {"spike_op": op, "kwargs": kwargs},
        },
        timeout=rpc_timeout,
    )
    bad = [f"tp{r.tp_rank}: {r.error}" for r in replies if not r.ok]
    if bad:
        raise RuntimeError(f"{op} failed: " + "; ".join(bad))
    return [r.value for r in replies]


def utility(engine, cmd, **kwargs):
    replies = engine.core_mgr.broadcast_utility_command_sync(cmd, **kwargs)
    bad = [r for r in replies if isinstance(r, dict) and r.get("error")]
    if bad:
        raise RuntimeError(f"{cmd} failed: {bad}")


def in_thread(fn, *a, **kw):
    box = {}

    def run():
        try:
            box["value"] = fn(*a, **kw)
        except BaseException as e:  # noqa: BLE001 - reported by the joiner
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def joined(t, box, timeout, what):
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"{what}: no reply within {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def one_round(engine, params, args, rnd):
    rec = {"round": rnd}
    step = "pause_generation"
    pg = None
    try:
        utility(engine, "pause_generation")

        step = "join"
        port = free_port()
        t0 = time.perf_counter()
        t, box = in_thread(
            spike_rpc,
            engine,
            "join",
            args.timeout,
            master_address="127.0.0.1",
            master_port=port,
            rank_offset=1,
            world_size=3,
            group_name=GROUP,
            timeout=args.timeout,
        )
        # The trainer's rendezvous waits for every rank, so it runs beside the
        # RPC: an RPC that fails is then reported instead of stalling the store.
        tp, boxp = in_thread(new_group, "127.0.0.1", port, 0, 3, GROUP, args.timeout)
        while t.is_alive() and tp.is_alive():
            time.sleep(0.01)
        if not t.is_alive() and "error" in box:
            raise box["error"]
        pg = joined(tp, boxp, args.timeout, "trainer join")
        rec["join"] = joined(t, box, args.timeout, "join")
        rec["join_s"] = time.perf_counter() - t0
        rec["group_formed"] = True

        step = "broadcast"
        t0 = time.perf_counter()
        chunks = []
        for i in range(0, len(params), args.chunk):
            batch = params[i : i + args.chunk]
            t, box = in_thread(
                spike_rpc,
                engine,
                "receive",
                args.timeout,
                names=[n for n, _ in batch],
                dtypes=[str(p.dtype).removeprefix("torch.") for _, p in batch],
                shapes=[list(p.shape) for _, p in batch],
            )
            for _, p in batch:
                dist.broadcast(p, src=0, group=pg)
            chunks.append(joined(t, box, args.timeout, f"receive[{i}]"))
        torch.cuda.synchronize()
        rec["receive_wall_s"] = time.perf_counter() - t0
        for tp in range(2):
            rec[f"tp{tp}_recv_s"] = sum(c[tp]["recv_s"] for c in chunks)
            rec[f"tp{tp}_apply_s"] = sum(c[tp]["apply_s"] for c in chunks)
            rec[f"tp{tp}_updated"] = sum(c[tp]["updated"] for c in chunks)
        rec["bytes"] = sum(c[0]["bytes"] for c in chunks)

        step = "clear_kv_cache"
        utility(engine, "clear_kv_cache")

        step = "destroy"
        t, box = in_thread(spike_rpc, engine, "destroy", args.timeout)
        dist.destroy_process_group(pg)
        pg = None
        rec["destroy"] = joined(t, box, args.timeout, "destroy")
        rec["destroyed"] = True

        step = "resume_generation"
        utility(engine, "resume_generation")

        step = "serve"
        rec["token_ids"], rec["text"] = serve_one(engine, args.max_tokens)
    except BaseException as e:  # noqa: BLE001 - recorded as the round's failure
        rec["failed_step"] = step
        rec["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--broadcast", default="Qwen/Qwen3-8B")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()

    torch.cuda.set_device(2)
    policies = {}
    for name in (args.broadcast, args.start):
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
        m = m.to("cuda:2").eval()
        policies[name] = [(n, p.detach().contiguous()) for n, p in m.named_parameters()]
    # Round 1 broadcasts --broadcast, round 2 --start, and so on: each round
    # changes every weight the engine holds.
    plan = [args.broadcast if r % 2 else args.start for r in range(1, args.rounds + 1)]

    engine = start_engine(args.start, args.port, args)
    start_ids, _ = serve_one(engine, args.max_tokens)
    rounds = []
    for rnd, ckpt in enumerate(plan, 1):
        rec = one_round(engine, policies[ckpt], args, rnd)
        rec["checkpoint"] = ckpt
        rounds.append(rec)
        print(
            "ROUND",
            json.dumps({k: v for k, v in rec.items() if k != "text"}),
            flush=True,
        )
        if "error" in rec:
            break
    engine.close()

    fresh = {}
    for i, ckpt in enumerate(sorted(set(plan))):
        eng = start_engine(ckpt, args.port + 1 + i, args)
        fresh[ckpt] = serve_one(eng, args.max_tokens)[0]
        eng.close()

    print("\nRESULT")
    print(
        f"start={args.start} output differs from fresh {args.broadcast}: "
        f"{start_ids != fresh.get(args.broadcast)}"
    )
    ok = True
    for rec in rounds:
        if "error" in rec:
            ok = False
            print(
                f"round {rec['round']}: FAILED at {rec['failed_step']}: {rec['error']}"
            )
            continue
        same = rec["token_ids"] == fresh[rec["checkpoint"]]
        ok &= same
        print(
            f"round {rec['round']} ({rec['checkpoint']}): group formed={rec['group_formed']} "
            f"join={rec['join_s']:.2f}s receive wall={rec['receive_wall_s']:.2f}s "
            f"(tp0 recv {rec['tp0_recv_s']:.2f}s apply {rec['tp0_apply_s']:.2f}s, "
            f"tp1 recv {rec['tp1_recv_s']:.2f}s apply {rec['tp1_apply_s']:.2f}s, "
            f"{rec['bytes'] / 2**30:.2f} GiB, updated {rec['tp0_updated']}/{rec['tp1_updated']}) "
            f"destroyed={rec['destroyed']} output==fresh: {same}"
        )
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
