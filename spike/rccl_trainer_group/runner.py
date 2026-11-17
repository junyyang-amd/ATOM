"""Engine side of the spike: each ATOM TP rank joins a trainer's RCCL group.

`SpikeRunner` is `RLHFModelRunner` plus three operations a launcher reaches
through `collective_rpc`. The public RPC path runs only an allowlist, so they
ride on `configure_hidden_states`, which is on it: a call whose
`mooncake_config` carries `spike_op` runs that operation, anything else goes to
the real method. Throwaway; never lands.
"""

import datetime
import inspect
import time

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    _new_process_group_helper,
    _world,
    rendezvous,
)

from atom.rollout.model_runner_ext import RLHFModelRunner

_PG_OPTIONS_KW = (
    "backend_options"
    if "backend_options" in inspect.signature(_new_process_group_helper).parameters
    else "pg_options"
)


def new_group(master_address, master_port, rank, world_size, group_name, timeout):
    """Join `group_name` as `rank`, the way SGLang's engines and SMG's trainer
    (`examples/rl/refit_from_trainer.py: _join_trainer_group`) both build it."""
    delta = datetime.timedelta(seconds=timeout)
    store, rank, world_size = next(
        rendezvous(
            f"tcp://{master_address}:{master_port}", rank, world_size, timeout=delta
        )
    )
    store.set_timeout(delta)
    store = PrefixStore(group_name, store)
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        Backend("nccl"),
        store,
        group_name=group_name,
        timeout=delta,
        **{_PG_OPTIONS_KW: None},
    )
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg


class SpikeRunner(RLHFModelRunner):
    _spike_pg = None

    def configure_hidden_states(self, aux_layer_ids, mooncake_config):
        op = (
            mooncake_config.get("spike_op")
            if isinstance(mooncake_config, dict)
            else None
        )
        if op is None:
            return super().configure_hidden_states(aux_layer_ids, mooncake_config)
        return getattr(self, f"_spike_{op}")(**mooncake_config["kwargs"])

    def _spike_join(
        self, master_address, master_port, rank_offset, world_size, group_name, timeout
    ):
        t0 = time.perf_counter()
        rank = rank_offset + self.rank
        self._spike_pg = new_group(
            master_address, master_port, rank, world_size, group_name, timeout
        )
        return {
            "rank": rank,
            "device": str(self.device),
            "seconds": time.perf_counter() - t0,
        }

    def _spike_receive(self, names, dtypes, shapes):
        """Receive one chunk by broadcast(src=0), then apply it."""
        t0 = time.perf_counter()
        tensors = []
        for dtype, shape in zip(dtypes, shapes):
            t = torch.empty(shape, dtype=getattr(torch, dtype), device=self.device)
            dist.broadcast(t, src=0, group=self._spike_pg)
            tensors.append(t)
        torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        # The launcher clears the KV pool through the clear_kv_cache utility
        # once every chunk is in, which also drops the prefix-cache index.
        updated = self.update_weights(list(zip(names, tensors)), clear_kv_cache=False)
        torch.cuda.synchronize(self.device)
        return {
            "recv_s": t1 - t0,
            "apply_s": time.perf_counter() - t1,
            "updated": updated,
            "bytes": sum(t.numel() * t.element_size() for t in tensors),
        }

    def _spike_destroy(self):
        t0 = time.perf_counter()
        dist.destroy_process_group(self._spike_pg)
        self._spike_pg = None
        return {"seconds": time.perf_counter() - t0}
