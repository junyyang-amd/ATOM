"""Spike-only timing wrapper, loaded through PYTHONPATH in every server process.

On the rank on device 0 it appends one JSON line per vision encode and per prefill
`run_model` to $SPIKE_TIMING_DIR/timing-<pid>.jsonl. Both are fenced with
torch.cuda.synchronize, so the wall times are GPU times.
"""

import importlib.abc
import importlib.util
import json
import os
import sys
import time

OUT = os.environ.get("SPIKE_TIMING_DIR")


def _log(record):
    with open(os.path.join(OUT, f"timing-{os.getpid()}.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def _patch_runtime(mod):
    import torch

    acquire = mod.VisionEmbeddingCache.acquire

    def timed_acquire(self, request_id, data, encode):
        if torch.cuda.current_device() != 0:
            return acquire(self, request_id, data, encode)

        def timed_encode(payload):
            torch.cuda.synchronize()
            start = time.time()
            values = encode(payload)
            torch.cuda.synchronize()
            end = time.time()
            _log(
                {
                    "kind": "encode",
                    "start": start,
                    "end": end,
                    "images": len(payload["embedding_spans"]),
                    "rows": values.shape[0],
                    "bytes_bf16": values.shape[0] * values.shape[1] * 2,
                    "pixels": list(payload["pixel_values"].shape),
                }
            )
            return values

        return acquire(self, request_id, data, timed_encode)

    mod.VisionEmbeddingCache.acquire = timed_acquire


def _patch_runner(mod):
    import torch

    run_model = mod.ModelRunner.run_model

    def timed_run_model(self, input_ids, batch=None):
        context = mod.get_forward_context().context
        if not context.is_prefill or torch.cuda.current_device() != 0:
            return run_model(self, input_ids, batch)
        torch.cuda.synchronize()
        start = time.time()
        out = run_model(self, input_ids, batch)
        torch.cuda.synchronize()
        _log(
            {
                "kind": "prefill",
                "start": start,
                "end": time.time(),
                "tokens": int(input_ids.shape[0]),
            }
        )
        return out

    mod.ModelRunner.run_model = timed_run_model


PATCHES = {
    "atom.model_engine.multimodal_runtime": _patch_runtime,
    "atom.model_engine.model_runner": _patch_runner,
}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        exec_module = spec.loader.exec_module

        def patched_exec(module):
            exec_module(module)
            PATCHES[name](module)

        spec.loader.exec_module = patched_exec
        return spec


if OUT:
    sys.meta_path.insert(0, _Finder())
