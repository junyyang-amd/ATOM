"""ATOM's OpenAI server on the RLHF runner, plus throwaway refit routes.

Same arguments as ``python -m atom.entrypoints.openai_server``. Adds:
  POST /spike/pause            pause_generation(mode="keep")
  POST /spike/update {"path"}  every safetensors tensor under path, one SHM bucket
  POST /spike/resume           resume_generation()
  GET  /spike/state            each engine's latest scheduler snapshot
Each POST answers {"ok", "s", "error"}; a refused command is ok=false.
"""

import glob
import os
import time

from atom.model_engine import arg_utils
from atom.rollout.async_engine import AsyncLLMEngine

# AsyncLLMEngine is LLMEngine with the RLHF runner, which carries the weight sync.
arg_utils.LLMEngine = AsyncLLMEngine

from atom.entrypoints.openai import api_server as srv

STATE_KEYS = (
    "role",
    "requests_running",
    "requests_waiting",
    "requests_parked_kv_load",
    "requests_finished",
    "preemptions",
    "scheduler_metrics",
)


def _timed(fn) -> dict:
    t0 = time.monotonic()
    try:
        fn()
        return {"ok": True, "s": round(time.monotonic() - t0, 3), "error": None}
    except Exception as exc:  # noqa: BLE001 - the refusal is the measurement
        return {
            "ok": False,
            "s": round(time.monotonic() - t0, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _checkpoint(path: str):
    from safetensors.torch import load_file

    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        yield from load_file(f).items()


@srv.app.post("/spike/pause")
def spike_pause():
    return _timed(lambda: srv.engine.pause_generation(timeout=120))


@srv.app.post("/spike/update")
def spike_update(body: dict):
    return _timed(
        lambda: srv.engine.load_weights(
            _checkpoint(body["path"]), bucket_size_mb=8192, mode="shm"
        )
    )


@srv.app.post("/spike/resume")
def spike_resume():
    return _timed(lambda: srv.engine.resume_generation(timeout=120))


@srv.app.get("/spike/state")
def spike_state():
    return [
        {k: snap.get(k) for k in STATE_KEYS}
        for _, snap in sorted(srv.engine.core_mgr.latest_metrics.items())
    ]


if __name__ == "__main__":
    srv.main()
