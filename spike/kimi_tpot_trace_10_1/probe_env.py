# Does a spawned child record GPU events when its parent initialised HIP first?
# rocprofiler-register setenv()s ROCPROFILER_REGISTER_LIBRARY in the parent;
# the child inherits it. Modes: fresh, spawn, spawn_unset (unset before spawn).
# Run in the image with GPU devices: python3 probe_env.py <mode>
import collections
import json
import os
import sys
import tempfile

VAR = "ROCPROFILER_REGISTER_LIBRARY"


def probe(tag):
    import torch
    from torch.profiler import ProfilerActivity, profile

    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            x @ x
        torch.cuda.synchronize()
    out = tempfile.mktemp(suffix=".json")
    p.export_chrome_trace(out)
    cats = collections.Counter(e.get("cat") for e in json.load(open(out))["traceEvents"])
    print(tag, "kernel", cats["kernel"], "cuda_runtime", cats["cuda_runtime"],
          f"{VAR}={os.environ.get(VAR)}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fresh":
        probe("fresh")
    else:
        import torch
        import torch.multiprocessing as mp

        torch.zeros(1, device="cuda")  # parent initialises HIP, as ATOM's EngineCore does
        probe("parent")
        if mode == "spawn_unset":
            os.unsetenv(VAR)  # set by C setenv, so absent from os.environ
        c = mp.get_context("spawn").Process(target=probe, args=(mode + "-child",))
        c.start()
        c.join()
