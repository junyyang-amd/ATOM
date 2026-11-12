"""Spike: transfer time of a bf16 device tensor from an encoder to prefill.

Two processes on two GPUs. The producer (encoder side) holds a 64 MiB bf16
tensor on its GPU; the consumer (prefill side) receives a prefix of it into
its GPU. Every cell is timed on one clock, from the producer's device tensor
to the consumer's device tensor:

  mooncake  consumer reads producer GPU memory with Mooncake TransferEngine
            (RDMA, as the KV connector configures it); timed in the consumer.
            The consumer prints its HCA's RDMA bytes received per byte
            read: 1.0 means the data crossed the NIC, not HIP IPC.
  http      consumer GETs the bytes; the producer copies them device to host
            and serves them; the consumer copies host to device; timed in the
            consumer.
  inline    producer copies device to host, base64-encodes into a JSON chat
            body and POSTs it to ATOMesh, which relays it to the consumer;
            the consumer decodes, copies host to device and answers; timed
            in the producer. inline-direct is the same POST straight to the
            consumer, to separate ATOMesh's share from the encoding's.

Usage (inside the recipe container; see run.sh):
  bench.py producer --gpu 4 --ib rocep137s0 --ip IP --port P
  bench.py consumer --gpu 5 --ib rocep153s0 --ip IP --port C \
      --producer http://IP:P --mesh http://IP:M --reps 10
The consumer prints one JSON line per cell and a table.
"""

import argparse
import asyncio
import base64
import json
import logging
import statistics
import time
import warnings

import aiohttp
import torch
from aiohttp import web

MiB = 1 << 20
SIZES_MB = (1, 4, 16, 64)
MAX_BYTES = max(SIZES_MB) * MiB
BODY_LIMIT = 512 * MiB


def checksum(t):
    return int(t.view(torch.int16).to(torch.int64).sum().item())


def make_engine(ip, ib):
    from mooncake.engine import TransferEngine

    te = TransferEngine()
    if te.initialize(ip, "P2PHANDSHAKE", "rdma", ib) != 0:
        raise RuntimeError(f"TransferEngine.initialize({ip}, {ib}) failed")
    return te


def worker_stub_routes(app):
    # What ATOMesh asks a worker at registration.
    async def ok(_):
        return web.json_response({"model_path": "spike"})

    async def models(_):
        return web.json_response({"object": "list", "data": [{"id": "spike"}]})

    app.router.add_get("/health", ok)
    app.router.add_get("/server_info", ok)
    app.router.add_get("/model_info", ok)
    app.router.add_get("/v1/models", models)


def producer(args):
    dev = torch.device("cuda", 0)
    g = torch.Generator(device=dev).manual_seed(0)
    src = torch.randn(MAX_BYTES // 2, generator=g, device=dev).to(torch.bfloat16)
    sums = {mb: checksum(src[: mb * MiB // 2]) for mb in SIZES_MB}
    te = make_engine(args.ip, args.ib)
    if te.register_memory(src.data_ptr(), MAX_BYTES) != 0:
        raise RuntimeError("register_memory failed")
    session = f"{args.ip}:{te.get_rpc_port()}"

    async def meta(_):
        return web.json_response(
            {"session": session, "ptr": src.data_ptr(), "sums": sums}
        )

    async def tensor(request):
        n = int(request.query["mb"]) * MiB // 2
        host = src[:n].cpu()
        return web.Response(body=memoryview(host.view(torch.uint8).numpy()))

    async def run_inline(request):
        mb, reps = int(request.query["mb"]), int(request.query["reps"])
        url = request.query["to"] + "/v1/chat/completions"
        out = []
        for i in range(reps):
            t0 = time.perf_counter()
            host = src[: mb * MiB // 2].cpu()
            b64 = base64.b64encode(host.view(torch.uint8).numpy())
            body = json.dumps(
                {
                    "model": "spike",
                    "messages": [{"role": "user", "content": "describe the image"}],
                    "max_tokens": 1,
                    "ec_payload": b64.decode("ascii"),
                    "ec_rid": f"{mb}-{i}",
                }
            )
            async with client.post(
                url, data=body, headers={"content-type": "application/json"}
            ) as r:
                if r.status != 200:
                    raise RuntimeError(f"mesh returned {r.status}: {await r.text()}")
                await r.read()
            out.append((time.perf_counter() - t0) * 1e3)
        return web.json_response(out)

    client = None

    async def on_startup(_):
        nonlocal client
        client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))

    app = web.Application(client_max_size=BODY_LIMIT)
    app.on_startup.append(on_startup)
    worker_stub_routes(app)
    app.router.add_get("/meta", meta)
    app.router.add_get("/tensor", tensor)
    app.router.add_get("/run_inline", run_inline)
    web.run_app(app, host=args.ip, port=args.port, print=None)


def nic_rx_bytes(ib):
    # RDMA payload bytes the HCA received: shows the read went over the NIC.
    with open(
        f"/sys/class/infiniband/{ib}/ports/1/hw_counters/rx_rdma_ucast_bytes"
    ) as f:
        return int(f.read())


def summarize(transport, mb, ms, **extra):
    row = {
        "transport": transport,
        "MiB": mb,
        "reps": len(ms),
        "median_ms": round(statistics.median(ms), 3),
        "max_ms": round(max(ms), 3),
        "min_ms": round(min(ms), 3),
        **extra,
    }
    print(json.dumps(row), flush=True)
    return row


async def consumer_driver(args, dst, received):
    dev = dst.device
    own_url = f"http://{args.ip}:{args.port}"
    te = make_engine(args.ip, args.ib)
    if te.register_memory(dst.data_ptr(), MAX_BYTES) != 0:
        raise RuntimeError("register_memory failed")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as http:
        async with http.get(args.producer + "/meta") as r:
            meta = await r.json()
        sums = {int(k): v for k, v in meta["sums"].items()}
        rows = []
        warm = args.warmup
        rx0, read_bytes = nic_rx_bytes(args.ib), 0
        for mb in SIZES_MB:
            nbytes = mb * MiB
            n = nbytes // 2

            ms = []
            for i in range(warm + args.reps):
                dst.zero_()
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                ret = te.transfer_sync_read(
                    meta["session"], dst.data_ptr(), meta["ptr"], nbytes
                )
                t1 = time.perf_counter()
                if ret != 0:
                    raise RuntimeError(f"transfer_sync_read returned {ret}")
                if checksum(dst[:n]) != sums[mb]:
                    raise RuntimeError(f"mooncake {mb} MiB: checksum mismatch")
                if i >= warm:
                    ms.append((t1 - t0) * 1e3)
            read_bytes += (warm + args.reps) * nbytes
            rows.append(summarize("mooncake", mb, ms))

            ms = []
            for i in range(warm + args.reps):
                t0 = time.perf_counter()
                async with http.get(args.producer + f"/tensor?mb={mb}") as r:
                    body = await r.read()
                got = torch.frombuffer(body, dtype=torch.bfloat16).to(dev)
                torch.cuda.synchronize(dev)
                t1 = time.perf_counter()
                if checksum(got) != sums[mb]:
                    raise RuntimeError(f"http {mb} MiB: checksum mismatch")
                if i >= warm:
                    ms.append((t1 - t0) * 1e3)
            rows.append(summarize("http", mb, ms))

            # inline: through ATOMesh; inline-direct: the same POST without it.
            for name, url in (("inline", args.mesh), ("inline-direct", own_url)):
                received.clear()
                q = f"?mb={mb}&reps={warm + args.reps}&to={url}"
                async with http.get(args.producer + "/run_inline" + q) as r:
                    if r.status != 200:
                        raise RuntimeError(f"run_inline: {r.status} {await r.text()}")
                    ms = (await r.json())[warm:]
                if len(received) != warm + args.reps or any(
                    checksum(t) != sums[mb] for t in received.values()
                ):
                    raise RuntimeError(f"{name} {mb} MiB: missing or corrupt payloads")
                rows.append(summarize(name, mb, ms))

    await asyncio.sleep(3)  # the HCA's counters refresh with a delay
    rx = (nic_rx_bytes(args.ib) - rx0) / read_bytes
    print(f"mooncake: {args.ib} RDMA bytes received / bytes read = {rx:.3f}")
    print(f"\n{'transport':<14}{'MiB':>5}{'median ms':>12}{'max ms':>10}")
    for r in rows:
        print(
            f"{r['transport']:<14}{r['MiB']:>5}{r['median_ms']:>12.3f}{r['max_ms']:>10.3f}"
        )


def consumer(args):
    dev = torch.device("cuda", 0)
    dst = torch.zeros(MAX_BYTES // 2, dtype=torch.bfloat16, device=dev)
    received = {}

    async def chat(request):
        data = json.loads(await request.read())
        raw = base64.b64decode(data["ec_payload"])
        t = torch.frombuffer(raw, dtype=torch.bfloat16).to(dev)
        torch.cuda.synchronize(dev)
        received[data["ec_rid"]] = t
        return web.json_response(
            {
                "id": data["ec_rid"],
                "object": "chat.completion",
                "model": "spike",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

    async def main():
        app = web.Application(client_max_size=BODY_LIMIT)
        worker_stub_routes(app)
        app.router.add_post("/v1/chat/completions", chat)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, args.ip, args.port).start()
        await wait_ready(args.mesh)
        received.clear()
        await consumer_driver(args, dst, received)
        await runner.cleanup()

    asyncio.run(main())


async def wait_ready(mesh, timeout=120):
    # Until ATOMesh has registered the consumer, a request through it fails.
    body = {"model": "spike", "messages": [{"role": "user", "content": "x"}]}
    body.update(
        max_tokens=1, ec_payload=base64.b64encode(b"\0\0").decode(), ec_rid="probe"
    )
    deadline = time.monotonic() + timeout
    async with aiohttp.ClientSession() as s:
        while time.monotonic() < deadline:
            try:
                async with s.post(mesh + "/v1/chat/completions", json=body) as r:
                    if r.status == 200:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(1)
    raise RuntimeError(f"{mesh} does not relay to the consumer after {timeout}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=("producer", "consumer"))
    p.add_argument("--ip", required=True)
    p.add_argument("--ib", required=True, help="RDMA device paired with the GPU")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--producer", help="producer URL (consumer only)")
    p.add_argument("--mesh", help="ATOMesh URL (consumer only)")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    a = p.parse_args()
    # frombuffer on read-only bytes: the tensor is only copied to the device.
    warnings.filterwarnings("ignore", message="The given buffer is not writable")
    # ATOMesh probes a worker for gRPC first; aiohttp logs each probe as an error.
    logging.getLogger("aiohttp.server").setLevel(logging.CRITICAL)
    (producer if a.role == "producer" else consumer)(a)
