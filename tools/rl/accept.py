#!/usr/bin/env python3
"""RL acceptance checks on pit2-p03-g24.

    python3 tools/rl/accept.py <check> [--dry-run]

Runs ``checks/<check>.py``, which exports ``ISSUE`` (its containers are named
``rl-<ISSUE>-<role>``), ``plan(h) -> list[str]`` (what ``--dry-run`` prints)
and ``run(h) -> Verdict``. Off the node, the script reruns itself there over
ssh once the node sees this tree. It writes ``OUT_DIR/<check>.json`` and
container logs under ``OUT_DIR/logs/<check>/``, prints ``PASS <check>`` or
``FAIL <check>: <reason>`` and exits 0 only on PASS. Stdlib only: the node's
own python3 runs it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

NODE = "junyan.yang@amd.com@pit2-p03-g24"
NODE_HOST = "pit2-p03-g24"
NODE_IP = "10.19.0.124"
TREE = Path(__file__).resolve().parents[2]
OUT_DIR = Path("/share_nfs/junyan_tmp_r120/rl-accept")
HF_HOME = "/share_nfs/junyan_tmp_r120/hf"
GATE_DIR = "/var/tmp/junyan_agent_team_gate"
ATOM_IMAGE = "rocm/atom-dev:nightly_202611081546"
# lmsysorg/sglang-rocm is untagged on the node; this is its image id.
SGLANG_IMAGE = "sha256:2456fff6cfaffb822026d089a0629a4859c84d2448864ffd6d5fa4f945ef2c86"
MESH_IMAGE = "atom-gate:local"
FIRST_PORT = 18200
GPU_COUNT = 8
GPU_BUSY_MB = 1024
READY_TIMEOUT_S = 900
CONTROL_TIMEOUT_S = 900
# SMG's long-rollout launch profile, with which every check runs ATOMesh.
LONG_ROLLOUT = [
    "--enable-rl",
    "--disable-health-check",
    "--disable-circuit-breaker",
    "--request-timeout-secs",
    "14400",
]
# Explicit device flags (no --privileged on this node), as ~/issue455/pd_up.sh.
GPU_FLAGS = [
    "--network", "host", "--ipc", "host", "--shm-size", "32g",
    "--device", "/dev/kfd", "--device", "/dev/dri", "--device", "/dev/infiniband",
    "--group-add", "video", "--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1:-1",
]  # fmt: skip
# The image lacks jsonschema, which atom imports; the log names the atom that runs.
ATOM_PRELUDE = (
    "python3 -c 'import jsonschema' 2>/dev/null || pip install -q jsonschema "
    "|| exit 1; python3 -c 'import atom; print(\"atom from\", atom.__file__)'; "
)
# The tree mounted at its own path, wherever it lives, and run from there.
TREE_FLAGS = ["-v", f"{TREE}:{TREE}", "-w", str(TREE), "-e", f"PYTHONPATH={TREE}"]
_TOPICS = ["a lighthouse", "the moon", "bread", "a river", "chess", "rain"]
_TOPICS += ["a violin", "glaciers"]
_FORMS = ["Write a short poem about", "Explain to a child", "List three facts about"]
_FORMS += ["Tell a story about"]
# 32 fixed prompts for "equals fresh": temperature 0, 64 new tokens, one at a time.
PROMPTS = [f"{form} {topic}." for form in _FORMS for topic in _TOPICS]
SAMPLING = {"temperature": 0, "max_new_tokens": 64}


class HarnessError(RuntimeError):
    pass


@dataclass
class Verdict:
    passed: bool
    reason: str = ""
    data: dict = field(default_factory=dict)


def http(method: str, url: str, body=None, timeout: float = CONTROL_TIMEOUT_S):
    """(status, parsed JSON or text) of one call; an HTTP error is an answer."""
    data = None if body is None else json.dumps(body).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw.decode(errors="replace")


def _git(*args: str) -> bytes:
    cmd = ["git", "-C", str(TREE), *args]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def tree_digest() -> str:
    """Digest of this tree's HEAD, its diff and its untracked files."""
    digest = hashlib.sha256(
        _git("rev-parse", "HEAD") + _git("diff", "HEAD", "--binary")
    )
    for name in _git("ls-files", "-o", "--exclude-standard", "-z").split(b"\0"):
        if name:
            digest.update(name + (TREE / name.decode()).read_bytes())
    return digest.hexdigest()


def wait_for_tree(want: str | None) -> None:
    """Wait until this host sees the tree the caller digested (NFS lags)."""
    for _ in range(30):
        if want is None or tree_digest() == want:
            return
        time.sleep(3)
    raise HarnessError(f"{NODE_HOST} still sees a different tree after 90 s")


class Harness:
    """Containers, ports and GPUs of one check, and the calls a check makes.

    Workers take GPUs from 0 upward and ports from FIRST_PORT upward; a busy
    GPU or port is refused, never skipped. With ``dry_run`` nothing runs: each
    command is appended to ``steps`` instead.
    """

    def __init__(self, issue: int, dry_run: bool = False):
        self.issue, self.dry_run = issue, dry_run
        self.steps: list[str] = []
        self.containers: list[str] = []
        self.mesh_url: str | None = None
        self._next_gpu, self._next_port = 0, FIRST_PORT
        self._mesh_built = False
        self._user = f"{os.getuid()}:{os.getgid()}"

    # -- processes ---------------------------------------------------------

    def sh(self, argv: list[str], check: bool = True) -> str:
        if self.dry_run:
            self.steps.append(shlex.join(argv))
            return ""
        res = subprocess.run(argv, capture_output=True, text=True, check=False)
        if check and res.returncode:
            raise HarnessError(
                f"{shlex.join(argv[:4])} ... exited {res.returncode}: {res.stderr[-2000:]}"
            )
        return res.stdout

    def take_gpus(self, n: int) -> list[int]:
        gpus = list(range(self._next_gpu, self._next_gpu + n))
        self._next_gpu += n
        self.refuse_busy(gpus)
        return gpus

    def refuse_busy(self, gpus: list[int]) -> None:
        if max(gpus) >= GPU_COUNT:
            raise HarnessError(f"needs GPU {max(gpus)}; the node has {GPU_COUNT}")
        if not self.dry_run:
            rows = json.loads(self.sh(["amd-smi", "monitor", "-v", "--json"]))
            used = {r["gpu"]: r["vram_used"]["value"] for r in rows}
            busy = [g for g in gpus if used.get(g, 0) > GPU_BUSY_MB]
            if busy:
                raise HarnessError(f"GPU {busy} busy (> {GPU_BUSY_MB} MB in use)")

    def take_port(self) -> int:
        port, self._next_port = self._next_port, self._next_port + 1
        if not self.dry_run:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    raise HarnessError(f"port {port} is already in use")
        return port

    def container(self, role: str, image: str, cmd: list[str], gpus=None, flags=()):
        """Start ``rl-<issue>-<role>`` detached; removed at teardown."""
        name = f"rl-{self.issue}-{role}"
        argv = ["docker", "run", "-d", "--name", name, *flags]
        if gpus is not None:
            argv += [
                *GPU_FLAGS,
                "-e",
                f"HIP_VISIBLE_DEVICES={','.join(map(str, gpus))}",
            ]
        argv += ["-v", "/share_nfs:/share_nfs", "-e", f"HF_HOME={HF_HOME}"]
        argv += ["-e", "AITER_LOG_LEVEL=WARNING", image, *cmd]
        if not self.dry_run:
            self.sh(["docker", "rm", "-f", name], check=False)
        self.containers.append(name)
        self.sh(argv)
        return name

    def wait_ready(self, name: str, url: str, path="/health", ok=lambda body: True):
        if self.dry_run:
            return
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                status, body = http("GET", url + path, timeout=10)
                if status == 200 and ok(body):
                    return
            except OSError:
                pass
            state = self.sh(["docker", "inspect", "-f", "{{.State.Running}}", name])
            if state.strip() != "true":
                raise HarnessError(
                    f"{name} exited before ready: {self.logs(name)[-2000:]}"
                )
            time.sleep(5)
        raise HarnessError(f"{name} not ready after {READY_TIMEOUT_S} s")

    def logs(self, name: str) -> str:
        cmd = ["docker", "logs", "--tail", "5000", name]
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
        )
        return res.stdout.decode(errors="replace")

    def remove_leftovers(self) -> None:
        """Remove ``rl-<issue>-*`` containers an interrupted run left behind:
        without a tty, ssh going away does not stop the run, so its teardown
        never ran and its workers still hold their GPUs."""
        names = self.sh(["docker", "ps", "-a", "--format", "{{.Names}}"])
        mine = [n for n in names.split() if n.startswith(f"rl-{self.issue}-")]
        if mine:
            self.sh(["docker", "rm", "-f", *mine], check=False)

    def teardown(self, log_dir: Path) -> None:
        if self.dry_run:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        for name in self.containers:
            (log_dir / f"{name}.log").write_text(self.logs(name))
            self.sh(["docker", "rm", "-f", name], check=False)

    # -- models and workers ------------------------------------------------

    def model_dir(self, repo: str) -> str:
        """The snapshot of ``repo`` (``Qwen/Qwen3-1.7B``), downloaded if missing."""
        hub = Path(HF_HOME, "hub", "models--" + repo.replace("/", "--"), "snapshots")
        if not sorted(hub.glob("*")):
            fetch = f"from huggingface_hub import snapshot_download; snapshot_download({repo!r})"
            self.sh(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--user",
                    self._user,
                    "-e",
                    f"HF_HOME={HF_HOME}",
                    "-v",
                    "/share_nfs:/share_nfs",
                    ATOM_IMAGE,
                    "python3",
                    "-c",
                    fetch,
                ]
            )
        snaps = sorted(hub.glob("*"))
        if snaps:
            return str(snaps[-1])
        if self.dry_run:
            return str(hub / "<downloaded>")
        raise HarnessError(f"{repo}: no snapshot under {hub} after download")

    def start_sglang_workers(self, n: int, model: str) -> list[str]:
        urls = []
        for _ in range(n):
            gpus, port = self.take_gpus(1), self.take_port()
            cmd = ["python3", "-m", "sglang.launch_server", "--model-path", model]
            cmd += ["--host", "0.0.0.0", "--port", str(port), "--tp-size", "1"]
            name = self.container(f"sglang-{port}", SGLANG_IMAGE, cmd, gpus)
            urls.append((name, f"http://{NODE_IP}:{port}"))
        for name, url in urls:
            self.wait_ready(name, url)
        return [url for _, url in urls]

    @staticmethod
    def atom_argv(model: str, port: int, engine_port: int, extra=()) -> list[str]:
        """Arguments of ``python -m atom.entrypoints.openai_server`` for a worker."""
        return [
            "--model",
            model,
            "--host",
            "0.0.0.0",
            "--server-port",
            str(port),
            "--port",
            str(engine_port),
            "--enable-rl",
            *extra,
        ]

    def _start_atom(self, model: str, gpus: list[int], extra=()) -> tuple[str, str]:
        port, engine_port = self.take_port(), self.take_port()
        argv = self.atom_argv(model, port, engine_port, extra)
        run = (
            ATOM_PRELUDE
            + "exec python3 -m atom.entrypoints.openai_server "
            + shlex.join(argv)
        )
        name = self.container(
            f"atom-{port}", ATOM_IMAGE, ["bash", "-c", run], gpus, TREE_FLAGS
        )
        return name, f"http://{NODE_IP}:{port}"

    def start_atom_workers(
        self, n: int, model: str, tp: int = 1, extra=()
    ) -> list[str]:
        """``n`` ATOM workers on the tree's code, ``--enable-rl``, ``tp`` GPUs each."""
        tp_args = ["-tp", str(tp)] if tp > 1 else []
        started = [
            self._start_atom(model, self.take_gpus(tp), [*tp_args, *extra])
            for _ in range(n)
        ]
        for name, url in started:
            self.wait_ready(name, url)
        return [url for _, url in started]

    def start_pd_pair(self, model: str, extra=()) -> list[str]:
        """P and D, one GPU each, KV over mooncake ``tcp``: ``[p_url, d_url]``."""
        started = []
        for role in ("producer", "consumer"):
            kv = {
                "kv_connector": "mooncake",
                "kv_role": f"kv_{role}",
                "proxy_ip": NODE_IP,
                "handshake_port": self.take_port(),
                "protocol": "tcp",
            }
            kv_args = ["--kv-transfer-config", json.dumps(kv)]
            started.append(
                self._start_atom(model, self.take_gpus(1), [*kv_args, *extra])
            )
        for name, url in started:
            self.wait_ready(name, url)
        return [url for _, url in started]

    def start_trainer(self, gpus: list[int], argv: list[str]) -> str:
        """Run ``argv`` on ``gpus`` in the ATOM image with the tree importable; its name."""
        for g in gpus:
            if g < self._next_gpu:
                raise HarnessError(f"trainer GPU {g} is already a worker's")
        self.refuse_busy(gpus)
        self._next_gpu = max(self._next_gpu, max(gpus) + 1)
        return self.container("trainer", ATOM_IMAGE, argv, gpus, TREE_FLAGS)

    # -- ATOMesh -----------------------------------------------------------

    def mesh_target(self) -> str:
        """The tree's cargo target dir, as the gate names it."""
        tag = hashlib.md5(str(TREE).encode()).hexdigest()[:12]
        return f"{GATE_DIR}/cargo-target/{tag}"

    def build_mesh(self) -> None:
        if self._mesh_built:
            return
        home = os.path.expanduser("~")
        name = f"rl-{self.issue}-mesh-build"
        self.containers.append(name)
        self.sh(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--user",
                self._user,
                "-e",
                f"HOME={home}",
                "-v",
                f"{home}:{home}",
                "-v",
                "/share_nfs:/share_nfs",
                "-v",
                f"{GATE_DIR}:{GATE_DIR}",
                "-v",
                f"{TREE}:{TREE}",
                "-w",
                f"{TREE}/atom/mesh",
                "-e",
                f"CARGO_HOME={home}/.cargo",
                "-e",
                f"RUSTUP_HOME={home}/.rustup",
                "-e",
                f"CARGO_TARGET_DIR={self.mesh_target()}",
                "-e",
                f"PATH={home}/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
                MESH_IMAGE,
                "cargo",
                "build",
                "--release",
                "--bin",
                "atomesh",
            ]
        )
        self._mesh_built = True

    def start_mesh(
        self, urls: list[str], backend="atom", pd=False, model_path=None
    ) -> str:
        """ATOMesh over ``urls`` (``[p_url, d_url]`` with ``pd``), long-rollout profile."""
        self.build_mesh()
        port, prom = self.take_port(), self.take_port()
        args = [
            "launch",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--prometheus-port",
            str(prom),
            "--backend",
            backend,
            "--policy",
            "round_robin",
            *LONG_ROLLOUT,
        ]
        if pd:
            args += ["--pd-disaggregation", "--prefill", urls[0], "--decode", urls[1]]
        else:
            args += ["--worker-urls", *urls]
        if model_path:
            args += ["--model-path", model_path]
        flags = [
            "--network",
            "host",
            "--user",
            self._user,
            "-v",
            f"{GATE_DIR}:{GATE_DIR}",
        ]
        binary = f"{self.mesh_target()}/release/atomesh"
        name = self.container(f"mesh-{port}", MESH_IMAGE, [binary, *args], flags=flags)
        self.mesh_url = f"http://127.0.0.1:{port}"

        def listed(body) -> bool:
            return isinstance(body, dict) and body.get("total") == len(urls)

        self.wait_ready(name, self.mesh_url, "/v1/rl/workers", listed)
        return self.mesh_url

    def call_mesh(self, method: str, path: str, body=None):
        return http(method, self.mesh_url + path, body)

    def rl_fanout(self, path: str, body, selector: str):
        query = urllib.parse.urlencode({"selector": selector})
        return self.call_mesh("POST", f"/v1/rl/engine/{path}?{query}", body)

    def rl_worker_call(self, worker_id: str, path: str, body=None):
        """GET without ``body``, else POST, through the per-worker proxy."""
        wid = urllib.parse.quote(worker_id, safe="")
        method = "GET" if body is None else "POST"
        return self.call_mesh(method, f"/v1/rl/workers/{wid}/engine/{path}", body)

    # -- outputs -----------------------------------------------------------

    def generate(self, url: str, prompts: list[str]) -> list[dict]:
        """SGLang-shaped ``/generate`` bodies, one prompt at a time, greedy."""
        out = []
        for prompt in prompts:
            req = {"text": prompt, "sampling_params": SAMPLING}
            status, body = http("POST", url + "/generate", req)
            if status != 200:
                raise HarnessError(f"/generate on {url} answered {status}: {body}")
            out.append(body)
        return out

    def equals_fresh(self, checkpoint: str, outputs: list[dict], launch) -> bool:
        """Whether ``outputs`` (``generate`` over ``PROMPTS``) have the token ids
        of a fresh server ``launch(checkpoint)`` starts and returns the URL of."""
        fresh = self.generate(launch(checkpoint), PROMPTS[: len(outputs)])
        return [o["output_ids"] for o in outputs] == [f["output_ids"] for f in fresh]

    def write_result(self, check: str, verdict: Verdict, elapsed_s: float) -> Path:
        dirty = bool(_git("status", "--porcelain").strip())
        commit = _git("rev-parse", "HEAD").decode().strip()
        record = {"check": check, "commit": commit, "dirty": dirty}
        record |= {"elapsed_s": round(elapsed_s, 1), **asdict(verdict)}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{check}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
        return path


def _exit_on_signal(signum, _frame):
    sys.exit(128 + signum)  # unwinds through teardown


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RL acceptance checks on " + NODE_HOST)
    ap.add_argument("check", help="module name under tools/rl/checks")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the steps; run nothing"
    )
    ap.add_argument("--tree-digest", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    sys.modules.setdefault("accept", sys.modules[__name__])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    check = importlib.import_module(f"checks.{args.check}")
    h = Harness(check.ISSUE, dry_run=args.dry_run)
    if args.dry_run:
        print("\n".join(check.plan(h)))
        return 0
    if socket.gethostname() != NODE_HOST:
        rerun = ["python3", "tools/rl/accept.py", args.check]
        rerun += ["--tree-digest", tree_digest()]
        remote = f"cd {shlex.quote(str(TREE))} && {shlex.join(rerun)}"
        ssh = ["ssh", "-o", "BatchMode=yes", NODE, remote]
        return subprocess.run(ssh, check=False).returncode
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _exit_on_signal)
    started = time.monotonic()
    try:
        wait_for_tree(args.tree_digest)
        h.remove_leftovers()
        verdict = check.run(h)
    except Exception as e:  # noqa: BLE001 - any failure is the check's FAIL
        verdict = Verdict(False, f"{type(e).__name__}: {e}")
    finally:
        h.teardown(OUT_DIR / "logs" / args.check)
    h.write_result(args.check, verdict, time.monotonic() - started)
    print(
        f"PASS {args.check}"
        if verdict.passed
        else f"FAIL {args.check}: {verdict.reason}"
    )
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
