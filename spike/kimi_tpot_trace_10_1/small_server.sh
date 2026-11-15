#!/bin/bash
# One profiled window on a Qwen3-0.6B ATOM server (tp 1) in the image; prints
# the event categories of its trace. UNSET=1 adds a sitecustomize that unsets
# ROCPROFILER_REGISTER_LIBRARY at every Python start, workers included.
# Run on the GPU host: bash small_server.sh <out-dir> [image]
set -euo pipefail
OUT=$(realpath -m "$1")
IMG=${2:-rocm/atom-dev:ROCm_10.1.0}
NAME=kimi-tpot-65-probe
PORT=8021
M=/share_nfs/models/Qwen/Qwen3-0.6B
[[ ! -e $OUT ]] || { echo "$OUT exists; use a fresh one" >&2; exit 1; }
mkdir -p "$OUT"
env=()
if [[ ${UNSET:-0} == 1 ]]; then
  mkdir "$OUT/site"
  echo 'import os; os.unsetenv("ROCPROFILER_REGISTER_LIBRARY")' > "$OUT/site/sitecustomize.py"
  env=(-e PYTHONPATH="$OUT/site")
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --ulimit memlock=-1:-1 --shm-size 16g \
  -v /share_nfs:/share_nfs "$IMG" sleep infinity >/dev/null
trap 'docker rm -f "$NAME" >/dev/null' EXIT
docker exec -d -w /app/ATOM -e ATOM_PROFILER_MORE=0 "${env[@]}" "$NAME" bash -c \
  "exec python3 -m atom.entrypoints.openai_server --model $M --server-port $PORT -tp 1 \
    --torch-profiler-dir $OUT > $OUT/server.log 2>&1"
for i in $(seq 1 120); do
  sleep 5
  curl -sf "localhost:$PORT/v1/models" >/dev/null && break
  [[ $i -lt 120 ]] || { tail -30 "$OUT/server.log"; exit 1; }
done
curl -sf -XPOST "localhost:$PORT/start_profile" >/dev/null
for j in 1 2 3; do
  curl -sf "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"prompt\":\"Hello, my name is\",\"max_tokens\":32}" >/dev/null
done
curl -sf -XPOST "localhost:$PORT/stop_profile" >/dev/null
for i in $(seq 1 30); do
  [[ -n $(find "$OUT" -name '*.pt.trace.json.gz') && -z $(find "$OUT" -name '*.pt.trace.json') ]] && break
  sleep 2
done
f=$(find "$OUT" -name '*.pt.trace.json.gz')
python3 -I -c "import collections, gzip, json, sys
ev = json.load(gzip.open(sys.argv[1]))['traceEvents']
print(sorted(collections.Counter(e.get('cat') for e in ev).items(), key=str))" "$f"
