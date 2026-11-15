#!/bin/bash
# One-node A/B of Kimi-K3 decode TPOT: one ATOM server on all 8 GPUs per
# launch, launches alternating between the two images, one 48-prompt cohort
# per ISL per launch (scripts/run_benchmark.sh), no profiler.
# Usage, on pit2-p03-g08 from the repository root:
#   bash spike/kimi_tpot_one_node/run.sh OUT_DIR
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=$(mkdir -p "$1" && cd "$1" && pwd)
launches=${LAUNCHES:-3}
read -r -a isls <<<"${ISLS:-8192 65536}"
read -r -a images <<<"${IMAGES:-rocm/atom-dev:nightly_202611021458 rocm/atom-dev:ROCm_10.1.0}"
model=/share_nfs/models/moonshotai/Kimi-K3
name=
trap '[[ -z $name ]] || docker rm -f "$name" >/dev/null 2>&1' EXIT INT TERM

for r in $(seq 1 "$launches"); do
  for img in "${images[@]}"; do
    tag=${img##*:}
    dir=$out/$tag-r$r
    name=kimi-tpot-64-$tag-r$r
    mkdir -p "$dir"
    echo "=== launch $r $tag $(date -u +%FT%TZ)"
    docker run -d --name "$name" --network host --ipc host \
      --device /dev/kfd --device /dev/dri --device /dev/infiniband \
      --group-add video --cap-add IPC_LOCK --cap-add NET_ADMIN --cap-add SYS_NICE \
      --ulimit memlock=-1:-1 --ulimit stack=67108864 --ulimit nofile=65536:524288 \
      --shm-size 128G --env-file "$here/decode.env" \
      -v /share_nfs:/share_nfs -v "$dir:/app/logs_claude" \
      --entrypoint /bin/bash "$img" "$here/serve.sh" >/dev/null
    for _ in $(seq 1 540); do
      curl -sf localhost:8020/v1/models >/dev/null && break
      [[ $(docker inspect -f '{{.State.Running}}' "$name") == true ]] ||
        { docker logs "$name" >"$dir/server.log" 2>&1; echo "server died: $dir/server.log"; exit 1; }
      sleep 10
    done
    curl -sf localhost:8020/v1/models >/dev/null || { echo "server not ready after 90 min"; exit 1; }
    for isl in "${isls[@]}"; do
      docker exec -e RANDOM_RANGE_RATIO=1.0 "$name" bash "$repo/scripts/run_benchmark.sh" \
        "$model" 8020 "$isl" 1024 48 1 0 --save-detailed >"$dir/bench-isl$isl.log" 2>&1
      mv "$dir/benchmark.json" "$dir/isl$isl.json"
    done
    docker logs "$name" >"$dir/server.log" 2>&1
    docker exec "$name" chown -R "$(id -u):$(id -g)" /app/logs_claude
    docker rm -f "$name" >/dev/null
    name=
  done
done
docker run --rm -v "$here:$here" -v "$out:$out" --entrypoint python3 "${images[0]}" \
  "$here/tpot.py" "$out"
