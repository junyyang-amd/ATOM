#!/usr/bin/env bash
# One run of bench.py on one node: producer on GPU $GPU_P, consumer on
# GPU $GPU_C, ATOMesh between them, inside the recipe container with the
# Mooncake RDMA docker flags. Run on the GPU node from the repository root:
#   ATOMESH=<atomesh binary built from this tree> bash spike/embedding_transport/run.sh
set -euo pipefail
IMG=${IMG:-rocm/atom-dev:nightly_202611011456}
NAME=${NAME:-epd-7-bench}
GPU_P=${GPU_P:-4} IB_P=${IB_P:-rocep137s0}
GPU_C=${GPU_C:-5} IB_C=${IB_C:-rocep153s0}
IP=${IP:-$(hostname -I | awk '{print $2}')}
PORT_P=${PORT_P:-18710} PORT_C=${PORT_C:-18711} PORT_M=${PORT_M:-18712}
REPS=${REPS:-10}
ATOMESH=${ATOMESH:?path to the atomesh binary built from this tree}
TREE=$(pwd -P)

args=(run -d --rm --name "$NAME" --network host --ipc host --security-opt label=disable
  --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 --shm-size 16g
  -e MC_GID_INDEX=1 -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 -e MC_DISABLE_HIP=1
  -e HOME="$HOME" -v "$HOME:$HOME" -v /share_nfs/junyan_tmp_r120:/share_nfs/junyan_tmp_r120
  -v "$(dirname "$ATOMESH"):/opt/spike-mesh:ro" -w "$TREE")
host_ionic="$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)"
[[ -z "$host_ionic" ]] || args+=(-v "$host_ionic:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro")
[[ ! -e /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so ]] || args+=(
  -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro)
[[ ! -e /etc/libibverbs.d/ionic.driver ]] || args+=(-v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro)
docker "${args[@]}" "$IMG" sleep infinity >/dev/null
trap 'docker rm -f "$NAME" >/dev/null' EXIT

B=spike/embedding_transport/bench.py
LOGS=${LOGS:-/share_nfs/junyan_tmp_r120/spike7-logs}
mkdir -p "$LOGS"
docker exec -d -e HIP_VISIBLE_DEVICES="$GPU_P" "$NAME" bash -c \
  "python $B producer --ip $IP --ib $IB_P --port $PORT_P > $LOGS/producer.log 2>&1"
wait_up() {
  for _ in $(seq 120); do
    curl -sf "$1/health" >/dev/null && return
    sleep 1
  done
  echo "$1 not up" >&2
  return 1
}
wait_up "http://$IP:$PORT_P"
# The consumer serves first: ATOMesh registers its worker once, at launch.
docker exec -e HIP_VISIBLE_DEVICES="$GPU_C" "$NAME" \
  python "$B" consumer --ip "$IP" --ib "$IB_C" --port "$PORT_C" \
  --producer "http://$IP:$PORT_P" --mesh "http://$IP:$PORT_M" --reps "$REPS" &
wait_up "http://$IP:$PORT_C"
docker exec -d "$NAME" bash -c "/opt/spike-mesh/$(basename "$ATOMESH") launch --host $IP \
  --port $PORT_M --worker-urls http://$IP:$PORT_C --log-level warn > $LOGS/mesh.log 2>&1"
wait $!
