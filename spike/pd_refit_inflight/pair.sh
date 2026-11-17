#!/bin/bash
# 1P1D Qwen3 behind the ATOMesh PD router, inside the rl-92-pd container.
#   pair.sh up <model_dir> <log_dir>   start P (GPU $P_GPU), D (GPU $D_GPU), router; wait ready
#   pair.sh down                       stop all three
# mooncake as ~/issue455/pd_up.sh configures it, over $PROTOCOL.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
IP=${IP:-10.19.0.124}
# rdma decodes garbage after the first token on this node (see the PR); tcp is
# correct against a plain server.
PROTOCOL=${PROTOCOL:-tcp}
P_GPU=${P_GPU:-0} D_GPU=${D_GPU:-1}
P_PORT=18110 D_PORT=18120 ROUTER_PORT=18100
MESH=${MESH:-/home/junyan.yang@amd.com/issue455/bin/atomesh-base}

down() {  # the container runs nothing else, so every python in it is ours
  pkill -f "atomesh-base launch" || true
  pkill -INT -f "pd_refit_inflight/server.py" || true
  for _ in $(seq 30); do pgrep python > /dev/null || return 0; sleep 1; done
  pkill -9 python || true
  sleep 3
}

worker() {  # gpu server_port engine_port kv_cfg log
  HIP_VISIBLE_DEVICES=$1 AITER_LOG_LEVEL=WARNING ATOM_METRICS_UPDATE_INTERVAL_S=0.2 \
    PYTHONPATH="$TREE" nohup python "$HERE/server.py" --model "$MODEL" \
    --server-port "$2" --port "$3" --kv-transfer-config "$4" > "$5" 2>&1 &
}

up() {
  MODEL=$1 LOG=$2
  mkdir -p "$LOG"
  producer="{\"kv_connector\":\"mooncake\",\"kv_role\":\"kv_producer\",\"proxy_ip\":\"$IP\",\"handshake_port\":6311,\"protocol\":\"$PROTOCOL\"}"
  consumer="{\"kv_connector\":\"mooncake\",\"kv_role\":\"kv_consumer\",\"proxy_ip\":\"$IP\",\"handshake_port\":6312,\"protocol\":\"$PROTOCOL\",\"ib_enable_alternate_hca\":true,\"ib_hca_count\":8}"
  worker "$P_GPU" $P_PORT 18106 "$producer" "$LOG/prefill.log"
  worker "$D_GPU" $D_PORT 18107 "$consumer" "$LOG/decode.log"
  for port in $P_PORT $D_PORT; do
    for i in $(seq 600); do
      curl -sf "http://127.0.0.1:$port/health" > /dev/null && break
      [[ $i == 600 ]] && { echo "worker :$port not ready" >&2; exit 1; }
      sleep 1
    done
  done
  nohup "$MESH" launch --host 0.0.0.0 --port $ROUTER_PORT --pd-disaggregation \
    --prefill "http://$IP:$P_PORT" --decode "http://$IP:$D_PORT" --policy round_robin \
    --model-path "$MODEL" --backend atom --log-level info --disable-circuit-breaker \
    --prometheus-port 29192 > "$LOG/router.log" 2>&1 &
  for i in $(seq 120); do
    curl -sf "http://127.0.0.1:$ROUTER_PORT/v1/models" > /dev/null && return 0
    sleep 1
  done
  echo "router not ready" >&2; exit 1
}

"$@"
