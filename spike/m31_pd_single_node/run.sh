#!/usr/bin/env bash
# MiniMax-M3.1 text through ATOMesh on one 8-GPU node, scored on GSM8K.
# Run on the GPU node from the repository root:
#   TOPO=pd DSPARK=1 ATOMESH=<atomesh built from this tree> bash spike/m31_pd_single_node/run.sh
# TOPO=single: one instance on GPUs 0-3 behind ATOMesh.
# TOPO=pd: prefill on GPUs 0-3, decode on 4-7 (TP4 each, Mooncake RDMA),
# ATOMesh --pd-disaggregation, as the CI cell minimax-m3-fp4-1p1d-tp4.
# DSPARK=0 drops the recipe's three DSpark flags on every server.
# MC_DISABLE_HIP=0 leaves out the CI cell's MC_DISABLE_HIP=1.
# Servers start once; GSM8K runs REPS times against them.
set -euo pipefail
TOPO=${TOPO:?single or pd}
DSPARK=${DSPARK:-1}
REPS=${REPS:-3}
LIMIT=${LIMIT:-200}
ATOMESH=${ATOMESH:?path to the atomesh binary built from this tree}
IMG=${IMG:-rocm/atom-dev:nightly_202611081546}
MODEL=${MODEL:-/share_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4}
IP=${IP:-$(hostname -I | awk '{print $2}')}
NAME=${NAME:-epd-5-$TOPO}
MC_DISABLE_HIP=${MC_DISABLE_HIP:-1}
OUT=${OUT:-/share_nfs/junyan_tmp_r120/spike5/$TOPO-dspark$DSPARK-hipoff$MC_DISABLE_HIP-$(date +%m%d-%H%M%S)}
PORT_P=18510 PORT_D=18520 PORT_M=18500 HANDSHAKE=16301
TREE=$(pwd -P)
SPIKE=$TREE/spike/m31_pd_single_node
mkdir -p "$OUT"

args=(run -d --name "$NAME" --network host --ipc host --security-opt label=disable
  --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 --shm-size 64g
  -e MC_GID_INDEX=1 -e MC_ENABLE_DEST_DEVICE_AFFINITY=1
  -e ATOM_HOST_IP="$IP" -e PYTHONUNBUFFERED=1
  -e AITER_LOG_LEVEL=WARNING -e ATOM_FORCE_ATTN_TRITON=1
  -e AITER_QUICK_REDUCE_QUANTIZATION=INT4 -e ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=1
  -e HOME="$HOME" -e HF_HOME=/share_nfs/junyan_tmp_r120/hf -e PYTHONPATH="$TREE"
  -v "$HOME:$HOME" -v /share_nfs:/share_nfs
  -v "$(dirname "$ATOMESH"):/opt/spike-mesh:ro" -w "$TREE")
[[ "$MC_DISABLE_HIP" == 0 ]] || args+=(-e MC_DISABLE_HIP=1)
host_ionic="$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)"
[[ -z "$host_ionic" ]] || args+=(-v "$host_ionic:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro")
[[ ! -e /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so ]] || args+=(
  -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro)
[[ ! -e /etc/libibverbs.d/ionic.driver ]] || args+=(-v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro)
docker "${args[@]}" "$IMG" sleep infinity >/dev/null
trap 'docker rm -f "$NAME" >/dev/null' EXIT
# The image predates the jsonschema dependency in pyproject.toml.
docker exec "$NAME" pip install -q "jsonschema>=4.25,<5"
docker exec "$NAME" python -c 'import atom, sys; print("atom from", atom.__file__)' | tee "$OUT/atom_origin.txt"

server_flags=(--model "$MODEL" --served-model-name minimax-m3.1 --tensor-parallel-size 4
  --trust-remote-code --default-chat-template-kwargs '{"thinking_mode": "enabled"}'
  --gpu-memory-utilization 0.9 --block-size 128 --max-model-len 131072 --max-num-seqs 128
  --max-num-batched-tokens 32768 --kv_cache_dtype fp8 --index-cache-dtype fp8
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}'
  --no-enable_prefix_caching)
[[ "$DSPARK" == 0 ]] || server_flags+=(--method dspark --draft-model "$MODEL/dspark" --num-speculative-tokens 7)

# serve <log> <gpus> <server port> <engine port> [extra flags...]
# Each server gets its own compile caches on the container's disk, as the CI
# launcher gives each role: two servers sharing ~/.cache/atom race on it.
serve() {
  local log=$1 gpus=$2 port=$3 engine=$4
  local c=/tmp/spike-cache/$log
  shift 4
  docker exec -d -e HIP_VISIBLE_DEVICES="$gpus" -e HOME=$c/home -e XDG_CACHE_HOME=$c/xdg \
    -e TORCHINDUCTOR_CACHE_DIR=$c/torchinductor -e TRITON_CACHE_DIR=$c/triton "$NAME" bash -c \
    "exec python -m atom.entrypoints.openai_server $(printf '%q ' "${server_flags[@]}" "$@") \
    --server-port $port --port $engine > $OUT/$log.log 2>&1"
}
wait_up() {
  until curl -sf "http://$IP:$1/health" >/dev/null; do
    docker exec "$NAME" pgrep -f "server-port $1" >/dev/null || { tail -30 "$OUT/$2.log"; exit 1; }
    sleep 10
  done
}

if [[ "$TOPO" == pd ]]; then
  producer="{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"$IP\",\"handshake_port\":$HANDSHAKE}"
  consumer="{\"kv_role\":\"kv_consumer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"$IP\",\"handshake_port\":$HANDSHAKE,\"ib_enable_alternate_hca\":true,\"ib_hca_count\":8}"
  serve prefill 0,1,2,3 $PORT_P 18506 --kv-transfer-config "$producer"
  serve decode 4,5,6,7 $PORT_D 18507 --kv-transfer-config "$consumer"
  wait_up $PORT_P prefill
  wait_up $PORT_D decode
  mesh_args=(--pd-disaggregation --prefill "http://$IP:$PORT_P" --decode "http://$IP:$PORT_D" --backend atom)
else
  serve server 0,1,2,3 $PORT_P 18506
  wait_up $PORT_P server
  mesh_args=(--worker-urls "http://$IP:$PORT_P")
fi
docker exec -d "$NAME" bash -c "exec /opt/spike-mesh/$(basename "$ATOMESH") launch --host 0.0.0.0 \
  --port $PORT_M $(printf '%q ' "${mesh_args[@]}") --policy random --disable-circuit-breaker \
  --log-level info > $OUT/mesh.log 2>&1"
until curl -sf "http://127.0.0.1:$PORT_M/health" >/dev/null; do sleep 2; done

# RDMA bytes received over every HCA on the node: which transport carried the KV.
rdma_rx() { cat /sys/class/infiniband/*/ports/1/hw_counters/rx_rdma_ucast_bytes | paste -sd+ | bc; }
for r in $(seq "$REPS"); do
  offset=$(stat -c %s "$OUT/mesh.log")
  rx0=$(rdma_rx)
  docker exec "$NAME" lm_eval --model local-chat-completions \
    --model_args "model=minimax-m3.1,base_url=http://127.0.0.1:$PORT_M/v1/chat/completions,tokenizer=$MODEL,tokenized_requests=False,num_concurrent=16,max_gen_toks=16384" \
    --tasks gsm8k --num_fewshot 5 --apply_chat_template --fewshot_as_multiturn --limit "$LIMIT" \
    --gen_kwargs "temperature=0,reasoning_effort=max,max_gen_toks=16384,until=<|no_stop_string|>" \
    --log_samples --output_path "$OUT/gsm8k-run$r" > "$OUT/gsm8k-run$r.log" 2>&1
  sleep 3 # the HCA counters refresh with a delay
  docker exec "$NAME" python3 "$SPIKE/score.py" --label "$TOPO dspark=$DSPARK mc_disable_hip=$MC_DISABLE_HIP run $r" \
    --results "$OUT/gsm8k-run$r" --mesh-log "$OUT/mesh.log" --offset "$offset" \
    --rdma-rx-bytes $(($(rdma_rx) - rx0)) | tee -a "$OUT/score.jsonl"
done
