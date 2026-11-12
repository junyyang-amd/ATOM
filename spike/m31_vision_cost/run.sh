#!/bin/bash
# On g15, from the repository root: bash spike/m31_vision_cost/run.sh
# M3.1 single instance per recipes/MiniMax-M3.md (TP4, DSpark) on GPUS; then
# MMStar 1 image/request at concurrency 1, and RUNS runs of 4 images/request at
# concurrency 32. TIMING=0 drops the wrapper (TTFT only).
set -euo pipefail
root=$(pwd)
spike=$root/spike/m31_vision_cost
out=${OUT:-/share_nfs/junyan_tmp_r120/spike6/$(date +%m%d-%H%M%S)}
model=/share_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4
name=epd-6-m31
mkdir -p "$out/timing"
docker run -d --name $name --network host --ipc host --device /dev/kfd --device /dev/dri \
  --group-add video --shm-size 64g -e HOME="$HOME" -e HF_HOME=/share_nfs/junyan_tmp_r120/hf \
  -v /share_nfs:/share_nfs -v "$HOME:$HOME" -w "$root" \
  "${IMG:-rocm/atom-dev:nightly_202611081546}" sleep infinity
trap 'docker rm -f $name >/dev/null' EXIT
# The image predates the jsonschema dependency in pyproject.toml.
docker exec $name pip install -q "jsonschema>=4.25,<5"
timing=""
[[ "${TIMING:-1}" == 0 ]] || timing="$out/timing"
docker exec -d -e PYTHONPATH="$spike:$root" -e SPIKE_TIMING_DIR="$timing" \
  -e HIP_VISIBLE_DEVICES="${GPUS:-0,1,2,3}" -e AITER_LOG_LEVEL=WARNING -e ATOM_FORCE_ATTN_TRITON=1 \
  -e AITER_QUICK_REDUCE_QUANTIZATION=INT4 -e ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=1 $name bash -c "
  python -m atom.entrypoints.openai_server --model $model --served-model-name minimax-m3.1 \
  --tensor-parallel-size 4 --server-port 8610 --port 8606 --trust-remote-code \
  --default-chat-template-kwargs '{\"thinking_mode\": \"enabled\"}' \
  --gpu-memory-utilization 0.9 --block-size 128 --max-model-len 131072 --max-num-seqs 128 \
  --max-num-batched-tokens 32768 --kv_cache_dtype fp8 --index-cache-dtype fp8 \
  --online_quant_config '{\"global_quant_config\": \"ptpc_fp8\", \"exclude_layer\": [\"lm_head\", \"model.embed_tokens\", \"vision_tower\", \"multi_modal_projector\", \"patch_merge_mlp\", \"*block_sparse_moe\"]}' \
  --no-enable_prefix_caching --method dspark --draft-model $model/dspark --num-speculative-tokens 7 \
  > $out/server.log 2>&1"
until curl -sf localhost:8610/health >/dev/null; do
  docker exec $name pgrep -f openai_server >/dev/null || { tail -30 "$out/server.log"; exit 1; }
  sleep 10
done
load() { docker exec -e PYTHONPATH="$root" $name python3 "$spike/load.py" "$@"; }
load --images 4 --requests 64 --out "$out/warmup.json"
load --images 1 --concurrency 1 --requests 64 --offset 1000 --max-tokens 1 --out "$out/mmstar-1img.json"
for r in $(seq "${RUNS:-5}"); do
  load --images 4 --requests 128 --offset $((r * 512)) --out "$out/a5-run$r.json"
done
docker exec $name python3 "$spike/report.py" "$out" mmstar-1img a5-run | tee "$out/table.md"
