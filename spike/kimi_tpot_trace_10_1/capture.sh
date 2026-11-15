#!/bin/bash
# Capture one kineto trace of a Kimi-K3 decode cohort on one node, in the
# image's own ATOM: the decode flags and env of ATOMesh CI Slurm 11899 minus
# --kv-transfer-config, one server on all 8 GPUs, 48 random prompts
# (ISL 8192, OSL 1024) at concurrency 48 with PROFILE=1.
# Run on the GPU host: bash capture.sh <trace-dir> [image]
set -euo pipefail
OUT=$(realpath -m "$1")
IMG=${2:-rocm/atom-dev:ROCm_10.1.0}
NAME=kimi-tpot-65-server
PORT=8020
H=/home/junyan.yang@amd.com
M=/share_nfs/models/moonshotai/Kimi-K3
[[ ! -e $OUT ]] || { echo "trace dir $OUT exists; use a fresh one" >&2; exit 1; }
mkdir -p "$OUT"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --ulimit memlock=-1:-1 --shm-size 64g \
  -e HF_HOME=/share_nfs/junyan_tmp_r120/hf \
  -v /share_nfs:/share_nfs -v "$H:$H" "$IMG" sleep infinity >/dev/null
trap 'docker rm -f "$NAME" >/dev/null' EXIT

sizes=$(seq -s, 1 480)
quant='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'
docker exec -d -w /app/ATOM \
  -e AITER_FLYDSL_STAGE2_FP8=1 -e AITER_LOG_LEVEL=WARNING \
  -e AITER_QUICK_REDUCE_QUANTIZATION=INT4 -e AITER_REUSE_IDENTICAL_COMM_GROUPS=0 \
  -e AITER_SITUV2_A4W4=1 -e ATOM_ENABLE_REPLAYSSM=1 -e ATOM_GDN_SSM_DTYPE=fp16 \
  -e ATOM_PD_RANK_MAPPING_POLICY=none -e ATOM_STATE_CHECKPOINT_DEMAND=1 \
  -e ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1 -e ATOM_USE_FLYDSL_GATHER_KV_B_PROJ=1 \
  -e PYTHONHASHSEED=0 -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -e ATOM_PROFILER_MORE=0 \
  "$NAME" bash -c "rm -rf /root/.cache/atom/*; exec python3 -m atom.entrypoints.openai_server \
    --model $M --host 0.0.0.0 --trust-remote-code --kv_cache_dtype fp8 --block-size 128 \
    --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192 --online_quant_config '$quant' \
    --method dspark --draft-model /share_nfs/models/Inferact/Kimi-K3-DSpark \
    --num-speculative-tokens 4 --spec-decode-acceptance-length 3.36 \
    --state-checkpoint-interval-tokens -1 --server-port $PORT -tp 8 \
    --decode-context-parallel-size 8 --max-num-seqs 96 \
    --cudagraph-capture-sizes [$sizes] --cudagraph-mode FULL --level 3 \
    --tool-call-parser kimi_k3 --torch-profiler-dir $OUT > $OUT/server.log 2>&1"

for i in $(seq 1 360); do  # model load + compile + 480 graph captures: up to an hour
  sleep 10
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null && break
  docker exec "$NAME" pgrep -f openai_server >/dev/null || { tail -30 "$OUT/server.log"; exit 1; }
  [[ $i -lt 360 ]] || { echo "server not ready after 1h" >&2; exit 1; }
done
echo "server ready"

docker exec -w /app/ATOM "$NAME" bash -c "mkdir -p /app/logs_claude &&
  bash scripts/run_benchmark.sh $M $PORT 8192 1024 48 1 1" > "$OUT/bench.log" 2>&1
tail -25 "$OUT/bench.log"

for i in $(seq 1 90); do  # exporter: .gz present and no orphan .json on all 8 ranks
  gz=$(find "$OUT" -name '*.pt.trace.json.gz' | wc -l)
  js=$(find "$OUT" -name '*.pt.trace.json' | wc -l)
  [[ $gz -ge 8 && $js -eq 0 ]] && break
  [[ $i -lt 90 ]] || { echo "export incomplete: gz=$gz json=$js" >&2; exit 1; }
  sleep 10
done
find "$OUT" -name '*.gz' -exec ls -la {} +
