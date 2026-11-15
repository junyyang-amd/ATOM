#!/bin/bash
# The decode openai_server command of CI Slurm 11899 / 12390 without
# --kv-transfer-config. Runs inside the image, on that image's own ATOM.
set -euo pipefail
sizes=$(seq -s, 1 480)
quant='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'
exec python3 -m atom.entrypoints.openai_server \
  --model /share_nfs/models/moonshotai/Kimi-K3 --host 0.0.0.0 --trust-remote-code \
  --kv_cache_dtype fp8 --block-size 128 --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192 --online_quant_config "$quant" \
  --method dspark --draft-model /share_nfs/models/Inferact/Kimi-K3-DSpark \
  --num-speculative-tokens 4 --spec-decode-acceptance-length 3.36 \
  --state-checkpoint-interval-tokens -1 --server-port 8020 -tp 8 \
  --decode-context-parallel-size 8 --max-num-seqs 96 \
  --cudagraph-capture-sizes "[$sizes]" --cudagraph-mode FULL --level 3 \
  --tool-call-parser kimi_k3
