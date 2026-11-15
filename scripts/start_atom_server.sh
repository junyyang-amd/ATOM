#!/bin/bash
# Start ATOM OpenAI-compatible server
# Usage: bash start_atom_server.sh [MODEL_PATH] [TP_SIZE] [PORT] [EXTRA_ARGS...]
#
# Examples:
#   bash start_atom_server.sh                                    # DeepSeek-R1-0528, tp=8, port=8000
#   bash start_atom_server.sh /data/Llama-3.1-8B-Instruct-FP8-KV 1 8000
#   bash start_atom_server.sh /data/DeepSeek-R1-0528 8 8000 --method mtp --num-speculative-tokens 3

set -euo pipefail

MODEL_PATH="${1:-/data/DeepSeek-R1-0528}"
TP_SIZE="${2:-8}"
PORT="${3:-8000}"
shift 3 2>/dev/null || true
# An array, not "$*": a value carrying spaces -- an --online_quant_config
# JSON, a chat-template kwargs blob -- reaches argv only if it is never
# re-split. Flattening it here fed argparse the first word and nothing else.
EXTRA_ARGS=("$@")
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
LOG_FILE="/app/logs_claude/atom_server.log"

export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-INFO}"
export KINETO_CONFIG="/home/ljin1/dk/libkineto.conf"

# rocm-smi where present; ROCm 10.1 images ship amd-smi only. Every GPU index
# below is this tool's.
if command -v rocm-smi > /dev/null; then SMI=rocm-smi; else SMI=amd-smi; fi

# === Which cards this run owns ===
# HIP_VISIBLE_DEVICES numbering is not the smi tool's (measured here: HIP 0 is
# GPU[3]) and the map does not survive a reboot, so resolve it through the PCI
# bus now. Unset means the whole box, which is all this script used to assume.
OWNED_GPUS=""
if [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
    OWNED_GPUS="$(SMI=$SMI python - <<'PY' || true
import os, re, subprocess, torch

mine = {
    f"{torch.cuda.get_device_properties(i).pci_bus_id:02X}"
    for i in range(torch.cuda.device_count())
}
if os.environ["SMI"] == "rocm-smi":
    smi = subprocess.run(["rocm-smi", "--showbus"], capture_output=True, text=True).stdout
    buses = re.findall(r"GPU\[(\d+)\].*?PCI Bus: \w+:(\w\w):", smi)
else:
    smi = subprocess.run(["amd-smi", "list", "--csv"], capture_output=True, text=True).stdout
    buses = re.findall(r"^(\d+),\w+:(\w\w):", smi, re.M)
print(" ".join(idx for idx, bus in buses if bus.upper() in mine))
PY
)"
    [ -n "$OWNED_GPUS" ] || {
        echo "ERROR: cannot map HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES to $SMI indices"
        exit 1
    }
    echo "Owned GPUs: $SMI [$OWNED_GPUS] <- HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
fi

# How many OWNED cards hold memory. Empty OWNED_GPUS counts every card.
# amd-smi prints one decimal (an idle card reads 0.1), rocm-smi an integer.
# amd-smi's CSV lines end in CRLF.
owned_in_use() {
    if [ "$SMI" = rocm-smi ]; then
        rocm-smi --showmemuse 2>/dev/null | awk -v owned="$OWNED_GPUS" '
            /VRAM%/ {
                idx = $0; sub(/^GPU\[/, "", idx); sub(/\].*/, "", idx)
                if (owned != "" && index(" " owned " ", " " idx " ") == 0) next
                if ($NF + 0 > 0) n++
            }
            END { print n + 0 }'
    else
        amd-smi monitor -v --csv 2>/dev/null | awk -F, -v owned="$OWNED_GPUS" '
            { sub(/\r$/, "") }
            NR == 1 { for (i = 1; i <= NF; i++) if ($i == "vram_percent") c = i; next }
            owned != "" && index(" " owned " ", " " $1 " ") == 0 { next }
            c && $c + 0 >= 1 { n++ }
            END { print n + 0 }'
    fi
}

# Ownership is decidable only on the server process: its EngineCore child
# clears HIP_VISIBLE_DEVICES. Empty means this run claims the whole box.
owns_our_cards() {
    [ -n "${HIP_VISIBLE_DEVICES:-}" ] || return 0
    [ "$(tr '\0' '\n' <"/proc/$1/environ" 2>/dev/null |
        sed -n 's/^HIP_VISIBLE_DEVICES=//p')" = "$HIP_VISIBLE_DEVICES" ]
}

# Children are reached through the tree, not by name: the worker calls itself
# ATOM::EngineCore and matches none of the patterns a pkill would use. An
# EngineCore already orphaned by a dead server is unreachable either way -- it
# carries neither the card set nor a parent -- and shows up as owned_in_use.
kill_tree() {
    local child
    for child in $(pgrep -P "$1" 2>/dev/null); do
        kill_tree "$child"
    done
    kill -9 "$1" 2>/dev/null || true
}

# === Pre-flight: ensure GPU is clean ===
echo "Pre-flight: cleaning up processes and GPU memory..."

# 1. Kill this run's servers and everything under them. A bare `pkill -f` hits
#    every ATOM process on the box, which on a shared one is someone else's.
for pid in $(pgrep -f 'atom\.entrypoints' 2>/dev/null || true); do
    if owns_our_cards "$pid"; then
        echo "  killing server $pid and its children"
        kill_tree "$pid"
    fi
done
sleep 3

# 3. Verify GPU memory is actually free
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    USED_GPUS=$(owned_in_use)
    if [ "$USED_GPUS" -eq 0 ]; then
        echo "GPU memory clear after ${i}s"
        break
    fi
    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "WARNING: owned GPU memory still in use after ${MAX_WAIT}s:"
        if [ "$SMI" = rocm-smi ]; then
            rocm-smi --showpidgpus 2>&1 | grep "PID.*is using" | grep -v "0 DRM" || true
        else
            amd-smi process --csv 2>/dev/null | awk -F, '
                { sub(/\r$/, "") }
                NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
                ("vram_mem" in c) && $c["vram_mem"] + 0 > 0 {
                    print "GPU[" $1 "] PID " $c["pid"] " VRAM " $c["vram_mem"] " B"
                }' || true
        fi
        # Only what we can BOTH attribute and reach. The pids the smi tool prints
        # are the host's and mean something else in this container, so the
        # blanket `kill -9` this used to do could hit an unrelated local pid --
        # and on a shared box that memory may not be ours to reclaim at all.
        # Our own leftovers are the ones we can name.
        for pid in $(pgrep -f 'atom\.entrypoints|multiprocessing\.spawn' || true); do
            echo "  force-killing our own leftover pid $pid"
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 5
    fi
    sleep 1
done

# 4. Clear stale compile cache
rm -rf ~/.cache/atom/*

# 5. Reclaim ROCm core dumps from a previous fault: one 8-rank fault writes
# ~283GB and fills the disk. HSA_ENABLE_COREDUMP=0 does NOT prevent them
# (measured), so clearing here is the only thing that works. Anchored to the
# repo, since the dumps land next to the server's CWD, not the caller's.
REPO_ROOT="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
rm -rf "$REPO_ROOT"/gpucore.* ./gpucore.*

# Write config header to log (truncates old content).
# Inherited env vars are dumped explicitly so you never have to wonder
# whether ATOM_USE_TRITON_MOE / V4_USE_REF_QUANT / etc. were set.
{
echo "========================================"
echo " ATOM Server Launcher"
echo "========================================"
echo " Model:          $MODEL_PATH"
echo " TP Size:        $TP_SIZE"
echo " Port:           $PORT"
echo " KV Cache dtype: $KV_CACHE_DTYPE"
echo " Max num seqs:   $MAX_NUM_SEQS"
echo " GPU mem util:   $GPU_MEM_UTIL"
echo " Extra args:     ${EXTRA_ARGS[*]:-none}"
echo " Date:           $(date)"
echo "----------------------------------------"
echo " Inherited env vars (ATOM_*, V4_*, AITER_*, HSA_*, AMD_*, HIP_*):"
env | grep -E '^(ATOM_|V4_|AITER_|HSA_|AMD_|HIP_|KV_CACHE|MAX_NUM_SEQS|MAX_MODEL_LEN|MAX_BATCHED_TOKENS|GPU_MEM_UTIL)' \
  | sort | sed 's/^/   /' || echo "   (none set)"
echo "========================================"
} | tee "$LOG_FILE"

python -m atom.entrypoints.openai_server \
    --model "$MODEL_PATH" \
    --kv_cache_dtype "$KV_CACHE_DTYPE" \
    -tp "$TP_SIZE" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --server-port "$PORT" \
    "${EXTRA_ARGS[@]}" \
    >> "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server started in background (PID: $SERVER_PID)"

# Wait for server ready with GPU verification
echo "Waiting for server to be ready..."
for i in $(seq 1 600); do
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        VRAM_COUNT=$(owned_in_use)
        if [ "$VRAM_COUNT" -gt 0 ]; then
            echo "Server is ready! (PID: $SERVER_PID, GPU VRAM loaded)"
            exit 0
        fi
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died. Check $LOG_FILE"
        exit 1
    fi
    if [ "$i" -eq 600 ]; then
        echo "ERROR: Server not ready after 600s (10 min)"
        exit 1
    fi
    sleep 1
done
