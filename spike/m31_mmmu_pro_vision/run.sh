#!/usr/bin/env bash
# One MMMU-Pro vision run of MiniMax-M3.1-Flash on direct ATOM, one factor
# changed from the A2 check's run at a time:
#
#   bash spike/m31_mmmu_pro_vision/run.sh <variant> <slot>
#
#   nightly       A2's run: the checkout's ATOM in the nightly image, lmms-eval's request
#   recipe-image  the recipe's image rocm/atom-dev:minimax-m31-flash with its own ATOM
#   prompt        as nightly, the request carrying MMMU-Pro's vision prompt
#
# slot 0 takes GPUs 0-3, slot 1 GPUs 4-7 of NODE (default pit2-p03-g07), so two
# runs share a node. The deployment and the client come from PR #35's launch
# library and m31 profile at HARNESS_REF. Prints the run's JSON line (score.py's
# fields); logs and samples stay in the run directory.
set -euo pipefail

if (($# != 2)); then
  echo "usage: $0 <nightly|recipe-image|prompt> <0|1>" >&2
  exit 2
fi
VARIANT=$1 SLOT=$2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
HARNESS_REF=${HARNESS_REF:-f8aec8e0c1aa66f1ba8b29de47a95b73b7d3997e}
NODE=${NODE:-pit2-p03-g07}
SPIKE_DIR=${SPIKE_DIR:-/share_nfs/junyan_tmp_r120/accept/spike47}
NIGHTLY=rocm/atom-dev:nightly_202611081546
RECIPE=rocm/atom-dev:minimax-m31-flash

case $VARIANT in
  nightly | recipe-image) PROMPT=lmms ;;
  prompt) PROMPT=official ;;
  *) echo "unknown variant $VARIANT" >&2; exit 2 ;;
esac
case $SLOT in
  0) ATOM_GPUS=0,1,2,3 ;;
  1) ATOM_GPUS=4,5,6,7 ;;
  *) echo "slot is 0 or 1" >&2; exit 2 ;;
esac

# PR #35's harness, unpacked once per ref.
HARNESS=$SPIKE_DIR/harness-${HARNESS_REF:0:12}
if [[ ! -d $HARNESS ]]; then
  mkdir -p "$HARNESS.tmp"
  git -C "$ROOT" archive "$HARNESS_REF" tests/accept/atomesh_mm | tar -x -C "$HARNESS.tmp"
  mv "$HARNESS.tmp" "$HARNESS"
fi
LIB=$HARNESS/tests/accept/atomesh_mm

ATOM_NODE=$NODE CLIENT_NODE=$NODE MESH_NODE=$NODE PORT_BASE=$((18600 + 100 * SLOT))
source "$LIB/models/m31.env"
source "$LIB/lib.sh"
ACCEPT_ISSUE=47-s$SLOT # containers epd-47-s<slot>-atom, epd-47-s<slot>-client

RUN_DIR=$SPIKE_DIR/runs/$VARIANT.s$SLOT.$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN_DIR"
log "run directory $RUN_DIR"
trap cleanup_containers EXIT

if [[ $VARIANT == recipe-image ]]; then
  # The image's own ATOM (/app/ATOM), as the recipe ran it, not the checkout.
  IMAGE=$RECIPE
  start_container atom
  env_opts=()
  for e in "${ATOM_ENV[@]}"; do env_opts+=(-e "$e"); done
  cmd=$(printf '%q ' python3 -m atom.entrypoints.openai_server "${ATOM_ARGS[@]}" \
    --server-port "$ATOM_PORT" --port "$ATOM_ENGINE_PORT")
  in_role atom -d "${env_opts[@]}" -- bash -c "cd /app/ATOM && $cmd > $RUN_DIR/atom.log 2>&1"
  log "waiting for atom (log $RUN_DIR/atom.log)"
  wait_http atom "http://127.0.0.1:$ATOM_PORT/health" 3600 atom.entrypoints.openai_server
else
  IMAGE=$NIGHTLY
  launch_atom atom "$ATOM_PORT" "$ATOM_ENGINE_PORT"
fi
in_role atom pip list 2>/dev/null | grep -E '^(torch|transformers|aiter|amd-aiter) ' >"$RUN_DIR/versions.txt" || true

IMAGE=$NIGHTLY
start_client
in_role client -e "ENDPOINT=$(role_url atom "$ATOM_PORT")" -e "SERVED_MODEL=$SERVED_MODEL" \
  -e "RUN_DIR=$RUN_DIR" -e "VARIANT=$VARIANT" -e "PROMPT=$PROMPT" -e "HARNESS_CHECKS=$LIB/checks" \
  -e "LIMIT=${LIMIT:-}" -e "LMMS_EVAL_SRC=$ACCEPT_DIR/lmms-eval" -- python3 "$HERE/vision.py"
