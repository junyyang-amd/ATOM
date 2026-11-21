#!/usr/bin/env bash
# DSpark acceptance of MiniMax-M3.1 on MMStar image chat, single instance or
# 1P1D, at this checkout's head.
#
#   bash spike/dspark_acceptance.sh <single|pd> [repeats]
#
# Brings the topology up through the EPD line's acceptance harness (its
# m31 profile: DSpark with 7 drafts, greedy decoding), with ATOM and ATOMesh
# from this checkout, then runs the harness's MMStar check <repeats> times
# (default 3) on that one deployment. Around each run it reads
# /debug/mtp_stats from the server that decodes (single: atom, pd: decode)
# and prints one JSON line per run: the MMStar score and the acceptance over
# that run alone (accepted / drafted, toks per verify step, the accepted
# count distribution and the per-position rate P(accepted >= k)).
#
# The harness is taken by `git archive` from HARNESS_REF (default the EPD
# commit it was validated at) into $SPIKE_DIR; nothing here is a test. The
# profile's variables (IMAGE, *_NODE, *_GPUS, PORT_BASE) override as in
# accept.sh.
set -euo pipefail

if (($# < 1 || $# > 2)); then
  echo "usage: $0 <single|pd> [repeats]" >&2
  exit 2
fi
TOPOLOGY=$1 REPEATS=${2:-3}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HARNESS_REF=${HARNESS_REF:-8e00a494c}
SPIKE_DIR=${SPIKE_DIR:-/share_nfs/junyan_tmp_r120/accept/spike-203}
sha=$(git -C "$ROOT" rev-parse --short "$HARNESS_REF")
HERE=$SPIKE_DIR/harness-$sha/tests/accept/atomesh_mm
if [[ ! -d $HERE ]]; then
  mkdir -p "$SPIKE_DIR/harness-$sha"
  git -C "$ROOT" archive "$sha" tests/accept/atomesh_mm | tar -x -C "$SPIKE_DIR/harness-$sha"
fi

MODEL=m31
ACCEPT_ISSUE=${ACCEPT_ISSUE:-203-$TOPOLOGY}
source "$HERE/models/$MODEL.env"
source "$HERE/lib.sh"
source "$HERE/topologies/$TOPOLOGY.sh"
case $TOPOLOGY in
  single) STATS_ROLE=atom STATS_PORT=$ATOM_PORT ;;
  pd) STATS_ROLE=decode STATS_PORT=$DECODE_PORT ;;
  *) echo "unknown topology $TOPOLOGY" >&2; exit 2 ;;
esac

RESULTS_DIR=$SPIKE_DIR/results
RUN_DIR=$RESULTS_DIR/$TOPOLOGY.$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN_DIR"
log "head $(git -C "$ROOT" rev-parse HEAD), harness $sha, image $IMAGE, run directory $RUN_DIR"
trap cleanup_containers EXIT

topology_up
start_client
stats() { in_role "$STATS_ROLE" curl -sf "http://127.0.0.1:$STATS_PORT/debug/mtp_stats"; }

for ((i = 1; i <= REPEATS; i++)); do
  run_dir=$RUN_DIR/run$i
  mkdir -p "$run_dir"
  before=$(stats)
  in_role client -e "ACCEPT_MODEL=$MODEL" -e "ACCEPT_TOPOLOGY=$TOPOLOGY" -e ACCEPT_CHECK=mmstar \
    -e "ENDPOINT=$ENDPOINT" -e "SERVED_MODEL=$SERVED_MODEL" -e "RUN_DIR=$run_dir" \
    -e "RESULTS_DIR=$run_dir" -e REF_TOPOLOGY= -e "LIMIT=${LIMIT:-}" \
    -- python3 "$HERE/checks/mmstar.py" >"$run_dir/check.out" || true
  after=$(stats)
  python3 - "$TOPOLOGY" "$i" "$before" "$after" "$run_dir/check.out" <<'EOF'
import json
import sys

topology, run, before, after, check_out = sys.argv[1:]
before, after = json.loads(before), json.loads(after)
check = json.loads(open(check_out).readline())
dist = {
    int(k): v - before["distribution"].get(k, 0)
    for k, v in after["distribution"].items()
}
steps = sum(dist.values())
drafted = after["total_draft_tokens"] - before["total_draft_tokens"]
accepted = after["total_accepted_tokens"] - before["total_accepted_tokens"]
k_max = max(dist)
print(json.dumps({
    "topology": topology,
    "run": int(run),
    "mmstar": check.get("score"),
    "mmstar_stderr": check.get("stderr"),
    "failed_requests": check.get("failed_requests"),
    "acceptance": accepted / drafted,
    "toks_per_step": 1 + accepted / steps,
    "verify_steps": steps,
    "accepted": accepted,
    "drafted": drafted,
    "distribution": {k: round(dist[k] / steps, 4) for k in sorted(dist)},
    "per_position": [
        round(sum(n for a, n in dist.items() if a >= k) / steps, 4)
        for k in range(1, k_max + 1)
    ],
}))
EOF
done
