# E/P/D: the PD deployment of topologies/pd.sh plus an encoder-only ATOM
# server (ENCODE_NODE/GPUS, ENCODE_ARGS) that ATOMesh sends each image
# request to before P; P reads the image embeddings from E over Mooncake on
# the HCAs of hca/<node>.map, as it moves KV to D. Checked against pd.
#
# Beside each role's HCA byte counters, the JSON line carries how many image
# requests reached P (image_requests), how many of them E answered
# (encode_served), how many went to P without E's embeddings
# (encode_fallbacks), and how many reads of E's embeddings P's ranks gave up
# for their own vision tower (read_fallbacks, one per rank and request). The
# run fails when E answered none, when every read P made fell back, or when
# E's HCAs sent no RDMA bytes: P then read the embeddings off RDMA, or not at
# all.

source "$HERE/topologies/pd.sh"
REF_TOPOLOGY=pd
HCA_ROLES=(encode prefill decode)

launch_encode() {
  local ATOM_ARGS=("${ENCODE_ARGS[@]}")
  launch_rdma encode "$ENCODE_PORT" "$ENCODE_ENGINE_PORT"
}

topology_up() {
  launch_encode
  launch_pd prefill kv_producer "$PREFILL_PORT" "$PREFILL_ENGINE_PORT"
  launch_pd decode kv_consumer "$DECODE_PORT" "$DECODE_ENGINE_PORT"
  launch_mesh mesh "$MESH_PORT" --backend atom --pd-disaggregation --policy cache_aware \
    --request-timeout-secs 14400 --encode "$(role_url encode "$ENCODE_PORT")" \
    --prefill "$(role_url prefill "$PREFILL_PORT")" --decode "$(role_url decode "$DECODE_PORT")"
  ENDPOINT=$(role_url mesh "$MESH_PORT")
}

# topology_findings <counter deltas>: the counts above, read from the logs in
# RUN_DIR, and the run's failures.
topology_findings() {
  local ranks
  IFS=, read -ra ranks <<<"$PREFILL_GPUS"
  python3 - "$1" "${#ranks[@]}" "$RUN_DIR" <<'EOF'
import json
import re
import sys
from pathlib import Path

deltas, ranks, run_dir = json.loads(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
ansi = re.compile(r"\x1b\[[0-9;]*m")


def lines(log):
    return ansi.sub("", (run_dir / log).read_text(errors="replace")).splitlines()


# ATOMesh logs a routing decision per pool and request, with the number of
# media in the request.
image_requests = sum(
    "routing decision" in line
    and 'pool="prefill"' in line
    and not re.search(r"media_digests=0\b", line)
    for line in lines("mesh.log")
)
served = sum(
    '"POST /v1/chat/completions HTTP/1.1" 200' in line for line in lines("encode.log")
)
read_fallbacks = sum(
    "are unavailable; running the vision tower" in line
    for line in lines("prefill.log")
)
failures = []
if not served:
    failures.append("E served no request")
elif read_fallbacks >= served * ranks:
    failures.append(f"every read of E's embeddings fell back ({read_fallbacks})")
if served and not deltas["encode_hca_tx_bytes"]:
    failures.append("E's HCAs sent no RDMA bytes: the embeddings left RDMA")
print(
    json.dumps(
        {
            "image_requests": image_requests,
            "encode_served": served,
            "encode_fallbacks": image_requests - served,
            "read_fallbacks": read_fallbacks,
            "failures": failures,
        }
    )
)
EOF
}
