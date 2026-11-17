#!/bin/bash
# The whole spike, inside the rl-92-pd container: five runs per order, each on
# a freshly started Qwen3-1.7B-Base pair, then the fresh Qwen3-1.7B reference.
#   spike.sh <out_dir> [runs_per_order]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1 RUNS=${2:-5}
HUB=${HF_HOME:-/share_nfs/junyan_tmp_r120/hf}/hub
OLD=$(ls -d "$HUB"/models--Qwen--Qwen3-1.7B-Base/snapshots/*)
NEW=$(ls -d "$HUB"/models--Qwen--Qwen3-1.7B/snapshots/*)
mkdir -p "$OUT"
for i in $(seq "$RUNS"); do
  for order in p d both; do
    tag=$order-$i
    "$HERE/pair.sh" down
    "$HERE/pair.sh" up "$OLD" "$OUT/logs/$tag" || { echo "$tag: pair did not start"; continue; }
    python "$HERE/drive.py" run --order "$order" --new "$NEW" --out "$OUT/run-$tag.json"
  done
done
"$HERE/pair.sh" down
"$HERE/pair.sh" up "$NEW" "$OUT/logs/reference"
python "$HERE/drive.py" reference "$OUT"/run-*.json --out "$OUT/ref.json"
"$HERE/pair.sh" down
python "$HERE/drive.py" table "$OUT/ref.json" "$OUT"/run-*.json
