#!/bin/bash
# The whole spike, inside the rl-92-pd container: five runs per order, each on
# a freshly started Qwen3-1.7B-Base pair, then fresh Qwen3-1.7B and
# Qwen3-1.7B-Base pairs' references.
#   spike.sh <out_dir> [runs_per_order]
set -uo pipefail
# rocm/atom-dev:nightly_202611081546 lacks jsonschema, which atom imports.
python -c 'import jsonschema' 2>/dev/null || pip install -q jsonschema || exit 1
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
for side in new old; do
  [[ $side == new ]] && ckpt=$NEW || ckpt=$OLD
  "$HERE/pair.sh" down
  "$HERE/pair.sh" up "$ckpt" "$OUT/logs/ref-$side"
  python "$HERE/drive.py" reference "$OUT"/run-*.json --out "$OUT/ref-$side.json"
done
"$HERE/pair.sh" down
python "$HERE/drive.py" table "$OUT/ref-new.json" "$OUT/ref-old.json" "$OUT"/run-*.json
