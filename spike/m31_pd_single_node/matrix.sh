#!/usr/bin/env bash
# Every cell of the spike, one after another, from the repository root:
#   ATOMESH=<atomesh built from this tree> OUT=<dir> bash spike/m31_pd_single_node/matrix.sh
# Cells: TOPO DSPARK MC_DISABLE_HIP REPS.
set -uo pipefail
OUT=${OUT:?output directory}
for cell in "pd 1 1 3" "pd 0 1 3" "single 1 1 3" "single 0 1 3" "pd 1 0 1"; do
  set -- $cell
  echo "=== TOPO=$1 DSPARK=$2 MC_DISABLE_HIP=$3 REPS=$4"
  TOPO=$1 DSPARK=$2 MC_DISABLE_HIP=$3 REPS=$4 OUT=$OUT/$1-dspark$2-hip$3 \
    bash spike/m31_pd_single_node/run.sh 2>&1 | grep -E '^\{|^atom from|Error'
done
