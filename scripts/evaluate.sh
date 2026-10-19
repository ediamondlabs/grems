#!/bin/bash
# Multi-seed dynamic evaluation of a GReMS checkpoint on the held-out test networks.
#
# Usage: bash scripts/evaluate.sh [MODEL] [TOPOLOGY] [RUN_ID]
#   MODEL     default checkpoints/checkpoints/variable_networks_GReMS_Final_Finetune/grems_final.pt
#   TOPOLOGY  rgg | ws | er | ba          (default rgg)
#   RUN_ID    label used in output file names (default GReMS_Final_Finetune)
#
# Environment knobs:
#   SEEDS="0 100 200 300 400"   evaluation seeds (paper: 5 seeds)
#   BASELINES=1                 also run DiSNet, MoDNN and EnSplit (default 0 = GReMS only, faster)
#   DEVICE=cpu|cuda             (default cpu, evaluation is dominated by baseline/env code)
#   SIZES="small ood large"     which size groups to run (default all three)
#   NUM_SEQ=10 NUM_STEPS=50     trajectories per network and steps per trajectory (paper values)
#
# Outputs: results/<RUN_ID>_<TOPOLOGY>[_s<seed>].json  (seed 0 has no suffix)
# Summarise them with:  python scripts/summarize_results.py results/<RUN_ID>_<TOPOLOGY>*.json
set -e
source "$(dirname "$0")/_common.sh"

MODEL="${1:-checkpoints/checkpoints/variable_networks_GReMS_Final_Finetune/grems_final.pt}"
TOPO="${2:-rgg}"
RUN_ID="${3:-GReMS_Final_Finetune}"
SEEDS="${SEEDS:-0 100 200 300 400}"
BASELINES="${BASELINES:-0}"
DEVICE="${DEVICE:-cpu}"
SIZES="${SIZES:-small ood large}"
NUM_SEQ="${NUM_SEQ:-10}"
NUM_STEPS="${NUM_STEPS:-50}"

[ -f "$MODEL" ] || { echo "ERROR: checkpoint not found: $MODEL" >&2; exit 1; }
[ -d "variable_networks_${TOPO}" ] || [ "$TOPO" = "ba" ] || bash scripts/setup_networks.sh "$TOPO"
[ "$TOPO" != "ba" ] || [ -d variable_networks ] || bash scripts/setup_networks.sh ba

FLAGS=()
[ "$BASELINES" = "1" ] || FLAGS+=(--no-baselines)
[[ " $SIZES " == *" small "* ]] || FLAGS+=(--no-small)
[[ " $SIZES " == *" ood "*   ]] || FLAGS+=(--no-ood-size)
[[ " $SIZES " == *" large "* ]] || FLAGS+=(--no-large)

for SEED in $SEEDS; do
    SUFFIX=""; [ "$SEED" -gt 0 ] && SUFFIX="_s${SEED}"
    echo "=== seed ${SEED} ==="
    $PYTHON evaluate_GReMS.py \
        --model "$MODEL" \
        --run-id "$RUN_ID" \
        --topology-tag "$TOPO" \
        --num-sequences "$NUM_SEQ" \
        --num-steps "$NUM_STEPS" \
        --frozen-warmup 0 \
        --seed "$SEED" \
        --device "$DEVICE" \
        --offline \
        --save-json "results/${RUN_ID}_${TOPO}${SUFFIX}.json" \
        "${FLAGS[@]}"
done
