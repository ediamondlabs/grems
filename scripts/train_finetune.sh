#!/bin/bash
# Stage 2: fine-tune the Stage-1 policy on N = 4-20 nodes (10M steps, 2x sampling weight on N >= 11).
# Produces the checkpoint released as  checkpoints/checkpoints/variable_networks_GReMS_Final_Finetune/
#
# Usage: DEVICE=cuda bash scripts/train_finetune.sh
#        SOURCE_RUN_ID=smoke DEVICE=cpu TIMESTEPS=20000 RUN_ID=smoke_ft bash scripts/train_finetune.sh
#
# Re-running the same command resumes the fine-tune from its own grems_latest.pt.
set -e
source "$(dirname "$0")/_common.sh"

DEVICE="${DEVICE:-cuda}"
TIMESTEPS="${TIMESTEPS:-10000000}"
RUN_ID="${RUN_ID:-GReMS_Final_Finetune}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-GReMS_FINAL}"
SEED="${SEED:-42}"

export VARIABLE_TRAIN_SIZES="4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"
# Stage 1 ended with the connectivity penalty fully ramped to 50. Do not restart
# the curriculum from 5, otherwise connectivity violations blow up (see docs).
export CONN_PENALTY_START=50.0

SOURCE="checkpoints/checkpoints/variable_networks_${SOURCE_RUN_ID}/grems_final.pt"
CKPT_DIR="variable_networks_${RUN_ID}"
LATEST="checkpoints/checkpoints/${CKPT_DIR}/grems_latest.pt"

if [ -f "$LATEST" ]; then
    echo "Resuming fine-tune from $LATEST"
    RESUME=(--resume "$LATEST")
elif [ -f "$SOURCE" ]; then
    echo "Starting fine-tune from $SOURCE"
    RESUME=(--resume "$SOURCE" --resume-step 0)
else
    echo "ERROR: $SOURCE not found. Run scripts/train_stage1.sh first (or use the released checkpoint)." >&2
    exit 1
fi

$PYTHON train_GReMS.py \
    --run-id "$RUN_ID" \
    --variable-networks \
    --topology-models rgg \
    --timesteps "$TIMESTEPS" \
    --rollout 512 \
    --batch-size 512 \
    --lr 3e-5 \
    --entropy-coeff 0.02 \
    --clip-eps 0.2 \
    --large-graph-size 11 \
    --eval-interval 200000 \
    --seed "$SEED" \
    --device "$DEVICE" \
    --offline \
    --best-model-path  "checkpoints/${CKPT_DIR}/grems_best.pt" \
    --final-model-path "checkpoints/${CKPT_DIR}/grems_final.pt" \
    "${RESUME[@]}"
