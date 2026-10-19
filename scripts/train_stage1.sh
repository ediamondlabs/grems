#!/bin/bash
# Stage 1: train GReMS from scratch on RGG networks with N = 4-10 nodes (30M steps).
# Produces the checkpoint released as  checkpoints/checkpoints/variable_networks_GReMS_FINAL/
#
# Usage: DEVICE=cuda bash scripts/train_stage1.sh            # full run (~1-2 days on one GPU)
#        DEVICE=cpu TIMESTEPS=20000 RUN_ID=smoke bash scripts/train_stage1.sh   # smoke test
#
# Re-running the same command resumes from grems_latest.pt automatically.
set -e
source "$(dirname "$0")/_common.sh"

DEVICE="${DEVICE:-cuda}"
TIMESTEPS="${TIMESTEPS:-30000000}"
RUN_ID="${RUN_ID:-GReMS_FINAL}"
SEED="${SEED:-42}"

export VARIABLE_TRAIN_SIZES="4,5,6,7,8,9,10"

# NOTE: train_GReMS.py prepends CHECKPOINT_DIR (= checkpoints/) to these paths,
# which is why released checkpoints live under checkpoints/checkpoints/.
CKPT_DIR="variable_networks_${RUN_ID}"
LATEST="checkpoints/checkpoints/${CKPT_DIR}/grems_latest.pt"
RESUME=()
if [ -f "$LATEST" ]; then
    echo "Resuming from $LATEST"
    RESUME=(--resume "$LATEST")
fi

$PYTHON train_GReMS.py \
    --run-id "$RUN_ID" \
    --variable-networks \
    --topology-models rgg \
    --timesteps "$TIMESTEPS" \
    --rollout 512 \
    --batch-size 512 \
    --lr 3e-4 \
    --entropy-coeff 0.02 \
    --clip-eps 0.2 \
    --eval-interval 200000 \
    --seed "$SEED" \
    --device "$DEVICE" \
    --offline \
    --best-model-path  "checkpoints/${CKPT_DIR}/grems_best.pt" \
    --final-model-path "checkpoints/${CKPT_DIR}/grems_final.pt" \
    "${RESUME[@]}"
