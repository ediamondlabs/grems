#!/bin/bash
# Train the EnSplit DDQN baseline (1500 episodes). Released checkpoint: checkpoints/ensplit/ensplit_best.pt
# Without a checkpoint, evaluate_GReMS.py falls back to a brute-force optimal EnSplit split.
set -e
source "$(dirname "$0")/_common.sh"
$PYTHON train_ensplit.py \
    --episodes 1500 \
    --eps-decay-episodes 1100 \
    --seed 42 \
    --checkpoint-dir "${CHECKPOINT_DIR:-checkpoints/ensplit}"
