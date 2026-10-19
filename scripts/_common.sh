#!/bin/bash
# Shared preamble: run from the repository root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p logs results
# Optional: pick a Python interpreter, e.g. PYTHON=python3.11 bash scripts/train_stage1.sh
PYTHON="${PYTHON:-python3}"
