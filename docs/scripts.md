# Scripts reference

All scripts live in `scripts/`, can be started from any directory and change to the repository root themselves.
Options are passed as environment variables.

| Script | Purpose | Options |
|--------|---------|---------|
| `setup_networks.sh [topo ...]` | Extract topology archives from `data/networks/` | topologies: `rgg ws er ba ba_m1 ba_m3 ba_m4` |
| `train_stage1.sh` | Stage 1 training, N = 4 to 10, 30M steps | `DEVICE` (cuda), `TIMESTEPS`, `RUN_ID` (GReMS_FINAL), `SEED` (42) |
| `train_finetune.sh` | Stage 2 fine tuning, N = 4 to 20, 10M steps | `DEVICE`, `TIMESTEPS`, `RUN_ID` (GReMS_Final_Finetune), `SOURCE_RUN_ID` (GReMS_FINAL), `SEED` |
| `evaluate.sh [MODEL] [TOPO] [RUN_ID]` | Multi seed evaluation | `SEEDS`, `BASELINES` (0/1), `DEVICE`, `SIZES` (`small ood large`), `NUM_SEQ`, `NUM_STEPS` |
| `train_ensplit.sh` | Train the EnSplit DDQN baseline | `CHECKPOINT_DIR` |
| `summarize_results.py FILES...` | Aggregate result JSONs over seeds into a table | `--csv out.csv` |

Set `PYTHON=/path/to/python` to choose the interpreter.

## Size groups in `evaluate.sh`

| Group | Node counts | Note |
|-------|-------------|------|
| `small` | 3 to 10 | seen sizes for both checkpoints (3 was never trained on) |
| `ood` | 11 to 19 | unseen by the Stage 1 model, trained for the fine tuned model |
| `large` | 20 | the largest supported size |

The label "ood" refers to the Stage 1 model.
For the Stage 2 model these sizes are in the training range, the test networks themselves are still held out.

## Direct use of the Python entry points

Equivalent explicit command for the most common evaluation:

```bash
python evaluate_GReMS.py \
    --model checkpoints/checkpoints/variable_networks_GReMS_Final_Finetune/grems_final.pt \
    --run-id GReMS_Final_Finetune --topology-tag rgg \
    --num-sequences 10 --num-steps 50 --seed 0 --offline --device cpu \
    --save-json results/GReMS_Final_Finetune_rgg.json
```

`python evaluate_GReMS.py --help` and `python train_GReMS.py --help` list every flag. The main ones:

| Flag | Meaning |
|------|---------|
| `--topology-tag` | topology family to evaluate: `rgg`, `ws`, `er`, `ba` |
| `--num-sequences`, `--num-steps` | trajectories per network and steps per trajectory |
| `--warmup-steps` | channel advances before recording starts (default 0) |
| `--no-baselines`, `--no-disnet`, `--no-modnn` | skip baselines for speed |
| `--ensplit-checkpoint` | use the trained DDQN instead of brute force EnSplit |
| `--no-small`, `--no-ood-size`, `--no-large` | skip a size group |
| `--networks-dir` | evaluate on a custom topology folder |
| `--offline` | use `data/offline_profiles/layer_profiles.json` (the paper runs use it) |
| `--resume`, `--resume-step` (train) | continue or fine tune from a checkpoint, reset the schedule |
| `--large-graph-size`, `--small-graph-size` (train) | oversample graphs at least or at most this size |
| `--topology-models` (train) | one or several of `rgg ba ws er` for mixed topology training |
