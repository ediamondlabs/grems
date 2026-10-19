# checkpoints/

```
checkpoints/
├── checkpoints/                                        (sic, see below)
│   ├── variable_networks_GReMS_FINAL/                  Stage 1: from scratch, N = 4 to 10, 30M steps
│   └── variable_networks_GReMS_Final_Finetune/         Stage 2: fine tuned, N = 4 to 20, 10M steps  <- headline model
└── ensplit/                                            EnSplit DDQN baseline
```

Each GReMS directory holds `grems_final.pt` (end of training), `grems_best.pt` (best by validation latency), `grems_latest.pt` (resume snapshot) and `learning_curve.csv`.
The doubled `checkpoints/checkpoints/` comes from `train_GReMS.py`, which prepends `CHECKPOINT_DIR` to the path given with `--best-model-path`.
Every `.pt` is a dictionary with the model weights under the key `model` (plus optimizer state, step counter and reward statistics for resuming). Load them with `torch.load(path, weights_only=False)` and only trust files from this repository.
