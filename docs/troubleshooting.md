# Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `Networks directory not found: .../variable_networks_rgg` | Topologies are not extracted. Run `bash scripts/setup_networks.sh`. |
| `Checkpoint not found` | Wrong path. Released checkpoints are in `checkpoints/checkpoints/variable_networks_<run>/` (note the doubled `checkpoints`). |
| `FileNotFoundError` for `data/offline_profiles/layer_profiles.json` | The file is needed by `--offline` (used by the paper scripts) and is tracked in git. Restore it with `git checkout data/offline_profiles/layer_profiles.json`. |
| `ModuleNotFoundError: No module named 'networkx'` (or torch, gymnasium) | Install requirements in the active environment: `pip install -r requirements.txt`. |
| `RuntimeError: ... size mismatch for ...` when loading a checkpoint | The code constants differ from the ones used for training (`MAX_NODES`, `MAX_GROUPS`, observation layout). Use the code revision that matches the checkpoint. |
| `torch.load` fails with `weights_only` or pickle errors | Checkpoints contain more than tensors. The code passes `weights_only=False`. Do this only for checkpoints you trust. |
| Evaluation is very slow | Baselines dominate the run time, especially DiSNet and MoDNN above 10 nodes. Use `BASELINES=0` for GReMS only, or `SIZES=small`. |
| Messages about baseline timeouts | The machine is too slow or busy for the per step limits. Free CPU cores or re-run. The timed out network is evaluated without that baseline. |
| Training: `CUDA out of memory` | Use `--device cpu` for a smoke test, or reduce `--batch-size`. The reference runs used a 16 GB GPU. |
| Training restarts from the beginning | The checkpoint directory differs between runs. The scripts resume from `checkpoints/checkpoints/variable_networks_<RUN_ID>/grems_latest.pt` so keep `RUN_ID` constant. |
| Fine tuning fails with connectivity penalty explosion | The curriculum was reset. Set `CONN_PENALTY_START=50` as `train_finetune.sh` does. |
| `ImportError` about `network`, `utils`, `inference` in a baseline | The vendored DiSNet code was removed or moved. Keep `third_party/disnet/` intact. |
| MkDocs: `mermaid` fences are shown as code | Use `mkdocs-material` and keep the `pymdownx.superfences` block of `mkdocs.yml`. |

If none of this helps, open an issue with the output of `python --version`, `pip freeze`, the exact command and the last 50 lines of `logs/*.log`.
