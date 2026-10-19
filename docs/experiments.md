# Experiments and outputs

## Main evaluation (`evaluate_GReMS.py`)

Output: one JSON per seed, `results/<RUN_ID>_<topology>[_s<seed>].json`. The files can be large (hundreds of MB for a full 3 to 20 node sweep) because they store every step of every trajectory.

Structure:

| Key | Content |
|-----|---------|
| `run_id`, `topology_tag`, `model_path`, `num_sequences`, `num_steps`, `obs_dim` | run description |
| `networks.<name>.sequences[i].steps[t]` | per step latencies `rl_lat_ms`, `frozen_lat_ms`, `disnet_lat_ms`, `modnn_lat_ms`, `ensplit_lat_ms`, channel statistics (`mean_load`, `frac_high`, `node_loads`, `link_bws_mbit`), plan (`hub_node`, `worker_nodes`) and `connected` |
| `size_summary.<N>` | per node count: `rl_mean`, `rl_std`, `disnet_mean`, `modnn_mean`, `ensplit_mean`, connectivity rate, win rates `rl_beat_*` |
| `step_curves` | latency as a function of the step index, averaged over networks |

`frozen_lat_ms` is the latency of the plan chosen at step 0 and kept while the channel changes. It shows the value of adapting.
`connected` is false if the plan uses a missing link. `rl_mean_conn` and the `*_conn` fields restrict the statistics to connected plans.

Aggregate seeds with `scripts/summarize_results.py`.

## Generalisation to other topology families

The same checkpoint is evaluated on WS, ER and BA test networks without any retraining:

```bash
for t in ws er ba; do bash scripts/evaluate.sh checkpoints/checkpoints/variable_networks_GReMS_Final_Finetune/grems_final.pt $t; done
```

Extra BA test sets with different attachment parameters m = 1, 3, 4 are in `data/networks/variable_networks_ba_m{1,3,4}.tar.gz` (test networks only). Evaluate them with `--networks-dir variable_networks_ba_m1 --topology-tag ba` in `evaluate_GReMS.py`.

## EnSplit baseline (`train_ensplit.py`)

Output: `checkpoints/ensplit/ensplit_best.pt`, `ensplit_final.pt` and `learning_curve.csv`.
