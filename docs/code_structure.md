# Code structure

The Python modules live flat in the repository root (they import each other by plain module name, for example `from config import ...`), so run every script from the repository root.

## Map

```
GReMS/
├── scripts/                 Reproducible entry points (shell) and result summariser
├── docs/, mkdocs.yml        This documentation
├── checkpoints/             Released weights and learning curves
├── data/
│   ├── networks/            Compressed topology archives (train and test graphs)
│   ├── offline_profiles/    layer_profiles.json, used with --offline
│   └── baselines.json       Original DiSNet testbed reference numbers (not used by the code)
├── third_party/disnet/      Vendored DiSNet code for the DiSNet and MoDNN baselines
└── *.py                     Core code, see below
```

Directories created at run time and not tracked by git: `variable_networks*/` (extracted topologies), `results/` (evaluation JSON), `plots/`, `logs/`.

## Core modules

| File | Role | Key objects |
|------|------|-------------|
| `config.py` | Single source of truth for constants, the VGG-16 description, layer timings, penalties, PPO and channel parameters. Several values are overridable by environment variables. | `MODEL_CFG`, `LAYER_IOT_FWD_MS`, `MAX_NODES`, `GE_*`, `NODE_UTIL_BONUS` |
| `environment.py` | Gymnasium environment. `reset()` picks a network and advances the channels, `step()` decodes the action, computes latency and shaped reward. | `SplitEnv` |
| `network_dynamics.py` | Channel models. | `ComputeLoadModel` (MMPP), `LinkBWModel` (Gilbert-Elliott) |
| `state_encoder.py` | Converts environment state to the 542 dimensional observation. | `encode_state`, `state_dim` |
| `action_space.py` | Action specification, decoding to a plan, connectivity and weak link penalties. | `ActionSpec`, `decode_action`, `compute_connectivity_penalty` |
| `gnn_model.py` | GraphSAGE actor critic with five heads. | `GNNActorCritic`, `build_model` |
| `reward.py` | `-ALPHA * log(latency_ms)`. | `compute_reward` |
| `run_rl_live.py` | Analytical latency model for a GReMS plan, including multi hop routing. | `measure_rl_plan_latency_ms` |
| `profiling.py` | Layer profiles, and conversion of a NetworkX topology into node and link profiles. | `generate_profiles_from_disnet_graph`, `load_layer_profiles` |
| `generate_variable_networks.py` | Random topology generator and path helper. | `generate_all`, `get_variable_network_paths` |
| `utils.py` | Logging, seeding, JSON helpers. | `set_seed`, `setup_logging` |

## Training and evaluation

| File | Role |
|------|------|
| `train_GReMS.py` | PPO training loop with GAE, reward normalisation, curriculum, resume and optional W&B. Entry point: `python train_GReMS.py --help`. |
| `evaluate_GReMS.py` | Dynamic multi sequence evaluator. Runs GReMS and the baselines on the same channel trajectories and writes a JSON per run. |

## Baselines

| File | Role |
|------|------|
| `run_disnet_live.py` | DiSNet plan and latency under the shared latency tables |
| `run_modnn_live.py` | MoDNN plan and latency |
| `run_ensplit_live.py` | EnSplit (brute force or DDQN) |
| `ensplit_ddqn.py`, `ensplit_env.py`, `train_ensplit.py` | EnSplit DDQN agent, environment and training script |
| `third_party/disnet/` | Original DiSNet functions used by the first two |

## Data flow in one training step

1. `train_GReMS.py` samples an environment from the pool (a `SplitEnv` bound to one topology file).
2. `SplitEnv.reset()` advances `ComputeLoadModel` and `LinkBWModel` and builds node and link profiles via `profiling.py`.
3. `state_encoder.encode_state()` produces the 542 dimensional observation.
4. `GNNActorCritic.act()` samples an action, `action_space.decode_action()` turns it into a plan.
5. `run_rl_live.measure_rl_plan_latency_ms()` returns the latency, `reward.compute_reward()` and the penalties in `environment.py` return the reward.
6. After 512 steps `ppo_update()` runs 4 epochs of clipped PPO.

## Conventions

- Latency is in milliseconds, bandwidth in Mbit/s in topology files and MB/s inside the latency model.
- A "group" is a vertical stage. A "chunk" is a horizontal slice of one group. A "worker" is a node that runs one chunk.
- Node indices are 0 based and refer to the NetworkX graph. Padded nodes have index at least `N`.
- Plans are dictionaries with keys `group_sizes`, `chunks`, `hub_node`, `assignment`.
