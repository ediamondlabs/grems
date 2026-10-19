# Extending GReMS

This page lists the usual extension points and the files to touch. Always re-run the smoke test of the [Quick start](quickstart.md) after a change.

## Use a different DNN than VGG-16

1. Describe the model in `config.py`: replace `MODEL_CFG`, `LAYER_FWD_MS`, `LAYER_KERNEL_CFG` and `LAYER_SPATIAL_OUT`. `NUM_LAYERS` follows from `MODEL_CFG`.
2. Check `state_encoder.py` (blocks 4 and 8 depend on the number of layers, so `state_dim()` changes) and `action_space.py` (the vertical weight decoder).
3. `run_rl_live.py` assumes a layer sequence of spatial layers followed by fully connected layers. Other topologies (residual connections) need a new dependency DAG.
4. The baselines hard code VGG-16 layer ranges (`run_disnet_live.py`, `run_modnn_live.py`). Update or disable them.
5. Retrain from scratch. Checkpoints are not transferable.

## Change the reward

Edit `reward.py` for the main term and the shaping block in `SplitEnv.step()` (`environment.py`). Penalty magnitudes are in `config.py`. When the scale of the reward changes keep the online normaliser in `train_GReMS.py` (`RunningMeanStd`).

## Add a channel model

`network_dynamics.py` contains the two models behind a small interface (`advance()`, getters for load or bandwidth, `resize()`). Implement the same methods in a new class and instantiate it in `SplitEnv.__init__` and `SplitEnv.reset()`. The observation already contains the effective load and bandwidth, so no encoder change is needed unless you add new state variables.

## Add a topology family

Add a generator in `generate_variable_networks.py` (see `_make_ba_graph` and the others), register its name in `SUPPORTED_TOPOLOGY_MODELS`, generate files with `--topology-model <name>`, and add the directory to the maps `_dir_map` in `evaluate_GReMS.py` and `_topo_dir_map` in `train_GReMS.py`. Each node needs a `weight` attribute and each edge a `weight` (bandwidth), as in the existing files.

## Add a baseline

Create `run_<name>_live.py` with a function that takes a NetworkX graph with the same attributes and returns a dictionary with `latency_ms`. Call it from `evaluate_one_network` in `evaluate_GReMS.py` next to the existing baselines and add its latency to the step record and to `_aggregate_by_size`.

## Plug in a real latency measurement

`measure_rl_plan_latency_ms` in `run_rl_live.py` takes a plan plus node and link state and returns a latency. Replacing it by lookups from measurements on your hardware is the easiest way to calibrate to a new testbed. Keep the signature and the plan keys unchanged and everything else works.

## Change the network architecture

`gnn_model.py` defines `PhysicalNetworkGNN` (the backbone) and `GNNActorCritic` (heads). `build_model` is the single construction point used by training and evaluation. If you change the parameter shapes, old checkpoints will not load.

## Coding conventions

- Python modules stay flat in the root, imports are plain (`from config import X`).
- Constants go to `config.py`, tunable ones read an environment variable.
- New experiments get a script in `scripts/` that sources `_common.sh` and writes to `results/`.
- Document new settings in [Configuration reference](configuration.md) and new commands in [Scripts reference](scripts.md).
