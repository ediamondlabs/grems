# GReMS: Graph-Reinforced Neural Micro-Splitting over Dynamic Heterogeneous IoT Networks

GReMS is a reinforcement learning agent that decides how to split a deep neural network (VGG-16) across a wireless network of heterogeneous IoT and edge devices.
For every inference request it chooses **where** to cut the model (vertical split into pipeline stages), **how** to parallelise each stage (horizontal split across workers), which node acts as the **hub**, and which physical node fills each worker role.
The goal is to minimise end-to-end inference latency while compute load and link bandwidth change over time.
A single trained policy handles topologies from 3 to 20 nodes.

It is compared against three baselines: **DiSNet**, **MoDNN** and **EnSplit**.

📖 **Full documentation:** in [docs/](docs/index.md), rendered as a website with MkDocs (see [Documentation](#documentation)).

---

## Quick start

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Unpack the exact train/test topologies used in the paper
bash scripts/setup_networks.sh

# 3. Evaluate the released checkpoint (a few minutes on a laptop CPU)
SEEDS="0" SIZES=small BASELINES=1 NUM_SEQ=1 NUM_STEPS=5 bash scripts/evaluate.sh
python scripts/summarize_results.py results/GReMS_Final_Finetune_rgg.json
```

This prints a table with the mean latency of GReMS, DiSNet, MoDNN and EnSplit for every network size.
Dropping the `NUM_*` and `SIZES` overrides runs the full protocol of the paper (10 trajectories of 50 steps, 5 seeds).

## Reproducing the paper

The step by step guide is in **[docs/reproducibility.md](docs/reproducibility.md)**.
In short:

| Goal | Command | Cost |
|------|---------|------|
| Re-evaluate released checkpoints | `bash scripts/evaluate.sh` | hours on CPU |
| Re-train Stage 1 (N = 4 to 10, 30M steps) | `DEVICE=cuda bash scripts/train_stage1.sh` | about 2 days on one GPU |
| Re-train Stage 2 (fine tune to N = 4 to 20, 10M steps) | `DEVICE=cuda bash scripts/train_finetune.sh` | about 16 hours on one GPU |

## Repository layout

```
GReMS/
├── README.md                    this file
├── mkdocs.yml, docs/            documentation site (MkDocs Material)
├── requirements.txt             runtime dependencies
├── scripts/                     reproducible entry points (training, evaluation, data setup)
├── checkpoints/                 released model weights and learning curves
├── data/                        topology archives, layer profiles, DiSNet reference numbers
├── third_party/disnet/          vendored DiSNet code used by the DiSNet and MoDNN baselines
│
├── train_GReMS.py               PPO training loop
├── evaluate_GReMS.py            dynamic multi-seed evaluator (GReMS plus baselines)
├── generate_variable_networks.py   topology generator (RGG, BA, WS, ER)
│
├── environment.py               Gymnasium environment SplitEnv
├── gnn_model.py                 GraphSAGE actor critic with five action heads
├── state_encoder.py             542 dimensional observation encoder
├── action_space.py              action specification and plan decoder
├── reward.py                    reward function
├── network_dynamics.py          MMPP compute load and Gilbert-Elliott bandwidth models
├── run_rl_live.py               analytical latency model for a split plan
├── profiling.py                 layer profiles and graph to profile conversion
├── config.py                    all hyperparameters and constants
├── utils.py                     logging, seeding, JSON helpers
│
├── run_disnet_live.py           DiSNet baseline
├── run_modnn_live.py            MoDNN baseline
├── run_ensplit_live.py          EnSplit baseline (brute force or DDQN)
├── ensplit_ddqn.py, ensplit_env.py, train_ensplit.py   EnSplit DDQN agent and training
```

A file by file description is in [docs/code_structure.md](docs/code_structure.md).

## Released checkpoints

| Path | Description |
|------|-------------|
| `checkpoints/checkpoints/variable_networks_GReMS_FINAL/grems_final.pt` | Stage 1, trained from scratch on N = 4 to 10 for 30M steps |
| `checkpoints/checkpoints/variable_networks_GReMS_Final_Finetune/grems_final.pt` | Stage 2, fine tuned on N = 4 to 20. **This is the model used for the headline results.** |
| `checkpoints/ensplit/ensplit_best.pt` | EnSplit DDQN baseline |

Each directory also contains `grems_best.pt`, `grems_latest.pt` and a `learning_curve.csv`.

## Documentation

```bash
pip install -r requirements-docs.txt
mkdocs serve          # browse at http://127.0.0.1:8000
mkdocs build          # static site in site/
```

The site covers installation, the problem formulation, the agent, every file and config option, how to reproduce each experiment, how to extend the project and troubleshooting.

## Hardware

Training ran on a single NVIDIA GeForce RTX 5060 Ti on the ETH student cluster.
Evaluation and the analytical latency model run on CPU.
Layer timings are calibrated to a Jetson Nano in MAXN mode (7253 ms for one full VGG-16 forward pass).
