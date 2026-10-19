# Quick start

All commands are run from the repository root with the environment from [Installation](installation.md) active.

## 1. Evaluate the released model on a few networks

```bash
SEEDS="0" SIZES=small BASELINES=1 NUM_SEQ=1 NUM_STEPS=5 bash scripts/evaluate.sh
```

What this does:

- loads `checkpoints/checkpoints/variable_networks_GReMS_Final_Finetune/grems_final.pt`,
- evaluates it on the held out RGG test networks with 3 to 10 nodes (10 networks per size),
- runs DiSNet, MoDNN and EnSplit on exactly the same channel states,
- writes `results/GReMS_Final_Finetune_rgg.json`.

Summarise the output:

```bash
python scripts/summarize_results.py results/GReMS_Final_Finetune_rgg.json
```

Example (shortened, one trajectory so numbers are noisy):

```
 N  GReMS (ms)  DiSNet (ms)  MoDNN (ms)  EnSplit (ms)
--  ----------  -----------  ----------  ------------
 4        2782         8256       29265          4074
 6        2314         6927       11084          3061
10        2056         5879       26788          4146
```

## 2. Train a tiny model (smoke test)

```bash
DEVICE=cpu TIMESTEPS=20480 RUN_ID=smoke bash scripts/train_stage1.sh
```

This checks that training, checkpointing and logging work. Expect 10 to 30 minutes on a CPU machine, because 350 training environments are built first and every PPO update simulates 512 steps. It is a plumbing test and the resulting policy is meaningless.
The checkpoint appears in `checkpoints/checkpoints/variable_networks_smoke/`.
The real run uses 30 million steps, see the [Reproducibility guide](reproducibility.md).

## 3. Load the policy in your own code

```python
import torch
from config import MAX_NODES
from environment import SplitEnv
from gnn_model import build_model

env = SplitEnv(num_nodes=MAX_NODES, use_offline_profiles=False, seed=0)
obs_dim = env.observation_space.shape[0]            # 542

path = "checkpoints/checkpoints/variable_networks_GReMS_Final_Finetune/grems_final.pt"
raw = torch.load(path, map_location="cpu", weights_only=False)
policy = build_model(obs_dim, env.spec_action)
policy.load_state_dict(raw["model"] if "model" in raw else raw)
policy.eval()
```

`evaluate_one_network` in `evaluate_GReMS.py` shows how a topology file is turned into an environment, how observations are encoded and how the policy output is decoded into a plan.
It is the best starting point for embedding the policy in another system.
