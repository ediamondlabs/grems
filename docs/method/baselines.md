# Baselines

All baselines are evaluated **live**. At every evaluation step they see the same compute load and link bandwidth state as GReMS, and their plan is scored with the same latency lookup tables.

| Method | File | Idea |
|--------|------|------|
| DiSNet | `run_disnet_live.py` | Four vertical groups (layers [0,4), [4,9), [9,14), [14,18)). Path selection, hub selection (`determine_opt_neighbours`) and the horizontal split ratio (`find_split_ratio`) come from the original DiSNet code. |
| MoDNN | `run_modnn_live.py` | The hub splits all 18 spatial layers across its neighbours. Fully connected layers run on the hub only. |
| EnSplit | `run_ensplit_live.py` | Vertical split only. A chain of 4 nodes is extracted between source and destination, then the best contiguous layer assignment is found. Without a checkpoint it is brute force optimal over at most 969 candidates. With `--ensplit-checkpoint` a trained DDQN picks the action. |

## DiSNet and MoDNN code

The decision functions of the two baselines are the original DiSNet implementation, vendored in `third_party/disnet/` (see its README for provenance). `run_disnet_live.py` and `run_modnn_live.py` import these modules inside a temporary `sys.modules` sandbox because the original files share module names such as `utils` and `network` with GReMS.

## EnSplit

`ensplit_ddqn.py` (agent), `ensplit_env.py` (environment) and `train_ensplit.py` reproduce the EnSplit DDQN: 128 neuron MLP, Huber loss, Adam with learning rate 1e-5, discount 0.9, 1500 episodes of 50 steps. The released agent is in `checkpoints/ensplit/`. By default the evaluator uses the **brute force** variant, which is an upper bound on what the DDQN can reach. Pass `--ensplit-checkpoint checkpoints/ensplit/ensplit_best.pt` to evaluate the learned agent.

## Fair comparison notes

- Bandwidth: baselines are capped at `DISNET_BW_CAP = 200` Mbit/s so that their link model covers the same range as the generated networks (`DISNET_PAPER_BW_CAP = 15` reproduces the original DiSNet testbed setting).
- Time limits: DiSNet and MoDNN use path enumeration that is exponential in graph size. Evaluations of networks above 10 nodes use a shorter per step time limit. If a baseline times out repeatedly on a network, it is skipped for that network (the log says so).
