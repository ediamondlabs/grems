# Problem and environment

## What is being optimised

A VGG-16 model with 21 layers (13 convolution, 5 max pool, 3 fully connected, input 224 x 224 x 3) must run inference on a wireless network of IoT devices.
No single device is fast enough, so the model is split over several devices.
GReMS produces a **split plan** for each request:

| Decision | Meaning |
|----------|---------|
| Vertical split | The 21 layers are cut into 1 to 6 consecutive groups (pipeline stages). |
| Horizontal split | Inside one spatial group the feature map is cut into chunks that run on parallel workers. Neighbouring chunks exchange the overlap rows that convolutions need. |
| Hub | The node that receives the input, distributes it to the first group and aggregates. |
| Assignment | Which physical node fills each worker slot of each group. |

The metric is **end to end latency in milliseconds**, estimated with an analytical model ([Latency model](latency_model.md)).

## The network

- **Topologies** are connected graphs with 3 to 20 nodes from four generator families: random geometric graph (RGG, used for training), Barabasi-Albert (BA), Watts-Strogatz (WS) and Erdos-Renyi (ER). BA, WS and ER are used to test generalisation.
- **Node capability** is a weight in [1, 10] relative to a Jetson Nano in MAXN mode. A node of weight `w` runs layer `l` in `LAYER_IOT_FWD_MS[l] / speedup(w)` milliseconds with `speedup(w) = 1 + 0.4 * (w - 1)`, a deliberately diminishing return (see `DEVICE_SCALING_DIMINISHING_RETURN`).
- **Link bandwidth**: the base bandwidth of every edge is drawn log uniformly between 5 and 200 Mbit/s.
- Topology files are pickled NetworkX graphs named `varnet_<N>nodes[_<tag>]_<i>.txt`. Test networks carry the tag `test`.

## Channel dynamics

Two Markov models make the conditions change over time. Both advance **once per episode** at `env.reset()`, so a channel state stays fixed during the steps of an episode. This lets the critic predict the whole discounted return from the first observation.

**Compute load, two state MMPP per node** (`ComputeLoadModel`)

| State | Background utilisation rho | Transition |
|-------|----------------------------|------------|
| LOW | uniform 0.10 to 0.35 | LOW to HIGH with probability 0.05 |
| HIGH | uniform 0.55 to 0.80 | HIGH to LOW with probability 0.20 |

The compute time of a node is multiplied by an M/M/1 style slowdown `min(1 / (1 - rho), 4)`.

**Link bandwidth, two state Gilbert-Elliott per link** (`LinkBWModel`)

| State | Bandwidth factor | Transition (paper setting) |
|-------|------------------|----------------------------|
| GOOD | 0.80 to 1.00 of base | GOOD to BAD with probability 0.15 |
| BAD | 0.15 to 0.25 of base | BAD to GOOD with probability 0.40 |

During evaluation the chains advance at **every step** of a 50 step trajectory, and all methods are scored on the same sequence of states.

## Episodes in training

`SplitEnv.reset()` samples a network from the training pool, advances both chains and returns the first observation.
An episode lasts `EPISODE_LENGTH = 10` steps. At each step the agent proposes a plan for the same channel state and receives the reward. The previous action is part of the next observation.

## Train and test protocol

| Item | Value |
|------|-------|
| Training topologies | RGG, 50 networks per size |
| Stage 1 sizes | N = 4 to 10 |
| Stage 2 sizes | N = 4 to 20, sizes 11 to 20 sampled with twice the weight |
| Test topologies | 10 held out networks per size, never used in training |
| Evaluation sizes | 3 to 20 (size 3 was never trained on in either stage) |
| Out of family tests | WS, ER, BA with the same sizes, plus BA with m = 1, 3, 4 |
