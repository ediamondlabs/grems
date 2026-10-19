"""config.py – Central configuration: PPO hyper-parameters, environment sizing, VGG-16 model, paths."""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROFILE_DIR = DATA_DIR / "offline_profiles"

# Ensure directories exist
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# Profiling file paths (JSON) — only used by profiling.py CLI to save
# custom timed runs; not required for training or evaluation.
LAYER_PROFILE_PATH = PROFILE_DIR / "layer_profiles.json"

# Trained model checkpoint
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "ppo_best.pt"

# Variable-topology networks checkpoint (separate from DiSNet checkpoint)
VARIABLE_NET_CHECKPOINT_DIR = CHECKPOINT_DIR / "variable_networks_gnn"
VARIABLE_NET_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
VARIABLE_NET_BEST_PATH = VARIABLE_NET_CHECKPOINT_DIR / "ppo_best.pt"
VARIABLE_NET_FINAL_PATH = VARIABLE_NET_CHECKPOINT_DIR / "ppo_final.pt"

# ──────────────────────────────────────────────
# Neural network (target model being split)
# ──────────────────────────────────────────────
# Full VGG-16 architecture as used by DiSNet (models/model_vgg16.py).
# 21 layers: 13 Conv + 5 MaxPool + 3 FC.  Input: 224×224×3.
# Each tuple: (type, in_ch, out_ch, kernel/size, activation_size, flops)
# FLOPs convention: out_ch × H_out × W_out × k × k × in_ch  (multiplications only)
MODEL_CFG = [
    # ── Block 1 (224×224) ──
    ("C",   3,   64, 3,  64 * 224 * 224,   64 * 224 * 224 * 3 * 3 * 3),    #  0
    ("C",  64,   64, 3,  64 * 224 * 224,   64 * 224 * 224 * 3 * 3 * 64),   #  1
    ("M",  64,   64, 2,  64 * 112 * 112,   0),                              #  2
    # ── Block 2 (112×112) ──
    ("C",  64,  128, 3, 128 * 112 * 112,  128 * 112 * 112 * 3 * 3 * 64),   #  3
    ("C", 128,  128, 3, 128 * 112 * 112,  128 * 112 * 112 * 3 * 3 * 128),  #  4
    ("M", 128,  128, 2, 128 *  56 *  56,   0),                              #  5
    # ── Block 3 (56×56) ──
    ("C", 128,  256, 3, 256 *  56 *  56,  256 *  56 *  56 * 3 * 3 * 128),  #  6
    ("C", 256,  256, 3, 256 *  56 *  56,  256 *  56 *  56 * 3 * 3 * 256),  #  7
    ("C", 256,  256, 3, 256 *  56 *  56,  256 *  56 *  56 * 3 * 3 * 256),  #  8
    ("M", 256,  256, 2, 256 *  28 *  28,   0),                              #  9
    # ── Block 4 (28×28) ──
    ("C", 256,  512, 3, 512 *  28 *  28,  512 *  28 *  28 * 3 * 3 * 256),  # 10
    ("C", 512,  512, 3, 512 *  28 *  28,  512 *  28 *  28 * 3 * 3 * 512),  # 11
    ("C", 512,  512, 3, 512 *  28 *  28,  512 *  28 *  28 * 3 * 3 * 512),  # 12
    ("M", 512,  512, 2, 512 *  14 *  14,   0),                              # 13
    # ── Block 5 (14×14) ──
    ("C", 512,  512, 3, 512 *  14 *  14,  512 *  14 *  14 * 3 * 3 * 512),  # 14
    ("C", 512,  512, 3, 512 *  14 *  14,  512 *  14 *  14 * 3 * 3 * 512),  # 15
    ("C", 512,  512, 3, 512 *  14 *  14,  512 *  14 *  14 * 3 * 3 * 512),  # 16
    ("M", 512,  512, 2, 512 *   7 *   7,   0),                              # 17
    # ── Fully connected ──
    ("D", 7 * 7 * 512, 4096, 1, 4096,  7 * 7 * 512 * 4096),               # 18
    ("D", 4096,        4096, 1, 4096,  4096 * 4096),                        # 19
    ("D", 4096,        1000, 1, 1000,  4096 * 1000),                        # 20
]
NUM_LAYERS = len(MODEL_CFG)  # 21

# ──────────────────────────────────────────────
# Per-layer CPU forward times (ms)
# ──────────────────────────────────────────────
# Measured on ETH tik42x (Intel Xeon E5-2690 @ 2.90 GHz, 16 threads)
# via  python3 profiling.py --device cpu
# Total ≈ 394.60 ms for a full VGG-16 forward pass.
LAYER_FWD_MS = [
     14.702,  #  0  conv1_1
     34.396,  #  1  conv1_2
      6.294,  #  2  pool1
     28.855,  #  3  conv2_1
     33.150,  #  4  conv2_2
      3.218,  #  5  pool2
     23.706,  #  6  conv3_1
     31.697,  #  7  conv3_2
     32.194,  #  8  conv3_3
      4.168,  #  9  pool3
     24.072,  # 10  conv4_1
     29.671,  # 11  conv4_2
     31.960,  # 12  conv4_3
      0.272,  # 13  pool4
     26.840,  # 14  conv5_1
     25.004,  # 15  conv5_2
     19.907,  # 16  conv5_3
      2.106,  # 17  pool5
     13.907,  # 18  fc6
      5.219,  # 19  fc7
      3.264,  # 20  fc8
]

# ──────────────────────────────────────────────
# VGG-16 spatial & kernel config (for FoV computation)
# ──────────────────────────────────────────────
# (kernel_size, stride, padding) per layer – matches vgg16 dict in field_DiSNet.py
LAYER_KERNEL_CFG = [
    (3, 1, 1),  #  0  conv1_1
    (3, 1, 1),  #  1  conv1_2
    (2, 2, 0),  #  2  pool1
    (3, 1, 1),  #  3  conv2_1
    (3, 1, 1),  #  4  conv2_2
    (2, 2, 0),  #  5  pool2
    (3, 1, 1),  #  6  conv3_1
    (3, 1, 1),  #  7  conv3_2
    (3, 1, 1),  #  8  conv3_3
    (2, 2, 0),  #  9  pool3
    (3, 1, 1),  # 10  conv4_1
    (3, 1, 1),  # 11  conv4_2
    (3, 1, 1),  # 12  conv4_3
    (2, 2, 0),  # 13  pool4
    (3, 1, 1),  # 14  conv5_1
    (3, 1, 1),  # 15  conv5_2
    (3, 1, 1),  # 16  conv5_3
    (2, 2, 0),  # 17  pool5
    (0, 0, 0),  # 18  fc6
    (0, 0, 0),  # 19  fc7
    (0, 0, 0),  # 20  fc8
]


def group_fov_overlap(l_start: int, l_end: int) -> int:
    """Extra input rows each worker needs on each interior boundary side for group [l_start, l_end)."""
    j, r = 1, 1
    for l in range(l_start, l_end):
        if l >= len(LAYER_KERNEL_CFG):
            break
        k, s = LAYER_KERNEL_CFG[l][0], LAYER_KERNEL_CFG[l][1]
        if k == 0:          # FC layer — no spatial receptive field, stop here
            break
        r += (k - 1) * j
        j *= s
    return r // 2


# Output spatial height of each layer (Conv3+pad1 preserves, Pool/2 halves)
LAYER_SPATIAL_OUT = [
    224, 224, 112,          # layers  0– 2
    112, 112,  56,          # layers  3– 5
     56,  56,  56, 28,      # layers  6– 9
     28,  28,  28, 14,      # layers 10–13
     14,  14,  14,  7,      # layers 14–17
      0,   0,   0,          # layers 18–20 (FC — no spatial dim)
]

# ──────────────────────────────────────────────
# IoT Reference LUT  (Jetson Nano MAXN, calibrated)
# ──────────────────────────────────────────────
# DiSNet measured Jetson Nano MAXN (CPU, float32, VGG-16): 7 253.3 ms total.
# ETH tik42x (Intel Xeon E5-2690) total from LAYER_FWD_MS above:  394.6 ms.
# Calibration scale: 7253.3 / 394.6 ≈ 18.38×
#
# IOT_REF_WEIGHT = 1  ↔  Jetson Nano MAXN speed (slowest reference device).
# A node of weight W executes layer l in:
#     LAYER_IOT_FWD_MS[l] * IOT_REF_WEIGHT / W   milliseconds
# which gives:
#   weight=1  → Jetson Nano MAXN speed  (~7 250 ms total)
#   weight=4  → ~1 810 ms total  (~4× faster than Nano)
#   weight=10 → ~725 ms total    (~10× faster than Nano)
#
# Real-world reference points (VGG-16 CPU inference, measured):
#   Jetson Nano MAXN       7 253 ms  →  weight ≈ 1.0  (DiSNet paper)
#   Intel Xeon E5-2697A v4 1 626 ms  →  weight ≈ 4.5  (measured)
#   Intel Xeon E5-2690      394 ms   →  weight ≈ 18   (this machine)
#   Apple M1                111 ms   →  weight ≈ 65   (measured)
#
# The [1, 10] training range is a conservative subset of the real spread.
_IOT_CALIBRATION_SCALE = 7253.297 / 394.6   # ≈ 18.38
IOT_REF_WEIGHT         = 1                   # reference node weight (Jetson Nano MAXN)
LAYER_IOT_FWD_MS       = [t * _IOT_CALIBRATION_SCALE for t in LAYER_FWD_MS]

# ──────────────────────────────────────────────
# IoT network
# ──────────────────────────────────────────────
NUM_NODES = 10                      # number of IoT / edge nodes (matches DiSNet networks)
DEFAULT_COMP_POWERS = [4, 2, 4, 8, 4, 2, 8, 7, 6, 9]   # from DiSNet network0 node caps
DEFAULT_TRANS_POWERS = [1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0]        # normalised (bandwidth from graph)

MAX_NODES = 20                      # obs/action-space padded to this; single model covers 3…20 nodes
VARIABLE_NETWORKS_DIR = PROJECT_ROOT / "variable_networks"
# Training sizes (held-out test: 20 nodes); override via VARIABLE_TRAIN_SIZES="4,5,...,10"
VARIABLE_TRAIN_SIZES = [int(_n) for _n in os.environ.get(
    "VARIABLE_TRAIN_SIZES",
    "3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
).split(",")]
VARIABLE_TEST_SIZES  = [20]

# ── DiSNet network comparison mode (legacy, not used by training or evaluation) ──
# The 4 original DiSNet testbed graphs; only needed if you copy them to data/disnet_networks/.
DISNET_NETWORKS_DIR = DATA_DIR / "disnet_networks"
DISNET_GRAPH_PATHS = [
    DISNET_NETWORKS_DIR / f"network{i}.txt" for i in range(4)
]
DISNET_GRAPH_PATH = DISNET_GRAPH_PATHS[0]   # default (backward compat)
DISNET_BW_CAP = 200.0      # Mbit/s – must be ≥ BW_RANGE upper bound in generate_variable_networks.py

DISNET_PAPER_BW_CAP = 15.0  # Mbit/s – original DiSNet paper BW cap for CSV baseline comparisons

# BW_NORM_SCALE: normalises raw link-bandwidth (MB/s) to ≈[0,1] in state encoder.
BW_NORM_SCALE = 25.0       # MB/s  (= 200 Mbit/s / 8)

# ──────────────────────────────────────────────
# Environment / action-space limits
# ──────────────────────────────────────────────
MAX_GROUPS = 6          # capacity ceiling for action vector; agent freely chooses 1…MAX_GROUPS
MAX_CHUNKS = MAX_NODES  # max horizontal chunks per group
MAX_PIPELINE_DEPTH = MAX_GROUPS

# ──────────────────────────────────────────────
# Latency model (matching DiSNet conventions)
# ──────────────────────────────────────────────
DEVICE_PACE_RATE = 1    # divisor applied to transfer times only (DiSNet convention)

# effective_speedup = 1 + DEVICE_SCALING_DIMINISHING_RETURN * (node_weight - 1)
DEVICE_SCALING_DIMINISHING_RETURN = 0.4

# ──────────────────────────────────────────────
# Background compute load (MMPP) — see network_dynamics.py for model details
# ──────────────────────────────────────────────
MMPP_TRANS_LH        = 0.05   # P(LOW→HIGH) per episode
MMPP_TRANS_HL        = 0.20   # P(HIGH→LOW) per episode
MMPP_LOAD_LOW_MIN    = 0.10   # ρ lower bound in LOW state
MMPP_LOAD_LOW_MAX    = 0.35   # ρ upper bound in LOW state
MMPP_LOAD_HIGH_MIN   = 0.55   # ρ lower bound in HIGH state
MMPP_LOAD_HIGH_MAX   = 0.80   # ρ upper bound in HIGH state
COMPUTE_SLOWDOWN_MAX = 4.0    # M/M/1 slowdown cap (guards against ρ → 1)


# ──────────────────────────────────────────────
# Gilbert-Elliott link bandwidth model — see network_dynamics.py for model details
# ──────────────────────────────────────────────
GE_TRANS_GB           = float(os.environ.get("GE_TRANS_GB",           "0.15"))
GE_TRANS_BG           = float(os.environ.get("GE_TRANS_BG",           "0.40"))
GE_BW_GOOD_FACTOR_MIN = float(os.environ.get("GE_BW_GOOD_FACTOR_MIN", "0.80"))
GE_BW_GOOD_FACTOR_MAX = float(os.environ.get("GE_BW_GOOD_FACTOR_MAX", "1.00"))
GE_BW_BAD_FACTOR_MIN  = float(os.environ.get("GE_BW_BAD_FACTOR_MIN",  "0.15"))
GE_BW_BAD_FACTOR_MAX  = float(os.environ.get("GE_BW_BAD_FACTOR_MAX",  "0.25"))

# ──────────────────────────────────────────────
# MDP episode length
# ──────────────────────────────────────────────
EPISODE_LENGTH = 10    # steps per episode (MMPP + G-E chains advance once at reset, not per step)

# Fixed overheads (ms) — account for setup, sync, and network stack costs.
# These are added independently per group/hop and are NOT scaled by device speed.
OVERHEAD_REMOTE_NODE_SETUP = 5.0   # per remote worker: connection, GPU memory alloc, etc.
OVERHEAD_PER_CHUNK = 1.5            # per chunk: packing/unpacking, serialization overhead
OVERHEAD_PER_DATA_TRANSFER = 0.5    # per send or recv operation: network stack latency

# ──────────────────────────────────────────────
# Reward
# ──────────────────────────────────────────────
ALPHA = 5.0             # reward = -ALPHA * log(latency_ms)
CONNECTIVITY_PENALTY_BETA = 30.0   # penalty per disconnected hub-worker pair

# Curriculum: connectivity penalty ramps linearly START→END over WARMUP_STEPS
CONN_PENALTY_START        = float(os.environ.get("CONN_PENALTY_START", "5.0"))  # low early: allows G=3/4 exploration
CONN_PENALTY_END          = 50.0       # high late: enforces robustness at convergence
CONN_PENALTY_WARMUP_STEPS = 500_000
MIN_DISTINCT_NODES = 2          # hub + at least 1 worker always required
SINGLE_NODE_PENALTY = 140.0     # keeps hub-only plans dominated over any valid split

# Penalty for a weak-but-nonzero hub link (CONNECTIVITY_PENALTY_BETA only fires at bw=0)
WEAK_LINK_BW_THRESHOLD = float(os.environ.get("WEAK_LINK_BW_THRESHOLD", "2.0"))  # MB/s
WEAK_LINK_PENALTY_BETA = float(os.environ.get("WEAK_LINK_PENALTY_BETA", "10.0"))
WORKER_DIVERSITY_PENALTY = 0.0   # disabled — memory penalty drives splitting

# ── Memory constraint ─────────────────────────────────────────────────────
# Dynamic threshold per topology: NUM_LAYERS / ceil(sqrt(num_nodes))
MAX_LAYERS_PER_NODE    = 7      # reference only — env uses dynamic threshold
MEMORY_OVERLOAD_PENALTY = 200.0 # strong enough that 1 overloaded node forces higher G

# Exploration bonuses
NODE_UTIL_BONUS  = float(os.environ.get("NODE_UTIL_BONUS", "0.0"))  # disabled in the released models (was 25.0 early in development)
VSPLIT_BONUS     = 0.0   # disabled — memory penalty drives G≥3
# Routing quality: 0.0 if all inter-group links BAD, 5.0 if all GOOD (low-variance signal)
ROUTING_QUALITY_BONUS = 5.0

# ──────────────────────────────────────────────
# Discrete action space
# ──────────────────────────────────────────────
N_VERT_WEIGHTS = 8      # vertical-split weight bins (per group: 0-7)
BW_CONNECTED_THRESHOLD = 1.0  # MB/s – above this = connected edge

# ──────────────────────────────────────────────
# PPO hyper-parameters
# ──────────────────────────────────────────────
PPO_LR = 3e-4                   # initial learning rate (annealed linearly to 0)
PPO_GAMMA = 0.99                # discount factor for multi-step MDP
PPO_GAE_LAMBDA = 0.95           # GAE lambda for advantage estimation
PPO_CLIP_EPS = 0.2              # clipping epsilon
PPO_ENTROPY_COEFF = 0.06         # GNN features prevent collapse

# Entropy coeff decays --entropy-coeff -> ENTROPY_COEFF_END over ENTROPY_DECAY_STEPS
ENTROPY_COEFF_END    = float(os.environ.get("ENTROPY_COEFF_END", "0.002"))
ENTROPY_DECAY_STEPS  = int(os.environ.get("ENTROPY_DECAY_STEPS", "2000000"))
PPO_VALUE_COEFF = float(os.environ.get("VALUE_COEFF", "0.5"))  # 1.0 caused critic oscillation; 0.5 is steadier

# Caps reward_rms's loaded Welford count on --resume-step so new data can
# still move it; 0 = disabled.
REWARD_RMS_RESUME_COUNT_CAP = float(os.environ.get("REWARD_RMS_RESUME_COUNT_CAP", "0"))
PPO_MAX_GRAD_NORM = 0.5         # gradient clipping
PPO_EPOCHS = 4                  # update epochs per rollout
PPO_BATCH_SIZE = 512            # larger batches → more topology diversity per GNN gradient step
NUM_ROLLOUT_STEPS = 512         # steps collected per rollout
TOTAL_TIMESTEPS = 6_000_000     # total training steps (can be changed in training file)

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
SEED = 42
LOG_INTERVAL = 10               # log every N rollouts
EVAL_EPISODES = 50              # episodes for evaluation
DEVICE = "cpu"                  # "cuda" if GPU available

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
