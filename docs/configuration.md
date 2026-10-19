# Configuration reference

Most constants live in `config.py`. A subset can be overridden with environment variables without editing code. The defaults are the paper settings.

## Environment variables

| Variable | Default (paper value) | Meaning |
|---|---|---|
| `GE_TRANS_GB` | 0.15 | P(GOOD to BAD) of the link chain |
| `GE_TRANS_BG` | 0.40 | P(BAD to GOOD) |
| `GE_BW_GOOD_FACTOR_MIN/MAX` | 0.80 / 1.00 | bandwidth fraction in GOOD state |
| `GE_BW_BAD_FACTOR_MIN/MAX` | 0.15 / 0.25 | bandwidth fraction in BAD state |
| `NODE_UTIL_BONUS` | 0.0 | reward bonus for using more nodes (disabled) |
| `CONN_PENALTY_START` | 5.0 (Stage 2 uses 50.0, set by its script) | start of the connectivity penalty ramp |
| `VARIABLE_TRAIN_SIZES` | 3,...,20 (the stage scripts set 4,...,10 and 4,...,20) | node counts sampled in training |
| `WEAK_LINK_BW_THRESHOLD` | 2.0 | MB/s below which a first hop link is penalised |
| `WEAK_LINK_PENALTY_BETA` | 10.0 | weight of that penalty |
| `ENTROPY_COEFF_END` | 0.002 | final entropy coefficient |
| `ENTROPY_DECAY_STEPS` | 2 000 000 | steps over which entropy decays |
| `VALUE_COEFF` | 0.5 | critic loss weight |
| `REWARD_RMS_RESUME_COUNT_CAP` | 0 (off) | caps the reward normaliser count on resume |

## Constants in `config.py`

| Group | Names | Notes |
|-------|-------|-------|
| Model | `MODEL_CFG`, `NUM_LAYERS = 21`, `LAYER_KERNEL_CFG`, `LAYER_SPATIAL_OUT` | VGG-16 definition with FLOPs and activation sizes |
| Timing | `LAYER_FWD_MS`, `LAYER_IOT_FWD_MS`, `IOT_REF_WEIGHT`, `DEVICE_SCALING_DIMINISHING_RETURN` | Xeon timings scaled to a Jetson Nano |
| Network | `MAX_NODES = 20`, `DISNET_BW_CAP = 200`, `BW_NORM_SCALE` | padding size and bandwidth normalisation |
| Channels | `MMPP_*`, `COMPUTE_SLOWDOWN_MAX`, `GE_*`, `EPISODE_LENGTH = 10` | see [Problem and environment](method/problem.md) |
| Action space | `MAX_GROUPS = 6`, `MAX_CHUNKS = MAX_NODES`, `N_VERT_WEIGHTS = 8` | sizes of the action heads |
| Reward | `ALPHA = 5`, `CONNECTIVITY_PENALTY_BETA`, `CONN_PENALTY_END = 50`, `CONN_PENALTY_WARMUP_STEPS = 500000`, `SINGLE_NODE_PENALTY = 140`, `MEMORY_OVERLOAD_PENALTY = 200`, `ROUTING_QUALITY_BONUS = 5` | see [Reward and training](method/training.md) |
| PPO | `PPO_LR`, `PPO_GAMMA`, `PPO_GAE_LAMBDA`, `PPO_CLIP_EPS`, `PPO_EPOCHS`, `PPO_BATCH_SIZE`, `NUM_ROLLOUT_STEPS`, `PPO_MAX_GRAD_NORM` | command line flags override the learning rate, batch size, clip and entropy |
| Overheads | `OVERHEAD_REMOTE_NODE_SETUP = 5 ms`, `OVERHEAD_PER_CHUNK = 1.5 ms`, `OVERHEAD_PER_DATA_TRANSFER = 0.5 ms` | added to the latency model |
| Paths | `PROJECT_ROOT`, `DATA_DIR`, `CHECKPOINT_DIR`, `LOG_DIR` | derived from the location of `config.py` |

Importing `config` creates `data/offline_profiles/`, `checkpoints/` and `logs/` if they are missing.

!!! warning "Changing a constant changes the problem"
    The released checkpoints were trained for one observation layout (542 dimensions), one action layout and one latency model.
    Changing `MAX_NODES`, `MAX_GROUPS`, `MAX_CHUNKS`, `N_VERT_WEIGHTS` or anything that alters the observation size makes the released weights incompatible.
    Changing penalties, channel parameters or timings keeps them loadable but results will no longer match the paper.
