"""reward.py – Reward: -ALPHA * log(latency_ms). Log-scale compresses the wide latency range
(~100 ms to 100 000 ms) into a stable signal; negative sign makes lower latency → higher reward."""

import numpy as np

from config import ALPHA


# --------------------------------------------------
# Reward function
# --------------------------------------------------

def compute_reward(latency: float, alpha: float = ALPHA) -> float:
    """Return -alpha * log(latency_ms); floor at 1 ms to avoid log(0)."""
    latency = max(latency, 1.0)   # floor only — no cap; reward_rms handles scale
    return -alpha * np.log(latency)
