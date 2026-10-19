"""ensplit_env.py – DDQN training environment for EnSplit; N_CHAIN_NODES=4 chain, C(19,3)=969 discrete actions.
LUT-based latency (same physics as GReMS/DiSNet/MoDNN). Energy credit tracked during training, dropped at eval."""

from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import List, Optional, Tuple

from config import (
    IOT_REF_WEIGHT,
    LAYER_IOT_FWD_MS,
    LAYER_SPATIAL_OUT,
    MODEL_CFG,
    NUM_LAYERS,
)

# ── Chain topology constants (match paper: N=4 nodes) ────────────────────────
N_CHAIN_NODES: int = 4      # UE + 3 network nodes
H_MIN_UE:      int = 2      # min layers at node 0 (privacy, eq. 7c)
L:             int = NUM_LAYERS   # 21 VGG-16 layers
EPISODE_STEPS: int = 50     # decisions per episode (Table I: 50 s @ 1/s)
NU:            float = 0.90 # energy credit limit (Table I: ν = 90 %)

# Resource ranges (IoT scale)
_NODE_W_MIN,  _NODE_W_MAX  = 1.0,  10.0   # compute weight
_UE_W_MIN,    _UE_W_MAX    = 1.0,   4.0   # UE is weaker
_BW_MIN,      _BW_MAX      = 1.0, 200.0   # Mbit/s (network links)
_UE_BW_MIN,   _UE_BW_MAX   = 1.0,  50.0  # UE uplink weaker


# ── Action enumeration ────────────────────────────────────────────────────────

def enumerate_actions(
    n_nodes: int = N_CHAIN_NODES,
    n_layers: int = L,
    h_min: int = H_MIN_UE,
) -> List[Tuple[int, ...]]:
    """Return all valid contiguous split-point tuples; s1≥h_min (UE privacy), each node gets ≥1 layer."""
    positions = range(h_min, n_layers)   # valid split-point values
    return list(combinations(positions, n_nodes - 1))


# ── Latency model ─────────────────────────────────────────────────────────────

def _activation_bytes(layer_idx: int) -> float:
    """Output activation size in bytes (float32) for inter-node transfer volume."""
    h = LAYER_SPATIAL_OUT[layer_idx]
    out_ch = MODEL_CFG[layer_idx][2]
    if h > 0:                       # spatial layer (Conv / Pool)
        return float(out_ch * h * h * 4)
    else:                           # FC layer
        return float(MODEL_CFG[layer_idx][4] * 4)


def compute_chain_latency_ms(
    splits: Tuple[int, ...],
    node_weights: np.ndarray,
    link_bw_mbitps: np.ndarray,
) -> float:
    """LUT-based end-to-end latency (ms) for a vertical split on a chain of nodes."""
    n = len(node_weights)
    boundaries = (0,) + tuple(splits) + (L,)
    total_ms = 0.0

    for node_idx in range(n):
        l_start = boundaries[node_idx]
        l_end   = boundaries[node_idx + 1]
        if l_start >= l_end:
            continue

        # Compute time: sum of LUT times scaled by node weight
        w = max(node_weights[node_idx], 1e-6)
        for l in range(l_start, l_end):
            total_ms += LAYER_IOT_FWD_MS[l] * IOT_REF_WEIGHT / w

        # Transfer time: output of last layer in segment → next node
        if node_idx < n - 1 and l_end < L:
            data_bytes = _activation_bytes(l_end - 1)
            bw = link_bw_mbitps[node_idx, node_idx + 1]
            if bw > 0.0:
                bw_Bps = bw * 1e6 / 8.0        # Mbit/s → bytes/s
                total_ms += data_bytes / bw_Bps * 1000.0
            else:
                total_ms += 1e6                 # no-link penalty

    return total_ms


# ── Environment ───────────────────────────────────────────────────────────────

class EnSplitEnv:
    """EnSplit DDQN environment; fixed N_CHAIN_NODES chain, resources re-sampled each step (time-varying, Table I)."""

    # Precomputed total compute cost proxy (sum of LUT times at ref weight)
    _LAYER_COST = np.array(LAYER_IOT_FWD_MS, dtype=np.float64)
    _TOTAL_COST = float(_LAYER_COST.sum())

    def __init__(
        self,
        n_nodes:  int = N_CHAIN_NODES,
        h_min:    int = H_MIN_UE,
        nu:       float = NU,
        seed:     int = 0,
    ):
        self.n_nodes = n_nodes
        self.h_min   = h_min
        self.nu      = nu
        self.rng     = np.random.RandomState(seed)

        self.actions  = enumerate_actions(n_nodes, L, h_min)
        self.n_actions = len(self.actions)

        # State: node_weights (N) + link_bw_flat (N×N) + energy_credit (1)
        self.state_dim = n_nodes + n_nodes * n_nodes + 1

        # Internal dynamic state
        self.node_weights:   np.ndarray = np.ones(n_nodes)
        self.link_bw:        np.ndarray = np.zeros((n_nodes, n_nodes))
        self._credit_used:   float = 0.0   # cumulative offloaded compute cost
        self._credit_budget: float = 0.0   # grows each step: step * total_cost
        self._step:          int   = 0

    # ── Resource sampling ─────────────────────────────────────────────────────

    def _sample_resources(self) -> None:
        n = self.n_nodes
        # Node weights: UE (0) is weaker
        w = self.rng.uniform(_NODE_W_MIN, _NODE_W_MAX, size=n)
        w[0] = self.rng.uniform(_UE_W_MIN, _UE_W_MAX)
        self.node_weights = w

        # Link BW: symmetric matrix, UE uplink weaker
        bw = self.rng.uniform(_BW_MIN, _BW_MAX, size=(n, n))
        bw = (bw + bw.T) / 2.0
        np.fill_diagonal(bw, 0.0)
        if n > 1:
            ue_bw = self.rng.uniform(_UE_BW_MIN, _UE_BW_MAX)
            bw[0, 1] = bw[1, 0] = ue_bw
        self.link_bw = bw

    # ── State encoding ────────────────────────────────────────────────────────

    def _encode_state(self) -> np.ndarray:
        credit_frac = np.clip(
            self._credit_used / max(self._credit_budget, 1e-9), 0.0, 1.0
        )
        return np.concatenate([
            self.node_weights / _NODE_W_MAX,
            self.link_bw.flatten() / _BW_MAX,
            [credit_frac],
        ]).astype(np.float32)

    # ── Energy credit helpers ─────────────────────────────────────────────────

    def _offloaded_cost(self, splits: Tuple[int, ...]) -> float:
        """Compute proxy (LUT ms at ref weight) for layers NOT on node 0."""
        s1 = splits[0]
        return float(self._LAYER_COST[s1:].sum())

    def _credit_violated(self, splits: Tuple[int, ...]) -> bool:
        """True if accepting this split would exceed the energy credit limit."""
        new_used   = self._credit_used + self._offloaded_cost(splits)
        new_budget = self._credit_budget + self._TOTAL_COST
        return (new_used / max(new_budget, 1e-9)) > self.nu

    # ── Gym-like API ──────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._credit_used   = 0.0
        self._credit_budget = 0.0
        self._step          = 0
        self._sample_resources()
        return self._encode_state()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one inference decision; doubles latency as penalty if energy credit constraint (eq. 7d) is violated."""
        splits = self.actions[action_idx]

        # Advance budget for this step
        self._credit_budget += self._TOTAL_COST
        self._step += 1

        if self._credit_violated(splits):
            # Force all-local execution as penalty (model paper's constraint)
            lat_ms = compute_chain_latency_ms(
                (L,) * (self.n_nodes - 1),   # all layers at node 0
                self.node_weights,
                self.link_bw,
            ) * 2.0
        else:
            self._credit_used += self._offloaded_cost(splits)
            lat_ms = compute_chain_latency_ms(
                splits, self.node_weights, self.link_bw
            )

        # Reward: -log(latency) — consistent with GReMS's reward convention
        reward = -float(np.log(max(lat_ms, 1.0)))

        # Re-sample resources for next step (time-varying, Table I)
        self._sample_resources()

        done = (self._step >= EPISODE_STEPS)
        obs  = self._encode_state()
        return obs, reward, done, {"latency_ms": float(lat_ms)}

    # ── Evaluation helper (no energy tracking, no resource re-sample) ─────────

    def greedy_latency_ms(
        self,
        action_idx: int,
        node_weights: Optional[np.ndarray] = None,
        link_bw: Optional[np.ndarray] = None,
    ) -> float:
        """Compute latency for action_idx under given (or current) resources; no side effects."""
        w  = node_weights if node_weights is not None else self.node_weights
        bw = link_bw      if link_bw      is not None else self.link_bw
        return compute_chain_latency_ms(self.actions[action_idx], w, bw)
