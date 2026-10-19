"""
network_dynamics.py — Dynamic channel models for IoT nodes and links.

Both models advance once per episode (at env.reset()) so dynamics are stable
within each episode, letting the critic predict the full discounted return.

ComputeLoadModel — two-state MMPP per node:
  LOW  (ρ ≈ 0.10–0.35)  HIGH (ρ ≈ 0.55–0.80)
  P(LOW→HIGH)=0.05  P(HIGH→LOW)=0.20  steady-state P(HIGH)≈0.20

LinkBWModel — two-state Gilbert-Elliott per link:
  GOOD (factor ≈ 0.80–1.00 × base)  BAD (factor ≈ 0.15–0.25 × base)
  P(GOOD→BAD)=0.15  P(BAD→GOOD)=0.40  steady-state P(BAD)≈0.27
  effective_bw(i,j) = base_bw(i,j) × factor(i,j)
"""

import numpy as np

from config import (
    GE_BW_BAD_FACTOR_MAX,
    GE_BW_BAD_FACTOR_MIN,
    GE_BW_GOOD_FACTOR_MAX,
    GE_BW_GOOD_FACTOR_MIN,
    GE_TRANS_BG,
    GE_TRANS_GB,
    MMPP_LOAD_HIGH_MAX,
    MMPP_LOAD_HIGH_MIN,
    MMPP_LOAD_LOW_MAX,
    MMPP_LOAD_LOW_MIN,
    MMPP_TRANS_HL,
    MMPP_TRANS_LH,
)

_LOW  = 0
_HIGH = 1
_GOOD = 0
_BAD  = 1


class ComputeLoadModel:
    """Independent two-state MMPP per node; advance() called once per episode."""

    def __init__(self, num_nodes: int, rng: np.random.Generator) -> None:
        self._n    = num_nodes
        self._rng  = rng
        # All nodes start in LOW state with minimum background load
        self._state = np.zeros(num_nodes, dtype=np.int8)
        self._util  = np.full(num_nodes, MMPP_LOAD_LOW_MIN, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def advance(self) -> None:
        """Step the MMPP chain and resample node loads; call once per env reset()."""
        rands = self._rng.uniform(size=self._n)

        for i in range(self._n):
            if self._state[i] == _LOW:
                if rands[i] < MMPP_TRANS_LH:
                    self._state[i] = _HIGH
            else:
                if rands[i] < MMPP_TRANS_HL:
                    self._state[i] = _LOW

        # Sample load from the current state's distribution
        for i in range(self._n):
            if self._state[i] == _LOW:
                self._util[i] = float(
                    self._rng.uniform(MMPP_LOAD_LOW_MIN, MMPP_LOAD_LOW_MAX)
                )
            else:
                self._util[i] = float(
                    self._rng.uniform(MMPP_LOAD_HIGH_MIN, MMPP_LOAD_HIGH_MAX)
                )

    def get_utilization(self, node_id: int) -> float:
        """Return current background utilization ρ ∈ [0, 1) for node_id."""
        return float(self._util[node_id])

    def resize(self, num_nodes: int) -> None:
        """Resize to a new node count; new nodes start in LOW state, existing states preserved."""
        if num_nodes == self._n:
            return
        if num_nodes < self._n:
            self._state = self._state[:num_nodes].copy()
            self._util  = self._util[:num_nodes].copy()
        else:
            extra = num_nodes - self._n
            self._state = np.concatenate(
                [self._state, np.zeros(extra, dtype=np.int8)]
            )
            self._util = np.concatenate(
                [self._util, np.full(extra, MMPP_LOAD_LOW_MIN, dtype=np.float32)]
            )
        self._n = num_nodes


# ─────────────────────────────────────────────────────────────────────────────


class LinkBWModel:
    """Independent two-state Gilbert-Elliott chain per link; advance() called once per episode."""

    def __init__(self, num_nodes: int, rng: np.random.Generator) -> None:
        self._n       = num_nodes
        self._rng     = rng
        self._base_bw = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        # All links start in GOOD state
        self._state   = np.zeros((num_nodes, num_nodes), dtype=np.int8)
        self._factor  = np.ones((num_nodes, num_nodes),  dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def reset(self, base_bw: np.ndarray) -> None:
        """Hard-init from base BW matrix with all links in GOOD; only for the first episode."""
        n = base_bw.shape[0]
        if n != self._n:
            self.resize(n)
        self._base_bw = np.array(base_bw, dtype=np.float32)
        self._state   = np.zeros((self._n, self._n), dtype=np.int8)
        self._resample()

    def set_base_bw(self, base_bw: np.ndarray) -> None:
        """Update base BW without resetting chain state; use for all episodes after the first."""
        n = base_bw.shape[0]
        if n != self._n:
            self.resize(n)
        self._base_bw = np.array(base_bw, dtype=np.float32)

    def advance(self) -> None:
        """Step G-E chains and resample BW factors; call once per env reset()."""
        # Only iterate upper triangle; mirror to lower.
        rands = self._rng.uniform(size=(self._n, self._n))
        for i in range(self._n):
            for j in range(i + 1, self._n):
                s = self._state[i, j]
                if s == _GOOD:
                    if rands[i, j] < GE_TRANS_GB:
                        self._state[i, j] = self._state[j, i] = _BAD
                else:
                    if rands[i, j] < GE_TRANS_BG:
                        self._state[i, j] = self._state[j, i] = _GOOD
        self._resample()

    def get_bandwidth_matrix(self) -> np.ndarray:
        """Return the full N×N effective BW matrix (Mbit/s)."""
        return self._base_bw * self._factor

    def get_link_states(self) -> np.ndarray:
        """Return the (N, N) link state matrix: 0 = GOOD, 1 = BAD."""
        return self._state.copy()

    def resize(self, num_nodes: int) -> None:
        """Adapt to a topology size change; new links start in GOOD state."""
        if num_nodes == self._n:
            return
        n = num_nodes
        new_base  = np.zeros((n, n), dtype=np.float32)
        new_state = np.zeros((n, n), dtype=np.int8)
        new_factor = np.ones((n, n), dtype=np.float32)
        c = min(self._n, n)
        new_base[:c, :c]   = self._base_bw[:c, :c]
        new_state[:c, :c]  = self._state[:c, :c]
        new_factor[:c, :c] = self._factor[:c, :c]
        self._base_bw = new_base
        self._state   = new_state
        self._factor  = new_factor
        self._n       = num_nodes

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _resample(self) -> None:
        """Resample BW factors from each link's current state distribution."""
        for i in range(self._n):
            for j in range(i + 1, self._n):
                if self._state[i, j] == _GOOD:
                    f = float(self._rng.uniform(
                        GE_BW_GOOD_FACTOR_MIN, GE_BW_GOOD_FACTOR_MAX))
                else:
                    f = float(self._rng.uniform(
                        GE_BW_BAD_FACTOR_MIN, GE_BW_BAD_FACTOR_MAX))
                self._factor[i, j] = self._factor[j, i] = f
