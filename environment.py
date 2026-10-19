"""
Gymnasium environment for GReMS split-inference optimisation.

Each episode: reset() advances MMPP compute-load and G-E link-BW chains once,
then holds channel state fixed for all EPISODE_LENGTH steps — stable dynamics
let the critic predict the full discounted return from obs_0. step() decodes
the action into a split plan, estimates latency, and returns a shaped reward.
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from network_dynamics import ComputeLoadModel, LinkBWModel
from config import (
    ALPHA,
    MAX_CHUNKS,
    MAX_GROUPS,
    NUM_LAYERS,
    NUM_NODES,
    SEED,
    BW_CONNECTED_THRESHOLD,
    MIN_DISTINCT_NODES,
    SINGLE_NODE_PENALTY,
    WORKER_DIVERSITY_PENALTY,
    NODE_UTIL_BONUS,
    MEMORY_OVERLOAD_PENALTY,
    MAX_NODES,
    EPISODE_LENGTH,
    ROUTING_QUALITY_BONUS,
    WEAK_LINK_BW_THRESHOLD,
    WEAK_LINK_PENALTY_BETA,
)
# Curriculum annealing defaults (overridden by train_GReMS.py via env.connectivity_penalty_beta)
from config import CONN_PENALTY_START as _CONN_PENALTY_START
from action_space import (
    ActionSpec, decode_action, compute_connectivity_penalty, compute_weak_link_penalty,
)
from reward import compute_reward
from state_encoder import encode_state, state_dim, get_path_hub_info
from profiling import load_layer_profiles
from utils import setup_logging

logger = setup_logging("environment")


class SplitEnv(gym.Env):
    """Gymnasium environment for RL-based split-inference placement on variable-topology networks."""

    # gym requires this field; no rendering is implemented
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_nodes: int = NUM_NODES,
        max_groups: int = MAX_GROUPS,
        max_chunks: int = MAX_CHUNKS,
        alpha: float = ALPHA,
        use_offline_profiles: bool = True,
        seed: Optional[int] = SEED,
        inject_node_profiles: Optional[list] = None,
        inject_link_profiles: Optional[dict] = None,
        network_id: Optional[int] = None,
        use_bg_load: bool = True,
    ):
        """network_id is encoded in the observation so the agent can distinguish topologies."""
        super().__init__()

        self.num_nodes = num_nodes
        self.max_groups = max_groups
        self.alpha = alpha
        self.use_offline = use_offline_profiles

        self._inject_node_profiles = inject_node_profiles
        self._inject_link_profiles = inject_link_profiles
        self._network_id = network_id
        self._use_bg_load = use_bg_load
        # Graph-level state (set by reset; used for path/hub-candidate features)
        self._graph = None
        self._src_node: int = 0
        self._dst_node: int = 0
        # Path/hub cache (set by reset; included in step() info for logging)
        self._path_nodes: List[int] = []
        self._hub_scores_norm: np.ndarray = np.array([], dtype=np.float32)

        # Global ceilings keep checkpoint shapes consistent across topology sizes
        from config import MAX_NODES as _MAX_NODES
        self.spec_action = ActionSpec(
            num_layers=NUM_LAYERS,
            max_groups=max_groups,
            max_chunks=MAX_CHUNKS,   # global ceiling
            num_nodes=_MAX_NODES,    # global ceiling (not per-env num_nodes)
        )

        obs_dim = state_dim(max_groups)  # fixed size regardless of node count
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            self.spec_action.multi_discrete_nvec
        )

        self.layer_profiles = None
        self.node_profiles = None
        self.link_profiles = None

        if self.use_offline:
            self.layer_profiles = load_layer_profiles()
            logger.info("Loaded offline layer profiles (%d layers).",
                        len(self.layer_profiles))

        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        else:
            self._np_random = np.random.default_rng()

        # Background compute load model (MMPP, per-node independent chains)
        self._load_model = ComputeLoadModel(
            num_nodes=self.num_nodes, rng=self._np_random
        )
        # Link bandwidth dynamics model (G-E, per-link independent chains)
        self._link_bw_model = LinkBWModel(
            num_nodes=self.num_nodes, rng=self._np_random
        )
        self._link_bw_initialized = False   # first reset() hard-inits; rest accumulate
        self._step_count: int = 0
        self._prev_action: Optional[np.ndarray] = None

        # Curriculum connectivity penalty — updated externally by train_GReMS.py
        self.connectivity_penalty_beta: float = _CONN_PENALTY_START

        self._current_obs: Optional[np.ndarray] = None
        self._base_adjacency: Optional[np.ndarray] = None  # physical edges (BW>0), fixed per episode
        self._adjacency: Optional[np.ndarray] = None       # G-E-degraded edges (BW>threshold), for obs

    # ─────────────────────────────────────────
    # Gym API
    # ─────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Advance MMPP/G-E chains, sample fresh channel state, return obs_0."""
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Advance MMPP and G-E chains once; channel state is stable for the full episode
        if self._use_bg_load:
            self._load_model.resize(self.num_nodes)
            self._load_model.advance()  # advance once per episode (stable within episode)
        self.node_profiles = self._sample_node_profiles()
        if self._inject_link_profiles is None:
            raise RuntimeError(
                "No link profiles injected. Pass inject_link_profiles from "
                "generate_profiles_from_disnet_graph()."
            )
        self.link_profiles = dict(self._inject_link_profiles)  # working copy; step() may overwrite BW

        base_bw = np.array(self.link_profiles["bandwidth"], dtype=np.float32)
        self._link_bw_model.resize(self.num_nodes)
        if not self._link_bw_initialized:
            # First episode: hard-init so chain starts from all-GOOD base state
            self._link_bw_model.reset(base_bw)
            self._link_bw_initialized = True
        else:
            # Subsequent episodes: update topology BW, preserve accumulated chain state
            self._link_bw_model.set_base_bw(base_bw)
        self._link_bw_model.advance()  # advance once per episode (stable within episode)
        eff_bw = self._link_bw_model.get_bandwidth_matrix()
        self.link_profiles["bandwidth"] = eff_bw.tolist()

        # Base adjacency from original BW — MMPP degrades throughput but never severs edges
        self._base_adjacency = (base_bw > 0.0).astype(np.float32)
        np.fill_diagonal(self._base_adjacency, 1.0)

        # Reset episode counter and previous action
        self._step_count = 0
        self._prev_action = np.zeros(
            len(self.spec_action.multi_discrete_nvec), dtype=np.float32
        )

        # ── (Re-)build NetworkX graph for path/hub-candidate features ──
        self._graph = self._build_nx_graph()
        n = self.num_nodes
        self._src_node = int(self._np_random.integers(0, n))
        self._dst_node = int(self._np_random.integers(0, n))
        if self._dst_node == self._src_node:
            self._dst_node = (self._src_node + 1) % n

        # Compute and cache path/hub info for this episode
        self._path_nodes, self._hub_scores_norm = get_path_hub_info(
            self.node_profiles, self.link_profiles,
            graph=self._graph, src_node=self._src_node, dst_node=self._dst_node,
        )

        # Layer profiles: use offline or synthetic
        if self.layer_profiles is None:
            self.layer_profiles = self._synthetic_layer_profiles()

        # Compute binary adjacency matrix from link bandwidth
        bw = np.array(self.link_profiles["bandwidth"])
        self._adjacency = (bw > BW_CONNECTED_THRESHOLD).astype(np.float32)
        np.fill_diagonal(self._adjacency, 1.0)

        self._current_obs = encode_state(
            self.node_profiles,
            self.link_profiles,
            self.layer_profiles,
            max_groups=self.max_groups,
            network_id=self._network_id,
            graph=self._graph,
            src_node=self._src_node,
            dst_node=self._dst_node,
            prev_action=self._prev_action,
        )

        return self._current_obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Decode action, compute shaped reward, return next obs; terminates after EPISODE_LENGTH steps."""
        action = np.asarray(action, dtype=np.int64)

        # Clamp hub/assignment IDs to actual node count (spec_action uses global MAX_NODES)
        from dataclasses import replace as _dc_replace
        decode_spec = _dc_replace(self.spec_action, num_nodes=self.num_nodes)
        decoded = decode_action(action, decode_spec,
                                adjacency=self._base_adjacency)

        # Use the Jetson-Nano-calibrated LUT (same model as DiSNet/MODNN baselines)
        from run_rl_live import measure_rl_plan_latency_ms
        latency = measure_rl_plan_latency_ms(
            plan=decoded,
            link_profiles=self.link_profiles,
            device="cpu",
            node_profiles=self.node_profiles,
        )
        base_reward = compute_reward(latency, self.alpha)

        # Connectivity penalty on physical topology only (MMPP degradation never triggers it)
        conn_penalty = 0.0
        if self._base_adjacency is not None:
            conn_penalty = compute_connectivity_penalty(
                decoded["group_sizes"],
                decoded["chunks"],
                decoded["hub_node"],
                decoded["assignment"],
                self._base_adjacency,
            )

        # Penalty for a weak-but-nonzero direct link to the hub
        weak_link_penalty = 0.0
        if self.link_profiles is not None:
            weak_link_penalty = compute_weak_link_penalty(
                decoded["group_sizes"],
                decoded["chunks"],
                decoded["hub_node"],
                decoded["assignment"],
                np.array(self.link_profiles["bandwidth"]),
                WEAK_LINK_BW_THRESHOLD,
            )

        reward = base_reward - self.connectivity_penalty_beta * conn_penalty
        reward -= WEAK_LINK_PENALTY_BETA * weak_link_penalty

        # Count nodes actually assigned to a group chunk (hub may or may not be among them)
        used_nodes: set = set()
        for g in range(len(decoded["group_sizes"])):
            if decoded["group_sizes"][g] == 0:
                continue
            if g >= len(decoded["assignment"]):
                continue
            n_c = decoded["chunks"][g] if g < len(decoded["chunks"]) else 1
            for c in range(min(n_c, len(decoded["assignment"][g]))):
                used_nodes.add(decoded["assignment"][g][c])
        num_distinct = len(used_nodes)

        # Require hub + at least 1 worker; scale up to MIN_DISTINCT_NODES on larger nets
        min_required = max(2, min(MIN_DISTINCT_NODES, self.num_nodes - 1))
        node_penalty = 0.0
        if num_distinct < min_required:
            node_penalty = SINGLE_NODE_PENALTY
        reward -= node_penalty

        # Penalise groups where all chunks land on the same node
        # (defeats the purpose of horizontal splitting)
        diversity_penalty = 0.0
        for g in range(len(decoded["group_sizes"])):
            if decoded["group_sizes"][g] == 0:
                continue
            n_c = decoded["chunks"][g] if g < len(decoded["chunks"]) else 1
            if n_c <= 1:
                continue
            assigned = decoded["assignment"][g] if g < len(decoded["assignment"]) else []
            active = assigned[:min(n_c, len(assigned))]
            if len(set(active)) <= 1:
                diversity_penalty += WORKER_DIVERSITY_PENALTY
        reward -= diversity_penalty

        # Penalise groups exceeding NUM_LAYERS/ceil(sqrt(N)) layers — guides agent to use more groups
        import math
        effective_threshold = NUM_LAYERS / math.ceil(math.sqrt(self.num_nodes))
        memory_penalty = 0.0
        for g_size in decoded["group_sizes"]:
            if g_size == 0:
                continue
            excess = max(0, g_size - effective_threshold)
            memory_penalty += (excess / NUM_LAYERS) * MEMORY_OVERLOAD_PENALTY
        reward -= memory_penalty

        # Bonus scales linearly with extra nodes used beyond min_required
        extra_nodes = max(0, num_distinct - min_required)
        max_extra   = max(1, self.num_nodes - min_required)
        node_util_bonus = NODE_UTIL_BONUS * (extra_nodes / max_extra)
        reward += node_util_bonus

        num_active_groups = sum(1 for gs in decoded["group_sizes"] if gs > 0)

        # Bonus for routing activations over GOOD-state G-E links between consecutive groups
        link_states = self._link_bw_model.get_link_states()  # (N,N): 0=GOOD, 1=BAD
        good_tx = 0
        total_tx = 0
        active_gs = [g for g in range(len(decoded["group_sizes"])) if decoded["group_sizes"][g] > 0]
        for idx in range(len(active_gs) - 1):
            g_a, g_b = active_gs[idx], active_gs[idx + 1]
            asgn = decoded["assignment"]
            node_a = asgn[g_a][0] if g_a < len(asgn) and asgn[g_a] else None
            node_b = asgn[g_b][0] if g_b < len(asgn) and asgn[g_b] else None
            if node_a is not None and node_b is not None and node_a != node_b:
                total_tx += 1
                if link_states[node_a, node_b] == 0:  # 0 = GOOD
                    good_tx += 1
        routing_quality_bonus = ROUTING_QUALITY_BONUS * (good_tx / total_tx) if total_tx > 0 else 0.0
        reward += routing_quality_bonus

        info = {
            "latency_ms": latency,
            "base_reward": base_reward,
            "conn_penalty": conn_penalty,
            "weak_link_penalty": weak_link_penalty,
            "node_penalty": node_penalty,
            "diversity_penalty": diversity_penalty,
            "memory_penalty": memory_penalty,
            "node_util_bonus": node_util_bonus,
            "routing_quality_bonus": routing_quality_bonus,
            "num_active_groups": num_active_groups,
            "hub_node": decoded["hub_node"],
            "num_distinct_nodes": num_distinct,
            "num_nodes": self.num_nodes,
            "network_id": self._network_id,
            "group_sizes": decoded["group_sizes"],
            "chunks": decoded["chunks"],
            "assignment": decoded["assignment"],
            # Path/hub diagnostic info (for training log)
            "src_node": self._src_node,
            "dst_node": self._dst_node,
            "path_nodes": self._path_nodes,
            "hub_scores_norm": self._hub_scores_norm,
            # MMPP background load diagnostics
            "mean_node_load": float(np.mean([n["current_load"] for n in self.node_profiles])),
            "frac_high_load": float(np.mean([1.0 if n["current_load"] > 0.5 else 0.0 for n in self.node_profiles])),
        }

        # Normalise action into [0,1] and store as prev_action for next obs
        nvec = np.array(self.spec_action.multi_discrete_nvec, dtype=np.float32)
        self._prev_action = action.astype(np.float32) / np.maximum(nvec - 1.0, 1.0)

        # Encode next observation
        self._current_obs = encode_state(
            self.node_profiles,
            self.link_profiles,
            self.layer_profiles,
            max_groups=self.max_groups,
            network_id=self._network_id,
            graph=self._graph,
            src_node=self._src_node,
            dst_node=self._dst_node,
            prev_action=self._prev_action,
        )

        # Advance step counter and check episode termination
        self._step_count += 1
        terminated = self._step_count >= EPISODE_LENGTH
        truncated = False

        return self._current_obs, float(reward), terminated, truncated, info

    def advance_dynamics(self, encode_obs: bool = True) -> Optional[np.ndarray]:
        """Advance MMPP/G-E chains and return updated obs; used in evaluation only (not called during training).
        encode_obs=False skips observation encoding when the caller builds its own (e.g. robustness sweeps)."""
        if self._use_bg_load:
            self._load_model.advance()
            self.node_profiles = self._sample_node_profiles()

        self._link_bw_model.advance()
        eff_bw = self._link_bw_model.get_bandwidth_matrix()
        self.link_profiles["bandwidth"] = eff_bw.tolist()

        bw = np.array(self.link_profiles["bandwidth"])
        self._adjacency = (bw > BW_CONNECTED_THRESHOLD).astype(np.float32)
        np.fill_diagonal(self._adjacency, 1.0)

        self._graph = self._build_nx_graph()

        if not encode_obs:
            return None

        self._path_nodes, self._hub_scores_norm = get_path_hub_info(
            self.node_profiles, self.link_profiles,
            graph=self._graph, src_node=self._src_node, dst_node=self._dst_node,
        )

        self._current_obs = encode_state(
            self.node_profiles,
            self.link_profiles,
            self.layer_profiles,
            max_groups=self.max_groups,
            network_id=self._network_id,
            graph=self._graph,
            src_node=self._src_node,
            dst_node=self._dst_node,
            prev_action=self._prev_action,
        )
        return self._current_obs.copy()

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _sample_node_profiles(self):
        """Return node profiles with randomised current loads."""
        if self._inject_node_profiles is None:
            raise RuntimeError(
                "No node profiles injected. Pass inject_node_profiles from "
                "generate_profiles_from_disnet_graph()."
            )
        import copy
        nodes = copy.deepcopy(self._inject_node_profiles)
        if self._use_bg_load:
            # Set load from MMPP model (already advanced in reset() before this call)
            for i, n in enumerate(nodes):
                n["current_load"] = self._load_model.get_utilization(i)
        else:
            # No background load: current_load=0.0 → _bg_slowdown()=1.0 (no slowdown)
            for n in nodes:
                n["current_load"] = 0.0
        return nodes

    def _synthetic_layer_profiles(self):
        """Fallback layer profiles from MODEL_CFG when use_offline_profiles=False; forward_time_ms is unused by the LUT latency model."""
        from config import MODEL_CFG
        profiles = []
        for idx, cfg in enumerate(MODEL_CFG):
            profiles.append({
                "layer_index": idx,
                "layer_type": cfg[0],
                "forward_time_ms": float(self._np_random.uniform(0.1, 5.0)),
                "peak_memory_bytes": 0,
                "flops": int(cfg[-1]),
                "activation_size": int(cfg[4]),
            })
        return profiles

    def _build_nx_graph(self):
        """Build a NetworkX graph from current node/link profiles."""
        try:
            import networkx as nx
            G = nx.Graph()
            n = self.num_nodes
            for i, np_ in enumerate(self.node_profiles):
                G.add_node(i, weight=np_.get("compute_capacity", 1.0))
            bw = np.array(self.link_profiles["bandwidth"])
            for i in range(min(n, bw.shape[0])):
                for j in range(i + 1, min(n, bw.shape[1])):
                    if bw[i][j] > BW_CONNECTED_THRESHOLD:
                        G.add_edge(i, j, weight=float(bw[i][j]))
            return G
        except Exception:
            return None


# ──────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    env = SplitEnv(use_offline_profiles=False)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    action = env.action_space.sample()
    obs2, reward, term, trunc, info = env.step(action)
    print(f"Reward: {reward:.4f}  Latency: {info['latency_ms']:.2f} ms")
    print(f"Groups: {info['group_sizes']}  Chunks: {info['chunks']}")
    print(f"Assignment: {info['assignment']}")
