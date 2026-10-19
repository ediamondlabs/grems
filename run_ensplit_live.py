"""run_ensplit_live.py – EnSplit vertical-split baseline: extract a N_CHAIN_NODES=4 chain from src→dst,
then brute-force the minimum-latency contiguous layer assignment (≤969 candidates). Optionally uses a trained DDQN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ensplit_env import (
    N_CHAIN_NODES,
    compute_chain_latency_ms,
    enumerate_actions,
    H_MIN_UE,
)

logger = logging.getLogger("ensplit_live")
logger.setLevel(logging.WARNING)
logger.propagate = False


# ── Chain extraction from IoT graph ──────────────────────────────────────────

def _series_effective_bw(
    G: nx.Graph,
    path_segment: List[int],
) -> float:
    """Harmonic-mean effective BW (Mbit/s) for a series of hops; bottleneck BW would undercount hop costs."""
    inv_sum = 0.0
    for u, v in zip(path_segment[:-1], path_segment[1:]):
        w = max(float(G[u][v].get("weight", 1.0)), 1e-9)
        inv_sum += 1.0 / w
    return 1.0 / inv_sum if inv_sum > 0 else 1.0


def _extract_chain(
    G:           nx.Graph,
    src:         int,
    dst:         int,
    n_chain:     int = N_CHAIN_NODES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a K-node chain (K ≤ n_chain) from src→dst; returns (node_weights (K,), link_bw (K,K))."""
    try:
        path = nx.shortest_path(G, src, dst)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # No path: single-node fallback
        w = float(G.nodes[src].get("weight", 1))
        return np.array([w]), np.zeros((1, 1))

    # Sub-sample if path is longer than n_chain nodes
    if len(path) > n_chain:
        # Choose n_chain evenly-spaced indices from the path
        idx = np.round(np.linspace(0, len(path) - 1, n_chain)).astype(int)
        chain_nodes = [path[i] for i in idx]
        # Build virtual link BW as bottleneck over the sub-path segments
        segments = [path[idx[k]:idx[k + 1] + 1] for k in range(len(idx) - 1)]
        bw_list  = [_series_effective_bw(G, seg) for seg in segments]
    else:
        chain_nodes = path
        bw_list = [
            float(G[chain_nodes[i]][chain_nodes[i + 1]].get("weight", 1.0))
            for i in range(len(chain_nodes) - 1)
        ]

    K = len(chain_nodes)
    node_weights = np.array(
        [max(1.0, float(G.nodes[n].get("weight", 1))) for n in chain_nodes],
        dtype=np.float64,
    )
    link_bw = np.zeros((K, K), dtype=np.float64)
    for i, bw in enumerate(bw_list):
        link_bw[i, i + 1] = bw
        link_bw[i + 1, i] = bw   # symmetric (GReMS graphs are undirected)

    return node_weights, link_bw


# ── Brute-force optimal vertical split ───────────────────────────────────────

def _best_vertical_split(
    node_weights: np.ndarray,
    link_bw:      np.ndarray,
    h_min:        int = H_MIN_UE,
) -> Tuple[float, Tuple[int, ...]]:
    """Enumerate all valid contiguous splits and return (min_latency_ms, best_splits); ≤969 candidates for N=4."""
    K = len(node_weights)
    actions = enumerate_actions(n_nodes=K, h_min=h_min)

    best_lat   = float("inf")
    best_split = actions[0] if actions else ()

    for splits in actions:
        lat = compute_chain_latency_ms(splits, node_weights, link_bw)
        if lat < best_lat:
            best_lat   = lat
            best_split = splits

    return best_lat, best_split


# ── DDQN inference (optional, requires trained checkpoint) ───────────────────

_ddqn_cache: Dict[str, object] = {}   # path → DDQNAgent (cached after first load)


def _ddqn_infer(
    checkpoint_path: str,
    node_weights:    np.ndarray,
    link_bw:         np.ndarray,
) -> float:
    """Forward pass of the trained DDQN; falls back to brute-force on chain size mismatch."""
    import torch
    from ensplit_ddqn import DDQNAgent
    from ensplit_env import EnSplitEnv, N_CHAIN_NODES

    K = len(node_weights)

    if checkpoint_path not in _ddqn_cache:
        env = EnSplitEnv(n_nodes=N_CHAIN_NODES)
        agent = DDQNAgent(state_dim=env.state_dim, n_actions=env.n_actions)
        agent.load(checkpoint_path)
        agent.main.eval()
        _ddqn_cache[checkpoint_path] = agent
        logger.info("EnSplit DDQN checkpoint loaded from %s", checkpoint_path)

    agent = _ddqn_cache[checkpoint_path]

    # State: padded to N_CHAIN_NODES if needed
    N = N_CHAIN_NODES
    if K != N:
        # Chain mismatch — fall back to brute-force for this graph
        lat, _ = _best_vertical_split(node_weights, link_bw)
        return lat

    # Build the same state encoding as EnSplitEnv._encode_state()
    # (no energy credit at eval time → credit_frac=0)
    state = np.concatenate([
        node_weights / 10.0,
        link_bw.flatten() / 200.0,
        [0.0],                       # energy credit = 0 at eval
    ]).astype(np.float32)

    action_idx = agent.select_action(state, greedy=True)
    env_ref = EnSplitEnv(n_nodes=N)
    return env_ref.greedy_latency_ms(action_idx, node_weights, link_bw)


# ── Public interface ──────────────────────────────────────────────────────────

def run_ensplit_on_graph(
    G:               nx.Graph,
    max_bandwidth:   float = 200.0,
    num_runs:        int   = 1,
    input_node:      int   = 0,
    output_node:     Optional[int] = None,
    device:          str   = "cpu",
    timeout:         int   = 30,
    checkpoint_path: Optional[str] = None,
) -> Dict:
    """Run EnSplit on graph G; returns dict with latency_ms, split_points, chain_nodes, mode."""
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return {"latency_ms": float("inf"), "split_points": [], "chain_nodes": [], "mode": "error"}

    if output_node is None:
        output_node = max(G.nodes())

    # Cap all link BW
    G_cap = G.copy()
    for u, v in G_cap.edges():
        G_cap[u][v]["weight"] = min(
            float(G_cap[u][v].get("weight", 1.0)), max_bandwidth
        )

    # Extract chain
    node_weights, link_bw = _extract_chain(
        G_cap, src=input_node, dst=output_node, n_chain=N_CHAIN_NODES
    )

    # EnSplit requires at least N_CHAIN_NODES nodes in the graph — skip smaller ones
    if n_nodes < N_CHAIN_NODES:
        return {"latency_ms": None, "split_points": [], "chain_nodes": [], "mode": "skip"}

    try:
        path_nodes = nx.shortest_path(G_cap, input_node, output_node)
    except Exception:
        path_nodes = [input_node]

    # Select split strategy
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        lat_ms = _ddqn_infer(checkpoint_path, node_weights, link_bw)
        mode   = "ddqn"
        splits = ()   # DDQN doesn't expose split points trivially
    else:
        lat_ms, splits = _best_vertical_split(node_weights, link_bw)
        mode = "brute_force"

    logger.debug(
        "EnSplit [%s] | chain=%d nodes | latency=%.2f ms | splits=%s",
        mode, len(node_weights), lat_ms, splits,
    )

    return {
        "latency_ms":  float(lat_ms),
        "split_points": list(splits),
        "chain_nodes":  path_nodes[:N_CHAIN_NODES],
        "mode":         mode,
    }
