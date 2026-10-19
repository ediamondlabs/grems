"""
Encodes system state into a fixed-size observation vector for the GReMS agent.

Aggregate stats (Blocks 1-5) are topology-size-invariant; per-node blocks
(6+) are zero-padded to MAX_NODES so the dimension is always fixed.

Observation layout (542 dims total):
  Block 1  node_stats         16  mean/std/min/max of [compute, load, eff_cap, degree]
  Block 2  link_stats          8  mean/std/min/max of [bandwidth, latency]
  Block 3  topo_meta           5  num_nodes_norm, edge_density, degree stats ×3
  Block 4  model_feats        12  cumul_flops + max_activation per group (MAX_GROUPS×2)
  Block 5  topo_struct        15  clustering, assortativity, path lengths, algebraic connectivity
  Block 6  per_node_feats     80  [compute, load, eff_cap, degree] per node, zero-padded
  Block 7  path_hub           40  [path_flag, hub_score_norm] per node, zero-padded
  Block 8  layer_split_costs  42  [activation_size_norm, cumul_flops_frac] per layer
  Block 9  bottleneck_bw     190  pairwise min-path-bandwidth upper triangle, zero-padded
  Block 10 prev_action       134  normalised previous action vector
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    MAX_CHUNKS,
    MAX_GROUPS,
    MAX_NODES,
    BW_CONNECTED_THRESHOLD,
    BW_NORM_SCALE,
)

# ─────────────────────────────────────────────────────
# Block sizes (kept as module constants for easy import)
# ─────────────────────────────────────────────────────
_N_STATS       = 4          # mean, std, min, max
_NODE_FEATS    = 4          # compute_capacity, current_load, effective_capacity, degree
_LINK_FEATS    = 2          # bandwidth, latency
MODEL_FEAT_DIM  = 2          # cumulative_flops, activation_size per group
TOPO_STRUCT_DIM = 15  # 11 structural features + 4 reserved (replaces network-ID one-hot)

NODE_STAT_DIM        = _N_STATS * _NODE_FEATS                # 16
LINK_STAT_DIM        = _N_STATS * _LINK_FEATS                # 8
TOPO_META_DIM        = 5                                     # num_nodes_norm, density, deg×3
MODEL_DIM            = MAX_GROUPS * MODEL_FEAT_DIM           # 12
PER_NODE_DIM         = MAX_NODES * _NODE_FEATS               # 80  (padded per-node feats)
PATH_HUB_DIM         = MAX_NODES * 2                         # 40  (Block 7)
LAYER_SPLIT_COST_DIM = 21 * 2                                # 42  (Block 8: activation + flops per layer)
BOTTLENECK_BW_DIM    = MAX_NODES * (MAX_NODES - 1) // 2     # 190 (Block 9: pairwise min-path-BW)

# Layout: 1(num_groups) + MAX_GROUPS(vert) + MAX_GROUPS(chunks) + 1(hub) + MAX_GROUPS*MAX_CHUNKS(assign)
PREV_ACTION_DIM = 1 + MAX_GROUPS + MAX_GROUPS + 1 + MAX_GROUPS * MAX_CHUNKS   # 134


def state_dim(max_groups: int = MAX_GROUPS) -> int:
    """Return the fixed observation length (independent of topology size)."""
    return (
        NODE_STAT_DIM
        + LINK_STAT_DIM
        + TOPO_META_DIM
        + max_groups * MODEL_FEAT_DIM
        + TOPO_STRUCT_DIM
        + PER_NODE_DIM
        + PATH_HUB_DIM           # Block 7
        + LAYER_SPLIT_COST_DIM   # Block 8: per-layer split costs
        + BOTTLENECK_BW_DIM      # Block 9: pairwise min-path-bandwidth
        + PREV_ACTION_DIM        # Block 10: previous action
    )



# ─────────────────────────────────────────────────────
# Individual encoders
# ─────────────────────────────────────────────────────

def _aggregate(arr: np.ndarray) -> np.ndarray:
    """Return [mean, std, min, max] of a 1-D array."""
    if len(arr) == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array([arr.mean(), arr.std(), arr.min(), arr.max()], dtype=np.float32)


def encode_node_stats(
    node_profiles: List[Dict],
    link_profiles: Optional[Dict] = None,
) -> np.ndarray:
    """Block 1: [mean,std,min,max] of [compute, load, eff_cap, degree] → (NODE_STAT_DIM,)."""
    num_nodes = len(node_profiles)

    degrees = np.zeros(num_nodes, dtype=np.float32)
    if link_profiles is not None:
        bw  = np.asarray(link_profiles["bandwidth"], dtype=np.float32)
        adj = (bw > BW_CONNECTED_THRESHOLD).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        degrees = adj.sum(axis=1) / max(num_nodes, 1)

    comp = np.array([n["compute_capacity"] for n in node_profiles], dtype=np.float32)
    load = np.array([n["current_load"]     for n in node_profiles], dtype=np.float32)
    eff  = comp * (1.0 - load)   # effective_capacity

    return np.concatenate([
        _aggregate(comp),
        _aggregate(load),
        _aggregate(eff),
        _aggregate(degrees),
    ])


def encode_link_stats(link_profiles: Dict) -> np.ndarray:
    """Block 2: [mean,std,min,max] of [bandwidth, latency] for all node pairs → (LINK_STAT_DIM,)."""
    bw  = np.asarray(link_profiles["bandwidth"], dtype=np.float32)
    lat = np.asarray(link_profiles["latency"],   dtype=np.float32)
    n   = bw.shape[0]

    mask     = ~np.eye(n, dtype=bool)
    bw_vals  = np.where(np.isfinite(bw[mask]),  bw[mask],  0.0) / BW_NORM_SCALE
    lat_vals = np.where(np.isfinite(lat[mask]), lat[mask], 0.0)  # not normalised: values are small (≈0.5 ms)

    return np.concatenate([
        _aggregate(bw_vals),
        _aggregate(lat_vals),
    ])


def encode_topo_meta(
    node_profiles: List[Dict],
    link_profiles: Dict,
) -> np.ndarray:
    """Block 3: [num_nodes_norm, edge_density, mean/min/max_degree_norm] → (TOPO_META_DIM,)."""
    n = len(node_profiles)
    bw  = np.asarray(link_profiles["bandwidth"], dtype=np.float32)
    adj = (bw > BW_CONNECTED_THRESHOLD).astype(np.float32)
    np.fill_diagonal(adj, 0.0)

    degree            = adj.sum(axis=1)
    max_possible_edges = max(n * (n - 1) / 2, 1.0)
    n_edges            = adj.sum() / 2.0

    return np.array([
        n / MAX_NODES,                          # relative network size
        n_edges / max_possible_edges,           # edge density
        degree.mean() / max(n - 1, 1),          # mean normalised degree
        degree.min()  / max(n - 1, 1),          # min normalised degree
        degree.max()  / max(n - 1, 1),          # max normalised degree
    ], dtype=np.float32)


def encode_model_features(
    layer_profiles: List[Dict],
    group_boundaries: Optional[List[int]] = None,
    max_groups: int = MAX_GROUPS,
) -> np.ndarray:
    """Block 4: cumulative FLOPs + max activation size per group → (max_groups, MODEL_FEAT_DIM)."""
    num_layers = len(layer_profiles)

    if group_boundaries is None:
        group_boundaries = np.linspace(0, num_layers, max_groups + 1, dtype=int).tolist()
    else:
        if group_boundaries[-1] != num_layers:
            group_boundaries = list(group_boundaries) + [num_layers]

    feats = np.zeros((max_groups, MODEL_FEAT_DIM), dtype=np.float32)
    for g in range(min(max_groups, len(group_boundaries) - 1)):
        start, end = group_boundaries[g], group_boundaries[g + 1]
        feats[g, 0] = sum(lp["flops"] for lp in layer_profiles[start:end])
        feats[g, 1] = max(
            (lp["activation_size"] for lp in layer_profiles[start:end]), default=0
        )
    return feats


def get_path_hub_info(
    node_profiles: List[Dict],
    link_profiles: Dict,
    graph=None,
    src_node: int = 0,
    dst_node: int = 0,
) -> Tuple[List[int], np.ndarray]:
    """Return (path_nodes, hub_scores_norm) for src→dst; shared by encoder and training log."""
    n = len(node_profiles)
    path_nodes: set = set()
    try:
        import networkx as nx
        if graph is not None and graph.number_of_nodes() > 0:
            G = graph
        else:
            bw = np.asarray(link_profiles["bandwidth"], dtype=np.float32)
            G = nx.Graph()
            for i in range(min(n, bw.shape[0])):
                G.add_node(i, weight=node_profiles[i].get("compute_capacity", 1.0))
            for i in range(min(n, bw.shape[0])):
                for j in range(i + 1, min(n, bw.shape[1])):
                    if bw[i][j] > BW_CONNECTED_THRESHOLD:
                        G.add_edge(i, j, weight=float(bw[i][j]))
        if G.has_node(src_node) and G.has_node(dst_node) and nx.has_path(G, src_node, dst_node):
            # weight=None → unweighted shortest path (fewest hops)
            sp = nx.shortest_path(G, src_node, dst_node, weight=None)
            path_nodes = set(sp)
    except Exception:
        path_nodes = set()

    bw_arr = np.asarray(link_profiles["bandwidth"], dtype=np.float32)
    hub_scores = np.zeros(n, dtype=np.float32)
    for i in range(min(n, bw_arr.shape[0])):
        cap = float(node_profiles[i].get("compute_capacity", 1.0))
        nbrs = [
            float(bw_arr[i][j])
            for j in path_nodes
            if j < bw_arr.shape[1] and i != j and bw_arr[i][j] > BW_CONNECTED_THRESHOLD
        ]
        hub_scores[i] = cap * (sum(nbrs) if nbrs else 0.0)

    max_score = hub_scores.max() if hub_scores.max() > 0 else 1.0
    return sorted(path_nodes), hub_scores / max_score


def encode_path_hub_features(
    node_profiles: List[Dict],
    link_profiles: Dict,
    max_nodes: int = MAX_NODES,
    graph=None,
    src_node: int = 0,
    dst_node: int = 0,
) -> np.ndarray:
    """Block 8: [path_flag, hub_score_norm] per node, zero-padded → (PATH_HUB_DIM,)."""
    path_nodes_list, hub_scores_norm = get_path_hub_info(
        node_profiles, link_profiles, graph, src_node, dst_node
    )
    path_set = set(path_nodes_list)
    n = len(node_profiles)
    out = np.zeros((max_nodes, 2), dtype=np.float32)
    for i in range(min(n, max_nodes)):
        out[i, 0] = 1.0 if i in path_set else 0.0
        out[i, 1] = hub_scores_norm[i] if i < len(hub_scores_norm) else 0.0
    return out.flatten()


def encode_per_node_features(
    node_profiles: List[Dict],
    link_profiles: Dict,
    max_nodes: int = MAX_NODES,
) -> np.ndarray:
    """Block 6: [compute_norm, load, eff_cap_norm, degree_norm] per node, zero-padded → (PER_NODE_DIM,)."""
    n = len(node_profiles)
    bw  = np.asarray(link_profiles["bandwidth"], dtype=np.float32)
    adj = (bw > BW_CONNECTED_THRESHOLD).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    degrees = adj.sum(axis=1)  # shape (n,)

    comp = np.array([nd["compute_capacity"] for nd in node_profiles], dtype=np.float32)
    load = np.array([nd["current_load"]     for nd in node_profiles], dtype=np.float32)
    eff  = comp * (1.0 - load)   # effective_capacity

    max_comp   = comp.max() if comp.max() > 0 else 1.0
    max_eff    = eff.max()  if eff.max()  > 0 else 1.0
    max_degree = max(max_nodes - 1, 1)

    out = np.zeros((max_nodes, 4), dtype=np.float32)
    for i in range(min(n, max_nodes)):
        out[i, 0] = comp[i] / max_comp      # compute capacity (normalised)
        out[i, 1] = load[i]                 # current load (already [0,1])
        out[i, 2] = eff[i]  / max_eff       # effective capacity (normalised)
        out[i, 3] = degrees[i] / max_degree # degree (normalised)

    return out.flatten()   # shape (max_nodes * 4,)


# ─────────────────────────────────────────────────────
# Block 5: Structural graph features
# ─────────────────────────────────────────────────────

def _encode_topo_structural(
    node_profiles: List[Dict],
    link_profiles: Dict,
    max_dims: int = TOPO_STRUCT_DIM,
    graph=None,
) -> np.ndarray:
    """Block 5: structural graph features → (TOPO_STRUCT_DIM,).
    Indices: 0=mean_clust, 1=std_clust, 2=transitivity, 3=assortativity,
    4=avg_path/N, 5=diameter/MAX_N, 6-8=degree max/min/std, 9=tri_density,
    10=algebraic_connectivity, 11-14=reserved zeros."""
    out = np.zeros(max_dims, dtype=np.float32)
    try:
        import networkx as nx

        G = graph
        if G is None or G.number_of_nodes() == 0:
            bw = np.asarray(link_profiles["bandwidth"], dtype=np.float32)
            n_bw = bw.shape[0]
            G = nx.Graph()
            for i in range(n_bw):
                G.add_node(i)
            for i in range(n_bw):
                for j in range(i + 1, n_bw):
                    if bw[i][j] > BW_CONNECTED_THRESHOLD:
                        G.add_edge(i, j)

        n = G.number_of_nodes()
        if n < 2:
            return out

        degrees = [d for _, d in G.degree()]
        max_deg = max(degrees) if degrees else 1

        # 0-1: clustering coefficients
        clust_vals = list(nx.clustering(G).values())
        out[0] = float(np.mean(clust_vals))
        out[1] = float(np.std(clust_vals))

        # 2: transitivity
        out[2] = float(nx.transitivity(G))

        # 3: degree assortativity, normalised to [0, 1].
        # nx returns float('nan') (not an exception) when all degrees are equal
        # (zero variance), so we must check isfinite rather than rely on try/except.
        try:
            _assort = nx.degree_assortativity_coefficient(G)
            out[3] = float((_assort + 1.0) / 2.0) if np.isfinite(_assort) else 0.5
        except Exception:
            out[3] = 0.5

        # 4: avg shortest path length (normalised)
        try:
            out[4] = float(nx.average_shortest_path_length(G)) / max(n, 1)
        except Exception:
            out[4] = 1.0   # disconnected – worst case

        # 5: diameter (normalised)
        try:
            out[5] = float(nx.diameter(G)) / MAX_NODES
        except Exception:
            out[5] = 1.0

        # 6-8: degree stats
        norm = max(n - 1, 1)
        out[6] = float(max(degrees)) / norm
        out[7] = float(min(degrees)) / norm
        out[8] = float(np.std(degrees)) / norm

        # 9: triangle density
        triangles = sum(nx.triangles(G).values()) // 3
        max_tri = max(n * (n - 1) * (n - 2) // 6, 1)
        out[9] = float(triangles) / max_tri

        # 10: algebraic connectivity (Fiedler) normalised by max degree
        try:
            _alg = nx.algebraic_connectivity(G)
            out[10] = float(_alg) / max(max_deg, 1) if np.isfinite(_alg) else 0.0
        except Exception:
            out[10] = 0.0

        # 11-14: reserved (zeros)

    except Exception:
        pass   # return zero vector on any unexpected failure

    # Belt-and-suspenders: replace any remaining NaN/Inf that slipped through
    # (e.g. from NetworkX returning silent NaN on degenerate graphs) with 0.
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def encode_layer_split_costs(layer_profiles: List[Dict]) -> np.ndarray:
    """Block 9: [activation_size_norm, cumul_flops_frac] per layer → (LAYER_SPLIT_COST_DIM,)."""
    costs = np.array(
        [float(lp.get("activation_size", 0)) for lp in layer_profiles],
        dtype=np.float32,
    )
    max_cost = costs.max() if costs.max() > 0 else 1.0
    act_norm = costs / max_cost

    flops = np.array(
        [float(lp.get("flops", 0)) for lp in layer_profiles],
        dtype=np.float32,
    )
    total_flops = flops.sum() if flops.sum() > 0 else 1.0
    cumul_flops = np.cumsum(flops) / total_flops   # fraction in [0, 1]

    return np.stack([act_norm, cumul_flops], axis=1).flatten().astype(np.float32)


def encode_bottleneck_bw(
    link_profiles: Dict,
    max_nodes: int = MAX_NODES,
) -> np.ndarray:
    """Block 10: min-path-bandwidth upper triangle, normalised by BW_NORM_SCALE → (BOTTLENECK_BW_DIM,)."""
    import networkx as nx

    bw  = np.asarray(link_profiles["bandwidth"], dtype=np.float32)
    n   = bw.shape[0]

    # Build weighted graph (weight = bandwidth)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if bw[i, j] > BW_CONNECTED_THRESHOLD:
                G.add_edge(i, j, weight=float(bw[i, j]))

    # For each pair, find max-bandwidth path (maximise minimum edge weight)
    # networkx.maximum_spanning_tree gives the path that maximises min edge BW
    padded = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    for i in range(min(n, max_nodes)):
        for j in range(i + 1, min(n, max_nodes)):
            try:
                # Shortest path weighted by inverse BW → bottleneck = min BW on path
                path = nx.shortest_path(G, i, j, weight=None)  # fewest hops
                min_bw = min(
                    bw[path[k], path[k + 1]] for k in range(len(path) - 1)
                ) if len(path) > 1 else bw[i, j]
                padded[i, j] = float(min_bw) / BW_NORM_SCALE
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                padded[i, j] = 0.0  # unreachable → worst case

    rows, cols = np.triu_indices(max_nodes, k=1)
    return padded[rows, cols].astype(np.float32)


# ─────────────────────────────────────────────────────
# Full state encoder
# ─────────────────────────────────────────────────────

def encode_state(
    node_profiles: List[Dict],
    link_profiles: Dict,
    layer_profiles: List[Dict],
    group_boundaries: Optional[List[int]] = None,
    max_groups: int = MAX_GROUPS,
    normalize: bool = True,
    network_id: Optional[int] = None,
    # max_nodes is accepted but ignored – retained for call-site compatibility
    max_nodes: Optional[int] = None,
    # Graph and routing context for Block 8
    graph=None,
    src_node: int = 0,
    dst_node: int = 0,
    # Block 11: previous action (normalised floats in [0,1], length PREV_ACTION_DIM)
    prev_action: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Assemble all observation blocks into a flat float32 vector of shape (state_dim(max_groups),)."""
    _max_n = max_nodes if max_nodes is not None else MAX_NODES

    # ── Blocks 1–4: aggregate statistics ──
    node_stats  = encode_node_stats(node_profiles, link_profiles)
    link_stats  = encode_link_stats(link_profiles)
    topo_meta   = encode_topo_meta(node_profiles, link_profiles)
    model_feats = encode_model_features(layer_profiles, group_boundaries, max_groups).flatten()

    if normalize:
        node_stats  = _zscore(node_stats)
        link_stats  = _zscore(link_stats)
        model_feats = _zscore(model_feats)
        # topo_meta is already normalised to [0,1] – skip z-score

    # Block 5: structural features; network_id in signature is ignored (backward compat)
    net_id_feats = _encode_topo_structural(
        node_profiles, link_profiles, TOPO_STRUCT_DIM, graph=graph
    )

    per_node      = encode_per_node_features(node_profiles, link_profiles, _max_n)  # Block 6
    path_hub      = encode_path_hub_features(                                        # Block 7
        node_profiles, link_profiles, _max_n,
        graph=graph, src_node=src_node, dst_node=dst_node,
    )
    split_costs   = encode_layer_split_costs(layer_profiles)                         # Block 8
    bottleneck_bw = encode_bottleneck_bw(link_profiles, _max_n)                     # Block 9

    # Block 11: pad/truncate prev_action to fixed PREV_ACTION_DIM
    if prev_action is not None:
        prev_act = np.asarray(prev_action, dtype=np.float32).ravel()
        if len(prev_act) < PREV_ACTION_DIM:
            prev_act = np.pad(prev_act, (0, PREV_ACTION_DIM - len(prev_act)))
        else:
            prev_act = prev_act[:PREV_ACTION_DIM]
    else:
        prev_act = np.zeros(PREV_ACTION_DIM, dtype=np.float32)

    return np.concatenate([
        node_stats,
        link_stats,
        topo_meta,
        model_feats,
        net_id_feats,
        per_node,
        path_hub,
        split_costs,
        bottleneck_bw,
        prev_act,
    ]).astype(np.float32)


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def _zscore(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu, std = arr.mean(), arr.std()
    return arr - mu if std < eps else (arr - mu) / std
