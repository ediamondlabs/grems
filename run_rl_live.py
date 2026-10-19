"""
run_rl_live.py – LUT-based latency estimator for the RL split plan.

Communication model (worker-to-worker, W2W):
  - Hub → group-0 workers: initial input rows + FoV overlap, once per episode.
  - Group g → group g+1: each worker k sends rows needed by each worker m (direct
    link or Dijkstra multi-hop); all (k→m) transfers happen as edges of a per-worker
    dependency DAG.
  - Final spatial group → FC node (or hub, if the plan ends on a spatial group):
    full activation, no overlap.
  - FC groups: agent-chosen node, no horizontal split, compute /= DEVICE_PACE_RATE.

Transfer formula: data_bytes / (bw_MBps × 1024²) × 1000 ms + latency_ms.
Plan keys: group_sizes, chunks, hub_node, assignment.

Latency model (paper Sec. "Latency Model"):
  Delta_{g,w} = delta_w^C + max_{u in W_{g-1,w}} { Delta_{g-1,u} + delta_{uw} }
"""

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np
from config import (
    COMPUTE_SLOWDOWN_MAX,
    DEVICE_PACE_RATE,
    IOT_REF_WEIGHT,
    LAYER_IOT_FWD_MS,
    LAYER_SPATIAL_OUT,
    MODEL_CFG,
    NUM_LAYERS,
    group_fov_overlap,
)
from config import (
    DEVICE_SCALING_DIMINISHING_RETURN,
    OVERHEAD_REMOTE_NODE_SETUP,
    OVERHEAD_PER_CHUNK,
    OVERHEAD_PER_DATA_TRANSFER,
)

# ── Helpers ────────────────────────────────────────────────────────────────

def _is_spatial(layer_idx: int) -> bool:
    """True if layer operates on a 4-D feature map (Conv or MaxPool)."""
    return MODEL_CFG[layer_idx][0] in ("C", "M")


def _multihop_transfer_ms(data_bytes: float, src: int, dst: int,
                           bw: np.ndarray, lat: np.ndarray) -> float:
    """Store-and-forward shortest-path transfer time (ms) via Dijkstra; inf if unreachable."""
    n = bw.shape[0]
    dist = np.full(n, np.inf)
    dist[src] = 0.0
    pq: List = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dst:
            return float(d)
        for v in range(n):
            if v == u or bw[u][v] <= 0:
                continue
            hop_t = data_bytes / (bw[u][v] * 1024.0 * 1024.0) * 1000.0 + float(lat[u][v])
            nd = d + hop_t
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return float(dist[dst])


def _transfer_ms(data_bytes: float, src: int, dst: int,
                  bw: np.ndarray, lat: np.ndarray) -> float:
    """Transfer time (ms): direct link if available, else shortest multi-hop path."""
    if src == dst:
        return 0.0
    bw_val = bw[src][dst]
    if bw_val > 0:
        if np.isinf(bw_val):
            return float(lat[src][dst])
        return data_bytes / (bw_val * 1024.0 * 1024.0) * 1000.0 + float(lat[src][dst])
    return _multihop_transfer_ms(data_bytes, src, dst, bw, lat)


def _pairwise_row_transfers(
    active_g:    List[int],
    node_frac_g: Dict[int, float],
    active_g1:   List[int],
    node_frac_g1: Dict[int, float],
    N_out:       int,
    overlap_g1:  int,
    C_out:       int,
    bw:          np.ndarray,
    lat:         np.ndarray,
) -> Dict[Tuple[int, int], float]:
    """Per-(sender,receiver) W2W transfer time (ms) at a spatial group boundary;
    same-node pairs omitted. Shared by both latency models below."""
    bytes_per_row = float(N_out * C_out) * 4.0   # W_out == N_out for VGG-16

    cum_g: List[tuple] = []
    c = 0.0
    for nid in active_g:
        s = c
        c += node_frac_g[nid]
        cum_g.append((s, c))

    cum_g1: List[tuple] = []
    c = 0.0
    for nid in active_g1:
        s = c
        c += node_frac_g1[nid]
        cum_g1.append((s, c))

    edges: Dict[Tuple[int, int], float] = {}

    for k, (ks, ke) in enumerate(cum_g):
        k_nid = active_g[k]
        k_r0  = ks * N_out            # first clean output row of worker k
        k_r1  = ke * N_out - 1        # last clean output row of worker k

        for m, (ms, me) in enumerate(cum_g1):
            m_nid = active_g1[m]
            if k_nid == m_nid:
                continue              # same node — no network transfer needed

            # m's required row range: proportional share ± FoV overlap, clamped
            m_r0 = max(0.0,           ms * N_out - overlap_g1)
            m_r1 = min(N_out - 1.0,  me * N_out - 1 + overlap_g1)

            i0 = max(k_r0, m_r0)  # intersection: rows k must send to m
            i1 = min(k_r1, m_r1)
            if i1 < i0:
                continue              # k has no rows that m needs

            rows = i1 - i0 + 1.0
            edges[(k_nid, m_nid)] = _transfer_ms(rows * bytes_per_row, k_nid, m_nid, bw, lat)

    return edges


def _inter_stage_transfer_ms(
    active_g:    List[int],   # ordered unique workers of group g
    node_frac_g: Dict[int, float],   # node → row-fraction for group g
    active_g1:   List[int],   # ordered unique workers of group g+1
    node_frac_g1: Dict[int, float],  # node → row-fraction for group g+1
    N_out:       int,         # output height of group g (= input height of g+1)
    overlap_g1:  int,         # group_fov_overlap(g1_l_start, g1_l_end)
    C_out:       int,         # output channels of group g's last layer
    bw:          np.ndarray,
    lat:         np.ndarray,
) -> float:
    """Bottleneck W2W transfer time (ms) at the group-g → group-g+1 boundary."""
    edges = _pairwise_row_transfers(active_g, node_frac_g, active_g1, node_frac_g1,
                                     N_out, overlap_g1, C_out, bw, lat)
    max_t = max(edges.values(), default=0.0)
    return max_t / DEVICE_PACE_RATE


# ── Pass 1: plan → per-group info (shared by both latency models) ──────────

def _build_groups(
    group_sizes: List[int],
    chunks:      List[int],
    hub:         int,
    assignment:  List[List[int]],
    node_weight_map: Dict[int, float],
) -> List[Optional[Dict]]:
    """Collect per-group plan info (lookahead needed at boundaries)."""

    def _node_weight(nid: int) -> float:
        return max(node_weight_map.get(nid, IOT_REF_WEIGHT), 1e-6)

    groups: List[Optional[Dict]] = []
    layer_offset = 0
    for g, g_size in enumerate(group_sizes):
        if g_size == 0:
            groups.append(None)
            continue

        g_end = min(layer_offset + g_size, NUM_LAYERS)

        n_chunks = max(1, chunks[g]) if g < len(chunks) else 1
        assigned = list(assignment[g][:n_chunks]) if g < len(assignment) else [hub]

        is_spatial = all(
            _is_spatial(layer_offset + i)
            for i in range(g_size)
            if layer_offset + i < NUM_LAYERS
        )
        if not is_spatial:
            n_chunks = 1
            # FC: agent-chosen node; hub as fallback if no assignment
            assigned = (list(assignment[g][:1])
                        if g < len(assignment) and assignment[g]
                        else [hub])

        n_act  = min(n_chunks, len(assigned))
        active_raw = assigned[:n_act]

        active = list(dict.fromkeys(active_raw))  # deduplicate, preserve order; fracs sum below
        remote_workers = [n for n in active if n != hub]
        has_remote = len(remote_workers) > 0

        H_in = LAYER_SPATIAL_OUT[layer_offset - 1] if layer_offset > 0 else 224

        if is_spatial and n_act > 1:
            _nw: Dict[int, float] = {}
            for nid in active_raw:
                _nw[nid] = _nw.get(nid, 0.0) + _node_weight(nid)
            _total_w = sum(_nw.values())
            node_frac = {nid: w / _total_w for nid, w in _nw.items()}
        else:
            node_frac = {nid: 1.0 for nid in active}

        groups.append({
            "g_size":        g_size,
            "layer_offset":  layer_offset,
            "g_end":         g_end,
            "is_spatial":    is_spatial,
            "active":        active,         # ordered, unique node IDs
            "remote_workers": remote_workers,
            "has_remote":    has_remote,
            "node_frac":     node_frac,
            "H_in":          H_in,
            "n_chunks":      n_chunks,
            "chunk_owners":  active_raw,     # one entry per chunk instance (duplicates allowed)
        })
        layer_offset += g_size

    return groups


def _group_compute_delta(
    info: Dict,
    hub: int,
    node_weight_map: Dict[int, float],
    node_load_map:   Dict[int, float],
) -> Dict[int, float]:
    """delta_w^C per worker (no group-wide max, unlike the bottleneck model)."""
    l0, g_end = info["layer_offset"], info["g_end"]
    spatial   = info["is_spatial"]
    active    = info["active"]
    frac      = info["node_frac"]
    H_in      = info["H_in"]

    def _node_weight(nid: int) -> float:
        return max(node_weight_map.get(nid, IOT_REF_WEIGHT), 1e-6)

    def _effective_speedup(w: float) -> float:
        return 1.0 + DEVICE_SCALING_DIMINISHING_RETURN * max(w - 1.0, 0.0)

    def _bg_slowdown(nid: int) -> float:
        rho = node_load_map.get(nid, 0.0)
        rho = max(0.0, min(rho, 1.0 - 1e-6))
        return min(1.0 / (1.0 - rho), COMPUTE_SLOWDOWN_MAX)

    delta: Dict[int, float] = {}
    if spatial and len(active) > 1:
        overlap = group_fov_overlap(l0, g_end)
        for nid, fr in frac.items():
            w_j      = _node_weight(nid)
            speedup  = _effective_speedup(w_j)
            slowdown = _bg_slowdown(nid)
            fov_frac = min(fr + overlap / H_in, 1.0)
            delta[nid] = sum(
                LAYER_IOT_FWD_MS[li] * fov_frac * IOT_REF_WEIGHT / speedup * slowdown
                for li in range(l0, g_end)
            )
    else:
        nid = active[0] if active else hub
        w_j      = _node_weight(nid)
        speedup  = _effective_speedup(w_j)
        slowdown = _bg_slowdown(nid)
        cms = sum(
            LAYER_IOT_FWD_MS[li] * IOT_REF_WEIGHT / speedup * slowdown
            for li in range(l0, g_end)
        )
        if not spatial:
            cms /= DEVICE_PACE_RATE
        delta[nid] = cms

    return delta


# ── Recursive Delta_{g,w} model ─────────────────────────────────────────────

def _measure_rl_plan_latency_ms_recursive(
    plan: Dict,
    link_profiles: Dict,
    node_profiles: Optional[List[Dict]],
) -> float:
    """Delta_{g,w} = delta_w^C + max_{u in W_{g-1,w}} { Delta_{g-1,u} + delta_{uw} }"""
    group_sizes: List[int]       = plan["group_sizes"]
    chunks:      List[int]       = plan["chunks"]
    hub:         int             = plan["hub_node"]
    assignment:  List[List[int]] = plan["assignment"]

    bw  = np.array(link_profiles["bandwidth"], dtype=float)   # MB/s
    lat = np.array(link_profiles["latency"],   dtype=float)   # ms

    node_weight_map: Dict[int, float] = {}
    node_load_map:   Dict[int, float] = {}
    if node_profiles:
        for entry in node_profiles:
            nid = int(entry["node_id"])
            node_weight_map[nid] = float(entry["compute_capacity"])
            node_load_map[nid]   = float(entry.get("current_load", 0.0))

    groups = _build_groups(group_sizes, chunks, hub, assignment, node_weight_map)

    delta_prev: Dict[int, float] = {hub: 0.0}   # Delta_{0,h} = 0
    prev_info: Optional[Dict] = None
    last_info: Optional[Dict] = None

    for info in groups:
        if info is None:
            continue

        compute_delta = _group_compute_delta(info, hub, node_weight_map, node_load_map)

        for nid in info["remote_workers"]:
            compute_delta[nid] = compute_delta.get(nid, 0.0) + OVERHEAD_REMOTE_NODE_SETUP
        for nid in info["chunk_owners"]:
            compute_delta[nid] = compute_delta.get(nid, 0.0) + OVERHEAD_PER_CHUNK

        delta_cur: Dict[int, float] = {}

        if prev_info is None:
            # W_{0,w} = {h}
            l0, g_end = info["layer_offset"], info["g_end"]
            for w in info["active"]:
                if w == hub:
                    edge = 0.0
                elif info["is_spatial"]:
                    if l0 == 0:
                        total_in_bytes = 3.0 * 224 * 224 * 4.0
                        W_in, C_in = 224, 3
                    else:
                        total_in_bytes = float(MODEL_CFG[l0 - 1][4]) * 4.0
                        W_in = LAYER_SPATIAL_OUT[l0 - 1]
                        C_in = MODEL_CFG[l0][1]
                    overlap = group_fov_overlap(l0, g_end)
                    overlap_bytes = 2 * overlap * W_in * C_in * 4.0
                    D_hw = info["node_frac"][w] * total_in_bytes + overlap_bytes
                    edge = (_transfer_ms(D_hw, hub, w, bw, lat) / DEVICE_PACE_RATE
                            + OVERHEAD_PER_DATA_TRANSFER)
                else:
                    # non-spatial first group: single node, full input tensor
                    total_in_bytes = 3.0 * 224 * 224 * 4.0
                    edge = (_transfer_ms(total_in_bytes, hub, w, bw, lat) / DEVICE_PACE_RATE
                            + OVERHEAD_PER_DATA_TRANSFER)
                delta_cur[w] = compute_delta.get(w, 0.0) + delta_prev[hub] + edge

        elif prev_info["is_spatial"] and info["is_spatial"]:
            N_out      = LAYER_SPATIAL_OUT[prev_info["g_end"] - 1]
            C_out      = MODEL_CFG[prev_info["g_end"] - 1][2]
            overlap_g1 = group_fov_overlap(info["layer_offset"], info["g_end"])
            edges = _pairwise_row_transfers(
                prev_info["active"], prev_info["node_frac"],
                info["active"],      info["node_frac"],
                N_out, overlap_g1, C_out, bw, lat,
            )
            for w in info["active"]:
                candidates = []
                if w in prev_info["active"]:
                    candidates.append(delta_prev[w])   # same-node continuation, zero transfer
                for u in prev_info["active"]:
                    t = edges.get((u, w))
                    if t is not None:
                        candidates.append(delta_prev[u] + t / DEVICE_PACE_RATE
                                           + OVERHEAD_PER_DATA_TRANSFER)
                pred_term = max(candidates) if candidates else max(delta_prev.values())
                delta_cur[w] = compute_delta.get(w, 0.0) + pred_term

        elif prev_info["is_spatial"] and not info["is_spatial"]:
            # spatial -> FC: full activation, no overlap; every worker of the
            # previous group is a predecessor (hub included, if it co-computed)
            dest      = info["active"][0]
            act_elems = float(MODEL_CFG[prev_info["g_end"] - 1][4])
            candidates = []
            if dest in prev_info["active"]:
                candidates.append(delta_prev[dest])
            for u in prev_info["active"]:
                if u == dest:
                    continue
                t = _transfer_ms(prev_info["node_frac"][u] * act_elems * 4.0, u, dest, bw, lat)
                candidates.append(delta_prev[u] + t / DEVICE_PACE_RATE + OVERHEAD_PER_DATA_TRANSFER)
            pred_term = max(candidates) if candidates else max(delta_prev.values())
            delta_cur[dest] = compute_delta.get(dest, 0.0) + pred_term

        else:  # FC -> FC
            fc_node = prev_info["active"][0]
            for w in info["active"]:
                if w == fc_node:
                    edge = 0.0
                else:
                    act_elems = float(MODEL_CFG[prev_info["g_end"] - 1][4])
                    edge = (_transfer_ms(act_elems * 4.0, fc_node, w, bw, lat) / DEVICE_PACE_RATE
                            + OVERHEAD_PER_DATA_TRANSFER)
                delta_cur[w] = compute_delta.get(w, 0.0) + delta_prev[fc_node] + edge

        delta_prev = delta_cur
        prev_info  = info
        last_info  = info

    if last_info is None:
        return 0.0

    # W_{G+1,h} = W_G: hub has no compute, only waits for stage G to send back.
    g_end     = last_info["g_end"]
    act_elems = float(MODEL_CFG[g_end - 1][4])

    if last_info["is_spatial"]:
        candidates = []
        if hub in last_info["active"]:
            candidates.append(delta_prev[hub])
        for u in last_info["remote_workers"]:
            t = _transfer_ms(last_info["node_frac"][u] * act_elems * 4.0, u, hub, bw, lat)
            candidates.append(delta_prev[u] + t / DEVICE_PACE_RATE + OVERHEAD_PER_DATA_TRANSFER)
        return float(max(candidates)) if candidates else float(max(delta_prev.values()))

    fc_node = last_info["active"][0]
    if fc_node == hub:
        return float(delta_prev[fc_node])
    t = _transfer_ms(act_elems * 4.0, fc_node, hub, bw, lat)
    return float(delta_prev[fc_node] + t / DEVICE_PACE_RATE + OVERHEAD_PER_DATA_TRANSFER)


# ── Main public function ───────────────────────────────────────────────────

def measure_rl_plan_latency_ms(
    plan: Dict,
    link_profiles: Dict,
    device,
    node_profiles: Optional[List[Dict]] = None,
    warmup: int = 3,    # unused – kept for API compatibility
    repeats: int = 10,  # unused – kept for API compatibility
) -> float:
    """LUT-based end-to-end latency (ms) for the RL split plan."""
    return _measure_rl_plan_latency_ms_recursive(plan, link_profiles, node_profiles)
