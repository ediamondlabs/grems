"""
Defines the discrete action space for the GReMS split-inference agent and decodes
raw action vectors into structured split plans (group sizes, chunk counts, hub node,
node assignments). Hub distributes the initial input to group-0 workers; inter-group
transfers use W2W multi-hop routing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config import NUM_LAYERS, NUM_NODES, MAX_GROUPS, MAX_CHUNKS, N_VERT_WEIGHTS


# ─────────────────────────────────────────────────
# Action specification
# ─────────────────────────────────────────────────

@dataclass
class ActionSpec:
    """Describes the multi-dimensional discrete action produced by the agent."""
    num_layers: int = NUM_LAYERS  # 21
    max_groups: int = MAX_GROUPS  # 6
    min_groups: int = 1
    max_chunks: int = MAX_CHUNKS  # 20
    num_nodes:  int = NUM_NODES   # max nodes in topology

    # ── Derived dimensions ──────────────────────

    @property
    def num_groups_dim(self) -> int:
        """One decision: how many vertical groups to activate."""
        return 1

    @property
    def num_groups_nvec(self) -> int:
        """Categories for num_groups: {min_groups, ..., max_groups}."""
        return self.max_groups - self.min_groups + 1

    @property
    def vertical_dim(self) -> int:
        """Number of weight decisions for vertical split (always max_groups slots)."""
        return self.max_groups

    @property
    def horizontal_dim(self) -> int:
        """Number of chunk-count decisions for horizontal split (always max_groups slots)."""
        return self.max_groups

    @property
    def hub_dim(self) -> int:
        """Number of categorical decisions for hub selection (always 1)."""
        return 1

    @property
    def assignment_dim(self) -> int:
        """Number of categorical decisions for node assignment."""
        return self.max_groups * self.max_chunks

    @property
    def total_dim(self) -> int:
        """Total number of independent categorical decisions."""
        return (
            self.num_groups_dim
            + self.vertical_dim
            + self.horizontal_dim
            + self.hub_dim
            + self.assignment_dim
        )

    # ── Per-head option counts ──────────────────

    @property
    def vertical_nvec(self) -> int:
        """Number of categories per vertical-split weight decision."""
        return N_VERT_WEIGHTS

    @property
    def horizontal_nvec(self) -> int:
        """Number of categories per horizontal-split chunk decision."""
        return self.max_chunks

    @property
    def hub_nvec(self) -> int:
        """Number of categories for hub selection."""
        return self.num_nodes

    @property
    def assignment_nvec(self) -> int:
        """Number of categories per assignment decision."""
        return self.num_nodes

    @property
    def multi_discrete_nvec(self) -> List[int]:
        """nvec array for gymnasium.spaces.MultiDiscrete; order matches decode_action."""
        num_g  = [self.num_groups_nvec]    * self.num_groups_dim
        vert   = [self.vertical_nvec]      * self.vertical_dim
        horiz  = [self.horizontal_nvec]    * self.horizontal_dim
        hub    = [self.hub_nvec]           * self.hub_dim
        assign = [self.assignment_nvec]    * self.assignment_dim
        return num_g + vert + horiz + hub + assign


# Global default spec
DEFAULT_SPEC = ActionSpec()


# ─────────────────────────────────────────────────
# Action decoding
# ─────────────────────────────────────────────────

def decode_vertical_split(
    raw: np.ndarray,
    num_layers: int = NUM_LAYERS,
    num_active_groups: Optional[int] = None,
) -> List[int]:
    """
    Convert discrete weight indices into layer counts per group.

    ``num_active_groups`` controls how many groups are activated; inactive
    groups receive 0 layers.  Every active group gets at least 1 layer;
    rounding errors are corrected so sizes always sum to ``num_layers``.
    """
    n = len(raw)
    active = num_active_groups if num_active_groups is not None else n
    active = int(np.clip(active, 1, min(n, num_layers)))

    weights = raw.astype(float)
    # Inactive groups contribute nothing
    weights[active:] = 0.0
    # Guarantee every active group has non-zero weight so it receives ≥ 1 layer
    for i in range(active):
        weights[i] = max(weights[i], 1.0)

    total = weights.sum()
    if total == 0:
        total = 1.0

    props = weights / total
    sizes = np.round(props * num_layers).astype(int)
    sizes = np.maximum(sizes, 0)
    sizes[active:] = 0  # hard-zero inactive groups after rounding

    # Enforce: every active group must have ≥ 1 layer
    for i in range(active):
        if sizes[i] == 0:
            sizes[i] = 1

    # Fix rounding error so layers sum to num_layers
    diff = num_layers - sizes.sum()
    if diff > 0:
        idx = int(np.argmax(sizes[:active]))
        sizes[idx] += diff
    elif diff < 0:
        sorted_idx = np.argsort(-sizes[:active])
        for si in sorted_idx:
            if diff == 0:
                break
            removable = sizes[si] - 1  # keep ≥ 1 per active group
            remove = min(removable, -diff)
            sizes[si] -= remove
            diff += remove

    return sizes.tolist()


def decode_horizontal_split(
    raw: np.ndarray,
    max_chunks: int = MAX_CHUNKS,
    min_chunks: int = 1,
    num_active_groups: Optional[int] = None,
) -> List[int]:
    """
    Convert discrete indices into chunk counts per group.

    Active groups: each ∈ [min_chunks, max_chunks].
    Inactive groups (index ≥ num_active_groups): 0.
    """
    chunks = (raw + 1).astype(int)
    chunks = np.clip(chunks, min_chunks, max_chunks)
    if num_active_groups is not None:
        chunks[num_active_groups:] = 0
    return chunks.tolist()


def decode_hub_selection(
    raw: np.ndarray,
    num_nodes: int = NUM_NODES,
) -> int:
    """Return the hub node index from a single raw action value."""
    return int(raw[0]) % num_nodes


def decode_node_assignment(
    raw: np.ndarray,
    chunks_per_group: List[int],
    num_nodes: int = NUM_NODES,
    max_groups: int = MAX_GROUPS,
    max_chunks: int = MAX_CHUNKS,
    hub_node: Optional[int] = None,
    adjacency: Optional[np.ndarray] = None,
) -> List[List[int]]:
    """
    Convert discrete node indices into assignment[group][chunk] = node_id.

    Enforces three hard constraints:
      1. Hub is first worker in group-0 for initial input distribution;
         inter-group transfers are worker-to-worker. Single-worker groups exempt.
      2. All chunks in a group use distinct nodes — chunk count is capped at
         num_nodes (or, for group 0, at the hub's direct-neighbor count
         including itself) since there is no valid way to seat more distinct
         workers than that many exist.
      3. Group-0 workers must be connected to the hub.
    """
    raw = raw.reshape(max_groups, max_chunks)

    # Pre-compute set of nodes connected to the hub (for remapping)
    connected: Optional[List[int]] = None
    if hub_node is not None and adjacency is not None:
        connected = sorted(
            i for i in range(num_nodes)
            if adjacency[hub_node][i] > 0.5 or i == hub_node
        )

    assignment: List[List[int]] = []
    for g in range(max_groups):
        row: List[int] = []
        used: set = set()
        n_c = chunks_per_group[g] if g < len(chunks_per_group) else 0
        if g == 0 and connected is not None:
            n_c = min(n_c, len(connected))
        else:
            n_c = min(n_c, num_nodes)
        if n_c == 0:
            assignment.append(row)
            continue

        # Hub is first worker in multi-worker groups; group-0 workers receive the
        # initial input from hub, subsequent groups receive activations via W2W multi-hop.
        if hub_node is not None and n_c > 1:
            row.append(hub_node)
            used.add(hub_node)

        # Group 0 workers must be hub-connected; the n_c cap above guarantees
        # enough distinct candidates remain for every slot below.
        candidates = connected if g == 0 and connected is not None else None

        for c in range(len(row), n_c):
            node_id = int(raw[g, c]) % num_nodes

            if candidates is not None:
                if node_id not in candidates or node_id in used:
                    node_id = next(cn for cn in candidates if cn not in used)
            elif node_id in used:
                while node_id in used:
                    node_id = (node_id + 1) % num_nodes

            used.add(node_id)
            row.append(node_id)
        assignment.append(row)
    return assignment


def decode_action(
    raw_action: np.ndarray,
    spec: ActionSpec = DEFAULT_SPEC,
    adjacency: Optional[np.ndarray] = None,
) -> Dict:
    """Decode a flat discrete action vector into a structured split plan dict."""
    # ── Head 0: num_active_groups ────────────────────────────────────────
    # raw_action[0] ∈ {0, …, num_groups_nvec-1}  maps to  {min_groups, …, max_groups}
    num_active_groups = int(raw_action[0]) + spec.min_groups
    num_active_groups = int(np.clip(num_active_groups, spec.min_groups, spec.max_groups))
    # Never use more groups than there are nodes — extra groups add overhead with no benefit
    num_active_groups = min(num_active_groups, spec.num_nodes)

    offset  = spec.num_groups_dim          # 1
    v_end   = offset + spec.vertical_dim
    h_end   = v_end  + spec.horizontal_dim
    hub_end = h_end  + spec.hub_dim

    raw_v   = raw_action[offset:v_end]
    raw_h   = raw_action[v_end:h_end]
    raw_hub = raw_action[h_end:hub_end]
    raw_a   = raw_action[hub_end:]

    group_sizes = decode_vertical_split(
        raw_v, spec.num_layers, num_active_groups=num_active_groups
    )
    chunks = decode_horizontal_split(
        raw_h, spec.max_chunks,
        min_chunks=1,
        num_active_groups=num_active_groups,
    )
    hub_node = decode_hub_selection(raw_hub, spec.num_nodes)
    assignment = decode_node_assignment(
        raw_a, chunks, spec.num_nodes, spec.max_groups, spec.max_chunks,
        hub_node=hub_node, adjacency=adjacency,
    )

    return {
        "group_sizes":        group_sizes,
        "chunks":             chunks,
        "hub_node":           hub_node,
        "assignment":         assignment,
        "num_active_groups":  num_active_groups,   # expose for reward / logging
    }


# ─────────────────────────────────────────────────
# Connectivity penalty (shaped reward)
# ─────────────────────────────────────────────────

def compute_connectivity_penalty(
    group_sizes: List[int],
    chunks: List[int],
    hub_node: int,
    assignment: List[List[int]],
    adjacency: np.ndarray,
) -> float:
    """Shaped-reward penalty: one point per group-0 worker with no direct link to the hub."""
    penalty = 0.0

    # Build ordered list of (group_idx, worker_list) for active groups only
    active: List[tuple] = []
    for g, gs in enumerate(group_sizes):
        if gs == 0 or g >= len(assignment) or not assignment[g]:
            continue
        n_c = chunks[g] if g < len(chunks) else 1
        workers = assignment[g][:min(n_c, len(assignment[g]))]
        active.append((g, workers))

    if not active:
        return 0.0

    # Every group-0 worker must be directly connected to hub.
    # Direct links are preferred for the initial input distribution;
    # multi-hop is the fallback in the latency model but incurs higher latency.
    _, workers0 = active[0]
    for w in workers0:
        if w != hub_node and adjacency[hub_node][w] < 0.5:
            penalty += 1.0

    return penalty


def compute_weak_link_penalty(
    group_sizes: List[int],
    chunks: List[int],
    hub_node: int,
    assignment: List[List[int]],
    bandwidth: np.ndarray,
    threshold: float,
) -> float:
    """Soft penalty for a group-0 worker whose direct link to the hub is
    connected but weak (bw>0, so compute_connectivity_penalty doesn't fire).
    Complements that hard check with a continuous signal — 0 at/above
    threshold, scaling toward 1 as bandwidth approaches 0."""
    if not group_sizes or group_sizes[0] == 0 or not assignment or not assignment[0]:
        return 0.0
    n_c = chunks[0] if chunks else 1
    workers0 = assignment[0][:min(n_c, len(assignment[0]))]

    penalty = 0.0
    for w in workers0:
        if w == hub_node:
            continue
        bw = bandwidth[hub_node][w]
        if 0.0 < bw < threshold:
            penalty += (threshold - bw) / threshold

    return penalty


# ─────────────────────────────────────────────────
# Random baseline
# ─────────────────────────────────────────────────

def sample_random_action(spec: ActionSpec = DEFAULT_SPEC) -> np.ndarray:
    """Sample a uniformly random discrete action vector."""
    nvec = spec.multi_discrete_nvec
    return np.array([np.random.randint(n) for n in nvec], dtype=np.int64)


# ─────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    spec = ActionSpec(num_nodes=10, max_chunks=10)
    print(f"total_dim    = {spec.total_dim}")
    print(f"num_groups   : 1 × {spec.num_groups_nvec} options  "
          f"({spec.min_groups}..{spec.max_groups} groups)")
    print(f"vertical     : {spec.vertical_dim} × {spec.vertical_nvec}")
    print(f"horizontal   : {spec.horizontal_dim} × {spec.horizontal_nvec}")
    print(f"hub          : {spec.hub_dim} × {spec.hub_nvec}")
    print(f"assignment   : {spec.assignment_dim} × {spec.assignment_nvec}")

    act = sample_random_action(spec)
    decoded = decode_action(act, spec)
    print(f"\nnum_active_groups = {decoded['num_active_groups']}")
    print(f"group_sizes = {decoded['group_sizes']}  (sum={sum(decoded['group_sizes'])})")
    print(f"chunks = {decoded['chunks']}")
    print(f"hub_node = {decoded['hub_node']}")
    print(f"assignment = {decoded['assignment']}")

    # Connectivity penalty with a dummy adjacency
    adj = np.zeros((10, 10))
    np.fill_diagonal(adj, 1.0)
    adj[0, 1] = adj[1, 0] = 1.0
    adj[1, 2] = adj[2, 1] = 1.0
    pen = compute_connectivity_penalty(
        decoded["group_sizes"], decoded["chunks"],
        decoded["hub_node"], decoded["assignment"], adj,
    )
    print(f"connectivity penalty = {pen}")
