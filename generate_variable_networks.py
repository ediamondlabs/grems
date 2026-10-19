# -*- coding: utf-8 -*-
"""generate_variable_networks.py – Random connected IoT graphs for RL training/evaluation.
Topology models: ba (Barabási-Albert, default), ws (Watts-Strogatz), rgg (Random Geometric Graph), er (Erdős-Rényi).
Output: variable_networks/varnet_<n>nodes[_<tag>]_<idx>.txt (pickle)."""

import argparse
import math
import pickle
import random
from pathlib import Path

import networkx as nx

# ──────────────────────────────────────────────────────────────────
# Output directory  (GReMS/variable_networks/)
# ──────────────────────────────────────────────────────────────────
NETWORKS_DIR = Path(__file__).resolve().parent / "variable_networks"
NETWORKS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# Network generation parameters  (follow DiSNet conventions)
# ──────────────────────────────────────────────────────────────────
# Edge BW log-uniform over BW_RANGE Mbit/s; equal probability per decade (IoT congested → near-LAN).
BW_RANGE = (5, 200)

# Compute weight ranges; BA hub (highest-degree) gets HUB range, others WORKER range.
COMP_RANGE = (1, 10)
HUB_COMP_RANGE    = (5, 10)  # gateway / edge-server: higher compute capacity
WORKER_COMP_RANGE = (1, 6)   # IoT worker nodes: resource-constrained

# Topology sizes: train on 3–10 nodes, held-out test on 20 nodes.
DEFAULT_SIZES = [3, 4, 5, 6, 7, 8, 9, 10, 20]
DEFAULT_PER_SIZE = 50         # 50 distinct random topologies per size

# ── Barabási-Albert: m = preferential attachment count (scale-free hub distribution). ──
BA_M = 2
EDGE_DENSITY = 0.35  # superseded by BA_M; kept for backward compatibility

# ── Watts-Strogatz: k = neighbours before rewiring; p=0.3 canonical small-world regime. ──
WS_K_MAX = 4       # capped at min(WS_K_MAX, n-1); forced even for ring symmetry
WS_P     = 0.3

# ── RGG: radius = factor × sqrt(log(n)/(π·n)); factor>1 ensures connectivity with high prob. ──
RGG_RADIUS_FACTOR = 1.6
RGG_MAX_RETRIES   = 30   # max attempts before bridging disconnected components

# ── Erdős-Rényi: p = p_factor·log(n)/n; p_factor≥2 ensures connectivity with high prob. ──
ER_P_FACTOR    = 2.5
ER_P_MIN       = 0.35     # floor for small n where log(n)/n is too small
ER_MAX_RETRIES = 30

# All supported topology models
SUPPORTED_TOPOLOGY_MODELS = ("ba", "ws", "rgg", "er")


# ──────────────────────────────────────────────────────────────────
# Per-model graph constructors  (topology only, no node/edge features)
# ──────────────────────────────────────────────────────────────────

def _make_ba_graph(n, rng, ba_m=None):
    """Barabási-Albert scale-free graph, m=min(ba_m, n-1)."""
    m = min(ba_m if ba_m is not None else BA_M, n - 1)
    seed = rng.randint(0, 2**31)
    G_ba = nx.barabasi_albert_graph(n, m, seed=seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(G_ba.edges())
    assert nx.is_connected(G), "BA graph unexpectedly disconnected"
    return G


def _make_ws_graph(n, rng):
    """Watts-Strogatz graph; k rounded down to even ≥2 so the ring lattice is symmetric."""
    k_raw = min(WS_K_MAX, n - 1)
    # Round down to even, keep at least 2
    k = max(2, k_raw - (k_raw % 2))
    # Clamp in case rounding pushed k above n-1
    k = min(k, n - 1)
    seed = rng.randint(0, 2**31)
    G_ws = nx.watts_strogatz_graph(n, k, WS_P, seed=seed)
    # WS with k≥2 is always connected (ring lattice base is connected)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(G_ws.edges())
    return G


def _make_rgg_graph(n, rng):
    """RGG in [0,1]²; radius increased on each retry until connected, then bridges components."""
    # Theoretical connectivity threshold: sqrt(log(n) / (pi * n))
    base_radius = math.sqrt(math.log(max(n, 2)) / (math.pi * max(n, 2)))
    radius = RGG_RADIUS_FACTOR * base_radius

    seed = rng.randint(0, 2**31)
    G_raw = None
    for _ in range(RGG_MAX_RETRIES):
        G_raw = nx.random_geometric_graph(n, radius, seed=seed)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(G_raw.edges())
        if nx.is_connected(G):
            return G
        radius *= 1.1
        seed = rng.randint(0, 2**31)

    # Fallback: bridge disconnected components using stored node positions.
    pos = nx.get_node_attributes(G_raw, "pos")
    components = list(nx.connected_components(G))
    while len(components) > 1:
        best_dist, best_u, best_v = float("inf"), None, None
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                for u in components[i]:
                    for v in components[j]:
                        d = math.dist(pos[u], pos[v]) if pos else 1.0
                        if d < best_dist:
                            best_dist, best_u, best_v = d, u, v
        G.add_edge(best_u, best_v)
        components = list(nx.connected_components(G))
    return G


def _make_er_graph(n, rng):
    """Erdős-Rényi graph; p = max(ER_P_MIN, ER_P_FACTOR·log(n)/n); bridges components if still disconnected."""
    p = max(ER_P_MIN, ER_P_FACTOR * math.log(max(n, 2)) / max(n, 2))
    p = min(p, 1.0)

    seed = rng.randint(0, 2**31)
    G = None
    for _ in range(ER_MAX_RETRIES):
        G_raw = nx.erdos_renyi_graph(n, p, seed=seed)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(G_raw.edges())
        if nx.is_connected(G):
            return G
        seed = rng.randint(0, 2**31)

    # Fallback: bridge components
    components = list(nx.connected_components(G))
    for i in range(1, len(components)):
        u = next(iter(components[i - 1]))
        v = next(iter(components[i]))
        G.add_edge(u, v)
    return G


# ──────────────────────────────────────────────────────────────────
# Core graph generator  (topology + IoT node/edge features)
# ──────────────────────────────────────────────────────────────────

def generate_random_connected_graph(
    n: int,
    bw_range: tuple = BW_RANGE,
    comp_range: tuple = COMP_RANGE,
    edge_density: float = EDGE_DENSITY,  # unused; kept for API compatibility
    rng: random.Random = None,
    topology_model: str = "ba",
    ba_m: int = None,
) -> nx.Graph:
    """Generate a random IoT graph (n nodes) with compute/BW features using the given topology model."""
    if rng is None:
        rng = random

    if topology_model not in SUPPORTED_TOPOLOGY_MODELS:
        raise ValueError(
            f"Unknown topology_model '{topology_model}'. "
            f"Choose from {SUPPORTED_TOPOLOGY_MODELS}."
        )

    # ── Build topology ──
    if topology_model == "ba":
        G = _make_ba_graph(n, rng, ba_m=ba_m)
    elif topology_model == "ws":
        G = _make_ws_graph(n, rng)
    elif topology_model == "rgg":
        G = _make_rgg_graph(n, rng)
    else:  # "er"
        G = _make_er_graph(n, rng)

    # ── Node compute: BA hub (highest-degree) gets HUB_COMP_RANGE; WS/RGG/ER use flat COMP_RANGE. ──
    comp_lo, comp_hi = COMP_RANGE
    if topology_model == "ba":
        degrees = dict(G.degree())
        hub_node = max(range(n), key=lambda i: (degrees[i], i))
        hub_lo, hub_hi = HUB_COMP_RANGE
        wkr_lo, wkr_hi = WORKER_COMP_RANGE
        for i in range(n):
            if i == hub_node:
                G.nodes[i]["weight"] = rng.randint(hub_lo, hub_hi)
            else:
                G.nodes[i]["weight"] = rng.randint(wkr_lo, wkr_hi)
    else:
        for i in range(n):
            G.nodes[i]["weight"] = rng.randint(comp_lo, comp_hi)

    # ── Log-uniform BW: equal probability mass per decade ──
    log_lo = math.log(bw_range[0])
    log_hi = math.log(bw_range[1])
    for u, v in G.edges():
        bw = int(math.exp(rng.uniform(log_lo, log_hi)))
        G[u][v]["weight"] = bw

    assert nx.is_connected(G), f"{topology_model} graph is disconnected after feature assignment"
    return G


# ──────────────────────────────────────────────────────────────────
# Save / Load helpers (pickle, same format as DiSNet network.py)
# ──────────────────────────────────────────────────────────────────

def save_variable_network(G: nx.Graph, name: str, directory: Path = NETWORKS_DIR) -> Path:
    """Save a NetworkX graph as a pickle to *directory/name*.txt."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.txt"
    with open(path, "wb") as f:
        f.write(pickle.dumps(G))
    return path


def load_variable_network(name: str, directory: Path = NETWORKS_DIR) -> nx.Graph:
    """Load a previously saved variable network by name."""
    path = directory / f"{name}.txt"
    with open(path, "rb") as f:
        return pickle.loads(f.read())


def get_variable_network_paths(
    directory: Path = NETWORKS_DIR,
    sizes: list = None,
    split: str = "train",
    tag: str = None,
) -> list:
    """Return sorted Path list from *directory* filtered by sizes, split, and tag."""
    paths = sorted(directory.glob("*.txt"))
    result = []
    for p in paths:
        # Filename formats:
        #   no tag : varnet_<n>nodes_<idx>          (parts len 3)
        #   tagged : varnet_<n>nodes_<tag>_<idx>    (parts len 4+)
        try:
            parts = p.stem.split("_")
            n = int(parts[1].replace("nodes", ""))
        except (IndexError, ValueError):
            continue
        # Tag filter
        if tag is None:
            pass   # no filtering
        elif tag == "":
            # training files: 3rd segment must be numeric (no tag word)
            if len(parts) < 3 or not parts[2].isdigit():
                continue
        else:
            # tagged files: 3rd segment must equal the requested tag
            if len(parts) < 4 or parts[2] != tag:
                continue
        if sizes is not None and n not in sizes:
            continue
        result.append((n, p))
    result.sort(key=lambda t: (t[0], t[1].stem))
    return [p for _, p in result]


# ──────────────────────────────────────────────────────────────────
# Network summary
# ──────────────────────────────────────────────────────────────────

def describe_network(G: nx.Graph, name: str = "") -> str:
    bws = [d["weight"] for _, _, d in G.edges(data=True)]
    comps = [G.nodes[i]["weight"] for i in G.nodes()]
    avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes()
    return (
        f"{name}  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}  "
        f"avg_deg={avg_deg:.2f}  "
        f"bw=[{min(bws)},{max(bws)}] Mbit/s  "
        f"comp=[{min(comps)},{max(comps)}]"
    )


# ──────────────────────────────────────────────────────────────────
# Topology visualisation
# ──────────────────────────────────────────────────────────────────

def plot_variable_networks(
    directory: Path = NETWORKS_DIR,
    out_dir: Path = None,
    tag: str = None,
) -> None:
    """Draw each network as a PNG (one per node count); tag filters files same as get_variable_network_paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available – skipping topology plots.")
        return

    if out_dir is None:
        out_dir = directory / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Group files by node count, filtered by tag ──
    all_paths = sorted(directory.glob("*.txt"))
    by_size: dict = {}
    for p in all_paths:
        try:
            parts = p.stem.split("_")
            n = int(parts[1].replace("nodes", ""))
        except (IndexError, ValueError):
            continue
        # Tag filter (mirrors get_variable_network_paths logic)
        if tag is None:
            pass
        elif tag == "":
            if len(parts) < 3 or not parts[2].isdigit():
                continue
        else:
            if len(parts) < 4 or parts[2] != tag:
                continue
        by_size.setdefault(n, []).append(p)

    if not by_size:
        print(f"No networks found in {directory}")
        return

    norm = mcolors.Normalize(vmin=COMP_RANGE[0], vmax=COMP_RANGE[1])
    cmap = __import__("matplotlib").colormaps["RdYlBu_r"]

    for node_count, paths in sorted(by_size.items()):
        n_nets = len(paths)
        fig, axes = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(
            1, n_nets, figsize=(5 * n_nets, 5), constrained_layout=True
        )
        if n_nets == 1:
            axes = [axes]

        is_test = node_count >= 20
        split_label = "UNSEEN TEST" if is_test else "TRAIN"
        split_color = "#B71C1C" if is_test else "#1B5E20"
        fig.suptitle(
            f"Variable Networks – {node_count} nodes   [{split_label}]",
            fontsize=14, fontweight="bold", color=split_color,
        )

        for ax, path in zip(axes, paths):
            with open(path, "rb") as f:
                G = pickle.loads(f.read())

            pos = nx.spring_layout(G, seed=42)

            comp_vals = [G.nodes[i]["weight"] for i in G.nodes()]
            node_colors = [cmap(norm(c)) for c in comp_vals]

            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                   node_size=600, alpha=0.9)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555555",
                                   width=1.5, alpha=0.7)
            node_labels = {i: f"{i}\n({G.nodes[i]['weight']})" for i in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                                    font_size=8, font_weight="bold")
            edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                         font_size=7, alpha=0.85)
            ax.set_title(path.stem, fontsize=9)
            ax.axis("off")

        out_path = out_dir / f"topology_{node_count}nodes.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"  [plot] saved → {out_path}")
        __import__("matplotlib.pyplot", fromlist=["close"]).close(fig)

    print(f"\nTopology plots saved to {out_dir}/")


# ──────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────

def generate_all(
    sizes: list = None,
    per_size: int = DEFAULT_PER_SIZE,
    seed: int = 42,
    directory: Path = NETWORKS_DIR,
    verbose: bool = True,
    plot: bool = True,
    tag: str = "",
    topology_model: str = "ba",
    ba_m: int = None,
) -> dict:
    """Generate per_size graphs per node count as pickles; non-BA models auto-tag filename with model name."""
    if sizes is None:
        sizes = DEFAULT_SIZES

    if topology_model not in SUPPORTED_TOPOLOGY_MODELS:
        raise ValueError(
            f"Unknown topology_model '{topology_model}'. "
            f"Choose from {SUPPORTED_TOPOLOGY_MODELS}."
        )

    # Auto-tag: non-BA models get their name as tag so BA files stay
    # backward-compatible (no tag in filename).
    effective_tag = tag if tag != "" else (topology_model if topology_model != "ba" else "")

    master_rng = random.Random(seed)
    generated = {}
    tag_part = f"_{effective_tag}" if effective_tag else ""

    for n in sizes:
        for idx in range(per_size):
            name = f"varnet_{n}nodes{tag_part}_{idx}"
            child_seed = master_rng.randint(0, 2**31)
            child_rng = random.Random(child_seed)
            G = generate_random_connected_graph(
                n, rng=child_rng, topology_model=topology_model, ba_m=ba_m
            )
            path = save_variable_network(G, name, directory)
            generated[name] = G
            if verbose:
                print(f"  [saved] {describe_network(G, name)} → {path}")

    print(f"\nGenerated {len(generated)} networks ({topology_model}) in {directory}")

    if plot:
        plot_out_dir = directory / ("plots" if topology_model == "ba" else "plots_" + topology_model)
        plot_variable_networks(directory=directory, out_dir=plot_out_dir, tag=effective_tag)

    return generated


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate variable-topology networks for RL training/evaluation"
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
        help="Node counts to generate (default: 3 4 5 6 7 8 9 10 20)",
    )
    parser.add_argument(
        "--per-size", type=int, default=DEFAULT_PER_SIZE,
        help="Number of distinct random topologies per size (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--topology-model", type=str, default="ba",
        choices=SUPPORTED_TOPOLOGY_MODELS,
        help=(
            "Topology generative model: "
            "ba=Barabási-Albert (default), "
            "ws=Watts-Strogatz small-world, "
            "rgg=Random Geometric Graph, "
            "er=Erdős-Rényi random"
        ),
    )
    parser.add_argument(
        "--ba-m", type=int, default=None,
        help=(
            "Barabási-Albert attachment count m (BA topology only). "
            "m=1: sparse/tree-like; m=2: default scale-free; m=3-4: denser. "
            "Use different values to test connectivity-level generalisation. "
            "Defaults to the global BA_M constant (currently 2)."
        ),
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help=(
            "Explicit filename tag (e.g. 'test' → varnet_10nodes_test_0). "
            "Defaults to the topology model name for non-BA models."
        ),
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(NETWORKS_DIR),
        help="Output directory for network files",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating topology PNG visualisations",
    )
    args = parser.parse_args()

    _ba_m_display = args.ba_m if args.ba_m is not None else BA_M
    print(f"Generating variable-topology networks …")
    print(f"  Topology model : {args.topology_model}"
          + (f"  (BA_M={_ba_m_display})" if args.topology_model == "ba" else ""))
    print(f"  Sizes          : {args.sizes}")
    print(f"  Per size       : {args.per_size}")
    print(f"  Seed           : {args.seed}")
    print(f"  Output dir     : {args.out_dir}")
    print(f"  Plots          : {'disabled' if args.no_plots else 'enabled'}\n")

    nets = generate_all(
        sizes=args.sizes,
        per_size=args.per_size,
        seed=args.seed,
        tag=args.tag,
        topology_model=args.topology_model,
        directory=Path(args.out_dir),
        plot=not args.no_plots,
        ba_m=args.ba_m,
    )
