"""profiling.py – Layer profiling utilities and graph-to-profile converter."""

import time
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import (
    MODEL_CFG,
    NUM_LAYERS,
    LAYER_PROFILE_PATH,
    LAYER_FWD_MS,
)
from utils import save_json, load_json, setup_logging

logger = setup_logging("profiling")

# ═══════════════════════════════════════════════════
# Project paths
# ═══════════════════════════════════════════════════
_GREMS_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════
# 1.  Layer profiling (compute time, memory, FLOPs)
# ═══════════════════════════════════════════════════

def _build_layer(cfg_tuple: tuple) -> nn.Module:
    """Instantiate a PyTorch layer from a DiSNet-style config tuple."""
    ltype = cfg_tuple[0]
    if ltype == "C":
        _, in_ch, out_ch, k, _, _ = cfg_tuple
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
    elif ltype == "M":
        _, _, _, k, _, _ = cfg_tuple
        return nn.MaxPool2d(kernel_size=k, stride=k)
    elif ltype == "D":
        _, size_in, size_out, _, _, _ = cfg_tuple
        return nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.ReLU(),
        )
    else:
        raise ValueError(f"Unknown layer type: {ltype}")


def _dummy_input_for_layer(cfg_tuple: tuple, batch_size: int = 1) -> torch.Tensor:
    """Create a dummy input tensor matching the expected shape."""
    ltype = cfg_tuple[0]
    if ltype == "C":
        in_ch = cfg_tuple[1]
        out_ch = cfg_tuple[2]
        act_size = cfg_tuple[4]
        spatial = int(round((act_size / out_ch) ** 0.5))
        return torch.randn(batch_size, in_ch, spatial, spatial)
    elif ltype == "M":
        in_ch = cfg_tuple[1]
        k = cfg_tuple[3]
        out_ch = cfg_tuple[2]
        act_size = cfg_tuple[4]
        out_spatial = int(round((act_size / out_ch) ** 0.5))
        in_spatial = out_spatial * k
        return torch.randn(batch_size, in_ch, in_spatial, in_spatial)
    elif ltype == "D":
        size_in = cfg_tuple[1]
        return torch.randn(batch_size, size_in)
    else:
        raise ValueError(f"Unknown layer type: {ltype}")


def profile_layer_compute(
    cfg_tuple: tuple,
    device: torch.device = torch.device("cpu"),
    warmup: int = 5,
    repeats: int = 20,
) -> Dict[str, float]:
    """Time a single layer's forward pass and measure peak memory."""
    layer = _build_layer(cfg_tuple).to(device).eval()
    x = _dummy_input_for_layer(cfg_tuple).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = layer(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    peak_mem = 0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device)

    # Parameter size (weights + biases) in bytes (float32)
    param_bytes = sum(p.numel() * 4 for p in layer.parameters())

    return {
        "forward_time_ms": float(np.median(times)),
        "peak_memory_bytes": int(peak_mem),
        "param_size_bytes": int(param_bytes),
        "flops": int(cfg_tuple[-1]),
        "activation_size": int(cfg_tuple[4]),
    }


def profile_all_layers(
    device: torch.device = torch.device("cpu"),
    warmup: int = 5,
    repeats: int = 20,
) -> List[Dict[str, Any]]:
    """Profile every layer defined in ``MODEL_CFG``."""
    results = []
    for idx, cfg in enumerate(MODEL_CFG):
        logger.info("Profiling layer %d / %d  (%s) …", idx + 1, NUM_LAYERS, cfg[0])
        info = profile_layer_compute(cfg, device=device, warmup=warmup, repeats=repeats)
        info["layer_index"] = idx
        info["layer_type"] = cfg[0]
        results.append(info)
    return results


# ═══════════════════════════════════════════════════
# 2.  Save / Load helpers
# ═══════════════════════════════════════════════════

def build_layer_profiles() -> List[Dict]:
    """Build layer profiles from MODEL_CFG + LAYER_FWD_MS (no JSON file needed)."""
    profiles = []
    for idx, cfg in enumerate(MODEL_CFG):
        profiles.append({
            "layer_index":       idx,
            "layer_type":        cfg[0],            # "C", "M", or "D"
            "activation_size":   int(cfg[4]),       # output elements
            "flops":             int(cfg[5]),       # multiply-accumulates
            "forward_time_ms":   LAYER_FWD_MS[idx], # CPU timing (ms)
            "peak_memory_bytes": 0,                 # not used by reward/encoder
            "param_size_bytes":  0,                 # not used by reward/encoder
        })
    return profiles


def save_layer_profiles(profiles: List[Dict], path: Path = LAYER_PROFILE_PATH) -> None:
    save_json(profiles, path)


def load_layer_profiles(path: Path = LAYER_PROFILE_PATH) -> List[Dict]:
    """Return layer profiles built from config; any JSON at *path* is ignored."""
    return build_layer_profiles()


# ═══════════════════════════════════════════════════
# 5.  DiSNet graph import
# ═══════════════════════════════════════════════════

def generate_profiles_from_disnet_graph(
    graph_path: str,
    max_bandwidth_cap: Optional[float] = None,
    comm_pace_factor: float = 1.0,
) -> Tuple[List[Dict], Dict]:
    """Load a pickled NetworkX graph → (node_profiles, link_profiles).

    BW is converted Mbit/s → MB/s and optionally capped; unconnected pairs get bw=0.
    """
    with open(graph_path, "rb") as f:
        G = pickle.loads(f.read())

    num_nodes = G.number_of_nodes()
    node_ids = sorted(G.nodes())

    # ── Node profiles ──
    node_profiles = []
    for nid in node_ids:
        node_profiles.append({
            "node_id": nid,
            "compute_capacity": float(G.nodes[nid]["weight"]),
            "available_memory_mb": 1024.0,
            "current_load": 0.0,
        })

    # ── Link profiles ──
    # bw = 0 for non-existing links so _transfer_ms routes via Dijkstra multi-hop.
    bandwidth = np.zeros((num_nodes, num_nodes))
    latency_mat = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        bandwidth[i][i] = float("inf")
        latency_mat[i][i] = 0.0

    for u, v, data in G.edges(data=True):
        bw_mbit = float(data["weight"])
        if max_bandwidth_cap is not None:
            bw_mbit = min(bw_mbit, max_bandwidth_cap)
        bw_mbps = bw_mbit / 8.0 * comm_pace_factor

        bandwidth[u][v] = bw_mbps
        bandwidth[v][u] = bw_mbps
        latency_mat[u][v] = 0.5
        latency_mat[v][u] = 0.5

    link_profiles = {
        "bandwidth": bandwidth.tolist(),
        "latency": latency_mat.tolist(),
    }

    logger.info(
        "Loaded DiSNet graph: %d nodes, %d edges from %s",
        num_nodes, G.number_of_edges(), graph_path,
    )

    return node_profiles, link_profiles


# ──────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile VGG-16 layers and save to disk")
    parser.add_argument("--device", type=str, default=None,
                        help="Profiling device (default: cuda if available, else cpu)")
    args = parser.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling VGG-16 layers on {dev} …")
    dev_t = torch.device(dev)
    layer_profiles = profile_all_layers(device=dev_t)
    save_layer_profiles(layer_profiles)

    total_fwd = sum(lp["forward_time_ms"] for lp in layer_profiles)
    print(f"  Saved {len(layer_profiles)} layer profiles to {LAYER_PROFILE_PATH}")
    print(f"  Total forward time: {total_fwd:.2f} ms")
    for lp in layer_profiles:
        print(f"    Layer {lp['layer_index']:2d} ({lp['layer_type']}): "
              f"fwd={lp['forward_time_ms']:.3f}ms  "
              f"params={lp['param_size_bytes']/1e6:.2f}MB  "
              f"act={lp['activation_size']*4/1e6:.2f}MB")
