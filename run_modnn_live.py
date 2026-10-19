"""run_modnn_live.py – LUT-based MODNN latency estimator; hub splits all 18 spatial layers across neighbours, FC layers run on hub alone."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from config import (
    DEVICE_PACE_RATE,
    IOT_REF_WEIGHT,
    LAYER_IOT_FWD_MS,
    LAYER_SPATIAL_OUT,
    MODEL_CFG,
)

# ── Project paths ──────────────────────────────────────────────────────────────────
_GREMS_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _GREMS_DIR / "third_party" / "disnet"   # vendored DiSNet sources

# Kernel size per VGG-16 layer: 3=conv3×3, 2=maxpool2×2, 0=FC
_KERNER_SIZE = [3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 2, 2, 2, 0, 0, 0]

logger = logging.getLogger("modnn_live")
logger.setLevel(logging.WARNING)
logger.propagate = False

_CONFLICT_NAMES = [
    "utils", "configurations", "network", "inference",
    "field_vgg", "field_DiSNet", "models", "models.model_vgg16",
]

_MODNN_MODS: Optional[Dict] = None


def _import_modnn() -> Dict:
    """Import DiSNet modules via sys.modules sandbox; cached after first call."""
    global _MODNN_MODS
    if _MODNN_MODS is not None:
        return _MODNN_MODS

    saved = {n: sys.modules.pop(n) for n in _CONFLICT_NAMES if n in sys.modules}
    added = []
    for p in (str(_PROJECT_ROOT / "main"), str(_PROJECT_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p); added.append(p)

    try:
        import configurations as _cfg
        import utils as _utils
        from network import read_graph
        from inference import get_partiton_info

        _MODNN_MODS = {
            "cfg":               _cfg,
            "read_graph":        read_graph,
            "get_partiton_info": get_partiton_info,
            "find_split_ratio":  _utils.find_split_ratio,
        }
    finally:
        for name in list(sys.modules):
            if (name in _CONFLICT_NAMES or name.startswith("field_")) \
                    and name not in saved:
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        for p in added:
            try: sys.path.remove(p)
            except ValueError: pass

    return _MODNN_MODS


# ── LUT-based timing ─────────────────────────────────────────────────────

def _lut_opt_modnn(
    partition_input, trans_rate_mbitps: float,
    comp_rate_weights: List[float], dpr: float,
) -> float:
    """LUT replacement for opt_modnn(): RECV at every layer, all 18 layers split.
    FC layers 18-20 run on hub alone. Returns infer_time_s (already /DPR)."""
    n_workers = len(comp_rate_weights)
    t_CLs = 0.0
    t_FLs = 0.0

    for i in range(21):
        if i < 18:   # spatial layers: all split equally
            H_full_in = 224 if i == 0 else LAYER_SPATIAL_OUT[i - 1]
            C_in      = MODEL_CFG[i][1]
            C_out     = MODEL_CFG[i][2]
            W_out     = LAYER_SPATIAL_OUT[i]
            pi        = partition_input[i]

            t_sub: List[float] = []
            for j in range(len(pi[0])):
                if pi[1][j] == 0:
                    break
                H_sub = pi[0][j][1] - pi[0][j][0] + 1

                t_rec = (32.0 * C_in * H_sub * H_full_in
                         / (1024.0 * 1024.0 * trans_rate_mbitps))

                w_j      = max(comp_rate_weights[j] if j < n_workers else 1.0, 1e-6)
                fraction = H_sub / max(H_full_in, 1)
                t_cmp    = (LAYER_IOT_FWD_MS[i] / 1000.0
                            * fraction * IOT_REF_WEIGHT / w_j)
                t_cmp_prop = t_cmp * dpr

                # SEND: trim FoV overlap rows
                if _KERNER_SIZE[i] == 3:
                    H_out_j = H_sub
                    rows    = H_out_j - 2 if j not in [0, 8] else H_out_j - 1
                    t_send  = (32.0 * C_out * rows * W_out
                               / (1024.0 * 1024.0 * trans_rate_mbitps))
                elif _KERNER_SIZE[i] == 2:
                    H_out_j = H_sub // 2
                    t_send  = (32.0 * C_out * H_out_j * W_out
                               / (1024.0 * 1024.0 * trans_rate_mbitps))
                else:
                    t_send = 0.0

                t_sub.append(t_rec + t_cmp_prop + t_send)

            if t_sub:
                t_CLs += max(t_sub)

        else:   # FC layers (18, 19, 20) — hub runs alone
            w    = max(comp_rate_weights[0] if comp_rate_weights else 1.0, 1e-6)
            t_fl = LAYER_IOT_FWD_MS[i] / 1000.0 * IOT_REF_WEIGHT / w
            t_FLs += t_fl

    return (t_CLs + t_FLs) / dpr


# max_par[0]: initial worker cap; max_par[3]: cap for layers>12 (14×14/7×7 maps).
_MAX_PAR_TABLE = {
    2:  [2, 2, 2, 2],  3:  [3, 3, 2, 2],  4:  [3, 3, 3, 3],
    5:  [4, 3, 3, 3],  6:  [4, 4, 3, 3],  7:  [5, 4, 3, 3],
    8:  [5, 4, 4, 3],  9:  [5, 5, 4, 3],  10: [6, 5, 4, 3],
    11: [6, 6, 4, 3],  12: [7, 7, 4, 3],
}


def _max_par_for(n_nodes: int, cfg) -> List[int]:
    if n_nodes in _MAX_PAR_TABLE:
        return list(_MAX_PAR_TABLE[n_nodes])
    return list(cfg.max_par_partitions)


# ── Core runner ────────────────────────────────────────────────────────────

def _run_modnn(G, input_node: int, max_bw: float,
               max_par: List[int], num_runs: int = 1) -> Dict:
    """Run MODNN on G: select workers → build partition table → LUT timing."""
    dm  = _import_modnn()
    dpr = DEVICE_PACE_RATE

    workers = [[nb, G.nodes[nb]["weight"], G[input_node][nb]["weight"]]
               for nb in G.neighbors(input_node)]
    workers.append([input_node, G.nodes[input_node]["weight"], 0])
    if len(workers) > max_par[0]:
        workers = workers[:max_par[0]]
    comp_rate, device_ids, throughput = dm["find_split_ratio"](workers)
    num_workers = len(workers)
    trans_rate  = min(throughput, max_bw)
    logger.info("MODNN %d workers (nodes=%s)  trans_rate=%.2f Mbit/s",
                num_workers, device_ids, trans_rate)

    num_devices_modnn = num_workers
    partition_input = []
    for layer_idx in range(18):
        if layer_idx > 12 and num_devices_modnn > max_par[3]:
            num_devices_modnn = max_par[3]
        partition_input.append(dm["get_partiton_info"](layer_idx, layer_idx, num_devices_modnn))

    infer_time_s = _lut_opt_modnn(partition_input, trans_rate, comp_rate, dpr)
    return {
        "latency_s":       infer_time_s,
        "latency_ms":      infer_time_s * 1000.0,
        "num_devices":     num_workers,
        "device_nodes":    device_ids,
        "trans_rate":      trans_rate,
        "all_latencies_s": [infer_time_s] * num_runs,
    }


# ── Public API ─────────────────────────────────────────────────────────────

def run_modnn_on_network(network_id: int = 0, max_bandwidth: float = 15.0,
                          num_runs: int = 1, input_node: int = 8) -> Dict:
    """Run MODNN on one of the four fixed benchmark networks (ID 0–3)."""
    dm = _import_modnn()
    old_cwd = os.getcwd(); os.chdir(str(_PROJECT_ROOT))
    try:    G = dm["read_graph"](network_id)
    finally: os.chdir(old_cwd)
    n = G.number_of_nodes()
    result = _run_modnn(G, input_node, max_bandwidth, _max_par_for(n, dm["cfg"]), num_runs)
    result.update({"network_id": network_id, "num_nodes": n})
    logger.info("Network %d: %.4f s (%.1f ms)", network_id, result["latency_s"], result["latency_ms"])
    return result


def run_modnn_on_graph(G, max_bandwidth: float = 15.0, num_runs: int = 1,
                        input_node: int = 0, device: str = "cpu") -> Dict:
    """Run MODNN on a pre-loaded NetworkX graph; node weight=compute rate, edge weight=bw Mbit/s."""
    dm = _import_modnn()
    n  = G.number_of_nodes()
    result = _run_modnn(G, input_node, max_bandwidth, _max_par_for(n, dm["cfg"]), num_runs)
    result["num_nodes"] = n
    logger.info("Variable graph (%d nodes): %.4f s (%.1f ms)", n, result["latency_s"], result["latency_ms"])
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run MODNN live inference")
    ap.add_argument("--network",   type=int,   required=True)
    ap.add_argument("--bandwidth", type=float, default=15.0)
    ap.add_argument("--runs",      type=int,   default=1)
    args = ap.parse_args()
    r = run_modnn_on_network(args.network, args.bandwidth, args.runs)
    print(f"Network {args.network}: {r['latency_s']:.4f} s "
          f"({r['latency_ms']:.1f} ms)  nodes={r['device_nodes']}")
