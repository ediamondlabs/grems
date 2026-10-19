"""
run_disnet_live.py – LUT-based latency estimator mirroring DiSNet's opt_DiSNet().

4 vertical groups [[0,4],[4,9],[9,14],[14,18]]; hub via determine_opt_neighbours(),
horizontal split via find_split_ratio(). DiSNet root shares module names with GReMS —
sys.modules is temporarily swapped on import to avoid conflicts.
"""

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
_GREMS_DIR    = Path(__file__).resolve().parent   # …/DiSNet/GReMS/
_PROJECT_ROOT = _GREMS_DIR / "third_party" / "disnet"   # vendored DiSNet sources

# Kernel size per VGG-16 layer: 3=conv3×3, 2=maxpool2×2, 0=FC
_KERNER_SIZE = [3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 2, 2, 2, 0, 0, 0]

logger = logging.getLogger("disnet_live")
logger.setLevel(logging.WARNING)
logger.propagate = False

# ── Module names that clash between DiSNet root and GReMS ─────────────────
_CONFLICT_NAMES = [
    "utils", "configurations", "network", "inference",
    "field_vgg", "field_DiSNet", "models", "models.model_vgg16",
]

_DISNET_MODS: Optional[Dict] = None   # cached after first import


def _import_disnet() -> Dict:
    """Import DiSNet modules without polluting GReMS's namespace (cached)."""
    global _DISNET_MODS
    if _DISNET_MODS is not None:
        return _DISNET_MODS

    saved = {n: sys.modules.pop(n) for n in _CONFLICT_NAMES if n in sys.modules}
    added = []
    for p in (str(_PROJECT_ROOT / "main"), str(_PROJECT_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p); added.append(p)
    try:
        import configurations as _cfg
        import utils as _utils
        from network import read_graph, select_path, determine_opt_neighbours
        from inference import get_partiton_info_DiSNet
        _DISNET_MODS = {
            "cfg": _cfg, "read_graph": read_graph,
            "select_path": select_path,
            "determine_opt_neighbours": determine_opt_neighbours,
            "get_partiton_info_DiSNet": get_partiton_info_DiSNet,
            "find_split_ratio": _utils.find_split_ratio,
            "select_subset":    _utils.select_subset,
            "cap_trans_rate":   _utils.cap_trans_rate,
        }
    finally:
        for name in list(sys.modules):
            if (name in _CONFLICT_NAMES or name.startswith("field_")) and name not in saved:
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        for p in added:
            try: sys.path.remove(p)
            except ValueError: pass

    return _DISNET_MODS


# Safe max_par_partitions per node count (from configurations.py comments).
_MAX_PAR_TABLE = {
    2:  [2, 2, 2, 2],  3:  [3, 3, 2, 2],  4:  [3, 3, 3, 3],
    5:  [4, 3, 3, 3],  6:  [4, 4, 3, 3],  7:  [5, 4, 3, 3],
    8:  [5, 4, 4, 3],  9:  [5, 5, 4, 3],  10: [6, 5, 4, 3],
    11: [6, 6, 4, 3],  12: [7, 7, 4, 3],
}


def _max_par_for(n_nodes: int, cfg) -> List[int]:
    """Return safe max_par_partitions for a graph with n_nodes nodes."""
    if n_nodes in _MAX_PAR_TABLE:
        return list(_MAX_PAR_TABLE[n_nodes])
    return list(cfg.max_par_partitions)


def _build_plan(G, max_par: List[int], max_bw: float,
                src: int, dst: int, energy: float) -> Dict:
    """Select path + per-group hub/split-ratio. Mirrors run_DiSNet.py loop."""
    dm = _import_disnet()

    selected_path  = dm["select_path"](G, src, dst, energy)
    cur_pos, split_ratio, devices = 0, [], []
    trans_rate_fwd, par_trans_rate, execution_path = [], [], []

    for max_n in max_par:
        hub = dm["determine_opt_neighbours"](G, selected_path, max_n, cur_pos, energy)
        execution_path.append(hub)
        cur_pos = selected_path.index(hub)

        neighbours = [[nb, G.nodes[nb]["weight"], G[hub][nb]["weight"]]
                      for nb in G.neighbors(hub)]
        nxt = (selected_path[cur_pos - 1] if cur_pos >= len(selected_path) - 1
               else selected_path[cur_pos + 1])
        neighbours.append([hub, G.nodes[hub]["weight"], G[hub][nxt]["weight"]])
        trans_rate_fwd.append(G[hub][nxt]["weight"])

        src_nb = neighbours if len(neighbours) < max_n else dm["select_subset"](neighbours, max_n)
        h_split, dev, throughput = dm["find_split_ratio"](src_nb)
        split_ratio.append(h_split); par_trans_rate.append(throughput); devices.append(dev)
        logger.debug("plan grp %d: hub=%d nxt=%d workers=%d split=%s bw=%.2f",
                     len(execution_path)-1, hub, nxt, len(dev),
                     [round(r, 3) for r in h_split], throughput)

    trans_rate_fwd = dm["cap_trans_rate"](max_bw, trans_rate_fwd)
    par_trans_rate = dm["cap_trans_rate"](max_bw, par_trans_rate)

    return {"selected_path": selected_path, "execution_path": execution_path,
            "split_ratio": split_ratio, "devices": devices,
            "trans_rate_fwd": trans_rate_fwd, "par_trans_rate": par_trans_rate}


def _create_partition_input(split_ratio, dm) -> list:
    """Fixed create_partition_input(): original has bitwise-& bug mapping layers 4-17
    all to split_ratio[3]. Correct mapping: 0-3→[0], 4-8→[1], 9-13→[2], 14-17→[3]."""
    partition_input = []
    for l in range(18):
        if l < 4:
            point = 0
        elif l < 9:
            point = 1
        elif l < 14:
            point = 2
        else:
            point = 3
        partition = dm["get_partiton_info_DiSNet"](l, l, split_ratio[point])
        partition_input.append(partition)
    return partition_input


# ── LUT-based timing ─────────────────────────────────────────────────────

def _lut_disnet_group(
    lr, partition_input_grp, trans_rate_mbitps: float,
    device_node_weights: List[float], hub_weight: float, dpr: float,
) -> float:
    """LUT replacement for opt_DiSNet() using LAYER_IOT_FWD_MS; returns group time already /DPR."""
    layers_start = int(lr[0])
    layers_full  = 21 if int(lr[1]) == 18 else int(lr[1])
    layers_iter  = layers_full - layers_start

    t_CLs = 0.0   # spatial (×DPR inside, /DPR at end → net = raw)
    t_FLs = 0.0   # FC hub-only (/DPR at end)

    logger.debug("lut_group [l%d..l%d]  hub_w=%.2f  par_bw=%.2f Mbit/s  "
                 "n_device_weights=%d",
                 layers_start, layers_full - 1, hub_weight,
                 trans_rate_mbitps, len(device_node_weights))

    for i in range(layers_iter):
        p = layers_start + i

        if p < layers_full - 3:   # spatial: horizontal split
            if i >= len(partition_input_grp):
                continue
            pi        = partition_input_grp[i]
            H_full_in = 224 if p == 0 else LAYER_SPATIAL_OUT[p - 1]
            C_in      = MODEL_CFG[p][1]
            C_out     = MODEL_CFG[p][2]
            W_out     = LAYER_SPATIAL_OUT[p]

            t_sub: List[float] = []
            for j in range(len(pi[0])):
                if pi[1][j] == 0:
                    break
                H_sub = pi[0][j][1] - pi[0][j][0] + 1

                t_rec = 0.0
                if i == 0:  # RECV only on first layer of group
                    t_rec = (32.0 * C_in * H_sub * H_full_in
                             / (1024.0 * 1024.0 * trans_rate_mbitps))

                w_j        = max(device_node_weights[j] if j < len(device_node_weights) else 1.0, 1e-6)
                fraction   = H_sub / max(H_full_in, 1)
                t_cmp      = LAYER_IOT_FWD_MS[p] / 1000.0 * fraction * IOT_REF_WEIGHT / w_j
                t_cmp_prop = t_cmp * dpr  # ×DPR mirrors opt_DiSNet structure

                # SEND: trim FoV overlap rows
                if _KERNER_SIZE[p] == 3:
                    H_out_j = H_sub
                    rows    = H_out_j - 2 if j not in [0, 8] else H_out_j - 1
                    t_send  = (32.0 * C_out * rows * W_out
                               / (1024.0 * 1024.0 * trans_rate_mbitps))
                elif _KERNER_SIZE[p] == 2:
                    H_out_j = H_sub // 2
                    t_send  = (32.0 * C_out * H_out_j * W_out
                               / (1024.0 * 1024.0 * trans_rate_mbitps))
                else:
                    t_send = 0.0

                logger.debug("  l%02d w%d H=%d frac=%.3f rec=%.4f cmp=%.4f snd=%.4f eff=%.4f s",
                             p, j, H_sub, fraction, t_rec/dpr, t_cmp, t_send/dpr,
                             (t_rec + t_cmp_prop + t_send) / dpr)
                t_sub.append(t_rec + t_cmp_prop + t_send)

            if t_sub:
                logger.debug("  l%02d BN worker=%d eff=%.4f s", p, int(np.argmax(t_sub)), max(t_sub)/dpr)
                t_CLs += max(t_sub)

        else:  # hub-only: FC layers 18-20 (and trailing layers in groups 0-2)
            w     = max(hub_weight, 1e-6)
            t_fl  = LAYER_IOT_FWD_MS[p] / 1000.0 * IOT_REF_WEIGHT / w
            t_FLs += t_fl

    group_s = (t_CLs + t_FLs) / dpr
    logger.debug("lut_group [l%d..l%d] spatial=%.4f hub=%.4f total=%.4f s",
                 layers_start, layers_full-1, t_CLs/dpr, t_FLs/dpr, group_s)
    return group_s


def _analytical_fwd_transfer_s(lr, trans_rate_mbitps: float) -> float:
    """Analytical replacement for trans_time_forward() using known shapes."""
    if int(lr[1]) >= 18:
        return 0.0
    last_l = int(lr[1]) - 1
    C      = MODEL_CFG[last_l][2]
    H      = LAYER_SPATIAL_OUT[last_l]
    return 32.0 * C * H * H / (1024.0 * 1024.0 * trans_rate_mbitps)


# ── Core runner ────────────────────────────────────────────────────────────

def _run_disnet(G, max_par: List[int], max_bw: float,
                src: int, dst: int, energy: float = 0.5,
                num_runs: int = 1) -> Dict:
    """Run DiSNet on G: build plan → LUT group timing → inter-group transfers."""
    dm, dpr = _import_disnet(), DEVICE_PACE_RATE
    lr_arr  = dm["cfg"].layer_range  # [[0,4],[4,9],[9,14],[14,18]]

    plan           = _build_plan(G, max_par, max_bw, src, dst, energy)
    exec_path      = plan["execution_path"]
    devices        = plan["devices"]
    trans_rate_fwd = plan["trans_rate_fwd"]
    par_trans_rate = plan["par_trans_rate"]
    # original create_partition_input() has bitwise-& bug; use corrected version above
    partition_input = _create_partition_input(plan["split_ratio"], dm)

    logger.info("DiSNet path %d→%d  hubs=%s  workers/grp=%s",
                src, dst, exec_path, [len(d) for d in devices])
    logger.debug("  split_ratios (per grp): %s",
                 [[round(r, 3) for r in sr] for sr in plan["split_ratio"]])
    logger.debug("  par_trans_rate=%s Mbit/s  trans_rate_fwd=%s Mbit/s",
                 [round(r, 2) for r in par_trans_rate],
                 [round(r, 2) for r in trans_rate_fwd])

    grp_times_s: List[float] = []
    fwd_times_s: List[float] = []

    for j, lr in enumerate(lr_arr):
        # Node weights for this group's assigned workers
        device_weights_j = [float(G.nodes[d]["weight"]) for d in devices[j]]
        hub_weight_j     = float(G.nodes[exec_path[j]]["weight"])

        group_time_s = _lut_disnet_group(
            lr,
            partition_input[int(lr[0]):int(lr[1])],
            par_trans_rate[j],
            device_weights_j,
            hub_weight_j,
            dpr,
        )
        grp_times_s.append(group_time_s)
        logger.info("  grp %d [l%d-l%d] hub=n%d workers=%s bw=%.1f time=%.4f s",
                    j, int(lr[0]), int(lr[1])-1, exec_path[j],
                    [f"n{d}" for d in devices[j]], par_trans_rate[j], group_time_s)

        # Inter-group forward transfer.
        fwd_s = 0.0
        if j < len(lr_arr) - 1:
            same_hub = exec_path[j] == exec_path[j + 1]
            same_dev = devices[j]   == devices[j + 1]
            if same_hub and same_dev:   fwd_s = 0.0
            elif same_hub:              fwd_s = _analytical_fwd_transfer_s(lr, trans_rate_fwd[j] * 10)  # local link
            else:                       fwd_s = _analytical_fwd_transfer_s(lr, trans_rate_fwd[j])
            fwd_s /= dpr
        fwd_times_s.append(fwd_s)

    total_s = float(np.sum(grp_times_s) + np.sum(fwd_times_s))
    logger.info("grp=%s fwd=%s TOTAL=%.4f s (%.1f ms)",
                [f"{t:.4f}" for t in grp_times_s], [f"{t:.4f}" for t in fwd_times_s],
                total_s, total_s * 1000.0)
    return {
        "latency_s":             total_s,
        "latency_ms":            total_s * 1000.0,
        "selected_path":         plan["selected_path"],
        "execution_path":        exec_path,
        "num_devices_per_group": [len(d) for d in devices],
        "all_latencies_s":       [total_s] * num_runs,
    }


# ── Public API ─────────────────────────────────────────────────────────────

def run_disnet_on_network(network_id: int = 0, max_bandwidth: float = 15.0,
                          num_runs: int = 1, input_node: int = 8,
                          output_node: int = 4,
                          energy_sensitivity: float = 0.5) -> Dict:
    """Run DiSNet on one of the four fixed benchmark networks (ID 0–3)."""
    dm = _import_disnet()
    old_cwd = os.getcwd(); os.chdir(str(_PROJECT_ROOT))
    try:    G = dm["read_graph"](network_id)
    finally: os.chdir(old_cwd)
    n, max_par = G.number_of_nodes(), _max_par_for(G.number_of_nodes(), dm["cfg"])
    result = _run_disnet(G, max_par, max_bandwidth, input_node, output_node, energy_sensitivity, num_runs)
    result.update({"network_id": network_id, "num_nodes": n})
    logger.info("Network %d: %.4f s (%.1f ms)", network_id, result["latency_s"], result["latency_ms"])
    return result


def run_disnet_on_graph(G, max_bandwidth: float = 15.0, num_runs: int = 1,
                        input_node: int = None, output_node: int = None,
                        energy_sensitivity: float = 0.5,
                        device: str = "cpu") -> Dict:
    """Run DiSNet on a pre-loaded NetworkX graph (variable-topology).
    G.nodes[i]["weight"]=compute rate, G[u][v]["weight"]=link bw (Mbit/s)."""
    dm = _import_disnet()
    n  = G.number_of_nodes()
    if input_node is None:  input_node  = 0
    if output_node is None: output_node = n - 1
    if input_node == output_node and n > 1: output_node = (input_node + 1) % n
    max_par = _max_par_for(n, dm["cfg"])
    result  = _run_disnet(G, max_par, max_bandwidth, input_node, output_node, energy_sensitivity, num_runs)
    result["num_nodes"] = n
    logger.info("Variable graph (%d nodes): %.4f s (%.1f ms)", n, result["latency_s"], result["latency_ms"])
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run DiSNet live inference")
    ap.add_argument("--network",   type=int,   required=True)
    ap.add_argument("--bandwidth", type=float, default=15.0)
    ap.add_argument("--runs",      type=int,   default=1)
    args = ap.parse_args()
    r = run_disnet_on_network(args.network, args.bandwidth, args.runs)
    print(f"Network {args.network}: {r['latency_s']:.4f} s ({r['latency_ms']:.1f} ms)  "
          f"path={r['selected_path']}  dev/grp={r['num_devices_per_group']}")
