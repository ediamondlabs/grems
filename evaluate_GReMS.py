"""evaluate_GReMS.py – Dynamic multi-sequence evaluator for the GReMS agent.

Runs num_sequences independent trajectories per test network; at every step the
MMPP compute-load and G-E link-BW chains advance so all baselines (DiSNet, MoDNN,
EnSplit) see the same channel state as GReMS. Results aggregated by node size and
saved to JSON.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TimeoutError
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch

from config import (
    BW_CONNECTED_THRESHOLD,
    CHECKPOINT_DIR,
    COMPUTE_SLOWDOWN_MAX,
    DEVICE,
    DEVICE_SCALING_DIMINISHING_RETURN,
    DISNET_BW_CAP,
    EPISODE_LENGTH,
    MAX_NODES,
    SEED,
    VARIABLE_NETWORKS_DIR,
    VARIABLE_TEST_SIZES,
    VARIABLE_TRAIN_SIZES,
)
from environment import SplitEnv
from gnn_model import build_model
from generate_variable_networks import get_variable_network_paths
from profiling import generate_profiles_from_disnet_graph
from utils import set_seed, setup_logging

logger = setup_logging("evaluate_GReMS")

_BASELINE_TIMEOUT_S     = 30  # per-step timeout for small graphs (n≤10)
_OOD_BASELINE_TIMEOUT_S = 10  # tighter timeout for n=11-19; nx.all_simple_paths is exponential

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_with_timeout(fn, *args, timeout: int = _BASELINE_TIMEOUT_S, **kwargs):
    # Avoid context-manager form: its __exit__ calls shutdown(wait=True), blocking
    # until the thread finishes even after TimeoutError (catastrophic for nx.all_simple_paths).
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(fn, *args, **kwargs)
        return fut.result(timeout=timeout)
    finally:
        ex.shutdown(wait=False)


def _build_step_graph(env: SplitEnv) -> nx.Graph:
    """Build a NetworkX graph from env's current MMPP+G-E state for baseline solvers."""
    G = nx.Graph()
    for i, prof in enumerate(env.node_profiles):
        capacity = float(prof["compute_capacity"])
        load     = float(prof.get("current_load", 0.0))
        speedup  = 1.0 + DEVICE_SCALING_DIMINISHING_RETURN * max(capacity - 1.0, 0.0)
        rho      = max(0.0, min(load, 1.0 - 1e-6))
        slowdown = min(1.0 / (1.0 - rho), COMPUTE_SLOWDOWN_MAX)
        eff_w    = max(speedup / slowdown, 1e-3)
        G.add_node(i, weight=eff_w)

    bw_mb = np.array(env.link_profiles["bandwidth"])
    adj   = env._adjacency
    n     = env.num_nodes

    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 1.0:
                bw_mbit = float(bw_mb[u, v]) * 8.0  # MB/s → Mbit/s
                G.add_edge(u, v, weight=max(1, int(round(bw_mbit))))

    return G


def _float(x) -> Optional[float]:
    """Return Python float or None (safe for JSON serialisation)."""
    return float(x) if x is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Per-network evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one_network(
    ac: torch.nn.Module,
    env: SplitEnv,
    num_sequences: int = 10,
    num_steps: int = 50,
    warmup_steps: int = 0,
    frozen_warmup: int = 0,
    run_baselines: bool = True,
    device: str = DEVICE,
    baseline_timeout_s: int = _BASELINE_TIMEOUT_S,
    max_consecutive_baseline_failures: int = 999,
    ensplit_checkpoint: Optional[str] = None,
    run_disnet: bool = True,
    run_modnn:  bool = True,
) -> Dict:
    """Evaluate GReMS (+baselines) on one network across num_sequences dynamic trajectories."""
    _run_disnet = _run_modnn = _run_ensplit = None
    if run_baselines:
        if run_disnet:
            try:
                from run_disnet_live import run_disnet_on_graph as _rd
                _run_disnet = _rd
            except Exception as e:
                logger.warning("DiSNet import failed — skipping DiSNet baseline: %s", e)
        if run_modnn:
            try:
                from run_modnn_live import run_modnn_on_graph as _rm
                _run_modnn = _rm
            except Exception as e:
                logger.warning("MoDNN import failed — skipping MoDNN baseline: %s", e)
        try:
            from run_ensplit_live import run_ensplit_on_graph as _re
            _run_ensplit = _re
        except Exception as e:
            logger.warning("EnSplit import failed — skipping EnSplit baseline: %s", e)

    from dataclasses import replace as _dc_replace
    from action_space import decode_action as _decode_action
    from run_rl_live import measure_rl_plan_latency_ms as _measure_lat

    ac.eval()
    sequences: List[Dict] = []
    edges: Optional[List] = None

    for seq_idx in range(num_sequences):
        # Fresh start: reset to all-GOOD G-E state
        obs, _ = env.reset()

        # Snapshot frozen action at pristine state (after frozen_warmup advances)
        for _ in range(frozen_warmup):
            obs = env.advance_dynamics()
        obs_t_frozen = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            frozen_action_init, _, _ = ac.act(obs_t_frozen, deterministic=True)
        frozen_action: Optional[np.ndarray] = frozen_action_init.cpu().numpy()

        # Additional burn-in for adaptive GReMS starting state
        for _ in range(warmup_steps):
            obs = env.advance_dynamics()

        if edges is None:
            _n   = env.num_nodes
            _adj = env._adjacency
            edges = [[int(u), int(v)]
                     for u in range(_n) for v in range(u + 1, _n)
                     if _adj[u, v] == 1.0]

        _decode_spec  = _dc_replace(env.spec_action, num_nodes=env.num_nodes)

        # Per-sequence give-up state for baselines (reset each new sequence)
        disnet_fail_streak:  int = 0
        modnn_fail_streak:   int = 0
        ensplit_fail_streak: int = 0
        disnet_gave_up:      bool = False
        modnn_gave_up:       bool = False
        ensplit_gave_up:     bool = False

        step_records: List[Dict] = []

        for step_num in range(num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            # ── Snapshot current channel state ──
            node_profs = env.node_profiles
            mean_load  = float(np.mean([p["current_load"] for p in node_profs]))
            frac_high  = float(np.mean([
                1.0 if p["current_load"] > 0.5 else 0.0 for p in node_profs
            ]))
            bw_mb = np.array(env.link_profiles["bandwidth"])
            adj   = env._adjacency
            mask  = (adj == 1.0) & ~np.eye(env.num_nodes, dtype=bool)
            mean_eff_bw_mbit = float(np.mean(bw_mb[mask]) * 8.0) if mask.any() else 0.0

            # ── Run baselines on same current channel state ──
            disnet_lat_ms:  Optional[float] = None
            modnn_lat_ms:   Optional[float] = None
            ensplit_lat_ms: Optional[float] = None

            _try_baselines = (
                run_baselines
                and (_run_disnet is not None or _run_modnn is not None
                     or _run_ensplit is not None)
                and not (disnet_gave_up and modnn_gave_up and ensplit_gave_up)
            )
            if _try_baselines:
                G_step = _build_step_graph(env)
                src    = env._src_node
                dst    = env._dst_node

                if _run_disnet is not None and not disnet_gave_up:
                    try:
                        res = _run_with_timeout(
                            _run_disnet, G_step,
                            max_bandwidth=DISNET_BW_CAP,
                            num_runs=1,
                            input_node=src,
                            output_node=dst,
                            device="cpu",
                            timeout=baseline_timeout_s,
                        )
                        disnet_lat_ms = _float(res.get("latency_ms"))
                        disnet_fail_streak = 0
                    except (_TimeoutError, Exception) as e:
                        logger.debug("DiSNet step %d seq %d failed: %s", step_num, seq_idx, e)
                        disnet_fail_streak += 1
                        if disnet_fail_streak >= max_consecutive_baseline_failures:
                            disnet_gave_up = True
                            logger.debug("DiSNet gave up after %d consecutive failures (seq %d)",
                                         disnet_fail_streak, seq_idx)

                if _run_modnn is not None and not modnn_gave_up:
                    try:
                        res = _run_with_timeout(
                            _run_modnn, G_step,
                            max_bandwidth=DISNET_BW_CAP,
                            num_runs=1,
                            input_node=src,
                            device="cpu",
                            timeout=baseline_timeout_s,
                        )
                        modnn_lat_ms = _float(res.get("latency_ms"))
                        modnn_fail_streak = 0
                    except (_TimeoutError, Exception) as e:
                        logger.debug("MoDNN step %d seq %d failed: %s", step_num, seq_idx, e)
                        modnn_fail_streak += 1
                        if modnn_fail_streak >= max_consecutive_baseline_failures:
                            modnn_gave_up = True
                            logger.debug("MoDNN gave up after %d consecutive failures (seq %d)",
                                         modnn_fail_streak, seq_idx)

                if _run_ensplit is not None and not ensplit_gave_up:
                    try:
                        res = _run_with_timeout(
                            _run_ensplit, G_step,
                            max_bandwidth=DISNET_BW_CAP,
                            num_runs=1,
                            input_node=src,
                            output_node=dst,
                            device="cpu",
                            checkpoint_path=ensplit_checkpoint,
                            timeout=baseline_timeout_s,
                        )
                        ensplit_lat_ms = _float(res.get("latency_ms"))
                        ensplit_fail_streak = 0
                    except (_TimeoutError, Exception) as e:
                        logger.debug("EnSplit step %d seq %d failed: %s", step_num, seq_idx, e)
                        ensplit_fail_streak += 1
                        if ensplit_fail_streak >= max_consecutive_baseline_failures:
                            ensplit_gave_up = True
                            logger.debug("EnSplit gave up after %d consecutive failures (seq %d)",
                                         ensplit_fail_streak, seq_idx)

            # ── GReMS acts on current obs ──
            with torch.no_grad():
                action, _, _ = ac.act(obs_t, deterministic=True)
            action_np = action.cpu().numpy()

            # ── Decode split plan for recording ──
            decoded      = _decode_action(action_np, _decode_spec,
                                          adjacency=env._base_adjacency)
            hub_node     = int(decoded["hub_node"])
            # Collect all non-hub workers across active groups (hub is always index 0)
            worker_nodes = sorted({
                int(node)
                for g in range(decoded["num_active_groups"])
                if g < len(decoded["assignment"])
                for node in decoded["assignment"][g][1:]
            })

            # ── Frozen GReMS: pristine step-0 plan evaluated on current channel state ──
            frozen_lat_ms: Optional[float] = None
            if frozen_action is not None:
                try:
                    frozen_decoded = _decode_action(frozen_action, _decode_spec,
                                                    adjacency=env._base_adjacency)
                    frozen_lat_ms = _float(_measure_lat(
                        plan=frozen_decoded,
                        link_profiles=env.link_profiles,
                        node_profiles=env.node_profiles,
                        device="cpu",
                    ))
                except Exception:
                    pass

            _, _, _, _, info = env.step(action_np)

            connected = info.get("conn_penalty", 0.0) == 0.0

            step_records.append({
                "step":             step_num,
                "rl_lat_ms":        _float(info["latency_ms"]),
                "frozen_lat_ms":    frozen_lat_ms,
                "disnet_lat_ms":    disnet_lat_ms,
                "modnn_lat_ms":     modnn_lat_ms,
                "ensplit_lat_ms":   ensplit_lat_ms,
                "mean_load":        mean_load,
                "frac_high":        frac_high,
                "mean_eff_bw_mbit": mean_eff_bw_mbit,
                "node_loads":       [float(p["current_load"]) for p in node_profs],
                "link_bws_mbit":    [float(bw_mb[u, v]) * 8.0 for u, v in (edges or [])],
                "connected":        connected,
                "hub_node":         hub_node,
                "worker_nodes":     worker_nodes,
            })

            # ── Advance both Markov chains for the next step ──
            obs = env.advance_dynamics()

        # ── Aggregate over steps in this sequence ──
        rl_lats      = [s["rl_lat_ms"]      for s in step_records]
        d_lats       = [s["disnet_lat_ms"]  for s in step_records if s["disnet_lat_ms"]  is not None]
        m_lats       = [s["modnn_lat_ms"]   for s in step_records if s["modnn_lat_ms"]   is not None]
        e_lats       = [s["ensplit_lat_ms"] for s in step_records if s["ensplit_lat_ms"] is not None]
        conn_steps   = [s for s in step_records if s["connected"]]
        rl_conn_lats = [s["rl_lat_ms"] for s in conn_steps]
        conn_rate    = float(len(conn_steps)) / len(step_records) if step_records else 0.0

        sequences.append({
            "seq_idx":                seq_idx,
            "steps":                  step_records,
            "rl_mean_lat_ms":         _float(np.mean(rl_lats)),
            "rl_mean_lat_ms_conn":    _float(np.mean(rl_conn_lats)) if rl_conn_lats else None,
            "disnet_mean_lat_ms":     _float(np.mean(d_lats)) if d_lats else None,
            "modnn_mean_lat_ms":      _float(np.mean(m_lats)) if m_lats else None,
            "ensplit_mean_lat_ms":    _float(np.mean(e_lats)) if e_lats else None,
            "connectivity_rate":      conn_rate,
            "n_connected_steps":      len(conn_steps),
            "n_total_steps":          len(step_records),
        })

    # ── Aggregate over all sequences ──
    all_steps   = [s for seq in sequences for s in seq["steps"]]
    all_rl      = [s["rl_lat_ms"]      for s in all_steps]
    all_d       = [s["disnet_lat_ms"]  for s in all_steps if s["disnet_lat_ms"]  is not None]
    all_m       = [s["modnn_lat_ms"]   for s in all_steps if s["modnn_lat_ms"]   is not None]
    all_e       = [s["ensplit_lat_ms"] for s in all_steps if s["ensplit_lat_ms"] is not None]
    conn_steps  = [s for s in all_steps if s["connected"]]
    all_rl_conn = [s["rl_lat_ms"] for s in conn_steps]

    n_total = len(all_steps)
    n_conn  = len(conn_steps)
    d_total = len(all_d)
    m_total = len(all_m)
    e_total = len(all_e)

    rl_beat_d = sum(
        1 for s in all_steps
        if s["disnet_lat_ms"] is not None and s["rl_lat_ms"] < s["disnet_lat_ms"]
    )
    rl_beat_m = sum(
        1 for s in all_steps
        if s["modnn_lat_ms"] is not None and s["rl_lat_ms"] < s["modnn_lat_ms"]
    )
    rl_beat_e = sum(
        1 for s in all_steps
        if s["ensplit_lat_ms"] is not None and s["rl_lat_ms"] < s["ensplit_lat_ms"]
    )
    # Win rate conditioned on connected plans only (fair comparison)
    rl_beat_d_conn = sum(
        1 for s in conn_steps
        if s["disnet_lat_ms"] is not None and s["rl_lat_ms"] < s["disnet_lat_ms"]
    )
    rl_beat_m_conn = sum(
        1 for s in conn_steps
        if s["modnn_lat_ms"] is not None and s["rl_lat_ms"] < s["modnn_lat_ms"]
    )
    rl_beat_e_conn = sum(
        1 for s in conn_steps
        if s["ensplit_lat_ms"] is not None and s["rl_lat_ms"] < s["ensplit_lat_ms"]
    )
    d_conn_total = sum(1 for s in conn_steps if s["disnet_lat_ms"]  is not None)
    m_conn_total = sum(1 for s in conn_steps if s["modnn_lat_ms"]   is not None)
    e_conn_total = sum(1 for s in conn_steps if s["ensplit_lat_ms"] is not None)

    return {
        "sequences":                sequences,
        "edges":                    edges or [],
        # All-steps stats (includes disconnected penalty steps)
        "rl_mean_lat_ms":           _float(np.mean(all_rl)),
        "rl_std_lat_ms":            _float(np.std(all_rl)),
        # Connected-only stats (for paper)
        "rl_mean_lat_ms_conn":      _float(np.mean(all_rl_conn)) if all_rl_conn else None,
        "rl_std_lat_ms_conn":       _float(np.std(all_rl_conn))  if all_rl_conn else None,
        "connectivity_rate":        _float(n_conn / n_total) if n_total > 0 else None,
        "n_connected_steps":        n_conn,
        "n_total_steps":            n_total,
        # Baselines
        "disnet_mean_lat_ms":       _float(np.mean(all_d)) if all_d else None,
        "modnn_mean_lat_ms":        _float(np.mean(all_m)) if all_m else None,
        "ensplit_mean_lat_ms":      _float(np.mean(all_e)) if all_e else None,
        # Win rate (all steps)
        "rl_beat_disnet_pct":       _float(100.0 * rl_beat_d / d_total) if d_total > 0 else None,
        "rl_beat_modnn_pct":        _float(100.0 * rl_beat_m / m_total) if m_total > 0 else None,
        "rl_beat_ensplit_pct":      _float(100.0 * rl_beat_e / e_total) if e_total > 0 else None,
        # Win rate (connected steps only)
        "rl_beat_disnet_pct_conn":  _float(100.0 * rl_beat_d_conn / d_conn_total) if d_conn_total > 0 else None,
        "rl_beat_modnn_pct_conn":   _float(100.0 * rl_beat_m_conn / m_conn_total) if m_conn_total > 0 else None,
        "rl_beat_ensplit_pct_conn": _float(100.0 * rl_beat_e_conn / e_conn_total) if e_conn_total > 0 else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_by_size(network_results: Dict) -> Dict:
    """Group per-network results by node count; compute size-level means."""
    buckets: Dict[int, Dict[str, list]] = {}

    for net_name, rec in network_results.items():
        n = rec["num_nodes"]
        if n not in buckets:
            buckets[n] = {"rl": [], "rl_conn": [], "disnet": [], "modnn": [], "ensplit": [],
                          "rl_beat_d": [], "rl_beat_m": [], "rl_beat_e": [],
                          "rl_beat_d_conn": [], "rl_beat_m_conn": [], "rl_beat_e_conn": [],
                          "conn_rate": []}
        b = buckets[n]
        b["rl"].append(rec["rl_mean_lat_ms"])
        if rec["rl_mean_lat_ms_conn"] is not None:
            b["rl_conn"].append(rec["rl_mean_lat_ms_conn"])
        if rec["disnet_mean_lat_ms"] is not None:
            b["disnet"].append(rec["disnet_mean_lat_ms"])
        if rec["modnn_mean_lat_ms"] is not None:
            b["modnn"].append(rec["modnn_mean_lat_ms"])
        if rec["ensplit_mean_lat_ms"] is not None:
            b["ensplit"].append(rec["ensplit_mean_lat_ms"])
        if rec["rl_beat_disnet_pct"] is not None:
            b["rl_beat_d"].append(rec["rl_beat_disnet_pct"])
        if rec["rl_beat_modnn_pct"] is not None:
            b["rl_beat_m"].append(rec["rl_beat_modnn_pct"])
        if rec["rl_beat_ensplit_pct"] is not None:
            b["rl_beat_e"].append(rec["rl_beat_ensplit_pct"])
        if rec["rl_beat_disnet_pct_conn"] is not None:
            b["rl_beat_d_conn"].append(rec["rl_beat_disnet_pct_conn"])
        if rec["rl_beat_modnn_pct_conn"] is not None:
            b["rl_beat_m_conn"].append(rec["rl_beat_modnn_pct_conn"])
        if rec["rl_beat_ensplit_pct_conn"] is not None:
            b["rl_beat_e_conn"].append(rec["rl_beat_ensplit_pct_conn"])
        if rec["connectivity_rate"] is not None:
            b["conn_rate"].append(rec["connectivity_rate"])

    summary: Dict[int, Dict] = {}
    for n, b in sorted(buckets.items()):
        summary[n] = {
            "rl_mean":                 _float(np.mean(b["rl"])),
            "rl_std":                  _float(np.std(b["rl"])),
            "rl_mean_conn":            _float(np.mean(b["rl_conn"])) if b["rl_conn"] else None,
            "rl_std_conn":             _float(np.std(b["rl_conn"]))  if b["rl_conn"] else None,
            "connectivity_rate":       _float(np.mean(b["conn_rate"])) if b["conn_rate"] else None,
            "disnet_mean":             _float(np.mean(b["disnet"])) if b["disnet"] else None,
            "disnet_std":              _float(np.std(b["disnet"]))  if b["disnet"] else None,
            "modnn_mean":              _float(np.mean(b["modnn"]))  if b["modnn"]  else None,
            "modnn_std":               _float(np.std(b["modnn"]))   if b["modnn"]  else None,
            "ensplit_mean":            _float(np.mean(b["ensplit"])) if b["ensplit"] else None,
            "ensplit_std":             _float(np.std(b["ensplit"]))  if b["ensplit"] else None,
            "rl_beat_disnet":          _float(np.mean(b["rl_beat_d"])) if b["rl_beat_d"] else None,
            "rl_beat_modnn":           _float(np.mean(b["rl_beat_m"])) if b["rl_beat_m"] else None,
            "rl_beat_ensplit":         _float(np.mean(b["rl_beat_e"])) if b["rl_beat_e"] else None,
            "rl_beat_disnet_conn":     _float(np.mean(b["rl_beat_d_conn"])) if b["rl_beat_d_conn"] else None,
            "rl_beat_modnn_conn":      _float(np.mean(b["rl_beat_m_conn"])) if b["rl_beat_m_conn"] else None,
            "rl_beat_ensplit_conn":    _float(np.mean(b["rl_beat_e_conn"])) if b["rl_beat_e_conn"] else None,
            "n_networks":              len(b["rl"]),
        }
    return summary


def _step_curves(network_results: Dict) -> Dict:
    """Average RL/DiSNet/MoDNN latency at each trajectory step across all networks & sequences.
    Returns mean, std, and repetition count per step for CI shading."""
    max_steps = max(
        (s["step"] for rec in network_results.values()
         for seq in rec["sequences"] for s in seq["steps"]),
        default=0,
    ) + 1

    step_rl:      List[List[float]] = [[] for _ in range(max_steps)]
    step_disnet:  List[List[float]] = [[] for _ in range(max_steps)]
    step_modnn:   List[List[float]] = [[] for _ in range(max_steps)]
    step_ensplit: List[List[float]] = [[] for _ in range(max_steps)]

    for rec in network_results.values():
        for seq in rec["sequences"]:
            for s in seq["steps"]:
                t = s["step"]
                step_rl[t].append(s["rl_lat_ms"])
                if s["disnet_lat_ms"] is not None:
                    step_disnet[t].append(s["disnet_lat_ms"])
                if s["modnn_lat_ms"] is not None:
                    step_modnn[t].append(s["modnn_lat_ms"])
                if s.get("ensplit_lat_ms") is not None:
                    step_ensplit[t].append(s["ensplit_lat_ms"])

    return {
        "rl":           [_float(np.mean(v)) if v else None for v in step_rl],
        "rl_std":       [_float(np.std(v))  if v else None for v in step_rl],
        "disnet":       [_float(np.mean(v)) if v else None for v in step_disnet],
        "disnet_std":   [_float(np.std(v))  if v else None for v in step_disnet],
        "modnn":        [_float(np.mean(v)) if v else None for v in step_modnn],
        "modnn_std":    [_float(np.std(v))  if v else None for v in step_modnn],
        "ensplit":      [_float(np.mean(v)) if v else None for v in step_ensplit],
        "ensplit_std":  [_float(np.std(v))  if v else None for v in step_ensplit],
        "n_reps":       max((len(v) for v in step_rl if v), default=0),
    }



def main_grems(
    model_path: str,
    run_id:          str  = "K1",
    topology_tag:    str  = "rgg",
    num_sequences:   int  = 10,
    num_steps:       int  = 50,
    warmup_steps:    int  = 0,
    frozen_warmup:   int  = 0,
    seed:            int  = SEED,
    device:          str  = DEVICE,
    use_offline:     bool = False,
    eval_small:      bool = True,
    eval_ood_size:   bool = True,
    eval_large:      bool = True,
    run_baselines:      bool = True,
    run_disnet:         bool = True,
    run_modnn:          bool = True,
    ensplit_checkpoint: Optional[str] = None,
    networks_dir:       Optional[Path] = None,
    save_json:          Optional[str]  = None,
) -> Dict:
    """Load a GReMS checkpoint and run dynamic sequence evaluation across all test networks."""
    set_seed(seed)

    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return {}

    # ── Resolve network directory ─────────────────────────────────────────
    _dir_map = {
        "ba":  VARIABLE_NETWORKS_DIR,
        "rgg": VARIABLE_NETWORKS_DIR.parent / "variable_networks_rgg",
        "ws":  VARIABLE_NETWORKS_DIR.parent / "variable_networks_ws",
        "er":  VARIABLE_NETWORKS_DIR.parent / "variable_networks_er",
    }
    _file_tag   = "test"
    _default_dir = _dir_map.get(topology_tag,
                                VARIABLE_NETWORKS_DIR.parent / f"variable_networks_{topology_tag}")

    if networks_dir is None:
        networks_dir = _default_dir
    networks_dir = Path(networks_dir)
    if not networks_dir.exists():
        logger.error("Networks directory not found: %s", networks_dir)
        return {}

    # ── Load model ─────────────────────────────────────────────────────────
    ref_env  = SplitEnv(num_nodes=MAX_NODES, use_offline_profiles=False, seed=seed)
    obs_d    = ref_env.observation_space.shape[0]
    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = raw_ckpt["model"] if isinstance(raw_ckpt, dict) and "model" in raw_ckpt else raw_ckpt
    ac = build_model(obs_d, ref_env.spec_action)
    ac.load_state_dict(state_dict)
    ac = ac.to(device)
    ac.eval()

    ckpt_obs_dim = raw_ckpt.get("obs_dim") if isinstance(raw_ckpt, dict) else None
    logger.info(
        "Loaded GReMS-%s  obs_dim=%d  ckpt.obs_dim=%s  params=%d",
        run_id, obs_d, ckpt_obs_dim,
        sum(p.numel() for p in ac.parameters() if p.requires_grad),
    )

    # ── Collect network paths ──────────────────────────────────────────────
    small_paths: List[Path] = []
    large_paths:    List[Path] = []
    ood_size_paths: List[Path] = []

    if eval_small:
        # Hardcoded, not VARIABLE_TRAIN_SIZES: that can overlap ood_size_paths (11-19) below
        small_paths = get_variable_network_paths(
            networks_dir, sizes=list(range(3, 11)), split="train", tag=_file_tag
        )
    if eval_ood_size:
        # split="ood_size" bypasses the train/test size filters; sizes kwarg selects n=11–19
        ood_size_paths = get_variable_network_paths(
            networks_dir, sizes=list(range(11, 20)), split="ood_size", tag=_file_tag
        )
    if eval_large:
        large_paths = get_variable_network_paths(
            networks_dir, sizes=VARIABLE_TEST_SIZES, split="test", tag=_file_tag
        )

    logger.info(
        "Topology '%s': %d small (3–10n) + %d OOD-size (11–19n) + %d large (20n)",
        topology_tag, len(small_paths), len(ood_size_paths), len(large_paths),
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    # Tuple: (path, run_baselines, do_disnet, do_modnn, timeout_s, max_fail)
    network_results: Dict[str, Dict] = {}

    all_paths = (
        [(p, True, True,  True,  _BASELINE_TIMEOUT_S,     999) for p in small_paths]
        + [(p, True, True,  True,  _OOD_BASELINE_TIMEOUT_S, 5)   for p in ood_size_paths]
        + [(p, True, True,  True,  _OOD_BASELINE_TIMEOUT_S, 5)   for p in large_paths]
    )

    for net_path, do_baselines, do_disnet, do_modnn, bl_timeout, bl_max_fail in all_paths:
        net_name = net_path.stem
        try:
            node_profiles, link_profiles = generate_profiles_from_disnet_graph(
                str(net_path), max_bandwidth_cap=DISNET_BW_CAP,
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", net_name, exc)
            continue

        num_nodes = len(node_profiles)
        env = SplitEnv(
            num_nodes=num_nodes,
            max_chunks=num_nodes,
            use_offline_profiles=use_offline,
            seed=seed,
            inject_node_profiles=node_profiles,
            inject_link_profiles=link_profiles,
            use_bg_load=True,          # MMPP always on for GReMS evaluation
        )

        rec = evaluate_one_network(
            ac, env,
            num_sequences=num_sequences,
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            frozen_warmup=frozen_warmup,
            run_baselines=do_baselines and run_baselines,
            run_disnet=run_disnet and do_disnet,
            run_modnn=run_modnn and do_modnn,
            device=device,
            baseline_timeout_s=bl_timeout,
            max_consecutive_baseline_failures=bl_max_fail,
            ensplit_checkpoint=ensplit_checkpoint,
        )
        rec["net_name"]  = net_name
        rec["num_nodes"] = num_nodes

        # ── Per-network formatted output ───────────────────────────────────
        rl_ms   = rec["rl_mean_lat_ms"]
        rl_conn = rec["rl_mean_lat_ms_conn"]
        d_ms    = rec["disnet_mean_lat_ms"]
        m_ms    = rec["modnn_mean_lat_ms"]
        e_ms    = rec.get("ensplit_mean_lat_ms")
        cr      = rec["connectivity_rate"] or 0.0

        split = "TEST (in-dist)" if num_nodes in VARIABLE_TRAIN_SIZES else "TEST (OOD size)"
        print(f"\n  {net_name}  ({num_nodes} nodes)  [{split}]")
        print(f"    GReMS:     {rl_ms/1000:.4f} s  ({rl_ms:.2f} ms)"
              f"  [conn={cr*100:.0f}%"
              + (f"  conn_lat={rl_conn:.1f} ms" if rl_conn is not None else "")
              + "]")

        if d_ms is not None:
            imp_d = (d_ms - rl_ms) / d_ms * 100.0
            beat_d = "✓ BETTER" if rl_ms < d_ms else "✗ WORSE"
            print(f"    DiSNet:    {d_ms/1000:.4f} s  ({d_ms:.2f} ms)  [live]")
            print(f"    vs DiSNet: {imp_d:+.1f}%  [{beat_d}]")

        if m_ms is not None:
            imp_m = (m_ms - rl_ms) / m_ms * 100.0
            beat_m = "✓ BETTER" if rl_ms < m_ms else "✗ WORSE"
            print(f"    MoDNN:     {m_ms/1000:.4f} s  ({m_ms:.2f} ms)  [live]")
            print(f"    vs MoDNN:  {imp_m:+.1f}%  [{beat_m}]")

        if e_ms is not None:
            imp_e = (e_ms - rl_ms) / e_ms * 100.0
            beat_e = "✓ BETTER" if rl_ms < e_ms else "✗ WORSE"
            print(f"    EnSplit:   {e_ms/1000:.4f} s  ({e_ms:.2f} ms)  [live]")
            print(f"    vs EnSplit: {imp_e:+.1f}%  [{beat_e}]")

        network_results[net_name] = rec

    # ── Aggregate ──────────────────────────────────────────────────────────
    size_summary = _aggregate_by_size(network_results)
    curves       = _step_curves(network_results)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  GReMS-{run_id} [{topology_tag.upper()}]  –  Summary by node size")
    print(f"  {'n':>2}  {'GReMS':>9}  {'DiSNet':>9}  {'MoDNN':>9}  {'EnSplit':>9}"
          f"  {'wD':>4}  {'wM':>4}  {'wE':>4}  {'conn':>4}")
    print("  " + "-" * 68)
    for n, s in sorted(size_summary.items()):
        rl  = f"{s['rl_mean']:7.1f} ms"
        d   = f"{s['disnet_mean']:7.1f} ms"      if s["disnet_mean"]               is not None else "      n/a"
        m   = f"{s['modnn_mean']:7.1f} ms"        if s["modnn_mean"]                is not None else "      n/a"
        e   = f"{s.get('ensplit_mean'):7.1f} ms"  if s.get("ensplit_mean")          is not None else "      n/a"
        wd  = f"{s['rl_beat_disnet_conn']:4.0f}%"      if s["rl_beat_disnet_conn"]           is not None else " n/a"
        wm  = f"{s['rl_beat_modnn_conn']:4.0f}%"       if s.get("rl_beat_modnn_conn")        is not None else " n/a"
        we  = f"{s.get('rl_beat_ensplit_conn'):4.0f}%" if s.get("rl_beat_ensplit_conn")       is not None else " n/a"
        cr  = f"{s['connectivity_rate']*100:4.0f}%"    if s["connectivity_rate"]             is not None else " n/a"
        print(f"  {n:2d}  {rl}  {d}  {m}  {e}  {wd}  {wm}  {we}  {cr}")

    beaten_d = sum(
        1 for r in network_results.values()
        if r["disnet_mean_lat_ms"] is not None and r["rl_mean_lat_ms"] < r["disnet_mean_lat_ms"]
    )
    total_with_d = sum(1 for r in network_results.values() if r["disnet_mean_lat_ms"] is not None)
    beaten_m = sum(
        1 for r in network_results.values()
        if r["modnn_mean_lat_ms"] is not None and r["rl_mean_lat_ms"] < r["modnn_mean_lat_ms"]
    )
    total_with_m = sum(1 for r in network_results.values() if r["modnn_mean_lat_ms"] is not None)

    print()
    if total_with_d:
        mark = "★" if beaten_d == total_with_d else " "
        print(f"  {mark} GReMS beats DiSNet on {beaten_d}/{total_with_d} networks")
    if total_with_m:
        mark = "★" if beaten_m == total_with_m else " "
        print(f"  {mark} GReMS beats MoDNN  on {beaten_m}/{total_with_m} networks")
    beaten_e = sum(
        1 for r in network_results.values()
        if r.get("ensplit_mean_lat_ms") is not None and r["rl_mean_lat_ms"] < r["ensplit_mean_lat_ms"]
    )
    total_with_e = sum(1 for r in network_results.values() if r.get("ensplit_mean_lat_ms") is not None)
    if total_with_e:
        mark = "★" if beaten_e == total_with_e else " "
        print(f"  {mark} GReMS beats EnSplit on {beaten_e}/{total_with_e} networks")
    print("=" * 72)

    # ── Save JSON ──────────────────────────────────────────────────────────
    output = {
        "run_id":          run_id,
        "topology_tag":    topology_tag,
        "model_path":      str(ckpt_path),
        "num_sequences":   num_sequences,
        "num_steps":       num_steps,
        "warmup_steps":    warmup_steps,
        "frozen_warmup":   frozen_warmup,
        "obs_dim":         obs_d,
        "networks":        network_results,
        "size_summary":    {str(k): v for k, v in size_summary.items()},
        "step_curves":     curves,
    }
    if save_json:
        save_path = Path(save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved → %s", save_path)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Dynamic multi-sequence GReMS evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str, required=True,
                   help="Path to grems_best.pt or grems_final.pt checkpoint.")
    p.add_argument("--run-id", type=str, default="K1",
                   help="Run identifier (used in output directory names).")
    p.add_argument("--topology-tag", type=str, default="rgg",
                   choices=["rgg", "ba", "ws", "er", "test"],
                   help="Which topology family to evaluate.")
    p.add_argument("--num-sequences", type=int, default=10,
                   help="Number of independent trajectories per network.")
    p.add_argument("--num-steps", type=int, default=50,
                   help="Steps per trajectory — both Markov chains advance at each step.")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="G-E/MMPP advances after reset() before adaptive GReMS recording; ~10 reaches steady-state.")
    p.add_argument("--frozen-warmup", type=int, default=0,
                   help="G-E/MMPP advances before frozen baseline takes its step-0 snapshot (default 0 = pristine all-GOOD state).")
    p.add_argument("--networks-dir", type=str, default=None,
                   help="Override topology directory (auto-inferred from --topology-tag).")
    p.add_argument("--save-json", type=str, default=None,
                   help="Save full results to this JSON file.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", type=str, default=DEVICE)
    p.add_argument("--offline", action="store_true",
                   help="Use offline layer profiles.")
    p.add_argument("--no-baselines", action="store_true", default=False,
                   help="Skip all baselines (faster; RL-only evaluation).")
    p.add_argument("--no-disnet", action="store_true", default=False,
                   help="Skip DiSNet baseline only.")
    p.add_argument("--no-modnn", action="store_true", default=False,
                   help="Skip MoDNN baseline only.")
    p.add_argument("--ensplit-checkpoint", type=str, default=None,
                   help="Path to ensplit_best.pt; if omitted, brute-force optimal split is used.")
    p.add_argument("--no-small", action="store_true", default=False,
                   help="Skip 3–10-node evaluation.")
    p.add_argument("--no-ood-size", action="store_true", default=False,
                   help="Skip 11–19-node OOD-size evaluation.")
    p.add_argument("--no-large", action="store_true", default=False,
                   help="Skip 20-node evaluation.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main_grems(
        model_path=args.model,
        run_id=args.run_id,
        topology_tag=args.topology_tag,
        num_sequences=args.num_sequences,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        frozen_warmup=args.frozen_warmup,
        seed=args.seed,
        device=args.device,
        use_offline=args.offline,
        eval_small=not args.no_small,
        eval_ood_size=not args.no_ood_size,
        eval_large=not args.no_large,
        run_baselines=not args.no_baselines,
        run_disnet=not args.no_disnet,
        run_modnn=not args.no_modnn,
        ensplit_checkpoint=args.ensplit_checkpoint,
        networks_dir=Path(args.networks_dir) if args.networks_dir else None,
        save_json=args.save_json,
    )
