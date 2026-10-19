"""
PPO training loop for GReMS.

Trains the GNN actor-critic on variable-topology networks with MMPP dynamic compute
load and Gilbert-Elliott link BW. Supports mixed-topology training (BA, WS, RGG, ER),
curriculum connectivity penalty annealing, checkpoint resume, and optional W&B logging.
Checkpoints are saved under CHECKPOINT_DIR/variable_networks_GReMS_<run-id>/.
"""

import argparse
from collections import deque
from typing import Dict, List, Optional

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    DISNET_BW_CAP,
    EPISODE_LENGTH,
    NUM_ROLLOUT_STEPS,
    PPO_BATCH_SIZE,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEFF,
    PPO_EPOCHS,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    PPO_VALUE_COEFF,
    SEED,
    TOTAL_TIMESTEPS,
    LOG_INTERVAL,
    VARIABLE_NET_BEST_PATH,
    VARIABLE_NET_FINAL_PATH,
    VARIABLE_NETWORKS_DIR,
    VARIABLE_TRAIN_SIZES,
    CONN_PENALTY_START,
    CONN_PENALTY_END,
    CONN_PENALTY_WARMUP_STEPS,
    ENTROPY_COEFF_END,
    ENTROPY_DECAY_STEPS,
    REWARD_RMS_RESUME_COUNT_CAP,
)
from environment import SplitEnv
from gnn_model import build_model
from utils import set_seed, setup_logging

logger = setup_logging("train_GReMS", log_file="train_GReMS.log")


# ─────────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────────

class RunningMeanStd:
    """Welford's online algorithm for tracking running mean and variance of rewards."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        batch_mean  = float(np.mean(x))
        batch_var   = float(np.var(x))
        batch_count = len(x)
        delta       = batch_mean - self.mean
        tot_count   = self.count + batch_count
        self.mean   = self.mean + delta * batch_count / tot_count
        m_a = self.var  * self.count
        m_b = batch_var * batch_count
        self.var    = (m_a + m_b + delta ** 2 * self.count * batch_count / tot_count) / tot_count
        self.count  = tot_count

    def state_dict(self) -> Dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: Dict) -> None:
        self.mean  = float(d["mean"])
        self.var   = float(d["var"])
        self.count = float(d["count"])


class RolloutBuffer:
    """Buffer for multi-step MDP rollouts with GAE advantage estimation."""

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[float] = []   # 1.0 if episode ended at this step

    def store(self, obs, action, log_prob, reward, value, done: bool):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(float(done))

    def get(self, device: str = "cpu", last_value: float = 0.0,
            reward_rms: Optional["RunningMeanStd"] = None):
        """Return tensors for PPO update using GAE(γ, λ)."""
        obs_t  = torch.as_tensor(np.array(self.obs),       dtype=torch.float32, device=device)
        act_t  = torch.as_tensor(np.array(self.actions),   dtype=torch.long,    device=device)
        logp_t = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=device)

        rewards = np.array(self.rewards, dtype=np.float64)

        # Normalize rewards with running statistics before GAE so the critic
        # always sees a zero-mean unit-variance target regardless of MMPP/G-E shifts.
        if reward_rms is not None:
            reward_rms.update(rewards)
            rewards = (rewards - reward_rms.mean) / np.sqrt(reward_rms.var + 1e-8)

        values  = np.array(self.values,  dtype=np.float64)
        dones   = np.array(self.dones,   dtype=np.float64)
        T = len(rewards)

        # GAE computation (handles episode boundaries via done masking)
        advantages = np.zeros(T, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            next_value = last_value if t == T - 1 else values[t + 1]
            next_value = next_value * next_non_terminal   # zero at episode end
            delta = rewards[t] + PPO_GAMMA * next_value - values[t]
            gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values

        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(returns,    dtype=torch.float32, device=device)

        # Normalise advantages for stable policy gradient
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        return obs_t, act_t, logp_t, adv_t, ret_t

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.obs)


# ─────────────────────────────────────────────────
# PPO update
# ─────────────────────────────────────────────────

def ppo_update(
    ac: nn.Module,
    optimizer: optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = PPO_CLIP_EPS,
    entropy_coeff: float = PPO_ENTROPY_COEFF,
    value_coeff: float = PPO_VALUE_COEFF,
    max_grad_norm: float = PPO_MAX_GRAD_NORM,
    epochs: int = PPO_EPOCHS,
    batch_size: int = PPO_BATCH_SIZE,
) -> Dict[str, float]:
    """Run PPO clipped-objective updates; return dict of mean losses."""
    n = obs.size(0)
    total_pg_loss = 0.0
    total_v_loss = 0.0
    total_ent = 0.0
    num_updates = 0

    for _ in range(epochs):
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            mb_obs = obs[idx]
            mb_act = actions[idx]
            mb_old_lp = old_log_probs[idx]
            mb_adv = advantages[idx]
            # Returns are NOT re-normalized here — rewards were already normalized in
            # buffer.get() so returns are in the same normalized scale as the critic target.
            mb_ret = returns[idx]

            new_lp, entropy, values = ac.evaluate(mb_obs, mb_act)

            # Policy loss (clipped surrogate)
            ratio = torch.exp(new_lp - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            pg_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            v_loss = nn.functional.mse_loss(values, mb_ret)

            # Entropy bonus (negative sign: maximising entropy = minimising ent_loss)
            ent_loss = -entropy.mean()

            # Weighted combination of all three objectives
            loss = pg_loss + value_coeff * v_loss + entropy_coeff * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_v_loss += v_loss.item()
            total_ent += entropy.mean().item()
            num_updates += 1

    return {
        "policy_loss": total_pg_loss / max(num_updates, 1),
        "value_loss": total_v_loss / max(num_updates, 1),
        "entropy": total_ent / max(num_updates, 1),
    }


# ─────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────

def train(
    run_id: str = "K1",
    total_timesteps: int = TOTAL_TIMESTEPS,
    rollout_steps: int = NUM_ROLLOUT_STEPS,
    lr: float = PPO_LR,
    seed: int = SEED,
    device: str = DEVICE,
    use_offline_profiles: bool = False,
    variable_networks: bool = False,
    topology_models: List[str] = None,
    log_interval: int = LOG_INTERVAL,
    wandb_project: str = "",
    wandb_run_name: str = "",
    entropy_coeff: float = PPO_ENTROPY_COEFF,
    clip_eps: float = PPO_CLIP_EPS,
    batch_size: int = PPO_BATCH_SIZE,
    best_model_path: str = "",
    final_model_path: str = "",
    resume_path: str = "",
    resume_step: int = None,
    eval_interval: int = 0,
    use_bg_load: bool = True,
    large_graph_size: int = 7,
    small_graph_size: int = 0,
    ppo_epochs: int = PPO_EPOCHS,
) -> nn.Module:
    """Train GReMS. Pass --variable-networks; saves best checkpoint."""
    set_seed(seed)

    # ── Build environment(s) ──
    if variable_networks:
        from profiling import generate_profiles_from_disnet_graph
        from generate_variable_networks import get_variable_network_paths

        _active_topos = topology_models if topology_models else ["ba"]
        _topo_dir_map = {"ba": VARIABLE_NETWORKS_DIR}
        for _tm in _active_topos:
            if _tm != "ba":
                _topo_dir_map[_tm] = VARIABLE_NETWORKS_DIR.parent / f"variable_networks_{_tm}"

        _paths_by_topo: Dict[str, list] = {}
        for _tm in _active_topos:
            _tdir = _topo_dir_map[_tm]
            _train_tag = "" if _tm == "ba" else _tm
            _paths = get_variable_network_paths(
                directory=_tdir,
                sizes=VARIABLE_TRAIN_SIZES,
                split="train",
                tag=_train_tag,
            )
            if not _paths:
                raise RuntimeError(
                    f"No training networks found in {_tdir} for topology '{_tm}'. "
                    "Run generate_variable_networks.py first."
                )
            _paths_by_topo[_tm] = _paths
            logger.info("Topology '%s': %d training networks in %s", _tm, len(_paths), _tdir)

        envs = []
        _topo_env_ranges: Dict[str, tuple] = {}
        global_idx = 0
        for _tm in _active_topos:
            _start = len(envs)
            for net_path in _paths_by_topo[_tm]:
                node_profiles, link_profiles = generate_profiles_from_disnet_graph(
                    str(net_path),
                    max_bandwidth_cap=DISNET_BW_CAP,
                )
                num_nodes = len(node_profiles)
                env = SplitEnv(
                    num_nodes=num_nodes,
                    max_chunks=num_nodes,
                    use_offline_profiles=use_offline_profiles,
                    seed=seed,
                    inject_node_profiles=node_profiles,
                    inject_link_profiles=link_profiles,
                    network_id=global_idx,
                    use_bg_load=use_bg_load,
                )
                envs.append(env)
                global_idx += 1
            _topo_env_ranges[_tm] = (_start, len(envs))
        assert len(envs) > 0

        logger.info(
            "Mixed-topology training: %d networks across %d topology families %s",
            len(envs), len(_active_topos), _active_topos,
        )

        ref_env = envs[0]
        ckpt_name = f"variable_networks_GReMS_{run_id}"
        ckpt_dir_path = CHECKPOINT_DIR / ckpt_name
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)
        best_path  = ckpt_dir_path / "grems_best.pt"
        final_path = ckpt_dir_path / "grems_final.pt"

    else:
        raise ValueError("Pass --variable-networks to select a topology.")

    # Optional output-path overrides
    if best_model_path:
        best_path = CHECKPOINT_DIR / best_model_path
        best_path.parent.mkdir(parents=True, exist_ok=True)
    if final_model_path:
        final_path = CHECKPOINT_DIR / final_model_path
        final_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Periodic seen-performance evaluation setup ──
    if eval_interval > 0:
        _eval_interval = eval_interval
    else:
        # Default: eval roughly every 100 rollouts, minimum every 500k steps
        _eval_interval = max(rollout_steps * 100, 500_000)
    _eval_episodes   = 5
    _eval_fixed_seed = 2025  # fixed so learning-curve comparisons are reproducible

    _eval_rng = np.random.default_rng(seed=42)
    _eval_fixed_by_topo: Dict[str, list] = {}
    _EVAL_PER_SIZE = 15  # max envs sampled per node-size bucket for the fixed eval set

    _eval_ranges = (
        _topo_env_ranges.items()
        if variable_networks and topology_models
        else [("ba", (0, len(envs)))]
    )
    for _tm, (_s, _e) in _eval_ranges:
        _size_buckets: Dict[int, list] = {}
        for _i in range(_s, _e):
            _nn = getattr(envs[_i], "num_nodes", 0)
            _size_buckets.setdefault(_nn, []).append(_i)
        _fixed_idxs = []
        for _nn in sorted(_size_buckets):
            _bucket = _size_buckets[_nn]
            _pick = _eval_rng.choice(
                _bucket,
                size=min(_EVAL_PER_SIZE, len(_bucket)),
                replace=False,
            )
            _fixed_idxs.extend(_pick.tolist())
        _eval_fixed_by_topo[_tm] = [envs[_i] for _i in _fixed_idxs]
        logger.info(
            "Fixed eval set '%s': %d envs across %d node-sizes",
            _tm, len(_eval_fixed_by_topo[_tm]), len(_size_buckets),
        )

    _curve_csv_path       = best_path.parent / "learning_curve.csv"
    _curve_header_written = _curve_csv_path.exists()
    _next_eval_step       = _eval_interval

    def _truncate_curve_csv(up_to_step: int):
        # On resume, drop eval rows beyond the resume step to avoid a gap in the curve.
        if not _curve_csv_path.exists():
            return
        import csv as _csv
        with open(_curve_csv_path, "r", newline="") as _rf:
            reader = _csv.DictReader(_rf)
            kept = [r for r in reader if int(r["step"]) <= up_to_step]
            fieldnames = reader.fieldnames
        if fieldnames is None:
            return
        with open(_curve_csv_path, "w", newline="") as _wf:
            writer = _csv.DictWriter(_wf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        logger.info("learning_curve.csv truncated to step<=%d (%d rows kept)",
                    up_to_step, len(kept))

    # ── Model ──
    obs_d = ref_env.observation_space.shape[0]
    ac = build_model(obs_d, ref_env.spec_action).to(device)
    logger.info("GReMS GNNActorCritic  obs_dim=%d  params=%d",
                obs_d, sum(p.numel() for p in ac.parameters() if p.requires_grad))
    optimizer = optim.Adam(ac.parameters(), lr=lr)

    buffer = RolloutBuffer()
    reward_rms = RunningMeanStd()
    best_reward = -float("inf")
    # Rolling 100-step windows used for logging metrics
    reward_history   = deque(maxlen=100)
    latency_history  = deque(maxlen=100)
    hub_history      = deque(maxlen=100)
    distinct_history = deque(maxlen=100)
    groups_history   = deque(maxlen=100)
    load_history     = deque(maxlen=100)
    frac_high_history = deque(maxlen=100)
    rqb_history      = deque(maxlen=100)

    global_step   = 0
    rollout_count = 0

    # ── Resume from checkpoint ──
    latest_path = best_path.parent / "grems_latest.pt"
    _resume = resume_path or (str(latest_path) if latest_path.exists() else "")
    if _resume:
        _ckpt = torch.load(_resume, map_location=device, weights_only=False)
        if isinstance(_ckpt, dict) and "model" in _ckpt:
            ac.load_state_dict(_ckpt["model"])
            optimizer.load_state_dict(_ckpt["optimizer"])
            global_step   = _ckpt.get("global_step", 0)
            rollout_count = _ckpt.get("rollout_count", 0)
            best_reward   = _ckpt.get("best_reward", -float("inf"))
            if "reward_rms" in _ckpt:
                reward_rms.load_state_dict(_ckpt["reward_rms"])
            # --resume-step explicitly overrides the checkpoint's step counter.
            # Default is None (not set); 0 is a valid explicit override.
            if resume_step is not None:
                global_step   = resume_step
                rollout_count = resume_step // rollout_steps
                best_reward   = -float("inf")
                logger.info("Step counter overridden to %d via --resume-step.", resume_step)
                if REWARD_RMS_RESUME_COUNT_CAP > 0 and reward_rms.count > REWARD_RMS_RESUME_COUNT_CAP:
                    logger.info("reward_rms.count capped %.0f -> %.0f", reward_rms.count, REWARD_RMS_RESUME_COUNT_CAP)
                    reward_rms.count = REWARD_RMS_RESUME_COUNT_CAP
        else:
            ac.load_state_dict(_ckpt)
            if resume_step:
                global_step   = resume_step
                rollout_count = resume_step // rollout_steps
                logger.warning("Bare state_dict — resuming weights only; step set to %d.", resume_step)
            else:
                logger.warning("Bare state_dict — step counter resets to 0. Pass --resume-step.")
        logger.info("Resumed from %s  (step=%d, best_reward=%.4f)", _resume, global_step, best_reward)
        _truncate_curve_csv(global_step)
        _curve_header_written = _curve_csv_path.exists()
        while _next_eval_step <= global_step:
            _next_eval_step += _eval_interval

    rollout_eps: list = []
    n_envs = len(envs)

    _env_obs:  List[Optional[np.ndarray]] = [None] * n_envs
    _env_done: List[bool]                 = [True] * n_envs  # True forces reset on first step

    _env_topo: List[str] = ["ba"] * n_envs
    if variable_networks and topology_models:
        for _tm, (_s, _e) in _topo_env_ranges.items():
            for _i in range(_s, _e):
                _env_topo[_i] = _tm

    # ── Topology-balanced + size-weighted sampling ──
    # Each topology family gets equal total weight; within a family, larger graphs
    # (>= large_graph_size nodes) or smaller graphs (<= small_graph_size nodes,
    # 0 = disabled) are upweighted 2×. Not meant to be used together.
    _env_weights = np.zeros(len(envs), dtype=np.float64)
    _n_topos = len(_topo_env_ranges) if variable_networks and topology_models else 1
    for _tm, (_s, _e) in (_topo_env_ranges.items() if variable_networks and topology_models
                          else [("ba", (0, len(envs)))]):
        _slice_weights = np.array(
            [2.0 if (getattr(envs[i], "num_nodes", 0) >= large_graph_size
                     or (small_graph_size and getattr(envs[i], "num_nodes", 0) <= small_graph_size))
             else 1.0
             for i in range(_s, _e)],
            dtype=np.float64,
        )
        _slice_weights /= _slice_weights.sum() * _n_topos
        _env_weights[_s:_e] = _slice_weights

    logger.info("Starting GReMS training  |  run=%s  timesteps=%d  rollout=%d  lr=%.1e  envs=%d  bg_load=%s",
                run_id, total_timesteps, rollout_steps, lr, n_envs, use_bg_load)

    # ── W&B initialisation ──
    _use_wandb = _WANDB_AVAILABLE and bool(wandb_project)
    if _use_wandb:
        try:
            _wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"GReMS-{run_id}",
                config={
                    "run_id":           run_id,
                    "total_timesteps":  total_timesteps,
                    "rollout_steps":    rollout_steps,
                    "lr":               lr,
                    "seed":             seed,
                    "n_envs":           n_envs,
                    "ppo_clip_eps":     clip_eps,
                    "ppo_epochs":       ppo_epochs,
                    "ppo_batch_size":   batch_size,
                    "ppo_entropy_coeff": entropy_coeff,
                    "ppo_gamma":        PPO_GAMMA,
                    "ppo_gae_lambda":   PPO_GAE_LAMBDA,
                    "use_bg_load":      use_bg_load,
                    "mode": "variable_networks" if variable_networks else "all_networks",
                },
            )
            logger.info("W&B run started: %s/%s", wandb_project, _wandb.run.name)
        except Exception as e:
            logger.warning("W&B init failed (%s) – continuing without W&B.", e)
            _use_wandb = False

    while global_step < total_timesteps:
        # ── Linear LR annealing ──
        frac = 1.0 - global_step / total_timesteps
        cur_lr = lr * frac
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # ── Curriculum connectivity penalty annealing ──
        conn_beta = CONN_PENALTY_START + (CONN_PENALTY_END - CONN_PENALTY_START) * min(
            1.0, global_step / CONN_PENALTY_WARMUP_STEPS
        )
        for _env in envs:
            _env.connectivity_penalty_beta = conn_beta

        # ── Curriculum entropy-coefficient decay ──
        cur_entropy_coeff = entropy_coeff + (ENTROPY_COEFF_END - entropy_coeff) * min(
            1.0, global_step / ENTROPY_DECAY_STEPS
        )

        # ── Collect rollout ──
        buffer.clear()
        rollout_eps = []
        _last_next_obs = None
        _last_done     = True

        for _ in range(rollout_steps):
            env_idx = np.random.choice(n_envs, p=_env_weights)
            env = envs[env_idx]
            _topo_name = _env_topo[env_idx]

            if _env_done[env_idx]:
                _env_obs[env_idx], _ = env.reset()

            obs   = _env_obs[env_idx]
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, log_prob, value = ac.act(obs_t)

            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated or truncated

            buffer.store(
                obs,
                action.cpu().numpy(),
                log_prob.item(),
                reward,
                value.item(),
                done,
            )

            _env_done[env_idx] = done
            _env_obs[env_idx]  = next_obs
            _last_next_obs     = next_obs
            _last_done         = done

            reward_history.append(reward)
            latency_history.append(info["latency_ms"])
            hub_history.append(info.get("hub_node", -1))
            distinct_history.append(info.get("num_distinct_nodes", 0))
            load_history.append(info.get("mean_node_load", 0.0))
            frac_high_history.append(info.get("frac_high_load", 0.0))
            rqb_history.append(info.get("routing_quality_bonus", 0.0))
            n_active_groups = sum(1 for gs in info.get("group_sizes", []) if gs > 0)
            groups_history.append(n_active_groups)

            if done:
                _g_sizes = info.get("group_sizes", [])
                _chunks  = info.get("chunks", [])
                _assign  = info.get("assignment", [])
                _workers = sorted(set(
                    w
                    for g, gs in enumerate(_g_sizes) if gs > 0
                    for w in (_assign[g][: _chunks[g]] if g < len(_assign) and g < len(_chunks) else [])
                ))
                rollout_eps.append({
                    "hub":     info.get("hub_node", -1),
                    "workers": _workers,
                    "groups":  sum(1 for gs in _g_sizes if gs > 0),
                    "lat_ms":  info.get("latency_ms", 0.0),
                    "reward":  reward,
                    "n_total": info.get("num_nodes", "?"),
                    "net_id":  info.get("network_id", "?"),
                    "topo":    _topo_name,
                    "g_sizes": _g_sizes,
                    "chunks":  info.get("chunks", []),
                    "assign":  info.get("assignment", []),
                    "nb":      info.get("node_util_bonus", 0.0),
                    "vb":      info.get("vsplit_bonus", 0.0),
                    "rqb":     info.get("routing_quality_bonus", 0.0),
                    "mp":      info.get("memory_penalty", 0.0),
                    "conn_p":  info.get("conn_penalty", 0.0),
                    "src":        info.get("src_node", -1),
                    "dst":        info.get("dst_node", -1),
                    "path":       list(info.get("path_nodes", [])),
                    "hub_scores": info.get("hub_scores_norm", np.array([])),
                    "mean_load":  info.get("mean_node_load", 0.0),
                    "frac_high":  info.get("frac_high_load", 0.0),
                })

            global_step += 1

        # ── Bootstrap value for GAE ──
        if _last_done or _last_next_obs is None:
            _last_value = 0.0
        else:
            with torch.no_grad():
                _last_obs_t = torch.as_tensor(
                    _last_next_obs, dtype=torch.float32, device=device
                )
                _, _, _lv = ac.act(_last_obs_t)
                _last_value = _lv.item()

        # ── PPO update ──
        obs_b, act_b, logp_b, adv_b, ret_b = buffer.get(device, _last_value, reward_rms=reward_rms)
        losses = ppo_update(
            ac, optimizer, obs_b, act_b, logp_b, adv_b, ret_b,
            clip_eps=clip_eps,
            entropy_coeff=cur_entropy_coeff,
            batch_size=batch_size,
            epochs=ppo_epochs,
        )

        rollout_count += 1

        # ── Logging ──
        mean_rew    = np.mean(reward_history)
        mean_lat    = np.mean(latency_history)
        mean_dist   = np.mean(distinct_history)
        mean_groups = np.mean(groups_history) if groups_history else 0.0
        vsplit_rate = float(np.mean([g > 1 for g in groups_history])) if groups_history else 0.0  # fraction of steps using >1 group
        mean_load   = float(np.mean(load_history)) if load_history else 0.0
        frac_high   = float(np.mean(frac_high_history)) if frac_high_history else 0.0
        mean_rqb    = float(np.mean(rqb_history)) if rqb_history else 0.0

        if _use_wandb:
            _wandb.log({
                "rollout/mean_reward":         mean_rew,
                "rollout/mean_latency_ms":     mean_lat,
                "rollout/mean_groups":          mean_groups,
                "rollout/vsplit_rate_pct":      vsplit_rate * 100,
                "rollout/mean_distinct_nodes":  mean_dist,
                "rollout/mean_node_load":       mean_load,
                "rollout/frac_high_load_pct":   frac_high * 100,
                "rollout/mean_routing_quality": mean_rqb,
                "train/policy_loss":            losses["policy_loss"],
                "train/value_loss":             losses["value_loss"],
                "train/entropy":                losses["entropy"],
                "train/learning_rate":          cur_lr,
                "train/best_reward":            best_reward,
                "train/conn_penalty_beta":      conn_beta,
                "train/entropy_coeff":          cur_entropy_coeff,
            }, step=global_step)

        if rollout_count % log_interval == 0 or global_step >= total_timesteps:
            warns = []
            if vsplit_rate == 0.0:
                warns.append("NO-VSPLIT")   # agent collapsed to single-group plans
            if len(set(list(hub_history)[-20:])) == 1 and len(hub_history) >= 20:
                warns.append(f"HUB-FIXED={list(hub_history)[-1]}")  # hub selection stuck
            warn_str = "  ⚠ " + " ".join(warns) if warns else ""
            logger.info(
                "step=%-7d  rollout=%-4d │ rew=%-7.2f lat=%-8.1fms │ "
                "grp=%.1f(%.0f%%vsplit) nodes=%.1f │ load=%.2f(%.0f%%hi) │ "
                "pg=%.3f v=%.3f H=%.1f ent_c=%.4f%s",
                global_step, rollout_count,
                mean_rew, mean_lat,
                mean_groups, vsplit_rate * 100, mean_dist,
                mean_load, frac_high * 100,
                losses["policy_loss"], losses["value_loss"], losses["entropy"],
                cur_entropy_coeff,
                warn_str,
            )

            sample = rollout_eps[-10:]
            if sample:
                logger.info(
                    "  %4s %2s %4s  G  %7s  %6s  %4s %4s %5s %4s %5s │ split-plan",
                    "net", "N", "hub", "lat(ms)", "rew", "nb", "vb", "mp", "topo", "load",
                )
                for ep in sample:
                    hub_s = ep["hub"]
                    hub_flag = "★" if hub_s in ep["workers"] else " "
                    g_sizes = ep["g_sizes"]
                    chunks  = ep["chunks"]
                    assign  = ep["assign"]
                    active  = [(g, gs) for g, gs in enumerate(g_sizes) if gs > 0]
                    if len(active) == 1:
                        g, gs = active[0]
                        nc = chunks[g] if g < len(chunks) else 1
                        wk = sorted(set(assign[g][:nc])) if g < len(assign) else []
                        plan = f"{wk}×{gs}L"
                    else:
                        cum = 0
                        parts = []
                        for g, gs in active:
                            nc = chunks[g] if g < len(chunks) else 1
                            wk = sorted(set(assign[g][:nc])) if g < len(assign) else []
                            parts.append(f"L{cum+1}-{cum+gs}→{wk}")
                            cum += gs
                        plan = " | ".join(parts)
                    conn_flag = " ✗conn" if ep.get("conn_p", 0) > 0 else ""
                    logger.info(
                        "  %4s %2s %3s%s  %d  %7.1f  %+6.2f  %4.2f %4.2f %5.2f %4s %5.2f │ %s%s",
                        ep["net_id"], ep["n_total"],
                        hub_s, hub_flag,
                        ep["groups"], ep["lat_ms"], ep["reward"],
                        ep["nb"], ep["vb"], ep.get("mp", 0.0),
                        ep.get("topo", "ba"),
                        ep.get("mean_load", 0.0),
                        plan, conn_flag,
                    )
                    _hub_sc = ep.get("hub_scores", [])
                    if len(_hub_sc) > 0:
                        ranked = sorted(enumerate(_hub_sc), key=lambda x: -x[1])[:3]
                        top_hubs = " ".join(f"n{i}={v:.2f}" for i, v in ranked if v > 0.01)
                    else:
                        top_hubs = "n/a"
                    path_str = "->".join(str(p) for p in ep.get("path", []))
                    logger.info(
                        "    src->dst=%s->%s  path=[%s]  top-hubs=[%s]",
                        ep.get("src", "?"), ep.get("dst", "?"), path_str, top_hubs,
                    )

        # ── Periodic seen-performance evaluation ──
        if global_step >= _next_eval_step:
            _next_eval_step += _eval_interval
            ac.eval()
            _row: Dict[str, float] = {"step": global_step}

            for _tm, _fixed_envs in _eval_fixed_by_topo.items():
                _lats = []
                for _ep_idx, _env in enumerate(_fixed_envs):
                    for _rep in range(_eval_episodes):
                        _obs, _ = _env.reset(seed=_eval_fixed_seed + _ep_idx * _eval_episodes + _rep)
                        for _ in range(EPISODE_LENGTH):
                            _obs_t = torch.as_tensor(_obs, dtype=torch.float32, device=device)
                            _act, _, _ = ac.act(_obs_t, deterministic=True)
                            _obs, _, _term, _trunc, _info = _env.step(_act.cpu().numpy())
                            _lats.append(_info["latency_ms"])
                            if _term or _trunc:
                                break
                _row[f"seen_lat_ms_{_tm}"] = float(np.mean(_lats))
            _row["seen_lat_ms_all"] = float(np.mean([
                v for k, v in _row.items()
                if k.startswith("seen_lat_ms_") and k != "seen_lat_ms_all"
            ]))

            ac.train()

            import csv as _csv
            with open(_curve_csv_path, "a", newline="") as _cf:
                _writer = _csv.DictWriter(_cf, fieldnames=sorted(_row.keys()))
                if not _curve_header_written:
                    _writer.writeheader()
                    _curve_header_written = True
                _writer.writerow(_row)

            if _use_wandb:
                _wandb.log({f"eval/{k}": v for k, v in _row.items() if k != "step"},
                           step=global_step)

            logger.info(
                "  [eval] step=%d  seen_lat=%.1f ms %s",
                global_step, _row["seen_lat_ms_all"],
                "  ".join(f"{k.replace('seen_lat_ms_','')[:3]}={v:.0f}" for k, v in _row.items()
                           if k.startswith("seen_lat_ms_") and k != "seen_lat_ms_all"),
            )

        # ── Checkpoint ──
        # best_path: saved whenever rolling mean reward improves
        if mean_rew > best_reward:
            best_reward = mean_rew
            torch.save({
                "model":         ac.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "global_step":   global_step,
                "rollout_count": rollout_count,
                "best_reward":   best_reward,
                "model_arch":    "gnn",
                "run_id":        run_id,
                "obs_dim":       obs_d,
                "reward_rms":    reward_rms.state_dict(),
            }, best_path)
            logger.info("  ✓ New best GReMS model saved (reward=%.4f)", best_reward)

        # latest_path: periodic snapshot for crash recovery (every 50 rollouts)
        if rollout_count % 50 == 0:
            torch.save({
                "model":         ac.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "global_step":   global_step,
                "rollout_count": rollout_count,
                "best_reward":   best_reward,
                "model_arch":    "gnn",
                "run_id":        run_id,
                "obs_dim":       obs_d,
                "reward_rms":    reward_rms.state_dict(),
            }, latest_path)

    # final_path: unconditional end-of-training save (may be worse than best)
    torch.save({
        "model":         ac.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "global_step":   global_step,
        "rollout_count": rollout_count,
        "best_reward":   best_reward,
        "model_arch":    "gnn",
        "run_id":        run_id,
        "obs_dim":       obs_d,
        "reward_rms":    reward_rms.state_dict(),
    }, final_path)
    logger.info("Training complete.  Final GReMS model → %s", final_path)

    if _use_wandb:
        _wandb.finish()

    return ac


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train GReMS – Split-learning Policy with Adaptive Reinforcement Learning")
    parser.add_argument("--run-id", type=str, default="K1",
                        help="Run identifier used in checkpoint directory name (default: K1).")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--rollout", type=int, default=NUM_ROLLOUT_STEPS)
    parser.add_argument("--lr", type=float, default=PPO_LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--offline", action="store_true",
                        help="Use offline profiling data (must exist)")
    parser.add_argument("--variable-networks", action="store_true",
                        help="Train on variable-topology networks (3–10 nodes)")
    parser.add_argument("--topology-models", nargs="+", default=None,
                        choices=["ba", "ws", "rgg", "er"],
                        help="Topology families (default: ba). Use 'ba ws rgg er' for full mix.")
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=PPO_BATCH_SIZE)
    parser.add_argument("--entropy-coeff", type=float, default=PPO_ENTROPY_COEFF)
    parser.add_argument("--clip-eps", type=float, default=PPO_CLIP_EPS)
    parser.add_argument("--best-model-path", type=str, default="")
    parser.add_argument("--final-model-path", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=0,
                        help="Steps between seen-performance evaluations (0 = auto ~500k).")
    parser.add_argument("--no-bg-load", action="store_true", default=False,
                        help="Disable MMPP dynamic channels (ablation only; not recommended for GReMS).")
    parser.add_argument("--large-graph-size", type=int, default=7)
    parser.add_argument("--small-graph-size", type=int, default=0,
                        help="Upweight graphs with <= this many nodes 2x (0 = disabled). "
                             "Not meant to be combined with --large-graph-size.")
    parser.add_argument("--ppo-epochs", type=int, default=PPO_EPOCHS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        run_id=args.run_id,
        total_timesteps=args.timesteps,
        rollout_steps=args.rollout,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        use_offline_profiles=args.offline,
        variable_networks=args.variable_networks,
        topology_models=args.topology_models,
        log_interval=args.log_interval,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        entropy_coeff=args.entropy_coeff,
        clip_eps=args.clip_eps,
        batch_size=args.batch_size,
        best_model_path=args.best_model_path,
        final_model_path=args.final_model_path,
        resume_path=args.resume,
        resume_step=args.resume_step,
        eval_interval=args.eval_interval,
        use_bg_load=not args.no_bg_load,
        large_graph_size=args.large_graph_size,
        small_graph_size=args.small_graph_size,
        ppo_epochs=args.ppo_epochs,
    )
