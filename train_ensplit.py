"""train_ensplit.py – Training script for the EnSplit DDQN baseline (1500 episodes × 50 steps, Table II).
Saves best/final checkpoints and learning_curve.csv to checkpoints/ensplit/."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch

from config import CHECKPOINT_DIR
from ensplit_env import EnSplitEnv, N_CHAIN_NODES, EPISODE_STEPS
from ensplit_ddqn import DDQNAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_ensplit")

# ── Defaults (Table II) ───────────────────────────────────────────────────────
N_EPISODES        = 1500
EPS_DECAY_EPISODES = 1100   # ε reaches eps_end after this many episodes (Table II)
LR                = 1e-5
GAMMA             = 0.9
PSI               = 1e-6
BATCH_SIZE        = 10
BUFFER_CAP        = 10_000
EPS_START         = 0.9
EPS_END           = 1e-5
LOG_INTERVAL      = 50    # print progress every N episodes


def train(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    env   = EnSplitEnv(n_nodes=N_CHAIN_NODES, seed=args.seed)
    agent = DDQNAgent(
        state_dim          = env.state_dim,
        n_actions          = env.n_actions,
        lr                 = args.lr,
        gamma              = args.gamma,
        psi                = args.psi,
        batch_size         = args.batch_size,
        buffer_cap         = args.buffer_cap,
        eps_start          = args.eps_start,
        eps_end            = args.eps_end,
        n_train_eps        = args.episodes,
        eps_decay_episodes = args.eps_decay_episodes,
        device             = device,
    )

    logger.info(
        "EnSplit DDQN | state_dim=%d  n_actions=%d  episodes=%d",
        env.state_dim, env.n_actions, args.episodes,
    )

    csv_path = ckpt_dir / "learning_curve.csv"
    best_path = ckpt_dir / "ensplit_best.pt"
    final_path = ckpt_dir / "ensplit_final.pt"

    best_reward: float = -np.inf
    history: list[dict] = []

    t0 = time.time()
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        ep_losses: list[float] = []
        ep_rewards: list[float] = []

        for _ in range(EPISODE_STEPS):
            action = agent.select_action(obs, greedy=False)
            next_obs, reward, done, info = env.step(action)

            agent.push(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            ep_rewards.append(reward)
            obs = next_obs
            if done:
                break

        # Decay ε after each episode (Table II: linear schedule reaches eps_end after ~1100 episodes)
        agent.decay_epsilon()

        mean_loss   = float(np.mean(ep_losses))   if ep_losses   else float("nan")
        mean_reward = float(np.mean(ep_rewards))

        history.append({"episode": ep, "mean_loss": mean_loss, "mean_reward": mean_reward})

        # Save best checkpoint
        if mean_reward > best_reward:
            best_reward = mean_reward
            agent.save(str(best_path))

        if ep % LOG_INTERVAL == 0 or ep == 1:
            elapsed = time.time() - t0
            logger.info(
                "ep %4d/%d | ε=%.4f | loss=%.4f | reward=%.4f | best=%.4f | %.1f s",
                ep, args.episodes, agent.eps, mean_loss, mean_reward, best_reward, elapsed,
            )

    # Save final checkpoint
    agent.save(str(final_path))
    logger.info("Training complete. Best reward=%.4f", best_reward)

    # Write learning curve CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "mean_loss", "mean_reward"])
        writer.writeheader()
        writer.writerows(history)
    logger.info("Learning curve saved to %s", csv_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EnSplit DDQN baseline")
    p.add_argument("--episodes",           type=int,   default=N_EPISODES)
    p.add_argument("--eps-decay-episodes", type=int,   default=EPS_DECAY_EPISODES)
    p.add_argument("--lr",                 type=float, default=LR)
    p.add_argument("--gamma",         type=float, default=GAMMA)
    p.add_argument("--psi",           type=float, default=PSI)
    p.add_argument("--batch-size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--buffer-cap",    type=int,   default=BUFFER_CAP)
    p.add_argument("--eps-start",     type=float, default=EPS_START)
    p.add_argument("--eps-end",       type=float, default=EPS_END)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--checkpoint-dir", type=str,
                   default=str(CHECKPOINT_DIR / "ensplit"))
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
