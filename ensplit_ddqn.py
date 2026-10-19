"""ensplit_ddqn.py – DDQN agent for EnSplit (Table II: 128-neuron MLP, Huber loss, Adam lr=1e-5, γ=0.9, ψ=1e-6).
Double-Q target: Y = r + γ·Q(s', argmax_a Q(s',a;θ_m); θ_t) — MAIN selects, TARGET evaluates."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── Network ───────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Feed-forward Q-network: state → Q-values; one hidden layer with 128 neurons (Table II)."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-capacity circular replay buffer storing (s, a, r, s', done) tuples."""

    def __init__(self, capacity: int = 10_000):
        self._buf: Deque[Tuple] = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,               dtype=torch.long),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones,                 dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ── DDQN agent ────────────────────────────────────────────────────────────────

class DDQNAgent:
    """DDQN agent matching EnSplit Table II; all hyper-params are keyword args with Table II defaults."""

    def __init__(
        self,
        state_dim:          int,
        n_actions:          int,
        lr:                 float = 1e-5,
        gamma:              float = 0.9,
        psi:                float = 1e-6,
        batch_size:         int   = 10,
        buffer_cap:         int   = 10_000,
        eps_start:          float = 0.9,
        eps_end:            float = 1e-5,
        n_train_eps:        int   = 1500,
        eps_decay_episodes: int   = 1100,
        device:             Optional[torch.device] = None,
    ):
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.psi        = psi
        self.batch_size = batch_size
        self.device     = device or torch.device("cpu")

        # ε schedule: linear decay over n_train_eps episodes
        self.eps        = eps_start
        self.eps_end    = eps_end
        # ε reaches eps_end after eps_decay_episodes (~1100, Table II), then stays flat.
        self._eps_decay = (eps_start - eps_end) / max(eps_decay_episodes - 1, 1)

        # Networks
        self.main   = QNetwork(state_dim, n_actions).to(self.device)
        self.target = QNetwork(state_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.main.state_dict())
        self.target.eval()

        # Optimizer and loss (Table II)
        self.optimizer = optim.Adam(self.main.parameters(), lr=lr)
        self.loss_fn   = nn.HuberLoss()

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_cap)

        self._train_steps: int = 0

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection; greedy=True always picks highest Q-value (eval mode)."""
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)

        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.main(s)
        return int(q.argmax(dim=1).item())

    # ── Learning ──────────────────────────────────────────────────────────────

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """One gradient-descent update using the Double Q-Network target; returns loss or None if buffer too small."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Current Q-values from MAIN
        q_current = self.main(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        #   arg max_a Q(s', a; θ_m)  ← MAIN selects best action
        #   Q(s', a*; θ_t)           ← TARGET evaluates it
        with torch.no_grad():
            best_actions = self.main(next_states).argmax(dim=1, keepdim=True)
            q_next       = self.target(next_states).gather(1, best_actions).squeeze(1)
            y_target     = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_current, y_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft target-network update: θ_t ← (1-ψ)·θ_t + ψ·θ_m
        with torch.no_grad():
            for p_m, p_t in zip(self.main.parameters(), self.target.parameters()):
                p_t.data.copy_((1.0 - self.psi) * p_t.data + self.psi * p_m.data)

        self._train_steps += 1
        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Call once per episode to linearly decay ε toward eps_end."""
        self.eps = max(self.eps_end, self.eps - self._eps_decay)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "main_state_dict":   self.main.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer":         self.optimizer.state_dict(),
            "eps":               self.eps,
            "train_steps":       self._train_steps,
        }, path)

    def load(self, path: str, map_location: Optional[torch.device] = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.main.load_state_dict(ckpt["main_state_dict"])
        self.target.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.eps          = ckpt.get("eps", self.eps_end)
        self._train_steps = ckpt.get("train_steps", 0)
        self.main.eval()
        self.target.eval()
