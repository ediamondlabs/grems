"""
GNN-based PPO actor-critic for the GReMS split-inference agent.

Parses the flat 542-dim observation into graph-structured inputs, runs a
BW-weighted GraphSAGE backbone over the physical network topology, and
produces five action heads: num_active_groups, vertical_weights,
horizontal_chunks, hub_selection (per-node scorer), and node_assignment
(pointer network). Padded nodes are masked to -inf so the agent never
selects ghost nodes.

Usage:
    action, log_prob, value  = gnn_ac.act(obs)
    log_prob, entropy, value = gnn_ac.evaluate(obs, action)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from action_space import ActionSpec, DEFAULT_SPEC
from config import MAX_NODES, MAX_GROUPS
from state_encoder import (
    NODE_STAT_DIM, LINK_STAT_DIM, TOPO_META_DIM, MODEL_DIM,
    TOPO_STRUCT_DIM, PER_NODE_DIM, PATH_HUB_DIM,
    LAYER_SPLIT_COST_DIM, BOTTLENECK_BW_DIM, PREV_ACTION_DIM,
)


# ─────────────────────────────────────────────────────────────────────────────
# Observation block offsets (must stay in sync with state_encoder.py layout)
# ─────────────────────────────────────────────────────────────────────────────

_OFF_NODE_STATS  = 0
_OFF_LINK_STATS  = _OFF_NODE_STATS   + NODE_STAT_DIM          #  16
_OFF_TOPO_META   = _OFF_LINK_STATS   + LINK_STAT_DIM          #  24
_OFF_MODEL_FEATS = _OFF_TOPO_META    + TOPO_META_DIM          #  29
_OFF_TOPO_STRUCT = _OFF_MODEL_FEATS  + MODEL_DIM              #  41
_OFF_PER_NODE    = _OFF_TOPO_STRUCT  + TOPO_STRUCT_DIM        #  56
_OFF_PATH_HUB    = _OFF_PER_NODE     + PER_NODE_DIM           # 136
_OFF_SPLIT_COST  = _OFF_PATH_HUB     + PATH_HUB_DIM           # 176
_OFF_BOTTLENECK  = _OFF_SPLIT_COST   + LAYER_SPLIT_COST_DIM   # 218
_OFF_PREV_ACTION = _OFF_BOTTLENECK   + BOTTLENECK_BW_DIM      # 408
# Total: 408 + PREV_ACTION_DIM (134) = 542

# Dimension constants used in the model
_GLOBAL_DIM    = NODE_STAT_DIM + LINK_STAT_DIM + TOPO_META_DIM + TOPO_STRUCT_DIM  # 44
_NODE_FEAT_DIM = 6   # comp_norm, load, eff_cap, degree_norm, path_flag, hub_score
_MODEL_FLAT_DIM = MODEL_DIM + LAYER_SPLIT_COST_DIM  # 12 + 42 = 54

# Upper-triangle indices — computed once, moved to device on first use
_TRIU_ROWS, _TRIU_COLS = torch.triu_indices(MAX_NODES, MAX_NODES, offset=1)


# ─────────────────────────────────────────────────────────────────────────────
# Observation parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_obs(obs: torch.Tensor) -> dict:
    """Split the flat 542-dim obs into graph-structured components (batched or single)."""
    squeeze = obs.dim() == 1
    if squeeze:
        obs = obs.unsqueeze(0)
    B, dev = obs.shape[0], obs.device

    rows = _TRIU_ROWS.to(dev)
    cols = _TRIU_COLS.to(dev)

    # ── Global aggregate features (topology-size invariant) ──
    node_stats  = obs[:, _OFF_NODE_STATS  : _OFF_LINK_STATS]   # (B, 16)
    link_stats  = obs[:, _OFF_LINK_STATS  : _OFF_TOPO_META]    # (B, 8)
    topo_meta   = obs[:, _OFF_TOPO_META   : _OFF_MODEL_FEATS]  # (B, 5)
    topo_struct = obs[:, _OFF_TOPO_STRUCT : _OFF_PER_NODE]     # (B, 15)
    global_feats = torch.cat([node_stats, link_stats, topo_meta, topo_struct], dim=-1)

    # ── Per-node features (Blocks 6 + 7) ──
    per_node_flat = obs[:, _OFF_PER_NODE : _OFF_PATH_HUB  ]    # (B, 80)
    path_hub_flat = obs[:, _OFF_PATH_HUB : _OFF_SPLIT_COST]    # (B, 40)
    per_node = per_node_flat.view(B, MAX_NODES, 4)             # (B, N, 4)
    path_hub = path_hub_flat.view(B, MAX_NODES, 2)             # (B, N, 2)
    node_feats = torch.cat([per_node, path_hub], dim=-1)       # (B, N, 6)

    # ── BW-weighted adjacency upper triangle (Block 9) ──
    bw_flat  = obs[:, _OFF_BOTTLENECK : _OFF_PREV_ACTION]      # (B, 190)
    prev_act = obs[:, _OFF_PREV_ACTION:                ]       # (B, 134)

    bw_adj = torch.zeros(B, MAX_NODES, MAX_NODES, device=dev)
    bw_adj[:, rows, cols] = bw_flat
    bw_adj[:, cols, rows] = bw_flat

    # ── Model / split features ──
    model_feats_flat = obs[:, _OFF_MODEL_FEATS : _OFF_TOPO_STRUCT]  # (B, 12)
    split_costs      = obs[:, _OFF_SPLIT_COST  : _OFF_BOTTLENECK]   # (B, 42)
    model_flat = torch.cat([model_feats_flat, split_costs], dim=-1) # (B, 54)

    # ── Node mask: topo_meta[:, 0] = num_nodes / MAX_NODES ──
    # Identify which node slots are real vs. zero-padded.
    num_nodes_norm = topo_meta[:, 0]                                 # (B,)
    actual_nn = (num_nodes_norm * MAX_NODES).round().long().clamp(1, MAX_NODES)
    node_mask = torch.arange(MAX_NODES, device=dev).unsqueeze(0) < actual_nn.unsqueeze(1)

    return dict(
        global_feats=global_feats,
        node_feats=node_feats,
        bw_adj=bw_adj,
        model_flat=model_flat,
        node_mask=node_mask,
        prev_act=prev_act,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE layer
# ─────────────────────────────────────────────────────────────────────────────

class SAGELayer(nn.Module):
    """GraphSAGE layer with BW-weighted neighbour aggregation and optional residual."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self  = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.norm      = nn.LayerNorm(out_dim)
        self.residual  = (in_dim == out_dim)

    def forward(self, h: torch.Tensor, edge_w: torch.Tensor) -> torch.Tensor:
        # h: (B, N, in_dim), edge_w: (B, N, N) BW weights
        deg    = edge_w.sum(-1, keepdim=True).clamp(min=1e-8)  # (B, N, 1)
        w_norm = edge_w / deg                                    # row-normalise
        agg    = torch.bmm(w_norm, h)                           # (B, N, in_dim)
        out    = F.relu(self.norm(self.lin_self(h) + self.lin_neigh(agg)))
        if self.residual:
            out = out + h
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Physical Network GNN backbone
# ─────────────────────────────────────────────────────────────────────────────

class PhysicalNetworkGNN(nn.Module):
    """GraphSAGE backbone; returns per-node embeddings and a mean-pooled graph embedding."""

    def __init__(
        self,
        node_feat_dim: int = _NODE_FEAT_DIM,
        hidden_dim:    int = 128,
        num_layers:    int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            [SAGELayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        bw_adj:     torch.Tensor,
        node_mask:  torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # node_mask averages graph_emb over real nodes only, avoiding dilution from padding
        h = self.input_proj(node_feats)   # (B, N, hidden)
        for layer in self.layers:
            h = layer(h, bw_adj)
        if node_mask is not None:
            mask_f    = node_mask.unsqueeze(-1).float()          # (B, N, 1)
            graph_emb = (h * mask_f).sum(dim=1) \
                        / mask_f.sum(dim=1).clamp(min=1.0)       # (B, hidden)
        else:
            graph_emb = h.mean(dim=1)
        return h, graph_emb


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(sizes, act=nn.ReLU):
    """Build an MLP: Linear → act → … → Linear (no act on last layer)."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# GNN Actor-Critic
# ─────────────────────────────────────────────────────────────────────────────

class GNNActorCritic(nn.Module):
    """PPO actor-critic with GNN backbone. obs_dim is accepted for API parity but unused."""

    def __init__(
        self,
        obs_dim:        int,                          # API compat — not used
        spec:           ActionSpec = DEFAULT_SPEC,
        gnn_hidden:     int = 128,
        gnn_layers:     int = 3,
        model_hidden:   int = 64,
        context_hidden: int = 256,
    ):
        super().__init__()
        self.spec = spec
        mg  = spec.max_groups
        mc  = spec.max_chunks
        nn_ = spec.num_nodes   # MAX_NODES = 20
        nv  = spec.vertical_nvec
        ng  = spec.num_groups_nvec

        self._mg = mg
        self._mc = mc
        self._nn = nn_
        self._nv = nv
        self._ng = ng

        # ── Physical network encoder ──
        self.gnn = PhysicalNetworkGNN(
            node_feat_dim=_NODE_FEAT_DIM,
            hidden_dim=gnn_hidden,
            num_layers=gnn_layers,
        )

        # ── DNN model encoder ──
        self.model_enc = _mlp([_MODEL_FLAT_DIM, model_hidden, model_hidden], act=nn.ReLU)

        # ── Shared context ──
        # global (44) + gnn_emb (128) + model_emb (64) + prev_action (134) = 370
        ctx_in = _GLOBAL_DIM + gnn_hidden + model_hidden + PREV_ACTION_DIM
        self.context_mlp = _mlp([ctx_in, context_hidden, context_hidden], act=nn.ReLU)

        # ── Actor: global decision heads ──
        self.head_num_groups = nn.Linear(context_hidden, ng)
        self.head_vertical   = nn.Linear(context_hidden, mg * nv)
        self.head_horizontal = nn.Linear(context_hidden, mg * mc)

        # ── Actor: hub head (per-node point-wise scorer) ──
        self.hub_scorer = nn.Linear(gnn_hidden, 1)

        # ── Actor: assignment head (pointer network) ──
        # Learnable query per assignment slot; conditioned on global context.
        # score[role_r, node_i] = (role_emb[r] + ctx_proj) · node_emb[i]
        self.assignment_roles    = nn.Embedding(mg * mc, gnn_hidden)
        self.assignment_ctx_proj = nn.Linear(context_hidden, gnn_hidden, bias=False)

        # ── Critic ──
        self.critic_net = _mlp([context_hidden, context_hidden, 1], act=nn.ReLU)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init with small gain — prevents the initial policy from being
        near-deterministic, giving PPO entropy bonuses room to explore early on."""
        for head in [
            self.head_num_groups, self.head_vertical,
            self.head_horizontal, self.hub_scorer,
        ]:
            nn.init.orthogonal_(head.weight, gain=0.01)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.assignment_ctx_proj.weight, gain=0.01)
        nn.init.orthogonal_(self.assignment_roles.weight, gain=0.01)

    # ──────────────────────────────────────────────
    # Core forward: returns 5-tuple logits + context
    # ──────────────────────────────────────────────

    def _forward(self, obs: torch.Tensor):
        """Run the full GNN forward pass; returns 5 logit tensors and the context vector."""
        parsed = _parse_obs(obs)
        B      = parsed["global_feats"].shape[0]

        node_emb, graph_emb = self.gnn(
            parsed["node_feats"], parsed["bw_adj"], parsed["node_mask"]
        )

        # ── DNN model encoder ──
        model_emb = self.model_enc(parsed["model_flat"])   # (B, model_hidden)

        # ── Shared context ──
        ctx_in  = torch.cat(
            [parsed["global_feats"], graph_emb, model_emb, parsed["prev_act"]], dim=-1
        )
        context = self.context_mlp(ctx_in)                 # (B, context_hidden)

        # ── Global decision heads ──
        ng_logits = self.head_num_groups(context).reshape(B, 1, self._ng)   # (B, 1, ng)
        v_logits  = self.head_vertical(context).reshape(B, self._mg, self._nv)
        h_logits  = self.head_horizontal(context).reshape(B, self._mg, self._mc)

        # ── Hub head: per-node scorer ──
        hub_scores = self.hub_scorer(node_emb).squeeze(-1)  # (B, MAX_NODES)
        hub_logits = hub_scores.unsqueeze(1)                 # (B, 1, MAX_NODES)

        # ── Assignment head: pointer network ──
        ctx_proj   = self.assignment_ctx_proj(context)
        role_q     = self.assignment_roles.weight.unsqueeze(0) + ctx_proj.unsqueeze(1)
        a_logits   = torch.bmm(role_q, node_emb.transpose(-2, -1))   # (B, mg*mc, MAX_NODES)

        # ── Mask padded nodes to –∞ so they are never sampled ──
        node_mask = parsed["node_mask"]                     # (B, MAX_NODES) bool
        NEG_INF   = torch.finfo(hub_logits.dtype).min
        hub_logits = hub_logits.masked_fill(~node_mask.unsqueeze(1), NEG_INF)
        a_logits   = a_logits.masked_fill(~node_mask.unsqueeze(1), NEG_INF)

        return ng_logits, v_logits, h_logits, hub_logits, a_logits, context

    # ──────────────────────────────────────────────
    # Public interface (mirrors ActorCritic exactly)
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or greedily pick) an action; returns (action, log_prob, value)."""
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)

        ng_l, v_l, h_l, hub_l, a_l, context = self._forward(obs)

        if deterministic:
            ng_a  = ng_l.argmax(-1)    # (B, 1)
            v_a   = v_l.argmax(-1)     # (B, mg)
            h_a   = h_l.argmax(-1)     # (B, mg)
            hub_a = hub_l.argmax(-1)   # (B, 1)
            a_a   = a_l.argmax(-1)     # (B, mg*mc)
        else:
            ng_a  = Categorical(logits=ng_l).sample()
            v_a   = Categorical(logits=v_l).sample()
            h_a   = Categorical(logits=h_l).sample()
            hub_a = Categorical(logits=hub_l).sample()
            a_a   = Categorical(logits=a_l).sample()

        action = torch.cat([ng_a, v_a, h_a, hub_a, a_a], dim=-1)  # (B, total_dim)

        # Log-prob of the chosen action
        log_prob = (
            Categorical(logits=ng_l).log_prob(ng_a).sum(-1)
            + Categorical(logits=v_l).log_prob(v_a).sum(-1)
            + Categorical(logits=h_l).log_prob(h_a).sum(-1)
            + Categorical(logits=hub_l).log_prob(hub_a).sum(-1)
            + Categorical(logits=a_l).log_prob(a_a).sum(-1)
        )

        value = self.critic_net(context).squeeze(-1)   # (B,)

        if squeeze:
            return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions for PPO loss; returns (log_prob, entropy, value)."""
        ng_l, v_l, h_l, hub_l, a_l, context = self._forward(obs)
        mg = self._mg

        # action layout: [ng(1) | vert(mg) | horiz(mg) | hub(1) | assign(mg*mc)]
        ng_a  = action[:, 0:1].long()
        v_a   = action[:, 1:1 + mg].long()
        h_a   = action[:, 1 + mg:1 + 2 * mg].long()
        hub_a = action[:, 1 + 2 * mg:2 + 2 * mg].long()
        a_a   = action[:, 2 + 2 * mg:].long()

        ng_d  = Categorical(logits=ng_l)
        v_d   = Categorical(logits=v_l)
        h_d   = Categorical(logits=h_l)
        hub_d = Categorical(logits=hub_l)
        a_d   = Categorical(logits=a_l)

        log_prob = (
            ng_d.log_prob(ng_a).sum(-1)
            + v_d.log_prob(v_a).sum(-1)
            + h_d.log_prob(h_a).sum(-1)
            + hub_d.log_prob(hub_a).sum(-1)
            + a_d.log_prob(a_a).sum(-1)
        )
        entropy = (
            ng_d.entropy().sum(-1)
            + v_d.entropy().sum(-1)
            + h_d.entropy().sum(-1)
            + hub_d.entropy().sum(-1)
            + a_d.entropy().sum(-1)
        )
        value = self.critic_net(context).squeeze(-1)
        return log_prob, entropy, value


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(obs_dim: int, spec: ActionSpec) -> nn.Module:
    """Instantiate the GNN actor-critic."""
    return GNNActorCritic(obs_dim, spec)


if __name__ == "__main__":
    from state_encoder import state_dim

    spec   = ActionSpec(num_nodes=MAX_NODES, max_chunks=MAX_NODES)
    obs_d  = state_dim()
    gnn_ac = GNNActorCritic(obs_d, spec)

    total_params = sum(p.numel() for p in gnn_ac.parameters() if p.requires_grad)
    print(f"GNNActorCritic | obs_dim={obs_d} | trainable params={total_params:,}")

    # Single-sample inference
    dummy = torch.randn(obs_d)
    action, lp, val = gnn_ac.act(dummy)
    print(f"act()  → action_dim={action.shape[-1]}  log_prob={lp.item():.3f}  "
          f"value={val.item():.3f}")

    # Batched evaluate (simulates PPO update)
    B = 16
    obs_b   = torch.randn(B, obs_d)
    act_b   = torch.stack([gnn_ac.act(obs_b[i])[0] for i in range(B)])
    lp_b, ent_b, val_b = gnn_ac.evaluate(obs_b, act_b)
    print(f"evaluate() → log_prob mean={lp_b.mean():.3f}  "
          f"entropy mean={ent_b.mean():.3f}  value mean={val_b.mean():.3f}")

    print(f"\nbuild_model() factory → {type(build_model(obs_d, spec)).__name__}")
