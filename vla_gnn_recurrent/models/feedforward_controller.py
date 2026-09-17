from __future__ import annotations

import torch
from torch import nn

from vla_gnn_recurrent.graph.types import GraphData
from vla_gnn_recurrent.models.actions import ActionPrior, compose_action
from vla_gnn_recurrent.models.gnn import EdgeAwareGNN


class FeedForwardController(nn.Module):
    """Graph -> GNN -> MLP -> bounded Delta EE baseline."""

    def __init__(
        self,
        gnn_hidden_dim: int = 128,
        action_hidden_dim: int = 128,
        max_step: float = 0.08,
        action_prior: ActionPrior = "none",
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.max_step = max_step
        self.action_prior = action_prior
        self.residual_scale = residual_scale
        self.gnn = EdgeAwareGNN(hidden_dim=gnn_hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, action_hidden_dim),
            nn.SiLU(),
            nn.Linear(action_hidden_dim, 3),
        )
        self._init_action_head()

    def forward(self, graph: GraphData) -> torch.Tensor:
        # graph_embedding: [1, 128], action: [3]
        raw_action = self.raw_action(graph).unsqueeze(0)
        return compose_action(raw_action, graph, self.max_step, self.action_prior, self.residual_scale)

    def raw_action(self, graph: GraphData) -> torch.Tensor:
        graph_embedding = self.gnn(graph)
        return self.action_head(graph_embedding).squeeze(0)

    def _init_action_head(self) -> None:
        final = self.action_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
