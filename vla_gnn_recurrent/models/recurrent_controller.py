from __future__ import annotations

import torch
from torch import nn

from vla_gnn_recurrent.graph.types import GraphData
from vla_gnn_recurrent.models.actions import ActionPrior, compose_action
from vla_gnn_recurrent.models.gnn import EdgeAwareGNN


class RecurrentController(nn.Module):
    """Graph_t -> GNN -> 2-layer GRU(h_t-1) -> bounded Delta EE."""

    def __init__(
        self,
        gnn_hidden_dim: int = 128,
        gru_hidden_dim: int = 256,
        gru_layers: int = 2,
        action_hidden_dim: int = 128,
        max_step: float = 0.08,
        action_prior: ActionPrior = "none",
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.max_step = max_step
        self.action_prior = action_prior
        self.residual_scale = residual_scale
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_layers = gru_layers
        self.gnn = EdgeAwareGNN(hidden_dim=gnn_hidden_dim)
        self.gru = nn.GRU(
            input_size=gnn_hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.action_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, action_hidden_dim),
            nn.SiLU(),
            nn.Linear(action_hidden_dim, 3),
        )
        self._init_action_head()

    def initial_hidden(self, device: torch.device | str) -> torch.Tensor:
        # [num_layers, batch=1, hidden_dim]
        return torch.zeros(self.gru_layers, 1, self.gru_hidden_dim, device=device)

    def forward(self, graph: GraphData, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # graph_embedding: [1, 128] -> sequence input: [1, 1, 128]
        raw_action, next_hidden = self.raw_action(graph, hidden)
        action = compose_action(raw_action.unsqueeze(0), graph, self.max_step, self.action_prior, self.residual_scale)
        return action, next_hidden

    def raw_action(self, graph: GraphData, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        graph_embedding = self.gnn(graph).unsqueeze(1)
        output, next_hidden = self.gru(graph_embedding, hidden)
        raw_action = self.action_head(output[:, -1, :]).squeeze(0)
        return raw_action, next_hidden

    def _init_action_head(self) -> None:
        final = self.action_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
