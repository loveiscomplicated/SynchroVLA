from __future__ import annotations

import torch
from torch import nn

from vla_gnn_recurrent.utils import clamp_delta


class FlatStateController(nn.Module):
    """Flat-state feed-forward baseline: state vector -> bounded Delta EE."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, max_step: float = 0.035) -> None:
        super().__init__()
        self.max_step = max_step
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return clamp_delta(self.raw_action(state), self.max_step).squeeze(0)

    def raw_action(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.net(state).squeeze(0)
