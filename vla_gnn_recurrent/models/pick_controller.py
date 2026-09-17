from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from vla_gnn_recurrent.graph.types import GraphData
from vla_gnn_recurrent.models.gnn import EdgeAwareGNN
from vla_gnn_recurrent.utils import clamp_delta

PickActionHeadType = Literal["vector", "direction_magnitude"]


@dataclass(frozen=True)
class PickAction:
    """Policy output for the planar MuJoCo pick stack."""

    delta_ee: torch.Tensor
    gripper: torch.Tensor
    raw: torch.Tensor
    magnitude: torch.Tensor


class PickRecurrentController(nn.Module):
    """Graph_t -> edge-aware GNN -> 2-layer GRU -> [Delta x, Delta z, gripper]."""

    def __init__(
        self,
        gnn_hidden_dim: int = 128,
        gru_hidden_dim: int = 256,
        gru_layers: int = 2,
        action_hidden_dim: int = 128,
        max_step: float = 0.045,
        action_head_type: PickActionHeadType = "vector",
    ) -> None:
        super().__init__()
        self.max_step = float(max_step)
        self.action_head_type = action_head_type
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
            nn.Linear(action_hidden_dim, _raw_action_dim(action_head_type)),
        )
        self._init_action_head()

    def initial_hidden(self, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(self.gru_layers, 1, self.gru_hidden_dim, device=device)

    def raw_action(self, graph: GraphData, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        graph_embedding = self.gnn(graph).unsqueeze(1)  # [1, 1, 128]
        output, next_hidden = self.gru(graph_embedding, hidden)
        return self.action_head(output[:, -1, :]).squeeze(0), next_hidden

    def forward(self, graph: GraphData, hidden: torch.Tensor | None = None) -> tuple[PickAction, torch.Tensor]:
        raw, next_hidden = self.raw_action(graph, hidden)
        return decode_pick_action(raw, self.max_step, self.action_head_type), next_hidden

    def _init_action_head(self) -> None:
        final = self.action_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)


class PickFeedForwardController(nn.Module):
    """Graph -> edge-aware GNN -> MLP -> [Delta x, Delta z, gripper]."""

    def __init__(
        self,
        gnn_hidden_dim: int = 128,
        action_hidden_dim: int = 128,
        max_step: float = 0.045,
        action_head_type: PickActionHeadType = "vector",
    ) -> None:
        super().__init__()
        self.max_step = float(max_step)
        self.action_head_type = action_head_type
        self.gnn = EdgeAwareGNN(hidden_dim=gnn_hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, action_hidden_dim),
            nn.SiLU(),
            nn.Linear(action_hidden_dim, _raw_action_dim(action_head_type)),
        )
        self._init_action_head()

    def raw_action(self, graph: GraphData) -> torch.Tensor:
        return self.action_head(self.gnn(graph)).squeeze(0)

    def forward(self, graph: GraphData) -> PickAction:
        return decode_pick_action(self.raw_action(graph), self.max_step, self.action_head_type)

    def _init_action_head(self) -> None:
        final = self.action_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)


def decode_pick_action(
    raw_action: torch.Tensor,
    max_step: float,
    action_head_type: PickActionHeadType = "vector",
) -> PickAction:
    """Decode raw action into a bounded planar Delta EE and [0, 1] gripper."""

    if action_head_type == "vector":
        delta_xz = clamp_delta(raw_action[..., 0:2], max_step)
        gripper_logit = raw_action[..., 2]
    elif action_head_type == "direction_magnitude":
        direction_raw = raw_action[..., 0:2]
        direction_norm = direction_raw.norm(dim=-1, keepdim=True)
        direction = direction_raw / direction_norm.clamp_min(1e-8)
        magnitude = torch.sigmoid(raw_action[..., 2:3]) * max_step
        delta_xz = direction * magnitude
        gripper_logit = raw_action[..., 3]
    else:
        raise ValueError(f"Unknown pick action head type: {action_head_type}")
    zeros = torch.zeros_like(delta_xz[..., 0:1])
    delta_ee = torch.cat([delta_xz[..., 0:1], zeros, delta_xz[..., 1:2]], dim=-1)
    gripper = torch.sigmoid(gripper_logit)
    return PickAction(delta_ee=delta_ee, gripper=gripper, raw=raw_action, magnitude=delta_xz.norm(dim=-1))


def gripper_logit(raw_action: torch.Tensor, action_head_type: PickActionHeadType) -> torch.Tensor:
    if action_head_type == "vector":
        return raw_action[..., 2]
    if action_head_type == "direction_magnitude":
        return raw_action[..., 3]
    raise ValueError(f"Unknown pick action head type: {action_head_type}")


def _raw_action_dim(action_head_type: PickActionHeadType) -> int:
    if action_head_type == "vector":
        return 3
    if action_head_type == "direction_magnitude":
        return 4
    raise ValueError(f"Unknown pick action head type: {action_head_type}")
