from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GateSummary:
    gate_mean: float
    gate_std: float
    per_layer_gate_mean: list[float]
    per_layer_gate_std: list[float]


class StateCorrector(nn.Module):
    """Lightweight event-time correction for a multi-layer GRU hidden state.

    The recurrent controller's hidden state has shape [num_layers, batch, hidden_dim].
    This module applies the same MLP parameters independently to every GRU layer:

        candidate, gate = f([h_old[layer], z_new])
        h_corrected = gate * h_old + (1 - gate) * candidate

    graph_embedding is expected to be [batch, graph_dim].
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        graph_dim: int = 128,
        num_layers: int = 2,
        mlp_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.graph_dim = graph_dim
        self.num_layers = num_layers
        input_dim = hidden_dim + graph_dim
        self.candidate_mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor, graph_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return corrected hidden state and gate tensor.

        hidden: [num_layers, batch, hidden_dim]
        graph_embedding: [batch, graph_dim]
        corrected/gate: [num_layers, batch, hidden_dim]
        """

        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden [layers, batch, hidden_dim], got {tuple(hidden.shape)}")
        if graph_embedding.ndim != 2:
            raise ValueError(f"Expected graph_embedding [batch, graph_dim], got {tuple(graph_embedding.shape)}")
        layers, batch, hidden_dim = hidden.shape
        if layers != self.num_layers or hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden [{self.num_layers}, batch, {self.hidden_dim}], got {tuple(hidden.shape)}"
            )
        if graph_embedding.shape[0] != batch or graph_embedding.shape[1] != self.graph_dim:
            raise ValueError(
                f"Expected graph_embedding [{batch}, {self.graph_dim}], got {tuple(graph_embedding.shape)}"
            )

        expanded_graph = graph_embedding.unsqueeze(0).expand(layers, batch, self.graph_dim)
        inputs = torch.cat([hidden, expanded_graph], dim=-1).reshape(layers * batch, self.hidden_dim + self.graph_dim)
        candidate = self.candidate_mlp(inputs).reshape(layers, batch, self.hidden_dim)
        gate = torch.sigmoid(self.gate_mlp(inputs)).reshape(layers, batch, self.hidden_dim)
        corrected = gate * hidden + (1.0 - gate) * candidate
        return corrected, gate

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def summarize_gate(gate: torch.Tensor) -> GateSummary:
    detached = gate.detach().float().cpu()
    per_layer_mean = detached.mean(dim=(1, 2))
    per_layer_std = detached.std(dim=(1, 2), unbiased=False)
    return GateSummary(
        gate_mean=float(detached.mean().item()),
        gate_std=float(detached.std(unbiased=False).item()),
        per_layer_gate_mean=[float(value.item()) for value in per_layer_mean],
        per_layer_gate_std=[float(value.item()) for value in per_layer_std],
    )
