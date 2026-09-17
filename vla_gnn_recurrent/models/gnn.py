from __future__ import annotations

import torch
from torch import nn

from vla_gnn_recurrent.graph.types import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, GraphData


class EdgeMessagePassingLayer(nn.Module):
    """One edge-aware message passing block."""

    def __init__(self, hidden_dim: int, edge_feature_dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_state: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        # node_state: [N, H], edge_index: [2, E], edge_features: [E, F_e]
        src = edge_index[0]
        dst = edge_index[1]
        source_state = node_state.index_select(0, src)
        messages = self.message_mlp(torch.cat([source_state, edge_features], dim=-1))

        aggregated = torch.zeros_like(node_state)
        aggregated.index_add_(0, dst, messages)

        degree = torch.zeros(node_state.shape[0], 1, device=node_state.device, dtype=node_state.dtype)
        degree.index_add_(0, dst, torch.ones(messages.shape[0], 1, device=node_state.device, dtype=node_state.dtype))
        aggregated = aggregated / degree.clamp_min(1.0)

        update = self.update_mlp(torch.cat([node_state, aggregated], dim=-1))
        return self.norm(node_state + update)


class EdgeAwareGNN(nn.Module):
    """3-layer edge-aware GNN with an EE/object/global readout."""

    def __init__(
        self,
        node_feature_dim: int = NODE_FEATURE_DIM,
        edge_feature_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [EdgeMessagePassingLayer(hidden_dim=hidden_dim, edge_feature_dim=edge_feature_dim) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.geometry_readout = nn.Linear(4, hidden_dim)

    def encode_nodes(self, graph: GraphData) -> torch.Tensor:
        # Returns node embeddings: [N, H]
        node_state = self.node_encoder(graph.node_features)
        for layer in self.layers:
            node_state = layer(node_state, graph.edge_index, graph.edge_features)
        return node_state

    def forward(self, graph: GraphData) -> torch.Tensor:
        # Returns a single graph/control embedding: [1, H]
        node_state = self.encode_nodes(graph)
        ee_state = node_state[graph.ee_node_index]
        target_state = node_state[graph.target_node_index]
        global_state = node_state.mean(dim=0)
        relative = graph.node_features[graph.target_node_index, 0:3] - graph.node_features[graph.ee_node_index, 0:3]
        distance = relative.norm().reshape(1)
        control_state = torch.cat([ee_state, target_state, global_state], dim=-1)
        geometry_state = torch.cat([relative, distance], dim=-1)
        return (self.readout(control_state) + self.geometry_readout(geometry_state)).unsqueeze(0)
