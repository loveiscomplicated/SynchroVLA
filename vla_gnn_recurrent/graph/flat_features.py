from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch

from vla_gnn_recurrent.graph.types import GraphData


EXPECTED_PICK_PLACE_NODE_NAMES = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "ee",
    "gripper",
    "ball",
    "destination",
    "task:pick_and_place",
)


@dataclass(frozen=True)
class FlatStateSchema:
    """Deterministic flat-state view of the manipulation graph.

    The flat vector receives the same physical graph tensors as the GNN, but
    without message-passing structure:

    [node_features in fixed node order, edge_features in builder edge order]
    """

    node_names: tuple[str, ...]
    node_feature_dim: int
    num_nodes: int
    edge_feature_dim: int
    num_edges: int
    input_dim: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def flat_state_from_graph(graph: GraphData, validate_pick_place: bool = True) -> torch.Tensor:
    if validate_pick_place and graph.node_names != EXPECTED_PICK_PLACE_NODE_NAMES:
        raise ValueError(f"Unexpected pick-place node order: {graph.node_names}")
    return torch.cat(
        [
            graph.node_features.reshape(-1),
            graph.edge_features.reshape(-1),
        ],
        dim=0,
    ).float()


def flat_state_schema_from_graph(graph: GraphData, validate_pick_place: bool = True) -> FlatStateSchema:
    if validate_pick_place and graph.node_names != EXPECTED_PICK_PLACE_NODE_NAMES:
        raise ValueError(f"Unexpected pick-place node order: {graph.node_names}")
    node_dim = int(graph.node_features.shape[-1])
    edge_dim = int(graph.edge_features.shape[-1])
    num_nodes = int(graph.node_features.shape[0])
    num_edges = int(graph.edge_features.shape[0])
    return FlatStateSchema(
        node_names=tuple(graph.node_names),
        node_feature_dim=node_dim,
        num_nodes=num_nodes,
        edge_feature_dim=edge_dim,
        num_edges=num_edges,
        input_dim=num_nodes * node_dim + num_edges * edge_dim,
    )


def flat_state_dim_from_graph(graph: GraphData) -> int:
    return flat_state_schema_from_graph(graph).input_dim
