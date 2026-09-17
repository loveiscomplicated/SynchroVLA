from __future__ import annotations

from dataclasses import dataclass

import torch

NODE_TYPES = ("end_effector", "object", "task", "joint", "gripper", "destination")
EDGE_TYPES = ("spatial", "task", "kinematic", "interaction")
TASK_TYPES = ("reach", "pick", "pick_and_place")

NODE_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(NODE_TYPES)}
EDGE_TYPE_TO_INDEX = {name: idx for idx, name in enumerate(EDGE_TYPES)}

# xyz position + xyz velocity + node type one-hot + task one-hot + two generic
# physical-state scalars. Manipulation graphs use the auxiliary scalars for
# gripper opening/contact state; synthetic reaching graphs leave them zero.
NODE_AUX_DIM = 2
NODE_FEATURE_DIM = 3 + 3 + len(NODE_TYPES) + len(TASK_TYPES) + NODE_AUX_DIM
EDGE_FEATURE_DIM = 3 + 1 + len(EDGE_TYPES) + 1


@dataclass(frozen=True)
class ObjectObservation:
    """Perception-facing object contract used by the synthetic graph builder.

    A future perception stack should produce this shape of state without
    requiring controller-side changes.
    """

    object_id: str
    position: torch.Tensor
    orientation: torch.Tensor | None = None
    velocity: torch.Tensor | None = None
    is_target: bool = True
    confidence: float = 1.0

    @classmethod
    def from_position(
        cls,
        object_id: str,
        position: torch.Tensor | list[float] | tuple[float, float, float],
        orientation: torch.Tensor | list[float] | tuple[float, float, float, float] | None = None,
        velocity: torch.Tensor | list[float] | tuple[float, float, float] | None = None,
        is_target: bool = True,
        confidence: float = 1.0,
    ) -> "ObjectObservation":
        pos = torch.as_tensor(position, dtype=torch.float32).reshape(3)
        quat = None if orientation is None else torch.as_tensor(orientation, dtype=torch.float32).reshape(4)
        vel = None if velocity is None else torch.as_tensor(velocity, dtype=torch.float32).reshape(3)
        return cls(
            object_id=object_id,
            position=pos,
            orientation=quat,
            velocity=vel,
            is_target=is_target,
            confidence=float(confidence),
        )


@dataclass(frozen=True)
class DestinationObservation:
    """Perception-facing destination/goal contract for manipulation tasks."""

    destination_id: str
    position: torch.Tensor
    orientation: torch.Tensor | None = None
    velocity: torch.Tensor | None = None
    confidence: float = 1.0

    @property
    def object_id(self) -> str:
        return self.destination_id

    @classmethod
    def from_position(
        cls,
        destination_id: str,
        position: torch.Tensor | list[float] | tuple[float, float, float],
        orientation: torch.Tensor | list[float] | tuple[float, float, float, float] | None = None,
        velocity: torch.Tensor | list[float] | tuple[float, float, float] | None = None,
        confidence: float = 1.0,
    ) -> "DestinationObservation":
        pos = torch.as_tensor(position, dtype=torch.float32).reshape(3)
        quat = None if orientation is None else torch.as_tensor(orientation, dtype=torch.float32).reshape(4)
        vel = None if velocity is None else torch.as_tensor(velocity, dtype=torch.float32).reshape(3)
        return cls(
            destination_id=destination_id,
            position=pos,
            orientation=quat,
            velocity=vel,
            confidence=float(confidence),
        )


@dataclass(frozen=True)
class GraphData:
    """Single-scene directed graph consumed by the GNN.

    node_features: [num_nodes, NODE_FEATURE_DIM]
    edge_index: [2, num_edges], row 0 = source, row 1 = destination
    edge_features: [num_edges, EDGE_FEATURE_DIM]
    """

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    node_types: tuple[str, ...]
    node_names: tuple[str, ...]
    ee_node_index: int
    target_node_index: int
    task_node_index: int

    def to(self, device: torch.device | str) -> "GraphData":
        return GraphData(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            node_types=self.node_types,
            node_names=self.node_names,
            ee_node_index=self.ee_node_index,
            target_node_index=self.target_node_index,
            task_node_index=self.task_node_index,
        )

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_features.shape[0])
