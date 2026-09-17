from __future__ import annotations

from dataclasses import dataclass

import torch

from vla_gnn_recurrent.graph.graph_builder import GraphBuilder
from vla_gnn_recurrent.graph.types import GraphData, NODE_AUX_DIM, TASK_TYPES
from vla_gnn_recurrent.sim.types import ManipulationObservation


@dataclass
class ManipulationGraphBuilder:
    """Build relational manipulation graphs from simulator/perception-style observations."""

    task: str = "reach"
    include_object: bool = False

    def build(self, observation: ManipulationObservation) -> GraphData:
        robot = observation.robot
        positions: list[torch.Tensor] = []
        velocities: list[torch.Tensor] = []
        node_types: list[str] = []
        node_names: list[str] = []
        task_features: list[torch.Tensor] = []
        aux_features: list[torch.Tensor] = []

        joint_indices: list[int] = []
        for idx in range(robot.joint_world_positions.shape[0]):
            joint_indices.append(len(positions))
            positions.append(robot.joint_world_positions[idx].float())
            velocities.append(torch.zeros(3))
            node_types.append("joint")
            node_names.append(f"joint_{idx}")
            task_features.append(_task_feature(None))
            aux_features.append(_aux_feature())

        ee_idx = len(positions)
        positions.append(robot.ee_position.float())
        velocities.append(robot.ee_velocity.float())
        node_types.append("end_effector")
        node_names.append("ee")
        task_features.append(_task_feature(self.task))
        aux_features.append(_aux_feature())

        gripper_idx = len(positions)
        positions.append(robot.gripper_position.float())
        velocities.append(robot.ee_velocity.float())
        node_types.append("gripper")
        node_names.append("gripper")
        task_features.append(_task_feature(None))
        gripper_opening = float(robot.gripper_state.reshape(-1)[0].item())
        aux_features.append(_aux_feature(gripper_opening, _contact_scalar(observation)))

        object_idx: int | None = None
        if self.include_object and observation.object is not None:
            object_idx = len(positions)
            positions.append(observation.object.position.float())
            velocities.append(_velocity_or_zero(observation.object.velocity))
            node_types.append("object")
            node_names.append(observation.object.object_id)
            task_features.append(_task_feature(self.task if self.task in {"pick", "pick_and_place"} else None))
            aux_features.append(_aux_feature(_grasped_scalar(observation), _contact_scalar(observation)))

        destination = observation.destination or observation.target
        destination_idx = len(positions)
        positions.append(destination.position.float())
        velocities.append(_velocity_or_zero(destination.velocity))
        node_types.append("destination")
        node_names.append(destination.object_id)
        task_features.append(_task_feature(self.task))
        aux_features.append(_aux_feature())

        task_idx = len(positions)
        positions.append(destination.position.float())
        velocities.append(torch.zeros(3))
        node_types.append("task")
        node_names.append(f"task:{self.task}")
        task_features.append(_task_feature(self.task))
        aux_features.append(_aux_feature())

        node_features = torch.stack(
            [
                GraphBuilder._node_feature(position, velocity, node_type, task_feature, aux_feature)
                for position, velocity, node_type, task_feature, aux_feature in zip(
                    positions, velocities, node_types, task_features, aux_features, strict=True
                )
            ],
            dim=0,
        )

        edge_pairs: list[tuple[int, int, str, float]] = []
        for src, dst in zip(joint_indices[:-1], joint_indices[1:], strict=True):
            _add_bidirectional(edge_pairs, src, dst, "kinematic")
        if joint_indices:
            _add_bidirectional(edge_pairs, joint_indices[-1], ee_idx, "kinematic")
        _add_bidirectional(edge_pairs, ee_idx, gripper_idx, "kinematic")
        _add_bidirectional(edge_pairs, ee_idx, destination_idx, "spatial")
        if object_idx is not None:
            _add_bidirectional(edge_pairs, ee_idx, object_idx, "spatial")
            _add_bidirectional(edge_pairs, object_idx, destination_idx, "spatial")
            contact = _contact_scalar(observation)
            _add_bidirectional(edge_pairs, gripper_idx, object_idx, "interaction", contact)
            _add_bidirectional(edge_pairs, task_idx, object_idx, "task")
        _add_bidirectional(edge_pairs, task_idx, destination_idx, "task")
        _add_bidirectional(edge_pairs, task_idx, ee_idx, "task")

        edge_index = torch.tensor([[src, dst] for src, dst, _, _ in edge_pairs], dtype=torch.long).t()
        edge_features = torch.stack(
            [
                GraphBuilder._edge_feature(positions[src], positions[dst], edge_type, contact)
                for src, dst, edge_type, contact in edge_pairs
            ],
            dim=0,
        )

        control_target_idx = object_idx if self.task == "pick" and object_idx is not None else destination_idx
        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_types=tuple(node_types),
            node_names=tuple(node_names),
            ee_node_index=ee_idx,
            target_node_index=control_target_idx,
            task_node_index=task_idx,
        )


def _add_bidirectional(
    edges: list[tuple[int, int, str, float]],
    a: int,
    b: int,
    edge_type: str,
    contact: float = 0.0,
) -> None:
    edges.append((a, b, edge_type, contact))
    edges.append((b, a, edge_type, contact))


def _velocity_or_zero(velocity: torch.Tensor | None) -> torch.Tensor:
    return torch.zeros(3) if velocity is None else velocity.float()


def _task_feature(task: str | None) -> torch.Tensor:
    feature = torch.zeros(len(TASK_TYPES), dtype=torch.float32)
    if task is not None:
        feature[TASK_TYPES.index(task)] = 1.0
    return feature


def _aux_feature(first: float = 0.0, second: float = 0.0) -> torch.Tensor:
    return torch.tensor([first, second], dtype=torch.float32).reshape(NODE_AUX_DIM)


def _contact_scalar(observation: ManipulationObservation) -> float:
    grasp = observation.grasp
    if grasp is None:
        return 0.0
    return float(grasp.left_contact or grasp.right_contact)


def _grasped_scalar(observation: ManipulationObservation) -> float:
    grasp = observation.grasp
    if grasp is None:
        return 0.0
    return float(grasp.object_grasped)
