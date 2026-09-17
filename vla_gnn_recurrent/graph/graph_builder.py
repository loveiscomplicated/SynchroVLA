from __future__ import annotations

from dataclasses import dataclass

import torch

from vla_gnn_recurrent.graph.types import (
    EDGE_FEATURE_DIM,
    EDGE_TYPE_TO_INDEX,
    NODE_AUX_DIM,
    NODE_FEATURE_DIM,
    TASK_TYPES,
    NODE_TYPE_TO_INDEX,
    GraphData,
    ObjectObservation,
)
from vla_gnn_recurrent.utils import tensor3


@dataclass
class GraphBuilder:
    """Build typed interaction graphs from explicit scene state."""

    include_joints: bool = False
    num_joints: int = 2
    target_velocity_feature: bool = True

    def build(
        self,
        ee_position: torch.Tensor | list[float] | tuple[float, float, float],
        target: ObjectObservation,
        ee_velocity: torch.Tensor | list[float] | tuple[float, float, float] | None = None,
    ) -> GraphData:
        ee_pos = tensor3(ee_position)
        ee_vel = torch.zeros(3) if ee_velocity is None else tensor3(ee_velocity)
        obj_pos = tensor3(target.position)
        obj_vel = (
            torch.zeros(3)
            if target.velocity is None or not self.target_velocity_feature
            else tensor3(target.velocity)
        )

        node_positions: list[torch.Tensor] = [ee_pos, obj_pos, obj_pos.clone()]
        node_velocities: list[torch.Tensor] = [ee_vel, obj_vel, torch.zeros(3)]
        node_types: list[str] = ["end_effector", "object", "task"]
        node_names: list[str] = ["ee", target.object_id, "task:reach"]
        task_features: list[torch.Tensor] = [
            self._task_feature(None),
            self._task_feature("reach" if target.is_target else None),
            self._task_feature("reach"),
        ]

        if self.include_joints:
            for idx in range(self.num_joints):
                alpha = float(idx + 1) / float(self.num_joints + 1)
                node_positions.append(ee_pos * alpha)
                node_velocities.append(ee_vel * alpha)
                node_types.append("joint")
                node_names.append(f"joint_{idx}")
                task_features.append(self._task_feature(None))

        node_features = torch.stack(
            [
                self._node_feature(pos, vel, node_type, task_feature)
                for pos, vel, node_type, task_feature in zip(
                    node_positions, node_velocities, node_types, task_features, strict=True
                )
            ],
            dim=0,
        )

        edge_pairs: list[tuple[int, int, str]] = []
        ee_idx = 0
        obj_idx = 1
        task_idx = 2

        edge_pairs.extend(
            [
                (ee_idx, obj_idx, "spatial"),
                (obj_idx, ee_idx, "spatial"),
                (task_idx, ee_idx, "task"),
                (ee_idx, task_idx, "task"),
                (task_idx, obj_idx, "task"),
                (obj_idx, task_idx, "task"),
            ]
        )

        if self.include_joints:
            joint_indices = list(range(3, 3 + self.num_joints))
            chain = joint_indices + [ee_idx]
            for src, dst in zip(chain[:-1], chain[1:], strict=True):
                edge_pairs.append((src, dst, "kinematic"))
                edge_pairs.append((dst, src, "kinematic"))

        edge_index = torch.tensor([[src, dst] for src, dst, _ in edge_pairs], dtype=torch.long).t()
        edge_features = torch.stack(
            [
                self._edge_feature(node_positions[src], node_positions[dst], edge_type)
                for src, dst, edge_type in edge_pairs
            ],
            dim=0,
        )

        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_types=tuple(node_types),
            node_names=tuple(node_names),
            ee_node_index=ee_idx,
            target_node_index=obj_idx,
            task_node_index=task_idx,
        )

    @staticmethod
    def _node_feature(
        position: torch.Tensor,
        velocity: torch.Tensor,
        node_type: str,
        task_feature: torch.Tensor,
        aux_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature = torch.zeros(NODE_FEATURE_DIM, dtype=torch.float32)
        feature[0:3] = position
        feature[3:6] = velocity
        feature[6 + NODE_TYPE_TO_INDEX[node_type]] = 1.0
        task_start = 6 + len(NODE_TYPE_TO_INDEX)
        feature[task_start : task_start + len(TASK_TYPES)] = task_feature
        if aux_feature is not None:
            feature[-NODE_AUX_DIM:] = aux_feature.reshape(NODE_AUX_DIM)
        return feature

    @staticmethod
    def _edge_feature(
        src_position: torch.Tensor,
        dst_position: torch.Tensor,
        edge_type: str,
        contact_state: float = 0.0,
    ) -> torch.Tensor:
        feature = torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32)
        relative = dst_position - src_position
        feature[0:3] = relative
        feature[3] = relative.norm()
        feature[4 + EDGE_TYPE_TO_INDEX[edge_type]] = 1.0
        feature[-1] = float(contact_state)
        return feature

    @staticmethod
    def _task_feature(task: str | None) -> torch.Tensor:
        feature = torch.zeros(len(TASK_TYPES), dtype=torch.float32)
        if task is not None:
            feature[TASK_TYPES.index(task)] = 1.0
        return feature
