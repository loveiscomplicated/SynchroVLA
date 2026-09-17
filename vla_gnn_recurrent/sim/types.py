from __future__ import annotations

from dataclasses import dataclass

import torch

from vla_gnn_recurrent.graph.types import DestinationObservation, ObjectObservation


@dataclass(frozen=True)
class RobotObservation:
    """Simulator-independent robot state consumed by graph/adapters."""

    joint_positions: torch.Tensor
    joint_velocities: torch.Tensor
    joint_world_positions: torch.Tensor
    ee_position: torch.Tensor
    ee_orientation: torch.Tensor
    ee_velocity: torch.Tensor
    gripper_state: torch.Tensor
    gripper_position: torch.Tensor


@dataclass(frozen=True)
class ManipulationObservation:
    """Controller-facing manipulation observation plus true state for metrics."""

    robot: RobotObservation
    target: ObjectObservation
    true_target: ObjectObservation
    object: ObjectObservation | None
    true_object: ObjectObservation | None
    destination: DestinationObservation | ObjectObservation | None
    true_destination: DestinationObservation | ObjectObservation | None
    step_count: int
    reanchored: bool
    observable_event: bool = False
    grasp: GraspObservation | None = None


@dataclass(frozen=True)
class IKResult:
    success: bool
    qpos: torch.Tensor
    final_error: float
    iterations: int
    clipped: bool = False


@dataclass(frozen=True)
class GraspObservation:
    left_contact: bool
    right_contact: bool
    contact_force: float
    object_grasped: bool
    object_height: float
    object_velocity: torch.Tensor
    num_contacts: int
