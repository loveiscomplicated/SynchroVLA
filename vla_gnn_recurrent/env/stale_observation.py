from __future__ import annotations

from dataclasses import dataclass

import torch

from vla_gnn_recurrent.env.reaching_env import ReachingObservation
from vla_gnn_recurrent.graph.types import ObjectObservation


@dataclass(frozen=True)
class ControllerObservation:
    """Controller-facing state plus ground-truth state for metrics."""

    ee_position: torch.Tensor
    ee_velocity: torch.Tensor
    observed_target: ObjectObservation
    true_target: ObjectObservation
    step_count: int
    reanchored: bool


class StaleTargetObserver:
    """Hold object observations stale between visual/object re-anchors."""

    def __init__(self, observation_interval: int = 1, target_velocity_feature: bool = False) -> None:
        if observation_interval < 1:
            raise ValueError("observation_interval must be >= 1")
        self.observation_interval = observation_interval
        self.target_velocity_feature = target_velocity_feature
        self._last_position: torch.Tensor | None = None
        self._last_velocity = torch.zeros(3)
        self._last_anchor_step = 0

    def reset(self) -> None:
        self._last_position = None
        self._last_velocity = torch.zeros(3)
        self._last_anchor_step = 0

    def observe(self, true_observation: ReachingObservation) -> ControllerObservation:
        reanchored = self._should_reanchor(true_observation.step_count)
        if reanchored:
            self._update_anchor(true_observation)

        if self._last_position is None:
            raise RuntimeError("StaleTargetObserver failed to initialize an observed target.")

        velocity = self._last_velocity.clone() if self.target_velocity_feature else torch.zeros(3)
        observed_target = ObjectObservation.from_position(
            object_id=true_observation.target.object_id,
            position=self._last_position.clone(),
            velocity=velocity,
            is_target=true_observation.target.is_target,
        )
        return ControllerObservation(
            ee_position=true_observation.ee_position.clone(),
            ee_velocity=true_observation.ee_velocity.clone(),
            observed_target=observed_target,
            true_target=true_observation.target,
            step_count=true_observation.step_count,
            reanchored=reanchored,
        )

    def _should_reanchor(self, step_count: int) -> bool:
        return self._last_position is None or step_count % self.observation_interval == 0

    def _update_anchor(self, true_observation: ReachingObservation) -> None:
        new_position = true_observation.target.position.clone()
        if self._last_position is None:
            observed_velocity = torch.zeros(3)
        else:
            delta_steps = max(true_observation.step_count - self._last_anchor_step, 1)
            observed_velocity = (new_position - self._last_position) / float(delta_steps)
        self._last_position = new_position
        self._last_velocity = observed_velocity
        self._last_anchor_step = true_observation.step_count

