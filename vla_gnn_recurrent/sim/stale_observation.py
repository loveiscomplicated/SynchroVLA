from __future__ import annotations

from dataclasses import replace

import torch

from vla_gnn_recurrent.graph.types import ObjectObservation
from vla_gnn_recurrent.sim.types import ManipulationObservation


class StaleManipulationObserver:
    """Hold object/destination observations stale while robot proprioception updates every step."""

    def __init__(self, observation_interval: int = 1, object_velocity_feature: bool = False) -> None:
        if observation_interval < 1:
            raise ValueError("observation_interval must be >= 1")
        self.observation_interval = observation_interval
        self.object_velocity_feature = object_velocity_feature
        self._target: ObjectObservation | None = None
        self._object: ObjectObservation | None = None
        self._destination: ObjectObservation | None = None
        self._last_anchor_step = 0
        self._pending_event = False
        self._event_true_step: int | None = None

    def reset(self) -> None:
        self._target = None
        self._object = None
        self._destination = None
        self._last_anchor_step = 0
        self._pending_event = False
        self._event_true_step = None

    def notify_true_event(self, step_count: int) -> None:
        self._pending_event = True
        self._event_true_step = int(step_count)

    def observe(self, true_observation: ManipulationObservation) -> ManipulationObservation:
        reanchored = self._should_reanchor(true_observation.step_count)
        if reanchored:
            self._target = self._copy_object(true_observation.true_target, self._target, true_observation.step_count)
            if true_observation.true_object is not None:
                self._object = self._copy_object(true_observation.true_object, self._object, true_observation.step_count)
            if true_observation.true_destination is not None:
                self._destination = self._copy_object(
                    true_observation.true_destination,
                    self._destination,
                    true_observation.step_count,
                )
            self._last_anchor_step = true_observation.step_count

        if self._target is None:
            raise RuntimeError("StaleManipulationObserver failed to initialize target observation.")

        observable_event = bool(
            self._pending_event
            and reanchored
            and self._event_true_step is not None
            and true_observation.step_count >= self._event_true_step
        )
        if observable_event:
            self._pending_event = False

        return ManipulationObservation(
            robot=true_observation.robot,
            target=self._target,
            true_target=true_observation.true_target,
            object=self._object,
            true_object=true_observation.true_object,
            destination=self._destination,
            true_destination=true_observation.true_destination,
            step_count=true_observation.step_count,
            reanchored=reanchored,
            observable_event=observable_event,
            grasp=true_observation.grasp,
        )

    def _should_reanchor(self, step_count: int) -> bool:
        return self._target is None or (
            step_count != self._last_anchor_step and step_count % self.observation_interval == 0
        )

    def _copy_object(
        self,
        current: ObjectObservation,
        previous: ObjectObservation | None,
        step_count: int,
    ) -> ObjectObservation:
        if previous is None or not self.object_velocity_feature:
            velocity = torch.zeros(3)
        else:
            delta_steps = max(step_count - self._last_anchor_step, 1)
            velocity = (current.position - previous.position) / float(delta_steps)
        return replace(current, position=current.position.clone(), velocity=velocity, confidence=current.confidence)
