from __future__ import annotations

from dataclasses import dataclass

import torch

from vla_gnn_recurrent.graph.types import ObjectObservation
from vla_gnn_recurrent.utils import clamp_delta, tensor3


@dataclass
class ReachingEnvConfig:
    max_steps: int = 32
    max_action: float = 0.08
    target_radius: float = 0.05
    initial_workspace: float = 0.8
    min_distance: float = 0.45
    max_distance: float = 1.2
    target_velocity_min: float = 0.006
    target_velocity_max: float = 0.018


@dataclass(frozen=True)
class ReachingObservation:
    ee_position: torch.Tensor
    ee_velocity: torch.Tensor
    target: ObjectObservation
    step_count: int


class ReachingEnv:
    """Tiny 3D closed-loop end-effector reaching environment."""

    def __init__(self, config: ReachingEnvConfig | None = None, seed: int = 0) -> None:
        self.config = config or ReachingEnvConfig()
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.ee_position = torch.zeros(3)
        self.ee_velocity = torch.zeros(3)
        self.target_position = torch.zeros(3)
        self.target_velocity = torch.zeros(3)
        self.target_motion_velocity = torch.zeros(3)
        self.step_count = 0

    def reset(
        self,
        ee_position: torch.Tensor | list[float] | tuple[float, float, float] | None = None,
        target_position: torch.Tensor | list[float] | tuple[float, float, float] | None = None,
        target_velocity: torch.Tensor | list[float] | tuple[float, float, float] | None = None,
        distance_range: tuple[float, float] | None = None,
    ) -> ReachingObservation:
        if ee_position is None or target_position is None:
            sampled_ee, sampled_target = self._sample_pair(distance_range)
            ee_position = sampled_ee if ee_position is None else tensor3(ee_position)
            target_position = sampled_target if target_position is None else tensor3(target_position)

        self.ee_position = tensor3(ee_position).clone()
        self.ee_velocity = torch.zeros(3)
        self.target_position = tensor3(target_position).clone()
        self.target_motion_velocity = torch.zeros(3) if target_velocity is None else tensor3(target_velocity).clone()
        self.target_velocity = self.target_motion_velocity.clone()
        self.step_count = 0
        return self.observe()

    def reset_moving_target(
        self,
        distance_range: tuple[float, float] | None = None,
        velocity_range: tuple[float, float] | None = None,
    ) -> ReachingObservation:
        ee_position, target_position = self._sample_pair(distance_range)
        target_velocity = self.sample_target_velocity(velocity_range)
        return self.reset(
            ee_position=ee_position,
            target_position=target_position,
            target_velocity=target_velocity,
            distance_range=distance_range,
        )

    def observe(self) -> ReachingObservation:
        return ReachingObservation(
            ee_position=self.ee_position.clone(),
            ee_velocity=self.ee_velocity.clone(),
            target=ObjectObservation.from_position(
                object_id="target_object",
                position=self.target_position.clone(),
                velocity=self.target_velocity.clone(),
                is_target=True,
            ),
            step_count=self.step_count,
        )

    def step(self, action: torch.Tensor) -> tuple[ReachingObservation, float, bool, dict[str, float | bool]]:
        bounded_action = clamp_delta(tensor3(action.detach().cpu()), self.config.max_action)
        previous = self.ee_position.clone()
        self.ee_position = self.ee_position + bounded_action
        self.ee_velocity = self.ee_position - previous

        previous_target = self.target_position.clone()
        self.target_position = self.target_position + self.target_motion_velocity
        self.target_velocity = self.target_position - previous_target
        self.step_count += 1

        distance = self.distance_to_target()
        success = distance <= self.config.target_radius
        timeout = self.step_count >= self.config.max_steps
        done = success or timeout
        info = {"distance": distance, "success": success, "timeout": timeout}
        return self.observe(), -distance, done, info

    def move_target(self, new_position: torch.Tensor | list[float] | tuple[float, float, float]) -> None:
        updated = tensor3(new_position).clone()
        self.target_velocity = updated - self.target_position
        self.target_position = updated
        self.target_motion_velocity = torch.zeros(3)

    def distance_to_target(self) -> float:
        return float((self.target_position - self.ee_position).norm().item())

    def optimal_action(self, lookahead: bool = False) -> torch.Tensor:
        target = self.target_position + self.target_motion_velocity if lookahead else self.target_position
        return clamp_delta(target - self.ee_position, self.config.max_action)

    def sample_perturbed_target(self, min_distance: float = 0.55, max_distance: float = 1.05) -> torch.Tensor:
        """Sample a new target that usually requires redirecting the current EE path."""
        old_direction = self.target_position - self.ee_position
        if float(old_direction.norm()) < 1e-6:
            old_direction = self._random_unit_vector()
        redirect = -old_direction / old_direction.norm().clamp_min(1e-6)
        noise = 0.35 * self._random_unit_vector()
        direction = redirect + noise
        direction = direction / direction.norm().clamp_min(1e-6)
        distance = self._uniform(min_distance, max_distance)
        return self.ee_position + direction * distance

    def sample_target_velocity(self, velocity_range: tuple[float, float] | None = None) -> torch.Tensor:
        min_velocity, max_velocity = velocity_range or (
            self.config.target_velocity_min,
            self.config.target_velocity_max,
        )
        return self._random_unit_vector() * self._uniform(min_velocity, max_velocity)

    def _sample_pair(self, distance_range: tuple[float, float] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        min_distance, max_distance = distance_range or (self.config.min_distance, self.config.max_distance)
        ee = self._uniform_vec(-self.config.initial_workspace, self.config.initial_workspace)
        direction = self._random_unit_vector()
        distance = self._uniform(min_distance, max_distance)
        target = ee + direction * distance
        return ee, target

    def _random_unit_vector(self) -> torch.Tensor:
        vec = torch.randn(3, generator=self.generator)
        return vec / vec.norm().clamp_min(1e-6)

    def _uniform_vec(self, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(3, generator=self.generator)

    def _uniform(self, low: float, high: float) -> float:
        return float(low + (high - low) * torch.rand((), generator=self.generator).item())
