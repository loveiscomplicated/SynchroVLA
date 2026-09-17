from __future__ import annotations

from dataclasses import dataclass

import torch

from vla_gnn_recurrent.env.reaching_env import ReachingEnv


@dataclass(frozen=True)
class EpisodeSpec:
    ee_position: torch.Tensor
    target_position: torch.Tensor


def sample_episode(env: ReachingEnv, distance_range: tuple[float, float] | None = None) -> EpisodeSpec:
    observation = env.reset(distance_range=distance_range)
    return EpisodeSpec(
        ee_position=observation.ee_position.clone(),
        target_position=observation.target.position.clone(),
    )

