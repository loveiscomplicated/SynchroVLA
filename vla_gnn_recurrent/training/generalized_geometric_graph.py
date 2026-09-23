"""Variable-sampling geometric graph feasibility experiment.

This module deliberately keeps the continuous part geometry in the oracle/evaluator
and exposes only sampled (s, point) observations to the policies.  The simulator's
existing planar convention is preserved: MuJoCo moves the EE in the x-z plane and
the scalar orientation state is the synthetic x-z planar yaw used by the prior
keypoint experiment.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv, MujocoReachConfig
from vla_gnn_recurrent.utils import DevicePreference, clamp_delta, ensure_dir, select_device, set_seed

GeneralizedModelName = Literal["local_concat", "local_set_attention", "geometric_gnn", "geometric_gnn_no_local"]
EvalCondition = Literal["iid", "sampling_ood", "shape_ood", "relative_orientation_ood"]


@dataclass
class GeneralizedGeometryConfig:
    output_dir: str = "artifacts/generalized_geometric_graph_feasibility"
    seeds: tuple[int, ...] = (2811, 2812, 2813)
    train_shapes: int = 48
    val_shapes: int = 12
    test_shapes: int = 12
    eval_shapes: int = 24
    max_steps: int = 20
    epochs: int = 24
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    message_passing_layers: int = 2
    attention_heads: int = 4
    resample_points: int = 8
    train_k_range: tuple[int, int] = (5, 9)
    sampling_ood_k_values: tuple[int, ...] = (3, 4, 10, 12)
    target_s: float = 0.5
    target_radius: float = 0.030
    yaw_success_threshold: float = 0.35
    max_delta_ee: float = 0.035
    max_delta_yaw: float = 0.35
    yaw_loss_weight: float = 0.25
    object_x_range: tuple[float, float] = (-0.14, 0.14)
    object_z_range: tuple[float, float] = (0.50, 0.74)
    object_yaw_range: tuple[float, float] = (-0.45, 0.45)
    part_offset_x_range: tuple[float, float] = (-0.035, 0.035)
    part_offset_z_range: tuple[float, float] = (-0.020, 0.030)
    part_yaw_range: tuple[float, float] = (-0.55, 0.55)
    length_range: tuple[float, float] = (0.08, 0.12)
    curvature_range: tuple[float, float] = (-0.004, 0.004)
    bend_range: tuple[float, float] = (-0.005, 0.005)
    asymmetry_range: tuple[float, float] = (-0.004, 0.004)
    shape_ood_length_range: tuple[float, float] = (0.13, 0.17)
    shape_ood_curvature_range: tuple[float, float] = (-0.018, 0.018)
    shape_ood_bend_range: tuple[float, float] = (-0.018, 0.018)
    shape_ood_asymmetry_range: tuple[float, float] = (-0.014, 0.014)
    train_ee_yaw_range: tuple[float, float] = (-0.45, 0.45)
    relative_ood_ee_yaw_abs_range: tuple[float, float] = (1.20, 2.40)
    device: DevicePreference = "auto"
    eval_device: DevicePreference = "auto"
    bootstrap_samples: int = 1000
    plot_cases: int = 3


@dataclass
class OrientationSanityConfig:
    source_output_dir: str = "artifacts/generalized_geometric_graph_feasibility"
    output_dir: str = "artifacts/generalized_geometric_graph_feasibility/orientation_sanity"
    seeds: tuple[int, ...] = (2811, 2812, 2813)
    wide_train_ee_yaw_range: tuple[float, float] = (-2.40, 2.40)
    wide_epochs: int = 24
    batch_size: int = 128
    device: DevicePreference = "auto"
    eval_device: DevicePreference = "auto"
    max_steps: int = 20
    orientation_bins: tuple[tuple[float, float], ...] = ((0.0, 0.45), (0.45, 0.90), (0.90, 1.20), (1.20, 1.80), (1.80, 2.40))
    sweep_episodes_per_bin: int = 48
    rollout_episodes: int = 48


@dataclass(frozen=True)
class PhysicalShape:
    shape_id: str
    object_center: tuple[float, float, float]
    object_yaw: float
    part_offset: tuple[float, float, float]
    part_yaw: float
    length: float
    curvature: float
    bend: float
    asymmetry: float


@dataclass(frozen=True)
class GeneralizedEpisode:
    episode_id: int
    shape: PhysicalShape
    arm_qpos: tuple[float, float, float, float]
    ee_yaw: float
    condition: str


@dataclass
class GeneralizedBatch:
    object_features: torch.Tensor
    ee_features: torch.Tensor
    point_features: torch.Tensor
    point_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "GeneralizedBatch":
        return GeneralizedBatch(
            object_features=self.object_features.to(device),
            ee_features=self.ee_features.to(device),
            point_features=self.point_features.to(device),
            point_mask=self.point_mask.to(device),
        )

    def index_select(self, indices: torch.Tensor) -> "GeneralizedBatch":
        return GeneralizedBatch(
            object_features=self.object_features.index_select(0, indices),
            ee_features=self.ee_features.index_select(0, indices),
            point_features=self.point_features.index_select(0, indices),
            point_mask=self.point_mask.index_select(0, indices),
        )

    @property
    def num_samples(self) -> int:
        return int(self.object_features.shape[0])


def wrap_angle(angle: float | torch.Tensor) -> float | torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotate_xz(vector: torch.Tensor, yaw: float | torch.Tensor) -> torch.Tensor:
    c = torch.cos(torch.as_tensor(yaw, dtype=vector.dtype, device=vector.device))
    s = torch.sin(torch.as_tensor(yaw, dtype=vector.dtype, device=vector.device))
    x = c * vector[..., 0] - s * vector[..., 2]
    z = s * vector[..., 0] + c * vector[..., 2]
    return torch.stack([x, vector[..., 1], z], dim=-1)


def curve_local(s: torch.Tensor, shape: PhysicalShape) -> tuple[torch.Tensor, torch.Tensor]:
    """Return C(s) and dC/ds in the part-local x-z plane."""
    u = s - 0.5
    x = shape.length * u
    z = shape.curvature * u.square() + shape.bend * torch.sin(2.0 * math.pi * s) + shape.asymmetry * u.pow(3)
    dx = torch.full_like(s, shape.length)
    dz = 2.0 * shape.curvature * u + 2.0 * math.pi * shape.bend * torch.cos(2.0 * math.pi * s) + 3.0 * shape.asymmetry * u.square()
    points = torch.stack([x, torch.zeros_like(x), z], dim=-1)
    tangent = torch.stack([dx, torch.zeros_like(dx), dz], dim=-1)
    return points, tangent


def curve_world(s: torch.Tensor, shape: PhysicalShape) -> tuple[torch.Tensor, torch.Tensor]:
    local, tangent = curve_local(s, shape)
    offset = torch.tensor(shape.part_offset, dtype=local.dtype, device=local.device)
    center = torch.tensor(shape.object_center, dtype=local.dtype, device=local.device)
    object_points = offset + rotate_xz(local, shape.part_yaw)
    object_tangent = rotate_xz(tangent, shape.part_yaw)
    world_points = center + rotate_xz(object_points, shape.object_yaw)
    world_tangent = rotate_xz(object_tangent, shape.object_yaw)
    return world_points, world_tangent


def interaction_frame(shape: PhysicalShape, target_s: float) -> tuple[torch.Tensor, float]:
    s = torch.tensor([float(target_s)], dtype=torch.float32)
    point, tangent = curve_world(s, shape)
    vector = tangent[0]
    yaw = float(math.atan2(float(vector[2]), float(vector[0])))
    return point[0], yaw


def sample_observation_parameters(
    rng: np.random.Generator,
    k_range: tuple[int, int] | None = None,
    k_values: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if k_values is not None:
        k = int(k_values[int(rng.integers(0, len(k_values)))])
    else:
        assert k_range is not None
        k = int(rng.integers(k_range[0], k_range[1] + 1))
    if k < 3:
        raise ValueError("At least three observations are required.")
    if k == 3:
        interior = np.array([0.5], dtype=np.float64)
    else:
        interior = np.sort(rng.uniform(0.02, 0.98, size=k - 2))
        if np.min(np.abs(interior - 0.5)) > 0.20:
            interior[int(np.argmin(np.abs(interior - 0.5)))] = 0.5
            interior.sort()
    values = np.concatenate([[0.0], interior, [1.0]])
    return torch.tensor(values, dtype=torch.float32)


def _localize_world(points: torch.Tensor, ee_position: torch.Tensor, ee_yaw: float) -> torch.Tensor:
    return rotate_xz(points - ee_position.reshape(1, 3), -ee_yaw)


def build_observation_from_state(
    shape: PhysicalShape,
    ee_position: torch.Tensor,
    ee_yaw: float,
    s_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points_world, _ = curve_world(s_values, shape)
    points_local = _localize_world(points_world, ee_position, ee_yaw)
    object_position = torch.tensor(shape.object_center, dtype=torch.float32)
    object_local = rotate_xz(object_position.reshape(1, 3), -ee_yaw)[0] - rotate_xz(
        ee_position.reshape(1, 3), -ee_yaw
    )[0]
    relative_object_yaw = float(wrap_angle(shape.object_yaw - ee_yaw))
    object_features = torch.cat(
        [
            object_local,
            torch.tensor([math.sin(relative_object_yaw), math.cos(relative_object_yaw)], dtype=torch.float32),
            torch.tensor([0.095, 0.040], dtype=torch.float32),
        ]
    )
    ee_features = torch.tensor([0.0], dtype=torch.float32)
    point_features = torch.cat([points_local, s_values.reshape(-1, 1)], dim=-1)
    return object_features, ee_features, point_features


def build_observation(
    env: MujocoManipulatorEnv,
    episode: GeneralizedEpisode,
    s_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return build_observation_from_state(
        episode.shape,
        env.robot_observation().ee_position.float(),
        episode.ee_yaw,
        s_values,
    )


def expert_action_local(
    env: MujocoManipulatorEnv,
    episode: GeneralizedEpisode,
    config: GeneralizedGeometryConfig,
) -> torch.Tensor:
    return expert_action_from_state(env.robot_observation().ee_position.float(), episode, config)


def expert_action_from_state(
    ee: torch.Tensor,
    episode: GeneralizedEpisode,
    config: GeneralizedGeometryConfig,
) -> torch.Tensor:
    target, target_yaw = interaction_frame(episode.shape, config.target_s)
    delta_world = target - ee
    delta_local = rotate_xz(delta_world.reshape(1, 3), -episode.ee_yaw)[0]
    delta_local = clamp_delta(delta_local, config.max_delta_ee)
    yaw_delta = max(-config.max_delta_yaw, min(config.max_delta_yaw, float(wrap_angle(target_yaw - episode.ee_yaw))))
    return torch.cat([delta_local, torch.tensor([yaw_delta], dtype=torch.float32)])


def apply_local_action(
    env: MujocoManipulatorEnv,
    episode: GeneralizedEpisode,
    action: torch.Tensor,
    config: GeneralizedGeometryConfig,
) -> GeneralizedEpisode:
    local_delta = clamp_delta(action[:3].detach().cpu().float(), config.max_delta_ee)
    world_delta = rotate_xz(local_delta.reshape(1, 3), episode.ee_yaw)[0]
    env.step_delta_ee(world_delta, gripper=0.0)
    yaw_delta = float(torch.clamp(action[3].detach().cpu(), -config.max_delta_yaw, config.max_delta_yaw).item())
    return GeneralizedEpisode(
        episode_id=episode.episode_id,
        shape=episode.shape,
        arm_qpos=episode.arm_qpos,
        ee_yaw=float(wrap_angle(episode.ee_yaw + yaw_delta)),
        condition=episode.condition,
    )


def _reset_env(env: MujocoManipulatorEnv, episode: GeneralizedEpisode, config: GeneralizedGeometryConfig) -> None:
    env.reset_pick_scene(
        object_x=float(episode.shape.object_center[0]),
        object_yaw=episode.shape.object_yaw,
        randomize_robot=False,
    )
    env._set_arm_qpos(env._clip_arm_qpos(np.asarray(episode.arm_qpos, dtype=np.float64)))
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = 0.0
    target, _ = interaction_frame(episode.shape, config.target_s)
    env.set_object_position(np.asarray(episode.shape.object_center, dtype=np.float64), yaw=episode.shape.object_yaw)
    env.set_target_position(target.detach().cpu().numpy())
    for _ in range(8):
        env.step_delta_ee(torch.zeros(3), gripper=0.0)


def _sample_shape(config: GeneralizedGeometryConfig, rng: np.random.Generator, shape_id: str, ood: bool = False) -> PhysicalShape:
    length_range = config.shape_ood_length_range if ood else config.length_range
    curvature_range = config.shape_ood_curvature_range if ood else config.curvature_range
    bend_range = config.shape_ood_bend_range if ood else config.bend_range
    asymmetry_range = config.shape_ood_asymmetry_range if ood else config.asymmetry_range
    return PhysicalShape(
        shape_id=shape_id,
        object_center=(float(rng.uniform(*config.object_x_range)), 0.0, float(rng.uniform(*config.object_z_range))),
        object_yaw=float(rng.uniform(*config.object_yaw_range)),
        part_offset=(float(rng.uniform(*config.part_offset_x_range)), 0.0, float(rng.uniform(*config.part_offset_z_range))),
        part_yaw=float(rng.uniform(*config.part_yaw_range)),
        length=float(rng.uniform(*length_range)),
        curvature=float(rng.uniform(*curvature_range)),
        bend=float(rng.uniform(*bend_range)),
        asymmetry=float(rng.uniform(*asymmetry_range)),
    )


def _sample_episode(
    shape: PhysicalShape,
    episode_id: int,
    rng: np.random.Generator,
    config: GeneralizedGeometryConfig,
    condition: str,
) -> GeneralizedEpisode:
    if condition == "relative_orientation_ood":
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        ee_yaw = sign * float(rng.uniform(*config.relative_ood_ee_yaw_abs_range))
    else:
        ee_yaw = float(rng.uniform(*config.train_ee_yaw_range))
    home = np.array([0.0, -0.35, 0.95, -0.45], dtype=np.float64)
    arm_qpos = home + rng.uniform(-0.25, 0.25, size=4)
    return GeneralizedEpisode(
        episode_id=episode_id,
        shape=shape,
        arm_qpos=tuple(float(v) for v in arm_qpos),
        ee_yaw=ee_yaw,
        condition=condition,
    )


def _sample_orientation_episode(
    shape: PhysicalShape,
    episode_id: int,
    rng: np.random.Generator,
    config: GeneralizedGeometryConfig,
    abs_range: tuple[float, float],
    condition: str,
) -> tuple[GeneralizedEpisode, float]:
    """Create an episode with a controlled target-vs-EE relative orientation."""
    sign = 1.0 if rng.random() >= 0.5 else -1.0
    relative_yaw = sign * float(rng.uniform(*abs_range))
    _, target_yaw = interaction_frame(shape, config.target_s)
    ee_yaw = float(wrap_angle(target_yaw - relative_yaw))
    home = np.array([0.0, -0.35, 0.95, -0.45], dtype=np.float64)
    arm_qpos = home + rng.uniform(-0.25, 0.25, size=4)
    return (
        GeneralizedEpisode(
            episode_id=episode_id,
            shape=shape,
            arm_qpos=tuple(float(v) for v in arm_qpos),
            ee_yaw=ee_yaw,
            condition=condition,
        ),
        relative_yaw,
    )


def _env_config(config: GeneralizedGeometryConfig) -> MujocoReachConfig:
    return MujocoReachConfig(
        max_steps=config.max_steps,
        control_substeps=25,
        max_delta_ee=config.max_delta_ee,
        target_radius=config.target_radius,
        pick_scene=True,
        kinematic_joint_control=True,
    )


def _collect_episode_samples(
    env: MujocoManipulatorEnv,
    episode: GeneralizedEpisode,
    config: GeneralizedGeometryConfig,
    rng: np.random.Generator,
    sampling_ood: bool = False,
) -> list[dict[str, Any]]:
    _reset_env(env, episode, config)
    current = episode
    samples: list[dict[str, Any]] = []
    for _ in range(config.max_steps):
        s_values = sample_observation_parameters(
            rng,
            k_values=config.sampling_ood_k_values if sampling_ood else None,
            k_range=None if sampling_ood else config.train_k_range,
        )
        object_features, ee_features, point_features = build_observation(env, current, s_values)
        action = expert_action_local(env, current, config)
        samples.append(
            {
                "object_features": object_features,
                "ee_features": ee_features,
                "point_features": point_features,
                "action": action,
                "episode_id": current.episode_id,
                "shape_id": current.shape.shape_id,
                "s_values": s_values,
            }
        )
        current = apply_local_action(env, current, action, config)
        target, target_yaw = interaction_frame(current.shape, config.target_s)
        distance = float((env.robot_observation().ee_position.float() - target).norm().item())
        if distance <= config.target_radius and abs(float(wrap_angle(target_yaw - current.ee_yaw))) <= config.yaw_success_threshold:
            break
    return samples


def generate_generalized_dataset(config: GeneralizedGeometryConfig, seed: int) -> dict[str, Any]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    env = MujocoManipulatorEnv(_env_config(config), seed=seed)
    try:
        split_sizes = {"train": config.train_shapes, "val": config.val_shapes, "test": config.test_shapes}
        split_samples: dict[str, list[dict[str, Any]]] = {}
        physical_shapes: dict[str, list[dict[str, Any]]] = {}
        for split, count in split_sizes.items():
            episodes: list[dict[str, Any]] = []
            samples: list[dict[str, Any]] = []
            for idx in range(count):
                shape = _sample_shape(config, rng, f"{split}_{idx:04d}")
                episode = _sample_episode(shape, idx, rng, config, "train")
                physical_shapes.setdefault(split, []).append(asdict(shape))
                samples.extend(_collect_episode_samples(env, episode, config, rng))
                episodes.append(asdict(episode))
            split_samples[split] = samples
        return {"config": asdict(config), "seed": seed, "splits": split_samples, "physical_shapes": physical_shapes}
    finally:
        env.close()


def _pad_samples(samples: list[dict[str, Any]], device: torch.device | str) -> GeneralizedBatch:
    batch = len(samples)
    max_k = max(int(sample["point_features"].shape[0]) for sample in samples)
    object_features = torch.stack([sample["object_features"] for sample in samples]).float()
    ee_features = torch.stack([sample["ee_features"] for sample in samples]).float()
    point_features = torch.zeros(batch, max_k, 4, dtype=torch.float32)
    point_mask = torch.zeros(batch, max_k, dtype=torch.bool)
    for idx, sample in enumerate(samples):
        points = sample["point_features"].float()
        point_features[idx, : points.shape[0]] = points
        point_mask[idx, : points.shape[0]] = True
    return GeneralizedBatch(object_features, ee_features, point_features, point_mask).to(device)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _resample_observed(points: torch.Tensor, mask: torch.Tensor, count: int) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, count, device=points.device, dtype=points.dtype)
    outputs = []
    for row, row_mask in zip(points, mask):
        valid = row[row_mask]
        valid = valid[torch.argsort(valid[:, 3])]
        s = valid[:, 3].contiguous()
        xyz = valid[:, :3]
        rows = []
        for query in grid:
            right = int(torch.searchsorted(s, query).clamp(0, len(s) - 1).item())
            left = max(right - 1, 0)
            if right == left:
                value = xyz[left]
            else:
                alpha = (query - s[left]) / (s[right] - s[left]).clamp_min(1e-6)
                value = xyz[left] + alpha * (xyz[right] - xyz[left])
            rows.append(torch.cat([value, query.reshape(1)]))
        outputs.append(torch.stack(rows))
    return torch.stack(outputs)


class LocalResampledConcat(nn.Module):
    def __init__(self, hidden_dim: int, resample_points: int, output_dim: int = 4) -> None:
        super().__init__()
        self.resample_points = resample_points
        input_dim = 7 + 1 + resample_points * 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: GeneralizedBatch) -> torch.Tensor:
        points = _resample_observed(batch.point_features, batch.point_mask, self.resample_points).flatten(1)
        return self.net(torch.cat([batch.object_features, batch.ee_features, points], dim=-1))


class LocalSetAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, output_dim: int = 4) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim + 8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, batch: GeneralizedBatch) -> torch.Tensor:
        state = self.point_encoder(batch.point_features)
        attended, _ = self.attention(state, state, state, key_padding_mask=~batch.point_mask)
        state = self.norm(state + attended)
        pooled = _masked_mean(state, batch.point_mask)
        return self.head(torch.cat([pooled, batch.object_features, batch.ee_features], dim=-1))


class GeometricMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, use_local_edges: bool) -> None:
        super().__init__()
        self.use_local_edges = use_local_edges
        edge_dim = 4 + 1 + 4
        self.message = nn.Sequential(nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def _edge(self, dst: torch.Tensor, src: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([dst, src, edge], dim=-1)
        return self.message(joined) * torch.sigmoid(self.gate(joined))

    def forward(
        self,
        object_state: torch.Tensor,
        ee_state: torch.Tensor,
        point_state: torch.Tensor,
        point_features: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz = point_features[:, :, :3]
        s = point_features[:, :, 3:4]
        b, k, _ = point_state.shape
        point_agg = self._edge(
            point_state,
            ee_state.unsqueeze(1).expand(-1, k, -1),
            _edge_features(xyz, torch.zeros_like(xyz), s, edge_type=0),
        )
        object_xyz = object_state.new_zeros(b, k, 3)
        point_agg = point_agg + self._edge(
            point_state,
            object_state.unsqueeze(1).expand(-1, k, -1),
            _edge_features(xyz, object_xyz, s, edge_type=1),
        )
        if self.use_local_edges and k > 1:
            right_state = torch.roll(point_state, shifts=-1, dims=1)
            right_xyz = torch.roll(xyz, shifts=-1, dims=1)
            right_s = torch.roll(s, shifts=-1, dims=1)
            valid = point_mask & torch.roll(point_mask, shifts=-1, dims=1)
            local_edge = _edge_features(right_xyz - xyz, torch.zeros_like(xyz), right_s - s, edge_type=2)
            local_msg = self._edge(point_state, right_state, local_edge) * valid.unsqueeze(-1)
            point_agg = point_agg + local_msg
        point_state = self.norm(point_state + self.update(torch.cat([point_state, point_agg], dim=-1)))
        point_state = point_state * point_mask.unsqueeze(-1)

        point_to_ee = self._edge(
            ee_state.unsqueeze(1).expand(-1, k, -1), point_state,
            _edge_features(xyz, torch.zeros_like(xyz), s, edge_type=3),
        )
        point_to_ee = point_to_ee * point_mask.unsqueeze(-1)
        ee_agg = point_to_ee.sum(dim=1) / point_mask.sum(dim=1, keepdim=True).clamp_min(1).to(point_to_ee.dtype)
        ee_state = self.norm(ee_state + self.update(torch.cat([ee_state, ee_agg], dim=-1)))

        point_to_object = self._edge(
            object_state.unsqueeze(1).expand(-1, k, -1), point_state,
            _edge_features(xyz, object_xyz, s, edge_type=1),
        ) * point_mask.unsqueeze(-1)
        object_agg = point_to_object.sum(dim=1) / point_mask.sum(dim=1, keepdim=True).clamp_min(1).to(point_to_object.dtype)
        object_state = self.norm(object_state + self.update(torch.cat([object_state, object_agg], dim=-1)))
        return object_state, ee_state, point_state


def _edge_features(relative: torch.Tensor, unused: torch.Tensor, delta_s: torch.Tensor, edge_type: int) -> torch.Tensor:
    del unused
    type_one_hot = torch.zeros(*relative.shape[:-1], 4, device=relative.device, dtype=relative.dtype)
    type_one_hot[..., edge_type] = 1.0
    return torch.cat([relative, relative.square().sum(dim=-1, keepdim=True), delta_s, type_one_hot], dim=-1)


class GeometricGNN(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, use_local_edges: bool, output_dim: int = 4) -> None:
        super().__init__()
        self.object_projector = nn.Sequential(nn.Linear(7, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.ee_projector = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.point_projector = nn.Sequential(nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList([GeometricMessageLayer(hidden_dim, use_local_edges) for _ in range(layers)])
        self.action_head = nn.Sequential(nn.Linear(hidden_dim * 3 + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))
        self.vector_coeff = nn.Sequential(nn.Linear(hidden_dim * 2 + 5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, batch: GeneralizedBatch) -> torch.Tensor:
        object_state = self.object_projector(batch.object_features)
        ee_state = self.ee_projector(batch.ee_features)
        order = torch.argsort(batch.point_features[:, :, 3], dim=1)
        gather_index = order.unsqueeze(-1).expand(-1, -1, batch.point_features.shape[-1])
        sorted_features = torch.gather(batch.point_features, 1, gather_index)
        sorted_mask = torch.gather(batch.point_mask, 1, order)
        point_state = self.point_projector(sorted_features) * sorted_mask.unsqueeze(-1)
        for layer in self.layers:
            object_state, ee_state, point_state = layer(
                object_state, ee_state, point_state, sorted_features, sorted_mask
            )
        geometry_state = _masked_mean(point_state, sorted_mask)
        xyz = sorted_features[:, :, :3]
        s = sorted_features[:, :, 3:4]
        ee_expand = ee_state.unsqueeze(1).expand_as(point_state)
        coeff_input = torch.cat([ee_expand, point_state, xyz, xyz.square().sum(dim=-1, keepdim=True), s], dim=-1)
        coeff = self.vector_coeff(coeff_input) * sorted_mask.unsqueeze(-1)
        vector_path = (coeff * xyz).sum(dim=1) / sorted_mask.sum(dim=1, keepdim=True).clamp_min(1).to(xyz.dtype)
        vector_features = torch.cat([vector_path, vector_path.norm(dim=-1, keepdim=True)], dim=-1)
        return self.action_head(torch.cat([ee_state, object_state, geometry_state, vector_features], dim=-1))


def build_model(name: GeneralizedModelName, config: GeneralizedGeometryConfig) -> nn.Module:
    if name == "local_concat":
        return LocalResampledConcat(config.hidden_dim, config.resample_points)
    if name == "local_set_attention":
        return LocalSetAttention(config.hidden_dim, config.attention_heads)
    if name == "geometric_gnn":
        return GeometricGNN(config.hidden_dim, config.message_passing_layers, True)
    if name == "geometric_gnn_no_local":
        return GeometricGNN(config.hidden_dim, config.message_passing_layers, False)
    raise ValueError(f"Unknown model: {name}")


def _model_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _loss(prediction: torch.Tensor, target: torch.Tensor, yaw_weight: float) -> torch.Tensor:
    return nn.functional.mse_loss(prediction[:, :3], target[:, :3]) + yaw_weight * nn.functional.mse_loss(
        prediction[:, 3], target[:, 3]
    )


def _evaluate_offline(model: nn.Module, samples: list[dict[str, Any]], device: torch.device, config: GeneralizedGeometryConfig) -> dict[str, float]:
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(samples), config.batch_size):
            batch_samples = samples[start : start + config.batch_size]
            batch = _pad_samples(batch_samples, device)
            target = torch.stack([sample["action"] for sample in batch_samples]).to(device)
            prediction = model(batch)
            errors.append(prediction.detach().cpu() - target.detach().cpu())
    error = torch.cat(errors, dim=0)
    return {
        "translation_l2": float(error[:, :3].norm(dim=-1).mean().item()),
        "rotation_abs_error": float(error[:, 3].abs().mean().item()),
    }


def train_model(
    model_name: GeneralizedModelName,
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]] | None,
    config: GeneralizedGeometryConfig,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    device = select_device(config.device)
    model = build_model(model_name, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(train_samples), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), config.batch_size):
            batch_samples = [train_samples[idx] for idx in order[start : start + config.batch_size]]
            batch = _pad_samples(batch_samples, device)
            target = torch.stack([sample["action"] for sample in batch_samples]).to(device)
            prediction = model(batch)
            loss = _loss(prediction, target, config.yaw_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch == 1 or epoch == config.epochs or epoch % max(config.epochs // 3, 1) == 0:
            val = _evaluate_offline(model, val_samples, device, config)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **val})
            print(f"[generalized:{model_name}] epoch={epoch} loss={np.mean(losses):.6f} val_xyz={val['translation_l2']:.4f}")
    summary = {
        "model": model_name,
        "model_state": model.state_dict(),
        "config": asdict(config),
        "parameter_count": _model_parameters(model),
        "history": history,
        "offline": {
            "train": _evaluate_offline(model, train_samples, device, config),
            "val": _evaluate_offline(model, val_samples, device, config),
            **({"test": _evaluate_offline(model, test_samples, device, config)} if test_samples is not None else {}),
        },
        "runtime_seconds": time.perf_counter() - start_time,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(summary, output_dir / f"{model_name}.pt")
    serializable = {key: value for key, value in summary.items() if key != "model_state"}
    (output_dir / f"{model_name}_summary.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return {"model": model, "summary": summary, "checkpoint_path": str(output_dir / f"{model_name}.pt")}


def _load_model(path: Path, config: GeneralizedGeometryConfig, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_name = checkpoint["model"]
    model = build_model(model_name, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _success_metrics(env: MujocoManipulatorEnv, episode: GeneralizedEpisode, config: GeneralizedGeometryConfig) -> tuple[float, float, bool]:
    target, target_yaw = interaction_frame(episode.shape, config.target_s)
    position_error = float((env.robot_observation().ee_position.float() - target).norm().item())
    yaw_error = abs(float(wrap_angle(target_yaw - episode.ee_yaw)))
    return position_error, yaw_error, position_error <= config.target_radius and yaw_error <= config.yaw_success_threshold


@torch.no_grad()
def rollout_model(
    model: nn.Module,
    env: MujocoManipulatorEnv,
    episode: GeneralizedEpisode,
    config: GeneralizedGeometryConfig,
    rng: np.random.Generator,
    sampling_ood: bool,
    device: torch.device,
) -> dict[str, Any]:
    _reset_env(env, episode, config)
    current = episode
    trajectory = [env.robot_observation().ee_position.float().tolist()]
    position_errors: list[float] = []
    yaw_errors: list[float] = []
    forward_ms: list[float] = []
    policy_ms: list[float] = []
    success = False
    for _ in range(config.max_steps):
        s_values = sample_observation_parameters(
            rng,
            k_values=config.sampling_ood_k_values if sampling_ood else None,
            k_range=None if sampling_ood else config.train_k_range,
        )
        obs_start = time.perf_counter()
        object_features, ee_features, point_features = build_observation(env, current, s_values)
        batch = _pad_samples(
            [{"object_features": object_features, "ee_features": ee_features, "point_features": point_features}], device
        )
        forward_start = time.perf_counter()
        action = model(batch)[0]
        if device.type != "cpu":
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize(device)
        forward_ms.append((time.perf_counter() - forward_start) * 1000.0)
        policy_ms.append((time.perf_counter() - obs_start) * 1000.0)
        current = apply_local_action(env, current, action, config)
        trajectory.append(env.robot_observation().ee_position.float().tolist())
        position_error, yaw_error, step_success = _success_metrics(env, current, config)
        position_errors.append(position_error)
        yaw_errors.append(yaw_error)
        success = success or step_success
        if step_success:
            break
    return {
        "success": bool(success),
        "final_position_error": position_errors[-1],
        "final_orientation_error": yaw_errors[-1],
        "mean_position_error": float(np.mean(position_errors)),
        "mean_orientation_error": float(np.mean(yaw_errors)),
        "steps": len(position_errors),
        "trajectory_length": float(np.linalg.norm(np.diff(np.asarray(trajectory), axis=0), axis=1).sum()) if len(trajectory) > 1 else 0.0,
        "trajectory": trajectory,
        "mean_forward_latency_ms": float(np.mean(forward_ms)),
        "mean_policy_latency_ms": float(np.mean(policy_ms)),
    }


def _contingency(results: list[dict[str, Any]], baseline: str, candidate: str) -> dict[str, int]:
    counts = {"both_success": 0, "baseline_only_success": 0, "candidate_only_success": 0, "both_fail": 0}
    for item in results:
        a = bool(item["models"][baseline]["success"])
        b = bool(item["models"][candidate]["success"])
        if a and b:
            counts["both_success"] += 1
        elif a:
            counts["baseline_only_success"] += 1
        elif b:
            counts["candidate_only_success"] += 1
        else:
            counts["both_fail"] += 1
    return counts


def _mcnemar_exact_p(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, candidate_only)
    probability = sum(math.comb(discordant, i) for i in range(tail + 1)) / (2.0 ** discordant)
    return float(min(1.0, 2.0 * probability))


def _bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    values_array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        means.append(float(values_array[rng.integers(0, len(values_array), size=len(values_array))].mean()))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _paired_summary(
    results: list[dict[str, Any]],
    baseline: str,
    candidate: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    delta_position = [
        float(item["models"][candidate]["final_position_error"] - item["models"][baseline]["final_position_error"])
        for item in results
    ]
    delta_orientation = [
        float(item["models"][candidate]["final_orientation_error"] - item["models"][baseline]["final_orientation_error"])
        for item in results
    ]
    counts = _contingency(results, baseline, candidate)
    return {
        "contingency": counts,
        "mcnemar_exact_p": _mcnemar_exact_p(counts["baseline_only_success"], counts["candidate_only_success"]),
        "delta_final_position": {
            "mean": float(np.mean(delta_position)) if delta_position else 0.0,
            "median": float(np.median(delta_position)) if delta_position else 0.0,
            "ci95": _bootstrap_ci(delta_position, bootstrap_samples, seed),
        },
        "delta_final_orientation": {
            "mean": float(np.mean(delta_orientation)) if delta_orientation else 0.0,
            "median": float(np.median(delta_orientation)) if delta_orientation else 0.0,
            "ci95": _bootstrap_ci(delta_orientation, bootstrap_samples, seed + 1),
        },
    }


def _summarize_rollout_results(results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "success_rate": float(np.mean([float(item["success"]) for item in results])),
        "final_position_error": float(np.mean([item["final_position_error"] for item in results])),
        "final_orientation_error": float(np.mean([item["final_orientation_error"] for item in results])),
        "mean_position_error": float(np.mean([item["mean_position_error"] for item in results])),
        "mean_orientation_error": float(np.mean([item["mean_orientation_error"] for item in results])),
        "steps": float(np.mean([item["steps"] for item in results])),
        "forward_latency_ms": float(np.mean([item["mean_forward_latency_ms"] for item in results])),
        "policy_latency_ms": float(np.mean([item["mean_policy_latency_ms"] for item in results])),
    }


def evaluate_condition(
    models: dict[str, nn.Module],
    config: GeneralizedGeometryConfig,
    shapes: list[PhysicalShape],
    condition: EvalCondition,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    envs = {name: MujocoManipulatorEnv(_env_config(config), seed=seed + idx) for idx, name in enumerate(models)}
    results = []
    try:
        for episode_id, shape in enumerate(shapes):
            episode = _sample_episode(shape, episode_id, rng, config, condition)
            model_results = {}
            for name, model in models.items():
                # Recreate the same per-step sampling sequence for every model in this episode.
                episode_sampling_rng = np.random.default_rng(seed + episode_id * 100003 + 17)
                model_results[name] = rollout_model(
                    model, envs[name], episode, config, episode_sampling_rng, condition == "sampling_ood", device
                )
            results.append({"episode_id": episode_id, "shape": asdict(shape), "models": model_results})
    finally:
        for env in envs.values():
            env.close()
    summaries = {name: _summarize_rollout_results([item["models"][name] for item in results]) for name in models}
    pairs = [("local_concat", "local_set_attention"), ("local_set_attention", "geometric_gnn"), ("local_concat", "geometric_gnn"), ("geometric_gnn", "geometric_gnn_no_local")]
    paired = {}
    for baseline, candidate in pairs:
        delta_pos = [item["models"][candidate]["final_position_error"] - item["models"][baseline]["final_position_error"] for item in results]
        delta_yaw = [item["models"][candidate]["final_orientation_error"] - item["models"][baseline]["final_orientation_error"] for item in results]
        del delta_pos, delta_yaw
        paired[f"{baseline}_vs_{candidate}"] = _paired_summary(
            results, baseline, candidate, bootstrap_samples=1000, seed=seed + len(paired)
        )
    return {"condition": condition, "num_episodes": len(results), "models": summaries, "paired": paired, "episodes": results}


def _orientation_bin_label(bounds: tuple[float, float]) -> str:
    return f"{bounds[0]:.2f}_{bounds[1]:.2f}"


@torch.no_grad()
def orientation_action_sweep(
    models: dict[str, nn.Module],
    shapes: list[PhysicalShape],
    config: GeneralizedGeometryConfig,
    sanity_config: OrientationSanityConfig,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Measure one-step yaw behavior in controlled relative-orientation bins."""
    rng = np.random.default_rng(seed)
    ee_position = torch.tensor([0.0, 0.0, 0.52], dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    per_bin: dict[str, dict[str, list[float]]] = {}
    model_names = tuple(models)
    for bin_index, bounds in enumerate(sanity_config.orientation_bins):
        label = _orientation_bin_label(bounds)
        per_bin[label] = {
            name: []
            for name in model_names
        }
        for local_index in range(sanity_config.sweep_episodes_per_bin):
            shape = shapes[(bin_index * sanity_config.sweep_episodes_per_bin + local_index) % len(shapes)]
            episode, relative_yaw = _sample_orientation_episode(
                shape,
                bin_index * sanity_config.sweep_episodes_per_bin + local_index,
                rng,
                config,
                bounds,
                f"orientation_bin_{label}",
            )
            s_values = sample_observation_parameters(rng, k_range=config.train_k_range)
            object_features, ee_features, point_features = build_observation_from_state(
                shape, ee_position, episode.ee_yaw, s_values
            )
            batch = _pad_samples(
                [{"object_features": object_features, "ee_features": ee_features, "point_features": point_features}],
                device,
            )
            target = expert_action_from_state(ee_position, episode, config)
            target_yaw_action = float(target[3])
            episode_models: dict[str, dict[str, float]] = {}
            for name, model in models.items():
                prediction = model(batch)[0].detach().cpu()
                yaw_prediction = float(prediction[3])
                yaw_error = abs(yaw_prediction - target_yaw_action)
                sign_matches = float(np.sign(yaw_prediction) == np.sign(target_yaw_action))
                metrics = {
                    "translation_action_l2": float((prediction[:3] - target[:3]).norm().item()),
                    "rotation_action_abs_error": yaw_error,
                    "rotation_sign_accuracy": sign_matches,
                    "predicted_rotation_abs": abs(yaw_prediction),
                    "target_rotation_abs": abs(target_yaw_action),
                    "predicted_rotation_clipped": float(abs(yaw_prediction) >= config.max_delta_yaw),
                }
                episode_models[name] = metrics
                for metric, value in metrics.items():
                    per_bin[label].setdefault(f"{name}:{metric}", []).append(value)
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "shape_id": shape.shape_id,
                    "bin": label,
                    "relative_yaw": relative_yaw,
                    "relative_yaw_abs": abs(relative_yaw),
                    "target_yaw_action": target_yaw_action,
                    "required_yaw_steps": int(math.ceil(abs(relative_yaw) / config.max_delta_yaw)),
                    "models": episode_models,
                }
            )

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    metrics = (
        "translation_action_l2",
        "rotation_action_abs_error",
        "rotation_sign_accuracy",
        "predicted_rotation_abs",
        "target_rotation_abs",
        "predicted_rotation_clipped",
    )
    for label in per_bin:
        summaries[label] = {}
        for name in model_names:
            summaries[label][name] = {
                metric: float(np.mean(per_bin[label].get(f"{name}:{metric}", [0.0]))) for metric in metrics
            }
    max_required_steps = max(int(row["required_yaw_steps"]) for row in rows) if rows else 0
    return {
        "device": str(device),
        "episodes_per_bin": sanity_config.sweep_episodes_per_bin,
        "bins": summaries,
        "max_required_yaw_steps": max_required_steps,
        "rows": rows,
    }


def _make_orientation_rollout_episodes(
    shapes: list[PhysicalShape],
    config: GeneralizedGeometryConfig,
    sanity_config: OrientationSanityConfig,
    seed: int,
) -> list[GeneralizedEpisode]:
    rng = np.random.default_rng(seed)
    episodes: list[GeneralizedEpisode] = []
    for episode_id in range(sanity_config.rollout_episodes):
        shape = shapes[episode_id % len(shapes)]
        episode, _ = _sample_orientation_episode(
            shape,
            episode_id,
            rng,
            config,
            config.relative_ood_ee_yaw_abs_range,
            "relative_orientation_ood",
        )
        episodes.append(episode)
    return episodes


def evaluate_fixed_episodes(
    models: dict[str, nn.Module],
    episodes: list[GeneralizedEpisode],
    config: GeneralizedGeometryConfig,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Roll out several models from exactly the same generated episode specs."""
    envs = {name: MujocoManipulatorEnv(_env_config(config), seed=seed + idx) for idx, name in enumerate(models)}
    results: list[dict[str, Any]] = []
    try:
        for episode in episodes:
            model_results = {}
            for name, model in models.items():
                episode_sampling_rng = np.random.default_rng(seed + episode.episode_id * 100003 + 17)
                model_results[name] = rollout_model(model, envs[name], episode, config, episode_sampling_rng, False, device)
            _, target_yaw = interaction_frame(episode.shape, config.target_s)
            results.append(
                {
                    "episode_id": episode.episode_id,
                    "shape": asdict(episode.shape),
                    "ee_yaw": episode.ee_yaw,
                    "relative_yaw": float(wrap_angle(target_yaw - episode.ee_yaw)),
                    "models": model_results,
                }
            )
    finally:
        for env in envs.values():
            env.close()
    summaries = {name: _summarize_rollout_results([row["models"][name] for row in results]) for name in models}
    paired = {}
    model_names = list(models)
    if len(model_names) >= 2:
        for index, (baseline, candidate) in enumerate(zip(model_names, model_names[1:])):
            paired[f"{baseline}_vs_{candidate}"] = _paired_summary(
                results, baseline, candidate, bootstrap_samples=1000, seed=seed + index
            )
    return {"num_episodes": len(results), "models": summaries, "paired": paired, "episodes": results}


def _load_source_generalized_config(source_output_dir: Path) -> GeneralizedGeometryConfig:
    payload = json.loads((source_output_dir / "config.json").read_text(encoding="utf-8"))
    fields = GeneralizedGeometryConfig.__dataclass_fields__
    return GeneralizedGeometryConfig(**{key: value for key, value in payload.items() if key in fields})


def _aggregate_orientation_sweep(seed_payloads: list[dict[str, Any]], model_names: tuple[str, ...], bins: tuple[tuple[float, float], ...]) -> dict[str, Any]:
    aggregate_bins: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for bounds in bins:
        label = _orientation_bin_label(bounds)
        aggregate_bins[label] = {}
        for name in model_names:
            rows = [payload["orientation_sweep"]["bins"][label][name] for payload in seed_payloads]
            aggregate_bins[label][name] = {
                metric: _mean_std([float(row[metric]) for row in rows]) for metric in rows[0]
            }
    wide_rows = [payload["wide_eval"] for payload in seed_payloads]
    wide_models = tuple(wide_rows[0]["models"])
    wide_summary = {
        name: {
            metric: _mean_std([float(row["models"][name][metric]) for row in wide_rows])
            for metric in (
                "success_rate",
                "final_position_error",
                "final_orientation_error",
                "mean_position_error",
                "mean_orientation_error",
                "steps",
                "forward_latency_ms",
                "policy_latency_ms",
            )
        }
        for name in wide_models
    }
    combined_episodes: list[dict[str, Any]] = []
    for row in wide_rows:
        combined_episodes.extend(row["episodes"])
    paired = {}
    for index, (baseline, candidate) in enumerate(zip(wide_models, wide_models[1:])):
        paired[f"{baseline}_vs_{candidate}"] = _paired_summary(
            combined_episodes, baseline, candidate, bootstrap_samples=1000, seed=12000 + index
        )
    max_required = [int(payload["orientation_sweep"]["max_required_yaw_steps"]) for payload in seed_payloads]
    return {
        "bins": aggregate_bins,
        "wide_eval": wide_summary,
        "wide_eval_paired": paired,
        "max_required_yaw_steps": _mean_std([float(value) for value in max_required]),
        "seeds": seed_payloads,
    }


def rotation_wrapping_sanity() -> dict[str, float]:
    """Check periodic aliases and the sin/cos boundary encoding used by observations."""
    angles = np.linspace(-math.pi, math.pi, 257, dtype=np.float64)
    alias_error = max(abs(float(wrap_angle(float(angle + 2.0 * math.pi))) - float(wrap_angle(float(angle)))) for angle in angles)
    epsilon = 1e-5
    boundary_a = np.array([math.sin(math.pi - epsilon), math.cos(math.pi - epsilon)])
    boundary_b = np.array([math.sin(-math.pi + epsilon), math.cos(-math.pi + epsilon)])
    return {
        "max_two_pi_alias_error": float(alias_error),
        "sin_cos_boundary_gap": float(np.linalg.norm(boundary_a - boundary_b)),
    }


def orientation_sanity_summary_markdown(payload: dict[str, Any], config: OrientationSanityConfig, source_config: GeneralizedGeometryConfig) -> str:
    lines = [
        "# Relative Orientation OOD Sanity Check",
        "",
        "This follow-up keeps the existing architectures fixed. It first bins one-step yaw behavior of the existing narrow-range checkpoints, then retrains only Local Set Attention with a widened EE-yaw training range.",
        "",
        f"- Narrow train EE yaw range: `{source_config.train_ee_yaw_range}`",
        f"- Widened train EE yaw range: `{config.wide_train_ee_yaw_range}`",
        f"- Max yaw action: `{source_config.max_delta_yaw}` rad; rollout horizon: `{source_config.max_steps}` steps",
        f"- Maximum required yaw steps observed: `{payload['max_required_yaw_steps']['mean']:.1f}`",
        f"- Evaluation device: `{payload['seeds'][0]['orientation_sweep']['device']}`",
        f"- Rotation wrap alias error: `{payload['rotation_wrapping']['max_two_pi_alias_error']:.2e}`; sin/cos boundary gap: `{payload['rotation_wrapping']['sin_cos_boundary_gap']:.2e}`",
        "",
        "## Narrow Checkpoint: Orientation Bins",
        "",
        "`rotation_sign` is the fraction of predictions with the correct signed yaw direction; `clip_frac` is the fraction of raw model outputs at or beyond the action clip.",
        "",
        "| Relative yaw abs. | Model | Yaw action error | Sign accuracy | Pred. abs yaw | Target abs yaw | Clip frac |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    model_names = tuple(payload["seeds"][0]["orientation_sweep"]["bins"][next(iter(payload["seeds"][0]["orientation_sweep"]["bins"]))])
    for bounds in config.orientation_bins:
        label = _orientation_bin_label(bounds)
        for name in model_names:
            row = payload["bins"][label][name]
            lines.append(
                f"| {label} | {name} | {row['rotation_action_abs_error']['mean']:.4f} +/- {row['rotation_action_abs_error']['std']:.4f} | {row['rotation_sign_accuracy']['mean']:.3f} +/- {row['rotation_sign_accuracy']['std']:.3f} | {row['predicted_rotation_abs']['mean']:.3f} | {row['target_rotation_abs']['mean']:.3f} | {row['predicted_rotation_clipped']['mean']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Narrow vs Wide Set Attention Rollout",
            "",
            "| Model | Success | Final position error | Final orientation error | Policy ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in payload["wide_eval"].items():
        lines.append(
            f"| {name} | {row['success_rate']['mean']:.3f} +/- {row['success_rate']['std']:.3f} | {row['final_position_error']['mean']:.4f} +/- {row['final_position_error']['std']:.4f} | {row['final_orientation_error']['mean']:.4f} +/- {row['final_orientation_error']['std']:.4f} | {row['policy_latency_ms']['mean']:.3f} |"
        )
    for name, row in payload["wide_eval_paired"].items():
        counts = row["contingency"]
        delta = row["delta_final_position"]
        lines.extend(
            [
                "",
                f"- `{name}`: both={counts['both_success']}, baseline-only={counts['baseline_only_success']}, candidate-only={counts['candidate_only_success']}, both-fail={counts['both_fail']}; paired final-position delta mean={delta['mean']:.5f}, CI95=[{delta['ci95'][0]:.5f}, {delta['ci95'][1]:.5f}], McNemar p={row['mcnemar_exact_p']:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The yaw action is deliberately clipped to a single-step maximum. The largest tested relative orientation therefore needs several valid control steps, but it is not unreachable within the rollout horizon. If the wide-range checkpoint recovers while the narrow checkpoint's signed yaw predictions degrade by bin, the original Relative Orientation OOD collapse is a training-coverage/extrapolation issue rather than evidence against Set Attention or GNN structure.",
            "",
        ]
    )
    return "\n".join(lines)


def run_orientation_sanity(config: OrientationSanityConfig) -> dict[str, Any]:
    source_dir = Path(config.source_output_dir)
    output_dir = ensure_dir(config.output_dir)
    source_config = _load_source_generalized_config(source_dir)
    (output_dir / "config.json").write_text(
        json.dumps({"orientation_sanity": asdict(config), "source_config": asdict(source_config)}, indent=2),
        encoding="utf-8",
    )
    device = select_device(config.eval_device)
    model_names: tuple[GeneralizedModelName, ...] = (
        "local_concat",
        "local_set_attention",
        "geometric_gnn",
        "geometric_gnn_no_local",
    )
    seed_payloads = []
    for seed in config.seeds:
        seed_dir = ensure_dir(output_dir / f"seed{seed}")
        dataset = torch.load(source_dir / f"seed{seed}" / "dataset.pt", map_location="cpu", weights_only=False)
        shapes = [PhysicalShape(**item) for item in dataset["physical_shapes"]["test"]]
        models = {
            name: _load_model(source_dir / f"seed{seed}" / "checkpoints" / f"{name}.pt", source_config, device)[0]
            for name in model_names
        }
        orientation_sweep = orientation_action_sweep(models, shapes, source_config, config, seed + 7000, device)
        (seed_dir / "orientation_sweep.json").write_text(json.dumps(orientation_sweep, indent=2), encoding="utf-8")

        wide_config = replace(
            source_config,
            train_ee_yaw_range=config.wide_train_ee_yaw_range,
            epochs=config.wide_epochs,
            batch_size=config.batch_size,
            device=config.device,
            eval_device=config.eval_device,
            output_dir=str(output_dir / f"seed{seed}" / "wide_training"),
        )
        wide_dataset = generate_generalized_dataset(wide_config, seed)
        torch.save(wide_dataset, seed_dir / "wide_dataset.pt")
        wide_training = train_model(
            "local_set_attention",
            wide_dataset["splits"]["train"],
            wide_dataset["splits"]["val"],
            wide_dataset["splits"]["test"],
            wide_config,
            seed,
            seed_dir / "wide_checkpoints",
        )
        narrow_set, _ = _load_model(
            source_dir / f"seed{seed}" / "checkpoints" / "local_set_attention.pt", source_config, device
        )
        wide_set = wide_training["model"].to(device).eval()
        rollout_episodes = _make_orientation_rollout_episodes(shapes, source_config, config, seed + 8000)
        wide_eval = evaluate_fixed_episodes(
            {"narrow_set": narrow_set, "wide_set": wide_set},
            rollout_episodes,
            source_config,
            seed + 9000,
            device,
        )
        wide_eval["episode_specs"] = [asdict(episode) for episode in rollout_episodes]
        (seed_dir / "wide_eval.json").write_text(json.dumps(wide_eval, indent=2), encoding="utf-8")
        seed_payloads.append(
            {
                "seed": seed,
                "orientation_sweep": orientation_sweep,
                "wide_eval": wide_eval,
                "wide_parameter_count": int(wide_training["summary"]["parameter_count"]),
                "wide_offline": wide_training["summary"]["offline"],
            }
        )

    aggregate = _aggregate_orientation_sweep(seed_payloads, model_names, config.orientation_bins)
    aggregate["rotation_wrapping"] = rotation_wrapping_sanity()
    (output_dir / "aggregated_results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(orientation_sanity_summary_markdown(aggregate, config, source_config), encoding="utf-8")
    return {"output_path": str(output_dir / "aggregated_results.json"), "summary_path": str(output_dir / "summary.md"), **aggregate}


def analytic_observed_action(
    env: MujocoManipulatorEnv,
    episode: GeneralizedEpisode,
    s_values: torch.Tensor,
    config: GeneralizedGeometryConfig,
) -> torch.Tensor:
    object_features, ee_features, point_features = build_observation(env, episode, s_values)
    del object_features, ee_features
    s = point_features[:, 3]
    xyz = point_features[:, :3]
    target_s = torch.tensor(config.target_s, dtype=s.dtype)
    right = int(torch.searchsorted(s, target_s).clamp(0, len(s) - 1).item())
    left = max(right - 1, 0)
    if right == left:
        target = xyz[left]
    else:
        alpha = (target_s - s[left]) / (s[right] - s[left]).clamp_min(1e-6)
        target = xyz[left] + alpha * (xyz[right] - xyz[left])
    tangent = xyz[min(right + 1, len(s) - 1)] - xyz[max(left - 1, 0)]
    target_yaw_local = math.atan2(float(tangent[2]), float(tangent[0]))
    delta = clamp_delta(target, config.max_delta_ee)
    return torch.cat([delta, torch.tensor([max(-config.max_delta_yaw, min(config.max_delta_yaw, target_yaw_local))])])


def run_sanity_checks(config: GeneralizedGeometryConfig, shapes: list[PhysicalShape], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    env = MujocoManipulatorEnv(_env_config(config), seed=seed)
    oracle_success = []
    analytic_success = []
    sampling_variance = []
    try:
        for idx, shape in enumerate(shapes):
            episode = _sample_episode(shape, idx, rng, config, "iid")
            _reset_env(env, episode, config)
            current = episode
            for _ in range(config.max_steps):
                action = expert_action_local(env, current, config)
                current = apply_local_action(env, current, action, config)
                _, _, success = _success_metrics(env, current, config)
                if success:
                    break
            oracle_success.append(float(success))
            _reset_env(env, episode, config)
            current = episode
            for _ in range(config.max_steps):
                s_values = sample_observation_parameters(rng, k_range=config.train_k_range)
                action = analytic_observed_action(env, current, s_values, config)
                current = apply_local_action(env, current, action, config)
                _, _, success = _success_metrics(env, current, config)
                if success:
                    break
            analytic_success.append(float(success))
            _reset_env(env, episode, config)
            actions = []
            for _ in range(8):
                s_values = sample_observation_parameters(rng, k_range=config.train_k_range)
                object_features, ee_features, point_features = build_observation(env, episode, s_values)
                actions.append(torch.cat([point_features.mean(dim=0), object_features[:1], ee_features]))
            stacked = torch.stack(actions)
            sampling_variance.append(float(stacked.var(dim=0).mean().item()))
    finally:
        env.close()
    return {
        "oracle_success_rate": float(np.mean(oracle_success)),
        "analytic_observed_success_rate": float(np.mean(analytic_success)),
        "raw_observation_sampling_feature_variance": float(np.mean(sampling_variance)),
    }


@torch.no_grad()
def sampling_consistency_check(
    models: dict[str, nn.Module],
    shapes: list[PhysicalShape],
    config: GeneralizedGeometryConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    action_variance: dict[str, list[float]] = {name: [] for name in models}
    pairwise_difference: dict[str, list[float]] = {name: [] for name in models}
    for shape in shapes:
        ee_position = torch.tensor([0.0, 0.0, 0.52], dtype=torch.float32)
        actions_by_model = {name: [] for name in models}
        sampling_sets = [sample_observation_parameters(rng, k_range=config.train_k_range) for _ in range(8)]
        for s_values in sampling_sets:
            object_features, ee_features, point_features = build_observation_from_state(
                shape, ee_position, 0.15, s_values
            )
            batch = _pad_samples(
                [{"object_features": object_features, "ee_features": ee_features, "point_features": point_features}],
                device,
            )
            for name, model in models.items():
                actions_by_model[name].append(model(batch)[0].detach().cpu())
        for name, action_rows in actions_by_model.items():
            stacked = torch.stack(action_rows)
            action_variance[name].append(float(stacked.var(dim=0, unbiased=False).mean().item()))
            differences = []
            for i in range(len(action_rows)):
                for j in range(i + 1, len(action_rows)):
                    differences.append(float((action_rows[i] - action_rows[j]).norm().item()))
            pairwise_difference[name].append(float(np.mean(differences)))
    return {
        "mean_action_variance": {name: float(np.mean(values)) for name, values in action_variance.items()},
        "mean_pairwise_action_difference": {name: float(np.mean(values)) for name, values in pairwise_difference.items()},
    }


@torch.no_grad()
def rigid_transform_equivariance_check(
    models: dict[str, nn.Module],
    shapes: list[PhysicalShape],
    config: GeneralizedGeometryConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    errors: dict[str, list[float]] = {name: [] for name in models}
    for shape in shapes:
        global_yaw = float(rng.uniform(-math.pi, math.pi))
        translation = torch.tensor([float(rng.uniform(-0.1, 0.1)), 0.0, float(rng.uniform(-0.1, 0.1))])
        center = rotate_xz(torch.tensor(shape.object_center).reshape(1, 3), global_yaw)[0] + translation
        transformed = PhysicalShape(
            shape_id=shape.shape_id + "_rigid",
            object_center=tuple(float(v) for v in center),
            object_yaw=float(wrap_angle(shape.object_yaw + global_yaw)),
            part_offset=shape.part_offset,
            part_yaw=shape.part_yaw,
            length=shape.length,
            curvature=shape.curvature,
            bend=shape.bend,
            asymmetry=shape.asymmetry,
        )
        ee_position = torch.tensor([0.0, 0.0, 0.52])
        transformed_ee = rotate_xz(ee_position.reshape(1, 3), global_yaw)[0] + translation
        s_values = sample_observation_parameters(rng, k_range=config.train_k_range)
        base_features = build_observation_from_state(shape, ee_position, 0.15, s_values)
        transformed_features = build_observation_from_state(
            transformed, transformed_ee, float(wrap_angle(0.15 + global_yaw)), s_values
        )
        base_batch = _pad_samples(
            [{"object_features": base_features[0], "ee_features": base_features[1], "point_features": base_features[2]}], device
        )
        transformed_batch = _pad_samples(
            [{"object_features": transformed_features[0], "ee_features": transformed_features[1], "point_features": transformed_features[2]}], device
        )
        for name, model in models.items():
            difference = (model(base_batch)[0] - model(transformed_batch)[0]).abs().max().item()
            errors[name].append(float(difference))
    return {"max_abs_action_difference": {name: float(max(values)) for name, values in errors.items()}}


def _plot_generalized_episode(
    episode: dict[str, Any],
    config: GeneralizedGeometryConfig,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    s = torch.linspace(0.0, 1.0, 100)
    curve, _ = curve_world(s, PhysicalShape(**episode["shape"]))
    target, _ = interaction_frame(PhysicalShape(**episode["shape"]), config.target_s)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(curve[:, 0], curve[:, 2], "k-", linewidth=1.5, label="continuous part")
    ax.scatter([target[0]], [target[2]], marker="*", s=120, label="interaction target")
    for name, result in episode["models"].items():
        trajectory = np.asarray(result["trajectory"], dtype=float)
        ax.plot(trajectory[:, 0], trajectory[:, 2], "-o", markersize=2, linewidth=1.0, label=name)
    ax.set_xlabel("world x")
    ax.set_ylabel("world z")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_generalized_experiment(config: GeneralizedGeometryConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    model_names: tuple[GeneralizedModelName, ...] = ("local_concat", "local_set_attention", "geometric_gnn", "geometric_gnn_no_local")
    seed_payloads = []
    for seed in config.seeds:
        seed_dir = ensure_dir(output_dir / f"seed{seed}")
        dataset = generate_generalized_dataset(config, seed)
        torch.save(dataset, seed_dir / "dataset.pt")
        dataset_dir = ensure_dir(output_dir / "datasets")
        torch.save(dataset, dataset_dir / f"seed{seed}.pt")
        models: dict[str, nn.Module] = {}
        summaries = {}
        for name in model_names:
            trained = train_model(
                name,
                dataset["splits"]["train"],
                dataset["splits"]["val"],
                dataset["splits"]["test"],
                config,
                seed,
                seed_dir / "checkpoints",
            )
            models[name] = trained["model"]
            summaries[name] = {key: value for key, value in trained["summary"].items() if key != "model_state"}
        test_shapes = [PhysicalShape(**item) for item in dataset["physical_shapes"]["test"]]
        iid_shapes = test_shapes[: config.eval_shapes]
        shape_ood_shapes = [_sample_shape(config, np.random.default_rng(seed + 777 + idx), f"shape_ood_{idx}", ood=True) for idx in range(config.eval_shapes)]
        orientation_shapes = test_shapes[: config.eval_shapes]
        condition_results = {}
        device = select_device(config.eval_device)
        for condition, shapes in (("iid", iid_shapes), ("sampling_ood", iid_shapes), ("shape_ood", shape_ood_shapes), ("relative_orientation_ood", orientation_shapes)):
            condition_results[condition] = evaluate_condition(models, config, shapes, condition, seed + 3000, device)
            condition_dir = ensure_dir(seed_dir / "eval" / condition)
            (condition_dir / "results.json").write_text(json.dumps(condition_results[condition], indent=2), encoding="utf-8")
            if config.plot_cases > 0:
                plots_dir = ensure_dir(seed_dir / "plots" / condition)
                plot_episodes = sorted(
                    condition_results[condition]["episodes"],
                    key=lambda item: max(result["final_position_error"] for result in item["models"].values()),
                    reverse=True,
                )[: config.plot_cases]
                for episode in plot_episodes:
                    _plot_generalized_episode(
                        episode,
                        config,
                        plots_dir / f"episode_{int(episode['episode_id']):04d}.png",
                    )
        sanity = run_sanity_checks(config, iid_shapes[: min(8, len(iid_shapes))], seed + 5000)
        sanity["sampling_consistency"] = sampling_consistency_check(
            models, iid_shapes[: min(8, len(iid_shapes))], config, device, seed + 6000
        )
        sanity["rigid_transform"] = rigid_transform_equivariance_check(
            models, iid_shapes[: min(8, len(iid_shapes))], config, device, seed + 7000
        )
        (seed_dir / "sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
        seed_payloads.append({"seed": seed, "summaries": summaries, "conditions": condition_results, "sanity": sanity})
    aggregate = _aggregate_generalized(seed_payloads, model_names)
    (output_dir / "aggregated_results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(generalized_summary_markdown(aggregate), encoding="utf-8")
    return {"output_path": str(output_dir / "aggregated_results.json"), "summary_path": str(output_dir / "summary.md"), **aggregate}


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def _aggregate_generalized(seed_payloads: list[dict[str, Any]], model_names: tuple[str, ...]) -> dict[str, Any]:
    conditions = ("iid", "sampling_ood", "shape_ood", "relative_orientation_ood")
    aggregate_conditions = {}
    aggregate_paired = {}
    for condition in conditions:
        aggregate_conditions[condition] = {}
        for name in model_names:
            rows = [item["conditions"][condition]["models"][name] for item in seed_payloads]
            aggregate_conditions[condition][name] = {
                metric: _mean_std([float(row[metric]) for row in rows])
                for metric in ("success_rate", "final_position_error", "final_orientation_error", "mean_position_error", "mean_orientation_error", "forward_latency_ms", "policy_latency_ms")
            }
        combined_episodes = []
        for item in seed_payloads:
            combined_episodes.extend(item["conditions"][condition]["episodes"])
        aggregate_paired[condition] = {
            f"{baseline}_vs_{candidate}": _paired_summary(
                combined_episodes,
                baseline,
                candidate,
                bootstrap_samples=1000,
                seed=9100 + pair_idx,
            )
            for pair_idx, (baseline, candidate) in enumerate(
                (
                    ("local_concat", "local_set_attention"),
                    ("local_set_attention", "geometric_gnn"),
                    ("local_concat", "geometric_gnn"),
                    ("geometric_gnn", "geometric_gnn_no_local"),
                )
            )
        }
    sampling_consistency = {
        metric: {
            name: _mean_std([float(item["sanity"]["sampling_consistency"][metric][name]) for item in seed_payloads])
            for name in model_names
        }
        for metric in ("mean_action_variance", "mean_pairwise_action_difference")
    }
    rigid_transform = {
        name: _mean_std([float(item["sanity"]["rigid_transform"]["max_abs_action_difference"][name]) for item in seed_payloads])
        for name in model_names
    }
    offline_test = {}
    for name in model_names:
        rows = [item["summaries"][name]["offline"]["test"] for item in seed_payloads]
        offline_test[name] = {
            metric: _mean_std([float(row[metric]) for row in rows])
            for metric in ("translation_l2", "rotation_abs_error")
        }
    return {
        "seeds": seed_payloads,
        "conditions": aggregate_conditions,
        "paired": aggregate_paired,
        "parameter_counts": {
            name: [int(item["summaries"][name]["parameter_count"]) for item in seed_payloads] for name in model_names
        },
        "offline_test": offline_test,
        "sanity": {
            key: _mean_std([float(item["sanity"][key]) for item in seed_payloads])
            for key in ("oracle_success_rate", "analytic_observed_success_rate", "raw_observation_sampling_feature_variance")
        },
        "sampling_consistency": sampling_consistency,
        "rigid_transform": rigid_transform,
    }


def generalized_summary_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Generalized Geometric Graph Feasibility", "", "The current MuJoCo action convention is x-z planar translation plus a synthetic planar yaw state; all neural models receive EE-local observed geometry only.", "", "## Success", "", "| Model | IID | Sampling OOD | Shape OOD | Relative Orientation OOD |", "|---|---:|---:|---:|---:|"]
    for name in ("local_concat", "local_set_attention", "geometric_gnn", "geometric_gnn_no_local"):
        values = [payload["conditions"][condition][name]["success_rate"] for condition in ("iid", "sampling_ood", "shape_ood", "relative_orientation_ood")]
        lines.append("| " + name + " | " + " | ".join(f"{value['mean']:.3f} +/- {value['std']:.3f}" for value in values) + " |")
    lines.extend(["", "## Efficiency", "", "| Model | Params | Forward ms (IID) | Policy ms (IID) |", "|---|---:|---:|---:|"])
    for name in ("local_concat", "local_set_attention", "geometric_gnn", "geometric_gnn_no_local"):
        params = payload["parameter_counts"][name]
        row = payload["conditions"]["iid"][name]
        lines.append(f"| {name} | {params} | {row['forward_latency_ms']['mean']:.3f} | {row['policy_latency_ms']['mean']:.3f} |")
    lines.extend(["", "## Offline Test", "", "| Model | Translation L2 | Rotation error |", "|---|---:|---:|"])
    for name in ("local_concat", "local_set_attention", "geometric_gnn", "geometric_gnn_no_local"):
        row = payload["offline_test"][name]
        lines.append(f"| {name} | {row['translation_l2']['mean']:.5f} +/- {row['translation_l2']['std']:.5f} | {row['rotation_abs_error']['mean']:.5f} +/- {row['rotation_abs_error']['std']:.5f} |")
    lines.extend(["", "## Sanity Baselines", "", json.dumps(payload["sanity"], indent=2), ""])
    lines.extend(["## Sampling Consistency", "", "| Model | Action variance | Pairwise action difference | Rigid-transform max error |", "|---|---:|---:|---:|"])
    for name in ("local_concat", "local_set_attention", "geometric_gnn", "geometric_gnn_no_local"):
        variance = payload["sampling_consistency"]["mean_action_variance"][name]["mean"]
        difference = payload["sampling_consistency"]["mean_pairwise_action_difference"][name]["mean"]
        rigid = payload["rigid_transform"][name]["mean"]
        lines.append(f"| {name} | {variance:.6f} | {difference:.6f} | {rigid:.8f} |")
    lines.extend(["", "## Pooled IID Paired Comparisons", ""])
    for name, comparison in payload["paired"]["iid"].items():
        counts = comparison["contingency"]
        delta = comparison["delta_final_position"]
        lines.append(
            f"- `{name}`: both={counts['both_success']}, baseline-only={counts['baseline_only_success']}, candidate-only={counts['candidate_only_success']}, both-fail={counts['both_fail']}; delta position mean={delta['mean']:.5f}, CI95=[{delta['ci95'][0]:.5f}, {delta['ci95'][1]:.5f}], McNemar p={comparison['mcnemar_exact_p']:.4f}"
        )
    lines.append("")
    return "\n".join(lines)
