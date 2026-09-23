"""Surface-aware pre-grasp feasibility study.

The task uses analytic, sparse surface observations and the planar x-z/wrist-yaw
convention of the existing MuJoCo manipulator.  Collision checks use MuJoCo
primitive geoms; target frames are only used by the expert and evaluator.
"""

from __future__ import annotations

import json
import hashlib
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import torch
from torch import nn

from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv, MujocoReachConfig
from vla_gnn_recurrent.training.generalized_geometric_graph import rotate_xz, wrap_angle
from vla_gnn_recurrent.utils import DevicePreference, clamp_delta, ensure_dir, select_device, set_seed

SurfaceModelName = Literal["centerline_set", "surface_set", "surface_graph", "surface_graph_no_local"]
MODEL_NAMES: tuple[SurfaceModelName, ...] = (
    "centerline_set", "surface_set", "surface_graph", "surface_graph_no_local"
)
SURFACE_POINTS = 32
GRAPH_K = 6
GRAPH_RADIUS = 0.040
TIP_K = 8
ACTION_DIM = 5
STATE_DIM = 14
MIN_GRIPPER_WIDTH = 0.065
MAX_GRIPPER_WIDTH = 0.105


@dataclass
class SurfaceFeasibilityConfig:
    output_dir: str = "artifacts/surface_graph_feasibility"
    seeds: tuple[int, ...] = (2811, 2812, 2813)
    train_episodes: int = 72
    validation_episodes: int = 16
    eval_episodes: int = 16
    smoke_episodes: int = 4
    max_steps: int = 20
    epochs: int = 20
    smoke_epochs: int = 2
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    attention_heads: int = 4
    message_passing_steps: int = 3
    graph_k: int = GRAPH_K
    graph_radius: float = GRAPH_RADIUS
    point_count: int = SURFACE_POINTS
    max_delta_ee: float = 0.035
    max_delta_rotation: float = 0.35
    max_delta_gripper: float = 0.25
    pregrasp_clearance: float = 0.075
    success_position_threshold: float = 0.025
    success_rotation_threshold: float = 0.20
    success_opening_threshold: float = 0.008
    device: DevicePreference = "auto"
    eval_device: DevicePreference = "auto"
    bootstrap_samples: int = 2000
    latency_warmup: int = 200
    latency_iterations: int = 2000
    control_substeps: int = 10
    training_object_x_range: tuple[float, float] = (-0.12, 0.12)
    training_object_z_range: tuple[float, float] = (0.54, 0.72)
    training_length_range: tuple[float, float] = (0.15, 0.22)
    training_half_width_range: tuple[float, float] = (0.028, 0.041)
    training_depth_range: tuple[float, float] = (0.010, 0.020)
    training_curvature_abs: float = 0.012
    training_asymmetry_abs: float = 0.22
    ood_half_width_range: tuple[float, float] = (0.024, 0.043)
    ood_curvature_abs: float = 0.022
    ood_asymmetry_abs: float = 0.42
    ood_depth_range: tuple[float, float] = (0.007, 0.023)
    plot_cases: int = 4


@dataclass(frozen=True)
class SurfaceShape:
    shape_id: str
    object_center: tuple[float, float, float]
    object_yaw: float
    part_yaw: float
    length: float
    curvature: float
    bend: float
    half_width: float
    half_depth: float
    thickness_amplitude: float
    thickness_phase: float
    asymmetry: float
    cross_section: str


@dataclass(frozen=True)
class SurfaceEpisodeSpec:
    episode_id: int
    shape: SurfaceShape
    arm_qpos: tuple[float, float, float, float]
    gripper_command: float
    condition: str = "iid"
    sample_identity: str = ""


@dataclass
class GraphTopology:
    src: torch.Tensor
    dst: torch.Tensor
    edge_type: torch.Tensor
    valid: torch.Tensor
    surface_pairs: torch.Tensor

    def to(self, device: torch.device | str) -> "GraphTopology":
        return GraphTopology(
            self.src.to(device), self.dst.to(device), self.edge_type.to(device),
            self.valid.to(device), self.surface_pairs.to(device),
        )


def _surface_scene_xml_path(output_dir: str | Path) -> Path:
    out = ensure_dir(Path(output_dir) / "geometry")
    path = out / "manipulator_surface_primitives_v2.xml"
    if path.exists():
        return path
    source = Path(__import__("dm_control").__file__).parent / "suite" / "manipulator.xml"
    xml = source.read_text(encoding="utf-8")
    common_dir = (source.parent / "common").resolve()
    xml = xml.replace('file="./common/', f'file="{common_dir}/')
    geoms: list[str] = ["\n    <!-- Surface task analytic collision proxies, updated per episode. -->"]
    for i in range(8):
        geoms.append(
            f'    <body name="surface_capsule_body_{i}" mocap="true">'
            f'<geom name="surface_capsule_{i}" type="capsule" size="0.03 0.01" contype="0" conaffinity="0" '
            'rgba="0.3 0.65 0.9 0.55"/></body>'
        )
        geoms.append(
            f'    <body name="surface_box_body_{i}" mocap="true">'
            f'<geom name="surface_box_{i}" type="box" size="0.01 0.01 0.03" contype="0" conaffinity="0" '
            'rgba="0.3 0.65 0.9 0.55"/></body>'
        )
    xml = xml.replace("  <worldbody>\n", "  <worldbody>\n" + "\n".join(geoms) + "\n", 1)
    path.write_text(xml, encoding="utf-8")
    return path


class SurfaceManipulatorEnv(MujocoManipulatorEnv):
    """Existing arm/gripper with parametric rigid surface collision geoms."""

    def __init__(self, config: MujocoReachConfig | None = None, seed: int = 0, output_dir: str | Path = "artifacts/surface_graph_feasibility") -> None:
        self.surface_scene_path = _surface_scene_xml_path(output_dir)
        super().__init__(config=config, seed=seed)
        self.surface_capsule_ids = np.array([self._geom_id(f"surface_capsule_{i}") for i in range(8)], dtype=np.int32)
        self.surface_box_ids = np.array([self._geom_id(f"surface_box_{i}") for i in range(8)], dtype=np.int32)
        self.surface_geom_ids = np.concatenate([self.surface_capsule_ids, self.surface_box_ids])
        self.surface_capsule_mocap_ids = np.array([self.model.body_mocapid[self._body_id(f"surface_capsule_body_{i}")] for i in range(8)], dtype=np.int32)
        self.surface_box_mocap_ids = np.array([self.model.body_mocapid[self._body_id(f"surface_box_body_{i}")] for i in range(8)], dtype=np.int32)
        # Include the full end-effector, palm, finger links, and fingertips.
        # The arm links are excluded: this pre-grasp experiment compares tool
        # clearance, and the stock articulated arm sweeps through the handle region.
        robot_roots = {int(self._body_id("hand"))}
        robot_geoms: list[int] = []
        for geom_id in range(self.model.ngeom):
            if geom_id in set(self.surface_geom_ids.tolist()):
                continue
            body_id = int(self.model.geom_bodyid[geom_id])
            while body_id > 0 and body_id not in robot_roots:
                body_id = int(self.model.body_parentid[body_id])
            if body_id in robot_roots:
                robot_geoms.append(geom_id)
        self.robot_collision_geom_ids = np.asarray(robot_geoms, dtype=np.int32)
        self._active_shape: SurfaceShape | None = None

    def _load_model(self) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_path(str(self.surface_scene_path))

    def set_surface_shape(self, shape: SurfaceShape) -> None:
        self._active_shape = shape
        for ids in (self.surface_capsule_mocap_ids, self.surface_box_mocap_ids):
            self.data.mocap_pos[ids] = np.array([2.0, 2.0, 2.0], dtype=np.float64)
            self.data.mocap_quat[ids] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        values = (np.arange(8, dtype=np.float64) + 0.5) / 8.0
        for i, s in enumerate(values):
            center, tangent = curve_world_np(s, shape)
            tangent = tangent / max(np.linalg.norm(tangent[[0, 2]]), 1e-8)
            normal = np.array([-tangent[2], 0.0, tangent[0]], dtype=np.float64)
            radius = profile_half_width(float(s), shape)
            offset = normal * shape.asymmetry * radius
            start, _ = curve_world_np(i / 8.0, shape)
            stop, _ = curve_world_np((i + 1) / 8.0, shape)
            half_length = max(float(np.linalg.norm((stop - start)[[0, 2]]) * 0.5), 0.005)
            if shape.cross_section == "flat":
                geom_id = int(self.surface_box_ids[i])
                self.model.geom_contype[geom_id] = 1
                self.model.geom_conaffinity[geom_id] = 1
                mocap_id = int(self.surface_box_mocap_ids[i])
                self.data.mocap_pos[mocap_id] = center + offset
                self.model.geom_size[geom_id] = np.array([half_length + 0.004, shape.half_depth, radius])
                basis = np.stack([tangent, np.array([0.0, 1.0, 0.0]), normal], axis=1)
                quat = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(quat, basis.reshape(-1))
                self.data.mocap_quat[mocap_id] = quat
                self.model.geom_contype[int(self.surface_capsule_ids[i])] = 0
                self.model.geom_conaffinity[int(self.surface_capsule_ids[i])] = 0
            else:
                geom_id = int(self.surface_capsule_ids[i])
                self.model.geom_contype[geom_id] = 1
                self.model.geom_conaffinity[geom_id] = 1
                mocap_id = int(self.surface_capsule_mocap_ids[i])
                self.data.mocap_pos[mocap_id] = center + offset
                self.model.geom_size[geom_id] = np.array([max(radius, shape.half_depth), half_length + 0.004, 0.0])
                # Capsule's local z axis follows the centerline.
                side = np.cross(tangent, normal)
                basis = np.stack([normal, side, tangent], axis=1)
                quat = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(quat, basis.reshape(-1))
                self.data.mocap_quat[mocap_id] = quat
                self.model.geom_contype[int(self.surface_box_ids[i])] = 0
                self.model.geom_conaffinity[int(self.surface_box_ids[i])] = 0
        mujoco.mj_forward(self.model, self.data)

    def contacts_with_surface(self) -> list[dict[str, Any]]:
        return _surface_distance_metrics(self, self._active_shape)[0] if self._active_shape is not None else []


def profile_half_width(s: float, shape: SurfaceShape) -> float:
    return float(shape.half_width * (1.0 + shape.thickness_amplitude * math.sin(2.0 * math.pi * s + shape.thickness_phase)))


def curve_world_np(s: float, shape: SurfaceShape) -> tuple[np.ndarray, np.ndarray]:
    u = float(s) - 0.5
    local = np.array([shape.length * u, 0.0, shape.curvature * u * u + shape.bend * math.sin(2.0 * math.pi * s)], dtype=np.float64)
    tangent_local = np.array([shape.length, 0.0, 2.0 * shape.curvature * u + 2.0 * math.pi * shape.bend * math.cos(2.0 * math.pi * s)], dtype=np.float64)
    theta = shape.object_yaw + shape.part_yaw
    c, s_ = math.cos(theta), math.sin(theta)
    rotate = np.array([[c, 0.0, -s_], [0.0, 1.0, 0.0], [s_, 0.0, c]], dtype=np.float64)
    center = np.asarray(shape.object_center, dtype=np.float64)
    return center + rotate @ local, rotate @ tangent_local


def curve_world(s_values: np.ndarray | torch.Tensor, shape: SurfaceShape) -> tuple[np.ndarray, np.ndarray]:
    rows = [curve_world_np(float(s), shape) for s in np.asarray(s_values).reshape(-1)]
    return np.stack([row[0] for row in rows]), np.stack([row[1] for row in rows])


def sample_surface_points_world(shape: SurfaceShape, count: int = SURFACE_POINTS, seed: int = 0, nonuniform: bool = False) -> np.ndarray:
    if count < 8 or count % 4:
        raise ValueError("Surface point count must be >=8 and divisible by 4.")
    rng = np.random.default_rng(seed)
    ring_count = 4
    longitudinal_count = count // ring_count
    t = (np.arange(longitudinal_count, dtype=np.float64) + 0.5) / longitudinal_count
    if nonuniform:
        t = np.power(t, 1.65)
    else:
        t = np.clip(t + rng.uniform(-0.015, 0.015, size=t.shape), 0.001, 0.999)
    points: list[np.ndarray] = []
    theta = shape.object_yaw + shape.part_yaw
    c, s_ = math.cos(theta), math.sin(theta)
    rot = np.array([[c, 0.0, -s_], [0.0, 1.0, 0.0], [s_, 0.0, c]], dtype=np.float64)
    for u in t:
        center, tangent = curve_world_np(float(u), shape)
        tangent /= max(np.linalg.norm(tangent[[0, 2]]), 1e-8)
        normal = np.array([-tangent[2], 0.0, tangent[0]], dtype=np.float64)
        width = profile_half_width(float(u), shape)
        asym_offset = normal * shape.asymmetry * width
        for j in range(ring_count):
            # The robot operates in the x-z plane. Sample the task-facing half
            # of each 3D cross section; this keeps the point graph local instead
            # of linking opposite front/back surfaces through a thin extrusion.
            angle = math.pi + math.pi * j / (ring_count - 1)
            cn, sy = math.cos(angle), math.sin(angle)
            if shape.cross_section == "flat":
                radial_n = width * cn / max(abs(cn), 0.65)
                radial_y = shape.half_depth * sy / max(abs(sy), 0.65)
            else:
                radial_n = width * cn
                radial_y = max(shape.half_depth, width * 0.70) * sy
            points.append(center + asym_offset + normal * radial_n + np.array([0.0, radial_y, 0.0]))
    return np.asarray(points, dtype=np.float32)


def sample_centerline_points_world(shape: SurfaceShape, count: int = SURFACE_POINTS) -> np.ndarray:
    s_values = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return curve_world(s_values, shape)[0].astype(np.float32)


def sample_shape(rng: np.random.Generator, shape_id: str, config: SurfaceFeasibilityConfig, ood: bool = False) -> SurfaceShape:
    width_range = config.ood_half_width_range if ood else config.training_half_width_range
    depth_range = config.ood_depth_range if ood else config.training_depth_range
    curvature_abs = config.ood_curvature_abs if ood else config.training_curvature_abs
    asym_abs = config.ood_asymmetry_abs if ood else config.training_asymmetry_abs
    cross_section = str(rng.choice(["flat", "rounded"] if ood else ["flat", "rounded", "flat"]))
    return SurfaceShape(
        shape_id=shape_id,
        object_center=(float(rng.uniform(*config.training_object_x_range)), 0.0, float(rng.uniform(*config.training_object_z_range))),
        object_yaw=float(rng.uniform(-0.30 if ood else -0.18, 0.30 if ood else 0.18)),
        part_yaw=float(rng.uniform(-0.55 if ood else -0.40, 0.55 if ood else 0.40)),
        length=float(rng.uniform(*config.training_length_range)),
        curvature=float(rng.uniform(-curvature_abs, curvature_abs)),
        bend=float(rng.uniform(-curvature_abs * 0.55, curvature_abs * 0.55)),
        half_width=float(rng.uniform(*width_range)),
        half_depth=float(rng.uniform(*depth_range)),
        thickness_amplitude=float(rng.uniform(0.0, 0.10 if ood else 0.06)),
        thickness_phase=float(rng.uniform(-math.pi, math.pi)),
        asymmetry=float(rng.uniform(-asym_abs, asym_abs)),
        cross_section=cross_section,
    )


def sample_episode_specs(count: int, seed: int, config: SurfaceFeasibilityConfig, condition: str = "iid", ood: bool = False) -> list[SurfaceEpisodeSpec]:
    rng = np.random.default_rng(seed)
    specs: list[SurfaceEpisodeSpec] = []
    for i in range(count):
        shape = sample_shape(rng, f"{condition}_{seed}_{i:04d}", config, ood=ood)
        q = np.array([0.0, -0.35, 0.95, -0.45], dtype=np.float64) + rng.uniform(-0.20, 0.20, size=4)
        identity = f"{shape.shape_id}:surface_sample_v1"
        specs.append(SurfaceEpisodeSpec(i, shape, tuple(float(x) for x in q), float(rng.uniform(0.0, 1.0)), condition, identity))
    return specs


def paired_surface_necessity_shapes(seed: int, config: SurfaceFeasibilityConfig) -> tuple[SurfaceShape, SurfaceShape]:
    rng = np.random.default_rng(seed)
    base = sample_shape(rng, "necessity_base", config)
    thin = replace(base, shape_id="necessity_thin", half_width=0.029, thickness_amplitude=0.02, asymmetry=-0.12, cross_section="flat")
    thick = replace(base, shape_id="necessity_thick", half_width=0.040, thickness_amplitude=0.10, asymmetry=0.25, cross_section="rounded")
    return thin, thick


def _surface_target(shape: SurfaceShape, ee_position: np.ndarray, clearance: float) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    center, tangent = curve_world_np(0.5, shape)
    tangent /= max(np.linalg.norm(tangent[[0, 2]]), 1e-8)
    normal = np.array([-tangent[2], 0.0, tangent[0]], dtype=np.float64)
    side_dot = float(np.dot((ee_position - center)[[0, 2]], normal[[0, 2]]))
    side = 1 if side_dot >= 0.0 else -1
    radius = profile_half_width(0.5, shape)
    signed_radius = radius * (1.0 + side * shape.asymmetry)
    offset = normal * shape.asymmetry * radius
    outward = normal * side
    surface_point = center + offset + normal * side * radius
    target_position = surface_point + clearance * outward
    target_yaw = math.atan2(float(outward[2]), float(outward[0]))
    opening_width = 2.0 * radius + 0.010
    return target_position, outward, target_yaw, opening_width, side


def tool_yaw(env: SurfaceManipulatorEnv) -> float:
    rotation = env.data.site_xmat[env.ee_site_id].reshape(3, 3)
    return float(wrap_angle(math.atan2(float(rotation[2, 0]), float(rotation[0, 0]))))


def gripper_width(env: SurfaceManipulatorEnv) -> float:
    left = np.mean([env.data.geom_xpos[env._geom_id(n)] for n in ("thumbtip1", "thumbtip2")], axis=0)
    right = np.mean([env.data.geom_xpos[env._geom_id(n)] for n in ("fingertip1", "fingertip2")], axis=0)
    jaw_axis = np.array([math.cos(tool_yaw(env)), 0.0, math.sin(tool_yaw(env))], dtype=np.float64)
    return float(abs(np.dot(left - right, jaw_axis)))


def opening_fraction(width: float) -> float:
    return float(np.clip((width - MIN_GRIPPER_WIDTH) / (MAX_GRIPPER_WIDTH - MIN_GRIPPER_WIDTH), 0.0, 1.0))


def opening_fraction_to_command(open_fraction: float) -> float:
    return float(np.clip(1.0 - open_fraction, 0.0, 1.0))


def _set_gripper_open_fraction_kinematic(env: SurfaceManipulatorEnv, open_fraction: float) -> None:
    # Native manipulator calibration: actuator command 0 (open) settles near q=-0.259,
    # and command 1 (closed) near q=+0.150.  The task uses this calibrated joint range.
    open_fraction = float(np.clip(open_fraction, 0.0, 1.0))
    q = 0.150 - 0.409 * open_fraction
    env.data.qpos[env.finger_qpos_ids] = q
    env.data.qvel[env.finger_dof_ids] = 0.0
    mujoco.mj_forward(env.model, env.data)


def localize_world(points: np.ndarray, ee_position: np.ndarray, ee_yaw: float) -> np.ndarray:
    tensor = torch.as_tensor(points, dtype=torch.float32)
    local = rotate_xz(tensor - torch.as_tensor(ee_position, dtype=torch.float32).reshape(1, 3), -ee_yaw)
    return local.cpu().numpy()


def world_action_from_local(local_delta: torch.Tensor | np.ndarray, yaw: float) -> np.ndarray:
    delta = torch.as_tensor(local_delta, dtype=torch.float32).reshape(1, 3)
    return rotate_xz(delta, yaw)[0].cpu().numpy()


def wrap_rotation_delta(delta: float) -> float:
    return float(wrap_angle(delta))


def equivalent_orientation_error(current_yaw: float, target_yaw: float) -> float:
    # The unique side rule removes physical 180-degree ambiguity; 2π aliases remain equivalent.
    return abs(float(wrap_angle(current_yaw - target_yaw)))


def clamp_gripper_action(delta: float, current_fraction: float, max_delta: float) -> tuple[float, float]:
    applied = float(np.clip(delta, -max_delta, max_delta))
    return applied, float(np.clip(current_fraction + applied, 0.0, 1.0))


def _common_state(env: SurfaceManipulatorEnv, shape: SurfaceShape, ee_pos: np.ndarray, yaw: float) -> np.ndarray:
    object_local = localize_world(np.asarray(shape.object_center, dtype=np.float32).reshape(1, 3), ee_pos, yaw)[0]
    rel_yaw = float(wrap_angle(shape.object_yaw - yaw))
    tip_world = np.stack([
        np.mean([env.data.geom_xpos[env._geom_id(n)] for n in ("thumbtip1", "thumbtip2")], axis=0),
        np.mean([env.data.geom_xpos[env._geom_id(n)] for n in ("fingertip1", "fingertip2")], axis=0),
    ]).astype(np.float32)
    tips_local = localize_world(tip_world, ee_pos, yaw).reshape(-1)
    return np.asarray([
        *object_local.tolist(), math.sin(rel_yaw), math.cos(rel_yaw),
        math.sin(yaw), math.cos(yaw), opening_fraction(gripper_width(env)),
        *tips_local.tolist(),
    ], dtype=np.float32)


def observation_inputs(env: SurfaceManipulatorEnv, shape: SurfaceShape, point_count: int = SURFACE_POINTS, sample_seed: int = 0, nonuniform: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ee_pos = env.robot_observation().ee_position.numpy().astype(np.float32)
    yaw = tool_yaw(env)
    world_surface = sample_surface_points_world(shape, point_count, sample_seed, nonuniform=nonuniform)
    world_centerline = sample_centerline_points_world(shape, point_count)
    surface_local = localize_world(world_surface, ee_pos, yaw).astype(np.float32)
    centerline_local = localize_world(world_centerline, ee_pos, yaw).astype(np.float32)
    state = _common_state(env, shape, ee_pos, yaw)
    return state, surface_local, centerline_local


def policy_geometry_points(model_name: SurfaceModelName, surface_points: np.ndarray,
                           centerline_points: np.ndarray) -> np.ndarray:
    return centerline_points if model_name == "centerline_set" else surface_points


def expert_action(env: SurfaceManipulatorEnv, shape: SurfaceShape, config: SurfaceFeasibilityConfig) -> tuple[np.ndarray, dict[str, Any]]:
    ee = env.robot_observation().ee_position.numpy().astype(np.float64)
    yaw = tool_yaw(env)
    target, normal, target_yaw, target_width, side = _surface_target(shape, ee, config.pregrasp_clearance)
    local_delta = rotate_xz(torch.as_tensor((target - ee).reshape(1, 3), dtype=torch.float32), -yaw)[0]
    local_delta = clamp_delta(local_delta, config.max_delta_ee)
    d_yaw = float(np.clip(wrap_angle(target_yaw - yaw), -config.max_delta_rotation, config.max_delta_rotation))
    target_fraction = opening_fraction(target_width)
    d_gripper = float(np.clip(target_fraction - opening_fraction(gripper_width(env)), -config.max_delta_gripper, config.max_delta_gripper))
    action = np.asarray([float(local_delta[0]), 0.0, float(local_delta[2]), d_yaw, d_gripper], dtype=np.float32)
    metadata = {"target_position": target, "target_normal": normal, "target_yaw": target_yaw,
                "target_width": target_width, "side": side, "surface_point_count": SURFACE_POINTS}
    return action, metadata


def _solve_ik_pose(env: SurfaceManipulatorEnv, target_position: np.ndarray, target_yaw: float) -> np.ndarray:
    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = env.data.qpos
    scratch.qvel[:] = 0.0
    q = scratch.qpos[env.arm_qpos_ids].copy()
    axes = np.array([0, 2], dtype=np.int32)
    orientation_scale = 0.08
    for _ in range(env.config.ik_max_iters):
        scratch.qpos[env.arm_qpos_ids] = q
        mujoco.mj_forward(env.model, scratch)
        position_error = target_position[axes] - scratch.site_xpos[env.ee_site_id][axes]
        yaw_error = float(wrap_angle(target_yaw - float(q.sum())))
        error = np.concatenate([position_error, [orientation_scale * yaw_error]])
        if np.linalg.norm(position_error) <= env.config.ik_tolerance and abs(yaw_error) <= 0.025:
            return q
        jacp = np.zeros((3, env.model.nv), dtype=np.float64)
        jacr = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(env.model, scratch, jacp, jacr, env.ee_site_id)
        joint_cols = env.arm_dof_ids
        j = np.vstack([jacp[np.ix_(axes, joint_cols)], np.ones((1, len(joint_cols))) * orientation_scale])
        dq = j.T @ np.linalg.solve(j @ j.T + env.config.ik_damping**2 * np.eye(3), error)
        q += np.clip(dq, -env.config.ik_max_joint_delta, env.config.ik_max_joint_delta)
        q = env._clip_arm_qpos(q)
    return q


def apply_local_action(env: SurfaceManipulatorEnv, action: np.ndarray | torch.Tensor, config: SurfaceFeasibilityConfig) -> None:
    action_t = torch.as_tensor(action, dtype=torch.float32).reshape(ACTION_DIM)
    robot = env.robot_observation()
    current_yaw = tool_yaw(env)
    local_delta = clamp_delta(action_t[:3], config.max_delta_ee)
    world_delta = world_action_from_local(local_delta, current_yaw)
    world_delta[1] = 0.0
    target_position = robot.ee_position.numpy().astype(np.float64) + world_delta
    d_yaw = float(np.clip(action_t[3].item(), -config.max_delta_rotation, config.max_delta_rotation))
    target_yaw = float(wrap_angle(current_yaw + d_yaw))
    q_target = _solve_ik_pose(env, target_position, target_yaw)
    current_open = opening_fraction(gripper_width(env))
    _, desired_open = clamp_gripper_action(float(action_t[4].item()), current_open, config.max_delta_gripper)
    env._track_joint_target(q_target, gripper=opening_fraction_to_command(desired_open))
    _set_gripper_open_fraction_kinematic(env, desired_open)
    env.step_count += 1
    mujoco.mj_forward(env.model, env.data)


def _reset_surface_env(env: SurfaceManipulatorEnv, spec: SurfaceEpisodeSpec) -> None:
    env.reset(randomize_robot=False)
    env.set_target_position(np.asarray([0.39, 0.0, 0.22], dtype=np.float64))
    env.set_surface_shape(spec.shape)
    env._set_arm_qpos(env._clip_arm_qpos(np.asarray(spec.arm_qpos, dtype=np.float64)))
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    # Set a reproducible initial aperture using the calibrated native finger range.
    _set_gripper_open_fraction_kinematic(env, 1.0 - float(spec.gripper_command))
    env.step_count = 0
    mujoco.mj_forward(env.model, env.data)


def make_env(config: SurfaceFeasibilityConfig, seed: int) -> SurfaceManipulatorEnv:
    sim = MujocoReachConfig(max_steps=config.max_steps, control_substeps=config.control_substeps,
                            max_delta_ee=config.max_delta_ee, target_radius=0.01, kinematic_joint_control=True)
    return SurfaceManipulatorEnv(sim, seed=seed, output_dir=config.output_dir)


def _torch_topology(points: torch.Tensor, state: torch.Tensor, config: SurfaceFeasibilityConfig, use_local_edges: bool) -> GraphTopology:
    batch, count, _ = points.shape
    device = points.device
    tips = state[:, 8:14].reshape(batch, 2, 3)
    surface_dist = torch.cdist(points, points)
    surface_dist = surface_dist + torch.eye(count, dtype=surface_dist.dtype, device=device).unsqueeze(0) * 1e6
    ss_dist, ss_idx = torch.topk(surface_dist, k=min(config.graph_k, count - 1), dim=-1, largest=False)
    ss_src = torch.arange(count, device=device).reshape(1, count, 1).expand(batch, -1, ss_idx.shape[-1]) + 3
    ss_dst = ss_idx + 3
    ss_valid = ss_dist <= config.graph_radius

    tip_dist = torch.cdist(tips, points)
    tip_k = min(TIP_K, count)
    tip_idx = torch.topk(tip_dist, k=tip_k, dim=-1, largest=False).indices
    tip_nodes = torch.arange(1, 3, device=device).reshape(1, 2, 1).expand(batch, -1, tip_k)
    tip_to_surface_src, tip_to_surface_dst = tip_nodes, tip_idx + 3

    ee_src = torch.zeros((batch, count), dtype=torch.long, device=device)
    surf_dst = torch.arange(count, device=device).reshape(1, count).expand(batch, -1) + 3
    src_parts = [torch.cat([ee_src, surf_dst], dim=1)]
    dst_parts = [torch.cat([surf_dst, ee_src], dim=1)]
    type_parts = [torch.ones((batch, count * 2), dtype=torch.long, device=device)]
    src_parts.append(torch.stack([torch.zeros(batch, dtype=torch.long, device=device), torch.ones(batch, dtype=torch.long, device=device)], dim=-1))
    dst_parts.append(torch.stack([torch.ones(batch, dtype=torch.long, device=device), torch.zeros(batch, dtype=torch.long, device=device)], dim=-1))
    type_parts.append(torch.zeros((batch, 2), dtype=torch.long, device=device))

    tip_src = tip_to_surface_src.reshape(batch, -1)
    tip_dst = tip_to_surface_dst.reshape(batch, -1)
    src_parts.extend([tip_src, tip_dst])
    dst_parts.extend([tip_dst, tip_src])
    tip_types = torch.full_like(tip_src, 2)
    type_parts.extend([tip_types, tip_types])
    valid_parts = [torch.ones_like(part, dtype=torch.bool) for part in src_parts[:-2]]
    valid_parts.extend([torch.ones_like(tip_src, dtype=torch.bool), torch.ones_like(tip_src, dtype=torch.bool)])
    if use_local_edges:
        src_parts.append(torch.cat([ss_src.reshape(batch, -1), ss_dst.reshape(batch, -1)], dim=1))
        dst_parts.append(torch.cat([ss_dst.reshape(batch, -1), ss_src.reshape(batch, -1)], dim=1))
        ss_types = torch.full((batch, ss_src.numel() // batch * 2), 3, dtype=torch.long, device=device)
        type_parts.append(ss_types)
        valid_parts.append(torch.cat([ss_valid.reshape(batch, -1), ss_valid.reshape(batch, -1)], dim=1))
    return GraphTopology(
        src=torch.cat(src_parts, dim=1), dst=torch.cat(dst_parts, dim=1),
        edge_type=torch.cat(type_parts, dim=1), valid=torch.cat(valid_parts, dim=1),
        surface_pairs=torch.stack([ss_src.reshape(batch, -1), ss_dst.reshape(batch, -1)], dim=-1),
    )


def build_graph_topology_numpy(points: np.ndarray, tips: np.ndarray, config: SurfaceFeasibilityConfig,
                               use_local_edges: bool = True, cached_surface_pairs: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Geometry-only directed kNN topology; no mesh or expert connectivity is used."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    tips = np.asarray(tips, dtype=np.float64).reshape(2, 3)
    count = len(xyz)
    src: list[int] = []
    dst: list[int] = []
    types: list[int] = []
    valid: list[bool] = []
    for tip_node in (1, 2):
        src.extend([0, tip_node]); dst.extend([tip_node, 0]); types.extend([0, 0]); valid.extend([True, True])
    for p in range(count):
        src.extend([0, p + 3]); dst.extend([p + 3, 0]); types.extend([1, 1]); valid.extend([True, True])
    for tip_idx in range(2):
        distances = np.linalg.norm(xyz - tips[tip_idx], axis=1)
        nearest = np.argsort(distances)[:min(TIP_K, count)]
        node = tip_idx + 1
        for idx in nearest:
            src.extend([node, int(idx) + 3]); dst.extend([int(idx) + 3, node]); types.extend([2, 2]); valid.extend([True, True])
    if use_local_edges:
        if cached_surface_pairs is None:
            distance = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
            np.fill_diagonal(distance, np.inf)
            nearest = np.argsort(distance, axis=1)[:, :min(config.graph_k, count - 1)]
            pairs = np.stack([np.repeat(np.arange(count), nearest.shape[1]), nearest.reshape(-1)], axis=1)
            pair_dist = distance[pairs[:, 0], pairs[:, 1]]
            pairs = pairs[pair_dist <= config.graph_radius]
        else:
            pairs = np.asarray(cached_surface_pairs, dtype=np.int64).reshape(-1, 2)
        for a, b in pairs:
            src.extend([int(a) + 3, int(b) + 3]); dst.extend([int(b) + 3, int(a) + 3]); types.extend([3, 3]); valid.extend([True, True])
    else:
        pairs = np.empty((0, 2), dtype=np.int64)
    return {"src": np.asarray(src, dtype=np.int64), "dst": np.asarray(dst, dtype=np.int64),
            "edge_type": np.asarray(types, dtype=np.int64), "valid": np.asarray(valid, dtype=bool),
            "surface_pairs": np.asarray(pairs, dtype=np.int64)}


def cached_topology_valid(cached_identity: str | None, current_identity: str | None) -> bool:
    return bool(cached_identity and current_identity and cached_identity == current_identity)


def topology_from_numpy(payload: dict[str, np.ndarray], device: torch.device | str) -> GraphTopology:
    src = torch.as_tensor(payload["src"], dtype=torch.long).unsqueeze(0)
    dst = torch.as_tensor(payload["dst"], dtype=torch.long).unsqueeze(0)
    edge_type = torch.as_tensor(payload["edge_type"], dtype=torch.long).unsqueeze(0)
    valid = torch.as_tensor(payload["valid"], dtype=torch.bool).unsqueeze(0)
    pairs = torch.as_tensor(payload["surface_pairs"], dtype=torch.long).reshape(1, -1, 2)
    return GraphTopology(src, dst, edge_type, valid, pairs).to(device)


class SurfaceSetAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, output_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim + STATE_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, state: torch.Tensor, points: torch.Tensor, topology: GraphTopology | None = None) -> torch.Tensor:
        del topology
        encoded = self.point_encoder(points)
        attended, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        pooled = self.norm(encoded + attended).mean(dim=1)
        return self.head(torch.cat([pooled, state], dim=-1))


class GNSProcessorLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int = 8) -> None:
        super().__init__()
        edge_input = hidden_dim * 2 + edge_dim
        self.edge_update = nn.Sequential(nn.Linear(edge_input, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gate = nn.Sequential(nn.Linear(edge_input, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1))
        self.node_update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor, positions: torch.Tensor, topology: GraphTopology) -> torch.Tensor:
        batch, node_count, hidden = nodes.shape
        src, dst = topology.src, topology.dst
        b_index = torch.arange(batch, device=nodes.device).reshape(batch, 1).expand_as(src)
        src_h = nodes[b_index, src]
        dst_h = nodes[b_index, dst]
        rel = positions[b_index, dst] - positions[b_index, src]
        dist = rel.norm(dim=-1, keepdim=True)
        edge_type = torch.nn.functional.one_hot(topology.edge_type.clamp(0, 3), num_classes=4).to(nodes.dtype)
        edge_input = torch.cat([dst_h, src_h, rel, dist, edge_type], dim=-1)
        message = self.edge_update(edge_input) * torch.sigmoid(self.gate(edge_input))
        message = message * topology.valid.unsqueeze(-1).to(message.dtype)
        flat_dst = (dst + b_index * node_count).reshape(-1)
        aggregate = nodes.new_zeros((batch * node_count, hidden))
        aggregate.index_add_(0, flat_dst, message.reshape(-1, hidden))
        degree = nodes.new_zeros((batch * node_count, 1))
        degree.index_add_(0, flat_dst, topology.valid.reshape(-1, 1).to(nodes.dtype))
        aggregate = aggregate / degree.clamp_min(1.0).sqrt()
        aggregate = aggregate.reshape(batch, node_count, hidden)
        return self.node_norm(nodes + self.node_update(torch.cat([nodes, aggregate], dim=-1)))


class SurfaceGraphNetwork(nn.Module):
    def __init__(self, hidden_dim: int, steps: int, use_local_edges: bool, config: SurfaceFeasibilityConfig, output_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.use_local_edges = bool(use_local_edges)
        self.config = config
        self.node_encoder = nn.Sequential(nn.Linear(7, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.context_encoder = nn.Sequential(nn.Linear(STATE_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList([GNSProcessorLayer(hidden_dim) for _ in range(steps)])
        self.action_head = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))

    def encode(self, state: torch.Tensor, points: torch.Tensor, topology: GraphTopology | None = None) -> torch.Tensor:
        batch, count, _ = points.shape
        tips = state[:, 8:14].reshape(batch, 2, 3)
        positions = torch.cat([points.new_zeros((batch, 1, 3)), tips, points], dim=1)
        roles = torch.zeros((batch, count + 3, 4), dtype=points.dtype, device=points.device)
        roles[:, 0, 0] = 1.0
        roles[:, 1:3, 1] = 1.0
        roles[:, 3:, 2] = 1.0
        raw = torch.cat([positions, roles], dim=-1)
        nodes = self.node_encoder(raw)
        context = self.context_encoder(state)
        nodes = torch.cat([nodes[:, :1] + context.unsqueeze(1), nodes[:, 1:]], dim=1)
        if topology is None:
            topology = _torch_topology(points, state, self.config, self.use_local_edges)
        for layer in self.layers:
            nodes = layer(nodes, positions, topology)
        return torch.cat([nodes[:, 0], nodes[:, 1], nodes[:, 2]], dim=-1)

    def forward(self, state: torch.Tensor, points: torch.Tensor, topology: GraphTopology | None = None) -> torch.Tensor:
        return self.action_head(self.encode(state, points, topology))


def build_surface_model(name: SurfaceModelName, config: SurfaceFeasibilityConfig) -> nn.Module:
    if name in ("centerline_set", "surface_set"):
        return SurfaceSetAttention(config.hidden_dim, config.attention_heads)
    if name == "surface_graph":
        return SurfaceGraphNetwork(config.hidden_dim, config.message_passing_steps, True, config)
    if name == "surface_graph_no_local":
        return SurfaceGraphNetwork(config.hidden_dim, config.message_passing_steps, False, config)
    raise ValueError(f"Unknown surface model {name}")


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _normalised_action_loss(pred: torch.Tensor, target: torch.Tensor, config: SurfaceFeasibilityConfig) -> torch.Tensor:
    scales = pred.new_tensor([config.max_delta_ee, config.max_delta_ee, config.max_delta_ee,
                              config.max_delta_rotation, config.max_delta_gripper])
    return torch.nn.functional.mse_loss((pred - target) / scales, torch.zeros_like(pred))


def train_surface_model(name: SurfaceModelName, dataset: dict[str, Any], config: SurfaceFeasibilityConfig,
                        seed: int, output_dir: str | Path, epochs: int | None = None,
                        device_preference: DevicePreference | None = None) -> dict[str, Any]:
    set_seed(seed)
    device = select_device(device_preference or config.device)
    model = build_surface_model(name, config).to(device)
    states = dataset["states"].to(device)
    train_points = torch.as_tensor(policy_geometry_points(
        name, dataset["surface_points"].numpy(), dataset["centerline_points"].numpy()), dtype=torch.float32, device=device)
    actions = dataset["actions"].to(device)
    train_indices = dataset["train_indices"].to(device)
    val_indices = dataset["validation_indices"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    num_epochs = int(epochs or config.epochs)
    best_val = float("inf")
    history: list[dict[str, float]] = []
    out_dir = ensure_dir(output_dir)
    checkpoint = out_dir / f"{name}.pt"
    for epoch in range(num_epochs):
        model.train()
        permutation = train_indices[torch.randperm(len(train_indices), device=device)]
        losses: list[float] = []
        for start in range(0, len(permutation), config.batch_size):
            idx = permutation[start:start + config.batch_size]
            pred = model(states[idx], train_points[idx])
            loss = _normalised_action_loss(pred, actions[idx], config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vals: list[float] = []
            for start in range(0, len(val_indices), config.batch_size):
                idx = val_indices[start:start + config.batch_size]
                vals.append(float(_normalised_action_loss(model(states[idx], train_points[idx]), actions[idx], config).cpu()))
        val = float(np.mean(vals))
        row = {"epoch": float(epoch + 1), "train_loss": float(np.mean(losses)), "validation_loss": val}
        history.append(row)
        if val < best_val:
            best_val = val
            torch.save({"model_name": name, "model_state": model.state_dict(), "config": asdict(config),
                        "seed": seed, "best_validation_loss": best_val, "parameters": model_parameter_count(model)}, checkpoint)
    payload = {"model": name, "seed": seed, "device": str(device), "epochs": num_epochs,
               "best_validation_loss": best_val, "parameters": model_parameter_count(model),
               "history": history, "checkpoint_path": str(checkpoint)}
    _write_json(out_dir / f"{name}_training.json", payload)
    return payload


def load_surface_model(path: str | Path, model_name: SurfaceModelName, config: SurfaceFeasibilityConfig,
                       device: torch.device | str) -> nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_surface_model(model_name, config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _distance_to_surface(env: SurfaceManipulatorEnv, shape: SurfaceShape) -> float:
    return _surface_distance_metrics(env, shape)[1]


def _surface_distance_metrics(env: SurfaceManipulatorEnv, shape: SurfaceShape | None) -> tuple[list[dict[str, Any]], float]:
    if shape is None:
        return [], 0.20
    active = env.surface_box_ids if shape.cross_section == "flat" else env.surface_capsule_ids
    rows: list[dict[str, Any]] = []
    best = 0.20
    segment = np.zeros(6, dtype=np.float64)
    # Use the full articulated robot geometry, including finger links and tips.
    for object_id in active:
        for robot_id in env.robot_collision_geom_ids:
            distance = float(mujoco.mj_geomDistance(env.model, env.data, int(object_id), int(robot_id), 0.20, segment))
            best = min(best, distance)
            if distance <= 1e-4:
                rows.append({"geom1": int(object_id), "geom2": int(robot_id), "distance": distance})
    return rows, best


def _state_errors(env: SurfaceManipulatorEnv, shape: SurfaceShape, config: SurfaceFeasibilityConfig,
                  target: tuple[np.ndarray, np.ndarray, float, float, int]) -> dict[str, float]:
    target_pos, _, target_yaw, target_width, _ = target
    ee_pos = env.robot_observation().ee_position.numpy().astype(np.float64)
    return {
        "position_error": float(np.linalg.norm((ee_pos - target_pos)[[0, 2]])),
        "orientation_error": equivalent_orientation_error(tool_yaw(env), target_yaw),
        "gripper_width_error": abs(gripper_width(env) - target_width),
    }


def _success_from_errors(errors: dict[str, float], collision: bool, config: SurfaceFeasibilityConfig) -> bool:
    return bool(
        errors["position_error"] < config.success_position_threshold
        and errors["orientation_error"] < config.success_rotation_threshold
        and errors["gripper_width_error"] < config.success_opening_threshold
        and not collision
    )


def _analytic_observed_action(env: SurfaceManipulatorEnv, surface_local: np.ndarray,
                              state: np.ndarray, config: SurfaceFeasibilityConfig) -> np.ndarray:
    """PCA surface estimate; consumes only observed points and current robot state."""
    xz = np.asarray(surface_local[:, [0, 2]], dtype=np.float64)
    center = np.median(xz, axis=0)
    covariance = np.cov(xz.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    # The EE is the local origin; orient the normal toward the current EE side.
    if float(np.dot(-center, normal)) < 0.0:
        normal = -normal
    longitudinal = (xz - center) @ tangent
    near_mid = np.abs(longitudinal - np.median(longitudinal)) <= max(float(np.ptp(longitudinal)) * 0.26, 1e-5)
    selected = xz[near_mid] if np.any(near_mid) else xz
    normal_coord = selected @ normal
    surface_proj = float(np.max(normal_coord))
    side_points = selected[normal_coord >= np.quantile(normal_coord, 0.86)]
    surface_center = np.median(side_points, axis=0) if len(side_points) else center + surface_proj * normal
    target_local = np.asarray([surface_center[0] + config.pregrasp_clearance * normal[0],
                               surface_center[1] + config.pregrasp_clearance * normal[1]], dtype=np.float32)
    delta = np.asarray([target_local[0], 0.0, target_local[1]], dtype=np.float32)
    delta_t = clamp_delta(torch.as_tensor(delta), config.max_delta_ee).numpy()
    target_yaw_local = math.atan2(float(normal[1]), float(normal[0]))
    d_yaw = float(np.clip(target_yaw_local, -config.max_delta_rotation, config.max_delta_rotation))
    observed_width = float(np.quantile(xz @ normal, 0.97) - np.quantile(xz @ normal, 0.03))
    desired_width = float(np.clip(observed_width + 0.010, MIN_GRIPPER_WIDTH, MAX_GRIPPER_WIDTH))
    current_open = opening_fraction(gripper_width(env))
    d_open = float(np.clip(opening_fraction(desired_width) - current_open,
                           -config.max_delta_gripper, config.max_delta_gripper))
    return np.asarray([delta_t[0], 0.0, delta_t[2], d_yaw, d_open], dtype=np.float32)


def _policy_action(model_name: SurfaceModelName, model: nn.Module, state: np.ndarray, points: np.ndarray,
                   config: SurfaceFeasibilityConfig, device: torch.device) -> np.ndarray:
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, STATE_DIM)
    points_t = torch.as_tensor(points, dtype=torch.float32, device=device).reshape(1, -1, 3)
    topology = None
    if model_name in ("surface_graph", "surface_graph_no_local"):
        tips = state[8:14].reshape(2, 3)
        topo_np = build_graph_topology_numpy(points, tips, config, use_local_edges=model_name == "surface_graph")
        topology = topology_from_numpy(topo_np, device)
    with torch.no_grad():
        output = model(state_t, points_t, topology)[0].detach().cpu().numpy()
    translation = clamp_delta(torch.as_tensor(output[:3], dtype=torch.float32), config.max_delta_ee).numpy()
    return np.asarray([
        translation[0], 0.0, translation[2],
        np.clip(output[3], -config.max_delta_rotation, config.max_delta_rotation),
        np.clip(output[4], -config.max_delta_gripper, config.max_delta_gripper),
    ], dtype=np.float32)


def rollout_surface_episode(env: SurfaceManipulatorEnv, spec: SurfaceEpisodeSpec, config: SurfaceFeasibilityConfig,
                            model_name: SurfaceModelName | None = None, model: nn.Module | None = None,
                            device: torch.device | None = None, controller: Literal["neural", "expert", "analytic"] = "neural",
                            point_count: int = SURFACE_POINTS, nonuniform: bool = False) -> dict[str, Any]:
    _reset_surface_env(env, spec)
    initial_ee = env.robot_observation().ee_position.numpy().astype(np.float64)
    target = _surface_target(spec.shape, initial_ee, config.pregrasp_clearance)
    trajectory = [initial_ee.tolist()]
    errors = _state_errors(env, spec.shape, config, target)
    trajectory_errors = [errors["position_error"]]
    collisions, min_clearance = _surface_distance_metrics(env, spec.shape)
    penetration_count = sum(c["distance"] < -1e-4 for c in collisions)
    steps_to_success: int | None = 0 if _success_from_errors(errors, bool(collisions), config) else None
    for step in range(config.max_steps):
        if steps_to_success is not None:
            break
        state, surface_points, centerline_points = observation_inputs(
            env, spec.shape, point_count, sample_seed=_stable_seed(spec.sample_identity), nonuniform=nonuniform
        )
        if controller == "expert":
            action, _ = expert_action(env, spec.shape, config)
        elif controller == "analytic":
            action = _analytic_observed_action(env, surface_points, state, config)
        else:
            if model is None or model_name is None or device is None:
                raise ValueError("Neural rollout requires model, model_name, and device.")
            points = policy_geometry_points(model_name, surface_points, centerline_points)
            action = _policy_action(model_name, model, state, points, config, device)
        apply_local_action(env, action, config)
        ee = env.robot_observation().ee_position.numpy().astype(np.float64)
        trajectory.append(ee.tolist())
        errors = _state_errors(env, spec.shape, config, target)
        trajectory_errors.append(errors["position_error"])
        contact_rows, step_clearance = _surface_distance_metrics(env, spec.shape)
        collisions.extend(contact_rows)
        penetration_count += sum(c["distance"] < -1e-4 for c in contact_rows)
        min_clearance = min(min_clearance, step_clearance)
        if _success_from_errors(errors, bool(collisions), config):
            steps_to_success = step + 1
    final_collision_rows, _ = _surface_distance_metrics(env, spec.shape)
    final_errors = _state_errors(env, spec.shape, config, target)
    traj = np.asarray(trajectory, dtype=np.float64)
    path_length = float(np.linalg.norm(np.diff(traj[:, [0, 2]], axis=0), axis=1).sum()) if len(traj) > 1 else 0.0
    return {
        "episode_id": spec.episode_id, "condition": spec.condition, "shape_id": spec.shape.shape_id,
        "success": _success_from_errors(final_errors, bool(collisions), config),
        "final_position_error": final_errors["position_error"],
        "final_orientation_error": final_errors["orientation_error"],
        "final_gripper_width_error": final_errors["gripper_width_error"],
        "final_gripper_width": gripper_width(env), "target_gripper_width": target[3],
        "final_yaw": tool_yaw(env),
        "minimum_safe_clearance": float(min_clearance),
        "collision": bool(collisions), "final_collision": bool(final_collision_rows),
        "trajectory_collision": bool(collisions), "penetration_count": int(penetration_count),
        "illegal_contact_count": int(len(collisions)), "trajectory_error": float(np.mean(trajectory_errors)),
        "trajectory_length": path_length, "steps_to_convergence": int(steps_to_success if steps_to_success is not None else config.max_steps),
        "trajectory": trajectory, "target_position": target[0].tolist(), "target_normal": target[1].tolist(),
        "target_yaw": target[2], "target_side": target[4], "initial_ee": initial_ee.tolist(),
        "shape": asdict(spec.shape),
    }


def _stable_seed(value: str) -> int:
    # Stable across Python processes (unlike hash()).
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little", signed=False)


def collect_expert_dataset(config: SurfaceFeasibilityConfig, seed: int, output_dir: str | Path,
                           train_episodes: int | None = None, validation_episodes: int | None = None) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    train_specs = sample_episode_specs(train_episodes or config.train_episodes, seed, config, "train")
    val_specs = sample_episode_specs(validation_episodes or config.validation_episodes, seed + 10000, config, "validation")
    env = make_env(config, seed)
    states: list[np.ndarray] = []
    surfaces: list[np.ndarray] = []
    centerlines: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    train_ids: list[int] = []
    val_ids: list[int] = []
    episode_metadata: list[dict[str, Any]] = []
    for split_name, specs, indices in (("train", train_specs, train_ids), ("validation", val_specs, val_ids)):
        for spec in specs:
            _reset_surface_env(env, spec)
            episode_start = len(states)
            for _ in range(config.max_steps):
                state, surface, centerline = observation_inputs(env, spec.shape, config.point_count,
                                                                sample_seed=_stable_seed(spec.sample_identity))
                action, _ = expert_action(env, spec.shape, config)
                states.append(state)
                surfaces.append(surface)
                centerlines.append(centerline)
                actions.append(action)
                apply_local_action(env, action, config)
                target = _surface_target(spec.shape, env.robot_observation().ee_position.numpy(), config.pregrasp_clearance)
                errors = _state_errors(env, spec.shape, config, target)
                if _success_from_errors(errors, bool(env.contacts_with_surface()), config):
                    break
            indices.extend(range(episode_start, len(states)))
            episode_metadata.append({"split": split_name, "spec": asdict(spec), "sample_start": episode_start,
                                     "sample_end": len(states), "expert_success": bool(_success_from_errors(
                                         _state_errors(env, spec.shape, config,
                                             _surface_target(spec.shape, env.robot_observation().ee_position.numpy(), config.pregrasp_clearance)),
                                         bool(env.contacts_with_surface()), config))})
    env.close()
    dataset = {
        "task": "surface_aware_pregrasp_alignment", "seed": seed,
        "states": torch.as_tensor(np.stack(states), dtype=torch.float32),
        "surface_points": torch.as_tensor(np.stack(surfaces), dtype=torch.float32),
        "centerline_points": torch.as_tensor(np.stack(centerlines), dtype=torch.float32),
        "actions": torch.as_tensor(np.stack(actions), dtype=torch.float32),
        "train_indices": torch.as_tensor(train_ids, dtype=torch.long),
        "validation_indices": torch.as_tensor(val_ids, dtype=torch.long),
        "model_input_fields": ["state", "sampled_geometry_points"],
        "episodes": episode_metadata,
    }
    dataset_path = out / "surface_alignment.pt"
    torch.save(dataset, dataset_path)
    _write_json(out / "geometry_statistics.json", {
        "task": dataset["task"], "seed": seed, "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
        "point_count": config.point_count, "train_samples": len(train_ids), "validation_samples": len(val_ids),
        "train_episodes": len(train_specs), "validation_episodes": len(val_specs), "model_input_fields": dataset["model_input_fields"],
        "expert_episode_success_rate": float(np.mean([item["expert_success"] for item in episode_metadata])),
        "episodes": episode_metadata,
    })
    return {**dataset, "dataset_path": str(dataset_path), "train_count": len(train_ids), "validation_count": len(val_ids)}


def run_task_sanity(config: SurfaceFeasibilityConfig, seed: int) -> dict[str, Any]:
    out = ensure_dir(Path(config.output_dir) / "surface_necessity")
    specs = sample_episode_specs(12, seed + 40_000, config, "sanity")
    env = make_env(config, seed + 40_000)
    oracle: list[dict[str, Any]] = []
    analytic: list[dict[str, Any]] = []
    for spec in specs:
        oracle.append(rollout_surface_episode(env, spec, config, controller="expert"))
        analytic.append(rollout_surface_episode(env, spec, config, controller="analytic"))
    env.close()
    thin, thick = paired_surface_necessity_shapes(seed, config)
    common = sample_episode_specs(1, seed + 41_000, config, "surface_necessity")[0]
    thin_spec = replace(common, shape=thin, sample_identity="necessity_shared_cloud_identity")
    thick_spec = replace(common, shape=thick, sample_identity="necessity_shared_cloud_identity")
    env = make_env(config, seed + 41_000)
    _reset_surface_env(env, thin_spec)
    initial_ee = env.robot_observation().ee_position.numpy().astype(np.float64)
    thin_target = _surface_target(thin, initial_ee, config.pregrasp_clearance)
    thin_points = sample_surface_points_world(thin, config.point_count, seed=7)
    _reset_surface_env(env, thick_spec)
    thick_target = _surface_target(thick, initial_ee, config.pregrasp_clearance)
    thick_points = sample_surface_points_world(thick, config.point_count, seed=7)
    env.close()
    diagnostic = {
        "same_centerline": bool(np.allclose(sample_centerline_points_world(thin), sample_centerline_points_world(thick))),
        "same_object_pose": thin.object_center == thick.object_center and thin.object_yaw == thick.object_yaw,
        "same_initial_arm_qpos": thin_spec.arm_qpos == thick_spec.arm_qpos,
        "surface_cloud_max_difference": float(np.max(np.linalg.norm(thin_points - thick_points, axis=1))),
        "target_position_delta": float(np.linalg.norm(thin_target[0] - thick_target[0])),
        "target_opening_delta": float(abs(thin_target[3] - thick_target[3])),
        "thin": {"shape": asdict(thin), "target_position": thin_target[0].tolist(), "target_opening": thin_target[3]},
        "thick": {"shape": asdict(thick), "target_position": thick_target[0].tolist(), "target_opening": thick_target[3]},
    }
    _write_json(out / "paired_diagnostic.json", diagnostic)
    payload = {
        "oracle_expert_success_rate": float(np.mean([row["success"] for row in oracle])),
        "analytic_observed_surface_success_rate": float(np.mean([row["success"] for row in analytic])),
        "oracle_rollouts": oracle, "analytic_rollouts": analytic,
        "surface_necessity": diagnostic,
        "oracle_target_is_model_input": False,
        "expert_and_evaluation_only_fields": ["target_position", "target_normal", "target_yaw", "target_width"],
    }
    _write_json(Path(config.output_dir) / "task_sanity.json", payload)
    return payload


def _episode_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("success", "final_position_error", "final_orientation_error", "final_gripper_width_error",
             "minimum_safe_clearance", "collision", "final_collision", "trajectory_collision", "penetration_count",
             "illegal_contact_count", "trajectory_error", "steps_to_convergence", "trajectory_length")
    return {name: float(np.mean([float(row[name]) for row in results])) if results else float("nan") for name in names} | {"episodes": len(results)}


def evaluate_models_paired(specs: list[SurfaceEpisodeSpec], models: dict[SurfaceModelName, nn.Module],
                           config: SurfaceFeasibilityConfig, device: torch.device, output_dir: str | Path,
                           point_count: int = SURFACE_POINTS, nonuniform: bool = False) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    env = make_env(config, 719)
    model_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    for spec in specs:
        for name, model in models.items():
            model_rows[name].append(rollout_surface_episode(env, spec, config, name, model, device,
                                                           point_count=point_count, nonuniform=nonuniform))
    env.close()
    per_model = {name: _episode_metrics(rows) for name, rows in model_rows.items()}
    comparisons: dict[str, Any] = {}
    for baseline, candidate in (("centerline_set", "surface_set"), ("surface_set", "surface_graph"),
                                ("surface_set", "surface_graph_no_local"),
                                ("surface_graph_no_local", "surface_graph")):
        if baseline not in model_rows or candidate not in model_rows:
            continue
        a, b = model_rows[baseline], model_rows[candidate]
        both = sum(bool(x["success"] and y["success"]) for x, y in zip(a, b, strict=True))
        a_only = sum(bool(x["success"] and not y["success"]) for x, y in zip(a, b, strict=True))
        b_only = sum(bool(y["success"] and not x["success"]) for x, y in zip(a, b, strict=True))
        neither = len(a) - both - a_only - b_only
        comparison = {"baseline": baseline, "candidate": candidate, "paired_success_contingency": {
            "both_success": both, "baseline_only_success": a_only, "candidate_only_success": b_only, "both_fail": neither},
            "mcnemar_exact_p": _mcnemar_exact_p(a_only, b_only), "paired_metrics": {}}
        for metric in ("final_position_error", "final_orientation_error", "final_gripper_width_error",
                       "collision", "trajectory_error", "trajectory_length", "success"):
            delta = np.asarray([float(y[metric]) - float(x[metric]) for x, y in zip(a, b, strict=True)], dtype=np.float64)
            comparison["paired_metrics"][metric] = {"candidate_minus_baseline_mean": float(delta.mean()) if len(delta) else 0.0,
                "ci95": _bootstrap_ci(delta, config.bootstrap_samples, seed=8341 + len(delta) + len(metric))}
        comparisons[f"{baseline}_vs_{candidate}"] = comparison
    episodes = [{"episode_id": spec.episode_id, "condition": spec.condition, "shape": asdict(spec.shape),
                 "models": {name: model_rows[name][i] for name in model_rows}}
                for i, spec in enumerate(specs)]
    payload = {"surface_points": point_count, "sampling_nonuniform": nonuniform, "models": per_model,
               "paired_comparisons": comparisons, "episodes": episodes}
    _write_json(out / "episode_results.json", payload)
    _write_json(out / "summary.json", {"surface_points": point_count, "sampling_nonuniform": nonuniform,
                                        "models": per_model, "paired_comparisons": comparisons})
    return payload


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(max(1, samples), len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _mcnemar_exact_p(baseline_only: int, candidate_only: int) -> float:
    n = baseline_only + candidate_only
    if n == 0:
        return 1.0
    k = min(baseline_only, candidate_only)
    probability = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * probability))


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _distribution_ms(values_ns: list[int]) -> dict[str, float]:
    values = np.asarray(values_ns, dtype=np.float64) / 1e6
    return {
        "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)), "max_ms": float(values.max()),
        "mean_ms": float(values.mean()), "deadline_miss_rate_over_5ms": float(np.mean(values > 5.0)),
    }


def benchmark_latency(model_name: SurfaceModelName, model: nn.Module, shape: SurfaceShape,
                      env: SurfaceManipulatorEnv, config: SurfaceFeasibilityConfig, device: torch.device,
                      point_count: int, mode: Literal["rebuilt", "cached"] = "rebuilt",
                      warmup: int | None = None, iterations: int | None = None) -> dict[str, Any]:
    warmup = int(config.latency_warmup if warmup is None else warmup)
    iterations = int(config.latency_iterations if iterations is None else iterations)
    if mode == "cached" and model_name not in ("surface_graph", "surface_graph_no_local"):
        raise ValueError("Cached topology is only meaningful for Surface Graph models.")
    use_local = model_name == "surface_graph"
    sample_identity = f"latency:{shape.shape_id}:surface_sample_v1"
    if mode == "cached" and not cached_topology_valid(sample_identity, sample_identity):
        raise ValueError("Cached Surface↔Surface topology requires unchanged point identities.")
    fixed_seed = _stable_seed(sample_identity)
    initial = env.robot_observation().ee_position.numpy().astype(np.float32)
    yaw = tool_yaw(env)
    base_surface = sample_surface_points_world(shape, 64, fixed_seed, nonuniform=True)
    take = np.linspace(0, 63, point_count, dtype=np.int64)
    cached_pairs: np.ndarray | None = None
    if mode == "cached" and use_local:
        first_local = localize_world(base_surface[take], initial, yaw)
        state0 = _common_state(env, shape, initial, yaw)
        cached_pairs = build_graph_topology_numpy(first_local, state0[8:14].reshape(2, 3), config, True)["surface_pairs"]
    components: dict[str, list[int]] = {key: [] for key in (
        "surface_sampling", "coordinate_transform", "graph_construction", "tensor_preparation",
        "network_forward", "local_action_to_world", "total_geometry_to_action")}
    final_action: np.ndarray | None = None
    total_calls = warmup + iterations
    for it in range(total_calls):
        total_start = time.perf_counter_ns()
        t0 = time.perf_counter_ns()
        # This stage represents the already detected task-relevant region, not RGB-D perception.
        raw_surface = sample_surface_points_world(shape, 64, fixed_seed, nonuniform=True)
        selected_world = raw_surface[take]
        t1 = time.perf_counter_ns()
        state_np = _common_state(env, shape, initial, yaw)
        points_np = localize_world(selected_world, initial, yaw).astype(np.float32)
        t2 = time.perf_counter_ns()
        topology_payload = None
        if model_name in ("surface_graph", "surface_graph_no_local"):
            use_cache = mode == "cached" and use_local
            topology_payload = build_graph_topology_numpy(
                points_np, state_np[8:14].reshape(2, 3), config, use_local_edges=use_local,
                cached_surface_pairs=cached_pairs if use_cache else None,
            )
        t3 = time.perf_counter_ns()
        state_t = torch.as_tensor(state_np, dtype=torch.float32).reshape(1, STATE_DIM).to(device)
        points_t = torch.as_tensor(points_np, dtype=torch.float32).reshape(1, point_count, 3).to(device)
        topology_t = topology_from_numpy(topology_payload, device) if topology_payload is not None else None
        t4 = time.perf_counter_ns()
        _sync_device(device)
        forward_start = time.perf_counter_ns()
        with torch.no_grad():
            output = model(state_t, points_t, topology_t)[0]
        _sync_device(device)
        forward_end = time.perf_counter_ns()
        action_local = output[:3].detach().cpu()
        action_world = world_action_from_local(action_local, yaw)
        # Preserve the current plane constraint and include the actual command transform.
        action_world[1] = 0.0
        final_action = np.concatenate([action_world, output[3:5].detach().cpu().numpy()])
        end = time.perf_counter_ns()
        if it >= warmup:
            components["surface_sampling"].append(t1 - t0)
            components["coordinate_transform"].append(t2 - t1)
            components["graph_construction"].append(t3 - t2)
            components["tensor_preparation"].append(t4 - t3)
            components["network_forward"].append(forward_end - forward_start)
            components["local_action_to_world"].append(end - forward_end)
            components["total_geometry_to_action"].append(end - total_start)
    summaries = {key: _distribution_ms(value) for key, value in components.items()}
    return {
        "model": model_name, "surface_points": point_count, "mode": mode, "device": str(device),
        "warmup_iterations": warmup, "timed_iterations": iterations, "sample_identity": sample_identity,
        "cached_topology_condition": "same surface point identities and rigid object geometry" if mode == "cached" else "rebuilt each control step",
        "sensor_to_command_latency": "NOT MEASURED", "components": summaries,
        "network_forward": summaries["network_forward"], "geometry_to_action": summaries["total_geometry_to_action"],
        "last_action": final_action.tolist() if final_action is not None else [],
    }


def benchmark_experiment_latency(config: SurfaceFeasibilityConfig, checkpoints: dict[str, dict[str, str]],
                                 shape: SurfaceShape, seed: int) -> dict[str, Any]:
    device = select_device(config.eval_device)
    root = ensure_dir(Path(config.output_dir) / "latency")
    env = make_env(config, seed + 90000)
    spec = sample_episode_specs(1, seed + 90001, config, "latency")[0]
    spec = replace(spec, shape=shape, condition="latency", sample_identity=f"latency:{shape.shape_id}:surface_sample_v1")
    _reset_surface_env(env, spec)
    results: dict[str, Any] = {"target_geometry_to_action_p99_ms": 5.0, "target_control_rate_hz": 100,
                               "control_period_ms": 10.0, "latency_target_is_provisional_design_budget": True,
                               "device": str(device), "sensor_to_command_latency": "NOT MEASURED", "benchmarks": {}}
    for count in (16, 32, 64):
        n_dir = ensure_dir(root / f"n{count}")
        for name in MODEL_NAMES:
            checkpoint = checkpoints[str(seed)][name]
            model = load_surface_model(checkpoint, name, config, device)
            print(f"[latency] {name} N={count} rebuilt: warmup={config.latency_warmup}, timed={config.latency_iterations}", flush=True)
            rebuilt = benchmark_latency(name, model, shape, env, config, device, count, "rebuilt")
            _write_json(n_dir / f"{name}_rebuilt.json", rebuilt)
            results["benchmarks"][f"{name}_n{count}_rebuilt"] = rebuilt
            if name in ("surface_graph", "surface_graph_no_local"):
                print(f"[latency] {name} N={count} cached", flush=True)
                cached = benchmark_latency(name, model, shape, env, config, device, count, "cached")
                _write_json(n_dir / f"{name}_cached.json", cached)
                results["benchmarks"][f"{name}_n{count}_cached"] = cached
    env.close()
    _write_json(root / "latency_results.json", results)
    return results


def plot_geometry_sample(shape: SurfaceShape, output_path: str | Path, seed: int = 3,
                        ee_position: np.ndarray | None = None, ee_yaw: float = 0.0,
                        topology: dict[str, np.ndarray] | None = None,
                        fingertip_positions: np.ndarray | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface = sample_surface_points_world(shape, SURFACE_POINTS, seed)
    centerline = sample_centerline_points_world(shape, SURFACE_POINTS)
    if ee_position is None:
        ee_position = np.asarray(shape.object_center, dtype=np.float64) + np.asarray([0.0, 0.0, 0.16])
    target = _surface_target(shape, ee_position, 0.03)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot(centerline[:, 0], centerline[:, 2], color="black", linewidth=1.3, label="centerline")
    ax.scatter(surface[:, 0], surface[:, 2], color="#2377b4", s=22, label="surface samples")
    if topology is not None:
        pairs = topology["surface_pairs"]
        for index, (a, b) in enumerate(pairs):
            ax.plot([surface[a, 0], surface[b, 0]], [surface[a, 2], surface[b, 2]], color="#78a6c8",
                    alpha=0.35, linewidth=0.7, label="surface graph edges" if index == 0 else None)
    ax.scatter([ee_position[0]], [ee_position[2]], marker="s", color="#ef8a17", label="EE")
    if fingertip_positions is None:
        jaw = np.asarray([math.cos(ee_yaw), 0.0, math.sin(ee_yaw)]) * 0.04
        fingertip_positions = np.stack([ee_position - jaw, ee_position + jaw])
    ax.scatter(fingertip_positions[:, 0], fingertip_positions[:, 2], marker="^", color="#db5963", label="left/right fingertips")
    ax.scatter([target[0][0]], [target[0][2]], marker="*", s=160, color="#7827a1", label="expert pre-grasp")
    normal = target[1]
    ax.arrow(target[0][0], target[0][2], normal[0] * 0.04, normal[2] * 0.04, head_width=0.008, color="#7827a1")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.2); ax.legend(fontsize=8)
    ax.set_xlabel("world x (m)"); ax.set_ylabel("world z (m)"); ax.set_title(f"{shape.shape_id}: sparse observed surface graph")
    fig.tight_layout(); Path(output_path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(output_path, dpi=150); plt.close(fig)


def plot_trajectory_cases(episodes: list[dict[str, Any]], output_dir: str | Path, count: int = 4) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ensure_dir(output_dir)
    selection: dict[str, dict[str, Any] | None] = {
        "centerline_success_surface_failure": None, "surface_success_centerline_failure": None,
        "set_only_success": None, "graph_only_success": None, "collision_failure": None,
        "wrong_orientation": None, "wrong_approach_side": None, "wrong_gripper_width": None,
    }
    for ep in episodes:
        m = ep["models"]
        c = m.get("centerline_set", {})
        s = m.get("surface_set", {})
        g = m.get("surface_graph", {})
        ng = m.get("surface_graph_no_local", {})
        if selection["centerline_success_surface_failure"] is None and c.get("success") and not s.get("success"):
            selection["centerline_success_surface_failure"] = ep
        if selection["surface_success_centerline_failure"] is None and s.get("success") and not c.get("success"):
            selection["surface_success_centerline_failure"] = ep
        if selection["set_only_success"] is None and s.get("success") and not g.get("success"):
            selection["set_only_success"] = ep
        if selection["graph_only_success"] is None and g.get("success") and not s.get("success"):
            selection["graph_only_success"] = ep
        if selection["collision_failure"] is None and any(x.get("collision") for x in m.values()):
            selection["collision_failure"] = ep
        if selection["wrong_orientation"] is None and any(x.get("final_orientation_error", 0) > 0.45 for x in m.values()):
            selection["wrong_orientation"] = ep
        if selection["wrong_approach_side"] is None and any(x.get("final_orientation_error", 0) > 1.35 for x in m.values()):
            selection["wrong_approach_side"] = ep
        if selection["wrong_gripper_width"] is None and any(x.get("final_gripper_width_error", 0) > 0.012 for x in m.values()):
            selection["wrong_gripper_width"] = ep
    written: dict[str, str] = {}
    for label, ep in selection.items():
        if ep is None:
            # Keep required diagnostic panels explicit when no example occurred.
            fig, ax = plt.subplots(figsize=(5.5, 4.4)); ax.text(0.5, 0.5, "No episode met this case criterion", ha="center", va="center")
            ax.set_axis_off(); path = out / f"{label}.png"; fig.savefig(path, dpi=140); plt.close(fig); written[label] = str(path); continue
        shape = SurfaceShape(**ep["models"]["surface_set"].get("shape", ep["models"]["surface_graph"].get("shape")))
        surface = sample_surface_points_world(shape, SURFACE_POINTS, _stable_seed(shape.shape_id))
        fig, ax = plt.subplots(figsize=(6.0, 4.6))
        ax.scatter(surface[:, 0], surface[:, 2], s=12, color="#91b9ce", label="surface")
        for name, result in ep["models"].items():
            traj = np.asarray(result["trajectory"])
            ax.plot(traj[:, 0], traj[:, 2], "-o", markersize=2.5, linewidth=1, label=name)
            ax.scatter([traj[0, 0]], [traj[0, 2]], marker="<", s=34)
            ax.scatter([traj[-1, 0]], [traj[-1, 2]], marker="s", s=24)
        reference = ep["models"].get("surface_set", next(iter(ep["models"].values())))
        target = reference["target_position"]
        ax.scatter([target[0]], [target[2]], marker="*", color="black", s=120, label="target")
        normal = reference["target_normal"]
        ax.arrow(target[0], target[2], normal[0] * 0.035, normal[2] * 0.035,
                 head_width=0.007, color="black", length_includes_head=True)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.2); ax.legend(fontsize=6)
        ax.set_title(label.replace("_", " ")); ax.set_xlabel("world x (m)"); ax.set_ylabel("world z (m)")
        fig.tight_layout(); path = out / f"{label}_episode_{ep['episode_id']:04d}.png"; fig.savefig(path, dpi=140); plt.close(fig); written[label] = str(path)
    return written


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)) if values else float("nan"),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def _pooled_pair(episode_groups: list[list[dict[str, Any]]], baseline: str, candidate: str,
                 config: SurfaceFeasibilityConfig, label_seed: int) -> dict[str, Any]:
    rows = [row for group in episode_groups for row in group]
    a_only = sum(bool(row["models"][baseline]["success"] and not row["models"][candidate]["success"]) for row in rows)
    b_only = sum(bool(row["models"][candidate]["success"] and not row["models"][baseline]["success"]) for row in rows)
    both = sum(bool(row["models"][candidate]["success"] and row["models"][baseline]["success"]) for row in rows)
    neither = len(rows) - a_only - b_only - both
    result: dict[str, Any] = {"baseline": baseline, "candidate": candidate,
        "episodes": len(rows), "paired_success_contingency": {"both_success": both, "baseline_only_success": a_only,
        "candidate_only_success": b_only, "both_fail": neither}, "mcnemar_exact_p": _mcnemar_exact_p(a_only, b_only),
        "paired_metrics": {}}
    for metric in ("success", "final_position_error", "final_orientation_error", "final_gripper_width_error",
                   "collision", "trajectory_error", "trajectory_length"):
        delta = np.asarray([float(row["models"][candidate][metric]) - float(row["models"][baseline][metric]) for row in rows])
        result["paired_metrics"][metric] = {"candidate_minus_baseline_mean": float(delta.mean()) if len(delta) else 0.0,
                                             "ci95": _bootstrap_ci(delta, config.bootstrap_samples, label_seed + len(metric))}
    return result


def _aggregate_main(seed_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"conditions": {}, "seed_count": len(seed_payloads), "seeds": sorted(int(k) for k in seed_payloads)}
    for condition in ("iid", "surface_shape_ood", "sampling_ood"):
        results = [seed_payloads[seed][condition] for seed in sorted(seed_payloads)]
        models = results[0]["models"]
        model_agg: dict[str, Any] = {}
        for name in models:
            model_agg[name] = {}
            for metric in models[name]:
                if metric == "episodes":
                    model_agg[name][metric] = int(sum(result["models"][name][metric] for result in results))
                else:
                    model_agg[name][metric] = _mean_std([float(result["models"][name][metric]) for result in results])
        out["conditions"][condition] = {"models": model_agg}
        for baseline, candidate, key in (("centerline_set", "surface_set", "centerline_vs_surface_set"),
                                         ("surface_set", "surface_graph", "surface_set_vs_graph"),
                                         ("surface_set", "surface_graph_no_local", "surface_set_vs_graph_no_local"),
                                         ("surface_graph_no_local", "surface_graph", "graph_vs_no_local")):
            out["conditions"][condition][key] = _pooled_pair([result["episodes"] for result in results], baseline, candidate,
                                                              SurfaceFeasibilityConfig(bootstrap_samples=2000), 9901)
    return out


def _pick_spatial_representation(aggregate: dict[str, Any], latency: dict[str, Any] | None) -> dict[str, Any]:
    condition_names = ("iid", "surface_shape_ood")
    surface_deltas = [aggregate["conditions"][c]["centerline_vs_surface_set"]["paired_metrics"]["success"]["candidate_minus_baseline_mean"] for c in condition_names]
    surface_cis = [aggregate["conditions"][c]["centerline_vs_surface_set"]["paired_metrics"]["success"]["ci95"] for c in condition_names]
    surface_supported = float(np.mean(surface_deltas)) > 0.0 and any(ci[0] > 0.0 for ci in surface_cis)
    if not surface_supported:
        return {"selected": "centerline_set", "surface_value_supported": False,
                "reason": "Surface Set did not show a stable positive paired success difference over Centerline Set."}
    graph_deltas = [aggregate["conditions"][c]["surface_set_vs_graph"]["paired_metrics"]["success"]["candidate_minus_baseline_mean"] for c in condition_names]
    local_deltas = [aggregate["conditions"][c]["graph_vs_no_local"]["paired_metrics"]["success"]["candidate_minus_baseline_mean"] for c in condition_names]
    graph_gain = float(np.mean(graph_deltas))
    local_gain = float(np.mean(local_deltas))
    graph_ci = [aggregate["conditions"][c]["surface_set_vs_graph"]["paired_metrics"]["success"]["ci95"] for c in condition_names]
    local_ci = [aggregate["conditions"][c]["graph_vs_no_local"]["paired_metrics"]["success"]["ci95"] for c in condition_names]
    latency_ok = False
    latency_summary: dict[str, float] = {}
    if latency is not None:
        graph = latency["benchmarks"]["surface_graph_n32_rebuilt"]["geometry_to_action"]
        latency_summary = {"p99_ms": graph["p99_ms"], "deadline_miss_rate_over_5ms": graph["deadline_miss_rate_over_5ms"]}
        latency_ok = graph["p99_ms"] <= 5.0 and graph["deadline_miss_rate_over_5ms"] <= 0.01
    graph_supported = graph_gain > 0.0 and any(ci[0] > 0.0 for ci in graph_ci)
    local_supported = local_gain > 0.0 and any(ci[0] > 0.0 for ci in local_ci)
    if graph_supported and local_supported and latency_ok:
        selected = "surface_graph"
        reason = "Paired closed-loop gains over Surface Set and no-local-edge ablation were positive with CI support, and N=32 rebuilt p99 met the 5 ms / 1% miss design budget."
    else:
        selected = "surface_set"
        reason = "Surface geometry was supported; the graph did not satisfy every paired local-edge and latency adoption condition."
    return {"selected": selected, "surface_value_supported": True, "graph_gain_mean": graph_gain,
            "local_edge_gain_mean": local_gain, "graph_gain_ci95_by_condition": graph_ci,
            "local_edge_gain_ci95_by_condition": local_ci, "latency_gate": latency_summary, "reason": reason}


def _accuracy_latency_operating_point(accuracy_curve: dict[str, Any], latency: dict[str, Any], model_name: str) -> dict[str, Any]:
    rows = []
    for count in (16, 32, 64):
        success = float(accuracy_curve[model_name][str(count)]["success_rate"])
        p99 = float(latency["benchmarks"][f"{model_name}_n{count}_rebuilt"]["geometry_to_action"]["p99_ms"])
        rows.append({"surface_points": count, "success_rate": success, "geometry_to_action_p99_ms": p99})
    maximum = max(row["success_rate"] for row in rows)
    eligible = [row for row in rows if row["success_rate"] >= maximum - 0.03]
    chosen = min(eligible, key=lambda row: (row["surface_points"], row["geometry_to_action_p99_ms"]))
    return {"model": model_name, "curve": rows, "chosen": chosen,
            "selection_rule": "smallest point count within 3 percentage points of the best observed success rate; p99 remains a hard 5 ms budget"}


def surface_summary_markdown(aggregate: dict[str, Any], task_sanity: dict[str, Any], latency: dict[str, Any],
                             accuracy_curve: dict[str, Any], selection: dict[str, Any], efficiency_rows: list[dict[str, Any]],
                             config: SurfaceFeasibilityConfig, device: str) -> str:
    def mean_metric(condition: str, model: str, metric: str) -> float:
        return float(aggregate["conditions"][condition]["models"][model][metric]["mean"])

    q1 = aggregate["conditions"]["iid"]["centerline_vs_surface_set"]["paired_metrics"]["success"]
    q1_ood = aggregate["conditions"]["surface_shape_ood"]["centerline_vs_surface_set"]["paired_metrics"]["success"]
    q2 = aggregate["conditions"]["iid"]["surface_set_vs_graph"]["paired_metrics"]["success"]
    q3 = aggregate["conditions"]["iid"]["graph_vs_no_local"]["paired_metrics"]["success"]
    q4_delta_iid = q1["candidate_minus_baseline_mean"]
    q4_delta_ood = q1_ood["candidate_minus_baseline_mean"]
    latency_graph = latency["benchmarks"]["surface_graph_n32_rebuilt"]["geometry_to_action"]
    q5 = latency_graph["p99_ms"] <= 5.0 and latency_graph["deadline_miss_rate_over_5ms"] <= 0.01
    chosen_model = selection["selected"] if selection["selected"] in ("surface_set", "surface_graph") else "surface_set"
    operating = _accuracy_latency_operating_point(accuracy_curve, latency, chosen_model)
    max_accuracy = max(row["success_rate"] for row in operating["curve"])
    q1_baseline_floor = mean_metric("iid", "centerline_set", "success") < 0.15 and mean_metric("surface_shape_ood", "centerline_set", "success") < 0.15
    q2_baseline_floor = mean_metric("iid", "surface_set", "success") < 0.15 and mean_metric("surface_shape_ood", "surface_set", "success") < 0.15
    q3_baseline_floor = mean_metric("iid", "surface_graph_no_local", "success") < 0.15
    q1_answer = "YES" if q1["ci95"][0] > 0 else ("UNCERTAIN" if q1_baseline_floor else ("NO" if q1["ci95"][1] <= 0 else "UNCERTAIN"))
    q2_answer = "YES" if q2["ci95"][0] > 0 else ("UNCERTAIN" if q2_baseline_floor else ("NO" if q2["ci95"][1] <= 0 else "UNCERTAIN"))
    q3_answer = "YES" if q3["ci95"][0] > 0 else ("UNCERTAIN" if q3_baseline_floor else ("NO" if q3["ci95"][1] <= 0 else "UNCERTAIN"))
    q4_answer = "YES" if q4_delta_ood > q4_delta_iid + 0.02 else ("UNCERTAIN" if q1_baseline_floor else ("NO" if q4_delta_ood <= q4_delta_iid else "UNCERTAIN"))
    q5_answer = "YES" if q5 else "NO"
    q6_answer = "YES" if operating["chosen"]["success_rate"] >= max_accuracy - 0.03 else "UNCERTAIN"
    lines = [
        "# Surface-aware Pre-grasp Alignment Feasibility",
        "",
        "## Experimental setup",
        "",
        f"- Training seeds: `{', '.join(str(s) for s in config.seeds)}`; device: `{device}`; fixed feed-forward supervised imitation; no recurrent state.",
        f"- Main comparison: N={config.point_count}; {config.train_episodes} training and {config.validation_episodes} validation physical episodes per seed; {config.eval_episodes} paired evaluation episodes per seed and condition.",
        "- Task geometry uses a curved parametric centerline plus flat/rounded primitive cross sections, variable thickness and asymmetry. The sparse oracle point cloud samples the task-facing half surface visible to the planar robot. Mesh adjacency, normals, target frame and target opening are absent from model inputs.",
        "- Both input families receive 32 nodes, the same current EE pose, fingertip positions, measured aperture, object pose context, training episodes and expert labels. Graph edges use only observed-point kNN (k=6, radius 0.040 m), EE-global relations and nearest fingertip relations.",
        "- MuJoCo's current arm is planar in world x-z. The pinch-frame local x axis spans the fingers; planar yaw is a rotation about world y. The task rollout uses a position-plus-yaw IK solve and the native calibrated gripper joint range. Collision and clearance use signed MuJoCo primitive distances over the full hand, palm and finger geometries; arm links are excluded. Arm pose is stepped kinematically through MuJoCo forward kinematics; this experiment does not model force closure, frictional grasping, lifting, or RGB-D perception.",
        f"- Success: position < {config.success_position_threshold:.3f} m, orientation < {config.success_rotation_threshold:.3f} rad, aperture error < {config.success_opening_threshold:.3f} m, and no trajectory collision. These thresholds are saved in `config.json`.",
        "- Oracle task sanity success: **{:.3f}**; analytic observed-surface controller: **{:.3f}**. Paired necessity target position delta: {:.3f} m; opening delta: {:.3f} m; centerlines and object pose match.".format(
            task_sanity["oracle_expert_success_rate"], task_sanity["analytic_observed_surface_success_rate"],
            task_sanity["surface_necessity"]["target_position_delta"], task_sanity["surface_necessity"]["target_opening_delta"]),
        "- Geometry-to-action p99 ≤ 5 ms at 100 Hz is a provisional design budget for this model-selection exercise, not an industry-standard claim. Full sensor-to-command latency is **NOT MEASURED**.",
        "",
        "## N=32 closed-loop results",
        "",
        "| Model | IID success | Surface Shape OOD | Sampling OOD | IID collision | IID position error (m) | IID orientation error (rad) | IID opening error (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        lines.append("| {} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} | {:.4f} | {:.4f} | {:.4f} |".format(
            name, mean_metric("iid", name, "success"), aggregate["conditions"]["iid"]["models"][name]["success"]["std"],
            mean_metric("surface_shape_ood", name, "success"), aggregate["conditions"]["surface_shape_ood"]["models"][name]["success"]["std"],
            mean_metric("sampling_ood", name, "success"), aggregate["conditions"]["sampling_ood"]["models"][name]["success"]["std"],
            mean_metric("iid", name, "collision"), mean_metric("iid", name, "final_position_error"),
            mean_metric("iid", name, "final_orientation_error"), mean_metric("iid", name, "final_gripper_width_error")))
    lines.extend(["", "## Paired comparisons", ""])
    for condition in ("iid", "surface_shape_ood", "sampling_ood"):
        lines.append(f"### {condition.replace('_', ' ')}")
        lines.append("")
        for label in ("centerline_vs_surface_set", "surface_set_vs_graph", "graph_vs_no_local"):
            pair = aggregate["conditions"][condition][label]
            metric = pair["paired_metrics"]["success"]
            lines.append(f"- `{label}`: candidate-minus-baseline success {metric['candidate_minus_baseline_mean']:+.3f}, episode-bootstrap 95% CI [{metric['ci95'][0]:+.3f}, {metric['ci95'][1]:+.3f}], McNemar exact p={pair['mcnemar_exact_p']:.4f}; contingency={pair['paired_success_contingency']}.")
        lines.append("")
    lines.extend(["## Efficiency table", "", "| Model | N | Edges | MP steps | Parameters | Graph build p50/p99 ms | Forward p50/p99 ms | Geometry-to-action p50/p99 ms | >5 ms | Success | Collision |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in efficiency_rows:
        lines.append("| {model} | {n} | {edges} | {steps} | {params} | {gp50:.3f}/{gp99:.3f} | {fp50:.3f}/{fp99:.3f} | {tp50:.3f}/{tp99:.3f} | {miss:.1%} | {success:.3f} | {collision:.3f} |".format(**row))
    lines.extend(["", f"Latency device: `{latency['device']}`; each forward/geometry run used batch size 1, {config.latency_warmup} warm-up and {config.latency_iterations} timed iterations with async synchronization. Cached graph mode reused only Surface↔Surface pairs while rebuilding EE/fingertip relations. Cached topology assumes stable surface point identity; it is not used for shape-changing or reordered clouds.",
                  "", "## Accuracy–latency operating point", "", f"- Candidate model: `{operating['model']}`. Curve: `{json.dumps(operating['curve'])}`.",
                  f"- Selected N: **{operating['chosen']['surface_points']}**, with success {operating['chosen']['success_rate']:.3f} and geometry-to-action p99 {operating['chosen']['geometry_to_action_p99_ms']:.3f} ms ({operating['selection_rule']}).",
                  "", "## Adoption decision", "", f"- Selected spatial representation for GRU feasibility: **{selection['selected']}**.", f"- Rule result: {selection['reason']}",
                  "- The oracle and analytic controller solve the task, but learned-policy success is near the floor. The paired results support carrying the simpler centerline input into the next GRU feasibility, while Q1–Q4 remain uncertain about representation value under this feed-forward training budget.", "",
                  "## Final questions", "",
                  f"- Q1 Surface information helps over centerline: **{q1_answer}** (IID delta {q1['candidate_minus_baseline_mean']:+.3f}, CI [{q1['ci95'][0]:+.3f}, {q1['ci95'][1]:+.3f}]).",
                  f"- Q2 Surface Graph improves on the same Surface Set points: **{q2_answer}** (IID delta {q2['candidate_minus_baseline_mean']:+.3f}, CI [{q2['ci95'][0]:+.3f}, {q2['ci95'][1]:+.3f}]).",
                  f"- Q3 Local Surface↔Surface edges add an independent gain: **{q3_answer}** (IID delta {q3['candidate_minus_baseline_mean']:+.3f}, CI [{q3['ci95'][0]:+.3f}, {q3['ci95'][1]:+.3f}]).",
                  f"- Q4 Surface gain is larger on Shape OOD than IID: **{q4_answer}** (deltas {q4_delta_iid:+.3f} IID, {q4_delta_ood:+.3f} OOD).",
                  f"- Q5 Surface Graph computation meets the fast-controller budget: **{q5_answer}** (N=32 rebuilt p99 {latency_graph['p99_ms']:.3f} ms; >5 ms {latency_graph['deadline_miss_rate_over_5ms']:.1%}).",
                  f"- Q6 Best accuracy–latency point among N=16/32/64: **{q6_answer}**, N={operating['chosen']['surface_points']}.",
                  f"- Q7 Spatial representation passed to GRU feasibility: **{selection['selected']}**.",
                  "", "## Artifacts", "", "- `config.json`, `task_sanity.json`, `surface_necessity/`, `seed2811..seed2813/dataset/`, `checkpoints/`, `eval/`, `paired/`, `latency/n16|n32|n64/`, and `plots/`.",
                  "- The result is a sparse oracle-geometry controller feasibility study. It does not support a full perception-to-control latency or grasp-physics claim.", ""])
    return "\n".join(lines)


def run_surface_feasibility_experiment(config: SurfaceFeasibilityConfig | None = None) -> dict[str, Any]:
    config = config or SurfaceFeasibilityConfig()
    root = Path(config.output_dir)
    ensure_dir(root)
    config_path = root / "config.json"
    if config_path.exists():
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        expected = json.loads(json.dumps(asdict(config), default=_json_default))
        if stored != expected:
            raise FileExistsError(f"Existing experiment uses a different config; preserving artifacts at {config_path}")
        return _finalize_saved_surface_run(config)
    _write_json(config_path, asdict(config))
    set_seed(config.seeds[0])
    if select_device(config.device).type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))

    sanity_path = root / "task_sanity.json"
    if sanity_path.exists():
        task_sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        print("[step 1] using already-written task sanity artifact", flush=True)
    else:
        print("[step 1] oracle, analytic observed-surface and paired necessity checks", flush=True)
        task_sanity = run_task_sanity(config, config.seeds[0])

    # Step 2: small one-seed fit and closed-loop smoke before the main training run.
    print("[step 2] four-model smoke fit, IID closed-loop rollout and latency path", flush=True)
    smoke_config = replace(config, epochs=config.smoke_epochs)
    smoke_dataset = collect_expert_dataset(smoke_config, config.seeds[0] + 50_000,
                                           root / "smoke" / "dataset", train_episodes=12, validation_episodes=4)
    smoke_train: dict[str, Any] = {}
    smoke_checkpoints: dict[str, str] = {}
    for name in MODEL_NAMES:
        print(f"[smoke train] {name}", flush=True)
        row = train_surface_model(name, smoke_dataset, smoke_config, config.seeds[0] + 50_000,
                                  root / "smoke" / "checkpoints", epochs=config.smoke_epochs)
        smoke_train[name] = row
        smoke_checkpoints[name] = row["checkpoint_path"]
    smoke_device = select_device(config.eval_device)
    smoke_models = {name: load_surface_model(smoke_checkpoints[name], name, config, smoke_device) for name in MODEL_NAMES}
    smoke_specs = sample_episode_specs(config.smoke_episodes, config.seeds[0] + 51_000, config, "iid_smoke")
    smoke_eval = evaluate_models_paired(smoke_specs, smoke_models, config, smoke_device, root / "smoke" / "eval" / "iid")
    smoke_env = make_env(config, config.seeds[0] + 52_000)
    smoke_spec = smoke_specs[0]
    _reset_surface_env(smoke_env, smoke_spec)
    smoke_latency: dict[str, Any] = {}
    for name in MODEL_NAMES:
        print(f"[smoke latency] {name} N=32", flush=True)
        smoke_latency[name] = benchmark_latency(name, smoke_models[name], smoke_spec.shape, smoke_env,
                                                config, smoke_device, 32, "rebuilt", warmup=10, iterations=30)
    smoke_env.close()
    _write_json(root / "smoke" / "smoke_summary.json", {"training": smoke_train,
        "iid_models": smoke_eval["models"], "iid_paired_comparisons": smoke_eval["paired_comparisons"],
        "latency": smoke_latency})
    del smoke_models

    # Step 3: same episodes and expert labels for every model, across the requested seeds.
    main_by_seed: dict[str, dict[str, Any]] = {}
    checkpoint_map: dict[str, dict[str, str]] = {}
    training_by_seed: dict[str, dict[str, Any]] = {}
    for seed in config.seeds:
        print(f"[main] seed {seed}: generating shared expert trajectories", flush=True)
        seed_dir = ensure_dir(root / f"seed{seed}")
        dataset = collect_expert_dataset(config, seed, seed_dir / "dataset")
        training: dict[str, Any] = {}
        checkpoint_map[str(seed)] = {}
        for name in MODEL_NAMES:
            print(f"[main train] seed={seed} model={name}", flush=True)
            row = train_surface_model(name, dataset, config, seed, seed_dir / "checkpoints")
            training[name] = row
            checkpoint_map[str(seed)][name] = row["checkpoint_path"]
        training_by_seed[str(seed)] = training
        device = select_device(config.eval_device)
        models = {name: load_surface_model(checkpoint_map[str(seed)][name], name, config, device) for name in MODEL_NAMES}
        conditions = {
            "iid": (sample_episode_specs(config.eval_episodes, seed + 60_000, config, "iid"), SURFACE_POINTS, False),
            "surface_shape_ood": (sample_episode_specs(config.eval_episodes, seed + 61_000, config,
                                                       "surface_shape_ood", ood=True), SURFACE_POINTS, False),
            "sampling_ood": (sample_episode_specs(config.eval_episodes, seed + 62_000, config,
                                                  "sampling_ood"), 16, True),
        }
        seed_result: dict[str, Any] = {"training": training}
        for condition, (specs, n_points, nonuniform) in conditions.items():
            print(f"[closed-loop] seed={seed} condition={condition} episodes={len(specs)} N={n_points}", flush=True)
            eval_result = evaluate_models_paired(
                specs, models, config, device, root / "eval" / condition / f"seed{seed}",
                point_count=n_points, nonuniform=nonuniform,
            )
            seed_result[condition] = eval_result
            pair_dir = ensure_dir(root / "paired" / condition)
            _write_json(pair_dir / f"seed{seed}_paired_summary.json", eval_result["paired_comparisons"])
        main_by_seed[str(seed)] = seed_result
        del models, dataset

    aggregate = _aggregate_main(main_by_seed)
    aggregate["training"] = training_by_seed
    aggregate["task_sanity"] = task_sanity
    _write_json(root / "aggregated_results.json", aggregate)

    # N curve is a fixed resolution ablation of the two surface checkpoints, not a new architecture search.
    print("[resolution] evaluating fixed Surface Set and Surface Graph checkpoints at N=16/32/64", flush=True)
    resolution_seed = config.seeds[0]
    resolution_device = select_device(config.eval_device)
    selected_names: tuple[SurfaceModelName, SurfaceModelName] = ("surface_set", "surface_graph")
    resolution_models = {name: load_surface_model(checkpoint_map[str(resolution_seed)][name], name, config, resolution_device)
                         for name in selected_names}
    resolution_iid = sample_episode_specs(max(4, config.eval_episodes // 2), resolution_seed + 70_000, config, "resolution_iid")
    resolution_ood = sample_episode_specs(max(4, config.eval_episodes // 2), resolution_seed + 71_000, config,
                                          "resolution_surface_shape_ood", ood=True)
    resolution_specs = resolution_iid + resolution_ood
    accuracy_curve: dict[str, Any] = {name: {} for name in selected_names}
    for point_count in (16, 32, 64):
        curve_result = evaluate_models_paired(resolution_specs, resolution_models, config, resolution_device,
            root / "eval" / "resolution" / f"n{point_count}", point_count=point_count)
        for name in selected_names:
            accuracy_curve[name][str(point_count)] = {
                **curve_result["models"][name], "success_rate": curve_result["models"][name]["success"]
            }
    _write_json(root / "eval" / "resolution" / "accuracy_curve.json", accuracy_curve)
    del resolution_models

    # Step 4: the full batch-1 benchmark runs after all fitting and closed-loop work.
    print("[latency] beginning isolated batch-size-1 warm-up/timed benchmarks", flush=True)
    latency_shape = resolution_iid[0].shape
    latency = benchmark_experiment_latency(config, checkpoint_map, latency_shape, resolution_seed)

    selection = _pick_spatial_representation(aggregate, latency)
    efficiency_rows: list[dict[str, Any]] = []
    diagnostic_env = make_env(config, resolution_seed + 80_000)
    _reset_surface_env(diagnostic_env, resolution_iid[0])
    state, surface_points, _ = observation_inputs(diagnostic_env, resolution_iid[0].shape, 32,
                                                   _stable_seed(resolution_iid[0].sample_identity))
    diagnostic_topology = build_graph_topology_numpy(surface_points, state[8:14].reshape(2, 3), config, True)
    no_local_topology = build_graph_topology_numpy(surface_points, state[8:14].reshape(2, 3), config, False)
    collision_shape = resolution_iid[0].shape
    collision_geoms = len(diagnostic_env.surface_box_ids if collision_shape.cross_section == "flat" else diagnostic_env.surface_capsule_ids)
    plot_geometry_sample(resolution_iid[0].shape, root / "plots" / "geometry" / "surface_graph_sample.png",
                         seed=_stable_seed(resolution_iid[0].sample_identity),
                         ee_position=diagnostic_env.robot_observation().ee_position.numpy(), ee_yaw=tool_yaw(diagnostic_env),
                         topology=diagnostic_topology,
                         fingertip_positions=np.stack([
                             np.mean([diagnostic_env.data.geom_xpos[diagnostic_env._geom_id(n)] for n in ("thumbtip1", "thumbtip2")], axis=0),
                             np.mean([diagnostic_env.data.geom_xpos[diagnostic_env._geom_id(n)] for n in ("fingertip1", "fingertip2")], axis=0),
                         ]))
    diagnostic_env.close()
    for name in MODEL_NAMES:
        lat = latency["benchmarks"][f"{name}_n32_rebuilt"]
        if name == "surface_graph":
            edges = int(len(diagnostic_topology["src"]))
        elif name == "surface_graph_no_local":
            edges = int(len(no_local_topology["src"]))
        else:
            edges = 0
        efficiency_rows.append({
            "model": name, "n": 32, "edges": edges,
            "steps": config.message_passing_steps if name.startswith("surface_graph") else 0,
            "params": training_by_seed[str(resolution_seed)][name]["parameters"],
            "gp50": lat["components"]["graph_construction"]["p50_ms"],
            "gp99": lat["components"]["graph_construction"]["p99_ms"],
            "fp50": lat["network_forward"]["p50_ms"], "fp99": lat["network_forward"]["p99_ms"],
            "tp50": lat["geometry_to_action"]["p50_ms"], "tp99": lat["geometry_to_action"]["p99_ms"],
            "miss": lat["geometry_to_action"]["deadline_miss_rate_over_5ms"],
            "success": aggregate["conditions"]["iid"]["models"][name]["success"]["mean"],
            "collision": aggregate["conditions"]["iid"]["models"][name]["collision"]["mean"],
        })
    for name in selected_names:
        for point_count in (16, 64):
            lat = latency["benchmarks"][f"{name}_n{point_count}_rebuilt"]
            efficiency_rows.append({
                "model": name, "n": point_count, "edges": int(len(diagnostic_topology["src"]) * point_count / 32) if name == "surface_graph" else 0,
                "steps": config.message_passing_steps if name == "surface_graph" else 0,
                "params": training_by_seed[str(resolution_seed)][name]["parameters"],
                "gp50": lat["components"]["graph_construction"]["p50_ms"], "gp99": lat["components"]["graph_construction"]["p99_ms"],
                "fp50": lat["network_forward"]["p50_ms"], "fp99": lat["network_forward"]["p99_ms"],
                "tp50": lat["geometry_to_action"]["p50_ms"], "tp99": lat["geometry_to_action"]["p99_ms"],
                "miss": lat["geometry_to_action"]["deadline_miss_rate_over_5ms"],
                "success": accuracy_curve[name][str(point_count)]["success_rate"],
                "collision": accuracy_curve[name][str(point_count)]["collision"],
            })
    for n in (16, 32, 64):
        cache = latency["benchmarks"][f"surface_graph_n{n}_cached"]
        efficiency_rows.append({
            "model": "surface_graph_cached", "n": n, "edges": int(len(diagnostic_topology["src"]) * n / 32),
            "steps": config.message_passing_steps, "params": training_by_seed[str(resolution_seed)]["surface_graph"]["parameters"],
            "gp50": cache["components"]["graph_construction"]["p50_ms"], "gp99": cache["components"]["graph_construction"]["p99_ms"],
            "fp50": cache["network_forward"]["p50_ms"], "fp99": cache["network_forward"]["p99_ms"],
            "tp50": cache["geometry_to_action"]["p50_ms"], "tp99": cache["geometry_to_action"]["p99_ms"],
            "miss": cache["geometry_to_action"]["deadline_miss_rate_over_5ms"],
            "success": accuracy_curve["surface_graph"][str(n)]["success_rate"],
            "collision": accuracy_curve["surface_graph"][str(n)]["collision"],
        })
    plot_cases = plot_trajectory_cases(main_by_seed[str(resolution_seed)]["iid"]["episodes"], root / "plots" / "failure_cases", config.plot_cases)
    plot_latency_curve(latency, accuracy_curve, root / "plots" / "latency_accuracy_curve.png")
    selection["operating_point"] = _accuracy_latency_operating_point(
        accuracy_curve, latency, selection["selected"] if selection["selected"] in selected_names else "surface_set")
    aggregate.update({"selection": selection, "accuracy_latency_curve": accuracy_curve,
                      "task_sanity": task_sanity, "latency": latency, "efficiency_rows": efficiency_rows,
                      "failure_case_plots": plot_cases})
    _write_json(root / "aggregated_results.json", aggregate)
    _write_json(root / "paired" / "pooled_paired_summary.json", {
        condition: {label: aggregate["conditions"][condition][label] for label in (
            "centerline_vs_surface_set", "surface_set_vs_graph", "surface_set_vs_graph_no_local", "graph_vs_no_local")}
        for condition in ("iid", "surface_shape_ood", "sampling_ood")})
    summary = surface_summary_markdown(aggregate, task_sanity, latency, accuracy_curve, selection,
                                       efficiency_rows, config, str(select_device(config.eval_device)))
    (root / "summary.md").write_text(summary, encoding="utf-8")
    print(f"[complete] summary: {root / 'summary.md'}", flush=True)
    return aggregate


def _finalize_saved_surface_run(config: SurfaceFeasibilityConfig) -> dict[str, Any]:
    """Finish report generation from completed fit/eval/latency artifacts without rerunning or overwriting them."""
    root = Path(config.output_dir)
    print("[resume] found matching config; finalizing existing measurements only", flush=True)
    aggregate = json.loads((root / "aggregated_results.json").read_text(encoding="utf-8"))
    task_sanity = json.loads((root / "task_sanity.json").read_text(encoding="utf-8"))
    training_by_seed = aggregate["training"]
    latency = json.loads((root / "latency" / "latency_results.json").read_text(encoding="utf-8"))
    resolution_seed = config.seeds[0]
    resolution_iid = sample_episode_specs(max(4, config.eval_episodes // 2), resolution_seed + 70_000,
                                         config, "resolution_iid")
    resolution_ood = sample_episode_specs(max(4, config.eval_episodes // 2), resolution_seed + 71_000,
                                          config, "resolution_surface_shape_ood", ood=True)
    selected_names: tuple[SurfaceModelName, SurfaceModelName] = ("surface_set", "surface_graph")
    accuracy_curve: dict[str, Any] = {name: {} for name in selected_names}
    for count in (16, 32, 64):
        saved = json.loads((root / "eval" / "resolution" / f"n{count}" / "episode_results.json").read_text(encoding="utf-8"))
        for name in selected_names:
            row = saved["models"][name]
            accuracy_curve[name][str(count)] = {**row, "success_rate": row["success"]}
    _write_json(root / "eval" / "resolution" / "accuracy_curve.json", accuracy_curve)

    diagnostic_env = make_env(config, resolution_seed + 80_000)
    diagnostic_spec = resolution_iid[0]
    _reset_surface_env(diagnostic_env, diagnostic_spec)
    state, surface_points, _ = observation_inputs(diagnostic_env, diagnostic_spec.shape, 32,
                                                   _stable_seed(diagnostic_spec.sample_identity))
    diagnostic_topology = build_graph_topology_numpy(surface_points, state[8:14].reshape(2, 3), config, True)
    no_local_topology = build_graph_topology_numpy(surface_points, state[8:14].reshape(2, 3), config, False)
    plot_geometry_sample(diagnostic_spec.shape, root / "plots" / "geometry" / "surface_graph_sample.png",
                         seed=_stable_seed(diagnostic_spec.sample_identity),
                         ee_position=diagnostic_env.robot_observation().ee_position.numpy(),
                         ee_yaw=tool_yaw(diagnostic_env), topology=diagnostic_topology,
                         fingertip_positions=np.stack([
                             np.mean([diagnostic_env.data.geom_xpos[diagnostic_env._geom_id(n)] for n in ("thumbtip1", "thumbtip2")], axis=0),
                             np.mean([diagnostic_env.data.geom_xpos[diagnostic_env._geom_id(n)] for n in ("fingertip1", "fingertip2")], axis=0),
                         ]))
    diagnostic_env.close()

    efficiency_rows: list[dict[str, Any]] = []
    for name in MODEL_NAMES:
        lat = latency["benchmarks"][f"{name}_n32_rebuilt"]
        edges = len(diagnostic_topology["src"]) if name == "surface_graph" else (
            len(no_local_topology["src"]) if name == "surface_graph_no_local" else 0)
        efficiency_rows.append({
            "model": name, "n": 32, "edges": int(edges),
            "steps": config.message_passing_steps if name.startswith("surface_graph") else 0,
            "params": training_by_seed[str(resolution_seed)][name]["parameters"],
            "gp50": lat["components"]["graph_construction"]["p50_ms"],
            "gp99": lat["components"]["graph_construction"]["p99_ms"],
            "fp50": lat["network_forward"]["p50_ms"], "fp99": lat["network_forward"]["p99_ms"],
            "tp50": lat["geometry_to_action"]["p50_ms"], "tp99": lat["geometry_to_action"]["p99_ms"],
            "miss": lat["geometry_to_action"]["deadline_miss_rate_over_5ms"],
            "success": aggregate["conditions"]["iid"]["models"][name]["success"]["mean"],
            "collision": aggregate["conditions"]["iid"]["models"][name]["collision"]["mean"],
        })
    for name in selected_names:
        for count in (16, 64):
            lat = latency["benchmarks"][f"{name}_n{count}_rebuilt"]
            efficiency_rows.append({
                "model": name, "n": count,
                "edges": int(len(diagnostic_topology["src"]) * count / 32) if name == "surface_graph" else 0,
                "steps": config.message_passing_steps if name == "surface_graph" else 0,
                "params": training_by_seed[str(resolution_seed)][name]["parameters"],
                "gp50": lat["components"]["graph_construction"]["p50_ms"], "gp99": lat["components"]["graph_construction"]["p99_ms"],
                "fp50": lat["network_forward"]["p50_ms"], "fp99": lat["network_forward"]["p99_ms"],
                "tp50": lat["geometry_to_action"]["p50_ms"], "tp99": lat["geometry_to_action"]["p99_ms"],
                "miss": lat["geometry_to_action"]["deadline_miss_rate_over_5ms"],
                "success": accuracy_curve[name][str(count)]["success_rate"],
                "collision": accuracy_curve[name][str(count)]["collision"],
            })
    for count in (16, 32, 64):
        cached = latency["benchmarks"][f"surface_graph_n{count}_cached"]
        efficiency_rows.append({
            "model": "surface_graph_cached", "n": count,
            "edges": int(len(diagnostic_topology["src"]) * count / 32),
            "steps": config.message_passing_steps,
            "params": training_by_seed[str(resolution_seed)]["surface_graph"]["parameters"],
            "gp50": cached["components"]["graph_construction"]["p50_ms"],
            "gp99": cached["components"]["graph_construction"]["p99_ms"],
            "fp50": cached["network_forward"]["p50_ms"], "fp99": cached["network_forward"]["p99_ms"],
            "tp50": cached["geometry_to_action"]["p50_ms"], "tp99": cached["geometry_to_action"]["p99_ms"],
            "miss": cached["geometry_to_action"]["deadline_miss_rate_over_5ms"],
            "success": accuracy_curve["surface_graph"][str(count)]["success_rate"],
            "collision": accuracy_curve["surface_graph"][str(count)]["collision"],
        })

    main_iid = json.loads((root / "eval" / "iid" / f"seed{resolution_seed}" / "episode_results.json").read_text(encoding="utf-8"))
    plot_cases = plot_trajectory_cases(main_iid["episodes"], root / "plots" / "failure_cases", config.plot_cases)
    plot_latency_curve(latency, accuracy_curve, root / "plots" / "latency_accuracy_curve.png")
    selection = _pick_spatial_representation(aggregate, latency)
    operating_model = selection["selected"] if selection["selected"] in selected_names else "surface_set"
    selection["operating_point"] = _accuracy_latency_operating_point(accuracy_curve, latency, operating_model)
    aggregate.update({"selection": selection, "accuracy_latency_curve": accuracy_curve,
                      "task_sanity": task_sanity, "latency": latency, "efficiency_rows": efficiency_rows,
                      "failure_case_plots": plot_cases})
    _write_json(root / "aggregated_results.json", aggregate)
    _write_json(root / "paired" / "pooled_paired_summary.json", {
        condition: {label: aggregate["conditions"][condition][label] for label in (
            "centerline_vs_surface_set", "surface_set_vs_graph", "surface_set_vs_graph_no_local", "graph_vs_no_local")}
        for condition in ("iid", "surface_shape_ood", "sampling_ood")})
    summary = surface_summary_markdown(aggregate, task_sanity, latency, accuracy_curve, selection,
                                       efficiency_rows, config, str(select_device(config.eval_device)))
    (root / "summary.md").write_text(summary, encoding="utf-8")
    print(f"[complete] summary: {root / 'summary.md'}", flush=True)
    return aggregate



def plot_latency_curve(latency: dict[str, Any], accuracy_curve: dict[str, Any], output_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = (16, 32, 64)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for model_name, label in (("surface_set", "Surface Set"), ("surface_graph", "Surface Graph")):
        p50 = []; p99 = []; success = []
        for n in points:
            row = latency["benchmarks"][f"{model_name}_n{n}_rebuilt"]
            p50.append(row["geometry_to_action"]["p50_ms"]); p99.append(row["geometry_to_action"]["p99_ms"])
            success.append(accuracy_curve[model_name][str(n)]["success_rate"])
        axes[0].plot(points, p50, "-o", label=f"{label} p50")
        axes[0].plot(points, p99, "--o", label=f"{label} p99")
        axes[1].plot(points, success, "-o", label=label)
    axes[0].axhline(5.0, color="red", linestyle=":", label="5 ms design budget")
    axes[0].set_xlabel("surface point count N"); axes[0].set_ylabel("geometry-to-action (ms)"); axes[0].grid(alpha=0.25); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("surface point count N"); axes[1].set_ylabel("closed-loop success rate"); axes[1].set_ylim(-0.03, 1.03); axes[1].grid(alpha=0.25); axes[1].legend(fontsize=8)
    fig.tight_layout(); Path(output_path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(output_path, dpi=150); plt.close(fig)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
