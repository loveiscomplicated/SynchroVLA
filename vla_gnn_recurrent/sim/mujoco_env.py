from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dm_control
import mujoco
import numpy as np
import torch

from vla_gnn_recurrent.graph.types import DestinationObservation, ObjectObservation
from vla_gnn_recurrent.sim.types import GraspObservation, IKResult, ManipulationObservation, RobotObservation
from vla_gnn_recurrent.utils import clamp_delta


@dataclass
class MujocoReachConfig:
    max_steps: int = 48
    control_substeps: int = 25
    max_delta_ee: float = 0.035
    target_radius: float = 0.045
    ik_damping: float = 0.05
    ik_max_iters: int = 40
    ik_tolerance: float = 0.006
    ik_max_joint_delta: float = 0.08
    joint_kp: float = 300.0
    joint_kd: float = 50.0
    motor_ctrl_limit: float = 3.2
    actuator_force_limit: float = 500.0
    joint_damping: float = 5.0
    joint_armature: float = 0.5
    kinematic_joint_control: bool = False
    pick_scene: bool = False
    support_position: tuple[float, float, float] = (0.0, 0.0, 0.36)
    support_size: tuple[float, float, float] = (0.07, 0.012, 0.012)
    object_radius: float = 0.022
    lift_height: float = 0.10
    lift_hold_steps: int = 8
    target_velocity_min: float = 0.002
    target_velocity_max: float = 0.006


class MujocoManipulatorEnv:
    """MuJoCo articulated reaching wrapper around dm_control's manipulator model.

    Robot/model source:
        dm_control.suite.manipulator.xml, a maintained MuJoCo planar manipulator
        with articulated arm, two-finger gripper, and object/contact assets.
    """

    ARM_JOINTS = ("arm_root", "arm_shoulder", "arm_elbow", "arm_wrist")
    FINGER_JOINTS = ("thumb", "finger")
    ARM_ACTUATORS = ("root", "shoulder", "elbow", "wrist")
    GRIPPER_ACTUATOR = "grasp"
    JOINT_BODIES = ("upper_arm", "middle_arm", "lower_arm", "hand")

    def __init__(self, config: MujocoReachConfig | None = None, seed: int = 0) -> None:
        self.config = config or MujocoReachConfig()
        self.xml_path = Path(dm_control.__file__).parent / "suite" / "manipulator.xml"
        self.model = self._load_model()
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.step_count = 0
        self.target_motion_velocity = np.zeros(3, dtype=np.float64)
        self.target_velocity = np.zeros(3, dtype=np.float64)
        self._renderer: mujoco.Renderer | None = None

        self.ee_site_id = self._site_id("pinch")
        self.grasp_site_id = self._site_id("grasp")
        self.target_site_id = self._site_id("target_ball")
        self.object_site_id = self._site_id("ball")
        self.ball_body_id = self._body_id("ball")
        self.ball_geom_id = self._geom_id("ball")
        self.thumb_geom_ids = {self._geom_id(name) for name in ("thumb1", "thumb2", "thumbtip1", "thumbtip2")}
        self.finger_geom_ids = {self._geom_id(name) for name in ("finger1", "finger2", "fingertip1", "fingertip2")}
        self.target_body_id = self._body_id("target_ball")
        self.arm_joint_ids = np.array([self._joint_id(name) for name in self.ARM_JOINTS], dtype=np.int32)
        self.finger_joint_ids = np.array([self._joint_id(name) for name in self.FINGER_JOINTS], dtype=np.int32)
        self.arm_qpos_ids = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_dof_ids = self.model.jnt_dofadr[self.arm_joint_ids]
        self.finger_qpos_ids = self.model.jnt_qposadr[self.finger_joint_ids]
        self.finger_dof_ids = self.model.jnt_dofadr[self.finger_joint_ids]
        self.arm_actuator_ids = np.array([self._actuator_id(name) for name in self.ARM_ACTUATORS], dtype=np.int32)
        self.gripper_actuator_id = self._actuator_id(self.GRIPPER_ACTUATOR)
        self._configure_actuators()
        self.joint_body_ids = [self._body_id(name) for name in self.JOINT_BODIES]
        self.arm_joint_ranges = self.model.jnt_range[self.arm_joint_ids].copy()
        self.home_qpos = np.array([0.0, -0.35, 0.95, -0.45], dtype=np.float64)
        self.workspace_low = np.array([-0.42, -0.001, 0.16], dtype=np.float64)
        self.workspace_high = np.array([0.42, 0.001, 0.95], dtype=np.float64)
        self.last_arm_qpos_before_control = self.data.qpos[self.arm_qpos_ids].copy()

    def reset(
        self,
        target_position: np.ndarray | torch.Tensor | None = None,
        randomize_robot: bool = True,
    ) -> ManipulationObservation:
        mujoco.mj_resetData(self.model, self.data)
        qpos = self.home_qpos.copy()
        if randomize_robot:
            qpos += self.rng.uniform(low=-0.25, high=0.25, size=4)
        self._set_arm_qpos(self._clip_arm_qpos(qpos))
        self.data.qpos[self.finger_qpos_ids] = np.array([0.0, 0.0], dtype=np.float64)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.step_count = 0
        self.target_motion_velocity[:] = 0.0
        self.target_velocity[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if target_position is None:
            target_position = self._sample_target_away_from_ee()
        self.set_target_position(np.asarray(target_position, dtype=np.float64))
        return self.observe(reanchored=True)

    def reset_moving_target(self) -> ManipulationObservation:
        observation = self.reset()
        self.target_motion_velocity = self.sample_target_velocity()
        return observation

    def reset_pick_scene(
        self,
        object_x: float = 0.0,
        object_yaw: float = 0.0,
        randomize_robot: bool = True,
    ) -> ManipulationObservation:
        self.reset(randomize_robot=randomize_robot)
        hold_q = self.data.qpos[self.arm_qpos_ids].copy()
        for _ in range(500):
            self.data.ctrl[self.arm_actuator_ids] = np.clip(
                hold_q,
                self.model.actuator_ctrlrange[self.arm_actuator_ids, 0],
                self.model.actuator_ctrlrange[self.arm_actuator_ids, 1],
            )
            self.data.ctrl[self.gripper_actuator_id] = self._gripper_ctrl(0.0)
            mujoco.mj_step(self.model, self.data)
        z = self.config.support_position[2] + self.config.support_size[2] + self.config.object_radius + 0.003
        self.set_object_position(np.array([object_x, 0.0, z], dtype=np.float64), yaw=object_yaw)
        self.data.qvel[:] = 0.0
        for _ in range(80):
            self.data.ctrl[self.arm_actuator_ids] = np.clip(
                self.data.qpos[self.arm_qpos_ids],
                self.model.actuator_ctrlrange[self.arm_actuator_ids, 0],
                self.model.actuator_ctrlrange[self.arm_actuator_ids, 1],
            )
            self.data.ctrl[self.gripper_actuator_id] = self._gripper_ctrl(0.0)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.observe(reanchored=True)

    def reset_pick_place_scene(
        self,
        object_x: float = -0.02,
        destination_x: float = 0.035,
        object_yaw: float = 0.0,
        randomize_robot: bool = True,
    ) -> ManipulationObservation:
        observation = self.reset_pick_scene(object_x=object_x, object_yaw=object_yaw, randomize_robot=randomize_robot)
        destination = self.pick_surface_object_center(destination_x)
        self.set_target_position(destination)
        return self.observe(reanchored=observation.reanchored)

    def pick_surface_object_center(self, x: float) -> np.ndarray:
        z = self.config.support_position[2] + self.config.support_size[2] + self.config.object_radius + 0.003
        return np.array([float(x), 0.0, z], dtype=np.float64)

    def observe(self, reanchored: bool = True, observable_event: bool = False) -> ManipulationObservation:
        robot = self.robot_observation()
        target = self.target_observation()
        destination = self.destination_observation()
        obj = self.object_observation()
        return ManipulationObservation(
            robot=robot,
            target=target,
            true_target=target,
            object=obj,
            true_object=obj,
            destination=destination,
            true_destination=destination,
            step_count=self.step_count,
            reanchored=reanchored,
            observable_event=observable_event,
            grasp=self.grasp_observation(),
        )

    def robot_observation(self) -> RobotObservation:
        mujoco.mj_forward(self.model, self.data)
        ee_position = torch.tensor(self.data.site_xpos[self.ee_site_id].copy(), dtype=torch.float32)
        ee_orientation = torch.tensor(self._site_quat(self.ee_site_id), dtype=torch.float32)
        joint_positions = torch.tensor(self.data.qpos[self.arm_qpos_ids].copy(), dtype=torch.float32)
        joint_velocities = torch.tensor(self.data.qvel[self.arm_dof_ids].copy(), dtype=torch.float32)
        joint_world_positions = torch.tensor(
            np.stack([self.data.xpos[body_id].copy() for body_id in self.joint_body_ids], axis=0),
            dtype=torch.float32,
        )
        gripper_q = self.data.qpos[self.finger_qpos_ids].copy()
        gripper_position = torch.tensor(self.data.site_xpos[self.grasp_site_id].copy(), dtype=torch.float32)
        return RobotObservation(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_world_positions=joint_world_positions,
            ee_position=ee_position,
            ee_orientation=ee_orientation,
            ee_velocity=self._ee_velocity_tensor(),
            gripper_state=torch.tensor([float(np.mean(gripper_q))], dtype=torch.float32),
            gripper_position=gripper_position,
        )

    def target_observation(self) -> ObjectObservation:
        return ObjectObservation.from_position(
            object_id="reach_target",
            position=torch.tensor(self.data.site_xpos[self.target_site_id].copy(), dtype=torch.float32),
            orientation=torch.tensor(self._site_quat(self.target_site_id), dtype=torch.float32),
            velocity=torch.tensor(self.target_velocity.copy(), dtype=torch.float32),
            is_target=True,
        )

    def destination_observation(self) -> DestinationObservation:
        return DestinationObservation.from_position(
            destination_id="destination",
            position=torch.tensor(self.data.site_xpos[self.target_site_id].copy(), dtype=torch.float32),
            orientation=torch.tensor(self._site_quat(self.target_site_id), dtype=torch.float32),
            velocity=torch.tensor(self.target_velocity.copy(), dtype=torch.float32),
        )

    def object_observation(self) -> ObjectObservation:
        return ObjectObservation.from_position(
            object_id="ball",
            position=torch.tensor(self.data.site_xpos[self.object_site_id].copy(), dtype=torch.float32),
            orientation=torch.tensor(self._site_quat(self.object_site_id), dtype=torch.float32),
            velocity=torch.zeros(3),
            is_target=False,
        )

    def grasp_observation(self) -> GraspObservation:
        left_contact = False
        right_contact = False
        contact_force = 0.0
        object_contacts = 0
        force = np.zeros(6, dtype=np.float64)
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            geoms = {int(contact.geom1), int(contact.geom2)}
            if self.ball_geom_id not in geoms:
                continue
            object_contacts += 1
            if geoms & self.thumb_geom_ids:
                left_contact = True
            if geoms & self.finger_geom_ids:
                right_contact = True
            mujoco.mj_contactForce(self.model, self.data, idx, force)
            contact_force += abs(float(force[0]))
        object_height = float(self.data.site_xpos[self.object_site_id][2])
        ball_x_dof = self.model.jnt_dofadr[self._joint_id("ball_x")]
        ball_z_dof = self.model.jnt_dofadr[self._joint_id("ball_z")]
        object_velocity = torch.tensor(
            [float(self.data.qvel[ball_x_dof]), 0.0, float(self.data.qvel[ball_z_dof])],
            dtype=torch.float32,
        )
        gripper_distance = float(np.linalg.norm(self.data.site_xpos[self.object_site_id] - self.data.site_xpos[self.ee_site_id]))
        object_grasped = bool(left_contact and right_contact and gripper_distance < 0.075)
        return GraspObservation(
            left_contact=left_contact,
            right_contact=right_contact,
            contact_force=contact_force,
            object_grasped=object_grasped,
            object_height=object_height,
            object_velocity=object_velocity,
            num_contacts=object_contacts,
        )

    def step_delta_ee(self, delta_ee: torch.Tensor, gripper: float = 0.0) -> tuple[ManipulationObservation, float, bool, dict[str, Any]]:
        bounded = clamp_delta(delta_ee.detach().cpu().float(), self.config.max_delta_ee).numpy().astype(np.float64)
        current = self.data.site_xpos[self.ee_site_id].copy()
        desired = current + bounded
        desired[1] = current[1]
        desired = np.clip(desired, self.workspace_low, self.workspace_high)
        ik = self.solve_ik(desired)
        self._track_joint_target(ik.qpos.numpy(), gripper=gripper)
        self._advance_target_motion()
        self.step_count += 1
        distance = self.distance_to_target()
        done = distance <= self.config.target_radius or self.step_count >= self.config.max_steps
        info = {
            "distance": distance,
            "success": distance <= self.config.target_radius,
            "timeout": self.step_count >= self.config.max_steps,
            "ik_success": ik.success,
            "ik_error": ik.final_error,
            "grasp": self.grasp_observation(),
        }
        return self.observe(reanchored=True), -distance, done, info

    def step_toward_ee_target(
        self,
        ee_target: np.ndarray | torch.Tensor,
        gripper: float = 0.0,
    ) -> tuple[ManipulationObservation, float, bool, dict[str, Any]]:
        current = self.data.site_xpos[self.ee_site_id].copy()
        target = np.asarray(ee_target, dtype=np.float64).reshape(3)
        delta = torch.tensor(target - current, dtype=torch.float32)
        return self.step_delta_ee(delta, gripper=gripper)

    def solve_ik(self, target_position: np.ndarray | torch.Tensor) -> IKResult:
        target = np.asarray(target_position, dtype=np.float64).reshape(3)
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0.0
        q = scratch.qpos[self.arm_qpos_ids].copy()
        clipped = False
        axes = np.array([0, 2], dtype=np.int32)

        for iteration in range(1, self.config.ik_max_iters + 1):
            scratch.qpos[self.arm_qpos_ids] = q
            mujoco.mj_forward(self.model, scratch)
            current = scratch.site_xpos[self.ee_site_id].copy()
            error = target[axes] - current[axes]
            if float(np.linalg.norm(error)) <= self.config.ik_tolerance:
                return IKResult(True, torch.tensor(q, dtype=torch.float32), float(np.linalg.norm(error)), iteration, clipped)

            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self.ee_site_id)
            j = jacp[np.ix_(axes, self.arm_dof_ids)]
            lhs = j @ j.T + (self.config.ik_damping**2) * np.eye(len(axes), dtype=np.float64)
            dq = j.T @ np.linalg.solve(lhs, error)
            dq = np.clip(dq, -self.config.ik_max_joint_delta, self.config.ik_max_joint_delta)
            q = q + dq
            q_clipped = self._clip_arm_qpos(q)
            clipped = clipped or bool(np.max(np.abs(q_clipped - q)) > 1e-9)
            q = q_clipped

        scratch.qpos[self.arm_qpos_ids] = q
        mujoco.mj_forward(self.model, scratch)
        final_error = float(np.linalg.norm(target[axes] - scratch.site_xpos[self.ee_site_id][axes]))
        return IKResult(False, torch.tensor(q, dtype=torch.float32), final_error, self.config.ik_max_iters, clipped)

    def expert_action(self) -> torch.Tensor:
        delta = self.target_observation().position - self.robot_observation().ee_position
        delta[1] = 0.0
        return clamp_delta(delta, self.config.max_delta_ee)

    def move_target(self, target_position: np.ndarray | torch.Tensor) -> None:
        previous = self.target_position_np()
        self.set_target_position(np.asarray(target_position, dtype=np.float64).reshape(3))
        self.target_velocity = self.target_position_np() - previous
        self.target_motion_velocity[:] = 0.0

    def set_object_position(self, position: np.ndarray | torch.Tensor, yaw: float = 0.0) -> None:
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        self.data.qpos[self.model.jnt_qposadr[self._joint_id("ball_x")]] = pos[0]
        self.data.qpos[self.model.jnt_qposadr[self._joint_id("ball_z")]] = pos[2]
        self.data.qpos[self.model.jnt_qposadr[self._joint_id("ball_y")]] = float(yaw)
        self.data.qvel[self.model.jnt_dofadr[self._joint_id("ball_x")]] = 0.0
        self.data.qvel[self.model.jnt_dofadr[self._joint_id("ball_z")]] = 0.0
        self.data.qvel[self.model.jnt_dofadr[self._joint_id("ball_y")]] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def sample_perturbed_target(self, min_distance: float = 0.18, max_distance: float = 0.36) -> np.ndarray:
        ee = self.data.site_xpos[self.ee_site_id].copy()
        old = self.target_position_np() - ee
        if np.linalg.norm(old[[0, 2]]) < 1e-6:
            old = self._random_xz_unit()
        direction = -old
        noise = 0.55 * self._random_xz_unit()
        direction = direction + noise
        direction[1] = 0.0
        direction = direction / max(np.linalg.norm(direction), 1e-6)
        distance = float(self.rng.uniform(min_distance, max_distance))
        target = ee + direction * distance
        return self._clip_target(target)

    def sample_target_velocity(self) -> np.ndarray:
        direction = self._random_xz_unit()
        speed = float(self.rng.uniform(self.config.target_velocity_min, self.config.target_velocity_max))
        return direction * speed

    def set_target_position(self, position: np.ndarray) -> None:
        self.model.body_pos[self.target_body_id] = self._clip_target(position)
        mujoco.mj_forward(self.model, self.data)

    def target_position_np(self) -> np.ndarray:
        return self.data.site_xpos[self.target_site_id].copy()

    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.target_position_np() - self.data.site_xpos[self.ee_site_id]))

    def render_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera="fixed")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _track_joint_target(self, q_target: np.ndarray, gripper: float) -> None:
        q_target = self._clip_arm_qpos(q_target)
        gripper_ctrl = self._gripper_ctrl(gripper)
        self.last_arm_qpos_before_control = self.data.qpos[self.arm_qpos_ids].copy()
        if self.config.kinematic_joint_control:
            previous_q = self.data.qpos[self.arm_qpos_ids].copy()
            for _ in range(self.config.control_substeps):
                q = self.data.qpos[self.arm_qpos_ids].copy()
                delta = np.clip(q_target - q, -self.config.ik_max_joint_delta, self.config.ik_max_joint_delta)
                self.data.qpos[self.arm_qpos_ids] = self._clip_arm_qpos(q + delta)
                self.data.qvel[self.arm_dof_ids] = (self.data.qpos[self.arm_qpos_ids] - previous_q) / max(
                    self.model.opt.timestep,
                    1e-6,
                )
                previous_q = self.data.qpos[self.arm_qpos_ids].copy()
                self.data.ctrl[self.gripper_actuator_id] = gripper_ctrl
                mujoco.mj_forward(self.model, self.data)
            return
        for _ in range(self.config.control_substeps):
            self.data.ctrl[self.arm_actuator_ids] = np.clip(
                q_target,
                self.model.actuator_ctrlrange[self.arm_actuator_ids, 0],
                self.model.actuator_ctrlrange[self.arm_actuator_ids, 1],
            )
            self.data.ctrl[self.gripper_actuator_id] = gripper_ctrl
            mujoco.mj_step(self.model, self.data)

    def _advance_target_motion(self) -> None:
        if np.linalg.norm(self.target_motion_velocity) <= 0.0:
            self.target_velocity[:] = 0.0
            return
        previous = self.target_position_np()
        next_position = self._clip_target(previous + self.target_motion_velocity)
        if np.any(np.isclose(next_position[[0, 2]], self.workspace_low[[0, 2]], atol=1e-6)) or np.any(
            np.isclose(next_position[[0, 2]], self.workspace_high[[0, 2]], atol=1e-6)
        ):
            self.target_motion_velocity *= -1.0
            next_position = self._clip_target(previous + self.target_motion_velocity)
        self.set_target_position(next_position)
        self.target_velocity = next_position - previous

    def _sample_target_away_from_ee(self) -> np.ndarray:
        ee = self.data.site_xpos[self.ee_site_id].copy()
        for _ in range(200):
            target = self._sample_target()
            if np.linalg.norm(target - ee) > 0.16:
                return target
        return self._sample_target()

    def _sample_target(self) -> np.ndarray:
        return np.array(
            [
                self.rng.uniform(-0.34, 0.34),
                0.001,
                self.rng.uniform(0.22, 0.62),
            ],
            dtype=np.float64,
        )

    def _clip_target(self, position: np.ndarray) -> np.ndarray:
        clipped = np.asarray(position, dtype=np.float64).reshape(3).copy()
        clipped[1] = 0.001
        clipped = np.clip(clipped, self.workspace_low, self.workspace_high)
        return clipped

    def _random_xz_unit(self) -> np.ndarray:
        vec = np.array([self.rng.normal(), 0.0, self.rng.normal()], dtype=np.float64)
        return vec / max(np.linalg.norm(vec), 1e-6)

    def _set_arm_qpos(self, qpos: np.ndarray) -> None:
        self.data.qpos[self.arm_qpos_ids] = qpos

    def _load_model(self) -> mujoco.MjModel:
        if not self.config.pick_scene:
            return mujoco.MjModel.from_xml_path(str(self.xml_path))
        return mujoco.MjModel.from_xml_path(str(self._pick_scene_xml_path()))

    def _pick_scene_xml_path(self) -> Path:
        output_dir = Path("artifacts/mujoco_prototype/generated_models")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "manipulator_pick_scene.xml"
        if path.exists() and path.stat().st_size > 0:
            return path
        text = self.xml_path.read_text(encoding="utf-8")
        common_dir = (self.xml_path.parent / "common").resolve()
        text = text.replace('file="./common/', f'file="{common_dir}/')
        support_pos = " ".join(f"{value:.6f}" for value in self.config.support_position)
        support_size = " ".join(f"{value:.6f}" for value in self.config.support_size)
        support_xml = (
            "\n"
            "    <!-- Added by VLA-GNN-Recurrent for physical grasp/lift validation. -->\n"
            f'    <geom name="grasp_support" type="box" pos="{support_pos}" size="{support_size}" '
            'rgba="0.25 0.25 0.25 1" friction="1.0 0.01 0.001" '
            'solref="0.004 1.0" solimp="0.92 0.98 0.001"/>\n'
        )
        text = text.replace("  <worldbody>\n", "  <worldbody>\n" + support_xml, 1)
        tmp_path = output_dir / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
        return path

    def _configure_actuators(self) -> None:
        self.model.actuator_ctrlrange[self.arm_actuator_ids, 0] = -self.config.motor_ctrl_limit
        self.model.actuator_ctrlrange[self.arm_actuator_ids, 1] = self.config.motor_ctrl_limit
        if self.config.kinematic_joint_control:
            return
        self.model.jnt_limited[self.arm_joint_ids[0]] = 1
        self.model.jnt_range[self.arm_joint_ids[0]] = np.array(
            [-self.config.motor_ctrl_limit, self.config.motor_ctrl_limit],
            dtype=np.float64,
        )
        self.model.dof_damping[self.arm_dof_ids] = self.config.joint_damping
        self.model.dof_armature[self.arm_dof_ids] = self.config.joint_armature
        self.model.actuator_gear[self.arm_actuator_ids, 0] = 1.0
        self.model.actuator_gaintype[self.arm_actuator_ids] = int(mujoco.mjtGain.mjGAIN_FIXED)
        self.model.actuator_biastype[self.arm_actuator_ids] = int(mujoco.mjtBias.mjBIAS_AFFINE)
        self.model.actuator_gainprm[self.arm_actuator_ids, 0] = self.config.joint_kp
        self.model.actuator_biasprm[self.arm_actuator_ids, 0] = 0.0
        self.model.actuator_biasprm[self.arm_actuator_ids, 1] = -self.config.joint_kp
        self.model.actuator_biasprm[self.arm_actuator_ids, 2] = -self.config.joint_kd
        self.model.actuator_forcelimited[self.arm_actuator_ids] = 1
        self.model.actuator_forcerange[self.arm_actuator_ids, 0] = -self.config.actuator_force_limit
        self.model.actuator_forcerange[self.arm_actuator_ids, 1] = self.config.actuator_force_limit
        self.model.geom_friction[self._geom_id("ball"), 0] = 1.0
        for name in ("thumb1", "thumb2", "thumbtip1", "thumbtip2", "finger1", "finger2", "fingertip1", "fingertip2"):
            self.model.geom_friction[self._geom_id(name), 0] = 1.0

    @staticmethod
    def _gripper_ctrl(command: float) -> float:
        command = float(np.clip(command, 0.0, 1.0))
        return -1.0 + 2.0 * command

    def _clip_arm_qpos(self, qpos: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos, dtype=np.float64).copy()
        limited = self.model.jnt_limited[self.arm_joint_ids].astype(bool)
        for idx, is_limited in enumerate(limited):
            if is_limited:
                low, high = self.arm_joint_ranges[idx]
                q[idx] = np.clip(q[idx], low, high)
        return q

    def _ee_velocity_tensor(self) -> torch.Tensor:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        velocity = jacp @ self.data.qvel
        return torch.tensor(velocity.copy(), dtype=torch.float32)

    def _site_quat(self, site_id: int) -> np.ndarray:
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id].reshape(9))
        return quat

    def _site_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _joint_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _actuator_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _geom_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
