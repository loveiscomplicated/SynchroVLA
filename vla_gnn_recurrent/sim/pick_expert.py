from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv, MujocoReachConfig
from vla_gnn_recurrent.utils import ensure_dir, set_seed


@dataclass
class ScriptedPickConfig:
    episodes: int = 100
    seed: int = 123
    max_steps: int = 260
    object_x_range: tuple[float, float] = (-0.015, 0.0)
    grasp_offset: tuple[float, float, float] = (0.0, 0.0, -0.008)
    adaptive_grasp_offset: bool = False
    negative_x_grasp_offset: tuple[float, float, float] = (-0.015, 0.0, 0.0)
    positive_x_grasp_offset: tuple[float, float, float] = (0.03, 0.0, -0.015)
    approach_height: float = 0.10
    lift_height: float = 0.18
    lift_success_height: float = 0.10
    lift_hold_steps: int = 8
    close_steps: int = 45
    verify_steps: int = 8
    randomize_robot: bool = False
    grasp_association_distance: float = 0.13
    render: bool = True
    output_dir: str = "artifacts/mujoco_prototype/pick"


PHASES = ("PRE_GRASP", "ALIGN", "CLOSE", "VERIFY", "LIFT", "SUCCESS", "FAILURE")


def run_scripted_pick_evaluation(config: ScriptedPickConfig) -> dict[str, Any]:
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    env_config = MujocoReachConfig(
        max_steps=config.max_steps,
        control_substeps=40,
        max_delta_ee=0.045,
        pick_scene=True,
        kinematic_joint_control=False,
    )
    env = MujocoManipulatorEnv(env_config, seed=config.seed)
    output_dir = ensure_dir(config.output_dir)
    results = []
    for episode_idx in range(config.episodes):
        object_x = float(rng.uniform(*config.object_x_range))
        result = run_scripted_pick_episode(
            env=env,
            config=config,
            object_x=object_x,
            render=config.render and episode_idx == 0,
            render_dir=output_dir / "success_frames",
        )
        result["episode"] = episode_idx + 1
        results.append(result)

    failure_example = next((item for item in results if not item["lift_success"]), None)
    if failure_example is not None and config.render:
        run_scripted_pick_episode(
            env=env,
            config=config,
            object_x=float(failure_example["object_x"]),
            render=True,
            render_dir=output_dir / "failure_frames",
        )

    summary = summarize_pick_results(results)
    payload = {
        "config": asdict(config),
        "env_config": asdict(env_config),
        "summary": summary,
        "episodes": results,
    }
    path = output_dir / "scripted_pick_eval.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if results:
        plot_pick_diagnostics(results[0], output_dir / "scripted_pick_diagnostics.png")
    env.close()
    payload["output_path"] = str(path)
    return payload


def run_scripted_pick_episode(
    env: MujocoManipulatorEnv,
    config: ScriptedPickConfig,
    object_x: float = 0.0,
    render: bool = False,
    render_dir: Path | None = None,
) -> dict[str, Any]:
    env.reset_pick_scene(object_x=object_x, randomize_robot=config.randomize_robot)
    render_paths: list[str] = []
    if render:
        render_dir = ensure_dir(render_dir or Path(config.output_dir) / "frames")

    object_start = env.data.site_xpos[env.object_site_id].copy()
    phase = "PRE_GRASP"
    phase_order = [phase]
    records: list[dict[str, Any]] = []
    failure_reason: str | None = None
    hold_count = 0
    approach_success = False
    grasp_success = False
    lift_success = False
    grasp_step: int | None = None
    lift_step: int | None = None
    ik_failures = 0
    close_counter = 0
    verify_counter = 0
    lift_target: np.ndarray | None = None

    for step_idx in range(config.max_steps):
        object_position = env.data.site_xpos[env.object_site_id].copy()
        grasp_offset = _select_grasp_offset(config, object_position)
        grasp_target = object_position + grasp_offset
        approach_target = grasp_target + np.array([0.0, 0.0, config.approach_height], dtype=np.float64)
        if phase == "PRE_GRASP":
            target = approach_target
            gripper = 0.0
            if _ee_distance(env, target) < 0.035:
                approach_success = True
                phase = "ALIGN"
                phase_order.append(phase)
        elif phase == "ALIGN":
            target = grasp_target
            gripper = 0.0
            if _ee_distance(env, target) < 0.035:
                phase = "CLOSE"
                phase_order.append(phase)
        elif phase == "CLOSE":
            target = grasp_target
            gripper = 1.0
            close_counter += 1
            if close_counter >= config.close_steps:
                phase = "VERIFY"
                phase_order.append(phase)
        elif phase == "VERIFY":
            target = grasp_target
            gripper = 1.0
            verify_counter += 1
            grasp_obs = env.grasp_observation()
            if grasp_obs.left_contact and grasp_obs.right_contact:
                grasp_success = True
                grasp_step = step_idx
                lift_target = env.data.site_xpos[env.ee_site_id].copy() + np.array(
                    [0.0, 0.0, config.lift_height],
                    dtype=np.float64,
                )
                phase = "LIFT"
                phase_order.append(phase)
            elif verify_counter >= config.verify_steps:
                failure_reason = "insufficient_contact_or_alignment_error"
                phase = "FAILURE"
                phase_order.append(phase)
        elif phase == "LIFT":
            target = lift_target if lift_target is not None else grasp_target
            gripper = 1.0
        else:
            target = lift_target if lift_target is not None else grasp_target
            gripper = 1.0

        _, _, _, info = env.step_toward_ee_target(target, gripper=gripper)
        if not bool(info["ik_success"]):
            ik_failures += 1
        grasp_obs = env.grasp_observation()
        object_height_delta = grasp_obs.object_height - float(object_start[2])
        object_to_ee = float(np.linalg.norm(env.data.site_xpos[env.object_site_id] - env.data.site_xpos[env.ee_site_id]))
        associated_with_gripper = object_to_ee <= config.grasp_association_distance
        if phase == "LIFT":
            if object_height_delta >= config.lift_success_height and associated_with_gripper:
                hold_count += 1
            else:
                hold_count = 0
            if hold_count >= config.lift_hold_steps:
                lift_success = True
                lift_step = step_idx
                phase = "SUCCESS"
                phase_order.append(phase)
        if phase == "LIFT" and step_idx > (grasp_step or 0) + 80 and object_height_delta < 0.04:
            failure_reason = "grasp_slip_or_lift_instability"
            phase = "FAILURE"
            phase_order.append(phase)
        if phase == "FAILURE" and failure_reason is None:
            failure_reason = "unknown_failure"

        records.append(
            {
                "step": step_idx,
                "phase": phase,
                "ee_position": env.data.site_xpos[env.ee_site_id].copy().tolist(),
                "ee_target": target.tolist(),
                "joint_position": env.data.qpos[env.arm_qpos_ids].copy().tolist(),
                "joint_target": env.data.ctrl[env.arm_actuator_ids].copy().tolist(),
                "gripper_command": gripper,
                "object_position": env.data.site_xpos[env.object_site_id].copy().tolist(),
                "object_height_delta": object_height_delta,
                "left_contact": grasp_obs.left_contact,
                "right_contact": grasp_obs.right_contact,
                "object_grasped": grasp_obs.object_grasped,
                "object_to_ee": object_to_ee,
                "associated_with_gripper": associated_with_gripper,
                "contact_force": grasp_obs.contact_force,
                "num_object_contacts": grasp_obs.num_contacts,
                "ik_success": bool(info["ik_success"]),
            }
        )

        if render and render_dir is not None and step_idx % 4 == 0:
            frame_path = render_dir / f"frame_{step_idx:03d}.png"
            plt.imsave(frame_path, env.render_frame())
            render_paths.append(str(frame_path))
        if phase in {"SUCCESS", "FAILURE"}:
            break

    if failure_reason is None and not lift_success:
        if ik_failures > 0:
            failure_reason = "ik_failure"
        elif not approach_success:
            failure_reason = "approach_timeout"
        elif not grasp_success:
            failure_reason = "insufficient_contact_or_alignment_error"
        else:
            failure_reason = "lift_instability"
    drop = bool(grasp_success and not lift_success)
    return {
        "object_x": object_x,
        "approach_success": approach_success,
        "grasp_success": grasp_success,
        "lift_success": lift_success,
        "drop": drop,
        "failure_reason": None if lift_success else failure_reason,
        "phase_order": phase_order,
        "steps": len(records),
        "grasp_step": grasp_step,
        "lift_step": lift_step,
        "ik_failures": ik_failures,
        "records": records,
        "render_frames": render_paths,
    }


def summarize_pick_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = max(len(results), 1)
    failure_counts: dict[str, int] = {}
    for result in results:
        if not result["lift_success"]:
            reason = str(result.get("failure_reason") or "unknown_failure")
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    grasp_steps = [result["grasp_step"] for result in results if result.get("grasp_step") is not None]
    lift_steps = [result["lift_step"] for result in results if result.get("lift_step") is not None]
    return {
        "approach_success_rate": sum(1.0 for item in results if item["approach_success"]) / count,
        "grasp_success_rate": sum(1.0 for item in results if item["grasp_success"]) / count,
        "lift_success_rate": sum(1.0 for item in results if item["lift_success"]) / count,
        "drop_rate": sum(1.0 for item in results if item["drop"]) / count,
        "mean_time_to_grasp": _mean(grasp_steps),
        "mean_time_to_lift": _mean(lift_steps),
        "failure_counts": failure_counts,
    }


def plot_pick_diagnostics(result: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = result["records"]
    xs = np.arange(len(records))
    ee = np.asarray([row["ee_position"] for row in records], dtype=float)
    ee_target = np.asarray([row["ee_target"] for row in records], dtype=float)
    joints = np.asarray([row["joint_position"] for row in records], dtype=float)
    joint_targets = np.asarray([row["joint_target"] for row in records], dtype=float)
    gripper = np.asarray([row["gripper_command"] for row in records], dtype=float)
    height = np.asarray([row["object_height_delta"] for row in records], dtype=float)
    contact = np.asarray([row["left_contact"] and row["right_contact"] for row in records], dtype=float)
    force = np.asarray([row["contact_force"] for row in records], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), constrained_layout=True)
    axes[0].plot(xs, ee[:, 0], label="ee x")
    axes[0].plot(xs, ee[:, 2], label="ee z")
    axes[0].plot(xs, ee_target[:, 0], "--", label="target x")
    axes[0].plot(xs, ee_target[:, 2], "--", label="target z")
    axes[0].set_ylabel("EE")
    axes[0].legend(loc="best")
    for idx in range(min(4, joints.shape[1])):
        axes[1].plot(xs, joints[:, idx], label=f"q{idx}")
        axes[1].plot(xs, joint_targets[:, idx], "--", alpha=0.5)
    axes[1].set_ylabel("joints")
    axes[1].legend(loc="best", ncol=2)
    axes[2].plot(xs, gripper, label="gripper command")
    axes[2].plot(xs, height, label="object height delta")
    axes[2].set_ylabel("grip/height")
    axes[2].legend(loc="best")
    axes[3].plot(xs, contact, label="both finger contact")
    axes[3].plot(xs, force, label="contact force")
    axes[3].set_xlabel("step")
    axes[3].set_ylabel("contact")
    axes[3].legend(loc="best")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _ee_distance(env: MujocoManipulatorEnv, target: np.ndarray) -> float:
    return float(np.linalg.norm(env.data.site_xpos[env.ee_site_id] - target))


def _select_grasp_offset(config: ScriptedPickConfig, object_position: np.ndarray) -> np.ndarray:
    if not config.adaptive_grasp_offset:
        return np.asarray(config.grasp_offset, dtype=np.float64)
    if float(object_position[0]) > 0.004:
        return np.asarray(config.positive_x_grasp_offset, dtype=np.float64)
    return np.asarray(config.negative_x_grasp_offset, dtype=np.float64)


def _mean(values: list[int | float | None]) -> float:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return 0.0
    return sum(numeric) / len(numeric)
