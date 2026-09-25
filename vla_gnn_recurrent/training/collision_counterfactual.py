"""Closed-loop, multi-step collision counterfactuals for the frozen spatial controller.

The dynamic benchmark scores success at its fixed horizon.  A transient earlier
success is recorded, but is not counted as a rescue unless final success holds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from vla_gnn_recurrent.training import contact_aware_performance as contact
from vla_gnn_recurrent.training import dynamic_visual_delay_feasibility as dynamic
from vla_gnn_recurrent.training import motion_controller_performance as motion
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import residual_recurrent_feasibility as residual
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device


OUTPUT = Path("artifacts/collision_multistep_counterfactual_v1")
INTERVENTIONS = ("no_inward", "inward_025", "inward_050", "yaw_first", "tangent_first")
MODES = ("single", "persistent")
STATE_FLAG = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass(frozen=True)
class ReplayConfig:
    pre_collision_steps: int = 10
    interventions: tuple[str, ...] = INTERVENTIONS
    modes: tuple[str, ...] = MODES
    yaw_threshold_rad: float = 0.20
    yaw_inward_scale: float = 0.0
    near_surface_m: float = 0.0646
    inward_scales: tuple[tuple[str, float], ...] = (("inward_025", 0.25), ("inward_050", 0.50))
    atol: float = 2e-5

    def validate(self) -> None:
        if self.pre_collision_steps < 1 or self.atol <= 0 or self.near_surface_m <= 0:
            raise ValueError("Pre-collision steps, tolerance, and near-surface zone must be positive")
        if not 0 <= self.yaw_inward_scale <= 1 or self.yaw_threshold_rad < 0:
            raise ValueError("Invalid yaw-first parameters")
        if any(not 0 <= scale <= 1 for _, scale in self.inward_scales):
            raise ValueError("Inward scales must lie in [0, 1]")
        if set(self.interventions) - set(INTERVENTIONS) - {name for name, _ in self.inward_scales}:
            raise ValueError("Unknown intervention")
        if set(self.modes) - set(MODES):
            raise ValueError("Unknown mode")


@dataclass
class Snapshot:
    timestep: int
    state: np.ndarray
    geom_size: np.ndarray
    geom_contype: np.ndarray
    geom_conaffinity: np.ndarray
    shape: sf.SurfaceShape
    step_count: int
    previous_action: np.ndarray


def capture(env: sf.SurfaceManipulatorEnv, shape: sf.SurfaceShape, timestep: int,
            previous_action: np.ndarray) -> Snapshot:
    state = np.empty(mujoco.mj_stateSize(env.model, STATE_FLAG), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, state, STATE_FLAG)
    return Snapshot(timestep, state, env.model.geom_size.copy(),
                    env.model.geom_contype.copy(), env.model.geom_conaffinity.copy(),
                    shape, env.step_count, previous_action.copy())


def restore(env: sf.SurfaceManipulatorEnv, snapshot: Snapshot) -> np.ndarray:
    env.model.geom_size[:] = snapshot.geom_size
    env.model.geom_contype[:] = snapshot.geom_contype
    env.model.geom_conaffinity[:] = snapshot.geom_conaffinity
    env._active_shape = snapshot.shape
    mujoco.mj_setState(env.model, env.data, snapshot.state, STATE_FLAG)
    env.step_count = snapshot.step_count
    mujoco.mj_forward(env.model, env.data)
    return snapshot.previous_action.copy()


def _geometry(env: sf.SurfaceManipulatorEnv, shape: sf.SurfaceShape,
              task: sf.SurfaceFeasibilityConfig, points: np.ndarray) -> dict:
    ee = env.robot_observation().ee_position.numpy().astype(np.float64)
    yaw = sf.tool_yaw(env)
    target = sf._surface_target(shape, ee, task.pregrasp_clearance)
    errors = sf._state_errors(env, shape, task, target)
    contacts, clearance = sf._surface_distance_metrics(env, shape)
    tips = [np.mean([env.data.geom_xpos[env._geom_id(n)] for n in names], axis=0)
            for names in (("thumbtip1", "thumbtip2"), ("fingertip1", "fingertip2"))]
    visible = contact.observed_geometry(points, task.pregrasp_clearance)
    tip_local = sf.localize_world(np.asarray(tips), ee, yaw)
    tip_surface_distance = min(float(np.linalg.norm(points-tip, axis=1).min())
                               for tip in tip_local)
    return {"ee_position": ee.tolist(), "ee_yaw": float(yaw),
            "left_tip": np.asarray(tips[0]).tolist(),
            "right_tip": np.asarray(tips[1]).tolist(),
            "target_position": target[0].tolist(), "target_yaw": float(target[2]),
            "target_opening": float(target[3]), "opening": float(sf.gripper_width(env)),
            "position_error": errors["position_error"],
            "yaw_error": errors["orientation_error"],
            "aperture_error": errors["gripper_width_error"],
            "clearance": float(clearance), "collision": bool(contacts),
            "success_now": sf._success_from_errors(errors, bool(contacts), task),
            "collision_robot_geoms": sorted({motion._geom_name(env, c["geom2"]) for c in contacts}),
            "collision_surface_geoms": sorted({motion._geom_name(env, c["geom1"]) for c in contacts}),
            "visible_tip_surface_distance": tip_surface_distance,
            "normal_local_xz": visible["normal_local_xz"].tolist(),
            "observed_yaw_proxy": visible["yaw_proxy_rad"]}


def decompose(action: np.ndarray, normal: np.ndarray) -> dict[str, Any]:
    translation = np.asarray(action[[0, 2]], dtype=np.float64)
    outward_signed = float(np.dot(translation, normal))
    tangent = translation-outward_signed*normal
    return {"inward_m": max(0.0, -outward_signed),
            "outward_m": max(0.0, outward_signed),
            "tangent_local_xz": tangent.tolist(),
            "tangent_norm_m": float(np.linalg.norm(tangent))}


def intervene(action: np.ndarray, points: np.ndarray, state: np.ndarray,
              intervention: str, config: ReplayConfig,
              clearance: float) -> tuple[np.ndarray, dict]:
    """Change inward translation only, using the existing observable PCA normal."""
    geometry = contact.observed_geometry(points, clearance)
    normal = np.asarray(geometry["normal_local_xz"], dtype=np.float64)
    before = decompose(action, normal)
    scales = dict(config.inward_scales)
    if intervention in ("no_inward", "tangent_first"):
        scale = 0.0
    elif intervention in scales:
        scale = scales[intervention]
    elif intervention == "yaw_first":
        tips = state[8:14].reshape(2, 3)
        tip_distance = min(float(np.linalg.norm(points-tip, axis=1).min()) for tip in tips)
        active = geometry["yaw_proxy_rad"] >= config.yaw_threshold_rad and tip_distance < config.near_surface_m
        scale = config.yaw_inward_scale if active else 1.0
    else:
        raise ValueError(intervention)
    result = action.astype(np.float64).copy()
    if before["inward_m"] > 0 and scale < 1:
        result[[0, 2]] += before["inward_m"]*(1-scale)*normal
    result = result.astype(np.float32)
    after = decompose(result, normal)
    tol = 2e-7
    if not np.allclose(result[[1, 3, 4]], action[[1, 3, 4]], atol=tol, rtol=0):
        raise AssertionError("Intervention changed non-translation actions")
    if not np.allclose(after["tangent_local_xz"], before["tangent_local_xz"], atol=tol, rtol=0):
        raise AssertionError("Intervention changed tangent translation")
    if abs(after["inward_m"]-before["inward_m"]*scale) > tol:
        raise AssertionError("Intervention did not apply the requested inward scale")
    if abs(after["outward_m"]-before["outward_m"]) > tol:
        raise AssertionError("Intervention changed outward translation")
    return result, {"active": bool(before["inward_m"] > 0 and scale < 1),
                    "scale": float(scale), "yaw_proxy_rad": geometry["yaw_proxy_rad"],
                    "before": before, "after": after,
                    "equivalent_to": "no_inward" if intervention == "tangent_first" else None}


@torch.no_grad()
def _policy_action(controller: motion.YawSpatialController,
                   state: np.ndarray, points: np.ndarray, previous: np.ndarray,
                   task: sf.SurfaceFeasibilityConfig,
                   device: torch.device) -> np.ndarray:
    state_t, points_t, topology = rf._graph_tensors(state, points, task, device)
    previous_t = torch.as_tensor(previous, dtype=torch.float32, device=device).reshape(1, 5)
    raw, _, _, _ = controller.step(state_t, points_t, previous_t, topology=topology)
    return rf.clipped_action(raw[0].detach().cpu().numpy(), task)


def _observation(env: sf.SurfaceManipulatorEnv, spec: dynamic.DynamicSpec,
                 task: sf.SurfaceFeasibilityConfig, timestep: int) -> tuple[sf.SurfaceShape, np.ndarray, np.ndarray]:
    shape = dynamic._shape_at(env, spec, timestep)
    state, points, _ = sf.observation_inputs(env, shape, task.point_count,
                                             sf._stable_seed(spec.initial.sample_identity))
    return shape, state, points


def _step_record(env: sf.SurfaceManipulatorEnv, spec: dynamic.DynamicSpec,
                 task: sf.SurfaceFeasibilityConfig, timestep: int,
                 shape: sf.SurfaceShape, state: np.ndarray, points: np.ndarray,
                 previous: np.ndarray, action: np.ndarray | None,
                 intervention_detail: dict | None = None,
                 seed: int | None = None) -> dict:
    row = {"episode_id": spec.identity,
           "seed": seed if seed is not None else _seed_from_spec(spec), "timestep": timestep,
           "dynamic": True, "direction": spec.direction,
           "speed_m_per_step": spec.speed_m_per_step,
           "object_center": list(shape.object_center),
           "surface_points_local": points.tolist(),
           "surface_points_world": sf.sample_surface_points_world(
               shape, task.point_count, sf._stable_seed(spec.initial.sample_identity)).tolist(),
           "state_14d": state.tolist(),
           "previous_executed_action": previous.tolist(),
           **_geometry(env, shape, task, points)}
    if action is not None:
        row["executed_action"] = action.tolist()
        row["translation_action"] = action[:3].tolist()
        row["yaw_action"] = float(action[3])
        row["gripper_action"] = float(action[4])
        row.update(decompose(action, np.asarray(row["normal_local_xz"])))
    else:
        row.update({"executed_action": None, "translation_action": None,
                    "yaw_action": None, "gripper_action": None,
                    "inward_m": None, "outward_m": None,
                    "tangent_local_xz": None, "tangent_norm_m": None})
    row["intervention"] = intervention_detail
    return row


def _seed_from_spec(spec: dynamic.DynamicSpec) -> int:
    # Shape IDs are generated as dynamic_heldout_{seed+110000}_{index}.
    return int(spec.initial.shape.shape_id.split("_")[2])-110_000


def _terminal(trace: list[dict], task: sf.SurfaceFeasibilityConfig) -> dict:
    final = trace[-1]
    collision = any(step["collision"] for step in trace)
    errors = {"position_error": final["position_error"],
              "orientation_error": final["yaw_error"],
              "gripper_width_error": final["aperture_error"]}
    first_success = next((step["timestep"] for step in trace
                          if step["success_now"] and not any(x["collision"] for x in trace
                                                                if x["timestep"] <= step["timestep"])), None)
    onset = next((step["timestep"] for step in trace if step["collision"]), None)
    return {"success": sf._success_from_errors(errors, collision, task),
            "collision": collision, "collision_step": onset,
            "final_position_error": final["position_error"],
            "final_yaw_error": final["yaw_error"],
            "final_aperture_error": final["aperture_error"],
            "min_clearance": float(min(step["clearance"] for step in trace)),
            "trajectory_length": final["timestep"],
            "steps_to_success": first_success}


def original_rollout(env: sf.SurfaceManipulatorEnv, spec: dynamic.DynamicSpec,
                     controller: motion.YawSpatialController,
                     task: sf.SurfaceFeasibilityConfig, device: torch.device,
                     seed: int | None = None,
                     ) -> tuple[list[dict], dict[int, Snapshot]]:
    sf._reset_surface_env(env, spec.initial)
    previous = np.zeros(5, dtype=np.float32)
    trace: list[dict] = []
    snapshots: dict[int, Snapshot] = {}
    for t in range(task.max_steps+1):
        shape, state, points = _observation(env, spec, task, t)
        snapshots[t] = capture(env, shape, t, previous)
        action = (_policy_action(controller, state, points, previous, task, device)
                  if t < task.max_steps else None)
        trace.append(_step_record(env, spec, task, t, shape, state, points,
                                  previous, action, seed=seed))
        if action is not None:
            sf.apply_local_action(env, action, task)
            previous = action
    return trace, snapshots


def replay_from(env: sf.SurfaceManipulatorEnv, snapshot: Snapshot,
                spec: dynamic.DynamicSpec, controller: motion.YawSpatialController,
                task: sf.SurfaceFeasibilityConfig, device: torch.device,
                intervention: str | None, mode: str, config: ReplayConfig,
                stop_on_collision: bool = True,
                seed: int | None = None) -> tuple[list[dict], dict]:
    previous = restore(env, snapshot)
    trace: list[dict] = []
    applied = 0
    for t in range(snapshot.timestep, task.max_steps+1):
        shape, state, points = _observation(env, spec, task, t)
        action = (_policy_action(controller, state, points, previous, task, device)
                  if t < task.max_steps else None)
        detail = None
        if action is not None and intervention is not None and (mode == "persistent" or t == snapshot.timestep):
            action, detail = intervene(action, points, state, intervention,
                                       config, task.pregrasp_clearance)
            applied += int(detail["active"])
        trace.append(_step_record(env, spec, task, t, shape, state, points,
                                  previous, action, detail, seed=seed))
        if trace[-1]["collision"] and stop_on_collision:
            break
        if action is not None:
            sf.apply_local_action(env, action, task)
            previous = action
    outcome = _terminal(trace, task)
    outcome["num_intervention_steps"] = applied
    return trace, outcome


def verify_reproduction(original: list[dict], reproduced: list[dict],
                        original_outcome: dict, replay_outcome: dict,
                        atol: float) -> dict:
    if len(original) != len(reproduced):
        raise AssertionError("No-intervention replay length differs from original")
    fields = ("ee_position", "ee_yaw", "left_tip", "right_tip",
              "opening", "position_error", "yaw_error", "aperture_error",
              "clearance", "executed_action", "surface_points_local")
    max_errors = {}
    for field in fields:
        diffs = []
        for a, b in zip(original, reproduced, strict=True):
            if a[field] is None:
                continue
            diffs.append(float(np.max(np.abs(np.asarray(a[field])-np.asarray(b[field])))))
        max_errors[field] = max(diffs, default=0.0)
    if any(value > atol for value in max_errors.values()):
        raise AssertionError(f"No-intervention replay state/action mismatch: {max_errors}")
    if (original_outcome["collision_step"] != replay_outcome["collision_step"] or
        original_outcome["collision"] != replay_outcome["collision"] or
        original_outcome["success"] != replay_outcome["success"]):
        raise AssertionError("No-intervention replay changed episode outcome")
    return {"max_absolute_differences": max_errors,
            "collision_step": replay_outcome["collision_step"],
            "success": replay_outcome["success"], "collision": replay_outcome["collision"],
            "tolerance": atol}


def verify_start_snapshot(env: sf.SurfaceManipulatorEnv, snapshot: Snapshot,
                          spec: dynamic.DynamicSpec,
                          controller: motion.YawSpatialController,
                          task: sf.SurfaceFeasibilityConfig, device: torch.device,
                          original: list[dict], atol: float,
                          seed: int | None = None) -> dict:
    """Check one regenerated transition from every candidate start snapshot."""
    previous = restore(env, snapshot)
    t = snapshot.timestep
    shape, state, points = _observation(env, spec, task, t)
    action = _policy_action(controller, state, points, previous, task, device)
    before = _step_record(env, spec, task, t, shape, state, points, previous, action,
                          seed=seed)
    sf.apply_local_action(env, action, task)
    shape_next, state_next, points_next = _observation(env, spec, task, t+1)
    after = _step_record(env, spec, task, t+1, shape_next, state_next,
                         points_next, action, None, seed=seed)
    compared = []
    for got, expected in ((before, original[t]), (after, original[t+1])):
        for key in ("ee_position", "ee_yaw", "left_tip", "right_tip",
                    "opening", "position_error", "yaw_error", "aperture_error",
                    "clearance", "surface_points_local", "previous_executed_action"):
            compared.append(float(np.max(np.abs(np.asarray(got[key])-
                                                np.asarray(expected[key])))))
    compared.append(float(np.max(np.abs(action-np.asarray(original[t]["executed_action"])))))
    error = max(compared)
    if error > atol or before["collision"] != original[t]["collision"] or after["collision"] != original[t+1]["collision"]:
        raise AssertionError(f"Start snapshot t={t} failed one-step reproduction: {error}")
    return {"start_step": t, "max_absolute_difference": error,
            "next_collision": after["collision"]}


def _verify_official_reference(seed: int, identity: str, outcome: dict,
                               atol: float) -> dict:
    path = motion.OUTPUT / f"seed{seed}/yaw_spatial_readout/dynamic_per_episode.csv"
    matches = [row for row in csv.DictReader(path.open()) if row["episode_id"] == identity]
    if not matches:
        return {"reference_path": str(path), "available": False,
                "reason": "Episode is not in the canonical held-out reference"}
    if len(matches) != 1:
        raise AssertionError(f"Duplicate official held-out reference for {identity}")
    row = matches[0]
    if (str(outcome["success"]).lower() != row["success"].lower() or
        str(outcome["collision"]).lower() != row["collision"].lower()):
        raise AssertionError(f"Official outcome mismatch for {identity}")
    differences = {key: abs(outcome[field]-float(row[reference])) for key, field, reference in (
        ("position", "final_position_error", "position_error"),
        ("yaw", "final_yaw_error", "yaw_error"),
        ("aperture", "final_aperture_error", "aperture_error"),
        ("clearance", "min_clearance", "minimum_clearance"))}
    if any(value > atol for value in differences.values()):
        raise AssertionError(f"Official metrics mismatch for {identity}: {differences}")
    return {"reference_path": str(path), "max_absolute_differences": differences,
            "success": outcome["success"], "collision": outcome["collision"]}


def _json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def _csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _compact(trace: list[dict], onset: int, before: int) -> list[dict]:
    return [{"step": row["timestep"]-onset, "timestep": row["timestep"],
             "position_error": row["position_error"], "yaw_error": row["yaw_error"],
             "clearance": row["clearance"], "inward_m": row["inward_m"],
             "tangent_norm_m": row["tangent_norm_m"],
             "yaw_action": row["yaw_action"], "collision": row["collision"]}
            for row in trace if max(0, onset-before) <= row["timestep"] <= onset]


def analyze_episode(seed: int, spec: dynamic.DynamicSpec, config: ReplayConfig,
                    output: Path, device: torch.device,
                    max_starts: int | None = None) -> dict:
    task = motion.task(seed, eval_device=device.type)
    controller = contact._final_controller(seed, device)
    env = sf.make_env(task, seed+864_211)
    try:
        original, snapshots = original_rollout(env, spec, controller, task, device,
                                                seed=seed)
        original_outcome = _terminal(original, task)
        official = _verify_official_reference(seed, spec.identity, original_outcome,
                                              config.atol)
        if not original_outcome["collision"]:
            return {"episode_id": spec.identity, "collision": False,
                    "original_outcome": original_outcome, "official_reproduction": official}
        onset = int(original_outcome["collision_step"])
        first_start = max(0, onset-config.pre_collision_steps)
        starts = list(range(first_start, onset))
        if max_starts is not None:
            starts = starts[-max_starts:]
        if not starts:
            raise AssertionError("Collision at reset has no pre-contact action to intervene on")
        # Native MuJoCo state and model geometry are restored, then every future
        # action and observation is regenerated. No recorded future state is used.
        reproduced, replay_outcome = replay_from(env, snapshots[first_start], spec,
            controller, task, device, None, "single", config, stop_on_collision=False,
            seed=seed)
        reproduction = verify_reproduction(original[first_start:], reproduced,
            _terminal(original[first_start:], task), replay_outcome, config.atol)
        snapshot_checks = [verify_start_snapshot(env, snapshots[start], spec,
            controller, task, device, original, config.atol, seed=seed) for start in starts]
        root = ensure_dir(output / f"episode_{seed}_{spec.initial.episode_id:04d}")
        metadata = {"episode_id": spec.identity, "seed": seed,
                    "episode_spec": asdict(spec), "controller": "frozen_yaw_spatial_plus_aperture",
                    "dynamic": True, "original_outcome": original_outcome,
                    "reproduction": reproduction, "official_reproduction": official,
                    "start_snapshot_checks": snapshot_checks,
                    "snapshot_method": "mjSTATE_INTEGRATION + mutable model geom arrays + previous action"}
        _json(root / "metadata.json", metadata)
        _json(root / "original_trace.json", original)
        _csv(root / "original_trace.csv", [{k: json.dumps(v) if isinstance(v, (list, dict)) else v
                                             for k, v in row.items()} for row in original])
        compact = _compact(original, onset, config.pre_collision_steps)
        _csv(root / "precontact_table.csv", compact)
        _json(root / "precontact_table.json", compact)
        results = []
        for intervention in config.interventions:
            for mode in config.modes:
                for start in starts:
                    trace, outcome = replay_from(env, snapshots[start], spec, controller,
                        task, device, intervention, mode, config, seed=seed)
                    # A changed action must be followed by a fresh simulator state,
                    # and the next policy call must consume that state and action.
                    if trace[0]["intervention"] and trace[0]["intervention"]["active"] and len(trace) > 1:
                        if not np.allclose(trace[1]["previous_executed_action"],
                                           trace[0]["executed_action"], atol=1e-7):
                            raise AssertionError("Closed-loop previous action was not updated")
                    result = {"episode_id": spec.identity, "seed": seed,
                              "original_outcome": "collision",
                              "original_collision_step": onset,
                              "intervention": intervention, "mode": mode,
                              "start_relative_step": start-onset,
                              "start_absolute_step": start,
                              **outcome,
                              "min_clearance": min(outcome["min_clearance"],
                                  min(row["clearance"] for row in original[:start])
                                  if start else math.inf),
                              "position_fail": outcome["final_position_error"] >= task.success_position_threshold,
                              "yaw_fail": outcome["final_yaw_error"] >= task.success_rotation_threshold,
                              "aperture_fail": outcome["final_aperture_error"] >= task.success_opening_threshold,
                              "collision_avoided_task_failed": not outcome["collision"] and not outcome["success"],
                              "outcome_class": ("success_rescued" if outcome["success"] else
                                  "still_collision" if outcome["collision"] else "collision_avoided_task_failed")}
                    result["new_failure_mode"] = ("+".join(name for name, failed in (
                        ("position", result["position_fail"]),
                        ("yaw", result["yaw_fail"]),
                        ("aperture", result["aperture_fail"])) if failed)
                        if result["collision_avoided_task_failed"] else None)
                    if not outcome["num_intervention_steps"] and (
                        outcome["collision_step"] != onset or not outcome["collision"]):
                        raise AssertionError("Zero-effect counterfactual changed the original collision")
                    changed_at = [row for row in trace if row["intervention"] and
                                  row["intervention"]["active"]]
                    if changed_at:
                        first_changed_t = changed_at[0]["timestep"]
                        next_row = next((row for row in trace if row["timestep"] ==
                                        first_changed_t+1), None)
                        result["first_post_intervention_ee_delta_m"] = (
                            float(np.linalg.norm(np.asarray(next_row["ee_position"])-
                                np.asarray(original[first_changed_t+1]["ee_position"])))
                            if next_row is not None else None)
                    else:
                        result["first_post_intervention_ee_delta_m"] = None
                    results.append(result)
                    label = f"start_{start-onset:+03d}"
                    folder = root / intervention / mode
                    _json(folder / f"{label}.json", {"result": result, "trace": trace})
        _csv(root / "results.csv", results)
        return {"episode_id": spec.identity, "collision": True,
                "original_outcome": original_outcome, "reproduction": reproduction,
                "results": results, "root": str(root)}
    finally:
        env.close()


def aggregate(episodes: list[dict], config: ReplayConfig) -> dict:
    collided = [episode for episode in episodes if episode["collision"]]
    summary = []
    for intervention in config.interventions:
        for mode in config.modes:
            rows = [row for ep in collided for row in ep["results"]
                    if row["intervention"] == intervention and row["mode"] == mode]
            per_episode = []
            for ep in collided:
                subset = [row for row in rows if row["episode_id"] == ep["episode_id"]]
                successful = [row for row in subset if row["success"]]
                steps = [row["start_absolute_step"] for row in successful]
                onset = ep["original_outcome"]["collision_step"]
                per_episode.append({"episode_id": ep["episode_id"],
                    "earliest_rescuable_step": min(steps) if steps else None,
                    "latest_rescuable_step": max(steps) if steps else None,
                    "earliest_rescuable_relative_step": min(steps)-onset if steps else None,
                    "latest_rescuable_relative_step": max(steps)-onset if steps else None,
                    "any_rescue": bool(successful),
                    "any_collision_avoided_task_failed": any(r["collision_avoided_task_failed"] for r in subset),
                    "all_still_collision": all(r["collision"] for r in subset)})
            summary.append({"intervention": intervention, "mode": mode,
                "original_collision_episodes": len(collided),
                "replay_trials": len(rows),
                "rescued_episodes": sum(r["any_rescue"] for r in per_episode),
                "collision_avoided_task_failed_episodes": sum(
                    r["any_collision_avoided_task_failed"] and not r["any_rescue"] for r in per_episode),
                "always_collision_episodes": sum(r["all_still_collision"] for r in per_episode),
                "success_trials": sum(r["success"] for r in rows),
                "collision_avoided_task_failed_trials": sum(r["collision_avoided_task_failed"] for r in rows),
                "still_collision_trials": sum(r["collision"] for r in rows),
                "episode_rescue_windows": per_episode,
                "equivalent_to": "no_inward" if intervention == "tangent_first" else None})
    return {"collision_episodes": len(collided), "interventions": summary}


def _specs_from_file(path: Path, seeds: tuple[int, ...]) -> list[tuple[int, dynamic.DynamicSpec]]:
    data = json.loads(path.read_text())
    entries = data.get("episodes", data) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("Episode file must be a JSON list or {'episodes': [...]} object")
    canonical = {(seed, spec.identity): spec for seed in seeds
                 for spec in dynamic.dynamic_specs(seed, motion.task(seed), 16,
                    dynamic.SELECTED_STEP_SPEEDS_M, "heldout")}
    result = []
    for entry in entries:
        if isinstance(entry, str):
            matches = [(seed, spec) for (seed, identity), spec in canonical.items() if identity == entry]
        elif isinstance(entry, dict) and "episode_id" in entry:
            matches = [(seed, spec) for (seed, identity), spec in canonical.items()
                       if identity == entry["episode_id"] and ("seed" not in entry or seed == int(entry["seed"]))]
        elif isinstance(entry, dict) and "spec" in entry:
            seed = int(entry["seed"])
            payload = entry["spec"]
            initial = payload["initial"]
            spec = dynamic.DynamicSpec(sf.SurfaceEpisodeSpec(
                initial["episode_id"], sf.SurfaceShape(**initial["shape"]),
                tuple(initial["arm_qpos"]), initial["gripper_command"],
                initial["condition"], initial["sample_identity"]),
                payload["direction"], payload["speed_m_per_step"])
            matches = [(seed, spec)]
        else:
            raise ValueError(f"Invalid episode entry: {entry}")
        if len(matches) != 1:
            raise ValueError(f"Episode entry did not resolve uniquely: {entry}")
        result.extend(matches)
    return result


def _episode_worker(payload: tuple) -> tuple[int, dict]:
    index, seed, spec, config, output, eval_device, max_starts = payload
    device = select_device(eval_device)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    return index, analyze_episode(seed, spec, config, output, device, max_starts)


def run(output: Path = OUTPUT, config: ReplayConfig = ReplayConfig(),
        seeds: tuple[int, ...] = motion.SEEDS, episodes_file: Path | None = None,
        eval_device: DevicePreference = "cpu", eval_workers: int = 1,
        max_episodes: int | None = None, max_starts: int | None = None) -> dict:
    config.validate()
    if eval_workers < 1:
        raise ValueError("--eval-workers must be positive")
    device = select_device(eval_device)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    expected = json.loads((motion.OUTPUT / "aggregate/checkpoint_provenance.json").read_text())
    hashes = {}
    for seed in seeds:
        paths = motion.paths(seed)
        actual = {"starting_motion_sha256": motion.sha(paths["motion"]),
                  "adapted_motion_sha256": motion.sha(paths["updated_motion"]),
                  "aperture_branch_sha256": motion.sha(paths["aperture"]),
                  "yaw_readout_sha256": motion.sha(motion.OUTPUT /
                    f"seed{seed}/yaw_spatial_readout/selected.pt"),
                  "base_ff_sha256": motion.sha(residual.BASE_CHECKPOINT)}
        if actual != expected[str(seed)]:
            raise AssertionError(f"Frozen checkpoint provenance mismatch for seed {seed}")
        hashes[str(seed)] = actual
    specs = (_specs_from_file(episodes_file, seeds) if episodes_file else
        [(seed, spec) for seed in seeds for spec in dynamic.dynamic_specs(seed,
            motion.task(seed), 16, dynamic.SELECTED_STEP_SPEEDS_M, "heldout")])
    if max_episodes is not None:
        specs = specs[:max_episodes]
    _json(output / "experiment_config.json", {"config": asdict(config),
        "seeds": seeds, "eval_device": str(device), "eval_workers": eval_workers,
        "episodes_file": str(episodes_file) if episodes_file else None,
        "episode_count": len(specs), "checkpoint_hashes": hashes,
        "success_semantics": "dynamic fixed-horizon final success; transient success recorded separately",
        "tangent_first": "equivalent to no_inward under the current action decomposition"})
    payloads = [(i, seed, spec, config, output, eval_device, max_starts)
                for i, (seed, spec) in enumerate(specs)]
    if eval_workers > 1:
        with ProcessPoolExecutor(max_workers=eval_workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            outcomes = list(pool.map(_episode_worker, payloads))
    else:
        outcomes = [_episode_worker(payload) for payload in payloads]
    episodes = [episode for _, episode in sorted(outcomes, key=lambda pair: pair[0])]
    for i, episode in enumerate(episodes):
        print(f"[{i+1}/{len(specs)}] {episode['episode_id']}: "
              f"{'collision' if episode['collision'] else 'no collision'}", flush=True)
    results = [row for ep in episodes if ep["collision"] for row in ep["results"]]
    _csv(output / "summary.csv", results)
    aggregate_result = aggregate(episodes, config)
    _json(output / "summary.json", aggregate_result)
    _csv(output / "aggregate.csv", [{k: v for k, v in row.items()
        if k != "episode_rescue_windows"} for row in aggregate_result["interventions"]])
    _json(output / "episode_rescue_windows.json", {ep["episode_id"]: [
        row for item in aggregate_result["interventions"]
        for row in item["episode_rescue_windows"] if row["episode_id"] == ep["episode_id"]]
        for ep in episodes if ep["collision"]})
    return aggregate_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--seeds", default="2811,2812,2813")
    parser.add_argument("--pre-collision-steps", type=int, default=10)
    parser.add_argument("--interventions", default=",".join(INTERVENTIONS))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--yaw-threshold", type=float, default=0.20)
    parser.add_argument("--yaw-inward-scale", type=float, default=0.0)
    parser.add_argument("--near-surface-m", type=float, default=0.0646)
    parser.add_argument("--inward-scales", default="inward_025:0.25,inward_050:0.50")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-starts", type=int)
    args = parser.parse_args()
    scales = tuple((name, float(value)) for name, value in
                   (part.split(":", 1) for part in args.inward_scales.split(",") if part))
    config = ReplayConfig(args.pre_collision_steps,
        tuple(x for x in args.interventions.split(",") if x),
        tuple(x for x in args.modes.split(",") if x),
        args.yaw_threshold, args.yaw_inward_scale, args.near_surface_m, scales)
    result = run(args.output_dir, config, tuple(int(x) for x in args.seeds.split(",")),
        args.episodes, args.eval_device, args.eval_workers, args.max_episodes, args.max_starts)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
