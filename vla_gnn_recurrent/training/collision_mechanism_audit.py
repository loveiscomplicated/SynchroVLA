"""Audit the production inward guard and unresolved near-target contacts.

Only diagnostic replay is performed. The learned controller and production
ContactController are loaded unchanged from their validated checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from vla_gnn_recurrent.training import collision_counterfactual as cf
from vla_gnn_recurrent.training import contact_aware_performance as contact
from vla_gnn_recurrent.training import dynamic_visual_delay_feasibility as dynamic
from vla_gnn_recurrent.training import motion_controller_performance as motion
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training.generalized_geometric_graph import wrap_angle
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device


OUTPUT = Path("artifacts/collision_mechanism_audit_v1")
LARGE = {(2812, 1), (2812, 10), (2813, 1), (2813, 8)}
NEAR = {(2811, 4), (2812, 6), (2813, 14)}
ONE_STEP_VARIANTS = ("original", "no_inward", "no_tangent", "no_translation",
                     "no_yaw", "half_tangent", "half_translation")


def _read_counterfactual(source: Path) -> tuple[dict, dict, list[dict], dict]:
    config = json.loads((source / "experiment_config.json").read_text())
    aggregate = json.loads((source / "summary.json").read_text())
    rows = list(csv.DictReader((source / "summary.csv").open()))
    windows = json.loads((source / "episode_rescue_windows.json").read_text())
    if aggregate["collision_episodes"] != 8:
        raise AssertionError("Expected eight authoritative collision episodes")
    if len(rows) != 600:
        raise AssertionError("Authoritative counterfactual sweep is incomplete")
    return config, aggregate, rows, windows


def _spec(seed: int, episode_id: int) -> dynamic.DynamicSpec:
    task = motion.task(seed)
    matches = [s for s in dynamic.dynamic_specs(seed, task, 16,
        dynamic.SELECTED_STEP_SPEEDS_M, "heldout")
        if s.initial.episode_id == episode_id]
    if len(matches) != 1:
        raise AssertionError((seed, episode_id))
    return matches[0]


def _load_episode(source: Path, seed: int, episode_id: int,
                  device: torch.device) -> tuple[Any, list[dict], dict, dict, dict]:
    root = source / f"episode_{seed}_{episode_id:04d}"
    metadata = json.loads((root / "metadata.json").read_text())
    stored = json.loads((root / "original_trace.json").read_text())
    if not metadata["reproduction"]["collision"] or max(
            metadata["reproduction"]["max_absolute_differences"].values()) > 2e-5:
        raise AssertionError(f"Unverified historical replay: {root}")
    spec = _spec(seed, episode_id)
    if spec.identity != metadata["episode_id"]:
        raise AssertionError("Episode specification changed")
    task = motion.task(seed, eval_device=device.type)
    controller = contact._final_controller(seed, device)
    env = sf.make_env(task, seed+864_211)
    try:
        regenerated, snapshots = cf.original_rollout(env, spec, controller,
                                                       task, device, seed=seed)
        parity = cf.verify_reproduction(stored, regenerated,
            cf._terminal(stored, task), cf._terminal(regenerated, task), 2e-5)
    finally:
        env.close()
    return spec, stored, snapshots, metadata, parity


def _verify_hashes(source_config: dict) -> None:
    expected = source_config["checkpoint_hashes"]
    current = json.loads((motion.OUTPUT / "aggregate/checkpoint_provenance.json").read_text())
    if expected != current:
        raise AssertionError("Frozen checkpoint manifest changed")
    for seed in motion.SEEDS:
        paths = motion.paths(seed)
        hashes = {"starting_motion_sha256": motion.sha(paths["motion"]),
                  "adapted_motion_sha256": motion.sha(paths["updated_motion"]),
                  "aperture_branch_sha256": motion.sha(paths["aperture"]),
                  "yaw_readout_sha256": motion.sha(motion.OUTPUT /
                      f"seed{seed}/yaw_spatial_readout/selected.pt"),
                  "base_ff_sha256": motion.sha(cf.residual.BASE_CHECKPOINT)}
        if hashes != expected[str(seed)]:
            raise AssertionError(f"Checkpoint bytes changed for seed {seed}")


def guard_definition(output: Path, zone: float) -> dict:
    task = motion.task(2811)
    source = Path(inspect.getsourcefile(contact.ContactController) or "")
    lines, line_number = inspect.getsourcelines(contact.ContactController.step)
    result = {"implementation_file": str(source),
        "class": "ContactController", "method": "step", "mode": "near_mixed",
        "method_first_line": line_number,
        "method_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
        "distance_proxy": "min Euclidean distance from either state[8:14] EE-local fingertip to the observed EE-local 32-point surface cloud",
        "position_proxy": "observed_geometry(points, pregrasp_clearance).position_proxy_m; norm of PCA-derived local target",
        "yaw_proxy": "observed_geometry(...).yaw_proxy_rad; absolute atan2 of EE-local PCA surface normal",
        "thresholds": {"distance_strict_less_m": zone,
                       "position_greater_equal_m": task.success_position_threshold,
                       "yaw_greater_equal_rad": task.success_rotation_threshold,
                       "inward_strict_greater_m": 0.0},
        "condition": "distance < zone AND position_proxy >= position_threshold AND yaw_proxy >= yaw_threshold AND inward > 0",
        "inward_definition": "-dot(EE-local action[[0,2]], observed_geometry normal_local_xz)",
        "modification": "action[[0,2]] += inward * normal_local_xz; full inward cancellation",
        "unchanged_at_guard_output_before_clipping": ["tangent translation", "outward translation", "yaw action", "gripper action"],
        "execution_note": "The evaluation path calls recurrent_feasibility.clipped_action after the guard; clipping can change executed tangent magnitude relative to the clipped baseline",
        "zone_provenance": str(contact.OUTPUT /
            "approach_safety/training_zone/zone.json")}
    cf._json(output / "guard_definition.json", result)
    return result


def _guard_observation(state: np.ndarray, points: np.ndarray,
                       task: sf.SurfaceFeasibilityConfig, zone: float) -> dict:
    geometry = contact.observed_geometry(points, task.pregrasp_clearance)
    tips = state[8:14].reshape(2, 3)
    distance = min(float(np.linalg.norm(points-tip, axis=1).min()) for tip in tips)
    predicates = {"distance_condition": distance < zone,
                  "position_condition": geometry["position_proxy_m"] >= task.success_position_threshold,
                  "yaw_condition": geometry["yaw_proxy_rad"] >= task.success_rotation_threshold}
    return {"guard_position_proxy": geometry["position_proxy_m"],
            "guard_yaw_proxy": geometry["yaw_proxy_rad"],
            "guard_distance_proxy": distance,
            "normal_local_xz": geometry["normal_local_xz"].tolist(),
            **predicates, "guard_condition": all(predicates.values())}


@torch.no_grad()
def production_guard_step(guard: contact.ContactController,
                          state: np.ndarray, points: np.ndarray,
                          previous: np.ndarray,
                          task: sf.SurfaceFeasibilityConfig,
                          device: torch.device) -> tuple[np.ndarray, dict]:
    state_t, points_t, topology = rf._graph_tensors(state, points, task, device)
    previous_t = torch.as_tensor(previous, dtype=torch.float32,
                                 device=device).reshape(1, 5)
    raw, _, _, _ = guard.step(state_t, points_t, previous_t, topology=topology)
    action = rf.clipped_action(raw[0].detach().cpu().numpy(), task)
    return action, {"diagnostic": dict(guard.last_diagnostic[0]),
                    "requested_action": list(guard.last_requested_action[0]),
                    "guarded_raw_action": raw[0].detach().cpu().tolist()}


class RecordingGuard:
    """Read-only recorder around the production guard implementation."""

    def __init__(self, base: motion.YawSpatialController, zone: float) -> None:
        self.production = contact.ContactController(base, "near_mixed", zone)
        self.history: list[dict] = []

    def step(self, *args: Any, **kwargs: Any) -> Any:
        result = self.production.step(*args, **kwargs)
        self.history.append({"diagnostic": dict(self.production.last_diagnostic[0]),
                             "requested_action": list(self.production.last_requested_action[0])})
        return result


def _rescue_lookup(source_rows: list[dict], identity: str) -> dict[int, dict]:
    selected = [row for row in source_rows if row["episode_id"] == identity and
                row["intervention"] == "no_inward" and row["mode"] == "persistent"]
    return {int(row["start_absolute_step"]): row for row in selected}


def _validate_window_lookup(identity: str, rescue: dict[int, dict],
                            windows: dict, source_config: dict) -> None:
    # Historical window rows inherit the intervention/mode ordering from the
    # config; each list item has no own intervention label.
    order = [(name, mode) for name in source_config["config"]["interventions"]
            for mode in source_config["config"]["modes"]]
    index = order.index(("no_inward", "persistent"))
    window = windows[identity][index]
    successful = sorted(step for step, row in rescue.items()
                        if row["success"].lower() == "true")
    if (window["earliest_rescuable_step"] != (min(successful) if successful else None) or
        window["latest_rescuable_step"] != (max(successful) if successful else None)):
        raise AssertionError(f"Rescue window mismatch for {identity}")


def _csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _label_contact_part(name: str) -> str:
    if name.startswith("thumb"):
        return "left_finger_or_tip"
    if name.startswith("finger"):
        return "right_finger_or_tip"
    if name.startswith("palm") or name.startswith("hand"):
        return "palm_or_hand"
    return "other_robot_part"


def _body_name(env: sf.SurfaceManipulatorEnv, geom_id: int) -> str:
    body_id = int(env.model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or str(body_id)


def _tip_distance(env: sf.SurfaceManipulatorEnv, shape: sf.SurfaceShape,
                  side: str) -> float:
    active = env.surface_box_ids if shape.cross_section == "flat" else env.surface_capsule_ids
    tip_names = (("thumbtip1", "thumbtip2") if side == "left" else
                 ("fingertip1", "fingertip2"))
    segment = np.zeros(6, dtype=np.float64)
    return min(float(mujoco.mj_geomDistance(env.model, env.data,
        int(surface_id), int(env._geom_id(tip_name)), 0.20, segment))
        for surface_id in active for tip_name in tip_names)


def extract_contacts(env: sf.SurfaceManipulatorEnv,
                     shape: sf.SurfaceShape) -> dict:
    """Separate official distance-query triggers from native MuJoCo contacts."""
    official, clearance = sf._surface_distance_metrics(env, shape)
    official_pairs = {frozenset((int(c["geom1"]), int(c["geom2"]))) for c in official}
    native = []
    for index in range(env.data.ncon):
        con = env.data.contact[index]
        geom1, geom2 = int(con.geom1), int(con.geom2)
        pair = frozenset((geom1, geom2))
        if pair not in official_pairs:
            continue
        robot_id = geom2 if geom2 in set(env.robot_collision_geom_ids.tolist()) else geom1
        surface_id = geom1 if robot_id == geom2 else geom2
        robot_name = motion._geom_name(env, robot_id)
        native.append({"native_contact_index": index, "geom1_id": geom1,
            "geom2_id": geom2, "geom1_name": motion._geom_name(env, geom1),
            "geom2_name": motion._geom_name(env, geom2),
            "robot_geom": robot_name, "robot_body": _body_name(env, robot_id),
            "surface_geom": motion._geom_name(env, surface_id),
            "surface_body": _body_name(env, surface_id),
            "robot_part": _label_contact_part(robot_name),
            "position_world": np.asarray(con.pos).tolist(),
            "normal_world_geom1_to_geom2": np.asarray(con.frame[:3]).tolist(),
            "native_contact_distance_m": float(con.dist),
            "official_trigger": True})
    official_rows = []
    for pair in official:
        geom1, geom2 = int(pair["geom1"]), int(pair["geom2"])
        segment = np.zeros(6, dtype=np.float64)
        distance = float(mujoco.mj_geomDistance(env.model, env.data, geom1,
                                                 geom2, .20, segment))
        robot_name = motion._geom_name(env, geom2)
        official_rows.append({"surface_geom": motion._geom_name(env, geom1),
            "surface_body": _body_name(env, geom1),
            "robot_geom": robot_name, "robot_body": _body_name(env, geom2),
            "robot_part": _label_contact_part(robot_name),
            "distance_m": distance, "nearest_points_world": segment.tolist(),
            "has_native_contact": frozenset((geom1, geom2)) in
                {frozenset((row["geom1_id"], row["geom2_id"])) for row in native}})
    return {"official_collision": bool(official), "official_clearance_m": float(clearance),
            "official_trigger_pairs": official_rows, "native_matching_contacts": native,
            "native_contacts_total": int(env.data.ncon),
            "contact_force": None,
            "force_limit": "mj_forward after kinematic action does not provide a validated current-step solved contact force"}


def _guard_trace_rows(trace: list[dict], history: list[dict],
                      task: sf.SurfaceFeasibilityConfig, zone: float) -> list[dict]:
    rows = []
    for index, step in enumerate(trace):
        observed = _guard_observation(np.asarray(step["state_14d"], dtype=np.float32),
            np.asarray(step["surface_points_local"], dtype=np.float32), task, zone)
        diagnostic = history[index]["diagnostic"] if index < len(history) else None
        rows.append({"timestep": step["timestep"], **observed,
            "production_approach_blocked": (diagnostic["approach_blocked"]
                if diagnostic is not None else None),
            "inward_m": step["inward_m"], "executed_action": step["executed_action"]})
    return rows


def _first_divergence(guard_trace: list[dict], no_inward_trace: list[dict],
                      guard_rows: list[dict], task: sf.SurfaceFeasibilityConfig,
                      atol: float = 2e-5) -> dict:
    blocked_before = False
    for index, (guard_step, inward_step) in enumerate(
            zip(guard_trace, no_inward_trace, strict=False)):
        if guard_step["collision"] or inward_step["collision"]:
            break
        action_a, action_b = guard_step["executed_action"], inward_step["executed_action"]
        if action_a is None or action_b is None:
            break
        state_delta = float(np.max(np.abs(np.asarray(guard_step["state_14d"])-
                                          np.asarray(inward_step["state_14d"]))))
        action_delta = float(np.max(np.abs(np.asarray(action_a)-np.asarray(action_b))))
        if state_delta > atol and action_delta <= atol:
            raise AssertionError("Counterfactual states diverged before action divergence")
        if action_delta > atol:
            observed = guard_rows[index]
            missed = [name for name in ("distance", "position", "yaw")
                      if not observed[f"{name}_condition"]]
            if observed["guard_condition"]:
                label = "ACTION_MISMATCH"
            elif blocked_before:
                label = "EARLY_RELEASE"
            else:
                label = "LATE_ACTIVATION"
            return {"first_action_divergence_step": guard_step["timestep"],
                "first_action_delta_max": action_delta,
                "first_state_delta_max": state_delta,
                "label": label, "false_predicates": missed,
                "guard_action": action_a, "no_inward_action": action_b,
                "guard_observation": observed,
                "production_blocked_before_divergence": blocked_before}
        blocked_before |= bool(guard_rows[index]["production_approach_blocked"])
    guard_end, inward_end = guard_trace[-1], no_inward_trace[-1]
    if guard_end["timestep"] != inward_end["timestep"] or (
        bool(guard_end["collision"]) != bool(inward_end["collision"])):
        raise AssertionError("Same-action trajectories had different terminal events")
    return {"first_action_divergence_step": None,
            "first_action_delta_max": 0.0, "first_state_delta_max": 0.0,
            "label": "GUARD_MATCHES_NO_INWARD", "false_predicates": [],
            "guard_action": None, "no_inward_action": None,
            "guard_observation": None,
            "production_blocked_before_divergence": blocked_before}


def audit_large_episode(source: Path, output: Path, seed: int, episode_id: int,
                        device: torch.device, zone: float,
                        source_rows: list[dict], windows: dict,
                        source_config: dict) -> tuple[list[dict], dict, list[dict], list[dict]]:
    spec, original, snapshots, metadata, parity = _load_episode(
        source, seed, episode_id, device)
    identity = spec.identity
    task = motion.task(seed, eval_device=device.type)
    onset = int(metadata["original_outcome"]["collision_step"])
    starts = list(range(max(0, onset-int(source_config["config"]["pre_collision_steps"])),
                        onset))
    rescue = _rescue_lookup(source_rows, identity)
    _validate_window_lookup(identity, rescue, windows, source_config)
    if sorted(rescue) != starts:
        raise AssertionError("Stored rescue starts differ from the original window")
    controller = contact._final_controller(seed, device)
    guard = contact.ContactController(controller, "near_mixed", zone)
    per_step = []
    for t in starts:
        step = original[t]
        state = np.asarray(step["state_14d"], dtype=np.float32)
        points = np.asarray(step["surface_points_local"], dtype=np.float32)
        previous = np.asarray(step["previous_executed_action"], dtype=np.float32)
        guarded, production = production_guard_step(guard, state, points, previous,
                                                     task, device)
        requested = np.asarray(production["requested_action"], dtype=np.float32)
        baseline = np.asarray(step["executed_action"], dtype=np.float32)
        if not np.allclose(rf.clipped_action(requested, task), baseline,
                           atol=2e-5, rtol=0):
            raise AssertionError("Production guard base action differs from frozen baseline")
        observed = _guard_observation(state, points, task, zone)
        normal = np.asarray(observed["normal_local_xz"])
        raw_decomp = cf.decompose(baseline, normal)
        guard_decomp = cf.decompose(guarded, normal)
        requested_decomp = cf.decompose(requested, normal)
        guarded_raw = np.asarray(production["guarded_raw_action"], dtype=np.float32)
        guarded_raw_decomp = cf.decompose(guarded_raw, normal)
        expected_blocked = observed["guard_condition"] and requested_decomp["inward_m"] > 1e-8
        if bool(production["diagnostic"]["approach_blocked"]) != expected_blocked:
            raise AssertionError("Audited predicate differs from ContactController.step")
        if not expected_blocked and not np.allclose(guarded, baseline, atol=2e-5):
            raise AssertionError("Inactive production guard changed the action")
        if expected_blocked:
            if guarded_raw_decomp["inward_m"] > 2e-5 or not np.allclose(
                guarded_raw[[1, 3, 4]], requested[[1, 3, 4]], atol=2e-5):
                raise AssertionError("Production guard changed action dimensions incorrectly")
            if not np.allclose(requested_decomp["tangent_local_xz"],
                               guarded_raw_decomp["tangent_local_xz"], atol=2e-5):
                raise AssertionError("Production guard changed raw tangent motion")
        stored = rescue[t]
        per_step.append({"episode_id": identity, "seed": seed,
            "absolute_step": t, "relative_to_collision": t-onset,
            "position_error_true": step["position_error"],
            "yaw_error_true": step["yaw_error"], "clearance": step["clearance"],
            **observed,
            "guard_action_modified": expected_blocked,
            "raw_inward_component": raw_decomp["inward_m"],
            "requested_raw_inward_component": requested_decomp["inward_m"],
            "guarded_inward_component": guard_decomp["inward_m"],
            "tangent_component": raw_decomp["tangent_norm_m"],
            "tangent_local_xz": raw_decomp["tangent_local_xz"],
            "executed_tangent_change_norm_m": float(np.linalg.norm(
                np.asarray(guard_decomp["tangent_local_xz"])-
                np.asarray(raw_decomp["tangent_local_xz"]))),
            "requested_raw_translation_norm_m": float(np.linalg.norm(requested[:3])),
            "requested_raw_translation_exceeds_limit": bool(
                np.linalg.norm(requested[:3]) > task.max_delta_ee+1e-8),
            "yaw_action": float(baseline[3]), "gripper_action": float(baseline[4]),
            "counterfactual_no_inward_success_if_started_here":
                stored["success"].lower() == "true",
            "counterfactual_no_inward_collision_if_started_here":
                stored["collision"].lower() == "true"})
    successful = [r for r in per_step if r["counterfactual_no_inward_success_if_started_here"]]
    active_steps = [r for r in per_step if r["guard_condition"]]
    active_in_window = [r for r in successful if r["guard_condition"]]
    false_rows = [r for r in successful if not r["guard_condition"]]
    first_rescuable = min((r["absolute_step"] for r in successful), default=None)
    first_guard = min((r["absolute_step"] for r in active_steps), default=None)
    false_counts = {name: sum(not r[f"{name}_condition"] for r in false_rows)
                    for name in ("distance", "position", "yaw")}
    false_counts["multiple"] = sum(sum(not r[f"{name}_condition"] for name in
        ("distance", "position", "yaw")) > 1 for r in false_rows)
    if false_counts["multiple"]:
        dominant = "multiple"
    else:
        dominant = max(("distance", "position", "yaw"),
                       key=lambda name: false_counts[name]) if false_rows else None
    summary = {"episode_id": identity, "seed": seed,
        "original_collision_step": onset,
        "rescue_earliest": first_rescuable-onset if first_rescuable is not None else None,
        "rescue_latest": max(r["absolute_step"] for r in successful)-onset if successful else None,
        "rescue_window_size": len(successful),
        "first_rescuable_step": first_rescuable,
        "first_guard_activation_step": first_guard,
        "first_guard_activation_relative": first_guard-onset if first_guard is not None else None,
        "guard_active_steps_in_rescue_window": len(active_in_window),
        "guard_coverage_ratio": len(active_in_window)/len(successful) if successful else None,
        "rescuable_guard_false_steps": len(false_rows),
        "false_predicate_counts": false_counts,
        "dominant_false_predicate": dominant,
        "activation_delay": first_guard-first_rescuable if first_guard is not None and
                                                     first_rescuable is not None else None,
        "original_reproduction_max_difference": max(parity["max_absolute_differences"].values())}
    root = ensure_dir(output / "guard_audit" / f"{seed}_{episode_id:04d}")
    env = sf.make_env(task, seed+864_211)
    comparisons = []
    labels: list[dict] = []
    try:
        full_guard = RecordingGuard(controller, zone)
        full_trace, full_outcome = cf.replay_from(env, snapshots[0], spec, full_guard,
            task, device, None, "persistent", cf.ReplayConfig(),
            stop_on_collision=True, seed=seed)
        summary["full_episode_guard_success"] = full_outcome["success"]
        summary["full_episode_guard_collision"] = full_outcome["collision"]
        summary["full_episode_guard_collision_step"] = full_outcome["collision_step"]
        historical_path = (contact.OUTPUT /
            f"approach_safety/near_mixed/heldout/seed{seed}/dynamic_per_episode.csv")
        historical = [row for row in csv.DictReader(historical_path.open())
                      if row["episode_id"] == identity]
        if len(historical) != 1:
            raise AssertionError("Historical production guard result is missing")
        historical = historical[0]
        if (full_outcome["success"] != (historical["success"].lower() == "true") or
            full_outcome["collision"] != (historical["collision"].lower() == "true")):
            raise AssertionError("Full-episode guard replay differs from historical evaluation")
        if not full_outcome["collision"]:
            for field, reference in (("final_position_error", "position_error"),
                                     ("final_yaw_error", "yaw_error"),
                                     ("final_aperture_error", "aperture_error")):
                if abs(full_outcome[field]-float(historical[reference])) > 2e-5:
                    raise AssertionError("Full-episode guard final metrics differ")
        summary["historical_guard_reproduced"] = True
        summary["historical_guard_reference"] = str(historical_path)
        cf._json(root / "full_episode_guard.json", {"outcome": full_outcome,
            "trace": full_trace,
            "guard_diagnostic": _guard_trace_rows(full_trace, full_guard.history, task, zone)})
        for start in starts:
            baseline, baseline_outcome = cf.replay_from(env, snapshots[start], spec,
                controller, task, device, None, "persistent", cf.ReplayConfig(),
                stop_on_collision=False, seed=seed)
            cf.verify_reproduction(original[start:], baseline,
                cf._terminal(original[start:], task), baseline_outcome, 2e-5)
            no_inward, inward_outcome = cf.replay_from(env, snapshots[start], spec,
                controller, task, device, "no_inward", "persistent",
                cf.ReplayConfig(), stop_on_collision=True, seed=seed)
            relative = start-onset
            stored = json.loads((source / f"episode_{seed}_{episode_id:04d}" /
                "no_inward/persistent" / f"start_{relative:+03d}.json").read_text())
            cf.verify_reproduction(stored["trace"], no_inward,
                cf._terminal(stored["trace"], task), inward_outcome, 2e-5)
            guarded = RecordingGuard(controller, zone)
            guard_trace, guard_outcome = cf.replay_from(env, snapshots[start], spec,
                guarded, task, device, None, "persistent", cf.ReplayConfig(),
                stop_on_collision=True, seed=seed)
            guard_rows = _guard_trace_rows(guard_trace, guarded.history, task, zone)
            divergence = _first_divergence(guard_trace, no_inward, guard_rows, task)
            if (divergence["label"] == "GUARD_MATCHES_NO_INWARD" and
                (guard_outcome["success"] != inward_outcome["success"] or
                 guard_outcome["collision"] != inward_outcome["collision"])):
                raise AssertionError("Same-action guard/no-inward replay outcomes differ")
            comparison = {"episode_id": identity, "seed": seed,
                "start_absolute_step": start, "start_relative_step": relative,
                "baseline_success": baseline_outcome["success"],
                "baseline_collision": baseline_outcome["collision"],
                "baseline_collision_step": baseline_outcome["collision_step"],
                "no_inward_success": inward_outcome["success"],
                "no_inward_collision": inward_outcome["collision"],
                "no_inward_collision_step": inward_outcome["collision_step"],
                "no_inward_min_clearance": inward_outcome["min_clearance"],
                "guard_success": guard_outcome["success"],
                "guard_collision": guard_outcome["collision"],
                "guard_collision_step": guard_outcome["collision_step"],
                "guard_min_clearance": guard_outcome["min_clearance"],
                "guard_modified_steps": sum(bool(r["production_approach_blocked"])
                                            for r in guard_rows),
                **divergence}
            comparisons.append(comparison)
            if divergence["label"] != "GUARD_MATCHES_NO_INWARD":
                labels.append({"episode_id": identity, "start_step": start,
                    "relative_step": relative, "label": divergence["label"],
                    "evidence_step": divergence["first_action_divergence_step"],
                    "false_predicates": divergence["false_predicates"],
                    "no_inward_success": inward_outcome["success"],
                    "guard_success": guard_outcome["success"]})
            cf._json(root / f"replay_start_{relative:+03d}.json",
                {"baseline": {"outcome": baseline_outcome, "trace": baseline},
                 "no_inward": {"outcome": inward_outcome, "trace": no_inward},
                 "guard": {"outcome": guard_outcome, "trace": guard_trace,
                           "diagnostic": guard_rows},
                 "first_divergence": divergence})
    finally:
        env.close()
    summary["current_guard_rescues"] = any(r["guard_success"] for r in comparisons)
    summary["persistent_no_inward_rescues"] = bool(successful)
    summary["guard_rescues_at_no_inward_rescue_starts"] = sum(
        r["guard_success"] and r["no_inward_success"] for r in comparisons)
    for name in ("distance", "position", "yaw"):
        if false_counts[name]:
            labels.append({"episode_id": identity, "start_step": None,
                "relative_step": None, "label": f"{name.upper()}_PROXY_MISS",
                "evidence_steps": [r["absolute_step"] for r in false_rows
                    if not r[f"{name}_condition"]],
                "false_predicates": [name], "no_inward_success": True,
                "guard_success": None})
    if false_counts["multiple"]:
        labels.append({"episode_id": identity, "start_step": None,
            "relative_step": None, "label": "MULTI_PREDICATE_MISS",
            "evidence_steps": [r["absolute_step"] for r in false_rows if sum(
                not r[f"{name}_condition"] for name in ("distance", "position", "yaw")) > 1],
            "false_predicates": [], "no_inward_success": True,
            "guard_success": None})
    summary["mechanism_labels"] = sorted(set(row["label"] for row in labels)) or [
        "GUARD_MATCHES_NO_INWARD" if all(r["label"] == "GUARD_MATCHES_NO_INWARD"
            for r in comparisons) else "UNRESOLVED"]
    cf._json(root / "summary.json", summary)
    _csv(root / "per_step_guard_audit.csv", [{k: json.dumps(v) if isinstance(v, (list, dict))
        else v for k, v in row.items()} for row in per_step])
    _csv(root / "guard_replay_comparison.csv", [{k: json.dumps(v) if isinstance(v, (list, dict))
        else v for k, v in row.items()} for row in comparisons])
    return per_step, summary, comparisons, labels


def _near_trace(env: sf.SurfaceManipulatorEnv, spec: dynamic.DynamicSpec,
                original: list[dict], snapshots: dict,
                task: sf.SurfaceFeasibilityConfig, onset: int) -> list[dict]:
    start = max(0, onset-10)
    result = []
    cum_tangent_signed = 0.0
    cum_tangent_abs = 0.0
    cum_inward = 0.0
    cum_yaw = 0.0
    cum_actual_tangent = 0.0
    cum_actual_inward = 0.0
    previous_left: float | None = None
    previous_right: float | None = None
    for t in range(start, onset+1):
        cf.restore(env, snapshots[t])
        shape, state, points = cf._observation(env, spec, task, t)
        stored = original[t]
        official, clearance = sf._surface_distance_metrics(env, shape)
        if bool(official) != bool(stored["collision"]) or abs(clearance-stored["clearance"]) > 2e-5:
            raise AssertionError("Native contact trace does not reproduce the recorded trace")
        left = _tip_distance(env, shape, "left")
        right = _tip_distance(env, shape, "right")
        ee = np.asarray(stored["ee_position"], dtype=np.float64)
        target = sf._surface_target(shape, ee, task.pregrasp_clearance)
        signed_yaw_error = float(wrap_angle(target[2]-stored["ee_yaw"]))
        action = (np.asarray(stored["executed_action"], dtype=np.float64)
                  if stored["executed_action"] is not None else None)
        normal = np.asarray(stored["normal_local_xz"], dtype=np.float64)
        tangent = np.asarray([-normal[1], normal[0]])
        tangent_signed = (float(np.dot(action[[0, 2]], tangent))
                          if action is not None else None)
        actual_tangent = None
        actual_inward = None
        if action is not None and t+1 < len(original):
            actual_world = np.asarray(original[t+1]["ee_position"], dtype=np.float64)-ee
            tangent_world = sf.world_action_from_local(
                np.asarray([tangent[0], 0., tangent[1]]), stored["ee_yaw"])
            normal_world = sf.world_action_from_local(
                np.asarray([normal[0], 0., normal[1]]), stored["ee_yaw"])
            actual_tangent = float(np.dot(actual_world, tangent_world))
            actual_inward = max(0.0, -float(np.dot(actual_world, normal_world)))
        translation_norm = (float(np.linalg.norm(action[:3]))
                            if action is not None else None)
        if action is not None:
            cum_tangent_signed += tangent_signed
            cum_tangent_abs += abs(tangent_signed)
            cum_inward += float(stored["inward_m"])
            cum_yaw += float(action[3])
            if actual_tangent is not None:
                cum_actual_tangent += actual_tangent
                cum_actual_inward += actual_inward
        position_ok = stored["position_error"] < task.success_position_threshold
        yaw_ok = stored["yaw_error"] < task.success_rotation_threshold
        aperture_ok = stored["aperture_error"] < task.success_opening_threshold
        row = {"episode_id": spec.identity, "absolute_step": t,
            "relative_to_collision": t-onset,
            "position_error": stored["position_error"],
            "yaw_error": stored["yaw_error"],
            "signed_yaw_error": signed_yaw_error,
            "aperture_error": stored["aperture_error"],
            "clearance": stored["clearance"],
            "inward_m": stored["inward_m"],
            "outward_m": stored["outward_m"],
            "tangent_norm_m": stored["tangent_norm_m"],
            "tangent_local_xz": stored["tangent_local_xz"],
            "tangent_signed_m": tangent_signed,
            "cumulative_tangent_signed_action_m": cum_tangent_signed,
            "cumulative_tangent_absolute_action_m": cum_tangent_abs,
            "cumulative_inward_action_m": cum_inward,
            "cumulative_yaw_action_rad": cum_yaw,
            "actual_ee_tangent_displacement_m": actual_tangent,
            "actual_ee_inward_displacement_m": actual_inward,
            "cumulative_actual_ee_tangent_displacement_m": cum_actual_tangent,
            "cumulative_actual_ee_inward_displacement_m": cum_actual_inward,
            "yaw_command_rad": float(action[3]) if action is not None else None,
            "yaw_command_sign": int(np.sign(action[3])) if action is not None else None,
            "yaw_error_sign": int(np.sign(signed_yaw_error)),
            "yaw_command_sign_reduces_error": (bool(np.sign(action[3]) ==
                np.sign(signed_yaw_error)) if action is not None else None),
            "gripper_command": float(action[4]) if action is not None else None,
            "opening": stored["opening"],
            "left_tip": stored["left_tip"], "right_tip": stored["right_tip"],
            "left_tip_surface_distance_m": left,
            "right_tip_surface_distance_m": right,
            "left_approach_distance_change_m": left-previous_left if previous_left is not None else None,
            "right_approach_distance_change_m": right-previous_right if previous_right is not None else None,
            "position_success_condition": position_ok,
            "yaw_success_condition": yaw_ok,
            "aperture_success_condition": aperture_ok,
            "all_geometric_conditions_except_collision": position_ok and yaw_ok and aperture_ok,
            "translation_action_norm_m": translation_norm,
            "action_norm": float(np.linalg.norm(action)) if action is not None else None,
            "yaw_action_abs_rad": abs(float(action[3])) if action is not None else None,
            "translation_margin_to_native_limit_m": (
                task.max_delta_ee-translation_norm if action is not None else None),
            "yaw_margin_to_native_limit_rad": (
                task.max_delta_rotation-abs(float(action[3])) if action is not None else None),
            "gripper_margin_to_native_limit_fraction": (
                task.max_delta_gripper-abs(float(action[4])) if action is not None else None),
            "collision": bool(official),
            "contacting_robot_geoms": stored["collision_robot_geoms"],
            "contacting_surface_geoms": stored["collision_surface_geoms"]}
        result.append(row)
        previous_left, previous_right = left, right
    return result


def _one_step_action(action: np.ndarray, normal: np.ndarray,
                     variant: str) -> np.ndarray:
    """Diagnostic only; preserves the original gripper command."""
    result = action.copy()
    original_yaw_gripper = action[[3, 4]].copy()
    xz = np.asarray(action[[0, 2]], dtype=np.float64)
    normal_component = float(np.dot(xz, normal))*normal
    tangent_component = xz-normal_component
    if variant == "original":
        pass
    elif variant == "no_inward":
        inward = max(0.0, -float(np.dot(xz, normal)))
        result[[0, 2]] = xz+inward*normal
    elif variant == "no_tangent":
        result[[0, 2]] = normal_component
    elif variant == "no_translation":
        result[:3] = 0.0
    elif variant == "no_yaw":
        result[3] = 0.0
    elif variant == "half_tangent":
        result[[0, 2]] = normal_component+.5*tangent_component
    elif variant == "half_translation":
        result[:3] *= .5
    else:
        raise ValueError(variant)
    if not np.isclose(result[4], original_yaw_gripper[1]):
        raise AssertionError("One-step diagnostic changed the gripper action")
    if variant != "no_yaw" and not np.isclose(result[3], original_yaw_gripper[0]):
        raise AssertionError("One-step diagnostic changed yaw unexpectedly")
    if variant in ("no_tangent", "half_tangent"):
        new_tangent = result[[0, 2]]-np.dot(result[[0, 2]], normal)*normal
        factor = 0 if variant == "no_tangent" else .5
        if not np.allclose(new_tangent, factor*tangent_component, atol=2e-7):
            raise AssertionError("Tangent intervention did not preserve normal motion")
    return result


def _component_counterfactual(env: sf.SurfaceManipulatorEnv,
                              spec: dynamic.DynamicSpec, original: list[dict],
                              snapshots: dict, task: sf.SurfaceFeasibilityConfig,
                              controller: motion.YawSpatialController,
                              device: torch.device, onset: int) -> list[dict]:
    start = onset-1
    stored = original[start]
    normal = np.asarray(stored["normal_local_xz"], dtype=np.float64)
    rows = []
    for variant in ONE_STEP_VARIANTS:
        previous = cf.restore(env, snapshots[start])
        shape, state, points = cf._observation(env, spec, task, start)
        baseline = cf._policy_action(controller, state, points, previous, task, device)
        if not np.allclose(baseline, stored["executed_action"], atol=2e-5):
            raise AssertionError("One-step baseline action changed")
        action = _one_step_action(baseline, normal, variant)
        sf.apply_local_action(env, action, task)
        post_shape, _, _ = cf._observation(env, spec, task, onset)
        contacts = extract_contacts(env, post_shape)
        left = _tip_distance(env, post_shape, "left")
        right = _tip_distance(env, post_shape, "right")
        if variant == "original":
            if not contacts["official_collision"] or abs(
                    contacts["official_clearance_m"]-original[onset]["clearance"]) > 2e-5:
                raise AssertionError("Original one-step contact failed to reproduce")
        rows.append({"episode_id": spec.identity, "variant": variant,
            "start_step": start, "contact_check_step": onset,
            "immediate_collision": contacts["official_collision"],
            "clearance_after_step": contacts["official_clearance_m"],
            "left_distance_after": left, "right_distance_after": right,
            "original_action": baseline.tolist(), "executed_action": action.tolist(),
            "first_contact_robot_geoms": [r["robot_geom"] for r in
                contacts["official_trigger_pairs"]],
            "first_contact_surface_geoms": [r["surface_geom"] for r in
                contacts["official_trigger_pairs"]]})
    return rows


def audit_near_episode(source: Path, output: Path, seed: int, episode_id: int,
                       device: torch.device) -> tuple[dict, list[dict], list[dict]]:
    spec, original, snapshots, metadata, parity = _load_episode(
        source, seed, episode_id, device)
    task = motion.task(seed, eval_device=device.type)
    controller = contact._final_controller(seed, device)
    onset = int(metadata["original_outcome"]["collision_step"])
    root = ensure_dir(output / "near_target" / f"{seed}_{episode_id:04d}")
    env = sf.make_env(task, seed+864_211)
    try:
        cf.restore(env, snapshots[onset])
        shape, _, _ = cf._observation(env, spec, task, onset)
        contacts = extract_contacts(env, shape)
        if not contacts["official_collision"]:
            raise AssertionError("Official onset has no triggering contact")
        native_contact_count = sum(r["has_native_contact"]
                                   for r in contacts["official_trigger_pairs"])
        trace = _near_trace(env, spec, original, snapshots, task, onset)
        one_step = _component_counterfactual(env, spec, original, snapshots,
                                             task, controller, device, onset)
    finally:
        env.close()
    cf._json(root / "contact.json", {"episode_id": spec.identity,
        "collision_step": onset, "direction": spec.direction,
        "speed_m_per_step": spec.speed_m_per_step, **contacts})
    _csv(root / "precontact_trace.csv", [{k: json.dumps(v) if isinstance(v, (list, dict))
        else v for k, v in row.items()} for row in trace])
    cf._json(root / "precontact_trace.json", trace)
    _csv(root / "component_counterfactual.csv", [{k: json.dumps(v) if isinstance(v, (list, dict))
        else v for k, v in row.items()} for row in one_step])
    cf._json(root / "component_counterfactual.json", one_step)
    pre = trace[-2]
    tail = trace[max(0, len(trace)-6):-1]
    parts = sorted({row["robot_part"] for row in contacts["official_trigger_pairs"]})
    sides = sorted({"left" if p.startswith("left") else "right" if p.startswith("right")
                    else "body" for p in parts})
    summary = {"episode_id": spec.identity, "seed": seed,
        "collision_step": onset,
        "target_motion_direction": spec.direction,
        "target_speed_m_per_step": spec.speed_m_per_step,
        "contact_robot_part": parts,
        "contact_target_geom": sorted({r["surface_geom"] for r in
                                        contacts["official_trigger_pairs"]}),
        "contact_side": sides,
        "official_trigger_pair_count": len(contacts["official_trigger_pairs"]),
        "official_pairs_with_native_contact": native_contact_count,
        "pre_position_error": pre["position_error"],
        "pre_yaw_error": pre["yaw_error"],
        "pre_clearance": pre["clearance"],
        "pre_inward": pre["inward_m"],
        "pre_tangent": pre["tangent_norm_m"],
        "pre_dyaw": pre["yaw_command_rad"],
        "pre_left_distance": pre["left_tip_surface_distance_m"],
        "pre_right_distance": pre["right_tip_surface_distance_m"],
        "last_five_cumulative_signed_tangent_action_m": sum(
            r["tangent_signed_m"] for r in tail),
        "last_five_cumulative_absolute_tangent_action_m": sum(
            abs(r["tangent_signed_m"]) for r in tail),
        "last_five_cumulative_inward_action_m": sum(r["inward_m"] for r in tail),
        "last_five_cumulative_yaw_action_rad": sum(r["yaw_command_rad"] for r in tail),
        "last_five_actual_ee_tangent_displacement_m": sum(
            r["actual_ee_tangent_displacement_m"] for r in tail),
        "last_five_actual_ee_inward_displacement_m": sum(
            r["actual_ee_inward_displacement_m"] for r in tail),
        "precontact_all_geometry_pass_steps": [r["absolute_step"] for r in trace[:-1]
            if r["all_geometric_conditions_except_collision"]],
        "one_step_avoided_variants": [r["variant"] for r in one_step
                                      if not r["immediate_collision"]],
        "one_step_only": "Immediate contact diagnostic; no variant is called a task rescue",
        "original_reproduction_max_difference": max(parity["max_absolute_differences"].values())}
    cf._json(root / "summary.json", summary)
    return summary, trace, one_step


def _large_worker(payload: tuple) -> tuple[int, tuple]:
    index, source, output, seed, episode_id, preference, zone, rows, windows, config = payload
    device = select_device(preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    return index, audit_large_episode(source, output, seed, episode_id,
                                      device, zone, rows, windows, config)


def _near_worker(payload: tuple) -> tuple[int, tuple]:
    index, source, output, seed, episode_id, preference = payload
    device = select_device(preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    return index, audit_near_episode(source, output, seed, episode_id, device)


def run(source: Path = cf.OUTPUT, output: Path = OUTPUT,
        eval_device: DevicePreference = "cpu", eval_workers: int = 1) -> dict:
    if eval_workers < 1:
        raise ValueError("--eval-workers must be positive")
    device = select_device(eval_device)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    source_config, aggregate, source_rows, windows = _read_counterfactual(source)
    _verify_hashes(source_config)
    zone_data = json.loads((contact.OUTPUT /
        "approach_safety/training_zone/zone.json").read_text())
    zone = float(zone_data["zone_m"])
    guard_definition(output, zone)
    cf._json(output / "audit_config.json", {"source": str(source),
        "source_config_sha256": hashlib.sha256((source / "experiment_config.json").read_bytes()).hexdigest(),
        "source_summary_sha256": hashlib.sha256((source / "summary.csv").read_bytes()).hexdigest(),
        "zone_m": zone, "device": str(device), "eval_workers": eval_workers,
        "large_episodes": sorted([list(x) for x in LARGE]),
        "near_episodes": sorted([list(x) for x in NEAR]),
        "checkpoints": source_config["checkpoint_hashes"]})
    guard_steps, guard_summaries, comparisons, labels = [], [], [], []
    large_payloads = [(index, source, output, seed, episode_id, eval_device,
        zone, source_rows, windows, source_config)
        for index, (seed, episode_id) in enumerate(sorted(LARGE))]
    if eval_workers > 1:
        with ProcessPoolExecutor(max_workers=eval_workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            large_outcomes = list(pool.map(_large_worker, large_payloads))
    else:
        large_outcomes = [_large_worker(payload) for payload in large_payloads]
    for index, (step_rows, summary, replay_rows, mechanism_rows) in sorted(large_outcomes):
        seed, episode_id = sorted(LARGE)[index]
        guard_steps.extend(step_rows)
        guard_summaries.append(summary)
        comparisons.extend(replay_rows)
        labels.extend(mechanism_rows)
        print(f"Guard audit {seed}/{episode_id:04d}: "
              f"{summary['guard_active_steps_in_rescue_window']}/"
              f"{summary['rescue_window_size']} original rescue steps covered", flush=True)
    guard_root = ensure_dir(output / "guard_audit")
    _csv(guard_root / "per_step_guard_audit.csv", [{k: json.dumps(v) if isinstance(v, (list, dict))
        else v for k, v in row.items()} for row in guard_steps])
    _csv(guard_root / "episode_guard_summary.csv", [{k: json.dumps(v) if isinstance(v, (list, dict))
        else v for k, v in row.items()} for row in guard_summaries])
    _csv(guard_root / "guard_replay_comparison.csv", [{k: json.dumps(v) if isinstance(v, (list, dict))
        else v for k, v in row.items()} for row in comparisons])
    cf._json(guard_root / "guard_mechanism_labels.json", labels)
    near_summaries, near_traces, components = [], [], []
    near_payloads = [(index, source, output, seed, episode_id, eval_device)
        for index, (seed, episode_id) in enumerate(sorted(NEAR))]
    if eval_workers > 1:
        with ProcessPoolExecutor(max_workers=eval_workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            near_outcomes = list(pool.map(_near_worker, near_payloads))
    else:
        near_outcomes = [_near_worker(payload) for payload in near_payloads]
    for index, (summary, trace, one_step) in sorted(near_outcomes):
        seed, episode_id = sorted(NEAR)[index]
        near_summaries.append(summary)
        near_traces.extend(trace)
        components.extend(one_step)
        print(f"Near-target audit {seed}/{episode_id:04d}: "
              f"{summary['contact_robot_part']}", flush=True)
    _csv(output / "near_target_summary.csv", [{k: json.dumps(v) if isinstance(v, (list, dict))
        else v for k, v in row.items()} for row in near_summaries])
    _csv(output / "near_target" / "precontact_trace.csv", [
        {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
        for row in near_traces])
    _csv(output / "near_target" / "component_counterfactual.csv", [
        {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
        for row in components])
    result = {"guard_episodes": guard_summaries,
        "near_target_episodes": near_summaries,
        "guard_replay_trials": len(comparisons),
        "near_target_one_step_trials": len(components),
        "guard_aggregate": {
            "rescuable_start_steps": sum(r["rescue_window_size"] for r in guard_summaries),
            "original_guard_condition_steps_in_rescue_windows": sum(
                r["guard_active_steps_in_rescue_window"] for r in guard_summaries),
            "rescuable_guard_false_steps": sum(r["rescuable_guard_false_steps"]
                                                for r in guard_summaries),
            "distance_false_count": sum(r["false_predicate_counts"]["distance"]
                                        for r in guard_summaries),
            "position_false_count": sum(r["false_predicate_counts"]["position"]
                                        for r in guard_summaries),
            "yaw_false_count": sum(r["false_predicate_counts"]["yaw"]
                                   for r in guard_summaries),
            "full_episode_guard_successes": sum(r["full_episode_guard_success"]
                                                for r in guard_summaries),
            "guard_success_at_no_inward_success_starts": sum(
                r["guard_success"] and r["no_inward_success"] for r in comparisons),
            "no_inward_success_starts": sum(r["no_inward_success"] for r in comparisons)},
        "guard_labels": labels,
        "interpretation_limit": "Diagnostic replay on the original held-out episodes; no threshold or controller tuning and no claim of deployed-policy improvement"}
    cf._json(output / "analysis_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=cf.OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    args = parser.parse_args()
    result = run(args.source_dir, args.output_dir, args.eval_device, args.eval_workers)
    print(json.dumps({"guard_replay_trials": result["guard_replay_trials"],
                      "near_target_one_step_trials": result["near_target_one_step_trials"]},
                     indent=2))


if __name__ == "__main__":
    main()
