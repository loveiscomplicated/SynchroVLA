"""Contact coordination diagnostics on the frozen 5D spatial controller."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_gnn_recurrent.training import dynamic_visual_delay_feasibility as dynamic
from vla_gnn_recurrent.training import motion_controller_performance as motion
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import residual_recurrent_feasibility as residual
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device


OUTPUT = Path("artifacts/contact_aware_performance_v1")
SEEDS = motion.SEEDS


def _write(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _csv(path: Path, rows: list[dict]) -> None:
    motion.write_csv(path, rows)


def _final_controller(seed: int, device: torch.device) -> motion.YawSpatialController:
    combined = motion.load_combined(seed, device, motion.paths(seed)["updated_motion"])
    return motion.YawSpatialController(combined, motion.OUTPUT /
        f"seed{seed}/yaw_spatial_readout/selected.pt", device)


def reproduce(output: Path = OUTPUT, eval_device: DevicePreference = "cpu",
              eval_workers: int = 1) -> dict:
    root = ensure_dir(output / "reproduction")
    expected = json.loads((motion.OUTPUT / "aggregate/three_seed_totals.json").read_text())
    checkpoints = json.loads((motion.OUTPUT / "aggregate/checkpoint_provenance.json").read_text())
    summaries = {}
    for seed in SEEDS:
        paths = motion.paths(seed)
        hashes = {"starting_motion_sha256": motion.sha(paths["motion"]),
                  "adapted_motion_sha256": motion.sha(paths["updated_motion"]),
                  "aperture_branch_sha256": motion.sha(paths["aperture"]),
                  "yaw_readout_sha256": motion.sha(motion.OUTPUT /
                      f"seed{seed}/yaw_spatial_readout/selected.pt"),
                  "base_ff_sha256": motion.sha(residual.BASE_CHECKPOINT)}
        if hashes != checkpoints[str(seed)]:
            raise AssertionError(f"Frozen checkpoint provenance changed for {seed}")
        _write(root / f"seed{seed}_checkpoint_hashes.json", hashes)
        summaries[seed] = {}
        for condition in ("static", "dynamic"):
            rows, summary = motion.official_evaluate(seed, condition,
                eval_device_preference=eval_device, eval_workers=eval_workers,
                motion_checkpoint=paths["updated_motion"],
                yaw_head_checkpoint=motion.OUTPUT /
                    f"seed{seed}/yaw_spatial_readout/selected.pt")
            old = json.loads((motion.OUTPUT /
                f"seed{seed}/yaw_spatial_readout/{condition}_summary.json").read_text())
            for key in ("success", "position_fail", "yaw_fail", "aperture_fail",
                        "collision_fail", "multiple_failures"):
                if summary[key] != old[key]:
                    raise AssertionError(f"Baseline reproduction drifted: {seed}/{condition}/{key}")
            for key in ("final_position_mean_m", "final_yaw_mean_rad",
                        "final_aperture_mean_m", "minimum_clearance_mean_m"):
                if abs(summary[key]-old[key]) > 1e-6:
                    raise AssertionError(f"Baseline metric drifted: {seed}/{condition}/{key}")
            _csv(root / f"seed{seed}_{condition}_per_episode.csv", rows)
            _write(root / f"seed{seed}_{condition}_summary.json", summary)
            summaries[seed][condition] = summary
    for condition, target in (("static", (46, 0, 1, 0, 1)),
                              ("dynamic", (38, 5, 2, 0, 8))):
        keys = ("success", "position_fail", "yaw_fail", "aperture_fail", "collision_fail")
        actual = tuple(sum(summaries[s][condition][key] for s in SEEDS) for key in keys)
        if actual != target or any(sum(summaries[s][condition][key] for s in SEEDS) !=
                                   expected[condition]["yaw_spatial_readout"][key] for key in keys):
            raise AssertionError(f"Three-seed baseline changed: {condition} {actual}")
    _write(root / "summary.json", summaries)
    return summaries


def observed_geometry(points: np.ndarray, clearance: float) -> dict:
    """PCA estimate from the canonical unordered EE-local point cloud only."""
    xz = np.asarray(points[:, [0, 2]], dtype=np.float64)
    center = np.median(xz, axis=0)
    values, vectors = np.linalg.eigh(np.cov(xz.T))
    tangent = vectors[:, int(np.argmax(values))]
    normal = np.asarray([-tangent[1], tangent[0]])
    if float(np.dot(-center, normal)) < 0:
        normal = -normal
    along = (xz-center) @ tangent
    near_middle = np.abs(along-np.median(along)) <= max(float(np.ptp(along))*.26, 1e-5)
    selected = xz[near_middle] if np.any(near_middle) else xz
    projections = selected @ normal
    side = selected[projections >= np.quantile(projections, .86)]
    surface_center = np.median(side, axis=0) if len(side) else center + np.max(projections)*normal
    target_local = surface_center + clearance*normal
    return {"normal_local_xz": normal, "target_local_xz": target_local,
            "position_proxy_m": float(np.linalg.norm(target_local)),
            "yaw_proxy_rad": abs(math.atan2(float(normal[1]), float(normal[0])))}


def _observed_points(seed: int, spec: Any, timestep: int, ee_position: np.ndarray,
                     ee_yaw: float, condition: str) -> np.ndarray:
    shape = dynamic.moving_shape(spec, timestep) if condition == "dynamic" else spec.shape
    initial = spec.initial if condition == "dynamic" else spec
    world = sf.sample_surface_points_world(shape, 32, sf._stable_seed(initial.sample_identity))
    return sf.localize_world(world, ee_position, ee_yaw).astype(np.float32)


def collision_audit(output: Path = OUTPUT) -> dict:
    if not (output / "reproduction/summary.json").exists():
        raise FileNotFoundError("Reproduce the frozen baseline before collision audit")
    root = ensure_dir(output / "collision_audit")
    traces, events, windows, episode_rows = [], [], [], []
    for seed in SEEDS:
        task = motion.task(seed)
        specs = {spec.identity: spec for spec in dynamic.dynamic_specs(seed, task, 16,
            dynamic.SELECTED_STEP_SPEEDS_M, "heldout")}
        source = motion.OUTPUT / f"seed{seed}/final_diagnostics/fixed_horizon_traces.csv"
        source_events = motion.OUTPUT / f"seed{seed}/final_diagnostics/collision_events.csv"
        old_events = {row["episode_id"]: row for row in csv.DictReader(source_events.open())
                      if row["condition"] == "dynamic"}
        per_episode = {}
        for row in csv.DictReader(source.open()):
            if row["condition"] != "dynamic":
                continue
            identity = row["episode_id"]
            t = int(row["timestep"])
            ee = np.asarray(ast.literal_eval(row["EE_position"]), dtype=np.float64)
            points = _observed_points(seed, specs[identity], t, ee, float(row["EE_yaw"]), "dynamic")
            geom = observed_geometry(points, task.pregrasp_clearance)
            tips = [np.asarray(ast.literal_eval(row[name]), dtype=np.float64)
                    for name in ("left_tip", "right_tip")]
            points_world = sf.sample_surface_points_world(dynamic.moving_shape(specs[identity], t),
                32, sf._stable_seed(specs[identity].initial.sample_identity))
            fingertip_distance = min(float(np.linalg.norm(points_world-tip, axis=1).min())
                                      for tip in tips)
            shape = dynamic.moving_shape(specs[identity], t)
            target_opening = sf._surface_target(shape, ee, task.pregrasp_clearance)[3]
            contacts = ast.literal_eval(row["contacts"])
            augmented = {"episode_id": identity, "seed": seed, "timestep": t,
                "controller": "frozen_best", "EE_position": row["EE_position"],
                "EE_yaw": float(row["EE_yaw"]),
                "target_position": row["target_position"],
                "target_yaw": float(row["target_yaw"]),
                "position_error": float(row["position_error"]),
                "yaw_error": float(row["yaw_error"]),
                "opening": float(row["opening_width_m"]),
                "target_opening": float(target_opening),
                "left_tip": row["left_tip"], "right_tip": row["right_tip"],
                "translation_action": row["translation_action"],
                "yaw_action": float(row["yaw_action"]),
                "requested_gripper_action": float(row["gripper_action"]),
                "executed_gripper_action": float(row["gripper_action"]),
                "clearance": float(row["clearance"]),
                "collision": row["collision"] == "True",
                "collision_robot_geom": sorted(set(c["robot_geom"] for c in contacts)),
                "collision_surface_geom": sorted(set(c["surface_geom"] for c in contacts)),
                "fingertip_distance_to_observed_surface_m": fingertip_distance,
                "observed_position_proxy_m": geom["position_proxy_m"],
                "observed_yaw_proxy_rad": geom["yaw_proxy_rad"],
                "surface_normal_local_xz": geom["normal_local_xz"].tolist(),
                "inward_translation_action_m": float(-np.dot(
                    np.asarray(ast.literal_eval(row["translation_action"]))[[0, 2]],
                    geom["normal_local_xz"]))}
            per_episode.setdefault(identity, []).append(augmented)
        for identity, rows in per_episode.items():
            for i, row in enumerate(rows):
                prior = rows[i-1] if i else None
                row["clearance_delta"] = (row["clearance"]-prior["clearance"] if prior else None)
                row["approach_velocity_proxy"] = (-row["clearance_delta"] if prior else None)
                row["opening_change_from_previous_m"] = (row["opening"]-prior["opening"] if prior else None)
                traces.append(row)
                if row["collision"] and (prior is None or not prior["collision"]):
                    if identity not in old_events or i != int(old_events[identity]["timestep"]):
                        raise AssertionError("Collision onset differs from prior validated trace")
                    context = rows[max(0, i-5):i+1]
                    windows.extend([{**r, "onset_timestep": i,
                                     "relative_to_onset": r["timestep"]-i} for r in context])
                    events.append({**row, "previous_step_clearance": prior["clearance"] if prior else None,
                        "previous_position_error": prior["position_error"] if prior else None,
                        "previous_yaw_error": prior["yaw_error"] if prior else None,
                        "previous_gripper_action": prior["executed_gripper_action"] if prior else None,
                        "previous_opening_m": prior["opening"] if prior else None})
            episode_rows.append({"episode_id": identity, "seed": seed,
                "collision": any(r["collision"] for r in rows),
                "minimum_clearance_m": min(r["clearance"] for r in rows),
                "first_substantial_close_step": next((r["timestep"] for r in rows
                    if r["executed_gripper_action"] < -.01), None),
                "first_substantial_close_position_error_m": next((r["position_error"] for r in rows
                    if r["executed_gripper_action"] < -.01), None),
                "first_substantial_close_yaw_error_rad": next((r["yaw_error"] for r in rows
                    if r["executed_gripper_action"] < -.01), None)})
    if len(events) != 8 or len(traces) != 48*motion.task(2811).max_steps:
        raise AssertionError("Baseline collision count or trace length drifted")
    _csv(root / "all_dynamic_timesteps.csv", traces)
    _csv(root / "collision_onsets.csv", events)
    _csv(root / "five_step_windows.csv", windows)
    _csv(root / "episode_closing_timing.csv", episode_rows)
    result = {"episodes": len(episode_rows), "timesteps": len(traces),
              "collision_onsets": len(events), "window_rows": len(windows),
              "source": "replayed validated final fixed-horizon traces; observed point cloud regenerated from same EpisodeSpec/sample seed",
              "target_opening_and_true_errors": "diagnostic-only oracle geometry, never passed to gate"}
    _write(root / "summary.json", result)
    return result


def classify_and_analyze(output: Path = OUTPUT) -> dict:
    root = ensure_dir(output / "collision_taxonomy")
    source = output / "collision_audit"
    trace = {(r["episode_id"], int(r["timestep"])): r for r in
             csv.DictReader((source / "all_dynamic_timesteps.csv").open())}
    events = []
    for event in csv.DictReader((source / "collision_onsets.csv").open()):
        prior = trace[(event["episode_id"], int(event["timestep"])-1)]
        position = float(event["previous_position_error"])
        yaw = float(event["previous_yaw_error"])
        inward = float(prior["inward_translation_action_m"])
        actual_close_m = float(event["opening_change_from_previous_m"])
        clearance_rate = float(event["approach_velocity_proxy"])
        task = motion.task(int(event["seed"]))
        mispositioned = position >= task.success_position_threshold
        misoriented = yaw >= task.success_rotation_threshold
        substantial_actual_close = actual_close_m < -.001
        if mispositioned and misoriented:
            category = "D_mixed_position_yaw"
        elif substantial_actual_close and (mispositioned or misoriented):
            category = "A_premature_closing"
        elif misoriented and not mispositioned:
            category = "B_orientation_approach"
        elif inward > 0 and clearance_rate > 0 and float(event["clearance"]) <= 0:
            category = "C_translation_overshoot"
        else:
            category = "E_ambiguous"
        events.append({"episode_id": event["episode_id"], "seed": int(event["seed"]),
            "onset_timestep": int(event["timestep"]), "taxonomy": category,
            "robot_geoms": event["collision_robot_geom"],
            "surface_geoms": event["collision_surface_geom"],
            "evidence": (f"prior position={position:.4f}m; yaw={yaw:.3f}rad; "
                         f"inward translation={inward:.4f}m; clearance drop={clearance_rate:.4f}m; "
                         f"actual opening change={actual_close_m:.8f}m"),
            "prior_position_error_m": position, "prior_yaw_error_rad": yaw,
            "prior_opening_m": float(event["previous_opening_m"]),
            "prior_clearance_m": float(event["previous_step_clearance"]),
            "onset_clearance_m": float(event["clearance"]),
            "clearance_drop_m": clearance_rate,
            "prior_inward_translation_action_m": inward,
            "prior_translation_action": prior["translation_action"],
            "prior_yaw_action": float(prior["yaw_action"]),
            "prior_gripper_action": float(event["previous_gripper_action"]),
            "actual_opening_change_m": actual_close_m,
            "prior_fingertip_surface_distance_m": float(
                prior["fingertip_distance_to_observed_surface_m"])})
    _csv(root / "episode_classification.csv", events)
    timing = list(csv.DictReader((source / "episode_closing_timing.csv").open()))
    official = {r["episode_id"]: r for seed in SEEDS for r in csv.DictReader((output /
        f"reproduction/seed{seed}_dynamic_per_episode.csv").open())}
    groups = {"collision": [r for r in timing if r["collision"] == "True"],
              "successful": [r for r in timing if official[r["episode_id"]]["success"] == "True"]}
    timing_summary = {}
    for name, rows in groups.items():
        has_close = [r for r in rows if r["first_substantial_close_step"]]
        timing_summary[name] = {"episodes": len(rows),
            "episodes_with_action_below_minus_0p01": len(has_close),
            "first_close_step_mean": float(np.mean([int(r["first_substantial_close_step"])
                                               for r in has_close])) if has_close else None,
            "first_close_position_error_mean_m": float(np.mean([
                float(r["first_substantial_close_position_error_m"]) for r in has_close])) if has_close else None,
            "first_close_yaw_error_mean_rad": float(np.mean([
                float(r["first_substantial_close_yaw_error_rad"]) for r in has_close])) if has_close else None}
    result = {"rule": {"thresholds": "existing position 0.025m, yaw 0.20rad; actual closing requires >1mm measured aperture decrease",
                       "priority": ["D both errors", "A actual close while misaligned",
                                    "B yaw only", "C inward translation plus falling clearance",
                                    "E ambiguous"],
                       "caution": "Commands do not prove motion; these are evidence-based classes, not causal attribution."},
              "counts": {name: sum(e["taxonomy"] == name for e in events)
                         for name in sorted(set(e["taxonomy"] for e in events))},
              "closing_timing": timing_summary}
    _write(root / "summary.json", result)
    return result


class ContactController:
    """Analytic coordination layer; the learned action heads remain unchanged."""
    def __init__(self, base: motion.YawSpatialController, mode: str,
                 approach_zone_m: float | None = None,
                 inward_cap_m: float | None = None) -> None:
        if mode not in ("none", "close_gate", "approach_guard", "approach_cap",
                        "orientation_first", "near_mixed", "both"):
            raise ValueError(mode)
        self.base, self.mode = base, mode
        self.motion = base.motion
        self.task = base.task
        self.approach_zone_m = (approach_zone_m if approach_zone_m is not None else
                                .5*self.task.pregrasp_clearance)
        self.inward_cap_m = inward_cap_m
        if mode == "approach_cap" and (inward_cap_m is None or inward_cap_m <= 0):
            raise ValueError("Approach cap needs a positive training-derived inward limit")
        self.last_diagnostic: list[dict] = []
        self.last_requested_action: list[list[float]] = []
        self.close_blocked_steps = 0
        self.approach_blocked_steps = 0

    @torch.no_grad()
    def step(self, state: torch.Tensor, points: torch.Tensor, previous: torch.Tensor,
             hidden: torch.Tensor | None = None, topology: sf.GraphTopology | None = None,
             reset_hidden: bool = False) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
        action, _, _, base = self.base.step(state, points, previous, hidden,
                                           topology, reset_hidden)
        self.last_requested_action = action.detach().cpu().tolist()
        output = action.clone()
        self.last_diagnostic = []
        for i in range(len(state)):
            geometry = observed_geometry(points[i].detach().cpu().numpy(),
                                         self.task.pregrasp_clearance)
            aligned = (geometry["position_proxy_m"] < self.task.success_position_threshold and
                       geometry["yaw_proxy_rad"] < self.task.success_rotation_threshold)
            close_blocked = False
            if self.mode in ("close_gate", "both") and not aligned and float(output[i, 4]) < 0:
                output[i, 4] = 0
                close_blocked = True
            approach_blocked = False
            if self.mode in ("approach_guard", "approach_cap", "orientation_first",
                             "near_mixed", "both"):
                tips = state[i, 8:14].detach().cpu().numpy().reshape(2, 3)
                cloud = points[i].detach().cpu().numpy()
                tip_distance = min(float(np.linalg.norm(cloud-tip, axis=1).min()) for tip in tips)
                # The zone is fixed either by the physical half-clearance rule or training data.
                near_surface = tip_distance < self.approach_zone_m
                normal = geometry["normal_local_xz"]
                translation = output[i, [0, 2]].detach().cpu().numpy()
                inward = float(-np.dot(translation, normal))
                limit = (self.inward_cap_m if self.mode == "approach_cap" else 0.0)
                defer_for_orientation = (self.mode in ("orientation_first", "near_mixed") and
                    geometry["position_proxy_m"] >= self.task.success_position_threshold and
                    geometry["yaw_proxy_rad"] >= self.task.success_rotation_threshold)
                if self.mode == "orientation_first":
                    active = defer_for_orientation
                elif self.mode == "near_mixed":
                    active = near_surface and defer_for_orientation
                else:
                    active = near_surface and not aligned
                if active and inward > limit:
                    corrected = translation + (inward-limit)*normal
                    output[i, 0] = float(corrected[0])
                    output[i, 2] = float(corrected[1])
                    approach_blocked = True
            self.last_diagnostic.append({"observed_position_proxy_m": geometry["position_proxy_m"],
                "observed_yaw_proxy_rad": geometry["yaw_proxy_rad"],
                "aligned": aligned, "close_blocked": close_blocked,
                "approach_blocked": approach_blocked})
            self.close_blocked_steps += int(close_blocked)
            self.approach_blocked_steps += int(approach_blocked)
        return output, output-base, None, base


def _eval_worker(payload: tuple) -> tuple[int, dict]:
    seed, condition, index, spec, mode, device_preference, approach_zone_m, inward_cap_m = payload
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task = motion.task(seed, eval_device=device_preference)
    controller = ContactController(_final_controller(seed, device), mode,
                                   approach_zone_m, inward_cap_m)
    env = sf.make_env(task, seed+864_211)
    try:
        if condition == "dynamic":
            row, trace = dynamic.rollout(env, spec, task, device, "residual_mlp", controller, 0)
            identity = spec.identity
        else:
            row, trace = residual.rollout_residual(env, spec, controller, task,
                rf.TemporalConfig(), device)
            identity = spec.sample_identity
        if any(step.get("oracle_action") is not None for step in trace):
            raise AssertionError("Evaluation called scripted oracle")
        result = {"seed": seed, "condition": condition, "episode_id": identity,
            "controller": mode, "success": bool(row["success"]),
            "collision": bool(row["collision"]),
            "position_error": float(row["final_tracking_error" if condition == "dynamic"
                                        else "final_position_error"]),
            "yaw_error": float(row["final_yaw_error" if condition == "dynamic"
                                   else "final_orientation_error"]),
            "aperture_error": float(row["final_aperture_error" if condition == "dynamic"
                                        else "final_gripper_width_error"]),
            "minimum_clearance": float(row["minimum_safe_clearance"]),
            "inference_latency_ms": float(row.get("mean_inference_ms") or np.mean(
                [step.get("inference_ms") or 0 for step in trace])),
            "convergence_time": (next((step["t"] for step in trace if
                step.get("tracking_error", step.get("position_error", math.inf)) <
                    task.success_position_threshold and
                step.get("yaw_error", step.get("orientation_error", math.inf)) <
                    task.success_rotation_threshold), None)),
            "close_blocked_steps": controller.close_blocked_steps,
            "approach_blocked_steps": controller.approach_blocked_steps}
        return index, result
    finally:
        env.close()


def evaluate(seed: int, condition: str, mode: str, split: str,
             eval_device: DevicePreference = "cpu", eval_workers: int = 1,
             approach_zone_m: float | None = None,
             inward_cap_m: float | None = None) -> tuple[list[dict], dict]:
    if eval_workers < 1:
        raise ValueError("eval_workers must be positive")
    task = motion.task(seed, eval_device=eval_device)
    if condition == "dynamic":
        specs = dynamic.dynamic_specs(seed, task, 16, dynamic.SELECTED_STEP_SPEEDS_M, split)
    elif split == "heldout":
        specs = sf.sample_episode_specs(16, seed+60_000, task, "iid")
    elif split == "validation":
        specs = sf.sample_episode_specs(16, seed+55_000, task, "iid")
    else:
        specs = sf.sample_episode_specs(16, seed+50_000, task, "iid")
    payloads = [(seed, condition, i, spec, mode, eval_device, approach_zone_m,
                 inward_cap_m)
                for i, spec in enumerate(specs)]
    if eval_workers > 1:
        with ProcessPoolExecutor(max_workers=eval_workers, mp_context=mp.get_context("spawn")) as pool:
            responses = list(pool.map(_eval_worker, payloads))
    else:
        responses = [_eval_worker(payload) for payload in payloads]
    rows = [row for _, row in sorted(responses, key=lambda pair: pair[0])]
    return rows, motion._summary(rows, task)


def analytic_close_gate(seed: int = 2811, split: str = "train", output: Path = OUTPUT,
                        eval_device: DevicePreference = "cpu", eval_workers: int = 1) -> dict:
    root = ensure_dir(output / "closing_timing/analytic_gate" / split / f"seed{seed}")
    results = {}
    for condition in ("static", "dynamic"):
        baseline_rows, baseline_summary = evaluate(seed, condition, "none", split,
            eval_device, eval_workers)
        rows, summary = evaluate(seed, condition, "close_gate", split, eval_device, eval_workers)
        if [r["episode_id"] for r in baseline_rows] != [r["episode_id"] for r in rows]:
            raise AssertionError("Close-gate comparison lost EpisodeSpec pairing")
        _csv(root / f"{condition}_baseline_per_episode.csv", baseline_rows)
        _write(root / f"{condition}_baseline_summary.json", baseline_summary)
        _csv(root / f"{condition}_per_episode.csv", rows)
        _write(root / f"{condition}_summary.json", summary)
        transitions = [{"episode_id": a["episode_id"],
            "baseline_success": a["success"], "gate_success": b["success"],
            "baseline_collision": a["collision"], "gate_collision": b["collision"],
            "baseline_position_error": a["position_error"],
            "gate_position_error": b["position_error"],
            "baseline_yaw_error": a["yaw_error"], "gate_yaw_error": b["yaw_error"],
            "baseline_aperture_error": a["aperture_error"],
            "gate_aperture_error": b["aperture_error"]}
            for a, b in zip(baseline_rows, rows, strict=True)]
        _csv(root / f"{condition}_paired_transitions.csv", transitions)
        results[condition] = {"baseline": baseline_summary, "close_gate": summary,
            "collision_fixed": [r["episode_id"] for r in transitions if
                r["baseline_collision"] and not r["gate_collision"]],
            "new_collision": [r["episode_id"] for r in transitions if
                not r["baseline_collision"] and r["gate_collision"]],
            "success_gained": [r["episode_id"] for r in transitions if
                not r["baseline_success"] and r["gate_success"]],
            "success_lost": [r["episode_id"] for r in transitions if
                r["baseline_success"] and not r["gate_success"]]}
    _write(root / "summary.json", results)
    return results


def clearance_analysis(output: Path = OUTPUT) -> dict:
    root = ensure_dir(output / "approach_safety/clearance_analysis")
    audit = output / "collision_audit"
    rows = list(csv.DictReader((audit / "all_dynamic_timesteps.csv").open()))
    events = list(csv.DictReader((output /
        "collision_taxonomy/episode_classification.csv").open()))
    official = {r["episode_id"]: r for seed in SEEDS for r in csv.DictReader((output /
        f"reproduction/seed{seed}_dynamic_per_episode.csv").open())}
    by_episode = {}
    for r in rows:
        by_episode.setdefault(r["episode_id"], []).append(r)
    transitions = []
    for identity, trace in by_episode.items():
        for i in range(1, len(trace)):
            previous, current = trace[i-1], trace[i]
            task = motion.task(int(previous["seed"]))
            misaligned = (float(previous["position_error"]) >= task.success_position_threshold
                          and float(previous["yaw_error"]) >= task.success_rotation_threshold)
            near = (float(previous["fingertip_distance_to_observed_surface_m"]) <
                    .5*task.pregrasp_clearance)
            inward = float(previous["inward_translation_action_m"])
            transitions.append({"episode_id": identity, "seed": int(previous["seed"]),
                "timestep": int(previous["timestep"]),
                "collision_next_step": current["collision"] == "True",
                "collision_current_step": previous["collision"] == "True",
                "episode_success": official[identity]["success"] == "True",
                "misaligned_position_and_yaw": misaligned,
                "near_surface_proxy": near,
                "prior_clearance_m": float(previous["clearance"]),
                "clearance_drop_m": float(previous["clearance"])-float(current["clearance"]),
                "inward_translation_action_m": inward,
                "translation_action_norm_m": float(np.linalg.norm(
                    ast.literal_eval(previous["translation_action"]))),
                "yaw_action_abs_rad": abs(float(previous["yaw_action"])),
                "prior_position_error_m": float(previous["position_error"]),
                "prior_yaw_error_rad": float(previous["yaw_error"]),
                "prior_tip_surface_distance_m": float(
                    previous["fingertip_distance_to_observed_surface_m"])})
    _csv(root / "paired_step_clearance.csv", transitions)
    risky = [r for r in transitions if not r["collision_current_step"] and
             r["misaligned_position_and_yaw"] and
             r["near_surface_proxy"] and r["inward_translation_action_m"] > 0]
    contact = [r for r in risky if r["collision_next_step"]]
    safe = [r for r in risky if not r["collision_next_step"]]
    result = {"episodes": len(by_episode), "paired_steps": len(transitions),
        "near_misaligned_inward_steps": len(risky),
        "next_step_contact_steps": len(contact),
        "other_steps": len(safe),
        "contact_inward_action_mean_m": float(np.mean([r["inward_translation_action_m"]
                                                       for r in contact])) if contact else None,
        "other_inward_action_mean_m": float(np.mean([r["inward_translation_action_m"]
                                                     for r in safe])) if safe else None,
        "contact_clearance_drop_mean_m": float(np.mean([r["clearance_drop_m"]
                                                       for r in contact])) if contact else None,
        "other_clearance_drop_mean_m": float(np.mean([r["clearance_drop_m"]
                                                     for r in safe])) if safe else None,
        "mixed_collision_events": [e["episode_id"] for e in events if
                                    e["taxonomy"] == "D_mixed_position_yaw"],
        "intervention_rule": "When observed tip-to-surface point distance < half the existing pregrasp clearance and observed PCA alignment fails, cancel only inward normal translation; preserve tangent, retreat, yaw, gripper."}
    _write(root / "summary.json", result)
    return result


def derive_training_approach_zone(output: Path = OUTPUT,
                                  eval_device: DevicePreference = "cpu") -> dict:
    """One frozen safety zone from training-only pre-contact visible geometry."""
    root = ensure_dir(output / "approach_safety/training_zone")
    if (root / "zone.json").exists():
        return json.loads((root / "zone.json").read_text())
    seed = 2811
    task = motion.task(seed, eval_device=eval_device)
    device = select_device(eval_device)
    controller = _final_controller(seed, device)
    specs = dynamic.dynamic_specs(seed, task, 72,
        dynamic.SELECTED_STEP_SPEEDS_M, "train")
    env = sf.make_env(task, seed+864_211)
    traces, outcomes, contact_predecessors = [], [], []
    try:
        for spec in specs:
            rows, outcome = motion.fixed_horizon_trace(env, spec, task,
                controller, device, "dynamic", "frozen_best_training_diagnostic")
            previous = None
            for row in rows:
                t = int(row["timestep"])
                ee = np.asarray(row["EE_position"], dtype=np.float64)
                shape = dynamic.moving_shape(spec, t)
                surface = sf.sample_surface_points_world(shape, 32,
                    sf._stable_seed(spec.initial.sample_identity))
                tip_distance = min(float(np.linalg.norm(surface-np.asarray(row[name]), axis=1).min())
                                   for name in ("left_tip", "right_tip"))
                points = sf.localize_world(surface, ee, float(row["EE_yaw"]))
                visible = observed_geometry(points, task.pregrasp_clearance)
                record = {"episode_id": spec.identity, "timestep": t,
                    "collision": bool(row["collision"]),
                    "position_error": float(row["position_error"]),
                    "yaw_error": float(row["yaw_error"]),
                    "clearance": float(row["clearance"]),
                    "visible_tip_surface_distance_m": tip_distance,
                    "visible_position_proxy_m": visible["position_proxy_m"],
                    "visible_yaw_proxy_rad": visible["yaw_proxy_rad"],
                    "translation_action": row["translation_action"],
                    "yaw_action": row["yaw_action"],
                    "gripper_action": row["gripper_action"]}
                traces.append(record)
                if record["collision"] and previous is not None and not previous["collision"]:
                    contact_predecessors.append({**previous, "onset_timestep": t,
                        "onset_clearance": record["clearance"]})
                previous = record
            outcomes.append({"episode_id": spec.identity, **outcome})
    finally:
        env.close()
    if not contact_predecessors:
        raise AssertionError("No training contact events to derive an approach zone")
    distances = np.asarray([r["visible_tip_surface_distance_m"]
                            for r in contact_predecessors])
    zone = min(task.pregrasp_clearance, float(np.quantile(distances, .95)))
    _csv(root / "fixed_horizon_training_traces.csv", traces)
    _csv(root / "fixed_horizon_training_outcomes.csv", outcomes)
    _csv(root / "contact_predecessors.csv", contact_predecessors)
    result = {"seed": seed, "training_episodes": len(specs),
        "contact_onsets": len(contact_predecessors),
        "pre_contact_visible_tip_distance_p50_m": float(np.quantile(distances, .5)),
        "pre_contact_visible_tip_distance_p95_m": float(np.quantile(distances, .95)),
        "zone_m": zone, "derivation": "min(existing pregrasp clearance, training p95 of observable fingertip-to-surface-point distance one step before contact)",
        "no_heldout_input": True}
    _write(root / "zone.json", result)
    return result


def derive_training_inward_cap(output: Path = OUTPUT,
                               eval_device: DevicePreference = "cpu") -> dict:
    """Choose one cap from successful training-only near-surface inward steps."""
    root = ensure_dir(output / "approach_safety/training_inward_cap")
    if (root / "cap.json").exists():
        return json.loads((root / "cap.json").read_text())
    zone = derive_training_approach_zone(output, eval_device)["zone_m"]
    seed = 2811
    task = motion.task(seed, eval_device=eval_device)
    device = select_device(eval_device)
    controller = _final_controller(seed, device)
    specs = dynamic.dynamic_specs(seed, task, 72,
        dynamic.SELECTED_STEP_SPEEDS_M, "train")
    env = sf.make_env(task, seed+864_211)
    records = []
    try:
        for spec in specs:
            trace, _ = motion.fixed_horizon_trace(env, spec, task,
                controller, device, "dynamic", "training_rate_diagnostic")
            for previous, current in zip(trace[:-1], trace[1:], strict=True):
                t = int(previous["timestep"])
                shape = dynamic.moving_shape(spec, t)
                ee = np.asarray(previous["EE_position"], dtype=np.float64)
                surface = sf.sample_surface_points_world(shape, 32,
                    sf._stable_seed(spec.initial.sample_identity))
                points = sf.localize_world(surface, ee, float(previous["EE_yaw"]))
                geom = observed_geometry(points, task.pregrasp_clearance)
                tip_distance = min(float(np.linalg.norm(surface-np.asarray(previous[name]),
                                                     axis=1).min())
                                   for name in ("left_tip", "right_tip"))
                translation = np.asarray(previous["translation_action"])[[0, 2]]
                inward = float(-np.dot(translation, geom["normal_local_xz"]))
                aligned = (geom["position_proxy_m"] < task.success_position_threshold and
                           geom["yaw_proxy_rad"] < task.success_rotation_threshold)
                if tip_distance < zone and not aligned and inward > 0 and not previous["collision"]:
                    records.append({"episode_id": spec.identity, "timestep": t,
                        "tip_distance_m": tip_distance,
                        "inward_action_m": inward,
                        "clearance_drop_m": previous["clearance"]-current["clearance"],
                        "contact_next_step": bool(current["collision"]),
                        "position_proxy_m": geom["position_proxy_m"],
                        "yaw_proxy_rad": geom["yaw_proxy_rad"]})
    finally:
        env.close()
    safe = np.asarray([r["inward_action_m"] for r in records if
                       not r["contact_next_step"]])
    contact = np.asarray([r["inward_action_m"] for r in records if
                          r["contact_next_step"]])
    if len(safe) < 5:
        raise AssertionError("Too few safe training steps for a stable inward cap")
    cap = float(np.quantile(safe, .90))
    _csv(root / "training_near_misaligned_inward_steps.csv", records)
    result = {"seed": seed, "zone_m": zone,
        "eligible_training_steps": len(records),
        "safe_steps": len(safe), "contact_next_steps": len(contact),
        "safe_inward_p50_m": float(np.quantile(safe, .5)),
        "safe_inward_p90_m": cap,
        "contact_inward_p50_m": float(np.quantile(contact, .5)) if len(contact) else None,
        "cap_m": cap,
        "derivation": "p90 inward normal action among training steps in the training-derived near zone, misaligned by observable geometry, with no next-step contact",
        "no_heldout_input": True}
    _write(root / "cap.json", result)
    return result


def analytic_approach_guard(seed: int = 2811, split: str = "train", output: Path = OUTPUT,
                            eval_device: DevicePreference = "cpu", eval_workers: int = 1,
                            training_zone: bool = False) -> dict:
    parent = "analytic_control_train_p95" if training_zone else "analytic_control"
    root = ensure_dir(output / "approach_safety" / parent / split / f"seed{seed}")
    zone = (derive_training_approach_zone(output, eval_device)["zone_m"]
            if training_zone else None)
    results = {}
    for condition in ("static", "dynamic"):
        baseline, before = evaluate(seed, condition, "none", split, eval_device, eval_workers)
        guarded, after = evaluate(seed, condition, "approach_guard", split,
                                  eval_device, eval_workers, zone)
        if [r["episode_id"] for r in baseline] != [r["episode_id"] for r in guarded]:
            raise AssertionError("Approach guard lost EpisodeSpec pairing")
        _csv(root / f"{condition}_baseline_per_episode.csv", baseline)
        _csv(root / f"{condition}_per_episode.csv", guarded)
        _write(root / f"{condition}_baseline_summary.json", before)
        _write(root / f"{condition}_summary.json", after)
        paired = [{"episode_id": a["episode_id"],
            "baseline_success": a["success"], "guard_success": b["success"],
            "baseline_collision": a["collision"], "guard_collision": b["collision"],
            "baseline_position_error": a["position_error"],
            "guard_position_error": b["position_error"],
            "baseline_yaw_error": a["yaw_error"], "guard_yaw_error": b["yaw_error"],
            "baseline_aperture_error": a["aperture_error"],
            "guard_aperture_error": b["aperture_error"]}
            for a, b in zip(baseline, guarded, strict=True)]
        _csv(root / f"{condition}_paired_transitions.csv", paired)
        results[condition] = {"baseline": before, "approach_guard": after,
            "approach_zone_m": zone if zone is not None else .5*motion.task(seed).pregrasp_clearance,
            "approach_blocked_steps": sum(r["approach_blocked_steps"] for r in guarded),
            "collision_fixed": [r["episode_id"] for r in paired if
                r["baseline_collision"] and not r["guard_collision"]],
            "new_collision": [r["episode_id"] for r in paired if
                not r["baseline_collision"] and r["guard_collision"]],
            "success_gained": [r["episode_id"] for r in paired if
                not r["baseline_success"] and r["guard_success"]],
            "success_lost": [r["episode_id"] for r in paired if
                r["baseline_success"] and not r["guard_success"]]}
    _write(root / "summary.json", results)
    return results


def analytic_approach_cap(seed: int = 2811, split: str = "validation", output: Path = OUTPUT,
                          eval_device: DevicePreference = "cpu", eval_workers: int = 1) -> dict:
    config = derive_training_inward_cap(output, eval_device)
    root = ensure_dir(output / "approach_safety/analytic_cap" / split / f"seed{seed}")
    results = {}
    for condition in ("static", "dynamic"):
        baseline, before = evaluate(seed, condition, "none", split, eval_device, eval_workers)
        limited, after = evaluate(seed, condition, "approach_cap", split,
            eval_device, eval_workers, config["zone_m"], config["cap_m"])
        if [r["episode_id"] for r in baseline] != [r["episode_id"] for r in limited]:
            raise AssertionError("Approach cap lost EpisodeSpec pairing")
        _csv(root / f"{condition}_baseline_per_episode.csv", baseline)
        _csv(root / f"{condition}_per_episode.csv", limited)
        _write(root / f"{condition}_baseline_summary.json", before)
        _write(root / f"{condition}_summary.json", after)
        paired = [{"episode_id": a["episode_id"],
            "baseline_success": a["success"], "cap_success": b["success"],
            "baseline_collision": a["collision"], "cap_collision": b["collision"],
            "baseline_position_error": a["position_error"],
            "cap_position_error": b["position_error"],
            "baseline_yaw_error": a["yaw_error"], "cap_yaw_error": b["yaw_error"],
            "baseline_aperture_error": a["aperture_error"],
            "cap_aperture_error": b["aperture_error"]}
            for a, b in zip(baseline, limited, strict=True)]
        _csv(root / f"{condition}_paired_transitions.csv", paired)
        results[condition] = {"baseline": before, "approach_cap": after,
            "zone_m": config["zone_m"], "inward_cap_m": config["cap_m"],
            "capped_steps": sum(r["approach_blocked_steps"] for r in limited),
            "collision_fixed": [r["episode_id"] for r in paired if
                r["baseline_collision"] and not r["cap_collision"]],
            "new_collision": [r["episode_id"] for r in paired if
                not r["baseline_collision"] and r["cap_collision"]],
            "success_gained": [r["episode_id"] for r in paired if
                not r["baseline_success"] and r["cap_success"]],
            "success_lost": [r["episode_id"] for r in paired if
                r["baseline_success"] and not r["cap_success"]]}
    _write(root / "summary.json", results)
    return results


def orientation_first_diagnostic(seed: int = 2811, split: str = "train",
                                 output: Path = OUTPUT,
                                 eval_device: DevicePreference = "cpu",
                                 eval_workers: int = 1,
                                 near_mixed: bool = False) -> dict:
    mode = "near_mixed" if near_mixed else "orientation_first"
    zone = derive_training_approach_zone(output, eval_device)["zone_m"] if near_mixed else None
    root = ensure_dir(output / "approach_safety" / mode / split / f"seed{seed}")
    results = {}
    for condition in ("static", "dynamic"):
        baseline, before = evaluate(seed, condition, "none", split,
                                    eval_device, eval_workers)
        guarded, after = evaluate(seed, condition, mode, split,
                                  eval_device, eval_workers, zone)
        if [r["episode_id"] for r in baseline] != [r["episode_id"] for r in guarded]:
            raise AssertionError("Orientation-first comparison lost EpisodeSpec pairing")
        _csv(root / f"{condition}_baseline_per_episode.csv", baseline)
        _csv(root / f"{condition}_per_episode.csv", guarded)
        _write(root / f"{condition}_baseline_summary.json", before)
        _write(root / f"{condition}_summary.json", after)
        pairs = [{"episode_id": a["episode_id"],
            "baseline_success": a["success"], "intervention_success": b["success"],
            "baseline_collision": a["collision"], "intervention_collision": b["collision"],
            "baseline_position_error": a["position_error"],
            "intervention_position_error": b["position_error"],
            "baseline_yaw_error": a["yaw_error"],
            "intervention_yaw_error": b["yaw_error"],
            "baseline_aperture_error": a["aperture_error"],
            "intervention_aperture_error": b["aperture_error"]}
            for a, b in zip(baseline, guarded, strict=True)]
        _csv(root / f"{condition}_paired_transitions.csv", pairs)
        results[condition] = {"baseline": before, mode: after,
            "approach_zone_m": zone,
            "blocked_inward_steps": sum(r["approach_blocked_steps"] for r in guarded),
            "collision_fixed": [r["episode_id"] for r in pairs if
                r["baseline_collision"] and not r["intervention_collision"]],
            "new_collision": [r["episode_id"] for r in pairs if
                not r["baseline_collision"] and r["intervention_collision"]],
            "success_gained": [r["episode_id"] for r in pairs if
                not r["baseline_success"] and r["intervention_success"]],
            "success_lost": [r["episode_id"] for r in pairs if
                r["baseline_success"] and not r["intervention_success"]]}
    _write(root / "summary.json", results)
    return results


def one_step_collision_counterfactual(output: Path = OUTPUT,
                                      eval_device: DevicePreference = "cpu") -> dict:
    """Replay identical pre-contact states, changing exactly one executed action."""
    root = ensure_dir(output / "collision_audit/one_step_counterfactual")
    events = list(csv.DictReader((output /
        "collision_taxonomy/episode_classification.csv").open()))
    device = select_device(eval_device)
    rows = []
    variants = ("baseline", "no_closing", "no_inward_translation", "no_yaw",
                "no_inward_and_no_yaw")
    for event in events:
        seed = int(event["seed"])
        task = motion.task(seed, eval_device=eval_device)
        onset = int(event["onset_timestep"])
        spec = {s.identity: s for s in dynamic.dynamic_specs(seed, task, 16,
            dynamic.SELECTED_STEP_SPEEDS_M, "heldout")}[event["episode_id"]]
        for variant in variants:
            controller = _final_controller(seed, device)
            env = sf.make_env(task, seed+864_211)
            previous = torch.zeros(1, 5, device=device)
            intervention = {}
            try:
                sf._reset_surface_env(env, spec.initial)
                for t in range(onset):
                    shape = dynamic._shape_at(env, spec, t)
                    state, points, _ = sf.observation_inputs(env, shape, 32,
                        sf._stable_seed(spec.initial.sample_identity))
                    state_t, points_t, topology = rf._graph_tensors(state, points,
                                                                     task, device)
                    action_t, _, _, _ = controller.step(state_t, points_t, previous,
                                                         topology=topology)
                    action = rf.clipped_action(action_t[0].detach().cpu().numpy(), task)
                    if t == onset-1:
                        original = action.copy()
                        geometry = observed_geometry(points, task.pregrasp_clearance)
                        normal = geometry["normal_local_xz"]
                        inward = float(-np.dot(action[[0, 2]], normal))
                        if variant == "no_closing":
                            action[4] = max(action[4], 0.0)
                        elif variant in ("no_inward_translation", "no_inward_and_no_yaw"):
                            if inward > 0:
                                action[[0, 2]] += inward*normal
                        if variant in ("no_yaw", "no_inward_and_no_yaw"):
                            action[3] = 0.0
                        intervention = {"pre_contact_original_action": original.tolist(),
                            "pre_contact_executed_action": action.tolist(),
                            "visible_inward_component_m": inward,
                            "visible_tip_surface_distance_m": min(float(np.linalg.norm(
                                sf.sample_surface_points_world(shape, 32,
                                  sf._stable_seed(spec.initial.sample_identity))-
                                np.asarray(env.data.geom_xpos[env._geom_id(geom)]), axis=1).min())
                                for geom in ("thumbtip1", "fingertip1"))}
                    sf.apply_local_action(env, action, task)
                    previous = torch.as_tensor(action, dtype=torch.float32,
                                               device=device).reshape(1, 5)
                onset_shape = dynamic._shape_at(env, spec, onset)
                contacts, clearance = sf._surface_distance_metrics(env, onset_shape)
                ee = env.robot_observation().ee_position.numpy().astype(np.float64)
                target = sf._surface_target(onset_shape, ee, task.pregrasp_clearance)
                errors = sf._state_errors(env, onset_shape, task, target)
                rows.append({"episode_id": spec.identity, "seed": seed,
                    "onset_timestep": onset, "variant": variant,
                    "contact_at_original_onset": bool(contacts),
                    "clearance_m": float(clearance),
                    "position_error_m": float(errors["position_error"]),
                    "yaw_error_rad": float(errors["orientation_error"]),
                    "aperture_error_m": float(errors["gripper_width_error"]),
                    "contacting_robot_geoms": sorted(set(
                        motion._geom_name(env, c["geom2"]) for c in contacts)),
                    "contacting_surface_geoms": sorted(set(
                        motion._geom_name(env, c["geom1"]) for c in contacts)),
                    **intervention})
            finally:
                env.close()
    _csv(root / "one_step_actions.csv", rows)
    groups = {}
    for event in events:
        identity = event["episode_id"]
        group = {r["variant"]: r for r in rows if r["episode_id"] == identity}
        if not group["baseline"]["contact_at_original_onset"]:
            raise AssertionError(f"One-step baseline did not reproduce contact: {identity}")
        groups[identity] = {variant: bool(group[variant]["contact_at_original_onset"])
                            for variant in variants}
    result = {"episodes": len(events), "variants": variants,
        "contact_remaining_by_variant": {variant: sum(groups[identity][variant]
            for identity in groups) for variant in variants},
        "episode_outcomes": groups,
        "interpretation_limit": "One-step intervention isolates immediate contact at the original onset; it does not measure full-policy success after the changed action."}
    _write(root / "summary.json", result)
    return result


def final_diagnostics(seed: int = 2811, output: Path = OUTPUT,
                      eval_device: DevicePreference = "cpu") -> dict:
    """Full fixed-horizon traces for the frozen near-mixed coordination rule."""
    root = ensure_dir(output / f"seed{seed}/final_diagnostics")
    zone = derive_training_approach_zone(output, eval_device)["zone_m"]
    device = select_device(eval_device)
    task = motion.task(seed, eval_device=eval_device)
    controller = ContactController(_final_controller(seed, device), "near_mixed", zone)
    env = sf.make_env(task, seed+864_211)
    traces, outcomes, events, windows = [], [], [], []
    try:
        for condition in ("static", "dynamic"):
            specs = (dynamic.dynamic_specs(seed, task, 16,
                dynamic.SELECTED_STEP_SPEEDS_M, "heldout") if condition == "dynamic"
                else sf.sample_episode_specs(16, seed+60_000, task, "iid"))
            for spec in specs:
                raw, outcome = motion.fixed_horizon_trace(env, spec, task,
                    controller, device, condition, "near_mixed_final")
                episode = spec.identity if condition == "dynamic" else spec.sample_identity
                augmented = []
                for i, row in enumerate(raw):
                    t = int(row["timestep"])
                    shape = (dynamic.moving_shape(spec, t) if condition == "dynamic"
                             else spec.shape)
                    ee = np.asarray(row["EE_position"], dtype=np.float64)
                    points = sf.sample_surface_points_world(shape, 32,
                        sf._stable_seed((spec.initial if condition == "dynamic"
                                         else spec).sample_identity))
                    tip_distance = min(float(np.linalg.norm(points-np.asarray(row[name]),
                                                 axis=1).min())
                                       for name in ("left_tip", "right_tip"))
                    previous = raw[i-1] if i else None
                    contacts = row["contacts"]
                    entry = {"episode_id": episode, "seed": seed,
                        "condition": condition, "timestep": t,
                        "controller": "near_mixed_final",
                        "EE_position": row["EE_position"], "EE_yaw": row["EE_yaw"],
                        "target_position": row["target_position"],
                        "target_yaw": row["target_yaw"],
                        "position_error": row["position_error"],
                        "yaw_error": row["yaw_error"],
                        "opening": row["opening_width_m"],
                        "target_opening": sf._surface_target(shape, ee,
                            task.pregrasp_clearance)[3],
                        "left_tip": row["left_tip"], "right_tip": row["right_tip"],
                        "translation_action": row["translation_action"],
                        "yaw_action": row["yaw_action"],
                        "requested_gripper_action": row["requested_gripper_action"],
                        "executed_gripper_action": row["gripper_action"],
                        "clearance": row["clearance"],
                        "clearance_delta": (row["clearance"]-previous["clearance"]
                                            if previous else None),
                        "approach_velocity_proxy": (previous["clearance"]-row["clearance"]
                                                     if previous else None),
                        "opening_change_from_previous_m": (row["opening_width_m"]-
                            previous["opening_width_m"] if previous else None),
                        "fingertip_distance_to_observed_surface_m": tip_distance,
                        "collision": row["collision"],
                        "collision_robot_geom": sorted(set(c["robot_geom"] for c in contacts)),
                        "collision_surface_geom": sorted(set(c["surface_geom"] for c in contacts)),
                        "coordination_diagnostic": row["coordination_diagnostic"],
                        "inference_latency_ms": row["inference_latency_ms"]}
                    augmented.append(entry)
                    traces.append(entry)
                    if row["collision"] and (previous is None or not previous["collision"]):
                        events.append({"episode_id": episode, "seed": seed,
                            "condition": condition, "onset_timestep": t,
                            "robot_geoms": entry["collision_robot_geom"],
                            "surface_geoms": entry["collision_surface_geom"],
                            "prior_position_error_m": previous["position_error"] if previous else None,
                            "prior_yaw_error_rad": previous["yaw_error"] if previous else None,
                            "prior_opening_m": previous["opening_width_m"] if previous else None,
                            "prior_clearance_m": previous["clearance"] if previous else None,
                            "onset_clearance_m": row["clearance"],
                            "prior_translation_action": previous["translation_action"] if previous else None,
                            "prior_yaw_action": previous["yaw_action"] if previous else None,
                            "prior_gripper_action": previous["gripper_action"] if previous else None,
                            "clearance_drop_m": entry["approach_velocity_proxy"]})
                        windows.extend([{**r, "onset_timestep": t,
                            "relative_to_onset": r["timestep"]-t}
                            for r in augmented[max(0, len(augmented)-6):]])
                outcomes.append({"episode_id": episode, "seed": seed,
                                 "condition": condition, **outcome})
    finally:
        env.close()
    _csv(root / "fixed_horizon_traces.csv", traces)
    _csv(root / "fixed_horizon_outcomes.csv", outcomes)
    if events:
        _csv(root / "collision_onsets.csv", events)
        _csv(root / "five_step_windows.csv", windows)
    official = {r["episode_id"]: r for r in csv.DictReader((output /
        f"approach_safety/near_mixed/heldout/seed{seed}/dynamic_per_episode.csv").open())}
    for row in [r for r in outcomes if r["condition"] == "dynamic"]:
        source = official[row["episode_id"]]
        if row["collision"] != (source["collision"] == "True") or \
           abs(row["final_position_error"]-float(source["position_error"])) > 1e-5:
            raise AssertionError("Final diagnostic trace differs from official dynamic rollout")
    result = {"seed": seed, "episodes_per_condition": 16,
        "timesteps": len(traces), "contact_onset_events": len(events),
        "dynamic_collision_episodes": sum(r["collision"] for r in outcomes if
                                          r["condition"] == "dynamic"),
        "dynamic_official_parity": True,
        "note": "Dynamic fixed horizon matches official evaluation. Static diagnostic continues after official first-success termination; target_opening and true errors are diagnostic only."}
    _write(root / "summary.json", result)
    return result


def aggregate_final(output: Path = OUTPUT) -> dict:
    root = ensure_dir(output / "aggregate")
    task = motion.task(2811)
    totals, episode_rows, transitions, failures = {}, [], [], []
    for condition in ("static", "dynamic"):
        totals[condition] = {}
        for controller in ("baseline", "near_mixed"):
            combined = []
            for seed in SEEDS:
                filename = (output / f"reproduction/seed{seed}_{condition}_per_episode.csv"
                    if controller == "baseline" else output /
                    f"approach_safety/near_mixed/heldout/seed{seed}/{condition}_per_episode.csv")
                rows = list(csv.DictReader(filename.open()))
                for source in rows:
                    row = {"episode_id": source["episode_id"], "seed": seed,
                        "condition": condition, "controller": controller,
                        "success": source["success"] == "True",
                        "collision": source["collision"] == "True",
                        "position_error": float(source["position_error"]),
                        "yaw_error": float(source["yaw_error"]),
                        "aperture_error": float(source["aperture_error"]),
                        "minimum_clearance": float(source["minimum_clearance"]),
                        "inference_latency_ms": float(source["inference_latency_ms"]),
                        "convergence_time": source.get("convergence_time", "")}
                    episode_rows.append(row)
                    combined.append(row)
                    if controller == "near_mixed":
                        flags = {"position_fail": row["position_error"] >=
                                 task.success_position_threshold,
                                 "yaw_fail": row["yaw_error"] >=
                                 task.success_rotation_threshold,
                                 "aperture_fail": row["aperture_error"] >=
                                 task.success_opening_threshold,
                                 "collision_fail": row["collision"]}
                        failures.append({"episode_id": row["episode_id"], "seed": seed,
                            "condition": condition, **flags,
                            "multiple_failures": sum(flags.values()) > 1,
                            "success": row["success"]})
            totals[condition][controller] = motion._summary(combined, task)
        for seed in SEEDS:
            before = {r["episode_id"]: r for r in episode_rows if r["seed"] == seed and
                      r["condition"] == condition and r["controller"] == "baseline"}
            after = {r["episode_id"]: r for r in episode_rows if r["seed"] == seed and
                     r["condition"] == condition and r["controller"] == "near_mixed"}
            if set(before) != set(after):
                raise AssertionError("Final aggregate compared unmatched EpisodeSpecs")
            for identity in before:
                a, b = before[identity], after[identity]
                transitions.append({"episode_id": identity, "seed": seed,
                    "condition": condition,
                    "baseline_success": a["success"], "final_success": b["success"],
                    "baseline_collision": a["collision"], "final_collision": b["collision"],
                    "baseline_position_error_m": a["position_error"],
                    "final_position_error_m": b["position_error"],
                    "baseline_yaw_error_rad": a["yaw_error"],
                    "final_yaw_error_rad": b["yaw_error"],
                    "baseline_aperture_error_m": a["aperture_error"],
                    "final_aperture_error_m": b["aperture_error"]})
    _csv(root / "full_controller_per_episode.csv", episode_rows)
    _csv(root / "paired_episode_transitions.csv", transitions)
    _csv(root / "final_failure_decomposition.csv", failures)
    pair_summary = {}
    for condition in ("static", "dynamic"):
        subset = [r for r in transitions if r["condition"] == condition]
        pair_summary[condition] = {"success_gained": [r["episode_id"] for r in subset if
            not r["baseline_success"] and r["final_success"]],
            "success_lost": [r["episode_id"] for r in subset if
                r["baseline_success"] and not r["final_success"]],
            "collision_fixed": [r["episode_id"] for r in subset if
                r["baseline_collision"] and not r["final_collision"]],
            "new_collision": [r["episode_id"] for r in subset if
                not r["baseline_collision"] and r["final_collision"]]}
    # Diagnose every non-collision position failure using its 20-step trace.
    position_root = ensure_dir(output / "position_overshoot")
    position_rows = []
    for seed in SEEDS:
        trace = {}
        for r in csv.DictReader((output /
            f"seed{seed}/final_diagnostics/fixed_horizon_traces.csv").open()):
            if r["condition"] == "dynamic":
                trace.setdefault(r["episode_id"], []).append(r)
        for failure in [r for r in failures if r["seed"] == seed and
                        r["condition"] == "dynamic" and r["position_fail"]]:
            rows = trace[failure["episode_id"]]
            errors = np.asarray([float(r["position_error"]) for r in rows])
            best = int(np.argmin(errors))
            if failure["collision_fail"]:
                category = "collision_overlap"
            elif errors[best] < task.success_position_threshold:
                category = "reached_then_left"
            else:
                category = "never_reached"
            position_rows.append({"episode_id": failure["episode_id"],
                "seed": seed, "collision": failure["collision_fail"],
                "category": category, "best_timestep": best,
                "best_position_error_m": float(errors[best]),
                "final_position_error_m": float(next(r["position_error"] for r in
                    episode_rows if r["episode_id"] == failure["episode_id"] and
                    r["controller"] == "near_mixed" and r["condition"] == "dynamic")),
                "post_best_translation_actions": [r["translation_action"] for r in rows[best+1:]],
                "post_best_yaw_actions": [float(r["yaw_action"]) for r in rows[best+1:]],
                "post_best_gripper_actions": [float(r["executed_gripper_action"]) for r in rows[best+1:]]})
    _csv(position_root / "position_failure_trajectories.csv", position_rows)
    position_summary = {"position_failure_episodes": len(position_rows),
        "non_collision_position_failures": sum(not r["collision"] for r in position_rows),
        "reached_then_left_non_collision": sum(r["category"] == "reached_then_left"
                                               for r in position_rows),
        "stabilization_rule_justified": False,
        "reason": "Only one non-collision position failure; no common repeated reach-then-leave pattern."}
    _write(position_root / "summary.json", position_summary)
    provenance = {seed: {"base_ff_sha256": motion.sha(residual.BASE_CHECKPOINT),
        "adapted_motion_sha256": motion.sha(motion.paths(seed)["updated_motion"]),
        "aperture_branch_sha256": motion.sha(motion.paths(seed)["aperture"]),
        "yaw_readout_sha256": motion.sha(motion.OUTPUT /
            f"seed{seed}/yaw_spatial_readout/selected.pt")} for seed in SEEDS}
    result = {"totals": totals, "paired": pair_summary,
        "intervention": {"name": "near_mixed",
            "condition": "both observable position and yaw PCA proxies exceed existing task tolerances, and observable fingertip-to-surface-point distance is within the training-derived approach zone",
            "modified_action": "cancel only inward normal translation; leave tangent, retreat, yaw, and gripper action unchanged",
            "approach_zone_m": derive_training_approach_zone(output)["zone_m"]},
        "checkpoint_provenance": provenance,
        "position_overshoot": position_summary}
    _write(root / "summary.json", result)
    return result


def confirmatory_evaluation(output: Path = OUTPUT,
                            eval_device: DevicePreference = "cpu",
                            eval_workers: int = 1) -> dict:
    """Frozen rule on disjoint new EpisodeSpecs from the unchanged task distribution."""
    root = ensure_dir(output / "aggregate/confirmatory_new_episodes")
    zone = derive_training_approach_zone(output, eval_device)["zone_m"]
    summaries, pairs = {}, []
    for seed in SEEDS:
        task = motion.task(seed, eval_device=eval_device)
        summaries[seed] = {}
        for condition in ("static", "dynamic"):
            if condition == "static":
                specs = sf.sample_episode_specs(16, seed+70_000, task,
                                                "iid_contact_confirm")
            else:
                bases = sf.sample_episode_specs(16, seed+120_000, task,
                                                "dynamic_contact_confirm")
                directions = tuple(dynamic.DIRECTIONS)
                speeds = dynamic.SELECTED_STEP_SPEEDS_M
                specs = [dynamic.DynamicSpec(base, directions[i % 4],
                    speeds[(i//4) % len(speeds)]) for i, base in enumerate(bases)]
            ids = [spec.identity if condition == "dynamic" else spec.sample_identity
                   for spec in specs]
            existing = {r["episode_id"] for r in csv.DictReader((output /
                f"reproduction/seed{seed}_{condition}_per_episode.csv").open())}
            if set(ids) & existing:
                raise AssertionError("Confirmatory EpisodeSpecs overlap previous held-out set")
            result = {}
            for mode in ("none", "near_mixed"):
                payloads = [(seed, condition, i, spec, mode, eval_device,
                             zone if mode == "near_mixed" else None, None)
                            for i, spec in enumerate(specs)]
                if eval_workers > 1:
                    with ProcessPoolExecutor(max_workers=eval_workers,
                            mp_context=mp.get_context("spawn")) as pool:
                        responses = list(pool.map(_eval_worker, payloads))
                else:
                    responses = [_eval_worker(item) for item in payloads]
                rows = [row for _, row in sorted(responses, key=lambda pair: pair[0])]
                result[mode] = rows
                _csv(root / f"seed{seed}_{condition}_{mode}_per_episode.csv", rows)
                summaries[seed][condition] = summaries[seed].get(condition, {})
                summaries[seed][condition][mode] = motion._summary(rows, task)
            for a, b in zip(result["none"], result["near_mixed"], strict=True):
                if a["episode_id"] != b["episode_id"]:
                    raise AssertionError("Confirmatory evaluation lost pairing")
                pairs.append({"episode_id": a["episode_id"], "seed": seed,
                    "condition": condition, "baseline_success": a["success"],
                    "final_success": b["success"],
                    "baseline_collision": a["collision"],
                    "final_collision": b["collision"],
                    "baseline_position_error": a["position_error"],
                    "final_position_error": b["position_error"],
                    "baseline_yaw_error": a["yaw_error"],
                    "final_yaw_error": b["yaw_error"],
                    "baseline_aperture_error": a["aperture_error"],
                    "final_aperture_error": b["aperture_error"]})
    _csv(root / "paired_transitions.csv", pairs)
    aggregate = {}
    for condition in ("static", "dynamic"):
        aggregate[condition] = {}
        for mode in ("none", "near_mixed"):
            aggregate[condition][mode] = {key: sum(summaries[seed][condition][mode][key]
                for seed in SEEDS) for key in ("success", "position_fail", "yaw_fail",
                                               "aperture_fail", "collision_fail")}
        subset = [r for r in pairs if r["condition"] == condition]
        aggregate[condition]["paired"] = {
            "success_gained": [r["episode_id"] for r in subset if
                not r["baseline_success"] and r["final_success"]],
            "success_lost": [r["episode_id"] for r in subset if
                r["baseline_success"] and not r["final_success"]],
            "collision_fixed": [r["episode_id"] for r in subset if
                r["baseline_collision"] and not r["final_collision"]],
            "new_collision": [r["episode_id"] for r in subset if
                not r["baseline_collision"] and r["final_collision"]]}
    result = {"new_episode_seed_offsets": {"static": 70_000, "dynamic": 120_000},
        "same_shape_sampler_and_dynamic_speed_direction_distribution": True,
        "approach_zone_m": zone, "per_seed": summaries, "aggregate": aggregate}
    _write(root / "summary.json", result)
    return result


def classify_final_contacts(output: Path = OUTPUT) -> dict:
    root = ensure_dir(output / "collision_taxonomy/final_controller")
    rows = []
    for seed in SEEDS:
        task = motion.task(seed)
        traces = {(r["condition"], r["episode_id"], int(r["timestep"])): r
                  for r in csv.DictReader((output /
                    f"seed{seed}/final_diagnostics/fixed_horizon_traces.csv").open())}
        dynamic_specs = {spec.identity: spec for spec in dynamic.dynamic_specs(
            seed, task, 16, dynamic.SELECTED_STEP_SPEEDS_M, "heldout")}
        static_specs = {spec.sample_identity: spec for spec in sf.sample_episode_specs(
            16, seed+60_000, task, "iid")}
        events = list(csv.DictReader((output /
            f"seed{seed}/final_diagnostics/collision_onsets.csv").open()))
        for event in events:
            condition = event["condition"]
            identity = event["episode_id"]
            onset = int(event["onset_timestep"])
            previous = traces[(condition, identity, onset-1)]
            current = traces[(condition, identity, onset)]
            spec = dynamic_specs[identity] if condition == "dynamic" else static_specs[identity]
            ee = np.asarray(ast.literal_eval(previous["EE_position"]))
            points = _observed_points(seed, spec, onset-1, ee,
                float(previous["EE_yaw"]), condition)
            normal = observed_geometry(points, task.pregrasp_clearance)["normal_local_xz"]
            translation = np.asarray(ast.literal_eval(previous["translation_action"]))[[0, 2]]
            inward = float(-np.dot(translation, normal))
            pos = float(previous["position_error"])
            yaw = float(previous["yaw_error"])
            opening_change = float(current["opening_change_from_previous_m"])
            clearance_drop = float(current["approach_velocity_proxy"])
            if pos >= task.success_position_threshold and yaw >= task.success_rotation_threshold:
                label = "D_mixed_position_yaw"
            elif opening_change < -.001 and (pos >= task.success_position_threshold or
                                               yaw >= task.success_rotation_threshold):
                label = "A_premature_closing"
            elif yaw >= task.success_rotation_threshold:
                label = "B_orientation_approach"
            elif inward > 0 and clearance_drop > 0 and float(current["clearance"]) <= 0:
                label = "C_translation_overshoot"
            else:
                label = "E_ambiguous"
            rows.append({"episode_id": identity, "seed": seed,
                "condition": condition, "onset_timestep": onset,
                "taxonomy": label, "robot_geoms": event["robot_geoms"],
                "surface_geoms": event["surface_geoms"],
                "evidence": (f"prior position={pos:.4f}m, yaw={yaw:.3f}rad, "
                             f"inward action={inward:.4f}m, clearance drop={clearance_drop:.4f}m, "
                             f"opening change={opening_change:.8f}m"),
                "prior_position_error_m": pos, "prior_yaw_error_rad": yaw,
                "prior_opening_m": float(previous["opening"]),
                "prior_clearance_m": float(previous["clearance"]),
                "onset_clearance_m": float(current["clearance"]),
                "prior_translation_action": previous["translation_action"],
                "prior_yaw_action": previous["yaw_action"],
                "prior_gripper_action": previous["executed_gripper_action"],
                "prior_inward_action_m": inward,
                "opening_change_m": opening_change})
    _csv(root / "event_classification.csv", rows)
    result = {"events": len(rows),
        "dynamic_events": sum(r["condition"] == "dynamic" for r in rows),
        "static_fixed_horizon_events": sum(r["condition"] == "static" for r in rows),
        "taxonomy": {label: sum(r["taxonomy"] == label for r in rows)
                     for label in sorted(set(r["taxonomy"] for r in rows))},
        "note": "Static fixed-horizon collision can occur after official early-success termination; labels are descriptive and ambiguous remains allowed."}
    _write(root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("reproduce", "collision-audit",
                                          "classify", "close-gate",
                                          "clearance-analysis", "approach-guard",
                                          "derive-approach-zone", "approach-guard-train-p95",
                                          "derive-inward-cap", "approach-cap",
                                          "one-step-counterfactual", "orientation-first",
                                          "near-mixed", "final-traces", "aggregate",
                                          "confirmatory", "final-contact-taxonomy"))
    parser.add_argument("--seed", type=int, default=2811)
    parser.add_argument("--split", choices=("train", "validation", "heldout"), default="train")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "reproduce":
        reproduce(args.output, args.eval_device, args.eval_workers)
    elif args.phase == "collision-audit":
        collision_audit(args.output)
    elif args.phase == "classify":
        classify_and_analyze(args.output)
    elif args.phase == "close-gate":
        analytic_close_gate(args.seed, args.split, args.output, args.eval_device,
                            args.eval_workers)
    elif args.phase == "clearance-analysis":
        clearance_analysis(args.output)
    elif args.phase == "derive-approach-zone":
        derive_training_approach_zone(args.output, args.eval_device)
    elif args.phase == "approach-guard-train-p95":
        analytic_approach_guard(args.seed, args.split, args.output, args.eval_device,
                                args.eval_workers, training_zone=True)
    elif args.phase == "derive-inward-cap":
        derive_training_inward_cap(args.output, args.eval_device)
    elif args.phase == "approach-cap":
        analytic_approach_cap(args.seed, args.split, args.output, args.eval_device,
                              args.eval_workers)
    elif args.phase == "one-step-counterfactual":
        one_step_collision_counterfactual(args.output, args.eval_device)
    elif args.phase == "orientation-first":
        orientation_first_diagnostic(args.seed, args.split, args.output,
                                     args.eval_device, args.eval_workers)
    elif args.phase == "near-mixed":
        orientation_first_diagnostic(args.seed, args.split, args.output,
                                     args.eval_device, args.eval_workers, near_mixed=True)
    elif args.phase == "final-traces":
        final_diagnostics(args.seed, args.output, args.eval_device)
    elif args.phase == "aggregate":
        aggregate_final(args.output)
    elif args.phase == "confirmatory":
        confirmatory_evaluation(args.output, args.eval_device, args.eval_workers)
    elif args.phase == "final-contact-taxonomy":
        classify_final_contacts(args.output)
    else:
        analytic_approach_guard(args.seed, args.split, args.output, args.eval_device,
                                args.eval_workers)


if __name__ == "__main__":
    main()
