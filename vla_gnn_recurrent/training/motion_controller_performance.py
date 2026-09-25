"""Performance-first motion adaptation with the validated aperture branch fixed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from torch import nn

from vla_gnn_recurrent.training import aperture_spatial_controller as spatial
from vla_gnn_recurrent.training import dynamic_visual_delay_feasibility as dynamic
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import residual_recurrent_feasibility as residual
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training.generalized_geometric_graph import wrap_angle
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


OUTPUT = Path("artifacts/motion_controller_performance_v1")
SEEDS = (2811, 2812, 2813)
MOTION_UPDATES = 400
VALIDATION_INTERVAL = 25
TRAIN_POLICY_EPISODES = 72


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task(seed: int, device: DevicePreference = "cpu",
         eval_device: DevicePreference = "cpu") -> sf.SurfaceFeasibilityConfig:
    return spatial._task(seed, device, eval_device)


def paths(seed: int, output: Path = OUTPUT) -> dict[str, Path]:
    if seed not in SEEDS:
        raise ValueError(seed)
    motion = (spatial.STAGE_B_5D if seed == 2811 else spatial.OUTPUT /
              f"replication/seed{seed}/residual_5d_mlp/stage_b/selected.pt")
    aperture = (spatial.OUTPUT / "separate_aperture_branch/selected.pt" if seed == 2811
                else spatial.OUTPUT / f"replication/seed{seed}/separate_aperture_branch/selected.pt")
    return {"motion": motion, "aperture": aperture,
            "updated_motion": output / f"seed{seed}/onpolicy_motion_retraining/selected.pt"}


def gripper_bound() -> float:
    audit = json.loads((spatial.PRIOR / "audit/gripper_residual_bound.json").read_text())
    return min(.25, float(audit["oracle_minus_ff_action_abs_p95_fraction"]))


def load_combined(seed: int, device: torch.device, motion_checkpoint: Path | None = None,
                  output: Path = OUTPUT, yaw_bound: float | None = None) -> spatial.SpanBranchController:
    task_config = task(seed)
    p = paths(seed, output)
    checkpoint = motion_checkpoint or p["motion"]
    design = replace(residual.ResidualConfig(), seed=seed, corrected_dimensions=5)
    motion = residual.load_controller("mlp", task_config, design, device, checkpoint)
    if yaw_bound is not None:
        if not 0 < yaw_bound <= task_config.max_delta_rotation:
            raise ValueError("Yaw residual bound must lie within the native action limit")
        motion.residual_bounds[3] = yaw_bound
    return spatial.SpanBranchController(motion, p["aperture"], task_config, device, gripper_bound())


def _summary(rows: list[dict], task_config: sf.SurfaceFeasibilityConfig) -> dict:
    return {"episodes": len(rows),
            "success": int(sum(r["success"] for r in rows)),
            "position_fail": int(sum(r["position_error"] >= task_config.success_position_threshold for r in rows)),
            "yaw_fail": int(sum(r["yaw_error"] >= task_config.success_rotation_threshold for r in rows)),
            "aperture_fail": int(sum(r["aperture_error"] >= task_config.success_opening_threshold for r in rows)),
            "collision_fail": int(sum(r["collision"] for r in rows)),
            "multiple_failures": int(sum(sum((r["position_error"] >= task_config.success_position_threshold,
                                                r["yaw_error"] >= task_config.success_rotation_threshold,
                                                r["aperture_error"] >= task_config.success_opening_threshold,
                                                r["collision"])) > 1 for r in rows)),
            "final_position_mean_m": float(np.mean([r["position_error"] for r in rows])),
            "final_yaw_mean_rad": float(np.mean([r["yaw_error"] for r in rows])),
            "final_aperture_mean_m": float(np.mean([r["aperture_error"] for r in rows])),
            "minimum_clearance_mean_m": float(np.mean([r["minimum_clearance"] for r in rows])),
            "inference_latency_mean_ms": float(np.mean([r["inference_latency_ms"] for r in rows]))}


def _official_worker(payload: tuple) -> tuple[int, dict]:
    seed, condition, index, spec, motion_path, output, eval_device_preference, yaw_bound, yaw_head_path = payload
    device = select_device(eval_device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task_config = task(seed, eval_device=eval_device_preference)
    controller = load_combined(seed, device, motion_path, output, yaw_bound)
    if yaw_head_path is not None:
        controller = YawSpatialController(controller, yaw_head_path, device)
    env = sf.make_env(task_config, seed + 864_211)
    try:
        if condition == "dynamic":
            row, trace = dynamic.rollout(env, spec, task_config, device,
                                         "residual_mlp", controller, 0)
            episode = spec.identity
        else:
            row, trace = residual.rollout_residual(env, spec, controller,
                task_config, rf.TemporalConfig(), device)
            episode = spec.sample_identity
        if any(step.get("oracle_action") is not None for step in trace):
            raise AssertionError("Evaluation invoked scripted expert")
        return index, {"episode_id": episode, "seed": seed, "condition": condition,
                       "success": bool(row["success"]), "collision": bool(row["collision"]),
                       "position_error": float(row["final_tracking_error" if condition == "dynamic"
                                                    else "final_position_error"]),
                       "yaw_error": float(row["final_yaw_error" if condition == "dynamic"
                                               else "final_orientation_error"]),
                       "aperture_error": float(row["final_aperture_error" if condition == "dynamic"
                                                    else "final_gripper_width_error"]),
                       "minimum_clearance": float(row["minimum_safe_clearance"]),
                       "inference_latency_ms": float(row.get("mean_inference_ms") or
                           np.mean([step.get("inference_ms") or 0 for step in trace]))}
    finally:
        env.close()


def official_evaluate(seed: int, condition: str, output: Path = OUTPUT,
                      eval_device_preference: DevicePreference = "cpu", eval_workers: int = 1,
                      motion_checkpoint: Path | None = None,
                      yaw_bound: float | None = None,
                      yaw_head_checkpoint: Path | None = None) -> tuple[list[dict], dict]:
    if eval_workers < 1:
        raise ValueError("eval_workers must be positive")
    task_config = task(seed, eval_device=eval_device_preference)
    specs = (dynamic.dynamic_specs(seed, task_config, 16,
             dynamic.SELECTED_STEP_SPEEDS_M, "heldout") if condition == "dynamic" else
             sf.sample_episode_specs(16, seed + 60_000, task_config, "iid"))
    payloads = [(seed, condition, i, spec, motion_checkpoint, output,
                 eval_device_preference, yaw_bound, yaw_head_checkpoint) for i, spec in enumerate(specs)]
    if eval_workers > 1:
        with ProcessPoolExecutor(max_workers=eval_workers, mp_context=mp.get_context("spawn")) as pool:
            outcomes = list(pool.map(_official_worker, payloads))
    else:
        outcomes = [_official_worker(payload) for payload in payloads]
    rows = [row for _, row in sorted(outcomes, key=lambda pair: pair[0])]
    return rows, _summary(rows, task_config)


def reproduce(output: Path = OUTPUT, eval_device_preference: DevicePreference = "cpu",
              eval_workers: int = 1) -> dict:
    root = ensure_dir(output / "reproduction")
    source = spatial.OUTPUT
    if not json.loads((source / "reproduction/rollout_verification.json").read_text())["all_match"]:
        raise AssertionError("Upstream aperture result not verified")
    expected = json.loads((source / "aggregate/three_seed_totals.json").read_text())
    results = {}
    for seed in SEEDS:
        root_seed = ensure_dir(root / f"seed{seed}")
        p = paths(seed, output)
        config = {"seed": seed, "motion_checkpoint": str(p["motion"]),
                  "motion_sha256": sha(p["motion"]),
                  "aperture_checkpoint": str(p["aperture"]),
                  "aperture_sha256": sha(p["aperture"]),
                  "base_sha256": sha(residual.BASE_CHECKPOINT),
                  "gripper_correction_bound": gripper_bound(),
                  "point_count": 32, "directed_edges": 100,
                  "task_config": asdict(task(seed))}
        if config["base_sha256"] != residual.BASE_SHA256:
            raise AssertionError("Canonical FF checksum changed")
        residual._write(root_seed / "provenance.json", config)
        results[seed] = {}
        prior_summary = json.loads((source / ("aggregate/summary.json" if seed == 2811 else
            f"replication/seed{seed}/aggregate/summary.json")).read_text())
        for condition in ("static", "dynamic"):
            rows, summary = official_evaluate(seed, condition, output,
                eval_device_preference, eval_workers)
            previous = prior_summary[condition]["branch_bound_p95"]
            for key in ("success", "position_fail", "yaw_fail", "aperture_fail",
                        "collision_fail", "multiple_failures"):
                if summary[key] != previous[key]:
                    raise AssertionError(f"Aperture baseline reproduction failed {seed}/{condition}/{key}")
            if abs(summary["final_position_mean_m"] - previous["position_error_mean"]) > 1e-6 or \
               abs(summary["final_yaw_mean_rad"] - previous["yaw_error_mean"]) > 1e-6 or \
               abs(summary["final_aperture_mean_m"] - previous["aperture_error_mean"]) > 1e-6:
                raise AssertionError("Aperture baseline final error drifted")
            write_csv(root_seed / f"{condition}_per_episode.csv", rows)
            residual._write(root_seed / f"{condition}_summary.json", summary)
            results[seed][condition] = summary
    for condition in ("static", "dynamic"):
        for key in ("success", "position_fail", "yaw_fail", "aperture_fail", "collision_fail"):
            value = sum(results[seed][condition][key] for seed in SEEDS)
            if value != expected[condition]["branch_bound_p95"][key]:
                raise AssertionError("Three-seed aperture totals changed")
    residual._write(root / "summary.json", results)
    return results


def _geom_name(env: sf.SurfaceManipulatorEnv, geom_id: int) -> str:
    return mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or str(geom_id)


@torch.no_grad()
def fixed_horizon_trace(env: sf.SurfaceManipulatorEnv, spec: Any,
                        task_config: sf.SurfaceFeasibilityConfig,
                        controller: spatial.SpanBranchController | residual.ResidualController,
                        device: torch.device, condition: str, controller_name: str) -> tuple[list[dict], dict]:
    """Diagnostic replay: same fresh observations, exactly max_steps, no oracle calls."""
    initial = spec.initial if condition == "dynamic" else spec
    sf._reset_surface_env(env, initial)
    previous = torch.zeros(1, 5, device=device)
    rows = []
    seen_collision = False
    minimum_clearance = math.inf
    motion = getattr(controller, "motion", controller)
    for t in range(task_config.max_steps):
        shape = dynamic._shape_at(env, spec, t) if condition == "dynamic" else spec.shape
        ee = env.robot_observation().ee_position.numpy().astype(np.float64)
        yaw = sf.tool_yaw(env)
        target = sf._surface_target(shape, ee, task_config.pregrasp_clearance)
        state, points, _ = sf.observation_inputs(env, shape, task_config.point_count,
                                                  sf._stable_seed(initial.sample_identity))
        contacts, clearance = sf._surface_distance_metrics(env, shape)
        errors = sf._state_errors(env, shape, task_config, target)
        seen_collision |= bool(contacts)
        minimum_clearance = min(minimum_clearance, clearance)
        state_t, points_t, topology = rf._graph_tensors(state, points, task_config, device)
        start = time.perf_counter_ns()
        z = motion.base.encode(state_t, points_t, topology)
        raw, correction, _, base = controller.step(state_t, points_t, previous,
                                                   topology=topology)
        features = torch.cat((z, base / motion.scales,
                              previous / motion.scales), dim=-1)
        raw_yaw_residual = float(motion.residual_head(motion.memory(features))[0, 3])
        action = rf.clipped_action(raw[0].cpu().numpy(), task_config)
        latency = (time.perf_counter_ns() - start) / 1e6
        left = ee + sf.world_action_from_local(state[8:11], yaw)
        right = ee + sf.world_action_from_local(state[11:14], yaw)
        episode_id = spec.identity if condition == "dynamic" else spec.sample_identity
        rows.append({"episode_id": episode_id, "timestep": t,
                     "controller": controller_name, "condition": condition,
                     "EE_position": ee.tolist(), "EE_yaw": yaw,
                     "target_position": target[0].tolist(), "target_yaw": float(target[2]),
                     "position_error": errors["position_error"],
                     "yaw_error": errors["orientation_error"],
                     "aperture_error": errors["gripper_width_error"],
                     "opening": float(state[7]), "opening_width_m": float(sf.gripper_width(env)),
                     "left_tip": left.tolist(), "right_tip": right.tolist(),
                     "fingertip_separation_m": float(np.linalg.norm(left - right)),
                     "object_relative_position": state[:3].tolist(),
                     "object_relative_yaw": float(math.atan2(state[3], state[4])),
                     "z": z[0].cpu().tolist(),
                     "translation_action": action[:3].tolist(), "yaw_action": float(action[3]),
                     "gripper_action": float(action[4]),
                     "requested_gripper_action": float(getattr(controller,
                         "last_requested_action", [action.tolist()])[0][4]),
                     "coordination_diagnostic": getattr(controller,
                         "last_diagnostic", None),
                     "base_yaw_action": float(base[0, 3]),
                     "raw_yaw_residual": raw_yaw_residual,
                     "bounded_yaw_residual": float(correction[0, 3]),
                     "clearance": float(clearance), "collision": bool(contacts),
                     "contacts": [{"surface_geom": _geom_name(env, c["geom1"]),
                                   "robot_geom": _geom_name(env, c["geom2"]),
                                   "distance": float(c["distance"])} for c in contacts],
                     "inference_latency_ms": latency})
        sf.apply_local_action(env, action, task_config)
        previous = torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, 5)
    shape = dynamic._shape_at(env, spec, task_config.max_steps) if condition == "dynamic" else spec.shape
    ee = env.robot_observation().ee_position.numpy().astype(np.float64)
    target = sf._surface_target(shape, ee, task_config.pregrasp_clearance)
    final_errors = sf._state_errors(env, shape, task_config, target)
    contacts, clearance = sf._surface_distance_metrics(env, shape)
    seen_collision |= bool(contacts)
    minimum_clearance = min(minimum_clearance, clearance)
    summary = {"episode_id": spec.identity if condition == "dynamic" else spec.sample_identity,
               "success": sf._success_from_errors(final_errors, seen_collision, task_config),
               "collision": seen_collision, "final_position_error": final_errors["position_error"],
               "final_yaw_error": final_errors["orientation_error"],
               "final_aperture_error": final_errors["gripper_width_error"],
               "minimum_clearance": minimum_clearance,
               "final_contacts": [{"surface_geom": _geom_name(env, c["geom1"]),
                                   "robot_geom": _geom_name(env, c["geom2"]),
                                   "distance": float(c["distance"])} for c in contacts]}
    return rows, summary


def distribution_shift(seed: int = 2811, output: Path = OUTPUT,
                       eval_device_preference: DevicePreference = "cpu") -> dict:
    if seed != 2811:
        raise ValueError("Development distribution audit is fixed to seed 2811")
    root = ensure_dir(output / "distribution_shift")
    device = select_device(eval_device_preference)
    task_config = task(seed, eval_device=eval_device_preference)
    current = load_combined(seed, device, output=output)
    old = current.motion
    specs = dynamic.dynamic_specs(seed, task_config, 16, dynamic.SELECTED_STEP_SPEEDS_M, "train")
    env = sf.make_env(task_config, seed + 864_211)
    paired, all_traces, summaries = [], [], []
    try:
        for spec in specs:
            old_rows, old_summary = fixed_horizon_trace(env, spec, task_config, old, device,
                                                         "dynamic", "old_5d")
            new_rows, new_summary = fixed_horizon_trace(env, spec, task_config, current, device,
                                                         "dynamic", "aperture_branch")
            summaries.extend([{**old_summary, "controller": "old_5d"},
                              {**new_summary, "controller": "aperture_branch"}])
            all_traces.extend(old_rows + new_rows)
            for a, b in zip(old_rows, new_rows, strict=True):
                paired.append({"episode_id": spec.identity, "timestep": a["timestep"],
                               "opening_abs_delta": abs(b["opening"]-a["opening"]),
                               "left_tip_world_delta_m": float(np.linalg.norm(np.asarray(b["left_tip"])-a["left_tip"])),
                               "right_tip_world_delta_m": float(np.linalg.norm(np.asarray(b["right_tip"])-a["right_tip"])),
                               "tip_separation_delta_m": abs(b["fingertip_separation_m"]-a["fingertip_separation_m"]),
                               "EE_position_delta_m": float(np.linalg.norm(np.asarray(b["EE_position"])-a["EE_position"])),
                               "EE_yaw_delta_rad": abs(float(wrap_angle(b["EE_yaw"]-a["EE_yaw"]))),
                               "object_relative_position_delta_m": float(np.linalg.norm(
                                   np.asarray(b["object_relative_position"])-a["object_relative_position"])),
                               "object_relative_yaw_delta_rad": abs(float(wrap_angle(
                                   b["object_relative_yaw"]-a["object_relative_yaw"]))),
                               "z_l2_delta": float(np.linalg.norm(np.asarray(b["z"])-a["z"])),
                               "translation_action_delta": float(np.linalg.norm(
                                   np.asarray(b["translation_action"])-a["translation_action"])),
                               "yaw_action_abs_delta": abs(b["yaw_action"]-a["yaw_action"]),
                               "gripper_action_abs_delta": abs(b["gripper_action"]-a["gripper_action"]),
                               "clearance_delta_m": b["clearance"]-a["clearance"]})
    finally:
        env.close()
    write_csv(root / "paired_timesteps.csv", paired)
    write_csv(root / "fixed_horizon_traces.csv", all_traces)
    write_csv(root / "episode_outcomes.csv", summaries)
    keys = [key for key in paired[0] if key not in ("episode_id", "timestep")]
    result = {"paired_episodes": len(specs), "paired_timesteps": len(paired),
              "metrics": {key: {"mean": float(np.mean([r[key] for r in paired])),
                                "p95_abs": float(np.quantile(np.abs([r[key] for r in paired]), .95))}
                          for key in keys},
              "paired_episode_outcomes": {"old_5d_success": sum(r["success"] for r in summaries
                                                        if r["controller"] == "old_5d"),
                                          "aperture_branch_success": sum(r["success"] for r in summaries
                                                        if r["controller"] == "aperture_branch")}}
    residual._write(root / "summary.json", result)
    return result


def collect_current_policy(seed: int = 2811, output: Path = OUTPUT,
                           eval_device_preference: DevicePreference = "cpu") -> dict:
    """One fixed 72-episode fresh collection, with current-state oracle relabeling."""
    if seed not in SEEDS:
        raise ValueError(seed)
    root = ensure_dir(output / f"seed{seed}/onpolicy_motion_retraining")
    save_path = root / "current_aperture_policy.pt"
    device = select_device(eval_device_preference)
    task_config = task(seed, eval_device=eval_device_preference)
    specs = dynamic.dynamic_specs(seed, task_config, TRAIN_POLICY_EPISODES,
                                  dynamic.SELECTED_STEP_SPEEDS_M, "train")
    if save_path.exists():
        episodes = torch.load(save_path, map_location="cpu", weights_only=False)
        metadata = json.loads((root / "collection.json").read_text())
        if [ep["identity"] for ep in episodes] != [spec.identity for spec in specs]:
            raise AssertionError("Saved current-policy EpisodeSpecs drifted")
        return metadata
    controller = load_combined(seed, device, output=output)
    episodes, rows = dynamic.collect_sequences(specs, task_config, device,
        "residual_mlp", controller, [0] * len(specs), seed)
    if len(episodes) != TRAIN_POLICY_EPISODES or any(len(ep["targets"]) != task_config.max_steps
        for ep in episodes):
        raise AssertionError("Current-policy collection size changed")
    if any(not torch.isfinite(ep["targets"]).all() for ep in episodes):
        raise AssertionError("Nonfinite on-policy oracle relabels")
    torch.save(episodes, save_path)
    result = {"seed": seed, "episodes": len(episodes),
              "valid_supervised_timesteps": int(sum(ep["mask"].sum() for ep in episodes)),
              "behavior_delay": 0, "collection_once": True,
              "motion_checkpoint_sha256": sha(paths(seed, output)["motion"]),
              "aperture_checkpoint_sha256": sha(paths(seed, output)["aperture"]),
              "aperture_bound": gripper_bound(),
              "episode_identities": [ep["identity"] for ep in episodes],
              "rollout_success": int(sum(row["success"] for row in rows)),
              "rollout_aperture_fail": int(sum(row["final_aperture_error"] >=
                                               task_config.success_opening_threshold for row in rows)),
              "training_labels": "current-state scripted oracle action, never used in closed-loop evaluation"}
    residual._write(root / "collection.json", result)
    write_csv(root / "collection_outcomes.csv", [{"episode_id": row["episode_id"],
        "success": row["success"], "collision": row["collision"],
        "final_position": row["final_tracking_error"], "final_yaw": row["final_yaw_error"],
        "final_aperture": row["final_aperture_error"]} for row in rows])
    return result


def train_motion(seed: int = 2811, output: Path = OUTPUT,
                 device_preference: DevicePreference = "auto",
                 yaw_bound: float | None = None) -> dict:
    """Exactly 400 motion-only updates on fixed old+current data; no recollection."""
    if seed not in SEEDS:
        raise ValueError(seed)
    data_root = output / f"seed{seed}/onpolicy_motion_retraining"
    root = ensure_dir(output / f"seed{seed}" /
        ("yaw_bound_retrained" if yaw_bound is not None else "onpolicy_motion_retraining"))
    collection = json.loads((data_root / "collection.json").read_text())
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task_config = task(seed, device_preference)
    source = dynamic.OUTPUT / f"seed{seed}/training"
    expert = torch.load(source / "expert.pt", map_location="cpu", weights_only=False)
    old_policy = torch.load(source / "policy_mlp.pt", map_location="cpu", weights_only=False)
    current = torch.load(data_root / "current_aperture_policy.pt", map_location="cpu", weights_only=False)
    validation = torch.load(source / "validation.pt", map_location="cpu", weights_only=False)
    if set(ep["identity"] for ep in validation) & set(ep["identity"] for ep in current):
        raise AssertionError("On-policy train and validation episodes overlap")
    if [ep["identity"] for ep in current] != collection["episode_identities"]:
        raise AssertionError("Current-policy collection changed after saving")
    design = replace(residual.ResidualConfig(), seed=seed + 710_000,
                     corrected_dimensions=5, updates=MOTION_UPDATES)
    batches, schedule = residual.make_schedule({"expert": expert + old_policy,
                                                "policy": current}, design,
                                               dynamic.dynamic_delayed_sequence)
    residual._write(root / "motion_schedule.json", schedule)
    set_seed(seed + 710_000)
    model = residual.load_controller("mlp", task_config, design, device,
                                     paths(seed, output)["motion"])
    if yaw_bound is not None:
        if not 0 < yaw_bound <= task_config.max_delta_rotation:
            raise ValueError("Invalid yaw correction bound")
        model.residual_bounds[3] = yaw_bound
    base_before = {key: value.detach().cpu().clone() for key, value in model.base.state_dict().items()}
    aperture_sha_before = sha(paths(seed, output)["aperture"])
    fifth_weight = model.residual_head.weight[4].detach().clone()
    fifth_bias = model.residual_head.bias[4].detach().clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=task_config.learning_rate,
                                  weight_decay=task_config.weight_decay)
    best, best_update, curve = math.inf, 0, []
    started = time.perf_counter()
    for step, cpu_batch in enumerate(batches, 1):
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        model.train()
        action, _, _, _, _ = model.forward_sequence(batch["states"], batch["points"],
                                                    batch["previous"])
        loss = .5 * residual.corrected_loss(action[:8], batch["targets"][:8],
                                            batch["mask"][:8], task_config, dimensions=4) + \
               .5 * residual.corrected_loss(action[8:], batch["targets"][8:],
                                            batch["mask"][8:], task_config, dimensions=4)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in model.base.parameters()):
            raise AssertionError("Frozen graph/FF received gradients")
        optimizer.step()
        with torch.no_grad():
            model.residual_head.weight[4].copy_(fifth_weight)
            model.residual_head.bias[4].copy_(fifth_bias)
        if step % VALIDATION_INTERVAL == 0:
            model.eval()
            val = residual.validation_metrics(model, validation, task_config, device)
            score = val["corrected_action_mse_4d"]
            curve.append({"update": step, "train_motion_4d_mse": float(loss.detach().cpu()),
                          "validation_motion_4d_mse": score,
                          "validation_full_5d_mse_diagnostic": val["corrected_action_mse_5d"]})
            if score < best:
                best, best_update = score, step
                torch.save({"model_state": model.state_dict(), "kind": "mlp",
                            "base_sha256": residual.BASE_SHA256, "design": asdict(design),
                            "seed": seed, "selected_update": step,
                            "validation_4d_mse": score,
                            "topology": "consistent_intended_100",
                            "aperture_branch_sha256": aperture_sha_before,
                            "starting_motion_sha256": sha(paths(seed, output)["motion"])},
                           root / "selected.pt")
    torch.save({"model_state": model.state_dict(), "kind": "mlp",
                "base_sha256": residual.BASE_SHA256, "design": asdict(design),
                "seed": seed, "selected_update": MOTION_UPDATES,
                "validation_4d_mse": curve[-1]["validation_motion_4d_mse"],
                "topology": "consistent_intended_100",
                "aperture_branch_sha256": aperture_sha_before,
                "starting_motion_sha256": sha(paths(seed, output)["motion"])},
               root / "final.pt")
    if any(not torch.equal(base_before[key], value.detach().cpu())
           for key, value in model.base.state_dict().items()) or \
       not torch.equal(model.residual_head.weight[4], fifth_weight) or \
       not torch.equal(model.residual_head.bias[4], fifth_bias):
        raise AssertionError("Frozen graph/FF or fifth residual projection changed")
    if sha(paths(seed, output)["aperture"]) != aperture_sha_before:
        raise AssertionError("Aperture checkpoint changed during motion retraining")
    write_csv(root / "learning_curve.csv", curve)
    result = {"seed": seed, "updates": MOTION_UPDATES, "selected_update": best_update,
              "yaw_residual_bound_rad": float(model.residual_bounds[3]),
              "best_validation_motion_4d_mse": best,
              "final_validation_motion_4d_mse": curve[-1]["validation_motion_4d_mse"],
              "schedule_sha256": schedule["schedule_sha256"],
              "batch_size": design.sequence_batch_size,
              "data": {"existing_expert": len(expert), "existing_policy": len(old_policy),
                       "new_current_policy": len(current)},
              "loss": "same normalized action MSE, motion dimensions 0:4 only",
              "optimizer": "fresh AdamW", "learning_rate": task_config.learning_rate,
              "weight_decay": task_config.weight_decay,
              "training_seconds": time.perf_counter() - started,
              "device": str(device),
              "starting_motion_sha256": sha(paths(seed, output)["motion"]),
              "selected_motion_sha256": sha(root / "selected.pt"),
              "aperture_checkpoint_sha256": aperture_sha_before}
    residual._write(root / "training_summary.json", result)
    return result


def compare_retrained(seed: int = 2811, output: Path = OUTPUT,
                      eval_device_preference: DevicePreference = "cpu",
                      eval_workers: int = 1) -> dict:
    root = ensure_dir(output / f"seed{seed}/onpolicy_motion_retraining")
    new_checkpoint = root / "selected.pt"
    if not new_checkpoint.exists():
        raise FileNotFoundError(new_checkpoint)
    results = {}
    for condition in ("static", "dynamic"):
        before_rows = list(csv.DictReader((output /
            f"reproduction/seed{seed}/{condition}_per_episode.csv").open()))
        after_rows, after_summary = official_evaluate(seed, condition, output,
            eval_device_preference, eval_workers, new_checkpoint)
        before_ids = [r["episode_id"] for r in before_rows]
        if before_ids != [r["episode_id"] for r in after_rows]:
            raise AssertionError("Paired evaluation EpisodeSpecs changed")
        before_summary = json.loads((output /
            f"reproduction/seed{seed}/{condition}_summary.json").read_text())
        write_csv(root / f"{condition}_after_per_episode.csv", after_rows)
        residual._write(root / f"{condition}_after_summary.json", after_summary)
        paired = [{"episode_id": old["episode_id"],
                   "before_success": old["success"] == "True", "after_success": new["success"],
                   "before_position_error": float(old["position_error"]),
                   "after_position_error": new["position_error"],
                   "before_yaw_error": float(old["yaw_error"]),
                   "after_yaw_error": new["yaw_error"],
                   "before_aperture_error": float(old["aperture_error"]),
                   "after_aperture_error": new["aperture_error"],
                   "before_collision": old["collision"] == "True",
                   "after_collision": new["collision"]}
                  for old, new in zip(before_rows, after_rows, strict=True)]
        write_csv(root / f"{condition}_paired.csv", paired)
        results[condition] = {"before": before_summary, "after": after_summary,
                              "success_gained": sum(not r["before_success"] and r["after_success"] for r in paired),
                              "success_lost": sum(r["before_success"] and not r["after_success"] for r in paired)}
    residual._write(root / "evaluation_summary.json", results)
    return results


def yaw_semantics(output: Path = OUTPUT) -> dict:
    root = ensure_dir(output / "yaw_audit")
    task_config = task(2811)
    result = {"robot_yaw": "tool_yaw from EE site rotation atan2(R[2,0], R[0,0])",
              "state_3_5": "sin and cos of wrap(object_yaw - EE_yaw)",
              "state_5_7": "sin and cos of EE_yaw",
              "surface_orientation": "object_yaw + part_yaw rotates centerline/surface",
              "target_yaw": "atan2(outward_normal_z, outward_normal_x) at centerline s=0.5; side chosen by EE position",
              "action_dimension_3": "signed local yaw delta in radians, clipped to native ±0.35 per step",
              "execution": "target_yaw = wrap(current_EE_yaw + clipped_delta); IK follows target",
              "oracle_action": "clip(wrap(target_yaw - current_EE_yaw), ±0.35)",
              "final_error": "abs(wrap(EE_yaw - target_yaw)); no π equivalence because unique approach side",
              "native_yaw_limit_rad": task_config.max_delta_rotation,
              "residual_yaw_bound_rad": .2 * task_config.max_delta_rotation,
              "success_yaw_threshold_rad": task_config.success_rotation_threshold,
              "wrap_boundary_example": {"target_deg": 179, "current_deg": -179,
                  "wrapped_error_deg": abs(math.degrees(float(wrap_angle(
                      math.radians(179) - math.radians(-179)))))}}
    if abs(result["wrap_boundary_example"]["wrapped_error_deg"] - 2) > 1e-9:
        raise AssertionError("Yaw wrap boundary regression")
    residual._write(root / "semantics.json", result)
    return result


def _yaw_split(split: str, seed: int = 2811) -> dict:
    task_config = task(seed)
    source = dynamic.OUTPUT / f"seed{seed}"
    names = ("expert", "policy_mlp") if split == "train" else (
        ("validation",) if split == "validation" else ("heldout_oracle",))
    specs = {spec.identity: spec for name, count in (("train", 72), ("validation", 16),
             ("heldout", 16)) for spec in dynamic.dynamic_specs(
                 seed, task_config, count, dynamic.SELECTED_STEP_SPEEDS_M, name)}
    states, points, oracle_action, uncapped_delta, target_yaw, current_yaw, ids, steps = [], [], [], [], [], [], [], []
    for name in names:
        path = (source / "benchmark_sanity/heldout_oracle.pt" if name == "heldout_oracle"
                else source / f"training/{name}.pt")
        for ep in torch.load(path, map_location="cpu", weights_only=False):
            spec = specs[ep["identity"]]
            for t, mask in enumerate(ep["mask"]):
                if not bool(mask):
                    continue
                state = ep["states"][t].numpy()
                shape = dynamic.moving_shape(spec, t)
                ee = ep["ee_world"][t].numpy()
                target = sf._surface_target(shape, ee, task_config.pregrasp_clearance)[2]
                yaw = math.atan2(float(state[5]), float(state[6]))
                delta = float(wrap_angle(target-yaw))
                action = float(ep["targets"][t, 3])
                if abs(float(np.clip(delta, -task_config.max_delta_rotation,
                                     task_config.max_delta_rotation)) - action) > 2e-4:
                    raise AssertionError("Yaw oracle action does not match audited semantics")
                states.append(state)
                points.append(ep["points"][t].numpy())
                oracle_action.append(action)
                uncapped_delta.append(delta)
                target_yaw.append(target)
                current_yaw.append(yaw)
                ids.append(ep["identity"])
                steps.append(t)
    return {"state": np.asarray(states, np.float32), "points": np.asarray(points, np.float32),
            "oracle_yaw_action": np.asarray(oracle_action, np.float32),
            "uncapped_delta": np.asarray(uncapped_delta, np.float32),
            "target_yaw": np.asarray(target_yaw, np.float32),
            "current_yaw": np.asarray(current_yaw, np.float32),
            "identity": ids, "timestep": steps}


class YawProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def yaw_spatial_features(state: torch.Tensor, points: torch.Tensor,
                         current_yaw_action: torch.Tensor) -> torch.Tensor:
    """Order-free observable moments, including pairwise surface covariance."""
    centered = points - points.mean(dim=-2, keepdim=True)
    covariance = (centered[..., 0] * centered[..., 2]).mean(dim=-1, keepdim=True)
    return torch.cat((state, points.mean(dim=-2), points.std(dim=-2, unbiased=False),
                      points.amin(dim=-2), points.amax(dim=-2), covariance,
                      current_yaw_action.unsqueeze(-1)), dim=-1)


class YawSpatialHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(28, 64), nn.SiLU(), nn.Linear(64, 64),
                                 nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class YawSpatialController:
    """Frozen motion and aperture paths plus one order-free yaw correction head."""
    def __init__(self, combined: spatial.SpanBranchController, checkpoint: Path,
                 device: torch.device) -> None:
        self.combined = combined
        self.motion = combined.motion
        self.task = combined.task
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.head = YawSpatialHead().to(device)
        self.head.load_state_dict(payload["model_state"])
        self.head.eval()
        self.feature_mean = torch.as_tensor(payload["feature_mean"], device=device)
        self.feature_scale = torch.as_tensor(payload["feature_scale"], device=device)
        self.bound = float(payload["correction_bound_rad"])

    @torch.no_grad()
    def step(self, state: torch.Tensor, points: torch.Tensor, previous: torch.Tensor,
             hidden: torch.Tensor | None = None, topology: sf.GraphTopology | None = None,
             reset_hidden: bool = False) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
        action, _, _, base = self.combined.step(state, points, previous, hidden,
                                               topology, reset_hidden)
        feature = yaw_spatial_features(state, points, action[:, 3])
        residual_yaw = self.bound * torch.tanh(self.head(
            (feature - self.feature_mean) / self.feature_scale))
        yaw = (action[:, 3] + residual_yaw).clamp(-self.task.max_delta_rotation,
                                                 self.task.max_delta_rotation)
        corrected = torch.cat((action[:, :3], yaw[:, None], action[:, 4:]), dim=-1)
        return corrected, corrected-base, None, base


def _yaw_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    error = np.abs(pred - true)
    return {"mae_rad": float(error.mean()), "p50_rad": float(np.quantile(error, .5)),
            "p90_rad": float(np.quantile(error, .9)), "p95_rad": float(np.quantile(error, .95)),
            "within_final_0p20_rad_proxy": float(np.mean(error < .20)),
            "correlation": float(np.corrcoef(true, pred)[0, 1]) if np.std(pred) > 1e-9 else None}


def yaw_probes(output: Path = OUTPUT, device_preference: DevicePreference = "auto") -> dict:
    root = ensure_dir(output / "yaw_probe")
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task_config = task(2811, device_preference)
    splits = {name: _yaw_split(name) for name in ("train", "validation", "heldout")}
    ids = {name: set(data["identity"]) for name, data in splits.items()}
    if any(ids[a] & ids[b] for a, b in (("train", "validation"),
            ("train", "heldout"), ("validation", "heldout"))):
        raise AssertionError("Yaw probe episode leakage")
    model = residual.load_controller("mlp", task_config,
        replace(residual.ResidualConfig(), corrected_dimensions=5), device,
        paths(2811, output)["updated_motion"]).eval()
    features = {}
    for name, data in splits.items():
        z = []
        with torch.no_grad():
            for start in range(0, len(data["state"]), 128):
                state_t = torch.from_numpy(data["state"][start:start+128]).to(device)
                point_t = torch.from_numpy(data["points"][start:start+128]).to(device)
                z.append(model.base.encode(state_t, point_t).cpu().numpy())
        features[name] = {"current_relative_yaw": data["state"][:, 3:7],
                          "robot_state": data["state"],
                          "frozen_z": np.concatenate(z),
                          "raw_graph_ordered_diagnostic": np.concatenate(
                              [data["state"], data["points"].reshape(len(data["state"]), -1)], axis=1)}
    results = {}
    for probe_name in features["train"]:
        root_probe = ensure_dir(root / probe_name)
        mean = features["train"][probe_name].mean(0, keepdims=True)
        std = features["train"][probe_name].std(0, keepdims=True).clip(1e-5)
        x = {name: torch.as_tensor((feat[probe_name]-mean)/std,
                                   dtype=torch.float32, device=device)
             for name, feat in features.items()}
        y = {name: torch.as_tensor(data["oracle_yaw_action"] / task_config.max_delta_rotation,
                                   dtype=torch.float32, device=device)
             for name, data in splits.items()}
        set_seed(2811 + 990_000 + len(probe_name))
        probe = YawProbe(x["train"].shape[1]).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(2811 + 990_000)
        best, selected, stale, curve = math.inf, 0, 0, []
        for step in range(1, 5001):
            index = torch.randint(len(x["train"]), (256,), generator=generator).to(device)
            probe.train()
            prediction = probe(x["train"][index])
            loss = (prediction-y["train"][index]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 25 == 0:
                probe.eval()
                with torch.no_grad():
                    val = float((probe(x["validation"])-y["validation"]).square().mean())
                curve.append({"update": step, "train_normalized_mse": float(loss.detach().cpu()),
                              "validation_normalized_mse": val})
                if val < best - 1e-5:
                    best, selected, stale = val, step, 0
                    torch.save({"model_state": probe.state_dict(), "input_dim": x["train"].shape[1],
                                "feature_mean": mean, "feature_std": std,
                                "action_scale": task_config.max_delta_rotation,
                                "selected_update": step}, root_probe / "selected.pt")
                else:
                    stale += 1
                if step >= 500 and stale >= 20:
                    break
        write_csv(root_probe / "learning_curve.csv", curve)
        payload = torch.load(root_probe / "selected.pt", map_location=device, weights_only=False)
        probe.load_state_dict(payload["model_state"])
        probe.eval()
        scores = {}
        for name in ("validation", "heldout"):
            with torch.no_grad():
                pred = probe(x[name]).cpu().numpy() * task_config.max_delta_rotation
            rows = [{"episode_id": identity, "timestep": t,
                     "true_yaw_action_rad": float(actual), "predicted_yaw_action_rad": float(estimate),
                     "error_rad": float(estimate-actual),
                     "current_yaw_rad": float(current), "target_yaw_rad": float(target)}
                    for identity, t, actual, estimate, current, target in zip(
                        splits[name]["identity"], splits[name]["timestep"],
                        splits[name]["oracle_yaw_action"], pred,
                        splits[name]["current_yaw"], splits[name]["target_yaw"], strict=True)]
            write_csv(root_probe / f"{name}_predictions.csv", rows)
            scores[name] = _yaw_metrics(splits[name]["oracle_yaw_action"], pred)
        results[probe_name] = {"selected_update": selected, "final_update": step,
                               "validation": scores["validation"], "heldout": scores["heldout"]}
    residual._write(root / "summary.json", results)
    residual._write(root / "split_audit.json", {name: {"episodes": len(ids[name]),
        "timesteps": len(splits[name]["state"])} for name in splits})
    return results


def yaw_counterfactual(output: Path = OUTPUT,
                       eval_device_preference: DevicePreference = "cpu") -> dict:
    root = ensure_dir(output / "yaw_counterfactual")
    device = select_device(eval_device_preference)
    task_config = task(2811, eval_device=eval_device_preference)
    model = residual.load_controller("mlp", task_config,
        replace(residual.ResidualConfig(), corrected_dimensions=5), device,
        paths(2811, output)["updated_motion"])
    yaw_spatial_path = output / "seed2811/yaw_spatial_readout/selected.pt"
    yaw_spatial = (YawSpatialController(load_combined(2811, device,
        paths(2811, output)["updated_motion"], output), yaw_spatial_path, device)
        if yaw_spatial_path.exists() else None)
    probes = {}
    for name in ("raw_graph_ordered_diagnostic", "frozen_z"):
        payload = torch.load(output / f"yaw_probe/{name}/selected.pt",
                             map_location=device, weights_only=False)
        probe = YawProbe(payload["input_dim"]).to(device)
        probe.load_state_dict(payload["model_state"])
        probes[name] = (probe.eval(), payload)
    source = dynamic.OUTPUT / "seed2811/benchmark_sanity/heldout_oracle.pt"
    episodes = torch.load(source, map_location="cpu", weights_only=False)
    specs = {spec.identity: spec for spec in dynamic.dynamic_specs(
        2811, task_config, 16, dynamic.SELECTED_STEP_SPEEDS_M, "heldout")}
    rows = []
    for ep in episodes:
        spec = specs[ep["identity"]]
        candidates = []
        for t in range(len(ep["states"])):
            state_t = ep["states"][t].numpy()
            ee_t = ep["ee_world"][t].numpy()
            yaw_t = math.atan2(float(state_t[5]), float(state_t[6]))
            target_t = sf._surface_target(dynamic.moving_shape(spec, t), ee_t,
                                          task_config.pregrasp_clearance)[2]
            candidates.append(abs(float(wrap_angle(target_t-yaw_t))))
        selected_t = int(np.argmin(candidates))
        original_state = ep["states"][selected_t].numpy()
        ee = ep["ee_world"][selected_t].numpy()
        base_shape = dynamic.moving_shape(spec, selected_t)
        ee_yaw = math.atan2(float(original_state[5]), float(original_state[6]))
        variants = {}
        for label, offset in (("A", -.15), ("B", .15)):
            shape = replace(spec.initial.shape, object_yaw=float(wrap_angle(
                spec.initial.shape.object_yaw + offset)), object_center=base_shape.object_center)
            state = original_state.copy()
            relative = float(wrap_angle(shape.object_yaw - ee_yaw))
            state[3:5] = [math.sin(relative), math.cos(relative)]
            points_world = sf.sample_surface_points_world(shape, 32,
                sf._stable_seed(spec.initial.sample_identity))
            points = sf.localize_world(points_world, ee, ee_yaw).astype(np.float32)
            target = sf._surface_target(shape, ee, task_config.pregrasp_clearance)
            state_t = torch.as_tensor(state[None], dtype=torch.float32, device=device)
            points_t = torch.as_tensor(points[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                z = model.base.encode(state_t, points_t).cpu().numpy()
                action, correction, _, base = model.step(state_t, points_t,
                                                          torch.zeros(1, 5, device=device))
                spatial_action = (yaw_spatial.step(state_t, points_t,
                    torch.zeros(1, 5, device=device))[0] if yaw_spatial is not None else None)
            predictions = {}
            for name, (probe, payload) in probes.items():
                feature = (z if name == "frozen_z" else
                           np.concatenate([state[None], points.reshape(1, -1)], axis=1))
                x = torch.as_tensor((feature-payload["feature_mean"])/payload["feature_std"],
                                    dtype=torch.float32, device=device)
                with torch.no_grad():
                    predictions[name] = float(probe(x).item() * payload["action_scale"])
            variants[label] = {"state": state, "points": points,
                               "target_yaw": float(target[2]), "side": int(target[4]),
                               "true_oracle_action": float(np.clip(wrap_angle(target[2]-ee_yaw),
                                        -task_config.max_delta_rotation, task_config.max_delta_rotation)),
                               "z": z[0], "motion_yaw_action": float(action[0, 3]),
                               "yaw_spatial_action": (float(spatial_action[0, 3])
                                                      if spatial_action is not None else None),
                               "base_yaw_action": float(base[0, 3]),
                               "bounded_yaw_residual": float(correction[0, 3]),
                               "predictions": predictions}
        a, b = variants["A"], variants["B"]
        if not np.array_equal(a["state"][:3], b["state"][:3]) or \
           not np.array_equal(a["state"][5:], b["state"][5:]):
            raise AssertionError("Yaw counterfactual changed robot state or object position")
        rows.append({"pair_id": ep["identity"], "source_timestep": selected_t,
                     "source_baseline_abs_yaw_error": candidates[selected_t],
                     "same_approach_side": a["side"] == b["side"],
                     "true_target_yaw_A": a["target_yaw"], "true_target_yaw_B": b["target_yaw"],
                     "true_target_yaw_delta": float(wrap_angle(b["target_yaw"]-a["target_yaw"])),
                     "true_oracle_action_delta": b["true_oracle_action"]-a["true_oracle_action"],
                     "raw_graph_probe_action_delta": b["predictions"]["raw_graph_ordered_diagnostic"] -
                                                    a["predictions"]["raw_graph_ordered_diagnostic"],
                     "frozen_z_probe_action_delta": b["predictions"]["frozen_z"]-
                                                    a["predictions"]["frozen_z"],
                     "motion_yaw_action_delta": b["motion_yaw_action"]-a["motion_yaw_action"],
                     "yaw_spatial_action_delta": (b["yaw_spatial_action"]-
                        a["yaw_spatial_action"] if yaw_spatial is not None else None),
                     "ff_yaw_action_delta": b["base_yaw_action"]-a["base_yaw_action"],
                     "z_l2_delta": float(np.linalg.norm(b["z"]-a["z"])),
                     "surface_cloud_l2_delta": float(np.linalg.norm(b["points"]-a["points"])),
                     "robot_state_other_max_delta": float(np.max(np.abs(
                         b["state"][5:]-a["state"][5:]))),
                     "relative_yaw_feature_l2_delta": float(np.linalg.norm(
                         b["state"][3:5]-a["state"][3:5]))})
    write_csv(root / "paired_response.csv", rows)
    same = [row for row in rows if row["same_approach_side"]]
    if len(same) < 8:
        raise AssertionError("Too few same-side yaw counterfactual pairs")
    result = {"pairs": len(rows), "same_side_pairs": len(same),
              "same_side_mean_true_target_delta_rad": float(np.mean([r["true_target_yaw_delta"] for r in same])),
              "same_side_mean_oracle_action_delta_rad": float(np.mean([r["true_oracle_action_delta"] for r in same])),
              **{key + "_same_side_mean": float(np.mean([r[key] for r in same])) for key in
                 ("raw_graph_probe_action_delta", "frozen_z_probe_action_delta",
                  "motion_yaw_action_delta", "ff_yaw_action_delta", "z_l2_delta")}}
    if yaw_spatial is not None:
        result["yaw_spatial_action_delta_same_side_mean"] = float(np.mean(
            [r["yaw_spatial_action_delta"] for r in same]))
    residual._write(root / "summary.json", result)
    return result


def yaw_capacity(output: Path = OUTPUT,
                 eval_device_preference: DevicePreference = "cpu") -> dict:
    root = ensure_dir(output / "yaw_capacity")
    device = select_device(eval_device_preference)
    task_config = task(2811, eval_device=eval_device_preference)
    model = residual.load_controller("mlp", task_config,
        replace(residual.ResidualConfig(), corrected_dimensions=5), device,
        paths(2811, output)["updated_motion"])
    source = dynamic.OUTPUT / "seed2811/training"
    episodes_by_source = {"expert": torch.load(source / "expert.pt", map_location="cpu", weights_only=False),
                          "old_policy": torch.load(source / "policy_mlp.pt", map_location="cpu", weights_only=False),
                          "current_policy": torch.load(output /
                             "seed2811/onpolicy_motion_retraining/current_aperture_policy.pt",
                             map_location="cpu", weights_only=False)}
    rows = []
    with torch.no_grad():
        for source_name, episodes in episodes_by_source.items():
            for ep in episodes:
                states = ep["states"][None].to(device)
                points = ep["points"][None].to(device)
                previous = ep["previous"][None].to(device)
                _, correction, _, base, _ = model.forward_sequence(states, points, previous)
                z = model.base.encode(states.flatten(0, 1), points.flatten(0, 1)).reshape(1, len(ep["states"]), -1)
                features = torch.cat((z, base/model.scales, previous/model.scales), dim=-1)
                raw = model.residual_head(model.memory(features))[0, :, 3]
                for t, valid in enumerate(ep["mask"]):
                    if not bool(valid):
                        continue
                    target = float(ep["targets"][t, 3]-base[0, t, 3].cpu())
                    bounded = float(correction[0, t, 3])
                    rows.append({"source": source_name, "episode_id": ep["identity"],
                                 "timestep": t, "oracle_minus_ff_yaw_correction_rad": target,
                                 "raw_predicted_yaw_residual": float(raw[t]),
                                 "bounded_yaw_residual_rad": bounded,
                                 "near_bound": abs(bounded) >= .069,
                                 "base_yaw_action_rad": float(base[0, t, 3]),
                                 "oracle_yaw_action_rad": float(ep["targets"][t, 3])})
    write_csv(root / "supervised_corrections.csv", rows)
    absolute = np.abs([r["oracle_minus_ff_yaw_correction_rad"] for r in rows])
    result = {"supervised_timesteps": len(rows), "native_yaw_action_limit_rad": task_config.max_delta_rotation,
              "current_yaw_residual_bound_rad": .2*task_config.max_delta_rotation,
              "target_correction_abs_p50_rad": float(np.quantile(absolute, .5)),
              "target_correction_abs_p90_rad": float(np.quantile(absolute, .9)),
              "target_correction_abs_p95_rad": float(np.quantile(absolute, .95)),
              "target_correction_abs_max_rad": float(np.max(absolute)),
              "supervised_fraction_exceeding_bound": float(np.mean(absolute > .07)),
              "supervised_predicted_near_bound_fraction": float(np.mean([r["near_bound"] for r in rows]))}
    # Rollout statistics use fixed-horizon held-out dynamic traces, without oracle calls.
    task_config = task(2811, eval_device=eval_device_preference)
    combined = load_combined(2811, device, paths(2811, output)["updated_motion"], output)
    specs = dynamic.dynamic_specs(2811, task_config, 16,
                                  dynamic.SELECTED_STEP_SPEEDS_M, "heldout")
    env = sf.make_env(task_config, 2811 + 864_211)
    trace_rows = []
    try:
        for spec in specs:
            trace, _ = fixed_horizon_trace(env, spec, task_config, combined, device,
                                            "dynamic", "adapted_motion")
            trace_rows.extend(trace)
    finally:
        env.close()
    write_csv(root / "dynamic_fixed_horizon_traces.csv", trace_rows)
    result["rollout_steps"] = len(trace_rows)
    result["rollout_near_bound_fraction"] = float(np.mean([
        abs(r["bounded_yaw_residual"]) >= .069 for r in trace_rows]))
    result["rollout_raw_abs_p95"] = float(np.quantile(np.abs([
        r["raw_yaw_residual"] for r in trace_rows]), .95))
    result["rollout_bounded_abs_p95_rad"] = float(np.quantile(np.abs([
        r["bounded_yaw_residual"] for r in trace_rows]), .95))
    residual._write(root / "summary.json", result)
    return result


def yaw_coverage(output: Path = OUTPUT) -> dict:
    """Describe orientation support before choosing a yaw intervention."""
    root = ensure_dir(output / "yaw_coverage")
    task_config = task(2811)
    rows = []
    for split in ("train", "validation", "heldout"):
        data = _yaw_split(split)
        for identity, t, target, current, action, uncapped in zip(
                data["identity"], data["timestep"], data["target_yaw"],
                data["current_yaw"], data["oracle_yaw_action"],
                data["uncapped_delta"], strict=True):
            rows.append({"split": split, "condition": "dynamic", "episode_id": identity,
                         "timestep": t, "relative_yaw_rad": float(uncapped),
                         "target_yaw_rad": float(target),
                         "current_yaw_rad": float(current),
                         "oracle_yaw_action_abs_rad": abs(float(action)),
                         "initial": t == 0})
    # Static held-out: measure the same quantities at reset, using observed EE yaw.
    env = sf.make_env(task_config, 2811 + 864_211)
    try:
        for spec in sf.sample_episode_specs(16, 2811 + 60_000, task_config, "iid"):
            sf._reset_surface_env(env, spec)
            yaw = sf.tool_yaw(env)
            ee = env.robot_observation().ee_position.numpy().astype(np.float64)
            target = sf._surface_target(spec.shape, ee, task_config.pregrasp_clearance)[2]
            delta = float(wrap_angle(target-yaw))
            rows.append({"split": "heldout", "condition": "static",
                         "episode_id": spec.sample_identity, "timestep": 0,
                         "relative_yaw_rad": delta, "target_yaw_rad": float(target),
                         "current_yaw_rad": yaw,
                         "oracle_yaw_action_abs_rad": abs(float(np.clip(delta,
                             -task_config.max_delta_rotation, task_config.max_delta_rotation))),
                         "initial": True})
    finally:
        env.close()
    write_csv(root / "orientation_rows.csv", rows)
    failures = {r["episode_id"]: float(r["yaw_error"]) >=
                task_config.success_rotation_threshold for r in csv.DictReader(
                    (output / "seed2811/onpolicy_motion_retraining/dynamic_after_per_episode.csv").open())}
    initial_rows = [r for r in rows if r["initial"] and r["condition"] == "dynamic"]
    train = [r for r in initial_rows if r["split"] == "train"]
    held = [r for r in initial_rows if r["split"] == "heldout"]
    train_abs = np.abs([r["relative_yaw_rad"] for r in train])
    fail = [r for r in held if failures[r["episode_id"]]]
    passed = [r for r in held if not failures[r["episode_id"]]]
    result = {"initial_dynamic_train_episodes": len(train),
              "initial_dynamic_heldout_episodes": len(held),
              "train_initial_abs_yaw_quantiles_rad": {str(q): float(np.quantile(train_abs, q))
                  for q in (0, .1, .5, .9, 1)},
              "train_initial_yaw_range_rad": [float(min(r["relative_yaw_rad"] for r in train)),
                                                  float(max(r["relative_yaw_rad"] for r in train))],
              "heldout_fail_initial_abs_yaw_mean_rad": float(np.mean(
                  np.abs([r["relative_yaw_rad"] for r in fail]))),
              "heldout_pass_initial_abs_yaw_mean_rad": float(np.mean(
                  np.abs([r["relative_yaw_rad"] for r in passed]))),
              "heldout_fail_outside_train_initial_range": int(sum(
                  r["relative_yaw_rad"] < min(x["relative_yaw_rad"] for x in train) or
                  r["relative_yaw_rad"] > max(x["relative_yaw_rad"] for x in train)
                  for r in fail)),
              "heldout_fail_episode_ids": [r["episode_id"] for r in fail]}
    for split in ("train", "validation", "heldout"):
        subset = [r for r in rows if r["condition"] == "dynamic" and r["split"] == split]
        result[split + "_action_abs_quantiles_rad"] = {str(q): float(np.quantile(
            [r["oracle_yaw_action_abs_rad"] for r in subset], q)) for q in (.5, .9, .95)}
        result[split + "_target_quadrants"] = [int(sum(
            (-math.pi + k * math.pi / 2) <= r["target_yaw_rad"] <
            (-math.pi + (k + 1) * math.pi / 2) for r in subset)) for k in range(4)]
    residual._write(root / "summary.json", result)
    return result


def yaw_bound_ablation(seed: int = 2811, output: Path = OUTPUT,
                      eval_device_preference: DevicePreference = "cpu",
                      eval_workers: int = 1) -> dict:
    """One frozen-weight bound change derived once from seed-2811 training corrections."""
    root = ensure_dir(output / f"seed{seed}/yaw_bound_ablation")
    capacity = json.loads((output / "yaw_capacity/summary.json").read_text())
    bound = min(float(capacity["native_yaw_action_limit_rad"]),
                float(capacity["target_correction_abs_p95_rad"]))
    checkpoint = paths(seed, output)["updated_motion"]
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    results = {}
    for condition in ("static", "dynamic"):
        reference = list(csv.DictReader((output /
            f"seed{seed}/onpolicy_motion_retraining/{condition}_after_per_episode.csv").open()))
        rows, summary = official_evaluate(seed, condition, output,
            eval_device_preference, eval_workers, checkpoint, yaw_bound=bound)
        if [r["episode_id"] for r in rows] != [r["episode_id"] for r in reference]:
            raise AssertionError("Yaw bound ablation is not episode paired")
        write_csv(root / f"{condition}_per_episode.csv", rows)
        residual._write(root / f"{condition}_summary.json", summary)
        results[condition] = summary
    result = {"seed": seed, "old_bound_rad": .2*task(seed).max_delta_rotation,
              "new_bound_rad": bound,
              "derivation": "min(native yaw action limit, seed-2811 training p95 absolute oracle-minus-FF yaw correction)",
              "motion_checkpoint_sha256": sha(checkpoint),
              "aperture_checkpoint_sha256": sha(paths(seed, output)["aperture"]),
              "weights_unchanged": True, "evaluation": results}
    residual._write(root / "summary.json", result)
    return result


def yaw_bound_retrained(seed: int = 2811, output: Path = OUTPUT,
                        device_preference: DevicePreference = "auto",
                        eval_device_preference: DevicePreference = "cpu",
                        eval_workers: int = 1) -> dict:
    capacity = json.loads((output / "yaw_capacity/summary.json").read_text())
    bound = min(float(capacity["native_yaw_action_limit_rad"]),
                float(capacity["target_correction_abs_p95_rad"]))
    root = ensure_dir(output / f"seed{seed}/yaw_bound_retrained")
    training = train_motion(seed, output, device_preference, bound)
    scores = {}
    for condition in ("static", "dynamic"):
        reference = list(csv.DictReader((output /
            f"seed{seed}/onpolicy_motion_retraining/{condition}_after_per_episode.csv").open()))
        rows, summary = official_evaluate(seed, condition, output,
            eval_device_preference, eval_workers, root / "selected.pt")
        if [r["episode_id"] for r in rows] != [r["episode_id"] for r in reference]:
            raise AssertionError("Retrained yaw bound is not paired")
        write_csv(root / f"{condition}_per_episode.csv", rows)
        residual._write(root / f"{condition}_summary.json", summary)
        scores[condition] = summary
    result = {"training": training, "evaluation": scores,
              "bound_rad": bound,
              "aperture_sha256": sha(paths(seed, output)["aperture"])}
    residual._write(root / "summary.json", result)
    return result


def train_yaw_spatial(seed: int = 2811, output: Path = OUTPUT,
                      device_preference: DevicePreference = "auto",
                      eval_device_preference: DevicePreference = "cpu",
                      eval_workers: int = 1) -> dict:
    """One fixed, small yaw-specific readout with motion/aperture completely frozen."""
    if seed not in SEEDS:
        raise ValueError(seed)
    root = ensure_dir(output / f"seed{seed}/yaw_spatial_readout")
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task_config = task(seed, device_preference)
    motion_checkpoint = paths(seed, output)["updated_motion"]
    combined = load_combined(seed, device, motion_checkpoint, output)
    source = dynamic.OUTPUT / f"seed{seed}/training"
    collections = {
        "train": [source / "expert.pt", source / "policy_mlp.pt",
                  output / f"seed{seed}/onpolicy_motion_retraining/current_aperture_policy.pt"],
        "validation": [source / "validation.pt"]}
    features, targets, identities = {}, {}, {}
    with torch.no_grad():
        for split, filenames in collections.items():
            xs, ys, ids = [], [], []
            for filename in filenames:
                for ep in torch.load(filename, map_location="cpu", weights_only=False):
                    state = ep["states"][None].to(device)
                    points = ep["points"][None].to(device)
                    previous = ep["previous"][None].to(device)
                    action, _, _, _, _ = combined.motion.forward_sequence(
                        state, points, previous)
                    mask = ep["mask"].to(device).bool()
                    xs.append(yaw_spatial_features(state, points, action[..., 3])[0][mask])
                    ys.append(torch.stack((action[0, mask, 3],
                                           ep["targets"].to(device)[mask, 3]), dim=-1))
                    ids.append(ep["identity"])
            features[split] = torch.cat(xs)
            targets[split] = torch.cat(ys)
            identities[split] = set(ids)
    if identities["train"] & identities["validation"]:
        raise AssertionError("Yaw readout episode leakage")
    mean = features["train"].mean(0, keepdim=True)
    scale = features["train"].std(0, keepdim=True).clamp_min(1e-5)
    x = {split: (value-mean)/scale for split, value in features.items()}
    capacity = json.loads((output / "yaw_capacity/summary.json").read_text())
    bound = min(task_config.max_delta_rotation,
                capacity["target_correction_abs_p95_rad"])
    set_seed(seed + 820_000)
    head = YawSpatialHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=task_config.learning_rate,
                                  weight_decay=task_config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed + 820_000)
    best, best_step, stale, curve = math.inf, 0, 0, []
    started = time.perf_counter()
    for step in range(1, 1601):
        indices = torch.randint(len(x["train"]), (256,), generator=generator).to(device)
        raw = head(x["train"][indices])
        prediction = (targets["train"][indices, 0] + bound*torch.tanh(raw)).clamp(
            -task_config.max_delta_rotation, task_config.max_delta_rotation)
        loss = ((prediction-targets["train"][indices, 1]) /
                task_config.max_delta_rotation).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            with torch.no_grad():
                val = (targets["validation"][:, 0] + bound*torch.tanh(
                    head(x["validation"]))).clamp(-task_config.max_delta_rotation,
                    task_config.max_delta_rotation)
                score = float(((val-targets["validation"][:, 1]) /
                               task_config.max_delta_rotation).square().mean())
            curve.append({"update": step, "train_normalized_yaw_mse": float(loss.detach()),
                          "validation_normalized_yaw_mse": score})
            if score < best - 1e-6:
                best, best_step, stale = score, step, 0
                torch.save({"model_state": head.state_dict(), "feature_mean": mean.cpu(),
                            "feature_scale": scale.cpu(), "correction_bound_rad": bound,
                            "motion_checkpoint_sha256": sha(motion_checkpoint),
                            "aperture_checkpoint_sha256": sha(paths(seed, output)["aperture"]),
                            "selected_update": step,
                            "topology": "canonical_32_points_100_edges_unchanged"},
                           root / "selected.pt")
            else:
                stale += 1
            if step >= 500 and stale >= 24:
                break
    write_csv(root / "learning_curve.csv", curve)
    scores = {}
    for condition in ("static", "dynamic"):
        rows, summary = official_evaluate(seed, condition, output,
            eval_device_preference, eval_workers, motion_checkpoint,
            yaw_head_checkpoint=root / "selected.pt")
        write_csv(root / f"{condition}_per_episode.csv", rows)
        residual._write(root / f"{condition}_summary.json", summary)
        scores[condition] = summary
    result = {"train_episodes": len(identities["train"]),
              "validation_episodes": len(identities["validation"]),
              "train_timesteps": len(x["train"]),
              "validation_timesteps": len(x["validation"]),
              "selected_update": best_step, "final_update": step,
              "best_validation_normalized_yaw_mse": best,
              "correction_bound_rad": bound,
              "training_seconds": time.perf_counter()-started,
              "device": str(device),
              "motion_checkpoint_sha256": sha(motion_checkpoint),
              "aperture_checkpoint_sha256": sha(paths(seed, output)["aperture"]),
              "evaluation": scores,
              "input": "canonical observable state and permutation-invariant surface moments including x-z covariance; no point IDs"}
    residual._write(root / "summary.json", result)
    return result


def motion_failure_analysis(output: Path = OUTPUT,
                            eval_device_preference: DevicePreference = "cpu") -> dict:
    """Matched fixed-horizon traces for trajectory, position and physical-contact review."""
    root = ensure_dir(output / "paired_trajectory_analysis")
    device = select_device(eval_device_preference)
    task_config = task(2811, eval_device=eval_device_preference)
    before = load_combined(2811, device, output=output)
    adapted = load_combined(2811, device, paths(2811, output)["updated_motion"], output)
    larger = load_combined(2811, device,
        output / "seed2811/yaw_bound_retrained/selected.pt", output)
    controllers = {"old_5d": before.motion, "aperture_baseline": before,
                   "motion_adapted": adapted, "yaw_bound_retrained": larger}
    env = sf.make_env(task_config, 2811 + 864_211)
    all_rows, outcomes = [], []
    try:
        for condition in ("static", "dynamic"):
            specs = (dynamic.dynamic_specs(2811, task_config, 16,
                dynamic.SELECTED_STEP_SPEEDS_M, "heldout") if condition == "dynamic"
                else sf.sample_episode_specs(16, 2811 + 60_000, task_config, "iid"))
            for spec in specs:
                for name, controller in controllers.items():
                    trace, outcome = fixed_horizon_trace(env, spec, task_config,
                        controller, device, condition, name)
                    all_rows.extend(trace)
                    outcomes.append({**outcome, "condition": condition,
                                     "controller": name})
    finally:
        env.close()
    write_csv(root / "fixed_horizon_traces.csv", all_rows)
    write_csv(root / "fixed_horizon_outcomes.csv", outcomes)
    by_run = {}
    for row in all_rows:
        by_run.setdefault((row["condition"], row["episode_id"], row["controller"]), []).append(row)
    contact_rows = []
    for key, trace in by_run.items():
        previous = None
        for row in trace:
            if row["collision"] and (previous is None or not previous["collision"]):
                prior = previous or row
                yaw_high = prior["yaw_error"] >= task_config.success_rotation_threshold
                position_high = prior["position_error"] >= task_config.success_position_threshold
                for contact in row["contacts"]:
                    robot_geom = contact["robot_geom"].lower()
                    if "tip" in robot_geom or "finger" in robot_geom or "thumb" in robot_geom:
                        category = "fingertip_contact"
                    elif yaw_high and not position_high:
                        category = "yaw_associated_contact"
                    elif position_high and not yaw_high:
                        category = "translation_associated_contact"
                    else:
                        category = "mixed_or_unresolved_contact"
                    contact_rows.append({"condition": key[0], "episode_id": key[1],
                        "controller": key[2], "timestep": row["timestep"],
                        "robot_geom": contact["robot_geom"],
                        "surface_geom": contact["surface_geom"],
                        "contact_distance_m": contact["distance"],
                        "prior_position_error_m": prior["position_error"],
                        "prior_yaw_error_rad": prior["yaw_error"],
                        "prior_opening_width_m": prior["opening_width_m"],
                        "prior_gripper_action": prior["gripper_action"],
                        "prior_clearance_m": prior["clearance"],
                        "descriptive_category": category})
            previous = row
    if contact_rows:
        write_csv(root / "contact_onsets.csv", contact_rows)
    official = {}
    for name, path in (("aperture_baseline", output /
        "reproduction/seed2811/dynamic_per_episode.csv"),
        ("motion_adapted", output /
        "seed2811/onpolicy_motion_retraining/dynamic_after_per_episode.csv"),
        ("yaw_bound_retrained", output /
        "seed2811/yaw_bound_retrained/dynamic_per_episode.csv")):
        official[name] = {r["episode_id"]: r for r in csv.DictReader(path.open())}
    position_rows = []
    for episode_id in official["aperture_baseline"]:
        names = ("aperture_baseline", "motion_adapted", "yaw_bound_retrained")
        records = {name: official[name][episode_id] for name in names}
        traces = {name: by_run[("dynamic", episode_id, name)] for name in names}
        before_fail = float(records["aperture_baseline"]["position_error"]) >= \
            task_config.success_position_threshold
        final_fail = float(records["yaw_bound_retrained"]["position_error"]) >= \
            task_config.success_position_threshold
        if not (before_fail or final_fail):
            continue
        def stats(trace: list[dict]) -> dict:
            error = np.asarray([r["position_error"] for r in trace])
            return {"initial_error_m": float(error[0]),
                    "peak_error_m": float(error.max()),
                    "minimum_error_m": float(error.min()),
                    "first_within_25mm_step": next((r["timestep"] for r in trace if
                        r["position_error"] < task_config.success_position_threshold), None),
                    "initial_translation_action": trace[0]["translation_action"],
                    "initial_yaw_action": trace[0]["yaw_action"],
                    "opening_at_step_5": trace[min(5, len(trace)-1)]["opening_width_m"],
                    "last_yaw_error_rad": trace[-1]["yaw_error"]}
        position_rows.append({"episode_id": episode_id,
            "baseline_position_fail": before_fail, "final_position_fail": final_fail,
            **{name + "_final_position_m": float(records[name]["position_error"])
               for name in names},
            **{name + "_stats": stats(traces[name]) for name in names}})
    if position_rows:
        write_csv(root / "position_failure_traces.csv", position_rows)
    result = {"traced_episodes_per_condition": 16,
              "controllers": list(controllers),
              "fixed_horizon_timesteps": len(all_rows),
              "contact_onsets": len(contact_rows),
              "position_cases": len(position_rows),
              "note": "Static fixed-horizon diagnostic continues after possible official early success; official outcome CSVs remain authoritative.",
              "contact_categories": {category: sum(r["descriptive_category"] == category
                   for r in contact_rows) for category in sorted(set(
                       r["descriptive_category"] for r in contact_rows))}}
    residual._write(root / "summary.json", result)
    return result


def final_trace_and_contacts(seed: int = 2811, output: Path = OUTPUT,
                             eval_device_preference: DevicePreference = "cpu") -> dict:
    """One event per physical collision onset for the frozen final spatial controller."""
    root = ensure_dir(output / f"seed{seed}/final_diagnostics")
    device = select_device(eval_device_preference)
    task_config = task(seed, eval_device=eval_device_preference)
    combined = load_combined(seed, device, paths(seed, output)["updated_motion"], output)
    controller = YawSpatialController(combined, output /
        f"seed{seed}/yaw_spatial_readout/selected.pt", device)
    traces, outcomes, events = [], [], []
    env = sf.make_env(task_config, seed + 864_211)
    try:
        for condition in ("static", "dynamic"):
            specs = (dynamic.dynamic_specs(seed, task_config, 16,
                dynamic.SELECTED_STEP_SPEEDS_M, "heldout") if condition == "dynamic"
                else sf.sample_episode_specs(16, seed + 60_000, task_config, "iid"))
            for spec in specs:
                rows, outcome = fixed_horizon_trace(env, spec, task_config,
                    controller, device, condition, "final_yaw_spatial")
                traces.extend(rows)
                outcomes.append({**outcome, "condition": condition})
                previous = None
                for row in rows:
                    if row["collision"] and (previous is None or not previous["collision"]):
                        prior = previous or row
                        yaw_high = prior["yaw_error"] >= task_config.success_rotation_threshold
                        position_high = prior["position_error"] >= task_config.success_position_threshold
                        closing = prior["gripper_action"] < -1e-4
                        if yaw_high and position_high:
                            category = "mixed_yaw_translation"
                        elif yaw_high:
                            category = "yaw_associated"
                        elif position_high:
                            category = "translation_associated"
                        elif closing:
                            category = "closing_associated"
                        else:
                            category = "near_target_contact_unresolved"
                        events.append({"episode_id": outcome["episode_id"],
                            "condition": condition, "timestep": row["timestep"],
                            "robot_geoms": sorted(set(c["robot_geom"] for c in row["contacts"])),
                            "surface_geoms": sorted(set(c["surface_geom"] for c in row["contacts"])),
                            "prior_position_error_m": prior["position_error"],
                            "prior_yaw_error_rad": prior["yaw_error"],
                            "prior_opening_width_m": prior["opening_width_m"],
                            "prior_gripper_action": prior["gripper_action"],
                            "prior_clearance_m": prior["clearance"],
                            "category_evidence_flag": category})
                    previous = row
                if outcome["collision"] and not any(e["episode_id"] == outcome["episode_id"]
                    and e["condition"] == condition for e in events):
                    events.append({"episode_id": outcome["episode_id"],
                        "condition": condition, "timestep": task_config.max_steps,
                        "robot_geoms": sorted(set(c["robot_geom"] for c in outcome["final_contacts"])),
                        "surface_geoms": sorted(set(c["surface_geom"] for c in outcome["final_contacts"])),
                        "prior_position_error_m": rows[-1]["position_error"],
                        "prior_yaw_error_rad": rows[-1]["yaw_error"],
                        "prior_opening_width_m": rows[-1]["opening_width_m"],
                        "prior_gripper_action": rows[-1]["gripper_action"],
                        "prior_clearance_m": rows[-1]["clearance"],
                        "category_evidence_flag": "final_step_unresolved"})
    finally:
        env.close()
    write_csv(root / "fixed_horizon_traces.csv", traces)
    write_csv(root / "fixed_horizon_outcomes.csv", outcomes)
    if events:
        write_csv(root / "collision_events.csv", events)
    # Dynamic official and fixed-horizon rollouts have identical horizon and EpisodeSpecs.
    official = {r["episode_id"]: r for r in csv.DictReader((output /
        f"seed{seed}/yaw_spatial_readout/dynamic_per_episode.csv").open())}
    dynamic_outcomes = [r for r in outcomes if r["condition"] == "dynamic"]
    for r in dynamic_outcomes:
        prior = official[r["episode_id"]]
        if r["collision"] != (prior["collision"] == "True") or \
           abs(r["final_position_error"]-float(prior["position_error"])) > 1e-5:
            raise AssertionError("Diagnostic dynamic trace diverged from official evaluation")
    result = {"seed": seed, "traces": len(traces),
              "dynamic_official_parity": True,
              "collision_episodes_dynamic": sum(r["collision"] for r in dynamic_outcomes),
              "collision_episodes_static_fixed_horizon": sum(r["collision"] for r in outcomes
                  if r["condition"] == "static"),
              "contact_onset_events": len(events),
              "event_categories": {name: sum(e["category_evidence_flag"] == name for e in events)
                  for name in sorted(set(e["category_evidence_flag"] for e in events))},
              "note": "Categories are evidence flags from pre-contact state, not proven causal attributions; static fixed horizon continues after official success."}
    residual._write(root / "summary.json", result)
    return result


def aggregate_results(output: Path = OUTPUT) -> dict:
    root = ensure_dir(output / "aggregate")
    locations = {"aperture_baseline": ("reproduction/seed{seed}", "{condition}_per_episode.csv"),
                 "motion_adapted": ("seed{seed}/onpolicy_motion_retraining",
                                    "{condition}_after_per_episode.csv"),
                 "larger_yaw_bound": ("seed{seed}/yaw_bound_retrained",
                                      "{condition}_per_episode.csv"),
                 "yaw_spatial_readout": ("seed{seed}/yaw_spatial_readout",
                                         "{condition}_per_episode.csv")}
    task_config = task(2811)
    totals, metric_rows, final_rows, failures, transitions = {}, [], [], [], []
    for condition in ("static", "dynamic"):
        totals[condition] = {}
        for name, (folder, filename) in locations.items():
            all_rows = []
            for seed in SEEDS:
                path = output / folder.format(seed=seed) / filename.format(condition=condition)
                original = list(csv.DictReader(path.open()))
                rows = [{"seed": seed, "condition": condition, "controller": name,
                         "episode_id": r["episode_id"],
                         "success": r["success"] == "True",
                         "collision": r["collision"] == "True",
                         "position_error": float(r["position_error"]),
                         "yaw_error": float(r["yaw_error"]),
                         "aperture_error": float(r["aperture_error"]),
                         "minimum_clearance": float(r["minimum_clearance"]),
                         "inference_latency_ms": float(r["inference_latency_ms"])}
                        for r in original]
                all_rows.extend(rows)
                summary = _summary(rows, task_config)
                metric_rows.append({"seed": seed, "condition": condition,
                                    "controller": name, **summary})
                if name == "yaw_spatial_readout":
                    final_rows.extend(rows)
                    for r in rows:
                        flags = {"position_fail": r["position_error"] >=
                                 task_config.success_position_threshold,
                                 "yaw_fail": r["yaw_error"] >=
                                 task_config.success_rotation_threshold,
                                 "aperture_fail": r["aperture_error"] >=
                                 task_config.success_opening_threshold,
                                 "collision_fail": r["collision"]}
                        failures.append({"seed": seed, "condition": condition,
                            "episode_id": r["episode_id"], **flags,
                            "multiple_failures": sum(flags.values()) > 1,
                            "success": r["success"]})
            totals[condition][name] = _summary(all_rows, task_config)
        for seed in SEEDS:
            folder, filename = locations["aperture_baseline"]
            before = {r["episode_id"]: r for r in csv.DictReader((output /
                folder.format(seed=seed) / filename.format(condition=condition)).open())}
            after = [r for r in final_rows if r["seed"] == seed and r["condition"] == condition]
            if set(before) != {r["episode_id"] for r in after}:
                raise AssertionError("Aggregate compared unmatched EpisodeSpecs")
            for r in after:
                b = before[r["episode_id"]]
                transitions.append({"seed": seed, "condition": condition,
                    "episode_id": r["episode_id"],
                    "before_success": b["success"] == "True",
                    "after_success": r["success"],
                    "before_collision": b["collision"] == "True",
                    "after_collision": r["collision"],
                    "before_position_error_m": float(b["position_error"]),
                    "after_position_error_m": r["position_error"],
                    "before_yaw_error_rad": float(b["yaw_error"]),
                    "after_yaw_error_rad": r["yaw_error"],
                    "before_aperture_error_m": float(b["aperture_error"]),
                    "after_aperture_error_m": r["aperture_error"]})
    write_csv(root / "seed_condition_metrics.csv", metric_rows)
    write_csv(root / "final_full_controller.csv", final_rows)
    write_csv(root / "failure_decomposition.csv", failures)
    write_csv(root / "paired_episode_transitions.csv", transitions)
    position_categories = []
    for seed in SEEDS:
        trace_path = output / f"seed{seed}/final_diagnostics/fixed_horizon_traces.csv"
        traces = {}
        for row in csv.DictReader(trace_path.open()):
            if row["condition"] == "dynamic":
                traces.setdefault(row["episode_id"], []).append(row)
        for outcome in [r for r in final_rows if r["seed"] == seed and
                        r["condition"] == "dynamic" and
                        r["position_error"] >= task_config.success_position_threshold]:
            rows = traces[outcome["episode_id"]]
            errors = [float(r["position_error"]) for r in rows]
            first_within = next((int(r["timestep"]) for r in rows if
                float(r["position_error"]) < task_config.success_position_threshold), None)
            if outcome["collision"]:
                category = "F_contact_interaction"
            elif first_within is not None:
                category = "C_overshoot_after_convergence"
            elif errors[-1] < errors[max(0, len(errors)-5)]:
                category = "B_slow_convergence"
            else:
                category = "G_other_unresolved"
            position_categories.append({"seed": seed,
                "episode_id": outcome["episode_id"], "primary_category": category,
                "initial_position_error_m": errors[0],
                "minimum_position_error_m": min(errors),
                "final_position_error_m": outcome["position_error"],
                "first_within_25mm_step": first_within,
                "collision": outcome["collision"],
                "final_yaw_error_rad": outcome["yaw_error"],
                "note": "Trajectory-based descriptive category, not a causal proof"})
    write_csv(root / "position_failure_categories.csv", position_categories)
    paired = {condition: {"gained": sum(r["condition"] == condition and
                                   not r["before_success"] and r["after_success"] for r in transitions),
                          "lost": sum(r["condition"] == condition and
                                 r["before_success"] and not r["after_success"] for r in transitions)}
              for condition in ("static", "dynamic")}
    provenance = {seed: {"starting_motion_sha256": sha(paths(seed, output)["motion"]),
        "adapted_motion_sha256": sha(paths(seed, output)["updated_motion"]),
        "aperture_branch_sha256": sha(paths(seed, output)["aperture"]),
        "yaw_readout_sha256": sha(output /
            f"seed{seed}/yaw_spatial_readout/selected.pt"),
        "base_ff_sha256": sha(residual.BASE_CHECKPOINT)} for seed in SEEDS}
    residual._write(root / "three_seed_totals.json", totals)
    residual._write(root / "paired_success_summary.json", paired)
    residual._write(root / "checkpoint_provenance.json", provenance)
    return {"totals": totals, "paired": paired, "provenance": provenance,
            "position_categories": position_categories}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("reproduce", "distribution", "collect", "train-motion",
                                          "compare-motion", "yaw-semantics", "yaw-probes",
                                          "yaw-counterfactual", "yaw-capacity", "yaw-coverage",
                                          "yaw-bound", "yaw-bound-retrain", "yaw-spatial",
                                          "failure-analysis", "final-traces", "aggregate"))
    parser.add_argument("--seed", type=int, default=2811)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "reproduce":
        reproduce(args.output, args.eval_device, args.eval_workers)
    elif args.phase == "distribution":
        distribution_shift(args.seed, args.output, args.eval_device)
    elif args.phase == "collect":
        collect_current_policy(args.seed, args.output, args.eval_device)
    elif args.phase == "train-motion":
        train_motion(args.seed, args.output, args.device)
    elif args.phase == "yaw-semantics":
        yaw_semantics(args.output)
    elif args.phase == "yaw-probes":
        yaw_probes(args.output, args.device)
    elif args.phase == "yaw-counterfactual":
        yaw_counterfactual(args.output, args.eval_device)
    elif args.phase == "yaw-capacity":
        yaw_capacity(args.output, args.eval_device)
    elif args.phase == "yaw-coverage":
        yaw_coverage(args.output)
    elif args.phase == "yaw-bound":
        yaw_bound_ablation(args.seed, args.output, args.eval_device, args.eval_workers)
    elif args.phase == "yaw-bound-retrain":
        yaw_bound_retrained(args.seed, args.output, args.device, args.eval_device,
                            args.eval_workers)
    elif args.phase == "yaw-spatial":
        train_yaw_spatial(args.seed, args.output, args.device, args.eval_device,
                          args.eval_workers)
    elif args.phase == "failure-analysis":
        motion_failure_analysis(args.output, args.eval_device)
    elif args.phase == "final-traces":
        final_trace_and_contacts(args.seed, args.output, args.eval_device)
    elif args.phase == "aggregate":
        aggregate_results(args.output)
    else:
        compare_retrained(args.seed, args.output, args.eval_device, args.eval_workers)


if __name__ == "__main__":
    main()
