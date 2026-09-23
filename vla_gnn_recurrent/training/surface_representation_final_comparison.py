"""Final paired spatial-representation comparison after FF DAgger stabilization.

This runner reuses the four existing feed-forward encoders, task geometry,
MuJoCo action application, and the Surface Set DAgger Round 1 artifacts.  It
does not introduce temporal state or a new spatial architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from torch import nn

from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_pregrasp_onpolicy_relabel as dagger
from vla_gnn_recurrent.training import surface_pregrasp_stabilization as stab
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


DEFAULT_OUTPUT = "artifacts/surface_representation_final_comparison"
STABILIZATION_ROOT = Path("artifacts/surface_pregrasp_stabilization/recovery_training")
DAGGER_ROOT = Path("artifacts/surface_pregrasp_onpolicy_relabel")
BASE_FEASIBILITY_ROOT = Path("artifacts/surface_graph_feasibility")
SEEDS = (2811, 2812, 2813)
UPDATES = 800
BATCH_SIZE = 128
MIX_EXPERT_FRACTION = 0.5
SEVERE_PENETRATION_M = -0.020
PAIR_COMPARISONS = (
    ("centerline_set", "surface_set", "centerline_vs_surface"),
    ("surface_set", "surface_graph", "surface_vs_graph"),
    ("surface_graph_no_local", "surface_graph", "graph_vs_no_local"),
)
FAILURE_BITS = ("collision_failure", "position_failure", "orientation_failure", "aperture_failure")
METRICS = (
    "success", "collision", "final_position_error", "final_orientation_error",
    "final_gripper_width_error", "trajectory_error", "trajectory_yaw_error",
    "trajectory_aperture_error", "trajectory_length", "minimum_safe_clearance",
)


def write_json(path: str | Path, payload: Any) -> None:
    sf._write_json(path, payload)


def state_dict_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def spec_signature(spec: sf.SurfaceEpisodeSpec) -> str:
    return json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"), default=list)


def assert_same_specs(per_model_specs: dict[str, list[sf.SurfaceEpisodeSpec]]) -> None:
    if not per_model_specs:
        raise ValueError("At least one model EpisodeSpec list is required.")
    signatures = {name: [spec_signature(s) for s in specs] for name, specs in per_model_specs.items()}
    reference = next(iter(signatures.values()))
    if any(items != reference for items in signatures.values()):
        raise ValueError("Paired models must use exactly the same ordered physical EpisodeSpecs.")


def validate_train_specs(specs: list[sf.SurfaceEpisodeSpec]) -> None:
    if not specs or any(spec.condition != "train" for spec in specs):
        raise ValueError("DAgger collection accepts training-split EpisodeSpecs only.")


def failure_flags(row: dict[str, Any], config: sf.SurfaceFeasibilityConfig) -> dict[str, bool]:
    """Multi-label failures matching the four-way success conjunction."""
    return {
        "collision_failure": bool(row.get("trajectory_collision", row.get("collision", False))),
        "position_failure": float(row["final_position_error"]) >= config.success_position_threshold,
        "orientation_failure": float(row["final_orientation_error"]) >= config.success_rotation_threshold,
        "aperture_failure": float(row["final_gripper_width_error"]) >= config.success_opening_threshold,
    }


def failure_decomposition(rows: list[dict[str, Any]], config: sf.SurfaceFeasibilityConfig) -> dict[str, Any]:
    flags = [failure_flags(row, config) for row in rows]
    failures = [any(item.values()) for item in flags]
    combos: dict[str, dict[str, Any]] = {}
    for bits in itertools.product((False, True), repeat=len(FAILURE_BITS)):
        names = [name for name, active in zip(FAILURE_BITS, bits, strict=True) if active]
        label = "+".join(names) if names else "success_all_conditions_pass"
        count = sum(tuple(item[name] for name in FAILURE_BITS) == bits for item in flags)
        combos[label] = {
            "conditions": names, "count": count,
            "fraction_all_episodes": count / max(len(rows), 1),
            "fraction_failed_episodes": count / max(sum(failures), 1),
        }
    per_condition: dict[str, Any] = {}
    for name in FAILURE_BITS:
        count = sum(item[name] for item in flags)
        per_condition[name] = {
            "episode_count": int(count), "fraction_all_episodes": count / max(len(rows), 1),
            "fraction_failed_episodes": count / max(sum(failures), 1),
        }
    margins: list[dict[str, Any]] = []
    for row, condition_flags in zip(rows, flags, strict=True):
        violations = {
            "position": float(row["final_position_error"]) / config.success_position_threshold,
            "orientation": float(row["final_orientation_error"]) / config.success_rotation_threshold,
            "aperture": float(row["final_gripper_width_error"]) / config.success_opening_threshold,
        }
        ranked = sorted(violations.items(), key=lambda kv: kv[1], reverse=True)
        margins.append({
            "episode_id": int(row["episode_id"]),
            "position_margin_m": float(row["final_position_error"] - config.success_position_threshold),
            "orientation_margin_rad": float(row["final_orientation_error"] - config.success_rotation_threshold),
            "aperture_margin_m": float(row["final_gripper_width_error"] - config.success_opening_threshold),
            "normalized_violations": violations,
            "dominant_continuous_violation": ranked[0][0],
            "dominant_normalized_violation": ranked[0][1],
            "second_continuous_violation": ranked[1][0],
            "second_normalized_violation": ranked[1][1],
            **condition_flags,
            "trajectory_collision": bool(row.get("trajectory_collision", row.get("collision", False))),
            "final_position_error_m": float(row["final_position_error"]),
            "final_orientation_error_rad": float(row["final_orientation_error"]),
            "final_aperture_error_m": float(row["final_gripper_width_error"]),
            "failure_conditions": [key for key in FAILURE_BITS if condition_flags[key]],
        })
    failed_count = sum(failures)
    aperture_only = combos["aperture_failure"]["count"]
    return {
        "episodes": len(rows), "failed_episodes": int(failed_count),
        "success_rate": float(1.0 - failed_count / max(len(rows), 1)),
        "thresholds": {
            "position_m_strictly_below": config.success_position_threshold,
            "orientation_rad_strictly_below": config.success_rotation_threshold,
            "aperture_m_strictly_below": config.success_opening_threshold,
            "trajectory_collision_required_false": True,
        },
        "failure_conditions": per_condition, "all_16_failure_combinations": combos,
        "aperture_only_failed_fraction": aperture_only / max(failed_count, 1),
        "aperture_bottleneck_warning": bool(aperture_only / max(failed_count, 1) >= 0.25),
        "threshold_margins": margins,
    }


def _point_array(data: dict[str, Any], name: str) -> torch.Tensor:
    key = "centerline_points" if name == "centerline_set" else "surface_points"
    if key not in data:
        raise KeyError(f"Dataset missing {key}, needed by {name}.")
    return data[key]


def _checkpoint_config_compatible(checkpoint: Path, config: sf.SurfaceFeasibilityConfig,
                                  model_name: str, seed: int) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = asdict(config)
    actual = payload.get("config", {})
    ignored = {"output_dir", "device", "eval_device", "latency_warmup", "latency_iterations"}
    mismatches = {key: (actual.get(key), value) for key, value in expected.items()
                  if key not in ignored and actual.get(key) != value}
    if payload.get("model_name") != model_name or payload.get("seed") != seed:
        mismatches["checkpoint_identity"] = (payload.get("model_name"), payload.get("seed"))
    if mismatches:
        raise ValueError(f"Base checkpoint configuration mismatch for {checkpoint}: {mismatches}")
    return payload


def _current_datasets(seed: int, config: sf.SurfaceFeasibilityConfig) -> tuple[dict[str, Any], dict[str, Any], list[sf.SurfaceEpisodeSpec], list[sf.SurfaceEpisodeSpec]]:
    base = STABILIZATION_ROOT / f"seed{seed}" / "dataset"
    train_data = torch.load(base / "expert" / "expert_states.pt", map_location="cpu", weights_only=False)
    val_data = torch.load(base / "validation" / "expert_states.pt", map_location="cpu", weights_only=False)
    train_specs = [stab._as_spec(x) for x in train_data["specs"]]
    val_specs = [stab._as_spec(x) for x in val_data["specs"]]
    validate_train_specs(train_specs)
    expected_train = sf.sample_episode_specs(config.train_episodes, seed, config, "train")
    expected_val = sf.sample_episode_specs(config.validation_episodes, seed + 10_000, config, "validation")
    if [spec_signature(s) for s in train_specs] != [spec_signature(s) for s in expected_train]:
        raise ValueError(f"Seed {seed} shared D_expert is not the canonical training EpisodeSpec set.")
    if [spec_signature(s) for s in val_specs] != [spec_signature(s) for s in expected_val]:
        raise ValueError(f"Seed {seed} validation expert split does not match the shared protocol.")
    return train_data, val_data, train_specs, val_specs


def _make_eval_specs(seed: int, config: sf.SurfaceFeasibilityConfig) -> dict[str, tuple[list[sf.SurfaceEpisodeSpec], int, bool]]:
    iid = sf.sample_episode_specs(config.eval_episodes, seed + 60_000, config, "iid")
    dagger_iid_path = DAGGER_ROOT / "evaluation" / "iid" / f"seed{seed}" / "paired_episode_results.json"
    dagger_iid = json.loads(dagger_iid_path.read_text(encoding="utf-8"))["same_episode_specs"]
    if [spec_signature(s) for s in iid] != [spec_signature(stab._as_spec(x)) for x in dagger_iid]:
        raise ValueError(f"Seed {seed} DAgger held-out IID EpisodeSpecs do not match deterministic regeneration.")
    shape_ood = sf.sample_episode_specs(config.eval_episodes, seed + 61_000, config, "surface_shape_ood", ood=True)
    sampling = sf.sample_episode_specs(config.eval_episodes, seed + 62_000, config, "sampling_ood")
    # Prove that regenerated OOD physical shapes are the prior fixed evaluation set.
    for condition, specs in (("surface_shape_ood", shape_ood), ("sampling_ood", sampling)):
        prior_path = BASE_FEASIBILITY_ROOT / "eval" / condition / f"seed{seed}" / "episode_results.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        prior_shapes = [json.dumps(item["shape"], sort_keys=True) for item in prior["episodes"]]
        current_shapes = [json.dumps(json.loads(json.dumps(asdict(s.shape), default=list)), sort_keys=True) for s in specs]
        if current_shapes != prior_shapes:
            raise ValueError(f"Seed {seed} {condition} EpisodeSpecs differ from the existing paired evaluation.")
    return {
        "iid": (iid, sf.SURFACE_POINTS, False),
        "surface_shape_ood": (shape_ood, sf.SURFACE_POINTS, False),
        "sampling_ood": (sampling, 16, True),
    }


def _tips_world(env: sf.SurfaceManipulatorEnv) -> np.ndarray:
    return np.stack([
        np.mean([env.data.geom_xpos[env._geom_id(name)] for name in ("thumbtip1", "thumbtip2")], axis=0),
        np.mean([env.data.geom_xpos[env._geom_id(name)] for name in ("fingertip1", "fingertip2")], axis=0),
    ]).astype(np.float64)


def _action_for_mode(model_name: sf.SurfaceModelName, model: nn.Module, state: np.ndarray,
                     points: np.ndarray, config: sf.SurfaceFeasibilityConfig,
                     device: torch.device, cached_pairs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    if model_name not in ("surface_graph", "surface_graph_no_local") or cached_pairs is None:
        return sf._policy_action(model_name, model, state, points, config, device), cached_pairs
    topology_np = sf.build_graph_topology_numpy(points, state[8:14].reshape(2, 3), config,
                                                use_local_edges=True, cached_surface_pairs=cached_pairs)
    topology = sf.topology_from_numpy(topology_np, device)
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, sf.STATE_DIM)
    points_t = torch.as_tensor(points, dtype=torch.float32, device=device).reshape(1, -1, 3)
    with torch.no_grad():
        output = model(state_t, points_t, topology)[0].detach().cpu().numpy()
    translation = sf.clamp_delta(torch.as_tensor(output[:3], dtype=torch.float32), config.max_delta_ee).numpy()
    action = np.asarray([translation[0], 0.0, translation[2],
                         np.clip(output[3], -config.max_delta_rotation, config.max_delta_rotation),
                         np.clip(output[4], -config.max_delta_gripper, config.max_delta_gripper)], dtype=np.float32)
    return action, cached_pairs


def rollout_detailed(env: sf.SurfaceManipulatorEnv, spec: sf.SurfaceEpisodeSpec,
                     config: sf.SurfaceFeasibilityConfig, model_name: sf.SurfaceModelName,
                     model: nn.Module, device: torch.device, point_count: int = sf.SURFACE_POINTS,
                     nonuniform: bool = False, topology_mode: str = "rebuilt") -> dict[str, Any]:
    if topology_mode not in ("rebuilt", "cached"):
        raise ValueError(topology_mode)
    if topology_mode == "cached" and model_name != "surface_graph":
        raise ValueError("Only the full Surface Graph has a reusable local surface topology.")
    sf._reset_surface_env(env, spec)
    initial = env.robot_observation().ee_position.numpy().astype(np.float64)
    target = sf._surface_target(spec.shape, initial, config.pregrasp_clearance)
    positions = [initial.tolist()]
    tips = [_tips_world(env).tolist()]
    errors = sf._state_errors(env, spec.shape, config, target)
    trajectory_pos = [errors["position_error"]]
    trajectory_yaw = [errors["orientation_error"]]
    trajectory_aperture = [errors["gripper_width_error"]]
    contacts, min_clearance = sf._surface_distance_metrics(env, spec.shape)
    collision_seen = bool(contacts)
    collision_count = len(contacts)
    penetration_count = sum(row["distance"] < -1e-4 for row in contacts)
    first_collision = 0 if contacts else None
    steps_to_success = 0 if sf._success_from_errors(errors, collision_seen, config) else None
    cached_pairs: np.ndarray | None = None
    cached_identity: str | None = None
    prediction_actions: list[list[float]] = []
    for step in range(config.max_steps):
        if steps_to_success is not None:
            break
        state, surface, centerline = sf.observation_inputs(
            env, spec.shape, point_count, sf._stable_seed(spec.sample_identity), nonuniform,
        )
        points = sf.policy_geometry_points(model_name, surface, centerline)
        if topology_mode == "cached":
            current_identity = spec.sample_identity
            if not sf.cached_topology_valid(cached_identity, current_identity):
                cached_pairs = None
            if cached_pairs is None:
                payload = sf.build_graph_topology_numpy(points, state[8:14].reshape(2, 3), config, True)
                cached_pairs = payload["surface_pairs"]
                cached_identity = current_identity
            action, cached_pairs = _action_for_mode(model_name, model, state, points, config, device, cached_pairs)
        else:
            action = sf._policy_action(model_name, model, state, points, config, device)
        prediction_actions.append(action.tolist())
        sf.apply_local_action(env, action, config)
        current = env.robot_observation().ee_position.numpy().astype(np.float64)
        positions.append(current.tolist())
        tips.append(_tips_world(env).tolist())
        errors = sf._state_errors(env, spec.shape, config, target)
        trajectory_pos.append(errors["position_error"])
        trajectory_yaw.append(errors["orientation_error"])
        trajectory_aperture.append(errors["gripper_width_error"])
        contacts, clearance = sf._surface_distance_metrics(env, spec.shape)
        if contacts and first_collision is None:
            first_collision = step + 1
        collision_count += len(contacts)
        penetration_count += sum(row["distance"] < -1e-4 for row in contacts)
        collision_seen = collision_seen or bool(contacts)
        min_clearance = min(min_clearance, clearance)
        if sf._success_from_errors(errors, collision_seen, config):
            steps_to_success = step + 1
    final_contacts, final_clearance = sf._surface_distance_metrics(env, spec.shape)
    final_errors = sf._state_errors(env, spec.shape, config, target)
    min_clearance = min(min_clearance, final_clearance)
    position_path = np.asarray(positions, dtype=np.float64)
    length = float(np.linalg.norm(np.diff(position_path[:, [0, 2]], axis=0), axis=1).sum()) if len(position_path) > 1 else 0.0
    return {
        "episode_id": spec.episode_id, "condition": spec.condition, "shape_id": spec.shape.shape_id,
        "success": bool(sf._success_from_errors(final_errors, collision_seen, config)),
        "collision": collision_seen, "trajectory_collision": collision_seen,
        "final_collision": bool(final_contacts), "first_collision_timestep": first_collision,
        "minimum_safe_clearance": float(min_clearance), "collision_count": int(collision_count),
        "penetration_count": int(penetration_count), "illegal_contact_count": int(collision_count),
        "final_position_error": final_errors["position_error"],
        "final_orientation_error": final_errors["orientation_error"],
        "final_gripper_width_error": final_errors["gripper_width_error"],
        "final_gripper_width": sf.gripper_width(env), "target_gripper_width": target[3],
        "final_yaw": sf.tool_yaw(env), "trajectory_error": float(np.mean(trajectory_pos)),
        "trajectory_yaw_error": float(np.mean(trajectory_yaw)),
        "trajectory_aperture_error": float(np.mean(trajectory_aperture)),
        "trajectory_length": length,
        "steps_to_convergence": int(steps_to_success if steps_to_success is not None else config.max_steps),
        "steps_until_collision": first_collision, "trajectory": positions,
        "fingertip_trajectories": tips, "policy_actions": prediction_actions,
        "target_position": target[0].tolist(), "target_normal": target[1].tolist(),
        "target_yaw": target[2], "target_side": target[4], "initial_ee": initial.tolist(),
        "shape": asdict(spec.shape), "spec": asdict(spec), "topology_mode": topology_mode,
    }


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int = 5000) -> list[float]:
    return sf._bootstrap_ci(values, samples, seed)


def paired_comparison(reference: list[dict[str, Any]], candidate: list[dict[str, Any]],
                      config: sf.SurfaceFeasibilityConfig, label: str, seed: int) -> dict[str, Any]:
    if len(reference) != len(candidate) or [x["episode_id"] for x in reference] != [x["episode_id"] for x in candidate]:
        raise ValueError("Paired models must use exactly the same episode order.")
    ref_only = sum(bool(a["success"] and not b["success"]) for a, b in zip(reference, candidate, strict=True))
    cand_only = sum(bool(b["success"] and not a["success"]) for a, b in zip(reference, candidate, strict=True))
    result: dict[str, Any] = {
        "comparison": label, "episodes": len(reference),
        "success_contingency": {
            "reference_only_success": ref_only, "candidate_only_success": cand_only,
            "both_success": sum(bool(a["success"] and b["success"]) for a, b in zip(reference, candidate, strict=True)),
            "both_fail": sum(bool(not a["success"] and not b["success"]) for a, b in zip(reference, candidate, strict=True)),
        },
        "success_mcnemar_exact_p": sf._mcnemar_exact_p(ref_only, cand_only),
        "collision_mcnemar_exact_p": sf._mcnemar_exact_p(
            sum(bool(a["collision"] and not b["collision"]) for a, b in zip(reference, candidate, strict=True)),
            sum(bool(b["collision"] and not a["collision"]) for a, b in zip(reference, candidate, strict=True)),
        ),
        "paired_metrics": {},
    }
    for metric in METRICS:
        delta = np.asarray([float(b[metric]) - float(a[metric]) for a, b in zip(reference, candidate, strict=True)])
        result["paired_metrics"][metric] = {
            "candidate_minus_reference_mean": float(delta.mean()),
            "bootstrap_95_ci": _bootstrap_ci(delta, seed + len(metric)),
        }
    return result


def collect_self_policy_states(model_name: sf.SurfaceModelName, model: nn.Module,
                              specs: list[sf.SurfaceEpisodeSpec],
                              expert_reference: dict[int, dict[str, Any]],
                              config: sf.SurfaceFeasibilityConfig, seed: int,
                              device: torch.device, output_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    validate_train_specs(specs)
    base_policy_hash = state_dict_hash(model)
    out = ensure_dir(output_dir)
    env = sf.make_env(config, seed + 4_051)
    records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    for spec in specs:
        sf._reset_surface_env(env, spec)
        initial = env.robot_observation().ee_position.numpy().astype(np.float64)
        target = sf._surface_target(spec.shape, initial, config.pregrasp_clearance)
        expert_path = np.asarray(expert_reference[spec.episode_id]["ee_positions"], dtype=np.float64)
        collision_seen = False
        first_collision = None
        min_clearance = 0.20
        positions = [initial.tolist()]
        count = 0
        errors = sf._state_errors(env, spec.shape, config, target)
        for timestep in range(config.max_steps):
            if sf._success_from_errors(errors, collision_seen, config):
                break
            state, surface, centerline = sf.observation_inputs(
                env, spec.shape, config.point_count, sf._stable_seed(spec.sample_identity),
            )
            ee = env.robot_observation().ee_position.numpy().astype(np.float64)
            before, clearance = sf._surface_distance_metrics(env, spec.shape)
            collision_now = bool(before)
            collision_seen = collision_seen or collision_now
            min_clearance = min(min_clearance, clearance)
            policy_action = sf._policy_action(model_name, model, state,
                sf.policy_geometry_points(model_name, surface, centerline), config, device)
            oracle_action, _ = sf.expert_action(env, spec.shape, config)
            trajectory_distance = dagger._reference_distance(ee, expert_path)
            deviation = dagger.classify_deviation(trajectory_distance, collision_now, clearance)
            severe = clearance < SEVERE_PENETRATION_M
            row = {
                "state": state.copy(), "surface_points": surface.copy(), "centerline_points": centerline.copy(),
                "action": oracle_action.copy(), "oracle_action": oracle_action.copy(),
                "policy_action": policy_action.copy(), "qpos": env.data.qpos.copy(),
                "episode_id": spec.episode_id, "timestep": timestep,
                "ee_position_world": ee.tolist(), "ee_yaw_rad": float(sf.tool_yaw(env)),
                "aperture_m": float(sf.gripper_width(env)), "collision": collision_now,
                "minimum_clearance_m": float(clearance),
                "position_error_m": errors["position_error"],
                "orientation_error_rad": errors["orientation_error"],
                "aperture_error_m": errors["gripper_width_error"],
                "distance_from_expert_trajectory_m": trajectory_distance,
                "deviation_bucket": deviation, "severe_penetration_excluded": severe,
            }
            metadata.append({key: value for key, value in row.items()
                             if key not in ("state", "surface_points", "centerline_points", "qpos", "action", "oracle_action", "policy_action")})
            if not severe:
                records.append(row)
                count += 1
            sf.apply_local_action(env, policy_action, config)
            current = env.robot_observation().ee_position.numpy().astype(np.float64)
            positions.append(current.tolist())
            after, after_clearance = sf._surface_distance_metrics(env, spec.shape)
            if after and first_collision is None:
                first_collision = timestep + 1
            collision_seen = collision_seen or bool(after)
            min_clearance = min(min_clearance, after_clearance)
            errors = sf._state_errors(env, spec.shape, config, target)
        final = sf._state_errors(env, spec.shape, config, target)
        final_contacts, final_clearance = sf._surface_distance_metrics(env, spec.shape)
        min_clearance = min(min_clearance, final_clearance)
        rollout_rows.append({
            "episode_id": spec.episode_id, "steps": len(positions) - 1, "collected_eligible_states": count,
            "success": bool(sf._success_from_errors(final, collision_seen, config)),
            "collision": collision_seen, "first_collision_timestep": first_collision,
            "minimum_safe_clearance": float(min_clearance), "final_position_error": final["position_error"],
            "final_orientation_error": final["orientation_error"],
            "final_gripper_width_error": final["gripper_width_error"],
            "trajectory": positions, "final_collision": bool(final_contacts),
        })
    env.close()
    if not records:
        raise RuntimeError(f"{model_name} seed={seed}: no eligible DAgger states collected.")
    data = {
        "task": "surface_representation_self_policy_dagger_round1", "seed": seed, "model": model_name,
        "states": torch.as_tensor(np.stack([r["state"] for r in records]), dtype=torch.float32),
        "surface_points": torch.as_tensor(np.stack([r["surface_points"] for r in records]), dtype=torch.float32),
        "centerline_points": torch.as_tensor(np.stack([r["centerline_points"] for r in records]), dtype=torch.float32),
        "actions": torch.as_tensor(np.stack([r["oracle_action"] for r in records]), dtype=torch.float32),
        "policy_actions": torch.as_tensor(np.stack([r["policy_action"] for r in records]), dtype=torch.float32),
        "qpos": torch.as_tensor(np.stack([r["qpos"] for r in records]), dtype=torch.float64),
        "episode_ids": torch.as_tensor([r["episode_id"] for r in records], dtype=torch.long),
        "steps": torch.as_tensor([r["timestep"] for r in records], dtype=torch.long),
    }
    eligible = [r for r in metadata if not r["severe_penetration_excluded"]]
    buckets = ("near_expert", "moderate_deviation", "large_deviation", "collision_near_collision")
    stats = {
        "model": model_name, "seed": seed, "training_split_only": True,
        "base_policy_state_dict_sha256": base_policy_hash,
        "training_episode_count": len(specs), "collected_policy_states": len(metadata),
        "eligible_policy_states": len(records), "excluded_severe_penetration_states": len(metadata) - len(records),
        "severe_penetration_rule": f"exclude pre-action min signed geom distance < {SEVERE_PENETRATION_M:.3f} m",
        "eligible_deviation_bucket_counts": {bucket: sum(r["deviation_bucket"] == bucket for r in eligible) for bucket in buckets},
        "eligible_deviation_bucket_proportions": {bucket: sum(r["deviation_bucket"] == bucket for r in eligible) / len(eligible) for bucket in buckets},
        "collected_deviation_bucket_counts": {bucket: sum(r["deviation_bucket"] == bucket for r in metadata) for bucket in buckets},
        "expert_path_distance_m": {
            "mean": float(np.mean([r["distance_from_expert_trajectory_m"] for r in metadata])),
            "p50": float(np.percentile([r["distance_from_expert_trajectory_m"] for r in metadata], 50)),
            "p90": float(np.percentile([r["distance_from_expert_trajectory_m"] for r in metadata], 90)),
            "max": float(np.max([r["distance_from_expert_trajectory_m"] for r in metadata])),
        },
        "rollout_metrics": {
            "success": float(np.mean([r["success"] for r in rollout_rows])),
            "collision": float(np.mean([r["collision"] for r in rollout_rows])),
            "final_position_error": float(np.mean([r["final_position_error"] for r in rollout_rows])),
            "final_orientation_error": float(np.mean([r["final_orientation_error"] for r in rollout_rows])),
            "final_aperture_error": float(np.mean([r["final_gripper_width_error"] for r in rollout_rows])),
        },
    }
    torch.save(data, out / "policy_relabelled_train_states.pt")
    write_json(out / "visited_state_metadata.json", metadata)
    write_json(out / "eligible_state_metadata.json", eligible)
    write_json(out / "collection_summary.json", {**stats, "episode_rollouts": rollout_rows})
    return data, eligible, stats


def _validation_loss(model: nn.Module, model_name: sf.SurfaceModelName, data: dict[str, Any],
                     config: sf.SurfaceFeasibilityConfig, device: torch.device) -> float:
    model.eval()
    states = data["states"].to(device)
    points = _point_array(data, model_name).to(device)
    actions = data["actions"].to(device)
    losses = []
    with torch.no_grad():
        for start in range(0, len(actions), config.batch_size):
            pred = model(states[start:start + config.batch_size], points[start:start + config.batch_size])
            losses.append(float(sf._normalised_action_loss(pred, actions[start:start + config.batch_size], config).cpu()))
    return float(np.mean(losses))


def train_dagger_round1(model_name: sf.SurfaceModelName, base_checkpoint: Path,
                        expert_data: dict[str, Any], val_data: dict[str, Any],
                        policy_data: dict[str, Any], policy_metadata: list[dict[str, Any]],
                        config: sf.SurfaceFeasibilityConfig, seed: int, device: torch.device,
                        output_dir: Path, updates: int = UPDATES) -> dict[str, Any]:
    if len(policy_metadata) != len(policy_data["actions"]):
        raise ValueError("On-policy metadata and oracle relabel arrays must stay aligned.")
    set_seed(seed + 121_117)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    model = sf.load_surface_model(base_checkpoint, model_name, config, device)
    initial_hash = state_dict_hash(model)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ex_state = expert_data["states"].to(device)
    ex_points = _point_array(expert_data, model_name).to(device)
    ex_actions = expert_data["actions"].to(device)
    po_state = policy_data["states"].to(device)
    po_points = _point_array(policy_data, model_name).to(device)
    po_actions = policy_data["actions"].to(device)
    batch_size = int(config.batch_size)
    expert_batch = batch_size // 2
    policy_batch = batch_size - expert_batch
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1_000_003)
    best_val = float("inf")
    history: list[dict[str, float]] = []
    losses: list[float] = []
    sampled_bucket_counts = {name: 0 for name in ("near_expert", "moderate_deviation", "large_deviation", "collision_near_collision")}
    output_dir = ensure_dir(output_dir)
    checkpoint = output_dir / f"{model_name}.pt"
    for update in range(1, updates + 1):
        expert_ids = torch.randint(len(ex_actions), (expert_batch,), generator=generator, device=device)
        policy_ids = torch.randint(len(po_actions), (policy_batch,), generator=generator, device=device)
        for idx in policy_ids.detach().cpu().tolist():
            sampled_bucket_counts[str(policy_metadata[idx]["deviation_bucket"])] += 1
        states = torch.cat([ex_state[expert_ids], po_state[policy_ids]], dim=0)
        points = torch.cat([ex_points[expert_ids], po_points[policy_ids]], dim=0)
        actions = torch.cat([ex_actions[expert_ids], po_actions[policy_ids]], dim=0)
        prediction = model(states, points)
        loss = sf._normalised_action_loss(prediction, actions, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if update % 25 == 0 or update == updates:
            val_loss = _validation_loss(model, model_name, val_data, config, device)
            history.append({"update": update, "recent_training_loss": float(np.mean(losses[-25:])),
                            "expert_validation_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model_name": model_name, "model_state": model.state_dict(),
                    "config": asdict(config), "seed": seed, "best_validation_loss": best_val,
                    "parameters": sf.model_parameter_count(model), "optimizer_updates": updates,
                    "continuation_condition": "dagger_round1", "base_checkpoint_sha256": initial_hash,
                    "self_policy_collection": True,
                }, checkpoint)
    draws = updates * batch_size
    summary = {
        "model": model_name, "seed": seed, "device": str(device), "optimizer": "AdamW fresh/reset",
        "same_start_checkpoint": str(base_checkpoint), "base_checkpoint_sha256": initial_hash,
        "optimizer_updates": updates, "batch_size": batch_size,
        "effective_expert_draws": updates * expert_batch, "effective_policy_draws": updates * policy_batch,
        "effective_expert_fraction": expert_batch / batch_size,
        "effective_policy_fraction": policy_batch / batch_size,
        "expert_dataset_states": int(len(ex_actions)), "policy_dataset_states": int(len(po_actions)),
        "policy_bucket_sampling_draws": sampled_bucket_counts,
        "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
        "checkpoint_selection": "lowest shared expert validation normalized loss, checked every 25 updates",
        "best_expert_validation_loss": best_val, "parameters": sf.model_parameter_count(model),
        "history": history, "checkpoint_path": str(checkpoint),
        "final_weights_sha256": state_dict_hash(model), "batch_draws": draws,
    }
    write_json(output_dir / "training.json", summary)
    return summary


def _metrics_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"episodes": len(rows)}
    for key in METRICS:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    result["first_collision_timestep_mean"] = float(np.mean([
        row["first_collision_timestep"] if row["first_collision_timestep"] is not None else (sf.SurfaceFeasibilityConfig().max_steps + 1)
        for row in rows
    ]))
    result["steps_to_convergence_mean"] = float(np.mean([row["steps_to_convergence"] for row in rows]))
    return result


def evaluate_paired(specs: list[sf.SurfaceEpisodeSpec], models: dict[str, nn.Module],
                    config: sf.SurfaceFeasibilityConfig, device: torch.device,
                    point_count: int, nonuniform: bool, output_dir: Path,
                    topology_mode: str = "rebuilt",
                    checkpoint_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    assert_same_specs({name: specs for name in models})
    env = sf.make_env(config, 864_211)
    per_model: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    episodes: list[dict[str, Any]] = []
    for spec in specs:
        row: dict[str, Any] = {"episode_id": spec.episode_id, "condition": spec.condition, "spec": asdict(spec), "models": {}}
        for name, model in models.items():
            metrics = rollout_detailed(env, spec, config, name, model, device, point_count, nonuniform, topology_mode)
            row["models"][name] = metrics
            per_model[name].append(metrics)
        episodes.append(row)
    env.close()
    paired: dict[str, Any] = {}
    for ref, cand, key in PAIR_COMPARISONS:
        if ref not in per_model or cand not in per_model:
            continue
        paired[key] = paired_comparison(per_model[ref], per_model[cand], config,
                                        f"{cand} - {ref}", 913_003 + len(specs))
    payload = {
        "surface_points": point_count if point_count != sf.SURFACE_POINTS or not nonuniform else "16 nonuniform",
        "sampling_nonuniform": nonuniform, "topology_mode": topology_mode,
        "checkpoint_hashes": checkpoint_hashes or {},
        "same_episode_specs": [asdict(spec) for spec in specs],
        "models": {name: _metrics_summary(rows) for name, rows in per_model.items()},
        "paired_comparisons": paired,
        "episodes": episodes,
    }
    ensure_dir(output_dir)
    write_json(output_dir / "episode_results.json", payload)
    return payload


def _plot_failure_breakdown(summary: dict[str, Any], output_path: Path) -> None:
    combos = summary["all_16_failure_combinations"]
    rows = [(label, values["count"]) for label, values in combos.items() if values["count"]]
    rows.sort(key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(11, max(4.2, 0.32 * len(rows))))
    ax.barh([r[0].replace("_failure", "") for r in rows][::-1], [r[1] for r in rows][::-1], color="#b56050")
    ax.set_xlabel("episodes")
    ax.set_title("DAgger Surface Set held-out IID multi-label failure combinations")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(); ensure_dir(output_path.parent); fig.savefig(output_path, dpi=160); plt.close(fig)


def _plot_pair_case(condition: str, seed: int, label: str, episode: dict[str, Any],
                    point_count: int, nonuniform: bool, output: Path) -> None:
    shape = sf.SurfaceShape(**episode["models"]["surface_set"].get("shape", next(iter(episode["models"].values()))["shape"]))
    spec = stab._as_spec(episode["spec"])
    points = sf.sample_surface_points_world(shape, point_count, sf._stable_seed(spec.sample_identity), nonuniform)
    center = sf.sample_centerline_points_world(shape, sf.SURFACE_POINTS)
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.scatter(points[:, 0], points[:, 2], s=12, alpha=0.55, color="#5f9db8", label="oracle surface samples")
    ax.plot(center[:, 0], center[:, 2], color="#333333", linewidth=1.0, alpha=0.7, label="centerline")
    plotted = set()
    for model_name, result in episode["models"].items():
        traj = np.asarray(result["trajectory"], dtype=np.float64)
        ax.plot(traj[:, 0], traj[:, 2], marker=".", markersize=3, linewidth=1.2, label=model_name)
        ax.scatter([traj[0, 0]], [traj[0, 2]], marker="<", s=38)
        ax.scatter([traj[-1, 0]], [traj[-1, 2]], marker="s", s=32)
        if model_name not in plotted and result.get("fingertip_trajectories"):
            tip_trace = np.asarray(result["fingertip_trajectories"], dtype=np.float64)
            for tip_i in (0, 1):
                ax.plot(tip_trace[:, tip_i, 0], tip_trace[:, tip_i, 2], linestyle=":", linewidth=0.6, alpha=0.5)
            plotted.add(model_name)
        collision_step = result.get("first_collision_timestep")
        if collision_step is not None and 0 <= collision_step < len(traj):
            ax.scatter([traj[collision_step, 0]], [traj[collision_step, 2]], marker="x", s=50, color="#c62828")
    reference = next(iter(episode["models"].values()))
    target = np.asarray(reference["target_position"])
    normal = np.asarray(reference["target_normal"])
    ax.scatter([target[0]], [target[2]], marker="*", s=160, color="#7028a0", label="expert target")
    ax.arrow(target[0], target[2], normal[0] * 0.04, normal[2] * 0.04, color="#7028a0", head_width=0.006)
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.2)
    ax.set(xlabel="world x (m)", ylabel="world z (m)", title=f"{condition} seed {seed}: {label}")
    ax.legend(fontsize=6, loc="best"); fig.tight_layout(); ensure_dir(output.parent); fig.savefig(output, dpi=150); plt.close(fig)


def _save_failure_cases(by_seed: dict[str, dict[str, Any]], root: Path) -> dict[str, str]:
    categories = {
        "centerline_fail_surface_success": ("centerline_set", "surface_set"),
        "surface_fail_centerline_success": ("surface_set", "centerline_set"),
        "set_fail_graph_success": ("surface_set", "surface_graph"),
        "graph_fail_set_success": ("surface_graph", "surface_set"),
        "no_local_fail_graph_success": ("surface_graph_no_local", "surface_graph"),
        "graph_fail_no_local_success": ("surface_graph", "surface_graph_no_local"),
    }
    written: dict[str, str] = {}
    for condition in ("iid", "surface_shape_ood", "sampling_ood"):
        for label, (a, b) in categories.items():
            selected = None
            for seed, data in by_seed.items():
                for ep in data["conditions"][condition]["episodes"]:
                    if not ep["models"][a]["success"] and ep["models"][b]["success"]:
                        selected = (int(seed), ep)
                        break
                if selected:
                    break
            if selected:
                seed, ep = selected
                point_value = by_seed[str(seed)]["conditions"][condition]["surface_points"]
                n = int(point_value if isinstance(point_value, int) else 16)
                nonuniform = condition == "sampling_ood"
                path = root / "failure_cases" / condition / f"{label}_seed{seed}_ep{ep['episode_id']:04d}.png"
                _plot_pair_case(condition, seed, label, ep, n, nonuniform, path)
                written[f"{condition}/{label}"] = str(path)
    return written


def _plot_latency_curve(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = [row["surface_points"] for row in rows]
    p50 = [row["geometry_to_action"]["p50_ms"] for row in rows]
    p99 = [row["geometry_to_action"]["p99_ms"] for row in rows]
    ax.plot(x, p50, marker="o", label="p50")
    ax.plot(x, p99, marker="o", label="p99")
    ax.axhline(5.0, linestyle="--", color="#ba3d3d", label="5 ms design budget")
    ax.set(xlabel="surface points N", ylabel="geometry-to-action latency (ms)", title="Selected representation: latency vs N")
    ax.grid(alpha=0.25); ax.legend(frameon=False); fig.tight_layout(); ensure_dir(path.parent); fig.savefig(path, dpi=160); plt.close(fig)


def _load_prior_dagger_surface(seed: int, config: sf.SurfaceFeasibilityConfig,
                               expected_base_hash: str, train_specs: list[sf.SurfaceEpisodeSpec],
                               output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Path]:
    prior_seed = DAGGER_ROOT / "training" / f"seed{seed}"
    checkpoint = prior_seed / "dagger_round1" / "surface_set.pt"
    training_json = prior_seed / "dagger_round1" / "training.json"
    policy_path = DAGGER_ROOT / "onpolicy_collection" / f"seed{seed}" / "onpolicy_relabelled_train_states.pt"
    metadata_path = DAGGER_ROOT / "onpolicy_collection" / f"seed{seed}" / "visited_state_metadata.json"
    base_path = STABILIZATION_ROOT / f"seed{seed}" / "base" / "checkpoints" / "surface_set.pt"
    for path in (checkpoint, training_json, policy_path, metadata_path, base_path):
        if not path.exists():
            raise FileNotFoundError(path)
    digest = state_dict_hash(sf.load_surface_model(base_path, "surface_set", config, torch.device("cpu")))
    if digest != expected_base_hash:
        raise ValueError(f"Seed {seed} Surface Set base weights changed relative to the DAgger run.")
    _checkpoint_config_compatible(checkpoint, config, "surface_set", seed)
    training = json.loads(training_json.read_text(encoding="utf-8"))
    if training["optimizer_updates"] != UPDATES or training["batch_size"] != BATCH_SIZE or training["effective_expert_fraction"] != 0.5:
        raise ValueError(f"Seed {seed} Surface Set Round 1 protocol is incompatible.")
    policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    eligible_metadata = [row for row in metadata if not row["severe_penetration_excluded"]]
    if len(policy["actions"]) != len(eligible_metadata):
        raise ValueError("Surface Set DAgger state/metadata records are not aligned.")
    episode_ids = set(int(x) for x in policy["episode_ids"].tolist())
    if not episode_ids.issubset({spec.episode_id for spec in train_specs}):
        raise ValueError("Surface Set policy collection leaked non-training episode IDs.")
    if any(spec.condition != "train" for spec in train_specs):
        raise ValueError("Surface Set policy dataset requires training EpisodeSpecs.")
    out = ensure_dir(output_dir)
    torch.save(policy, out / "policy_relabelled_train_states.pt")
    write_json(out / "visited_state_metadata.json", metadata)
    write_json(out / "eligible_state_metadata.json", eligible_metadata)
    artifact_root = out.parent.parent.parent
    new_checkpoint = ensure_dir(artifact_root / "training" / f"seed{seed}" / "surface_set" / "dagger_round1") / "surface_set.pt"
    shutil.copy2(checkpoint, new_checkpoint)
    shutil.copy2(training_json, new_checkpoint.parent / "training.json")
    stats_path = DAGGER_ROOT / "onpolicy_collection" / f"seed{seed}" / "collection_summary.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["reused_prior_surface_set_collection"] = True
    write_json(out / "collection_summary.json", stats)
    training["checkpoint_path"] = str(new_checkpoint)
    training["reused_verified_prior_round1"] = True
    write_json(new_checkpoint.parent / "training.json", training)
    return policy, eligible_metadata, training, new_checkpoint


def _base_summary(seed: int, model_name: str, checkpoint: Path, training: dict[str, Any], reused: bool) -> dict[str, Any]:
    return {
        "model": model_name, "seed": seed, "checkpoint_path": str(checkpoint),
        "optimizer_updates": int(training.get("optimizer_updates", training.get("epochs", -1))),
        "batch_size": int(training.get("batch_size", BATCH_SIZE)),
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
        "validation_checkpoint_selection": "lowest expert-state normalized action loss",
        "reused_verified_checkpoint": reused, "parameters": int(training["parameters"]),
        "best_validation_loss": float(training["best_validation_loss"]),
    }


def _summary_markdown(result: dict[str, Any]) -> str:
    model_order = sf.MODEL_NAMES
    lines = [
        "# Surface representation final comparison", "",
        "## 1. DAgger Surface Set failure decomposition", "",
        f"Held-out IID: **{result['failure_decomposition']['episodes']} episodes**, success {result['failure_decomposition']['success_rate']:.3f}; aperture-only share among failures {result['failure_decomposition']['aperture_only_failed_fraction']:.3f} (warning={result['failure_decomposition']['aperture_bottleneck_warning']}).",
        "Thresholds: position < 0.025 m, orientation < 0.200 rad, aperture < 0.008 m, and no trajectory collision. Failure labels are multi-label.", "",
        "## 2. Training fairness", "",
        "All models use the same 72 training EpisodeSpecs, 411/400/417 expert states by seed (the same physical data across encodings), 16 validation EpisodeSpecs, AdamW, batch 128, LR 3e-4, weight decay 1e-4, and 800 updates for both base and Round 1. The verified Surface Set base and its prior self-policy Round 1 artifacts were reused; the other base encoders were trained with the same fixed-update protocol. Each representation collected its own D_policy from its own base rollout. Dagger batches draw 64 expert and 64 eligible on-policy labels uniformly with replacement.", "",
        "## 3. Main results", "",
        "| Model | IID success | IID collision | IID position (m) | IID yaw (rad) | IID aperture (m) | Shape OOD success | Shape OOD collision | Sampling OOD success | Sampling OOD collision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pooled = result["pooled_metrics"]
    for name in model_order:
        iid, shape, sampling = (pooled[c][name] for c in ("iid", "surface_shape_ood", "sampling_ood"))
        lines.append(f"| {name} | {iid['success']:.3f} | {iid['collision']:.3f} | {iid['final_position_error']:.4f} | {iid['final_orientation_error']:.3f} | {iid['final_gripper_width_error']:.4f} | {shape['success']:.3f} | {shape['collision']:.3f} | {sampling['success']:.3f} | {sampling['collision']:.3f} |")
    lines.extend(["", "## 4. Paired comparisons", "", "Deltas are candidate minus reference; negative error/collision is better. Confidence intervals use episode bootstrap; success also reports exact McNemar p.", ""])
    for condition in ("iid", "surface_shape_ood", "sampling_ood"):
        lines.append(f"### {condition}")
        lines.append("")
        lines.append("| Comparison | Success Δ | Collision Δ | Position Δ m | Yaw Δ rad | Aperture Δ m | Trajectory position Δ m | Success McNemar p |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, _, key in PAIR_COMPARISONS:
            pair = result["paired_pooled"][condition][key]
            metrics = pair["paired_metrics"]
            vals = [metrics[x]["candidate_minus_reference_mean"] for x in ("success", "collision", "final_position_error", "final_orientation_error", "final_gripper_width_error", "trajectory_error")]
            ci = metrics["collision"]["bootstrap_95_ci"]
            lines.append(f"| {pair['comparison']} | {vals[0]:+.3f} | {vals[1]:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] | {vals[2]:+.4f} | {vals[3]:+.3f} | {vals[4]:+.4f} | {vals[5]:+.4f} | {pair['success_mcnemar_exact_p']:.4f} |")
        lines.append("")
    if result.get("surface_set_vs_no_local_graph"):
        lines.extend(["### Additional selection diagnostic: no-local graph vs Surface Set", "",
                      "| Condition | Success Δ | Collision Δ | Position Δ m | Yaw Δ rad | Collision 95% CI |",
                      "|---|---:|---:|---:|---:|---:|"])
        for condition in ("iid", "surface_shape_ood", "sampling_ood"):
            pair = result["surface_set_vs_no_local_graph"][condition]
            metrics = pair["paired_metrics"]
            lines.append(f"| {condition} | {metrics['success']['candidate_minus_reference_mean']:+.3f} | {metrics['collision']['candidate_minus_reference_mean']:+.3f} | {metrics['final_position_error']['candidate_minus_reference_mean']:+.4f} | {metrics['final_orientation_error']['candidate_minus_reference_mean']:+.3f} | {metrics['collision']['bootstrap_95_ci']} |")
        lines.append("")
    lines.extend(["## 5. Latency", "", "| Model | Parameters | Preprocess p99 ms | Graph build p50/p99 ms | Forward p99 ms | Geometry-to-action p50/p99 ms | >5 ms |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, row in result["latency_n32"].items():
        comp = row["components"]
        prep = max(comp["surface_sampling"]["p99_ms"], comp["coordinate_transform"]["p99_ms"], comp["tensor_preparation"]["p99_ms"])
        graph_build = comp["graph_construction"]
        g2a = row["geometry_to_action"]
        lines.append(f"| {name} | {row['parameters']} | {prep:.3f} | {graph_build['p50_ms']:.3f}/{graph_build['p99_ms']:.3f} | {comp['network_forward']['p99_ms']:.3f} | {g2a['p50_ms']:.3f}/{g2a['p99_ms']:.3f} | {100*row['deadline_miss_rate_gt_5ms']:.2f}% |")
    chosen = result.get("decision", {}).get("selected_representation", "UNDECIDED")
    lines.extend([
        "", f"Selected spatial representation: **{chosen}**.",
        f"N sweep: {result.get('n_sweep_decision', {}).get('selected_operating_point', 'not run')}",
        "",
        "| N | Success | Collision | Position m | Yaw rad | Aperture m | Geometry-to-action p50/p99 ms | >5 ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for n, row in result.get("n_sweep_decision", {}).get("runs", {}).items():
        metrics = row["metrics"]
        g2a = row["geometry_to_action"]
        miss = g2a["deadline_miss_rate_over_5ms"]
        lines.append(f"| {n} | {metrics['success']:.3f} | {metrics['collision']:.3f} | {metrics['final_position_error']:.4f} | {metrics['final_orientation_error']:.3f} | {metrics['final_gripper_width_error']:.4f} | {g2a['p50_ms']:.3f}/{g2a['p99_ms']:.3f} | {100*miss:.2f}% |")
    lines.extend([
        "",
        "Latency is geometry-to-action over already available task-relevant geometry. Target is a project-specific design budget of 100 Hz and p99 ≤ 5 ms; it is not claimed as an industry standard. `sensor-to-command latency = NOT MEASURED` (no camera capture, detector, segmentation, or tracking).", "",
        f"Latency device: `{result.get('latency_n32', {}).get('surface_graph_no_local', {}).get('device', 'not recorded')}`, batch 1, warm-up {result.get('latency_n32', {}).get('surface_graph_no_local', {}).get('warmup_iterations', 'n/a')}, timed {result.get('latency_n32', {}).get('surface_graph_no_local', {}).get('timed_iterations', 'n/a')}. Cached topology: {result.get('cached_graph', {}).get('reason', 'not measured')}.", "",
        "## 6. Final questions", "",
    ])
    for question, answer in result.get("answers", {}).items():
        lines.append(f"- **{question}** {answer['answer']} — {answer['evidence']}")
    lines.extend(["", "The comparison uses only the existing feed-forward encoders; no recurrent state or spatial architecture was added. The measured winner is frozen for the later `Selected spatial encoder + FF vs GRU` feasibility experiment.", ""])
    return "\n".join(lines)


def run_experiment(output_dir: str = DEFAULT_OUTPUT, seeds: tuple[int, ...] = SEEDS,
                   device_preference: DevicePreference = "auto",
                   eval_device_preference: DevicePreference = "auto",
                   latency_warmup: int = 200, latency_iterations: int = 2000) -> dict[str, Any]:
    root = ensure_dir(output_dir)
    old_config = json.loads((Path("artifacts/surface_pregrasp_stabilization/config.json")).read_text())
    config_fields = old_config["base_experiment_config"]
    tuple_fields = {"seeds", "training_object_x_range", "training_object_z_range", "training_length_range",
                    "training_half_width_range", "training_depth_range", "ood_half_width_range", "ood_depth_range"}
    values = {k: tuple(v) if k in tuple_fields else v for k, v in config_fields.items()}
    values.update(output_dir=str(root), device=device_preference, eval_device=eval_device_preference,
                  latency_warmup=latency_warmup, latency_iterations=latency_iterations,
                  seeds=tuple(seeds), batch_size=BATCH_SIZE)
    config = sf.SurfaceFeasibilityConfig(**values)
    config_json = {
        "experiment": "surface_representation_final_comparison", "seeds": list(seeds),
        "models": list(sf.MODEL_NAMES), "no_recurrent_state": True,
        "thresholds": {"position_m": config.success_position_threshold,
                       "orientation_rad": config.success_rotation_threshold,
                       "aperture_m": config.success_opening_threshold,
                       "trajectory_collision_required_false": True},
        "base_training": {"updates": UPDATES, "batch_size": BATCH_SIZE,
                          "optimizer": "AdamW", "learning_rate": config.learning_rate,
                          "weight_decay": config.weight_decay,
                          "checkpoint_selection": "lowest expert-state validation normalized loss, every 20 base updates"},
        "dagger_round1": {"updates": UPDATES, "batch_size": BATCH_SIZE,
                          "optimizer": "fresh AdamW", "expert_policy_fraction": [0.5, 0.5],
                          "checkpoint_selection": "lowest expert-state validation normalized loss, every 25 updates",
                          "severe_penetration_exclusion_m": SEVERE_PENETRATION_M},
        "evaluation": {"episode_count_per_seed_condition": config.eval_episodes,
                       "iid_seed_offset": 60_000, "shape_ood_seed_offset": 61_000,
                       "sampling_ood_seed_offset": 62_000, "sampling_ood_points": 16,
                       "sampling_ood_nonuniform": True},
        "latency": {"control_rate_hz_design_target": 100, "geometry_to_action_p99_budget_ms": 5.0,
                    "warmup": latency_warmup, "timed_iterations": latency_iterations,
                    "device_preference": device_preference, "eval_device_preference": eval_device_preference,
                    "sensor_to_command_latency": "NOT MEASURED"},
        "full_config": asdict(config),
    }
    write_json(root / "config.json", config_json)

    # Stage A first: the held-out Round 1 Surface Set failure gate.
    print("[stage A] DAgger Surface Set held-out IID failure decomposition", flush=True)
    device = select_device(eval_device_preference)
    failure_rows: list[dict[str, Any]] = []
    failure_per_seed: dict[str, Any] = {}
    for seed in seeds:
        train_data, val_data, train_specs, val_specs = _current_datasets(seed, config)
        del train_data, val_data, val_specs
        eval_specs = _make_eval_specs(seed, config)["iid"][0]
        ckpt = DAGGER_ROOT / "training" / f"seed{seed}" / "dagger_round1" / "surface_set.pt"
        _checkpoint_config_compatible(ckpt, config, "surface_set", seed)
        model = sf.load_surface_model(ckpt, "surface_set", config, device)
        rows = []
        env = sf.make_env(config, seed + 885_900)
        for spec in eval_specs:
            rows.append(rollout_detailed(env, spec, config, "surface_set", model, device))
        env.close()
        failure_rows.extend(rows)
        for row in rows:
            row["training_seed"] = seed
            row["failure_flags"] = failure_flags(row, config)
        per_seed = failure_decomposition(rows, config)
        failure_per_seed[str(seed)] = per_seed
        del model
    failure_summary = failure_decomposition(failure_rows, config)
    failure_summary["per_seed"] = failure_per_seed
    failure_path = ensure_dir(root / "failure_decomposition")
    write_json(failure_path / "dagger_surface_set.json", failure_summary)
    write_json(failure_path / "threshold_margins.json", failure_summary["threshold_margins"])
    write_json(failure_path / "combination_table.json", failure_summary["all_16_failure_combinations"])
    write_json(failure_path / "episode_rows.json", failure_rows)
    _plot_failure_breakdown(failure_summary, failure_path / "plots" / "failure_combinations.png")
    print(f"[stage A] success={failure_summary['success_rate']:.3f}, aperture-only failed share={failure_summary['aperture_only_failed_fraction']:.3f}", flush=True)

    # Train/evaluate all four encoders from a matched base and their own policy-state DAgger data.
    device_train = select_device(device_preference)
    by_seed: dict[str, dict[str, Any]] = {}
    checkpoint_map: dict[str, dict[str, str]] = {}
    base_training_all: dict[str, Any] = {}
    dagger_training_all: dict[str, Any] = {}
    collection_all: dict[str, Any] = {}
    dataset_audit: dict[str, Any] = {}
    for seed in seeds:
        print(f"[seed {seed}] loading common physical expert/validation datasets", flush=True)
        train_data, val_data, train_specs, val_specs = _current_datasets(seed, config)
        eval_conditions = _make_eval_specs(seed, config)
        train_root = ensure_dir(root / "training" / f"seed{seed}")
        collection_root = ensure_dir(root / "onpolicy_collection" / f"seed{seed}")
        expert_copy = ensure_dir(train_root / "shared_expert_data")
        # Save references to the same physical data for every model.
        torch.save(train_data, expert_copy / "expert_states.pt")
        torch.save(val_data, expert_copy / "validation_states.pt")
        write_json(expert_copy / "episode_specs.json", [asdict(s) for s in train_specs])
        expert_reference_dir = ensure_dir(collection_root / "expert_reference")
        _, expert_reference = stab.collect_supervised_rows(config, train_specs, seed + 991_001, expert_reference_dir)

        base_paths: dict[str, Path] = {}
        base_rows: dict[str, Any] = {}
        for name in sf.MODEL_NAMES:
            if name == "surface_set":
                base = STABILIZATION_ROOT / f"seed{seed}" / "base" / "checkpoints" / "surface_set.pt"
                training_json = STABILIZATION_ROOT / f"seed{seed}" / "base" / "checkpoints" / "surface_set_training.json"
                payload = _checkpoint_config_compatible(base, config, name, seed)
                train_summary = json.loads(training_json.read_text(encoding="utf-8"))
                if train_summary.get("optimizer_updates") != UPDATES or train_summary.get("batch_size") != BATCH_SIZE:
                    raise ValueError("Surface Set base training did not match the fixed 800 update protocol.")
                # Verify expert states are bit-identical to this experiment's common D_expert.
                original_expert = torch.load(STABILIZATION_ROOT / f"seed{seed}" / "dataset" / "expert" / "expert_states.pt",
                                             map_location="cpu", weights_only=False)
                if not torch.equal(original_expert["states"], train_data["states"]) or not torch.equal(original_expert["actions"], train_data["actions"]):
                    raise ValueError("Surface Set base and final comparison D_expert differ.")
                base_copy = ensure_dir(train_root / name / "base") / f"{name}.pt"
                shutil.copy2(base, base_copy)
                shutil.copy2(training_json, base_copy.parent / f"{name}_training.json")
                base_paths[name] = base_copy
                base_rows[name] = _base_summary(seed, name, base_copy, train_summary, True)
            else:
                base_out = ensure_dir(train_root / name / "base")
                saved_ckpt = base_out / f"{name}.pt"
                saved_summary = base_out / f"{name}_training.json"
                if saved_ckpt.exists() and saved_summary.exists():
                    candidate = json.loads(saved_summary.read_text(encoding="utf-8"))
                    _checkpoint_config_compatible(saved_ckpt, config, name, seed)
                    if candidate.get("optimizer_updates") != UPDATES or candidate.get("batch_size") != BATCH_SIZE:
                        raise ValueError(f"Existing {name} base artifact has incompatible training budget.")
                    train_summary = candidate
                    train_summary["checkpoint_path"] = str(saved_ckpt)
                    print(f"[base train] seed={seed} model={name}: reuse verified fixed-budget checkpoint", flush=True)
                else:
                    print(f"[base train] seed={seed} model={name}, updates={UPDATES}", flush=True)
                    train_summary = stab.train_fixed_updates(name, train_data, val_data, config,
                                                             seed, base_out, UPDATES, device_preference)
                base_paths[name] = Path(train_summary["checkpoint_path"])
                base_rows[name] = _base_summary(seed, name, base_paths[name], train_summary, False)
            _checkpoint_config_compatible(base_paths[name], config, name, seed)

        # Surface Set reuses its verified prior policy states, base hash, and Round 1 checkpoint.
        base_surface_model = sf.load_surface_model(base_paths["surface_set"], "surface_set", config, device_train)
        surface_hash = state_dict_hash(base_surface_model)
        s_policy_out = collection_root / "surface_set"
        s_policy, s_meta, s_train, s_ckpt = _load_prior_dagger_surface(
            seed, config, surface_hash, train_specs, s_policy_out,
        )
        models_policy: dict[str, nn.Module] = {"surface_set": base_surface_model}
        policies: dict[str, dict[str, Any]] = {"surface_set": s_policy}
        metadata_by_model: dict[str, list[dict[str, Any]]] = {"surface_set": s_meta}
        collection_stats: dict[str, Any] = {"surface_set": json.loads((s_policy_out / "collection_summary.json").read_text())}
        dagger_rows: dict[str, Any] = {"surface_set": s_train}
        checkpoints: dict[str, str] = {"surface_set": str(s_ckpt)}

        for name in sf.MODEL_NAMES:
            if name == "surface_set":
                continue
            model = sf.load_surface_model(base_paths[name], name, config, device_train)
            collect_dir = collection_root / name
            policy_path = collect_dir / "policy_relabelled_train_states.pt"
            eligible_path = collect_dir / "eligible_state_metadata.json"
            collection_summary_path = collect_dir / "collection_summary.json"
            expected_hash = state_dict_hash(model)
            if policy_path.exists() and eligible_path.exists() and collection_summary_path.exists():
                policy = torch.load(policy_path, map_location="cpu", weights_only=False)
                metadata = json.loads(eligible_path.read_text(encoding="utf-8"))
                stats = json.loads(collection_summary_path.read_text(encoding="utf-8"))
                observed_ids = set(int(x) for x in policy["episode_ids"].tolist())
                if (stats.get("model") != name or stats.get("seed") != seed
                        or stats.get("base_policy_state_dict_sha256", expected_hash) != expected_hash
                        or len(policy["actions"]) != len(metadata)
                        or not observed_ids.issubset({s.episode_id for s in train_specs})):
                    raise ValueError(f"Existing {name} policy collection failed seed/split/base-hash validation.")
                stats["base_policy_state_dict_sha256"] = expected_hash
                write_json(collection_summary_path, stats)
                print(f"[on-policy collect] seed={seed} model={name}: reuse verified own-policy rollout", flush=True)
            else:
                print(f"[on-policy collect] seed={seed} model={name}", flush=True)
                policy, metadata, stats = collect_self_policy_states(
                    name, model, train_specs, expert_reference, config, seed,
                    device_train, collect_dir,
                )
            models_policy[name] = model
            policies[name] = policy
            metadata_by_model[name] = metadata
            collection_stats[name] = stats

        # The persisted self-policy datasets are model-specific; no policy data are shared across encoders.
        bucket_proportions = {}
        for name in sf.MODEL_NAMES:
            row = collection_stats[name]
            if name == "surface_set":
                eligible = sum(not item["severe_penetration_excluded"] for item in s_meta)
                counts = {bucket: sum(item["deviation_bucket"] == bucket and not item["severe_penetration_excluded"] for item in s_meta)
                          for bucket in ("near_expert", "moderate_deviation", "large_deviation", "collision_near_collision")}
                row.update({"eligible_policy_states": eligible, "collected_policy_states": len(s_meta),
                            "eligible_deviation_bucket_counts": counts,
                            "eligible_deviation_bucket_proportions": {k: v / max(eligible, 1) for k, v in counts.items()},
                            "reused_prior_surface_set_collection": True})
            bucket_proportions[name] = row.get("eligible_deviation_bucket_proportions", {})
        collection_all[str(seed)] = collection_stats

        seed_base: dict[str, Any] = {name: base_rows[name] for name in sf.MODEL_NAMES}
        seed_dagger: dict[str, Any] = {"surface_set": dagger_rows["surface_set"]}
        for name in sf.MODEL_NAMES:
            if name == "surface_set":
                continue
            print(f"[DAgger train] seed={seed} model={name}, updates={UPDATES}", flush=True)
            dagger_out = ensure_dir(train_root / name / "dagger_round1")
            dagger_ckpt = dagger_out / f"{name}.pt"
            dagger_summary_path = dagger_out / "training.json"
            if dagger_ckpt.exists() and dagger_summary_path.exists():
                summary = json.loads(dagger_summary_path.read_text(encoding="utf-8"))
                _checkpoint_config_compatible(dagger_ckpt, config, name, seed)
                expected_base = state_dict_hash(sf.load_surface_model(base_paths[name], name, config, torch.device("cpu")))
                if (summary.get("optimizer_updates") != UPDATES or summary.get("batch_size") != BATCH_SIZE
                        or summary.get("base_checkpoint_sha256") != expected_base):
                    raise ValueError(f"Existing {name} DAgger artifact has incompatible base or budget.")
                summary["checkpoint_path"] = str(dagger_ckpt)
                print(f"[DAgger train] seed={seed} model={name}: reuse verified Round 1 checkpoint", flush=True)
            else:
                summary = train_dagger_round1(
                    name, base_paths[name], train_data, val_data, policies[name], metadata_by_model[name],
                    config, seed, device_train, dagger_out,
                )
            seed_dagger[name] = summary
            checkpoints[name] = summary["checkpoint_path"]
        # Verified Surface Set checkpoint was copied under the new artifact tree.
        seed_base["surface_set"]["state_dict_sha256"] = surface_hash
        for name in sf.MODEL_NAMES:
            checkpoint = Path(checkpoints[name]) if name != "surface_set" else Path(checkpoints["surface_set"])
            if name == "surface_set":
                checkpoint = train_root / "surface_set" / "dagger_round1" / "surface_set.pt"
            checkpoints[name] = str(checkpoint)
            model_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            seed_dagger.setdefault(name, {
                "model": name, "seed": seed, "optimizer_updates": int(model_payload.get("optimizer_updates", UPDATES)),
                "batch_size": BATCH_SIZE, "best_expert_validation_loss": model_payload.get("best_validation_loss"),
                "parameters": model_payload.get("parameters"), "checkpoint_path": str(checkpoint),
                "reused_verified_prior_round1": name == "surface_set",
            })
        checkpoint_map[str(seed)] = checkpoints
        base_training_all[str(seed)] = seed_base
        dagger_training_all[str(seed)] = seed_dagger
        dataset_audit[str(seed)] = {
            "train_episode_specs_sha256": hashlib.sha256("\n".join(spec_signature(s) for s in train_specs).encode()).hexdigest(),
            "validation_episode_specs_sha256": hashlib.sha256("\n".join(spec_signature(s) for s in val_specs).encode()).hexdigest(),
            "training_episodes": len(train_specs), "expert_states": int(len(train_data["actions"])),
            "validation_states": int(len(val_data["actions"])),
            "same_D_expert_for_all_models": True,
            "self_policy_dataset_per_model": {name: int(len(policies[name]["actions"])) for name in sf.MODEL_NAMES},
            "effective_sampling_ratio_per_model": {name: [0.5, 0.5] for name in sf.MODEL_NAMES},
            "eligible_deviation_sampling_proportions": bucket_proportions,
        }

        device_eval = select_device(eval_device_preference)
        models_eval = {name: sf.load_surface_model(checkpoints[name], name, config, device_eval) for name in sf.MODEL_NAMES}
        eval_checkpoint_hashes = {name: state_dict_hash(model) for name, model in models_eval.items()}
        seed_result: dict[str, Any] = {"base_training": seed_base, "dagger_training": seed_dagger,
                                       "conditions": {}, "evaluation_episode_specs": {}}
        for condition, (specs, point_count, nonuniform) in eval_conditions.items():
            assert_same_specs({name: specs for name in sf.MODEL_NAMES})
            eval_dir = root / "evaluation" / condition / f"seed{seed}"
            eval_path = eval_dir / "episode_results.json"
            expected_specs = [spec_signature(spec) for spec in specs]
            reuse_eval = False
            if eval_path.exists():
                saved = json.loads(eval_path.read_text(encoding="utf-8"))
                saved_specs = [spec_signature(stab._as_spec(item)) for item in saved.get("same_episode_specs", [])]
                reuse_eval = (saved_specs == expected_specs
                              and saved.get("checkpoint_hashes") == eval_checkpoint_hashes
                              and saved.get("sampling_nonuniform") == nonuniform
                              and len(saved.get("episodes", [])) == len(specs))
            if reuse_eval:
                print(f"[evaluation] seed={seed} condition={condition}: reuse exact paired result", flush=True)
                eval_result = saved
            else:
                print(f"[evaluation] seed={seed} condition={condition}, episodes={len(specs)}", flush=True)
                eval_result = evaluate_paired(
                    specs, models_eval, config, device_eval, point_count, nonuniform,
                    eval_dir, checkpoint_hashes=eval_checkpoint_hashes,
                )
            seed_result["conditions"][condition] = eval_result
            seed_result["evaluation_episode_specs"][condition] = [asdict(s) for s in specs]
        by_seed[str(seed)] = seed_result
        del models_eval, models_policy, train_data, val_data, policies

    # Aggregate pooled per-condition metrics and paired comparisons across the three training seeds.
    pooled_metrics: dict[str, dict[str, Any]] = {}
    paired_pooled: dict[str, dict[str, Any]] = {}
    for condition in ("iid", "surface_shape_ood", "sampling_ood"):
        pooled_metrics[condition] = {}
        for name in sf.MODEL_NAMES:
            episode_rows = [ep["models"][name] for per_seed in by_seed.values() for ep in per_seed["conditions"][condition]["episodes"]]
            pooled_metrics[condition][name] = _metrics_summary(episode_rows)
        paired_pooled[condition] = {}
        for ref, cand, key in PAIR_COMPARISONS:
            ref_rows = [ep["models"][ref] for per_seed in by_seed.values() for ep in per_seed["conditions"][condition]["episodes"]]
            cand_rows = [ep["models"][cand] for per_seed in by_seed.values() for ep in per_seed["conditions"][condition]["episodes"]]
            paired_pooled[condition][key] = paired_comparison(ref_rows, cand_rows, config,
                                                               f"{cand} - {ref}", 802_119 + len(condition))
    direct_surface_graph_no_local: dict[str, Any] = {}
    for condition in ("iid", "surface_shape_ood", "sampling_ood"):
        set_rows = [ep["models"]["surface_set"] for per_seed in by_seed.values() for ep in per_seed["conditions"][condition]["episodes"]]
        no_local_rows = [ep["models"]["surface_graph_no_local"] for per_seed in by_seed.values() for ep in per_seed["conditions"][condition]["episodes"]]
        direct_surface_graph_no_local[condition] = paired_comparison(
            set_rows, no_local_rows, config, "surface_graph_no_local - surface_set", 824_119 + len(condition),
        )

    # Latency is timed on one common inference device and only after all training has finished.
    print("[latency] same-device N=32 benchmark for all four Round 1 checkpoints", flush=True)
    device_eval = select_device(eval_device_preference)
    latency_shape = sf.sample_episode_specs(1, seeds[0] + 60_000, config, "iid")[0].shape
    latency_spec = sf.sample_episode_specs(1, seeds[0] + 60_000, config, "iid")[0]
    latency_env = sf.make_env(config, seeds[0] + 900_000)
    sf._reset_surface_env(latency_env, latency_spec)
    latency_n32: dict[str, Any] = {}
    for name in sf.MODEL_NAMES:
        model = sf.load_surface_model(checkpoint_map[str(seeds[0])][name], name, config, device_eval)
        bench = sf.benchmark_latency(name, model, latency_shape, latency_env, config, device_eval,
                                     sf.SURFACE_POINTS, "rebuilt", latency_warmup, latency_iterations)
        g2a = bench["geometry_to_action"]
        bench["deadline_miss_rate_gt_5ms"] = float(g2a["deadline_miss_rate_over_5ms"])
        bench["parameters"] = sf.model_parameter_count(model)
        bench["batch_size"] = 1
        latency_n32[name] = bench
        write_json(root / "latency" / "n32" / f"{name}_rebuilt.json", bench)
        del model
    latency_env.close()

    # Mean differences with paired CI guide the final choice; a graph must clear both usefulness and budget.
    def ci_upper(condition: str, key: str, metric: str) -> float:
        return float(paired_pooled[condition][key]["paired_metrics"][metric]["bootstrap_95_ci"][1])
    def ci_lower(condition: str, key: str, metric: str) -> float:
        return float(paired_pooled[condition][key]["paired_metrics"][metric]["bootstrap_95_ci"][0])
    graph_latency = latency_n32["surface_graph"]["geometry_to_action"]["p99_ms"]
    graph_budget_ok = graph_latency <= 5.0
    graph_improves = any(
        ci_upper(condition, "surface_vs_graph", metric) < 0.0
        for condition in ("iid", "surface_shape_ood")
        for metric in ("collision", "final_position_error", "final_orientation_error")
    )
    graph_success_improves = any(
        paired_pooled[condition]["surface_vs_graph"]["paired_metrics"]["success"]["candidate_minus_reference_mean"] > 0.0
        and paired_pooled[condition]["surface_vs_graph"]["success_mcnemar_exact_p"] < 0.05
        for condition in ("iid", "surface_shape_ood")
    )
    no_local_latency = latency_n32["surface_graph_no_local"]["geometry_to_action"]["p99_ms"]
    no_local_budget_ok = no_local_latency <= 5.0
    no_local_collision_clear = all(
        direct_surface_graph_no_local[condition]["paired_metrics"]["collision"]["bootstrap_95_ci"][1] < 0.0
        for condition in ("iid", "surface_shape_ood")
    )
    no_local_continuous_nonworse = all(
        direct_surface_graph_no_local[condition]["paired_metrics"]["final_position_error"]["candidate_minus_reference_mean"] <= 0.0
        for condition in ("iid", "surface_shape_ood")
    )
    surface_advantage = any(
        ci_upper(condition, "centerline_vs_surface", metric) < 0.0
        for condition in ("iid", "surface_shape_ood")
        for metric in ("collision", "final_position_error", "final_orientation_error")
    ) or any(
        paired_pooled[condition]["centerline_vs_surface"]["paired_metrics"]["success"]["candidate_minus_reference_mean"] > 0.0
        and paired_pooled[condition]["centerline_vs_surface"]["success_mcnemar_exact_p"] < 0.05
        for condition in ("iid", "surface_shape_ood")
    )
    if no_local_budget_ok and no_local_collision_clear and no_local_continuous_nonworse:
        selected = "surface_graph_no_local"
        reason = ("The existing no-local Surface Graph reduced paired collision in both IID and Shape OOD versus Surface Set, "
                  "did not worsen mean position error, and met the p99 budget; the local Surface↔Surface edges did not help.")
    elif graph_budget_ok and (graph_improves or graph_success_improves):
        selected = "surface_graph"
        reason = "Graph showed paired closed-loop improvement over Surface Set and met p99 geometry-to-action budget."
    else:
        selected = "surface_set" if surf_advantage else "centerline_set"
        reason = ("Surface Set had paired evidence of improvement over Centerline; Graph did not satisfy the usefulness+latency gate."
                  if surf_advantage else "Surface Set did not show stable paired benefit over Centerline; use the simpler representation.")

    decision = {"selected_representation": selected, "reason": reason,
                "graph_p99_ms": graph_latency, "graph_budget_ok": graph_budget_ok,
                "graph_meaningful_gain_over_set": bool(graph_improves or graph_success_improves),
                "no_local_graph_p99_ms": no_local_latency, "no_local_graph_budget_ok": no_local_budget_ok,
                "no_local_graph_clear_collision_gain_vs_set": no_local_collision_clear,
                "surface_meaningful_gain_over_centerline": bool(surface_advantage)}
    write_json(root / "paired" / "pooled_comparisons.json", paired_pooled)
    write_json(root / "paired" / "surface_vs_no_local_graph_pooled.json", direct_surface_graph_no_local)
    per_seed_pairs: dict[str, Any] = {}
    for seed, seed_data in by_seed.items():
        per_seed_pairs[seed] = {}
        for condition, condition_data in seed_data["conditions"].items():
            per_seed_pairs[seed][condition] = condition_data["paired_comparisons"]
            for _, _, pair_dir in PAIR_COMPARISONS:
                ensure_dir(root / "paired" / pair_dir / condition)
                write_json(root / "paired" / pair_dir / condition / f"seed{seed}.json",
                           condition_data["paired_comparisons"][pair_dir])
    write_json(root / "paired" / "per_seed_comparisons.json", per_seed_pairs)
    for condition, comparison in direct_surface_graph_no_local.items():
        ensure_dir(root / "paired" / "surface_vs_no_local_graph" / condition)
        write_json(root / "paired" / "surface_vs_no_local_graph" / condition / "pooled.json", comparison)

    failure_cases = _save_failure_cases(by_seed, root)

    # N sweep on the selected surface family; same exact IID EpisodeSpecs and checkpoints, only N changes.
    n_sweep: dict[str, Any] = {"selected_representation": selected, "same_episode_set": True, "runs": {}}
    latency_curve_rows: list[dict[str, Any]] = []
    if selected in ("surface_set", "surface_graph", "surface_graph_no_local"):
        n_rows_by_count: dict[int, list[dict[str, Any]]] = {16: [], 32: [], 64: []}
        n_specs_by_seed = {seed: _make_eval_specs(seed, config)["iid"][0] for seed in seeds}
        n_sweep["training_seeds"] = list(seeds)
        n_sweep["episode_count"] = sum(len(v) for v in n_specs_by_seed.values())
        n_sweep["same_episode_specs_by_seed"] = {
            str(seed): [asdict(spec) for spec in specs] for seed, specs in n_specs_by_seed.items()
        }
        for point_count in (16, 32, 64):
            per_seed_rows: dict[str, list[dict[str, Any]]] = {}
            for n_seed in seeds:
                specs = n_specs_by_seed[n_seed]
                n_model = sf.load_surface_model(checkpoint_map[str(n_seed)][selected], selected, config, device_eval)
                n_env = sf.make_env(config, n_seed + 990_000)
                rows = [rollout_detailed(n_env, spec, config, selected, n_model, device_eval, point_count, False)
                        for spec in specs]
                n_env.close()
                del n_model
                per_seed_rows[str(n_seed)] = rows
                n_rows_by_count[point_count].extend(rows)
            # Benchmark one checkpoint on the common device; accuracy is pooled over all three paired seeds.
            latency_model = sf.load_surface_model(checkpoint_map[str(seeds[0])][selected], selected, config, device_eval)
            latency_env = sf.make_env(config, seeds[0] + 990_500)
            sf._reset_surface_env(latency_env, latency_spec)
            n_latency = sf.benchmark_latency(selected, latency_model, latency_shape, latency_env, config,
                                             device_eval, point_count, "rebuilt", latency_warmup, latency_iterations)
            latency_env.close(); del latency_model
            metrics = _metrics_summary(n_rows_by_count[point_count])
            g2a = n_latency["geometry_to_action"]
            n_row = {
                "surface_points": point_count, "metrics": metrics, "latency": n_latency,
                "checkpoint_hashes": {str(seed): state_dict_hash(sf.load_surface_model(
                    checkpoint_map[str(seed)][selected], selected, config, torch.device("cpu"))) for seed in seeds},
                "same_episode_count": len(n_rows_by_count[point_count]),
                "per_seed_episode_results": per_seed_rows,
                "episode_results": n_rows_by_count[point_count],
            }
            write_json(root / "n_sweep" / f"n{point_count}.json", n_row)
            n_sweep["runs"][str(point_count)] = {"metrics": metrics, "geometry_to_action": g2a}
            latency_curve_rows.append({"surface_points": point_count, "geometry_to_action": g2a})
        n_sweep["paired_comparisons"] = {}
        for a, b in ((16, 32), (16, 64), (32, 64)):
            n_sweep["paired_comparisons"][f"n{b}_minus_n{a}"] = paired_comparison(
                n_rows_by_count[a], n_rows_by_count[b], config,
                f"N={b} - N={a}", 990_600 + a + b,
            )
        feasible = [(int(n), row) for n, row in n_sweep["runs"].items()
                    if row["geometry_to_action"]["p99_ms"] <= 5.0]
        if feasible:
            best_success = max(row["metrics"]["success"] for _, row in feasible)
            episode_count = int(n_sweep["episode_count"])
            standard_error = math.sqrt(max(best_success * (1.0 - best_success), 0.0) / max(episode_count, 1))
            within_success = [(n, row) for n, row in feasible
                              if row["metrics"]["success"] >= best_success - standard_error]
            best_n, best = min(within_success, key=lambda item: (
                item[1]["metrics"]["collision"], item[1]["metrics"]["final_position_error"],
                item[1]["geometry_to_action"]["p99_ms"], item[0],
            ))
            n_sweep["selection_rule"] = (
                "Among p99<=5ms points within one Bernoulli standard error of the best success, "
                "minimize trajectory collision, then final position error, p99 latency, and N."
            )
            n_sweep["best_observed_success"] = best_success
            n_sweep["success_one_standard_error"] = standard_error
            n_sweep["selected_operating_point"] = f"N={best_n} (collision-first among points within one SE of best success)"
        else:
            n_sweep["selected_operating_point"] = "No N met p99 <= 5 ms; no budget-feasible operating point."
        write_json(root / "n_sweep" / "summary.json", n_sweep)
        _plot_latency_curve(latency_curve_rows, root / "plots" / "n_sweep_latency.png")
    else:
        n_sweep["selected_operating_point"] = "not run: Centerline Set selected; surface N sweep is not decision-relevant."

    cached: dict[str, Any] = {"run": False, "reason": "not selected as final representation"}
    if selected == "surface_graph_no_local":
        cached = {
            "run": False,
            "reason": "not applicable to the selected no-local ablation: it has no Surface↔Surface edges to cache; EE↔Surface and fingertip↔Surface relations are state-dependent and rebuilt each step",
            "selected_representation": selected,
            "surface_surface_edge_count": 0,
            "same_episode_rebuilt_rollout_used": True,
        }
        write_json(root / "n_sweep" / "cached_graph_correctness.json", cached)
    if selected == "surface_graph":
        cached = {"run": True, "device": str(device_eval), "episodes_by_seed": {}, "prediction_max_abs_difference": 0.0}
        n_graph = sf.load_surface_model(checkpoint_map[str(seeds[0])]["surface_graph"], "surface_graph", config, device_eval)
        # Same live input: explicitly verify cached pairs reproduce rebuilt topology predictions.
        check_spec = _make_eval_specs(seeds[0], config)["iid"][0][0]
        env = sf.make_env(config, seeds[0] + 990_100); sf._reset_surface_env(env, check_spec)
        state, points, _ = sf.observation_inputs(env, check_spec.shape, sf.SURFACE_POINTS, sf._stable_seed(check_spec.sample_identity))
        rebuilt_np = sf.build_graph_topology_numpy(points, state[8:14].reshape(2, 3), config, True)
        cached_np = sf.build_graph_topology_numpy(points, state[8:14].reshape(2, 3), config, True,
                                                  cached_surface_pairs=rebuilt_np["surface_pairs"])
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device_eval).reshape(1, sf.STATE_DIM)
        points_t = torch.as_tensor(points, dtype=torch.float32, device=device_eval).reshape(1, -1, 3)
        with torch.no_grad():
            pred_a = n_graph(state_t, points_t, sf.topology_from_numpy(rebuilt_np, device_eval))
            pred_b = n_graph(state_t, points_t, sf.topology_from_numpy(cached_np, device_eval))
        same_input_diff = float(torch.max(torch.abs(pred_a - pred_b)).item())
        cached["same_input_prediction_max_abs_difference"] = same_input_diff
        if same_input_diff > 1e-6:
            raise AssertionError(f"Cached and rebuilt graph predictions differ ({same_input_diff:g}).")
        for seed in seeds:
            specs = _make_eval_specs(seed, config)["iid"][0]
            rebuilt_model = sf.load_surface_model(checkpoint_map[str(seed)]["surface_graph"], "surface_graph", config, device_eval)
            cached_model = sf.load_surface_model(checkpoint_map[str(seed)]["surface_graph"], "surface_graph", config, device_eval)
            rebuilt_env = sf.make_env(config, seed + 990_200)
            cached_env = sf.make_env(config, seed + 990_200)
            rebuilt_rows = [rollout_detailed(rebuilt_env, spec, config, "surface_graph", rebuilt_model, device_eval) for spec in specs]
            cached_rows = [rollout_detailed(cached_env, spec, config, "surface_graph", cached_model, device_eval, topology_mode="cached") for spec in specs]
            rebuilt_env.close(); cached_env.close()
            action_diffs = []
            for a, b in zip(rebuilt_rows, cached_rows, strict=True):
                actions_a = np.asarray(a["policy_actions"], dtype=np.float64)
                actions_b = np.asarray(b["policy_actions"], dtype=np.float64)
                common = min(len(actions_a), len(actions_b))
                action_diffs.append(float(np.max(np.abs(actions_a[:common] - actions_b[:common]))) if common else 0.0)
            max_action_diff = max(action_diffs, default=0.0)
            metric_equal = all(
                np.isclose(float(a[m]), float(b[m]), atol=1e-6, rtol=0.0)
                for a, b in zip(rebuilt_rows, cached_rows, strict=True)
                for m in ("success", "collision", "final_position_error", "final_orientation_error", "final_gripper_width_error")
            )
            cached["episodes_by_seed"][str(seed)] = {
                "episodes": len(specs), "max_action_abs_difference": max_action_diff,
                "metrics_equal_at_1e-6": metric_equal,
                "rebuilt_metrics": _metrics_summary(rebuilt_rows), "cached_metrics": _metrics_summary(cached_rows),
            }
            cached["prediction_max_abs_difference"] = max(cached["prediction_max_abs_difference"], max_action_diff)
            if not metric_equal or max_action_diff > 1e-6:
                raise AssertionError("Cached/rebuilt graph closed-loop outputs do not match.")
            latency_cached_model = sf.load_surface_model(checkpoint_map[str(seed)]["surface_graph"], "surface_graph", config, device_eval)
            latency_env = sf.make_env(config, seed + 990_300); sf._reset_surface_env(latency_env, specs[0])
            bench_rebuilt = sf.benchmark_latency("surface_graph", latency_cached_model, specs[0].shape,
                                                latency_env, config, device_eval, 32, "rebuilt",
                                                latency_warmup, latency_iterations)
            bench_cached = sf.benchmark_latency("surface_graph", latency_cached_model, specs[0].shape,
                                               latency_env, config, device_eval, 32, "cached",
                                               latency_warmup, latency_iterations)
            latency_env.close()
            write_json(root / "latency" / "cached_vs_rebuilt" / f"seed{seed}.json",
                       {"rebuilt": bench_rebuilt, "cached": bench_cached})
        env.close()
        write_json(root / "n_sweep" / "cached_graph_correctness.json", cached)

    # Capture all seed-level paired results in one aggregate artifact.
    result = {
        "seeds": list(seeds), "model_order": list(sf.MODEL_NAMES), "failure_decomposition": failure_summary,
        "dataset_audit": dataset_audit, "base_training": base_training_all,
        "dagger_training": dagger_training_all, "onpolicy_collection": collection_all,
        "per_seed": by_seed, "pooled_metrics": pooled_metrics, "paired_pooled": paired_pooled,
        "latency_n32": latency_n32, "decision": decision, "n_sweep_decision": n_sweep,
        "cached_graph": cached, "failure_cases": failure_cases,
        "sensor_to_command_latency": "NOT MEASURED",
    }
    result["answers"] = _final_answers(result)
    write_json(root / "aggregated_results.json", result)
    (root / "summary.md").write_text(_summary_markdown(result), encoding="utf-8")
    write_json(root / "latency" / "device_and_budget.json", {
        "device": str(device_eval), "batch_size": 1, "warmup_iterations": latency_warmup,
        "timed_iterations": latency_iterations, "budget_ms_p99": 5.0,
        "control_rate_hz_design_target": 100, "sensor_to_command_latency": "NOT MEASURED",
    })
    return result


def _final_answers(result: dict[str, Any]) -> dict[str, Any]:
    failure = result["failure_decomposition"]
    iid = result["paired_pooled"]["iid"]
    shape = result["paired_pooled"]["surface_shape_ood"]
    sampling = result["paired_pooled"]["sampling_ood"]
    center_surface = iid["centerline_vs_surface"]
    set_graph = iid["surface_vs_graph"]
    graph_local = iid["graph_vs_no_local"]
    means = result["pooled_metrics"]
    latency_ok = result["latency_n32"]["surface_graph"]["geometry_to_action"]["p99_ms"] <= 5.0
    sel = result["decision"]["selected_representation"]
    surface_signal = any(
        center_surface["paired_metrics"][metric]["bootstrap_95_ci"][1] < 0.0
        for metric in ("collision", "final_position_error", "final_orientation_error")
    ) or (center_surface["paired_metrics"]["success"]["candidate_minus_reference_mean"] > 0.0
          and center_surface["success_mcnemar_exact_p"] < 0.05)
    surface_ci_spans_zero = any(
        center_surface["paired_metrics"][metric]["bootstrap_95_ci"][0] <= 0.0
        <= center_surface["paired_metrics"][metric]["bootstrap_95_ci"][1]
        for metric in ("collision", "final_position_error", "final_orientation_error")
    )
    return {
        "Q1 remaining Surface Set failure mode": {
            "answer": "YES" if (failure["failure_conditions"]["collision_failure"]["fraction_failed_episodes"] > 0.5 and failure["failure_conditions"]["position_failure"]["fraction_failed_episodes"] > 0.5 and failure["failure_conditions"]["orientation_failure"]["fraction_failed_episodes"] > 0.5) else ("UNCERTAIN" if failure["aperture_bottleneck_warning"] else "NO"),
            "evidence": f"aperture-only={failure['aperture_only_failed_fraction']:.3f} of failures; collision/position/orientation among failures={failure['failure_conditions']['collision_failure']['fraction_failed_episodes']:.3f}/{failure['failure_conditions']['position_failure']['fraction_failed_episodes']:.3f}/{failure['failure_conditions']['orientation_failure']['fraction_failed_episodes']:.3f}",
        },
        "Q2 Surface information vs Centerline": {
            "answer": "YES" if surface_signal else ("UNCERTAIN" if surface_ci_spans_zero else "NO"),
            "evidence": f"IID surface-set minus centerline position={center_surface['paired_metrics']['final_position_error']['candidate_minus_reference_mean']:+.4f} m CI {center_surface['paired_metrics']['final_position_error']['bootstrap_95_ci']}; collision={center_surface['paired_metrics']['collision']['candidate_minus_reference_mean']:+.3f} CI {center_surface['paired_metrics']['collision']['bootstrap_95_ci']}; Shape OOD position delta={shape['centerline_vs_surface']['paired_metrics']['final_position_error']['candidate_minus_reference_mean']:+.4f} m.",
        },
        "Q3 Main Surface benefit dimension": {
            "answer": "UNCERTAIN" if not surface_signal else "YES",
            "evidence": f"Position is the clearest gain: {center_surface['paired_metrics']['final_position_error']['candidate_minus_reference_mean']:+.4f} m, CI {center_surface['paired_metrics']['final_position_error']['bootstrap_95_ci']}; collision {center_surface['paired_metrics']['collision']['candidate_minus_reference_mean']:+.3f}, CI {center_surface['paired_metrics']['collision']['bootstrap_95_ci']}; yaw {center_surface['paired_metrics']['final_orientation_error']['candidate_minus_reference_mean']:+.3f} rad, CI {center_surface['paired_metrics']['final_orientation_error']['bootstrap_95_ci']}.",
        },
        "Q4 Surface Graph vs Surface Set": {
            "answer": "YES" if set_graph["paired_metrics"]["collision"]["bootstrap_95_ci"][1] < 0.0 or (set_graph["paired_metrics"]["success"]["candidate_minus_reference_mean"] > 0.0 and set_graph["success_mcnemar_exact_p"] < 0.05) else ("NO" if set_graph["paired_metrics"]["collision"]["candidate_minus_reference_mean"] >= 0 and set_graph["paired_metrics"]["success"]["candidate_minus_reference_mean"] <= 0 else "UNCERTAIN"),
            "evidence": f"IID graph-set success Δ={set_graph['paired_metrics']['success']['candidate_minus_reference_mean']:+.3f}, collision Δ={set_graph['paired_metrics']['collision']['candidate_minus_reference_mean']:+.3f} CI {set_graph['paired_metrics']['collision']['bootstrap_95_ci']}; Shape OOD collision Δ={shape['surface_vs_graph']['paired_metrics']['collision']['candidate_minus_reference_mean']:+.3f}.",
        },
        "Q5 Surface local edges": {
            "answer": "YES" if graph_local["paired_metrics"]["collision"]["candidate_minus_reference_mean"] < 0 and graph_local["paired_metrics"]["collision"]["bootstrap_95_ci"][1] < 0 else ("NO" if graph_local["paired_metrics"]["collision"]["candidate_minus_reference_mean"] >= 0 else "UNCERTAIN"),
            "evidence": f"Full graph minus no-local IID collision Δ={graph_local['paired_metrics']['collision']['candidate_minus_reference_mean']:+.3f} CI {graph_local['paired_metrics']['collision']['bootstrap_95_ci']}; success Δ={graph_local['paired_metrics']['success']['candidate_minus_reference_mean']:+.3f}.",
        },
        "Q6 Graph additional benefit in Shape OOD": {
            "answer": "YES" if shape["surface_vs_graph"]["paired_metrics"]["collision"]["candidate_minus_reference_mean"] < iid["surface_vs_graph"]["paired_metrics"]["collision"]["candidate_minus_reference_mean"] else "NO",
            "evidence": f"Graph-set collision Δ: IID {iid['surface_vs_graph']['paired_metrics']['collision']['candidate_minus_reference_mean']:+.3f}; Shape OOD {shape['surface_vs_graph']['paired_metrics']['collision']['candidate_minus_reference_mean']:+.3f}.",
        },
        "Q7 Sampling OOD robustness": {
            "answer": "UNCERTAIN",
            "evidence": "No single model leads both completion and safety: " + ", ".join(f"{name}: success={means['sampling_ood'][name]['success']:.3f}, collision={means['sampling_ood'][name]['collision']:.3f}" for name in sf.MODEL_NAMES),
        },
        "Q8 Graph latency budget": {
            "answer": "YES" if latency_ok else "NO",
            "evidence": f"N=32 full graph p99={result['latency_n32']['surface_graph']['geometry_to_action']['p99_ms']:.3f} ms, no-local graph p99={result['latency_n32']['surface_graph_no_local']['geometry_to_action']['p99_ms']:.3f} ms; deadline misses={result['latency_n32']['surface_graph']['deadline_miss_rate_gt_5ms']:.4f}/{result['latency_n32']['surface_graph_no_local']['deadline_miss_rate_gt_5ms']:.4f} vs project design budget 5 ms.",
        },
        "Q9 Representation passed to later GRU feasibility": {
            "answer": "YES" if sel in ("surface_graph", "surface_graph_no_local", "surface_set", "centerline_set") else "UNCERTAIN",
            "evidence": f"selected `{sel}`; {result['decision']['reason']} N operating point: {result['n_sweep_decision'].get('selected_operating_point')}.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-workers", type=int, default=0, help="Reserved for spawn-safe episode parallelism; 0 is serial.")
    parser.add_argument("--latency-warmup", type=int, default=200)
    parser.add_argument("--latency-iterations", type=int, default=2000)
    args = parser.parse_args()
    if args.eval_workers != 0:
        print("[note] This final paired runner uses deterministic serial MuJoCo episodes; --eval-workers currently must be 0.", flush=True)
        raise SystemExit(2)
    run_experiment(args.output_dir, tuple(args.seeds), args.device, args.eval_device,
                   args.latency_warmup, args.latency_iterations)


if __name__ == "__main__":
    main()
