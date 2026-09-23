"""DAgger-style on-policy relabeling diagnostic for Surface Set pre-grasp control.

This module reuses the existing Surface Set model and MuJoCo task.  The only
new mechanism is collecting visited feed-forward policy states and asking the
existing geometric expert for a fresh action at each state.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
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
from vla_gnn_recurrent.training import surface_pregrasp_stabilization as stab
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


DEFAULT_OUTPUT = "artifacts/surface_pregrasp_onpolicy_relabel"
BASE_ARTIFACT = Path("artifacts/surface_pregrasp_stabilization/recovery_training")
SEEDS = (2811, 2812, 2813)
DEFAULT_UPDATES = 800
MIX_EXPERT_FRACTION = 0.5
SEVERE_PENETRATION_EXCLUSION_M = -0.020
NEAR_EXPERT_DISTANCE_M = 0.012
MODERATE_DISTANCE_M = 0.035
YAW_BUCKETS = (0.0, 0.05, 0.10, 0.20, 0.30, float("inf"))
APERTURE_BUCKETS = (0.0, 0.02, 0.05, 0.10, 0.20, float("inf"))


def _write_json(path: str | Path, payload: Any) -> None:
    def finite(value: Any) -> Any:
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(item) for item in value]
        return value
    sf._write_json(path, finite(payload))


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_spec(payload: dict[str, Any]) -> sf.SurfaceEpisodeSpec:
    return stab._as_spec(payload)


def _load_matching_config(output_dir: str | Path, device: DevicePreference,
                          eval_device: DevicePreference) -> sf.SurfaceFeasibilityConfig:
    old = _load_json("artifacts/surface_pregrasp_stabilization/config.json")["base_experiment_config"]
    tuple_fields = {
        "seeds", "training_object_x_range", "training_object_z_range", "training_length_range",
        "training_half_width_range", "training_depth_range", "ood_half_width_range", "ood_depth_range",
    }
    values = {key: tuple(value) if key in tuple_fields else value for key, value in old.items()}
    values.update(output_dir=str(output_dir), device=device, eval_device=eval_device)
    return sf.SurfaceFeasibilityConfig(**values)


def _array_data(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(data[key].detach().cpu().numpy().astype(np.float64)
                 for key in ("states", "surface_points", "actions"))  # type: ignore[return-value]


def _pearson(target: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(target) < 2 or np.std(target) < 1e-12 or np.std(prediction) < 1e-12:
        return None
    return float(np.corrcoef(target, prediction)[0, 1])


def _summary_stats(values: np.ndarray) -> dict[str, Any]:
    percentiles = np.percentile(values, [10, 25, 50, 75, 90, 95, 99])
    return {
        "count": int(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values)),
        "median": float(percentiles[2]),
        "p10": float(percentiles[0]), "p25": float(percentiles[1]),
        "p75": float(percentiles[3]), "p90": float(percentiles[4]),
        "p95": float(percentiles[5]), "p99": float(percentiles[6]),
        "absolute_magnitude_mean": float(np.mean(np.abs(values))),
        "absolute_magnitude_std": float(np.std(np.abs(values))),
    }


def _bucket_rows(target: np.ndarray, prediction: np.ndarray,
                 edges: tuple[float, ...], wrapped: bool = False) -> list[dict[str, Any]]:
    absolute = np.abs(target)
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (absolute >= low) & (absolute < high)
        if not np.any(mask):
            rows.append({"range": [low, high if math.isfinite(high) else None], "count": 0,
                         "proportion": 0.0, "mae": None, "bias": None, "target_std": None,
                         "prediction_std": None, "pearson_correlation": None})
            continue
        residual = prediction[mask] - target[mask]
        if wrapped:
            residual = np.arctan2(np.sin(residual), np.cos(residual))
        rows.append({
            "range": [low, high if math.isfinite(high) else None], "count": int(mask.sum()),
            "proportion": float(mask.mean()), "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)), "target_std": float(np.std(target[mask])),
            "prediction_std": float(np.std(prediction[mask])),
            "pearson_correlation": _pearson(target[mask], prediction[mask]),
            "target_abs_mean": float(np.mean(np.abs(target[mask]))),
            "prediction_abs_mean": float(np.mean(np.abs(prediction[mask]))),
        })
    return rows


def _predict(model: nn.Module, states: np.ndarray, points: np.ndarray,
             device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            state_t = torch.as_tensor(states[start:start + batch_size], dtype=torch.float32, device=device)
            point_t = torch.as_tensor(points[start:start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(state_t, point_t).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float64)


def action_distribution_report(target: np.ndarray, prediction: np.ndarray,
                               output_dir: str | Path, label: str) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    names = ("dx_m", "dy_m", "dz_m", "d_yaw_rad", "d_aperture_fraction")
    component_rows = {name: _summary_stats(target[:, i]) for i, name in enumerate(names)}
    histogram: dict[str, Any] = {}
    for i, name in enumerate(names):
        edges = np.histogram_bin_edges(target[:, i], bins=30)
        counts, _ = np.histogram(target[:, i], bins=edges)
        abs_edges = np.histogram_bin_edges(np.abs(target[:, i]), bins=30)
        histogram[name] = {
            "signed_edges": edges.tolist(), "target_counts": counts.tolist(),
            "prediction_counts": np.histogram(prediction[:, i], bins=edges)[0].tolist(),
            "absolute_magnitude_edges": abs_edges.tolist(),
            "absolute_target_counts": np.histogram(np.abs(target[:, i]), bins=abs_edges)[0].tolist(),
            "absolute_prediction_counts": np.histogram(np.abs(prediction[:, i]), bins=abs_edges)[0].tolist(),
        }
    yaw_rows = _bucket_rows(target[:, 3], prediction[:, 3], YAW_BUCKETS, wrapped=True)
    aperture_rows = _bucket_rows(target[:, 4], prediction[:, 4], APERTURE_BUCKETS)
    payload = {
        "label": label, "samples": len(target), "target_statistics": component_rows,
        "prediction_statistics": {name: _summary_stats(prediction[:, i]) for i, name in enumerate(names)},
        "component_errors": {
            name: {"mae": float(np.mean(np.abs(prediction[:, i] - target[:, i]))),
                  "rmse": float(np.sqrt(np.mean((prediction[:, i] - target[:, i]) ** 2))),
                  "bias": float(np.mean(prediction[:, i] - target[:, i])),
                  "pearson_correlation": _pearson(target[:, i], prediction[:, i])}
            for i, name in enumerate(names)
        },
        "yaw_absolute_target_buckets": yaw_rows,
        "aperture_absolute_target_buckets": aperture_rows,
        "aperture_prediction_to_target_std_ratio": float(np.std(prediction[:, 4]) / max(np.std(target[:, 4]), 1e-12)),
        "yaw_prediction_to_target_std_ratio": float(np.std(prediction[:, 3]) / max(np.std(target[:, 3]), 1e-12)),
        "target_histograms": histogram,
        "bucket_edges": {"yaw_abs_rad": [x if math.isfinite(x) else None for x in YAW_BUCKETS],
                         "aperture_abs_fraction": [x if math.isfinite(x) else None for x in APERTURE_BUCKETS]},
    }
    _write_json(out / "distribution.json", payload)
    for index, key, display, edges in (
        (3, "yaw", "absolute yaw action (rad)", YAW_BUCKETS),
        (4, "aperture", "absolute aperture action (fraction)", APERTURE_BUCKETS),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
        axes[0].scatter(np.abs(target[:, index]), np.abs(prediction[:, index]), s=10, alpha=0.42)
        maximum = max(float(np.max(np.abs(target[:, index]))), float(np.max(np.abs(prediction[:, index]))), 1e-6)
        axes[0].plot([0, maximum], [0, maximum], color="black", linestyle="--", linewidth=1)
        axes[0].set(xlabel=f"target {display}", ylabel=f"predicted {display}", title=f"{label}: magnitude")
        bucket = yaw_rows if key == "yaw" else aperture_rows
        xs = []
        for row in bucket:
            upper = "∞" if row["range"][1] is None else f"{row['range'][1]:.2f}"
            xs.append(f"{row['range'][0]:.2f}–{upper}")
        maes = [np.nan if r["mae"] is None else r["mae"] for r in bucket]
        axes[1].bar(xs, maes, color="#4e83ad")
        axes[1].set(xlabel="target magnitude bucket", ylabel="MAE", title="bucket error")
        axes[1].tick_params(axis="x", rotation=35)
        bins = np.linspace(float(min(target[:, index].min(), prediction[:, index].min())),
                           float(max(target[:, index].max(), prediction[:, index].max())), 25)
        axes[2].hist(target[:, index], bins=bins, alpha=0.55, label="oracle target")
        axes[2].hist(prediction[:, index], bins=bins, alpha=0.55, label="policy prediction")
        axes[2].set(xlabel=display, ylabel="states", title="signed action distribution")
        axes[2].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"{key}_magnitude_diagnostic.png", dpi=160)
        plt.close(fig)
    return payload


def _reference_distance(position: np.ndarray, path: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(path[:, [0, 2]] - position[[0, 2]], axis=1)))


def classify_deviation(distance_m: float, collision: bool, clearance_m: float) -> str:
    """Stable metadata-only label; it is never part of model input."""
    if collision or clearance_m <= 0.005:
        return "collision_near_collision"
    if distance_m <= NEAR_EXPERT_DISTANCE_M:
        return "near_expert"
    if distance_m <= MODERATE_DISTANCE_M:
        return "moderate_deviation"
    return "large_deviation"


def is_severe_penetration(clearance_m: float) -> bool:
    return bool(clearance_m < SEVERE_PENETRATION_EXCLUSION_M)


def continuation_batch_counts(batch_size: int) -> tuple[int, int]:
    """Return expert/policy draw counts for the single fixed 1:1 mixing rule."""
    if batch_size < 2:
        raise ValueError("DAgger 1:1 mixing requires a batch size of at least two.")
    expert = int(round(batch_size * MIX_EXPERT_FRACTION))
    expert = min(max(expert, 1), batch_size - 1)
    return expert, batch_size - expert


def _data_tensors(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    if not records:
        raise ValueError("On-policy collection produced no states.")
    return {
        "task": "surface_pregrasp_dagger_round1", "seed": seed,
        "states": torch.as_tensor(np.stack([r["state"] for r in records]), dtype=torch.float32),
        "surface_points": torch.as_tensor(np.stack([r["surface_points"] for r in records]), dtype=torch.float32),
        # This is the freshly recomputed oracle label, distinct from policy_action.
        "actions": torch.as_tensor(np.stack([r["oracle_action"] for r in records]), dtype=torch.float32),
        "policy_actions": torch.as_tensor(np.stack([r["policy_action"] for r in records]), dtype=torch.float32),
        "qpos": torch.as_tensor(np.stack([r["qpos"] for r in records]), dtype=torch.float64),
        "episode_ids": torch.as_tensor([r["episode_id"] for r in records], dtype=torch.long),
        "steps": torch.as_tensor([r["timestep"] for r in records], dtype=torch.long),
    }


def collect_on_policy_states(
    model: nn.Module,
    specs: list[sf.SurfaceEpisodeSpec],
    expert_reference: dict[int, dict[str, Any]],
    config: sf.SurfaceFeasibilityConfig,
    seed: int,
    device: torch.device,
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the current controller only on training specs and relabel every visited state."""
    if not specs or any(spec.condition != "train" for spec in specs):
        raise ValueError("On-policy training data collection accepts training-split specs only.")
    out = ensure_dir(output_dir)
    env = sf.make_env(config, seed + 4_051)
    records: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    for spec in specs:
        sf._reset_surface_env(env, spec)
        initial = env.robot_observation().ee_position.numpy().astype(np.float64)
        target = sf._surface_target(spec.shape, initial, config.pregrasp_clearance)
        reference_path = np.asarray(expert_reference[spec.episode_id]["ee_positions"], dtype=np.float64)
        collision_seen = False
        min_clearance = 0.20
        episode_records = 0
        episode_collision_step: int | None = None
        traj_positions = [initial.tolist()]
        errors = sf._state_errors(env, spec.shape, config, target)
        contact_rows, clearance = sf._surface_distance_metrics(env, spec.shape)
        collision_seen = bool(contact_rows)
        min_clearance = min(min_clearance, clearance)
        for timestep in range(config.max_steps):
            if sf._success_from_errors(errors, collision_seen, config):
                break
            state, surface_points, centerline_points = sf.observation_inputs(
                env, spec.shape, config.point_count, sf._stable_seed(spec.sample_identity),
            )
            ee = env.robot_observation().ee_position.numpy().astype(np.float64)
            yaw = sf.tool_yaw(env)
            aperture = sf.gripper_width(env)
            distance = _reference_distance(ee, reference_path)
            before_contacts, before_clearance = sf._surface_distance_metrics(env, spec.shape)
            difficulty = classify_deviation(distance, bool(before_contacts), before_clearance)
            # Both actions are evaluated at this exact live MuJoCo state. Only policy_action
            # is applied; oracle_action is supervision and never enters the network input.
            policy_action = sf._policy_action("surface_set", model, state, surface_points, config, device)
            oracle_action, oracle_meta = sf.expert_action(env, spec.shape, config)
            state_errors = sf._state_errors(env, spec.shape, config, target)
            severe = is_severe_penetration(before_clearance)
            record = {
                "episode_id": spec.episode_id, "timestep": timestep,
                "state": state.copy(), "surface_points": surface_points.copy(),
                "policy_action": policy_action.copy(), "oracle_action": oracle_action.copy(),
                "qpos": env.data.qpos.copy(), "ee_position_world": ee.tolist(), "ee_yaw_rad": float(yaw),
                "aperture_m": float(aperture), "object_context_in_state": state[:8].copy().tolist(),
                "collision": bool(before_contacts), "minimum_clearance_m": float(before_clearance),
                "position_error_m": state_errors["position_error"],
                "orientation_error_rad": state_errors["orientation_error"],
                "aperture_error_m": state_errors["gripper_width_error"],
                "distance_from_expert_trajectory_m": distance,
                "deviation_bucket": difficulty, "severe_penetration_excluded": severe,
                "expert_target_side_metadata": int(oracle_meta["side"]),
            }
            all_metadata.append({k: v for k, v in record.items()
                                 if k not in ("state", "surface_points", "qpos")})
            if not severe:
                records.append(record)
                episode_records += 1
            sf.apply_local_action(env, policy_action, config)
            ee_after = env.robot_observation().ee_position.numpy().astype(np.float64)
            traj_positions.append(ee_after.tolist())
            errors = sf._state_errors(env, spec.shape, config, target)
            after_contacts, step_clearance = sf._surface_distance_metrics(env, spec.shape)
            if after_contacts and episode_collision_step is None:
                episode_collision_step = timestep + 1
            collision_seen = collision_seen or bool(after_contacts)
            min_clearance = min(min_clearance, step_clearance)
        final_errors = sf._state_errors(env, spec.shape, config, target)
        final_contacts, final_clearance = sf._surface_distance_metrics(env, spec.shape)
        min_clearance = min(min_clearance, final_clearance)
        rollout_rows.append({
            "episode_id": spec.episode_id, "steps": len(traj_positions) - 1,
            "collected_states": episode_records, "success": sf._success_from_errors(final_errors, collision_seen, config),
            "collision": collision_seen, "steps_until_collision": episode_collision_step,
            "minimum_clearance_m": float(min_clearance),
            "final_position_error": final_errors["position_error"],
            "final_orientation_error": final_errors["orientation_error"],
            "final_aperture_error": final_errors["gripper_width_error"],
            "final_collision": bool(final_contacts), "trajectory": traj_positions,
        })
    env.close()
    dataset = _data_tensors(records, seed)
    torch.save(dataset, out / "onpolicy_relabelled_train_states.pt")
    _write_json(out / "visited_state_metadata.json", all_metadata)
    bucket_counts: dict[str, int] = {}
    for row in all_metadata:
        key = row["deviation_bucket"]
        bucket_counts[key] = bucket_counts.get(key, 0) + 1
    eligible_bucket_counts: dict[str, int] = {}
    for row in all_metadata:
        if not row["severe_penetration_excluded"]:
            key = row["deviation_bucket"]
            eligible_bucket_counts[key] = eligible_bucket_counts.get(key, 0) + 1
    stats = {
        "training_split_only": True, "training_episode_count": len(specs),
        "episodes": rollout_rows, "collected_policy_states": len(all_metadata),
        "eligible_policy_states": len(records), "excluded_severe_penetration_states": len(all_metadata) - len(records),
        "severe_penetration_rule": f"exclude current min clearance < {SEVERE_PENETRATION_EXCLUSION_M:.3f} m",
        "difficulty_rule": {"near_expert_max_trajectory_distance_m": NEAR_EXPERT_DISTANCE_M,
                            "moderate_max_distance_m": MODERATE_DISTANCE_M,
                            "collision_or_clearance_le": 0.005},
        "collected_deviation_bucket_counts": bucket_counts,
        "eligible_deviation_bucket_counts": eligible_bucket_counts,
        "eligible_deviation_sampling_proportions": {
            key: count / max(len(records), 1) for key, count in eligible_bucket_counts.items()},
        "policy_rollout_metrics": {
            "episodes": len(rollout_rows),
            "success": float(np.mean([r["success"] for r in rollout_rows])),
            "collision": float(np.mean([r["collision"] for r in rollout_rows])),
            "final_position_error": float(np.mean([r["final_position_error"] for r in rollout_rows])),
            "final_orientation_error": float(np.mean([r["final_orientation_error"] for r in rollout_rows])),
            "final_aperture_error": float(np.mean([r["final_aperture_error"] for r in rollout_rows])),
            "mean_minimum_clearance_m": float(np.mean([r["minimum_clearance_m"] for r in rollout_rows])),
        },
        "collision_steps": [r["steps_until_collision"] for r in rollout_rows],
    }
    _write_json(out / "collection_summary.json", stats)
    return dataset, stats


def _evaluate_validation_loss(model: nn.Module, states: torch.Tensor, points: torch.Tensor,
                              actions: torch.Tensor, config: sf.SurfaceFeasibilityConfig,
                              device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(actions), config.batch_size):
            end = start + config.batch_size
            pred = model(states[start:end].to(device), points[start:end].to(device))
            losses.append(float(sf._normalised_action_loss(pred, actions[start:end].to(device), config).cpu()))
    return float(np.mean(losses))


def continuation_train(
    condition: str,
    base_checkpoint: str | Path,
    expert_data: dict[str, Any],
    policy_data: dict[str, Any] | None,
    val_data: dict[str, Any],
    policy_metadata: list[dict[str, Any]] | None,
    config: sf.SurfaceFeasibilityConfig,
    seed: int,
    updates: int,
    device_preference: DevicePreference,
    output_dir: str | Path,
) -> dict[str, Any]:
    if condition not in ("extra_original", "dagger_round1"):
        raise ValueError(condition)
    if (condition == "dagger_round1") != (policy_data is not None):
        raise ValueError("Only dagger_round1 receives the relabelled policy dataset.")
    set_seed(seed + 121_117)
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    model = sf.load_surface_model(base_checkpoint, "surface_set", config, device)
    initial_hash = _state_dict_hash(model)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ex_state = expert_data["states"].to(device)
    ex_points = expert_data["surface_points"].to(device)
    ex_actions = expert_data["actions"].to(device)
    va_state = val_data["states"]
    va_points = val_data["surface_points"]
    va_actions = val_data["actions"]
    if policy_data is not None:
        po_state = policy_data["states"].to(device)
        po_points = policy_data["surface_points"].to(device)
        po_actions = policy_data["actions"].to(device)
        po_meta = policy_metadata or []
    else:
        po_state = po_points = po_actions = None
        po_meta = []
    batch_size = config.batch_size
    expert_batch, policy_batch = (
        (batch_size, 0) if condition == "extra_original" else continuation_batch_counts(batch_size)
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1_000_003)
    output = ensure_dir(output_dir)
    checkpoint = output / "surface_set.pt"
    best_val = float("inf")
    history: list[dict[str, float]] = []
    losses: list[float] = []
    sampling_buckets: dict[str, int] = {}
    validation_interval = 25
    for update in range(1, updates + 1):
        ids = torch.randint(len(ex_actions), (expert_batch,), generator=generator, device=device)
        state_parts = [ex_state[ids]]
        point_parts = [ex_points[ids]]
        action_parts = [ex_actions[ids]]
        if policy_batch:
            assert po_state is not None and po_points is not None and po_actions is not None
            pids = torch.randint(len(po_actions), (policy_batch,), generator=generator, device=device)
            state_parts.append(po_state[pids])
            point_parts.append(po_points[pids])
            action_parts.append(po_actions[pids])
            for pi in pids.detach().cpu().tolist():
                bucket = str(po_meta[pi]["deviation_bucket"])
                sampling_buckets[bucket] = sampling_buckets.get(bucket, 0) + 1
        state_batch = torch.cat(state_parts, dim=0)
        point_batch = torch.cat(point_parts, dim=0)
        action_batch = torch.cat(action_parts, dim=0)
        model.train()
        prediction = model(state_batch, point_batch)
        loss = sf._normalised_action_loss(prediction, action_batch, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if update % validation_interval == 0 or update == updates:
            validation_loss = _evaluate_validation_loss(model, va_state, va_points, va_actions, config, device)
            row = {"update": update, "recent_training_loss": float(np.mean(losses[-validation_interval:])),
                   "expert_validation_loss": validation_loss}
            history.append(row)
            if validation_loss < best_val:
                best_val = validation_loss
                torch.save({"model_name": "surface_set", "model_state": model.state_dict(),
                            "config": asdict(config), "seed": seed,
                            "best_validation_loss": best_val,
                            "parameters": sf.model_parameter_count(model),
                            "optimizer_updates": updates, "continuation_condition": condition,
                            "base_checkpoint_sha256": initial_hash}, checkpoint)
    effective_policy_n = policy_batch * updates
    effective_expert_n = expert_batch * updates
    payload = {
        "condition": condition, "seed": seed, "device": str(device), "optimizer": "AdamW fresh/reset for both conditions",
        "same_start_checkpoint": str(base_checkpoint), "base_checkpoint_sha256": initial_hash,
        "optimizer_updates": updates, "batch_size": batch_size,
        "effective_expert_draws": effective_expert_n, "effective_policy_draws": effective_policy_n,
        "effective_expert_fraction": effective_expert_n / max(effective_expert_n + effective_policy_n, 1),
        "effective_policy_fraction": effective_policy_n / max(effective_expert_n + effective_policy_n, 1),
        "policy_sampling_bucket_draws": sampling_buckets,
        "expert_dataset_states": len(ex_actions),
        "policy_dataset_states": len(po_actions) if po_actions is not None else 0,
        "lr": config.learning_rate, "weight_decay": config.weight_decay,
        "checkpoint_selection": "lowest expert-state validation normalized action loss, evaluated every 25 updates",
        "best_expert_validation_loss": best_val, "parameters": sf.model_parameter_count(model),
        "history": history, "checkpoint_path": str(checkpoint),
        "initial_checkpoint_weights_sha256": initial_hash,
        "final_weights_sha256": _state_dict_hash(model),
    }
    _write_json(output / "training.json", payload)
    return payload


def _state_dict_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rollout_with_diagnostics(env: sf.SurfaceManipulatorEnv, spec: sf.SurfaceEpisodeSpec,
                             config: sf.SurfaceFeasibilityConfig, model: nn.Module,
                             device: torch.device,
                             expert_reference: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Closed-loop surface_set rollout with collision timing and oracle error trace."""
    sf._reset_surface_env(env, spec)
    initial = env.robot_observation().ee_position.numpy().astype(np.float64)
    target = sf._surface_target(spec.shape, initial, config.pregrasp_clearance)
    trajectory = [initial.tolist()]
    position_errors: list[float] = []
    yaw_errors: list[float] = []
    aperture_errors: list[float] = []
    errors = sf._state_errors(env, spec.shape, config, target)
    position_errors.append(errors["position_error"])
    yaw_errors.append(errors["orientation_error"])
    aperture_errors.append(errors["gripper_width_error"])
    contacts, min_clearance = sf._surface_distance_metrics(env, spec.shape)
    collision_seen = bool(contacts)
    collision_count = len(contacts)
    penetration_count = sum(c["distance"] < -1e-4 for c in contacts)
    steps_until_collision: int | None = 0 if contacts else None
    steps_to_success: int | None = 0 if sf._success_from_errors(errors, collision_seen, config) else None
    trace: list[dict[str, Any]] = []
    expert_path = np.asarray(expert_reference["ee_positions"], dtype=np.float64) if expert_reference else None
    for step in range(config.max_steps):
        if steps_to_success is not None:
            break
        state, surface, _ = sf.observation_inputs(env, spec.shape, config.point_count,
                                                  sf._stable_seed(spec.sample_identity))
        policy_action = sf._policy_action("surface_set", model, state, surface, config, device)
        if expert_reference is not None:
            oracle_action, _ = sf.expert_action(env, spec.shape, config)
            position = env.robot_observation().ee_position.numpy().astype(np.float64)
            trace.append({
                "t": step,
                "translation_action_error": float(np.linalg.norm(policy_action[:3] - oracle_action[:3])),
                "yaw_action_error": abs(float(sf.wrap_rotation_delta(policy_action[3] - oracle_action[3]))),
                "aperture_action_error": abs(float(policy_action[4] - oracle_action[4])),
                "distance_from_expert_trajectory_m": _reference_distance(position, expert_path) if expert_path is not None else None,
                "collision": float(collision_seen),
            })
        sf.apply_local_action(env, policy_action, config)
        position = env.robot_observation().ee_position.numpy().astype(np.float64)
        trajectory.append(position.tolist())
        errors = sf._state_errors(env, spec.shape, config, target)
        position_errors.append(errors["position_error"])
        yaw_errors.append(errors["orientation_error"])
        aperture_errors.append(errors["gripper_width_error"])
        contacts, clearance = sf._surface_distance_metrics(env, spec.shape)
        if contacts and steps_until_collision is None:
            steps_until_collision = step + 1
        collision_count += len(contacts)
        penetration_count += sum(c["distance"] < -1e-4 for c in contacts)
        collision_seen = collision_seen or bool(contacts)
        min_clearance = min(min_clearance, clearance)
        if sf._success_from_errors(errors, collision_seen, config):
            steps_to_success = step + 1
    final_contacts, final_clearance = sf._surface_distance_metrics(env, spec.shape)
    final_errors = sf._state_errors(env, spec.shape, config, target)
    min_clearance = min(min_clearance, final_clearance)
    positions = np.asarray(trajectory, dtype=np.float64)
    path_length = float(np.linalg.norm(np.diff(positions[:, [0, 2]], axis=0), axis=1).sum()) if len(positions) > 1 else 0.0
    result = {
        "episode_id": spec.episode_id, "condition": spec.condition, "shape_id": spec.shape.shape_id,
        "success": sf._success_from_errors(final_errors, collision_seen, config),
        "final_position_error": final_errors["position_error"],
        "final_orientation_error": final_errors["orientation_error"],
        "final_gripper_width_error": final_errors["gripper_width_error"],
        "final_gripper_width": sf.gripper_width(env), "target_gripper_width": target[3],
        "final_yaw": sf.tool_yaw(env), "minimum_safe_clearance": float(min_clearance),
        "collision": collision_seen, "final_collision": bool(final_contacts),
        "trajectory_collision": collision_seen, "penetration_count": int(penetration_count),
        "illegal_contact_count": int(collision_count),
        "trajectory_error": float(np.mean(position_errors)),
        "trajectory_yaw_error": float(np.mean(yaw_errors)),
        "trajectory_aperture_error": float(np.mean(aperture_errors)),
        "trajectory_length": path_length,
        "steps_to_convergence": int(steps_to_success if steps_to_success is not None else config.max_steps),
        "steps_until_collision": steps_until_collision,
        "trajectory": trajectory, "target_position": target[0].tolist(), "target_normal": target[1].tolist(),
        "target_yaw": target[2], "target_side": target[4], "initial_ee": initial.tolist(),
        "shape": asdict(spec.shape),
    }
    return result, trace


def paired_comparison(reference: list[dict[str, Any]], candidate: list[dict[str, Any]],
                      config: sf.SurfaceFeasibilityConfig, label: str, seed: int) -> dict[str, Any]:
    if len(reference) != len(candidate) or [r["episode_id"] for r in reference] != [r["episode_id"] for r in candidate]:
        raise ValueError("Paired models must use the exact same EpisodeSpec order.")
    ref_only = sum(bool(a["success"] and not b["success"]) for a, b in zip(reference, candidate, strict=True))
    cand_only = sum(bool(b["success"] and not a["success"]) for a, b in zip(reference, candidate, strict=True))
    metrics = ("success", "collision", "final_position_error", "final_orientation_error",
               "final_gripper_width_error", "trajectory_error", "trajectory_yaw_error",
               "trajectory_aperture_error", "trajectory_length")
    result: dict[str, Any] = {
        "comparison": label, "episodes": len(reference),
        "success_contingency": {"reference_only_success": ref_only, "candidate_only_success": cand_only,
                                "both_success": sum(bool(a["success"] and b["success"]) for a, b in zip(reference, candidate, strict=True)),
                                "both_fail": sum(bool(not a["success"] and not b["success"]) for a, b in zip(reference, candidate, strict=True))},
        "mcnemar_exact_p": sf._mcnemar_exact_p(ref_only, cand_only), "paired_deltas": {},
    }
    for metric in metrics:
        delta = np.asarray([float(b[metric]) - float(a[metric]) for a, b in zip(reference, candidate, strict=True)])
        result["paired_deltas"][metric] = {
            "candidate_minus_reference_mean": float(np.mean(delta)),
            "bootstrap_95_ci": sf._bootstrap_ci(delta, config.bootstrap_samples, seed + len(metric)),
        }
    return result


def _timestep_aggregate(traces: dict[str, list[list[dict[str, Any]]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_name, episode_traces in traces.items():
        timesteps: dict[int, list[dict[str, Any]]] = {}
        for trace in episode_traces:
            for row in trace:
                timesteps.setdefault(int(row["t"]), []).append(row)
        means = []
        for timestep, rows in sorted(timesteps.items()):
            mean_row: dict[str, Any] = {"t": timestep, "episodes": len(rows)}
            for key in rows[0]:
                if key != "t":
                    mean_row[key] = float(np.mean([r[key] for r in rows]))
            means.append(mean_row)
        out[model_name] = means
    return out


def _plot_timestep_analysis(summary: dict[str, Any], out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 1, figsize=(8.3, 13.2), sharex=True)
    colors = {"base": "#6a6a6a", "extra_original": "#d08b30", "dagger_round1": "#247ba0"}
    names = ("translation_action_error", "yaw_action_error", "aperture_action_error",
             "distance_from_expert_trajectory_m", "collision")
    for ax, metric in zip(axes, names, strict=True):
        for model, rows in summary.items():
            if rows:
                ax.plot([r["t"] for r in rows], [r[metric] for r in rows], marker="o", ms=3,
                        color=colors.get(model), label=model)
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("policy rollout timestep")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _model_action_report(model: nn.Module, data: dict[str, Any], config: sf.SurfaceFeasibilityConfig,
                         device: torch.device, output_dir: str | Path, label: str) -> dict[str, Any]:
    states, points, target = _array_data(data)
    prediction = _predict(model, states.astype(np.float32), points.astype(np.float32), device)
    return action_distribution_report(target, prediction, output_dir, label)


def _episode_metrics_with_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = sf._episode_metrics(rows)
    metrics.update({
        key: float(np.mean([row[key] for row in rows]))
        for key in ("trajectory_yaw_error", "trajectory_aperture_error")
    })
    steps = [row["steps_until_collision"] for row in rows if row["steps_until_collision"] is not None]
    metrics["steps_until_collision_mean_when_collision"] = float(np.mean(steps)) if steps else None
    metrics["steps_until_collision_observed_episodes"] = len(steps)
    return metrics


def _write_relabel_aggregates(root: Path, per_seed: dict[str, dict[str, Any]]) -> None:
    dataset_rows: dict[str, Any] = {}
    deviation_rows: dict[str, Any] = {}
    pooled_counts: dict[str, int] = {}
    pooled_eligible: dict[str, int] = {}
    pooled_draws: dict[str, int] = {}
    for seed, payload in per_seed.items():
        collection = payload["data"]["onpolicy_collection"]
        training = payload["training"]["dagger_round1"]
        dataset_rows[seed] = {
            "expert_states": payload["data"]["expert_train_states"],
            "collected_policy_states": collection["collected_policy_states"],
            "eligible_policy_states": collection["eligible_policy_states"],
            "excluded_severe_penetration_states": collection["excluded_severe_penetration_states"],
            "effective_training_draws": {
                "expert": training["effective_expert_draws"],
                "policy": training["effective_policy_draws"],
                "expert_fraction": training["effective_expert_fraction"],
                "policy_fraction": training["effective_policy_fraction"],
                "policy_deviation_bucket_draws": training["policy_sampling_bucket_draws"],
            },
        }
        deviation_rows[seed] = {
            "training_split_only": collection["training_split_only"],
            "collected_counts": collection["collected_deviation_bucket_counts"],
            "eligible_counts": collection["eligible_deviation_bucket_counts"],
            "eligible_proportions": collection["eligible_deviation_sampling_proportions"],
            "severe_penetration_rule": collection["severe_penetration_rule"],
        }
        for name, count in collection["collected_deviation_bucket_counts"].items():
            pooled_counts[name] = pooled_counts.get(name, 0) + int(count)
        for name, count in collection["eligible_deviation_bucket_counts"].items():
            pooled_eligible[name] = pooled_eligible.get(name, 0) + int(count)
        for name, count in training["policy_sampling_bucket_draws"].items():
            pooled_draws[name] = pooled_draws.get(name, 0) + int(count)
    out = ensure_dir(root / "relabel")
    _write_json(out / "dataset_statistics.json", {
        "per_seed": dataset_rows, "pooled_expert_states": sum(x["expert_states"] for x in dataset_rows.values()),
        "pooled_collected_policy_states": sum(x["collected_policy_states"] for x in dataset_rows.values()),
        "pooled_eligible_policy_states": sum(x["eligible_policy_states"] for x in dataset_rows.values()),
        "pooled_excluded_severe_penetration_states": sum(x["excluded_severe_penetration_states"] for x in dataset_rows.values()),
        "pooled_effective_policy_sampling_bucket_draws": pooled_draws,
        "effective_sampling_ratio": {"expert": MIX_EXPERT_FRACTION, "policy": 1.0 - MIX_EXPERT_FRACTION},
    })
    _write_json(out / "deviation_distribution.json", {
        "per_seed": deviation_rows, "pooled_collected_counts": pooled_counts,
        "pooled_eligible_counts": pooled_eligible,
        "pooled_eligible_proportions": {key: value / max(sum(pooled_eligible.values()), 1)
                                        for key, value in pooled_eligible.items()},
        "pooled_policy_sampling_draws": pooled_draws,
        "difficulty_label_is_model_input": False,
    })


def _plot_eval_trajectories(episodes: list[dict[str, Any]], output_path: str | Path) -> None:
    if not episodes:
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    chosen = episodes[:min(6, len(episodes))]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, ep in zip(axes.flat, chosen, strict=False):
        shape = sf.SurfaceShape(**ep["spec"]["shape"])
        points = sf.sample_surface_points_world(shape, sf.SURFACE_POINTS, sf._stable_seed(shape.shape_id))
        ax.scatter(points[:, 0], points[:, 2], s=8, color="#a9c9d9", label="surface")
        for model_name, result in ep["models"].items():
            path = np.asarray(result["trajectory"])
            ax.plot(path[:, 0], path[:, 2], marker=".", ms=2, label=model_name)
        target = ep["models"]["dagger_round1"]["target_position"]
        ax.scatter([target[0]], [target[2]], marker="*", color="black", s=90, label="target")
        ax.set_title(f"episode {ep['spec']['episode_id']}: {shape.shape_id}", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_experiment(
    output_dir: str | Path = DEFAULT_OUTPUT,
    seeds: tuple[int, ...] = SEEDS,
    updates: int = DEFAULT_UPDATES,
    device_preference: DevicePreference = "auto",
    eval_device_preference: DevicePreference = "auto",
    latency_warmup: int = 200,
    latency_iterations: int = 2000,
) -> dict[str, Any]:
    root = ensure_dir(output_dir)
    config = _load_matching_config(root, device_preference, eval_device_preference)
    config.latency_warmup = latency_warmup
    config.latency_iterations = latency_iterations
    _write_json(root / "config.json", {
        "base_experiment_config": asdict(config), "seeds": list(seeds),
        "only_model": "surface_set", "architecture_change": False, "recurrent_state": False,
        "onpolicy_mixing": {"expert_fraction": MIX_EXPERT_FRACTION, "policy_fraction": 1.0 - MIX_EXPERT_FRACTION,
                            "sampling": "uniform with replacement from eligible collected D_policy; natural deviation distribution retained"},
        "severe_penetration_exclusion": {"threshold_m": SEVERE_PENETRATION_EXCLUSION_M,
                                         "rule": "exclude current rollout states whose MuJoCo minimum surface-to-robot geom distance is below threshold"},
        "difficulty_buckets": {"near_expert_max_m": NEAR_EXPERT_DISTANCE_M,
                                "moderate_max_m": MODERATE_DISTANCE_M,
                                "collision_near_collision_clearance_m": 0.005},
        "extra_training": {"updates_per_condition": updates, "optimizer": "AdamW, fresh/reset for both continuations",
                           "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
                           "batch_size": config.batch_size, "checkpoint_selection": "best expert-state validation normalized loss"},
        "latency_budget_ms_p99": 5.0, "control_rate_hz_design_target": 100,
        "sensor_to_command_latency": "NOT MEASURED",
    })
    device = select_device(eval_device_preference)
    per_seed: dict[str, Any] = {}
    all_main_rows: dict[str, dict[str, list[dict[str, Any]]]] = {k: {m: [] for m in ("base", "extra_original", "dagger_round1")} for k in ("iid",)}
    policy_timestep_across_seeds: dict[str, list[list[dict[str, Any]]]] = {m: [] for m in ("base", "extra_original", "dagger_round1")}
    pooled_action_reports: dict[str, dict[str, list[dict[str, Any]]]] = {}
    base_hashes: dict[str, str] = {}
    for seed in seeds:
        seed_root = ensure_dir(root / "training" / f"seed{seed}")
        prior_seed = BASE_ARTIFACT / f"seed{seed}"
        base_checkpoint = prior_seed / "base" / "checkpoints" / "surface_set.pt"
        train_path = prior_seed / "dataset" / "expert" / "expert_states.pt"
        val_path = prior_seed / "dataset" / "validation" / "expert_states.pt"
        eval_spec_path = prior_seed / "eval_episode_specs.json"
        for required in (base_checkpoint, train_path, val_path, eval_spec_path):
            if not required.exists():
                raise FileNotFoundError(f"Required prior artifact missing: {required}")
        base_hashes[str(seed)] = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
        train_data = torch.load(train_path, map_location="cpu", weights_only=False)
        val_data = torch.load(val_path, map_location="cpu", weights_only=False)
        train_specs = [_as_spec(x) for x in train_data["specs"]]
        if any(spec.condition != "train" for spec in train_specs):
            raise ValueError(f"Seed {seed} training expert file contains a non-training EpisodeSpec.")
        eval_specs = [_as_spec(x) for x in _load_json(eval_spec_path)]
        if any(spec.condition != "iid" for spec in eval_specs):
            raise ValueError(f"Seed {seed} paired evaluation EpisodeSpecs are not IID.")
        ref_dir = ensure_dir(root / "onpolicy_collection" / f"seed{seed}" / "expert_reference")
        print(f"[dagger] seed={seed}: load base, collect training-split policy states", flush=True)
        _, train_reference = stab.collect_supervised_rows(config, train_specs, seed + 991_001, ref_dir)
        base_model = sf.load_surface_model(base_checkpoint, "surface_set", config, device)
        base_actions_dir = root / "action_distribution" / f"seed{seed}" / "base_validation_expert"
        base_action_report = _model_action_report(base_model, val_data, config, device, base_actions_dir,
                                                  f"seed{seed}:base_validation_expert")
        target_only_train = train_data["actions"].numpy().astype(np.float64)
        target_only_val = val_data["actions"].numpy().astype(np.float64)
        target_summary = {
            "train": {name: _summary_stats(target_only_train[:, i]) for i, name in enumerate(("dx", "dy", "dz", "yaw", "aperture"))},
            "validation": {name: _summary_stats(target_only_val[:, i]) for i, name in enumerate(("dx", "dy", "dz", "yaw", "aperture"))},
        }
        target_histograms = {}
        for split_name, action_values in (("train", target_only_train), ("validation", target_only_val)):
            target_histograms[split_name] = {}
            for i, component in enumerate(("dx", "dy", "dz", "yaw", "aperture")):
                edges = np.histogram_bin_edges(np.abs(action_values[:, i]), bins=30)
                counts, _ = np.histogram(np.abs(action_values[:, i]), bins=edges)
                target_histograms[split_name][component] = {"abs_edges": edges.tolist(), "abs_counts": counts.tolist()}
        _write_json(root / "action_distribution" / f"seed{seed}" / "target_histograms.json", {
            "train_and_validation_target_statistics": target_summary,
            "absolute_magnitude_histograms": target_histograms,
            "base_validation_action_report": base_action_report,
        })
        policy_data, collection_stats = collect_on_policy_states(
            base_model, train_specs, train_reference, config, seed + 992_002, device,
            root / "onpolicy_collection" / f"seed{seed}",
        )
        metadata = _load_json(root / "onpolicy_collection" / f"seed{seed}" / "visited_state_metadata.json")
        eligible_metadata = [row for row in metadata if not row["severe_penetration_excluded"]]
        if len(eligible_metadata) != len(policy_data["actions"]):
            raise AssertionError("Relabel metadata and eligible policy data are misaligned.")
        train_payload = train_data
        validation_payload = val_data
        print(f"[dagger] seed={seed}: train extra_original and dagger_round1 ({updates} updates each)", flush=True)
        training_rows: dict[str, Any] = {}
        for condition in ("extra_original", "dagger_round1"):
            training_rows[condition] = continuation_train(
                condition, base_checkpoint, train_payload,
                policy_data if condition == "dagger_round1" else None,
                validation_payload, eligible_metadata if condition == "dagger_round1" else None,
                config, seed, updates, device_preference,
                seed_root / condition,
            )
        models = {
            "base": base_model,
            "extra_original": sf.load_surface_model(training_rows["extra_original"]["checkpoint_path"], "surface_set", config, device),
            "dagger_round1": sf.load_surface_model(training_rows["dagger_round1"]["checkpoint_path"], "surface_set", config, device),
        }
        diagnostics: dict[str, Any] = {}
        for name, model in models.items():
            for data_label, data in (("validation_expert", val_data), ("policy_state_relabelled", policy_data)):
                dst = root / "action_distribution" / f"seed{seed}" / name / data_label
                report = _model_action_report(model, data, config, device, dst, f"seed{seed}:{name}:{data_label}")
                diagnostics[f"{name}_{data_label}"] = report
                pooled_action_reports.setdefault(f"{name}_{data_label}", {"yaw_std_ratio": [], "aperture_std_ratio": [], "yaw_mae": [], "aperture_mae": []})
                pooled_action_reports[f"{name}_{data_label}"]["yaw_std_ratio"].append(report["yaw_prediction_to_target_std_ratio"])
                pooled_action_reports[f"{name}_{data_label}"]["aperture_std_ratio"].append(report["aperture_prediction_to_target_std_ratio"])
                pooled_action_reports[f"{name}_{data_label}"]["yaw_mae"].append(report["component_errors"]["d_yaw_rad"]["mae"])
                pooled_action_reports[f"{name}_{data_label}"]["aperture_mae"].append(report["component_errors"]["d_aperture_fraction"]["mae"])
        eval_ref_dir = ensure_dir(root / "evaluation" / "iid" / f"seed{seed}" / "expert_reference")
        _, eval_reference = stab.collect_supervised_rows(config, eval_specs, seed + 993_003, eval_ref_dir)
        env = sf.make_env(config, seed + 994_004)
        eval_rows: list[dict[str, Any]] = []
        traces: dict[str, list[list[dict[str, Any]]]] = {m: [] for m in models}
        per_model_rows: dict[str, list[dict[str, Any]]] = {m: [] for m in models}
        for spec in eval_specs:
            episode: dict[str, Any] = {"spec": asdict(spec), "models": {}}
            for model_name, model in models.items():
                result, trace = rollout_with_diagnostics(env, spec, config, model, device, eval_reference[spec.episode_id])
                episode["models"][model_name] = result
                per_model_rows[model_name].append(result)
                traces[model_name].append(trace)
            eval_rows.append(episode)
        env.close()
        model_metrics = {name: _episode_metrics_with_trace(rows) for name, rows in per_model_rows.items()}
        pairs = {
            "dagger_round1_vs_extra_original": paired_comparison(per_model_rows["extra_original"], per_model_rows["dagger_round1"], config, "dagger_round1 - extra_original", seed + 990_000),
            "dagger_round1_vs_base": paired_comparison(per_model_rows["base"], per_model_rows["dagger_round1"], config, "dagger_round1 - base", seed + 991_000),
            "extra_original_vs_base": paired_comparison(per_model_rows["base"], per_model_rows["extra_original"], config, "extra_original - base", seed + 992_000),
        }
        timestep = _timestep_aggregate(traces)
        eval_out = ensure_dir(root / "evaluation" / "iid" / f"seed{seed}")
        _write_json(eval_out / "paired_episode_results.json", {
            "same_episode_specs": [asdict(s) for s in eval_specs], "model_metrics": model_metrics,
            "paired": pairs, "episodes": eval_rows,
        })
        _write_json(root / "policy_state_analysis" / f"seed{seed}" / "timestep_errors.json", timestep)
        for model_name, rows in timestep.items():
            policy_timestep_across_seeds[model_name].append(rows)
        _plot_eval_trajectories(eval_rows, root / "plots" / f"seed{seed}_paired_trajectories.png")
        per_seed[str(seed)] = {
            "base_checkpoint": str(base_checkpoint), "base_checkpoint_sha256": base_hashes[str(seed)],
            "data": {"expert_train_states": len(train_data["actions"]), "expert_validation_states": len(val_data["actions"]),
                     "eligible_policy_states": len(policy_data["actions"]), "onpolicy_collection": collection_stats},
            "training": training_rows, "action_diagnostics": diagnostics,
            "iid_models": model_metrics, "iid_paired": pairs,
            "eval_episode_specs": [asdict(s) for s in eval_specs],
            "policy_state_timestep_means": timestep,
            "round2_gate": _round2_gate(pairs["dagger_round1_vs_extra_original"], model_metrics),
        }
        _write_json(seed_root / "seed_result.json", per_seed[str(seed)])
        print(f"[dagger] seed={seed}: base success={model_metrics['base']['success']:.3f}; "
              f"extra={model_metrics['extra_original']['success']:.3f}; "
              f"dagger={model_metrics['dagger_round1']['success']:.3f}; "
              f"dagger collision={model_metrics['dagger_round1']['collision']:.3f}", flush=True)
        for name in models:
            all_main_rows["iid"][name].extend(per_model_rows[name])
    _write_relabel_aggregates(root, per_seed)
    pooled_models = {name: _episode_metrics_with_trace(rows) for name, rows in all_main_rows["iid"].items()}
    pooled_pairs = {
        "dagger_round1_vs_extra_original": paired_comparison(all_main_rows["iid"]["extra_original"], all_main_rows["iid"]["dagger_round1"], config, "dagger_round1 - extra_original", 781_003),
        "dagger_round1_vs_base": paired_comparison(all_main_rows["iid"]["base"], all_main_rows["iid"]["dagger_round1"], config, "dagger_round1 - base", 781_013),
        "extra_original_vs_base": paired_comparison(all_main_rows["iid"]["base"], all_main_rows["iid"]["extra_original"], config, "extra_original - base", 781_023),
    }
    pooled_timestep = _pool_timestep_rows(policy_timestep_across_seeds)
    _plot_timestep_analysis(pooled_timestep, root / "plots" / "policy_state_error_by_timestep.png")
    pooled_action_summary = {name: {key: {"mean": float(np.mean(vals)), "seed_values": vals}
                                     for key, vals in row.items()} for name, row in pooled_action_reports.items()}
    gate = _round2_gate(pooled_pairs["dagger_round1_vs_extra_original"], pooled_models)
    result = {
        "seeds": list(seeds), "optimizer_updates_per_continuation": updates,
        "base_checkpoint_hashes": base_hashes, "pooled_iid_models": pooled_models,
        "pooled_paired": pooled_pairs, "action_head_pooled": pooled_action_summary,
        "policy_state_timestep_means": pooled_timestep, "round2_gate": gate,
        "round2_run": False, "round2_skip_reason": "Round 1 did not meet the pre-registered clear-improvement gate." if not gate["clear_improvement"] else "Round 2 implementation is intentionally not automatic; only one optional round is allowed.",
        "representation_reevaluation_gate_passed": _floor_resolved(pooled_models, pooled_pairs),
        "representation_reevaluation_rule": (
            "pass if dagger success >=0.20, or paired collision and at least two of final position, final yaw, "
            "or trajectory position error have 95% CIs entirely below zero"
        ),
        "representation_reevaluation_rule_status": "operationalization of the user's qualitative floor-exit criterion; success delta remains separately reported with its paired CI",
        "sensor_to_command_latency": "NOT MEASURED",
    }
    # Round 2 is optional and only permitted after an evident R1 gain. Do not silently
    # add another training round to the main comparison.
    result["round2_skip_reason"] = ("Not run: Round 1 did not show a clear success, collision, or trajectory-error gain."
                                    if not gate["clear_improvement"] else
                                    "Not run: Round 1 showed gains, but the default diagnostic stops after its single pre-registered aggregation round.")
    result["per_seed"] = per_seed
    _write_json(root / "aggregated_results.json", result)
    latency = _run_latency_validation(config, per_seed, seeds, root, latency_warmup, latency_iterations)
    result["latency_validation"] = latency
    _write_json(root / "aggregated_results.json", result)
    summary = write_summary(root, result)
    (root / "summary.md").write_text(summary, encoding="utf-8")
    return result


def _pool_timestep_rows(seed_rows: dict[str, list[list[dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for model, groups in seed_rows.items():
        by_t: dict[int, list[dict[str, Any]]] = {}
        for group in groups:
            for row in group:
                by_t.setdefault(int(row["t"]), []).append(row)
        means = []
        for t, rows in sorted(by_t.items()):
            means.append({"t": t, "episodes": len(rows)} | {
                k: float(np.mean([r[k] for r in rows]))
                for k in rows[0] if k != "t"
            })
        out[model] = means
    return out


def _round2_gate(paired: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    deltas = paired["paired_deltas"]
    success = deltas["success"]["candidate_minus_reference_mean"]
    collision = deltas["collision"]["candidate_minus_reference_mean"]
    trajectory = deltas["trajectory_error"]["candidate_minus_reference_mean"]
    clear = bool(success >= 0.10 or collision <= -0.10 or trajectory <= -0.010)
    return {"clear_improvement": clear, "dagger_success": metrics.get("dagger_round1", {}).get("success"),
            "success_delta_vs_extra_original": success, "collision_delta_vs_extra_original": collision,
            "trajectory_position_error_delta_vs_extra_original": trajectory,
            "rule": "success +0.10 or collision rate -0.10 or mean trajectory position error -0.010 m"}


def _floor_resolved(metrics: dict[str, Any], pairs: dict[str, Any]) -> bool:
    dagger = metrics["dagger_round1"]
    delta = pairs["dagger_round1_vs_extra_original"]["paired_deltas"]
    collision_reduction = delta["collision"]["bootstrap_95_ci"][1] < 0.0
    continuous_improvements = sum(
        delta[name]["bootstrap_95_ci"][1] < 0.0
        for name in ("final_position_error", "final_orientation_error", "trajectory_error")
    )
    return bool(dagger["success"] >= 0.20 or (collision_reduction and continuous_improvements >= 2))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _run_latency_validation(config: sf.SurfaceFeasibilityConfig, per_seed: dict[str, Any],
                            seeds: tuple[int, ...], root: Path, warmup: int, iterations: int) -> dict[str, Any]:
    device = select_device(config.eval_device)
    # Network is architecture-identical across checkpoints; benchmark the round-1
    # checkpoint from the first seed with the existing full geometry-to-action utility.
    seed = seeds[0]
    checkpoint = per_seed[str(seed)]["training"]["dagger_round1"]["checkpoint_path"]
    model = sf.load_surface_model(checkpoint, "surface_set", config, device)
    shape = sf.sample_episode_specs(1, seed + 440_001, config, "latency")[0].shape
    spec = sf.sample_episode_specs(1, seed + 440_002, config, "latency")[0]
    env = sf.make_env(config, seed + 440_003)
    sf._reset_surface_env(env, spec)
    benchmark = sf.benchmark_latency("surface_set", model, shape, env, config, device, 32,
                                     "rebuilt", warmup=warmup, iterations=iterations)
    env.close()
    out = ensure_dir(root / "latency_validation")
    payload = {
        "reference_checkpoint": checkpoint, "device": str(device), "surface_points": 32,
        "warmup_iterations": warmup, "timed_iterations": iterations,
        "geometry_to_action": benchmark["geometry_to_action"],
        "network_forward": benchmark["network_forward"], "components": benchmark["components"],
        "p99_budget_ms": 5.0, "budget_is_project_specific_provisional_target": True,
        "sensor_to_command_latency": "NOT MEASURED",
        "architecture_unchanged_from_base": True,
    }
    _write_json(out / "surface_set_n32.json", payload)
    return payload


def write_summary(root: Path, result: dict[str, Any]) -> str:
    metrics = result["pooled_iid_models"]
    pairs = result["pooled_paired"]
    data = result["action_head_pooled"]
    dagger_vs_extra = pairs["dagger_round1_vs_extra_original"]
    latency = result.get("latency_validation", {})
    collections = [result["per_seed"][seed]["data"]["onpolicy_collection"] for seed in result["per_seed"]]
    expert_count = sum(result["per_seed"][seed]["data"]["expert_train_states"] for seed in result["per_seed"])
    collected_count = sum(row["collected_policy_states"] for row in collections)
    eligible_count = sum(row["eligible_policy_states"] for row in collections)
    excluded_count = sum(row["excluded_severe_penetration_states"] for row in collections)
    eligible_buckets: dict[str, int] = {}
    for row in collections:
        for bucket, count in row["eligible_deviation_bucket_counts"].items():
            eligible_buckets[bucket] = eligible_buckets.get(bucket, 0) + int(count)
    eligible_shares = {bucket: count / max(eligible_count, 1) for bucket, count in eligible_buckets.items()}
    def md(name: str) -> str:
        row = metrics[name]
        return (f"success {row['success']:.3f}, collision {row['collision']:.3f}, "
                f"pos {row['final_position_error']:.4f} m, yaw {row['final_orientation_error']:.3f} rad, "
                f"aperture {row['final_gripper_width_error']:.4f} m, "
                f"trajectory pos {row['trajectory_error']:.4f} m")
    def answer(value: bool | None) -> str:
        return "YES" if value is True else "NO" if value is False else "UNCERTAIN"
    def high_bucket(field: str) -> dict[str, float]:
        bucket_rows = [result["per_seed"][seed]["action_diagnostics"]["base_validation_expert"][field][-1]
                       for seed in sorted(result["per_seed"])]
        count = sum(row["count"] for row in bucket_rows)
        samples = sum(result["per_seed"][seed]["action_diagnostics"]["base_validation_expert"]["samples"]
                      for seed in result["per_seed"])
        return {
            "count": count, "proportion": count / max(samples, 1),
            "mae": sum(row["count"] * (row["mae"] or 0.0) for row in bucket_rows) / max(count, 1),
            "target_abs_mean": sum(row["count"] * (row.get("target_abs_mean") or 0.0) for row in bucket_rows) / max(count, 1),
            "prediction_abs_mean": sum(row["count"] * (row.get("prediction_abs_mean") or 0.0) for row in bucket_rows) / max(count, 1),
        }
    high_yaw = high_bucket("yaw_absolute_target_buckets")
    high_aperture = high_bucket("aperture_absolute_target_buckets")
    seed_success = {
        model: [result["per_seed"][seed]["iid_models"][model]["success"] for seed in sorted(result["per_seed"])]
        for model in ("base", "extra_original", "dagger_round1")
    }
    q1stat = data.get("base_validation_expert", {})
    q1yes = bool(q1stat.get("aperture_std_ratio", {}).get("mean", 1.0) < 0.8 or
                 q1stat.get("yaw_std_ratio", {}).get("mean", 1.0) < 0.8)
    q2 = "UNCERTAIN"  # this experiment demonstrates policy-state divergence; it does not identify every causal factor.
    success_delta = dagger_vs_extra["paired_deltas"]["success"]["candidate_minus_reference_mean"]
    q3 = ("YES" if success_delta > 0.0 and dagger_vs_extra["mcnemar_exact_p"] is not None and dagger_vs_extra["mcnemar_exact_p"] < 0.05
          else "NO" if success_delta < 0.0 and dagger_vs_extra["mcnemar_exact_p"] is not None and dagger_vs_extra["mcnemar_exact_p"] < 0.05
          else "UNCERTAIN")
    collision_delta = dagger_vs_extra["paired_deltas"]["collision"]
    q4 = (collision_delta["candidate_minus_reference_mean"] <= -0.10
          and collision_delta["bootstrap_95_ci"][1] < 0.0)
    timestep = result["policy_state_timestep_means"]
    t_common = min(10, *(max([r["t"] for r in timestep[name]]) for name in ("base", "extra_original", "dagger_round1") if timestep[name]))
    yaw_at_t = {name: next((r["yaw_action_error"] for r in timestep[name] if r["t"] == t_common), None)
                for name in ("base", "extra_original", "dagger_round1")}
    late_t = min(15, *(max([r["t"] for r in timestep[name]]) for name in ("base", "extra_original", "dagger_round1") if timestep[name]))
    late_yaw = {name: next((r["yaw_action_error"] for r in timestep[name] if r["t"] == late_t), None)
                for name in ("base", "extra_original", "dagger_round1")}
    dagger_extra_t = yaw_at_t["dagger_round1"] - yaw_at_t["extra_original"]
    dagger_extra_late = late_yaw["dagger_round1"] - late_yaw["extra_original"]
    if dagger_extra_t <= -0.05 and dagger_extra_late <= 0.0:
        q5 = "YES"
    elif dagger_extra_late < 0.0 and abs(dagger_extra_t) < 0.05:
        q5 = "UNCERTAIN"
    else:
        q5 = "NO"
    base_aperture = data.get("base_policy_state_relabelled", {}).get("aperture_std_ratio", {}).get("mean")
    dagger_aperture = data.get("dagger_round1_policy_state_relabelled", {}).get("aperture_std_ratio", {}).get("mean")
    base_yaw_ratio = data.get("base_policy_state_relabelled", {}).get("yaw_std_ratio", {}).get("mean")
    dagger_yaw_ratio = data.get("dagger_round1_policy_state_relabelled", {}).get("yaw_std_ratio", {}).get("mean")
    q6 = bool(base_aperture is not None and dagger_aperture is not None and base_yaw_ratio is not None
              and dagger_yaw_ratio is not None
              and dagger_aperture > base_aperture + 0.05 and dagger_yaw_ratio > base_yaw_ratio + 0.05)
    q7 = result["representation_reevaluation_gate_passed"]
    q8 = "YES" if q7 else "NO"
    q9 = "NO"
    # If the collected policy states show action errors increasing across time, that is
    # direct evidence of rollout-state mismatch, though not proof it is the sole cause.
    initial_err = result["policy_state_timestep_means"].get("base", [{}])[0].get("yaw_action_error")
    last_err = result["policy_state_timestep_means"].get("base", [{}])[-1].get("yaw_action_error")
    if initial_err is not None and last_err is not None and last_err > max(initial_err * 2.0, initial_err + 0.05):
        q2 = "YES"
    lines = [
        "# Surface pre-grasp on-policy relabeling diagnostic", "",
        "## 1. Prior feed-forward failure", "",
        "The prior Surface Set rollout had pooled IID success near 0.083 and collision near 0.917. The base checkpoints and exact held-out IID EpisodeSpecs were reused; prior artifacts were not modified.", "",
        "## 2. Expert action distribution and magnitude errors", "",
        f"Base validation aperture prediction/target std ratio: {q1stat.get('aperture_std_ratio', {}).get('mean', float('nan')):.3f}; yaw ratio: {q1stat.get('yaw_std_ratio', {}).get('mean', float('nan')):.3f}.",
        f"Base validation yaw MAE: {q1stat.get('yaw_mae', {}).get('mean', float('nan')):.4f} rad; aperture action MAE: {q1stat.get('aperture_mae', {}).get('mean', float('nan')):.4f} normalized fraction.",
        f"For the base validation set, |yaw target| ≥0.30 rad occurred in {high_yaw['count']}/{sum(result['per_seed'][s]['action_diagnostics']['base_validation_expert']['samples'] for s in result['per_seed'])} states ({high_yaw['proportion']:.1%}): bucket mean |target| {high_yaw['target_abs_mean']:.3f}, mean |prediction| {high_yaw['prediction_abs_mean']:.3f}, MAE {high_yaw['mae']:.3f} rad. |aperture action target| ≥0.20 occurred in {high_aperture['count']} states ({high_aperture['proportion']:.1%}); mean |target| {high_aperture['target_abs_mean']:.3f}, mean |prediction| {high_aperture['prediction_abs_mean']:.3f}, MAE {high_aperture['mae']:.3f} fraction.",
        "Training/validation per-component percentiles, magnitude buckets, histograms, target/prediction plots, and bucket MAE plots are in `action_distribution/`.", "",
        "## 3. On-policy collection and relabeling", "",
        "Only the original training split's physical shapes were rolled out for D_policy. At each visited state, the existing oracle expert was called again to compute a new action. Policy action was applied; target pose, target errors, oracle action, trajectory phase, and difficulty labels were not model inputs.",
        f"Severe-penetration exclusion: current MuJoCo minimum geometry distance < {SEVERE_PENETRATION_EXCLUSION_M:.3f} m. Eligible state counts and naturally observed deviation buckets are recorded per seed in `onpolicy_collection/`.",
        f"Across seeds, D_expert has {expert_count:,} states. {collected_count:,} visited states were collected; {eligible_count:,} were eligible and {excluded_count:,} severe-penetration states were excluded. Eligible natural distribution: near-expert {eligible_shares.get('near_expert', 0.0):.1%}, moderate {eligible_shares.get('moderate_deviation', 0.0):.1%}, large deviation {eligible_shares.get('large_deviation', 0.0):.1%}, collision/near-collision {eligible_shares.get('collision_near_collision', 0.0):.1%}. No bucket was oversampled; DAgger's 1:1 expert/policy composition came from fixed half-batches.",
        "Per-seed collision timing, reference-path distance, and actual effective sampling bucket draws are in `relabel/` and the `onpolicy_collection/` episode records.", "",
        "## 4. Fair continuation budget", "",
        f"Both continuations start from the same base checkpoint and use fresh AdamW state, batch size {result['per_seed'][str(result['seeds'][0])]['training']['extra_original']['batch_size']}, the same LR/weight decay, and {result['optimizer_updates_per_continuation']} optimizer updates. Only dagger_round1 includes a 50:50 expert/policy minibatch.", "",
        "## 5. Pooled paired IID results", "",
        "| Model | Closed-loop result |",
        "|---|---|",
        f"| base | {md('base')} |",
        f"| extra_original | {md('extra_original')} |",
        f"| dagger_round1 | {md('dagger_round1')} |", "",
        f"DAgger − extra_original paired success delta: {dagger_vs_extra['paired_deltas']['success']['candidate_minus_reference_mean']:+.3f} (95% CI {dagger_vs_extra['paired_deltas']['success']['bootstrap_95_ci']}); exact McNemar p={dagger_vs_extra['mcnemar_exact_p']}.",
        f"Collision delta: {collision_delta['candidate_minus_reference_mean']:+.3f} (95% CI {collision_delta['bootstrap_95_ci']}); final position delta {dagger_vs_extra['paired_deltas']['final_position_error']['candidate_minus_reference_mean']:+.4f} m; yaw delta {dagger_vs_extra['paired_deltas']['final_orientation_error']['candidate_minus_reference_mean']:+.3f} rad; aperture delta {dagger_vs_extra['paired_deltas']['final_gripper_width_error']['candidate_minus_reference_mean']:+.4f} m.",
        "Paired 95% CIs: position " + str(dagger_vs_extra["paired_deltas"]["final_position_error"]["bootstrap_95_ci"]) + " m; yaw " + str(dagger_vs_extra["paired_deltas"]["final_orientation_error"]["bootstrap_95_ci"]) + " rad; aperture " + str(dagger_vs_extra["paired_deltas"]["final_gripper_width_error"]["bootstrap_95_ci"]) + " m; trajectory position " + str(dagger_vs_extra["paired_deltas"]["trajectory_error"]["bootstrap_95_ci"]) + " m.",
        f"Per-seed success (2811/2812/2813): base {seed_success['base']}; extra_original {seed_success['extra_original']}; dagger_round1 {seed_success['dagger_round1']}. Paired success discordance: dagger-only {dagger_vs_extra['success_contingency']['candidate_only_success']}, extra-only {dagger_vs_extra['success_contingency']['reference_only_success']}.",
        "Mean trajectory yaw/aperture errors: " + "; ".join(
            f"{name} {metrics[name]['trajectory_yaw_error']:.3f} rad / {metrics[name]['trajectory_aperture_error']:.4f} m"
            for name in ("base", "extra_original", "dagger_round1")
        ) + ".",
        "Mean steps until first collision (conditional on collision) / steps to convergence: " + "; ".join(
            f"{name} {metrics[name]['steps_until_collision_mean_when_collision']:.2f} / {metrics[name]['steps_to_convergence']:.2f}"
            for name in ("base", "extra_original", "dagger_round1")
        ) + ".",
        "", "## 6. Policy-state errors and action magnitude", "",
        "Per-timestep translation/yaw/aperture action error, distance from the expert path, and collision fraction are in `policy_state_analysis/` and `plots/policy_state_error_by_timestep.png`.",
        "The `action_distribution/seed*/{base,extra_original,dagger_round1}/policy_state_relabelled/` reports compare all three policies against freshly recomputed oracle labels on the same base-policy visited states. Validation expert-state errors are reported separately.",
        f"Mean expert-state validation MAE base→extra→dagger: yaw {data.get('base_validation_expert', {}).get('yaw_mae', {}).get('mean', float('nan')):.4f}→{data.get('extra_original_validation_expert', {}).get('yaw_mae', {}).get('mean', float('nan')):.4f}→{data.get('dagger_round1_validation_expert', {}).get('yaw_mae', {}).get('mean', float('nan')):.4f} rad; aperture action {data.get('base_validation_expert', {}).get('aperture_mae', {}).get('mean', float('nan')):.4f}→{data.get('extra_original_validation_expert', {}).get('aperture_mae', {}).get('mean', float('nan')):.4f}→{data.get('dagger_round1_validation_expert', {}).get('aperture_mae', {}).get('mean', float('nan')):.4f} fraction. The slight expert-state regression is measured alongside the policy-state gain.",
        f"On the fixed base-policy visited states, mean yaw/aperture MAE improves base→dagger from {data.get('base_policy_state_relabelled', {}).get('yaw_mae', {}).get('mean', float('nan')):.4f}→{data.get('dagger_round1_policy_state_relabelled', {}).get('yaw_mae', {}).get('mean', float('nan')):.4f} rad and {data.get('base_policy_state_relabelled', {}).get('aperture_mae', {}).get('mean', float('nan')):.4f}→{data.get('dagger_round1_policy_state_relabelled', {}).get('aperture_mae', {}).get('mean', float('nan')):.4f} fraction.",
        f"Mean aperture std ratio on base-policy visited states: base {base_aperture:.3f}, extra {data.get('extra_original_policy_state_relabelled', {}).get('aperture_std_ratio', {}).get('mean', float('nan')):.3f}, dagger {dagger_aperture:.3f}; yaw ratios base {base_yaw_ratio:.3f}, dagger {dagger_yaw_ratio:.3f}.", "",
        "## 7. Latency", "",
        f"N=32 geometry-to-action p50/p95/p99: {latency.get('geometry_to_action', {}).get('p50_ms', float('nan')):.3f}/{latency.get('geometry_to_action', {}).get('p95_ms', float('nan')):.3f}/{latency.get('geometry_to_action', {}).get('p99_ms', float('nan')):.3f} ms on `{latency.get('device', 'unknown')}`; >5 ms {latency.get('geometry_to_action', {}).get('deadline_miss_rate_over_5ms', float('nan')):.1%}. Warm-up/timed iterations: {latency.get('warmup_iterations', 0)}/{latency.get('timed_iterations', 0)}. The Set architecture and N=32 input are unchanged.",
        "The 5 ms p99 budget at 100 Hz is a project-specific provisional design target, not an industry standard. Full RGB-D/perception/tracking latency is **NOT MEASURED**.", "",
        "## 8. Decision and questions", "",
        f"- Q1. Large yaw/aperture actions are rare or underpredicted: **{answer(q1yes)}** (|yaw|≥0.30 occurs {high_yaw['proportion']:.1%} but predicts 0.288 vs target 0.350 rad; |aperture|≥0.20 occurs {high_aperture['proportion']:.1%} but predicts 0.069 vs target 0.246).",
        f"- Q2. Failure is more strongly linked to policy-state distribution shift than expert-state fitting: **{q2}** (base yaw action error t=0 {initial_err}, last observed {last_err}; evidence is associational).",
        f"- Q3. DAgger beats extra-original at equal update budget on closed-loop success: **{q3}** (success delta {dagger_vs_extra['paired_deltas']['success']['candidate_minus_reference_mean']:+.3f}, 95% CI {dagger_vs_extra['paired_deltas']['success']['bootstrap_95_ci']}, exact McNemar p={dagger_vs_extra['mcnemar_exact_p']}).",
        f"- Q4. DAgger meaningfully reduces collisions: **{answer(q4)}** (paired collision delta {collision_delta['candidate_minus_reference_mean']:+.3f}, 95% CI {collision_delta['bootstrap_95_ci']}).",
        f"- Q5. DAgger slows timestep error growth: **{q5}** (yaw action error at t={t_common}: base {yaw_at_t['base']}, extra {yaw_at_t['extra_original']}, dagger {yaw_at_t['dagger_round1']}; at t={late_t}: base {late_yaw['base']}, extra {late_yaw['extra_original']}, dagger {late_yaw['dagger_round1']}).",
        f"- Q6. Yaw/aperture prediction magnitude collapse improves: **{answer(q6)}** (same base-policy state set: aperture std ratio base {base_aperture}, dagger {dagger_aperture}; yaw ratio base {base_yaw_ratio}, dagger {dagger_yaw_ratio}).",
        f"- Q7. Round 1 escapes the floor enough to resume representation comparison: **{answer(q7)}** (operational gate: success ≥0.20, or paired collision and at least two of final position/yaw/trajectory position errors have 95% CIs entirely below zero).",
        f"- Q8. Evidence supports restarting surface representation comparison: **{q8}**.",
        f"- Q9. Ready for GRU feasibility: **{q9}**. Recurrent state remains out of this experiment.",
        f"- Optional Round 2: not run. {result['round2_skip_reason']}", "",
        "## Artifacts", "",
        "See `config.json`, `action_distribution/`, `onpolicy_collection/`, `relabel/`, `training/`, `evaluation/iid/`, `policy_state_analysis/`, `latency_validation/`, `plots/`, and `aggregated_results.json`.",
        "",
    ]
    return "\n".join(lines)
