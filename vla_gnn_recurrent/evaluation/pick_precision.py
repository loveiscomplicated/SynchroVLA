from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from vla_gnn_recurrent.graph.manipulation_graph_builder import ManipulationGraphBuilder
from vla_gnn_recurrent.graph.types import GraphData
from vla_gnn_recurrent.models.pick_controller import PickFeedForwardController, PickRecurrentController
from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv
from vla_gnn_recurrent.training.pick_bc import (
    PickModelKind,
    _pick_env_config,
    _reactive_pick_expert_action,
    load_pick_dataset,
    load_pick_policy,
)
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


DISTANCE_BINS = (
    ("contact_alignment", 0.0, 0.020),
    ("near", 0.020, 0.050),
    ("medium", 0.050, 0.100),
    ("far", 0.100, float("inf")),
)


@dataclass
class PickPrecisionConfig:
    dataset_path: str = "artifacts/mujoco_learned_pick/demos/pick_demos_300.pt"
    output_dir: str = "artifacts/mujoco_pick_precision/diagnostics"
    ff_checkpoint_path: str | None = "artifacts/mujoco_learned_pick/checkpoints_ff_300/graph_feedforward.pt"
    gru_checkpoint_path: str | None = "artifacts/mujoco_learned_pick/checkpoints_gru_300_motion/graph_recurrent.pt"
    extra_checkpoints: tuple[tuple[str, str], ...] = ()
    split: str = "test"
    device: DevicePreference = "auto"
    max_step: float = 0.045
    saturation_fraction: float = 0.90
    failure_seed: int = 4242
    failure_steps: int = 180
    run_failure_diagnostic: bool = True


def analyze_pick_precision(config: PickPrecisionConfig) -> dict[str, Any]:
    dataset = load_pick_dataset(config.dataset_path)
    device = select_device(config.device)
    output_dir = ensure_dir(config.output_dir)
    max_step = float(dataset.get("env_config", {}).get("max_delta_ee", config.max_step))

    expert_rows = _expert_rows(dataset, config.split, max_step, config.saturation_fraction)
    policy_rows: list[dict[str, Any]] = []
    checkpoints = [
        ("FF-current", config.ff_checkpoint_path),
        ("GRU-current", config.gru_checkpoint_path),
    ]
    checkpoints.extend((label, path) for label, path in config.extra_checkpoints)
    for label, path in checkpoints:
        if path is None or not Path(path).exists():
            continue
        model, checkpoint = load_pick_policy(path, device)
        policy_rows.extend(
            _policy_rows(
                model=model,
                model_kind=checkpoint["model_kind"],
                label=label,
                dataset=dataset,
                split=config.split,
                device=device,
                max_step=max_step,
                saturation_fraction=config.saturation_fraction,
                step_reset=False,
            )
        )
        if isinstance(model, PickRecurrentController):
            policy_rows.extend(
                _policy_rows(
                    model=model,
                    model_kind=checkpoint["model_kind"],
                    label=f"{label}-step-reset-offline",
                    dataset=dataset,
                    split=config.split,
                    device=device,
                    max_step=max_step,
                    saturation_fraction=config.saturation_fraction,
                    step_reset=True,
                )
            )

    payload: dict[str, Any] = {
        "config": asdict(config),
        "max_step": max_step,
        "distance_bins": [{"name": name, "low": low, "high": high} for name, low, high in DISTANCE_BINS],
        "expert_by_phase": _summarize_rows(expert_rows, group_key="phase", model_key=None),
        "expert_by_distance_bin": _summarize_rows(expert_rows, group_key="distance_bin", model_key=None),
        "policy_by_phase": _summarize_rows(policy_rows, group_key="phase", model_key="model"),
        "policy_by_distance_bin": _summarize_rows(policy_rows, group_key="distance_bin", model_key="model"),
        "expert_gripper_distribution": _gripper_distribution(expert_rows),
        "created_at_unix": time.time(),
    }
    if config.run_failure_diagnostic and config.gru_checkpoint_path and Path(config.gru_checkpoint_path).exists():
        payload["gru_failure_diagnostic"] = run_pick_failure_diagnostic(
            checkpoint_path=config.gru_checkpoint_path,
            output_dir=output_dir / "failure_diagnostic",
            seed=config.failure_seed,
            device=config.device,
            max_steps=config.failure_steps,
        )

    json_path = output_dir / "pick_precision_analysis.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot_expert_histogram(expert_rows, output_dir / "expert_action_magnitude_histogram.png")
    _plot_magnitude_vs_distance(expert_rows, policy_rows, output_dir / "action_magnitude_vs_distance.png")
    _plot_saturation(policy_rows, output_dir / "policy_saturation_by_distance.png")
    payload["output_path"] = str(json_path)
    return payload


@torch.no_grad()
def run_pick_failure_diagnostic(
    checkpoint_path: str,
    output_dir: Path,
    seed: int,
    device: DevicePreference,
    max_steps: int = 180,
    object_x: float | None = None,
) -> dict[str, Any]:
    """Roll out the policy, logging expert corrective actions without executing them."""

    set_seed(seed)
    output_dir = ensure_dir(output_dir)
    torch_device = select_device(device)
    policy, checkpoint = load_pick_policy(checkpoint_path, torch_device)
    env = MujocoManipulatorEnv(_pick_env_config(max_steps=max_steps), seed=seed)
    graph_builder = ManipulationGraphBuilder(task="pick", include_object=True)
    rng = np.random.default_rng(seed)
    sampled_object_x = float(object_x if object_x is not None else rng.uniform(-0.015, 0.0))
    env.reset_pick_scene(object_x=sampled_object_x, randomize_robot=False)
    object_start = env.data.site_xpos[env.object_site_id].copy()
    hidden = policy.initial_hidden(torch_device) if isinstance(policy, PickRecurrentController) else None
    records: list[dict[str, Any]] = []
    lift_success = False
    lift_hold = 0

    for step_idx in range(max_steps):
        graph = graph_builder.build(env.observe()).to(torch_device)
        ee_before = env.data.site_xpos[env.ee_site_id].copy()
        object_before = env.data.site_xpos[env.object_site_id].copy()
        expert_delta, expert_gripper, expert_mode = _reactive_pick_expert_action(env, object_start)
        if isinstance(policy, PickRecurrentController):
            action, hidden = policy(graph, hidden)
            hidden_norm = float(hidden.norm().detach().cpu().item())
        elif isinstance(policy, PickFeedForwardController):
            action = policy(graph)
            hidden_norm = 0.0
        else:
            raise ValueError(f"Unsupported model type: {type(policy)}")
        policy_delta = action.delta_ee.detach().cpu()
        policy_xz = torch.stack([policy_delta[0], policy_delta[2]])
        expert_xz = torch.stack([expert_delta[0], expert_delta[2]])
        _, _, _, info = env.step_delta_ee(policy_delta, gripper=float(action.gripper.detach().cpu().item()))
        grasp = env.grasp_observation()
        ee_after = env.data.site_xpos[env.ee_site_id].copy()
        object_after = env.data.site_xpos[env.object_site_id].copy()
        object_to_ee = float(np.linalg.norm(object_after - ee_after))
        object_height_delta = float(grasp.object_height - object_start[2])
        if object_height_delta >= env.config.lift_height and object_to_ee <= 0.13:
            lift_hold += 1
        else:
            lift_hold = 0
        if lift_hold >= env.config.lift_hold_steps:
            lift_success = True
        records.append(
            {
                "step": step_idx,
                "ee_position_before": ee_before.tolist(),
                "object_position_before": object_before.tolist(),
                "ee_position_after": ee_after.tolist(),
                "object_position_after": object_after.tolist(),
                "relative_ee_object": (ee_after - object_after).tolist(),
                "policy_delta_xz": policy_xz.tolist(),
                "policy_delta_norm": float(policy_xz.norm().item()),
                "expert_counterfactual_delta_xz": expert_xz.tolist(),
                "expert_counterfactual_norm": float(expert_xz.norm().item()),
                "policy_vs_expert_cosine": _cosine(policy_xz, expert_xz),
                "gripper": float(action.gripper.detach().cpu().item()),
                "expert_counterfactual_gripper": float(expert_gripper),
                "expert_counterfactual_mode": expert_mode,
                "left_contact": grasp.left_contact,
                "right_contact": grasp.right_contact,
                "object_grasped": grasp.object_grasped,
                "contact_force": grasp.contact_force,
                "object_to_ee": object_to_ee,
                "object_height_delta": object_height_delta,
                "hidden_norm": hidden_norm,
                "actual_ee_motion": (ee_after - ee_before).tolist(),
                "ik_success": bool(info["ik_success"]),
            }
        )
        if lift_success:
            break

    env.close()
    diagnostic = {
        "checkpoint_path": checkpoint_path,
        "model_kind": checkpoint["model_kind"],
        "seed": seed,
        "object_x": sampled_object_x,
        "lift_success": lift_success,
        "steps": len(records),
        "near_window": [row for row in records if float(row["object_to_ee"]) <= 0.12],
        "records": records,
    }
    path = output_dir / "gru_failure_diagnostic.json"
    path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    _plot_failure_diagnostic(records, output_dir / "gru_failure_action_trace.png")
    return {"output_path": str(path), "lift_success": lift_success, "steps": len(records)}


def _expert_rows(
    dataset: dict[str, Any],
    split: str,
    max_step: float,
    saturation_fraction: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_id in dataset["split"][split]:
        episode = dataset["episodes"][episode_id]
        for graph, delta, gripper, phase in zip(
            episode["graphs"],
            episode["delta_xz"],
            episode["gripper"],
            episode["phase_labels_diagnostics_only"],
            strict=True,
        ):
            distance = _graph_ee_object_distance(graph)
            mag = float(delta.norm().item())
            rows.append(
                {
                    "model": "Expert",
                    "phase": str(phase),
                    "distance_bin": _distance_bin(distance),
                    "distance": distance,
                    "expert_magnitude": mag,
                    "predicted_magnitude": mag,
                    "delta_x": float(delta[0].item()),
                    "delta_z": float(delta[1].item()),
                    "cosine": 1.0 if mag > 1e-8 else 0.0,
                    "component_error_x": 0.0,
                    "component_error_z": 0.0,
                    "l2_error": 0.0,
                    "saturated": mag >= saturation_fraction * max_step,
                    "gripper": float(gripper.item()),
                    "object_height": float(graph.node_features[graph.target_node_index, 2].item()),
                    "contact": _graph_contact(graph),
                }
            )
    return rows


@torch.no_grad()
def _policy_rows(
    model: nn.Module,
    model_kind: PickModelKind,
    label: str,
    dataset: dict[str, Any],
    split: str,
    device: torch.device,
    max_step: float,
    saturation_fraction: float,
    step_reset: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for episode_id in dataset["split"][split]:
        episode = dataset["episodes"][episode_id]
        hidden = model.initial_hidden(device) if isinstance(model, PickRecurrentController) else None
        for graph, target_delta, phase in zip(
            episode["graphs"],
            episode["delta_xz"],
            episode["phase_labels_diagnostics_only"],
            strict=True,
        ):
            graph = graph.to(device)
            if isinstance(model, PickRecurrentController):
                if step_reset:
                    hidden = model.initial_hidden(device)
                action, hidden = model(graph, hidden)
            elif isinstance(model, PickFeedForwardController):
                action = model(graph)
            else:
                raise ValueError(f"Unsupported model kind {model_kind}")
            pred_xz = torch.stack([action.delta_ee[0], action.delta_ee[2]]).detach().cpu()
            target_delta = target_delta.detach().cpu()
            distance = _graph_ee_object_distance(graph)
            rows.append(
                {
                    "model": label,
                    "phase": str(phase),
                    "distance_bin": _distance_bin(distance),
                    "distance": distance,
                    "expert_magnitude": float(target_delta.norm().item()),
                    "predicted_magnitude": float(pred_xz.norm().item()),
                    "delta_x": float(pred_xz[0].item()),
                    "delta_z": float(pred_xz[1].item()),
                    "cosine": _cosine(pred_xz, target_delta),
                    "component_error_x": float((pred_xz[0] - target_delta[0]).item()),
                    "component_error_z": float((pred_xz[1] - target_delta[1]).item()),
                    "l2_error": float((pred_xz - target_delta).norm().item()),
                    "saturated": float(pred_xz.norm().item()) >= saturation_fraction * max_step,
                    "object_height": float(graph.node_features[graph.target_node_index, 2].detach().cpu().item()),
                    "contact": _graph_contact(graph),
                }
            )
    return rows


def _summarize_rows(
    rows: list[dict[str, Any]],
    group_key: str,
    model_key: str | None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row[group_key]) if model_key is None else f"{row[model_key]}::{row[group_key]}"
        grouped.setdefault(key, []).append(row)
    return {key: _row_stats(values) for key, values in sorted(grouped.items())}


def _row_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    expert_mag = [float(row["expert_magnitude"]) for row in rows]
    predicted_mag = [float(row["predicted_magnitude"]) for row in rows]
    return {
        "count": float(len(rows)),
        "expert_mag_mean": _mean(expert_mag),
        "expert_mag_median": _percentile(expert_mag, 50),
        "expert_mag_p90": _percentile(expert_mag, 90),
        "expert_mag_p95": _percentile(expert_mag, 95),
        "pred_mag_mean": _mean(predicted_mag),
        "pred_mag_median": _percentile(predicted_mag, 50),
        "pred_mag_p90": _percentile(predicted_mag, 90),
        "pred_mag_p95": _percentile(predicted_mag, 95),
        "cosine_mean": _mean([float(row["cosine"]) for row in rows]),
        "l2_error_mean": _mean([float(row["l2_error"]) for row in rows]),
        "component_error_x_mean": _mean([float(row["component_error_x"]) for row in rows]),
        "component_error_z_mean": _mean([float(row["component_error_z"]) for row in rows]),
        "saturation_ratio": _mean([1.0 if row["saturated"] else 0.0 for row in rows]),
    }


def _gripper_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(row["gripper"]) for row in rows if "gripper" in row]
    open_fraction = _mean([1.0 if value < 0.5 else 0.0 for value in values])
    closed_fraction = _mean([1.0 if value >= 0.5 else 0.0 for value in values])
    return {"open_fraction": open_fraction, "closed_fraction": closed_fraction, "count": float(len(values))}


def _graph_ee_object_distance(graph: GraphData) -> float:
    features = graph.node_features.detach().cpu()
    ee = features[graph.ee_node_index, 0:3]
    obj = features[graph.target_node_index, 0:3]
    return float((obj - ee).norm().item())


def _graph_contact(graph: GraphData) -> float:
    features = graph.node_features.detach().cpu()
    return float(features[graph.target_node_index, -1].item())


def _distance_bin(distance: float) -> str:
    for name, low, high in DISTANCE_BINS:
        if distance > low and distance <= high:
            return name
    return "contact_alignment" if distance <= 0.0 else "far"


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item() * b.norm().item())
    if denom < 1e-8:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / max(len(finite), 1)


def _percentile(values: list[float], percentile: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    return float(np.percentile(np.asarray(finite, dtype=np.float64), percentile))


def _plot_expert_histogram(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist([row["expert_magnitude"] for row in rows], bins=30, color="#4C78A8", alpha=0.85)
    ax.axvline(0.045, color="black", linestyle="--", linewidth=1, label="max step")
    ax.set_xlabel("expert |Delta EE|")
    ax.set_ylabel("count")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_magnitude_vs_distance(
    expert_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    rng = np.random.default_rng(0)
    for label, rows in [("Expert", expert_rows), *[(name, [r for r in policy_rows if r["model"] == name]) for name in sorted({r["model"] for r in policy_rows})]]:
        if not rows:
            continue
        indices = np.arange(len(rows))
        if len(indices) > 2500:
            indices = rng.choice(indices, size=2500, replace=False)
        distances = [rows[int(idx)]["distance"] for idx in indices]
        mags = [rows[int(idx)]["predicted_magnitude"] for idx in indices]
        ax.scatter(distances, mags, s=8, alpha=0.35, label=label)
    ax.set_xlabel("EE-object distance")
    ax.set_ylabel("|Delta EE|")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_saturation(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = _summarize_rows(rows, group_key="distance_bin", model_key="model")
    labels = list(grouped)
    values = [grouped[label]["saturation_ratio"] for label in labels]
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(labels)), 4), constrained_layout=True)
    ax.bar(np.arange(len(labels)), values, color="#F58518")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("saturation ratio")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_failure_diagnostic(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.asarray([row["step"] for row in records], dtype=float)
    policy_mag = np.asarray([row["policy_delta_norm"] for row in records], dtype=float)
    expert_mag = np.asarray([row["expert_counterfactual_norm"] for row in records], dtype=float)
    distance = np.asarray([row["object_to_ee"] for row in records], dtype=float)
    grip = np.asarray([row["gripper"] for row in records], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    axes[0].plot(xs, policy_mag, label="policy")
    axes[0].plot(xs, expert_mag, label="expert counterfactual")
    axes[0].set_ylabel("|Delta EE|")
    axes[0].legend(loc="best")
    axes[1].plot(xs, distance)
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("EE-object dist")
    axes[2].plot(xs, grip)
    axes[2].set_ylabel("gripper")
    axes[2].set_xlabel("step")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)
