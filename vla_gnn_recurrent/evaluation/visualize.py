from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(result: dict[str, Any], save_path: str | Path, title: str) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = np.asarray(result["trajectory"], dtype=float)
    distances = np.asarray(result["distances"], dtype=float)
    true_targets = np.asarray(result.get("true_targets", result.get("targets", [])), dtype=float)
    observed_targets = np.asarray(result.get("observed_targets", []), dtype=float)
    original_target = np.asarray(result["original_target"], dtype=float)
    perturbed = result.get("perturbed_target")
    perturbed_target = None if perturbed is None else np.asarray(perturbed, dtype=float)
    perturb_step = result.get("perturb_step")

    fig = plt.figure(figsize=(11, 5), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_dist = fig.add_subplot(1, 2, 2)

    ax3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="#2563eb", linewidth=2.2, label="EE trajectory")
    if true_targets.ndim == 2 and len(true_targets) > 1:
        ax3d.plot(
            true_targets[:, 0],
            true_targets[:, 1],
            true_targets[:, 2],
            color="#dc2626",
            linewidth=1.8,
            alpha=0.8,
            label="true target",
        )
    if observed_targets.ndim == 2 and len(observed_targets) > 0:
        reanchor_steps = result.get("reanchor_steps", [])
        reanchor_points = []
        for step in reanchor_steps:
            if step < len(observed_targets):
                reanchor_points.append(observed_targets[step])
        if reanchor_points:
            reanchors = np.asarray(reanchor_points, dtype=float)
            ax3d.scatter(
                reanchors[:, 0],
                reanchors[:, 1],
                reanchors[:, 2],
                color="#f97316",
                s=38,
                label="re-anchor",
            )
        ax3d.scatter(
            observed_targets[:, 0],
            observed_targets[:, 1],
            observed_targets[:, 2],
            color="#9333ea",
            s=16,
            alpha=0.35,
            label="observed target",
        )
    ax3d.scatter(*trajectory[0], color="#16a34a", s=55, label="start")
    ax3d.scatter(*trajectory[-1], color="#1d4ed8", s=45, label="final EE")
    ax3d.scatter(*original_target, color="#dc2626", marker="x", s=90, label="target A")

    if perturbed_target is not None:
        ax3d.scatter(*perturbed_target, color="#9333ea", marker="^", s=80, label="target B")
    if perturb_step is not None and perturb_step < len(trajectory):
        point = trajectory[int(perturb_step)]
        ax3d.scatter(*point, color="#f97316", s=70, label="perturb step")

    ax3d.set_title(title)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.legend(loc="best")
    _set_equal_axes(ax3d, trajectory, original_target, perturbed_target, true_targets, observed_targets)

    ax_dist.plot(np.arange(len(distances)), distances, color="#0f172a", linewidth=2)
    if perturb_step is not None:
        ax_dist.axvline(int(perturb_step), color="#f97316", linestyle="--", label="perturb")
        ax_dist.legend(loc="best")
    ax_dist.set_title("Distance to current target")
    ax_dist.set_xlabel("timestep")
    ax_dist.set_ylabel("distance")
    ax_dist.grid(True, alpha=0.25)

    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_error_comparison(
    feedforward_result: dict[str, Any],
    recurrent_result: dict[str, Any],
    save_path: str | Path,
    title: str,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ff_distances = np.asarray(feedforward_result["distances"], dtype=float)
    gru_distances = np.asarray(recurrent_result["distances"], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(np.arange(len(ff_distances)), ff_distances, color="#2563eb", linewidth=2, label="Feed-forward")
    ax.plot(np.arange(len(gru_distances)), gru_distances, color="#dc2626", linewidth=2, label="GRU")
    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel("||true target - EE||")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_stale_aggregate(
    summaries: list[dict[str, Any]],
    save_path: str | Path,
    metric: str = "mean_tracking_error",
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[int, float]]] = {}
    for item in summaries:
        model = item["model_kind"]
        interval = int(item["observation_interval"])
        value = float(item["summary"][metric])
        grouped.setdefault(model, []).append((interval, value))

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    colors = {"feedforward": "#2563eb", "recurrent": "#dc2626"}
    labels = {"feedforward": "Feed-forward", "recurrent": "GRU"}
    for model, values in grouped.items():
        values = sorted(values, key=lambda item: item[0])
        xs = [item[0] for item in values]
        ys = [item[1] for item in values]
        ax.plot(xs, ys, marker="o", linewidth=2, color=colors.get(model), label=labels.get(model, model))

    ax.set_title(metric.replace("_", " "))
    ax.set_xlabel("observation interval N")
    ax.set_ylabel(metric)
    ax.set_xticks(sorted({int(item["observation_interval"]) for item in summaries}))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_condition_aggregate(
    summaries: list[dict[str, Any]],
    save_path: str | Path,
    metric: str,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[int, float]]] = {}
    for item in summaries:
        condition = item["condition"]
        interval = int(item["observation_interval"])
        value = float(item["summary"][metric])
        grouped.setdefault(condition, []).append((interval, value))

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for condition, values in grouped.items():
        values = sorted(values, key=lambda item: item[0])
        xs = [item[0] for item in values]
        ys = [item[1] for item in values]
        ax.plot(xs, ys, marker="o", linewidth=2, label=_condition_label(condition))
    ax.set_title(metric.replace("_", " "))
    ax.set_xlabel("observation interval N")
    ax.set_ylabel(metric)
    ax.set_xticks(sorted({int(item["observation_interval"]) for item in summaries}))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_perturbation_recovery(
    condition_results: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    window_before: int = 5,
    window_after: int = 20,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(-window_before, window_after + 1)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for condition, episodes in condition_results.items():
        aligned = _aligned_series(episodes, key="distances", window_before=window_before, window_after=window_after)
        if aligned.size == 0:
            continue
        mean = np.nanmean(aligned, axis=0)
        ax.plot(xs, mean, linewidth=2, label=_condition_label(condition))
    ax.axvline(0, color="#f97316", linestyle="--", linewidth=1.6, label="observable event")
    ax.set_title("Perturbation recovery")
    ax.set_xlabel("steps relative to observable event")
    ax.set_ylabel("distance to true target")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_action_redirection(
    condition_results: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
    window_before: int = 5,
    window_after: int = 10,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(-window_before, window_after + 1)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for condition, episodes in condition_results.items():
        aligned = _aligned_series(
            episodes,
            key="action_target_cosines",
            window_before=window_before,
            window_after=window_after,
        )
        if aligned.size == 0:
            continue
        mean = np.nanmean(aligned, axis=0)
        ax.plot(xs, mean, linewidth=2, label=_condition_label(condition))
    ax.axvline(0, color="#f97316", linestyle="--", linewidth=1.6, label="observable event")
    ax.set_title("Action redirection")
    ax.set_xlabel("steps relative to observable event")
    ax.set_ylabel("cos(action, new-target direction)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_perturbation_magnitude(
    summaries: list[dict[str, Any]],
    save_path: str | Path,
    metric: str = "mean_recovery_time_observable",
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    order = {"small": 0, "medium": 1, "large": 2}
    grouped: dict[str, list[tuple[str, float]]] = {}
    for item in summaries:
        condition = item["condition"]
        level = item["perturbation_level"]
        value = float(item["summary"][metric])
        grouped.setdefault(condition, []).append((level, value))

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for condition, values in grouped.items():
        values = sorted(values, key=lambda item: order.get(item[0], 99))
        xs = [item[0] for item in values]
        ys = [item[1] for item in values]
        ax.plot(xs, ys, marker="o", linewidth=2, label=_condition_label(condition))
    ax.set_title(metric.replace("_", " "))
    ax.set_xlabel("perturbation magnitude")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_gate_by_magnitude(
    condition_results: dict[str, list[dict[str, Any]]],
    save_path: str | Path,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    levels = ["small", "medium", "large"]
    means = []
    stds = []
    for level in levels:
        gates = []
        for episodes in condition_results.values():
            for episode in episodes:
                if episode.get("perturbation_level") != level:
                    continue
                for event in episode.get("correction_events", []):
                    if event.get("gate_mean") is not None:
                        gates.append(float(event["gate_mean"]))
        if gates:
            values = np.asarray(gates, dtype=float)
            means.append(float(values.mean()))
            stds.append(float(values.std()))
        else:
            means.append(np.nan)
            stds.append(0.0)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.errorbar(levels, means, yerr=stds, marker="o", linewidth=2, capsize=4)
    ax.set_title("Learned correction gate")
    ax.set_xlabel("perturbation magnitude")
    ax.set_ylabel("mean gate value")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _aligned_series(
    episodes: list[dict[str, Any]],
    key: str,
    window_before: int,
    window_after: int,
) -> np.ndarray:
    rows = []
    width = window_before + window_after + 1
    for episode in episodes:
        event_step = episode.get("observable_event_step")
        if event_step is None:
            event_step = episode.get("perturb_true_step")
        if event_step is None:
            continue
        series = np.asarray(episode.get(key, []), dtype=float)
        row = np.full(width, np.nan, dtype=float)
        for out_idx, rel_step in enumerate(range(-window_before, window_after + 1)):
            src_idx = int(event_step) + rel_step
            if 0 <= src_idx < len(series):
                row[out_idx] = series[src_idx]
        rows.append(row)
    if not rows:
        return np.empty((0, width), dtype=float)
    return np.stack(rows, axis=0)


def _condition_label(condition: str) -> str:
    labels = {
        "feedforward": "Feed-forward",
        "gru_normal": "GRU-Normal",
        "gru_step_reset": "GRU-Step-Reset",
        "gru_event_reset": "GRU-Zero-Reset",
        "gru_learned_correction": "GRU-Learned-Correction",
        "gru_fixed_decay_0.25": "GRU-Fixed-Decay 0.25",
        "gru_fixed_decay_0.5": "GRU-Fixed-Decay 0.5",
        "gru_fixed_decay_0.75": "GRU-Fixed-Decay 0.75",
    }
    return labels.get(condition, condition)


def _set_equal_axes(
    ax: Any,
    trajectory: np.ndarray,
    original: np.ndarray,
    perturbed: np.ndarray | None,
    true_targets: np.ndarray | None = None,
    observed_targets: np.ndarray | None = None,
) -> None:
    points = [trajectory, original.reshape(1, 3)]
    if perturbed is not None:
        points.append(perturbed.reshape(1, 3))
    if true_targets is not None and true_targets.ndim == 2 and len(true_targets) > 0:
        points.append(true_targets)
    if observed_targets is not None and observed_targets.ndim == 2 and len(observed_targets) > 0:
        points.append(observed_targets)
    all_points = np.concatenate(points, axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
