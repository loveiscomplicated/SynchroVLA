from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from vla_gnn_recurrent.env.reaching_env import ReachingEnv, ReachingEnvConfig
from vla_gnn_recurrent.env.stale_observation import StaleTargetObserver
from vla_gnn_recurrent.graph.graph_builder import GraphBuilder
from vla_gnn_recurrent.models.actions import ActionPrior
from vla_gnn_recurrent.models.recurrent_controller import RecurrentController
from vla_gnn_recurrent.models.state_corrector import StateCorrector, summarize_gate
from vla_gnn_recurrent.training.trainer import ModelKind, TaskKind, build_controller
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed

RecurrentMode = Literal["normal", "step_reset", "event_reset", "fixed_decay", "learned_correction"]
PerturbationLevel = Literal["default", "small", "medium", "large"]

PERTURBATION_RANGES: dict[PerturbationLevel, tuple[float, float]] = {
    "default": (0.55, 1.05),
    "small": (0.25, 0.45),
    "medium": (0.55, 0.85),
    "large": (0.95, 1.25),
}


@dataclass
class EvalConfig:
    model_kind: ModelKind
    checkpoint_path: str
    episodes: int = 8
    max_steps: int = 36
    seed: int = 101
    device: DevicePreference = "auto"
    perturb: bool = False
    perturb_step: int = 10
    output_dir: str = "artifacts/eval"
    task: TaskKind = "static"
    observation_interval: int = 1
    target_velocity_feature: bool = False
    target_velocity_min: float = 0.006
    target_velocity_max: float = 0.018
    recurrent_mode: RecurrentMode = "normal"
    recovery_threshold: float | None = None
    correction_checkpoint_path: str | None = None
    fixed_decay_alpha: float = 0.5
    perturbation_level: PerturbationLevel = "default"


def load_controller(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_kind = checkpoint["model_kind"]
    env_config = checkpoint.get("env_config", {})
    train_config = checkpoint.get("config", {})
    max_step = float(env_config.get("max_action", 0.08))
    action_prior: ActionPrior = train_config.get("action_prior", "geometric")
    model = build_controller(model_kind, max_step=max_step, action_prior=action_prior).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def load_state_corrector(checkpoint_path: str | Path, device: torch.device) -> tuple[StateCorrector, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    corrector_config = checkpoint.get("corrector_config", {})
    corrector = StateCorrector(
        hidden_dim=int(corrector_config.get("hidden_dim", 256)),
        graph_dim=int(corrector_config.get("graph_dim", 128)),
        num_layers=int(corrector_config.get("num_layers", 2)),
        mlp_hidden_dim=int(corrector_config.get("mlp_hidden_dim", 128)),
    ).to(device)
    corrector.load_state_dict(checkpoint["corrector_state"])
    corrector.eval()
    return corrector, checkpoint


def evaluate_controller(config: EvalConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = select_device(config.device)
    model, _ = load_controller(config.checkpoint_path, device)
    corrector: StateCorrector | None = None
    if config.recurrent_mode == "learned_correction":
        if config.correction_checkpoint_path is None:
            raise ValueError("learned_correction mode requires correction_checkpoint_path.")
        corrector, _ = load_state_corrector(config.correction_checkpoint_path, device)
    env_config = ReachingEnvConfig(
        max_steps=config.max_steps,
        target_velocity_min=config.target_velocity_min,
        target_velocity_max=config.target_velocity_max,
    )
    env = ReachingEnv(config=env_config, seed=config.seed)
    graph_builder = GraphBuilder(include_joints=False, target_velocity_feature=config.target_velocity_feature)

    results = []
    for episode_idx in range(config.episodes):
        distance_range = (1.05, 1.35) if config.perturb else None
        result = run_episode(
            model=model,
            model_kind=config.model_kind,
            env=env,
            graph_builder=graph_builder,
            device=device,
            perturb_step=config.perturb_step if config.perturb else None,
            distance_range=distance_range,
            task=config.task,
            observation_interval=config.observation_interval,
            target_velocity_feature=config.target_velocity_feature,
            recurrent_mode=config.recurrent_mode,
            recovery_threshold=config.recovery_threshold or env_config.target_radius,
            corrector=corrector,
            fixed_decay_alpha=config.fixed_decay_alpha,
            perturbation_level=config.perturbation_level,
        )
        result["episode"] = episode_idx + 1
        results.append(result)

    summary = _summarize(results)
    payload = {
        "config": asdict(config),
        "device": str(device),
        "summary": summary,
        "episodes": results,
    }
    output_dir = ensure_dir(config.output_dir)
    mode_name = _mode_name(config)
    output_path = output_dir / f"{config.model_kind}_{mode_name}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["output_path"] = str(output_path)
    return payload


@torch.no_grad()
def run_episode(
    model: nn.Module,
    model_kind: ModelKind,
    env: ReachingEnv,
    graph_builder: GraphBuilder,
    device: torch.device,
    perturb_step: int | None = None,
    distance_range: tuple[float, float] | None = None,
    task: TaskKind = "static",
    observation_interval: int = 1,
    target_velocity_feature: bool = False,
    recurrent_mode: RecurrentMode = "normal",
    recovery_threshold: float | None = None,
    corrector: StateCorrector | None = None,
    fixed_decay_alpha: float = 0.5,
    perturbation_level: PerturbationLevel = "default",
) -> dict[str, Any]:
    if recurrent_mode == "learned_correction" and corrector is None:
        raise ValueError("learned_correction mode requires a StateCorrector.")
    if recurrent_mode == "fixed_decay" and not 0.0 <= fixed_decay_alpha <= 1.0:
        raise ValueError("fixed_decay_alpha must be in [0, 1].")

    if task == "moving":
        observation = env.reset_moving_target(distance_range=distance_range or (0.75, 1.15))
    else:
        observation = env.reset(distance_range=distance_range)
    observer = StaleTargetObserver(
        observation_interval=observation_interval,
        target_velocity_feature=target_velocity_feature,
    )
    original_target = observation.target.position.clone()
    perturbed_target: torch.Tensor | None = None
    hidden = model.initial_hidden(device) if isinstance(model, RecurrentController) else None
    recovery_threshold = recovery_threshold or env.config.target_radius

    trajectory = [observation.ee_position.tolist()]
    true_targets = [observation.target.position.tolist()]
    observed_targets: list[list[float]] = []
    reanchor_steps: list[int] = []
    distances = [env.distance_to_target()]
    actions: list[list[float]] = []
    action_target_cosines: list[float] = []
    hidden_diagnostics: list[dict[str, float | int | bool]] = []
    correction_events: list[dict[str, Any]] = []
    distance_before_perturb: float | None = None
    distance_after_perturb: float | None = None
    recovery_distance_after: float | None = None
    recovery_time: int | None = None
    recovery_time_observable: int | None = None
    perturb_true_step: int | None = None
    observable_event_step: int | None = None
    pending_observable_event = False
    max_error_after_perturb = 0.0
    success_after_perturb = False
    done = False

    for step_idx in range(env.config.max_steps):
        if perturb_step is not None and step_idx == perturb_step:
            distance_before_perturb = env.distance_to_target()
            min_distance, max_distance = PERTURBATION_RANGES[perturbation_level]
            perturbed_target = env.sample_perturbed_target(min_distance=min_distance, max_distance=max_distance)
            env.move_target(perturbed_target)
            distance_after_perturb = env.distance_to_target()
            max_error_after_perturb = distance_after_perturb
            perturb_true_step = env.step_count
            pending_observable_event = True

        observation = observer.observe(env.observe())
        observed_targets.append(observation.observed_target.position.tolist())
        if observation.reanchored:
            reanchor_steps.append(observation.step_count)
        event_is_observable = bool(
            pending_observable_event
            and observation.reanchored
            and perturb_true_step is not None
            and observation.step_count >= perturb_true_step
        )
        if event_is_observable and observable_event_step is None:
            observable_event_step = observation.step_count
        graph = graph_builder.build(
            ee_position=observation.ee_position,
            ee_velocity=observation.ee_velocity,
            target=observation.observed_target,
        ).to(device)

        if model_kind == "recurrent":
            reset_applied = False
            correction_applied = False
            correction_kind = "none"
            if recurrent_mode == "step_reset":
                hidden = model.initial_hidden(device)  # type: ignore[union-attr]
                reset_applied = True
            elif recurrent_mode == "event_reset" and event_is_observable:
                hidden = model.initial_hidden(device)  # type: ignore[union-attr]
                reset_applied = True
            elif recurrent_mode == "fixed_decay" and event_is_observable:
                hidden = apply_fixed_decay(hidden, fixed_decay_alpha)
                correction_applied = True
                correction_kind = f"fixed_decay_{fixed_decay_alpha:g}"
            elif recurrent_mode == "learned_correction" and event_is_observable:
                pre_correction_hidden = hidden.clone()
                graph_embedding = model.gnn(graph)  # type: ignore[union-attr]
                hidden, gate = corrector(hidden, graph_embedding)  # type: ignore[arg-type]
                gate_summary = summarize_gate(gate)
                correction_applied = True
                correction_kind = "learned_correction"
                correction_events.append(
                    {
                        "step": int(observation.step_count),
                        "correction_kind": correction_kind,
                        "hidden_pre_correction_norm": float(pre_correction_hidden.detach().norm().cpu().item()),
                        "hidden_post_correction_norm": float(hidden.detach().norm().cpu().item()),
                        "hidden_correction_delta_norm": float(
                            (hidden.detach() - pre_correction_hidden.detach()).norm().cpu().item()
                        ),
                        "gate_mean": gate_summary.gate_mean,
                        "gate_std": gate_summary.gate_std,
                        "per_layer_gate_mean": gate_summary.per_layer_gate_mean,
                        "per_layer_gate_std": gate_summary.per_layer_gate_std,
                        "perturbation_level": perturbation_level,
                        "distance_after_perturb": distance_after_perturb,
                    }
                )
            hidden_input = hidden.clone() if hidden is not None else None
            action, hidden = model(graph, hidden)  # type: ignore[misc]
            _append_hidden_diagnostic(
                hidden_diagnostics,
                step=observation.step_count,
                hidden_input=hidden_input,
                hidden_output=hidden,
                reset_applied=reset_applied,
                correction_applied=correction_applied,
                correction_kind=correction_kind,
                event_observable=event_is_observable,
            )
        else:
            action = model(graph)  # type: ignore[misc]
        if event_is_observable:
            pending_observable_event = False

        direction_to_true_target = env.target_position - observation.ee_position
        action_target_cosines.append(_cosine(action.detach().cpu(), direction_to_true_target.detach().cpu()))
        observation, _, done, info = env.step(action.detach())
        actions.append(action.detach().cpu().tolist())
        trajectory.append(observation.ee_position.tolist())
        true_targets.append(observation.target.position.tolist())
        distances.append(float(info["distance"]))

        if perturb_true_step is not None:
            max_error_after_perturb = max(max_error_after_perturb, float(info["distance"]))
            success_after_perturb = success_after_perturb or float(info["distance"]) <= recovery_threshold
        if perturb_step is not None and step_idx == perturb_step + 5:
            recovery_distance_after = env.distance_to_target()
        if perturb_true_step is not None and step_idx >= perturb_step and recovery_time is None:
            if float(info["distance"]) <= recovery_threshold:
                recovery_time = step_idx - perturb_step + 1
        if observable_event_step is not None and recovery_time_observable is None:
            if float(info["distance"]) <= recovery_threshold:
                recovery_time_observable = env.step_count - observable_event_step
        if done:
            break

    if recovery_distance_after is None and perturb_step is not None:
        recovery_distance_after = env.distance_to_target()

    final_distance = env.distance_to_target()
    return {
        "success": final_distance <= env.config.target_radius,
        "final_distance": final_distance,
        "steps": len(actions),
        "trajectory": trajectory,
        "targets": true_targets,
        "true_targets": true_targets,
        "observed_targets": observed_targets,
        "reanchor_steps": reanchor_steps,
        "actions": actions,
        "distances": distances,
        "tracking_error": _mean(distances),
        "action_smoothness": _action_smoothness(actions),
        "action_target_cosines": action_target_cosines,
        "hidden_diagnostics": hidden_diagnostics,
        "correction_events": correction_events,
        "original_target": original_target.tolist(),
        "perturbed_target": None if perturbed_target is None else perturbed_target.tolist(),
        "perturb_step": perturb_step,
        "perturb_true_step": perturb_true_step,
        "observable_event_step": observable_event_step,
        "distance_before_perturb": distance_before_perturb,
        "distance_after_perturb": distance_after_perturb,
        "max_error_after_perturb": max_error_after_perturb,
        "recovery_distance_after": recovery_distance_after,
        "recovery_time": recovery_time,
        "recovery_time_observable": recovery_time_observable,
        "recovery_threshold": recovery_threshold,
        "success_after_perturb": success_after_perturb,
        "redirection_cosine_1": _redirection_cosine(action_target_cosines, observable_event_step, perturb_true_step, horizon=1),
        "redirection_cosine_2": _redirection_cosine(action_target_cosines, observable_event_step, perturb_true_step, horizon=2),
        "redirection_cosine_4": _redirection_cosine(action_target_cosines, observable_event_step, perturb_true_step, horizon=4),
        "observation_interval": observation_interval,
        "target_velocity_feature": target_velocity_feature,
        "recurrent_mode": recurrent_mode,
        "fixed_decay_alpha": fixed_decay_alpha,
        "perturbation_level": perturbation_level,
        "task": task,
        "done": done,
    }


def apply_fixed_decay(hidden: torch.Tensor | None, alpha: float) -> torch.Tensor | None:
    if hidden is None:
        return None
    return hidden * float(alpha)


def _summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    count = max(len(results), 1)
    successes = sum(1.0 for result in results if result["success"])
    return {
        "success_rate": successes / count,
        "mean_final_distance": sum(float(result["final_distance"]) for result in results) / count,
        "std_final_distance": _std([float(result["final_distance"]) for result in results]),
        "mean_tracking_error": sum(float(result["tracking_error"]) for result in results) / count,
        "std_tracking_error": _std([float(result["tracking_error"]) for result in results]),
        "mean_steps": sum(float(result["steps"]) for result in results) / count,
        "mean_action_smoothness": sum(float(result["action_smoothness"]) for result in results) / count,
        "mean_recovery_delta": _mean_recovery_delta(results),
        "mean_recovery_time": _mean_recovery_time(results),
        "mean_recovery_time_observable": _mean_optional(results, "recovery_time_observable"),
        "mean_distance_before_perturb": _mean_optional(results, "distance_before_perturb"),
        "mean_distance_after_perturb": _mean_optional(results, "distance_after_perturb"),
        "mean_max_error_after_perturb": _mean_optional(results, "max_error_after_perturb"),
        "post_perturb_success_rate": sum(1.0 for result in results if result.get("success_after_perturb")) / count,
        "mean_redirection_cosine_1": _mean_optional(results, "redirection_cosine_1"),
        "mean_redirection_cosine_2": _mean_optional(results, "redirection_cosine_2"),
        "mean_redirection_cosine_4": _mean_optional(results, "redirection_cosine_4"),
    }


def _mean_recovery_delta(results: list[dict[str, Any]]) -> float:
    deltas = []
    for result in results:
        before = result.get("distance_after_perturb")
        after = result.get("recovery_distance_after")
        if before is not None and after is not None:
            deltas.append(float(before) - float(after))
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.std(unbiased=True).item())


def _mean_optional(results: list[dict[str, Any]], key: str) -> float:
    values = [float(result[key]) for result in results if result.get(key) is not None]
    return _mean(values)


def _action_smoothness(actions: list[list[float]]) -> float:
    if len(actions) < 2:
        return 0.0
    action_tensor = torch.tensor(actions, dtype=torch.float32)
    diffs = action_tensor[1:] - action_tensor[:-1]
    return float(diffs.norm(dim=-1).mean().item())


def _cosine(action: torch.Tensor, direction: torch.Tensor) -> float:
    if float(action.norm()) < 1e-8 or float(direction.norm()) < 1e-8:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(action, direction, dim=0).item())


def _redirection_cosine(
    cosines: list[float],
    observable_event_step: int | None,
    perturb_true_step: int | None,
    horizon: int,
) -> float | None:
    event_step = observable_event_step if observable_event_step is not None else perturb_true_step
    if event_step is None:
        return None
    start = max(int(event_step), 0)
    end = min(start + horizon, len(cosines))
    if start >= end:
        return None
    return max(cosines[start:end])


def _append_hidden_diagnostic(
    diagnostics: list[dict[str, float | int | bool | str]],
    step: int,
    hidden_input: torch.Tensor | None,
    hidden_output: torch.Tensor | None,
    reset_applied: bool,
    correction_applied: bool,
    correction_kind: str,
    event_observable: bool,
) -> None:
    input_norm = 0.0 if hidden_input is None else float(hidden_input.detach().norm().cpu().item())
    output_norm = 0.0 if hidden_output is None else float(hidden_output.detach().norm().cpu().item())
    if hidden_input is None or hidden_output is None:
        delta_norm = 0.0
    else:
        delta_norm = float((hidden_output.detach() - hidden_input.detach()).norm().cpu().item())
    diagnostics.append(
        {
            "step": int(step),
            "hidden_input_norm": input_norm,
            "hidden_output_norm": output_norm,
            "hidden_delta_norm": delta_norm,
            "reset_applied": bool(reset_applied),
            "correction_applied": bool(correction_applied),
            "correction_kind": correction_kind,
            "event_observable": bool(event_observable),
        }
    )


def _mean_recovery_time(results: list[dict[str, Any]]) -> float:
    times = [float(result["recovery_time"]) for result in results if result.get("recovery_time") is not None]
    if not times:
        return 0.0
    return sum(times) / len(times)


def _mode_name(config: EvalConfig) -> str:
    pieces = [config.task, f"N{config.observation_interval}"]
    if config.model_kind == "recurrent":
        pieces.append(config.recurrent_mode)
        if config.recurrent_mode == "fixed_decay":
            pieces.append(f"alpha{config.fixed_decay_alpha:g}")
    if config.perturb:
        if config.perturbation_level != "default":
            pieces.append(config.perturbation_level)
        pieces.append("perturb")
    return "_".join(pieces)
