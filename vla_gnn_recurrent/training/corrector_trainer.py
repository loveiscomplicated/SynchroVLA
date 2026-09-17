from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vla_gnn_recurrent.env.reaching_env import ReachingEnv, ReachingEnvConfig
from vla_gnn_recurrent.env.stale_observation import StaleTargetObserver
from vla_gnn_recurrent.evaluation.evaluate import PERTURBATION_RANGES, PerturbationLevel, load_controller
from vla_gnn_recurrent.graph.graph_builder import GraphBuilder
from vla_gnn_recurrent.models.recurrent_controller import RecurrentController
from vla_gnn_recurrent.models.state_corrector import StateCorrector, summarize_gate
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


@dataclass
class CorrectorTrainConfig:
    base_checkpoint: str = "artifacts/stale_final/checkpoints/recurrent.pt"
    episodes: int = 240
    max_steps: int = 36
    learning_rate: float = 1e-3
    seed: int = 31
    device: DevicePreference = "auto"
    output_dir: str = "artifacts/state_corrector/checkpoints"
    log_every: int = 20
    observation_intervals: tuple[int, ...] = (1, 4)
    perturbation_levels: tuple[PerturbationLevel, ...] = ("small", "medium", "large")
    horizon: int = 6
    target_velocity_min: float = 0.006
    target_velocity_max: float = 0.018
    corrector_hidden_dim: int = 128


def train_state_corrector(config: CorrectorTrainConfig) -> tuple[StateCorrector, dict[str, Any]]:
    if config.horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not config.observation_intervals:
        raise ValueError("At least one observation interval is required.")
    if not config.perturbation_levels:
        raise ValueError("At least one perturbation level is required.")

    set_seed(config.seed)
    device = select_device(config.device)
    controller, checkpoint = load_controller(config.base_checkpoint, device)
    if not isinstance(controller, RecurrentController):
        raise ValueError("State correction training requires a recurrent checkpoint.")
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config.get("action_prior", "none") != "none":
        raise ValueError("Correction-only training expects a learned-only base controller (action_prior='none').")

    _freeze_module(controller)
    controller.eval()
    corrector = StateCorrector(
        hidden_dim=controller.gru_hidden_dim,
        graph_dim=controller.gnn.hidden_dim,
        num_layers=controller.gru_layers,
        mlp_hidden_dim=config.corrector_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=config.learning_rate)

    env_config = ReachingEnvConfig(
        max_steps=config.max_steps,
        target_velocity_min=config.target_velocity_min,
        target_velocity_max=config.target_velocity_max,
    )
    env = ReachingEnv(config=env_config, seed=config.seed)
    graph_builder = GraphBuilder(include_joints=False, target_velocity_feature=False)

    history: list[dict[str, Any]] = []
    for episode_idx in range(config.episodes):
        observation_interval = config.observation_intervals[episode_idx % len(config.observation_intervals)]
        perturbation_level = config.perturbation_levels[episode_idx % len(config.perturbation_levels)]
        loss, record = _train_one_episode(
            config=config,
            controller=controller,
            corrector=corrector,
            optimizer=optimizer,
            env=env,
            graph_builder=graph_builder,
            observation_interval=observation_interval,
            perturbation_level=perturbation_level,
            device=device,
        )
        history.append(record)
        if config.log_every > 0 and ((episode_idx + 1) % config.log_every == 0 or episode_idx == 0):
            recent = history[-min(config.log_every, len(history)) :]
            trained = [item for item in recent if item["trained"]]
            mean_loss = sum(float(item["loss"]) for item in trained) / max(len(trained), 1)
            mean_gate = sum(float(item["gate_mean"]) for item in trained) / max(len(trained), 1)
            print(
                f"[corrector] episode {episode_idx + 1:04d} loss={mean_loss:.5f} "
                f"gate={mean_gate:.3f} trained={len(trained)}/{len(recent)} "
                f"N={observation_interval} level={perturbation_level}"
            )

    output_dir = ensure_dir(config.output_dir)
    checkpoint_path = output_dir / "corrector.pt"
    corrector_config = {
        "hidden_dim": controller.gru_hidden_dim,
        "graph_dim": controller.gnn.hidden_dim,
        "num_layers": controller.gru_layers,
        "mlp_hidden_dim": config.corrector_hidden_dim,
    }
    saved = {
        "corrector_state": corrector.state_dict(),
        "corrector_config": corrector_config,
        "base_checkpoint": config.base_checkpoint,
        "train_config": asdict(config),
        "parameter_count": corrector.parameter_count(),
        "history": history,
    }
    torch.save(saved, checkpoint_path)
    _write_history_json(output_dir / "corrector_history.json", saved)

    trained_records = [item for item in history if item["trained"]]
    summary = {
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "base_checkpoint": config.base_checkpoint,
        "episodes": config.episodes,
        "trained_episodes": len(trained_records),
        "parameter_count": corrector.parameter_count(),
        "last_loss": history[-1]["loss"] if history else 0.0,
        "mean_gate": sum(float(item["gate_mean"]) for item in trained_records) / max(len(trained_records), 1),
        "history_path": str(output_dir / "corrector_history.json"),
    }
    return corrector, summary


def _train_one_episode(
    config: CorrectorTrainConfig,
    controller: RecurrentController,
    corrector: StateCorrector,
    optimizer: torch.optim.Optimizer,
    env: ReachingEnv,
    graph_builder: GraphBuilder,
    observation_interval: int,
    perturbation_level: PerturbationLevel,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    env.reset_moving_target(distance_range=(1.05, 1.35))
    observer = StaleTargetObserver(observation_interval=observation_interval, target_velocity_feature=False)
    hidden = controller.initial_hidden(device)
    pending_observable_event = False
    perturb_true_step: int | None = None
    observable_event_step: int | None = None
    latest_step = max(5, min(10, config.max_steps - config.horizon - 2))
    perturb_step = random.randint(4, latest_step)
    trained = False
    loss_value = 0.0
    gate_mean = 0.0
    gate_std = 0.0
    per_layer_gate_mean: list[float] = []

    for step_idx in range(config.max_steps):
        if step_idx == perturb_step:
            min_distance, max_distance = PERTURBATION_RANGES[perturbation_level]
            env.move_target(env.sample_perturbed_target(min_distance=min_distance, max_distance=max_distance))
            perturb_true_step = env.step_count
            pending_observable_event = True

        observation = observer.observe(env.observe())
        graph = graph_builder.build(
            ee_position=observation.ee_position,
            ee_velocity=observation.ee_velocity,
            target=observation.observed_target,
        ).to(device)
        target_action = env.optimal_action(lookahead=True).to(device)
        event_is_observable = bool(
            pending_observable_event
            and observation.reanchored
            and perturb_true_step is not None
            and observation.step_count >= perturb_true_step
        )

        if event_is_observable:
            observable_event_step = observation.step_count
            loss, gate = _train_event_horizon(
                config=config,
                controller=controller,
                corrector=corrector,
                optimizer=optimizer,
                env=env,
                graph_builder=graph_builder,
                observer=observer,
                graph=graph,
                hidden=hidden,
                first_target_action=target_action,
                device=device,
            )
            gate_summary = summarize_gate(gate)
            loss_value = loss
            gate_mean = gate_summary.gate_mean
            gate_std = gate_summary.gate_std
            per_layer_gate_mean = gate_summary.per_layer_gate_mean
            trained = True
            pending_observable_event = False
            break

        with torch.no_grad():
            _, hidden = controller.raw_action(graph, hidden)
        _, _, done, _ = env.step(target_action.detach())
        if done:
            break

    record = {
        "loss": loss_value,
        "trained": trained,
        "observation_interval": observation_interval,
        "perturbation_level": perturbation_level,
        "perturb_step": perturb_step,
        "perturb_true_step": perturb_true_step,
        "observable_event_step": observable_event_step,
        "gate_mean": gate_mean,
        "gate_std": gate_std,
        "per_layer_gate_mean": per_layer_gate_mean,
    }
    return loss_value, record


def _train_event_horizon(
    config: CorrectorTrainConfig,
    controller: RecurrentController,
    corrector: StateCorrector,
    optimizer: torch.optim.Optimizer,
    env: ReachingEnv,
    graph_builder: GraphBuilder,
    observer: StaleTargetObserver,
    graph,
    hidden: torch.Tensor,
    first_target_action: torch.Tensor,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        graph_embedding = controller.gnn(graph)
    corrected_hidden, gate = corrector(hidden, graph_embedding)
    hidden = corrected_hidden
    losses: list[torch.Tensor] = []
    current_graph = graph
    target_action = first_target_action

    for horizon_idx in range(config.horizon):
        predicted_action, hidden = controller.raw_action(current_graph, hidden)
        losses.append(torch.nn.functional.mse_loss(predicted_action, target_action))
        _, _, done, _ = env.step(target_action.detach())
        if done or horizon_idx == config.horizon - 1:
            break
        observation = observer.observe(env.observe())
        current_graph = graph_builder.build(
            ee_position=observation.ee_position,
            ee_velocity=observation.ee_velocity,
            target=observation.observed_target,
        ).to(device)
        target_action = env.optimal_action(lookahead=True).to(device)

    loss = torch.stack(losses).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(corrector.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach().cpu().item()), gate.detach()


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _write_history_json(path: Path, checkpoint: dict[str, Any]) -> None:
    serializable = dict(checkpoint)
    serializable.pop("corrector_state", None)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
