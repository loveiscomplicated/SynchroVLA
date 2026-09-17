from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from vla_gnn_recurrent.env.reaching_env import ReachingEnv, ReachingEnvConfig
from vla_gnn_recurrent.env.stale_observation import StaleTargetObserver
from vla_gnn_recurrent.graph.graph_builder import GraphBuilder
from vla_gnn_recurrent.models.actions import ActionPrior
from vla_gnn_recurrent.models.feedforward_controller import FeedForwardController
from vla_gnn_recurrent.models.recurrent_controller import RecurrentController
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed

ModelKind = Literal["recurrent", "feedforward"]
RolloutPolicy = Literal["teacher", "model"]
TaskKind = Literal["static", "moving"]


@dataclass
class TrainConfig:
    model_kind: ModelKind
    episodes: int = 120
    max_steps: int = 28
    learning_rate: float = 3e-4
    seed: int = 7
    device: DevicePreference = "auto"
    output_dir: str = "artifacts/checkpoints"
    log_every: int = 20
    rollout_policy: RolloutPolicy = "teacher"
    task: TaskKind = "static"
    observation_intervals: tuple[int, ...] = (1,)
    target_velocity_feature: bool = False
    action_prior: ActionPrior = "none"
    target_velocity_min: float = 0.006
    target_velocity_max: float = 0.018
    bptt_steps: int = 4
    sequence_repeats: int = 4


def build_controller(model_kind: ModelKind, max_step: float, action_prior: ActionPrior = "none") -> nn.Module:
    if model_kind == "recurrent":
        return RecurrentController(max_step=max_step, action_prior=action_prior)
    if model_kind == "feedforward":
        return FeedForwardController(max_step=max_step, action_prior=action_prior)
    raise ValueError(f"Unknown model kind: {model_kind}")


def train_controller(config: TrainConfig) -> tuple[nn.Module, dict[str, object]]:
    if config.bptt_steps < 1:
        raise ValueError("bptt_steps must be >= 1")
    if config.sequence_repeats < 1:
        raise ValueError("sequence_repeats must be >= 1")
    set_seed(config.seed)
    device = select_device(config.device)
    env_config = ReachingEnvConfig(
        max_steps=config.max_steps,
        target_velocity_min=config.target_velocity_min,
        target_velocity_max=config.target_velocity_max,
    )
    env = ReachingEnv(config=env_config, seed=config.seed)
    graph_builder = GraphBuilder(include_joints=False, target_velocity_feature=config.target_velocity_feature)
    model = build_controller(
        config.model_kind,
        max_step=env_config.max_action,
        action_prior=config.action_prior,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    sequences = []
    for episode_idx in range(config.episodes):
        observation_interval = config.observation_intervals[episode_idx % len(config.observation_intervals)]
        observer = StaleTargetObserver(
            observation_interval=observation_interval,
            target_velocity_feature=config.target_velocity_feature,
        )
        if config.task == "moving":
            env.reset_moving_target(distance_range=(0.75, 1.15))
        else:
            env.reset()
        graphs, target_actions, steps = _collect_episode_sequence(
            config=config,
            model=model,
            env=env,
            graph_builder=graph_builder,
            observer=observer,
            device=torch.device("cpu") if config.rollout_policy == "teacher" else device,
        )
        final_distance = env.distance_to_target()
        success = final_distance <= env_config.target_radius
        sequences.append(
            {
                "graphs": graphs,
                "target_actions": target_actions,
                "steps": steps,
                "final_distance": final_distance,
                "success": success,
                "observation_interval": observation_interval,
            }
        )

    history: list[dict[str, float | int | bool]] = []
    update_idx = 0
    for epoch_idx in range(config.sequence_repeats):
        for sequence_idx, sequence in enumerate(sequences):
            model.train()
            episode_losses = _train_on_sequence(
                config=config,
                model=model,
                optimizer=optimizer,
                graphs=sequence["graphs"],
                target_actions=sequence["target_actions"],
                device=device,
            )
            update_idx += 1
            record = {
                "episode": sequence_idx + 1,
                "epoch": epoch_idx + 1,
                "loss": sum(episode_losses) / max(len(episode_losses), 1),
                "final_distance": sequence["final_distance"],
                "success": bool(sequence["success"]),
                "steps": sequence["steps"],
                "observation_interval": sequence["observation_interval"],
            }
            history.append(record)
            if config.log_every > 0 and (update_idx % config.log_every == 0 or update_idx == 1):
                recent = history[-min(config.log_every, len(history)) :]
                mean_loss = sum(float(item["loss"]) for item in recent) / len(recent)
                mean_dist = sum(float(item["final_distance"]) for item in recent) / len(recent)
                success_rate = sum(1.0 for item in recent if item["success"]) / len(recent)
                print(
                    f"[{config.model_kind}] update {update_idx:04d} "
                    f"loss={mean_loss:.5f} final_dist={mean_dist:.3f} success={success_rate:.2f} "
                    f"prior={config.action_prior} task={config.task}"
                )

    checkpoint_dir = ensure_dir(config.output_dir)
    checkpoint_path = checkpoint_dir / f"{config.model_kind}.pt"
    checkpoint = {
        "model_kind": config.model_kind,
        "model_state": model.state_dict(),
        "config": asdict(config),
        "env_config": asdict(env_config),
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)
    _write_history_json(checkpoint_dir / f"{config.model_kind}_history.json", checkpoint)

    summary = {
        "model_kind": config.model_kind,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "episodes": config.episodes,
        "last_loss": history[-1]["loss"],
        "last_final_distance": history[-1]["final_distance"],
        "recent_success_rate": sum(1.0 for item in history[-20:] if item["success"]) / min(20, len(history)),
    }
    return model, summary


def _initial_hidden(model: nn.Module, device: torch.device) -> torch.Tensor | None:
    if isinstance(model, RecurrentController):
        return model.initial_hidden(device)
    return None


@torch.no_grad()
def _collect_episode_sequence(
    config: TrainConfig,
    model: nn.Module,
    env: ReachingEnv,
    graph_builder: GraphBuilder,
    observer: StaleTargetObserver,
    device: torch.device,
) -> tuple[list, list[torch.Tensor], int]:
    graphs = []
    target_actions = []
    hidden = _initial_hidden(model, device)
    steps = 0

    for _ in range(config.max_steps):
        observation = observer.observe(env.observe())
        graph = graph_builder.build(
            ee_position=observation.ee_position,
            ee_velocity=observation.ee_velocity,
            target=observation.observed_target,
        ).to(device)
        target_action = env.optimal_action(lookahead=config.task == "moving").to(device)
        graphs.append(graph)
        target_actions.append(target_action)

        if config.rollout_policy == "model":
            if config.model_kind == "recurrent":
                env_action, hidden = model(graph, hidden)  # type: ignore[misc]
            else:
                env_action = model(graph)  # type: ignore[misc]
        else:
            env_action = target_action

        _, _, done, _ = env.step(env_action.detach())
        steps += 1
        if done:
            break
    return graphs, target_actions, steps


def _train_on_sequence(
    config: TrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    graphs: list,
    target_actions: list[torch.Tensor],
    device: torch.device,
) -> list[float]:
    hidden = _initial_hidden(model, device)
    chunk_losses: list[torch.Tensor] = []
    episode_losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)

    for graph, target_action in zip(graphs, target_actions, strict=True):
        graph = graph.to(device)
        target_action = target_action.to(device)
        prediction, hidden = _training_prediction(config, model, graph, hidden)
        chunk_losses.append(_action_loss(prediction, target_action))
        if len(chunk_losses) >= config.bptt_steps:
            episode_losses.append(_apply_sequence_update(model, optimizer, chunk_losses))
            chunk_losses = []
            if hidden is not None:
                hidden = hidden.detach()

    if chunk_losses:
        episode_losses.append(_apply_sequence_update(model, optimizer, chunk_losses))
    return episode_losses


def _apply_sequence_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    losses: list[torch.Tensor],
) -> float:
    chunk_loss = torch.stack(losses).mean()
    chunk_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(chunk_loss.detach().cpu().item())


def _training_prediction(
    config: TrainConfig,
    model: nn.Module,
    graph,
    hidden: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if config.action_prior == "none":
        if isinstance(model, RecurrentController):
            return model.raw_action(graph, hidden)
        if isinstance(model, FeedForwardController):
            return model.raw_action(graph), hidden
    if isinstance(model, RecurrentController):
        action, next_hidden = model(graph, hidden)
        return action, next_hidden
    return model(graph), hidden  # type: ignore[misc]


def _action_loss(action: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(action, target_action)


def _write_history_json(path: Path, checkpoint: dict[str, object]) -> None:
    serializable = dict(checkpoint)
    serializable.pop("model_state", None)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
