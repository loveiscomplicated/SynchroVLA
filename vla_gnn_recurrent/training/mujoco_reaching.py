from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from vla_gnn_recurrent.graph.manipulation_graph_builder import ManipulationGraphBuilder
from vla_gnn_recurrent.models.feedforward_controller import FeedForwardController
from vla_gnn_recurrent.models.recurrent_controller import RecurrentController
from vla_gnn_recurrent.models.state_corrector import StateCorrector
from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv, MujocoReachConfig
from vla_gnn_recurrent.sim.stale_observation import StaleManipulationObserver
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed

MujocoPolicyKind = Literal["graph_recurrent", "graph_feedforward"]


@dataclass
class MujocoReachTrainConfig:
    model_kind: MujocoPolicyKind = "graph_recurrent"
    episodes: int = 180
    max_steps: int = 48
    learning_rate: float = 5e-4
    seed: int = 41
    device: DevicePreference = "auto"
    output_dir: str = "artifacts/mujoco_prototype/checkpoints"
    log_every: int = 20
    bptt_steps: int = 8
    sequence_repeats: int = 8
    observation_interval: int = 1
    demo_modes: tuple[str, ...] = ("static", "moving", "perturb")
    perturb_step: int = 8


def train_mujoco_reaching(config: MujocoReachTrainConfig) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(config.seed)
    device = select_device(config.device)
    env_config = MujocoReachConfig(max_steps=config.max_steps)
    env = MujocoManipulatorEnv(env_config, seed=config.seed)
    graph_builder = ManipulationGraphBuilder(task="reach", include_object=False)
    model = _build_policy(config.model_kind, env_config.max_delta_ee).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    sequences = []
    for episode_idx in range(config.episodes):
        observer = StaleManipulationObserver(observation_interval=config.observation_interval)
        demo_mode = config.demo_modes[episode_idx % len(config.demo_modes)]
        graphs, actions, final_distance, success, steps = _collect_expert_sequence(
            env=env,
            graph_builder=graph_builder,
            observer=observer,
            max_steps=config.max_steps,
            demo_mode=demo_mode,
            perturb_step=config.perturb_step,
        )
        sequences.append(
            {
                "graphs": graphs,
                "actions": actions,
                "final_distance": final_distance,
                "success": success,
                "steps": steps,
                "demo_mode": demo_mode,
            }
        )

    history: list[dict[str, Any]] = []
    update_idx = 0
    for epoch_idx in range(config.sequence_repeats):
        for sequence_idx, sequence in enumerate(sequences):
            losses = _train_sequence(
                model=model,
                model_kind=config.model_kind,
                optimizer=optimizer,
                graphs=sequence["graphs"],
                target_actions=sequence["actions"],
                device=device,
                bptt_steps=config.bptt_steps,
            )
            update_idx += 1
            record = {
                "epoch": epoch_idx + 1,
                "episode": sequence_idx + 1,
                "loss": sum(losses) / max(len(losses), 1),
                "expert_final_distance": sequence["final_distance"],
                "expert_success": bool(sequence["success"]),
                "steps": sequence["steps"],
            }
            history.append(record)
            if config.log_every > 0 and (update_idx % config.log_every == 0 or update_idx == 1):
                recent = history[-min(config.log_every, len(history)) :]
                mean_loss = sum(float(item["loss"]) for item in recent) / len(recent)
                expert_success = sum(1.0 for item in recent if item["expert_success"]) / len(recent)
                print(
                    f"[mujoco:{config.model_kind}] update {update_idx:04d} "
                    f"loss={mean_loss:.5f} expert_success={expert_success:.2f}"
                )

    output_dir = ensure_dir(config.output_dir)
    checkpoint_path = output_dir / f"{config.model_kind}.pt"
    checkpoint = {
        "model_kind": config.model_kind,
        "model_state": model.state_dict(),
        "config": asdict(config),
        "env_config": asdict(env_config),
        "model_source": str(env.xml_path),
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)
    history_path = output_dir / f"{config.model_kind}_history.json"
    serializable = dict(checkpoint)
    serializable.pop("model_state", None)
    history_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    summary = {
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "episodes": config.episodes,
        "recent_loss": sum(float(item["loss"]) for item in history[-20:]) / min(20, len(history)),
        "expert_success_rate": sum(1.0 for item in sequences if item["success"]) / len(sequences),
    }
    env.close()
    return model, summary


def load_mujoco_policy(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    env_config = checkpoint.get("env_config", {})
    model = _build_policy(checkpoint["model_kind"], float(env_config.get("max_delta_ee", 0.035))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def evaluate_mujoco_reaching(
    checkpoint_path: str | Path,
    episodes: int = 32,
    max_steps: int = 48,
    seed: int = 101,
    device_preference: DevicePreference = "auto",
    observation_interval: int = 1,
    moving_target: bool = False,
    perturb: bool = False,
    perturb_step: int = 16,
    output_dir: str | Path = "artifacts/mujoco_prototype/eval",
    render: bool = False,
    corrector_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    device = select_device(device_preference)
    model, checkpoint = load_mujoco_policy(checkpoint_path, device)
    corrector = _load_corrector(corrector_checkpoint_path, device) if corrector_checkpoint_path is not None else None
    env_config = MujocoReachConfig(max_steps=max_steps)
    env = MujocoManipulatorEnv(env_config, seed=seed)
    graph_builder = ManipulationGraphBuilder(task="reach", include_object=False)
    results = []
    for episode_idx in range(episodes):
        result = run_mujoco_reaching_episode(
            model=model,
            model_kind=checkpoint["model_kind"],
            env=env,
            graph_builder=graph_builder,
            device=device,
            observation_interval=observation_interval,
            moving_target=moving_target,
            perturb=perturb,
            perturb_step=perturb_step,
            render=render and episode_idx == 0,
            render_dir=Path(output_dir) / "frames",
            corrector=corrector,
        )
        result["episode"] = episode_idx + 1
        results.append(result)
    summary = _summarize(results)
    output_dir = ensure_dir(output_dir)
    name = _eval_name(checkpoint["model_kind"], observation_interval, moving_target, perturb)
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "model_kind": checkpoint["model_kind"],
        "device": str(device),
        "observation_interval": observation_interval,
        "moving_target": moving_target,
        "perturb": perturb,
        "state_corrector": str(corrector_checkpoint_path) if corrector_checkpoint_path is not None else None,
        "summary": summary,
        "episodes": results,
    }
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["output_path"] = str(path)
    env.close()
    return payload


@torch.no_grad()
def run_mujoco_reaching_episode(
    model: nn.Module,
    model_kind: MujocoPolicyKind,
    env: MujocoManipulatorEnv,
    graph_builder: ManipulationGraphBuilder,
    device: torch.device,
    observation_interval: int = 1,
    moving_target: bool = False,
    perturb: bool = False,
    perturb_step: int = 16,
    render: bool = False,
    render_dir: Path | None = None,
    corrector: StateCorrector | None = None,
) -> dict[str, Any]:
    env.reset_moving_target() if moving_target else env.reset()
    observer = StaleManipulationObserver(observation_interval=observation_interval)
    hidden = model.initial_hidden(device) if isinstance(model, RecurrentController) else None
    trajectory = [env.robot_observation().ee_position.tolist()]
    targets = [env.target_observation().position.tolist()]
    observed_targets: list[list[float]] = []
    distances = [env.distance_to_target()]
    actions: list[list[float]] = []
    latencies_ms: list[float] = []
    reanchor_steps: list[int] = []
    perturb_true_step: int | None = None
    observable_event_step: int | None = None
    recovery_time: int | None = None
    frames: list[str] = []
    if render:
        render_dir = ensure_dir(render_dir or Path("artifacts/mujoco_prototype/frames"))

    done = False
    for step_idx in range(max_steps := env.config.max_steps):
        if perturb and step_idx == perturb_step:
            env.move_target(env.sample_perturbed_target())
            perturb_true_step = env.step_count
            observer.notify_true_event(env.step_count)

        true_observation = env.observe()
        observation = observer.observe(true_observation)
        if observation.reanchored:
            reanchor_steps.append(observation.step_count)
        if observation.observable_event and observable_event_step is None:
            observable_event_step = observation.step_count

        graph = graph_builder.build(observation).to(device)
        start = time.perf_counter()
        if isinstance(model, RecurrentController):
            if corrector is not None and observation.observable_event and hidden is not None:
                graph_embedding = model.gnn(graph)
                hidden, _ = corrector(hidden, graph_embedding)
            action, hidden = model(graph, hidden)
        elif isinstance(model, FeedForwardController):
            action = model(graph)
        else:
            raise ValueError(f"Unsupported MuJoCo policy kind: {model_kind}")
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        observation_after, _, done, info = env.step_delta_ee(action.detach())
        actions.append(action.detach().cpu().tolist())
        trajectory.append(observation_after.robot.ee_position.tolist())
        targets.append(observation_after.true_target.position.tolist())
        observed_targets.append(observation.target.position.tolist())
        distances.append(float(info["distance"]))
        if render and render_dir is not None and step_idx % 2 == 0:
            try:
                frame = env.render_frame()
                frame_path = render_dir / f"frame_{step_idx:03d}.png"
                import matplotlib.pyplot as plt

                plt.imsave(frame_path, frame)
                frames.append(str(frame_path))
            except Exception as exc:  # pragma: no cover - depends on local OpenGL backend
                frames.append(f"render_failed:{type(exc).__name__}:{exc}")
                render = False

        if observable_event_step is not None and recovery_time is None and float(info["distance"]) <= env.config.target_radius:
            recovery_time = env.step_count - observable_event_step
        if done:
            break

    final_distance = env.distance_to_target()
    return {
        "success": final_distance <= env.config.target_radius,
        "final_distance": final_distance,
        "steps": len(actions),
        "trajectory": trajectory,
        "targets": targets,
        "observed_targets": observed_targets,
        "reanchor_steps": reanchor_steps,
        "distances": distances,
        "tracking_error": _mean(distances),
        "action_smoothness": _action_smoothness(actions),
        "mean_latency_ms": _mean(latencies_ms),
        "actions": actions,
        "perturb_true_step": perturb_true_step,
        "observable_event_step": observable_event_step,
        "recovery_time_observable": recovery_time,
        "frames": frames,
        "done": done,
        "max_steps": max_steps,
    }


@torch.no_grad()
def _collect_expert_sequence(
    env: MujocoManipulatorEnv,
    graph_builder: ManipulationGraphBuilder,
    observer: StaleManipulationObserver,
    max_steps: int,
    demo_mode: str = "static",
    perturb_step: int = 8,
) -> tuple[list, list[torch.Tensor], float, bool, int]:
    if demo_mode == "moving":
        env.reset_moving_target()
    else:
        env.reset()
    graphs = []
    actions = []
    for step_idx in range(max_steps):
        if demo_mode == "perturb" and step_idx == perturb_step:
            env.move_target(env.sample_perturbed_target())
            observer.notify_true_event(env.step_count)
        observation = observer.observe(env.observe())
        graphs.append(graph_builder.build(observation))
        action = env.expert_action()
        actions.append(action)
        _, _, done, _ = env.step_delta_ee(action)
        if done:
            break
    final_distance = env.distance_to_target()
    return graphs, actions, final_distance, final_distance <= env.config.target_radius, step_idx + 1


def _train_sequence(
    model: nn.Module,
    model_kind: MujocoPolicyKind,
    optimizer: torch.optim.Optimizer,
    graphs: list,
    target_actions: list[torch.Tensor],
    device: torch.device,
    bptt_steps: int,
) -> list[float]:
    model.train()
    hidden = model.initial_hidden(device) if isinstance(model, RecurrentController) else None
    losses: list[torch.Tensor] = []
    loss_values: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for graph, target_action in zip(graphs, target_actions, strict=True):
        graph = graph.to(device)
        target_action = target_action.to(device)
        if isinstance(model, RecurrentController):
            prediction, hidden = model.raw_action(graph, hidden)
        else:
            prediction = model.raw_action(graph)  # type: ignore[assignment]
        losses.append(torch.nn.functional.mse_loss(prediction, target_action))
        if len(losses) >= bptt_steps:
            loss_values.append(_apply_update(model, optimizer, losses))
            losses = []
            if hidden is not None:
                hidden = hidden.detach()
    if losses:
        loss_values.append(_apply_update(model, optimizer, losses))
    return loss_values


def _apply_update(model: nn.Module, optimizer: torch.optim.Optimizer, losses: list[torch.Tensor]) -> float:
    loss = torch.stack(losses).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu().item())


def _build_policy(model_kind: MujocoPolicyKind, max_step: float) -> nn.Module:
    if model_kind == "graph_recurrent":
        return RecurrentController(max_step=max_step, action_prior="none")
    if model_kind == "graph_feedforward":
        return FeedForwardController(max_step=max_step, action_prior="none")
    raise ValueError(f"Unknown MuJoCo policy kind: {model_kind}")


def _load_corrector(checkpoint_path: str | Path | None, device: torch.device) -> StateCorrector | None:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("corrector_config", {})
    corrector = StateCorrector(
        hidden_dim=int(config.get("hidden_dim", 256)),
        graph_dim=int(config.get("graph_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        mlp_hidden_dim=int(config.get("mlp_hidden_dim", 128)),
    ).to(device)
    corrector.load_state_dict(checkpoint["corrector_state"])
    corrector.eval()
    return corrector


def _summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    count = max(len(results), 1)
    return {
        "success_rate": sum(1.0 for result in results if result["success"]) / count,
        "mean_final_distance": _mean([result["final_distance"] for result in results]),
        "mean_steps": _mean([result["steps"] for result in results]),
        "mean_tracking_error": _mean([result["tracking_error"] for result in results]),
        "mean_action_smoothness": _mean([result["action_smoothness"] for result in results]),
        "mean_latency_ms": _mean([result["mean_latency_ms"] for result in results]),
        "mean_recovery_time_observable": _mean(
            [result["recovery_time_observable"] for result in results if result["recovery_time_observable"] is not None]
        ),
    }


def _action_smoothness(actions: list[list[float]]) -> float:
    if len(actions) < 2:
        return 0.0
    tensor = torch.tensor(actions, dtype=torch.float32)
    return float((tensor[1:] - tensor[:-1]).norm(dim=-1).mean().item())


def _mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _eval_name(model_kind: str, observation_interval: int, moving_target: bool, perturb: bool) -> str:
    pieces = [model_kind, f"N{observation_interval}"]
    if moving_target:
        pieces.append("moving")
    if perturb:
        pieces.append("perturb")
    return "_".join(pieces)
