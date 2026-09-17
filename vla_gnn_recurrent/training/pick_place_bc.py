from __future__ import annotations

import json
import time
from collections import Counter
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
from vla_gnn_recurrent.graph.flat_features import flat_state_from_graph, flat_state_schema_from_graph
from vla_gnn_recurrent.graph.types import GraphData
from vla_gnn_recurrent.models.pick_controller import (
    FlatPickFeedForwardController,
    FlatPickRecurrentController,
    PickFeedForwardController,
    PickRecurrentController,
)
from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv
from vla_gnn_recurrent.training.pick_bc import (
    PickModelKind,
    RecurrentEvalMode,
    _build_pick_model,
    _direction_magnitude_loss,
    _episode_split,
    _gripper_pos_weight,
    _mean,
    _mean_key,
    _mean_losses,
    _offline_metrics,
    _pick_env_config,
    _precision_weight,
    decode_pick_action,
    gripper_logit,
    load_pick_dataset,
    load_pick_policy,
)
from vla_gnn_recurrent.utils import DevicePreference, clamp_delta, ensure_dir, select_device, set_seed

OBJECT_SITE_X_OFFSET = -0.0366


@dataclass
class PickPlaceDemoConfig:
    episodes: int = 100
    seed: int = 321
    output_path: str = "artifacts/mujoco_pick_place/demos/pick_place_demos.pt"
    metadata_path: str = "artifacts/mujoco_pick_place/demos/pick_place_demos_metadata.json"
    object_x_range: tuple[float, float] = (-0.0048, -0.0001)
    destination_x_range: tuple[float, float] = (-0.060, -0.053)
    min_separation: float = 0.012
    max_steps: int = 340
    max_attempts_multiplier: int = 3
    render: bool = False


@dataclass
class PickPlaceTrainConfig:
    dataset_path: str = "artifacts/mujoco_pick_place/demos/pick_place_demos.pt"
    model_kind: PickModelKind = "graph_recurrent_dir_mag"
    output_dir: str = "artifacts/mujoco_pick_place/checkpoints"
    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    bptt_steps: int = 32
    motion_loss_weight: float = 10.0
    direction_loss_weight: float = 1.0
    magnitude_loss_weight: float = 1.0
    gripper_loss_weight: float = 1.0
    precision_weighting: bool = True
    near_distance: float = 0.05
    medium_distance: float = 0.10
    near_weight: float = 4.0
    medium_weight: float = 2.0
    seed: int = 321
    device: DevicePreference = "auto"
    initial_checkpoint_path: str | None = None
    log_every: int = 50


@dataclass
class PickPlaceEvalConfig:
    checkpoint_path: str = "artifacts/mujoco_pick_place/checkpoints/graph_recurrent_dir_mag.pt"
    episodes: int = 100
    seed: int = 992
    output_dir: str = "artifacts/mujoco_pick_place/eval"
    recurrent_mode: RecurrentEvalMode = "normal"
    device: DevicePreference = "auto"
    max_steps: int = 340
    render: bool = True
    object_x_range: tuple[float, float] = (-0.0048, -0.0001)
    destination_x_range: tuple[float, float] = (-0.060, -0.053)
    min_separation: float = 0.012


@dataclass
class PlacementMetricConfig:
    placement_threshold: float = 0.040
    alignment_threshold: float = 0.025
    high_velocity_threshold: float = 0.080


@dataclass
class PickPlaceAlignmentDiagnosticsConfig:
    checkpoint_path: str = "artifacts/mujoco_pick_place/checkpoints_gru_300_1epoch/graph_recurrent_dir_mag.pt"
    episodes: int = 100
    seed: int = 1201
    output_dir: str = "artifacts/mujoco_pick_place_alignment/diagnosis"
    device: DevicePreference = "auto"
    max_steps: int = 420
    object_x_range: tuple[float, float] = (-0.0048, -0.0001)
    destination_x_range: tuple[float, float] = (-0.060, -0.053)
    min_separation: float = 0.012


@dataclass
class PickPlaceAlignmentDaggerConfig:
    dataset_path: str = "artifacts/mujoco_pick_place/demos/pick_place_demos_300.pt"
    policy_checkpoint_path: str = "artifacts/mujoco_pick_place/checkpoints_gru_300_1epoch/graph_recurrent_dir_mag.pt"
    output_path: str = "artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger.pt"
    metadata_path: str = "artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger_metadata.json"
    episodes: int = 100
    seed: int = 1301
    device: DevicePreference = "auto"
    max_steps: int = 420
    object_x_range: tuple[float, float] = (-0.0048, -0.0001)
    destination_x_range: tuple[float, float] = (-0.060, -0.053)
    min_separation: float = 0.012
    destination_start_threshold: float = 0.120
    release_tail_steps: int = 56
    correction_repeat_count: int = 3


@dataclass
class PickPlaceExpertConfig:
    max_steps: int = 340
    grasp_offset: tuple[float, float, float] = (0.0, 0.0, -0.008)
    approach_height: float = 0.10
    lift_height: float = 0.18
    transport_height: float = 0.18
    release_hold_steps: int = 20
    close_steps: int = 45
    verify_steps: int = 10
    release_steps: int = 36
    placement_threshold: float = 0.040
    destination_neighborhood: float = 0.055
    lift_success_height: float = 0.08
    randomize_robot: bool = False


def generate_pick_place_demonstrations(config: PickPlaceDemoConfig) -> dict[str, Any]:
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    env_config = _pick_env_config(max_steps=config.max_steps)
    env = MujocoManipulatorEnv(env_config, seed=config.seed)
    graph_builder = ManipulationGraphBuilder(task="pick_and_place", include_object=True)
    expert_config = PickPlaceExpertConfig(max_steps=config.max_steps)
    episodes: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = config.episodes * config.max_attempts_multiplier

    while len(episodes) < config.episodes and attempts < max_attempts:
        attempts += 1
        object_x, destination_x = _sample_pick_place_positions(rng, config)
        episode = collect_pick_place_demo_episode(
            env=env,
            graph_builder=graph_builder,
            expert_config=expert_config,
            object_x=object_x,
            destination_x=destination_x,
            episode_id=len(episodes),
            render=config.render and len(episodes) == 0,
            render_dir=Path(config.output_path).parent / "expert_frames",
        )
        if episode["placement_success"]:
            episodes.append(episode)

    env.close()
    if len(episodes) < config.episodes:
        raise RuntimeError(f"Only collected {len(episodes)} successful pick-place demos after {attempts} attempts.")

    split = _episode_split(len(episodes), seed=config.seed)
    payload = {
        "task": "pick_and_place",
        "config": asdict(config),
        "expert_config": asdict(expert_config),
        "env_config": asdict(env_config),
        "graph_builder": {"task": "pick_and_place", "include_object": True},
        "episodes": episodes,
        "split": split,
    }
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    metadata = pick_place_demo_metadata(payload)
    metadata_path = Path(config.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"dataset_path": str(output_path), "metadata_path": str(metadata_path), "metadata": metadata}


def evaluate_scripted_pick_place(config: PickPlaceDemoConfig) -> dict[str, Any]:
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    env = MujocoManipulatorEnv(_pick_env_config(max_steps=config.max_steps), seed=config.seed)
    graph_builder = ManipulationGraphBuilder(task="pick_and_place", include_object=True)
    expert_config = PickPlaceExpertConfig(max_steps=config.max_steps)
    output_dir = ensure_dir(Path(config.output_path).parent / "expert_eval")
    results: list[dict[str, Any]] = []
    for episode_idx in range(config.episodes):
        object_x, destination_x = _sample_pick_place_positions(rng, config)
        result = run_scripted_pick_place_episode(
            env=env,
            expert_config=expert_config,
            object_x=object_x,
            destination_x=destination_x,
            graph_builder=graph_builder,
            collect_graphs=False,
            episode_id=episode_idx,
            render=config.render and episode_idx == 0,
            render_dir=output_dir / "frames",
        )
        result["episode"] = episode_idx + 1
        result.pop("graphs", None)
        result.pop("delta_xz", None)
        result.pop("gripper", None)
        results.append(result)
    env.close()
    payload = {
        "config": asdict(config),
        "expert_config": asdict(expert_config),
        "summary": summarize_pick_place_results(results),
        "episodes": results,
    }
    path = output_dir / "scripted_pick_place_eval.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_pick_place_rollout(results, output_dir / "scripted_pick_place_eval.png")
    payload["output_path"] = str(path)
    return payload


def collect_pick_place_demo_episode(
    env: MujocoManipulatorEnv,
    graph_builder: ManipulationGraphBuilder,
    expert_config: PickPlaceExpertConfig,
    object_x: float,
    destination_x: float,
    episode_id: int,
    render: bool = False,
    render_dir: Path | None = None,
) -> dict[str, Any]:
    result = run_scripted_pick_place_episode(
        env=env,
        expert_config=expert_config,
        object_x=object_x,
        destination_x=destination_x,
        graph_builder=graph_builder,
        collect_graphs=True,
        episode_id=episode_id,
        render=render,
        render_dir=render_dir,
    )
    return result


def _is_flat_pick_model(model: nn.Module) -> bool:
    return isinstance(model, (FlatPickFeedForwardController, FlatPickRecurrentController))


def _is_recurrent_pick_model(model: nn.Module) -> bool:
    return isinstance(model, (PickRecurrentController, FlatPickRecurrentController))


def _model_observation(model: nn.Module, graph: GraphData, device: torch.device) -> GraphData | torch.Tensor:
    if _is_flat_pick_model(model):
        return flat_state_from_graph(graph).to(device)
    return graph.to(device)


def _raw_pick_action(
    model: nn.Module,
    observation: GraphData | torch.Tensor,
    hidden: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(model, (PickRecurrentController, FlatPickRecurrentController)):
        raw, hidden = model.raw_action(observation, hidden)  # type: ignore[arg-type]
        return raw, hidden
    if isinstance(model, (PickFeedForwardController, FlatPickFeedForwardController)):
        return model.raw_action(observation), hidden  # type: ignore[arg-type]
    raise ValueError(f"Unsupported model type: {type(model)}")


def _decoded_pick_action(
    model: nn.Module,
    observation: GraphData | torch.Tensor,
    hidden: torch.Tensor | None,
) -> tuple[Any, torch.Tensor | None]:
    if isinstance(model, (PickRecurrentController, FlatPickRecurrentController)):
        action, hidden = model(observation, hidden)  # type: ignore[arg-type]
        return action, hidden
    if isinstance(model, (PickFeedForwardController, FlatPickFeedForwardController)):
        return model(observation), hidden  # type: ignore[arg-type]
    raise ValueError(f"Unsupported model type: {type(model)}")


def run_scripted_pick_place_episode(
    env: MujocoManipulatorEnv,
    expert_config: PickPlaceExpertConfig,
    object_x: float,
    destination_x: float,
    graph_builder: ManipulationGraphBuilder | None = None,
    collect_graphs: bool = False,
    episode_id: int = 0,
    render: bool = False,
    render_dir: Path | None = None,
) -> dict[str, Any]:
    env.reset_pick_place_scene(object_x=object_x, destination_x=destination_x, randomize_robot=expert_config.randomize_robot)
    if graph_builder is None:
        graph_builder = ManipulationGraphBuilder(task="pick_and_place", include_object=True)
    if render:
        render_dir = ensure_dir(render_dir or Path("artifacts/mujoco_pick_place/expert_frames"))
    object_start = env.data.site_xpos[env.object_site_id].copy()
    destination = env.target_position_np().copy()
    phase = "APPROACH_OBJECT"
    phase_order = [phase]
    close_counter = 0
    verify_counter = 0
    release_counter = 0
    stable_counter = 0
    lift_target: np.ndarray | None = None
    records: list[dict[str, Any]] = []
    graphs: list[GraphData] = []
    delta_xz: list[torch.Tensor] = []
    grippers: list[float] = []
    phases: list[str] = []
    frames: list[str] = []
    approach_success = False
    grasp_success = False
    lift_success = False
    transport_success = False
    release_success = False
    placement_success = False
    drop = False
    grasp_step: int | None = None
    lift_step: int | None = None
    release_step: int | None = None
    placement_step: int | None = None
    ik_failures = 0
    failure_reason: str | None = None

    for step_idx in range(expert_config.max_steps):
        object_position = env.data.site_xpos[env.object_site_id].copy()
        ee_position = env.data.site_xpos[env.ee_site_id].copy()
        grasp_target = object_position + np.asarray(expert_config.grasp_offset, dtype=np.float64)
        approach_target = grasp_target + np.array([0.0, 0.0, expert_config.approach_height], dtype=np.float64)
        destination_above = destination + np.array([0.0, 0.0, expert_config.transport_height], dtype=np.float64)
        place_target = destination + np.asarray(expert_config.grasp_offset, dtype=np.float64)

        if phase == "APPROACH_OBJECT":
            target = approach_target
            gripper = 0.0
            if _distance_xz(ee_position, approach_target) < 0.035:
                approach_success = True
                phase = "ALIGN_OBJECT"
                phase_order.append(phase)
        elif phase == "ALIGN_OBJECT":
            target = grasp_target
            gripper = 0.0
            if _distance_xz(ee_position, grasp_target) < 0.030:
                phase = "GRASP"
                phase_order.append(phase)
        elif phase == "GRASP":
            target = grasp_target
            gripper = 1.0
            close_counter += 1
            if close_counter >= expert_config.close_steps:
                phase = "VERIFY"
                phase_order.append(phase)
        elif phase == "VERIFY":
            target = grasp_target
            gripper = 1.0
            verify_counter += 1
            grasp = env.grasp_observation()
            if grasp.left_contact and grasp.right_contact:
                grasp_success = True
                grasp_step = step_idx
                lift_target = ee_position + np.array([0.0, 0.0, expert_config.lift_height], dtype=np.float64)
                phase = "LIFT"
                phase_order.append(phase)
            elif verify_counter >= expert_config.verify_steps:
                failure_reason = "failed_grasp"
                phase = "FAILURE"
                phase_order.append(phase)
        elif phase == "LIFT":
            target = lift_target if lift_target is not None else ee_position + np.array([0.0, 0.0, expert_config.lift_height])
            gripper = 1.0
            if object_position[2] - object_start[2] >= expert_config.lift_success_height:
                lift_success = True
                if lift_step is None:
                    lift_step = step_idx
                phase = "TRANSPORT"
                phase_order.append(phase)
        elif phase == "TRANSPORT":
            target = destination_above
            gripper = 1.0
            if _distance_xz(object_position, destination_above) < expert_config.destination_neighborhood:
                transport_success = True
                phase = "ALIGN_DESTINATION"
                phase_order.append(phase)
        elif phase == "ALIGN_DESTINATION":
            target = place_target
            gripper = 1.0
            if _distance_xz(ee_position, place_target) < 0.028:
                phase = "RELEASE"
                phase_order.append(phase)
        elif phase == "RELEASE":
            target = place_target
            gripper = 0.0
            release_counter += 1
            if release_step is None:
                release_step = step_idx
            if release_counter >= expert_config.release_steps:
                release_success = True
                phase = "STABILIZE"
                phase_order.append(phase)
        elif phase == "STABILIZE":
            target = place_target
            gripper = 0.0
        else:
            target = ee_position
            gripper = 0.0

        if collect_graphs:
            graph = graph_builder.build(env.observe())
            current = env.data.site_xpos[env.ee_site_id].copy()
            delta = clamp_delta(torch.tensor(target - current, dtype=torch.float32), env.config.max_delta_ee)
            graphs.append(graph)
            delta_xz.append(torch.stack([delta[0], delta[2]]))
            grippers.append(float(gripper))
            phases.append(phase)

        _, _, _, info = env.step_toward_ee_target(target, gripper=gripper)
        if not bool(info["ik_success"]):
            ik_failures += 1
        grasp = env.grasp_observation()
        object_position = env.data.site_xpos[env.object_site_id].copy()
        ee_position = env.data.site_xpos[env.ee_site_id].copy()
        object_to_ee = float(np.linalg.norm(object_position - ee_position))
        object_height_delta = float(grasp.object_height - object_start[2])
        placement_error = _distance_xz(object_position, destination)
        placing_near_goal = phase in {"ALIGN_DESTINATION", "RELEASE", "STABILIZE"} and (
            placement_error <= expert_config.placement_threshold * 1.5
        )
        if grasp_success and not release_success and not placing_near_goal and object_height_delta < 0.015 and step_idx > (grasp_step or 0) + 20:
            drop = True
            failure_reason = "drop_during_transport"
            phase = "FAILURE"
            phase_order.append(phase)
        if phase == "STABILIZE":
            if placement_error <= expert_config.placement_threshold and object_height_delta < 0.040 and not grasp.object_grasped:
                stable_counter += 1
            else:
                stable_counter = 0
            if stable_counter >= expert_config.release_hold_steps:
                placement_success = True
                placement_step = step_idx
                phase = "SUCCESS"
                phase_order.append(phase)

        records.append(
            {
                "step": step_idx,
                "phase": phase,
                "ee_position": ee_position.tolist(),
                "object_position": object_position.tolist(),
                "destination_position": destination.tolist(),
                "ee_target": target.tolist(),
                "gripper_command": gripper,
                "object_to_ee": object_to_ee,
                "object_height_delta": object_height_delta,
                "placement_error": placement_error,
                "object_velocity": grasp.object_velocity.detach().cpu().tolist(),
                "object_velocity_norm": float(grasp.object_velocity.norm().item()),
                "left_contact": grasp.left_contact,
                "right_contact": grasp.right_contact,
                "object_grasped": grasp.object_grasped,
                "contact_force": grasp.contact_force,
                "ik_success": bool(info["ik_success"]),
            }
        )
        if render and render_dir is not None and step_idx % 4 == 0:
            frame_path = render_dir / f"frame_{step_idx:03d}.png"
            plt.imsave(frame_path, env.render_frame())
            frames.append(str(frame_path))
        if phase in {"SUCCESS", "FAILURE"}:
            break

    if failure_reason is None and not placement_success:
        failure_reason = classify_pick_place_failure(
            approach_success=approach_success,
            grasp_success=grasp_success,
            lift_success=lift_success,
            transport_success=transport_success,
            release_success=release_success,
            records=records,
            ik_failures=ik_failures,
        )
    return {
        "episode_id": int(episode_id),
        "object_x": float(object_x),
        "destination_x": float(destination_x),
        "graphs": graphs,
        "delta_xz": torch.stack(delta_xz, dim=0) if delta_xz else torch.empty(0, 2),
        "gripper": torch.tensor(grippers, dtype=torch.float32),
        "phase_labels_diagnostics_only": phases,
        "steps": len(records),
        "approach_success": approach_success,
        "grasp_success": grasp_success,
        "lift_success": lift_success,
        "transport_success": transport_success,
        "release_success": release_success,
        "placement_success": placement_success,
        "drop": drop,
        "failure_reason": None if placement_success else failure_reason,
        "phase_order": phase_order,
        "grasp_step": grasp_step,
        "lift_step": lift_step,
        "release_step": release_step,
        "placement_step": placement_step,
        "ik_failures": ik_failures,
        "records": records,
        "render_frames": frames,
    }


def train_pick_place_policy(config: PickPlaceTrainConfig) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(config.seed)
    device = select_device(config.device)
    dataset = load_pick_dataset(config.dataset_path)
    max_step = float(dataset["env_config"].get("max_delta_ee", 0.045))
    flat_schema = flat_state_schema_from_graph(dataset["episodes"][0]["graphs"][0])
    model = _build_pick_model(config.model_kind, max_step=max_step, flat_input_dim=flat_schema.input_dim).to(device)
    if config.initial_checkpoint_path is not None:
        checkpoint = torch.load(config.initial_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    gripper_pos_weight = _gripper_pos_weight(dataset, "train").to(device)
    start_time = time.perf_counter()
    history: list[dict[str, Any]] = []
    update_idx = 0
    for epoch in range(1, config.epochs + 1):
        train_ids = list(dataset["split"]["train"])
        rng = np.random.default_rng(config.seed + epoch)
        rng.shuffle(train_ids)
        for episode_id in train_ids:
            losses, updates = train_pick_place_episode(
                model=model,
                model_kind=config.model_kind,
                episode=dataset["episodes"][episode_id],
                optimizer=optimizer,
                device=device,
                config=config,
                gripper_pos_weight=gripper_pos_weight,
            )
            update_idx += updates
            if losses:
                history.append({"epoch": epoch, "episode": int(episode_id), **_mean_losses(losses)})
            if config.log_every and update_idx > 0 and update_idx % config.log_every == 0:
                recent = history[-min(len(history), config.log_every) :]
                print(
                    f"[pick_place:{config.model_kind}] update={update_idx} epoch={epoch} "
                    f"loss={_mean_key(recent, 'loss'):.5f} motion={_mean_key(recent, 'motion_loss'):.5f} "
                    f"gripper={_mean_key(recent, 'gripper_loss'):.5f}"
                )
        validation = evaluate_pick_place_offline(model, dataset, "val", device, max_step)
        history.append({"epoch": epoch, "episode": -1, "validation": validation})
        print(
            f"[pick_place:{config.model_kind}] epoch={epoch} val_l2={validation['motion_l2_error']:.4f} "
            f"val_grip={validation['gripper_accuracy']:.3f} val_cos={validation['action_cosine']:.3f}"
        )

    offline = {split: evaluate_pick_place_offline(model, dataset, split, device, max_step) for split in ("train", "val", "test")}
    output_dir = ensure_dir(config.output_dir)
    checkpoint_path = output_dir / f"{config.model_kind}.pt"
    checkpoint = {
        "model_kind": config.model_kind,
        "model_state": model.state_dict(),
        "config": asdict(config),
        "env_config": dataset["env_config"],
        "split": dataset["split"],
        "offline": offline,
        "history": history,
        "flat_input_dim": flat_schema.input_dim if config.model_kind.startswith("flat_") else None,
        "flat_schema": flat_schema.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "runtime_seconds": time.perf_counter() - start_time,
        "gripper_pos_weight": float(gripper_pos_weight.item()),
    }
    torch.save(checkpoint, checkpoint_path)
    summary_path = output_dir / f"{config.model_kind}_summary.json"
    serializable = dict(checkpoint)
    serializable.pop("model_state", None)
    summary_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return model, {"checkpoint_path": str(checkpoint_path), "summary_path": str(summary_path), **serializable}


def train_pick_place_episode(
    model: nn.Module,
    model_kind: PickModelKind,
    episode: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: PickPlaceTrainConfig,
    gripper_pos_weight: torch.Tensor,
) -> tuple[list[dict[str, float]], int]:
    model.train()
    hidden = model.initial_hidden(device) if _is_recurrent_pick_model(model) else None
    graphs = episode["graphs"]
    deltas = episode["delta_xz"]
    grippers = episode["gripper"]
    losses: list[dict[str, float]] = []
    updates = 0
    for start_idx in range(0, len(graphs), config.bptt_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        total_motion = torch.tensor(0.0, device=device)
        total_gripper = torch.tensor(0.0, device=device)
        chunk_graphs = graphs[start_idx : start_idx + config.bptt_steps]
        for offset, graph in enumerate(chunk_graphs):
            idx = start_idx + offset
            observation = _model_observation(model, graph, device)
            target_delta = deltas[idx].to(device)
            target_gripper = grippers[idx].to(device)
            raw, hidden = _raw_pick_action(model, observation, hidden)
            action = decode_pick_action(raw, model.max_step, model.action_head_type)
            pred_delta = torch.stack([action.delta_ee[0], action.delta_ee[2]])
            if model.action_head_type == "direction_magnitude":
                motion_loss = _direction_magnitude_loss(
                    raw=raw,
                    pred_delta=pred_delta,
                    target_delta=target_delta,
                    max_step=model.max_step,
                    direction_loss_weight=config.direction_loss_weight,
                    magnitude_loss_weight=config.magnitude_loss_weight,
                )
            else:
                motion_loss = torch.nn.functional.smooth_l1_loss(pred_delta, target_delta)
            weight_graph = graph.to(device)
            if config.precision_weighting:
                motion_loss = motion_loss * pick_place_precision_weight(
                    weight_graph,
                    config.near_distance,
                    config.medium_distance,
                    config.near_weight,
                    config.medium_weight,
                )
            else:
                motion_loss = motion_loss * _precision_weight(
                    weight_graph,
                    False,
                    config.near_distance,
                    config.medium_distance,
                    config.near_weight,
                    config.medium_weight,
                )
            gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                gripper_logit(raw, model.action_head_type).reshape(()),
                target_gripper.reshape(()),
                pos_weight=gripper_pos_weight,
            )
            total_motion = total_motion + motion_loss
            total_gripper = total_gripper + gripper_loss
            total_loss = total_loss + config.motion_loss_weight * motion_loss + config.gripper_loss_weight * gripper_loss
        if not chunk_graphs:
            continue
        total_loss = total_loss / len(chunk_graphs)
        total_motion = total_motion / len(chunk_graphs)
        total_gripper = total_gripper / len(chunk_graphs)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if hidden is not None:
            hidden = hidden.detach()
        updates += 1
        losses.append(
            {
                "loss": float(total_loss.detach().cpu().item()),
                "motion_loss": float(total_motion.detach().cpu().item()),
                "gripper_loss": float(total_gripper.detach().cpu().item()),
            }
        )
    return losses, updates


@torch.no_grad()
def evaluate_pick_place_offline(
    model: nn.Module,
    dataset: dict[str, Any],
    split_name: str,
    device: torch.device,
    max_step: float,
) -> dict[str, float]:
    model.eval()
    motion_errors: list[float] = []
    motion_mse: list[float] = []
    cosines: list[float] = []
    magnitudes: list[float] = []
    grip_targets: list[int] = []
    grip_preds: list[int] = []
    grip_probs: list[float] = []
    release_targets: list[int] = []
    release_preds: list[int] = []
    for episode_id in dataset["split"][split_name]:
        episode = dataset["episodes"][episode_id]
        hidden = model.initial_hidden(device) if _is_recurrent_pick_model(model) else None
        prev_target = 0
        prev_pred = 0
        for graph, target_delta, target_gripper in zip(episode["graphs"], episode["delta_xz"], episode["gripper"], strict=True):
            observation = _model_observation(model, graph, device)
            target_delta = target_delta.to(device)
            action, hidden = _decoded_pick_action(model, observation, hidden)
            pred_xz = torch.stack([action.delta_ee[0], action.delta_ee[2]])
            motion_errors.append(float((pred_xz - target_delta).norm().item()))
            motion_mse.append(float(torch.nn.functional.mse_loss(pred_xz, target_delta).item()))
            denom = float(pred_xz.norm().item() * target_delta.norm().item())
            cosines.append(0.0 if denom < 1e-8 else float(torch.dot(pred_xz, target_delta).item() / denom))
            prob = float(action.gripper.item())
            pred = int(prob >= 0.5)
            target = int(float(target_gripper) >= 0.5)
            if prev_target == 1 and target == 0:
                release_targets.append(1)
                release_preds.append(int(prev_pred == 1 and pred == 0))
            prev_target = target
            prev_pred = pred
            grip_targets.append(target)
            grip_preds.append(pred)
            grip_probs.append(prob)
            magnitudes.append(float(pred_xz.norm().item()))
    metrics = _offline_metrics(motion_errors, motion_mse, cosines, magnitudes, grip_targets, grip_preds, grip_probs)
    metrics["release_transition_accuracy"] = sum(release_preds) / max(len(release_targets), 1)
    metrics["release_transition_count"] = float(len(release_targets))
    return metrics


@torch.no_grad()
def evaluate_pick_place_policy(config: PickPlaceEvalConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = select_device(config.device)
    model, checkpoint = load_pick_policy(config.checkpoint_path, device)
    env = MujocoManipulatorEnv(_pick_env_config(max_steps=config.max_steps), seed=config.seed)
    graph_builder = ManipulationGraphBuilder(task="pick_and_place", include_object=True)
    rng = np.random.default_rng(config.seed)
    output_dir = ensure_dir(config.output_dir)
    results: list[dict[str, Any]] = []
    success_rendered = False
    failure_rendered = False
    for episode_idx in range(config.episodes):
        object_x, destination_x = _sample_pick_place_positions(rng, config)
        render = bool(config.render and episode_idx == 0)
        render_dir = output_dir / f"{checkpoint['model_kind']}_{config.recurrent_mode}_frames_ep{episode_idx:03d}" if render else None
        result = run_closed_loop_pick_place_episode(
            model=model,
            model_kind=checkpoint["model_kind"],
            env=env,
            graph_builder=graph_builder,
            device=device,
            object_x=object_x,
            destination_x=destination_x,
            recurrent_mode=config.recurrent_mode,
            render=render,
            render_dir=render_dir,
        )
        result["episode"] = episode_idx + 1
        results.append(result)
        if config.render and result["placement_success"] and not success_rendered:
            result["success_rerender"] = run_closed_loop_pick_place_episode(
                model=model,
                model_kind=checkpoint["model_kind"],
                env=env,
                graph_builder=graph_builder,
                device=device,
                object_x=object_x,
                destination_x=destination_x,
                recurrent_mode=config.recurrent_mode,
                render=True,
                render_dir=output_dir / f"{checkpoint['model_kind']}_{config.recurrent_mode}_success_frames",
            )["render_frames"]
            success_rendered = True
        if config.render and not result["placement_success"] and not failure_rendered:
            result["failure_rerender"] = run_closed_loop_pick_place_episode(
                model=model,
                model_kind=checkpoint["model_kind"],
                env=env,
                graph_builder=graph_builder,
                device=device,
                object_x=object_x,
                destination_x=destination_x,
                recurrent_mode=config.recurrent_mode,
                render=True,
                render_dir=output_dir / f"{checkpoint['model_kind']}_{config.recurrent_mode}_failure_frames",
            )["render_frames"]
            failure_rendered = True
    env.close()
    summary = summarize_pick_place_results(results)
    payload = {
        "checkpoint_path": config.checkpoint_path,
        "model_kind": checkpoint["model_kind"],
        "recurrent_mode": config.recurrent_mode,
        "device": str(device),
        "config": asdict(config),
        "summary": summary,
        "episodes": results,
    }
    path = output_dir / f"{checkpoint['model_kind']}_{config.recurrent_mode}_eval.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_pick_place_rollout(results, output_dir / f"{checkpoint['model_kind']}_{config.recurrent_mode}_closed_loop.png")
    payload["output_path"] = str(path)
    return payload


@torch.no_grad()
def diagnose_pick_place_alignment(config: PickPlaceAlignmentDiagnosticsConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = select_device(config.device)
    model, checkpoint = load_pick_policy(config.checkpoint_path, device)
    env = MujocoManipulatorEnv(_pick_env_config(max_steps=config.max_steps), seed=config.seed)
    graph_builder = ManipulationGraphBuilder(task="pick_and_place", include_object=True)
    expert_config = PickPlaceExpertConfig(max_steps=config.max_steps)
    rng = np.random.default_rng(config.seed)
    output_dir = ensure_dir(config.output_dir)
    policy_results: list[dict[str, Any]] = []
    expert_results: list[dict[str, Any]] = []

    for episode_idx in range(config.episodes):
        object_x, destination_x = _sample_pick_place_positions(rng, config)
        policy = run_closed_loop_pick_place_episode(
            model=model,
            model_kind=checkpoint["model_kind"],
            env=env,
            graph_builder=graph_builder,
            device=device,
            object_x=object_x,
            destination_x=destination_x,
            recurrent_mode="normal",
            render=False,
        )
        policy["episode"] = episode_idx + 1
        expert = run_scripted_pick_place_episode(
            env=env,
            expert_config=expert_config,
            object_x=object_x,
            destination_x=destination_x,
            graph_builder=graph_builder,
            collect_graphs=True,
            episode_id=episode_idx,
            render=False,
        )
        expert["episode"] = episode_idx + 1
        expert["_expert_delta_xz"] = expert["delta_xz"].detach().cpu().tolist()
        expert.pop("graphs", None)
        expert.pop("delta_xz", None)
        expert.pop("gripper", None)
        policy_results.append(policy)
        expert_results.append(expert)

    env.close()
    failure_sources = Counter(
        placement_diagnostics(item)["failure_source"]
        for item in policy_results
        if not item["placement_success"]
    )
    payload = {
        "config": asdict(config),
        "metric_thresholds": asdict(PlacementMetricConfig()),
        "checkpoint_path": config.checkpoint_path,
        "model_kind": checkpoint["model_kind"],
        "policy_summary": summarize_pick_place_results(policy_results),
        "expert_summary": summarize_pick_place_results(expert_results),
        "failure_source_counts": dict(failure_sources),
        "expert_vs_policy": compare_destination_behavior(expert_results, policy_results),
        "action_bins": {
            "expert": destination_action_bin_stats(expert_results, use_expert_actions=True),
            "policy": destination_action_bin_stats(policy_results, use_expert_actions=False),
        },
        "policy_episodes": policy_results,
        "expert_episodes": expert_results,
    }
    path = output_dir / "placement_failure_diagnosis.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_destination_diagnostics(expert_results, policy_results, output_dir)
    payload["output_path"] = str(path)
    return payload


def compare_destination_behavior(
    expert_results: list[dict[str, Any]],
    policy_results: list[dict[str, Any]],
) -> dict[str, Any]:
    expert_diag = [placement_diagnostics(item) for item in expert_results]
    policy_diag = [placement_diagnostics(item) for item in policy_results]
    return {
        "expert": _summarize_destination_diagnostics(expert_diag, expert_results),
        "policy": _summarize_destination_diagnostics(policy_diag, policy_results),
    }


def destination_action_bin_stats(
    results: list[dict[str, Any]],
    use_expert_actions: bool,
    max_step: float = 0.045,
) -> dict[str, dict[str, float]]:
    bins = {
        "far": {"lo": 0.12, "hi": float("inf"), "magnitudes": [], "cosines": [], "saturated": 0, "count": 0},
        "medium": {"lo": 0.06, "hi": 0.12, "magnitudes": [], "cosines": [], "saturated": 0, "count": 0},
        "near_destination": {"lo": 0.04, "hi": 0.06, "magnitudes": [], "cosines": [], "saturated": 0, "count": 0},
        "release_neighborhood": {"lo": 0.0, "hi": 0.04, "magnitudes": [], "cosines": [], "saturated": 0, "count": 0},
    }
    for episode in results:
        records = episode.get("records", [])
        expert_actions = episode.get("_expert_delta_xz", [])
        for idx, record in enumerate(records):
            distance = float(record.get("pre_placement_error", record.get("placement_error", 1e9)))
            if use_expert_actions:
                if idx >= len(expert_actions):
                    continue
                action_xz = np.asarray(expert_actions[idx], dtype=np.float64).reshape(2)
            else:
                delta = np.asarray(record.get("delta_ee", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
                action_xz = delta[[0, 2]]
            magnitude = float(np.linalg.norm(action_xz))
            object_pos = np.asarray(record["object_position"], dtype=np.float64)
            destination = np.asarray(record["destination_position"], dtype=np.float64)
            target_xz = (destination - object_pos)[[0, 2]]
            cosine = _cosine_np(action_xz, target_xz)
            for bucket in bins.values():
                if bucket["lo"] <= distance < bucket["hi"]:
                    bucket["count"] += 1
                    bucket["magnitudes"].append(magnitude)
                    bucket["cosines"].append(cosine)
                    bucket["saturated"] += int(magnitude >= 0.9 * max_step)
                    break
    return {
        name: {
            "count": float(bucket["count"]),
            "mean_action_magnitude": _mean(bucket["magnitudes"]),
            "mean_direction_cosine": _mean(bucket["cosines"]),
            "saturation_ratio": float(bucket["saturated"] / max(bucket["count"], 1)),
        }
        for name, bucket in bins.items()
    }


def plot_destination_diagnostics(
    expert_results: list[dict[str, Any]],
    policy_results: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    output = ensure_dir(output_dir)
    expert_diag = [placement_diagnostics(item) for item in expert_results]
    policy_diag = [placement_diagnostics(item) for item in policy_results]
    for key, filename, xlabel in (
        ("pre_release_placement_error", "pre_release_distance.png", "pre-release placement error"),
        ("pre_release_object_velocity_norm", "release_velocity.png", "pre-release object velocity"),
        ("settle_drift", "settle_drift.png", "final distance - pre-release distance"),
    ):
        expert_values = [float(item[key]) for item in expert_diag if item[key] is not None]
        policy_values = [float(item[key]) for item in policy_diag if item[key] is not None]
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.hist(expert_values, bins=20, alpha=0.55, label="expert")
        ax.hist(policy_values, bins=20, alpha=0.55, label="GRU")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("episodes")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)


def _summarize_destination_diagnostics(
    diagnostics: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, float]:
    count = max(len(results), 1)
    return {
        "mean_pre_release_placement_error": _mean([item["pre_release_placement_error"] for item in diagnostics]),
        "mean_release_velocity": _mean([item["pre_release_object_velocity_norm"] for item in diagnostics]),
        "mean_settle_drift": _mean([item["settle_drift"] for item in diagnostics]),
        "destination_alignment_success_rate": sum(
            1.0 for item in diagnostics if bool(item["destination_alignment_success"])
        )
        / count,
        "mean_release_step": _mean([item["release_step"] for item in diagnostics]),
        "placement_success_rate": sum(1.0 for item in results if item["placement_success"]) / count,
    }


def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


@torch.no_grad()
def aggregate_pick_place_alignment_dagger(config: PickPlaceAlignmentDaggerConfig) -> dict[str, Any]:
    """Append destination-neighborhood policy states labeled by the scripted expert.

    The expert labels are never executed during collection; the simulator follows
    the learned policy, and the expert only supplies supervised targets for the
    graph state the policy actually visited.
    """

    set_seed(config.seed)
    dataset = load_pick_dataset(config.dataset_path)
    device = select_device(config.device)
    policy, checkpoint = load_pick_policy(config.policy_checkpoint_path, device)
    env = MujocoManipulatorEnv(_pick_env_config(max_steps=config.max_steps), seed=config.seed)
    graph_builder = ManipulationGraphBuilder(task="pick_and_place", include_object=True)
    rng = np.random.default_rng(config.seed)
    base_count = len(dataset["episodes"])
    correction_episodes: list[dict[str, Any]] = []
    attempts = 0

    while len(correction_episodes) < config.episodes and attempts < config.episodes * 3:
        attempts += 1
        object_x, destination_x = _sample_pick_place_positions(rng, config)
        episode = _collect_pick_place_alignment_correction_episode(
            policy=policy,
            model_kind=checkpoint["model_kind"],
            env=env,
            graph_builder=graph_builder,
            device=device,
            object_x=object_x,
            destination_x=destination_x,
            episode_id=base_count + len(correction_episodes),
            config=config,
        )
        if episode["steps"] > 0:
            correction_episodes.append(episode)

    env.close()
    if len(correction_episodes) < config.episodes:
        raise RuntimeError(
            f"Only collected {len(correction_episodes)} alignment corrections after {attempts} attempts."
        )

    aggregate = dict(dataset)
    aggregate["episodes"] = list(dataset["episodes"]) + correction_episodes
    correction_ids = [int(ep["episode_id"]) for ep in correction_episodes]
    aggregate["split"] = {
        "train": list(dataset["split"]["train"]) + correction_ids * max(int(config.correction_repeat_count), 1),
        "val": list(dataset["split"]["val"]),
        "test": list(dataset["split"]["test"]),
    }
    aggregate["alignment_dagger"] = {
        "source_checkpoint": config.policy_checkpoint_path,
        "episodes": config.episodes,
        "seed": config.seed,
        "destination_start_threshold": config.destination_start_threshold,
        "release_tail_steps": config.release_tail_steps,
        "correction_repeat_count": config.correction_repeat_count,
        "labeler": "scripted_pick_place_expert_labels_only_policy_controls_simulator",
        "policy_input_exclusions": [
            "expert phase",
            "expert waypoint",
            "time-to-release",
            "phase completion flags",
        ],
    }
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(aggregate, output_path)
    metadata = pick_place_demo_metadata(aggregate)
    metadata["alignment_dagger"] = aggregate["alignment_dagger"]
    metadata["correction_episode_count"] = len(correction_episodes)
    metadata["correction_length"] = {
        "mean": _mean([ep["steps"] for ep in correction_episodes]),
        "min": min(int(ep["steps"]) for ep in correction_episodes),
        "max": max(int(ep["steps"]) for ep in correction_episodes),
    }
    metadata_path = Path(config.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "dataset_path": str(output_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def _collect_pick_place_alignment_correction_episode(
    policy: nn.Module,
    model_kind: PickModelKind,
    env: MujocoManipulatorEnv,
    graph_builder: ManipulationGraphBuilder,
    device: torch.device,
    object_x: float,
    destination_x: float,
    episode_id: int,
    config: PickPlaceAlignmentDaggerConfig,
) -> dict[str, Any]:
    env.reset_pick_place_scene(object_x=object_x, destination_x=destination_x, randomize_robot=False)
    hidden = policy.initial_hidden(device) if _is_recurrent_pick_model(policy) else None
    expert_config = PickPlaceExpertConfig(max_steps=config.max_steps)
    object_start = env.data.site_xpos[env.object_site_id].copy()
    destination = env.target_position_np().copy()
    graphs: list[GraphData] = []
    delta_xz: list[torch.Tensor] = []
    grippers: list[float] = []
    modes: list[str] = []
    records: list[dict[str, Any]] = []
    started = False
    tail_after_open: int | None = None
    approach_success = False
    grasp_success = False
    lift_success = False
    transport_success = False
    release_success = False
    placement_success = False
    first_grasp_step: int | None = None
    first_release_step: int | None = None
    stable_counter = 0
    previous_gripper = 0.0
    ik_failures = 0

    for step_idx in range(config.max_steps):
        graph = graph_builder.build(env.observe())
        expert_delta, expert_gripper, mode = _reactive_pick_place_destination_expert_action(
            env,
            object_start=object_start,
            expert_config=expert_config,
        )
        policy_observation = _model_observation(policy, graph, device)
        action, hidden = _decoded_pick_action(policy, policy_observation, hidden)

        grasp_before = env.grasp_observation()
        object_pos_before = env.data.site_xpos[env.object_site_id].copy()
        placement_before = _distance_xz(object_pos_before, destination)
        object_to_ee_before = float(
            np.linalg.norm(object_pos_before - env.data.site_xpos[env.ee_site_id])
        )
        object_height_delta_before = float(grasp_before.object_height - object_start[2])
        if object_to_ee_before <= 0.08:
            approach_success = True
        if grasp_before.left_contact and grasp_before.right_contact:
            grasp_success = True
            if first_grasp_step is None:
                first_grasp_step = step_idx
        if object_height_delta_before >= expert_config.lift_success_height and object_to_ee_before <= 0.13:
            lift_success = True
        if lift_success and placement_before <= expert_config.destination_neighborhood:
            transport_success = True
        if lift_success and placement_before <= config.destination_start_threshold:
            started = True
        if started:
            graphs.append(graph)
            delta_xz.append(torch.stack([expert_delta[0], expert_delta[2]]))
            grippers.append(float(expert_gripper))
            modes.append(mode)

        _, _, _, info = env.step_delta_ee(
            action.delta_ee.detach().cpu(),
            gripper=float(action.gripper.detach().cpu().item()),
        )
        if not bool(info["ik_success"]):
            ik_failures += 1
        grasp = env.grasp_observation()
        object_pos = env.data.site_xpos[env.object_site_id].copy()
        ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        placement_error = _distance_xz(object_pos, destination)
        object_to_ee = float(np.linalg.norm(object_pos - ee_pos))
        object_height_delta = float(grasp.object_height - object_start[2])
        gripper_value = float(action.gripper.detach().cpu().item())
        opening_now = previous_gripper >= 0.5 and gripper_value < 0.5
        if opening_now and grasp_success and lift_success and transport_success:
            release_success = True
            if first_release_step is None:
                first_release_step = step_idx
            if tail_after_open is None:
                tail_after_open = 0
        if (
            grasp_success
            and lift_success
            and transport_success
            and release_success
            and placement_error <= expert_config.placement_threshold
            and object_height_delta < 0.045
            and not grasp.object_grasped
        ):
            stable_counter += 1
        else:
            stable_counter = 0
        if stable_counter >= expert_config.release_hold_steps:
            placement_success = True
        records.append(
            {
                "step": step_idx,
                "placement_error": placement_error,
                "object_to_ee": object_to_ee,
                "object_height_delta": object_height_delta,
                "policy_gripper": gripper_value,
                "expert_label_gripper": float(expert_gripper),
                "expert_label_mode": mode,
                "object_velocity_norm": float(grasp.object_velocity.norm().item()),
            }
        )
        previous_gripper = gripper_value
        if tail_after_open is not None:
            tail_after_open += 1
            if tail_after_open >= config.release_tail_steps:
                break
        if placement_success:
            break

    return {
        "episode_id": int(episode_id),
        "object_x": float(object_x),
        "destination_x": float(destination_x),
        "graphs": graphs,
        "delta_xz": torch.stack(delta_xz, dim=0) if delta_xz else torch.empty(0, 2),
        "gripper": torch.tensor(grippers, dtype=torch.float32),
        "phase_labels_diagnostics_only": modes,
        "steps": len(graphs),
        "approach_success": approach_success,
        "grasp_success": grasp_success,
        "lift_success": lift_success,
        "transport_success": transport_success,
        "release_success": release_success,
        "placement_success": placement_success,
        "drop": bool(grasp_success and not release_success and not placement_success),
        "grasp_step": first_grasp_step,
        "release_step": first_release_step,
        "placement_step": None,
        "ik_failures": ik_failures,
        "dagger_policy_visited": True,
        "dagger_expert_labels_only": True,
        "records": records,
    }


def _reactive_pick_place_destination_expert_action(
    env: MujocoManipulatorEnv,
    object_start: np.ndarray,
    expert_config: PickPlaceExpertConfig,
) -> tuple[torch.Tensor, float, str]:
    object_position = env.data.site_xpos[env.object_site_id].copy()
    ee_position = env.data.site_xpos[env.ee_site_id].copy()
    destination = env.target_position_np().copy()
    grasp = env.grasp_observation()
    object_height_delta = float(grasp.object_height - object_start[2])
    placement_error = _distance_xz(object_position, destination)
    object_velocity = float(grasp.object_velocity.norm().item())
    place_target = destination + np.asarray(expert_config.grasp_offset, dtype=np.float64)
    ee_place_error = _distance_xz(ee_position, place_target)
    thresholds = PlacementMetricConfig()

    if object_height_delta < expert_config.lift_success_height * 0.5 and not grasp.object_grasped:
        target = object_position + np.asarray(expert_config.grasp_offset, dtype=np.float64)
        gripper = 1.0
        mode = "label_recover_grasp"
    elif (
        placement_error <= thresholds.alignment_threshold
        and object_velocity <= thresholds.high_velocity_threshold
        and ee_place_error <= 0.030
    ):
        target = place_target
        gripper = 0.0
        mode = "label_release"
    else:
        target = place_target
        gripper = 1.0
        mode = "label_align_and_stabilize"
    delta = clamp_delta(torch.tensor(target - ee_position, dtype=torch.float32), env.config.max_delta_ee)
    return delta, gripper, mode


@torch.no_grad()
def run_closed_loop_pick_place_episode(
    model: nn.Module,
    model_kind: PickModelKind,
    env: MujocoManipulatorEnv,
    graph_builder: ManipulationGraphBuilder,
    device: torch.device,
    object_x: float,
    destination_x: float,
    recurrent_mode: RecurrentEvalMode = "normal",
    render: bool = False,
    render_dir: Path | None = None,
) -> dict[str, Any]:
    env.reset_pick_place_scene(object_x=object_x, destination_x=destination_x, randomize_robot=False)
    if render:
        render_dir = ensure_dir(render_dir or Path("artifacts/mujoco_pick_place/eval/frames"))
    hidden = model.initial_hidden(device) if _is_recurrent_pick_model(model) else None
    object_start = env.data.site_xpos[env.object_site_id].copy()
    destination = env.target_position_np().copy()
    records: list[dict[str, Any]] = []
    frames: list[str] = []
    approach_success = False
    alignment_success = False
    grasp_success = False
    lift_success = False
    transport_success = False
    release_success = False
    placement_success = False
    drop = False
    first_gripper_close_step: int | None = None
    first_grasp_step: int | None = None
    first_release_step: int | None = None
    gripper_open_step: int | None = None
    valid_release_step: int | None = None
    gripper_open_event = False
    valid_release_success = False
    pre_release_placement_error: float | None = None
    pre_release_object_velocity_norm: float | None = None
    pre_release_ee_velocity_norm: float | None = None
    post_release_placement_error: float | None = None
    post_release_object_velocity_norm: float | None = None
    placement_step: int | None = None
    stable_counter = 0
    ik_failures = 0
    latencies_ms: list[float] = []
    expert_config = PickPlaceExpertConfig(max_steps=env.config.max_steps)
    previous_gripper_value = 0.0

    for step_idx in range(env.config.max_steps):
        pre_grasp = env.grasp_observation()
        pre_robot = env.robot_observation()
        pre_object_pos = env.data.site_xpos[env.object_site_id].copy()
        pre_ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        pre_object_velocity = pre_grasp.object_velocity.detach().cpu().numpy()
        pre_ee_velocity = pre_robot.ee_velocity.detach().cpu().numpy()
        pre_step_placement_error = _distance_xz(pre_object_pos, destination)
        graph = graph_builder.build(env.observe())
        observation = _model_observation(model, graph, device)
        start = time.perf_counter()
        if _is_recurrent_pick_model(model):
            if recurrent_mode == "step_reset":
                hidden = model.initial_hidden(device)
        action, hidden = _decoded_pick_action(model, observation, hidden)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        delta_ee = action.delta_ee.detach().cpu()
        gripper_value = float(action.gripper.detach().cpu().item())
        opening_now = bool(
            first_gripper_close_step is not None
            and previous_gripper_value >= 0.5
            and gripper_value < 0.5
        )
        _, _, _, info = env.step_delta_ee(delta_ee, gripper=gripper_value)
        if not bool(info["ik_success"]):
            ik_failures += 1
        grasp = env.grasp_observation()
        object_pos = env.data.site_xpos[env.object_site_id].copy()
        ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        actual_ee_motion = ee_pos - pre_ee_pos
        object_to_ee = float(np.linalg.norm(object_pos - ee_pos))
        object_height_delta = float(grasp.object_height - object_start[2])
        placement_error = _distance_xz(object_pos, destination)
        if object_to_ee <= 0.08:
            approach_success = True
        if object_to_ee <= 0.025:
            alignment_success = True
        if gripper_value >= 0.5 and first_gripper_close_step is None:
            first_gripper_close_step = step_idx
        if grasp.left_contact and grasp.right_contact:
            grasp_success = True
            if first_grasp_step is None:
                first_grasp_step = step_idx
        if object_height_delta >= expert_config.lift_success_height and object_to_ee <= 0.13:
            lift_success = True
        if lift_success and placement_error <= expert_config.destination_neighborhood:
            transport_success = True
        if opening_now and gripper_open_step is None:
            gripper_open_event = True
            gripper_open_step = step_idx
            pre_release_placement_error = float(pre_step_placement_error)
            pre_release_object_velocity_norm = float(np.linalg.norm(pre_object_velocity))
            pre_release_ee_velocity_norm = float(np.linalg.norm(pre_ee_velocity))
            post_release_placement_error = float(placement_error)
            post_release_object_velocity_norm = float(grasp.object_velocity.norm().item())
            if grasp_success and lift_success and transport_success:
                valid_release_success = True
                release_success = True
                valid_release_step = step_idx
                first_release_step = step_idx
        placing_near_goal = placement_error <= expert_config.placement_threshold * 1.5
        if grasp_success and not release_success and not placing_near_goal and object_height_delta < 0.015 and step_idx > (first_grasp_step or 0) + 20:
            drop = True
        if (
            grasp_success
            and lift_success
            and transport_success
            and release_success
            and placement_error <= expert_config.placement_threshold
            and object_height_delta < 0.045
            and not grasp.object_grasped
        ):
            stable_counter += 1
        else:
            stable_counter = 0
        if stable_counter >= expert_config.release_hold_steps:
            placement_success = True
            placement_step = step_idx
            break
        records.append(
            {
                "step": step_idx,
                "ee_position": ee_pos.tolist(),
                "object_position": object_pos.tolist(),
                "destination_position": destination.tolist(),
                "delta_ee": delta_ee.tolist(),
                "predicted_action_magnitude": float(delta_ee.norm().item()),
                "actual_ee_motion": actual_ee_motion.tolist(),
                "actual_ee_motion_magnitude": float(np.linalg.norm(actual_ee_motion)),
                "gripper": gripper_value,
                "gripper_open_event": opening_now,
                "valid_release_event": bool(opening_now and grasp_success and lift_success and transport_success),
                "object_to_ee": object_to_ee,
                "object_height_delta": object_height_delta,
                "pre_placement_error": float(pre_step_placement_error),
                "placement_error": placement_error,
                "pre_object_velocity": pre_object_velocity.tolist(),
                "pre_object_velocity_norm": float(np.linalg.norm(pre_object_velocity)),
                "object_velocity": grasp.object_velocity.detach().cpu().tolist(),
                "object_velocity_norm": float(grasp.object_velocity.norm().item()),
                "pre_ee_velocity": pre_ee_velocity.tolist(),
                "pre_ee_velocity_norm": float(np.linalg.norm(pre_ee_velocity)),
                "left_contact": grasp.left_contact,
                "right_contact": grasp.right_contact,
                "object_grasped": grasp.object_grasped,
                "contact_force": grasp.contact_force,
                "ik_success": bool(info["ik_success"]),
            }
        )
        previous_gripper_value = gripper_value
        if render and render_dir is not None and step_idx % 4 == 0:
            frame_path = render_dir / f"frame_{step_idx:03d}.png"
            plt.imsave(frame_path, env.render_frame())
            frames.append(str(frame_path))

    failure_reason = None if placement_success else classify_pick_place_failure(
        approach_success=approach_success,
        grasp_success=grasp_success,
        lift_success=lift_success,
        transport_success=transport_success,
        release_success=release_success,
        records=records,
        ik_failures=ik_failures,
    )
    return {
        "object_x": float(object_x),
        "destination_x": float(destination_x),
        "approach_success": approach_success,
        "alignment_success": alignment_success,
        "grasp_success": grasp_success,
        "lift_success": lift_success,
        "transport_success": transport_success,
        "release_success": release_success,
        "gripper_open_event": gripper_open_event,
        "valid_release_success": valid_release_success,
        "placement_success": placement_success,
        "drop": drop or bool(grasp_success and not placement_success and not release_success),
        "failure_reason": failure_reason,
        "steps": len(records),
        "first_gripper_close_step": first_gripper_close_step,
        "first_grasp_step": first_grasp_step,
        "first_release_step": first_release_step,
        "gripper_open_step": gripper_open_step,
        "valid_release_step": valid_release_step,
        "pre_release_placement_error": pre_release_placement_error,
        "pre_release_object_velocity_norm": pre_release_object_velocity_norm,
        "pre_release_ee_velocity_norm": pre_release_ee_velocity_norm,
        "post_release_placement_error": post_release_placement_error,
        "post_release_object_velocity_norm": post_release_object_velocity_norm,
        "placement_step": placement_step,
        "ik_failures": ik_failures,
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "records": records,
        "render_frames": frames,
    }


def summarize_pick_place_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = max(len(results), 1)
    failures = Counter(str(item["failure_reason"]) for item in results if not item["placement_success"])
    diagnostics = [placement_diagnostics(item) for item in results]
    return {
        "approach_success_rate": sum(1.0 for item in results if item["approach_success"]) / count,
        "alignment_success_rate": sum(1.0 for item in results if item.get("alignment_success", item["grasp_success"])) / count,
        "grasp_success_rate": sum(1.0 for item in results if item["grasp_success"]) / count,
        "lift_success_rate": sum(1.0 for item in results if item["lift_success"]) / count,
        "transport_success_rate": sum(1.0 for item in results if item["transport_success"]) / count,
        "gripper_open_rate": sum(1.0 for item in results if _has_gripper_open_event(item)) / count,
        "valid_release_rate": sum(1.0 for item in results if _has_valid_release(item)) / count,
        "release_success_rate": sum(1.0 for item in results if _has_valid_release(item)) / count,
        "successful_placement_release_rate": sum(1.0 for item in results if item["placement_success"]) / count,
        "placement_success_rate": sum(1.0 for item in results if item["placement_success"]) / count,
        "drop_rate": sum(1.0 for item in results if item["drop"]) / count,
        "mean_steps": _mean([item["steps"] for item in results]),
        "mean_time_to_grasp": _mean([item.get("first_grasp_step", item.get("grasp_step")) for item in results]),
        "mean_time_to_gripper_open": _mean([item.get("gripper_open_step", item.get("first_release_step", item.get("release_step"))) for item in results]),
        "mean_time_to_release": _mean([item.get("valid_release_step", item.get("first_release_step", item.get("release_step"))) for item in results]),
        "mean_time_to_place": _mean([item["placement_step"] for item in results]),
        "mean_final_placement_error": _mean([_final_record_value(item, "placement_error") for item in results]),
        "mean_action_smoothness": _mean([_episode_action_smoothness(item) for item in results]),
        "mean_pre_release_placement_error": _mean([diag["pre_release_placement_error"] for diag in diagnostics]),
        "mean_release_velocity": _mean([diag["pre_release_object_velocity_norm"] for diag in diagnostics]),
        "mean_settle_drift": _mean([diag["settle_drift"] for diag in diagnostics]),
        "destination_alignment_success_rate": sum(
            1.0 for diag in diagnostics if bool(diag["destination_alignment_success"])
        )
        / count,
        "placement_failure_sources": dict(
            Counter(
                str(diag["failure_source"])
                for diag in diagnostics
                if str(diag["failure_source"]) != "success"
            )
        ),
        "mean_latency_ms": _mean([item.get("mean_latency_ms") for item in results]),
        "failure_counts": dict(failures),
    }


def classify_pick_place_failure(
    approach_success: bool,
    grasp_success: bool,
    lift_success: bool,
    transport_success: bool,
    release_success: bool,
    records: list[dict[str, Any]],
    ik_failures: int,
) -> str:
    if ik_failures > max(len(records) // 2, 1):
        return "IK/control failure"
    if not approach_success:
        return "object approach"
    if not grasp_success:
        return "failed grasp"
    if not lift_success:
        return "drop during lift"
    if not transport_success:
        return "transport drift"
    if not release_success:
        last_record = records[-1] if records else {}
        last_gripper = float(last_record.get("gripper", last_record.get("gripper_command", 1.0)))
        return "late/no release" if last_gripper >= 0.5 else "placement alignment"
    final_error = float(records[-1]["placement_error"]) if records else 1e9
    if final_error > PickPlaceExpertConfig().placement_threshold:
        return "placement miss"
    return "placement instability"


def placement_diagnostics(
    episode: dict[str, Any],
    thresholds: PlacementMetricConfig | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or PlacementMetricConfig()
    records = episode.get("records", [])
    final_distance = _final_record_value(episode, "placement_error")
    release_step = _release_step(episode)
    release_record = _record_at_step(records, release_step)
    previous_record = _record_before_step(records, release_step)
    pre_release_distance = _first_float(
        episode.get("pre_release_placement_error"),
        release_record.get("pre_placement_error") if release_record else None,
        previous_record.get("placement_error") if previous_record else None,
    )
    post_release_distance = _first_float(
        episode.get("post_release_placement_error"),
        release_record.get("placement_error") if release_record else None,
    )
    pre_release_velocity = _first_float(
        episode.get("pre_release_object_velocity_norm"),
        release_record.get("pre_object_velocity_norm") if release_record else None,
        previous_record.get("object_velocity_norm") if previous_record else None,
    )
    pre_release_ee_velocity = _first_float(
        episode.get("pre_release_ee_velocity_norm"),
        release_record.get("pre_ee_velocity_norm") if release_record else None,
        previous_record.get("pre_ee_velocity_norm") if previous_record else None,
    )
    settle_drift = None
    if final_distance is not None and pre_release_distance is not None:
        settle_drift = float(final_distance - pre_release_distance)
    destination_alignment_success = (
        pre_release_distance is not None and pre_release_distance <= thresholds.alignment_threshold
    )
    failure_source = classify_placement_failure_source(
        episode,
        pre_release_distance=pre_release_distance,
        pre_release_velocity=pre_release_velocity,
        thresholds=thresholds,
    )
    return {
        "release_step": release_step,
        "pre_release_placement_error": pre_release_distance,
        "post_release_placement_error": post_release_distance,
        "final_placement_error": final_distance,
        "pre_release_object_velocity_norm": pre_release_velocity,
        "pre_release_ee_velocity_norm": pre_release_ee_velocity,
        "settle_drift": settle_drift,
        "destination_alignment_success": destination_alignment_success,
        "failure_source": failure_source,
    }


def classify_placement_failure_source(
    episode: dict[str, Any],
    pre_release_distance: float | None = None,
    pre_release_velocity: float | None = None,
    thresholds: PlacementMetricConfig | None = None,
) -> str:
    thresholds = thresholds or PlacementMetricConfig()
    if bool(episode.get("placement_success")):
        return "success"
    if not _has_valid_release(episode):
        return "other"
    if pre_release_distance is None:
        return "other"
    if pre_release_distance > thresholds.placement_threshold:
        return "pre_release_alignment_failure"
    if pre_release_distance > thresholds.alignment_threshold:
        return "premature_release"
    if pre_release_velocity is not None and pre_release_velocity > thresholds.high_velocity_threshold:
        return "high_velocity_release"
    return "post_release_dynamics_failure"


def _has_gripper_open_event(episode: dict[str, Any]) -> bool:
    if "gripper_open_event" in episode:
        return bool(episode["gripper_open_event"])
    return episode.get("gripper_open_step", episode.get("first_release_step", episode.get("release_step"))) is not None


def _has_valid_release(episode: dict[str, Any]) -> bool:
    if "valid_release_success" in episode:
        return bool(episode["valid_release_success"])
    return bool(
        episode.get("release_success")
        and episode.get("grasp_success")
        and episode.get("lift_success")
        and episode.get("transport_success", True)
    )


def _release_step(episode: dict[str, Any]) -> int | None:
    for key in ("valid_release_step", "gripper_open_step", "first_release_step", "release_step"):
        value = episode.get(key)
        if value is not None:
            return int(value)
    return None


def _record_at_step(records: list[dict[str, Any]], step: int | None) -> dict[str, Any] | None:
    if step is None:
        return None
    for record in records:
        if int(record.get("step", -1)) == int(step):
            return record
    return None


def _record_before_step(records: list[dict[str, Any]], step: int | None) -> dict[str, Any] | None:
    if step is None:
        return records[-1] if records else None
    previous = [record for record in records if int(record.get("step", -1)) < int(step)]
    return previous[-1] if previous else None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        return float(value)
    return None


def pick_place_demo_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    episodes = payload["episodes"]
    lengths = [int(ep["steps"]) for ep in episodes]
    grippers = torch.cat([ep["gripper"] for ep in episodes], dim=0)
    deltas = torch.cat([ep["delta_xz"] for ep in episodes], dim=0)
    phases = Counter(phase for ep in episodes for phase in ep["phase_labels_diagnostics_only"])
    closing = 0
    release = 0
    for ep in episodes:
        values = [int(float(v) >= 0.5) for v in ep["gripper"]]
        closing += sum(1 for a, b in zip(values[:-1], values[1:], strict=True) if a == 0 and b == 1)
        release += sum(1 for a, b in zip(values[:-1], values[1:], strict=True) if a == 1 and b == 0)
    return {
        "num_episodes": len(episodes),
        "split": payload["split"],
        "episode_length": {"mean": _mean(lengths), "min": min(lengths), "max": max(lengths)},
        "object_x_range": payload["config"]["object_x_range"],
        "destination_x_range": payload["config"]["destination_x_range"],
        "min_separation": payload["config"]["min_separation"],
        "gripper_distribution": {
            "open_fraction": float((grippers < 0.5).float().mean().item()),
            "closed_fraction": float((grippers >= 0.5).float().mean().item()),
            "closing_transition_count": closing,
            "release_transition_count": release,
        },
        "delta_xz": {
            "mean_norm": float(deltas.norm(dim=-1).mean().item()),
            "max_norm": float(deltas.norm(dim=-1).max().item()),
        },
        "phase_counts_diagnostics_only": dict(phases),
        "policy_input_exclusions": [
            "phase_labels_diagnostics_only",
            "expert waypoint",
            "expert target pose",
            "phase completion flags",
        ],
    }


def pick_place_precision_weight(
    graph: GraphData,
    near_distance: float,
    medium_distance: float,
    near_weight: float,
    medium_weight: float,
) -> torch.Tensor:
    features = graph.node_features
    ee = features[graph.ee_node_index, 0:3]
    destination = features[graph.target_node_index, 0:3]
    distances = [(destination - ee).norm()]
    for idx, node_type in enumerate(graph.node_types):
        if node_type == "object":
            obj = features[idx, 0:3]
            distances.append((obj - ee).norm())
            distances.append((destination - obj).norm())
    distance = torch.stack(distances).min()
    if float(distance.detach().cpu().item()) <= near_distance:
        value = near_weight
    elif float(distance.detach().cpu().item()) <= medium_distance:
        value = medium_weight
    else:
        value = 1.0
    return torch.tensor(value, device=features.device, dtype=features.dtype)


def plot_pick_place_rollout(results: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = results[0]["records"] if results else []
    if not records:
        path.write_text("no records", encoding="utf-8")
        return path
    xs = np.arange(len(records))
    placement = np.asarray([row["placement_error"] for row in records], dtype=float)
    height = np.asarray([row["object_height_delta"] for row in records], dtype=float)
    gripper = np.asarray([row.get("gripper", row.get("gripper_command", 0.0)) for row in records], dtype=float)
    contact = np.asarray([row["left_contact"] and row["right_contact"] for row in records], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    axes[0].plot(xs, placement)
    axes[0].axhline(PickPlaceExpertConfig().placement_threshold, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("placement error")
    axes[1].plot(xs, height)
    axes[1].axhline(PickPlaceExpertConfig().lift_success_height, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("object height")
    axes[2].plot(xs, gripper, label="gripper")
    axes[2].plot(xs, contact, label="both contact")
    axes[2].legend(loc="best")
    axes[2].set_xlabel("step")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _sample_pick_place_positions(rng: np.random.Generator, config: PickPlaceDemoConfig | PickPlaceEvalConfig) -> tuple[float, float]:
    for _ in range(200):
        object_x = float(rng.uniform(*config.object_x_range))
        destination_x = float(rng.uniform(*config.destination_x_range))
        if abs(destination_x - (object_x + OBJECT_SITE_X_OFFSET)) >= config.min_separation:
            return object_x, destination_x
    return float(config.object_x_range[0]), float(config.destination_x_range[1])


def _distance_xz(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b))[[0, 2]]))


def _final_record_value(item: dict[str, Any], key: str) -> float | None:
    records = item.get("records", [])
    if not records:
        return None
    return float(records[-1][key])


def _episode_action_smoothness(item: dict[str, Any]) -> float | None:
    records = item.get("records", [])
    deltas = [np.asarray(record.get("delta_ee"), dtype=np.float64) for record in records if "delta_ee" in record]
    if len(deltas) < 2:
        return None
    return float(np.mean([np.linalg.norm(curr - prev) for prev, curr in zip(deltas[:-1], deltas[1:], strict=True)]))
