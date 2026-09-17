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
from vla_gnn_recurrent.graph.types import GraphData
from vla_gnn_recurrent.models.pick_controller import PickFeedForwardController, PickRecurrentController
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
    model = _build_pick_model(config.model_kind, max_step=max_step).to(device)
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
    hidden = model.initial_hidden(device) if isinstance(model, PickRecurrentController) else None
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
            graph = graph.to(device)
            target_delta = deltas[idx].to(device)
            target_gripper = grippers[idx].to(device)
            if isinstance(model, PickRecurrentController):
                raw, hidden = model.raw_action(graph, hidden)
            elif isinstance(model, PickFeedForwardController):
                raw = model.raw_action(graph)
            else:
                raise ValueError(f"Unsupported model kind: {model_kind}")
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
            if config.precision_weighting:
                motion_loss = motion_loss * pick_place_precision_weight(
                    graph,
                    config.near_distance,
                    config.medium_distance,
                    config.near_weight,
                    config.medium_weight,
                )
            else:
                motion_loss = motion_loss * _precision_weight(
                    graph,
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
        hidden = model.initial_hidden(device) if isinstance(model, PickRecurrentController) else None
        prev_target = 0
        prev_pred = 0
        for graph, target_delta, target_gripper in zip(episode["graphs"], episode["delta_xz"], episode["gripper"], strict=True):
            graph = graph.to(device)
            target_delta = target_delta.to(device)
            if isinstance(model, PickRecurrentController):
                action, hidden = model(graph, hidden)
            elif isinstance(model, PickFeedForwardController):
                action = model(graph)
            else:
                raise ValueError(f"Unsupported model: {type(model)}")
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
    hidden = model.initial_hidden(device) if isinstance(model, PickRecurrentController) else None
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
    placement_step: int | None = None
    stable_counter = 0
    ik_failures = 0
    latencies_ms: list[float] = []
    expert_config = PickPlaceExpertConfig(max_steps=env.config.max_steps)

    for step_idx in range(env.config.max_steps):
        graph = graph_builder.build(env.observe()).to(device)
        start = time.perf_counter()
        if isinstance(model, PickRecurrentController):
            if recurrent_mode == "step_reset":
                hidden = model.initial_hidden(device)
            action, hidden = model(graph, hidden)
        elif isinstance(model, PickFeedForwardController):
            action = model(graph)
        else:
            raise ValueError(f"Unsupported model kind: {model_kind}")
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        delta_ee = action.delta_ee.detach().cpu()
        gripper_value = float(action.gripper.detach().cpu().item())
        _, _, _, info = env.step_delta_ee(delta_ee, gripper=gripper_value)
        if not bool(info["ik_success"]):
            ik_failures += 1
        grasp = env.grasp_observation()
        object_pos = env.data.site_xpos[env.object_site_id].copy()
        ee_pos = env.data.site_xpos[env.ee_site_id].copy()
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
        if first_gripper_close_step is not None and gripper_value < 0.5 and step_idx > first_gripper_close_step + 10:
            release_success = True
            if first_release_step is None:
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
                "gripper": gripper_value,
                "object_to_ee": object_to_ee,
                "object_height_delta": object_height_delta,
                "placement_error": placement_error,
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
        "placement_success": placement_success,
        "drop": drop or bool(grasp_success and not placement_success and not release_success),
        "failure_reason": failure_reason,
        "steps": len(records),
        "first_gripper_close_step": first_gripper_close_step,
        "first_grasp_step": first_grasp_step,
        "first_release_step": first_release_step,
        "placement_step": placement_step,
        "ik_failures": ik_failures,
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "records": records,
        "render_frames": frames,
    }


def summarize_pick_place_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = max(len(results), 1)
    failures = Counter(str(item["failure_reason"]) for item in results if not item["placement_success"])
    return {
        "approach_success_rate": sum(1.0 for item in results if item["approach_success"]) / count,
        "alignment_success_rate": sum(1.0 for item in results if item.get("alignment_success", item["grasp_success"])) / count,
        "grasp_success_rate": sum(1.0 for item in results if item["grasp_success"]) / count,
        "lift_success_rate": sum(1.0 for item in results if item["lift_success"]) / count,
        "transport_success_rate": sum(1.0 for item in results if item["transport_success"]) / count,
        "release_success_rate": sum(1.0 for item in results if item["release_success"]) / count,
        "placement_success_rate": sum(1.0 for item in results if item["placement_success"]) / count,
        "drop_rate": sum(1.0 for item in results if item["drop"]) / count,
        "mean_steps": _mean([item["steps"] for item in results]),
        "mean_time_to_grasp": _mean([item.get("first_grasp_step", item.get("grasp_step")) for item in results]),
        "mean_time_to_release": _mean([item.get("first_release_step", item.get("release_step")) for item in results]),
        "mean_time_to_place": _mean([item["placement_step"] for item in results]),
        "mean_final_placement_error": _mean([_final_record_value(item, "placement_error") for item in results]),
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
    distances = [(features[graph.target_node_index, 0:3] - ee).norm()]
    for idx, node_type in enumerate(graph.node_types):
        if node_type == "object":
            distances.append((features[idx, 0:3] - ee).norm())
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
