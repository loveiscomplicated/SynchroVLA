from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from vla_gnn_recurrent.graph.manipulation_graph_builder import ManipulationGraphBuilder
from vla_gnn_recurrent.graph.types import GraphData
from vla_gnn_recurrent.models.pick_controller import (
    PickAction,
    PickFeedForwardController,
    PickRecurrentController,
    decode_pick_action,
    gripper_logit,
)
from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv, MujocoReachConfig
from vla_gnn_recurrent.sim.pick_expert import ScriptedPickConfig
from vla_gnn_recurrent.utils import DevicePreference, clamp_delta, ensure_dir, select_device, set_seed

PickModelKind = Literal["graph_recurrent", "graph_feedforward", "graph_recurrent_dir_mag", "graph_feedforward_dir_mag"]
RecurrentEvalMode = Literal["normal", "step_reset"]


@dataclass
class PickDemoConfig:
    episodes: int = 100
    seed: int = 123
    output_path: str = "artifacts/mujoco_learned_pick/demos/pick_demos.pt"
    metadata_path: str = "artifacts/mujoco_learned_pick/demos/pick_demos_metadata.json"
    object_x_range: tuple[float, float] = (-0.015, 0.0)
    max_steps: int = 260
    max_attempts_multiplier: int = 2


@dataclass
class PickTrainConfig:
    dataset_path: str = "artifacts/mujoco_learned_pick/demos/pick_demos.pt"
    model_kind: PickModelKind = "graph_recurrent"
    output_dir: str = "artifacts/mujoco_learned_pick/checkpoints"
    epochs: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    bptt_steps: int = 16
    motion_loss_weight: float = 1.0
    direction_loss_weight: float = 1.0
    magnitude_loss_weight: float = 1.0
    gripper_loss_weight: float = 1.0
    precision_weighting: bool = False
    near_distance: float = 0.05
    medium_distance: float = 0.10
    near_weight: float = 4.0
    medium_weight: float = 2.0
    seed: int = 123
    device: DevicePreference = "auto"
    log_every: int = 50
    initial_checkpoint_path: str | None = None


@dataclass
class PickDaggerConfig:
    dataset_path: str = "artifacts/mujoco_learned_pick/demos/pick_demos_300.pt"
    policy_checkpoint_path: str = "artifacts/mujoco_learned_pick/checkpoints_gru_300_motion/graph_recurrent.pt"
    output_path: str = "artifacts/mujoco_learned_pick/demos/pick_demos_300_dagger.pt"
    metadata_path: str = "artifacts/mujoco_learned_pick/demos/pick_demos_300_dagger_metadata.json"
    episodes: int = 50
    seed: int = 1777
    device: DevicePreference = "auto"
    max_steps: int = 140
    object_x_range: tuple[float, float] = (-0.015, 0.0)


@dataclass
class PickEvalConfig:
    checkpoint_path: str = "artifacts/mujoco_learned_pick/checkpoints/graph_recurrent.pt"
    episodes: int = 100
    seed: int = 991
    output_dir: str = "artifacts/mujoco_learned_pick/eval"
    recurrent_mode: RecurrentEvalMode = "normal"
    device: DevicePreference = "auto"
    max_steps: int = 180
    render: bool = True
    object_x_range: tuple[float, float] = (-0.015, 0.0)


def generate_pick_demonstrations(config: PickDemoConfig) -> dict[str, Any]:
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    env_config = _pick_env_config(max_steps=config.max_steps)
    env = MujocoManipulatorEnv(env_config, seed=config.seed)
    graph_builder = ManipulationGraphBuilder(task="pick", include_object=True)
    expert_config = ScriptedPickConfig(
        max_steps=config.max_steps,
        object_x_range=config.object_x_range,
        render=False,
    )
    episodes: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = config.episodes * config.max_attempts_multiplier

    while len(episodes) < config.episodes and attempts < max_attempts:
        attempts += 1
        object_x = float(rng.uniform(*config.object_x_range))
        demo = _collect_pick_demo_episode(
            env=env,
            graph_builder=graph_builder,
            expert_config=expert_config,
            object_x=object_x,
            episode_id=len(episodes),
        )
        if demo["lift_success"]:
            episodes.append(demo)

    env.close()
    if len(episodes) < config.episodes:
        raise RuntimeError(f"Only collected {len(episodes)} successful demos after {attempts} attempts.")

    split = _episode_split(len(episodes), seed=config.seed)
    payload = {
        "config": asdict(config),
        "env_config": asdict(env_config),
        "graph_builder": {"task": "pick", "include_object": True},
        "episodes": episodes,
        "split": split,
    }
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    metadata = _demo_metadata(payload)
    metadata_path = Path(config.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"dataset_path": str(output_path), "metadata_path": str(metadata_path), "metadata": metadata}


@torch.no_grad()
def aggregate_pick_dagger(config: PickDaggerConfig) -> dict[str, Any]:
    """Append policy-visited states labeled by a reactive Cartesian expert to the train split."""

    set_seed(config.seed)
    dataset = load_pick_dataset(config.dataset_path)
    device = select_device(config.device)
    policy, checkpoint = load_pick_policy(config.policy_checkpoint_path, device)
    env_config = _pick_env_config(max_steps=config.max_steps)
    env = MujocoManipulatorEnv(env_config, seed=config.seed)
    graph_builder = ManipulationGraphBuilder(task="pick", include_object=True)
    rng = np.random.default_rng(config.seed)
    new_episodes: list[dict[str, Any]] = []
    base_count = len(dataset["episodes"])

    for local_idx in range(config.episodes):
        object_x = float(rng.uniform(*config.object_x_range))
        episode = _collect_dagger_episode(
            policy=policy,
            model_kind=checkpoint["model_kind"],
            env=env,
            graph_builder=graph_builder,
            device=device,
            object_x=object_x,
            episode_id=base_count + local_idx,
        )
        new_episodes.append(episode)

    env.close()
    aggregate = dict(dataset)
    aggregate["episodes"] = list(dataset["episodes"]) + new_episodes
    aggregate["split"] = {
        "train": list(dataset["split"]["train"]) + [int(ep["episode_id"]) for ep in new_episodes],
        "val": list(dataset["split"]["val"]),
        "test": list(dataset["split"]["test"]),
    }
    aggregate["dagger"] = {
        "source_checkpoint": config.policy_checkpoint_path,
        "episodes": config.episodes,
        "seed": config.seed,
        "labeler": "reactive_cartesian_pick_expert_without_phase_input_to_policy",
    }
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(aggregate, output_path)
    metadata = _demo_metadata(aggregate)
    metadata["dagger"] = aggregate["dagger"]
    metadata_path = Path(config.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "dataset_path": str(output_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
        "dagger_episode_lengths": [int(ep["steps"]) for ep in new_episodes],
    }


def train_pick_policy(config: PickTrainConfig) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(config.seed)
    device = select_device(config.device)
    dataset = load_pick_dataset(config.dataset_path)
    env_config = dataset["env_config"]
    max_step = float(env_config.get("max_delta_ee", 0.045))
    model = _build_pick_model(config.model_kind, max_step=max_step).to(device)
    if config.initial_checkpoint_path is not None:
        checkpoint = torch.load(config.initial_checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["model_kind"] != config.model_kind:
            raise ValueError(
                f"Initial checkpoint kind {checkpoint['model_kind']} does not match requested {config.model_kind}."
            )
        model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    gripper_pos_weight = _gripper_pos_weight(dataset, split_name="train").to(device)
    start_time = time.perf_counter()

    history: list[dict[str, Any]] = []
    update_idx = 0
    for epoch in range(1, config.epochs + 1):
        train_ids = list(dataset["split"]["train"])
        rng = np.random.default_rng(config.seed + epoch)
        rng.shuffle(train_ids)
        for episode_id in train_ids:
            episode = dataset["episodes"][episode_id]
            losses, updates = _train_episode(
                model=model,
                model_kind=config.model_kind,
                episode=episode,
                optimizer=optimizer,
                device=device,
                bptt_steps=config.bptt_steps,
                gripper_pos_weight=gripper_pos_weight,
                motion_loss_weight=config.motion_loss_weight,
                direction_loss_weight=config.direction_loss_weight,
                magnitude_loss_weight=config.magnitude_loss_weight,
                gripper_loss_weight=config.gripper_loss_weight,
                precision_weighting=config.precision_weighting,
                near_distance=config.near_distance,
                medium_distance=config.medium_distance,
                near_weight=config.near_weight,
                medium_weight=config.medium_weight,
            )
            update_idx += updates
            if losses:
                history.append({"epoch": epoch, "episode": int(episode_id), **_mean_losses(losses)})
            if config.log_every and update_idx > 0 and update_idx % config.log_every == 0:
                recent = history[-min(len(history), config.log_every) :]
                print(
                    f"[pick:{config.model_kind}] update={update_idx} epoch={epoch} "
                    f"loss={_mean_key(recent, 'loss'):.5f} motion={_mean_key(recent, 'motion_loss'):.5f} "
                    f"gripper={_mean_key(recent, 'gripper_loss'):.5f}"
                )

        validation = evaluate_pick_offline(model, config.model_kind, dataset, "val", device, max_step)
        history.append({"epoch": epoch, "episode": -1, "validation": validation})
        print(
            f"[pick:{config.model_kind}] epoch={epoch} val_motion_l2={validation['motion_l2_error']:.4f} "
            f"val_grip_acc={validation['gripper_accuracy']:.3f} val_cos={validation['action_cosine']:.3f}"
        )

    offline = {
        split: evaluate_pick_offline(model, config.model_kind, dataset, split, device, max_step)
        for split in ("train", "val", "test")
    }
    output_dir = ensure_dir(config.output_dir)
    checkpoint_path = output_dir / f"{config.model_kind}.pt"
    checkpoint = {
        "model_kind": config.model_kind,
        "model_state": model.state_dict(),
        "config": asdict(config),
        "env_config": env_config,
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


@torch.no_grad()
def evaluate_pick_offline(
    model: nn.Module,
    model_kind: PickModelKind,
    dataset: dict[str, Any],
    split_name: str,
    device: torch.device,
    max_step: float,
) -> dict[str, float]:
    model.eval()
    motion_errors: list[float] = []
    motion_mse: list[float] = []
    cosines: list[float] = []
    grip_targets: list[int] = []
    grip_preds: list[int] = []
    grip_probs: list[float] = []
    magnitudes: list[float] = []
    for episode_id in dataset["split"][split_name]:
        episode = dataset["episodes"][episode_id]
        hidden = model.initial_hidden(device) if isinstance(model, PickRecurrentController) else None
        for graph, target_delta, target_gripper in zip(
            episode["graphs"],
            episode["delta_xz"],
            episode["gripper"],
            strict=True,
        ):
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
            grip_targets.append(target)
            grip_preds.append(pred)
            grip_probs.append(prob)
            magnitudes.append(float(pred_xz.norm().item()))
    return _offline_metrics(motion_errors, motion_mse, cosines, magnitudes, grip_targets, grip_preds, grip_probs)


@torch.no_grad()
def evaluate_pick_policy(config: PickEvalConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = select_device(config.device)
    model, checkpoint = load_pick_policy(config.checkpoint_path, device)
    model_kind: PickModelKind = checkpoint["model_kind"]
    env_config = _pick_env_config(max_steps=config.max_steps)
    env = MujocoManipulatorEnv(env_config, seed=config.seed)
    graph_builder = ManipulationGraphBuilder(task="pick", include_object=True)
    rng = np.random.default_rng(config.seed)
    results = []
    output_dir = ensure_dir(config.output_dir)
    success_rendered = False
    failure_rendered = False
    for episode_idx in range(config.episodes):
        object_x = float(rng.uniform(*config.object_x_range))
        render_dir = None
        render = bool(config.render and episode_idx == 0)
        if render:
            render_dir = output_dir / f"{model_kind}_{config.recurrent_mode}_frames_ep{episode_idx:03d}"
        result = run_closed_loop_pick_episode(
            model=model,
            model_kind=model_kind,
            env=env,
            graph_builder=graph_builder,
            device=device,
            object_x=object_x,
            recurrent_mode=config.recurrent_mode,
            render=render,
            render_dir=render_dir,
        )
        result["episode"] = episode_idx + 1
        results.append(result)
        if config.render and result["lift_success"] and not success_rendered:
            result["success_rerender"] = run_closed_loop_pick_episode(
                model=model,
                model_kind=model_kind,
                env=env,
                graph_builder=graph_builder,
                device=device,
                object_x=object_x,
                recurrent_mode=config.recurrent_mode,
                render=True,
                render_dir=output_dir / f"{model_kind}_{config.recurrent_mode}_success_frames",
            )["render_frames"]
            success_rendered = True
        if config.render and not result["lift_success"] and not failure_rendered:
            result["failure_rerender"] = run_closed_loop_pick_episode(
                model=model,
                model_kind=model_kind,
                env=env,
                graph_builder=graph_builder,
                device=device,
                object_x=object_x,
                recurrent_mode=config.recurrent_mode,
                render=True,
                render_dir=output_dir / f"{model_kind}_{config.recurrent_mode}_failure_frames",
            )["render_frames"]
            failure_rendered = True
    env.close()

    summary = summarize_pick_policy_results(results)
    payload = {
        "checkpoint_path": config.checkpoint_path,
        "model_kind": model_kind,
        "recurrent_mode": config.recurrent_mode,
        "device": str(device),
        "config": asdict(config),
        "env_config": asdict(env_config),
        "summary": summary,
        "episodes": results,
    }
    name = f"{model_kind}_{config.recurrent_mode}_eval.json"
    path = output_dir / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_closed_loop_pick(results, output_dir / f"{model_kind}_{config.recurrent_mode}_closed_loop.png")
    payload["output_path"] = str(path)
    return payload


@torch.no_grad()
def run_closed_loop_pick_episode(
    model: nn.Module,
    model_kind: PickModelKind,
    env: MujocoManipulatorEnv,
    graph_builder: ManipulationGraphBuilder,
    device: torch.device,
    object_x: float,
    recurrent_mode: RecurrentEvalMode = "normal",
    render: bool = False,
    render_dir: Path | None = None,
) -> dict[str, Any]:
    env.reset_pick_scene(object_x=object_x, randomize_robot=False)
    hidden = model.initial_hidden(device) if isinstance(model, PickRecurrentController) else None
    object_start = env.data.site_xpos[env.object_site_id].copy()
    frames: list[str] = []
    if render:
        render_dir = ensure_dir(render_dir or Path("artifacts/mujoco_learned_pick/eval/frames"))

    records: list[dict[str, Any]] = []
    approach_success = False
    alignment_success = False
    grasp_initiated = False
    grasp_success = False
    lift_success = False
    drop = False
    lift_hold = 0
    first_gripper_close_step: int | None = None
    first_grasp_step: int | None = None
    lift_step: int | None = None
    ik_failures = 0
    latencies_ms: list[float] = []

    for step_idx in range(env.config.max_steps):
        ee_before = env.data.site_xpos[env.ee_site_id].copy()
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
        grasp_obs = env.grasp_observation()
        object_pos = env.data.site_xpos[env.object_site_id].copy()
        ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        actual_ee_motion = ee_pos - ee_before
        object_to_ee = float(np.linalg.norm(object_pos - ee_pos))
        rel = ee_pos - object_pos
        object_height_delta = float(grasp_obs.object_height - object_start[2])
        associated = object_to_ee <= 0.13
        if object_to_ee <= 0.08:
            approach_success = True
        aligned_now = _alignment_success(rel, object_height_delta)
        alignment_success = alignment_success or aligned_now
        if gripper_value >= 0.5:
            grasp_initiated = True
            if first_gripper_close_step is None:
                first_gripper_close_step = step_idx
        if grasp_obs.left_contact and grasp_obs.right_contact:
            grasp_success = True
            if first_grasp_step is None:
                first_grasp_step = step_idx
        if object_height_delta >= env.config.lift_height and associated:
            lift_hold += 1
        else:
            lift_hold = 0
        if lift_hold >= env.config.lift_hold_steps:
            lift_success = True
            lift_step = step_idx
            break
        if grasp_success and object_height_delta < 0.02 and step_idx > (first_grasp_step or 0) + 70:
            drop = True

        records.append(
            {
                "step": step_idx,
                "ee_position": ee_pos.tolist(),
                "object_position": object_pos.tolist(),
                "delta_ee": delta_ee.tolist(),
                "predicted_action_magnitude": float(delta_ee.norm().item()),
                "actual_ee_motion": actual_ee_motion.tolist(),
                "actual_ee_motion_magnitude": float(np.linalg.norm(actual_ee_motion)),
                "gripper": gripper_value,
                "object_to_ee": object_to_ee,
                "relative_ee_object": rel.tolist(),
                "alignment_success": aligned_now,
                "object_height_delta": object_height_delta,
                "left_contact": grasp_obs.left_contact,
                "right_contact": grasp_obs.right_contact,
                "object_grasped": grasp_obs.object_grasped,
                "contact_force": grasp_obs.contact_force,
                "ik_success": bool(info["ik_success"]),
            }
        )
        if render and render_dir is not None and step_idx % 4 == 0:
            frame_path = render_dir / f"frame_{step_idx:03d}.png"
            plt.imsave(frame_path, env.render_frame())
            frames.append(str(frame_path))

    failure_reason = None if lift_success else _classify_pick_failure(
        approach_success=approach_success,
        grasp_initiated=grasp_initiated,
        grasp_success=grasp_success,
        first_gripper_close_step=first_gripper_close_step,
        first_grasp_step=first_grasp_step,
        records=records,
        ik_failures=ik_failures,
    )
    return {
        "object_x": object_x,
        "approach_success": approach_success,
        "alignment_success": alignment_success,
        "grasp_initiated": grasp_initiated,
        "grasp_success": grasp_success,
        "lift_success": lift_success,
        "drop": drop or bool(grasp_success and not lift_success),
        "failure_reason": failure_reason,
        "steps": len(records),
        "first_gripper_close_step": first_gripper_close_step,
        "first_grasp_step": first_grasp_step,
        "lift_step": lift_step,
        "ik_failures": ik_failures,
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "records": records,
        "render_frames": frames,
    }


def summarize_pick_policy_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = max(len(results), 1)
    failures = Counter(str(item["failure_reason"]) for item in results if not item["lift_success"])
    return {
        "approach_success_rate": sum(1.0 for item in results if item["approach_success"]) / count,
        "alignment_success_rate": sum(1.0 for item in results if item.get("alignment_success")) / count,
        "grasp_initiation_rate": sum(1.0 for item in results if item["grasp_initiated"]) / count,
        "grasp_success_rate": sum(1.0 for item in results if item["grasp_success"]) / count,
        "lift_success_rate": sum(1.0 for item in results if item["lift_success"]) / count,
        "drop_rate": sum(1.0 for item in results if item["drop"]) / count,
        "mean_steps": _mean([item["steps"] for item in results]),
        "mean_time_to_gripper_close": _mean([item["first_gripper_close_step"] for item in results]),
        "mean_time_to_grasp": _mean([item["first_grasp_step"] for item in results]),
        "mean_time_to_lift": _mean([item["lift_step"] for item in results]),
        "mean_latency_ms": _mean([item["mean_latency_ms"] for item in results]),
        "mean_close_distance": _mean(_close_distances(results)),
        "close_alignment_success_rate": _close_alignment_success_rate(results),
        "saturation_ratio": _saturation_ratio(results, near_only=False),
        "near_contact_saturation_ratio": _saturation_ratio(results, near_only=True),
        "mean_near_contact_action_magnitude": _mean(_near_contact_action_magnitudes(results)),
        "failure_counts": dict(failures),
    }


def _alignment_success(
    relative_ee_object: np.ndarray,
    object_height_delta: float,
    planar_threshold: float = 0.040,
    vertical_threshold: float = 0.060,
) -> bool:
    """Deterministic grasp-alignment proxy before physical contact/lift.

    The planar pick setup uses x/z motion, so alignment means the EE is close
    enough to the object center in controllable x/z coordinates while the
    object has not already been knocked meaningfully below the table.
    """

    return (
        abs(float(relative_ee_object[0])) <= planar_threshold
        and abs(float(relative_ee_object[2])) <= vertical_threshold
        and float(object_height_delta) >= -0.050
    )


def _close_distances(results: list[dict[str, Any]]) -> list[float]:
    distances: list[float] = []
    for item in results:
        close_step = item.get("first_gripper_close_step")
        if close_step is None:
            continue
        for row in item.get("records", []):
            if int(row["step"]) == int(close_step):
                distances.append(float(row["object_to_ee"]))
                break
    return distances


def _close_alignment_success_rate(results: list[dict[str, Any]], close_distance_threshold: float = 0.025) -> float:
    distances = _close_distances(results)
    return sum(1.0 for distance in distances if distance <= close_distance_threshold) / max(len(distances), 1)


def _saturation_ratio(
    results: list[dict[str, Any]],
    near_only: bool,
    max_step: float = 0.045,
    near_distance: float = 0.050,
    saturation_fraction: float = 0.90,
) -> float:
    total = 0
    saturated = 0
    for item in results:
        for row in item.get("records", []):
            if near_only and float(row["object_to_ee"]) > near_distance:
                continue
            total += 1
            if float(row["predicted_action_magnitude"]) >= saturation_fraction * max_step:
                saturated += 1
    return float(saturated / max(total, 1))


def _near_contact_action_magnitudes(
    results: list[dict[str, Any]],
    near_distance: float = 0.050,
) -> list[float]:
    return [
        float(row["predicted_action_magnitude"])
        for item in results
        for row in item.get("records", [])
        if float(row["object_to_ee"]) <= near_distance
    ]


def load_pick_dataset(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_pick_policy(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    max_step = float(checkpoint.get("env_config", {}).get("max_delta_ee", 0.045))
    model = _build_pick_model(checkpoint["model_kind"], max_step=max_step).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def plot_closed_loop_pick(results: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    representative = results[0] if results else {"records": []}
    records = representative["records"]
    if not records:
        path.write_text("no records", encoding="utf-8")
        return path
    xs = np.arange(len(records))
    object_height = np.asarray([row["object_height_delta"] for row in records], dtype=float)
    gripper = np.asarray([row["gripper"] for row in records], dtype=float)
    object_to_ee = np.asarray([row["object_to_ee"] for row in records], dtype=float)
    both_contact = np.asarray([row["left_contact"] and row["right_contact"] for row in records], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    axes[0].plot(xs, object_height)
    axes[0].axhline(0.10, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("object lift")
    axes[1].plot(xs, gripper, label="gripper")
    axes[1].plot(xs, both_contact, label="both contact")
    axes[1].legend(loc="best")
    axes[1].set_ylabel("grasp")
    axes[2].plot(xs, object_to_ee)
    axes[2].axhline(0.08, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("object to EE")
    axes[2].set_xlabel("step")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _collect_pick_demo_episode(
    env: MujocoManipulatorEnv,
    graph_builder: ManipulationGraphBuilder,
    expert_config: ScriptedPickConfig,
    object_x: float,
    episode_id: int,
) -> dict[str, Any]:
    env.reset_pick_scene(object_x=object_x, randomize_robot=expert_config.randomize_robot)
    object_start = env.data.site_xpos[env.object_site_id].copy()
    phase = "PRE_GRASP"
    phases: list[str] = []
    graphs: list[GraphData] = []
    delta_xz: list[torch.Tensor] = []
    grippers: list[float] = []
    close_counter = 0
    verify_counter = 0
    lift_target: np.ndarray | None = None
    approach_success = False
    grasp_success = False
    lift_success = False
    hold_count = 0
    grasp_step: int | None = None
    lift_step: int | None = None
    ik_failures = 0

    for step_idx in range(expert_config.max_steps):
        object_position = env.data.site_xpos[env.object_site_id].copy()
        grasp_target = object_position + np.asarray(expert_config.grasp_offset, dtype=np.float64)
        approach_target = grasp_target + np.array([0.0, 0.0, expert_config.approach_height], dtype=np.float64)
        if phase == "PRE_GRASP":
            target = approach_target
            gripper = 0.0
            if _ee_distance(env, target) < 0.035:
                approach_success = True
                phase = "ALIGN"
        elif phase == "ALIGN":
            target = grasp_target
            gripper = 0.0
            if _ee_distance(env, target) < 0.035:
                phase = "CLOSE"
        elif phase == "CLOSE":
            target = grasp_target
            gripper = 1.0
            close_counter += 1
            if close_counter >= expert_config.close_steps:
                phase = "VERIFY"
        elif phase == "VERIFY":
            target = grasp_target
            gripper = 1.0
            verify_counter += 1
            grasp_obs = env.grasp_observation()
            if grasp_obs.left_contact and grasp_obs.right_contact:
                grasp_success = True
                grasp_step = step_idx
                lift_target = env.data.site_xpos[env.ee_site_id].copy() + np.array(
                    [0.0, 0.0, expert_config.lift_height],
                    dtype=np.float64,
                )
                phase = "LIFT"
            elif verify_counter >= expert_config.verify_steps:
                break
        elif phase == "LIFT":
            target = lift_target if lift_target is not None else grasp_target
            gripper = 1.0
        else:
            target = grasp_target
            gripper = 1.0

        graph = graph_builder.build(env.observe())
        current = env.data.site_xpos[env.ee_site_id].copy()
        delta = clamp_delta(torch.tensor(target - current, dtype=torch.float32), env.config.max_delta_ee)
        graphs.append(graph)
        delta_xz.append(torch.stack([delta[0], delta[2]]))
        grippers.append(float(gripper))
        phases.append(phase)

        _, _, _, info = env.step_delta_ee(delta, gripper=gripper)
        if not bool(info["ik_success"]):
            ik_failures += 1
        grasp_obs = env.grasp_observation()
        object_height_delta = grasp_obs.object_height - float(object_start[2])
        object_to_ee = float(np.linalg.norm(env.data.site_xpos[env.object_site_id] - env.data.site_xpos[env.ee_site_id]))
        if phase == "LIFT":
            if object_height_delta >= expert_config.lift_success_height and object_to_ee <= expert_config.grasp_association_distance:
                hold_count += 1
            else:
                hold_count = 0
            if hold_count >= expert_config.lift_hold_steps:
                lift_success = True
                lift_step = step_idx
                break

    return {
        "episode_id": episode_id,
        "object_x": float(object_x),
        "graphs": graphs,
        "delta_xz": torch.stack(delta_xz, dim=0) if delta_xz else torch.empty(0, 2),
        "gripper": torch.tensor(grippers, dtype=torch.float32),
        "phase_labels_diagnostics_only": phases,
        "steps": len(graphs),
        "approach_success": approach_success,
        "grasp_success": grasp_success,
        "lift_success": lift_success,
        "grasp_step": grasp_step,
        "lift_step": lift_step,
        "ik_failures": ik_failures,
    }


@torch.no_grad()
def _collect_dagger_episode(
    policy: nn.Module,
    model_kind: PickModelKind,
    env: MujocoManipulatorEnv,
    graph_builder: ManipulationGraphBuilder,
    device: torch.device,
    object_x: float,
    episode_id: int,
) -> dict[str, Any]:
    env.reset_pick_scene(object_x=object_x, randomize_robot=False)
    hidden = policy.initial_hidden(device) if isinstance(policy, PickRecurrentController) else None
    graphs: list[GraphData] = []
    delta_xz: list[torch.Tensor] = []
    grippers: list[float] = []
    diagnostic_modes: list[str] = []
    lift_success = False
    approach_success = False
    grasp_success = False
    object_start = env.data.site_xpos[env.object_site_id].copy()
    lift_hold = 0
    grasp_step: int | None = None
    lift_step: int | None = None
    ik_failures = 0

    for step_idx in range(env.config.max_steps):
        graph = graph_builder.build(env.observe())
        expert_delta, expert_gripper, mode = _reactive_pick_expert_action(env, object_start)
        graphs.append(graph)
        delta_xz.append(torch.stack([expert_delta[0], expert_delta[2]]))
        grippers.append(float(expert_gripper))
        diagnostic_modes.append(mode)

        policy_graph = graph.to(device)
        if isinstance(policy, PickRecurrentController):
            action, hidden = policy(policy_graph, hidden)
        elif isinstance(policy, PickFeedForwardController):
            action = policy(policy_graph)
        else:
            raise ValueError(f"Unsupported model kind: {model_kind}")
        _, _, _, info = env.step_delta_ee(action.delta_ee.detach().cpu(), gripper=float(action.gripper.detach().cpu().item()))
        if not bool(info["ik_success"]):
            ik_failures += 1
        grasp = env.grasp_observation()
        object_to_ee = float(np.linalg.norm(env.data.site_xpos[env.object_site_id] - env.data.site_xpos[env.ee_site_id]))
        object_height_delta = float(grasp.object_height - object_start[2])
        if object_to_ee <= 0.08:
            approach_success = True
        if grasp.left_contact and grasp.right_contact:
            grasp_success = True
            if grasp_step is None:
                grasp_step = step_idx
        if object_height_delta >= env.config.lift_height and object_to_ee <= 0.13:
            lift_hold += 1
        else:
            lift_hold = 0
        if lift_hold >= env.config.lift_hold_steps:
            lift_success = True
            lift_step = step_idx
            break
        if object_height_delta < -0.08:
            break

    return {
        "episode_id": episode_id,
        "object_x": float(object_x),
        "graphs": graphs,
        "delta_xz": torch.stack(delta_xz, dim=0) if delta_xz else torch.empty(0, 2),
        "gripper": torch.tensor(grippers, dtype=torch.float32),
        "phase_labels_diagnostics_only": diagnostic_modes,
        "steps": len(graphs),
        "approach_success": approach_success,
        "grasp_success": grasp_success,
        "lift_success": lift_success,
        "grasp_step": grasp_step,
        "lift_step": lift_step,
        "ik_failures": ik_failures,
        "dagger_policy_visited": True,
    }


def _reactive_pick_expert_action(env: MujocoManipulatorEnv, object_start: np.ndarray) -> tuple[torch.Tensor, float, str]:
    object_position = env.data.site_xpos[env.object_site_id].copy()
    ee_position = env.data.site_xpos[env.ee_site_id].copy()
    grasp = env.grasp_observation()
    object_to_ee = float(np.linalg.norm(object_position - ee_position))
    object_height_delta = float(grasp.object_height - object_start[2])
    grasp_target = object_position + np.array([0.0, 0.0, -0.008], dtype=np.float64)
    approach_target = grasp_target + np.array([0.0, 0.0, 0.10], dtype=np.float64)
    if object_height_delta > 0.04 or (grasp.left_contact and grasp.right_contact):
        target = ee_position + np.array([0.0, 0.0, 0.18], dtype=np.float64)
        gripper = 1.0
        mode = "reactive_lift"
    elif object_to_ee > 0.12 and ee_position[2] > object_position[2] + 0.06:
        target = approach_target
        gripper = 0.0
        mode = "reactive_approach"
    elif _ee_distance(env, grasp_target) > 0.030:
        target = grasp_target
        gripper = 0.0
        mode = "reactive_align"
    else:
        target = grasp_target
        gripper = 1.0
        mode = "reactive_close"
    delta = clamp_delta(torch.tensor(target - ee_position, dtype=torch.float32), env.config.max_delta_ee)
    return delta, gripper, mode


def _train_episode(
    model: nn.Module,
    model_kind: PickModelKind,
    episode: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bptt_steps: int,
    gripper_pos_weight: torch.Tensor,
    motion_loss_weight: float,
    direction_loss_weight: float,
    magnitude_loss_weight: float,
    gripper_loss_weight: float,
    precision_weighting: bool,
    near_distance: float,
    medium_distance: float,
    near_weight: float,
    medium_weight: float,
) -> tuple[list[dict[str, float]], int]:
    model.train()
    graphs = episode["graphs"]
    deltas = episode["delta_xz"]
    grippers = episode["gripper"]
    hidden = model.initial_hidden(device) if isinstance(model, PickRecurrentController) else None
    losses: list[dict[str, float]] = []
    updates = 0
    for start_idx in range(0, len(graphs), bptt_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        total_motion = torch.tensor(0.0, device=device)
        total_gripper = torch.tensor(0.0, device=device)
        chunk_graphs = graphs[start_idx : start_idx + bptt_steps]
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
            sample_weight = _precision_weight(
                graph,
                enabled=precision_weighting,
                near_distance=near_distance,
                medium_distance=medium_distance,
                near_weight=near_weight,
                medium_weight=medium_weight,
            )
            if model.action_head_type == "direction_magnitude":
                motion_loss = _direction_magnitude_loss(
                    raw=raw,
                    pred_delta=pred_delta,
                    target_delta=target_delta,
                    max_step=model.max_step,
                    direction_loss_weight=direction_loss_weight,
                    magnitude_loss_weight=magnitude_loss_weight,
                )
            else:
                motion_loss = torch.nn.functional.smooth_l1_loss(pred_delta, target_delta)
            motion_loss = motion_loss * sample_weight
            gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                gripper_logit(raw, model.action_head_type).reshape(()),
                target_gripper.reshape(()),
                pos_weight=gripper_pos_weight,
            )
            total_motion = total_motion + motion_loss
            total_gripper = total_gripper + gripper_loss
            total_loss = total_loss + motion_loss_weight * motion_loss + gripper_loss_weight * gripper_loss
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


def _offline_metrics(
    motion_errors: list[float],
    motion_mse: list[float],
    cosines: list[float],
    magnitudes: list[float],
    grip_targets: list[int],
    grip_preds: list[int],
    grip_probs: list[float],
) -> dict[str, float]:
    tp = sum(1 for p, t in zip(grip_preds, grip_targets, strict=True) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(grip_preds, grip_targets, strict=True) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(grip_preds, grip_targets, strict=True) if p == 0 and t == 1)
    correct = sum(1 for p, t in zip(grip_preds, grip_targets, strict=True) if p == t)
    count = max(len(grip_targets), 1)
    return {
        "motion_l2_error": _mean(motion_errors),
        "motion_mse": _mean(motion_mse),
        "action_cosine": _mean(cosines),
        "predicted_action_magnitude": _mean(magnitudes),
        "gripper_accuracy": correct / count,
        "gripper_precision": tp / max(tp + fp, 1),
        "gripper_recall": tp / max(tp + fn, 1),
        "target_closed_fraction": sum(grip_targets) / count,
        "predicted_closed_fraction": sum(grip_preds) / count,
        "mean_gripper_probability": _mean(grip_probs),
        "num_steps": float(count),
    }


def _direction_magnitude_loss(
    raw: torch.Tensor,
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    max_step: float,
    direction_loss_weight: float,
    magnitude_loss_weight: float,
    zero_direction_threshold: float = 1e-5,
) -> torch.Tensor:
    target_magnitude = target_delta.norm()
    pred_magnitude_fraction = torch.sigmoid(raw[2])
    target_magnitude_fraction = (target_magnitude / float(max_step)).clamp(0.0, 1.0)
    magnitude_loss = torch.nn.functional.smooth_l1_loss(pred_magnitude_fraction, target_magnitude_fraction)
    if float(target_magnitude.detach().cpu().item()) <= zero_direction_threshold:
        direction_loss = torch.zeros((), device=raw.device, dtype=raw.dtype)
    else:
        pred_direction = raw[0:2] / raw[0:2].norm().clamp_min(1e-8)
        target_direction = target_delta / target_magnitude.clamp_min(1e-8)
        direction_loss = 1.0 - torch.dot(pred_direction, target_direction).clamp(-1.0, 1.0)
    return direction_loss_weight * direction_loss + magnitude_loss_weight * magnitude_loss


def _precision_weight(
    graph: GraphData,
    enabled: bool,
    near_distance: float,
    medium_distance: float,
    near_weight: float,
    medium_weight: float,
) -> torch.Tensor:
    if not enabled:
        return torch.tensor(1.0, device=graph.node_features.device, dtype=graph.node_features.dtype)
    distance = (
        graph.node_features[graph.target_node_index, 0:3] - graph.node_features[graph.ee_node_index, 0:3]
    ).norm()
    if float(distance.detach().cpu().item()) <= near_distance:
        value = near_weight
    elif float(distance.detach().cpu().item()) <= medium_distance:
        value = medium_weight
    else:
        value = 1.0
    return torch.tensor(value, device=graph.node_features.device, dtype=graph.node_features.dtype)


def _demo_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    episodes = payload["episodes"]
    lengths = [int(ep["steps"]) for ep in episodes]
    grippers = torch.cat([ep["gripper"] for ep in episodes], dim=0)
    deltas = torch.cat([ep["delta_xz"] for ep in episodes], dim=0)
    phases = Counter(phase for ep in episodes for phase in ep["phase_labels_diagnostics_only"])
    return {
        "num_episodes": len(episodes),
        "split": payload["split"],
        "episode_length": {
            "mean": _mean(lengths),
            "min": min(lengths),
            "max": max(lengths),
        },
        "object_x_range": payload["config"]["object_x_range"],
        "gripper_distribution": {
            "open_fraction": float((grippers < 0.5).float().mean().item()),
            "closed_fraction": float((grippers >= 0.5).float().mean().item()),
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


def _episode_split(num_episodes: int, seed: int) -> dict[str, list[int]]:
    indices = np.arange(num_episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_end = int(0.8 * num_episodes)
    val_end = int(0.9 * num_episodes)
    return {
        "train": sorted(int(idx) for idx in indices[:train_end]),
        "val": sorted(int(idx) for idx in indices[train_end:val_end]),
        "test": sorted(int(idx) for idx in indices[val_end:]),
    }


def _build_pick_model(model_kind: PickModelKind, max_step: float) -> nn.Module:
    if model_kind == "graph_recurrent":
        return PickRecurrentController(max_step=max_step)
    if model_kind == "graph_feedforward":
        return PickFeedForwardController(max_step=max_step)
    if model_kind == "graph_recurrent_dir_mag":
        return PickRecurrentController(max_step=max_step, action_head_type="direction_magnitude")
    if model_kind == "graph_feedforward_dir_mag":
        return PickFeedForwardController(max_step=max_step, action_head_type="direction_magnitude")
    raise ValueError(f"Unknown pick model kind: {model_kind}")


def _pick_env_config(max_steps: int) -> MujocoReachConfig:
    return MujocoReachConfig(
        max_steps=max_steps,
        control_substeps=40,
        max_delta_ee=0.045,
        pick_scene=True,
        kinematic_joint_control=False,
    )


def _gripper_pos_weight(dataset: dict[str, Any], split_name: str) -> torch.Tensor:
    values = torch.cat([dataset["episodes"][idx]["gripper"] for idx in dataset["split"][split_name]], dim=0)
    positives = float((values >= 0.5).sum().item())
    negatives = float((values < 0.5).sum().item())
    return torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32)


def _mean_losses(losses: list[dict[str, float]]) -> dict[str, float]:
    return {
        "loss": _mean([item["loss"] for item in losses]),
        "motion_loss": _mean([item["motion_loss"] for item in losses]),
        "gripper_loss": _mean([item["gripper_loss"] for item in losses]),
    }


def _mean_key(records: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in records if key in item]
    return _mean(values)


def _mean(values: list[int | float | None]) -> float:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(numeric) / max(len(numeric), 1)


def _classify_pick_failure(
    approach_success: bool,
    grasp_initiated: bool,
    grasp_success: bool,
    first_gripper_close_step: int | None,
    first_grasp_step: int | None,
    records: list[dict[str, Any]],
    ik_failures: int,
) -> str:
    if ik_failures > max(len(records) // 2, 1):
        return "IK/control failure"
    if not approach_success:
        return "approach error"
    if not grasp_initiated:
        return "failure to close"
    min_distance = min((row["object_to_ee"] for row in records), default=1e9)
    if first_gripper_close_step is not None and min_distance > 0.09:
        return "premature gripper close"
    if first_gripper_close_step is not None and first_gripper_close_step > 120:
        return "late gripper close"
    if not grasp_success:
        return "alignment error"
    if first_grasp_step is not None and max((row["object_height_delta"] for row in records), default=0.0) < 0.05:
        return "weak/unstable grasp"
    return "lift direction error"


def _ee_distance(env: MujocoManipulatorEnv, target: np.ndarray) -> float:
    return float(np.linalg.norm(env.data.site_xpos[env.ee_site_id] - target))
