from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from mujoco_dynamic_2x2_eval import (
    EFFECT_METRICS,
    SUMMARY_METRICS,
    evaluate_dynamic_checkpoint,
    make_dynamic_episode_plan,
)
from mujoco_static_2x2_convergence import (
    MODEL_KINDS,
    MODEL_ORDER,
    _plot_aggregate_curves,
    _plot_run_curves,
    _save_checkpoint,
    _tail_text,
    _utc_now,
    _write_epoch_metrics,
    _write_run_state,
    completed_run,
    run_child_and_wait,
)
from vla_gnn_recurrent.graph.flat_features import flat_state_schema_from_graph
from vla_gnn_recurrent.graph.manipulation_graph_builder import ManipulationGraphBuilder
from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv
from vla_gnn_recurrent.training.pick_bc import (
    PickModelKind,
    _build_pick_model,
    _gripper_pos_weight,
    _mean_key,
    _pick_env_config,
    load_pick_dataset,
)
from vla_gnn_recurrent.training.pick_place_bc import (
    PickPlaceExpertConfig,
    PickPlaceTrainConfig,
    _distance_xz,
    classify_pick_place_failure,
    evaluate_pick_place_offline,
    pick_place_demo_metadata,
    summarize_pick_place_results,
    train_pick_place_episode,
)
from vla_gnn_recurrent.utils import DevicePreference, clamp_delta, ensure_dir, select_device, set_seed


@dataclass
class DynamicPickPlaceDemoConfig:
    episodes: int = 500
    seed: int = 4301
    output_path: str = "artifacts/dynamic_2x2_convergence/demos/dynamic_pick_place_demos.pt"
    metadata_path: str = "artifacts/dynamic_2x2_convergence/demos/dynamic_pick_place_demos_metadata.json"
    object_x_range: tuple[float, float] = (-0.0048, -0.0001)
    destination_x_range: tuple[float, float] = (-0.060, -0.053)
    min_separation: float = 0.012
    max_steps: int = 420
    max_attempts_multiplier: int = 4
    perturb_delay_steps: int = 8
    perturbed_destination_x_range: tuple[float, float] = (-0.046, -0.026)
    min_perturb_delta: float = 0.012
    render: bool = False


def dynamic_validation_score(validation_summary: dict[str, Any], offline_validation: dict[str, Any]) -> tuple[float, ...]:
    """Lexicographic dynamic checkpoint score. Higher is better for every component."""

    return (
        float(validation_summary.get("placement_success_rate", 0.0)),
        float(validation_summary.get("successful_placement_release_rate", 0.0)),
        float(validation_summary.get("valid_release_rate", 0.0)),
        float(validation_summary.get("dynamic_recovery_rate", 0.0)),
        float(validation_summary.get("transport_success_rate", 0.0)),
        float(validation_summary.get("lift_success_rate", 0.0)),
        float(validation_summary.get("grasp_success_rate", 0.0)),
        -float(offline_validation.get("motion_l2_error", 0.0)),
    )


def generate_dynamic_demonstrations(config: DynamicPickPlaceDemoConfig) -> dict[str, Any]:
    set_seed(config.seed)
    env_config = _pick_env_config(max_steps=config.max_steps)
    env = MujocoManipulatorEnv(env_config, seed=config.seed)
    graph_builder = ManipulationGraphBuilder(task="pick_and_place", include_object=True)
    expert_config = PickPlaceExpertConfig(max_steps=config.max_steps)
    episodes: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = config.episodes * config.max_attempts_multiplier
    plan = make_dynamic_episode_plan(
        episodes=max_attempts,
        seed=config.seed,
        max_steps=config.max_steps,
        object_x_range=config.object_x_range,
        destination_x_range=config.destination_x_range,
        min_separation=config.min_separation,
        dynamic_destination_x_range=config.perturbed_destination_x_range,
        min_perturb_delta=config.min_perturb_delta,
    )

    try:
        progress = tqdm(total=config.episodes, desc="dynamic demos", unit="ep")
        for item in plan:
            if len(episodes) >= config.episodes:
                break
            attempts += 1
            episode = run_dynamic_scripted_pick_place_episode(
                env=env,
                graph_builder=graph_builder,
                expert_config=expert_config,
                object_x=float(item["object_x"]),
                destination_x=float(item["destination_x"]),
                perturbed_destination_x=float(item["perturbed_destination_x"]),
                perturb_delay_steps=config.perturb_delay_steps,
                episode_id=len(episodes),
                render=config.render and len(episodes) == 0,
                render_dir=Path(config.output_path).parent / "expert_frames",
            )
            episode["dynamic_demo_success"] = _dynamic_demo_success(episode, expert_config)
            if episode["dynamic_demo_success"]:
                episodes.append(episode)
                progress.update(1)
                progress.set_postfix(steps=episode["steps"], recover=episode.get("dynamic_recovery_steps"))
        progress.close()
    finally:
        env.close()

    if len(episodes) < config.episodes:
        raise RuntimeError(
            f"Only collected {len(episodes)} successful dynamic pick-place demos after {attempts} attempts."
        )

    split = _episode_split(len(episodes), seed=config.seed)
    payload = {
        "task": "dynamic_pick_and_place",
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
    metadata["dynamic"] = {
        "perturb_delay_steps": config.perturb_delay_steps,
        "perturbed_destination_x_range": list(config.perturbed_destination_x_range),
        "min_perturb_delta": config.min_perturb_delta,
        "attempts": attempts,
        "destination_perturbation_rate": sum(1.0 for ep in episodes if ep["destination_perturbed"]) / len(episodes),
        "strict_placement_success_rate": sum(1.0 for ep in episodes if ep["placement_success"]) / len(episodes),
        "demo_success_rate": sum(1.0 for ep in episodes if ep.get("dynamic_demo_success")) / len(episodes),
        "mean_dynamic_recovery_steps": _mean(
            [float(ep["dynamic_recovery_steps"]) for ep in episodes if ep["dynamic_recovery_steps"] is not None]
        ),
    }
    metadata_path = Path(config.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"dataset_path": str(output_path), "metadata_path": str(metadata_path), "metadata": metadata}


def _dynamic_demo_success(episode: dict[str, Any], expert_config: PickPlaceExpertConfig) -> bool:
    records = episode.get("records", [])
    final_error = float(records[-1]["placement_error"]) if records else 1e9
    return bool(
        episode.get("destination_perturbed")
        and episode.get("transport_success")
        and episode.get("release_success")
        and final_error <= expert_config.placement_threshold
    )


def _episode_split(count: int, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    indices = list(range(count))
    rng.shuffle(indices)
    train_end = int(round(count * 0.8))
    val_end = int(round(count * 0.9))
    return {
        "train": sorted(indices[:train_end]),
        "val": sorted(indices[train_end:val_end]),
        "test": sorted(indices[val_end:]),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_dynamic_scripted_pick_place_episode(
    env: MujocoManipulatorEnv,
    graph_builder: ManipulationGraphBuilder,
    expert_config: PickPlaceExpertConfig,
    object_x: float,
    destination_x: float,
    perturbed_destination_x: float,
    perturb_delay_steps: int,
    episode_id: int,
    render: bool = False,
    render_dir: Path | None = None,
) -> dict[str, Any]:
    env.reset_pick_place_scene(object_x=object_x, destination_x=destination_x, randomize_robot=expert_config.randomize_robot)
    if render:
        import matplotlib.pyplot as plt

        render_dir = ensure_dir(render_dir or Path("artifacts/dynamic_2x2_convergence/expert_frames"))
    object_start = env.data.site_xpos[env.object_site_id].copy()
    original_destination = env.target_position_np().copy()
    destination = original_destination.copy()
    perturbed_destination: np.ndarray | None = None
    perturb_step: int | None = None
    first_recovery_step: int | None = None
    phase = "APPROACH_OBJECT"
    phase_order = [phase]
    close_counter = 0
    verify_counter = 0
    release_counter = 0
    stable_counter = 0
    demo_stable_counter = 0
    lift_target: np.ndarray | None = None
    records: list[dict[str, Any]] = []
    graphs: list[Any] = []
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
    demo_placement_step: int | None = None
    ik_failures = 0
    failure_reason: str | None = None

    for step_idx in range(expert_config.max_steps):
        if (
            perturb_step is None
            and lift_success
            and lift_step is not None
            and step_idx >= lift_step + perturb_delay_steps
            and phase not in {"RELEASE", "STABILIZE", "SUCCESS", "FAILURE"}
        ):
            perturbed_destination = env.pick_surface_object_center(perturbed_destination_x)
            env.move_target(perturbed_destination)
            destination = env.target_position_np().copy()
            perturb_step = step_idx
            transport_success = False
            stable_counter = 0
            phase_order.append("DESTINATION_PERTURBATION")
            if phase in {"ALIGN_DESTINATION"}:
                phase = "TRANSPORT"
                phase_order.append(phase)

        object_position = env.data.site_xpos[env.object_site_id].copy()
        ee_position = env.data.site_xpos[env.ee_site_id].copy()
        destination = env.target_position_np().copy()
        grasp_target = object_position + np.asarray(expert_config.grasp_offset, dtype=np.float64)
        approach_target = grasp_target + np.array([0.0, 0.0, expert_config.approach_height], dtype=np.float64)
        destination_above = destination + np.array([0.0, 0.0, expert_config.transport_height], dtype=np.float64)
        place_target = destination + np.asarray(expert_config.grasp_offset, dtype=np.float64)
        stabilize_target = place_target + np.array([0.060, 0.0, 0.0], dtype=np.float64)

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
            target = stabilize_target
            gripper = 0.0
        else:
            target = ee_position
            gripper = 0.0

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
        destination = env.target_position_np().copy()
        object_to_ee = float(np.linalg.norm(object_position - ee_position))
        object_height_delta = float(grasp.object_height - object_start[2])
        placement_error = _distance_xz(object_position, destination)
        if perturb_step is not None and first_recovery_step is None and placement_error <= expert_config.placement_threshold:
            first_recovery_step = step_idx
        placing_near_goal = phase in {"ALIGN_DESTINATION", "RELEASE", "STABILIZE"} and (
            placement_error <= expert_config.placement_threshold * 1.5
        )
        if grasp_success and not release_success and not placing_near_goal and object_height_delta < 0.015 and step_idx > (grasp_step or 0) + 20:
            drop = True
            failure_reason = "drop_during_transport"
            phase = "FAILURE"
            phase_order.append(phase)
        if phase == "STABILIZE":
            if (
                placement_error <= expert_config.placement_threshold
                and object_height_delta < 0.040
                and (not grasp.object_grasped or gripper <= 0.5)
            ):
                stable_counter += 1
            else:
                stable_counter = 0
            if stable_counter >= expert_config.release_hold_steps:
                placement_success = True
                placement_step = step_idx
                phase = "SUCCESS"
                phase_order.append(phase)
            if placement_error <= expert_config.placement_threshold and object_height_delta < 0.045:
                demo_stable_counter += 1
            else:
                demo_stable_counter = 0
            if demo_stable_counter >= expert_config.release_hold_steps and not placement_success:
                demo_placement_step = step_idx
                phase = "DEMO_SUCCESS"
                phase_order.append(phase)

        records.append(
            {
                "step": step_idx,
                "phase": phase,
                "ee_position": ee_position.tolist(),
                "object_position": object_position.tolist(),
                "destination_position": destination.tolist(),
                "original_destination_position": original_destination.tolist(),
                "perturbed_destination_position": perturbed_destination.tolist() if perturbed_destination is not None else None,
                "destination_perturbed": perturb_step is not None,
                "perturb_step": perturb_step,
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
            import matplotlib.pyplot as plt

            frame_path = render_dir / f"frame_{step_idx:03d}.png"
            plt.imsave(frame_path, env.render_frame())
            frames.append(str(frame_path))
        if phase in {"SUCCESS", "DEMO_SUCCESS", "FAILURE"}:
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
        "perturbed_destination_x": float(perturbed_destination_x),
        "destination_perturbed": perturb_step is not None,
        "perturb_step": perturb_step,
        "dynamic_recovery_step": first_recovery_step,
        "dynamic_recovery_steps": None if perturb_step is None or first_recovery_step is None else first_recovery_step - perturb_step,
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
        "valid_release_success": bool(grasp_success and lift_success and transport_success and release_success),
        "gripper_open_event": release_step is not None,
        "placement_success": placement_success,
        "drop": drop,
        "failure_reason": None if placement_success else failure_reason,
        "phase_order": phase_order,
        "grasp_step": grasp_step,
        "lift_step": lift_step,
        "release_step": release_step,
        "valid_release_step": release_step if grasp_success and lift_success and transport_success and release_success else None,
        "placement_step": placement_step,
        "demo_placement_step": demo_placement_step,
        "ik_failures": ik_failures,
        "records": records,
        "render_frames": frames,
    }


def run_all(args: argparse.Namespace) -> None:
    output_root = ensure_dir(Path(args.output_root))
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    if args.generate_dataset_if_missing and not Path(args.dataset_path).exists():
        generate_dynamic_demonstrations(
            DynamicPickPlaceDemoConfig(
                episodes=args.demo_episodes,
                seed=args.demo_seed,
                output_path=args.dataset_path,
                metadata_path=str(Path(args.dataset_path).with_name(Path(args.dataset_path).stem + "_metadata.json")),
                max_steps=args.max_steps,
                perturb_delay_steps=args.perturb_delay_steps,
                perturbed_destination_x_range=(args.perturbed_destination_x_min, args.perturbed_destination_x_max),
                min_perturb_delta=args.min_perturb_delta,
            )
        )
    run_order = [(model, seed) for seed in args.train_seeds for model in args.models]
    with tqdm(total=len(run_order), desc="dynamic 2x2 convergence", unit="run", disable=not args.progress) as progress:
        for model_label, train_seed in run_order:
            run_dir = output_root / "runs" / f"{model_label}_seed{train_seed}"
            progress.set_postfix_str(f"{model_label} seed={train_seed}")
            if completed_run(run_dir) and not args.force:
                print(f"[skip] {model_label} seed={train_seed} already complete at {run_dir}")
                progress.update(1)
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "run-one",
                "--model",
                model_label,
                "--train-seed",
                str(train_seed),
                "--dataset-path",
                args.dataset_path,
                "--output-root",
                str(output_root),
                "--min-epochs",
                str(args.min_epochs),
                "--max-epochs",
                str(args.max_epochs),
                "--patience",
                str(args.patience),
                "--val-episodes",
                str(args.val_episodes),
                "--val-seed",
                str(args.val_seed),
                "--test-episodes",
                str(args.test_episodes),
                "--test-seed",
                str(args.test_seed),
                "--max-steps",
                str(args.max_steps),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
                "--bptt-steps",
                str(args.bptt_steps),
                "--motion-loss-weight",
                str(args.motion_loss_weight),
                "--direction-loss-weight",
                str(args.direction_loss_weight),
                "--magnitude-loss-weight",
                str(args.magnitude_loss_weight),
                "--gripper-loss-weight",
                str(args.gripper_loss_weight),
                "--near-distance",
                str(args.near_distance),
                "--medium-distance",
                str(args.medium_distance),
                "--near-weight",
                str(args.near_weight),
                "--medium-weight",
                str(args.medium_weight),
                "--device",
                args.device,
                "--eval-device",
                args.eval_device,
                "--eval-workers",
                str(args.eval_workers),
                "--perturb-delay-steps",
                str(args.perturb_delay_steps),
                "--perturbed-destination-x-min",
                str(args.perturbed_destination_x_min),
                "--perturbed-destination-x-max",
                str(args.perturbed_destination_x_max),
                "--min-perturb-delta",
                str(args.min_perturb_delta),
            ]
            if not args.precision_weighting:
                command.append("--no-precision-weighting")
            if not args.progress:
                command.append("--no-progress")
            if args.force:
                command.append("--force")
            stdout_log = output_root / "logs" / f"{model_label}_seed{train_seed}.stdout.log"
            stderr_log = output_root / "logs" / f"{model_label}_seed{train_seed}.stderr.log"
            metadata_path = run_dir / "child_process_metadata.json"
            print(f"[run] {model_label} seed={train_seed}")
            exit_code = run_child_and_wait(command, stdout_log, stderr_log, metadata_path, stream_output=args.stream_child_output)
            if exit_code != 0:
                raise RuntimeError(
                    f"{model_label} seed={train_seed} failed with exit code {exit_code}. "
                    f"stderr tail:\n{_tail_text(stderr_log, 80)}"
                )
            progress.update(1)
    summarize_experiment(output_root)


def run_one(args: argparse.Namespace) -> None:
    if args.model not in MODEL_KINDS:
        raise ValueError(f"Unknown model label: {args.model}")
    model_kind = MODEL_KINDS[args.model]
    run_dir = ensure_dir(Path(args.output_root) / "runs" / f"{args.model}_seed{args.train_seed}")
    if completed_run(run_dir) and not args.force:
        print(f"[skip] completed run at {run_dir}")
        return

    set_seed(args.train_seed)
    device = select_device(args.device)
    dataset = load_pick_dataset(args.dataset_path)
    max_step = float(dataset["env_config"].get("max_delta_ee", 0.045))
    flat_schema = flat_state_schema_from_graph(dataset["episodes"][0]["graphs"][0])
    model = _build_pick_model(model_kind, max_step=max_step, flat_input_dim=flat_schema.input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    gripper_pos_weight = _gripper_pos_weight(dataset, "train").to(device)
    train_config = PickPlaceTrainConfig(
        dataset_path=args.dataset_path,
        model_kind=model_kind,
        output_dir=str(run_dir / "checkpoints"),
        epochs=1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bptt_steps=args.bptt_steps,
        motion_loss_weight=args.motion_loss_weight,
        direction_loss_weight=args.direction_loss_weight,
        magnitude_loss_weight=args.magnitude_loss_weight,
        gripper_loss_weight=args.gripper_loss_weight,
        precision_weighting=args.precision_weighting,
        near_distance=args.near_distance,
        medium_distance=args.medium_distance,
        near_weight=args.near_weight,
        medium_weight=args.medium_weight,
        seed=args.train_seed,
        device=args.device,
        log_every=0,
    )
    config_payload = {
        "protocol": {
            "task": "dynamic_pick_and_place",
            "min_epochs": args.min_epochs,
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.patience,
            "checkpoint_selection": [
                "placement_success_rate",
                "successful_placement_release_rate",
                "valid_release_rate",
                "dynamic_recovery_rate",
                "transport_success_rate",
                "lift_success_rate",
                "grasp_success_rate",
                "offline_validation_motion_l2_error",
            ],
            "perturb_delay_steps": args.perturb_delay_steps,
            "perturbed_destination_x_range": [args.perturbed_destination_x_min, args.perturbed_destination_x_max],
            "min_perturb_delta": args.min_perturb_delta,
        },
        "model_label": args.model,
        "model_kind": model_kind,
        "train_seed": args.train_seed,
        "validation_seed": args.val_seed,
        "test_seed": args.test_seed,
        "dataset_path": args.dataset_path,
        "device": str(device),
        "eval_device": args.eval_device,
        "eval_workers": args.eval_workers,
        "train_config": asdict(train_config),
        "flat_schema": flat_schema.to_dict(),
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameter_count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    best_score: tuple[float, ...] | None = None
    best_epoch = 0
    best_validation_metrics: dict[str, Any] = {}
    epochs_since_improvement = 0
    optimizer_steps = 0
    processed_transitions = 0
    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    best_checkpoint_path = checkpoints_dir / "best.pt"

    epoch_iter = tqdm(range(1, args.max_epochs + 1), desc=f"{args.model} seed={args.train_seed}", unit="epoch", disable=not args.progress)
    for epoch in epoch_iter:
        epoch_start = time.perf_counter()
        train_ids = list(dataset["split"]["train"])
        rng = np.random.default_rng(args.train_seed + epoch)
        rng.shuffle(train_ids)
        epoch_losses: list[dict[str, float]] = []
        epoch_updates = 0
        epoch_transitions = 0
        train_iter = tqdm(train_ids, desc=f"train epoch {epoch}", unit="ep", leave=False, disable=not args.progress)
        for episode_id in train_iter:
            episode = dataset["episodes"][episode_id]
            losses, updates = train_pick_place_episode(
                model=model,
                model_kind=model_kind,
                episode=episode,
                optimizer=optimizer,
                device=device,
                config=train_config,
                gripper_pos_weight=gripper_pos_weight,
            )
            epoch_losses.extend(losses)
            epoch_updates += updates
            epoch_transitions += int(len(episode["graphs"]))
            if losses:
                train_iter.set_postfix(loss=f"{_mean_key(epoch_losses, 'loss'):.4f}")
        optimizer_steps += epoch_updates
        processed_transitions += epoch_transitions
        train_loss = _mean_key(epoch_losses, "loss")
        offline_validation = evaluate_pick_place_offline(model, dataset, "val", device, max_step)
        epoch_checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
        _save_checkpoint(
            path=epoch_checkpoint_path,
            model=model,
            optimizer=optimizer,
            model_kind=model_kind,
            epoch=epoch,
            config=config_payload,
            dataset=dataset,
            flat_input_dim=flat_schema.input_dim if model_kind.startswith("flat_") else None,
            flat_schema=flat_schema.to_dict(),
            parameter_count=parameter_count,
            trainable_parameter_count=trainable_parameter_count,
            offline_validation=offline_validation,
        )
        validation = evaluate_dynamic_checkpoint(
            checkpoint_path=epoch_checkpoint_path,
            episodes=args.val_episodes,
            seed=args.val_seed,
            max_steps=args.max_steps,
            output_dir=run_dir / "validation",
            eval_device=args.eval_device,
            workers=args.eval_workers,
            perturb_delay_steps=args.perturb_delay_steps,
            perturbed_destination_x_range=(args.perturbed_destination_x_min, args.perturbed_destination_x_max),
            min_perturb_delta=args.min_perturb_delta,
            progress=args.progress,
        )
        score = dynamic_validation_score(validation["summary"], offline_validation)
        improved = best_score is None or score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_validation_metrics = {
                "offline": offline_validation,
                "closed_loop_dynamic": validation["summary"],
                "selection_score": list(score),
            }
            shutil.copyfile(epoch_checkpoint_path, best_checkpoint_path)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        row = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "epoch_optimizer_steps": epoch_updates,
            "processed_transitions": processed_transitions,
            "epoch_processed_transitions": epoch_transitions,
            "training_loss": train_loss,
            "training_motion_loss": _mean_key(epoch_losses, "motion_loss"),
            "training_gripper_loss": _mean_key(epoch_losses, "gripper_loss"),
            "validation_loss": float(offline_validation["motion_l2_error"]),
            "offline_delta_ee_l2": float(offline_validation["motion_l2_error"]),
            "offline_action_cosine": float(offline_validation["action_cosine"]),
            "offline_predicted_action_magnitude": float(offline_validation["predicted_action_magnitude"]),
            "offline_gripper_accuracy": float(offline_validation["gripper_accuracy"]),
            "offline_gripper_precision": float(offline_validation["gripper_precision"]),
            "offline_gripper_recall": float(offline_validation["gripper_recall"]),
            "offline_release_transition_accuracy": float(offline_validation["release_transition_accuracy"]),
            "validation_grasp_success_rate": float(validation["summary"]["grasp_success_rate"]),
            "validation_lift_success_rate": float(validation["summary"]["lift_success_rate"]),
            "validation_transport_success_rate": float(validation["summary"]["transport_success_rate"]),
            "validation_valid_release_rate": float(validation["summary"]["valid_release_rate"]),
            "validation_successful_placement_release_rate": float(validation["summary"]["successful_placement_release_rate"]),
            "validation_placement_success_rate": float(validation["summary"]["placement_success_rate"]),
            "validation_dynamic_recovery_rate": float(validation["summary"].get("dynamic_recovery_rate", 0.0)),
            "validation_mean_dynamic_recovery_steps": validation["summary"].get("mean_dynamic_recovery_steps"),
            "validation_mean_episode_length": float(validation["summary"]["mean_steps"]),
            "mean_inference_latency_ms": float(validation["summary"]["mean_latency_ms"] or 0.0),
            "parameter_count": parameter_count,
            "checkpoint_path": str(epoch_checkpoint_path),
            "training_runtime": float(time.perf_counter() - epoch_start),
            "best_epoch": best_epoch,
            "epochs_since_improvement": epochs_since_improvement,
            "improved": improved,
        }
        rows.append(row)
        epoch_iter.set_postfix(
            loss=f"{train_loss:.4f}",
            val_l2=f"{offline_validation['motion_l2_error']:.4f}",
            place=f"{validation['summary']['placement_success_rate']:.2f}",
            recover=f"{validation['summary'].get('dynamic_recovery_rate', 0.0):.2f}",
            best=best_epoch,
            stale=epochs_since_improvement,
        )
        _write_epoch_metrics(run_dir / "epoch_metrics.csv", rows)
        (run_dir / "epoch_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        _write_run_state(run_dir, False, best_checkpoint_path, best_epoch, best_validation_metrics, epochs_since_improvement, rows)
        if epoch >= args.min_epochs and epochs_since_improvement >= args.patience:
            convergence_status = "early_stopped"
            break
    else:
        convergence_status = "censored_at_max_epoch" if best_epoch == args.max_epochs else "stopped_at_max_epoch_no_recent_improvement"

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_test = evaluate_dynamic_checkpoint(
        checkpoint_path=best_checkpoint_path,
        episodes=args.test_episodes,
        seed=args.test_seed,
        max_steps=args.max_steps,
        output_dir=run_dir / "test",
        eval_device=args.eval_device,
        workers=args.eval_workers,
        perturb_delay_steps=args.perturb_delay_steps,
        perturbed_destination_x_range=(args.perturbed_destination_x_min, args.perturbed_destination_x_max),
        min_perturb_delta=args.min_perturb_delta,
        progress=args.progress,
    )
    offline = {split: evaluate_pick_place_offline(model, dataset, split, device, max_step) for split in ("train", "val", "test")}
    result = {
        "completed": True,
        "model_label": args.model,
        "model_kind": model_kind,
        "train_seed": args.train_seed,
        "validation_seed": args.val_seed,
        "test_seed": args.test_seed,
        "dataset_path": args.dataset_path,
        "device": str(device),
        "eval_device": args.eval_device,
        "eval_workers": args.eval_workers,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_validation_metrics": best_validation_metrics,
        "epochs_since_improvement": epochs_since_improvement,
        "hit_max_epoch": bool(len(rows) >= args.max_epochs),
        "convergence_status": convergence_status,
        "optimizer_steps": optimizer_steps,
        "processed_transitions": processed_transitions,
        "runtime_seconds": float(time.perf_counter() - start_time),
        "offline": offline,
        "final_test": final_test,
        "epoch_metrics_path": str(run_dir / "epoch_metrics.csv"),
    }
    (run_dir / "run_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _plot_run_curves(rows, run_dir / "convergence_curves.png")
    _write_run_state(run_dir, True, best_checkpoint_path, best_epoch, best_validation_metrics, epochs_since_improvement, rows)


def summarize_experiment(output_root: Path) -> dict[str, Any]:
    runs = _load_run_results(output_root)
    aggregate = _aggregate_final_metrics(runs)
    effects = _factorial_effects(runs)
    payload = {
        "output_root": str(output_root),
        "model_order": list(MODEL_ORDER),
        "runs": [
            {
                "model_label": run["model_label"],
                "train_seed": run["train_seed"],
                "best_epoch": run["best_epoch"],
                "convergence_status": run["convergence_status"],
                "hit_max_epoch": run["hit_max_epoch"],
                "parameter_count": run["parameter_count"],
                "final_test_summary": run["final_test"]["summary"],
            }
            for run in runs
        ],
        "aggregate": aggregate,
        "effects": effects,
    }
    summary_dir = ensure_dir(output_root / "summary")
    (summary_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_final_metrics_csv(runs, summary_dir / "final_metrics_by_seed.csv")
    (summary_dir / "factorial_effects.json").write_text(json.dumps(effects, indent=2), encoding="utf-8")
    _plot_aggregate_curves(output_root, runs, summary_dir / "aggregate_convergence_curves.png")
    print(json.dumps({"summary_path": str(summary_dir / "summary.json"), "completed_runs": len(runs)}, indent=2))
    return payload


def _load_run_results(output_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted((output_root / "runs").glob("*_seed*/run_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("completed"):
            results.append(payload)
    return results


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0.0}
    return {
        "mean": float(sum(values) / len(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "n": float(len(values)),
    }


def _aggregate_final_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODEL_ORDER}
    for run in runs:
        by_model.setdefault(run["model_label"], []).append(run)
    aggregate: dict[str, Any] = {}
    for model, model_runs in by_model.items():
        aggregate[model] = {
            "seeds": [int(run["train_seed"]) for run in model_runs],
            "parameter_count": _mean_std([float(run["parameter_count"]) for run in model_runs]),
            "best_epoch": _mean_std([float(run["best_epoch"]) for run in model_runs]),
            "metrics": {
                metric: _mean_std([float(run["final_test"]["summary"].get(metric, 0.0) or 0.0) for run in model_runs])
                for metric in SUMMARY_METRICS
            },
            "convergence_status": {str(run["train_seed"]): run["convergence_status"] for run in model_runs},
        }
    return aggregate


def _factorial_effects(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed_model: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        by_seed_model.setdefault(int(run["train_seed"]), {})[run["model_label"]] = run
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for seed, seed_runs in by_seed_model.items():
        if not all(model in seed_runs for model in MODEL_ORDER):
            continue
        per_seed[str(seed)] = {}
        for metric in EFFECT_METRICS:
            flat_ff = float(seed_runs["flat_ff"]["final_test"]["summary"].get(metric, 0.0) or 0.0)
            flat_gru = float(seed_runs["flat_gru"]["final_test"]["summary"].get(metric, 0.0) or 0.0)
            graph_ff = float(seed_runs["graph_ff"]["final_test"]["summary"].get(metric, 0.0) or 0.0)
            graph_gru = float(seed_runs["graph_gru"]["final_test"]["summary"].get(metric, 0.0) or 0.0)
            per_seed[str(seed)][metric] = {
                "graph_effect_under_ff": graph_ff - flat_ff,
                "graph_effect_under_gru": graph_gru - flat_gru,
                "recurrence_effect_on_flat": flat_gru - flat_ff,
                "recurrence_effect_on_graph": graph_gru - graph_ff,
                "graph_recurrence_interaction": graph_gru - graph_ff - flat_gru + flat_ff,
            }
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for metric in EFFECT_METRICS:
        aggregate[metric] = {}
        for effect_name in (
            "graph_effect_under_ff",
            "graph_effect_under_gru",
            "recurrence_effect_on_flat",
            "recurrence_effect_on_graph",
            "graph_recurrence_interaction",
        ):
            aggregate[metric][effect_name] = _mean_std(
                [float(seed_effects[metric][effect_name]) for seed_effects in per_seed.values() if metric in seed_effects]
            )
    return {"per_seed": per_seed, "aggregate": aggregate}


def _write_final_metrics_csv(runs: list[dict[str, Any]], path: Path) -> None:
    fields = ["model_label", "train_seed", "best_epoch", "convergence_status", "parameter_count", *SUMMARY_METRICS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {
                "model_label": run["model_label"],
                "train_seed": run["train_seed"],
                "best_epoch": run["best_epoch"],
                "convergence_status": run["convergence_status"],
                "parameter_count": run["parameter_count"],
            }
            row.update({metric: run["final_test"]["summary"].get(metric, 0.0) for metric in SUMMARY_METRICS})
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convergence-aware dynamic Flat/Graph x FF/GRU ablation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("generate-demos")
    demo_parser.add_argument("--episodes", type=int, default=500)
    demo_parser.add_argument("--seed", type=int, default=4301)
    demo_parser.add_argument("--output-path", default="artifacts/dynamic_2x2_convergence/demos/dynamic_pick_place_demos.pt")
    demo_parser.add_argument("--metadata-path", default="artifacts/dynamic_2x2_convergence/demos/dynamic_pick_place_demos_metadata.json")
    demo_parser.add_argument("--max-steps", type=int, default=420)
    demo_parser.add_argument("--max-attempts-multiplier", type=int, default=4)
    demo_parser.add_argument("--perturb-delay-steps", type=int, default=8)
    demo_parser.add_argument("--perturbed-destination-x-min", type=float, default=-0.046)
    demo_parser.add_argument("--perturbed-destination-x-max", type=float, default=-0.026)
    demo_parser.add_argument("--min-perturb-delta", type=float, default=0.012)
    demo_parser.add_argument("--render", action="store_true")

    for command in ("run-all", "run-one"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--dataset-path", default="artifacts/dynamic_2x2_convergence/demos/dynamic_pick_place_demos.pt")
        sub.add_argument("--output-root", default="artifacts/dynamic_2x2_convergence")
        sub.add_argument("--min-epochs", type=int, default=5)
        sub.add_argument("--max-epochs", type=int, default=30)
        sub.add_argument("--patience", type=int, default=5)
        sub.add_argument("--val-episodes", type=int, default=20)
        sub.add_argument("--val-seed", type=int, default=4601)
        sub.add_argument("--test-episodes", type=int, default=100)
        sub.add_argument("--test-seed", type=int, default=4701)
        sub.add_argument("--max-steps", type=int, default=420)
        sub.add_argument("--learning-rate", type=float, default=3e-4)
        sub.add_argument("--weight-decay", type=float, default=1e-4)
        sub.add_argument("--bptt-steps", type=int, default=32)
        sub.add_argument("--motion-loss-weight", type=float, default=10.0)
        sub.add_argument("--direction-loss-weight", type=float, default=1.0)
        sub.add_argument("--magnitude-loss-weight", type=float, default=1.0)
        sub.add_argument("--gripper-loss-weight", type=float, default=1.0)
        sub.add_argument("--no-precision-weighting", dest="precision_weighting", action="store_false")
        sub.set_defaults(precision_weighting=True)
        sub.add_argument("--near-distance", type=float, default=0.05)
        sub.add_argument("--medium-distance", type=float, default=0.10)
        sub.add_argument("--near-weight", type=float, default=4.0)
        sub.add_argument("--medium-weight", type=float, default=2.0)
        sub.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        sub.add_argument("--eval-device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
        sub.add_argument("--eval-workers", type=int, default=1)
        sub.add_argument("--perturb-delay-steps", type=int, default=8)
        sub.add_argument("--perturbed-destination-x-min", type=float, default=-0.046)
        sub.add_argument("--perturbed-destination-x-max", type=float, default=-0.026)
        sub.add_argument("--min-perturb-delta", type=float, default=0.012)
        sub.add_argument("--force", action="store_true")
        sub.add_argument("--no-progress", dest="progress", action="store_false")
        sub.set_defaults(progress=True)
    run_all_parser = subparsers.choices["run-all"]
    run_all_parser.add_argument("--models", nargs="+", choices=list(MODEL_ORDER), default=list(MODEL_ORDER))
    run_all_parser.add_argument("--train-seeds", nargs="+", type=int, default=[1601, 1602, 1603])
    run_all_parser.add_argument("--stream-child-output", action="store_true")
    run_all_parser.add_argument("--generate-dataset-if-missing", action="store_true")
    run_all_parser.add_argument("--demo-episodes", type=int, default=500)
    run_all_parser.add_argument("--demo-seed", type=int, default=4301)
    run_one_parser = subparsers.choices["run-one"]
    run_one_parser.add_argument("--model", required=True, choices=list(MODEL_ORDER))
    run_one_parser.add_argument("--train-seed", type=int, required=True)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output-root", default="artifacts/dynamic_2x2_convergence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate-demos":
        result = generate_dynamic_demonstrations(
            DynamicPickPlaceDemoConfig(
                episodes=args.episodes,
                seed=args.seed,
                output_path=args.output_path,
                metadata_path=args.metadata_path,
                max_steps=args.max_steps,
                max_attempts_multiplier=args.max_attempts_multiplier,
                perturb_delay_steps=args.perturb_delay_steps,
                perturbed_destination_x_range=(args.perturbed_destination_x_min, args.perturbed_destination_x_max),
                min_perturb_delta=args.min_perturb_delta,
                render=args.render,
            )
        )
        print(json.dumps(result, indent=2))
    elif args.command == "run-all":
        run_all(args)
    elif args.command == "run-one":
        run_one(args)
    elif args.command == "summarize":
        summarize_experiment(Path(args.output_root))
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
