from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from mujoco_static_2x2_convergence import MODEL_KINDS, MODEL_ORDER, chunk_sequence, completed_run, run_child_and_wait
from vla_gnn_recurrent.graph.manipulation_graph_builder import ManipulationGraphBuilder
from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv
from vla_gnn_recurrent.training.pick_bc import PickModelKind, RecurrentEvalMode, _pick_env_config, load_pick_policy
from vla_gnn_recurrent.training.pick_place_bc import (
    PickPlaceEvalConfig,
    PickPlaceExpertConfig,
    _decoded_pick_action,
    _distance_xz,
    _is_recurrent_pick_model,
    _model_observation,
    _sample_pick_place_positions,
    classify_pick_place_failure,
    summarize_pick_place_results,
)
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


SUMMARY_METRICS = (
    "grasp_success_rate",
    "lift_success_rate",
    "transport_success_rate",
    "valid_release_rate",
    "placement_success_rate",
    "drop_rate",
    "dynamic_perturbation_rate",
    "dynamic_recovery_rate",
    "mean_dynamic_recovery_steps",
    "mean_final_placement_error",
    "mean_steps",
    "mean_latency_ms",
)
EFFECT_METRICS = (
    "placement_success_rate",
    "valid_release_rate",
    "transport_success_rate",
    "dynamic_recovery_rate",
)


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0.0}
    return {
        "mean": float(sum(values) / len(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "n": float(len(values)),
    }


def checkpoint_path_for(static_root: Path, model_label: str, seed: int) -> Path:
    return static_root / "runs" / f"{model_label}_seed{seed}" / "checkpoints" / "best.pt"


def static_run_dir_for(static_root: Path, model_label: str, seed: int) -> Path:
    return static_root / "runs" / f"{model_label}_seed{seed}"


def make_dynamic_episode_plan(
    episodes: int,
    seed: int,
    max_steps: int,
    object_x_range: tuple[float, float] = (-0.0048, -0.0001),
    destination_x_range: tuple[float, float] = (-0.060, -0.053),
    min_separation: float = 0.012,
    dynamic_destination_x_range: tuple[float, float] = (0.052, 0.068),
    min_perturb_delta: float = 0.090,
) -> list[dict[str, Any]]:
    if dynamic_destination_x_range[0] >= dynamic_destination_x_range[1]:
        raise ValueError(f"Invalid perturbed destination x range: {dynamic_destination_x_range}")
    if min_perturb_delta < 0.0:
        raise ValueError(f"min_perturb_delta must be non-negative, got {min_perturb_delta}")
    rng = np.random.default_rng(seed)
    eval_config = PickPlaceEvalConfig(
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        render=False,
        object_x_range=object_x_range,
        destination_x_range=destination_x_range,
        min_separation=min_separation,
    )
    plan: list[dict[str, Any]] = []
    for episode_idx in range(episodes):
        object_x, destination_x = _sample_pick_place_positions(rng, eval_config)
        for _ in range(100):
            perturbed_destination_x = float(rng.uniform(*dynamic_destination_x_range))
            if abs(perturbed_destination_x - destination_x) >= min_perturb_delta:
                break
        else:
            perturbed_destination_x = float(dynamic_destination_x_range[1])
        plan.append(
            {
                "episode_idx": episode_idx,
                "episode": episode_idx + 1,
                "episode_seed": int(seed * 100_000 + episode_idx),
                "object_x": float(object_x),
                "destination_x": float(destination_x),
                "perturbed_destination_x": float(perturbed_destination_x),
            }
        )
    return plan


@torch.no_grad()
def run_dynamic_pick_place_episode(
    model: nn.Module,
    model_kind: PickModelKind,
    env: MujocoManipulatorEnv,
    graph_builder: ManipulationGraphBuilder,
    device: torch.device,
    object_x: float,
    destination_x: float,
    perturbed_destination_x: float,
    perturb_delay_steps: int = 8,
    recurrent_mode: RecurrentEvalMode = "normal",
) -> dict[str, Any]:
    env.reset_pick_place_scene(object_x=object_x, destination_x=destination_x, randomize_robot=False)
    hidden = model.initial_hidden(device) if _is_recurrent_pick_model(model) else None
    object_start = env.data.site_xpos[env.object_site_id].copy()
    original_destination = env.target_position_np().copy()
    destination = original_destination.copy()
    perturbed_destination: np.ndarray | None = None
    perturb_step: int | None = None
    first_recovery_step: int | None = None
    records: list[dict[str, Any]] = []
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
    first_lift_step: int | None = None
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
        if (
            perturb_step is None
            and lift_success
            and first_lift_step is not None
            and step_idx >= first_lift_step + perturb_delay_steps
        ):
            perturbed_destination = env.pick_surface_object_center(perturbed_destination_x)
            env.move_target(perturbed_destination)
            destination = env.target_position_np().copy()
            perturb_step = step_idx

        pre_grasp = env.grasp_observation()
        pre_robot = env.robot_observation()
        pre_object_pos = env.data.site_xpos[env.object_site_id].copy()
        pre_ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        pre_object_velocity = pre_grasp.object_velocity.detach().cpu().numpy()
        pre_ee_velocity = pre_robot.ee_velocity.detach().cpu().numpy()
        destination = env.target_position_np().copy()
        pre_step_placement_error = _distance_xz(pre_object_pos, destination)
        graph = graph_builder.build(env.observe())
        observation = _model_observation(model, graph, device)
        start = time.perf_counter()
        if _is_recurrent_pick_model(model) and recurrent_mode == "step_reset":
            hidden = model.initial_hidden(device)
        action, hidden = _decoded_pick_action(model, observation, hidden)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        delta_ee = action.delta_ee.detach().cpu()
        gripper_value = float(action.gripper.detach().cpu().item())
        opening_now = bool(first_gripper_close_step is not None and previous_gripper_value >= 0.5 and gripper_value < 0.5)
        _, _, _, info = env.step_delta_ee(delta_ee, gripper=gripper_value)
        if not bool(info["ik_success"]):
            ik_failures += 1
        grasp = env.grasp_observation()
        object_pos = env.data.site_xpos[env.object_site_id].copy()
        ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        destination = env.target_position_np().copy()
        actual_ee_motion = ee_pos - pre_ee_pos
        object_to_ee = float(np.linalg.norm(object_pos - ee_pos))
        object_height_delta = float(grasp.object_height - object_start[2])
        placement_error = _distance_xz(object_pos, destination)
        if perturb_step is not None and first_recovery_step is None and placement_error <= expert_config.placement_threshold:
            first_recovery_step = step_idx
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
            if first_lift_step is None:
                first_lift_step = step_idx
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
            and (not grasp.object_grasped or gripper_value < 0.5)
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
                "original_destination_position": original_destination.tolist(),
                "perturbed_destination_position": perturbed_destination.tolist() if perturbed_destination is not None else None,
                "destination_perturbed": perturb_step is not None,
                "perturb_step": perturb_step,
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
        "perturbed_destination_x": float(perturbed_destination_x),
        "destination_perturbed": perturb_step is not None,
        "perturb_step": perturb_step,
        "dynamic_recovery_step": first_recovery_step,
        "dynamic_recovery_steps": None if perturb_step is None or first_recovery_step is None else first_recovery_step - perturb_step,
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
        "first_lift_step": first_lift_step,
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
        "render_frames": [],
    }


def summarize_dynamic_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_pick_place_results(results)
    count = max(len(results), 1)
    recoveries = [item.get("dynamic_recovery_steps") for item in results if item.get("dynamic_recovery_steps") is not None]
    summary.update(
        {
            "dynamic_perturbation_rate": sum(1.0 for item in results if item.get("destination_perturbed")) / count,
            "dynamic_recovery_rate": sum(1.0 for item in results if item.get("dynamic_recovery_steps") is not None) / count,
            "mean_dynamic_recovery_steps": _mean(recoveries),
        }
    )
    return summary


def _dynamic_eval_worker(payload: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    device = select_device(payload["eval_device"])
    try:
        torch.set_num_threads(int(payload.get("torch_num_threads", 1)))
    except RuntimeError:
        pass
    model, checkpoint = load_pick_policy(payload["checkpoint_path"], device)
    model_kind: PickModelKind = checkpoint["model_kind"]
    graph_builder = ManipulationGraphBuilder(task="pick_and_place", include_object=True)
    env = MujocoManipulatorEnv(_pick_env_config(max_steps=int(payload["max_steps"])), seed=int(payload["worker_seed"]))
    try:
        with torch.no_grad():
            for job in payload["jobs"]:
                try:
                    set_seed(int(job["episode_seed"]))
                    result = run_dynamic_pick_place_episode(
                        model=model,
                        model_kind=model_kind,
                        env=env,
                        graph_builder=graph_builder,
                        device=device,
                        object_x=float(job["object_x"]),
                        destination_x=float(job["destination_x"]),
                        perturbed_destination_x=float(job["perturbed_destination_x"]),
                        perturb_delay_steps=int(payload["perturb_delay_steps"]),
                    )
                    result["episode"] = int(job["episode"])
                    result["episode_seed"] = int(job["episode_seed"])
                    result["episode_idx"] = int(job["episode_idx"])
                    results.append(result)
                except Exception as exc:  # pragma: no cover - worker failure path.
                    errors.append(
                        {
                            "episode": int(job["episode"]),
                            "episode_idx": int(job["episode_idx"]),
                            "episode_seed": int(job["episode_seed"]),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
    finally:
        env.close()
    return {"results": results, "errors": errors}


def evaluate_dynamic_checkpoint(
    checkpoint_path: Path,
    episodes: int,
    seed: int,
    max_steps: int,
    output_dir: Path,
    eval_device: DevicePreference,
    workers: int,
    perturb_delay_steps: int,
    perturbed_destination_x_range: tuple[float, float],
    min_perturb_delta: float,
    progress: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = make_dynamic_episode_plan(
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        dynamic_destination_x_range=perturbed_destination_x_range,
        min_perturb_delta=min_perturb_delta,
    )
    worker_count = max(1, min(int(workers), len(plan) if plan else 1))
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if worker_count <= 1:
        payload = {
            "checkpoint_path": str(checkpoint_path),
            "eval_device": eval_device,
            "max_steps": max_steps,
            "worker_seed": seed,
            "jobs": plan,
            "perturb_delay_steps": perturb_delay_steps,
            "torch_num_threads": 1,
        }
        worker_result = _dynamic_eval_worker(payload)
        results.extend(worker_result["results"])
        errors.extend(worker_result["errors"])
    else:
        chunks = chunk_sequence(plan, worker_count)
        context = mp.get_context("spawn")
        payloads = [
            {
                "checkpoint_path": str(checkpoint_path),
                "eval_device": eval_device,
                "max_steps": max_steps,
                "worker_seed": int(seed + worker_idx),
                "jobs": chunk,
                "perturb_delay_steps": perturb_delay_steps,
                "torch_num_threads": 1,
            }
            for worker_idx, chunk in enumerate(chunks)
        ]
        with ProcessPoolExecutor(max_workers=len(payloads), mp_context=context) as executor:
            futures = [executor.submit(_dynamic_eval_worker, payload) for payload in payloads]
            future_iter = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"dynamic eval {checkpoint_path.parent.parent.name}",
                unit="worker",
                leave=False,
                disable=not progress,
            )
            for future in future_iter:
                payload = future.result()
                results.extend(payload["results"])
                errors.extend(payload["errors"])
                if results:
                    future_iter.set_postfix(
                        episodes=len(results),
                        place=f"{sum(1 for item in results if item['placement_success']) / len(results):.2f}",
                    )
    results.sort(key=lambda item: int(item["episode_idx"]))
    if errors:
        error_path = output_dir / "dynamic_eval_errors.json"
        error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        raise RuntimeError(f"Dynamic evaluation failed for {len(errors)} episodes. See {error_path}")
    summary = summarize_dynamic_results(results)
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "seed": seed,
        "episodes": results,
        "episode_plan": plan,
        "summary": summary,
        "eval_device": eval_device,
        "eval_workers": worker_count,
        "perturb_delay_steps": perturb_delay_steps,
        "perturbed_destination_x_range": list(perturbed_destination_x_range),
        "min_perturb_delta": min_perturb_delta,
    }
    (output_dir / "dynamic_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_eval_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_dynamic_2x2(output_root: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for path in sorted((output_root / "runs").glob("*_seed*/dynamic_eval.json")):
        payload = _load_eval_summary(path)
        run_name = path.parent.name
        model_label, seed_text = run_name.rsplit("_seed", 1)
        runs.append({"model_label": model_label, "seed": int(seed_text), **payload})
    aggregate: dict[str, Any] = {}
    for model in MODEL_ORDER:
        model_runs = [run for run in runs if run["model_label"] == model]
        aggregate[model] = {
            "seeds": [run["seed"] for run in model_runs],
            "metrics": {
                metric: _mean_std([float(run["summary"].get(metric, 0.0) or 0.0) for run in model_runs])
                for metric in SUMMARY_METRICS
            },
        }
    effects = compute_effects(runs)
    payload = {"runs": [{"model_label": run["model_label"], "seed": run["seed"], "summary": run["summary"]} for run in runs], "aggregate": aggregate, "effects": effects}
    summary_dir = ensure_dir(output_root / "summary")
    (summary_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_final_metrics_csv(runs, summary_dir / "final_metrics_by_seed.csv")
    return payload


def compute_effects(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        by_seed.setdefault(int(run["seed"]), {})[run["model_label"]] = run
    per_seed: dict[str, Any] = {}
    for seed, seed_runs in by_seed.items():
        if not all(model in seed_runs for model in MODEL_ORDER):
            continue
        per_seed[str(seed)] = {}
        for metric in EFFECT_METRICS:
            flat_ff = float(seed_runs["flat_ff"]["summary"].get(metric, 0.0) or 0.0)
            flat_gru = float(seed_runs["flat_gru"]["summary"].get(metric, 0.0) or 0.0)
            graph_ff = float(seed_runs["graph_ff"]["summary"].get(metric, 0.0) or 0.0)
            graph_gru = float(seed_runs["graph_gru"]["summary"].get(metric, 0.0) or 0.0)
            per_seed[str(seed)][metric] = {
                "graph_effect_under_ff": graph_ff - flat_ff,
                "graph_effect_under_gru": graph_gru - flat_gru,
                "recurrence_effect_on_flat": flat_gru - flat_ff,
                "recurrence_effect_on_graph": graph_gru - graph_ff,
                "graph_recurrence_interaction": graph_gru - graph_ff - flat_gru + flat_ff,
            }
    aggregate: dict[str, Any] = {}
    for metric in EFFECT_METRICS:
        aggregate[metric] = {}
        for effect in (
            "graph_effect_under_ff",
            "graph_effect_under_gru",
            "recurrence_effect_on_flat",
            "recurrence_effect_on_graph",
            "graph_recurrence_interaction",
        ):
            aggregate[metric][effect] = _mean_std(
                [
                    float(seed_effects[metric][effect])
                    for seed_effects in per_seed.values()
                    if metric in seed_effects
                ]
            )
    return {"per_seed": per_seed, "aggregate": aggregate}


def write_final_metrics_csv(runs: list[dict[str, Any]], path: Path) -> None:
    fields = ["model_label", "seed", *SUMMARY_METRICS]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {"model_label": run["model_label"], "seed": run["seed"]}
            row.update({metric: run["summary"].get(metric, 0.0) for metric in SUMMARY_METRICS})
            writer.writerow(row)


def run_one(args: argparse.Namespace) -> None:
    static_root = Path(args.static_root)
    run_dir = static_run_dir_for(static_root, args.model, args.train_seed)
    if not args.allow_incomplete_static and not completed_run(run_dir):
        raise RuntimeError(
            f"Static convergence run is not complete for {args.model} seed={args.train_seed}: {run_dir}. "
            "Use --allow-incomplete-static only for quick provisional diagnostics."
        )
    checkpoint = checkpoint_path_for(static_root, args.model, args.train_seed)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing static best checkpoint: {checkpoint}")
    output_dir = ensure_dir(Path(args.output_root) / "runs" / f"{args.model}_seed{args.train_seed}")
    result_path = output_dir / "dynamic_eval.json"
    if result_path.exists() and not args.force:
        print(f"[skip] {args.model} seed={args.train_seed} already evaluated")
        return
    payload = evaluate_dynamic_checkpoint(
        checkpoint_path=checkpoint,
        episodes=args.episodes,
        seed=args.eval_seed,
        max_steps=args.max_steps,
        output_dir=output_dir,
        eval_device=args.eval_device,
        workers=args.eval_workers,
        perturb_delay_steps=args.perturb_delay_steps,
        perturbed_destination_x_range=(args.perturbed_destination_x_min, args.perturbed_destination_x_max),
        min_perturb_delta=args.min_perturb_delta,
        progress=args.progress,
    )
    print(json.dumps({"output": str(result_path), "summary": payload["summary"]}, indent=2))


def run_all(args: argparse.Namespace) -> None:
    output_root = ensure_dir(args.output_root)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    run_order = [(model, seed) for seed in args.train_seeds for model in args.models]
    with tqdm(total=len(run_order), desc="dynamic 2x2 eval", unit="run", disable=not args.progress) as progress:
        for model, seed in run_order:
            run_dir = output_root / "runs" / f"{model}_seed{seed}"
            result_path = run_dir / "dynamic_eval.json"
            if result_path.exists() and not args.force:
                print(f"[skip] {model} seed={seed}")
                progress.update(1)
                continue
            static_root = Path(args.static_root)
            static_run_dir = static_run_dir_for(static_root, model, seed)
            checkpoint = checkpoint_path_for(static_root, model, seed)
            if not checkpoint.exists():
                if args.allow_missing:
                    print(f"[missing] {model} seed={seed}: {checkpoint}")
                    progress.update(1)
                    continue
                raise FileNotFoundError(f"Missing static best checkpoint: {checkpoint}")
            if not args.allow_incomplete_static and not completed_run(static_run_dir):
                if args.allow_missing:
                    print(f"[incomplete] {model} seed={seed}: {static_run_dir}")
                    progress.update(1)
                    continue
                raise RuntimeError(
                    f"Static convergence run is not complete for {model} seed={seed}: {static_run_dir}. "
                    "Use --allow-incomplete-static only for quick provisional diagnostics."
                )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "run-one",
                "--model",
                model,
                "--train-seed",
                str(seed),
                "--static-root",
                args.static_root,
                "--output-root",
                args.output_root,
                "--episodes",
                str(args.episodes),
                "--eval-seed",
                str(args.eval_seed),
                "--max-steps",
                str(args.max_steps),
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
            if not args.progress:
                command.append("--no-progress")
            if args.force:
                command.append("--force")
            if args.allow_incomplete_static:
                command.append("--allow-incomplete-static")
            stdout_log = output_root / "logs" / f"{model}_seed{seed}.stdout.log"
            stderr_log = output_root / "logs" / f"{model}_seed{seed}.stderr.log"
            metadata_path = run_dir / "child_process_metadata.json"
            print(f"[run] {model} seed={seed}")
            exit_code = run_child_and_wait(command, stdout_log, stderr_log, metadata_path, stream_output=args.stream_child_output)
            if exit_code != 0:
                raise RuntimeError(f"{model} seed={seed} failed. See {stderr_log}")
            progress.update(1)
    payload = summarize_dynamic_2x2(output_root)
    print(json.dumps({"summary_path": str(Path(args.output_root) / "summary" / "summary.json"), "runs": len(payload["runs"])}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic destination-perturbation Flat/Graph x FF/GRU evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run-all", "run-one"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--static-root", default="artifacts/static_2x2_convergence")
        sub.add_argument("--output-root", default="artifacts/dynamic_2x2_eval")
        sub.add_argument("--episodes", type=int, default=100)
        sub.add_argument("--eval-seed", type=int, default=3701)
        sub.add_argument("--max-steps", type=int, default=420)
        sub.add_argument("--eval-device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
        sub.add_argument("--eval-workers", type=int, default=4)
        sub.add_argument("--perturb-delay-steps", type=int, default=8)
        sub.add_argument("--perturbed-destination-x-min", type=float, default=0.052)
        sub.add_argument("--perturbed-destination-x-max", type=float, default=0.068)
        sub.add_argument("--min-perturb-delta", type=float, default=0.090)
        sub.add_argument("--force", action="store_true")
        sub.add_argument(
            "--allow-incomplete-static",
            action="store_true",
            help="Evaluate provisional best.pt files from static runs that have not written a completed result marker.",
        )
        sub.add_argument("--no-progress", dest="progress", action="store_false")
        sub.set_defaults(progress=True)
    run_all_parser = subparsers.choices["run-all"]
    run_all_parser.add_argument("--models", nargs="+", choices=list(MODEL_ORDER), default=list(MODEL_ORDER))
    run_all_parser.add_argument("--train-seeds", nargs="+", type=int, default=[1601, 1602, 1603])
    run_all_parser.add_argument("--allow-missing", action="store_true")
    run_all_parser.add_argument("--stream-child-output", action="store_true")
    run_one_parser = subparsers.choices["run-one"]
    run_one_parser.add_argument("--model", required=True, choices=list(MODEL_ORDER))
    run_one_parser.add_argument("--train-seed", type=int, required=True)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output-root", default="artifacts/dynamic_2x2_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run-all":
        run_all(args)
    elif args.command == "run-one":
        run_one(args)
    elif args.command == "summarize":
        payload = summarize_dynamic_2x2(Path(args.output_root))
        print(json.dumps({"summary_path": str(Path(args.output_root) / "summary" / "summary.json"), "runs": len(payload["runs"])}, indent=2))
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
