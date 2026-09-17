from __future__ import annotations

import argparse
import multiprocessing as mp
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

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
    load_pick_policy,
)
from vla_gnn_recurrent.training.pick_place_bc import (
    PickPlaceEvalConfig,
    PickPlaceTrainConfig,
    _sample_pick_place_positions,
    evaluate_pick_place_offline,
    run_closed_loop_pick_place_episode,
    summarize_pick_place_results,
    train_pick_place_episode,
)
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


MODEL_KINDS: dict[str, PickModelKind] = {
    "flat_ff": "flat_feedforward_dir_mag",
    "flat_gru": "flat_recurrent_dir_mag",
    "graph_ff": "graph_feedforward_dir_mag",
    "graph_gru": "graph_recurrent_dir_mag",
}
MODEL_ORDER = ("flat_ff", "flat_gru", "graph_ff", "graph_gru")
PRIMARY_METRICS = (
    "grasp_success_rate",
    "lift_success_rate",
    "transport_success_rate",
    "valid_release_rate",
    "successful_placement_release_rate",
    "placement_success_rate",
)
SUMMARY_METRICS = (
    *PRIMARY_METRICS,
    "drop_rate",
    "mean_steps",
    "mean_final_placement_error",
    "mean_action_smoothness",
    "mean_latency_ms",
)
EFFECT_METRICS = (
    "placement_success_rate",
    "successful_placement_release_rate",
    "valid_release_rate",
    "transport_success_rate",
    "lift_success_rate",
    "grasp_success_rate",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0.0}
    return {
        "mean": float(sum(values) / len(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "n": float(len(values)),
    }


def validation_score(validation_summary: dict[str, Any], offline_validation: dict[str, Any]) -> tuple[float, ...]:
    """Lexicographic checkpoint score. Higher is better for every component."""

    return (
        float(validation_summary.get("placement_success_rate", 0.0)),
        float(validation_summary.get("successful_placement_release_rate", 0.0)),
        float(validation_summary.get("valid_release_rate", 0.0)),
        float(validation_summary.get("transport_success_rate", 0.0)),
        float(validation_summary.get("lift_success_rate", 0.0)),
        float(validation_summary.get("grasp_success_rate", 0.0)),
        -float(offline_validation.get("motion_l2_error", 0.0)),
    )


def completed_run(run_dir: Path) -> bool:
    result_path = run_dir / "run_result.json"
    if not result_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    checkpoint_path = payload.get("best_checkpoint_path")
    return bool(payload.get("completed")) and isinstance(checkpoint_path, str) and bool(checkpoint_path) and Path(checkpoint_path).exists()


def run_child_and_wait(
    command: list[str],
    stdout_log: Path,
    stderr_log: Path,
    metadata_path: Path,
    stream_output: bool = False,
) -> int:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    start = time.perf_counter()
    if stream_output:
        result = subprocess.run(command, check=False)
    else:
        with stdout_log.open("w", encoding="utf-8") as stdout_file, stderr_log.open("w", encoding="utf-8") as stderr_file:
            result = subprocess.run(command, stdout=stdout_file, stderr=stderr_file, check=False)
    metadata = {
        "command": command,
        "exit_code": int(result.returncode),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_sec": float(time.perf_counter() - start),
        "stdout_log": None if stream_output else str(stdout_log),
        "stderr_log": None if stream_output else str(stderr_log),
        "stream_output": stream_output,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return int(result.returncode)


def run_all(args: argparse.Namespace) -> None:
    output_root = ensure_dir(args.output_root)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    run_order: list[tuple[str, int]] = [(model, seed) for seed in args.train_seeds for model in args.models]
    completed_count = 0
    with tqdm(
        total=len(run_order),
        desc="static 2x2 convergence",
        unit="run",
        disable=not args.progress,
    ) as progress:
        for model_label, train_seed in run_order:
            run_dir = output_root / "runs" / f"{model_label}_seed{train_seed}"
            progress.set_postfix_str(f"{model_label} seed={train_seed}")
            if completed_run(run_dir) and not args.force:
                print(f"[skip] {model_label} seed={train_seed} already complete at {run_dir}")
                completed_count += 1
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
            ]
            if not args.precision_weighting:
                command.append("--no-precision-weighting")
            if not args.progress:
                command.append("--no-progress")
            stdout_log = output_root / "logs" / f"{model_label}_seed{train_seed}.stdout.log"
            stderr_log = output_root / "logs" / f"{model_label}_seed{train_seed}.stderr.log"
            metadata_path = run_dir / "child_process_metadata.json"
            print(f"[run] {model_label} seed={train_seed}")
            exit_code = run_child_and_wait(
                command,
                stdout_log,
                stderr_log,
                metadata_path,
                stream_output=args.stream_child_output,
            )
            if exit_code != 0:
                stderr_tail = _tail_text(stderr_log, line_count=80)
                raise RuntimeError(
                    f"{model_label} seed={train_seed} failed with exit code {exit_code}. "
                    f"stderr tail:\n{stderr_tail}"
                )
            completed_count += 1
            progress.update(1)
    summarize_experiment(output_root=output_root, fixed_budget_root=Path(args.fixed_budget_root))


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
            "min_epochs": args.min_epochs,
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.patience,
            "checkpoint_selection": [
                "placement_success_rate",
                "successful_placement_release_rate",
                "valid_release_rate",
                "transport_success_rate",
                "lift_success_rate",
                "grasp_success_rate",
                "offline_validation_motion_l2_error",
            ],
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

    epoch_iter = tqdm(
        range(1, args.max_epochs + 1),
        desc=f"{args.model} seed={args.train_seed}",
        unit="epoch",
        disable=not args.progress,
    )
    for epoch in epoch_iter:
        epoch_start = time.perf_counter()
        train_ids = list(dataset["split"]["train"])
        rng = np.random.default_rng(args.train_seed + epoch)
        rng.shuffle(train_ids)
        epoch_losses: list[dict[str, float]] = []
        epoch_updates = 0
        epoch_transitions = 0
        train_iter = tqdm(
            train_ids,
            desc=f"train epoch {epoch}",
            unit="ep",
            leave=False,
            disable=not args.progress,
        )
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
        validation = evaluate_closed_loop_checkpoint(
            checkpoint_path=epoch_checkpoint_path,
            episodes=args.val_episodes,
            seed=args.val_seed,
            max_steps=args.max_steps,
            output_dir=run_dir / "validation",
            keep_episodes=False,
            eval_device=args.eval_device,
            workers=args.eval_workers,
            progress=args.progress,
            progress_desc=f"val epoch {epoch}",
        )
        score = validation_score(validation["summary"], offline_validation)
        improved = best_score is None or score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_validation_metrics = {
                "offline": offline_validation,
                "closed_loop": validation["summary"],
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
            "validation_successful_placement_release_rate": float(
                validation["summary"]["successful_placement_release_rate"]
            ),
            "validation_placement_success_rate": float(validation["summary"]["placement_success_rate"]),
            "validation_mean_episode_length": float(validation["summary"]["mean_steps"]),
            "mean_inference_latency_ms": float(validation["summary"]["mean_latency_ms"]),
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
            best=best_epoch,
            stale=epochs_since_improvement,
        )
        _write_epoch_metrics(run_dir / "epoch_metrics.csv", rows)
        (run_dir / "epoch_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        _write_run_state(
            run_dir=run_dir,
            completed=False,
            best_checkpoint_path=best_checkpoint_path,
            best_epoch=best_epoch,
            best_validation_metrics=best_validation_metrics,
            epochs_since_improvement=epochs_since_improvement,
            rows=rows,
        )
        if epoch >= args.min_epochs and epochs_since_improvement >= args.patience:
            convergence_status = "early_stopped"
            break
    else:
        convergence_status = "censored_at_max_epoch" if best_epoch == args.max_epochs else "stopped_at_max_epoch_no_recent_improvement"

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_test = evaluate_closed_loop_checkpoint(
        checkpoint_path=best_checkpoint_path,
        episodes=args.test_episodes,
        seed=args.test_seed,
        max_steps=args.max_steps,
        output_dir=run_dir / "test",
        keep_episodes=True,
        eval_device=args.eval_device,
        workers=args.eval_workers,
        progress=args.progress,
        progress_desc="final test",
    )
    offline = {
        split: evaluate_pick_place_offline(model, dataset, split, device, max_step)
        for split in ("train", "val", "test")
    }
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
    _write_run_state(
        run_dir=run_dir,
        completed=True,
        best_checkpoint_path=best_checkpoint_path,
        best_epoch=best_epoch,
        best_validation_metrics=best_validation_metrics,
        epochs_since_improvement=epochs_since_improvement,
        rows=rows,
    )


def make_eval_episode_plan(episodes: int, seed: int, max_steps: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    eval_config = PickPlaceEvalConfig(episodes=episodes, seed=seed, max_steps=max_steps, render=False)
    plan: list[dict[str, Any]] = []
    for episode_idx in range(episodes):
        object_x, destination_x = _sample_pick_place_positions(rng, eval_config)
        plan.append(
            {
                "episode_idx": episode_idx,
                "episode": episode_idx + 1,
                "episode_seed": int(seed * 100_000 + episode_idx),
                "object_x": float(object_x),
                "destination_x": float(destination_x),
            }
        )
    return plan


def chunk_sequence(items: list[dict[str, Any]], chunks: int) -> list[list[dict[str, Any]]]:
    chunks = max(1, min(int(chunks), len(items) if items else 1))
    return [items[index::chunks] for index in range(chunks) if items[index::chunks]]


def _closed_loop_eval_worker(payload: dict[str, Any]) -> dict[str, Any]:
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
                    result = run_closed_loop_pick_place_episode(
                        model=model,
                        model_kind=model_kind,
                        env=env,
                        graph_builder=graph_builder,
                        device=device,
                        object_x=float(job["object_x"]),
                        destination_x=float(job["destination_x"]),
                        recurrent_mode="normal",
                        render=False,
                        render_dir=None,
                    )
                    result["episode"] = int(job["episode"])
                    result["episode_seed"] = int(job["episode_seed"])
                    result["episode_idx"] = int(job["episode_idx"])
                    results.append(result)
                except Exception as exc:  # pragma: no cover - exercised only on worker failure.
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


@torch.no_grad()
def evaluate_closed_loop_checkpoint(
    checkpoint_path: Path,
    episodes: int,
    seed: int,
    max_steps: int,
    output_dir: Path,
    keep_episodes: bool,
    eval_device: DevicePreference,
    workers: int,
    progress: bool = False,
    progress_desc: str = "closed loop",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = make_eval_episode_plan(episodes=episodes, seed=seed, max_steps=max_steps)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    worker_count = max(1, min(int(workers), len(plan) if plan else 1))
    if worker_count <= 1:
        device = select_device(eval_device)
        model, checkpoint = load_pick_policy(checkpoint_path, device)
        model_kind: PickModelKind = checkpoint["model_kind"]
        graph_builder = ManipulationGraphBuilder(task="pick_and_place", include_object=True)
        env = MujocoManipulatorEnv(_pick_env_config(max_steps=max_steps), seed=seed)
        try:
            episode_iter = tqdm(plan, desc=progress_desc, unit="ep", leave=False, disable=not progress)
            for job in episode_iter:
                set_seed(int(job["episode_seed"]))
                result = run_closed_loop_pick_place_episode(
                    model=model,
                    model_kind=model_kind,
                    env=env,
                    graph_builder=graph_builder,
                    device=device,
                    object_x=float(job["object_x"]),
                    destination_x=float(job["destination_x"]),
                    recurrent_mode="normal",
                    render=False,
                    render_dir=None,
                )
                result["episode"] = int(job["episode"])
                result["episode_seed"] = int(job["episode_seed"])
                result["episode_idx"] = int(job["episode_idx"])
                results.append(result)
                episode_iter.set_postfix(
                    place=f"{sum(1 for item in results if item['placement_success']) / len(results):.2f}",
                    release=f"{sum(1 for item in results if item['valid_release_success']) / len(results):.2f}",
                )
        finally:
            env.close()
    else:
        chunks = chunk_sequence(plan, worker_count)
        worker_payloads = [
            {
                "checkpoint_path": str(checkpoint_path),
                "eval_device": eval_device,
                "max_steps": max_steps,
                "worker_seed": int(seed + worker_idx),
                "jobs": chunk,
                "torch_num_threads": 1,
            }
            for worker_idx, chunk in enumerate(chunks)
        ]
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(worker_payloads), mp_context=context) as executor:
            futures = [executor.submit(_closed_loop_eval_worker, payload) for payload in worker_payloads]
            future_iter = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=progress_desc,
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
        error_path = output_dir / "parallel_eval_errors.json"
        error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        raise RuntimeError(f"Closed-loop evaluation failed for {len(errors)} episodes. See {error_path}")
    summary = summarize_pick_place_results(results)
    payload = {
        "episodes": results if keep_episodes else [],
        "summary": summary,
        "episode_count": episodes,
        "seed": seed,
        "episode_plan": plan,
        "checkpoint_path": str(checkpoint_path),
        "eval_device": eval_device,
        "eval_workers": worker_count,
    }
    if keep_episodes:
        (output_dir / "closed_loop_test.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_kind: PickModelKind,
    epoch: int,
    config: dict[str, Any],
    dataset: dict[str, Any],
    flat_input_dim: int | None,
    flat_schema: dict[str, Any],
    parameter_count: int,
    trainable_parameter_count: int,
    offline_validation: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": model_kind,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "env_config": dataset["env_config"],
            "split": dataset["split"],
            "flat_input_dim": flat_input_dim,
            "flat_schema": flat_schema,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "offline_validation": offline_validation,
        },
        path,
    )


def _write_epoch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_run_state(
    run_dir: Path,
    completed: bool,
    best_checkpoint_path: Path,
    best_epoch: int,
    best_validation_metrics: dict[str, Any],
    epochs_since_improvement: int,
    rows: list[dict[str, Any]],
) -> None:
    payload = {
        "completed": completed,
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation_metrics,
        "epochs_since_improvement": epochs_since_improvement,
        "last_epoch": int(rows[-1]["epoch"]) if rows else 0,
        "updated_at": _utc_now(),
    }
    (run_dir / "run_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_experiment(output_root: Path, fixed_budget_root: Path | None = None) -> dict[str, Any]:
    runs = _load_run_results(output_root)
    aggregate = _aggregate_final_metrics(runs)
    effects = _factorial_effects(runs)
    fixed_budget = _fixed_budget_comparison(fixed_budget_root) if fixed_budget_root is not None else {}
    comparison = _fixed_vs_convergence_comparison(aggregate, fixed_budget, runs)
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
        "fixed_budget": fixed_budget,
        "fixed_budget_vs_convergence": comparison,
    }
    summary_dir = ensure_dir(output_root / "summary")
    (summary_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_final_metrics_csv(runs, summary_dir / "final_metrics_by_seed.csv")
    (summary_dir / "factorial_effects.json").write_text(json.dumps(effects, indent=2), encoding="utf-8")
    (summary_dir / "fixed_budget_vs_convergence.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
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
                metric: _mean_std([float(run["final_test"]["summary"].get(metric, 0.0)) for run in model_runs])
                for metric in SUMMARY_METRICS
            },
            "convergence_status": {
                str(run["train_seed"]): run["convergence_status"]
                for run in model_runs
            },
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
        means = {
            model: seed_runs[model]["final_test"]["summary"]
            for model in MODEL_ORDER
        }
        for metric in EFFECT_METRICS:
            flat_ff = float(means["flat_ff"].get(metric, 0.0))
            flat_gru = float(means["flat_gru"].get(metric, 0.0))
            graph_ff = float(means["graph_ff"].get(metric, 0.0))
            graph_gru = float(means["graph_gru"].get(metric, 0.0))
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
                [
                    float(seed_effects[metric][effect_name])
                    for seed_effects in per_seed.values()
                    if metric in seed_effects
                ]
            )
    return {"per_seed": per_seed, "aggregate": aggregate}


def _fixed_budget_comparison(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {}
    paths = {
        "flat_ff": root / "eval_seed1601/flat_ff/flat_feedforward_dir_mag_normal_eval.json",
        "flat_gru": root / "eval_seed1601/flat_gru/flat_recurrent_dir_mag_normal_eval.json",
        "graph_ff": root / "eval_seed1601/graph_ff/graph_feedforward_dir_mag_normal_eval.json",
        "graph_gru": root / "eval_seed1601/graph_gru/graph_recurrent_dir_mag_normal_eval.json",
    }
    summaries = {}
    for model, path in paths.items():
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            summaries[model] = payload.get("summary", {})
    return summaries


def _fixed_vs_convergence_comparison(
    aggregate: dict[str, Any],
    fixed_budget: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    best_epoch_by_model = {
        model: aggregate.get(model, {}).get("best_epoch", {}).get("mean", 0.0)
        for model in MODEL_ORDER
    }
    stopped_by = {
        run["model_label"]: {
            **{
                str(existing["train_seed"]): existing["convergence_status"]
                for existing in runs
                if existing["model_label"] == run["model_label"]
            }
        }
        for run in runs
    }
    comparison = {}
    for model in MODEL_ORDER:
        comparison[model] = {
            "fixed_budget_placement_success_rate": fixed_budget.get(model, {}).get("placement_success_rate"),
            "convergence_placement_success_rate": aggregate.get(model, {})
            .get("metrics", {})
            .get("placement_success_rate", {})
            .get("mean"),
            "best_epoch_mean": best_epoch_by_model.get(model, 0.0),
            "stopped_by": stopped_by.get(model, {}),
        }
    return comparison


def _write_final_metrics_csv(runs: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "model_label",
        "train_seed",
        "best_epoch",
        "convergence_status",
        "parameter_count",
        *SUMMARY_METRICS,
    ]
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


def _plot_run_curves(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    specs = [
        ("training_loss", "Training loss"),
        ("validation_loss", "Offline validation L2"),
        ("validation_placement_success_rate", "Closed-loop placement"),
        ("validation_valid_release_rate", "Closed-loop valid release"),
    ]
    for axis, (key, title) in zip(axes.reshape(-1), specs, strict=True):
        axis.plot(epochs, [float(row[key]) for row in rows], marker="o")
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_aggregate_curves(output_root: Path, runs: list[dict[str, Any]], path: Path) -> None:
    if not runs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("training_loss", "Training loss"),
        ("validation_loss", "Offline validation L2"),
        ("validation_placement_success_rate", "Closed-loop placement"),
        ("validation_valid_release_rate", "Closed-loop valid release"),
    ]
    for axis, (key, title) in zip(axes.reshape(-1), specs, strict=True):
        for model in MODEL_ORDER:
            model_rows = []
            for run in runs:
                if run["model_label"] != model:
                    continue
                rows_path = output_root / "runs" / f"{model}_seed{run['train_seed']}" / "epoch_metrics.json"
                if rows_path.exists():
                    model_rows.append(json.loads(rows_path.read_text(encoding="utf-8")))
            if not model_rows:
                continue
            max_epoch = max(len(rows) for rows in model_rows)
            xs: list[int] = []
            ys: list[float] = []
            for epoch_idx in range(max_epoch):
                values = [float(rows[epoch_idx][key]) for rows in model_rows if epoch_idx < len(rows)]
                if values:
                    xs.append(epoch_idx + 1)
                    ys.append(_mean(values))
            axis.plot(xs, ys, marker="o", label=model)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _tail_text(path: Path, line_count: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convergence-aware static Flat/Graph x FF/GRU ablation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run-all", "run-one"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--dataset-path", default="artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger.pt")
        sub.add_argument("--output-root", default="artifacts/static_2x2_convergence")
        sub.add_argument("--min-epochs", type=int, default=5)
        sub.add_argument("--max-epochs", type=int, default=30)
        sub.add_argument("--patience", type=int, default=5)
        sub.add_argument("--val-episodes", type=int, default=20)
        sub.add_argument("--val-seed", type=int, default=2601)
        sub.add_argument("--test-episodes", type=int, default=100)
        sub.add_argument("--test-seed", type=int, default=2701)
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
        sub.add_argument("--force", action="store_true")
        sub.add_argument("--no-progress", dest="progress", action="store_false")
        sub.set_defaults(progress=True)
    run_all_parser = subparsers.choices["run-all"]
    run_all_parser.add_argument("--models", nargs="+", choices=list(MODEL_ORDER), default=list(MODEL_ORDER))
    run_all_parser.add_argument("--train-seeds", nargs="+", type=int, default=[1601, 1602, 1603])
    run_all_parser.add_argument("--fixed-budget-root", default="artifacts/static_2x2_ablation")
    run_all_parser.add_argument("--stream-child-output", action="store_true")
    run_one_parser = subparsers.choices["run-one"]
    run_one_parser.add_argument("--model", required=True, choices=list(MODEL_ORDER))
    run_one_parser.add_argument("--train-seed", type=int, required=True)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output-root", default="artifacts/static_2x2_convergence")
    summarize_parser.add_argument("--fixed-budget-root", default="artifacts/static_2x2_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run-all":
        run_all(args)
    elif args.command == "run-one":
        run_one(args)
    elif args.command == "summarize":
        summarize_experiment(output_root=Path(args.output_root), fixed_budget_root=Path(args.fixed_budget_root))
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
