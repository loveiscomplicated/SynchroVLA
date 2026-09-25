"""Confirmatory residual study with physically re-expressed stale vision.

The development artifact remains on its original EE-local copy corruption.  This
module changes only the delayed observation transform and seed-specific inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import recurrent_hidden_diagnostic as hd
from vla_gnn_recurrent.training import residual_recurrent_feasibility as residual
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_representation_final_comparison as final
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device


OUTPUT = Path("artifacts/residual_recurrent_replication")
SEEDS = (2811, 2812, 2813)
DELAYS = (0, 2, 4, 8)
PATTERN = (0, 0, 1, 3, 0, 2, 4, 1)


def _angle(state: np.ndarray) -> float:
    return math.atan2(float(state[5]), float(state[6]))


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    return sf.rotate_xz(torch.as_tensor(points, dtype=torch.float32), angle).numpy()


def reexpress_visual(state: np.ndarray, points: np.ndarray, old_state: np.ndarray,
                     old_points: np.ndarray, current_ee_world: np.ndarray,
                     old_ee_world: np.ndarray, age: int) -> tuple[np.ndarray, np.ndarray]:
    """Place old world visual geometry in the current EE frame; retain proprioception.

    `age=0` bypasses all arithmetic so the canonical fresh graph is bitwise equal.
    """
    if age == 0:
        return state.copy(), points.copy()
    old_yaw, current_yaw = _angle(old_state), _angle(state)
    world_points = _rotate(old_points, old_yaw) + np.asarray(old_ee_world, dtype=np.float32)
    world_center = _rotate(old_state[:3], old_yaw) + np.asarray(old_ee_world, dtype=np.float32)
    observed = state.copy()
    observed[:3] = sf.localize_world(world_center.reshape(1, 3), current_ee_world, current_yaw)[0]
    old_object_yaw = old_yaw + math.atan2(float(old_state[3]), float(old_state[4]))
    relative_yaw = old_object_yaw - current_yaw
    observed[3:5] = (math.sin(relative_yaw), math.cos(relative_yaw))
    return observed, sf.localize_world(world_points, current_ee_world, current_yaw).astype(np.float32)


class PhysicalVisualDelayObserver:
    def __init__(self, delay: int = 0, pattern: tuple[int, ...] | None = None) -> None:
        if delay < 0 or (pattern is not None and (not pattern or any(x < 0 for x in pattern))):
            raise ValueError("Delay and pattern must be nonnegative; pattern must be nonempty.")
        self.delay, self.pattern = delay, pattern
        self.history: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def observe(self, state: np.ndarray, points: np.ndarray,
                ee_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        self.history.append((state.copy(), points.copy(), np.asarray(ee_world).copy()))
        t = len(self.history) - 1
        requested_age = self.pattern[t % len(self.pattern)] if self.pattern else self.delay
        source_t = max(0, t - requested_age)
        old_state, old_points, old_ee = self.history[source_t]
        age = t - source_t
        observed_state, observed_points = reexpress_visual(
            state, points, old_state, old_points, ee_world, old_ee, age)
        return observed_state, observed_points, age


def corrected_delayed_sequence(ep: dict, delay: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same transform to saved ordered sequences as rollout uses.

    Training episodes contain EE-local observations, not EE world positions. The
    static episode object center reconstructs EE world pose from the true state.
    """
    states, points = ep["states"], ep["points"]
    if delay == 0:
        return states.clone(), points.clone()
    center = np.asarray(ep["shape"].object_center, dtype=np.float32)
    state_np, point_np = states.numpy(), points.numpy()
    ee_world = np.stack([center - _rotate(s[:3], _angle(s)) for s in state_np])
    transformed_states, transformed_points = [], []
    for t, (state, pts) in enumerate(zip(state_np, point_np, strict=True)):
        source_t = max(0, t - delay)
        observed_state, observed_points = reexpress_visual(
            state, pts, state_np[source_t], point_np[source_t],
            ee_world[t], ee_world[source_t], t - source_t)
        transformed_states.append(observed_state)
        transformed_points.append(observed_points)
    return torch.from_numpy(np.stack(transformed_states)), torch.from_numpy(np.stack(transformed_points))


def source_paths(seed: int) -> tuple[Path, Path, Path]:
    if seed == 2811:
        return (Path("artifacts/recurrent_recovery_priority_controlled/policy_pool.pt"),
                residual.OLD_POLICY, residual.NEW_POLICY)
    upstream = Path(f"artifacts/near_contact_translation_objective_multiseed/seed{seed}/upstream")
    return (upstream / "policy_pool.pt",
            upstream / "onpolicy_collection/old_base_gru/policy_relabelled_trajectories.pt",
            upstream / "onpolicy_collection/new_final_gru/policy_relabelled_trajectories.pt")


def prepare(seed: int, root: Path, device_preference: DevicePreference,
            eval_device_preference: DevicePreference) -> dict[str, Any]:
    design = replace(residual.ResidualConfig(), seed=seed)
    task = rf._task_config(root, (seed,), device_preference, eval_device_preference, 16)
    if task.point_count != 32 or not task.symmetric_robot_edges or task.message_passing_steps != 3:
        raise AssertionError("Canonical 100-edge graph configuration changed.")
    if hd._sha256(residual.BASE_CHECKPOINT) != residual.BASE_SHA256:
        raise AssertionError("Frozen FF checkpoint changed.")
    expert, validation, train_specs, validation_specs = final._current_datasets(seed, task)
    pool_path, old_path, new_path = source_paths(seed)
    expected_train_signatures = [final.spec_signature(s) for s in train_specs]
    for source_path in (old_path, new_path):
        source_specs = json.loads((source_path.parent / "episode_specs.json").read_text())
        source_signatures = [final.spec_signature(final.stab._as_spec(x)) for x in source_specs]
        if source_signatures != expected_train_signatures:
            raise AssertionError(f"Policy source EpisodeSpecs differ from training specs: {source_path}")
    pool = torch.load(pool_path, map_location="cpu", weights_only=False)
    data = residual.source_episodes(expert, pool, old_path, new_path)
    for source in data.values():
        for ep in source:
            ep["shape"] = train_specs[ep["episode_id"] % len(train_specs)].shape
    validation_data = residual.validation_episodes(validation)
    specs = sf.sample_episode_specs(design.heldout_episodes, seed + 60_000, task, "iid")
    if {s.sample_identity for s in specs} & {s.sample_identity for s in train_specs}:
        raise AssertionError("Held-out episodes overlap training episodes.")
    provenance = {"seed": seed, "design": asdict(design), "task": asdict(task),
                  "base_checkpoint": str(residual.BASE_CHECKPOINT), "base_sha256": residual.BASE_SHA256,
                  "policy_pool": str(pool_path), "policy_pool_sha256": hd._sha256(pool_path),
                  "old_policy": str(old_path), "old_policy_sha256": hd._sha256(old_path),
                  "new_policy": str(new_path), "new_policy_sha256": hd._sha256(new_path),
                  "training_episode_signatures": expected_train_signatures,
                  "validation_episode_signatures": [final.spec_signature(s) for s in validation_specs],
                  "heldout_episode_signatures": [final.spec_signature(s) for s in specs],
                  "delay_semantics": "old visual world geometry re-expressed in current EE frame",
                  "optimizer": "AdamW", "learning_rate": task.learning_rate,
                  "weight_decay": task.weight_decay,
                  "checkpoint_selection": "minimum unweighted expert-validation normalized 5D action MSE",
                  "delay_train_mixture": {"delays": list(design.train_delays),
                                          "expert_per_delay_per_update": 2,
                                          "policy_per_delay_per_update": 2},
                  "action_normalization": residual.action_scales(task).tolist(),
                  "perturbation_x_m": rf.TemporalConfig().perturbation_x_m,
                  "perturbation_step": rf.TemporalConfig().perturbation_step,
                  "variable_delay_pattern": list(PATTERN)}
    return {"design": design, "task": task, "data": data, "validation": validation_data,
            "specs": specs, "provenance": provenance}


def _save(path: Path, value: Any) -> None:
    residual._write(path, value)


def _condition(models: dict, specs: list, task: sf.SurfaceFeasibilityConfig,
               device: torch.device, root: Path, seed: int, delay: int = 0,
               reset: bool = False, pattern: tuple[int, ...] | None = None,
               perturbation: bool = False) -> dict:
    record = root / "summary.json"
    if record.exists():
        return json.loads(record.read_text())
    return residual.evaluate_condition(
        models, specs, task, rf.TemporalConfig(), device, root, delay=delay,
        reset_gru=reset, pattern=pattern, perturbation=perturbation,
        observer_factory=PhysicalVisualDelayObserver, seed=seed)


def run_seed(seed: int, output: Path = OUTPUT,
             device_preference: DevicePreference = "auto",
             eval_device_preference: DevicePreference = "cpu") -> dict:
    if seed not in SEEDS:
        raise ValueError(f"Confirmatory seeds are {SEEDS}")
    root = ensure_dir(output / ("seed2811_recheck" if seed == 2811 else f"seed{seed}"))
    prepared = prepare(seed, root, device_preference, eval_device_preference)
    design, task, specs = prepared["design"], prepared["task"], prepared["specs"]
    device, eval_device = select_device(device_preference), select_device(eval_device_preference)
    prepared["provenance"]["selected_training_device"] = str(device)
    prepared["provenance"]["selected_evaluation_device"] = str(eval_device)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    _save(root / "configs/frozen_protocol.json", prepared["provenance"])
    _save(root / "configs/heldout_episode_specs.json", [asdict(s) for s in specs])
    batches, schedule = residual.make_schedule(prepared["data"], design, corrected_delayed_sequence)
    _save(root / "logs/schedule.json", schedule)
    if seed == 2811:
        old_schedule = json.loads((residual.OUTPUT / "logs/schedule.json").read_text())
        if schedule["draws_sha256"] != old_schedule["draws_sha256"]:
            raise AssertionError("Corrected seed2811 changed sampled draws.")
    training = {}
    for kind in ("mlp", "gru"):
        record = root / "logs" / f"{kind}_training.json"
        if record.exists():
            training[f"residual_{kind}"] = json.loads(record.read_text())
            if training[f"residual_{kind}"]["schedule_sha256"] != schedule["schedule_sha256"] or \
                    training[f"residual_{kind}"]["base_checkpoint_sha256"] != residual.BASE_SHA256:
                raise AssertionError("Existing checkpoint used different training schedule.")
        else:
            training[f"residual_{kind}"] = residual.train_branch(
                kind, batches, schedule, prepared["validation"], task, design, device, root)
    del batches
    if training["residual_mlp"]["schedule_sha256"] != training["residual_gru"]["schedule_sha256"]:
        raise AssertionError("MLP/GRU sampled training tensors differ.")
    models = {"ff": sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME, task, eval_device),
              "residual_mlp": residual.load_controller("mlp", task, design, eval_device,
                                                       Path(training["residual_mlp"]["checkpoint"])),
              "residual_gru": residual.load_controller("gru", task, design, eval_device,
                                                       Path(training["residual_gru"]["checkpoint"]))}
    fresh = _condition(models, specs, task, eval_device, root / "metrics/fresh", seed)
    if seed == 2811:
        before = json.loads(residual.HISTORICAL_FRESH.read_text())
        after = json.loads((root / "metrics/fresh/per_episode/ff.json").read_text())
        for old_row, new_row in zip(before, after, strict=True):
            for key in ("success", "collision", "final_position_error", "final_orientation_error",
                        "final_gripper_width_error", "trajectory_error", "minimum_safe_clearance"):
                if not np.isclose(old_row[key], new_row[key], atol=1e-7):
                    raise AssertionError(f"Frozen FF fresh runner changed: {key}")
    gate = residual._fresh_gate(fresh)
    summary: dict[str, Any] = {"seed": seed, "audit": "BUG FOUND; corrected physical visual delay",
                               "training": training, "schedule": schedule, "fresh": fresh,
                               "fresh_gate": gate, "fixed_delay": {},
                               "hidden_reset": {}, "variable_delay": None, "perturbation": None}
    _save(root / "summary.json", summary)
    if not gate["passed"]:
        return summary
    for delay in DELAYS[1:]:
        summary["fixed_delay"][str(delay)] = _condition(
            models, specs, task, eval_device, root / f"metrics/delay{delay}", seed, delay)
        _save(root / "summary.json", summary)
    for delay in DELAYS:
        summary["hidden_reset"][str(delay)] = _condition(
            {"ff": models["ff"], "residual_gru": models["residual_gru"]}, specs, task, eval_device,
            root / f"metrics/reset_delay{delay}", seed, delay, reset=True)
        _save(root / "summary.json", summary)
    summary["variable_delay"] = _condition(models, specs, task, eval_device,
                                            root / "metrics/variable_delay", seed, pattern=PATTERN)
    _save(root / "summary.json", summary)
    summary["perturbation"] = _condition(models, specs, task, eval_device,
                                         root / "metrics/perturbation", seed, perturbation=True)
    _save(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    args = parser.parse_args()
    if args.eval_workers != 1:
        raise ValueError("This paired MuJoCo evaluator uses one process-local environment.")
    run_seed(args.seed, args.output, args.device, args.eval_device)


if __name__ == "__main__":
    main()
