"""Constant-velocity moving-surface benchmark for frozen residual controllers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import recurrent_hidden_diagnostic as hd
from vla_gnn_recurrent.training import residual_recurrent_feasibility as residual
from vla_gnn_recurrent.training import residual_recurrent_replication as physical
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_representation_final_comparison as final
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device


OUTPUT = Path("artifacts/dynamic_visual_delay_feasibility")
DIRECTIONS = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
              "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}
CANDIDATE_STEP_SPEEDS_M = (0.001, 0.002, 0.004)
SELECTED_STEP_SPEEDS_M = (0.002, 0.004)
TRAIN_DELAYS = (0, 1, 2, 4)
EVAL_DELAYS = (0, 1, 2, 4, 8)
STAGE_A_UPDATES = 800
STAGE_B_UPDATES = 400


@dataclass(frozen=True)
class DynamicSpec:
    initial: sf.SurfaceEpisodeSpec
    direction: str
    speed_m_per_step: float

    @property
    def identity(self) -> str:
        return f"{self.initial.sample_identity}:{self.direction}:{self.speed_m_per_step:.6f}"


def moving_shape(spec: DynamicSpec, step: int) -> sf.SurfaceShape:
    if step < 0 or spec.direction not in DIRECTIONS:
        raise ValueError((step, spec.direction))
    center = np.asarray(spec.initial.shape.object_center, dtype=np.float64)
    moved = center + np.asarray(DIRECTIONS[spec.direction]) * spec.speed_m_per_step * step
    return replace(spec.initial.shape, object_center=tuple(float(x) for x in moved))


def dynamic_specs(seed: int, task: sf.SurfaceFeasibilityConfig, count: int,
                  speeds: tuple[float, ...], condition: str) -> list[DynamicSpec]:
    base_seed = seed + (90_000 if condition == "train" else 100_000 if condition == "validation" else 110_000)
    bases = sf.sample_episode_specs(count, base_seed, task, f"dynamic_{condition}")
    directions = tuple(DIRECTIONS)
    # Exactly balanced direction/speed exposure if count is a multiple of 4*speeds.
    return [DynamicSpec(base, directions[i % len(directions)], speeds[(i // len(directions)) % len(speeds)])
            for i, base in enumerate(bases)]


def _shape_at(env: sf.SurfaceManipulatorEnv, spec: DynamicSpec, step: int) -> sf.SurfaceShape:
    shape = moving_shape(spec, step)
    env.set_surface_shape(shape)
    return shape


@torch.no_grad()
def rollout(env: sf.SurfaceManipulatorEnv, spec: DynamicSpec, task: sf.SurfaceFeasibilityConfig,
            device: torch.device, controller: str = "oracle", model: Any = None,
            delay: int = 0, reset_hidden: bool = False, relabel: bool = False) -> tuple[dict, list[dict]]:
    if controller not in ("oracle", "ff", "residual_mlp", "residual_gru"):
        raise ValueError(controller)
    if controller != "oracle" and model is None:
        raise ValueError("A learned controller needs a model.")
    sf._reset_surface_env(env, spec.initial)
    observer = physical.PhysicalVisualDelayObserver(delay)
    previous = torch.zeros(1, sf.ACTION_DIM, device=device)
    hidden = None
    trace: list[dict] = []
    tracking: list[float] = []
    yaw_tracking: list[float] = []
    seen_collision = False
    minimum_clearance = math.inf
    latency = []
    for t in range(task.max_steps):
        shape = _shape_at(env, spec, t)
        ee = env.robot_observation().ee_position.numpy().astype(np.float64)
        target = sf._surface_target(shape, ee, task.pregrasp_clearance)
        state, points, _ = sf.observation_inputs(env, shape, task.point_count,
                                                  sf._stable_seed(spec.initial.sample_identity))
        observed_state, observed_points, age = observer.observe(state, points, ee)
        errors = sf._state_errors(env, shape, task, target)
        contacts, clearance = sf._surface_distance_metrics(env, shape)
        seen_collision |= bool(contacts)
        minimum_clearance = min(minimum_clearance, clearance)
        oracle = sf.expert_action(env, shape, task)[0] if controller == "oracle" or relabel else None
        if controller == "oracle":
            action = oracle.copy()
            base, residual_action, hidden_norm = None, None, None
            inference_ms = None
        else:
            graph_state, graph_points, topology = rf._graph_tensors(observed_state, observed_points, task, device)
            start = time.perf_counter_ns()
            if controller == "ff":
                raw = model(graph_state, graph_points, topology)
                base = raw[0].detach().cpu().numpy()
                residual_action, hidden_norm = None, None
            else:
                raw, correction, hidden, base_raw = model.step(
                    graph_state, graph_points, previous, hidden, topology,
                    reset_hidden=reset_hidden and controller == "residual_gru")
                base = base_raw[0].detach().cpu().numpy()
                residual_action = correction[0].detach().cpu().numpy()
                hidden_norm = float(hidden.norm().detach().cpu()) if hidden is not None else None
            action = rf.clipped_action(raw[0].detach().cpu().numpy(), task)
            inference_ms = (time.perf_counter_ns() - start) / 1e6
            latency.append(inference_ms)
        trace.append({"episode_id": spec.identity, "t": t, "controller": controller,
                      "direction": spec.direction, "speed_m_per_step": spec.speed_m_per_step,
                      "delay": delay, "observation_age": age,
                      "true_object_center_world": list(shape.object_center),
                      "observed_object_center_world": (np.asarray(shape.object_center) -
                          np.asarray(DIRECTIONS[spec.direction]) * spec.speed_m_per_step * age).tolist(),
                      "ee_position_world": ee.tolist(), "true_state": state.tolist(),
                      "true_surface_points": points.tolist(), "observed_state": observed_state.tolist(),
                      "observed_surface_points": observed_points.tolist(),
                      "target_position_world": target[0].tolist(), "target_yaw": target[2],
                      "oracle_action": None if oracle is None else oracle.tolist(),
                      "ff_base_action": None if base is None else base.tolist(),
                      "residual_action": None if residual_action is None else residual_action.tolist(),
                      "hidden_norm": hidden_norm,
                      "previous_executed_action": previous[0].detach().cpu().tolist(),
                      "executed_action": action.tolist(), "tracking_error": errors["position_error"],
                      "yaw_error": errors["orientation_error"], "aperture_error": errors["gripper_width_error"],
                      "clearance_m": float(clearance), "collision": bool(contacts),
                      "valid_supervision": clearance >= rf.TemporalConfig().severe_penetration_m,
                      "inference_ms": inference_ms})
        tracking.append(errors["position_error"])
        yaw_tracking.append(errors["orientation_error"])
        sf.apply_local_action(env, action, task)
        previous = torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, -1)
    final_shape = _shape_at(env, spec, task.max_steps)
    ee = env.robot_observation().ee_position.numpy().astype(np.float64)
    target = sf._surface_target(final_shape, ee, task.pregrasp_clearance)
    final_errors = sf._state_errors(env, final_shape, task, target)
    final_contacts, final_clearance = sf._surface_distance_metrics(env, final_shape)
    seen_collision |= bool(final_contacts)
    minimum_clearance = min(minimum_clearance, final_clearance)
    tracking.append(final_errors["position_error"])
    yaw_tracking.append(final_errors["orientation_error"])
    row = {"episode_id": spec.identity, "controller": controller, "direction": spec.direction,
           "speed_m_per_step": spec.speed_m_per_step, "delay": delay,
           "unobserved_displacement_m": spec.speed_m_per_step * delay,
           "success": sf._success_from_errors(final_errors, seen_collision, task),
           "collision": seen_collision, "minimum_safe_clearance": float(minimum_clearance),
           "final_tracking_error": final_errors["position_error"],
           "mean_tracking_error": float(np.mean(tracking)),
           "median_tracking_error": float(np.median(tracking)),
           "p95_tracking_error": float(np.quantile(tracking, .95)),
           "maximum_tracking_error": float(np.max(tracking)),
           "final_yaw_error": final_errors["orientation_error"],
           "trajectory_yaw_error": float(np.mean(yaw_tracking)),
           "final_aperture_error": final_errors["gripper_width_error"],
           "mean_residual_norm": float(np.mean([np.linalg.norm(s["residual_action"]) for s in trace]))
           if controller.startswith("residual") else 0.0,
           "mean_inference_ms": float(np.mean(latency)) if latency else None}
    return row, trace


def oracle_sweep(output: Path = OUTPUT, seed: int = 2811, episodes_per_combination: int = 4) -> dict:
    root = ensure_dir(output / "benchmark_sanity")
    task = rf._task_config(output, (seed,), "cpu", "cpu", 16)
    base = sf.sample_episode_specs(episodes_per_combination, seed + 80_000, task, "dynamic_oracle_sanity")
    env = sf.make_env(task, seed + 864_211)
    period = env.model.opt.timestep * task.control_substeps
    rows = []
    try:
        for speed in CANDIDATE_STEP_SPEEDS_M:
            for direction in DIRECTIONS:
                for episode in base:
                    spec = DynamicSpec(episode, direction, speed)
                    result, _ = rollout(env, spec, task, select_device("cpu"))
                    rows.append(result)
    finally:
        env.close()
    by_speed = {}
    for speed in CANDIDATE_STEP_SPEEDS_M:
        selected = [r for r in rows if r["speed_m_per_step"] == speed]
        by_speed[str(speed)] = {"episodes": len(selected),
                                "success": float(np.mean([r["success"] for r in selected])),
                                "collision": float(np.mean([r["collision"] for r in selected])),
                                "mean_tracking_error": float(np.mean([r["mean_tracking_error"] for r in selected])),
                                "meters_per_step": speed, "nominal_meters_per_second": speed / period}
    result = {"nominal_control_period_s": period, "model_timestep_s": env.model.opt.timestep,
              "kinematic_control_time_does_not_advance": True, "by_speed": by_speed,
              "per_episode": rows}
    residual._write(root / "oracle_sweep.json", result)
    return result


def staleness_sanity(output: Path = OUTPUT, seed: int = 2811,
                     speeds: tuple[float, ...] = (0.002, 0.004)) -> dict:
    """Numerically prove speed × effective visual age using the real graph sampler."""
    task = rf._task_config(output, (seed,), "cpu", "cpu", 16)
    initial = sf.sample_episode_specs(1, seed + 80_000, task, "dynamic_oracle_sanity")[0]
    env = sf.make_env(task, seed + 864_211)
    rows = []
    try:
        for speed in speeds:
            for direction in DIRECTIONS:
                spec = DynamicSpec(initial, direction, speed)
                for delay in EVAL_DELAYS:
                    sf._reset_surface_env(env, initial)
                    observer = physical.PhysicalVisualDelayObserver(delay)
                    for t in range(9):
                        shape = _shape_at(env, spec, t)
                        ee = env.robot_observation().ee_position.numpy().astype(np.float64)
                        state, points, _ = sf.observation_inputs(
                            env, shape, task.point_count, sf._stable_seed(initial.sample_identity))
                        observed_state, observed_points, age = observer.observe(state, points, ee)
                    graph_state, graph_points, topology = rf._graph_tensors(
                        observed_state, observed_points, task, select_device("cpu"))
                    rf.assert_canonical_topology(topology, task)
                    expected = speed * age
                    measured_center = float(np.linalg.norm(state[:3] - observed_state[:3]))
                    measured_surface = float(np.mean(np.linalg.norm(points - observed_points, axis=1)))
                    rows.append({"speed_m_per_step": speed,
                                 "nominal_speed_m_per_second": speed / (env.model.opt.timestep*task.control_substeps),
                                 "direction": direction, "requested_delay": delay, "effective_age": age,
                                 "expected_stale_displacement_m": expected,
                                 "measured_center_displacement_m": measured_center,
                                 "measured_surface_displacement_m": measured_surface,
                                 "center_error_m": measured_center - expected,
                                 "surface_error_m": measured_surface - expected,
                                 "graph_nodes": int(graph_points.shape[1]) + 3,
                                 "graph_edges": int(topology.src.shape[1]),
                                 "fresh_exact": bool(np.array_equal(state, observed_state) and
                                                     np.array_equal(points, observed_points)) if delay == 0 else None})
    finally:
        env.close()
    if any(abs(row["center_error_m"]) > 1e-6 or abs(row["surface_error_m"]) > 1e-6 or
           (row["requested_delay"] > 0 and row["measured_center_displacement_m"] < 1e-4)
           for row in rows):
        raise AssertionError("Moving-target delayed graph does not match speed × visual age.")
    result = {"rows": rows, "max_absolute_center_error_m": max(abs(x["center_error_m"]) for x in rows),
              "max_absolute_surface_error_m": max(abs(x["surface_error_m"]) for x in rows),
              "delay_zero_exact": all(x["fresh_exact"] for x in rows if x["requested_delay"] == 0)}
    residual._write(output / "benchmark_sanity/staleness_check.json", result)
    return result


def dynamic_delayed_sequence(ep: dict, delay: int) -> tuple[torch.Tensor, torch.Tensor]:
    states, points = ep["states"], ep["points"]
    if delay == 0:
        return states.clone(), points.clone()
    true_states = states.numpy()
    true_points = points.numpy()
    ee_world = ep["ee_world"].numpy()
    observed_states, observed_points = [], []
    for t, (state, visual) in enumerate(zip(true_states, true_points, strict=True)):
        source_t = max(0, t - delay)
        result = physical.reexpress_visual(state, visual, true_states[source_t],
                                           true_points[source_t], ee_world[t], ee_world[source_t], t-source_t)
        observed_states.append(result[0])
        observed_points.append(result[1])
    return torch.from_numpy(np.stack(observed_states)), torch.from_numpy(np.stack(observed_points))


def _trace_episode(spec: DynamicSpec, trace: list[dict], behavior_delay: int) -> dict:
    if any(row["oracle_action"] is None for row in trace):
        raise AssertionError("Training sequence lacks current-state oracle labels.")
    if any(not np.allclose(row["true_object_center_world"], moving_shape(spec, row["t"]).object_center)
           for row in trace):
        raise AssertionError("Oracle state is not the true current dynamic shape.")
    return {"states": torch.tensor([r["true_state"] for r in trace], dtype=torch.float32),
            "points": torch.tensor([r["true_surface_points"] for r in trace], dtype=torch.float32),
            "ee_world": torch.tensor([r["ee_position_world"] for r in trace], dtype=torch.float32),
            "targets": torch.tensor([r["oracle_action"] for r in trace], dtype=torch.float32),
            "previous": torch.tensor([r["previous_executed_action"] for r in trace], dtype=torch.float32),
            "mask": torch.tensor([r["valid_supervision"] for r in trace], dtype=torch.bool),
            "behavior_delay": behavior_delay, "episode_id": spec.initial.episode_id,
            "direction": spec.direction, "speed_m_per_step": spec.speed_m_per_step,
            "identity": spec.identity}


def collect_sequences(specs: list[DynamicSpec], task: sf.SurfaceFeasibilityConfig,
                      device: torch.device, controller: str = "oracle", model: Any = None,
                      behavior_delays: list[int] | None = None, seed: int = 2811) -> tuple[list[dict], list[dict]]:
    if behavior_delays is None:
        behavior_delays = [0] * len(specs)
    if len(behavior_delays) != len(specs):
        raise ValueError("One behavior delay is required per dynamic spec.")
    env = sf.make_env(task, seed + 864_211)
    episodes, rows = [], []
    try:
        for spec, delay in zip(specs, behavior_delays, strict=True):
            result, trace = rollout(env, spec, task, device, controller, model,
                                    delay, relabel=True)
            episodes.append(_trace_episode(spec, trace, delay))
            rows.append(result)
    finally:
        env.close()
    return episodes, rows


def behavior_delay_assignment(specs: list[DynamicSpec]) -> list[int]:
    # Every speed/direction stratum cycles through the four delays independently.
    counter: dict[tuple[float, str], int] = {}
    delays = []
    for spec in specs:
        key = (spec.speed_m_per_step, spec.direction)
        n = counter.get(key, 0)
        delays.append(TRAIN_DELAYS[n % len(TRAIN_DELAYS)])
        counter[key] = n + 1
    return delays


def collection_metadata(episodes: list[dict]) -> dict:
    strata: dict[str, int] = {}
    for episode in episodes:
        key = f"{episode['speed_m_per_step']:.6f}|{episode['direction']}|{episode['behavior_delay']}"
        strata[key] = strata.get(key, 0) + int(episode["mask"].sum())
    return {"episodes": len(episodes), "valid_supervised_timesteps": sum(int(ep["mask"].sum()) for ep in episodes),
            "valid_timesteps_by_speed_direction_behavior_delay": strata,
            "episode_identities": [ep["identity"] for ep in episodes]}


def _load_or_collect(path: Path, specs: list[DynamicSpec], task: sf.SurfaceFeasibilityConfig,
                     device: torch.device, controller: str, model: Any = None,
                     behavior_delays: list[int] | None = None, seed: int = 2811) -> list[dict]:
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if [ep["identity"] for ep in saved] != [spec.identity for spec in specs]:
            raise AssertionError(f"Saved dynamic sequences differ from frozen specs: {path}")
        return saved
    episodes, rows = collect_sequences(specs, task, device, controller, model, behavior_delays, seed)
    ensure_dir(path.parent)
    torch.save(episodes, path)
    residual._write(path.with_suffix(".json"), {"collection": collection_metadata(episodes), "rows": rows,
                                                 "controller": controller})
    return episodes


def _eligible_policy_by_delay(episodes: list[dict]) -> dict[int, list[int]]:
    return {delay: [i for i, ep in enumerate(episodes) if ep["behavior_delay"] == delay]
            for delay in TRAIN_DELAYS}


def _dynamic_protocol(task: sf.SurfaceFeasibilityConfig, seed: int,
                      training_device: torch.device, eval_device: torch.device,
                      control_period_s: float) -> dict:
    train = dynamic_specs(seed, task, 72, SELECTED_STEP_SPEEDS_M, "train")
    validation = dynamic_specs(seed, task, 16, SELECTED_STEP_SPEEDS_M, "validation")
    heldout = dynamic_specs(seed, task, 16, SELECTED_STEP_SPEEDS_M, "heldout")
    identities = [[x.identity for x in split] for split in (train, validation, heldout)]
    if len(set(sum(identities, []))) != sum(map(len, identities)):
        raise AssertionError("Dynamic train/validation/heldout splits overlap.")
    return {"seed": seed, "training_device": str(training_device), "evaluation_device": str(eval_device),
            "base_checkpoint": str(residual.BASE_CHECKPOINT), "base_checkpoint_sha256": residual.BASE_SHA256,
            "graph_points": task.point_count, "graph_edges": 100,
            "motion_directions": list(DIRECTIONS), "motion_step_speeds_m": list(SELECTED_STEP_SPEEDS_M),
            "nominal_control_period_s": control_period_s,
            "nominal_speeds_m_per_s": [x/control_period_s for x in SELECTED_STEP_SPEEDS_M],
            "training_delays": list(TRAIN_DELAYS), "evaluation_delays": list(EVAL_DELAYS),
            "stage_a_updates": STAGE_A_UPDATES, "stage_b_updates": STAGE_B_UPDATES,
            "sequence_batch_size": 16, "max_sequence_length": 32,
            "stage_a_batch": "2+2 expert/reference sequences per delay",
            "stage_b_batch": "2 expert + 2 current-policy sequences per delay",
            "policy_collection_episodes_per_branch": 72,
            "optimizer": "fresh AdamW per stage", "learning_rate": task.learning_rate,
            "weight_decay": task.weight_decay, "validation_interval": 25,
            "selection": "minimum unweighted expert-validation 5D action MSE",
            "residual_hidden_size": 128, "mlp_width": 384,
            "residual_fraction_of_action_limit": .2, "corrected_dimensions": 4,
            "severe_penetration_mask_below_m": rf.TemporalConfig().severe_penetration_m,
            "dynamic_splits": {"train": identities[0], "validation": identities[1], "heldout": identities[2]}}


def _save_protocol(path: Path, protocol: dict) -> None:
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise AssertionError(f"Frozen protocol changed: {path}")
    residual._write(path, protocol)


def _train_stage(kind: str, root: Path, expert: list[dict], policy: list[dict],
                 validation: list[dict], task: sf.SurfaceFeasibilityConfig,
                 design: residual.ResidualConfig, device: torch.device,
                 starting_checkpoint: Path | None = None,
                 restrict_policy_delay: bool = False) -> dict:
    log = root / "logs" / f"{kind}_training.json"
    eligibility = None
    if restrict_policy_delay:
        eligibility = {"expert": {delay: list(range(len(expert))) for delay in TRAIN_DELAYS},
                       "policy": _eligible_policy_by_delay(policy)}
    batches, schedule = residual.make_schedule(
        {"expert": expert, "policy": policy}, design, dynamic_delayed_sequence, eligibility)
    residual._write(root / "logs/schedule.json", schedule)
    if log.exists():
        record = json.loads(log.read_text())
        if record["schedule_sha256"] != schedule["schedule_sha256"] or \
                record["starting_checkpoint_sha256"] != (hd._sha256(starting_checkpoint) if starting_checkpoint else None):
            raise AssertionError("Saved stage checkpoint differs from frozen data or start weights.")
        return record
    return residual.train_branch(kind, batches, schedule, validation, task, design,
                                 device, root, starting_checkpoint=starting_checkpoint)


def _paired_rows(reference: list[dict], candidate: list[dict], seed: int) -> dict:
    if [r["episode_id"] for r in reference] != [r["episode_id"] for r in candidate]:
        raise AssertionError("Paired dynamic trajectories differ.")
    rng = np.random.default_rng(seed)
    metrics = ("success", "collision", "final_tracking_error", "mean_tracking_error",
               "median_tracking_error", "p95_tracking_error", "maximum_tracking_error",
               "final_yaw_error", "minimum_safe_clearance")
    result = {}
    for key in metrics:
        differences = np.asarray([float(b[key])-float(a[key]) for a,b in zip(reference,candidate,strict=True)])
        draws = rng.integers(0,len(differences),size=(5000,len(differences)))
        result[key] = {"mean_candidate_minus_reference": float(differences.mean()),
                       "paired_episode_bootstrap_95_ci": np.quantile(differences[draws].mean(1),[.025,.975]).tolist()}
        if key in ("success","collision"):
            reference_only = sum(bool(a[key] and not b[key]) for a,b in zip(reference,candidate,strict=True))
            candidate_only = sum(bool(b[key] and not a[key]) for a,b in zip(reference,candidate,strict=True))
            result[key].update({"reference_only":reference_only,"candidate_only":candidate_only,
                                "exact_mcnemar_p":sf._mcnemar_exact_p(reference_only,candidate_only)})
    return result


def evaluate_dynamic(models: dict, specs: list[DynamicSpec], task: sf.SurfaceFeasibilityConfig,
                     device: torch.device, root: Path, seed: int, delay: int,
                     reset_gru: bool = False) -> dict:
    record = root / "summary.json"
    if record.exists():
        return json.loads(record.read_text())
    env = sf.make_env(task, seed + 864_211)
    rows: dict[str,list[dict]] = {name:[] for name in models}
    traces: dict[str,list[dict]] = {name:[] for name in models}
    try:
        for spec in specs:
            for name, model in models.items():
                row, trace = rollout(env,spec,task,device,name,model,delay,
                                     reset_hidden=reset_gru and name=="residual_gru")
                rows[name].append(row)
                if len(traces[name]) < 8:
                    traces[name].append({"episode_id":spec.identity,"steps":trace})
    finally:
        env.close()
    if any([r["episode_id"] for r in data] != [s.identity for s in specs] for data in rows.values()):
        raise AssertionError("Dynamic evaluation episodes differ between controllers.")
    for name in models:
        if any(step["oracle_action"] is not None for ep in traces[name] for step in ep["steps"]):
            raise AssertionError("Closed-loop evaluation called scripted expert.")
        residual._write(root / "per_episode" / f"{name}.json", rows[name])
        residual._write(root / "traces" / f"{name}.json", traces[name])
    keys = ("success","collision","minimum_safe_clearance","final_tracking_error","mean_tracking_error",
            "median_tracking_error","p95_tracking_error","maximum_tracking_error","final_yaw_error",
            "trajectory_yaw_error","final_aperture_error","mean_residual_norm","mean_inference_ms")
    means = {name:{key:float(np.mean([r[key] for r in data])) if key!="mean_inference_ms" or name!="ff"
                 else float(np.mean([r[key] for r in data])) for key in keys}
             for name,data in rows.items()}
    result = {"seed":seed,"delay":delay,"reset_gru":reset_gru,"episodes":len(specs),"means":means,
              "paired":{name:_paired_rows(rows["ff"],data,seed+delay+i)
                        for i,(name,data) in enumerate(rows.items()) if name!="ff"}}
    if "residual_mlp" in rows and "residual_gru" in rows:
        result["paired_gru_minus_mlp"] = _paired_rows(rows["residual_mlp"],rows["residual_gru"],seed+delay+100)
    residual._write(record,result)
    return result


def run_seed(seed: int, output: Path = OUTPUT, device_preference: DevicePreference = "auto",
             eval_device_preference: DevicePreference = "cpu") -> dict:
    if seed not in (2811,2812,2813):
        raise ValueError(seed)
    sanity = json.loads((output / "benchmark_sanity/staleness_check.json").read_text())
    oracle = json.loads((output / "benchmark_sanity/oracle_sweep.json").read_text())
    if not sanity["delay_zero_exact"] or oracle["by_speed"]["0.002"]["success"] < .8 or \
            oracle["by_speed"]["0.004"]["success"] < .8:
        raise AssertionError("Benchmark sanity gate has not passed.")
    root = ensure_dir(output / f"seed{seed}")
    device, eval_device = select_device(device_preference), select_device(eval_device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2,torch.get_num_threads()))
    task = rf._task_config(root,(seed,),device_preference,eval_device_preference,16)
    if hd._sha256(residual.BASE_CHECKPOINT)!=residual.BASE_SHA256 or task.point_count!=32 or not task.symmetric_robot_edges:
        raise AssertionError("Canonical frozen FF/topology changed.")
    protocol = _dynamic_protocol(task,seed,device,eval_device,oracle["nominal_control_period_s"])
    _save_protocol(output / "audit/frozen_protocol.json", {k:v for k,v in protocol.items() if k not in
        ("seed","training_device","evaluation_device","dynamic_splits")})
    _save_protocol(root / "config.json",protocol)
    train_specs = dynamic_specs(seed,task,72,SELECTED_STEP_SPEEDS_M,"train")
    val_specs = dynamic_specs(seed,task,16,SELECTED_STEP_SPEEDS_M,"validation")
    eval_specs = dynamic_specs(seed,task,16,SELECTED_STEP_SPEEDS_M,"heldout")
    expert = _load_or_collect(root / "training/expert.pt",train_specs,task,eval_device,"oracle",seed=seed)
    validation = _load_or_collect(root / "training/validation.pt",val_specs,task,eval_device,"oracle",seed=seed)
    heldout_oracle = _load_or_collect(root / "benchmark_sanity/heldout_oracle.pt",eval_specs,task,eval_device,"oracle",seed=seed)
    oracle_rows = json.loads((root / "benchmark_sanity/heldout_oracle.json").read_text())["rows"]
    if np.mean([r["success"] for r in oracle_rows]) < .8 or np.mean([r["collision"] for r in oracle_rows]) > .1:
        raise AssertionError("Seed-specific heldout dynamic oracle feasibility failed.")
    stage_a_design = replace(residual.ResidualConfig(),seed=seed,updates=STAGE_A_UPDATES)
    stage_a = {}
    for kind in ("mlp","gru"):
        stage_a[kind] = _train_stage(kind,root/"training/stage_a",expert,expert,validation,
                                     task,stage_a_design,device)
    if stage_a["mlp"]["schedule_sha256"] != stage_a["gru"]["schedule_sha256"]:
        raise AssertionError("Stage A branches received different sampled tensors.")
    behavior_delays = behavior_delay_assignment(train_specs)
    policy = {}
    for kind in ("mlp","gru"):
        checkpoint = Path(stage_a[kind]["checkpoint"])
        model = residual.load_controller(kind,task,stage_a_design,eval_device,checkpoint)
        policy[kind] = _load_or_collect(root / f"training/policy_{kind}.pt",train_specs,task,eval_device,
                                        f"residual_{kind}",model,behavior_delays,seed)
    stage_b_design = replace(residual.ResidualConfig(),seed=seed,updates=STAGE_B_UPDATES)
    stage_b = {}
    for kind in ("mlp","gru"):
        stage_b[kind] = _train_stage(kind,root/f"training/stage_b_{kind}",expert,policy[kind],validation,
                                     task,stage_b_design,device,Path(stage_a[kind]["checkpoint"]),True)
    stage_b_mlp_draws = json.loads((root/"training/stage_b_mlp/logs/schedule.json").read_text())["draws_sha256"]
    stage_b_gru_draws = json.loads((root/"training/stage_b_gru/logs/schedule.json").read_text())["draws_sha256"]
    if stage_b_mlp_draws != stage_b_gru_draws:
        raise AssertionError("Stage B branches used different episode/delay draw indices.")
    models = {"ff":sf.load_surface_model(residual.BASE_CHECKPOINT,rf.GRAPH_NAME,task,eval_device),
              **{f"residual_{kind}":residual.load_controller(kind,task,stage_b_design,eval_device,
                Path(stage_b[kind]["checkpoint"])) for kind in ("mlp","gru")}}
    fresh = evaluate_dynamic(models,eval_specs,task,eval_device,root/"evaluation/delay0",seed,0)
    fresh_gate = {"oracle_success":float(np.mean([r["success"] for r in oracle_rows])),
                  "oracle_collision":float(np.mean([r["collision"] for r in oracle_rows])),
                  "all_learned_fail": all(fresh["means"][name]["success"]==0 for name in models),
                  "all_learned_collision_at_least_half":all(fresh["means"][name]["collision"]>=.5 for name in models)}
    fresh_gate["passed"] = not (fresh_gate["all_learned_fail"] or fresh_gate["all_learned_collision_at_least_half"])
    summary = {"seed":seed,"protocol":protocol,"oracle_heldout":fresh_gate,"stage_a":stage_a,"stage_b":stage_b,
               "fresh":fresh,"fresh_gate":fresh_gate,"fixed_delay":{},"carry_reset":{}}
    residual._write(root/"summary.json",summary)
    if not fresh_gate["passed"]:
        return summary
    for delay in EVAL_DELAYS[1:]:
        summary["fixed_delay"][str(delay)] = evaluate_dynamic(
            models,eval_specs,task,eval_device,root/f"evaluation/delay{delay}",seed,delay)
        residual._write(root/"summary.json",summary)
    for delay in EVAL_DELAYS:
        summary["carry_reset"][str(delay)] = evaluate_dynamic(
            {"ff":models["ff"],"residual_gru":models["residual_gru"]},eval_specs,task,eval_device,
            root/f"evaluation/reset_delay{delay}",seed,delay,reset_gru=True)
        residual._write(root/"summary.json",summary)
    return summary


def direction_history_probe(seed: int, output: Path = OUTPUT) -> dict:
    """Opposite histories with the same focal stale graph and zero action history.

    This controlled open-loop diagnostic isolates the carried hidden state. It is
    not a closed-loop performance metric or a privileged main-model input.
    """
    root = output / f"seed{seed}"
    task = rf._task_config(root,(seed,),"cpu","cpu",16)
    device = select_device("cpu")
    design = replace(residual.ResidualConfig(),seed=seed,updates=STAGE_B_UPDATES)
    models = {kind:residual.load_controller(kind,task,design,device,
              root/f"training/stage_b_{kind}/checkpoints/{kind}.pt") for kind in ("mlp","gru")}
    bases = sf.sample_episode_specs(2,seed+120_000,task,"direction_probe")
    env = sf.make_env(task,seed+864_211)
    results = []
    try:
        for base in bases:
            for speed in SELECTED_STEP_SPEEDS_M:
                for axis in ("x","z"):
                    axis_vector = np.array([1.,0.,0.]) if axis=="x" else np.array([0.,0.,1.])
                    for delay in (4,8):
                        focal_t = delay + 4
                        pair = {}
                        for sign in (1,-1):
                            shifted_center = np.asarray(base.shape.object_center)-sign*axis_vector*speed*4
                            initial = replace(base,shape=replace(base.shape,
                                object_center=tuple(float(v) for v in shifted_center)))
                            spec = DynamicSpec(initial,("+" if sign>0 else "-")+axis,speed)
                            sf._reset_surface_env(env,initial)
                            observer = physical.PhysicalVisualDelayObserver(delay)
                            previous = torch.zeros(1,sf.ACTION_DIM,device=device)
                            hidden = None
                            history = []
                            for t in range(focal_t+1):
                                shape = _shape_at(env,spec,t)
                                ee = env.robot_observation().ee_position.numpy().astype(np.float64)
                                state,points,_ = sf.observation_inputs(
                                    env,shape,task.point_count,sf._stable_seed(base.sample_identity))
                                observed,visual,age = observer.observe(state,points,ee)
                                state_t,points_t,topology=rf._graph_tensors(observed,visual,task,device)
                                _,gru_residual,hidden,_=models["gru"].step(
                                    state_t,points_t,previous,hidden,topology)
                                _,reset_residual,_,_=models["gru"].step(
                                    state_t,points_t,previous,None,topology,reset_hidden=True)
                                _,mlp_residual,_,_=models["mlp"].step(
                                    state_t,points_t,previous,None,topology)
                                history.append({"t":t,"true_center":list(shape.object_center),
                                                "observed_state":observed.tolist(),
                                                "observed_points":visual.tolist(),
                                                "gru_carry_residual":gru_residual[0].detach().cpu().tolist(),
                                                "gru_reset_residual":reset_residual[0].detach().cpu().tolist(),
                                                "mlp_residual":mlp_residual[0].detach().cpu().tolist(),
                                                "hidden_norm":float(hidden.norm().detach().cpu())})
                            pair[sign]=history
                        plus,minus=pair[1][-1],pair[-1][-1]
                        observed_diff=max(np.max(np.abs(np.asarray(plus["observed_state"])-minus["observed_state"])),
                                          np.max(np.abs(np.asarray(plus["observed_points"])-minus["observed_points"])))
                        def world_projection(row: dict, residual_key: str) -> float:
                            yaw=math.atan2(row["observed_state"][5],row["observed_state"][6])
                            world=sf.world_action_from_local(np.asarray(row[residual_key][:3]),yaw)
                            return float(np.dot(world,axis_vector))
                        plus_carry=world_projection(plus,"gru_carry_residual")
                        minus_carry=world_projection(minus,"gru_carry_residual")
                        results.append({"base_episode":base.episode_id,"speed_m_per_step":speed,
                                        "axis":axis,"delay":delay,"focal_t":focal_t,
                                        "focal_observed_graph_max_abs_diff":float(observed_diff),
                                        "focal_true_target_separation_m":float(np.linalg.norm(
                                            np.asarray(plus["true_center"])-minus["true_center"])),
                                        "plus_gru_carry_axis_residual":plus_carry,
                                        "minus_gru_carry_axis_residual":minus_carry,
                                        "plus_gru_reset_axis_residual":world_projection(plus,"gru_reset_residual"),
                                        "minus_gru_reset_axis_residual":world_projection(minus,"gru_reset_residual"),
                                        "plus_mlp_axis_residual":world_projection(plus,"mlp_residual"),
                                        "minus_mlp_axis_residual":world_projection(minus,"mlp_residual"),
                                        "gru_carry_directional_separation":plus_carry-minus_carry,
                                        "histories":{"plus":pair[1],"minus":pair[-1]}})
    finally:
        env.close()
    if any(row["focal_observed_graph_max_abs_diff"]>1e-6 for row in results):
        raise AssertionError("Opposite-direction probe did not match focal observed geometry.")
    summary={"seed":seed,"pairs":len(results),
             "mean_directional_residual_separation":float(np.mean([r["gru_carry_directional_separation"] for r in results])),
             "fraction_expected_residual_sign":float(np.mean([r["gru_carry_directional_separation"]>0 for r in results])),
             "maximum_focal_observed_graph_difference":max(r["focal_observed_graph_max_abs_diff"] for r in results),
             "rows":results}
    residual._write(root/"direction_history_probe.json",summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-sweep", action="store_true")
    parser.add_argument("--staleness-sanity", action="store_true")
    parser.add_argument("--seed", type=int, choices=(2811,2812,2813))
    parser.add_argument("--direction-probe", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    args = parser.parse_args()
    if args.eval_workers != 1:
        raise ValueError("Paired MuJoCo rollout uses a single process-local environment.")
    if args.oracle_sweep:
        oracle_sweep(args.output)
    elif args.staleness_sanity:
        staleness_sanity(args.output)
    elif args.direction_probe and args.seed is not None:
        direction_history_probe(args.seed,args.output)
    elif args.seed is not None:
        run_seed(args.seed,args.output,args.device,args.eval_device)
    else:
        raise ValueError("Run --oracle-sweep, --staleness-sanity, or --seed after both sanity gates.")


if __name__ == "__main__":
    main()
