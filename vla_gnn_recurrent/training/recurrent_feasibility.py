"""Paired feed-forward/GRU feasibility with the fixed no-local Surface Graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from torch import nn

from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_pregrasp_stabilization as stab
from vla_gnn_recurrent.training import surface_representation_final_comparison as final
from vla_gnn_recurrent.utils import DevicePreference, clamp_delta, ensure_dir, select_device, set_seed


ROOT = Path("artifacts/recurrent_feasibility_canonical100")
PRIOR = Path("artifacts/surface_representation_final_comparison")
SEEDS = (2811, 2812, 2813)
GRAPH_NAME = "surface_graph_no_local"


@dataclass(frozen=True)
class TemporalConfig:
    sequence_length: int = 32
    sequence_batch_size: int = 16
    gru_hidden: int = 256
    base_updates: int = 800
    dagger_updates: int = 800
    expert_fraction: float = 0.5
    severe_penetration_m: float = -0.020
    stale_lengths: tuple[int, ...] = (0, 1, 2, 4, 8)
    delay_lengths: tuple[int, ...] = (0, 1, 2, 4, 8)
    stale_start: int = 3
    perturbation_step: int = 5
    perturbation_x_m: float = 0.025
    recovery_position_m: float = 0.025
    recovery_yaw_rad: float = 0.200
    latency_warmup: int = 200
    latency_iterations: int = 2000


class GraphGRUPolicy(nn.Module):
    """Exactly the existing graph encoder/readout followed by one GRU."""

    def __init__(self, config: sf.SurfaceFeasibilityConfig, hidden: int = 256) -> None:
        super().__init__()
        self.graph = sf.SurfaceGraphNetwork(config.hidden_dim, config.message_passing_steps, False, config)
        # Its original FF head is bypassed when exposing the existing readout.
        self.graph.action_head.requires_grad_(False)
        self.gru = nn.GRU(config.hidden_dim * 3, hidden, num_layers=1, batch_first=True)
        self.action_head = nn.Sequential(nn.Linear(hidden, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, sf.ACTION_DIM))
        self.hidden_size = hidden

    def forward_sequence(self, states: torch.Tensor, points: torch.Tensor,
                         hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim != 3 or points.ndim != 4 or states.shape[:2] != points.shape[:2]:
            raise ValueError("Sequence inputs must be [batch,time,state] and [batch,time,points,xyz].")
        batch, length = states.shape[:2]
        embedding = self.graph.encode(states.reshape(batch * length, -1), points.reshape(batch * length, points.shape[2], 3))
        recurrent, next_hidden = self.gru(embedding.reshape(batch, length, -1), hidden)
        return self.action_head(recurrent), next_hidden

    def step(self, state: torch.Tensor, points: torch.Tensor,
             hidden: torch.Tensor | None, topology: sf.GraphTopology | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.graph.encode(state, points, topology)
        recurrent, next_hidden = self.gru(embedding[:, None], hidden)
        return self.action_head(recurrent[:, 0]), next_hidden


def _episodes(data: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
    """Keep complete ordered trajectories; an explicit loss mask excludes severe states."""
    ids = data["episode_ids"].tolist()
    steps = data["steps"].tolist()
    if any((ids[i], steps[i]) > (ids[i + 1], steps[i + 1]) for i in range(len(ids) - 1)):
        raise ValueError("DAgger observations must remain in episode/timestep order.")
    groups: list[list[int]] = []
    for i, (episode, step) in enumerate(zip(ids, steps, strict=True)):
        if not groups or episode != ids[groups[-1][-1]] or step != steps[groups[-1][-1]] + 1:
            groups.append([])
        groups[-1].append(i)
    return [{key: (data[key][indices] if key in data else torch.ones(len(indices), dtype=torch.bool))
             for key in ("states", "surface_points", "actions", "episode_ids", "steps", "loss_mask")}
            for indices in groups]


def sample_sequence_batch(episodes: list[dict[str, torch.Tensor]], count: int, max_length: int,
                          generator: np.random.Generator, device: torch.device,
                          lengths_out: list[int] | None = None,
                          episode_weights: np.ndarray | None = None,
                          picks_out: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if episode_weights is None:
        picks = generator.integers(len(episodes), size=count)
    else:
        weights = np.asarray(episode_weights, dtype=np.float64)
        if weights.shape != (len(episodes),) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
            raise ValueError("Episode sampling weights must be a nonnegative probability vector.")
        picks = generator.choice(len(episodes), size=count, p=weights)
    if picks_out is not None:
        picks_out.extend(int(pick) for pick in picks)
    slices: list[dict[str, torch.Tensor]] = []
    for pick in picks:
        episode = episodes[int(pick)]
        length = min(len(episode["actions"]), max_length)
        start = int(generator.integers(len(episode["actions"]) - length + 1))
        if lengths_out is not None:
            lengths_out.append(length)
        slices.append({key: value[start:start + length] for key, value in episode.items()})
    longest = max(len(s["actions"]) for s in slices)
    states = torch.zeros(count, longest, sf.STATE_DIM, device=device)
    points = torch.zeros(count, longest, sf.SURFACE_POINTS, 3, device=device)
    actions = torch.zeros(count, longest, sf.ACTION_DIM, device=device)
    mask = torch.zeros(count, longest, dtype=torch.bool, device=device)
    for i, sequence in enumerate(slices):
        length = len(sequence["actions"])
        states[i, :length] = sequence["states"].to(device)
        points[i, :length] = sequence["surface_points"].to(device)
        actions[i, :length] = sequence["actions"].to(device)
        mask[i, :length] = sequence["loss_mask"].to(device)
    return states, points, actions, mask


def sequence_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  config: sf.SurfaceFeasibilityConfig) -> torch.Tensor:
    scales = pred.new_tensor((config.max_delta_ee,) * 3 + (config.max_delta_rotation, config.max_delta_gripper))
    squared = ((pred - target) / scales).square().mean(-1)
    return squared[mask].mean()


def mixed_sequence_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        expert_count: int, config: sf.SurfaceFeasibilityConfig) -> torch.Tensor:
    return 0.5 * sequence_loss(pred[:expert_count], target[:expert_count], mask[:expert_count], config) + \
           0.5 * sequence_loss(pred[expert_count:], target[expert_count:], mask[expert_count:], config)


def train_gru(model: GraphGRUPolicy, expert: dict[str, Any], validation: dict[str, Any],
              policy: dict[str, Any] | None, config: sf.SurfaceFeasibilityConfig,
              temporal: TemporalConfig, seed: int, device: torch.device, updates: int,
              output: Path, *, sampling_seed: int | None = None,
              validation_interval: int | None = None, stage_label: str | None = None,
              policy_episode_weights: np.ndarray | None = None,
              separate_source_rngs: bool = False) -> dict[str, Any]:
    model.train()
    expert_episodes = _episodes(expert)
    policy_episodes = _episodes(policy) if policy is not None else []
    val_episodes = _episodes(validation)
    effective_sampling_seed = (seed + (700_001 if policy is not None else 300_001)
                               if sampling_seed is None else sampling_seed)
    rng = np.random.default_rng(effective_sampling_seed)
    policy_rng = np.random.default_rng(effective_sampling_seed + 1) if separate_source_rngs else rng
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_val = float("inf")
    history = []
    validation_interval = validation_interval or (25 if policy_episodes else 20)
    effective_stage = stage_label or ("dagger" if policy is not None else "base")
    exposure = {"expert_valid_supervised_timesteps": 0, "policy_valid_supervised_timesteps": 0,
                "expert_sampled_sequence_lengths": [], "policy_sampled_sequence_lengths": []}
    sampled_policy_indices: list[int] = []
    sampled_expert_indices: list[int] = []
    checkpoint = ensure_dir(output) / "gru.pt"
    for update in range(1, updates + 1):
        expert_count = temporal.sequence_batch_size if not policy_episodes else temporal.sequence_batch_size // 2
        batches = [sample_sequence_batch(expert_episodes, expert_count, temporal.sequence_length, rng, device,
                                         exposure["expert_sampled_sequence_lengths"],
                                         picks_out=sampled_expert_indices)]
        if policy_episodes:
            batches.append(sample_sequence_batch(policy_episodes, temporal.sequence_batch_size - expert_count,
                                                 temporal.sequence_length, policy_rng, device,
                                                 exposure["policy_sampled_sequence_lengths"],
                                                 episode_weights=policy_episode_weights,
                                                 picks_out=sampled_policy_indices))
        if len(batches) > 1 and batches[0][0].shape[1] != batches[-1][0].shape[1]:
            width = max(b[0].shape[1] for b in batches)
            batches = [tuple(torch.cat([x, x.new_zeros((x.shape[0], width - x.shape[1], *x.shape[2:]))], 1) if x.shape[1] < width else x
                             for x in batch) for batch in batches]
        states = torch.cat([b[0] for b in batches], 0)
        points = torch.cat([b[1] for b in batches], 0)
        actions = torch.cat([b[2] for b in batches], 0)
        mask = torch.cat([b[3] for b in batches], 0)
        exposure["expert_valid_supervised_timesteps"] += int(mask[:expert_count].sum().item())
        if policy_episodes:
            exposure["policy_valid_supervised_timesteps"] += int(mask[expert_count:].sum().item())
        prediction, _ = model.forward_sequence(states, points)
        if policy_episodes:
            # FF mixes labels 50:50 by state; source-weighted loss prevents
            # longer policy trajectories from dominating recurrent updates.
            loss = mixed_sequence_loss(prediction, actions, mask, expert_count, config)
        else:
            loss = sequence_loss(prediction, actions, mask, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update % validation_interval == 0 or update == updates:
            model.eval()
            with torch.no_grad():
                weighted_loss = 0.0
                valid_count = 0
                for episode in val_episodes:
                    states_v = episode["states"][None].to(device)
                    points_v = episode["surface_points"][None].to(device)
                    actions_v = episode["actions"][None].to(device)
                    pred_v, _ = model.forward_sequence(states_v, points_v)
                    count_v = len(episode["actions"])
                    weighted_loss += float(sequence_loss(pred_v, actions_v, torch.ones_like(actions_v[..., 0], dtype=torch.bool), config).cpu()) * count_v
                    valid_count += count_v
            value = weighted_loss / valid_count
            history.append({"update": update, "train_loss": float(loss.detach().cpu()), "validation_loss": value})
            if value < best_val:
                best_val = value
                torch.save({"model_state": model.state_dict(), "seed": seed, "updates": update,
                            "best_validation_loss": value, "hidden_size": temporal.gru_hidden,
                            "topology": "consistent_intended_100" if config.symmetric_robot_edges else "legacy_98_train_100_rollout",
                            "config": asdict(config),
                            "stage": effective_stage}, checkpoint)
            model.train()
    def length_distribution(lengths: list[int]) -> dict[str, Any]:
        counts = {str(length): lengths.count(length) for length in sorted(set(lengths))}
        return {"draws": len(lengths), "histogram": counts,
                "mean": float(np.mean(lengths)) if lengths else None,
                "p50": float(np.percentile(lengths, 50)) if lengths else None,
                "p95": float(np.percentile(lengths, 95)) if lengths else None}

    selected = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = {"checkpoint": str(checkpoint), "updates": updates, "best_validation_loss": best_val,
              "selected_update": int(selected["updates"]), "final_update_loss": history[-1]["train_loss"],
              "stage": effective_stage,
              "expert_sequences": len(expert_episodes), "policy_sequences": len(policy_episodes),
              "expert_policy_sequence_fraction": [1.0, 0.0] if policy is None else [0.5, 0.5],
              "sequence_batch_size": temporal.sequence_batch_size, "sequence_length": temporal.sequence_length,
              "validation_selection_rule": "lowest expert validation normalized per-state MSE",
              "validation_interval_updates": validation_interval,
              "expert_valid_supervised_timesteps": exposure["expert_valid_supervised_timesteps"],
              "policy_valid_supervised_timesteps": exposure["policy_valid_supervised_timesteps"],
              "actual_valid_supervised_timesteps": exposure["expert_valid_supervised_timesteps"] + exposure["policy_valid_supervised_timesteps"],
              "sampled_expert_sequence_lengths": length_distribution(exposure["expert_sampled_sequence_lengths"]),
              "sampled_policy_sequence_lengths": length_distribution(exposure["policy_sampled_sequence_lengths"]),
              "sampled_policy_episode_index_counts": {
                  str(index): sampled_policy_indices.count(index) for index in sorted(set(sampled_policy_indices))},
              "sampled_expert_episode_index_sha256": hashlib.sha256(
                  np.asarray(sampled_expert_indices, dtype=np.int64).tobytes()).hexdigest(),
              "separate_source_rngs": separate_source_rngs,
              "initialization_seed": seed, "sampling_seed": effective_sampling_seed,
              "hidden_initialization": "zero at each sampled contiguous segment; full episode at evaluation",
              "history": history}
    sf._write_json(output / "training.json", result)
    return result


class ObservationCorruptor:
    """Snapshots include graph node features and sampled EE-local surface geometry."""

    def __init__(self, mode: str, severity: int = 0, start: int = 3) -> None:
        if mode not in ("fresh", "stale", "delay", "perturbation") or severity < 0:
            raise ValueError((mode, severity))
        self.mode, self.severity, self.start = mode, severity, start
        self.history: list[tuple[np.ndarray, np.ndarray]] = []

    def observe(self, state: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, bool]:
        self.history.append((state.copy(), points.copy()))
        step = len(self.history) - 1
        if self.mode == "delay":
            observed = max(0, step - self.severity)
        elif self.mode == "stale" and self.start <= step < self.start + self.severity:
            observed = max(0, self.start - 1)
        else:
            observed = step
        old_state, old_points = self.history[observed]
        return old_state.copy(), old_points.copy(), step - observed, observed != step


def _graph_tensors(state: np.ndarray, points: np.ndarray, config: sf.SurfaceFeasibilityConfig,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor, sf.GraphTopology]:
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, sf.STATE_DIM)
    points_t = torch.as_tensor(points, dtype=torch.float32, device=device).reshape(1, -1, 3)
    topology_np = sf.build_graph_topology_numpy(points, state[8:14].reshape(2, 3), config, False)
    topology = sf.topology_from_numpy(topology_np, device)
    if config.symmetric_robot_edges:
        assert_canonical_topology(topology, config)
    return state_t, points_t, topology


def assert_canonical_topology(topology: sf.GraphTopology, config: sf.SurfaceFeasibilityConfig) -> None:
    if config.point_count != 32 or topology.src.shape[1] != 100:
        raise AssertionError("Canonical no-local graph requires N=32 and exactly 100 directed edges.")
    edges = set(zip(topology.src[0].detach().cpu().tolist(), topology.dst[0].detach().cpu().tolist(), strict=True))
    if not {(0, 1), (1, 0), (0, 2), (2, 0)}.issubset(edges):
        raise AssertionError("Canonical graph requires symmetric EE-fingertip edges.")
    if torch.any(topology.edge_type == 3):
        raise AssertionError("Canonical no-local graph must not contain surface-surface edges.")


def clipped_action(raw: np.ndarray, config: sf.SurfaceFeasibilityConfig) -> np.ndarray:
    translation = clamp_delta(torch.as_tensor(raw[:3], dtype=torch.float32), config.max_delta_ee).numpy()
    return np.asarray([translation[0], 0.0, translation[2],
                       np.clip(raw[3], -config.max_delta_rotation, config.max_delta_rotation),
                       np.clip(raw[4], -config.max_delta_gripper, config.max_delta_gripper)], dtype=np.float32)


def parameter_audit(model: nn.Module, config: sf.SurfaceFeasibilityConfig) -> dict[str, int]:
    """Distinguish registered, requires-grad, and loss-path parameters."""
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    state = torch.zeros(1, sf.STATE_DIM, device=device)
    points = torch.zeros(1, sf.SURFACE_POINTS, 3, device=device)
    if isinstance(model, GraphGRUPolicy):
        action, _ = model.step(state, points, None)
    else:
        action = model(state, points)
    action.sum().backward()
    counts = {
        "registered": sum(p.numel() for p in model.parameters()),
        "trainable_requires_grad": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "active_in_forward_loss": sum(p.numel() for p in model.parameters() if p.requires_grad and p.grad is not None),
        "inactive_registered": sum(p.numel() for p in model.parameters() if p.grad is None),
    }
    model.zero_grad(set_to_none=True)
    return counts


@torch.no_grad()
def controller_step(model: nn.Module, controller: str, state: np.ndarray, points: np.ndarray,
                    hidden: torch.Tensor | None, config: sf.SurfaceFeasibilityConfig,
                    device: torch.device, diagnostics: dict[str, Any] | None = None) -> tuple[np.ndarray, torch.Tensor | None]:
    state_t, points_t, topology = _graph_tensors(state, points, config, device)
    if controller == "gru":
        assert isinstance(model, GraphGRUPolicy)
        previous = hidden if hidden is not None else state_t.new_zeros((1, 1, model.hidden_size))
        raw, hidden = model.step(state_t, points_t, hidden, topology)
        if diagnostics is not None:
            diagnostics["hidden_delta_norm"] = float((hidden - previous).norm().detach().cpu())
    elif controller == "ff":
        raw = model(state_t, points_t, topology)
        hidden = None
    else:
        raise ValueError(controller)
    raw_numpy = raw[0].detach().cpu().numpy()
    if diagnostics is not None:
        diagnostics["raw_predicted_action"] = raw_numpy.tolist()
    return clipped_action(raw_numpy, config), hidden


def hidden_reset_due(policy: str, timestep: int) -> bool:
    if policy == "normal":
        return timestep == 0
    if not policy.startswith("reset_"):
        raise ValueError(f"Unknown hidden-state policy: {policy}")
    interval = int(policy.removeprefix("reset_"))
    if interval < 1:
        raise ValueError("Hidden reset interval must be positive.")
    return timestep % interval == 0


def rollout(env: sf.SurfaceManipulatorEnv, spec: sf.SurfaceEpisodeSpec, model: nn.Module,
            controller: str, config: sf.SurfaceFeasibilityConfig, temporal: TemporalConfig,
            device: torch.device, mode: str = "fresh", severity: int = 0,
            collect: bool = False, hidden_policy: str = "normal",
            record_dynamics: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if controller == "gru":
        hidden_reset_due(hidden_policy, 0)
    elif hidden_policy != "normal":
        raise ValueError("Hidden reset policies require a GRU controller.")
    sf._reset_surface_env(env, spec)
    shape = spec.shape
    initial = env.robot_observation().ee_position.numpy().astype(np.float64)
    target = sf._surface_target(shape, initial, config.pregrasp_clearance)
    observer = ObservationCorruptor(mode, severity, temporal.stale_start)
    hidden = None
    trace: list[dict[str, Any]] = []
    initial_errors = sf._state_errors(env, shape, config, target)
    initial_contacts, min_clearance = sf._surface_distance_metrics(env, shape)
    seen_collision = bool(initial_contacts)
    first_collision: int | None = 0 if initial_contacts else None
    first_success: int | None = 0 if sf._success_from_errors(initial_errors, seen_collision, config) else None
    recovery_time: int | None = None
    post_perturb_errors: list[float] = []
    trajectory = [initial.tolist()]
    trajectory_pos = [initial_errors["position_error"]]
    trajectory_yaw = [initial_errors["orientation_error"]]
    trajectory_aperture = [initial_errors["gripper_width_error"]]
    for step in range(config.max_steps):
        if first_success is not None and mode != "perturbation":
            break
        perturbation = mode == "perturbation" and step == temporal.perturbation_step
        if perturbation:
            center = list(shape.object_center)
            center[0] += temporal.perturbation_x_m
            shape = replace(shape, object_center=tuple(center), shape_id=shape.shape_id + "_shifted")
            env.set_surface_shape(shape)
            target = sf._surface_target(shape, initial, config.pregrasp_clearance)
            first_success = None
        state, points, _ = sf.observation_inputs(env, shape, config.point_count,
                                                  sf._stable_seed(spec.sample_identity))
        observed_state, observed_points, age, stale = observer.observe(state, points)
        contacts, clearance = sf._surface_distance_metrics(env, shape)
        collision_now = bool(contacts)
        seen_collision = seen_collision or collision_now
        min_clearance = min(min_clearance, clearance)
        if first_collision is None and collision_now:
            first_collision = step
        errors = sf._state_errors(env, shape, config, target)
        oracle = sf.expert_action(env, shape, config)[0] if collect else None
        reset_hidden = controller == "gru" and hidden_reset_due(hidden_policy, step)
        if reset_hidden:
            hidden = None
        action_diagnostics: dict[str, Any] = {}
        action, hidden = controller_step(model, controller, observed_state, observed_points, hidden,
                                         config, device, diagnostics=action_diagnostics)
        severe = clearance < temporal.severe_penetration_m
        trace.append({
            "episode_id": spec.episode_id, "t": step,
            "true_ee_pose": env.robot_observation().ee_position.numpy().tolist(),
            "true_yaw": float(sf.tool_yaw(env)), "true_aperture": float(sf.gripper_width(env)),
            "observed_graph_timestamp": step - age, "observation_age": age,
            "observed_state": observed_state.tolist() if collect or record_dynamics else None,
            "observed_surface_points": observed_points.tolist() if collect or record_dynamics else None,
            "true_state": state.tolist() if collect or record_dynamics else None,
            "true_surface_points": points.tolist() if collect or record_dynamics else None,
            "target_pose": target[0].tolist(), "target_yaw": float(target[2]),
            "position_error": errors["position_error"], "orientation_error": errors["orientation_error"],
            "aperture_error": errors["gripper_width_error"],
            "predicted_action": action.tolist(), "oracle_action": oracle.tolist() if oracle is not None else None,
            "raw_predicted_action": action_diagnostics["raw_predicted_action"],
            "processed_action": action.tolist(), "hidden_policy": hidden_policy,
            "hidden_reset": reset_hidden,
            "hidden_delta_norm": action_diagnostics.get("hidden_delta_norm"),
            "collision": collision_now, "clearance_m": float(clearance),
            "severe_penetration_excluded": bool(severe),
            "success_state": sf._success_from_errors(errors, seen_collision, config),
            "perturbation_flag": perturbation, "stale_flag": stale,
            "hidden_norm": float(hidden.norm().detach().cpu()) if hidden is not None else None,
        })
        sf.apply_local_action(env, action, config)
        if record_dynamics:
            next_state, next_points, _ = sf.observation_inputs(
                env, shape, config.point_count, sf._stable_seed(spec.sample_identity))
            trace[-1]["next_true_state"] = next_state.tolist()
            trace[-1]["next_true_surface_points"] = next_points.tolist()
        current = env.robot_observation().ee_position.numpy().astype(np.float64)
        trajectory.append(current.tolist())
        after_contacts, after_clearance = sf._surface_distance_metrics(env, shape)
        seen_collision = seen_collision or bool(after_contacts)
        min_clearance = min(min_clearance, after_clearance)
        if first_collision is None and after_contacts:
            first_collision = step + 1
        next_errors = sf._state_errors(env, shape, config, target)
        trajectory_pos.append(next_errors["position_error"])
        trajectory_yaw.append(next_errors["orientation_error"])
        trajectory_aperture.append(next_errors["gripper_width_error"])
        if perturbation or (mode == "perturbation" and step >= temporal.perturbation_step):
            post_perturb_errors.append(next_errors["position_error"])
            if recovery_time is None and next_errors["position_error"] < temporal.recovery_position_m and next_errors["orientation_error"] < temporal.recovery_yaw_rad:
                recovery_time = step + 1 - temporal.perturbation_step
        if first_success is None and sf._success_from_errors(next_errors, seen_collision, config):
            first_success = step + 1
        if mode == "perturbation" and first_success is not None and step + 1 >= temporal.perturbation_step + 1:
            break
    final_contacts, final_clearance = sf._surface_distance_metrics(env, shape)
    min_clearance = min(min_clearance, final_clearance)
    final_errors = sf._state_errors(env, shape, config, target)
    positions = np.asarray(trajectory)
    row = {
        "episode_id": spec.episode_id, "condition": mode, "severity": severity,
        "spec_signature": final.spec_signature(spec),
        "success": sf._success_from_errors(final_errors, seen_collision, config),
        "collision": seen_collision, "final_collision": bool(final_contacts),
        "first_collision_timestep": first_collision, "minimum_safe_clearance": float(min_clearance),
        "final_position_error": final_errors["position_error"],
        "final_orientation_error": final_errors["orientation_error"],
        "final_gripper_width_error": final_errors["gripper_width_error"],
        "trajectory_error": float(np.mean(trajectory_pos)),
        "trajectory_yaw_error": float(np.mean(trajectory_yaw)),
        "trajectory_aperture_error": float(np.mean(trajectory_aperture)),
        "trajectory_length": float(np.linalg.norm(np.diff(positions[:, [0, 2]], axis=0), axis=1).sum()),
        "steps_to_convergence": first_success if first_success is not None else config.max_steps,
        "steps_executed": len(trace), "recovery_time": recovery_time,
        "post_perturb_peak_position_error": max(post_perturb_errors) if post_perturb_errors else None,
        "post_perturb_min_position_error": min(post_perturb_errors) if post_perturb_errors else None,
        "post_perturb_overshoot": max(post_perturb_errors) - post_perturb_errors[0] if post_perturb_errors else None,
    }
    return row, trace


def collect_recurrent_policy_data(model: GraphGRUPolicy, specs: list[sf.SurfaceEpisodeSpec],
                                  config: sf.SurfaceFeasibilityConfig, temporal: TemporalConfig,
                                  seed: int, device: torch.device, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    final.validate_train_specs(specs)
    env = sf.make_env(config, seed + 4_051)
    kept: list[dict[str, Any]] = []
    all_states: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    try:
        for spec in specs:
            row, trace = rollout(env, spec, model, "gru", config, temporal, device, collect=True)
            rollout_rows.append(row)
            for item in trace:
                meta = {key: item[key] for key in ("episode_id", "t", "clearance_m", "collision", "severe_penetration_excluded")}
                metadata.append(meta)
                all_states.append(item)
                if not item["severe_penetration_excluded"]:
                    kept.append(item)
    finally:
        env.close()
    if not kept:
        raise RuntimeError("No eligible recurrent DAgger states.")
    data = {
        "states": torch.as_tensor(np.asarray([r["true_state"] for r in all_states]), dtype=torch.float32),
        "surface_points": torch.as_tensor(np.asarray([r["true_surface_points"] for r in all_states]), dtype=torch.float32),
        "actions": torch.as_tensor(np.asarray([r["oracle_action"] for r in all_states]), dtype=torch.float32),
        "policy_actions": torch.as_tensor(np.asarray([r["predicted_action"] for r in all_states]), dtype=torch.float32),
        "episode_ids": torch.as_tensor([r["episode_id"] for r in all_states], dtype=torch.long),
        "steps": torch.as_tensor([r["t"] for r in all_states], dtype=torch.long),
        "loss_mask": torch.as_tensor([not r["severe_penetration_excluded"] for r in all_states], dtype=torch.bool),
    }
    sequences = _episodes(data)
    output = ensure_dir(output)
    torch.save(data, output / "policy_relabelled_trajectories.pt")
    sf._write_json(output / "episode_specs.json", [asdict(spec) for spec in specs])
    sf._write_json(output / "visited_metadata.json", metadata)
    stats = {
        "split": "train", "physical_episode_count": len(specs), "visited_states": len(metadata),
        "eligible_states": len(kept), "excluded_severe_penetration": len(metadata) - len(kept),
        "severe_penetration_rule_m": temporal.severe_penetration_m,
        "ordered_contiguous_training_sequences": len(sequences),
        "sequence_lengths": [len(s["actions"]) for s in sequences],
        "rollout_success": float(np.mean([r["success"] for r in rollout_rows])),
        "rollout_collision": float(np.mean([r["collision"] for r in rollout_rows])),
    }
    sf._write_json(output / "collection_summary.json", stats)
    return data, stats


def _load_gru(path: Path, config: sf.SurfaceFeasibilityConfig,
              temporal: TemporalConfig, device: torch.device) -> GraphGRUPolicy:
    payload = torch.load(path, map_location=device, weights_only=False)
    expected = "consistent_intended_100" if config.symmetric_robot_edges else "legacy_98_train_100_rollout"
    if payload.get("topology") != expected:
        raise ValueError(f"GRU checkpoint topology mismatch: {path}")
    model = GraphGRUPolicy(config, temporal.gru_hidden).to(device)
    model.load_state_dict(payload["model_state"])
    return model.eval()


def _metric_means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("success", "collision", "final_position_error", "final_orientation_error",
            "final_gripper_width_error", "trajectory_error", "trajectory_yaw_error",
            "trajectory_aperture_error", "trajectory_length", "steps_to_convergence", "minimum_safe_clearance")
    summary: dict[str, Any] = {"episodes": len(rows)}
    for key in keys:
        summary[key] = float(np.mean([r[key] for r in rows]))
    recovered = [r["recovery_time"] for r in rows if r["recovery_time"] is not None]
    if rows[0]["condition"] == "perturbation":
        summary["recovery_fraction"] = len(recovered) / len(rows)
        summary["recovery_time_mean_among_recovered"] = float(np.mean(recovered)) if recovered else None
        summary["recovery_time_horizon_censored_mean"] = float(np.mean([r["recovery_time"] if r["recovery_time"] is not None else 20 for r in rows]))
        for key in ("post_perturb_peak_position_error", "post_perturb_min_position_error", "post_perturb_overshoot"):
            summary[key] = float(np.mean([r[key] for r in rows]))
    return summary


def evaluate_condition(specs: list[sf.SurfaceEpisodeSpec], ff: nn.Module, gru: GraphGRUPolicy,
                       config: sf.SurfaceFeasibilityConfig, temporal: TemporalConfig,
                       device: torch.device, mode: str, severity: int, output: Path) -> dict[str, Any]:
    env = sf.make_env(config, 864_211)
    rows: dict[str, list[dict[str, Any]]] = {"ff": [], "gru": []}
    traces: dict[str, list[dict[str, Any]]] = {"ff": [], "gru": []}
    try:
        for spec in specs:
            for name, model in (("ff", ff), ("gru", gru)):
                row, trace = rollout(env, spec, model, name, config, temporal, device, mode, severity)
                rows[name].append(row)
                if len(traces[name]) < 4:
                    traces[name].append({"episode_id": spec.episode_id, "spec": asdict(spec), "steps": trace})
    finally:
        env.close()
    if [r["spec_signature"] for r in rows["ff"]] != [r["spec_signature"] for r in rows["gru"]]:
        raise AssertionError("Paired controllers received different physical EpisodeSpecs.")
    paired = final.paired_comparison(rows["ff"], rows["gru"], config, "GRU - FF", 813_032 + severity)
    if mode == "perturbation":
        for key in ("recovery_time", "post_perturb_peak_position_error", "post_perturb_overshoot"):
            deltas = np.asarray([(b[key] if b[key] is not None else config.max_steps) -
                                 (a[key] if a[key] is not None else config.max_steps)
                                 for a, b in zip(rows["ff"], rows["gru"], strict=True)], dtype=float)
            paired["paired_metrics"][key] = {"candidate_minus_reference_mean": float(deltas.mean()),
                                              "bootstrap_95_ci": sf._bootstrap_ci(deltas, 61_201, config.bootstrap_samples)}
    output = ensure_dir(output)
    sf._write_json(output / "paired_results.json", {"ff": rows["ff"], "gru": rows["gru"], "paired": paired})
    sf._write_json(output / "selected_trajectories.json", traces)
    return {"ff": _metric_means(rows["ff"]), "gru": _metric_means(rows["gru"]), "paired": paired,
            "rows": rows}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_controller(model: nn.Module, controller: str, spec: sf.SurfaceEpisodeSpec,
                         config: sf.SurfaceFeasibilityConfig, temporal: TemporalConfig,
                         device: torch.device) -> dict[str, Any]:
    env = sf.make_env(config, 936_187)
    sf._reset_surface_env(env, spec)
    components = defaultdict(list)
    hidden = None
    try:
        with torch.no_grad():
            for i in range(temporal.latency_warmup + temporal.latency_iterations):
                start = time.perf_counter_ns()
                state, points, _ = sf.observation_inputs(env, spec.shape, config.point_count,
                                                          sf._stable_seed(spec.sample_identity))
                after_observation = time.perf_counter_ns()
                state_t, points_t, topology = _graph_tensors(state, points, config, device)
                after_graph = time.perf_counter_ns()
                _sync(device)
                forward_start = time.perf_counter_ns()
                if controller == "gru":
                    raw, hidden = model.step(state_t, points_t, hidden, topology)
                    hidden = hidden.detach()
                else:
                    raw = model(state_t, points_t, topology)
                _sync(device)
                forward_end = time.perf_counter_ns()
                action = clipped_action(raw[0].detach().cpu().numpy(), config)
                sf.world_action_from_local(action[:3], sf.tool_yaw(env))
                end = time.perf_counter_ns()
                if i >= temporal.latency_warmup:
                    for key, value in (("observation_preparation", after_observation - start),
                                       ("graph_and_tensor", after_graph - after_observation),
                                       ("network_forward", forward_end - forward_start),
                                       ("geometry_to_action", end - start)):
                        components[key].append(value / 1e6)
    finally:
        env.close()
    return {"device": str(device), "controller": controller,
            "batch_size": 1, "warmup": temporal.latency_warmup, "iterations": temporal.latency_iterations,
            "components": {key: sf._distribution_ms([round(value * 1e6) for value in values])
                           for key, values in components.items()},
            "sensor_to_command_latency": "NOT MEASURED"}


def _task_config(output: Path, seeds: tuple[int, ...], device: DevicePreference,
                 eval_device: DevicePreference, eval_episodes: int) -> sf.SurfaceFeasibilityConfig:
    prior = json.loads(Path("artifacts/surface_pregrasp_stabilization/config.json").read_text())
    fields = prior["base_experiment_config"]
    tuple_fields = ("seeds", "training_object_x_range", "training_object_z_range", "training_length_range",
                    "training_half_width_range", "training_depth_range", "ood_half_width_range", "ood_depth_range")
    values = {k: tuple(v) if k in tuple_fields else v for k, v in fields.items()}
    values.update(output_dir=str(output), device=device, eval_device=eval_device, point_count=32,
                  symmetric_robot_edges=True)
    return sf.SurfaceFeasibilityConfig(**values)


def _restrict_episodes(data: dict[str, Any], specs: list[sf.SurfaceEpisodeSpec]) -> dict[str, Any]:
    """Select complete physical episodes for smoke runs, preserving timestep order."""
    selected = {spec.episode_id for spec in specs}
    keep = torch.as_tensor([int(x) in selected for x in data["episode_ids"].tolist()], dtype=torch.bool)
    size = len(data["episode_ids"])
    result = {key: (value[keep] if isinstance(value, torch.Tensor) and len(value.shape) > 0
                    and value.shape[0] == size else value) for key, value in data.items()}
    result["specs"] = [asdict(spec) for spec in specs]
    if "episodes" in result:
        result["episodes"] = [row for row in result["episodes"] if row["spec"]["episode_id"] in selected]
    return result


def _expert_reference(seed: int, specs: list[sf.SurfaceEpisodeSpec]) -> dict[int, dict[str, Any]]:
    path = PRIOR / "onpolicy_collection" / f"seed{seed}" / "expert_reference" / "expert_states.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    recorded = {int(row["spec"]["episode_id"]): row for row in data["episodes"]}
    if not {spec.episode_id for spec in specs}.issubset(recorded):
        raise ValueError("Missing historical expert reference for canonical physical training specs.")
    return {spec.episode_id: {"ee_positions": recorded[spec.episode_id]["expert_rollout"]["trajectory"]}
            for spec in specs}


def _plot_condition_curves(aggregate: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode in ("stale", "delay"):
        entries = aggregate["pooled"].get(mode, {})
        x = sorted(int(key) for key in entries)
        if not x:
            axes[0 if mode == "stale" else 1].set_visible(False)
            continue
        for controller in ("ff", "gru"):
            axes[0 if mode == "stale" else 1].plot(
                x, [entries[str(n)][controller]["success"] for n in x], marker="o", label=controller)
        ax = axes[0 if mode == "stale" else 1]
        ax.set(title=mode, xlabel="Observation age / severity (steps)", ylabel="Success fraction", ylim=(0, 1))
        ax.grid(alpha=0.3)
        ax.legend()
    ensure_dir(output.parent)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_saved_trace_pair(condition_dir: Path, output: Path) -> None:
    """Plot true errors, action magnitude, and observation age from paired traces."""
    payload = json.loads((condition_dir / "selected_trajectories.json").read_text())
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for controller, color in (("ff", "tab:blue"), ("gru", "tab:orange")):
        steps = payload[controller][0]["steps"]
        t = [r["t"] for r in steps]
        axes[0].plot(t, [r["position_error"] for r in steps], color=color, label=f"{controller} position")
        axes[0].plot(t, [r["orientation_error"] for r in steps], color=color, linestyle="--", label=f"{controller} yaw")
        axes[1].plot(t, [float(np.linalg.norm(r["predicted_action"])) for r in steps], color=color, label=controller)
        axes[2].step(t, [r["observation_age"] for r in steps], color=color, where="post", label=controller)
        for ax in axes:
            for row in steps:
                if row["perturbation_flag"]:
                    ax.axvline(row["t"], color="black", alpha=0.3)
    axes[0].set_ylabel("Position m / yaw rad")
    axes[1].set_ylabel("Action norm")
    axes[2].set_ylabel("Observation age (steps)")
    axes[2].set_xlabel("Control step")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    ensure_dir(output.parent)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _summary_markdown(result: dict[str, Any]) -> str:
    lines = ["# Recurrent feasibility: fixed Robot–Surface Interaction Graph", "",
             "## Architecture and protocol", "",
             f"Graph embedding: {result['architecture']['embedding_dim']} (EE + two fingertips, each 64). ",
             f"FF: existing two-layer action head; {result['architecture']['ff_parameters']} parameters. ",
             f"GRU: one layer, hidden {result['architecture']['gru_hidden']}, 256→64→5 action head; {result['architecture']['gru_parameter_audit']['registered']} registered, {result['architecture']['gru_parameter_audit']['trainable_requires_grad']} trainable/active parameters. The graph module retains its historical FF head in the state dict, frozen and bypassed by the GRU.",
             "Both use the unchanged 32-point no-local Surface Graph, existing 5D action, action clipping, IK and collision checks.", "",
             f"Training: {result['temporal']['base_updates']} base expert updates and {result['temporal']['dagger_updates']} DAgger updates per seed for each controller. GRU draws {result['temporal']['sequence_batch_size']} ordered sequences/update (length at most {result['temporal']['sequence_length']}); DAgger mixes expert and self-policy sequences 1:1. Hidden state starts at zero for each contiguous training segment and each episode, and persists through the full rollout. Both controllers are trained from scratch under the canonical symmetric 100-edge topology. AdamW, LR 3e-4, weight decay 1e-4.", "",
             "The GRU has more trainable parameters. Both graph modules follow the historical seed initialization convention, but their learned graph weights are independently updated. Sequence draws and FF state draws have different numbers of supervised frames per update; the logged counts limit a strict capacity or sample-budget attribution.", "",
             "## Evaluation", "",
             "Stale: the complete graph snapshot from immediately before the stale window is repeated while MuJoCo continues stepping. Delay: at step t the controller receives the complete graph from max(0,t−d). The full graph snapshot includes robot state and EE-local surface points. Perturbation support shifts object/surface center +0.025 m in world x at step 5, but perturbation is reserved until stale/delay provide a reason to evaluate it.", "",
             "All corruption schedules and physical EpisodeSpecs are paired. Evaluation actions come only from the neural policy. Oracle actions are used during training collection, not closed-loop evaluation.", "",
             "## Results", "",
             "| Condition | Severity | FF success | GRU success | FF collision | GRU collision | Success Δ (GRU−FF) | Collision Δ |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in ("fresh", "stale", "delay", "perturbation"):
        for severity, row in sorted(result["pooled"].get(mode, {}).items(), key=lambda kv: int(kv[0])):
            pair = row["paired"]["paired_metrics"]
            lines.append(f"| {mode} | {severity} | {row['ff']['success']:.3f} | {row['gru']['success']:.3f} | {row['ff']['collision']:.3f} | {row['gru']['collision']:.3f} | {pair['success']['candidate_minus_reference_mean']:+.3f} | {pair['collision']['candidate_minus_reference_mean']:+.3f} |")
    lines.extend(["", "Paired episode bootstrap 95% intervals and exact success disagreement counts are in `summary.json` and each condition's `paired_results.json`. Each condition also records final position, yaw, aperture, trajectory error, clearance and steps. Perturbation records recovery time, peak error and overshoot.", "",
                  "## Latency", "", "| Controller | Parameters | Forward p50/p95/p99 ms | Geometry-to-action p50/p95/p99 ms | >5 ms |", "|---|---:|---:|---:|---:|"])
    for controller, row in result.get("latency", {}).items():
        forward = row["components"]["network_forward"]
        total = row["components"]["geometry_to_action"]
        lines.append(f"| {controller} | {result['architecture'][controller + '_parameters']} | {forward['p50_ms']:.3f}/{forward['p95_ms']:.3f}/{forward['p99_ms']:.3f} | {total['p50_ms']:.3f}/{total['p95_ms']:.3f}/{total['p99_ms']:.3f} | {100*total['deadline_miss_rate_over_5ms']:.2f}% |")
    lines.extend(["", "Project design target: 100 Hz, geometry-to-action p99 ≤5 ms. sensor-to-command latency = NOT MEASURED.", "",
                  "## Interpretation", "", result.get("interpretation", "Fresh-behavior gate has not been assessed."), "",
                  "This experiment uses oracle task-relevant surface observations and a geometric pre-grasp task. Full perception, contact-rich grasping, and transfer to other tasks were not measured.", ""])
    return "\n".join(lines)


def run_experiment(output: Path = Path("artifacts/recurrent_feasibility_canonical100"), seeds: tuple[int, ...] = SEEDS,
                   device_preference: DevicePreference = "auto", eval_device_preference: DevicePreference = "auto",
                   temporal: TemporalConfig = TemporalConfig(), eval_episodes: int = 16,
                   smoke: bool = False, train_episodes: int | None = None,
                   validation_episodes: int | None = None) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Experiment artifact already exists: {output}")
    root = ensure_dir(output)
    config = _task_config(root, seeds, device_preference, eval_device_preference, eval_episodes)
    if not config.symmetric_robot_edges or config.point_count != 32:
        raise AssertionError("New recurrent experiment must use the canonical 100-edge N=32 topology.")
    if config.max_steps > temporal.sequence_length:
        raise ValueError("This run requires full-episode sequences; increase sequence_length or add burn-in.")
    train_device = select_device(device_preference)
    eval_device = select_device(eval_device_preference)
    if train_device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    sf._write_json(root / "config.json", {"task": asdict(config), "temporal": asdict(temporal),
                                          "training_device": str(train_device), "evaluation_device": str(eval_device),
                                          "fixed_graph_variant": GRAPH_NAME, "surface_points": 32,
                                          "topology": "consistent_intended_100", "local_surface_edges": False,
                                          "smoke": smoke, "training_episode_limit": train_episodes,
                                          "validation_episode_limit": validation_episodes,
                                          "eval_workers": 1, "sensor_to_command_latency": "NOT MEASURED"})
    aggregate: dict[str, Any] = {"seeds": seeds, "temporal": asdict(temporal), "per_seed": {}, "pooled": {},
                                 "latency": {}}
    pooled_rows: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"ff": [], "gru": []})
    checkpoint_map: dict[int, tuple[Path, Path]] = {}
    for seed in seeds:
        print(f"[recurrent] seed {seed}: loading paired expert data", flush=True)
        expert, val, train_specs, _ = final._current_datasets(seed, config)
        if train_episodes is not None:
            train_specs = train_specs[:train_episodes]
            expert = _restrict_episodes(expert, train_specs)
        if validation_episodes is not None:
            val_specs = sf.sample_episode_specs(config.validation_episodes, seed + 10_000, config, "validation")[:validation_episodes]
            val = _restrict_episodes(val, val_specs)
        seed_root = ensure_dir(root / "training" / f"seed{seed}")
        sf._write_json(seed_root / "shared_train_episode_specs.json", [asdict(spec) for spec in train_specs])
        ff_base = stab.train_fixed_updates(GRAPH_NAME, expert, val, config, seed,
                                           seed_root / "ff_base", temporal.base_updates, device_preference)
        ff_base_path = Path(ff_base["checkpoint_path"])
        final._checkpoint_config_compatible(ff_base_path, config, GRAPH_NAME, seed)
        ff_base_model = sf.load_surface_model(ff_base_path, GRAPH_NAME, config, eval_device)
        ff_policy, ff_metadata, ff_collection = final.collect_self_policy_states(
            GRAPH_NAME, ff_base_model, train_specs, _expert_reference(seed, train_specs),
            config, seed, eval_device, root / "onpolicy_collection" / f"seed{seed}" / "ff")
        ff_dagger = final.train_dagger_round1(
            GRAPH_NAME, ff_base_path, expert, val, ff_policy, ff_metadata,
            config, seed, train_device, seed_root / "ff_dagger_round1", updates=temporal.dagger_updates)
        ff_checkpoint = Path(ff_dagger["checkpoint_path"])
        final._checkpoint_config_compatible(ff_checkpoint, config, GRAPH_NAME, seed)
        ff = sf.load_surface_model(ff_checkpoint, GRAPH_NAME, config, eval_device)
        set_seed(seed)
        gru = GraphGRUPolicy(config, temporal.gru_hidden).to(train_device)
        base = train_gru(gru, expert, val, None, config, temporal, seed, train_device,
                         temporal.base_updates, seed_root / "gru_base")
        gru = _load_gru(Path(base["checkpoint"]), config, temporal, eval_device)
        policy_data, collection = collect_recurrent_policy_data(
            gru, train_specs, config, temporal, seed, eval_device, root / "onpolicy_collection" / f"seed{seed}" / "gru")
        gru = gru.to(train_device)
        dagger = train_gru(gru, expert, val, policy_data, config, temporal, seed, train_device,
                           temporal.dagger_updates, seed_root / "gru_dagger_round1")
        gru = _load_gru(Path(dagger["checkpoint"]), config, temporal, eval_device)
        specs = sf.sample_episode_specs(config.eval_episodes, seed + 60_000, config, "iid")[:eval_episodes]
        condition = evaluate_condition(specs, ff, gru, config, temporal, eval_device, "fresh", 0,
                                       root / "fresh" / f"seed{seed}")
        rows = condition.pop("rows")
        for name in rows:
            pooled_rows[("fresh", 0)][name].extend(rows[name])
        aggregate["per_seed"][str(seed)] = {"ff_base_training": ff_base, "ff_dagger_training": ff_dagger,
                                            "ff_onpolicy_collection": ff_collection,
                                            "gru_base_training": base, "gru_dagger_training": dagger,
                                            "gru_onpolicy_collection": collection,
                                            "training_episode_count": len(train_specs),
                                            "conditions": {"fresh": {"0": condition}}}
        print(f"[recurrent] seed {seed}: fresh FF={condition['ff']['success']:.3f}, GRU={condition['gru']['success']:.3f}, collision FF={condition['ff']['collision']:.3f}, GRU={condition['gru']['collision']:.3f}", flush=True)
        checkpoint_map[seed] = (ff_checkpoint, Path(dagger["checkpoint"]))
        del ff, gru
    fresh_rows = pooled_rows[("fresh", 0)]
    fresh_ff, fresh_gru = _metric_means(fresh_rows["ff"]), _metric_means(fresh_rows["gru"])
    gate_checks = {
        "success_not_far_below_ff": fresh_gru["success"] >= fresh_ff["success"] - 0.10,
        "collision_not_far_above_ff": fresh_gru["collision"] <= fresh_ff["collision"] + 0.15,
        "position_within_1_5x_ff": fresh_gru["final_position_error"] <= 1.5 * fresh_ff["final_position_error"],
        "yaw_within_1_5x_ff": fresh_gru["final_orientation_error"] <= 1.5 * fresh_ff["final_orientation_error"],
    }
    gate_passed = all(gate_checks.values())
    aggregate["fresh_gate_checks"] = gate_checks
    print(f"[recurrent] pooled fresh gate: {gate_checks}", flush=True)
    if gate_passed or smoke:
        for seed in seeds:
            ff_checkpoint, gru_checkpoint = checkpoint_map[seed]
            ff = sf.load_surface_model(ff_checkpoint, GRAPH_NAME, config, eval_device)
            gru = _load_gru(gru_checkpoint, config, temporal, eval_device)
            specs = sf.sample_episode_specs(config.eval_episodes, seed + 60_000, config, "iid")[:eval_episodes]
            conditions = (("stale", (1,)),) if smoke else (("stale", temporal.stale_lengths),
                                                           ("delay", temporal.delay_lengths))
            for mode, levels in conditions:
                aggregate["per_seed"][str(seed)]["conditions"][mode] = {}
                for severity in levels:
                    if mode in ("stale", "delay") and severity == 0:
                        continue
                    row = evaluate_condition(specs, ff, gru, config, temporal, eval_device, mode, severity,
                                             root / mode / f"{mode}_{severity}" / f"seed{seed}")
                    result_rows = row.pop("rows")
                    for name in result_rows:
                        pooled_rows[(mode, severity)][name].extend(result_rows[name])
                    aggregate["per_seed"][str(seed)]["conditions"][mode][str(severity)] = row
                    print(f"[recurrent] seed {seed} {mode} {severity}: FF={row['ff']['success']:.3f}, GRU={row['gru']['success']:.3f}", flush=True)
            del ff, gru
    first_seed = seeds[0]
    first_ff, first_gru = checkpoint_map[first_seed]
    latency_spec = sf.sample_episode_specs(config.eval_episodes, first_seed + 60_000, config, "iid")[0]
    for name, model in (("ff", sf.load_surface_model(first_ff, GRAPH_NAME, config, eval_device)),
                        ("gru", _load_gru(first_gru, config, temporal, eval_device))):
        latency = benchmark_controller(model, name, latency_spec, config, temporal, eval_device)
        aggregate["latency"][name] = latency
        sf._write_json(root / "latency" / f"{name}.json", latency)
    for (mode, severity), rows in pooled_rows.items():
        pair = final.paired_comparison(rows["ff"], rows["gru"], config, "GRU - FF", 71_092 + severity)
        aggregate["pooled"].setdefault(mode, {})[str(severity)] = {
            "ff": _metric_means(rows["ff"]), "gru": _metric_means(rows["gru"]), "paired": pair,
        }
    if "fresh" in aggregate["pooled"]:
        aggregate["pooled"].setdefault("stale", {})["0"] = aggregate["pooled"]["fresh"]["0"]
        aggregate["pooled"].setdefault("delay", {})["0"] = aggregate["pooled"]["fresh"]["0"]
    ff_params = sf.model_parameter_count(sf.build_surface_model(GRAPH_NAME, config))
    gru_params = sf.model_parameter_count(GraphGRUPolicy(config, temporal.gru_hidden))
    ff_audit = parameter_audit(sf.build_surface_model(GRAPH_NAME, config), config)
    gru_audit = parameter_audit(GraphGRUPolicy(config, temporal.gru_hidden), config)
    aggregate["architecture"] = {"embedding_dim": config.hidden_dim * 3,
                                  "gru_hidden": temporal.gru_hidden, "gru_layers": 1,
                                  "action_dim": sf.ACTION_DIM, "ff_parameters": ff_params,
                                  "gru_parameters": gru_params, "parameter_ratio_gru_to_ff": gru_params / ff_params,
                                  "ff_parameter_audit": ff_audit, "gru_parameter_audit": gru_audit,
                                  "graph_initialization_seed_policy": "seed (same as historical FF base)",
                                  "learned_graph_weights_shared_between_controllers": False}
    fresh = aggregate["pooled"]["fresh"]["0"]
    aggregate["fresh_gate_passed"] = gate_passed
    aggregate["interpretation"] = (
        "Fresh-observation gate failed; temporal corruption results were stopped. This experiment does not establish recurrent robustness."
        if not gate_passed else
        "Fresh-observation behavior was established. Inspect paired confidence intervals and degradation curves before attributing differences to temporal memory; the GRU has substantially more parameters."
    )
    if gate_passed or smoke:
        _plot_condition_curves(aggregate, root / "plots" / "degradation_success.png")
    sf._write_json(root / "summary.json", aggregate)
    (root / "summary.md").write_text(_summary_markdown(aggregate), encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/recurrent_feasibility_canonical100"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--base-updates", type=int, default=800)
    parser.add_argument("--dagger-updates", type=int, default=800)
    parser.add_argument("--latency-warmup", type=int, default=200)
    parser.add_argument("--latency-iterations", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--validation-episodes", type=int)
    args = parser.parse_args()
    if args.eval_workers != 1:
        parser.error("Episode workers are currently serial; each MuJoCo environment remains process-local.")
    temporal = replace(TemporalConfig(), base_updates=args.base_updates,
                       dagger_updates=args.dagger_updates, latency_warmup=args.latency_warmup,
                       latency_iterations=args.latency_iterations)
    run_experiment(args.output, tuple(args.seeds), args.device, args.eval_device, temporal, args.eval_episodes,
                   args.smoke, args.train_episodes, args.validation_episodes)


if __name__ == "__main__":
    main()
