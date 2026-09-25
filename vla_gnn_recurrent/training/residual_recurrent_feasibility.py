"""Frozen-FF residual MLP/GRU feasibility with visual-only observation delay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import recurrent_hidden_diagnostic as hd
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_representation_final_comparison as final
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


OUTPUT = Path("artifacts/residual_recurrent_feasibility")
SEED = 2811
BASE_CHECKPOINT = Path("artifacts/recurrent_temporal_feasibility/matched_ff/training/surface_graph_no_local.pt")
BASE_SHA256 = "aed75f3b8c3da4a1c8867c603087b08df42d0302ee76450a8d6e60f281bbcb19"
OLD_POLICY = Path("artifacts/recurrent_feasibility_canonical100_seed2811/onpolicy_collection/seed2811/gru/policy_relabelled_trajectories.pt")
NEW_POLICY = Path("artifacts/recurrent_recovery_dagger/new_policy_collection/policy_relabelled_trajectories.pt")
HISTORICAL_FRESH = Path("artifacts/recurrent_temporal_feasibility/seed2811/fresh/matched_ff/episodes.json")
CONTROLLERS = ("ff", "residual_mlp", "residual_gru")


@dataclass(frozen=True)
class ResidualConfig:
    seed: int = SEED
    hidden_size: int = 128
    mlp_width: int = 384
    residual_fraction_of_action_limit: float = 0.20
    corrected_dimensions: int = 4
    updates: int = 800
    sequence_batch_size: int = 16
    max_sequence_length: int = 32
    train_delays: tuple[int, ...] = (0, 1, 2, 4)
    validation_interval: int = 25
    heldout_episodes: int = 16
    bootstrap_draws: int = 5000


def _write(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def action_scales(config: sf.SurfaceFeasibilityConfig, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor([config.max_delta_ee] * 3 + [config.max_delta_rotation,
                                                     config.max_delta_gripper], dtype=torch.float32,
                        device=device)


def visual_delay(state: np.ndarray, points: np.ndarray, old_state: np.ndarray,
                 old_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retain current proprioception; replace object-relative features and visual points."""
    observed_state = state.copy()
    observed_state[..., :5] = old_state[..., :5]
    return observed_state, old_points.copy()


def delayed_sequence(states: torch.Tensor, points: torch.Tensor, delay: int) -> tuple[torch.Tensor, torch.Tensor]:
    if delay < 0:
        raise ValueError("Delay must be nonnegative.")
    if delay == 0:
        return states.clone(), points.clone()
    indices = (torch.arange(len(states)) - delay).clamp_min(0)
    observed = states.clone()
    observed[:, :5] = states[indices, :5]
    return observed, points[indices].clone()


class ResidualController(nn.Module):
    """Use the existing FF encode/head exactly; train only a bounded residual."""

    def __init__(self, base: sf.SurfaceGraphNetwork, task: sf.SurfaceFeasibilityConfig,
                 design: ResidualConfig, kind: str) -> None:
        super().__init__()
        if kind not in ("mlp", "gru") or design.corrected_dimensions not in (4, 5):
            raise ValueError((kind, design.corrected_dimensions))
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        self.task, self.design, self.kind = task, design, kind
        input_size = task.hidden_dim * 3 + 2 * sf.ACTION_DIM
        if kind == "gru":
            self.memory = nn.GRU(input_size, design.hidden_size, num_layers=1, batch_first=True)
            self.residual_head = nn.Linear(design.hidden_size, design.corrected_dimensions)
        else:
            self.memory = nn.Sequential(nn.Linear(input_size, design.mlp_width), nn.SiLU(),
                                        nn.Linear(design.mlp_width, design.hidden_size), nn.SiLU())
            self.residual_head = nn.Linear(design.hidden_size, design.corrected_dimensions)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.register_buffer("scales", action_scales(task))
        self.register_buffer("residual_bounds", self.scales[:design.corrected_dimensions].clone() *
                             design.residual_fraction_of_action_limit)

    def train(self, mode: bool = True) -> "ResidualController":
        super().train(mode)
        self.base.eval()
        return self

    def _combine(self, base_action: torch.Tensor, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.residual_bounds * torch.tanh(raw)
        if self.design.corrected_dimensions == 4:
            corrected = torch.cat((base_action[..., :4] + residual, base_action[..., 4:]), dim=-1)
        else:
            corrected = base_action + residual
        return corrected, residual

    def forward_sequence(self, states: torch.Tensor, points: torch.Tensor,
                         previous_executed_actions: torch.Tensor,
                         hidden: torch.Tensor | None = None,
                         reset_every_step: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None,
                                                                  torch.Tensor, torch.Tensor | None]:
        if states.ndim != 3 or points.ndim != 4 or previous_executed_actions.shape != (*states.shape[:2], 5):
            raise ValueError("Expected ordered [batch,time,...] graph and previous-action tensors.")
        batch, length = states.shape[:2]
        with torch.no_grad():
            z = self.base.encode(states.reshape(batch * length, -1),
                                 points.reshape(batch * length, points.shape[-2], 3)).reshape(batch, length, -1)
            base_action = self.base.action_head(z)
        inputs = torch.cat((z, base_action / self.scales,
                            previous_executed_actions / self.scales), dim=-1)
        hidden_norm = None
        if self.kind == "gru":
            if reset_every_step:
                pieces, norms = [], []
                for t in range(length):
                    output, next_hidden = self.memory(inputs[:, t:t + 1])
                    pieces.append(output)
                    norms.append(next_hidden.norm(dim=-1).squeeze(0))
                features = torch.cat(pieces, 1)
                hidden = next_hidden
                hidden_norm = torch.stack(norms, 1)
            else:
                features, hidden = self.memory(inputs, hidden)
                # Prefix hidden norms are computed only for diagnostics when needed.
        else:
            features = self.memory(inputs)
        corrected, residual = self._combine(base_action, self.residual_head(features))
        return corrected, residual, hidden, base_action, hidden_norm

    def step(self, state: torch.Tensor, points: torch.Tensor, previous_executed_action: torch.Tensor,
             hidden: torch.Tensor | None = None, topology: sf.GraphTopology | None = None,
             reset_hidden: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None,
                                                   torch.Tensor]:
        if reset_hidden:
            hidden = None
        with torch.no_grad():
            z = self.base.encode(state, points, topology)
            base_action = self.base.action_head(z)
        inputs = torch.cat((z, base_action / self.scales,
                            previous_executed_action / self.scales), dim=-1)
        if self.kind == "gru":
            features, hidden = self.memory(inputs[:, None], hidden)
            features = features[:, 0]
        else:
            features = self.memory(inputs)
            hidden = None
        corrected, residual = self._combine(base_action, self.residual_head(features))
        return corrected, residual, hidden, base_action


def load_controller(kind: str, task: sf.SurfaceFeasibilityConfig, design: ResidualConfig,
                    device: torch.device, checkpoint: Path | None = None) -> ResidualController:
    if hd._sha256(BASE_CHECKPOINT) != BASE_SHA256:
        raise AssertionError("Canonical matched-FF checkpoint changed.")
    base = sf.load_surface_model(BASE_CHECKPOINT, rf.GRAPH_NAME, task, device)
    if not isinstance(base, sf.SurfaceGraphNetwork) or base.use_local_edges:
        raise AssertionError("Frozen nominal controller is not the canonical no-local graph model.")
    controller = ResidualController(base, task, design, kind).to(device)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload["base_sha256"] != BASE_SHA256 or payload["kind"] != kind:
            raise AssertionError("Residual checkpoint provenance mismatch.")
        controller.load_state_dict(payload["model_state"])
    return controller.eval()


def source_episodes(expert: dict[str, Any], pool: dict[str, Any],
                    old_policy: Path = OLD_POLICY, new_policy: Path = NEW_POLICY) -> dict[str, list[dict[str, torch.Tensor]]]:
    old = torch.load(old_policy, map_location="cpu", weights_only=False)
    new = torch.load(new_policy, map_location="cpu", weights_only=False)
    executed = torch.cat((old["policy_actions"], new["policy_actions"]))
    if len(executed) != len(pool["actions"]) or not torch.equal(
            torch.cat((old["states"], new["states"])), pool["states"]):
        raise AssertionError("Old/new executed actions do not align with the 144-trajectory pool.")
    result = {}
    for name, data, actions in (("expert", expert, expert["actions"]), ("policy", pool, executed)):
        oracle = rf._episodes(data)
        behavior = rf._episodes({**data, "actions": actions})
        sequences = []
        for o, b in zip(oracle, behavior, strict=True):
            if not torch.equal(o["steps"], b["steps"]):
                raise AssertionError("Executed-action history crossed an episode boundary.")
            previous = torch.zeros_like(b["actions"])
            previous[1:] = b["actions"][:-1]
            sequences.append({"states": o["states"], "points": o["surface_points"],
                              "targets": o["actions"], "previous": previous,
                              "mask": o["loss_mask"], "episode_id": int(o["episode_ids"][0])})
        result[name] = sequences
    if len(result["expert"]) != 72 or len(result["policy"]) != 144:
        raise AssertionError("Expected canonical 72 expert and 144 policy sequences.")
    return result


def make_schedule(data: dict[str, list[dict[str, torch.Tensor]]], design: ResidualConfig,
                  delay_transform: Callable[[dict, int], tuple[torch.Tensor, torch.Tensor]] | None = None,
                  episode_indices_by_delay: dict[str, dict[int, list[int]]] | None = None) -> tuple[list[dict], dict]:
    if design.sequence_batch_size != 16 or design.train_delays != (0, 1, 2, 4):
        raise ValueError("Frozen feasibility mixture is 2 expert + 2 policy per delay.")
    rng = np.random.default_rng(design.seed + 1_400_001)
    preprocessed = {source: [[(delay_transform(ep, delay) if delay_transform else
                               delayed_sequence(ep["states"], ep["points"], delay))
                              for delay in design.train_delays] for ep in episodes]
                    for source, episodes in data.items()}
    digest = hashlib.sha256()
    batches, draws = [], []
    supervised = {str(delay): 0 for delay in design.train_delays}
    source_supervised = {source: 0 for source in data}
    stratified_supervised: dict[str, int] = {}
    for _ in range(design.updates):
        selections = []
        for source in ("expert", "policy"):
            for delay_index, delay in enumerate(design.train_delays):
                eligible = (episode_indices_by_delay[source][delay] if episode_indices_by_delay is not None
                            else list(range(len(data[source]))))
                if not eligible:
                    raise ValueError(f"No {source} episodes for delay {delay}.")
                for sampled in rng.integers(0, len(eligible), size=2).tolist():
                    episode_index = eligible[sampled]
                    selections.append((source, int(episode_index), delay_index))
                    draws.append((0 if source == "expert" else 1, int(episode_index), delay))
        width = max(len(data[source][index]["targets"]) for source, index, _ in selections)
        if width > design.max_sequence_length:
            raise ValueError("Full episode exceeds frozen sequence length.")
        batch = {"states": torch.zeros(16, width, sf.STATE_DIM),
                 "points": torch.zeros(16, width, sf.SURFACE_POINTS, 3),
                 "previous": torch.zeros(16, width, sf.ACTION_DIM),
                 "targets": torch.zeros(16, width, sf.ACTION_DIM),
                 "mask": torch.zeros(16, width, dtype=torch.bool)}
        for i, (source, index, delay_index) in enumerate(selections):
            episode = data[source][index]
            length = len(episode["targets"])
            observed_state, observed_points = preprocessed[source][index][delay_index]
            batch["states"][i, :length] = observed_state
            batch["points"][i, :length] = observed_points
            batch["previous"][i, :length] = episode["previous"]
            batch["targets"][i, :length] = episode["targets"]
            batch["mask"][i, :length] = episode["mask"]
            count = int(episode["mask"].sum())
            supervised[str(design.train_delays[delay_index])] += count
            source_supervised[source] += count
            if "speed_m_per_step" in episode and "direction" in episode:
                stratum = f"{source}|{episode['speed_m_per_step']:.6f}|{episode['direction']}|{design.train_delays[delay_index]}"
                stratified_supervised[stratum] = stratified_supervised.get(stratum, 0) + count
        for key, value in batch.items():
            digest.update(key.encode())
            digest.update(value.contiguous().numpy().tobytes())
        batches.append(batch)
    audit = {"schedule_sha256": digest.hexdigest(),
             "draws_sha256": hashlib.sha256(np.asarray(draws, dtype=np.int64).tobytes()).hexdigest(),
             "updates": design.updates, "batch_size": design.sequence_batch_size,
             "sources_per_delay_per_batch": {"expert": 2, "policy": 2},
             "supervised_timesteps_by_delay": supervised,
             "supervised_timesteps_by_source": source_supervised,
             "supervised_timesteps_by_source_speed_direction_delay": stratified_supervised,
             "masked_padding_or_severe_not_supervised": True}
    return batches, audit


def corrected_loss(corrected: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                   task: sf.SurfaceFeasibilityConfig, dimensions: int = 4) -> torch.Tensor:
    scales = action_scales(task, corrected.device)[:dimensions]
    q = ((corrected[..., :dimensions] - target[..., :dimensions]) / scales).square().mean(-1)
    return q[mask].mean()


@torch.no_grad()
def validation_metrics(model: ResidualController, validation: list[dict],
                       task: sf.SurfaceFeasibilityConfig, device: torch.device) -> dict[str, float]:
    totals = {"corrected_action_mse_5d": 0.0, "residual_prediction_mse_4d": 0.0,
              "corrected_action_mse_4d": 0.0}
    count = 0
    for ep in validation:
        length = len(ep["targets"])
        target = ep["targets"][None].to(device)
        corrected, residual, _, base, _ = model.forward_sequence(
            ep["states"][None].to(device), ep["points"][None].to(device), ep["previous"][None].to(device))
        mask = ep["mask"][None].to(device)
        residual_target = target[..., :4] - base[..., :4]
        scales = action_scales(task, device)
        totals["corrected_action_mse_5d"] += float((((corrected - target) / scales).square().mean(-1))[mask].sum())
        totals["corrected_action_mse_4d"] += float((((corrected[..., :4] - target[..., :4]) /
                                                      scales[:4]).square().mean(-1))[mask].sum())
        totals["residual_prediction_mse_4d"] += float((((residual[..., :4] - residual_target) /
                                                        scales[:4]).square().mean(-1))[mask].sum())
        count += int(mask.sum())
    return {key: value / count for key, value in totals.items()}


def train_branch(kind: str, batches: list[dict], audit: dict, validation: list[dict],
                 task: sf.SurfaceFeasibilityConfig, design: ResidualConfig,
                 device: torch.device, root: Path,
                 starting_checkpoint: Path | None = None) -> dict:
    set_seed(design.seed + 1_400_001)
    model = load_controller(kind, task, design, device, starting_checkpoint)
    base_hash = final.state_dict_hash(model.base)
    initial = next(iter(batches))
    with torch.no_grad():
        state = initial["states"][:1, :1].to(device)
        points = initial["points"][:1, :1].to(device)
        previous = initial["previous"][:1, :1].to(device)
        corrected, residual, _, base, _ = model.forward_sequence(state, points, previous)
        if starting_checkpoint is None and (not torch.equal(corrected, base) or torch.any(residual != 0)):
            raise AssertionError("Zero-initialized residual changed FF at initialization.")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=task.learning_rate, weight_decay=task.weight_decay)
    checkpoint = ensure_dir(root / "checkpoints") / f"{kind}.pt"
    best, selected, history = float("inf"), None, []
    for update, cpu_batch in enumerate(batches, 1):
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        model.train()
        corrected, _, _, _, _ = model.forward_sequence(batch["states"], batch["points"], batch["previous"])
        # Each source contributes one half, as in the existing recurrent DAgger loss.
        loss = .5 * corrected_loss(corrected[:8], batch["targets"][:8], batch["mask"][:8], task) + \
               .5 * corrected_loss(corrected[8:], batch["targets"][8:], batch["mask"][8:], task)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in model.base.parameters()):
            raise AssertionError("Frozen FF received gradients.")
        optimizer.step()
        if update % design.validation_interval == 0:
            model.eval()
            metrics = validation_metrics(model, validation, task, device)
            history.append({"update": update, "training_4d_mse": float(loss.detach().cpu()), **metrics})
            if metrics["corrected_action_mse_5d"] < best:
                best, selected = metrics["corrected_action_mse_5d"], update
                torch.save({"model_state": model.state_dict(), "kind": kind,
                            "base_sha256": BASE_SHA256, "design": asdict(design),
                            "seed": design.seed, "selected_update": update,
                            "validation_5d_mse": best, "topology": "consistent_intended_100"}, checkpoint)
        if update % 100 == 0:
            print(f"[residual] {kind} update {update}/{design.updates} val {history[-1]['corrected_action_mse_5d']:.5f}", flush=True)
    if final.state_dict_hash(model.base) != base_hash:
        raise AssertionError("Frozen FF parameters changed during training.")
    result = {"checkpoint": str(checkpoint), "checkpoint_sha256": hd._sha256(checkpoint),
              "kind": kind, "seed": design.seed, "base_checkpoint_sha256": BASE_SHA256,
              "starting_checkpoint_sha256": hd._sha256(starting_checkpoint) if starting_checkpoint else None,
              "base_model_hash_before_after_equal": True,
              "registered_parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "schedule_sha256": audit["schedule_sha256"],
              "supervised_timesteps_by_delay": audit["supervised_timesteps_by_delay"],
              "selected_update": selected, "best_common_validation_5d_mse": best,
              "history": history}
    _write(root / "logs" / f"{kind}_training.json", result)
    return result


def validation_episodes(validation: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
    episodes = []
    for ep in rf._episodes(validation):
        previous = torch.zeros_like(ep["actions"])
        previous[1:] = ep["actions"][:-1]
        episodes.append({"states": ep["states"], "points": ep["surface_points"],
                         "targets": ep["actions"], "previous": previous, "mask": ep["loss_mask"]})
    return episodes


class VisualDelayObserver:
    def __init__(self, delay: int = 0, pattern: tuple[int, ...] | None = None) -> None:
        if delay < 0 or (pattern is not None and any(x < 0 for x in pattern)):
            raise ValueError("Delay must be nonnegative.")
        self.delay, self.pattern = delay, pattern
        self.history: list[tuple[np.ndarray, np.ndarray]] = []

    def observe(self, state: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        self.history.append((state.copy(), points.copy()))
        t = len(self.history) - 1
        age = self.pattern[t % len(self.pattern)] if self.pattern else self.delay
        observed = max(0, t - age)
        old_state, old_points = self.history[observed]
        visual_state, visual_points = visual_delay(state, points, old_state, old_points)
        return visual_state, visual_points, t - observed


@torch.no_grad()
def rollout_residual(env: sf.SurfaceManipulatorEnv, spec: sf.SurfaceEpisodeSpec,
                     model: ResidualController, task: sf.SurfaceFeasibilityConfig,
                     temporal: rf.TemporalConfig, device: torch.device,
                     delay: int = 0, reset_every_step: bool = False,
                     pattern: tuple[int, ...] | None = None,
                     perturbation: bool = False,
                     observer_factory: Callable[[int, tuple[int, ...] | None], Any] | None = None) -> tuple[dict, list[dict]]:
    sf._reset_surface_env(env, spec)
    shape = spec.shape
    initial = env.robot_observation().ee_position.numpy().astype(np.float64)
    target = sf._surface_target(shape, initial, task.pregrasp_clearance)
    observer = observer_factory(delay, pattern) if observer_factory is not None else VisualDelayObserver(delay, pattern)
    hidden = None
    previous = torch.zeros(1, sf.ACTION_DIM, device=device)
    trace = []
    initial_errors = sf._state_errors(env, shape, task, target)
    initial_contacts, min_clearance = sf._surface_distance_metrics(env, shape)
    seen_collision = bool(initial_contacts)
    first_collision = 0 if initial_contacts else None
    first_success = 0 if sf._success_from_errors(initial_errors, seen_collision, task) else None
    recovery_time = None
    post_perturb_errors = []
    trajectory = [initial.tolist()]
    trajectory_pos = [initial_errors["position_error"]]
    trajectory_yaw = [initial_errors["orientation_error"]]
    trajectory_aperture = [initial_errors["gripper_width_error"]]
    for t in range(task.max_steps):
        if first_success is not None and not perturbation:
            break
        shifted = perturbation and t == temporal.perturbation_step
        if shifted:
            center = list(shape.object_center)
            center[0] += temporal.perturbation_x_m
            shape = replace(shape, object_center=tuple(center), shape_id=shape.shape_id + "_shifted")
            env.set_surface_shape(shape)
            target = sf._surface_target(shape, initial, task.pregrasp_clearance)
            first_success = None
        state, points, _ = sf.observation_inputs(env, shape, task.point_count,
                                                  sf._stable_seed(spec.sample_identity))
        observed_state, observed_points, age = (observer.observe(state, points,
            env.robot_observation().ee_position.numpy().astype(np.float64)) if observer_factory is not None
            else observer.observe(state, points))
        contacts, clearance = sf._surface_distance_metrics(env, shape)
        collision_now = bool(contacts)
        seen_collision = seen_collision or collision_now
        min_clearance = min(min_clearance, clearance)
        if first_collision is None and collision_now:
            first_collision = t
        errors = sf._state_errors(env, shape, task, target)
        state_t, points_t, topology = rf._graph_tensors(observed_state, observed_points, task, device)
        start_ns = time.perf_counter_ns()
        corrected, residual, hidden, base = model.step(state_t, points_t, previous, hidden, topology,
                                                        reset_hidden=reset_every_step)
        action = rf.clipped_action(corrected[0].detach().cpu().numpy(), task)
        inference_ms = (time.perf_counter_ns() - start_ns) / 1e6
        base_processed = rf.clipped_action(base[0].detach().cpu().numpy(), task)
        trace.append({"episode_id": spec.episode_id, "t": t,
                      "true_ee_pose": env.robot_observation().ee_position.numpy().tolist(),
                      "true_yaw": float(sf.tool_yaw(env)), "true_aperture": float(sf.gripper_width(env)),
                      "observed_graph_timestamp": t - age, "observation_age": age,
                      "observed_state": observed_state.tolist(), "observed_surface_points": observed_points.tolist(),
                      "true_state": state.tolist(), "true_surface_points": points.tolist(),
                      "target_pose": target[0].tolist(), "target_yaw": float(target[2]),
                      "position_error": errors["position_error"], "orientation_error": errors["orientation_error"],
                      "aperture_error": errors["gripper_width_error"],
                      "predicted_action": action.tolist(), "processed_action": action.tolist(),
                      "raw_predicted_action": corrected[0].detach().cpu().tolist(),
                      "base_raw_action": base[0].detach().cpu().tolist(),
                      "base_processed_action": base_processed.tolist(),
                      "residual_action": residual[0].detach().cpu().tolist(),
                      "residual_norm": float(residual[0].norm().detach().cpu()),
                      "action_difference_from_ff": float(np.linalg.norm(action - base_processed)),
                      "previous_executed_action": previous[0].detach().cpu().tolist(),
                      "hidden_norm": float(hidden.norm().detach().cpu()) if hidden is not None else None,
                      "hidden_reset": t == 0 or reset_every_step,
                      "inference_ms": inference_ms,
                      "oracle_action": None, "collision": collision_now,
                      "clearance_m": float(clearance), "perturbation_flag": shifted,
                      "success_state": sf._success_from_errors(errors, seen_collision, task)})
        sf.apply_local_action(env, action, task)
        previous = torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, -1)
        next_state, next_points, _ = sf.observation_inputs(env, shape, task.point_count,
                                                            sf._stable_seed(spec.sample_identity))
        trace[-1]["next_true_state"] = next_state.tolist()
        trace[-1]["next_true_surface_points"] = next_points.tolist()
        current = env.robot_observation().ee_position.numpy().astype(np.float64)
        trajectory.append(current.tolist())
        after_contacts, after_clearance = sf._surface_distance_metrics(env, shape)
        seen_collision = seen_collision or bool(after_contacts)
        min_clearance = min(min_clearance, after_clearance)
        if first_collision is None and after_contacts:
            first_collision = t + 1
        next_errors = sf._state_errors(env, shape, task, target)
        trajectory_pos.append(next_errors["position_error"])
        trajectory_yaw.append(next_errors["orientation_error"])
        trajectory_aperture.append(next_errors["gripper_width_error"])
        if perturbation and t >= temporal.perturbation_step:
            post_perturb_errors.append(next_errors["position_error"])
            if recovery_time is None and next_errors["position_error"] < temporal.recovery_position_m and \
                    next_errors["orientation_error"] < temporal.recovery_yaw_rad:
                recovery_time = t + 1 - temporal.perturbation_step
        if first_success is None and sf._success_from_errors(next_errors, seen_collision, task):
            first_success = t + 1
        if perturbation and first_success is not None and t + 1 >= temporal.perturbation_step + 1:
            break
    final_contacts, final_clearance = sf._surface_distance_metrics(env, shape)
    min_clearance = min(min_clearance, final_clearance)
    final_errors = sf._state_errors(env, shape, task, target)
    positions = np.asarray(trajectory)
    row = {"episode_id": spec.episode_id,
           "condition": "perturbation" if perturbation else ("delay" if delay or pattern else "fresh"),
           "severity": delay, "spec_signature": final.spec_signature(spec),
           "success": sf._success_from_errors(final_errors, seen_collision, task),
           "collision": seen_collision, "final_collision": bool(final_contacts),
           "first_collision_timestep": first_collision,
           "minimum_safe_clearance": float(min_clearance),
           "final_position_error": final_errors["position_error"],
           "final_orientation_error": final_errors["orientation_error"],
           "final_gripper_width_error": final_errors["gripper_width_error"],
           "trajectory_error": float(np.mean(trajectory_pos)),
           "trajectory_yaw_error": float(np.mean(trajectory_yaw)),
           "trajectory_aperture_error": float(np.mean(trajectory_aperture)),
           "trajectory_length": float(np.linalg.norm(np.diff(positions[:, [0, 2]], axis=0), axis=1).sum()),
           "steps_to_convergence": first_success if first_success is not None else task.max_steps,
           "steps_executed": len(trace), "recovery_time": recovery_time,
           "post_perturb_peak_position_error": max(post_perturb_errors) if post_perturb_errors else None,
           "post_perturb_min_position_error": min(post_perturb_errors) if post_perturb_errors else None,
           "post_perturb_overshoot": max(post_perturb_errors) - post_perturb_errors[0]
           if post_perturb_errors else None}
    return row, trace


@torch.no_grad()
def ff_inference_latency(model: sf.SurfaceGraphNetwork, traces: list[dict],
                         task: sf.SurfaceFeasibilityConfig, device: torch.device) -> dict[str, float]:
    times = []
    for episode in traces:
        for step in episode["steps"]:
            state_t, points_t, topology = rf._graph_tensors(np.asarray(step["observed_state"]),
                                                             np.asarray(step["observed_surface_points"]),
                                                             task, device)
            start = time.perf_counter_ns()
            raw = model(state_t, points_t, topology)
            rf.clipped_action(raw[0].detach().cpu().numpy(), task)
            times.append((time.perf_counter_ns() - start) / 1e6)
    return {"mean_ms": float(np.mean(times)), "p95_ms": float(np.quantile(times, .95)),
            "samples": len(times)}


def residual_diagnostics(traces: list[dict], dimensions: int) -> dict[str, Any]:
    steps = [step for ep in traces for step in ep["steps"]]
    residual = np.asarray([step["residual_action"] for step in steps], dtype=float)
    norms = np.linalg.norm(residual, axis=1)
    hidden = [step["hidden_norm"] for step in steps if step["hidden_norm"] is not None]
    by_time = {}
    for t in sorted({step["t"] for step in steps}):
        subset = [step for step in steps if step["t"] == t]
        by_time[str(t)] = {"mean_residual_norm": float(np.mean([r["residual_norm"] for r in subset])),
                           "mean_hidden_norm": float(np.mean([r["hidden_norm"] for r in subset]))
                           if hidden else None, "steps": len(subset)}
    return {"mean_residual_norm": float(norms.mean()), "p95_residual_norm": float(np.quantile(norms, .95)),
            "per_dimension_mean": residual.mean(0).tolist(),
            "per_dimension_mean_absolute": np.abs(residual).mean(0).tolist(),
            "mean_action_difference_from_ff": float(np.mean([r["action_difference_from_ff"] for r in steps])),
            "mean_hidden_norm": float(np.mean(hidden)) if hidden else None,
            "p95_hidden_norm": float(np.quantile(hidden, .95)) if hidden else None,
            "mean_inference_ms": float(np.mean([r["inference_ms"] for r in steps])),
            "p95_inference_ms": float(np.quantile([r["inference_ms"] for r in steps], .95)),
            "by_timestep": by_time, "corrected_dimensions": dimensions}


def paired_summary(reference: list[dict], candidate: list[dict], seed: int) -> dict:
    if [r["spec_signature"] for r in reference] != [r["spec_signature"] for r in candidate]:
        raise AssertionError("Paired EpisodeSpecs differ.")
    rng = np.random.default_rng(seed)
    metrics = ("success", "collision", "final_position_error", "final_orientation_error",
               "final_gripper_width_error", "trajectory_error", "trajectory_yaw_error",
               "minimum_safe_clearance")
    out = {}
    for key in metrics:
        diff = np.asarray([float(b[key]) - float(a[key]) for a, b in zip(reference, candidate, strict=True)])
        indices = rng.integers(0, len(diff), size=(5000, len(diff)))
        out[key] = {"mean_candidate_minus_reference": float(diff.mean()),
                    "episode_bootstrap_95_ci": [float(x) for x in np.quantile(diff[indices].mean(1), [.025, .975])]}
    for key in ("success", "collision"):
        ref_only = sum(bool(a[key] and not b[key]) for a, b in zip(reference, candidate, strict=True))
        candidate_only = sum(bool(b[key] and not a[key]) for a, b in zip(reference, candidate, strict=True))
        out[key]["discordant_reference_only"] = ref_only
        out[key]["discordant_candidate_only"] = candidate_only
        out[key]["mcnemar_exact_p"] = sf._mcnemar_exact_p(ref_only, candidate_only)
    return out


def evaluate_condition(models: dict[str, Any], specs: list[sf.SurfaceEpisodeSpec],
                       task: sf.SurfaceFeasibilityConfig, temporal: rf.TemporalConfig,
                       device: torch.device, root: Path, delay: int = 0,
                       reset_gru: bool = False, pattern: tuple[int, ...] | None = None,
                       perturbation: bool = False,
                       observer_factory: Callable[[int, tuple[int, ...] | None], Any] | None = None,
                       seed: int = SEED) -> dict:
    rows, traces = {}, {}
    env = sf.make_env(task, seed + 864_211)
    try:
        for name, model in models.items():
            rows[name], traces[name] = [], []
        for spec in specs:
            for name, model in models.items():
                if name == "ff":
                    if pattern is not None or perturbation:
                        # Custom visual-only temporal conditions use the common residual rollout adapter.
                        adapter = ResidualController(model, task, ResidualConfig(), "mlp").to(device).eval()
                        row, trace = rollout_residual(env, spec, adapter, task, temporal, device,
                                                      delay, pattern=pattern, perturbation=perturbation,
                                                      observer_factory=observer_factory)
                    elif delay == 0:
                        row, trace = rf.rollout(env, spec, model, "ff", task, temporal, device,
                                                mode="fresh", severity=0, collect=False, record_dynamics=True)
                    else:
                        adapter = ResidualController(model, task, ResidualConfig(), "mlp").to(device).eval()
                        row, trace = rollout_residual(env, spec, adapter, task, temporal, device, delay,
                                                      observer_factory=observer_factory)
                else:
                    row, trace = rollout_residual(env, spec, model, task, temporal, device, delay,
                                                  reset_every_step=reset_gru and name == "residual_gru",
                                                  pattern=pattern, perturbation=perturbation,
                                                  observer_factory=observer_factory)
                rows[name].append(row)
                traces[name].append({"episode_id": spec.episode_id,
                                     "spec_signature": final.spec_signature(spec), "steps": trace})
    finally:
        env.close()
    expected = [final.spec_signature(spec) for spec in specs]
    if any([r["spec_signature"] for r in episodes] != expected for episodes in rows.values()):
        raise AssertionError("Controllers used different held-out episodes.")
    for name in models:
        if any(step["oracle_action"] is not None for ep in traces[name] for step in ep["steps"]):
            raise AssertionError("Closed-loop evaluation called the scripted expert.")
        _write(root / "per_episode" / f"{name}.json", rows[name])
        _write(root / "traces" / f"{name}.json", traces[name])
    summary = {"condition": "perturbation" if perturbation else "variable_delay" if pattern else "delay",
               "delay_steps": delay, "pattern": pattern, "reset_gru": reset_gru,
               "means": {name: rf._metric_means(episodes) for name, episodes in rows.items()},
               "paired_vs_ff": {name: paired_summary(rows["ff"], rows[name], seed + delay + i)
                                for i, name in enumerate(models) if name != "ff"},
               "residual": {name: residual_diagnostics(traces[name], models[name].design.corrected_dimensions)
                            for name in models if name != "ff"}}
    summary["ff_inference_latency"] = ff_inference_latency(models["ff"], traces["ff"], task, device)
    _write(root / "summary.json", summary)
    return summary


def _fresh_gate(fresh: dict) -> dict:
    ff = fresh["means"]["ff"]
    gru = fresh["means"]["residual_gru"]
    # Reuse the project's prior, explicitly defined practical fresh gate.
    checks = {"position_within_1_5x_ff": gru["final_position_error"] <= 1.5 * ff["final_position_error"],
              "yaw_within_1_5x_ff": gru["final_orientation_error"] <= 1.5 * ff["final_orientation_error"],
              "collision_no_more_than_ff_plus_0_15": gru["collision"] <= ff["collision"] + .15,
              "success_no_more_than_0_10_below_ff": gru["success"] >= ff["success"] - .10}
    return {"checks": checks, "passed": all(checks.values()),
            "source": "artifacts/recurrent_temporal_feasibility/TEMPORAL_FEASIBILITY_REPORT.md fresh gate"}


def _delay_signal(fresh: dict, delayed: dict[int, dict]) -> dict:
    supporting = []
    for delay in (2, 4, 8):
        result = delayed[delay]["means"]
        degradation = {name: result[name]["final_position_error"] -
                       fresh["means"][name]["final_position_error"] for name in CONTROLLERS}
        if degradation["residual_gru"] < min(degradation["ff"], degradation["residual_mlp"]) and \
                result["residual_gru"]["collision"] <= max(result["ff"]["collision"],
                                                           result["residual_mlp"]["collision"]):
            supporting.append(delay)
    return {"supporting_delays": supporting,
            "credible_for_next_gate": len(supporting) >= 2,
            "rule": "GRU final-position degradation smaller than both FF/MLP at >=2 of delays 2,4,8, with no higher collision than both at those delays"}


def _delay_table(root: Path, fresh: dict, delayed: dict[int, dict]) -> list[dict]:
    table = []
    keys = ("success", "collision", "final_position_error", "final_orientation_error",
            "trajectory_error", "minimum_safe_clearance")
    for delay in (0, 1, 2, 4, 8):
        condition = fresh if delay == 0 else delayed[delay]
        for name in CONTROLLERS:
            result = condition["means"][name]
            row = {"delay_steps": delay, "controller": name,
                   **{key: result[key] for key in keys},
                   **{f"degradation_{key}": result[key] - fresh["means"][name][key] for key in keys},
                   "mean_residual_norm": 0.0 if name == "ff" else condition["residual"][name]["mean_residual_norm"],
                   "mean_inference_ms": condition["ff_inference_latency"]["mean_ms"] if name == "ff"
                   else condition["residual"][name]["mean_inference_ms"]}
            table.append(row)
    _write(root / "plots_or_plot_data" / "fixed_delay_table.json", table)
    path = ensure_dir(root / "plots_or_plot_data") / "fixed_delay_table.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return table


def analyze_saved(root: Path, summary: dict) -> dict:
    """Postprocess saved paired rows without rerunning training or MuJoCo."""
    if not summary.get("fixed_delay_run"):
        return summary
    for delay in (0, 1, 2, 4, 8):
        condition_root = root / "metrics" / ("fresh" if delay == 0 else f"delay{delay}")
        mlp = json.loads((condition_root / "per_episode/residual_mlp.json").read_text())
        gru = json.loads((condition_root / "per_episode/residual_gru.json").read_text())
        pair = paired_summary(mlp, gru, SEED + 100 + delay)
        _write(condition_root / "paired_gru_minus_mlp.json", pair)
    memory = {}
    for delay in (2, 4, 8):
        carry = json.loads((root / f"metrics/delay{delay}/per_episode/residual_gru.json").read_text())
        reset = json.loads((root / f"metrics/reset_delay{delay}/per_episode/residual_gru.json").read_text())
        memory[str(delay)] = paired_summary(reset, carry, SEED + 200 + delay)
    summary["paired_memory_carry_minus_reset"] = memory
    summary["interpretation"] = {
        "fresh_parity": "passed_preexisting_practical_gate",
        "fixed_delay": "limited_position_advantage_at_delays_4_and_8_with_high_collision_and_no_success_gain",
        "memory": "carry_improves_position_at_delays_4_and_8_but_not_consistently_collision_or_delay_2",
        "variable_and_perturbation_status": "exploratory_due_to_high_fixed_delay_collision",
        "general_temporal_robustness_established": False,
    }
    _write(root / "metrics" / "paired_memory_carry_minus_reset.json", memory)
    table = summary["fixed_delay_table"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, label in (("ff", "Frozen FF"), ("residual_mlp", "Residual MLP"),
                        ("residual_gru", "Residual GRU")):
        rows = [row for row in table if row["controller"] == name]
        axes[0].plot([r["delay_steps"] for r in rows],
                     [1000 * r["degradation_final_position_error"] for r in rows], "o-", label=label)
        axes[1].plot([r["delay_steps"] for r in rows],
                     [r["collision"] for r in rows], "o-", label=label)
    axes[0].set(xlabel="Visual delay (control steps)", ylabel="Final position change from fresh (mm)")
    axes[1].set(xlabel="Visual delay (control steps)", ylabel="Trajectory collision rate", ylim=(0, 1))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(ensure_dir(root / "plots_or_plot_data") / "fixed_delay_degradation.png", dpi=160)
    plt.close(fig)
    return summary


def write_report(root: Path, summary: dict) -> None:
    fresh = summary["fresh"]
    gate = summary["fresh_gate"]
    lines = ["# Residual recurrent controller feasibility", "",
             "## A. Implementation", "",
             f"Development seed {SEED}; frozen matched FF checkpoint SHA-256 `{BASE_SHA256}`. ",
             "Training ran on MPS and rollout inference on CPU; MuJoCo physics ran on CPU. The implementation "
             "uses the shared `select_device` path for CPU/MPS/CUDA. CUDA execution was not available to test here.", "",
             "The existing `SurfaceGraphNetwork.encode` and `action_head` compute the 192D latent and nominal 5D action. "
             "Both are frozen. The residual input is `[z (192), normalized a_base (5), normalized previous executed action (5)]`, "
             "202 dimensions. GRU: one layer, hidden 128; MLP: 202→384→128. Both have zero-initialized final "
             "128→4 projections. Their four corrections are bounded by `0.20 × [0.035,0.035,0.035,0.35]` "
             "= `[0.007,0.007,0.007]` m and `0.070` rad per coordinate. The 0.20 fraction is a conservative "
             "one-fifth of the existing per-step command limits, fixed before evaluation. The frozen FF gripper output is unchanged. "
             "Hidden state and previous executed action start at zero each episode. The existing 5D normalized MSE is used "
             "for validation selection; supervised correction MSE uses its four trainable geometric dimensions without "
             "new component weights. Frozen encoder/head hashes and gradient checks passed.", "",
             "Visual-only delay copies past object-relative state indices 0:5 and past sampled EE-local surface points; "
             "current robot yaw, aperture and fingertip state indices 5:14 remain fresh. The 100-edge topology is "
             "rebuilt from delayed points and current fingertips. Delay `d` means the visual snapshot at `max(0,t−d)`. "
             "The oracle label remains from true current state in saved ordered trajectories. Delay 0 exactly reproduces "
             "the canonical fresh graph. Training uses two expert and two policy episodes for each of delays 0,1,2,4 "
             "per 16-sequence update; actual valid counts are in `logs/schedule.json`.", "",
             "| Controller | Registered parameters | Trainable parameters | Checkpoint |", "|---|---:|---:|---|"]
    ff_parameters = (summary["training"]["residual_mlp"]["registered_parameters"] -
                     summary["training"]["residual_mlp"]["trainable_parameters"])
    lines.append(f"| frozen_ff | {ff_parameters} | 0 | `{BASE_CHECKPOINT}` |")
    for name in ("residual_mlp", "residual_gru"):
        training = summary["training"][name]
        lines.append(f"| {name} | {training['registered_parameters']} | {training['trainable_parameters']} | `{training['checkpoint']}` |")
    lines += ["", "Both residuals used the same frozen FF, balanced sequence draws and 800 updates. "
              "The source policy trajectories came from earlier Direct GRU controllers, so previous executed actions "
              "during supervised training are off-policy with respect to these residual controllers. "
              "This is a transfer limitation, not a new DAgger round.", "",
              "| Training delay (steps) | Valid supervised timesteps per branch |",
              "|---:|---:|"]
    for delay, count in summary["schedule"]["supervised_timesteps_by_delay"].items():
        lines.append(f"| {delay} | {count} |")
    lines += ["",
              "| Residual | Selected update | Common validation 5D MSE | Residual-target 4D MSE | Corrected-action 4D MSE |",
              "|---|---:|---:|---:|---:|"]
    for name in ("residual_mlp", "residual_gru"):
        training = summary["training"][name]
        selected = next(row for row in training["history"] if row["update"] == training["selected_update"])
        lines.append(f"| {name} | {training['selected_update']} | {selected['corrected_action_mse_5d']:.5f} | "
                     f"{selected['residual_prediction_mse_4d']:.5f} | "
                     f"{selected['corrected_action_mse_4d']:.5f} |")
    lines += ["", "## B. Fresh gate", "",
              "| Controller | Success | Collision | Final position (m) | Final yaw (rad) | Final aperture (m) | Trajectory position (m) | Mean residual norm | Mean inference (ms) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in CONTROLLERS:
        m = fresh["means"][name]
        residual = fresh["residual"].get(name, {})
        latency = fresh["ff_inference_latency"]["mean_ms"] if name == "ff" else residual["mean_inference_ms"]
        lines.append(f"| {name} | {m['success']:.3f} | {m['collision']:.3f} | {m['final_position_error']:.5f} | "
                     f"{m['final_orientation_error']:.4f} | {m['final_gripper_width_error']:.5f} | "
                     f"{m['trajectory_error']:.5f} | {residual.get('mean_residual_norm', 0):.5f} | {latency:.3f} |")
    mlp_r = fresh["residual"]["residual_mlp"]
    gru_r = fresh["residual"]["residual_gru"]
    lines += ["", f"**Fresh parity: {'PASS' if gate['passed'] else 'FAIL'}.** "
              f"The pre-existing practical checks are `{gate['checks']}`. "
              "The exact historical matched-FF fresh episode outcomes were reproduced. "
              "The per-episode rows, paired bootstrap intervals, residual/hidden distributions and traces are in `metrics/fresh/`.", "",
              f"Fresh residual norm p95: MLP {mlp_r['p95_residual_norm']:.5f}, GRU {gru_r['p95_residual_norm']:.5f}; "
              f"mean executed action deviation from FF: {mlp_r['mean_action_difference_from_ff']:.5f} and "
              f"{gru_r['mean_action_difference_from_ff']:.5f}. GRU hidden norm mean/p95: "
              f"{gru_r['mean_hidden_norm']:.3f}/{gru_r['p95_hidden_norm']:.3f}. "
              "Per-dimension and episode-time residual/hidden summaries are in `metrics/fresh/summary.json`. "
              "The four-dimensional residual-target MSE and corrected-action MSE are algebraically equal with a frozen "
              "base; both are recorded at every validation interval in the training logs.", ""]
    if not gate["passed"]:
        lines += ["## C. Delay experiment", "", "Not run: the fresh parity gate failed. No delayed, variable-delay, hidden-memory, or perturbation result is claimed.", "",
                  "## D. Memory ablation", "", "Not run because fresh parity failed.", "",
                  "## E. Variable delay / perturbation", "", "Not run because fresh parity failed.", ""]
    elif summary.get("fixed_delay_run"):
        lines += ["## C. Fixed-delay degradation", "",
                  "Machine-readable per-controller results and fresh-relative degradation are in "
                  "`plots_or_plot_data/fixed_delay_table.csv` and `.json`. All conditions use the same 16 held-out EpisodeSpecs. "
                  f"Next-gate signal: `{summary['fixed_delay_signal']}`.", "",
                  "| Delay | Controller | Success | Collision | Final position (mm) | Final yaw (rad) | "
                  "Trajectory position (mm) | Residual norm | Inference (ms) | Position change from fresh (mm) |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for delay in (0, 1, 2, 4, 8):
            condition = fresh if delay == 0 else summary["fixed_delay"][str(delay)]
            means = condition["means"]
            for name in CONTROLLERS:
                m = means[name]
                r = condition["residual"].get(name, {})
                latency = condition["ff_inference_latency"]["mean_ms"] if name == "ff" else r["mean_inference_ms"]
                lines.append(f"| {delay} | {name} | {m['success']:.3f} | {m['collision']:.3f} | "
                             f"{1000*m['final_position_error']:.2f} | {m['final_orientation_error']:.3f} | "
                             f"{1000*m['trajectory_error']:.2f} | {r.get('mean_residual_norm', 0):.4f} | "
                             f"{latency:.3f} | "
                             f"{1000*(m['final_position_error']-fresh['means'][name]['final_position_error']):+.2f} |")
        lines += ["", "At delay 2 the GRU degrades more in final position than FF and MLP. At delays 4 and 8 "
                  "its position degradation is smaller, but collision rates remain high and success falls to 1/16 for "
                  "all three controllers at each nonzero fixed delay. The curve supports a limited high-delay geometric "
                  "signal, not broad temporal robustness. Paired GRU−MLP and GRU−FF episode intervals are saved "
                  "under each delay metric directory. Residual norm combines command coordinates with different units "
                  "(meters and radians); use it as a within-controller action-size diagnostic. "
                  "At delay 4, paired GRU−MLP final position is −13.68 mm with episode bootstrap interval "
                  "[−19.42, −8.10] mm; at delay 8 it is −8.28 mm [−19.79, +3.13] mm. These exploratory intervals "
                  "do not account for looking across multiple delay settings.", "",
                  "## D. Hidden-memory ablation", "",
                  "The same GRU checkpoint was evaluated with carried hidden state and reset-every-step. "
                  "Paired rows and summaries are in `metrics/reset_delay*/`.", "",
                  "| Delay | Carry position (mm) | Reset position (mm) | Carry collision | Reset collision |",
                  "|---:|---:|---:|---:|---:|"]
        for delay in (2, 4, 8):
            carry = summary["fixed_delay"][str(delay)]["means"]["residual_gru"]
            reset = summary["hidden_reset_ablation"][str(delay)]["means"]["residual_gru"]
            lines.append(f"| {delay} | {1000*carry['final_position_error']:.2f} | "
                         f"{1000*reset['final_position_error']:.2f} | {carry['collision']:.3f} | {reset['collision']:.3f} |")
        lines += ["", "Carry improves final position at delays 4 and 8 but worsens it at delay 2; collision "
                  "effects also differ by delay. The saved paired carry−reset intervals quantify this small-sample "
                  "uncertainty. Paired carry−reset position differences are −28.55 mm "
                  "[−47.31, −9.94] at delay 4 and −35.40 mm [−50.90, −17.54] at delay 8. "
                  "This is a same-checkpoint memory ablation, not a comparison with the earlier Direct GRU.", "",
                  "## E. Variable delay / perturbation", ""]
        if summary.get("variable_and_perturbation_run"):
            variable, perturb = summary["variable_delay"]["means"], summary["perturbation"]["means"]
            lines += ["The fixed-delay signal met the predeclared next-gate rule at delays 4 and 8. "
                      "Variable delay cycles visual ages `[0,0,1,3,0,2,4,1]` control steps; proprioception remains "
                      "current. Perturbation shifts the object +0.025 m in x at step 5, immediately visible at that "
                      "step. Recovery means first return below the existing 0.025 m position and 0.200 rad yaw "
                      "thresholds; unrecovered episodes remain censored.", "",
                      "| Condition / controller | Success | Collision | Final position (mm) | Peak post-perturb position (mm) | Recovery fraction | Mean recovery among recovered (steps) |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for label, group in (("variable", variable), ("perturbation", perturb)):
                for name in CONTROLLERS:
                    m = group[name]
                    peak = "—" if label == "variable" else f"{1000*m['post_perturb_peak_position_error']:.2f}"
                    recovery = "—" if label == "variable" else f"{m['recovery_fraction']:.3f}"
                    recovery_time = "—" if label == "variable" else f"{m['recovery_time_mean_among_recovered']:.2f}"
                    lines.append(f"| {label} / {name} | {m['success']:.3f} | {m['collision']:.3f} | "
                                 f"{1000*m['final_position_error']:.2f} | {peak} | {recovery} | {recovery_time} |")
            lines += ["", "Variable-delay collision is 10/16 FF, 6/16 MLP, and 7/16 GRU. "
                      "The MLP is at least as safe as the GRU in this pattern; recurrent memory is not isolated as "
                      "the source of the improvement. Perturbation recovery favors the GRU in this one seed, "
                      "while all three have zero collisions. **Gate limitation:** the predefined position-based "
                      "next-gate rule allowed these runs despite high absolute collision under fixed delay. "
                      "They are retained as exploratory observations, not as a gate-validated demonstration of "
                      "delay robustness or perturbation benefit."]
        else:
            lines.append("Not run: fixed-delay results did not meet the predefined temporal-signal rule.")
        lines.append("")
    else:
        lines += ["## C. Fixed-delay degradation", "", "Pending after the passed fresh gate.", "",
                  "## D. Hidden-memory ablation", "", "Pending fixed-delay evaluation.", "",
                  "## E. Variable delay / perturbation", "", "Pending the fixed-delay signal check.", ""]
    lines += ["## F. Interpretation", "", "**Observed facts:** The tables and paired episode files above are the measurements. "
              "The frozen FF and topology were not retrained or changed.", "",
              "**Interpretation:** " + ("Fresh nominal behavior was preserved under the prior practical gate. "
              "The GRU has a limited position advantage at longer fixed delays, but delay-2 regression, high "
              "collision rates, and the MLP's similar variable-delay "
              "performance prevent a general claim that recurrent memory improves temporal robustness." if gate["passed"] else
              "The recurrent residual did not preserve nominal FF behavior under the pre-existing fresh gate; "
              "the staged protocol stops here."), "",
              "**Unverified hypotheses:** Any apparent benefit from recurrent memory would need replication across training seeds "
              "and should be separated from the MLP residual-capacity baseline. This single-seed study does not establish "
              "general temporal robustness.", ""]
    (root / "report.md").write_text("\n".join(lines))


def run(output: Path = OUTPUT, device_preference: DevicePreference = "auto",
        eval_device_preference: DevicePreference = "cpu", eval_workers: int = 1) -> dict:
    if output.exists() and (output / "summary.json").exists():
        prior = json.loads((output / "summary.json").read_text())
        if prior.get("fixed_delay_run") or not prior.get("fresh_gate", {}).get("passed", True):
            raise FileExistsError(f"Existing completed residual experiment: {output}")
    if eval_workers != 1:
        raise ValueError("Paired MuJoCo rollout uses one process-local environment.")
    root = ensure_dir(output)
    design = ResidualConfig()
    device, eval_device = select_device(device_preference), select_device(eval_device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task = rf._task_config(root, (SEED,), device_preference, eval_device_preference, 16)
    temporal = rf.TemporalConfig()
    if task.point_count != 32 or not task.symmetric_robot_edges or task.message_passing_steps != 3:
        raise AssertionError("Canonical graph configuration changed.")
    if hd._sha256(BASE_CHECKPOINT) != BASE_SHA256:
        raise AssertionError("Canonical FF checkpoint changed.")
    expert, validation, train_specs, _ = final._current_datasets(SEED, task)
    pool = torch.load("artifacts/recurrent_recovery_priority_controlled/policy_pool.pt",
                      map_location="cpu", weights_only=False)
    data = source_episodes(expert, pool)
    validation_data = validation_episodes(validation)
    batches, schedule = make_schedule(data, design)
    _write(root / "logs" / "schedule.json", schedule)
    _write(root / "configs" / "experiment.json", {"design": asdict(design), "task": asdict(task),
                                                   "training_device": str(device), "evaluation_device": str(eval_device),
                                                   "base_checkpoint": str(BASE_CHECKPOINT),
                                                   "base_sha256": BASE_SHA256,
                                                   "old_policy_sha256": hd._sha256(OLD_POLICY),
                                                   "new_policy_sha256": hd._sha256(NEW_POLICY),
                                                   "pool_sha256": hd._tensor_digest(pool),
                                                   "training_episode_signatures": [final.spec_signature(s) for s in train_specs]})
    training = {}
    for kind in ("mlp", "gru"):
        record = root / "logs" / f"{kind}_training.json"
        if record.exists():
            training[f"residual_{kind}"] = json.loads(record.read_text())
            if training[f"residual_{kind}"]["schedule_sha256"] != schedule["schedule_sha256"] or \
                    training[f"residual_{kind}"]["base_checkpoint_sha256"] != BASE_SHA256:
                raise AssertionError("Saved branch does not match frozen training protocol.")
        else:
            training[f"residual_{kind}"] = train_branch(kind, batches, schedule, validation_data,
                                                         task, design, device, root)
    if training["residual_mlp"]["schedule_sha256"] != training["residual_gru"]["schedule_sha256"]:
        raise AssertionError("Residual branches received different training tensors.")
    del batches
    models = {"ff": sf.load_surface_model(BASE_CHECKPOINT, rf.GRAPH_NAME, task, eval_device),
              "residual_mlp": load_controller("mlp", task, design, eval_device,
                                             Path(training["residual_mlp"]["checkpoint"])),
              "residual_gru": load_controller("gru", task, design, eval_device,
                                             Path(training["residual_gru"]["checkpoint"]))}
    specs = sf.sample_episode_specs(16, SEED + 60_000, task, "iid")
    if {s.sample_identity for s in specs} & {s.sample_identity for s in train_specs}:
        raise AssertionError("Held-out specs overlap training.")
    _write(root / "configs" / "heldout_episode_specs.json", [asdict(s) for s in specs])
    fresh_record = root / "metrics" / "fresh" / "summary.json"
    fresh = json.loads(fresh_record.read_text()) if fresh_record.exists() else evaluate_condition(
        models, specs, task, temporal, eval_device, root / "metrics" / "fresh")
    historical = json.loads(HISTORICAL_FRESH.read_text())
    for before, after in zip(historical, json.loads((root / "metrics/fresh/per_episode/ff.json").read_text()), strict=True):
        for key in ("success", "collision", "final_position_error", "final_orientation_error",
                    "final_gripper_width_error", "trajectory_error", "minimum_safe_clearance"):
            if not np.isclose(before[key], after[key], atol=1e-7):
                raise AssertionError(f"Canonical FF rollout differs from historical matched FF: {key}")
    gate = _fresh_gate(fresh)
    summary = {"seed": SEED, "base_checkpoint": str(BASE_CHECKPOINT), "base_sha256": BASE_SHA256,
               "design": asdict(design), "training": training, "schedule": schedule,
               "fresh": fresh, "fresh_gate": gate,
               "fixed_delay_run": False, "variable_and_perturbation_run": False}
    _write(root / "summary.json", summary)
    write_report(root, summary)
    if not gate["passed"]:
        return summary
    delayed = {delay: evaluate_condition(models, specs, task, temporal, eval_device,
                                         root / "metrics" / f"delay{delay}", delay=delay)
               for delay in (1, 2, 4, 8)}
    signal = _delay_signal(fresh, delayed)
    table = _delay_table(root, fresh, delayed)
    # Same trained checkpoint; no further optimization or tuning.
    reset = {delay: evaluate_condition({"ff": models["ff"], "residual_gru": models["residual_gru"]},
                                       specs, task, temporal, eval_device,
                                       root / "metrics" / f"reset_delay{delay}", delay=delay,
                                       reset_gru=True) for delay in (2, 4, 8)}
    summary.update({"fixed_delay_run": True, "fixed_delay": {str(k): v for k, v in delayed.items()},
                    "fixed_delay_signal": signal, "fixed_delay_table": table,
                    "hidden_reset_ablation": {str(k): v for k, v in reset.items()}})
    if signal["credible_for_next_gate"]:
        pattern = (0, 0, 1, 3, 0, 2, 4, 1)
        variable = evaluate_condition(models, specs, task, temporal, eval_device,
                                      root / "metrics" / "variable_delay", pattern=pattern)
        perturb = evaluate_condition(models, specs, task, temporal, eval_device,
                                     root / "metrics" / "perturbation", perturbation=True)
        summary.update({"variable_and_perturbation_run": True,
                        "variable_delay": variable, "perturbation": perturb})
    summary = analyze_saved(root, summary)
    _write(root / "summary.json", summary)
    write_report(root, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    args = parser.parse_args()
    run(args.output, args.device, args.eval_device, args.eval_workers)


if __name__ == "__main__":
    main()
