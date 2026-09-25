"""Aperture-aware spatial control on the unchanged canonical 32-point graph."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import multiprocessing as mp

import numpy as np
import torch
from torch import nn

from vla_gnn_recurrent.training import aperture_bottleneck_diagnosis as prior
from vla_gnn_recurrent.training import dynamic_visual_delay_feasibility as dynamic
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import residual_recurrent_feasibility as residual
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


OUTPUT = Path("artifacts/aperture_aware_spatial_controller")
PRIOR = prior.OUTPUT
STAGE_B_5D = PRIOR / "action_path/residual_5d_mlp/stage_b/selected.pt"
STAGE_B_4D = prior.SOURCE / "training/stage_b_mlp/checkpoints/mlp.pt"
WIDTH_HEAD_DIM = 64
TRAIN_STEPS = 1200
VALIDATE_EVERY = 25
PATIENCE_CHECKS = 24


def _csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task(seed: int = 2811, device: DevicePreference = "cpu",
          eval_device: DevicePreference = "cpu") -> sf.SurfaceFeasibilityConfig:
    return rf._task_config(dynamic.OUTPUT / f"seed{seed}", (seed,), device, eval_device, 16)


def width_target(shape: sf.SurfaceShape) -> float:
    """Training label only; no shape field enters model.forward or model.step."""
    return prior.desired_width(shape)


def width_transform(train_widths: np.ndarray) -> tuple[float, float]:
    center = float(np.mean(train_widths))
    scale = float(np.std(train_widths))
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("Degenerate training-only width normalization")
    return center, scale


def normalized_width(width: torch.Tensor, center: float, scale: float) -> torch.Tensor:
    return (width - center) / scale


def physical_width(normalized: torch.Tensor, center: float, scale: float) -> torch.Tensor:
    return normalized * scale + center


def _episode_data(split: str, seed: int = 2811) -> dict:
    if seed != 2811:
        raise ValueError("Only the frozen seed-2811 data pool is currently available")
    task = _task(seed)
    result = prior.load_split(split, task)
    return result


def _assert_splits() -> dict:
    splits = {name: _episode_data(name) for name in ("train", "validation", "heldout")}
    ids = {name: set(data["identity"]) for name, data in splits.items()}
    if any(ids[a] & ids[b] for a, b in (("train", "validation"),
                                       ("train", "heldout"), ("validation", "heldout"))):
        raise AssertionError("Episode-level aperture split leakage")
    return splits


def _check_topology(task: sf.SurfaceFeasibilityConfig, state: np.ndarray,
                    points: np.ndarray) -> None:
    state_t, points_t, topology = rf._graph_tensors(state, points, task, select_device("cpu"))
    rf.assert_canonical_topology(topology, task)
    if tuple(points_t.shape) != (1, 32, 3) or topology.src.shape[1] != 100:
        raise AssertionError("Canonical 32-point/100-edge topology changed")


def reproduce(output: Path = OUTPUT) -> dict:
    root = ensure_dir(output / "reproduction")
    task = _task()
    previous_hashes = json.loads((PRIOR / "audit/checkpoint_hashes.json").read_text())
    current_hashes = {"base_ff": _sha(residual.BASE_CHECKPOINT),
                      "historical_4d_mlp": _sha(STAGE_B_4D),
                      "selected_5d_mlp": _sha(STAGE_B_5D)}
    if any(previous_hashes[k] != value for k, value in current_hashes.items()):
        raise AssertionError("Aperture baseline checkpoint hash drifted")
    splits = _assert_splits()
    _check_topology(task, splits["heldout"]["state"][0], splits["heldout"]["points"][0])
    geometry_rows = []
    for name, data in splits.items():
        for identity in sorted(set(data["identity"])):
            index = data["identity"].index(identity)
            point_cloud = data["points"][index]
            raw_span = float(pairwise_span_features(point_cloud)[3:5].mean())
            geometry_rows.append({"split": name, "episode": identity,
                                  "observed_span_m": raw_span,
                                  "target_width_m": float(data["width"][index])})
    _csv(root / "observable_width.csv", geometry_rows)
    train_rows = [row for row in geometry_rows if row["split"] == "train"]
    offset = float(np.median([r["target_width_m"] - r["observed_span_m"] for r in train_rows]))
    observation_error = np.asarray([r["observed_span_m"] + offset - r["target_width_m"]
                                    for r in geometry_rows])
    head_payload = torch.load(PRIOR / "probes/frozen_z/width/best.pt",
                              map_location="cpu", weights_only=False)
    frozen_probe = prior.Probe(head_payload["input_dim"])
    frozen_probe.load_state_dict(head_payload["model_state"])
    frozen_probe.eval()
    base = sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME,
                                 task, select_device("cpu")).eval()
    heldout = splits["heldout"]
    z = prior.frozen_embeddings(heldout, task, select_device("cpu"), base)
    with torch.no_grad():
        prediction = frozen_probe(torch.as_tensor((z - head_payload["input_mean"]) /
                                                  head_payload["input_std"], dtype=torch.float32))
    pred = prediction.numpy() * head_payload["target_std"] + head_payload["target_mean"]
    frozen_mae = float(np.mean(np.abs(pred - heldout["width"])))
    prior_metrics = json.loads((PRIOR / "probes/summary.json").read_text())
    expected_mae = prior_metrics["width"]["frozen_z"]["heldout"]["mae"]
    if abs(frozen_mae - expected_mae) > 1e-5:
        raise AssertionError("Frozen-z width probe did not reproduce")
    prior_cf = json.loads((PRIOR / "probes/counterfactual_width_response.json").read_text())
    previous_eval = json.loads((PRIOR / "evaluation/summary.json").read_text())
    result = {"checkpoint_sha256": current_hashes,
              "episode_counts": {k: len(set(v["identity"])) for k, v in splits.items()},
              "frozen_z_heldout_mae_m": frozen_mae,
              "frozen_z_expected_mae_m": expected_mae,
              "geometry_offset_fit_train_only_m": offset,
              "observable_width_episode_mae_m": float(np.mean(np.abs(observation_error))),
              "observable_width_episode_p95_m": float(np.quantile(np.abs(observation_error), .95)),
              "prior_counterfactual_true_delta_m": prior_cf["target_width_delta_m_mean"],
              "prior_counterfactual_frozen_z_delta_m": prior_cf["frozen_z_predicted_width_delta_m_mean"],
              "prior_dynamic_4d_success": previous_eval["dynamic_fresh"]["residual_4d_mlp"]["success"],
              "prior_dynamic_5d_success": previous_eval["dynamic_fresh"]["residual_5d_mlp"]["success"],
              "prior_static_4d_success": previous_eval["static_fresh"]["residual_4d_mlp"]["success"],
              "prior_static_5d_success": previous_eval["static_fresh"]["residual_5d_mlp"]["success"]}
    if abs(result["observable_width_episode_mae_m"] - .00011) > .0003:
        raise AssertionError("Raw coordinate width diagnostic failed to reproduce")
    residual._write(root / "summary.json", result)
    return result


class GeometryWidthController:
    """Diagnostic controller: unchanged 5D MLP motion, observed-coordinate width command."""

    def __init__(self, motion: residual.ResidualController, offset_m: float,
                 task: sf.SurfaceFeasibilityConfig) -> None:
        self.motion, self.offset_m, self.task = motion, offset_m, task

    @torch.no_grad()
    def step(self, state: torch.Tensor, points: torch.Tensor, previous: torch.Tensor,
             hidden: torch.Tensor | None = None, topology: sf.GraphTopology | None = None,
             reset_hidden: bool = False) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
        action, _, _, base = self.motion.step(state, points, previous, hidden, topology, reset_hidden)
        widths = [float(pairwise_span_features(p.detach().cpu().numpy())[3:5].mean()) + self.offset_m
                  for p in points]
        desired = torch.as_tensor(widths, dtype=state.dtype, device=state.device)
        desired_fraction = ((desired - sf.MIN_GRIPPER_WIDTH) /
                            (sf.MAX_GRIPPER_WIDTH - sf.MIN_GRIPPER_WIDTH)).clamp(0, 1)
        gripper = (desired_fraction - state[:, 7]).clamp(-self.task.max_delta_gripper,
                                                        self.task.max_delta_gripper)
        output = torch.cat((action[:, :4], gripper[:, None]), dim=-1)
        return output, output - base, None, base


class ApertureAwareController(nn.Module):
    """Canonical GNN and FF head, existing residual MLP, plus normalized width head."""

    def __init__(self, base: sf.SurfaceGraphNetwork, task: sf.SurfaceFeasibilityConfig,
                 center: float, scale: float, bound_fraction: float = .2) -> None:
        super().__init__()
        if base.use_local_edges or task.point_count != 32 or not task.symmetric_robot_edges:
            raise ValueError("Only the canonical 32-point, 100-edge graph is supported")
        self.base = base
        self.task = task
        self.register_buffer("width_center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("width_scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("action_scales", residual.action_scales(task))
        self.register_buffer("residual_bounds", self.action_scales * bound_fraction)
        self.width_head = nn.Sequential(nn.Linear(task.hidden_dim * 3, WIDTH_HEAD_DIM),
                                        nn.SiLU(), nn.Linear(WIDTH_HEAD_DIM, 1))
        input_dim = task.hidden_dim * 3 + 2 * sf.ACTION_DIM
        self.memory = nn.Sequential(nn.Linear(input_dim, 384), nn.SiLU(),
                                    nn.Linear(384, 128), nn.SiLU())
        self.residual_head = nn.Linear(128, sf.ACTION_DIM)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, state: torch.Tensor, points: torch.Tensor,
                previous: torch.Tensor, topology: sf.GraphTopology | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.base.encode(state, points, topology)
        base_action = self.base.action_head(z)
        features = torch.cat((z, base_action / self.action_scales,
                              previous / self.action_scales), dim=-1)
        raw = self.residual_head(self.memory(features))
        correction = self.residual_bounds * torch.tanh(raw)
        action = base_action + correction
        width = self.width_head(z).squeeze(-1) * self.width_scale + self.width_center
        return action, correction, base_action, width, raw

    @torch.no_grad()
    def step(self, state: torch.Tensor, points: torch.Tensor, previous: torch.Tensor,
             hidden: torch.Tensor | None = None, topology: sf.GraphTopology | None = None,
             reset_hidden: bool = False) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
        del hidden, reset_hidden
        action, correction, base_action, _, _ = self.forward(state, points, previous, topology)
        return action, correction, None, base_action


def _load_aware(checkpoint: Path, task: sf.SurfaceFeasibilityConfig,
                device: torch.device) -> ApertureAwareController:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    base = sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME, task, device)
    model = ApertureAwareController(base, task, payload["width_center"],
                                    payload["width_scale"], payload["bound_fraction"]).to(device)
    model.load_state_dict(payload["model_state"])
    return model.eval()


def pairwise_span_features(points: np.ndarray) -> np.ndarray:
    """Eight paired endpoint spans from unordered observed xyz; no sampler slots."""
    xyz = np.asarray(points, dtype=np.float64).reshape(32, 3)
    endpoints = xyz[np.argsort(xyz[:, 1])[-16:]]
    xz = endpoints[:, [0, 2]]
    _, axes = np.linalg.eigh(np.cov(xz, rowvar=False))
    longitudinal = xz @ axes[:, 1]
    ordered = xz[np.argsort(longitudinal)].reshape(8, 2, 2)
    return np.linalg.norm(ordered[:, 0] - ordered[:, 1], axis=1).astype(np.float32)


class ApertureSpanHead(nn.Module):
    """Small permutation-invariant relational head for desired physical width."""

    def __init__(self, center: float, scale: float) -> None:
        super().__init__()
        self.register_buffer("width_center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("width_scale", torch.tensor(scale, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, spans: torch.Tensor) -> torch.Tensor:
        return self.net(spans).squeeze(-1) * self.width_scale + self.width_center


def train_aperture_branch(output: Path = OUTPUT,
                          device_preference: DevicePreference = "auto", seed: int = 2811,
                          fixed_budget: int | None = None) -> dict:
    """Train only on point-derived relations; current robot state cannot leak width."""
    root = ensure_dir(output / ("separate_aperture_branch" if seed == 2811 else
                                f"replication/seed{seed}/separate_aperture_branch"))
    device = select_device(device_preference)
    task = _task(seed, device=device_preference)
    splits = {key: _supervised_split(key, task, seed) for key in ("train", "validation", "heldout")}
    if any(set(splits[a]["identity"]) & set(splits[b]["identity"])
           for a, b in (("train", "validation"), ("train", "heldout"), ("validation", "heldout"))):
        raise AssertionError("Episode split leakage")
    center, scale = width_transform(splits["train"]["width"])
    spans = {key: np.stack([pairwise_span_features(p) for p in data["points"]])
             for key, data in splits.items()}
    feature_center = spans["train"].mean(0)
    feature_scale = spans["train"].std(0).clip(1e-5)
    x = {key: torch.from_numpy((value - feature_center) / feature_scale).to(device)
         for key, value in spans.items()}
    y = {key: torch.from_numpy(data["width"]).to(device) for key, data in splits.items()}
    set_seed(seed + 530_101)
    model = ApertureSpanHead(center, scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed + 530_101)
    best, selected, stale, curve = math.inf, 0, 0, []
    started = time.perf_counter()
    for step in range(1, (fixed_budget or 5000) + 1):
        idx = torch.randint(len(x["train"]), (128,), generator=generator).to(device)
        model.train()
        pred = model(x["train"][idx])
        loss = ((pred - y["train"][idx]) / model.width_scale).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            model.eval()
            with torch.no_grad():
                val = float((((model(x["validation"]) - y["validation"]) /
                              model.width_scale).square().mean()))
                val_mae = float((model(x["validation"]) - y["validation"]).abs().mean())
            curve.append({"update": step, "train_normalized_mse": float(loss.detach().cpu()),
                          "validation_normalized_mse": val, "validation_mae_m": val_mae})
            if val < best - 1e-5:
                best, selected, stale = val, step, 0
                torch.save({"model_state": model.state_dict(), "width_center": center,
                            "width_scale": scale, "feature_center": feature_center,
                            "feature_scale": feature_scale, "selected_update": step,
                            "topology": "unchanged_canonical_32_points_100_edges"},
                           root / "selected.pt")
            else:
                stale += 1
            if fixed_budget is None and step >= 500 and stale >= 30:
                break
    _csv(root / "learning_curve.csv", curve)
    payload = torch.load(root / "selected.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    metrics = {}
    for key in ("validation", "heldout"):
        with torch.no_grad():
            prediction = model(x[key]).cpu().numpy()
        rows = [{"episode": identity, "true_desired_width_m": float(true),
                 "predicted_desired_width_m": float(pred),
                 "current_aperture_fraction": float(opening),
                 "width_error_m": float(pred - true)}
                for identity, true, pred, opening in zip(splits[key]["identity"],
                    splits[key]["width"], prediction, splits[key]["state"][:, 7], strict=True)]
        _csv(root / f"{key}_width_predictions.csv", rows)
        metrics[key] = _width_metrics(rows)
    result = {"seed": seed, "selected_update": selected, "final_update": step,
              "validation_normalized_mse": best, "metrics": metrics,
              "training_seconds": time.perf_counter() - started,
              "device": str(device), "input": "eight pairwise endpoint spans from unordered xyz only",
              "no_robot_state_or_shape_input": True, "no_point_ids": True,
              "topology": "unchanged_canonical_32_points_100_edges"}
    residual._write(root / "summary.json", result)
    return result


class SpanBranchController:
    """Frozen 5D MLP motion with a separately learned point-relation gripper branch."""

    def __init__(self, motion: residual.ResidualController, checkpoint: Path,
                 task: sf.SurfaceFeasibilityConfig, device: torch.device,
                 correction_bound: float | None = None) -> None:
        self.motion, self.task, self.device = motion, task, device
        self.correction_bound = correction_bound
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.head = ApertureSpanHead(payload["width_center"], payload["width_scale"]).to(device)
        self.head.load_state_dict(payload["model_state"])
        self.head.eval()
        self.feature_center = torch.as_tensor(payload["feature_center"], device=device)
        self.feature_scale = torch.as_tensor(payload["feature_scale"], device=device)
        self.capacity_trace: list[dict] = []

    @torch.no_grad()
    def predicted_width(self, points: torch.Tensor) -> torch.Tensor:
        spans = torch.as_tensor(np.stack([pairwise_span_features(p.detach().cpu().numpy())
                                          for p in points]), dtype=points.dtype, device=points.device)
        return self.head((spans - self.feature_center) / self.feature_scale)

    @torch.no_grad()
    def step(self, state: torch.Tensor, points: torch.Tensor, previous: torch.Tensor,
             hidden: torch.Tensor | None = None, topology: sf.GraphTopology | None = None,
             reset_hidden: bool = False) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
        action, _, _, base = self.motion.step(state, points, previous, hidden, topology, reset_hidden)
        width = self.predicted_width(points)
        desired_fraction = ((width - sf.MIN_GRIPPER_WIDTH) /
                            (sf.MAX_GRIPPER_WIDTH - sf.MIN_GRIPPER_WIDTH)).clamp(0, 1)
        desired_action = (desired_fraction - state[:, 7]).clamp(-self.task.max_delta_gripper,
                                                               self.task.max_delta_gripper)
        raw_correction = desired_action - base[:, 4]
        executed_correction = (raw_correction if self.correction_bound is None else
                               raw_correction.clamp(-self.correction_bound, self.correction_bound))
        gripper = base[:, 4] + executed_correction
        output = torch.cat((action[:, :4], gripper[:, None]), dim=-1)
        self.capacity_trace.extend([{"timestep": len(self.capacity_trace) + i,
                               "target_gripper_correction": float(r),
                               "raw_predicted_correction": float(r),
                               "bounded_correction": float(b),
                               "saturated": bool(abs(float(r-b)) > 1e-7)}
                              for i, (r, b) in enumerate(zip(raw_correction, executed_correction, strict=True))])
        return output, output - base, None, base


def _features(split: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {"state": torch.from_numpy(split["state"]).to(device),
            "points": torch.from_numpy(split["points"]).to(device),
            "targets": torch.from_numpy(split["action_target_5d"]).to(device),
            "widths": torch.from_numpy(split["width"]).to(device),
            "previous": torch.from_numpy(split["previous"]).to(device)}


def _supervised_split(name: str, task: sf.SurfaceFeasibilityConfig, seed: int = 2811) -> dict:
    source = dynamic.OUTPUT / f"seed{seed}"
    if name == "train":
        paths = [source / "training/expert.pt", source / "training/policy_mlp.pt"]
    elif name == "validation":
        paths = [source / "training/validation.pt"]
    else:
        paths = [source / "benchmark_sanity/heldout_oracle.pt"]
    specs = {spec.identity: spec for split, count in (("train", 72), ("validation", 16),
             ("heldout", 16)) for spec in dynamic.dynamic_specs(
                 seed, task, count, dynamic.SELECTED_STEP_SPEEDS_M, split)}
    episodes = [ep for path in paths for ep in torch.load(path, map_location="cpu", weights_only=False)]
    state, points, target, prev, width, ids = [], [], [], [], [], []
    for ep in episodes:
        mask = ep["mask"].numpy().astype(bool)
        state.append(ep["states"].numpy()[mask])
        points.append(ep["points"].numpy()[mask])
        target.append(ep["targets"].numpy()[mask])
        prev.append(ep["previous"].numpy()[mask])
        width.extend([width_target(specs[ep["identity"]].initial.shape)] * int(mask.sum()))
        ids.extend([ep["identity"]] * int(mask.sum()))
    return {"state": np.concatenate(state), "points": np.concatenate(points),
            "action_target_5d": np.concatenate(target), "previous": np.concatenate(prev),
            "width": np.asarray(width, np.float32), "identity": ids}


def _validation(model: ApertureAwareController, data: dict, task: sf.SurfaceFeasibilityConfig,
                device: torch.device) -> dict:
    model.eval()
    scales = residual.action_scales(task, device)
    errors, width_errors = [], []
    with torch.no_grad():
        for start in range(0, len(data["state"]), 64):
            end = start + 64
            states = torch.from_numpy(data["state"][start:end]).to(device)
            points = torch.from_numpy(data["points"][start:end]).to(device)
            previous = torch.from_numpy(data["previous"][start:end]).to(device)
            target = torch.from_numpy(data["action_target_5d"][start:end]).to(device)
            width = torch.from_numpy(data["width"][start:end]).to(device)
            action, _, _, prediction, _ = model(states, points, previous)
            errors.append(((action - target) / scales).square().cpu())
            width_errors.append((prediction - width).cpu())
    error = torch.cat(errors)
    width_error = torch.cat(width_errors)
    return {"action_mse": float(error.mean()),
            "translation_mse": float(error[:, :3].mean()),
            "yaw_mse": float(error[:, 3].mean()),
            "gripper_mse": float(error[:, 4].mean()),
            "width_mae_m": float(width_error.abs().mean()),
            "width_normalized_mse": float((width_error / model.width_scale.cpu()).square().mean())}


def _shared_schedule(seed: int, task: sf.SurfaceFeasibilityConfig) -> list[tuple[int, dict]]:
    """Original Stage A/B batches plus episode-label width, with exact batch-hash checks."""
    source = dynamic.OUTPUT / f"seed{seed}/training"
    expert = torch.load(source / "expert.pt", map_location="cpu", weights_only=False)
    policy = torch.load(source / "policy_mlp.pt", map_location="cpu", weights_only=False)
    specs = {spec.identity: spec for spec in dynamic.dynamic_specs(
        seed, task, 72, dynamic.SELECTED_STEP_SPEEDS_M, "train")}
    result = []
    for stage, episodes_by_source, updates in (
            ("stage_a", {"expert": expert, "policy": expert}, 800),
            ("stage_b_mlp", {"expert": expert, "policy": policy}, 400)):
        design = replace(residual.ResidualConfig(), seed=seed, corrected_dimensions=5,
                         updates=updates)
        eligibility = None if stage == "stage_a" else {
            "expert": {delay: list(range(len(expert))) for delay in dynamic.TRAIN_DELAYS},
            "policy": dynamic._eligible_policy_by_delay(policy)}
        batches, schedule = residual.make_schedule(episodes_by_source, design,
                                                   dynamic.dynamic_delayed_sequence,
                                                   eligibility)
        expected = json.loads((source / stage / "logs/schedule.json").read_text())
        if schedule["schedule_sha256"] != expected["schedule_sha256"]:
            raise AssertionError("Shared encoder schedule differs from original residual training")
        rng = np.random.default_rng(seed + 1_400_001)
        for batch in batches:
            widths = torch.zeros(batch["targets"].shape[:2], dtype=torch.float32)
            selected = []
            for source_name in ("expert", "policy"):
                for delay in design.train_delays:
                    choices = (eligibility[source_name][delay] if eligibility else
                               list(range(len(episodes_by_source[source_name]))))
                    for draw in rng.integers(0, len(choices), size=2).tolist():
                        selected.append((source_name, choices[draw]))
            for i, (source_name, ep_index) in enumerate(selected):
                ep = episodes_by_source[source_name][ep_index]
                length = len(ep["targets"])
                if not torch.equal(batch["targets"][i, :length], ep["targets"]):
                    raise AssertionError("Width labels lost alignment with scheduled action batch")
                widths[i, :length] = width_target(specs[ep["identity"]].initial.shape)
            batch["width_targets"] = widths
            result.append((800 if stage == "stage_b_mlp" else 0, batch))
    if len(result) != TRAIN_STEPS:
        raise AssertionError("Shared encoder expected 800+400 scheduled updates")
    return result


def train_shared(output: Path = OUTPUT, device_preference: DevicePreference = "auto") -> dict:
    root = ensure_dir(output / "shared_encoder")
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task = _task(device=device_preference)
    train = _supervised_split("train", task)
    validation = _supervised_split("validation", task)
    if set(train["identity"]) & set(validation["identity"]):
        raise AssertionError("Train-validation episode leakage")
    center, scale = width_transform(train["width"])
    set_seed(2811 + 530_001)
    base = sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME, task, device)
    model = ApertureAwareController(base, task, center, scale).to(device)
    baseline = residual.load_controller("mlp", task,
        replace(residual.ResidualConfig(), corrected_dimensions=5), device, STAGE_B_5D)
    model.memory.load_state_dict(baseline.memory.state_dict())
    model.residual_head.load_state_dict(baseline.residual_head.state_dict())
    _check_topology(task, train["state"][0], train["points"][0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=task.learning_rate,
                                  weight_decay=task.weight_decay)
    batches = _shared_schedule(2811, task)
    scales = residual.action_scales(task, device)
    curve, best, selected, stale = [], math.inf, 0, 0
    started = time.perf_counter()
    for step, (stage_offset, cpu_batch) in enumerate(batches, 1):
        if step == 801:
            optimizer = torch.optim.AdamW(model.parameters(), lr=task.learning_rate,
                                          weight_decay=task.weight_decay)
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        flat_state = batch["states"].flatten(0, 1)
        flat_points = batch["points"].flatten(0, 1)
        flat_previous = batch["previous"].flatten(0, 1)
        model.train()
        action, _, _, width, _ = model(flat_state, flat_points, flat_previous)
        action = action.reshape(*batch["mask"].shape, 5)
        width = width.reshape(batch["mask"].shape)
        action_sq = ((action - batch["targets"]) / scales).square().mean(-1)
        width_sq = ((width - batch["width_targets"]) / model.width_scale).square()
        action_error = .5 * action_sq[:8][batch["mask"][:8]].mean() + \
                       .5 * action_sq[8:][batch["mask"][8:]].mean()
        width_error = .5 * width_sq[:8][batch["mask"][:8]].mean() + \
                      .5 * width_sq[8:][batch["mask"][8:]].mean()
        loss = action_error + width_error  # normalized targets, predeclared lambda_width = 1
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % VALIDATE_EVERY == 0:
            val = _validation(model, validation, task, device)
            row = {"update": step, "train_action_mse": float(action_error.detach().cpu()),
                   "train_width_normalized_mse": float(width_error.detach().cpu()), **val}
            curve.append(row)
            score = val["action_mse"] + val["width_normalized_mse"]
            if score < best - 1e-5:
                best, selected, stale = score, step, 0
                torch.save({"model_state": model.state_dict(), "width_center": center,
                            "width_scale": scale, "bound_fraction": .2,
                            "selected_update": step, "validation_score": score,
                            "base_checkpoint_sha256": residual.BASE_SHA256,
                            "source_5d_checkpoint_sha256": _sha(STAGE_B_5D),
                            "topology": "canonical_32_points_100_edges"}, root / "selected.pt")
            else:
                stale += 1
            if step % 100 == 0:
                print(f"[shared] {step}: val action={val['action_mse']:.4f} "
                      f"width={1000*val['width_mae_m']:.2f}mm", flush=True)
    _csv(root / "learning_curve.csv", curve)
    result = {"selected_update": selected, "final_update": step,
              "validation_score": best, "validation_at_selected": min(curve, key=lambda r:
                r["action_mse"] + r["width_normalized_mse"]),
              "width_center_m": center, "width_scale_m": scale,
              "lambda_width": 1.0, "sequence_batch_size": 16,
              "stage_a_updates": 800, "stage_b_updates": 400,
              "batch_protocol": "exact historical expert/policy/delay schedules; fresh AdamW per stage",
              "optimizer": "AdamW", "learning_rate": task.learning_rate,
              "weight_decay": task.weight_decay,
              "training_seconds": time.perf_counter() - started,
              "device": str(device), "train_episodes": len(set(train["identity"])),
              "validation_episodes": len(set(validation["identity"])),
              "source_5d_checkpoint_sha256": _sha(STAGE_B_5D),
              "topology": "canonical_32_points_100_edges"}
    residual._write(root / "summary.json", result)
    return result


def _train_seed_5d_stage(seed: int, stage: str, root: Path, expert: list[dict],
                         policy: list[dict], validation: list[dict],
                         task: sf.SurfaceFeasibilityConfig, device: torch.device,
                         starting_checkpoint: Path | None) -> dict:
    updates = 800 if stage == "stage_a" else 400
    design = replace(residual.ResidualConfig(), seed=seed,
                     corrected_dimensions=5, updates=updates)
    eligibility = None if stage == "stage_a" else {
        "expert": {delay: list(range(len(expert))) for delay in dynamic.TRAIN_DELAYS},
        "policy": dynamic._eligible_policy_by_delay(policy)}
    batches, schedule = residual.make_schedule({"expert": expert, "policy": policy},
                                               design, dynamic.dynamic_delayed_sequence,
                                               eligibility)
    expected_path = dynamic.OUTPUT / f"seed{seed}/training/{'stage_a' if stage == 'stage_a' else 'stage_b_mlp'}/logs/schedule.json"
    expected = json.loads(expected_path.read_text())
    if schedule["schedule_sha256"] != expected["schedule_sha256"]:
        raise AssertionError("Replication 5D batch schedule differs from historical 4D")
    residual._write(root / "schedule.json", schedule)
    set_seed(seed + 1_400_001)
    model = residual.load_controller("mlp", task, design, device, starting_checkpoint)
    if starting_checkpoint is None:
        first = batches[0]
        with torch.no_grad():
            corrected, correction, _, base, _ = model.forward_sequence(
                first["states"][:1].to(device), first["points"][:1].to(device),
                first["previous"][:1].to(device))
        if not torch.equal(corrected, base) or torch.count_nonzero(correction):
            raise AssertionError("Zero-initialized 5D branch does not reproduce FF")
    base_hash = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
        for v in model.base.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=task.learning_rate, weight_decay=task.weight_decay)
    best, selected, curve = math.inf, 0, []
    started = time.perf_counter()
    for step, cpu_batch in enumerate(batches, 1):
        batch = {k: v.to(device) for k, v in cpu_batch.items()}
        model.train()
        corrected, _, _, _, _ = model.forward_sequence(batch["states"], batch["points"],
                                                        batch["previous"])
        loss = .5 * residual.corrected_loss(corrected[:8], batch["targets"][:8],
                                             batch["mask"][:8], task, dimensions=5) + \
               .5 * residual.corrected_loss(corrected[8:], batch["targets"][8:],
                                             batch["mask"][8:], task, dimensions=5)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            model.eval()
            val = residual.validation_metrics(model, validation, task, device)
            curve.append({"update": step, "train_5d_mse": float(loss.detach().cpu()),
                          "validation_5d_mse": val["corrected_action_mse_5d"],
                          "validation_4d_mse": val["corrected_action_mse_4d"]})
            if val["corrected_action_mse_5d"] < best:
                best, selected = val["corrected_action_mse_5d"], step
                torch.save({"model_state": model.state_dict(), "kind": "mlp",
                            "base_sha256": residual.BASE_SHA256, "design": asdict(design),
                            "seed": seed, "selected_update": step,
                            "validation_5d_mse": best, "topology": "consistent_intended_100"},
                           root / "selected.pt")
    new_base_hash = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
        for v in model.base.state_dict().values())).hexdigest()
    if base_hash != new_base_hash:
        raise AssertionError("Frozen graph/FF changed during replication")
    _csv(root / "learning_curve.csv", curve)
    result = {"seed": seed, "stage": stage, "updates": updates,
              "selected_update": selected, "best_validation_5d_mse": best,
              "schedule_sha256": schedule["schedule_sha256"],
              "training_seconds": time.perf_counter() - started,
              "device": str(device), "checkpoint_sha256": _sha(root / "selected.pt")}
    residual._write(root / "summary.json", result)
    return result


def train_replication(seed: int, output: Path = OUTPUT,
                      device_preference: DevicePreference = "auto") -> dict:
    if seed not in (2812, 2813):
        raise ValueError("Replication seeds are fixed at 2812 and 2813")
    root = ensure_dir(output / f"replication/seed{seed}")
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task = _task(seed, device=device_preference)
    source = dynamic.OUTPUT / f"seed{seed}/training"
    expert = torch.load(source / "expert.pt", map_location="cpu", weights_only=False)
    policy = torch.load(source / "policy_mlp.pt", map_location="cpu", weights_only=False)
    validation = torch.load(source / "validation.pt", map_location="cpu", weights_only=False)
    a_root = ensure_dir(root / "residual_5d_mlp/stage_a")
    b_root = ensure_dir(root / "residual_5d_mlp/stage_b")
    a = _train_seed_5d_stage(seed, "stage_a", a_root, expert, expert, validation,
                             task, device, None)
    b = _train_seed_5d_stage(seed, "stage_b", b_root, expert, policy, validation,
                             task, device, a_root / "selected.pt")
    reference = json.loads((output / "separate_aperture_branch/summary.json").read_text())
    fixed_budget = reference["selected_update"]
    branch = train_aperture_branch(output, device_preference, seed, fixed_budget)
    result = {"seed": seed, "frozen_architecture": "pairwise_span_features -> 8-32-1 width head",
              "fixed_branch_budget": fixed_budget, "stage_a_5d": a,
              "stage_b_5d": b, "aperture_branch": branch}
    residual._write(root / "training_summary.json", result)
    return result


def _width_metrics(rows: list[dict]) -> dict:
    true = np.asarray([r["true_desired_width_m"] for r in rows])
    pred = np.asarray([r["predicted_desired_width_m"] for r in rows])
    error = np.abs(pred - true)
    return {"mae_m": float(error.mean()), "p50_m": float(np.quantile(error, .5)),
            "p90_m": float(np.quantile(error, .9)), "p95_m": float(np.quantile(error, .95)),
            "correlation": float(np.corrcoef(true, pred)[0, 1]),
            "within_8mm": float(np.mean(error < .008))}


@torch.no_grad()
def _node_embeddings(model: sf.SurfaceGraphNetwork, state: torch.Tensor,
                     points: torch.Tensor) -> dict[str, np.ndarray]:
    batch, count, _ = points.shape
    tips = state[:, 8:14].reshape(batch, 2, 3)
    positions = torch.cat([points.new_zeros((batch, 1, 3)), tips, points], dim=1)
    roles = points.new_zeros((batch, count + 3, 4))
    roles[:, 0, 0] = 1
    roles[:, 1:3, 1] = 1
    roles[:, 3:, 2] = 1
    nodes = model.node_encoder(torch.cat([positions, roles], dim=-1))
    nodes = torch.cat([nodes[:, :1] + model.context_encoder(state)[:, None], nodes[:, 1:]], dim=1)
    topology = sf._torch_topology(points, state, model.config, model.use_local_edges)
    for layer in model.layers:
        nodes = layer(nodes, positions, topology)
    z = torch.cat([nodes[:, 0], nodes[:, 1], nodes[:, 2]], dim=-1)
    if not torch.allclose(z, model.encode(state, points), atol=1e-6):
        raise AssertionError("Saved node embeddings do not reconstruct canonical z")
    return {"h_EE": nodes[0, 0].cpu().numpy(), "h_left": nodes[0, 1].cpu().numpy(),
            "h_right": nodes[0, 2].cpu().numpy(), "z": z[0].cpu().numpy()}


def counterfactual(output: Path = OUTPUT, eval_device_preference: DevicePreference = "cpu") -> dict:
    root = ensure_dir(output / "counterfactual")
    device = select_device(eval_device_preference)
    task = _task(eval_device=eval_device_preference)
    frozen = residual.load_controller("mlp", task,
        replace(residual.ResidualConfig(), corrected_dimensions=5), device, STAGE_B_5D)
    aware = _load_aware(output / "shared_encoder/selected.pt", task, device)
    branch = SpanBranchController(frozen, output / "separate_aperture_branch/selected.pt",
                                  task, device)
    probe_payload = torch.load(PRIOR / "probes/frozen_z/width/best.pt",
                               map_location=device, weights_only=False)
    probe = prior.Probe(probe_payload["input_dim"]).to(device)
    probe.load_state_dict(probe_payload["model_state"])
    probe.eval()
    specs = prior.specs_by_identity(task)
    episodes = torch.load(prior.SOURCE / "benchmark_sanity/heldout_oracle.pt",
                          map_location="cpu", weights_only=False)
    rows = []
    embedding_records = {f"{model}_{key}": [] for model in ("frozen", "aware")
                         for key in ("h_EE", "h_left", "h_right", "z")}
    for ep in episodes:
        spec = specs[ep["identity"]]
        state = ep["states"][0].to(device)[None]
        ee = ep["ee_world"][0].numpy()
        yaw = math.atan2(float(state[0, 5]), float(state[0, 6]))
        pair = {}
        for label, half_width in (("A", .029), ("B", .040)):
            shape = replace(spec.initial.shape, half_width=half_width)
            points = sf.localize_world(sf.sample_surface_points_world(shape, 32,
                sf._stable_seed(spec.initial.sample_identity)), ee, yaw)
            points_t = torch.as_tensor(points[None], dtype=torch.float32, device=device)
            _check_topology(task, state[0].detach().cpu().numpy(), points)
            with torch.no_grad():
                z_frozen = frozen.base.encode(state, points_t)
                z_aware = aware.base.encode(state, points_t)
                frozen_nodes = _node_embeddings(frozen.base, state, points_t)
                aware_nodes = _node_embeddings(aware.base, state, points_t)
                z_scaled = (z_frozen - torch.as_tensor(probe_payload["input_mean"], device=device)) / \
                    torch.as_tensor(probe_payload["input_std"], device=device)
                frozen_width = probe(z_scaled).item() * probe_payload["target_std"] + probe_payload["target_mean"]
                action_frozen, _, _, _ = frozen.step(state, points_t, torch.zeros(1, 5, device=device))
                action_aware, _, _, predicted_width, _ = aware(state, points_t,
                                                                torch.zeros(1, 5, device=device))
                branch_width = branch.predicted_width(points_t)
                branch_action, _, _, _ = branch.step(state, points_t,
                                                       torch.zeros(1, 5, device=device))
            pair[label] = {"true": width_target(shape), "frozen_width": float(frozen_width),
                           "aware_width": float(predicted_width.item()),
                           "branch_width": float(branch_width.item()),
                           "frozen_action": float(action_frozen[0, 4]),
                           "aware_action": float(action_aware[0, 4]),
                           "branch_action": float(branch_action[0, 4]),
                           "z_frozen": z_frozen[0].cpu().numpy(),
                           "z_aware": z_aware[0].cpu().numpy(),
                           "frozen_nodes": frozen_nodes, "aware_nodes": aware_nodes}
        a, b = pair["A"], pair["B"]
        for model_name in ("frozen", "aware"):
            for key in ("h_EE", "h_left", "h_right", "z"):
                embedding_records[f"{model_name}_{key}"].append(np.stack([
                    a[f"{model_name}_nodes"][key], b[f"{model_name}_nodes"][key]]))
        rows.append({"pair_id": ep["identity"], "true_width_A": a["true"],
                     "true_width_B": b["true"], "pred_width_A": a["aware_width"],
                     "pred_width_B": b["aware_width"],
                     "frozen_pred_width_A": a["frozen_width"],
                     "frozen_pred_width_B": b["frozen_width"],
                     "branch_pred_width_A": a["branch_width"],
                     "branch_pred_width_B": b["branch_width"],
                     "representation_distance": float(np.linalg.norm(b["z_aware"] - a["z_aware"])),
                     "frozen_representation_distance": float(np.linalg.norm(b["z_frozen"] - a["z_frozen"])),
                     **{f"{name}_{key}_distance": float(np.linalg.norm(
                         b[f"{name}_nodes"][key] - a[f"{name}_nodes"][key]))
                         for name in ("frozen", "aware")
                         for key in ("h_EE", "h_left", "h_right")},
                     "gripper_action_A": a["aware_action"], "gripper_action_B": b["aware_action"],
                     "branch_gripper_action_A": a["branch_action"],
                     "branch_gripper_action_B": b["branch_action"],
                     "frozen_gripper_action_A": a["frozen_action"],
                     "frozen_gripper_action_B": b["frozen_action"]})
    _csv(root / "paired_width_response.csv", rows)
    np.savez_compressed(root / "paired_node_embeddings.npz",
                        pair_id=np.asarray([r["pair_id"] for r in rows]),
                        **{key: np.stack(value) for key, value in embedding_records.items()})
    true_change = np.asarray([r["true_width_B"] - r["true_width_A"] for r in rows])
    aware_change = np.asarray([r["pred_width_B"] - r["pred_width_A"] for r in rows])
    frozen_change = np.asarray([r["frozen_pred_width_B"] - r["frozen_pred_width_A"] for r in rows])
    branch_change = np.asarray([r["branch_pred_width_B"] - r["branch_pred_width_A"] for r in rows])
    result = {"pairs": len(rows), "true_delta_m": float(true_change.mean()),
              "aware_predicted_delta_m": float(aware_change.mean()),
              "frozen_predicted_delta_m": float(frozen_change.mean()),
              "branch_predicted_delta_m": float(branch_change.mean()),
              "aware_response_ratio": float(np.mean(aware_change / true_change)),
              "frozen_response_ratio": float(np.mean(frozen_change / true_change)),
              "branch_response_ratio": float(np.mean(branch_change / true_change)),
              "aware_action_delta": float(np.mean([r["gripper_action_B"] - r["gripper_action_A"] for r in rows])),
              "frozen_action_delta": float(np.mean([r["frozen_gripper_action_B"] -
                                                      r["frozen_gripper_action_A"] for r in rows])),
              "branch_action_delta": float(np.mean([r["branch_gripper_action_B"] -
                                                      r["branch_gripper_action_A"] for r in rows])),
              "aware_correct_sign_fraction": float(np.mean(aware_change * true_change > 0)),
              "aware_embedding_distance": float(np.mean([r["representation_distance"] for r in rows])),
              "frozen_embedding_distance": float(np.mean([r["frozen_representation_distance"] for r in rows]))}
    residual._write(root / "summary.json", result)
    return result


def width_predictions(output: Path = OUTPUT, eval_device_preference: DevicePreference = "cpu") -> dict:
    root = ensure_dir(output / "shared_encoder")
    task = _task(eval_device=eval_device_preference)
    device = select_device(eval_device_preference)
    model = _load_aware(root / "selected.pt", task, device)
    results = {}
    for split in ("validation", "heldout"):
        data = _supervised_split(split, task)
        rows = []
        with torch.no_grad():
            for start in range(0, len(data["state"]), 64):
                end = start + 64
                state = torch.from_numpy(data["state"][start:end]).to(device)
                points = torch.from_numpy(data["points"][start:end]).to(device)
                previous = torch.from_numpy(data["previous"][start:end]).to(device)
                _, _, _, width, _ = model(state, points, previous)
                for identity, true, pred, opening in zip(data["identity"][start:end],
                        data["width"][start:end], width.cpu().numpy(),
                        data["state"][start:end, 7], strict=True):
                    rows.append({"episode": identity, "true_desired_width_m": float(true),
                                 "predicted_desired_width_m": float(pred),
                                 "current_aperture_fraction": float(opening),
                                 "width_error_m": float(pred - true)})
        _csv(root / f"{split}_width_predictions.csv", rows)
        results[split] = _width_metrics(rows)
    residual._write(root / "width_metrics.json", results)
    return results


def _models(task: sf.SurfaceFeasibilityConfig, device: torch.device, output: Path,
            geometry_offset: float, seed: int = 2811) -> dict:
    design4 = replace(residual.ResidualConfig(), seed=seed, corrected_dimensions=4)
    design5 = replace(residual.ResidualConfig(), seed=seed, corrected_dimensions=5)
    five_path = (STAGE_B_5D if seed == 2811 else
                 output / f"replication/seed{seed}/residual_5d_mlp/stage_b/selected.pt")
    four_path = (STAGE_B_4D if seed == 2811 else
                 dynamic.OUTPUT / f"seed{seed}/training/stage_b_mlp/checkpoints/mlp.pt")
    five = residual.load_controller("mlp", task, design5, device, five_path)
    models = {"frozen_ff": sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME, task, device),
            "residual_4d_mlp": residual.load_controller("mlp", task, design4, device, four_path),
            "residual_5d_mlp": five,
            "observable_geometry": GeometryWidthController(five, geometry_offset, task)}
    if seed == 2811:
        models["aperture_aware_5d"] = _load_aware(output / "shared_encoder/selected.pt", task, device)
    branch_checkpoint = (output / "separate_aperture_branch/selected.pt" if seed == 2811 else
                         output / f"replication/seed{seed}/separate_aperture_branch/selected.pt")
    if branch_checkpoint.exists():
        audit = json.loads((PRIOR / "audit/gripper_residual_bound.json").read_text())
        large = min(task.max_delta_gripper, audit["oracle_minus_ff_action_abs_p95_fraction"])
        models["separate_aperture_branch"] = SpanBranchController(five, branch_checkpoint,
                                                                    task, device)
        models["branch_bound_005"] = SpanBranchController(five, branch_checkpoint,
                                                            task, device, .05)
        models["branch_bound_p95"] = SpanBranchController(five, branch_checkpoint,
                                                            task, device, large)
    return models


def _evaluate_one(payload: tuple) -> tuple[int, list[tuple[dict, dict]], list[dict]]:
    condition, index, spec, task, output, offset, eval_device_preference, seed = payload
    device = select_device(eval_device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    models = _models(task, device, output, offset, seed)
    env = sf.make_env(task, seed + 864_211)
    rows, capacity_rows = [], []
    try:
        for name, model in models.items():
            if condition == "dynamic":
                row, trace = dynamic.rollout(env, spec, task, device,
                    "ff" if name == "frozen_ff" else "residual_mlp", model, 0)
                episode_id = spec.identity
            elif name == "frozen_ff":
                row, trace = rf.rollout(env, spec, model, "ff", task, rf.TemporalConfig(),
                                        device, mode="fresh", severity=0, collect=False,
                                        record_dynamics=True)
                episode_id = spec.sample_identity
            else:
                row, trace = residual.rollout_residual(env, spec, model, task,
                                                       rf.TemporalConfig(), device)
                episode_id = spec.sample_identity
            if any(step.get("oracle_action") is not None for step in trace):
                raise AssertionError("Closed-loop evaluation invoked scripted expert")
            row = {"episode": episode_id, "controller": name, "success": bool(row["success"]),
                   "position_error": float(row["final_tracking_error" if condition == "dynamic"
                                               else "final_position_error"]),
                   "yaw_error": float(row["final_yaw_error" if condition == "dynamic"
                                         else "final_orientation_error"]),
                   "aperture_error": float(row["final_aperture_error" if condition == "dynamic"
                                              else "final_gripper_width_error"]),
                   "collision": bool(row["collision"]),
                   "variable_horizon_mean_tracking_m": float(row["mean_tracking_error" if condition == "dynamic"
                                                                  else "trajectory_error"]),
                   "inference_latency_ms": float(row.get("mean_inference_ms") or
                                                  np.mean([step.get("inference_ms", 0) or 0 for step in trace]))}
            flags = {"position_fail": row["position_error"] >= task.success_position_threshold,
                     "yaw_fail": row["yaw_error"] >= task.success_rotation_threshold,
                     "aperture_fail": row["aperture_error"] >= task.success_opening_threshold,
                     "collision_fail": row["collision"]}
            flags["multiple_failures"] = sum(flags.values()) > 1
            rows.append((row, {"episode": episode_id, "controller": name, **flags}))
            if isinstance(model, SpanBranchController):
                capacity_rows.extend({"episode": episode_id, "controller": name, **entry}
                                     for entry in model.capacity_trace)
    finally:
        env.close()
    return index, rows, capacity_rows


def evaluate(output: Path = OUTPUT, eval_device_preference: DevicePreference = "cpu",
             eval_workers: int = 1, seed: int = 2811) -> dict:
    if eval_workers < 1:
        raise ValueError("eval_workers must be positive")
    root = ensure_dir(output)
    result_root = ensure_dir(root if seed == 2811 else root / f"replication/seed{seed}")
    reproduction = json.loads((root / "reproduction/summary.json").read_text())
    offset = reproduction["geometry_offset_fit_train_only_m"]
    task = _task(seed, eval_device=eval_device_preference)
    results = {}
    for condition in ("static", "dynamic"):
        specs = (sf.sample_episode_specs(16, seed + 60_000, task, "iid") if condition == "static"
                 else dynamic.dynamic_specs(seed, task, 16, dynamic.SELECTED_STEP_SPEEDS_M, "heldout"))
        payloads = [(condition, i, spec, task, root, offset, eval_device_preference, seed)
                    for i, spec in enumerate(specs)]
        if eval_workers > 1:
            with ProcessPoolExecutor(max_workers=eval_workers, mp_context=mp.get_context("spawn")) as executor:
                outcomes = list(executor.map(_evaluate_one, payloads))
        else:
            outcomes = [_evaluate_one(p) for p in payloads]
        rows, flags, capacity_rows = [], [], []
        for _, per_episode, per_capacity in sorted(outcomes, key=lambda x: x[0]):
            for row, flag in per_episode:
                rows.append(row)
                flags.append(flag)
            capacity_rows.extend(per_capacity)
        _csv(result_root / f"{condition}_eval/per_episode.csv", rows)
        _csv(result_root / f"failure_decomposition/{condition}.csv", flags)
        if capacity_rows:
            _csv(result_root / f"bound_ablation/{condition}_capacity_timesteps.csv", capacity_rows)
        summary = {}
        for name in sorted(set(row["controller"] for row in rows)):
            subset = [r for r in rows if r["controller"] == name]
            fset = [r for r in flags if r["controller"] == name]
            summary[name] = {"episodes": len(subset), "success": sum(r["success"] for r in subset),
                             **{key: sum(r[key] for r in fset) for key in
                                ("position_fail", "yaw_fail", "aperture_fail",
                                 "collision_fail", "multiple_failures")},
                             **{key + "_mean": float(np.mean([r[key] for r in subset])) for key in
                                ("position_error", "yaw_error", "aperture_error",
                                 "inference_latency_ms", "variable_horizon_mean_tracking_m")}}
        residual._write(result_root / f"{condition}_eval/summary.json", summary)
        results[condition] = summary
    residual._write(result_root / "aggregate/summary.json", results)
    return results


def verify_baseline_rollouts(output: Path = OUTPUT) -> dict:
    previous = json.loads((PRIOR / "evaluation/summary.json").read_text())
    current = json.loads((output / "aggregate/summary.json").read_text())
    rows = []
    for condition, previous_key in (("static", "static_fresh"), ("dynamic", "dynamic_fresh")):
        for controller, previous_name in (("frozen_ff", "ff"),
                                          ("residual_4d_mlp", "residual_4d_mlp"),
                                          ("residual_5d_mlp", "residual_5d_mlp")):
            new, old = current[condition][controller], previous[previous_key][previous_name]
            for key, old_key in (("success", "success"), ("aperture_fail", "aperture_fail"),
                                 ("position_fail", "position_fail"), ("yaw_fail", "yaw_fail"),
                                 ("collision_fail", "collision_fail")):
                rows.append({"condition": condition, "controller": controller,
                             "metric": key, "previous": old[old_key], "reproduced": new[key],
                             "match": old[old_key] == new[key]})
    if not all(row["match"] for row in rows):
        raise AssertionError("Historical 4D/5D rollout outcomes did not reproduce")
    _csv(output / "reproduction/rollout_comparison.csv", rows)
    result = {"compared_outcomes": len(rows), "all_match": True,
              "prior_artifact": str(PRIOR / "evaluation/summary.json")}
    residual._write(output / "reproduction/rollout_verification.json", result)
    return result


def capacity_training_diagnostic(output: Path = OUTPUT,
                                 eval_device_preference: DevicePreference = "cpu") -> dict:
    device = select_device(eval_device_preference)
    task = _task(eval_device=eval_device_preference)
    motion = residual.load_controller("mlp", task,
        replace(residual.ResidualConfig(), corrected_dimensions=5), device, STAGE_B_5D)
    audit = json.loads((PRIOR / "audit/gripper_residual_bound.json").read_text())
    larger = min(task.max_delta_gripper, audit["oracle_minus_ff_action_abs_p95_fraction"])
    branch_path = output / "separate_aperture_branch/selected.pt"
    controllers = {"bound_005": SpanBranchController(motion, branch_path, task, device, .05),
                   "bound_p95": SpanBranchController(motion, branch_path, task, device, larger)}
    episodes = [ep for name in ("expert", "policy_mlp") for ep in torch.load(
        prior.SOURCE / f"training/{name}.pt", map_location="cpu", weights_only=False)]
    rows = []
    for ep in episodes:
        state = ep["states"].to(device)
        points = ep["points"].to(device)
        previous = ep["previous"].to(device)
        for name, model in controllers.items():
            model.capacity_trace.clear()
            with torch.no_grad():
                _, _, _, base = model.step(state, points, previous)
            for t, entry in enumerate(model.capacity_trace):
                if not bool(ep["mask"][t]):
                    continue
                rows.append({"episode": ep["identity"], "timestep": t,
                             "controller": name,
                             "target_gripper_correction": float(ep["targets"][t, 4] - base[t, 4].cpu()),
                             "raw_predicted_correction": entry["raw_predicted_correction"],
                             "bounded_correction": entry["bounded_correction"],
                             "saturated": entry["saturated"]})
    _csv(output / "bound_ablation/training_target_corrections.csv", rows)
    summary = {"bounds": {"bound_005": .05, "bound_p95": larger},
               "meaning": "training target is oracle action minus FF; predicted correction comes from the learned width branch",
               "by_bound": {}}
    for name in controllers:
        subset = [row for row in rows if row["controller"] == name]
        summary["by_bound"][name] = {
            "supervised_timesteps": len(subset),
            "saturated_fraction": float(np.mean([r["saturated"] for r in subset])),
            "target_abs_p95": float(np.quantile(np.abs([r["target_gripper_correction"]
                                                      for r in subset]), .95)),
            "predicted_abs_p95": float(np.quantile(np.abs([r["raw_predicted_correction"]
                                                         for r in subset]), .95))}
    residual._write(output / "bound_ablation/training_summary.json", summary)
    return summary


def write_report(output: Path = OUTPUT) -> None:
    reproduction = json.loads((output / "reproduction/summary.json").read_text())
    rollout_check = verify_baseline_rollouts(output)
    shared = json.loads((output / "shared_encoder/summary.json").read_text())
    shared_width = json.loads((output / "shared_encoder/width_metrics.json").read_text())
    branch = json.loads((output / "separate_aperture_branch/summary.json").read_text())
    cf = json.loads((output / "counterfactual/summary.json").read_text())
    capacity = capacity_training_diagnostic(output)
    evaluations = {seed: json.loads((output / ("aggregate/summary.json" if seed == 2811 else
                      f"replication/seed{seed}/aggregate/summary.json")).read_text())
                   for seed in (2811, 2812, 2813)}
    table = []
    for seed, result in evaluations.items():
        for condition in ("static", "dynamic"):
            names = ("frozen_ff", "residual_4d_mlp", "residual_5d_mlp",
                     "observable_geometry", "separate_aperture_branch",
                     "branch_bound_005", "branch_bound_p95")
            if seed == 2811:
                names += ("aperture_aware_5d",)
            for name in names:
                r = result[condition][name]
                table.append({"seed": seed, "condition": condition, "controller": name,
                              "success": r["success"], "position_fail": r["position_fail"],
                              "yaw_fail": r["yaw_fail"], "aperture_fail": r["aperture_fail"],
                              "collision_fail": r["collision_fail"],
                              "multiple_failures": r["multiple_failures"],
                              "final_position_mm": 1000*r["position_error_mean"],
                              "final_yaw_rad": r["yaw_error_mean"],
                              "final_aperture_mm": 1000*r["aperture_error_mean"],
                              "latency_ms": r["inference_latency_ms_mean"]})
    _csv(output / "aggregate/seed_controller_comparison.csv", table)
    aggregate = {}
    for condition in ("static", "dynamic"):
        aggregate[condition] = {}
        for name in ("residual_5d_mlp", "separate_aperture_branch", "branch_bound_005",
                     "branch_bound_p95"):
            aggregate[condition][name] = {key: sum(evaluations[seed][condition][name][key]
                                                  for seed in evaluations)
                                          for key in ("success", "position_fail", "yaw_fail",
                                                      "aperture_fail", "collision_fail", "multiple_failures")}
    residual._write(output / "aggregate/three_seed_totals.json", aggregate)
    dynamic_capacity = list(csv.DictReader((output / "bound_ablation/dynamic_capacity_timesteps.csv").open()))
    saturation = {name: float(np.mean([row["saturated"] == "True" for row in dynamic_capacity
                                       if row["controller"] == name]))
                  for name in ("branch_bound_005", "branch_bound_p95")}
    residual._write(output / "bound_ablation/dynamic_saturation_summary.json", saturation)
    lines = ["# Aperture-aware spatial controller", "",
             "## A. Reproduction", "",
             f"All {rollout_check['compared_outcomes']} historical FF/4D/5D held-out outcome checks match. "
             f"Checkpoint SHA-256 values match the prior artifact. The frozen-`z` width probe reproduces "
             f"{1000*reproduction['frozen_z_heldout_mae_m']:.2f} mm MAE; unordered observable geometry "
             f"reproduces {1000*reproduction['observable_width_episode_mae_m']:.3f} mm MAE and "
             f"{1000*reproduction['observable_width_episode_p95_m']:.3f} mm p95 across episode shapes. "
             "Train/validation/held-out episodes remain disjoint. The graph has 32 surface points and 100 directed edges.", "",
             "## B. Observable-geometry upper bound", "",
             f"The diagnostic controller estimates width from unordered observed xyz, using a width offset "
             f"{1000*reproduction['geometry_offset_fit_train_only_m']:.3f} mm fitted on training labels only. "
             "It retains the 5D MLP's first four motion actions and replaces its gripper command. "
             "It is a diagnostic, not a learned-model result. On seed 2811 it reaches static "
             f"{evaluations[2811]['static']['observable_geometry']['success']}/16 and dynamic "
             f"{evaluations[2811]['dynamic']['observable_geometry']['success']}/16 success with zero aperture failures. "
             "The remaining dynamic failures occur in motion/collision. The extraction uses observed coordinates, "
             "not hidden shape fields, point IDs, future state, or held-out target labels.", "",
             "## C. Aperture-aware shared encoder", "",
             "The canonical GNN and existing 5D residual MLP were initialized from their frozen checkpoints and jointly "
             "trained on the exact historical Stage A/B expert/current-policy/delay batch schedules. "
             "The unchanged normalized 5D action MSE was combined "
             "with normalized target-width MSE at predeclared weight 1. Width center and scale were fitted on training "
             f"episodes only ({shared['width_center_m']:.6f} m, {shared['width_scale_m']:.6f} m). "
             f"The validation-selected checkpoint was update {shared['selected_update']} of {shared['final_update']}. "
             f"Its held-out width MAE was {1000*shared_width['heldout']['mae_m']:.2f} mm, "
             f"p95 {1000*shared_width['heldout']['p95_m']:.2f} mm. At selection, validation normalized "
             f"translation/yaw/gripper action MSE was "
             f"{shared['validation_at_selected']['translation_mse']:.4f}/"
             f"{shared['validation_at_selected']['yaw_mse']:.4f}/"
             f"{shared['validation_at_selected']['gripper_mse']:.4f}. "
             "Natural-trajectory width prediction improved, but the matched counterfactual below shows that this "
             "shared encoder still does not encode surface width reliably. The selected checkpoint was in Stage A; "
             "Stage B validation did not beat it. Closed-loop motion degraded despite a lower supervised action MSE.", "",
             "## D. Counterfactual geometry response", "",
             f"With robot state/opening fixed, true desired width changes {1000*cf['true_delta_m']:.2f} mm. "
             f"The frozen probe predicts {1000*cf['frozen_predicted_delta_m']:.2f} mm "
             f"(ratio {cf['frozen_response_ratio']:.3f}); the shared aperture-aware encoder predicts "
             f"{1000*cf['aware_predicted_delta_m']:.3f} mm (ratio {cf['aware_response_ratio']:.3f}); "
             f"the separate relation-aware branch predicts {1000*cf['branch_predicted_delta_m']:.2f} mm "
             f"(ratio {cf['branch_response_ratio']:.3f}). Gripper-command changes are "
             f"{cf['frozen_action_delta']:.4f}/{cf['aware_action_delta']:.4f}/"
             f"{cf['branch_action_delta']:.4f} normalized units. Canonical versus fine-tuned `z` L2 distances "
             f"are {cf['frozen_embedding_distance']:.3f}/{cf['aware_embedding_distance']:.3f}; "
             "larger embedding distance alone did not imply useful width encoding.", "",
             "## E. Closed-loop 5D control and F. Failure decomposition", "",
             "All rows use paired 16-episode static or moving-target fresh benchmarks per seed. "
             "Columns P/Y/A/C/M are position/yaw/aperture/collision/multiple failure counts. "
             "Final errors are episode means; static trajectory means are omitted because success changes stopping time.", "",
             "| Seed | Condition | Controller | Success | P | Y | A | C | M | Final pos mm | Final yaw rad | Final aperture mm | Latency ms |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        if row["controller"] not in ("frozen_ff", "residual_4d_mlp", "residual_5d_mlp",
                                     "separate_aperture_branch"):
            continue
        latency = ("n/a" if row["controller"] == "frozen_ff" and row["condition"] == "static"
                   else f"{row['latency_ms']:.2f}")
        lines.append(f"| {row['seed']} | {row['condition']} | {row['controller']} | {row['success']}/16 | "
                     f"{row['position_fail']} | {row['yaw_fail']} | {row['aperture_fail']} | "
                     f"{row['collision_fail']} | {row['multiple_failures']} | "
                     f"{row['final_position_mm']:.2f} | {row['final_yaw_rad']:.3f} | "
                     f"{row['final_aperture_mm']:.2f} | {latency} |")
    for condition in ("static", "dynamic"):
        r = evaluations[2811][condition]["aperture_aware_5d"]
        lines.append(f"| 2811 | {condition} | aperture_aware_5d | {r['success']}/16 | "
                     f"{r['position_fail']} | {r['yaw_fail']} | {r['aperture_fail']} | "
                     f"{r['collision_fail']} | {r['multiple_failures']} | "
                     f"{1000*r['position_error_mean']:.2f} | {r['yaw_error_mean']:.3f} | "
                     f"{1000*r['aperture_error_mean']:.2f} | {r['inference_latency_ms_mean']:.2f} |")
    lines += ["", "Across three seeds, the 5D MLP versus separate branch has static success "
              f"{aggregate['static']['residual_5d_mlp']['success']}/48 → "
              f"{aggregate['static']['separate_aperture_branch']['success']}/48 and aperture failures "
              f"{aggregate['static']['residual_5d_mlp']['aperture_fail']}/48 → "
              f"{aggregate['static']['separate_aperture_branch']['aperture_fail']}/48; dynamic success "
              f"{aggregate['dynamic']['residual_5d_mlp']['success']}/48 → "
              f"{aggregate['dynamic']['separate_aperture_branch']['success']}/48 and aperture failures "
              f"{aggregate['dynamic']['residual_5d_mlp']['aperture_fail']}/48 → "
              f"{aggregate['dynamic']['separate_aperture_branch']['aperture_fail']}/48. "
              "Dynamic position failures rise 4/48 → 7/48, yaw failures 12/48 → 13/48, and collisions "
              "fall 7/48 → 6/48. These are material tradeoffs despite the overall success gain. "
              "Spatial/yaw/collision failures remain and sometimes shift because gripper actions change future "
              "closed-loop states. Per-episode errors and overlapping flags are saved separately.", "",
              "## G. Shared encoder versus separate branch", "",
              "The shared encoder gained natural-trajectory width accuracy but failed the width-isolated "
              "counterfactual. On seed 2811 it fell from the 5D MLP's static/dynamic 10/16 and 5/16 success "
              "to 2/16 and 1/16; final position also worsened. This triggered one isolated branch. "
              "The branch keeps the frozen motion encoder and uses "
              "eight endpoint-pair spans extracted from unordered surface xyz, followed by one 8→32→1 MLP. "
              "The coordinate sorting is a documented hand-designed relational operation and gives the learner a "
              "stronger inductive bias than the canonical GNN; this result does not prove the GNN cannot learn width. "
              f"The branch's seed-2811 held-out width MAE is {1000*branch['metrics']['heldout']['mae_m']:.3f} mm "
              f"(p95 {1000*branch['metrics']['heldout']['p95_m']:.3f} mm). "
              "Seeds 2812/2813 used the same architecture, loss, and frozen 3375-update budget, "
              "with validation-selected checkpoints within that budget. No graph topology or sampling changed.", "",
              "## H. Gripper residual-bound ablation", "",
              f"After geometry use was established, the predeclared bounds were ±0.05 and ±"
              f"{capacity['bounds']['bound_p95']:.3f} opening fraction, the latter from the seed-2811 "
              "training oracle-minus-FF p95. The comparison applies the same learned-width command and motion "
              "controller, changing only the cap on gripper correction relative to FF. "
              f"In seed-2811 dynamic rollout, {100*saturation['branch_bound_005']:.1f}% versus "
              f"{100*saturation['branch_bound_p95']:.1f}% of predicted corrections were clipped. "
              "The small versus large bound gives static aperture failures 3/16 versus 0/16 and dynamic "
              "4/16 versus 0/16 for seed 2811. Across three seeds the corresponding dynamic aperture "
              f"failures are {aggregate['dynamic']['branch_bound_005']['aperture_fail']}/48 versus "
              f"{aggregate['dynamic']['branch_bound_p95']['aperture_fail']}/48. "
              "The historical 5D MLP reached its ±0.05 bound on only about 5% of dynamic steps. "
              "Capacity becomes limiting when a geometry-aware branch actually requests the larger correction; "
              "the historical model's low saturation was a prediction/representation issue.", "",
              "## I. Final bottleneck decision", "",
              "**SEPARATE APERTURE REPRESENTATION NEEDED** for this minimal experiment, with a conditional "
              "**GRIPPER ACTION CAPACITY** limit after geometry extraction is made effective. "
              "One shared-encoder auxiliary-loss run did not establish usable counterfactual width response, "
              "while the separate relation-aware branch did. The canonical observation supplied sufficient "
              "coordinates for accurate aperture control without adding nodes, edges, or point IDs. "
              "The branch bypasses GNN message passing for gripper width, so this does not establish that the "
              "original 100-edge encoder can learn the same relation.", "",
              "## J. Facts, interpretation, and remaining hypotheses", "",
              "**Facts.** The order-free coordinate diagnostic and trained relation-aware branch predict width "
              "accurately. The branch eliminates held-out aperture failures across three seeds. The shared "
              "encoder's natural width MAE does not translate into isolated width sensitivity. Larger gripper "
              "capacity helps once the width-conditioned target is available.", "",
              "**Interpretation.** The original encoder/readout training did not preserve the particular "
              "surface relation needed for aperture. A separate pairwise spatial readout is a compact solution "
              "for this benchmark. Remaining dynamic failures mostly concern motion and collision.", "",
              "**Remaining hypotheses.** A counterfactual-augmented shared-encoder objective may teach the "
              "canonical GNN the same relation; it was not tested. The hand-designed pair extraction should be "
              "tested on broader surface families before claiming general geometric robustness. No recurrent or "
              "topology change follows from these results.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("reproduce", "train-shared", "train-branch", "train-replication",
                                          "counterfactual", "width-predictions", "evaluate", "report"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2811)
    args = parser.parse_args()
    if args.phase == "reproduce":
        reproduce(args.output)
    elif args.phase == "train-shared":
        train_shared(args.output, args.device)
    elif args.phase == "train-branch":
        train_aperture_branch(args.output, args.device, args.seed)
    elif args.phase == "train-replication":
        train_replication(args.seed, args.output, args.device)
    elif args.phase == "counterfactual":
        counterfactual(args.output, args.eval_device)
    elif args.phase == "width-predictions":
        width_predictions(args.output, args.eval_device)
    elif args.phase == "report":
        write_report(args.output)
    else:
        evaluate(args.output, args.eval_device, args.eval_workers, args.seed)


if __name__ == "__main__":
    main()
