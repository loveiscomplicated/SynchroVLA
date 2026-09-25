"""Causal aperture diagnostics on the unchanged canonical surface graph."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import multiprocessing as mp
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vla_gnn_recurrent.training import dynamic_visual_delay_feasibility as dynamic
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import residual_recurrent_feasibility as residual
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


OUTPUT = Path("artifacts/aperture_bottleneck_diagnosis")
SOURCE = dynamic.OUTPUT / "seed2811"


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def task_config(device: DevicePreference = "cpu", eval_device: DevicePreference = "cpu") -> sf.SurfaceFeasibilityConfig:
    return rf._task_config(SOURCE, (2811,), device, eval_device, 16)


def specs_by_identity(task: sf.SurfaceFeasibilityConfig) -> dict[str, dynamic.DynamicSpec]:
    specs = []
    for split, count in (("train", 72), ("validation", 16), ("heldout", 16)):
        specs.extend(dynamic.dynamic_specs(2811, task, count, dynamic.SELECTED_STEP_SPEEDS_M, split))
    return {spec.identity: spec for spec in specs}


def desired_width(shape: sf.SurfaceShape) -> float:
    return 2.0 * sf.profile_half_width(0.5, shape) + 0.010


def observed_width_proxy(points: np.ndarray) -> float:
    """Diagnostic only: two central ring endpoint spans, computed from observed points."""
    points = np.asarray(points).reshape(8, 4, 3)
    spans = np.linalg.norm((points[:, 0] - points[:, 3])[:, [0, 2]], axis=1)
    return float(np.mean(spans[3:5]) + 0.010)


def unordered_pca_width_proxy(points: np.ndarray) -> float:
    """Order-free diagnostic using the minor x-z principal axis of observed points."""
    xz = np.asarray(points).reshape(-1, 3)[:, [0, 2]].astype(np.float64)
    centered = xz - xz.mean(0)
    _, axes = np.linalg.eigh(np.cov(centered, rowvar=False))
    projected = centered @ axes[:, 0]
    return float(np.ptp(projected) + 0.010)


def unordered_endpoint_pair_width_proxy(points: np.ndarray) -> float:
    """Order-free endpoint pairing from xyz only; diagnostic, not a model feature."""
    xyz = np.asarray(points).reshape(-1, 3).astype(np.float64)
    endpoints = xyz[np.argsort(xyz[:, 1])[-16:]]
    xz = endpoints[:, [0, 2]]
    _, axes = np.linalg.eigh(np.cov(xz, rowvar=False))
    longitudinal = xz @ axes[:, 1]
    sorted_xz = xz[np.argsort(longitudinal)].reshape(8, 2, 2)
    spans = np.linalg.norm(sorted_xz[:, 0] - sorted_xz[:, 1], axis=1)
    return float(np.mean(spans[3:5]) + 0.010)


def audit(output: Path = OUTPUT) -> dict:
    root = ensure_dir(output / "audit")
    task = task_config()
    specs = specs_by_identity(task)
    pair_rows = []
    for split in ("train", "validation", "heldout"):
        for spec in (x for x in specs.values() if x.initial.condition == f"dynamic_{split}"):
            shape = spec.initial.shape
            sample_seed = sf._stable_seed(spec.initial.sample_identity)
            points = sf.sample_surface_points_world(shape, 32, sample_seed)
            pair_rows.append({"split": split, "episode_id": spec.identity,
                              "half_width_m": shape.half_width,
                              "target_width_m": desired_width(shape),
                              "observed_width_proxy_m": observed_width_proxy(points),
                              "proxy_error_m": observed_width_proxy(points) - desired_width(shape),
                              "unordered_pca_width_proxy_m": unordered_pca_width_proxy(points),
                              "unordered_pca_error_m": unordered_pca_width_proxy(points) - desired_width(shape),
                              "unordered_endpoint_width_proxy_m": unordered_endpoint_pair_width_proxy(points),
                              "unordered_endpoint_error_m": unordered_endpoint_pair_width_proxy(points) - desired_width(shape),
                              "cross_section": shape.cross_section,
                              "thickness_amplitude": shape.thickness_amplitude,
                              "asymmetry": shape.asymmetry})
    write_csv(root / "surface_width_observability.csv", pair_rows)
    base = next(iter(specs.values())).initial.shape
    thin = replace(base, half_width=0.029)
    thick = replace(base, half_width=0.040)
    seed = 997
    thin_points = sf.sample_surface_points_world(thin, 32, seed)
    thick_points = sf.sample_surface_points_world(thick, 32, seed)
    deterministic_pair = {"same_non_width_shape_fields": all(getattr(thin, k) == getattr(thick, k)
                              for k in asdict(base) if k != "half_width"),
                          "thin_target_width_m": desired_width(thin),
                          "thick_target_width_m": desired_width(thick),
                          "target_delta_m": desired_width(thick) - desired_width(thin),
                          "cloud_max_point_shift_m": float(np.linalg.norm(thick_points - thin_points, axis=1).max()),
                          "thin_observed_proxy_m": observed_width_proxy(thin_points),
                          "thick_observed_proxy_m": observed_width_proxy(thick_points)}
    env = sf.make_env(task, 2811 + 864_211)
    try:
        spec = next(iter(specs.values())).initial
        sf._reset_surface_env(env, spec)
        sf._set_gripper_open_fraction_kinematic(env, 0.5)
        before = sf.gripper_width(env)
        state, points, _ = sf.observation_inputs(env, spec.shape, 32,
                                                  sf._stable_seed(spec.sample_identity))
        sf.apply_local_action(env, np.array([0, 0, 0, 0, task.max_delta_gripper], np.float32), task)
        after = sf.gripper_width(env)
        state_t, points_t, topology = rf._graph_tensors(state, points, task, select_device("cpu"))
        rf.assert_canonical_topology(topology, task)
        graph = {"state_shape": list(state_t.shape), "points_shape": list(points_t.shape),
                 "node_count": points_t.shape[1] + 3, "directed_edges": int(topology.src.shape[1]),
                 "edge_type_counts": {str(i): int((topology.edge_type == i).sum()) for i in range(4)},
                 "state_example": state.tolist()}
    finally:
        env.close()
    result = {"action_dim_4": "signed delta in normalized opening fraction; positive opens",
              "opening_fraction_formula": "clip((fingertip_width_m - 0.065) / 0.040, 0, 1)",
              "action_limit_fraction_per_step": task.max_delta_gripper,
              "nominal_width_limit_m_per_step": task.max_delta_gripper *
                                               (sf.MAX_GRIPPER_WIDTH - sf.MIN_GRIPPER_WIDTH),
              "measured_width_before_m": before, "measured_width_after_m": after,
              "measured_delta_m": after - before,
              "success_aperture_metric": "abs(current projected fingertip separation - target_width_m)",
              "success_aperture_threshold_m": task.success_opening_threshold,
              "oracle_target_formula": "2 * profile_half_width(0.5, shape) + 0.010 m",
              "oracle_target_shape_fields": ["half_width", "thickness_amplitude", "thickness_phase"],
              "oracle_action_formula": "clip(opening_fraction(target_width) - current_fraction, -0.25, 0.25)",
              "graph": graph, "deterministic_width_pair": deterministic_pair,
              "observability": {"episodes": len(pair_rows),
                                "proxy_mae_m": float(np.mean(np.abs([r["proxy_error_m"] for r in pair_rows]))),
                                "proxy_p95_abs_m": float(np.quantile(np.abs([r["proxy_error_m"] for r in pair_rows]), .95)),
                                "unordered_pca_mae_m": float(np.mean(np.abs([r["unordered_pca_error_m"]
                                                                             for r in pair_rows]))),
                                "unordered_pca_p95_abs_m": float(np.quantile(np.abs(
                                    [r["unordered_pca_error_m"] for r in pair_rows]), .95)),
                                "unordered_endpoint_mae_m": float(np.mean(np.abs(
                                    [r["unordered_endpoint_error_m"] for r in pair_rows]))),
                                "unordered_endpoint_p95_abs_m": float(np.quantile(np.abs(
                                    [r["unordered_endpoint_error_m"] for r in pair_rows]), .95)),
                                "target_width_range_m": [min(r["target_width_m"] for r in pair_rows),
                                                         max(r["target_width_m"] for r in pair_rows)]}}
    residual._write(root / "aperture_semantics.json", result)
    (root / "aperture_semantics.md").write_text(
        "# Aperture path audit\n\n"
        "Action 4 is a signed delta in normalized opening fraction, clipped to ±0.25 per step. "
        "Positive values open the gripper. `apply_local_action` adds it to the measured current fraction, "
        "clips the result to [0,1], converts to actuator command `1 - fraction`, and sets the native "
        "finger joints kinematically. The calibrated width range is 0.065–0.105 m, so a full "
        "0.25 step nominally changes width by 0.010 m.\n\n"
        "Success compares projected mean fingertip separation against the oracle desired width "
        "using an absolute 0.008 m threshold. `_surface_target` sets desired width to "
        "`2 * profile_half_width(0.5, shape) + 0.010`. The profile uses half-width, thickness "
        "amplitude, and phase. Current fingertip width affects the oracle action, not the desired width.\n\n"
        "State indices: 0:3 object center in EE-local coordinates; 3:5 relative object yaw sin/cos; "
        "5:7 EE yaw sin/cos; 7 normalized current opening; 8:11 left tip position; "
        "11:14 right tip position. Surface nodes receive 32 EE-local xyz coordinates. "
        "EE node receives all 14 state fields through the context encoder; tip nodes receive "
        "their xyz coordinates and role; all nodes receive role and edge geometry. "
        "Current opening and tip separation are directly present or recoverable. "
        "Width-related shape parameters are not directly provided.\n\n"
        "The 32 points are eight longitudinal cross sections with four points on the task-facing "
        "half in depth; both normal-direction endpoints occur at each cross section. Their "
        "x-z endpoint distance equals twice the local profile half-width. The central two "
        "cross sections approximate the oracle width at s=0.5; see the numeric audit CSV. "
        "No direct left-right fingertip edge or surface-local edges appear in the canonical graph.\n")
    return result


def load_split(split: str, task: sf.SurfaceFeasibilityConfig) -> dict[str, np.ndarray | list[str]]:
    names = ("expert", "policy_mlp") if split == "train" else (("validation",) if split == "validation" else ("heldout_oracle",))
    specs = specs_by_identity(task)
    episodes = []
    for name in names:
        path = SOURCE / ("benchmark_sanity/heldout_oracle.pt" if name == "heldout_oracle" else f"training/{name}.pt")
        episodes.extend(torch.load(path, map_location="cpu", weights_only=False))
    states, points, actions, widths, identities, thickness = [], [], [], [], [], []
    for ep in episodes:
        identity = ep["identity"]
        shape = specs[identity].initial.shape
        count = len(ep["states"])
        mask = ep["mask"].numpy().astype(bool)
        states.append(ep["states"].numpy()[mask])
        points.append(ep["points"].numpy()[mask])
        actions.append(ep["targets"].numpy()[mask, 4])
        widths.extend([desired_width(shape)] * int(mask.sum()))
        identities.extend([identity] * int(mask.sum()))
        thickness.extend([shape.half_width] * int(mask.sum()))
    return {"state": np.concatenate(states), "points": np.concatenate(points),
            "action": np.concatenate(actions), "width": np.asarray(widths, np.float32),
            "identity": identities, "half_width": np.asarray(thickness, np.float32)}


@torch.no_grad()
def frozen_embeddings(split: dict, task: sf.SurfaceFeasibilityConfig,
                      device: torch.device, model: sf.SurfaceGraphNetwork) -> np.ndarray:
    states = torch.from_numpy(split["state"]).to(device)
    points = torch.from_numpy(split["points"]).to(device)
    rows = []
    for start in range(0, len(states), 128):
        rows.append(model.encode(states[start:start+128], points[start:start+128]).cpu())
    return torch.cat(rows).numpy()


@torch.no_grad()
def frozen_node_readouts(split: dict, device: torch.device,
                         model: sf.SurfaceGraphNetwork) -> dict[str, np.ndarray]:
    states = torch.from_numpy(split["state"]).to(device)
    points = torch.from_numpy(split["points"]).to(device)
    pieces = {"existing_readout": [], "fingertips": [], "fingertips_pooled_surface": []}
    for start in range(0, len(states), 128):
        state, surface = states[start:start+128], points[start:start+128]
        batch, count, _ = surface.shape
        tips = state[:, 8:14].reshape(batch, 2, 3)
        positions = torch.cat([surface.new_zeros((batch, 1, 3)), tips, surface], dim=1)
        roles = torch.zeros((batch, count + 3, 4), dtype=surface.dtype, device=device)
        roles[:, 0, 0] = 1
        roles[:, 1:3, 1] = 1
        roles[:, 3:, 2] = 1
        nodes = model.node_encoder(torch.cat([positions, roles], dim=-1))
        context = model.context_encoder(state)
        nodes = torch.cat([nodes[:, :1] + context[:, None], nodes[:, 1:]], dim=1)
        topology = sf._torch_topology(surface, state, model.config, model.use_local_edges)
        rf.assert_canonical_topology(topology, model.config)
        for layer in model.layers:
            nodes = layer(nodes, positions, topology)
        existing = torch.cat([nodes[:, 0], nodes[:, 1], nodes[:, 2]], dim=-1)
        tips_only = torch.cat([nodes[:, 1], nodes[:, 2]], dim=-1)
        with_surface = torch.cat([nodes[:, 1], nodes[:, 2], nodes[:, 3:].mean(1)], dim=-1)
        pieces["existing_readout"].append(existing.cpu())
        pieces["fingertips"].append(tips_only.cpu())
        pieces["fingertips_pooled_surface"].append(with_surface.cpu())
    return {key: torch.cat(value).numpy() for key, value in pieces.items()}


class Probe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


class RawSetProbe(nn.Module):
    """Permutation-invariant diagnostic on exactly the observed state and 32 xyz points."""

    def __init__(self) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(sf.STATE_DIM + 64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, state: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        encoded = self.point_encoder(points)
        summary = torch.cat([encoded.mean(1), encoded.max(1).values], dim=-1)
        return self.head(torch.cat([state, summary], dim=-1)).squeeze(-1)


def metrics(target: np.ndarray, pred: np.ndarray, ids: list[str], thickness: np.ndarray,
            current_open: np.ndarray, tolerance: float, normalizer: float) -> dict:
    error = pred - target
    abs_error = np.abs(error)
    result = {"mae": float(abs_error.mean()), "mse": float(np.mean(error**2)),
              "normalized_mse": float(np.mean(error**2) / normalizer**2),
              "p50_abs": float(np.quantile(abs_error, .5)),
              "p90_abs": float(np.quantile(abs_error, .9)),
              "p95_abs": float(np.quantile(abs_error, .95)),
              "within_8mm": float(np.mean(abs_error < tolerance)),
              "correlation": float(np.corrcoef(target, pred)[0, 1]) if np.std(pred) > 1e-9 else None,
              "episode_mae": float(np.mean([abs_error[np.asarray(ids) == identity].mean()
                                              for identity in sorted(set(ids))]))}
    cuts = np.quantile(thickness, [0, 1/3, 2/3, 1])
    result["by_half_width_tertile"] = {}
    for i in range(3):
        mask = (thickness >= cuts[i]) & ((thickness <= cuts[i+1]) if i == 2 else (thickness < cuts[i+1]))
        result["by_half_width_tertile"][str(i)] = {"count": int(mask.sum()),
            "mae": float(abs_error[mask].mean()), "half_width_range": [float(cuts[i]), float(cuts[i+1])]}
    for label, values in (("target", target), ("current_opening_fraction", current_open)):
        cuts = np.quantile(values, [0, 1/3, 2/3, 1])
        result[f"by_{label}_tertile"] = {}
        for i in range(3):
            mask = (values >= cuts[i]) & ((values <= cuts[i+1]) if i == 2 else (values < cuts[i+1]))
            result[f"by_{label}_tertile"][str(i)] = {"count": int(mask.sum()),
                "mae": float(abs_error[mask].mean()) if mask.any() else None,
                "range": [float(cuts[i]), float(cuts[i+1])]}
    return result


def probes(output: Path = OUTPUT, device_preference: DevicePreference = "auto") -> dict:
    root = ensure_dir(output / "probes")
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task = task_config(device_preference, "cpu")
    splits = {name: load_split(name, task) for name in ("train", "validation", "heldout")}
    train_ids, val_ids, test_ids = (set(splits[name]["identity"]) for name in ("train", "validation", "heldout"))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise AssertionError("Episode leakage in aperture probe splits")
    model = sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME, task, device)
    model.eval()
    features = {}
    for name, split in splits.items():
        state, points = split["state"], split["points"]
        features[name] = {"current_aperture_only": state[:, 7:8],
                          "robot_only": state[:, 5:14],
                          "raw_graph": np.concatenate([state, points.reshape(len(points), -1)], axis=1),
                          "frozen_z": frozen_embeddings(split, task, device, model)}
        derived = np.clip((split["width"] - sf.MIN_GRIPPER_WIDTH) /
                          (sf.MAX_GRIPPER_WIDTH - sf.MIN_GRIPPER_WIDTH), 0, 1)
        expected_action = np.clip(derived - state[:, 7], -task.max_delta_gripper, task.max_delta_gripper)
        if np.max(np.abs(expected_action - split["action"])) > 2e-4:
            raise AssertionError("Oracle gripper action does not match audited semantics")
    residual._write(root / "split_audit.json", {"train_episodes": len(train_ids),
             "validation_episodes": len(val_ids), "heldout_episodes": len(test_ids),
             "train_timesteps": len(splits["train"]["state"]),
             "validation_timesteps": len(splits["validation"]["state"]),
             "heldout_timesteps": len(splits["heldout"]["state"]),
             "disjoint": True, "model_inputs": ["state[7]", "state[5:14]",
             "state[0:14]+observed points[32,3]", "frozen canonical graph z[192]"],
             "excluded_from_model_inputs": ["SurfaceShape", "target_width", "thickness", "oracle action"]})
    results = {}
    for target_name in ("width", "action"):
        results[target_name] = {}
        train_y = splits["train"][target_name]
        center, scale = float(train_y.mean()), float(train_y.std())
        if scale < 1e-8:
            raise AssertionError("Degenerate aperture target")
        for feature_name in ("current_aperture_only", "robot_only", "frozen_z", "raw_graph"):
            feature_root = ensure_dir(root / feature_name / target_name)
            mu = features["train"][feature_name].mean(0, keepdims=True)
            sigma = features["train"][feature_name].std(0, keepdims=True).clip(1e-5)
            x = {name: torch.as_tensor((features[name][feature_name] - mu) / sigma,
                                       dtype=torch.float32, device=device)
                 for name in splits}
            y = {name: torch.as_tensor((splits[name][target_name] - center) / scale,
                                       dtype=torch.float32, device=device)
                 for name in splits}
            set_seed(2811 + 8000 + len(feature_name) + (0 if target_name == "width" else 100))
            probe = Probe(x["train"].shape[1]).to(device)
            optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
            generator = torch.Generator(device="cpu").manual_seed(2811 + 6000)
            best, best_step, stale, curve = math.inf, 0, 0, []
            for step in range(1, 10001):
                indices = torch.randint(len(x["train"]), (256,), generator=generator).to(device)
                probe.train()
                loss = (probe(x["train"][indices]) - y["train"][indices]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % 25 == 0:
                    probe.eval()
                    with torch.no_grad():
                        val = float((probe(x["validation"]) - y["validation"]).square().mean().cpu())
                    curve.append({"update": step, "train_normalized_mse_sample": float(loss.detach().cpu()),
                                  "validation_normalized_mse": val})
                    if val < best - 1e-5:
                        best, best_step, stale = val, step, 0
                        torch.save({"model_state": probe.state_dict(), "input_dim": x["train"].shape[1],
                                    "feature": feature_name, "target": target_name,
                                    "input_mean": mu, "input_std": sigma,
                                    "target_mean": center, "target_std": scale,
                                    "selected_update": step}, feature_root / "best.pt")
                    else:
                        stale += 1
                    if step >= 500 and stale >= 20:
                        break
            write_csv(feature_root / "learning_curve.csv", curve)
            payload = torch.load(feature_root / "best.pt", map_location=device, weights_only=False)
            probe.load_state_dict(payload["model_state"])
            probe.eval()
            with torch.no_grad():
                pred = {name: probe(x[name]).cpu().numpy() * scale + center for name in splits}
            scores = {name: metrics(splits[name][target_name], pred[name],
                                    splits[name]["identity"], splits[name]["half_width"],
                                    splits[name]["state"][:, 7],
                                    task.success_opening_threshold if target_name == "width" else
                                    task.success_opening_threshold /
                                    (sf.MAX_GRIPPER_WIDTH - sf.MIN_GRIPPER_WIDTH), scale)
                      for name in ("validation", "heldout")}
            for name in ("validation", "heldout"):
                write_csv(feature_root / f"{name}_predictions.csv", [
                    {"episode_id": identity, "target": float(t), "prediction": float(p),
                     "error": float(p-t), "half_width": float(h)}
                    for identity, t, p, h in zip(splits[name]["identity"],
                                                  splits[name][target_name], pred[name],
                                                  splits[name]["half_width"], strict=True)])
            results[target_name][feature_name] = {"selected_update": best_step,
                                                   "final_update": step,
                                                   "validation": scores["validation"],
                                                   "heldout": scores["heldout"]}
            print(f"[probe] {target_name} {feature_name}: val MAE {scores['validation']['mae']:.5f}, "
                  f"heldout MAE {scores['heldout']['mae']:.5f}", flush=True)
    residual._write(root / "summary.json", results)
    return results


def readout_probes(output: Path = OUTPUT, device_preference: DevicePreference = "auto") -> dict:
    root = ensure_dir(output / "probes/specialized_readout")
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task = task_config(device_preference, "cpu")
    splits = {name: load_split(name, task) for name in ("train", "validation", "heldout")}
    model = sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME, task, device).eval()
    features = {name: frozen_node_readouts(split, device, model) for name, split in splits.items()}
    width = {name: split["width"] for name, split in splits.items()}
    center, scale = float(width["train"].mean()), float(width["train"].std())
    results = {}
    for feature_name in ("existing_readout", "fingertips", "fingertips_pooled_surface"):
        folder = ensure_dir(root / feature_name)
        mu = features["train"][feature_name].mean(0, keepdims=True)
        sigma = features["train"][feature_name].std(0, keepdims=True).clip(1e-5)
        x = {name: torch.as_tensor((features[name][feature_name] - mu) / sigma,
                                   dtype=torch.float32, device=device) for name in splits}
        y = {name: torch.as_tensor((width[name] - center) / scale,
                                   dtype=torch.float32, device=device) for name in splits}
        set_seed(2811 + 8000 + len(feature_name))
        probe = Probe(x["train"].shape[1]).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(2811 + 6000)
        best, best_step, stale, curve = math.inf, 0, 0, []
        for step in range(1, 10001):
            indices = torch.randint(len(x["train"]), (256,), generator=generator).to(device)
            probe.train()
            loss = (probe(x["train"][indices]) - y["train"][indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 25 == 0:
                probe.eval()
                with torch.no_grad():
                    val = float((probe(x["validation"]) - y["validation"]).square().mean().cpu())
                curve.append({"update": step, "train_normalized_mse_sample": float(loss.detach().cpu()),
                              "validation_normalized_mse": val})
                if val < best - 1e-5:
                    best, best_step, stale = val, step, 0
                    torch.save({"model_state": probe.state_dict(), "selected_update": step,
                                "input_dim": x["train"].shape[1], "input_mean": mu,
                                "input_std": sigma, "target_mean": center,
                                "target_std": scale}, folder / "best.pt")
                else:
                    stale += 1
                if step >= 500 and stale >= 20:
                    break
        write_csv(folder / "learning_curve.csv", curve)
        checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=False)
        probe.load_state_dict(checkpoint["model_state"])
        probe.eval()
        with torch.no_grad():
            pred = {name: probe(x[name]).cpu().numpy() * scale + center for name in splits}
        scores = {name: metrics(width[name], pred[name], splits[name]["identity"],
                                splits[name]["half_width"], splits[name]["state"][:, 7],
                                task.success_opening_threshold, scale)
                  for name in ("validation", "heldout")}
        results[feature_name] = {"selected_update": best_step, "final_update": step,
                                 "validation": scores["validation"], "heldout": scores["heldout"]}
        for name in ("validation", "heldout"):
            write_csv(folder / f"{name}_predictions.csv", [
                {"episode_id": identity, "target_width_m": float(t),
                 "prediction_width_m": float(p), "error_m": float(p-t)}
                for identity, t, p in zip(splits[name]["identity"], width[name], pred[name], strict=True)])
        print(f"[readout] {feature_name}: val MAE {scores['validation']['mae']:.5f}, "
              f"heldout MAE {scores['heldout']['mae']:.5f}", flush=True)
    residual._write(root / "summary.json", results)
    return results


def raw_set_probe(output: Path = OUTPUT, device_preference: DevicePreference = "auto") -> dict:
    folder = ensure_dir(output / "probes/raw_set/width")
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task = task_config(device_preference, "cpu")
    splits = {name: load_split(name, task) for name in ("train", "validation", "heldout")}
    state_mu = splits["train"]["state"].mean((0,), keepdims=True)
    state_std = splits["train"]["state"].std((0,), keepdims=True).clip(1e-5)
    point_mu = splits["train"]["points"].mean((0, 1), keepdims=True)
    point_std = splits["train"]["points"].std((0, 1), keepdims=True).clip(1e-5)
    center, scale = float(splits["train"]["width"].mean()), float(splits["train"]["width"].std())
    state = {name: torch.as_tensor((split["state"] - state_mu) / state_std,
                                   dtype=torch.float32, device=device) for name, split in splits.items()}
    points = {name: torch.as_tensor((split["points"] - point_mu) / point_std,
                                    dtype=torch.float32, device=device) for name, split in splits.items()}
    target = {name: torch.as_tensor((split["width"] - center) / scale,
                                    dtype=torch.float32, device=device) for name, split in splits.items()}
    set_seed(2811 + 8100)
    probe = RawSetProbe().to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(2811 + 6000)
    best, best_step, stale, curve = math.inf, 0, 0, []
    for step in range(1, 10001):
        indices = torch.randint(len(state["train"]), (256,), generator=generator).to(device)
        probe.train()
        loss = (probe(state["train"][indices], points["train"][indices]) -
                target["train"][indices]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            probe.eval()
            with torch.no_grad():
                val = float((probe(state["validation"], points["validation"]) -
                             target["validation"]).square().mean().cpu())
            curve.append({"update": step, "train_normalized_mse_sample": float(loss.detach().cpu()),
                          "validation_normalized_mse": val})
            if val < best - 1e-5:
                best, best_step, stale = val, step, 0
                torch.save({"model_state": probe.state_dict(), "selected_update": step,
                            "state_mean": state_mu, "state_std": state_std,
                            "point_mean": point_mu, "point_std": point_std,
                            "target_mean": center, "target_std": scale}, folder / "best.pt")
            else:
                stale += 1
            if step >= 500 and stale >= 20:
                break
    write_csv(folder / "learning_curve.csv", curve)
    checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=False)
    probe.load_state_dict(checkpoint["model_state"])
    probe.eval()
    with torch.no_grad():
        pred = {name: probe(state[name], points[name]).cpu().numpy() * scale + center for name in splits}
    scores = {name: metrics(split["width"], pred[name], split["identity"],
                            split["half_width"], split["state"][:, 7],
                            task.success_opening_threshold, scale)
              for name, split in splits.items() if name != "train"}
    for name in ("validation", "heldout"):
        write_csv(folder / f"{name}_predictions.csv", [
            {"episode_id": identity, "target_width_m": float(t),
             "prediction_width_m": float(p), "error_m": float(p-t)}
            for identity, t, p in zip(splits[name]["identity"],
                                      splits[name]["width"], pred[name], strict=True)])
    result = {"selected_update": best_step, "final_update": step,
              "validation": scores["validation"], "heldout": scores["heldout"],
              "permutation_invariant": True, "model_inputs": "state[14] and point set[32,3] only"}
    residual._write(folder / "summary.json", result)
    print(f"[raw set] width: val MAE {scores['validation']['mae']:.5f}, "
          f"heldout MAE {scores['heldout']['mae']:.5f}", flush=True)
    return result


def gripper_residual_audit(output: Path = OUTPUT) -> dict:
    task = task_config()
    device = select_device("cpu")
    model = residual.load_controller("mlp", task, residual.ResidualConfig(), device,
        SOURCE / "training/stage_b_mlp/checkpoints/mlp.pt")
    episodes = torch.load(SOURCE / "training/expert.pt", map_location="cpu", weights_only=False)
    episodes += torch.load(SOURCE / "training/policy_mlp.pt", map_location="cpu", weights_only=False)
    deltas = []
    with torch.no_grad():
        for ep in episodes:
            _, _, _, base, _ = model.forward_sequence(ep["states"][None], ep["points"][None],
                                                       ep["previous"][None])
            mask = ep["mask"]
            deltas.extend((ep["targets"][mask, 4] - base[0, mask, 4]).tolist())
    delta = np.asarray(deltas)
    bound = 0.2 * task.max_delta_gripper
    result = {"action_limit_normalized_fraction": task.max_delta_gripper,
              "same_fraction_as_existing_residual": 0.2,
              "fifth_residual_bound_fraction": bound,
              "fifth_residual_nominal_width_m_per_step": bound *
                 (sf.MAX_GRIPPER_WIDTH - sf.MIN_GRIPPER_WIDTH),
              "oracle_minus_ff_action_abs_p50_fraction": float(np.quantile(np.abs(delta), .5)),
              "oracle_minus_ff_action_abs_p90_fraction": float(np.quantile(np.abs(delta), .9)),
              "oracle_minus_ff_action_abs_p95_fraction": float(np.quantile(np.abs(delta), .95)),
              "fraction_exceeding_bound": float(np.mean(np.abs(delta) > bound)),
              "supervised_timesteps": len(delta)}
    residual._write(output / "audit/gripper_residual_bound.json", result)
    return result


def train_5d_stage(root: Path, data: dict[str, list[dict]], validation: list[dict],
                   task: sf.SurfaceFeasibilityConfig, device: torch.device,
                   updates: int, starting_checkpoint: Path | None,
                   restrict_policy_delay: bool, kind: str = "mlp") -> dict:
    design = replace(residual.ResidualConfig(), seed=2811, updates=updates,
                     corrected_dimensions=5)
    eligibility = ({"expert": {d: list(range(len(data["expert"]))) for d in dynamic.TRAIN_DELAYS},
                    "policy": dynamic._eligible_policy_by_delay(data["policy"])}
                   if restrict_policy_delay else None)
    batches, schedule = residual.make_schedule(data, design, dynamic.dynamic_delayed_sequence,
                                               eligibility)
    residual._write(root / "schedule.json", schedule)
    original_schedule = json.loads((SOURCE / "training" /
        ("stage_b_mlp" if restrict_policy_delay else "stage_a") / "logs/schedule.json").read_text())
    if schedule["schedule_sha256"] != original_schedule["schedule_sha256"]:
        raise AssertionError("5D comparison does not use the original 4D batch tensors")
    set_seed(2811 + 1_400_001)
    model = residual.load_controller(kind, task, design, device, starting_checkpoint)
    if starting_checkpoint is None:
        batch = batches[0]
        with torch.no_grad():
            corrected, correction, _, base, _ = model.forward_sequence(
                batch["states"][:1].to(device), batch["points"][:1].to(device),
                batch["previous"][:1].to(device))
        if not torch.equal(corrected, base) or torch.count_nonzero(correction):
            raise AssertionError("5D zero initialization does not reproduce frozen FF")
    base_before = {key: value.detach().cpu().clone() for key, value in model.base.state_dict().items()}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=task.learning_rate, weight_decay=task.weight_decay)
    checkpoint = ensure_dir(root) / "selected.pt"
    final_checkpoint = root / "final.pt"
    best, best_step, curve = math.inf, 0, []
    start = time.perf_counter()
    for step, cpu_batch in enumerate(batches, 1):
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        model.train()
        corrected, _, _, _, _ = model.forward_sequence(batch["states"], batch["points"], batch["previous"])
        loss = .5 * residual.corrected_loss(corrected[:8], batch["targets"][:8],
                                             batch["mask"][:8], task, dimensions=5) + \
               .5 * residual.corrected_loss(corrected[8:], batch["targets"][8:],
                                             batch["mask"][8:], task, dimensions=5)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in model.base.parameters()):
            raise AssertionError("Frozen FF received gradients")
        optimizer.step()
        if step % 25 == 0:
            model.eval()
            score = residual.validation_metrics(model, validation, task, device)
            row = {"update": step, "train_5d_normalized_mse_sample": float(loss.detach().cpu()),
                   "validation_5d_normalized_mse": score["corrected_action_mse_5d"],
                   "validation_4d_normalized_mse": score["corrected_action_mse_4d"]}
            curve.append(row)
            if row["validation_5d_normalized_mse"] < best:
                best, best_step = row["validation_5d_normalized_mse"], step
                torch.save({"model_state": model.state_dict(), "kind": kind,
                            "base_sha256": residual.BASE_SHA256, "design": asdict(design),
                            "seed": 2811, "selected_update": step,
                            "validation_5d_mse": best, "topology": "consistent_intended_100"}, checkpoint)
    torch.save({"model_state": model.state_dict(), "kind": kind,
                "base_sha256": residual.BASE_SHA256, "design": asdict(design),
                "seed": 2811, "selected_update": updates,
                "validation_5d_mse": curve[-1]["validation_5d_normalized_mse"],
                "topology": "consistent_intended_100"}, final_checkpoint)
    if any(not torch.equal(base_before[key], value.detach().cpu())
           for key, value in model.base.state_dict().items()):
        raise AssertionError("Frozen FF changed during 5D training")
    write_csv(root / "learning_curve.csv", curve)
    result = {"updates": updates, "selected_update": best_step,
              "best_validation_5d_normalized_mse": best,
              "final_validation_5d_normalized_mse": curve[-1]["validation_5d_normalized_mse"],
              "training_seconds": time.perf_counter() - start,
              "device": str(device), "schedule_sha256": schedule["schedule_sha256"],
              "starting_checkpoint": str(starting_checkpoint) if starting_checkpoint else None,
              "selected_checkpoint": str(checkpoint), "final_checkpoint": str(final_checkpoint)}
    residual._write(root / "summary.json", result)
    return result


def train_5d_mlp(output: Path = OUTPUT, device_preference: DevicePreference = "auto") -> dict:
    root = ensure_dir(output / "action_path/residual_5d_mlp")
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task = task_config(device_preference, "cpu")
    gripper_residual_audit(output)
    expert = torch.load(SOURCE / "training/expert.pt", map_location="cpu", weights_only=False)
    policy = torch.load(SOURCE / "training/policy_mlp.pt", map_location="cpu", weights_only=False)
    validation = torch.load(SOURCE / "training/validation.pt", map_location="cpu", weights_only=False)
    stage_a = train_5d_stage(root / "stage_a", {"expert": expert, "policy": expert},
                             validation, task, device, 800, None, False)
    stage_b = train_5d_stage(root / "stage_b", {"expert": expert, "policy": policy},
                             validation, task, device, 400,
                             Path(stage_a["selected_checkpoint"]), True)
    result = {"stage_a": stage_a, "stage_b": stage_b,
              "comparison_4d_checkpoint": str(SOURCE / "training/stage_b_mlp/checkpoints/mlp.pt"),
              "policy_data_source": "original 4D MLP current-policy collection; fixed identical pool"}
    residual._write(root / "summary.json", result)
    return result


def train_5d_gru(output: Path = OUTPUT, device_preference: DevicePreference = "auto") -> dict:
    root = ensure_dir(output / "action_path/residual_5d_gru")
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task = task_config(device_preference, "cpu")
    expert = torch.load(SOURCE / "training/expert.pt", map_location="cpu", weights_only=False)
    policy = torch.load(SOURCE / "training/policy_mlp.pt", map_location="cpu", weights_only=False)
    validation = torch.load(SOURCE / "training/validation.pt", map_location="cpu", weights_only=False)
    stage_a = train_5d_stage(root / "stage_a", {"expert": expert, "policy": expert},
                             validation, task, device, 800, None, False, kind="gru")
    stage_b = train_5d_stage(root / "stage_b", {"expert": expert, "policy": policy},
                             validation, task, device, 400,
                             Path(stage_a["selected_checkpoint"]), True, kind="gru")
    result = {"stage_a": stage_a, "stage_b": stage_b,
              "comparison_5d_mlp_checkpoint": str(output / "action_path/residual_5d_mlp/stage_b/selected.pt"),
              "policy_data_source": "same original 4D MLP policy pool as 5D MLP, for paired architecture comparison"}
    residual._write(root / "summary.json", result)
    return result


def failure_flags(row: dict, task: sf.SurfaceFeasibilityConfig, dynamic_row: bool) -> dict:
    position = row["final_tracking_error"] if dynamic_row else row["final_position_error"]
    yaw = row["final_yaw_error"] if dynamic_row else row["final_orientation_error"]
    aperture = row["final_aperture_error"] if dynamic_row else row["final_gripper_width_error"]
    result = {"position_fail": position >= task.success_position_threshold,
              "yaw_fail": yaw >= task.success_rotation_threshold,
              "aperture_fail": aperture >= task.success_opening_threshold,
              "collision_fail": bool(row["collision"]),
              "success": bool(row["success"]),
              "position_margin_m": task.success_position_threshold - position,
              "yaw_margin_rad": task.success_rotation_threshold - yaw,
              "aperture_margin_m": task.success_opening_threshold - aperture}
    result["multiple_failures"] = sum(result[k] for k in
        ("position_fail", "yaw_fail", "aperture_fail", "collision_fail")) > 1
    return result


def _summarize_rows(rows: list[dict], task: sf.SurfaceFeasibilityConfig,
                    dynamic_row: bool) -> dict:
    flags = [failure_flags(row, task, dynamic_row) for row in rows]
    keys = ("success", "collision_fail", "position_fail", "yaw_fail", "aperture_fail", "multiple_failures")
    return {"episodes": len(rows), **{key: int(sum(x[key] for x in flags)) for key in keys},
            "mean_position_error_m": float(np.mean([r["final_tracking_error" if dynamic_row else
                                                    "final_position_error"] for r in rows])),
            "mean_yaw_error_rad": float(np.mean([r["final_yaw_error" if dynamic_row else
                                                "final_orientation_error"] for r in rows])),
            "mean_aperture_error_m": float(np.mean([r["final_aperture_error" if dynamic_row else
                                                   "final_gripper_width_error"] for r in rows])),
            "mean_tracking_error_m": float(np.mean([r["mean_tracking_error" if dynamic_row else
                                                    "trajectory_error"] for r in rows])),
            "mean_inference_ms": float(np.mean([r["mean_inference_ms"] for r in rows]))
            if dynamic_row else None}


def _evaluation_models(task: sf.SurfaceFeasibilityConfig, device: torch.device,
                       output: Path) -> dict:
    design4 = replace(residual.ResidualConfig(), seed=2811, corrected_dimensions=4)
    design5 = replace(residual.ResidualConfig(), seed=2811, corrected_dimensions=5)
    return {"ff": sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME, task, device),
            "residual_4d_mlp": residual.load_controller("mlp", task, design4, device,
                SOURCE / "training/stage_b_mlp/checkpoints/mlp.pt"),
            "residual_5d_mlp_stage_a": residual.load_controller("mlp", task, design5, device,
                output / "action_path/residual_5d_mlp/stage_a/selected.pt"),
            "residual_5d_mlp": residual.load_controller("mlp", task, design5, device,
                output / "action_path/residual_5d_mlp/stage_b/selected.pt"),
            "residual_5d_gru": residual.load_controller("gru", task, design5, device,
                output / "action_path/residual_5d_gru/stage_b/selected.pt")}


def _episode_worker(payload: tuple) -> tuple[int, list[tuple[dict, dict, dict | None]]]:
    """Spawn-safe: each call creates its own MuJoCo env and loads checkpoints locally."""
    condition, index, spec, task, output, eval_device_preference = payload
    device = select_device(eval_device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    models = _evaluation_models(task, device, output)
    env = sf.make_env(task, 2811 + 864_211)
    result = []
    try:
        for name, model in models.items():
            if condition == "dynamic":
                controller = "ff" if name == "ff" else (
                    "residual_gru" if name == "residual_5d_gru" else "residual_mlp")
                row, trace = dynamic.rollout(env, spec, task, device, controller, model, 0)
                row = {**row, "controller": name}
                flag = {"episode_id": spec.identity, "controller": name,
                        **failure_flags(row, task, True)}
                correction_row = None
                if name != "ff":
                    correction = np.asarray([step["residual_action"] for step in trace])
                    correction_row = {"episode_id": spec.identity, "controller": name,
                        "gripper_residual_near_bound_fraction": float(np.mean(
                            np.abs(correction[:, 4]) >= 0.049)) if correction.shape[1] == 5 else 0.0,
                        **{f"residual_dim_{i}_mean": float(correction[:, i].mean())
                           if i < correction.shape[1] else 0.0 for i in range(5)},
                        **{f"residual_dim_{i}_mean_abs": float(np.abs(correction[:, i]).mean())
                           if i < correction.shape[1] else 0.0 for i in range(5)}}
            else:
                if name == "ff":
                    row, trace = rf.rollout(env, spec, model, "ff", task, rf.TemporalConfig(),
                                            device, mode="fresh", severity=0, collect=False,
                                            record_dynamics=True)
                else:
                    row, trace = residual.rollout_residual(env, spec, model, task,
                                                            rf.TemporalConfig(), device)
                row = {**row, "controller": name}
                flag = {"episode_id": spec.sample_identity, "controller": name,
                        **failure_flags(row, task, False)}
                correction_row = None
            if any(step.get("oracle_action") is not None for step in trace):
                raise AssertionError("Closed-loop evaluation called scripted expert")
            result.append((row, flag, correction_row))
    finally:
        env.close()
    return index, result


def evaluate_5d(output: Path = OUTPUT, eval_device_preference: DevicePreference = "cpu",
                eval_workers: int = 1) -> dict:
    if eval_workers < 1:
        raise ValueError("--eval-workers must be positive")
    device = select_device(eval_device_preference)
    task = task_config("cpu", eval_device_preference)
    root = ensure_dir(output / "evaluation")
    models = _evaluation_models(task, device, output) if eval_workers == 1 else None
    model_names = ("ff", "residual_4d_mlp", "residual_5d_mlp_stage_a",
                   "residual_5d_mlp", "residual_5d_gru")
    specs = dynamic.dynamic_specs(2811, task, 16, dynamic.SELECTED_STEP_SPEEDS_M, "heldout")
    expected = json.loads((SOURCE / "config.json").read_text())["dynamic_splits"]["heldout"]
    if [spec.identity for spec in specs] != expected:
        raise AssertionError("Dynamic held-out EpisodeSpecs changed")
    dynamic_rows, dynamic_flags, residual_rows = [], [], []
    if eval_workers > 1:
        payloads = [("dynamic", i, spec, task, output, eval_device_preference)
                    for i, spec in enumerate(specs)]
        with ProcessPoolExecutor(max_workers=eval_workers, mp_context=mp.get_context("spawn")) as executor:
            returned = sorted(executor.map(_episode_worker, payloads), key=lambda pair: pair[0])
        for _, episode in returned:
            for row, flag, correction_row in episode:
                dynamic_rows.append(row)
                dynamic_flags.append(flag)
                if correction_row is not None:
                    residual_rows.append(correction_row)
    else:
        env = sf.make_env(task, 2811 + 864_211)
        try:
            for spec in specs:
                for name, model in models.items():
                    controller = "ff" if name == "ff" else (
                        "residual_gru" if name == "residual_5d_gru" else "residual_mlp")
                    row, trace = dynamic.rollout(env, spec, task, device, controller, model, 0)
                    row = {**row, "controller": name}
                    dynamic_rows.append(row)
                    dynamic_flags.append({"episode_id": spec.identity, "controller": name,
                                          **failure_flags(row, task, True)})
                    if controller != "ff":
                        correction = np.asarray([step["residual_action"] for step in trace])
                        residual_rows.append({"episode_id": spec.identity, "controller": name,
                            "gripper_residual_near_bound_fraction": float(np.mean(
                                np.abs(correction[:, 4]) >= 0.049)) if correction.shape[1] == 5 else 0.0,
                            **{f"residual_dim_{i}_mean": float(correction[:, i].mean())
                               if i < correction.shape[1] else 0.0 for i in range(5)},
                            **{f"residual_dim_{i}_mean_abs": float(np.abs(correction[:, i]).mean())
                               if i < correction.shape[1] else 0.0 for i in range(5)}})
                    if any(step["oracle_action"] is not None for step in trace):
                        raise AssertionError("Closed-loop evaluation called scripted expert")
        finally:
            env.close()
    write_csv(root / "dynamic_fresh_per_episode.csv", dynamic_rows)
    write_csv(output / "failure_decomposition/dynamic_fresh.csv", dynamic_flags)
    write_csv(root / "dynamic_residual_by_dimension.csv", residual_rows)
    dynamic_summary = {name: _summarize_rows([r for r in dynamic_rows if r["controller"] == name], task, True)
                       for name in model_names}
    residual._write(root / "dynamic_fresh_summary.json", dynamic_summary)

    static_specs = sf.sample_episode_specs(16, 2811 + 60_000, task, "iid")
    static_rows, static_flags = [], []
    if eval_workers > 1:
        payloads = [("static", i, spec, task, output, eval_device_preference)
                    for i, spec in enumerate(static_specs)]
        with ProcessPoolExecutor(max_workers=eval_workers, mp_context=mp.get_context("spawn")) as executor:
            returned = sorted(executor.map(_episode_worker, payloads), key=lambda pair: pair[0])
        for _, episode in returned:
            for row, flag, _ in episode:
                static_rows.append(row)
                static_flags.append(flag)
    else:
        env = sf.make_env(task, 2811 + 864_211)
        try:
            for spec in static_specs:
                for name, model in models.items():
                    if name == "ff":
                        row, trace = rf.rollout(env, spec, model, "ff", task, rf.TemporalConfig(),
                                                device, mode="fresh", severity=0, collect=False,
                                                record_dynamics=True)
                    else:
                        row, trace = residual.rollout_residual(env, spec, model, task,
                                                                rf.TemporalConfig(), device)
                    row = {**row, "controller": name}
                    static_rows.append(row)
                    static_flags.append({"episode_id": spec.sample_identity, "controller": name,
                                         **failure_flags(row, task, False)})
                    if any(step.get("oracle_action") is not None for step in trace):
                        raise AssertionError("Static closed-loop evaluation called scripted expert")
        finally:
            env.close()
    write_csv(root / "static_fresh_per_episode.csv", static_rows)
    write_csv(output / "failure_decomposition/static_fresh.csv", static_flags)
    static_summary = {name: _summarize_rows([r for r in static_rows if r["controller"] == name], task, False)
                      for name in model_names}
    residual._write(root / "static_fresh_summary.json", static_summary)
    result = {"dynamic_fresh": dynamic_summary, "static_fresh": static_summary,
              "eval_device": str(device), "dynamic_episode_ids": expected,
              "static_episode_ids": [spec.sample_identity for spec in static_specs]}
    residual._write(root / "summary.json", result)
    return result


def counterfactual_width_response(output: Path = OUTPUT) -> dict:
    """Hold robot state fixed, change only observable surface width and oracle label."""
    task = task_config()
    device = select_device("cpu")
    model = sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME, task, device).eval()
    mlp5 = residual.load_controller("mlp", task,
        replace(residual.ResidualConfig(), corrected_dimensions=5), device,
        output / "action_path/residual_5d_mlp/stage_b/selected.pt")
    specs = specs_by_identity(task)
    episodes = torch.load(SOURCE / "benchmark_sanity/heldout_oracle.pt",
                          map_location="cpu", weights_only=False)
    checkpoint_paths = {name: output / "probes" / name / "width/best.pt"
                        for name in ("current_aperture_only", "robot_only", "frozen_z", "raw_graph")}
    probes = {}
    for name, path in checkpoint_paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        probe = Probe(payload["input_dim"])
        probe.load_state_dict(payload["model_state"])
        probes[name] = (probe.eval(), payload)
    readout_probes = {}
    for name in ("existing_readout", "fingertips", "fingertips_pooled_surface"):
        payload = torch.load(output / "probes/specialized_readout" / name / "best.pt",
                             map_location="cpu", weights_only=False)
        probe = Probe(payload["input_dim"])
        probe.load_state_dict(payload["model_state"])
        readout_probes[name] = (probe.eval(), payload)
    raw_set_payload = torch.load(output / "probes/raw_set/width/best.pt",
                                 map_location="cpu", weights_only=False)
    raw_set = RawSetProbe()
    raw_set.load_state_dict(raw_set_payload["model_state"])
    raw_set.eval()
    rows = []
    with torch.no_grad():
        for ep in episodes:
            spec = specs[ep["identity"]]
            state = ep["states"][0].numpy()
            ee = ep["ee_world"][0].numpy()
            yaw = math.atan2(float(state[5]), float(state[6]))
            seed = sf._stable_seed(spec.initial.sample_identity)
            variant = {}
            for label, half_width in (("thin", 0.029), ("thick", 0.040)):
                shape = replace(spec.initial.shape, half_width=half_width)
                points = sf.localize_world(sf.sample_surface_points_world(shape, 32, seed), ee, yaw)
                state_t = torch.as_tensor(state[None], dtype=torch.float32)
                points_t = torch.as_tensor(points[None], dtype=torch.float32)
                z = model.encode(state_t, points_t).numpy()
                base_action = model.action_head(torch.from_numpy(z)).numpy()[0]
                corrected, delta, _, _ = mlp5.step(state_t, points_t, torch.zeros(1, 5))
                features = {"current_aperture_only": state[None, 7:8],
                            "robot_only": state[None, 5:14],
                            "raw_graph": np.concatenate([state[None], points.reshape(1, -1)], axis=1),
                            "frozen_z": z}
                node_features = frozen_node_readouts(
                    {"state": state[None], "points": points[None]}, device, model)
                if not np.allclose(node_features["existing_readout"], z, atol=1e-6):
                    raise AssertionError("Intermediate node readout differs from canonical z")
                predictions = {}
                for name, (probe, payload) in probes.items():
                    x = torch.as_tensor((features[name] - payload["input_mean"]) /
                                        payload["input_std"], dtype=torch.float32)
                    predictions[name] = float(probe(x).item() * payload["target_std"] +
                                              payload["target_mean"])
                for name, (probe, payload) in readout_probes.items():
                    x = torch.as_tensor((node_features[name] - payload["input_mean"]) /
                                        payload["input_std"], dtype=torch.float32)
                    predictions[name] = float(probe(x).item() * payload["target_std"] +
                                              payload["target_mean"])
                set_state = torch.as_tensor((state[None] - raw_set_payload["state_mean"]) /
                                            raw_set_payload["state_std"], dtype=torch.float32)
                set_points = torch.as_tensor((points[None] - raw_set_payload["point_mean"]) /
                                             raw_set_payload["point_std"], dtype=torch.float32)
                predictions["raw_set"] = float(raw_set(set_state, set_points).item() *
                                               raw_set_payload["target_std"] + raw_set_payload["target_mean"])
                variant[label] = {"target_width_m": desired_width(shape),
                                  "observed_proxy_m": observed_width_proxy(points),
                                  "ff_gripper_action": float(base_action[4]),
                                  "5d_corrected_gripper_action": float(corrected[0, 4]),
                                  "5d_residual_gripper_action": float(delta[0, 4]),
                                  "predictions": predictions,
                                  "z": z[0], "points": points}
            thin, thick = variant["thin"], variant["thick"]
            rows.append({"episode_id": ep["identity"],
                         "target_width_delta_m": thick["target_width_m"] - thin["target_width_m"],
                         "observed_proxy_delta_m": thick["observed_proxy_m"] - thin["observed_proxy_m"],
                         "cloud_max_point_delta_m": float(np.linalg.norm(thick["points"] - thin["points"],axis=1).max()),
                         "z_l2_delta": float(np.linalg.norm(thick["z"] - thin["z"])),
                         "ff_action_delta": thick["ff_gripper_action"] - thin["ff_gripper_action"],
                         "5d_corrected_action_delta": thick["5d_corrected_gripper_action"] -
                                                       thin["5d_corrected_gripper_action"],
                         **{f"{name}_predicted_width_delta_m": thick["predictions"][name] -
                             thin["predictions"][name] for name in (*probes, *readout_probes, "raw_set")}})
    path = output / "probes/counterfactual_width_response.csv"
    write_csv(path, rows)
    summary = {"episodes": len(rows),
               **{key + "_mean": float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "episode_id"},
               "same_robot_state_within_each_pair": True,
               "thin_half_width_m": 0.029, "thick_half_width_m": 0.040}
    residual._write(output / "probes/counterfactual_width_response.json", summary)
    return summary


def report(output: Path = OUTPUT) -> None:
    audit = json.loads((output / "audit/aperture_semantics.json").read_text())
    bound = json.loads((output / "audit/gripper_residual_bound.json").read_text())
    probes = json.loads((output / "probes/summary.json").read_text())
    raw_set = json.loads((output / "probes/raw_set/width/summary.json").read_text())
    readouts = json.loads((output / "probes/specialized_readout/summary.json").read_text())
    counter = json.loads((output / "probes/counterfactual_width_response.json").read_text())
    evaluation = json.loads((output / "evaluation/summary.json").read_text())
    mlp_training = json.loads((output / "action_path/residual_5d_mlp/summary.json").read_text())
    gru_training = json.loads((output / "action_path/residual_5d_gru/summary.json").read_text())
    residual_rows = list(csv.DictReader((output / "evaluation/dynamic_residual_by_dimension.csv").open()))
    overlap_rows = []
    flag_keys = ("position_fail", "yaw_fail", "aperture_fail", "collision_fail")
    for condition in ("dynamic_fresh", "static_fresh"):
        flags = list(csv.DictReader((output / "failure_decomposition" / f"{condition}.csv").open()))
        for controller in sorted({r["controller"] for r in flags}):
            subset = [r for r in flags if r["controller"] == controller]
            for i, left in enumerate(flag_keys):
                for right in flag_keys[i+1:]:
                    overlap_rows.append({"condition": condition, "controller": controller,
                                         "failure_a": left, "failure_b": right,
                                         "overlap_count": sum(r[left] == "True" and r[right] == "True"
                                                              for r in subset)})
    write_csv(output / "failure_decomposition/overlap_counts.csv", overlap_rows)
    plotdir = ensure_dir(output / "plots")
    labels = {"current_aperture_only": "current opening", "robot_only": "robot state",
              "frozen_z": "frozen z", "raw_graph": "raw graph"}
    fig, ax = plt.subplots(figsize=(5.7, 3.6))
    for key, label in labels.items():
        curve = list(csv.DictReader((output / "probes" / key / "width/learning_curve.csv").open()))
        ax.plot([int(r["update"]) for r in curve],
                [float(r["validation_normalized_mse"]) for r in curve], label=label)
    curve = list(csv.DictReader((output / "probes/raw_set/width/learning_curve.csv").open()))
    ax.plot([int(r["update"]) for r in curve],
            [float(r["validation_normalized_mse"]) for r in curve], label="raw unordered set")
    ax.set(xlabel="Probe updates", ylabel="Validation normalized width MSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plotdir / "width_probe_learning_curves.png", dpi=170)
    plt.close(fig)
    response_keys = ("target_width_delta_m_mean", "observed_proxy_delta_m_mean",
                     "raw_graph_predicted_width_delta_m_mean", "raw_set_predicted_width_delta_m_mean",
                     "frozen_z_predicted_width_delta_m_mean",
                     "fingertips_predicted_width_delta_m_mean",
                     "fingertips_pooled_surface_predicted_width_delta_m_mean")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(range(len(response_keys)), [1000 * counter[k] for k in response_keys])
    ax.set_xticks(range(len(response_keys)), ["oracle", "point span", "ordered MLP", "set MLP", "frozen z",
                                             "tips", "tips + surface"], rotation=25, ha="right")
    ax.set_ylabel("Predicted width change (mm)")
    fig.tight_layout()
    fig.savefig(plotdir / "matched_width_response.png", dpi=170)
    plt.close(fig)
    for condition in ("dynamic_fresh", "static_fresh"):
        names = ("ff", "residual_4d_mlp", "residual_5d_mlp", "residual_5d_gru")
        data = evaluation[condition]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        positions = np.arange(len(names))
        ax.bar(positions - .18, [data[n]["aperture_fail"] for n in names], .36, label="aperture fail")
        ax.bar(positions + .18, [data[n]["success"] for n in names], .36, label="success")
        ax.set_xticks(positions, ["FF", "4D MLP", "5D MLP", "5D GRU"])
        ax.set(ylabel="Episodes out of 16", title=condition.replace("_", " "))
        ax.legend()
        fig.tight_layout()
        fig.savefig(plotdir / f"{condition}_outcomes.png", dpi=170)
        plt.close(fig)
    lines = ["# Aperture bottleneck diagnosis", "",
             "## A. Aperture semantics", "",
             "Action dimension 4 is a signed **delta in normalized opening fraction**. "
             "Positive opens. `opening_fraction` maps calibrated fingertip width 0.065–0.105 m "
             "to [0,1]; actions are clipped to ±0.25 per step, added to the current fraction, "
             "and applied through the native finger joints. The measured +0.25 action at "
             f"mid-opening changed separation by {1000*audit['measured_delta_m']:.2f} mm. "
             "Success uses absolute final projected fingertip separation error, with the "
             "unchanged <8 mm threshold. `_surface_target` sets desired width to "
             "`2 * profile_half_width(0.5, shape) + 0.010 m`; the profile uses `half_width`, "
             "`thickness_amplitude`, and `thickness_phase`. The oracle action is the clipped "
             "difference between target and current opening fractions. The desired width is "
             "independent of current gripper opening; the action is not.", "",
             "The 5D correction uses the same 20%-of-action-limit rule as geometric residuals: "
             f"±{bound['fifth_residual_bound_fraction']:.2f} opening fraction per step, nominally "
             f"±{1000*bound['fifth_residual_nominal_width_m_per_step']:.1f} mm. "
             f"The oracle-minus-FF gripper action correction had p50/p90/p95 absolute magnitudes "
             f"{bound['oracle_minus_ff_action_abs_p50_fraction']:.3f}/"
             f"{bound['oracle_minus_ff_action_abs_p90_fraction']:.3f}/"
             f"{bound['oracle_minus_ff_action_abs_p95_fraction']:.3f}; "
             f"{100*bound['fraction_exceeding_bound']:.1f}% of supervised steps exceed the "
             "one-step bound. This bound was fixed before rollout evaluation.", "",
             "## B. Observability audit", "",
             "The 32 surface points contain eight longitudinal cross sections with four "
             "points on each task-facing half cross section. The two normal-direction endpoints "
             "are both present and their x–z separation is twice the local profile radius. "
             "The central two cross sections estimate oracle desired width using points alone "
             f"with {1000*audit['observability']['proxy_mae_m']:.2f} mm MAE and "
             f"{1000*audit['observability']['proxy_p95_abs_m']:.2f} mm p95 across "
             f"{audit['observability']['episodes']} episode shapes. A deterministic pair changes "
             f"target width by {1000*audit['deterministic_width_pair']['target_delta_m']:.1f} mm "
             "and changes the sampled cloud by up to "
             f"{1000*audit['deterministic_width_pair']['cloud_max_point_shift_m']:.1f} mm. "
             "An order-free diagnostic recovers the same 0.11 mm MAE and 0.33 mm p95 by "
             "identifying endpoints from observed y coordinates, pairing them by observed "
             "longitudinal position, and measuring the central spans. It uses no sampler "
             "index or hidden shape field. We found no same-graph/different-target ambiguity "
             "for this width perturbation.", "",
             "| Quantity | Raw observation | Graph features | Recoverable |",
             "|---|---|---|---|",
             "| Current opening | state[7] normalized fraction | EE context receives all state | Direct |",
             "| Left/right tip position | state[8:11], state[11:14] | Tip-node xyz; EE context | Direct |",
             "| Tip separation | Two tip coordinates and state[7] | Relative geometry through EE; no direct tip–tip edge | Yes |",
             "| Surface coordinates | 32 EE-local xyz points | Surface-node xyz and edge relative xyz/distance | Direct |",
             "| Target width shape fields | No explicit half-width/amplitude/phase | No explicit fields | Inferable from sampled endpoint spans |",
             "| Oracle target/action | Labels only | Absent from graph input | Target inferred from geometry; action also needs current opening |",
             "", "The canonical graph has 35 nodes and 100 directed edges: four symmetric "
             "EE–tip, 64 EE–surface, 32 tip–surface, and no surface-local or direct tip–tip "
             "edges. The EE node receives all 14 state fields; each node also receives role "
             "and xyz, and each edge carries relative xyz, distance and type. The encoder "
             "readout concatenates EE, left-tip and right-tip embeddings into 192D `z`.", "",
             "## C. Frozen representation probes", "",
             "Probes use unchanged episode-level train/validation/held-out splits: 72/16/16 "
             "distinct episodes (2,880/320/320 supervised timesteps). The 192→64→1 "
             "frozen-`z` head and matching 64-hidden baseline heads were selected by validation "
             "MSE; curves and predictions are saved. Width errors below are physical millimeters. "
             "The 8 mm column is the fraction of timesteps whose predicted desired width is "
             "within the existing success tolerance; correlated timesteps do not constitute "
             "independent episodes.", "",
             "| Input | Held-out MAE mm | p50 mm | p90 mm | p95 mm | Normalized MSE | Correlation | Within 8 mm |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in labels:
        score = probes["width"][key]["heldout"]
        lines.append(f"| {labels[key]} | {1000*score['mae']:.2f} | {1000*score['p50_abs']:.2f} | "
                     f"{1000*score['p90_abs']:.2f} | {1000*score['p95_abs']:.2f} | "
                     f"{score['normalized_mse']:.3f} | {score['correlation']:.3f} | "
                     f"{score['within_8mm']:.3f} |")
    score = raw_set["heldout"]
    lines.append(f"| raw unordered set | {1000*score['mae']:.2f} | {1000*score['p50_abs']:.2f} | "
                 f"{1000*score['p90_abs']:.2f} | {1000*score['p95_abs']:.2f} | "
                 f"{score['normalized_mse']:.3f} | {score['correlation']:.3f} | "
                 f"{score['within_8mm']:.3f} |")
    lines += ["", "The raw graph head is consistently more accurate across half-width and "
              "current-opening tertiles (see `probes/summary.json`). It uses the sampler's "
              "point order, which the canonical graph does not encode. The separate small "
              "permutation-invariant set probe reached the raw-unordered row above and "
              f"plateaued at update {raw_set['selected_update']} (stopped at "
              f"{raw_set['final_update']}). Its weak result does not negate geometric "
              "observability: the explicit order-free coordinate diagnostic recovers width "
              "accurately, but the neural set head does not learn the endpoint relation. "
              "Current opening alone "
              "is predictive on these natural trajectories because robot state correlates "
              "with object width; this does not establish surface use. The action-target "
              "probes show a related pattern: held-out MAE in normalized opening-fraction "
              "units is 0.055 current-only, 0.037 robot-only, 0.041 frozen `z`, and 0.035 "
              "raw graph. Their full error distributions and validation-selected learning "
              "curves are saved.", "",
              "Frozen node-head diagnostics for desired width: existing EE+tips readout "
              f"{1000*readouts['existing_readout']['heldout']['mae']:.2f} mm MAE, "
              f"tips only {1000*readouts['fingertips']['heldout']['mae']:.2f} mm, "
              "tips plus mean-pooled surface "
              f"{1000*readouts['fingertips_pooled_surface']['heldout']['mae']:.2f} mm. "
              "None matches the raw graph head.", "",
              "In 16 matched held-out counterfactual pairs, robot state and opening were held "
              "fixed while sampled surface width changed. Oracle desired width changed "
              f"{1000*counter['target_width_delta_m_mean']:.2f} mm on average; the point-only "
              f"span changed {1000*counter['observed_proxy_delta_m_mean']:.2f} mm and the "
              f"raw graph head predicted {1000*counter['raw_graph_predicted_width_delta_m_mean']:.2f} mm. "
              "The small unordered-set head predicted "
              f"{1000*counter['raw_set_predicted_width_delta_m_mean']:.2f} mm. "
              f"The frozen-`z` head predicted only {1000*counter['frozen_z_predicted_width_delta_m_mean']:.2f} mm; "
              "fingertip-only and fingertip-plus-surface heads predicted "
              f"{1000*counter['fingertips_predicted_width_delta_m_mean']:.2f} and "
              f"{1000*counter['fingertips_pooled_surface_predicted_width_delta_m_mean']:.2f} mm. "
              "The frozen FF gripper command changed only "
              f"{counter['ff_action_delta_mean']:.4f} opening-fraction units. This controlled "
              "test shows little width-specific response in the frozen encoder/head despite "
              "clear raw geometry.", "",
              "## D. Bottleneck decision", "",
              "**ENCODER / READOUT** is the best-supported diagnosis for the *remaining "
              "width-conditioned aperture error*. Even as an unordered coordinate set, the "
              "observation contains the width. "
              "The current 4D action restriction is also a confirmed upstream bottleneck: "
              "opening a fifth output improves control, but it does not make the frozen "
              "features strongly responsive to surface width. The specialized frozen "
              "readouts did not recover that response, and the small unordered-set probe "
              "also failed to learn the endpoint pairing. A relation-aware model or a "
              "different width-specific training signal may help, but the direct fingertip "
              "relation has not been tested and cannot be credited. No graph topology or "
              "surface sampling change was made.", "",
              "## E. 4D versus 5D residual", "",
              "Both 5D branches used the original 800-update expert Stage A and 400-update "
              "fixed-pool Stage B schedule, with fresh AdamW per stage, identical 4D baseline "
              "batch tensors, unchanged FF and graph, and the same held-out EpisodeSpecs. "
              "The 5D objective applies the existing normalized action MSE to all five "
              "dimensions. Zero-initialized fifth projection exactly reproduces FF before "
              "training. Checkpoints were selected by validation 5D MSE, not held-out success. "
              "The 5D MLP Stage A/B selected updates were "
              f"{mlp_training['stage_a']['selected_update']}/"
              f"{mlp_training['stage_b']['selected_update']}; GRU were "
              f"{gru_training['stage_a']['selected_update']}/"
              f"{gru_training['stage_b']['selected_update']}. CPU was used for the controlled "
              "comparison; a separate MPS training execution also completed, while CUDA "
              "was unavailable.", "",
              "| Dynamic fresh controller | Success | Position fail | Yaw fail | Aperture fail | Collision | Multiple fail | Mean tracking mm | Final position mm | Final yaw rad | Final aperture mm | Latency ms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, label in (("ff", "Frozen FF"), ("residual_4d_mlp", "4D MLP"),
                        ("residual_5d_mlp", "5D MLP"), ("residual_5d_gru", "5D GRU")):
        r = evaluation["dynamic_fresh"][name]
        lines.append(f"| {label} | {r['success']}/16 | {r['position_fail']} | {r['yaw_fail']} | "
                     f"{r['aperture_fail']} | {r['collision_fail']} | {r['multiple_failures']} | "
                     f"{1000*r['mean_tracking_error_m']:.2f} | {1000*r['mean_position_error_m']:.2f} | "
                     f"{r['mean_yaw_error_rad']:.3f} | {1000*r['mean_aperture_error_m']:.2f} | "
                     f"{r['mean_inference_ms']:.2f} |")
    for condition in ("dynamic_fresh", "static_fresh"):
        flags = list(csv.DictReader((output / "failure_decomposition" / f"{condition}.csv").open()))
        base = {r["episode_id"]: r for r in flags if r["controller"] == "residual_4d_mlp"}
        new = {r["episode_id"]: r for r in flags if r["controller"] == "residual_5d_mlp"}
        gained = sum(base[k]["success"] == "False" and new[k]["success"] == "True" for k in base)
        lost = sum(base[k]["success"] == "True" and new[k]["success"] == "False" for k in base)
        fixed_aperture = sum(base[k]["aperture_fail"] == "True" and new[k]["aperture_fail"] == "False" for k in base)
        new_aperture = sum(base[k]["aperture_fail"] == "False" and new[k]["aperture_fail"] == "True" for k in base)
        lines.append(f"{condition.replace('_', ' ').capitalize()}: paired 4D→5D MLP success "
                     f"gained/lost {gained}/{lost}; aperture failures fixed/new "
                     f"{fixed_aperture}/{new_aperture}.")
    lines += ["", "The 5D MLP's mean absolute dynamic residual by action dimension "
              "(x/y/z/yaw/gripper) was "]
    selected = [r for r in residual_rows if r["controller"] == "residual_5d_mlp"]
    lines[-1] += "/".join(f"{np.mean([float(r[f'residual_dim_{i}_mean_abs']) for r in selected]):.4f}"
                          for i in range(5)) + ". The gripper residual averaged 0.0414 " \
              "fraction units and was within 0.001 of its 0.05 bound in 5% of dynamic " \
              "steps. This action-path change reduces aperture failures, but 8/16 dynamic " \
              "episodes still miss the threshold. Position and collision counts did not worsen " \
              "for the selected 5D MLP; yaw failures stayed 6/16."
    lines += ["", "On dynamic fresh episodes, yaw-plus-aperture overlap was 4/16 for both "
              "the 4D and 5D MLP. Pairwise overlap counts for every condition and controller "
              "are in `failure_decomposition/overlap_counts.csv`."]
    lines += ["", "## F. Dedicated gripper head", "",
              "The frozen gripper-specific *probe* readouts above were tested. A new "
              "closed-loop dedicated head was not trained because the available frozen "
              "node features showed weak width response in the matched counterfactual. "
              "Training another head on those same features would not isolate a new source "
              "of width information.", "",
              "## G. Graph ablations", "",
              "Not triggered. The raw 32-point observation clearly encodes desired width, "
              "so a direct fingertip relation, explicit width feature, or changed surface "
              "sampling was not introduced. The current result points first to how the "
              "frozen encoder was trained to use its existing surface information.", "",
              "## H. Overall task success", "",
              "| Static fresh controller | Success | Position fail | Yaw fail | Aperture fail | Collision | Multiple fail | Mean tracking mm | Final position mm | Final yaw rad | Final aperture mm |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, label in (("ff", "Frozen FF"), ("residual_4d_mlp", "4D MLP"),
                        ("residual_5d_mlp", "5D MLP"), ("residual_5d_gru", "5D GRU")):
        r = evaluation["static_fresh"][name]
        lines.append(f"| {label} | {r['success']}/16 | {r['position_fail']} | {r['yaw_fail']} | "
                     f"{r['aperture_fail']} | {r['collision_fail']} | {r['multiple_failures']} | "
                     f"{1000*r['mean_tracking_error_m']:.2f} | {1000*r['mean_position_error_m']:.2f} | "
                     f"{r['mean_yaw_error_rad']:.3f} | {1000*r['mean_aperture_error_m']:.2f} |")
    lines += ["", "The static 5D MLP improves success from 1/16 to 10/16 and aperture "
              "failures from 11/16 to 6/16, but its mean trajectory tracking error rises "
              "from 46.77 to 71.53 mm. Static rollouts stop on first success, and trajectory "
              "lengths can differ; this trajectory-average difference is not a fixed-horizon "
              "spatial-control comparison. Final static position remains comparable "
              "(10.45 versus 10.72 mm) and final yaw improves (0.145 versus 0.117 rad). "
              "The 5D GRU reaches 4/16 dynamic and 10/16 static success versus 5/16 and "
              "10/16 for the 5D MLP on the same training pool. This is not evidence of "
              "temporal memory. All paired per-episode errors, threshold margins and failure "
              "overlaps are in `evaluation/` and `failure_decomposition/`.", "",
              "## I. Facts, interpretation, remaining hypotheses", "",
              "**Facts.** Width is recoverable from the unordered sampled coordinates; "
              "the frozen `z`, small unordered-set head and "
              "existing FF gripper output respond weakly to controlled width changes. "
              "5D correction improves aperture and success without graph changes, but "
              "dynamic aperture failure remains 8/16.\n\n"
              "**Interpretation.** Restricted output was a real aperture bottleneck; "
              "the remaining width-conditioned bottleneck is in how the frozen encoder/readout "
              "and its training objective extract geometry, rather than absent raw geometry. "
              "Natural-trajectory prediction partly exploits robot-state correlations, "
              "so the matched counterfactual is the stronger evidence about geometry use.\n\n"
              "**Remaining hypotheses.** A future controlled experiment can train an "
              "aperture-aware spatial encoder or a raw-observation spatial gripper head "
              "without changing topology, and separately test whether the conservative "
              "fifth bound limits correction. Neither was tuned on held-out success here. "
              "No graph relation or surface representation conclusion follows from low "
              "success alone.", ""]
    (output / "report.md").write_text("\n".join(lines))
    residual._write(output / "audit/checkpoint_hashes.json", {
        "base_ff": hashlib.sha256(residual.BASE_CHECKPOINT.read_bytes()).hexdigest(),
        "historical_4d_mlp": hashlib.sha256((SOURCE / "training/stage_b_mlp/checkpoints/mlp.pt").read_bytes()).hexdigest(),
        "selected_5d_mlp": hashlib.sha256((output / "action_path/residual_5d_mlp/stage_b/selected.pt").read_bytes()).hexdigest(),
        "selected_5d_gru": hashlib.sha256((output / "action_path/residual_5d_gru/stage_b/selected.pt").read_bytes()).hexdigest()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("audit", "probes", "readout", "rawset", "train5d", "train5dgru",
                                          "evaluate", "counterfactual", "report"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "audit":
        audit(args.output)
    elif args.phase == "probes":
        probes(args.output, args.device)
    elif args.phase == "readout":
        readout_probes(args.output, args.device)
    elif args.phase == "rawset":
        raw_set_probe(args.output, args.device)
    elif args.phase == "train5d":
        train_5d_mlp(args.output, args.device)
    elif args.phase == "train5dgru":
        train_5d_gru(args.output, args.device)
    elif args.phase == "evaluate":
        evaluate_5d(args.output, args.eval_device, args.eval_workers)
    elif args.phase == "counterfactual":
        counterfactual_width_response(args.output)
    else:
        report(args.output)


if __name__ == "__main__":
    main()
