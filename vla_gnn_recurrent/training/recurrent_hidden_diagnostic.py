"""Read-only hidden-state reset diagnostic for the canonical seed-2811 GRU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_representation_final_comparison as final
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device


GATE2 = Path("artifacts/recurrent_feasibility_canonical100_seed2811")
OUTPUT = Path("artifacts/recurrent_hidden_diagnostic")
POLICIES = ("normal", "reset_1", "reset_4", "reset_8")
CHECKPOINT = GATE2 / "training/seed2811/gru_dagger_round1/gru.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(data: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("states", "surface_points", "actions", "episode_ids", "steps"):
        value = data[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(value.tobytes())
    if "loss_mask" in data:
        digest.update(b"loss_mask")
        digest.update(data["loss_mask"].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _model_digest(model: rf.GraphGRUPolicy) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _action_errors(predicted: np.ndarray, target: np.ndarray,
                   config: sf.SurfaceFeasibilityConfig) -> dict[str, float]:
    difference = predicted - target
    yaw_difference = float(np.arctan2(np.sin(difference[3]), np.cos(difference[3])))
    scales = np.array([config.max_delta_ee] * 3 + [config.max_delta_rotation, config.max_delta_gripper])
    return {"translation_l2": float(np.linalg.norm(difference[:3])),
            "yaw_absolute": abs(yaw_difference),
            "aperture_absolute": abs(float(difference[4])),
            "normalized_mse": float(np.mean((difference / scales) ** 2))}


@torch.no_grad()
def replay_fixed_sequences(model: rf.GraphGRUPolicy, data: dict[str, Any],
                           config: sf.SurfaceFeasibilityConfig, device: torch.device,
                           policies: tuple[str, ...] = ("normal", "reset_1")) -> dict[str, Any]:
    """Replay exactly one immutable graph stream under each hidden policy."""
    input_hash = _tensor_digest(data)
    records: dict[str, list[dict[str, Any]]] = {policy: [] for policy in policies}
    for episode in rf._episodes(data):
        episode_id = int(episode["episode_ids"][0])
        length = len(episode["actions"])
        for policy in policies:
            hidden = None
            for index in range(length):
                timestep = int(episode["steps"][index])
                reset = rf.hidden_reset_due(policy, index)
                if reset:
                    hidden = None
                state = episode["states"][index].numpy()
                points = episode["surface_points"][index].numpy()
                state_t, points_t, topology = rf._graph_tensors(state, points, config, device)
                previous = hidden if hidden is not None else state_t.new_zeros((1, 1, model.hidden_size))
                output, hidden = model.step(state_t, points_t, hidden, topology)
                raw = output[0].detach().cpu().numpy()
                processed = rf.clipped_action(raw, config)
                target = episode["actions"][index].numpy()
                records[policy].append({
                    "episode_id": episode_id, "timestep": timestep,
                    "progress": index / max(length - 1, 1), "sequence_length": length,
                    "input_sha256": input_hash, "hidden_policy": policy, "hidden_reset": reset,
                    "eligible_oracle_label": bool(episode["loss_mask"][index]),
                    "raw_predicted_action": raw.tolist(), "processed_action": processed.tolist(),
                    "oracle_action": target.tolist(),
                    "hidden_norm": float(hidden.norm().detach().cpu()),
                    "hidden_delta_norm": float((hidden - previous).norm().detach().cpu()),
                    **_action_errors(processed, target, config),
                })
    if _tensor_digest(data) != input_hash:
        raise AssertionError("Fixed-sequence replay modified the input observations.")
    for policy in policies[1:]:
        if [(r["episode_id"], r["timestep"], r["input_sha256"]) for r in records[policy]] != [
                (r["episode_id"], r["timestep"], r["input_sha256"]) for r in records[policies[0]]]:
            raise AssertionError("Hidden policies received different fixed graph sequences.")
    metrics = ("translation_l2", "yaw_absolute", "aperture_absolute", "normalized_mse")
    summary: dict[str, Any] = {}
    for policy, all_rows in records.items():
        rows = [row for row in all_rows if row["eligible_oracle_label"]]
        summary[policy] = {
            "samples": len(rows), "all_context_states": len(all_rows),
            "masked_severe_states": len(all_rows) - len(rows),
            "episodes": len({r["episode_id"] for r in all_rows}),
            "overall": {key: float(np.mean([r[key] for r in rows])) for key in metrics},
            "early_progress": {key: float(np.mean([r[key] for r in rows if r["progress"] < 0.5])) for key in metrics},
            "late_progress": {key: float(np.mean([r[key] for r in rows if r["progress"] >= 0.5])) for key in metrics},
            "by_timestep": [
                {"timestep": timestep, "count": sum(r["timestep"] == timestep for r in rows),
                 **{key: float(np.mean([r[key] for r in rows if r["timestep"] == timestep])) for key in metrics}}
                for timestep in sorted({r["timestep"] for r in rows})
            ],
        }
    return {"input_sha256": input_hash, "records": records, "summary": summary}


def _plot_diagnostics(traces: dict[str, list[dict[str, Any]]], representative_id: int,
                      fixed: dict[str, Any], fixed_policy: dict[str, Any], output: Path) -> None:
    ensure_dir(output)
    colors = {"normal": "tab:blue", "reset_1": "tab:orange", "reset_4": "tab:green", "reset_8": "tab:red"}
    representative = {policy: next(item for item in rows if item["episode_id"] == representative_id)["steps"]
                      for policy, rows in traces.items()}
    for filename, ylabel, extractor in (
        ("representative_yaw_error.png", "Yaw error (rad)", lambda row: row["orientation_error"]),
        ("representative_raw_delta_yaw.png", "Raw predicted Δyaw (rad)", lambda row: row["raw_predicted_action"][3]),
    ):
        fig, ax = plt.subplots(figsize=(8, 4))
        for policy, rows in representative.items():
            ax.plot([r["t"] for r in rows], [extractor(r) for r in rows], label=policy, color=colors[policy])
        ax.set(xlabel="Control step", ylabel=ylabel, title=f"Held-out episode {representative_id}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    normal = representative["normal"]
    ax.plot([r["t"] for r in normal], [r["hidden_norm"] for r in normal], label="||h_t||")
    ax.plot([r["t"] for r in normal], [r["hidden_delta_norm"] for r in normal], label="||h_t − h_(t−1)||")
    ax.set(xlabel="Control step", ylabel="Norm", title=f"Normal carry, episode {representative_id}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "representative_hidden_norms.png", dpi=150)
    plt.close(fig)
    for name, source in (("expert", fixed), ("base_gru_policy", fixed_policy)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, (metric, label) in zip(axes, (("translation_l2", "Translation L2 (m)"),
                                             ("yaw_absolute", "Yaw MAE (rad)"),
                                             ("aperture_absolute", "Aperture MAE")), strict=True):
            for policy in ("normal", "reset_1"):
                rows = [r for r in source["records"][policy] if r["eligible_oracle_label"]]
                bins = np.linspace(0, 1, 6)
                means = [np.mean([r[metric] for r in rows if (left <= r["progress"] < right or
                         (right == 1 and r["progress"] == 1))]) for left, right in zip(bins[:-1], bins[1:], strict=True)]
                ax.plot((bins[:-1] + bins[1:]) / 2, means, marker="o", label=policy)
            ax.set(xlabel="Normalized trajectory progress", ylabel=label)
            ax.grid(alpha=0.25)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(output / f"fixed_{name}_action_error.png", dpi=150)
        plt.close(fig)


def run(output: Path = OUTPUT, checkpoint: Path = CHECKPOINT,
        gate2: Path = GATE2, device_preference: DevicePreference = "cpu") -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Diagnostic artifact already exists: {output}")
    root = ensure_dir(output)
    checkpoint_hash = _sha256(checkpoint)
    device = select_device(device_preference)
    config = rf._task_config(root, (2811,), device_preference, device_preference, 16)
    if config.point_count != 32 or not config.symmetric_robot_edges:
        raise AssertionError("The hidden-state diagnostic requires the canonical N=32 100-edge graph.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("seed") != 2811 or payload.get("topology") != "consistent_intended_100":
        raise ValueError("Expected the canonical Gate 2 seed-2811 GRU checkpoint.")
    model = rf._load_gru(checkpoint, config, rf.TemporalConfig(), device)
    model_hash = _model_digest(model)
    specs = sf.sample_episode_specs(16, 2811 + 60_000, config, "iid")
    gate2_rows = json.loads((gate2 / "fresh/seed2811/paired_results.json").read_text())
    signatures = [final.spec_signature(spec) for spec in specs]
    if signatures != [row["spec_signature"] for row in gate2_rows["gru"]]:
        raise AssertionError("Held-out EpisodeSpecs differ from Gate 2.")
    sf._write_json(root / "config.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                                          "seed": 2811, "episode_specs": [asdict(spec) for spec in specs],
                                          "topology": "consistent_intended_100", "point_count": 32,
                                          "local_surface_edges": False, "hidden_policies": POLICIES,
                                          "eval_device": str(device), "retraining": False})
    rows: dict[str, list[dict[str, Any]]] = {policy: [] for policy in POLICIES}
    traces: dict[str, list[dict[str, Any]]] = {policy: [] for policy in POLICIES}
    env = sf.make_env(config, 2811 + 864_211)
    try:
        for spec in specs:
            for policy in POLICIES:
                row, trace = rf.rollout(env, spec, model, "gru", config, rf.TemporalConfig(), device,
                                        hidden_policy=policy)
                rows[policy].append(row)
                traces[policy].append({"episode_id": spec.episode_id, "spec_signature": final.spec_signature(spec),
                                       "steps": trace})
    finally:
        env.close()
    for policy in POLICIES:
        if [r["spec_signature"] for r in rows[policy]] != signatures:
            raise AssertionError("Hidden-policy runs did not receive identical EpisodeSpecs.")
        path = ensure_dir(root / "closed_loop" / policy)
        sf._write_json(path / "episodes.json", rows[policy])
        sf._write_json(root / "traces" / f"{policy}.json", traces[policy])
    for original, repeated in zip(gate2_rows["gru"], rows["normal"], strict=True):
        for key in ("success", "collision", "first_collision_timestep", "steps_executed"):
            if original[key] != repeated[key]:
                raise AssertionError(f"Normal carry did not reproduce Gate 2: {key}")
        for key in ("final_position_error", "final_orientation_error", "final_gripper_width_error"):
            if not np.isclose(original[key], repeated[key], atol=1e-7):
                raise AssertionError(f"Normal carry did not reproduce Gate 2: {key}")
    with (root / "paired_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("episode_id", "policy", "success", "collision",
            "final_position_error", "final_orientation_error", "final_gripper_width_error",
            "trajectory_error", "steps_executed"))
        writer.writeheader()
        for policy in POLICIES:
            for row in rows[policy]:
                writer.writerow({key: policy if key == "policy" else row.get(key) for key in writer.fieldnames})
    paired = {policy: final.paired_comparison(rows["normal"], rows[policy], config,
                                               f"{policy} - normal", 2811 + int(policy.split("_")[-1]))
              for policy in POLICIES[1:]}
    _, validation, _, _ = final._current_datasets(2811, config)
    fixed = replay_fixed_sequences(model, validation, config, device)
    sf._write_json(root / "fixed_sequence/expert/records.json", fixed["records"])
    sf._write_json(root / "fixed_sequence/expert/summary.json", fixed["summary"])
    policy_path = gate2 / "onpolicy_collection/seed2811/gru/policy_relabelled_trajectories.pt"
    policy_data = torch.load(policy_path, map_location="cpu", weights_only=False)
    train_data, _, train_specs, _ = final._current_datasets(2811, config)
    del train_data
    if set(policy_data["episode_ids"].tolist()) != {spec.episode_id for spec in train_specs}:
        raise AssertionError("Fixed base-GRU policy trajectories differ from the training split.")
    fixed_policy = replay_fixed_sequences(model, policy_data, config, device)
    sf._write_json(root / "fixed_sequence/base_gru_policy/records.json", fixed_policy["records"])
    sf._write_json(root / "fixed_sequence/base_gru_policy/summary.json", fixed_policy["summary"])
    representative_id = traces["normal"][0]["episode_id"]
    _plot_diagnostics(traces, representative_id, fixed, fixed_policy, root / "plots")
    if _sha256(checkpoint) != checkpoint_hash or _model_digest(model) != model_hash:
        raise AssertionError("Diagnostic altered checkpoint or model weights.")
    summary = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
               "model_state_sha256": model_hash, "seed": 2811, "episodes": len(specs),
               "topology": "consistent_intended_100", "point_count": 32,
               "local_surface_edges": False, "retraining": False,
               "gate2_normal_reproduced": True,
               "closed_loop": {policy: rf._metric_means(rows[policy]) for policy in POLICIES},
               "paired_vs_normal": paired, "fixed_expert": fixed["summary"],
               "fixed_input_sha256": fixed["input_sha256"],
               "fixed_base_gru_policy": fixed_policy["summary"],
               "fixed_base_gru_policy_input_sha256": fixed_policy["input_sha256"],
               "fixed_base_gru_policy_provenance": "training-split base-GRU DAgger trajectories; in-sample for final GRU",
               "representative_episode_id": representative_id}
    sf._write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--gate2", type=Path, default=GATE2)
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    args = parser.parse_args()
    if args.eval_workers != 1:
        parser.error("Diagnostic rollouts are serial; each MuJoCo environment stays process-local.")
    run(args.output, args.checkpoint, args.gate2, args.eval_device)


if __name__ == "__main__":
    main()
