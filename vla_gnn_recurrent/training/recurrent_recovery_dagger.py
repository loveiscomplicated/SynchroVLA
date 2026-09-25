"""One controlled recurrent recovery continuation from the canonical Gate 2 GRU."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import recurrent_hidden_diagnostic as hd
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_representation_final_comparison as final
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


GATE2 = Path("artifacts/recurrent_feasibility_canonical100_seed2811")
OUTPUT = Path("artifacts/recurrent_recovery_dagger")
SEED = 2811
BRANCHES = ("branch_a_expert_only", "branch_b_new_dagger", "branch_c_old_dagger")
START = GATE2 / "training/seed2811/gru_dagger_round1/gru.pt"
OLD_POLICY = GATE2 / "onpolicy_collection/seed2811/gru/policy_relabelled_trajectories.pt"
FF_REFERENCE = GATE2 / "training/seed2811/ff_dagger_round1/surface_graph_no_local.pt"


def branch_policy_data(branch: str, new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any] | None:
    if branch == BRANCHES[0]:
        return None
    if branch == BRANCHES[1]:
        return new
    if branch == BRANCHES[2]:
        return old
    raise ValueError(branch)


def validate_policy_data(data: dict[str, Any], specs: list[sf.SurfaceEpisodeSpec],
                         config: sf.SurfaceFeasibilityConfig) -> dict[str, Any]:
    required = ("states", "surface_points", "actions", "episode_ids", "steps", "loss_mask")
    count = len(data["actions"])
    if any(key not in data or len(data[key]) != count for key in required):
        raise ValueError("Policy trajectory arrays are missing or misaligned.")
    if data["surface_points"].shape[1:] != (32, 3) or config.point_count != 32 or not config.symmetric_robot_edges:
        raise ValueError("Recovery trajectories must use the canonical N=32 graph input.")
    episodes = rf._episodes(data)
    ids = {int(spec.episode_id) for spec in specs}
    if {int(x) for x in data["episode_ids"].tolist()} != ids or len(episodes) != len(specs):
        raise ValueError("Policy trajectories must match every training EpisodeSpec exactly once.")
    for episode in episodes:
        if episode["steps"].tolist() != list(range(len(episode["steps"]))):
            raise ValueError("A policy trajectory starts late or bridges a timestep gap.")
    eligible = int(data["loss_mask"].sum())
    if eligible == 0:
        raise ValueError("No eligible policy labels remain after severe-penetration masking.")
    return {"visited": count, "eligible": eligible, "masked": count - eligible,
            "trajectories": len(episodes), "sequence_lengths": [len(ep["actions"]) for ep in episodes],
            "input_sha256": hd._tensor_digest(data)}


def _visit_distribution(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = Counter(int(row["episode_id"]) for row in metadata)
    seen_contact: set[int] = set()
    after_contact = 0
    for row in metadata:
        episode = int(row["episode_id"])
        if episode in seen_contact and not row["severe_penetration_excluded"]:
            after_contact += 1
        if row["collision"]:
            seen_contact.add(episode)
    total = len(metadata)
    counts = {
        "late_t_ge_10": sum(int(row["t"]) >= 10 for row in metadata),
        "collision_states": sum(bool(row["collision"]) for row in metadata),
        "near_contact_distance_le_0_01m": sum(float(row["clearance_m"]) <= 0.01 for row in metadata),
        "eligible_after_first_collision": after_contact,
        "severe_masked": sum(bool(row["severe_penetration_excluded"]) for row in metadata),
    }
    return {"visited": total, "trajectories": len(lengths),
            "length_histogram": dict(sorted(Counter(lengths.values()).items())),
            "length_mean": float(np.mean(list(lengths.values()))),
            "counts": counts, "fractions_of_visited": {key: value / total for key, value in counts.items()}}


def _verify_eval_specs(specs: list[sf.SurfaceEpisodeSpec], gate2: Path) -> None:
    prior = json.loads((gate2 / "fresh/seed2811/paired_results.json").read_text())
    signatures = [final.spec_signature(spec) for spec in specs]
    if signatures != [row["spec_signature"] for row in prior["gru"]] or signatures != [
            row["spec_signature"] for row in prior["ff"]]:
        raise AssertionError("Fresh evaluation EpisodeSpecs differ from Gate 2.")


def _selected_cases(rows: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    original, new = rows["original_gru"], rows["branch_b_new_dagger"]
    median_index = sorted(range(len(original)), key=lambda i: original[i]["final_position_error"])[len(original) // 2]
    disagreement = [i for i in range(len(original)) if original[i]["collision"] != new[i]["collision"]]
    if disagreement:
        clearest = max(disagreement, key=lambda i: abs(original[i]["final_position_error"] - new[i]["final_position_error"]))
    else:
        clearest = max(range(len(original)), key=lambda i: abs(original[i]["final_position_error"] - new[i]["final_position_error"]))
    return {"previous_yaw_drift": 0, "median_original_position_error": original[median_index]["episode_id"],
            "largest_collision_or_position_difference": original[clearest]["episode_id"]}


def _plot_cases(traces: dict[str, list[dict[str, Any]]], cases: dict[str, int], output: Path) -> None:
    ensure_dir(output)
    colors = {"original_gru": "tab:blue", "branch_a_expert_only": "tab:orange",
              "branch_b_new_dagger": "tab:green", "branch_c_old_dagger": "tab:red",
              "ff_reference": "tab:purple"}
    for label, episode_id in cases.items():
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        for name, episodes in traces.items():
            steps = next(row["steps"] for row in episodes if row["episode_id"] == episode_id)
            t = [row["t"] for row in steps]
            axes[0].plot(t, [row["position_error"] for row in steps], label=name, color=colors[name])
            axes[1].plot(t, [row["orientation_error"] for row in steps], label=name, color=colors[name])
            axes[2].plot(t, [row["raw_predicted_action"][3] for row in steps], label=name, color=colors[name])
            for row in steps:
                if row["collision"]:
                    axes[1].scatter(row["t"], row["orientation_error"], color=colors[name], marker="x")
        axes[0].set_ylabel("Position error (m)")
        axes[1].set_ylabel("Yaw error (rad)")
        axes[2].set(xlabel="Control step", ylabel="Raw Δyaw (rad)")
        for ax in axes:
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=8, ncol=2)
        fig.suptitle(f"{label}: held-out episode {episode_id}")
        fig.tight_layout()
        fig.savefig(output / f"{label}_episode{episode_id}.png", dpi=150)
        plt.close(fig)


def run(output: Path = OUTPUT, gate2: Path = GATE2,
        device_preference: DevicePreference = "auto", eval_device_preference: DevicePreference = "cpu",
        updates: int = 800) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Recovery artifact already exists: {output}")
    root = ensure_dir(output)
    start = gate2 / START.relative_to(GATE2)
    old_path = gate2 / OLD_POLICY.relative_to(GATE2)
    ff_path = gate2 / FF_REFERENCE.relative_to(GATE2)
    start_hash = hd._sha256(start)
    config = rf._task_config(root, (SEED,), device_preference, eval_device_preference, 16)
    if not config.symmetric_robot_edges or config.point_count != 32:
        raise AssertionError("Recovery experiment requires the canonical 100-edge N=32 graph.")
    temporal = replace(rf.TemporalConfig(), dagger_updates=updates)
    train_device = select_device(device_preference)
    eval_device = select_device(eval_device_preference)
    if train_device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    expert, validation, train_specs, _ = final._current_datasets(SEED, config)
    prior_specs = json.loads((gate2 / "training/seed2811/shared_train_episode_specs.json").read_text())
    if [final.spec_signature(spec) for spec in train_specs] != [final.spec_signature(sf.SurfaceEpisodeSpec(
            episode_id=row["episode_id"], shape=sf.SurfaceShape(**row["shape"]),
            arm_qpos=tuple(row["arm_qpos"]), gripper_command=row["gripper_command"],
            condition=row["condition"], sample_identity=row["sample_identity"])) for row in prior_specs]:
        raise AssertionError("Training EpisodeSpecs differ from Gate 2.")
    original = rf._load_gru(start, config, temporal, eval_device)
    start_model_hash = hd._model_digest(original)
    new_data, new_collection = rf.collect_recurrent_policy_data(
        original, train_specs, config, temporal, SEED, eval_device, root / "new_policy_collection")
    old_data = torch.load(old_path, map_location="cpu", weights_only=False)
    new_audit = validate_policy_data(new_data, train_specs, config)
    old_audit = validate_policy_data(old_data, train_specs, config)
    if new_audit["input_sha256"] == old_audit["input_sha256"]:
        raise AssertionError("New final-policy states unexpectedly match old base-policy states exactly.")
    new_metadata = json.loads((root / "new_policy_collection/visited_metadata.json").read_text())
    old_metadata = json.loads((gate2 / "onpolicy_collection/seed2811/gru/visited_metadata.json").read_text())
    distribution = {"new_final_policy": _visit_distribution(new_metadata),
                    "old_base_policy": _visit_distribution(old_metadata)}
    sf._write_json(root / "new_policy_collection/distribution_comparison.json", distribution)
    sf._write_json(root / "config.json", {
        "starting_checkpoint": str(start), "starting_checkpoint_sha256": start_hash,
        "starting_model_sha256": start_model_hash, "model_seed": SEED,
        "continuation_seed": SEED + 1_100_001, "sampling_seed": SEED + 1_100_001,
        "optimizer": "fresh AdamW for each branch", "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay, "updates_per_branch": updates,
        "validation_interval_updates": 25, "validation_selection": "lowest expert validation normalized per-state MSE",
        "sequence_batch_size": temporal.sequence_batch_size, "sequence_length": temporal.sequence_length,
        "expert_policy_loss_weight": [0.5, 0.5], "topology": "consistent_intended_100",
        "point_count": 32, "local_surface_edges": False,
        "training_episode_specs": [asdict(spec) for spec in train_specs],
        "old_policy_dataset": str(old_path), "old_policy_input_sha256": old_audit["input_sha256"],
        "new_policy_dataset": str(root / "new_policy_collection/policy_relabelled_trajectories.pt"),
        "new_policy_input_sha256": new_audit["input_sha256"],
        "training_device": str(train_device), "evaluation_device": str(eval_device),
        "sensor_to_command_latency": "NOT MEASURED"})
    branch_results: dict[str, dict[str, Any]] = {}
    checkpoint_paths: dict[str, Path] = {}
    for branch in BRANCHES:
        set_seed(SEED + 1_100_001)
        model = rf._load_gru(start, config, temporal, train_device)
        if hd._model_digest(model) != start_model_hash:
            raise AssertionError(f"{branch} did not start from the same Gate 2 GRU weights.")
        policy = branch_policy_data(branch, new_data, old_data)
        out = root / branch
        result = rf.train_gru(model, expert, validation, policy, config, temporal, SEED,
                              train_device, updates, out,
                              sampling_seed=SEED + 1_100_001, validation_interval=25,
                              stage_label=branch)
        result["starting_checkpoint_sha256"] = start_hash
        result["starting_model_sha256"] = start_model_hash
        result["policy_source"] = None if policy is None else ("new_final_policy" if branch == BRANCHES[1]
                                                   else "old_base_policy")
        result["policy_input_sha256"] = None if policy is None else hd._tensor_digest(policy)
        result["optimizer_state_policy"] = "fresh AdamW; no restored state"
        sf._write_json(out / "training.json", result)
        branch_results[branch] = result
        checkpoint_paths[branch] = Path(result["checkpoint"])
        print(f"[recovery] {branch}: {updates} updates, selected {result['selected_update']}, "
              f"valid expert={result['expert_valid_supervised_timesteps']}, "
              f"policy={result['policy_valid_supervised_timesteps']}", flush=True)
    if {row["updates"] for row in branch_results.values()} != {updates}:
        raise AssertionError("Continuation branches used different update counts.")
    models: dict[str, Any] = {"original_gru": original,
                              "ff_reference": sf.load_surface_model(ff_path, rf.GRAPH_NAME, config, eval_device)}
    models.update({branch: rf._load_gru(path, config, temporal, eval_device)
                   for branch, path in checkpoint_paths.items()})
    specs = sf.sample_episode_specs(16, SEED + 60_000, config, "iid")
    _verify_eval_specs(specs, gate2)
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    traces: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    env = sf.make_env(config, SEED + 864_211)
    try:
        for spec in specs:
            for name, model in models.items():
                controller = "ff" if name == "ff_reference" else "gru"
                row, trace = rf.rollout(env, spec, model, controller, config, temporal, eval_device)
                rows[name].append(row)
                traces[name].append({"episode_id": spec.episode_id, "spec_signature": final.spec_signature(spec),
                                     "steps": trace})
    finally:
        env.close()
    for name in models:
        if [row["spec_signature"] for row in rows[name]] != [final.spec_signature(spec) for spec in specs]:
            raise AssertionError(f"{name} received different held-out IID EpisodeSpecs.")
        sf._write_json(root / "fresh_evaluation" / name / "episodes.json", rows[name])
        sf._write_json(root / "traces" / f"{name}.json", traces[name])
    prior = json.loads((gate2 / "fresh/seed2811/paired_results.json").read_text())
    for name, prior_name in (("original_gru", "gru"), ("ff_reference", "ff")):
        for old, new in zip(prior[prior_name], rows[name], strict=True):
            if old["success"] != new["success"] or old["collision"] != new["collision"] or not np.isclose(
                    old["final_position_error"], new["final_position_error"], atol=1e-7):
                raise AssertionError(f"{name} did not reproduce Gate 2 fresh evaluation.")
    comparisons = {"new_vs_expert": (BRANCHES[0], BRANCHES[1]),
                   "new_vs_old": (BRANCHES[2], BRANCHES[1]),
                   "new_vs_original": ("original_gru", BRANCHES[1]),
                   "new_vs_ff": ("ff_reference", BRANCHES[1])}
    paired = {}
    for label, (reference, candidate) in comparisons.items():
        paired[label] = final.paired_comparison(rows[reference], rows[candidate], config,
                                                f"{candidate} - {reference}", SEED + len(paired) * 31)
        sf._write_json(root / "fresh_evaluation" / "paired" / f"{label}.json", paired[label])
    fixed_summary: dict[str, Any] = {}
    for name, model in models.items():
        if name == "ff_reference":
            continue
        fixed_summary[name] = {}
        for source, data in (("validation_expert", validation), ("new_final_policy", new_data)):
            replay = hd.replay_fixed_sequences(model, data, config, eval_device, policies=("normal",))
            fixed_summary[name][source] = replay["summary"]["normal"]
            sf._write_json(root / "fixed_sequence" / source / name / "summary.json", replay["summary"]["normal"])
            sf._write_json(root / "fixed_sequence" / source / name / "records.json", replay["records"]["normal"])
    cases = _selected_cases(rows)
    _plot_cases(traces, cases, root / "plots")
    ff_mean = rf._metric_means(rows["ff_reference"])
    new_mean = rf._metric_means(rows[BRANCHES[1]])
    fresh_gate = {"collision_within_0_15_of_ff": new_mean["collision"] <= ff_mean["collision"] + 0.15,
                  "position_within_1_5x_ff": new_mean["final_position_error"] <= 1.5 * ff_mean["final_position_error"],
                  "yaw_within_1_5x_ff": new_mean["final_orientation_error"] <= 1.5 * ff_mean["final_orientation_error"]}
    if hd._sha256(start) != start_hash or hd._model_digest(original) != start_model_hash:
        raise AssertionError("Recovery study modified the starting checkpoint or original GRU model.")
    summary = {"starting_checkpoint": str(start), "starting_checkpoint_sha256": start_hash,
               "seed": SEED, "new_policy_collection": new_collection,
               "new_policy_audit": new_audit, "old_policy_audit": old_audit,
               "distribution_comparison": distribution, "branches": branch_results,
               "fresh": {name: rf._metric_means(result) for name, result in rows.items()},
               "paired": paired, "fixed_sequence": fixed_summary,
               "selected_case_episode_ids": cases, "fresh_safety_gate": fresh_gate,
               "ready_for_temporal_robustness": all(fresh_gate.values()),
               "retraining_from_scratch": False, "temporal_corruption_evaluated": False}
    sf._write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--gate2", type=Path, default=GATE2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--updates", type=int, default=800)
    args = parser.parse_args()
    if args.eval_workers != 1:
        parser.error("Diagnostic rollouts are serial; each MuJoCo environment remains process-local.")
    run(args.output, args.gate2, args.device, args.eval_device, args.updates)


if __name__ == "__main__":
    main()
