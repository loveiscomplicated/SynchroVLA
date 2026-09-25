"""Uniform versus recovery-prioritized sampling from one fixed recurrent DAgger pool."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
from vla_gnn_recurrent.training import recurrent_recovery_dagger as previous
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_representation_final_comparison as final
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


OUTPUT = Path("artifacts/recurrent_recovery_priority_controlled")
CATEGORIES = ("ordinary", "late", "near_contact", "pre_collision", "recovery")
LATE_STEP = 10
NEAR_CONTACT_M = 0.010  # chosen after inspecting the combined clearance distribution
PRE_COLLISION_STEPS = 3
RECOVERY_MIN_STEP = 4
RECOVERY_POSITION_M = 0.050
RECOVERY_YAW_RAD = 0.400


def _load_sources(gate2: Path, previous_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict], list[dict]]:
    old = torch.load(gate2 / previous.OLD_POLICY.relative_to(previous.GATE2), map_location="cpu", weights_only=False)
    new = torch.load(previous_root / "new_policy_collection/policy_relabelled_trajectories.pt",
                     map_location="cpu", weights_only=False)
    old_meta = json.loads((gate2 / "onpolicy_collection/seed2811/gru/visited_metadata.json").read_text())
    new_meta = json.loads((previous_root / "new_policy_collection/visited_metadata.json").read_text())
    return old, new, old_meta, new_meta


def combine_pool(old: dict[str, Any], new: dict[str, Any], old_meta: list[dict],
                 new_meta: list[dict]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep complete sequences and give overlapping old/new episode IDs separate identities."""
    if len(old_meta) != len(old["actions"]) or len(new_meta) != len(new["actions"]):
        raise ValueError("Metadata and trajectory tensors are not aligned.")
    offset = int(old["episode_ids"].max()) + 1
    for data, rows in ((old, old_meta), (new, new_meta)):
        if [(int(a), int(b)) for a, b in zip(data["episode_ids"], data["steps"], strict=True)] != [
                (int(row["episode_id"]), int(row["t"])) for row in rows]:
            raise ValueError("Policy metadata does not match tensor sequence order.")
        if any(bool(mask) == bool(row["severe_penetration_excluded"])
               for mask, row in zip(data["loss_mask"], rows, strict=True)):
            raise ValueError("Eligibility mask disagrees with penetration metadata.")
    keys = ("states", "surface_points", "actions", "episode_ids", "steps", "loss_mask")
    pool = {key: torch.cat((old[key], new[key] + offset if key == "episode_ids" else new[key]))
            for key in keys}
    metadata = ([{**row, "source_dataset_id": "old_base_gru", "pool_episode_id": int(row["episode_id"])}
                 for row in old_meta] +
                [{**row, "source_dataset_id": "new_final_gru", "pool_episode_id": int(row["episode_id"]) + offset}
                 for row in new_meta])
    episodes = rf._episodes(pool)
    if len(episodes) != len({row["pool_episode_id"] for row in metadata}):
        raise ValueError("Combined pool bridged a trajectory boundary.")
    for episode in episodes:
        if episode["steps"].tolist() != list(range(len(episode["steps"]))):
            raise ValueError("Combined pool contains a timestep gap.")
    return pool, metadata


def _geometry_errors(state: torch.Tensor, shape: sf.SurfaceShape,
                     config: sf.SurfaceFeasibilityConfig) -> tuple[float, float]:
    values = state.numpy()
    yaw = float(np.arctan2(values[5], values[6]))
    # state[:3] is the object center in the EE frame; invert that rigid transform.
    ee = np.asarray(shape.object_center) - sf.world_action_from_local(values[:3], yaw)
    target = sf._surface_target(shape, ee, config.pregrasp_clearance)
    position = float(np.linalg.norm((ee - target[0])[[0, 2]]))
    orientation = sf.equivalent_orientation_error(yaw, target[2])
    return position, orientation


def difficulty_rows(pool: dict[str, Any], metadata: list[dict[str, Any]],
                    specs: list[sf.SurfaceEpisodeSpec], config: sf.SurfaceFeasibilityConfig) -> list[dict[str, Any]]:
    shapes = {int(spec.episode_id): spec.shape for spec in specs}
    first_collision = {}
    for row in metadata:
        if row["collision"]:
            key = row["pool_episode_id"]
            first_collision[key] = min(first_collision.get(key, int(row["t"])), int(row["t"]))
    result = []
    for index, row in enumerate(metadata):
        source_id = int(row["episode_id"])
        t = int(row["t"])
        distance = float(row["clearance_m"])
        eligible = bool(pool["loss_mask"][index])
        position, yaw = _geometry_errors(pool["states"][index], shapes[source_id], config)
        collision_t = first_collision.get(row["pool_episode_id"])
        flags = {
            "late": eligible and t >= LATE_STEP,
            "near_contact": eligible and -0.020 <= distance <= NEAR_CONTACT_M,
            "pre_collision": eligible and collision_t is not None and
                             collision_t - PRE_COLLISION_STEPS <= t < collision_t,
            "recovery": eligible and t >= RECOVERY_MIN_STEP and distance >= -0.020 and
                        (position >= RECOVERY_POSITION_M or yaw >= RECOVERY_YAW_RAD),
        }
        flags["ordinary"] = eligible and not any(flags.values())
        result.append({**row, "eligible": eligible, "position_error_m": position, "yaw_error_rad": yaw,
                       "categories": flags, "priority_score": sum(flags[key] for key in CATEGORIES[1:])})
    return result


def difficulty_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible"]]
    clearances = [float(row["clearance_m"]) for row in eligible]
    categories = {}
    for category in CATEGORIES:
        subset = [row for row in eligible if row["categories"][category]]
        categories[category] = {
            "eligible_states": len(subset),
            "trajectories": len({row["pool_episode_id"] for row in subset}),
            "mean_timestep": float(np.mean([row["t"] for row in subset])) if subset else None,
            "collision_prevalence": float(np.mean([row["collision"] for row in subset])) if subset else None,
            "old_fraction": float(np.mean([row["source_dataset_id"] == "old_base_gru" for row in subset])) if subset else None,
            "new_fraction": float(np.mean([row["source_dataset_id"] == "new_final_gru" for row in subset])) if subset else None,
        }
    overlap = {f"{a}&{b}": sum(row["categories"][a] and row["categories"][b] for row in eligible)
               for i, a in enumerate(CATEGORIES[1:]) for b in CATEGORIES[i + 2:]}
    return {"visited": len(rows), "eligible": len(eligible), "masked": len(rows) - len(eligible),
            "trajectories": len({row["pool_episode_id"] for row in rows}),
            "clearance_quantiles_m": {str(q): float(np.quantile(clearances, q))
                                      for q in (0, .05, .1, .25, .5, .75, .9, .95, 1)},
            "definitions": {"late_timestep_at_least": LATE_STEP, "near_contact_m": [-.020, NEAR_CONTACT_M],
                            "pre_first_collision_steps": PRE_COLLISION_STEPS,
                            "recovery_timestep_at_least": RECOVERY_MIN_STEP,
                            "recovery_position_error_m_at_least": RECOVERY_POSITION_M,
                            "recovery_yaw_error_rad_at_least": RECOVERY_YAW_RAD},
            "categories": categories, "overlaps": overlap,
            "source_counts": dict(Counter(row["source_dataset_id"] for row in eligible))}


def sampling_weights(rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    """Favor difficult trajectories within equal eligible-length strata."""
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["pool_episode_id"]].append(row)
    episodes = list(by_episode.values())
    scores = np.array([np.mean([row["priority_score"] for row in ep if row["eligible"]])
                       for ep in episodes], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Each trajectory needs eligible states for a difficulty score.")
    priority = np.zeros(len(episodes), dtype=bool)
    priority[np.argsort(-scores, kind="stable")[:max(10, len(episodes) // 4)]] = True
    valid_lengths = np.array([sum(row["eligible"] for row in ep) for ep in episodes])
    prioritized = np.zeros(len(episodes), dtype=np.float64)
    stratum_rules = {}
    for length in sorted(set(valid_lengths)):
        indices = np.flatnonzero(valid_lengths == length)
        selected = indices[priority[indices]]
        count, selected_count = len(indices), len(selected)
        # Retain the uniform probability mass of each valid-length stratum.
        # Cap each trajectory at 3x its uniform probability to avoid a single
        # hard episode dominating a sparse stratum.
        alpha = 0.0 if selected_count == 0 else min(
            .5, 2.0 / max(count / selected_count - 1.0, 1e-12))
        prioritized[indices] = (1.0 - alpha) / len(episodes)
        if selected_count:
            prioritized[selected] += alpha * count / (len(episodes) * selected_count)
        stratum_rules[str(int(length))] = {"trajectories": count, "priority_trajectories": selected_count,
                                          "targeted_fraction": alpha}
    assert np.isclose(prioritized.sum(), 1.0)
    assert np.isclose(np.dot(prioritized, valid_lengths), np.mean(valid_lengths))
    return prioritized, {"priority_trajectory_count": int(priority.sum()),
                         "ordinary_trajectory_count": int((~priority).sum()),
                         "max_sampling_probability_ratio_to_uniform": float(prioritized.max() * len(episodes)),
                         "score": "mean over eligible timesteps of late + near_contact + pre_collision + recovery flags",
                         "selected_score_min": float(scores[priority].min()),
                         "formula": "within each eligible-length stratum, up to 0.5 uniform-over-priority plus remaining uniform; preserve stratum mass; cap trajectory probability at 3x uniform",
                         "strata": stratum_rules,
                         "priority_trajectory_indices": np.flatnonzero(priority).tolist()}


def exposure_audit(training: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["pool_episode_id"]].append(row)
    episodes = list(by_episode.values())
    picks = {int(k): int(v) for k, v in training["sampled_policy_episode_index_counts"].items()}
    category_counts = {category: sum(count * sum(row["categories"][category] for row in episodes[index])
                                     for index, count in picks.items()) for category in CATEGORIES}
    visited = sum(count * len(episodes[index]) for index, count in picks.items())
    eligible = sum(count * sum(row["eligible"] for row in episodes[index]) for index, count in picks.items())
    if eligible != training["policy_valid_supervised_timesteps"]:
        raise AssertionError("Sampler exposure differs from the supervised loss mask.")
    provenance = dict(Counter({source: sum(count * sum(row["source_dataset_id"] == source for row in episodes[index])
                                           for index, count in picks.items())
                               for source in ("old_base_gru", "new_final_gru")}))
    return {"sampled_sequences": sum(picks.values()), "unique_trajectories": len(picks),
            "max_trajectory_reuse": max(picks.values()), "p50_trajectory_reuse": float(np.median(list(picks.values()))),
            "visited_context_timesteps": visited, "valid_policy_timesteps": eligible,
            "category_valid_timestep_exposure": category_counts, "source_timestep_exposure": provenance}


def _category_errors(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = {(row["pool_episode_id"], int(row["t"])): row for row in rows}
    metrics = ("translation_l2", "yaw_absolute", "aperture_absolute", "normalized_mse")
    result = {}
    for category in CATEGORIES:
        subset = [record for record in records if metadata[(record["episode_id"], record["timestep"])]["categories"][category]]
        result[category] = {"samples": len(subset), **{key: float(np.mean([r[key] for r in subset])) if subset else None
                                                  for key in metrics}}
    return result


def _plot_results(root: Path, rows: dict[str, list[dict[str, Any]]], traces: dict[str, list[dict[str, Any]]],
                  category_errors: dict[str, Any]) -> None:
    output = ensure_dir(root / "plots")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, metric, label in zip(axes, ("collision", "final_position_error", "final_orientation_error"),
                                 ("Collision", "Final position (m)", "Final yaw (rad)"), strict=True):
        ax.bar(list(rows), [np.mean([r[metric] for r in values]) for values in rows.values()])
        ax.tick_params(axis="x", labelrotation=25)
        ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(output / "fresh_metrics.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for model, color in (("uniform", "tab:blue"), ("recovery_prioritized", "tab:orange")):
        for ax, key in zip(axes, ("position_error", "orientation_error", "hidden_norm"), strict=True):
            sample = next(item for item in traces[model] if item["episode_id"] == 0)["steps"]
            ax.plot([r["t"] for r in sample], [r[key] for r in sample], label=model, color=color)
        axes[0].legend()
    for ax, title in zip(axes, ("Position error (m)", "Yaw error (rad)", "Hidden norm"), strict=True):
        ax.set(xlabel="Step", ylabel=title)
    fig.tight_layout()
    fig.savefig(output / "episode0_trace.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    names = CATEGORIES
    for model in ("uniform", "recovery_prioritized"):
        ax.plot(names, [category_errors[model][cat]["yaw_absolute"] for cat in names], marker="o", label=model)
    ax.set(ylabel="Processed yaw-action MAE (rad)", xlabel="Training-pool category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "category_yaw_error.png", dpi=150)
    plt.close(fig)


def run(output: Path = OUTPUT, gate2: Path = previous.GATE2,
        previous_root: Path = previous.OUTPUT, device_preference: DevicePreference = "auto",
        eval_device_preference: DevicePreference = "cpu", updates: int = 800,
        audit_only: bool = False) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Priority artifact already exists: {output}")
    root = ensure_dir(output)
    start = gate2 / previous.START.relative_to(previous.GATE2)
    ff_path = gate2 / previous.FF_REFERENCE.relative_to(previous.GATE2)
    config = rf._task_config(root, (previous.SEED,), device_preference, eval_device_preference, 16)
    temporal = replace(rf.TemporalConfig(), dagger_updates=updates)
    if not config.symmetric_robot_edges or config.point_count != 32:
        raise AssertionError("Canonical N=32 symmetric graph required.")
    expert, validation, train_specs, _ = final._current_datasets(previous.SEED, config)
    old, new, old_meta, new_meta = _load_sources(gate2, previous_root)
    previous.validate_policy_data(old, train_specs, config)
    previous.validate_policy_data(new, train_specs, config)
    pool, metadata = combine_pool(old, new, old_meta, new_meta)
    rows = difficulty_rows(pool, metadata, train_specs, config)
    audit = difficulty_audit(rows)
    weights, sampler = sampling_weights(rows)
    audit["sampler"] = sampler
    sf._write_json(root / "difficulty_audit.json", audit)
    if audit_only:
        return audit
    torch.save(pool, root / "policy_pool.pt")
    sf._write_json(root / "policy_pool_metadata.json", rows)
    train_device = select_device(device_preference)
    eval_device = select_device(eval_device_preference)
    if train_device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    start_hash = hd._sha256(start)
    original = rf._load_gru(start, config, temporal, eval_device)
    start_model_hash = hd._model_digest(original)
    seed = previous.SEED + 1_300_001
    sf._write_json(root / "config.json", {
        "starting_checkpoint": str(start), "starting_checkpoint_sha256": start_hash,
        "starting_model_sha256": start_model_hash, "model_seed": previous.SEED,
        "continuation_seed": seed, "sampling_seed": seed,
        "policy_pool_sha256": hd._tensor_digest(pool),
        "source_dataset_hashes": {"old": hd._tensor_digest(old), "new": hd._tensor_digest(new)},
        "optimizer": "fresh AdamW for each branch", "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay, "updates_per_branch": updates,
        "expert_policy_loss_weight": [0.5, 0.5], "sequence_batch_size": temporal.sequence_batch_size,
        "sequence_length": temporal.sequence_length, "validation_interval": 25,
        "source_sampling_rng": "separate fixed expert and policy streams; expert draws identical between branches",
        "validation_selection_rule": "lowest expert validation normalized per-state MSE",
        "topology": "consistent_intended_100", "point_count": 32, "local_surface_edges": False,
        "training_episode_specs": [asdict(spec) for spec in train_specs],
        "evaluation_episode_seed": previous.SEED + 60_000,
        "training_device": str(train_device), "evaluation_device": str(eval_device),
        "priority_sampler": sampler})
    trainings = {}
    checkpoints = {}
    exposures = {}
    for branch in ("uniform", "recovery_prioritized"):
        set_seed(seed)
        model = rf._load_gru(start, config, temporal, train_device)
        if hd._model_digest(model) != start_model_hash:
            raise AssertionError("Continuation weights diverged before training.")
        result = rf.train_gru(model, expert, validation, pool, config, temporal, previous.SEED,
                              train_device, updates, root / branch, sampling_seed=seed,
                              validation_interval=25, stage_label=branch,
                              policy_episode_weights=weights if branch == "recovery_prioritized" else None,
                              separate_source_rngs=True)
        result["starting_checkpoint_sha256"] = start_hash
        result["starting_model_sha256"] = start_model_hash
        result["policy_pool_sha256"] = hd._tensor_digest(pool)
        result["optimizer_state_policy"] = "fresh AdamW; no restored state"
        trainings[branch] = result
        checkpoints[branch] = Path(result["checkpoint"])
        exposures[branch] = exposure_audit(result, rows)
        sf._write_json(root / branch / "exposure.json", exposures[branch])
        print(f"[priority] {branch}: {updates} updates, policy valid {result['policy_valid_supervised_timesteps']}, "
              f"priority exposure {exposures[branch]['category_valid_timestep_exposure']}", flush=True)
    if len({result["updates"] for result in trainings.values()}) != 1:
        raise AssertionError("Update counts differ.")
    if len({result["sampled_expert_episode_index_sha256"] for result in trainings.values()}) != 1:
        raise AssertionError("Expert sequence draws diverged between U and R.")
    if len({result["expert_valid_supervised_timesteps"] for result in trainings.values()}) != 1:
        raise AssertionError("Expert valid supervision differs between U and R.")
    specs = sf.sample_episode_specs(16, previous.SEED + 60_000, config, "iid")
    previous._verify_eval_specs(specs, gate2)
    models = {"original_gru": original,
              "uniform": rf._load_gru(checkpoints["uniform"], config, temporal, eval_device),
              "recovery_prioritized": rf._load_gru(checkpoints["recovery_prioritized"], config, temporal, eval_device),
              "ff_reference": sf.load_surface_model(ff_path, rf.GRAPH_NAME, config, eval_device)}
    eval_rows = {name: [] for name in models}
    traces = {name: [] for name in models}
    env = sf.make_env(config, previous.SEED + 864_211)
    try:
        for spec in specs:
            for name, model in models.items():
                controller = "ff" if name == "ff_reference" else "gru"
                row, trace = rf.rollout(env, spec, model, controller, config, temporal, eval_device)
                eval_rows[name].append(row)
                traces[name].append({"episode_id": spec.episode_id,
                                     "spec_signature": final.spec_signature(spec), "steps": trace})
    finally:
        env.close()
    for name in models:
        if [r["spec_signature"] for r in eval_rows[name]] != [final.spec_signature(s) for s in specs]:
            raise AssertionError("Held-out IID EpisodeSpecs diverged.")
        sf._write_json(root / "fresh_eval" / name / "episodes.json", eval_rows[name])
        sf._write_json(root / "traces" / f"{name}.json", traces[name])
    prior = json.loads((gate2 / "fresh/seed2811/paired_results.json").read_text())
    for name, prior_name in (("original_gru", "gru"), ("ff_reference", "ff")):
        for old_row, new_row in zip(prior[prior_name], eval_rows[name], strict=True):
            if old_row["success"] != new_row["success"] or old_row["collision"] != new_row["collision"] or not np.isclose(
                    old_row["final_position_error"], new_row["final_position_error"], atol=1e-7):
                raise AssertionError("Canonical reference fresh rollout parity failed.")
    paired = final.paired_comparison(eval_rows["uniform"], eval_rows["recovery_prioritized"], config,
                                     "recovery_prioritized - uniform", seed)
    sf._write_json(root / "fresh_eval/paired_recovery_minus_uniform.json", paired)
    fixed = {}
    category_errors = {}
    for name in ("uniform", "recovery_prioritized"):
        model = models[name]
        replay = hd.replay_fixed_sequences(model, pool, config, eval_device, policies=("normal",))
        records = replay["records"]["normal"]
        category_errors[name] = _category_errors(records, rows)
        sf._write_json(root / "fixed_sequence" / name / "policy_pool_errors.json", category_errors[name])
        sf._write_json(root / "fixed_sequence" / name / "policy_pool_records.json", records)
        expert_replay = hd.replay_fixed_sequences(model, validation, config, eval_device, policies=("normal",))
        fixed[name] = {"policy_pool": replay["summary"]["normal"],
                       "validation_expert": expert_replay["summary"]["normal"]}
        sf._write_json(root / "fixed_sequence" / name / "expert_validation.json", fixed[name]["validation_expert"])
    _plot_results(root, eval_rows, traces, category_errors)
    means = {name: rf._metric_means(values) for name, values in eval_rows.items()}
    ff = means["ff_reference"]
    recovery = means["recovery_prioritized"]
    safety_gate = {"collision_within_0_15_of_ff": recovery["collision"] <= ff["collision"] + .15,
                   "position_within_1_5x_ff": recovery["final_position_error"] <= 1.5 * ff["final_position_error"],
                   "yaw_within_1_5x_ff": recovery["final_orientation_error"] <= 1.5 * ff["final_orientation_error"]}
    if hd._sha256(start) != start_hash or hd._model_digest(original) != start_model_hash:
        raise AssertionError("Starting GRU checkpoint changed.")
    summary = {"difficulty_audit": audit, "training": trainings, "exposure": exposures,
               "fresh": means, "paired": paired, "fixed_sequence": fixed,
               "category_errors": category_errors, "fresh_safety_gate": safety_gate,
               "ready_for_temporal_robustness": all(safety_gate.values()),
               "training_seed_count": 1, "heldout_episode_count": 16,
               "temporal_corruption_evaluated": False}
    sf._write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--gate2", type=Path, default=previous.GATE2)
    parser.add_argument("--previous-root", type=Path, default=previous.OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.eval_workers != 1:
        parser.error("Diagnostic rollouts are serial; each MuJoCo environment remains process-local.")
    run(args.output, args.gate2, args.previous_root, args.device, args.eval_device, args.updates, args.audit_only)


if __name__ == "__main__":
    main()
