"""Frozen seed-2812/2813 replication of the seed-2811 objective ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vla_gnn_recurrent.training import near_contact_translation_objective as objective
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import recurrent_hidden_diagnostic as hd
from vla_gnn_recurrent.training import recurrent_recovery_dagger as dagger
from vla_gnn_recurrent.training import recurrent_recovery_priority as priority
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_pregrasp_stabilization as stab
from vla_gnn_recurrent.training import surface_representation_final_comparison as final
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


OUTPUT = Path("artifacts/near_contact_translation_objective_multiseed")
REFERENCE = objective.OUTPUT
SEEDS = (2812, 2813)
BRANCHES = objective.BRANCHES
UPDATES = objective.UPDATES


def _write(path: Path, value: Any) -> None:
    objective._write(path, value)


def _sha(path: Path) -> str:
    return hd._sha256(path)


def _require_canonical(config: sf.SurfaceFeasibilityConfig) -> None:
    if config.point_count != 32 or not config.symmetric_robot_edges:
        raise AssertionError("Frozen N=32 symmetric graph has changed.")
    topology = sf._torch_topology(torch.zeros(1, 32, 3), torch.zeros(1, sf.STATE_DIM), config,
                                  use_local_edges=False)
    if topology.src.shape[1] != 100 or torch.any(topology.edge_type == 3):
        raise AssertionError("Frozen 100-edge topology has changed.")


def reconstruct_upstream(seed: int, root: Path, config: sf.SurfaceFeasibilityConfig,
                         temporal: rf.TemporalConfig, train_device: torch.device,
                         eval_device: torch.device, device_preference: DevicePreference,
                         expert: dict, validation: dict, train_specs: list) -> tuple[Path, dict, list]:
    """Reproduce the Gate-2 stages, then collect the final-GRU pool; no further training."""
    upstream = root / "upstream"
    start = upstream / "training" / "gru_dagger_round1" / "gru.pt"
    old_path = upstream / "onpolicy_collection" / "old_base_gru" / "policy_relabelled_trajectories.pt"
    new_path = upstream / "onpolicy_collection" / "new_final_gru" / "policy_relabelled_trajectories.pt"
    pool_path = upstream / "policy_pool.pt"
    _write(upstream / "shared_train_episode_specs.json", [asdict(s) for s in train_specs])
    provenance = {"seed": seed, "reused": [], "generated": []}
    if not start.exists():
        # These FF stages precede the GRU in the historical Gate-2 protocol.
        ff_base = stab.train_fixed_updates(rf.GRAPH_NAME, expert, validation, config, seed,
                                            upstream / "training" / "ff_base", temporal.base_updates,
                                            device_preference)
        ff_base_path = Path(ff_base["checkpoint_path"])
        final._checkpoint_config_compatible(ff_base_path, config, rf.GRAPH_NAME, seed)
        ff_model = sf.load_surface_model(ff_base_path, rf.GRAPH_NAME, config, eval_device)
        ff_policy, ff_metadata, _ = final.collect_self_policy_states(
            rf.GRAPH_NAME, ff_model, train_specs, rf._expert_reference(seed, train_specs), config,
            seed, eval_device, upstream / "onpolicy_collection" / "ff")
        ff_dagger = final.train_dagger_round1(
            rf.GRAPH_NAME, ff_base_path, expert, validation, ff_policy, ff_metadata, config,
            seed, train_device, upstream / "training" / "ff_dagger_round1",
            updates=temporal.dagger_updates)
        final._checkpoint_config_compatible(Path(ff_dagger["checkpoint_path"]), config,
                                            rf.GRAPH_NAME, seed)
        provenance["generated"].extend(["ff_base", "ff_policy", "ff_dagger_round1"])
        set_seed(seed)
        gru = rf.GraphGRUPolicy(config, temporal.gru_hidden).to(train_device)
        base = rf.train_gru(gru, expert, validation, None, config, temporal, seed, train_device,
                            temporal.base_updates, upstream / "training" / "gru_base")
        base_path = Path(base["checkpoint"])
        base_model = rf._load_gru(base_path, config, temporal, eval_device)
        old, _ = rf.collect_recurrent_policy_data(base_model, train_specs, config, temporal, seed,
                                                   eval_device, old_path.parent)
        gru = base_model.to(train_device)
        rf.train_gru(gru, expert, validation, old, config, temporal, seed, train_device,
                     temporal.dagger_updates, start.parent)
        provenance["generated"].extend(["gru_base", "old_base_gru_policy", "gru_dagger_round1"])
        print(f"[multiseed] seed {seed}: Gate-2 recurrent checkpoint generated", flush=True)
    else:
        provenance["reused"].append("gru_dagger_round1")
    if not old_path.exists():
        base_path = upstream / "training" / "gru_base" / "gru.pt"
        if not base_path.exists():
            raise FileNotFoundError("Missing old base-GRU checkpoint for policy collection.")
        base_model = rf._load_gru(base_path, config, temporal, eval_device)
        rf.collect_recurrent_policy_data(base_model, train_specs, config, temporal, seed,
                                         eval_device, old_path.parent)
        provenance["generated"].append("old_base_gru_policy")
    else:
        provenance["reused"].append("old_base_gru_policy")
    if not new_path.exists():
        final_model = rf._load_gru(start, config, temporal, eval_device)
        rf.collect_recurrent_policy_data(final_model, train_specs, config, temporal, seed,
                                         eval_device, new_path.parent)
        provenance["generated"].append("new_final_gru_policy")
    else:
        provenance["reused"].append("new_final_gru_policy")
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    new = torch.load(new_path, map_location="cpu", weights_only=False)
    dagger.validate_policy_data(old, train_specs, config)
    dagger.validate_policy_data(new, train_specs, config)
    old_meta = json.loads((old_path.parent / "visited_metadata.json").read_text())
    new_meta = json.loads((new_path.parent / "visited_metadata.json").read_text())
    if not pool_path.exists():
        pool, metadata = priority.combine_pool(old, new, old_meta, new_meta)
        metadata = priority.difficulty_rows(pool, metadata, train_specs, config)
        torch.save(pool, pool_path)
        _write(upstream / "policy_pool_metadata.json", metadata)
        provenance["generated"].append("combined_144_policy_pool")
    else:
        pool = torch.load(pool_path, map_location="cpu", weights_only=False)
        metadata = json.loads((upstream / "policy_pool_metadata.json").read_text())
        provenance["reused"].append("combined_144_policy_pool")
    if len(rf._episodes(old)) != 72 or len(rf._episodes(new)) != 72 or len(rf._episodes(pool)) != 144:
        raise AssertionError("Frozen 72+72 policy trajectory pool was not reproduced.")
    provenance.update({"gate2_checkpoint": str(start), "gate2_checkpoint_sha256": _sha(start),
                       "old_policy_sha256": _sha(old_path), "new_policy_sha256": _sha(new_path),
                       "pool_sha256": hd._tensor_digest(pool)})
    _write(upstream / "provenance.json", provenance)
    return start, pool, metadata


def build_schedule(seed: int, expert: dict, pool: dict, expert_near: np.ndarray,
                   policy_near: np.ndarray, temporal: rf.TemporalConfig) -> tuple[list[dict], dict]:
    """One uniform full-episode draw schedule, replayed byte-for-byte by both losses."""
    ex_eps, po_eps = rf._episodes(expert), rf._episodes(pool)
    if max(len(ep["actions"]) for ep in ex_eps + po_eps) > temporal.sequence_length:
        raise ValueError("Full-episode window assumption changed.")
    ex_flags, po_flags = objective.episode_flags(expert, expert_near), objective.episode_flags(pool, policy_near)
    rng_ex = np.random.default_rng(seed + 1_300_001)
    rng_po = np.random.default_rng(seed + 1_300_002)
    digest = hashlib.sha256()
    batches, ex_picks, po_picks = [], [], []
    counts = {"expert_valid_supervised_timesteps": 0, "policy_valid_supervised_timesteps": 0,
              "near_contact_supervised_timesteps": 0, "ordinary_supervised_timesteps": 0}
    for _ in range(UPDATES):
        epicks, ppicks = [], []
        ex = rf.sample_sequence_batch(ex_eps, temporal.sequence_batch_size // 2,
                                      temporal.sequence_length, rng_ex, torch.device("cpu"), picks_out=epicks)
        po = rf.sample_sequence_batch(po_eps, temporal.sequence_batch_size // 2,
                                      temporal.sequence_length, rng_po, torch.device("cpu"), picks_out=ppicks)
        ex_picks.extend(epicks)
        po_picks.extend(ppicks)
        width = max(ex[0].shape[1], po[0].shape[1])
        ex, po = objective._pad_batch(ex, width), objective._pad_batch(po, width)
        mask = torch.cat((ex[3], po[3]), 0)
        near = torch.cat((objective._flag_batch(ex_flags, epicks, width),
                          objective._flag_batch(po_flags, ppicks, width)), 0) & mask
        batch = {"states": torch.cat((ex[0], po[0]), 0),
                 "points": torch.cat((ex[1], po[1]), 0),
                 "actions": torch.cat((ex[2], po[2]), 0), "mask": mask, "near": near}
        for key, value in batch.items():
            digest.update(key.encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.contiguous().numpy().tobytes())
        batches.append(batch)
        counts["expert_valid_supervised_timesteps"] += int(ex[3].sum())
        counts["policy_valid_supervised_timesteps"] += int(po[3].sum())
        counts["near_contact_supervised_timesteps"] += int(near.sum())
        counts["ordinary_supervised_timesteps"] += int(mask.sum() - near.sum())
    audit = {"seed": seed, "updates": UPDATES, "sequence_batch_size": temporal.sequence_batch_size,
             "maximum_sequence_length": temporal.sequence_length, "uniform_policy_sampling": True,
             "all_windows_are_complete_episodes": True,
             "expert_episode_index_sha256": hashlib.sha256(np.asarray(ex_picks, dtype=np.int64).tobytes()).hexdigest(),
             "policy_episode_index_sha256": hashlib.sha256(np.asarray(po_picks, dtype=np.int64).tobytes()).hexdigest(),
             "schedule_tensor_sha256": digest.hexdigest(), **counts}
    return batches, audit


def train_branch(seed: int, name: str, start: Path, config: sf.SurfaceFeasibilityConfig,
                 temporal: rf.TemporalConfig, validation: dict, val_near: list[torch.Tensor],
                 batches: list[dict], audit: dict, device: torch.device,
                 root: Path, start_hash: str) -> dict:
    if name not in BRANCHES:
        raise ValueError(name)
    set_seed(seed + 1_300_001)
    model = rf._load_gru(start, config, temporal, device)
    if hd._model_digest(model) != start_hash:
        raise AssertionError("Branch did not load the common seed-specific starting weights.")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    checkpoint = ensure_dir(root / name) / "gru.pt"
    best, selected, history = float("inf"), None, []
    for update, cpu_batch in enumerate(batches, 1):
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        pred, _ = model.forward_sequence(batch["states"], batch["points"])
        if name == "baseline_mse":
            loss = rf.mixed_sequence_loss(pred, batch["actions"], batch["mask"],
                                          temporal.sequence_batch_size // 2, config)
        else:
            loss = objective.mixed_weighted_loss(pred, batch["actions"], batch["mask"],
                                                  batch["near"], temporal.sequence_batch_size // 2, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update % 25 == 0:
            model.eval()
            score = objective.validation_metrics(model, validation, val_near, config, device)
            history.append({"update": update, "train_loss": float(loss.detach().cpu()), **score})
            if score["unweighted_mse"] < best:
                best, selected = score["unweighted_mse"], update
                torch.save({"model_state": model.state_dict(), "seed": seed, "updates": update,
                            "best_validation_loss": best, "hidden_size": temporal.gru_hidden,
                            "topology": "consistent_intended_100", "config": asdict(config),
                            "stage": name}, checkpoint)
            model.train()
        if update % 100 == 0:
            print(f"[multiseed] seed {seed} {name} update {update}/{UPDATES}", flush=True)
    result = {"checkpoint": str(checkpoint), "checkpoint_sha256": _sha(checkpoint),
              "starting_model_sha256": start_hash, "schedule_tensor_sha256": audit["schedule_tensor_sha256"],
              "exposure": {key: audit[key] for key in ("expert_valid_supervised_timesteps",
                            "policy_valid_supervised_timesteps", "near_contact_supervised_timesteps",
                            "ordinary_supervised_timesteps")}, "updates": UPDATES,
              "selected_update": selected, "best_common_unweighted_validation_mse": best,
              "optimizer": "fresh AdamW", "learning_rate": config.learning_rate,
              "weight_decay": config.weight_decay, "validation_interval": 25,
              "selection_rule": "lowest common unweighted expert validation MSE", "history": history}
    _write(root / name / "training.json", result)
    return result


def failure_counts(rows: list[dict], config: sf.SurfaceFeasibilityConfig) -> dict[str, int]:
    counts = {key: 0 for key in ("position_only", "yaw_only", "aperture_only", "collision_only",
                                  "collision_any", "mixed", "success")}
    mapping = {"position_failure": "position_only", "orientation_failure": "yaw_only",
               "aperture_failure": "aperture_only", "collision_failure": "collision_only"}
    for row in rows:
        flags = final.failure_flags(row, config)
        active = [key for key, value in flags.items() if value]
        if flags["collision_failure"]:
            counts["collision_any"] += 1
        if not active:
            counts["success"] += 1
        elif len(active) == 1:
            counts[mapping[active[0]]] += 1
        else:
            counts["mixed"] += 1
    return counts


def evaluate(seed: int, models: dict, config: sf.SurfaceFeasibilityConfig,
             temporal: rf.TemporalConfig, device: torch.device, root: Path,
             train_specs: list) -> tuple[dict, dict]:
    specs = sf.sample_episode_specs(16, seed + 60_000, config, "iid")
    if {s.sample_identity for s in specs} & {s.sample_identity for s in train_specs}:
        raise AssertionError("Held-out EpisodeSpecs overlap training.")
    expected = [final.spec_signature(s) for s in specs]
    _write(root / "fresh_eval" / "episode_specs.json", [asdict(s) for s in specs])
    rows, traces = {name: [] for name in models}, {name: [] for name in models}
    env = sf.make_env(config, seed + 864_211)
    try:
        for spec in specs:
            for name, model in models.items():
                row, trace = rf.rollout(env, spec, model, "gru", config, temporal, device,
                                        mode="fresh", severity=0, collect=False, record_dynamics=True)
                rows[name].append(row)
                traces[name].append({"episode_id": spec.episode_id,
                                     "spec_signature": final.spec_signature(spec), "steps": trace})
    finally:
        env.close()
    for name in models:
        if [r["spec_signature"] for r in rows[name]] != expected:
            raise AssertionError("Fresh evaluation used different EpisodeSpecs.")
        if any(step["oracle_action"] is not None for ep in traces[name] for step in ep["steps"]):
            raise AssertionError("Fresh rollout called scripted expert.")
        _write(root / "fresh_eval" / name / "episodes.json", rows[name])
        _write(root / "traces" / f"{name}.json", traces[name])
    return rows, traces


def paired(rows: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1_300_001)
    result = {"episodes": 16, "weighted_minus_baseline": {}, "discordant": {}}
    for key in objective.METRICS:
        delta = np.asarray([float(b[key]) - float(a[key]) for a, b in
                            zip(rows[BRANCHES[0]], rows[BRANCHES[1]], strict=True)])
        indices = rng.integers(0, len(delta), size=(5000, len(delta)))
        means = delta[indices].mean(1)
        result["weighted_minus_baseline"][key] = {"mean": float(delta.mean()),
            "episode_bootstrap_95_ci": [float(x) for x in np.quantile(means, [.025, .975])]}
    for key in ("collision", "success"):
        m_only = sum(bool(a[key] and not b[key]) for a, b in zip(*[rows[n] for n in BRANCHES], strict=True))
        t_only = sum(bool(b[key] and not a[key]) for a, b in zip(*[rows[n] for n in BRANCHES], strict=True))
        result["discordant"][key] = {"baseline_only": m_only, "weighted_only": t_only,
                                     "mcnemar_exact_p": sf._mcnemar_exact_p(m_only, t_only)}
    return result


def run_seed(seed: int, output: Path, device_preference: DevicePreference,
             eval_device_preference: DevicePreference) -> dict:
    if seed not in SEEDS:
        raise ValueError("Only confirmatory seeds 2812 and 2813 are authorized.")
    root = ensure_dir(output / f"seed{seed}")
    device, eval_device = select_device(device_preference), select_device(eval_device_preference)
    config = rf._task_config(root, (seed,), device_preference, eval_device_preference, 16)
    temporal = replace(rf.TemporalConfig(), dagger_updates=UPDATES)
    _require_canonical(config)
    if (config.learning_rate, config.weight_decay, temporal.sequence_batch_size,
            temporal.sequence_length) != (3e-4, 1e-4, 16, 32):
        raise AssertionError("Frozen optimization or sequence protocol changed.")
    expert, validation, train_specs, val_specs = final._current_datasets(seed, config)
    start, pool, metadata = reconstruct_upstream(seed, root, config, temporal, device, eval_device,
                                                   device_preference, expert, validation, train_specs)
    if (objective.ALPHA, objective.NEAR_LOW, objective.NEAR_HIGH) != (2.0, -.020, .010):
        raise AssertionError("Frozen objective constants changed.")
    policy_mask = pool["loss_mask"].numpy().astype(bool)
    policy_near = objective.near_contact(np.asarray([r["clearance_m"] for r in metadata]), policy_mask)
    if not np.array_equal(policy_near, np.asarray([r["categories"]["near_contact"] for r in metadata])):
        raise AssertionError("Contact labels differ from deterministic diagnostic categories.")
    ex_clearance = objective.expert_clearance(expert, train_specs, config, seed + 91)
    val_clearance = objective.expert_clearance(validation, val_specs, config, seed + 92)
    expert_near = objective.near_contact(ex_clearance, np.ones(len(ex_clearance), dtype=bool))
    val_near = objective.episode_flags(validation, objective.near_contact(
        val_clearance, np.ones(len(val_clearance), dtype=bool)))
    batches, audit = build_schedule(seed, expert, pool, expert_near, policy_near, temporal)
    _write(root / "schedule_audit.json", audit)
    start_hash = hd._model_digest(rf._load_gru(start, config, temporal, device))
    config_record = {"seed": seed, "alpha": 2.0, "near_contact_range_m": [-.020, .010],
                     "starting_checkpoint": str(start), "starting_checkpoint_sha256": _sha(start),
                     "starting_model_sha256": start_hash, "training_device": str(device),
                     "evaluation_device": str(eval_device), "graph": "N=32 symmetric 100-edge no local surface edges",
                     "policy_trajectories": len(rf._episodes(pool)), "expert_trajectories": len(rf._episodes(expert)),
                     "updates": UPDATES, "schedule_tensor_sha256": audit["schedule_tensor_sha256"]}
    _write(root / "config.json", config_record)
    training = {}
    for name in BRANCHES:
        checkpoint = root / name / "gru.pt"
        record = root / name / "training.json"
        if checkpoint.exists() and record.exists():
            training[name] = json.loads(record.read_text())
            if training[name]["schedule_tensor_sha256"] != audit["schedule_tensor_sha256"] or \
                    training[name]["starting_model_sha256"] != start_hash:
                raise AssertionError("Existing branch does not match frozen schedule/start.")
        else:
            training[name] = train_branch(seed, name, start, config, temporal, validation, val_near,
                                          batches, audit, device, root, start_hash)
    if training[BRANCHES[0]]["exposure"] != training[BRANCHES[1]]["exposure"] or \
            training[BRANCHES[0]]["schedule_tensor_sha256"] != training[BRANCHES[1]]["schedule_tensor_sha256"]:
        raise AssertionError("Exposure or sampled windows differ between branches.")
    del batches
    models = {name: rf._load_gru(Path(training[name]["checkpoint"]), config, temporal, eval_device)
              for name in BRANCHES}
    validation_objectives = {name: objective.validation_metrics(model, validation, val_near, config,
                                                                 eval_device) for name, model in models.items()}
    fixed = objective.fixed_replay(models, pool, validation, metadata, config, eval_device, root)
    rows, traces = evaluate(seed, models, config, temporal, eval_device, root, train_specs)
    paired_result = paired(rows, seed)
    _write(root / "fresh_eval" / "paired.json", paired_result)
    style = objective.trajectory_style(traces, rows, config, root)
    failure = {name: failure_counts(rows[name], config) for name in BRANCHES}
    summary = {"seed": seed, "config": config_record, "schedule": audit,
               "training": training, "fixed_sequence": fixed,
               "validation_objectives": validation_objectives,
               "fresh": {name: rf._metric_means(rows[name]) for name in BRANCHES},
               "paired": paired_result, "failure_counts": failure,
               "trajectory_style_means": {name: style[name]["means"] for name in BRANCHES},
               "temporal_corruption_evaluated": False}
    _write(root / "summary.json", summary)
    return summary


def aggregate(output: Path, summaries: dict[int, dict], reference_hashes: dict[str, str]) -> dict:
    config = rf._task_config(output, (2811,), "cpu", "cpu", 16)
    reference = json.loads((REFERENCE / "summary.json").read_text())
    all_summaries = {2811: reference, **summaries}
    rows = {}
    for seed in (2811, *SEEDS):
        root = REFERENCE if seed == 2811 else output / f"seed{seed}"
        rows[seed] = {name: json.loads((root / "fresh_eval" / name / "episodes.json").read_text())
                      for name in BRANCHES}
        if [r["spec_signature"] for r in rows[seed][BRANCHES[0]]] != [
                r["spec_signature"] for r in rows[seed][BRANCHES[1]]]:
            raise AssertionError("Aggregate paired EpisodeSpecs differ.")
    seed_table = []
    for seed in (2811, *SEEDS):
        item = all_summaries[seed]
        record = {"seed": seed}
        for name, tag in zip(BRANCHES, ("M", "T"), strict=True):
            record.update({f"near_translation_{tag}": item["fixed_sequence"][name]["categories"]["near_contact"]["translation_l2"],
                           f"collision_{tag}": item["fresh"][name]["collision"],
                           f"clearance_{tag}": item["fresh"][name]["minimum_safe_clearance"],
                           f"success_{tag}": item["fresh"][name]["success"],
                           f"position_{tag}": item["fresh"][name]["final_position_error"],
                           f"trajectory_position_{tag}": item["fresh"][name]["trajectory_error"],
                           f"yaw_{tag}": item["fresh"][name]["final_orientation_error"],
                           f"trajectory_yaw_{tag}": item["fresh"][name]["trajectory_yaw_error"],
                           f"aperture_{tag}": item["fresh"][name]["final_gripper_width_error"],
                           f"aperture_only_{tag}": failure_counts(rows[seed][name], config)["aperture_only"]})
        seed_table.append(record)
    metric_keys = ("collision", "minimum_safe_clearance", "final_position_error", "trajectory_error",
                   "final_orientation_error", "trajectory_yaw_error", "final_gripper_width_error", "success")
    rng = np.random.default_rng(1_302_811)
    hierarchical = {}
    for key in metric_keys:
        per_seed = [np.asarray([float(b[key]) - float(a[key]) for a, b in
                    zip(rows[seed][BRANCHES[0]], rows[seed][BRANCHES[1]], strict=True)])
                    for seed in (2811, *SEEDS)]
        seed_means = [float(x.mean()) for x in per_seed]
        draws = []
        for _ in range(5000):
            sampled_seeds = rng.integers(0, 3, size=3)
            draws.append(float(np.mean([per_seed[i][rng.integers(0, len(per_seed[i]), size=len(per_seed[i]))].mean()
                                        for i in sampled_seeds])))
        hierarchical[key] = {"per_seed_T_minus_M": seed_means, "mean_seed_level_difference": float(np.mean(seed_means)),
                             "hierarchical_bootstrap_95_ci": [float(x) for x in np.quantile(draws, [.025, .975])]}
    result = {"seeds": [2811, 2812, 2813], "seed2811_role": "exploratory_reference_unchanged",
              "seed_table": seed_table, "hierarchical_descriptive": hierarchical,
              "reference_sha256": reference_hashes,
              "all_episode_files": {str(seed): {name: str((REFERENCE if seed == 2811 else output / f"seed{seed}") /
                                           "fresh_eval" / name / "episodes.json") for name in BRANCHES}
                                    for seed in (2811, *SEEDS)}}
    result["replication"] = {
        "mechanism_improved_seeds": [r["seed"] for r in seed_table if r["near_translation_T"] < r["near_translation_M"]],
        "clearance_improved_seeds": [r["seed"] for r in seed_table if r["clearance_T"] > r["clearance_M"]],
        "collision_reduced_seeds": [r["seed"] for r in seed_table if r["collision_T"] < r["collision_M"]],
        "joint_mechanism_and_clearance_improved_seeds": [r["seed"] for r in seed_table
                                                         if r["near_translation_T"] < r["near_translation_M"]
                                                         and r["clearance_T"] > r["clearance_M"]],
        "success_decreased_seeds": [r["seed"] for r in seed_table if r["success_T"] < r["success_M"]],
        "decision": "SEED-2811 RESULT DOES NOT REPLICATE",
    }
    _write(output / "aggregate" / "summary.json", result)
    _write(output / "aggregate" / "tables" / "three_seed_table.json", seed_table)
    return result


def plot_aggregate(output: Path, table: list[dict]) -> None:
    out = ensure_dir(output / "aggregate" / "plots")
    specs = (("near_translation", "Near-contact translation L2 (m)", "01_near_translation.png"),
             ("clearance", "Mean minimum safe clearance (m)", "02_clearance.png"),
             ("collision", "Collision rate", "03_collision.png"))
    x = np.arange(len(table))
    for key, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - .18, [r[f"{key}_M"] for r in table], width=.36, label="M")
        ax.bar(x + .18, [r[f"{key}_T"] for r in table], width=.36, label="T")
        ax.set(xticks=x, xticklabels=[str(r["seed"]) for r in table], xlabel="Training seed", ylabel=ylabel)
        ax.legend(); fig.tight_layout(); fig.savefig(out / filename, dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, ylabel in ((axes[0], "success", "Success rate"),
                            (axes[1], "aperture_only", "Aperture-only failures")):
        ax.bar(x - .18, [r[f"{key}_M"] for r in table], width=.36, label="M")
        ax.bar(x + .18, [r[f"{key}_T"] for r in table], width=.36, label="T")
        ax.set(xticks=x, xticklabels=[str(r["seed"]) for r in table], xlabel="Training seed", ylabel=ylabel)
        ax.legend()
    fig.tight_layout(); fig.savefig(out / "04_success_aperture_failures.png", dpi=160); plt.close(fig)


def run(output: Path = OUTPUT, device_preference: DevicePreference = "mps",
        eval_device_preference: DevicePreference = "cpu", eval_workers: int = 1) -> dict:
    if eval_workers != 1:
        raise ValueError("Frozen paired evaluation uses one process-local MuJoCo environment.")
    if output == REFERENCE:
        raise ValueError("Replication output must be separate from seed-2811 artifacts.")
    if select_device(device_preference).type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    reference_files = (REFERENCE / "summary.json", REFERENCE / "translation_weighted" / "gru.pt",
                       objective.CONTROLLED / "uniform" / "gru.pt")
    reference_hashes = {str(p): _sha(p) for p in reference_files}
    ensure_dir(output)
    summaries = {}
    for seed in SEEDS:
        summaries[seed] = run_seed(seed, output, device_preference, eval_device_preference)
    result = aggregate(output, summaries, reference_hashes)
    plot_aggregate(output, result["seed_table"])
    if {str(p): _sha(p) for p in reference_files} != reference_hashes:
        raise AssertionError("Historical seed-2811 artifacts changed during replication.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    args = parser.parse_args()
    run(args.output, args.device, args.eval_device, args.eval_workers)


if __name__ == "__main__":
    main()
