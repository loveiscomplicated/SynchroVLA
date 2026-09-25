"""Matched FF continuation and paired temporal evaluation of a fixed GRU candidate."""

from __future__ import annotations

import argparse
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
from vla_gnn_recurrent.training import recurrent_hidden_diagnostic as hd
from vla_gnn_recurrent.training import recurrent_recovery_priority as priority
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_representation_final_comparison as final
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


OUTPUT = Path("artifacts/recurrent_temporal_feasibility")
SEED = 2811
GATE2 = priority.previous.GATE2
PRIORITY_ROOT = priority.OUTPUT
CONTINUATION_SEED = SEED + 1_300_001
METRICS = ("success", "collision", "final_position_error", "final_orientation_error",
           "final_gripper_width_error", "trajectory_error", "trajectory_yaw_error")


def _weights_and_pool(pool_path: Path, metadata_path: Path) -> tuple[dict[str, Any], list[dict], np.ndarray]:
    pool = torch.load(pool_path, map_location="cpu", weights_only=False)
    rows = json.loads(metadata_path.read_text())
    if len(rows) != len(pool["actions"]):
        raise ValueError("Priority metadata and policy pool are misaligned.")
    weights, _ = priority.sampling_weights(rows)
    return pool, rows, weights


def train_matched_ff(start: Path, expert: dict[str, Any], validation: dict[str, Any],
                     pool: dict[str, Any], weights: np.ndarray, config: sf.SurfaceFeasibilityConfig,
                     temporal: rf.TemporalConfig, device: torch.device, output: Path,
                     updates: int = 800) -> dict[str, Any]:
    """Use the recurrent candidate's *same sequence draws*, then flatten eligible FF labels."""
    set_seed(CONTINUATION_SEED)
    model = sf.load_surface_model(start, rf.GRAPH_NAME, config, device)
    initial_hash = final.state_dict_hash(model)
    if model.use_local_edges or not config.symmetric_robot_edges or config.point_count != 32:
        raise AssertionError("Matched FF must use the canonical no-local 100-edge graph.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ex_episodes, po_episodes = rf._episodes(expert), rf._episodes(pool)
    expert_rng = np.random.default_rng(CONTINUATION_SEED)
    policy_rng = np.random.default_rng(CONTINUATION_SEED + 1)
    expert_picks: list[int] = []
    policy_picks: list[int] = []
    expert_valid = policy_valid = 0
    best = float("inf")
    selected = 0
    history = []
    checkpoint = ensure_dir(output) / "surface_graph_no_local.pt"
    batch_count = temporal.sequence_batch_size // 2
    for update in range(1, updates + 1):
        ex = rf.sample_sequence_batch(ex_episodes, batch_count, temporal.sequence_length,
                                      expert_rng, device, picks_out=expert_picks)
        po = rf.sample_sequence_batch(po_episodes, temporal.sequence_batch_size - batch_count,
                                      temporal.sequence_length, policy_rng, device,
                                      episode_weights=weights, picks_out=policy_picks)
        if ex[3].sum() == 0 or po[3].sum() == 0:
            raise RuntimeError("A sampled source has no eligible supervised state.")
        expert_valid += int(ex[3].sum())
        policy_valid += int(po[3].sum())
        # Mean per valid state in each source, then exactly 50:50 source weighting.
        ex_pred = model(ex[0][ex[3]], ex[1][ex[3]])
        po_pred = model(po[0][po[3]], po[1][po[3]])
        loss = .5 * sf._normalised_action_loss(ex_pred, ex[2][ex[3]], config) + \
               .5 * sf._normalised_action_loss(po_pred, po[2][po[3]], config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update % 25 == 0 or update == updates:
            model.eval()
            with torch.no_grad():
                val_state = validation["states"]
                val_points = validation["surface_points"]
                val_actions = validation["actions"]
                weighted = 0.0
                for start_index in range(0, len(val_actions), config.batch_size):
                    stop = min(start_index + config.batch_size, len(val_actions))
                    prediction = model(val_state[start_index:stop].to(device),
                                       val_points[start_index:stop].to(device))
                    weighted += float(sf._normalised_action_loss(
                        prediction, val_actions[start_index:stop].to(device), config).detach().cpu()) * (stop - start_index)
                value = weighted / len(val_actions)
            history.append({"update": update, "train_loss": float(loss.detach().cpu()),
                            "expert_validation_loss": value})
            if value < best:
                best, selected = value, update
                torch.save({"model_name": rf.GRAPH_NAME, "model_state": model.state_dict(),
                            "config": asdict(config), "seed": SEED,
                            "best_validation_loss": best, "optimizer_updates": updates,
                            "selected_update": update, "topology": "consistent_intended_100",
                            "continuation_condition": "matched_recovery_priority_ff"}, checkpoint)
            model.train()
    result = {"starting_checkpoint": str(start), "starting_model_hash": initial_hash,
              "checkpoint": str(checkpoint), "checkpoint_sha256": hd._sha256(checkpoint),
              "optimizer": "fresh AdamW", "optimizer_updates": updates,
              "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
              "expert_valid_supervised_states": expert_valid,
              "policy_valid_supervised_states": policy_valid,
              "expert_policy_loss_weight": [0.5, 0.5],
              "expert_sequence_draw_sha256": hashlib.sha256(np.asarray(expert_picks, dtype=np.int64).tobytes()).hexdigest(),
              "policy_sequence_draw_counts": {str(i): policy_picks.count(i) for i in sorted(set(policy_picks))},
              "selected_update": selected, "best_validation_loss": best,
              "final_update_loss": history[-1]["train_loss"], "history": history}
    sf._write_json(output / "training.json", result)
    return result


def _eval_models(models: dict[str, tuple[Any, str]], specs: list[sf.SurfaceEpisodeSpec],
                 config: sf.SurfaceFeasibilityConfig, temporal: rf.TemporalConfig,
                 device: torch.device, output: Path, mode: str, severity: int) -> dict[str, Any]:
    rows = {name: [] for name in models}
    traces = {name: [] for name in models}
    env = sf.make_env(config, SEED + 864_211)
    try:
        for spec in specs:
            for name, (model, controller) in models.items():
                before = final.state_dict_hash(model)
                row, trace = rf.rollout(env, spec, model, controller, config, temporal, device,
                                        mode=mode, severity=severity, record_dynamics=True)
                if final.state_dict_hash(model) != before:
                    raise AssertionError("Evaluation changed model weights.")
                rows[name].append(row)
                traces[name].append({"episode_id": spec.episode_id,
                                     "spec_signature": final.spec_signature(spec), "steps": trace})
    finally:
        env.close()
    expected = [final.spec_signature(spec) for spec in specs]
    for name in rows:
        if [r["spec_signature"] for r in rows[name]] != expected:
            raise AssertionError("Models received different held-out EpisodeSpecs.")
        for episode in traces[name]:
            for step in episode["steps"]:
                t = step["t"]
                expected_age = (min(severity, t) if mode == "delay" else
                                t - (temporal.stale_start - 1) if mode == "stale" and
                                temporal.stale_start <= t < temporal.stale_start + severity else 0)
                if step["observation_age"] != expected_age or step["observed_graph_timestamp"] != t - expected_age:
                    raise AssertionError("Observation corruption age differs from the fixed schedule.")
                if step["oracle_action"] is not None or step["next_true_state"] is None:
                    raise AssertionError("Evaluation must log transitions without calling the scripted expert.")
        sf._write_json(output / name / "episodes.json", rows[name])
        sf._write_json(output / name / "traces.json", traces[name])
    pair = final.paired_comparison(rows["matched_ff"], rows["gru"], config,
                                   "GRU - matched FF", SEED + 11 * severity + len(mode))
    sf._write_json(output / "paired.json", pair)
    return {"rows": rows, "traces": traces, "means": {name: rf._metric_means(values) for name, values in rows.items()},
            "paired": pair}


def _stale_recovery(trace: list[dict[str, Any]], severity: int,
                    temporal: rf.TemporalConfig, config: sf.SurfaceFeasibilityConfig) -> dict[str, Any]:
    if severity == 0:
        return {"window_observed": False, "recovery_steps": None}
    start, end = temporal.stale_start, temporal.stale_start + severity - 1
    during = [r for r in trace if start <= r["t"] <= end]
    returned = [r for r in trace if r["t"] >= end + 1]
    recovered = next((r["t"] - (end + 1) for r in returned
                      if r["position_error"] < config.success_position_threshold and
                      r["orientation_error"] < config.success_rotation_threshold), None)
    return {"window_observed": bool(during), "window_completed": len(during) == severity,
            "end_position_error": during[-1]["position_error"] if during else None,
            "end_yaw_error": during[-1]["orientation_error"] if during else None,
            "peak_position_error": max((r["position_error"] for r in during), default=None),
            "peak_yaw_error": max((r["orientation_error"] for r in during), default=None),
            "recovery_steps": recovered, "recovered_after_return": recovered is not None,
            "fresh_return_timestep": end + 1}


def _augment_recovery(result: dict[str, Any], mode: str, severity: int,
                      temporal: rf.TemporalConfig, config: sf.SurfaceFeasibilityConfig) -> None:
    if mode == "stale":
        for name in result["rows"]:
            for row, trace in zip(result["rows"][name], result["traces"][name], strict=True):
                row["stale_recovery"] = _stale_recovery(trace["steps"], severity, temporal, config)
    elif mode == "perturbation":
        for name in result["rows"]:
            for row, trace in zip(result["rows"][name], result["traces"][name], strict=True):
                after = [step for step in trace["steps"] if step["t"] >= temporal.perturbation_step]
                row["post_perturb_peak_yaw_error"] = max((s["orientation_error"] for s in after), default=None)
                row["collision_after_perturbation"] = bool(any(s["collision"] for s in after) or
                                                          row["first_collision_timestep"] is not None and
                                                          row["first_collision_timestep"] >= temporal.perturbation_step)


def _degradation(means: dict[str, Any], fresh: dict[str, Any]) -> dict[str, float]:
    return {key: float(means[key] - fresh[key]) for key in METRICS}


def _temporal_plots(root: Path, aggregate: dict[str, Any]) -> None:
    out = ensure_dir(root / "plots")
    for mode in ("stale", "delay"):
        conditions = aggregate[mode]
        levels = sorted(int(level) for level in conditions)
        for filename, metrics in (("collision_success", ("collision", "success")),
                                  ("position_yaw_degradation", ("final_position_error", "final_orientation_error"))):
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
            for ax, metric in zip(axes, metrics, strict=True):
                for model in ("matched_ff", "gru"):
                    field = "degradation" if "degradation" in filename else "means"
                    ax.plot(levels, [conditions[str(level)][field][model][metric] for level in levels],
                            marker="o", label=model)
                ax.set(xlabel=f"{mode} steps", ylabel=metric)
                ax.grid(alpha=.25)
            axes[0].legend()
            fig.tight_layout()
            fig.savefig(out / f"{mode}_{filename}.png", dpi=150)
            plt.close(fig)
    if "perturbation" in aggregate:
        traces = aggregate["perturbation"]["traces"]
        fig, ax = plt.subplots(figsize=(8, 4))
        for name in ("matched_ff", "gru"):
            episode = traces[name][0]["steps"]
            ax.plot([r["t"] for r in episode], [r["position_error"] for r in episode], label=name)
        ax.axvline(aggregate["config"]["perturbation_step"], color="black", linestyle="--")
        ax.set(xlabel="step", ylabel="position error (m)", title="Paired episode 0, not selected by outcome")
        ax.legend(); fig.tight_layout()
        fig.savefig(out / "perturbation_episode0.png", dpi=150)
        plt.close(fig)


def run_fresh(output: Path = OUTPUT, device_preference: DevicePreference = "auto",
              eval_device_preference: DevicePreference = "cpu", updates: int = 800) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Temporal study artifact already exists: {output}")
    root = ensure_dir(output)
    config = rf._task_config(root, (SEED,), device_preference, eval_device_preference, 16)
    temporal = rf.TemporalConfig()
    device, eval_device = select_device(device_preference), select_device(eval_device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    expert, validation, train_specs, _ = final._current_datasets(SEED, config)
    pool, meta, weights = _weights_and_pool(PRIORITY_ROOT / "policy_pool.pt", PRIORITY_ROOT / "policy_pool_metadata.json")
    if not config.symmetric_robot_edges or config.point_count != 32 or len(rf._episodes(pool)) != 144:
        raise AssertionError("Fixed N=32 canonical pool is required.")
    if {int(r["episode_id"]) for r in meta} != {int(s.episode_id) for s in train_specs}:
        raise AssertionError("Policy pool must contain training EpisodeSpecs only.")
    priority_config = json.loads((PRIORITY_ROOT / "config.json").read_text())
    if hd._tensor_digest(pool) != priority_config["policy_pool_sha256"]:
        raise AssertionError("Priority pool differs from the GRU candidate's supervision pool.")
    start_ff = GATE2 / "training/seed2811/ff_dagger_round1/surface_graph_no_local.pt"
    gru_path = PRIORITY_ROOT / "recovery_prioritized/gru.pt"
    if not gru_path.exists():
        raise AssertionError("GRU checkpoint is unavailable.")
    sf._write_json(root / "config.json", {
        "seed": SEED, "task": asdict(config), "temporal": asdict(temporal),
        "topology": "consistent_intended_100", "surface_points": 32, "local_edges": False,
        "historical_ff_checkpoint": str(start_ff), "historical_ff_sha256": hd._sha256(start_ff),
        "gru_candidate_checkpoint": str(gru_path), "gru_candidate_sha256": hd._sha256(gru_path),
        "policy_pool_sha256": hd._tensor_digest(pool), "priority_rule": priority_config["priority_sampler"],
        "training_device": str(device), "evaluation_device": str(eval_device),
        "training_episode_ids": [s.episode_id for s in train_specs],
        "sensor_to_command_latency": "NOT MEASURED"})
    result = train_matched_ff(start_ff, expert, validation, pool, weights, config, temporal, device,
                              root / "matched_ff/training", updates)
    prior_gru_training = json.loads((PRIORITY_ROOT / "recovery_prioritized/training.json").read_text())
    if updates == prior_gru_training["updates"]:
        if result["expert_sequence_draw_sha256"] != prior_gru_training["sampled_expert_episode_index_sha256"]:
            raise AssertionError("Matched FF did not receive GRU's expert sequence schedule.")
        if result["expert_valid_supervised_states"] != prior_gru_training["expert_valid_supervised_timesteps"] or \
                result["policy_valid_supervised_states"] != prior_gru_training["policy_valid_supervised_timesteps"]:
            raise AssertionError("Matched FF supervised timestep exposure differs from GRU.")
        if result["policy_sequence_draw_counts"] != prior_gru_training["sampled_policy_episode_index_counts"]:
            raise AssertionError("Matched FF did not receive GRU's policy sequence schedule.")
    specs = sf.sample_episode_specs(16, SEED + 60_000, config, "iid")
    priority.previous._verify_eval_specs(specs, GATE2)
    models = {"historical_ff": (sf.load_surface_model(start_ff, rf.GRAPH_NAME, config, eval_device), "ff"),
              "matched_ff": (sf.load_surface_model(Path(result["checkpoint"]), rf.GRAPH_NAME, config, eval_device), "ff"),
              "gru": (rf._load_gru(gru_path, config, temporal, eval_device), "gru")}
    fresh = _eval_models(models, specs, config, temporal, eval_device, root / "seed2811/fresh", "fresh", 0)
    prior_gru = json.loads((PRIORITY_ROOT / "fresh_eval/recovery_prioritized/episodes.json").read_text())
    for a, b in zip(prior_gru, fresh["rows"]["gru"], strict=True):
        if a["success"] != b["success"] or a["collision"] != b["collision"] or not np.isclose(
                a["final_position_error"], b["final_position_error"], atol=1e-7):
            raise AssertionError("GRU candidate fresh rollout differs from priority study.")
    old, matched, gru = (fresh["means"][name] for name in ("historical_ff", "matched_ff", "gru"))
    ff_gate = {"success": matched["success"] >= old["success"] - .10,
               "collision": matched["collision"] <= old["collision"] + .15,
               "position": matched["final_position_error"] <= 1.5 * old["final_position_error"],
               "yaw": matched["final_orientation_error"] <= 1.5 * old["final_orientation_error"]}
    gru_gate = {"success": gru["success"] >= matched["success"] - .10,
                "collision": gru["collision"] <= matched["collision"] + .15,
                "position": gru["final_position_error"] <= 1.5 * matched["final_position_error"],
                "yaw": gru["final_orientation_error"] <= 1.5 * matched["final_orientation_error"]}
    summary = {"seed": SEED, "matched_ff_training": result, "fresh": fresh["means"],
               "fresh_paired": fresh["paired"], "fresh_gate": {"matched_ff_vs_historical": ff_gate,
                                                           "gru_vs_matched_ff": gru_gate},
               "fresh_gate_passed": all(ff_gate.values()) and all(gru_gate.values()),
               "temporal_run_started": False}
    sf._write_json(root / "summary.json", summary)
    return summary


def run_temporal(output: Path = OUTPUT, eval_device_preference: DevicePreference = "cpu") -> dict[str, Any]:
    root = output
    summary = json.loads((root / "summary.json").read_text())
    if not summary["fresh_gate_passed"]:
        raise RuntimeError("Fresh gate failed; temporal evaluation is prohibited.")
    if summary.get("temporal_run_started"):
        raise FileExistsError("Temporal evaluation artifacts already exist; use a new output namespace.")
    config = rf._task_config(root, (SEED,), "auto", eval_device_preference, 16)
    temporal = rf.TemporalConfig()
    device = select_device(eval_device_preference)
    specs = sf.sample_episode_specs(16, SEED + 60_000, config, "iid")
    priority.previous._verify_eval_specs(specs, GATE2)
    ff_path = Path(summary["matched_ff_training"]["checkpoint"])
    gru_path = PRIORITY_ROOT / "recovery_prioritized/gru.pt"
    models = {"matched_ff": (sf.load_surface_model(ff_path, rf.GRAPH_NAME, config, device), "ff"),
              "gru": (rf._load_gru(gru_path, config, temporal, device), "gru")}
    fresh = summary["fresh"]
    aggregate: dict[str, Any] = {"seed": SEED, "config": asdict(temporal), "fresh": fresh,
                                 "stale": {}, "delay": {}, "perturbation": {}}
    for mode, levels in (("stale", temporal.stale_lengths), ("delay", temporal.delay_lengths)):
        for severity in levels:
            if severity == 0:
                aggregate[mode]["0"] = {"means": {name: fresh[name] for name in models},
                                          "degradation": {name: {metric: 0.0 for metric in METRICS} for name in models}}
                continue
            result = _eval_models(models, specs, config, temporal, device,
                                  root / f"seed2811/{mode}/{mode}_{severity}", mode, severity)
            _augment_recovery(result, mode, severity, temporal, config)
            for name in models:
                sf._write_json(root / f"seed2811/{mode}/{mode}_{severity}" / name / "episodes.json", result["rows"][name])
            entry = {"means": result["means"], "paired": result["paired"],
                     "degradation": {name: _degradation(result["means"][name], fresh[name]) for name in models}}
            if mode == "stale":
                entry["recovery"] = {name: {
                    "window_completed": sum(r["stale_recovery"]["window_completed"] for r in result["rows"][name]),
                    "recovered_after_return": sum(r["stale_recovery"]["recovered_after_return"] for r in result["rows"][name]),
                    "mean_recovery_steps_among_recovered": float(np.mean([
                        r["stale_recovery"]["recovery_steps"] for r in result["rows"][name]
                        if r["stale_recovery"]["recovery_steps"] is not None])) if any(
                            r["stale_recovery"]["recovery_steps"] is not None for r in result["rows"][name]) else None,
                } for name in models}
            aggregate[mode][str(severity)] = entry
            print(f"[temporal] {mode} {severity}: FF collision {entry['means']['matched_ff']['collision']:.3f}, "
                  f"GRU {entry['means']['gru']['collision']:.3f}", flush=True)
    perturb = _eval_models(models, specs, config, temporal, device,
                           root / "seed2811/perturbation", "perturbation", 0)
    _augment_recovery(perturb, "perturbation", 0, temporal, config)
    for name in models:
        sf._write_json(root / "seed2811/perturbation" / name / "episodes.json", perturb["rows"][name])
    aggregate["perturbation"] = {"means": perturb["means"], "paired": perturb["paired"],
                                  "recovery": {name: {"recovered": sum(r["recovery_time"] is not None for r in perturb["rows"][name]),
                                                      "mean_recovery_steps_among_recovered": float(np.mean([
                                                          r["recovery_time"] for r in perturb["rows"][name] if r["recovery_time"] is not None]))
                                                      if any(r["recovery_time"] is not None for r in perturb["rows"][name]) else None,
                                                      "collision_after_perturbation": sum(r["collision_after_perturbation"] for r in perturb["rows"][name]),
                                                      "peak_position_error_mean": float(np.mean([
                                                          r["post_perturb_peak_position_error"] for r in perturb["rows"][name]])),
                                                      "peak_yaw_error_mean": float(np.mean([
                                                          r["post_perturb_peak_yaw_error"] for r in perturb["rows"][name]]))}
                                               for name in models},
                                  "traces": perturb["traces"]}
    latency_spec = specs[0]
    aggregate["latency"] = {}
    for name, (model, controller) in models.items():
        bench = rf.benchmark_controller(model, controller, latency_spec, config, temporal, device)
        aggregate["latency"][name] = bench
        sf._write_json(root / "latency" / f"{name}.json", bench)
    _temporal_plots(root, aggregate)
    aggregate["perturbation"].pop("traces")
    sf._write_json(root / "seed2811/temporal_summary.json", aggregate)
    summary["temporal_run_started"] = True
    summary["temporal"] = aggregate
    sf._write_json(root / "summary.json", summary)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fresh", "temporal"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--updates", type=int, default=800)
    args = parser.parse_args()
    if args.eval_workers != 1:
        parser.error("Paired diagnostic rollout is serial; each MuJoCo environment stays process-local.")
    if args.phase == "fresh":
        result = run_fresh(args.output, args.device, args.eval_device, args.updates)
        print(f"[temporal] fresh gate {result['fresh_gate_passed']}: {result['fresh_gate']}", flush=True)
    else:
        run_temporal(args.output, args.eval_device)


if __name__ == "__main__":
    main()
