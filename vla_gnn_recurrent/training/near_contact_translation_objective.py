"""Seed-2811 objective ablation using the controlled Uniform GRU protocol."""

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
import mujoco
import numpy as np
import torch
from scipy.stats import spearmanr

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import recurrent_hidden_diagnostic as hd
from vla_gnn_recurrent.training import recurrent_recovery_priority as priority
from vla_gnn_recurrent.training import recurrent_recovery_dagger as dagger
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_representation_final_comparison as final
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


OUTPUT = Path("artifacts/near_contact_translation_objective")
CONTROLLED = Path("artifacts/recurrent_recovery_priority_controlled")
SEED = 2811
CONTINUATION_SEED = SEED + 1_300_001
UPDATES = 800
ALPHA = 2.0
NEAR_LOW, NEAR_HIGH = -.020, .010
BRANCHES = ("baseline_mse", "translation_weighted")
METRICS = ("collision", "minimum_safe_clearance", "final_position_error", "trajectory_error",
           "final_orientation_error", "trajectory_yaw_error", "success", "final_gripper_width_error",
           "steps_to_convergence")


def _write(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def near_contact(clearance: np.ndarray | torch.Tensor, eligible: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return eligible & (clearance >= NEAR_LOW) & (clearance <= NEAR_HIGH)


def weighted_sequence_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                           near: torch.Tensor, config: sf.SurfaceFeasibilityConfig,
                           alpha: float = ALPHA) -> torch.Tensor:
    if alpha == 1.0:
        return rf.sequence_loss(pred, target, mask, config)
    scales = pred.new_tensor((config.max_delta_ee,) * 3 + (config.max_delta_rotation, config.max_delta_gripper))
    q = ((pred - target) / scales).square()
    ordinary = q.mean(-1)
    weighted = (alpha * q[..., :3].sum(-1) + q[..., 3:].sum(-1)) / (3.0 * alpha + 2.0)
    return torch.where(near & mask, weighted, ordinary)[mask].mean()


def mixed_weighted_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        near: torch.Tensor, expert_count: int,
                        config: sf.SurfaceFeasibilityConfig) -> torch.Tensor:
    return .5 * weighted_sequence_loss(pred[:expert_count], target[:expert_count], mask[:expert_count],
                                        near[:expert_count], config) + \
           .5 * weighted_sequence_loss(pred[expert_count:], target[expert_count:], mask[expert_count:],
                                        near[expert_count:], config)


def expert_clearance(data: dict[str, Any], specs: list[sf.SurfaceEpisodeSpec],
                     config: sf.SurfaceFeasibilityConfig, seed: int) -> np.ndarray:
    """Read saved expert qpos into a scratch MuJoCo state; no episode is advanced."""
    env = sf.make_env(config, seed)
    by_id = {s.episode_id: s for s in specs}
    clearances = np.empty(len(data["actions"]), dtype=np.float64)
    active_id = None
    try:
        for i, (episode_id, qpos) in enumerate(zip(data["episode_ids"].tolist(), data["qpos"], strict=True)):
            if episode_id != active_id:
                env.set_surface_shape(by_id[episode_id].shape)
                active_id = episode_id
            env.data.qpos[:] = qpos.numpy()
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            clearances[i] = sf._surface_distance_metrics(env, by_id[episode_id].shape)[1]
    finally:
        env.close()
    return clearances


def episode_flags(data: dict[str, Any], flag: np.ndarray) -> list[torch.Tensor]:
    episodes = rf._episodes(data)
    if len(flag) != len(data["actions"]):
        raise ValueError("Contact labels and actions have different lengths.")
    result, offset = [], 0
    for episode in episodes:
        length = len(episode["actions"])
        result.append(torch.as_tensor(flag[offset:offset + length], dtype=torch.bool))
        offset += length
    if offset != len(flag):
        raise AssertionError("Episode contact labels did not exhaust source data.")
    return result


def _pad_batch(batch: tuple[torch.Tensor, ...], width: int) -> tuple[torch.Tensor, ...]:
    if batch[0].shape[1] == width:
        return batch
    return tuple(torch.cat([x, x.new_zeros((x.shape[0], width - x.shape[1], *x.shape[2:]))], 1) for x in batch)


def _flag_batch(flags: list[torch.Tensor], picks: list[int], width: int) -> torch.Tensor:
    out = torch.zeros(len(picks), width, dtype=torch.bool)
    for i, index in enumerate(picks):
        sequence = flags[index]
        out[i, :len(sequence)] = sequence
    return out


def build_schedule(expert: dict[str, Any], policy: dict[str, Any],
                   expert_near: np.ndarray, policy_near: np.ndarray,
                   temporal: rf.TemporalConfig,
                   historical: dict[str, Any]) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    ex_eps, po_eps = rf._episodes(expert), rf._episodes(policy)
    if max(len(ep["actions"]) for ep in ex_eps + po_eps) > temporal.sequence_length:
        raise ValueError("Historical full-episode draw replay needs max length <= 32.")
    ex_flags = episode_flags(expert, expert_near)
    po_flags = episode_flags(policy, policy_near)
    ex_rng = np.random.default_rng(CONTINUATION_SEED)
    po_rng = np.random.default_rng(CONTINUATION_SEED + 1)
    batches = []
    expert_picks, policy_picks = [], []
    ex_lengths, po_lengths = [], []
    expert_valid = policy_valid = expert_contact = policy_contact = 0
    digest = hashlib.sha256()
    for _ in range(UPDATES):
        epicks, ppicks = [], []
        ex = rf.sample_sequence_batch(ex_eps, temporal.sequence_batch_size // 2, temporal.sequence_length,
                                      ex_rng, torch.device("cpu"), ex_lengths, picks_out=epicks)
        po = rf.sample_sequence_batch(po_eps, temporal.sequence_batch_size // 2, temporal.sequence_length,
                                      po_rng, torch.device("cpu"), po_lengths, picks_out=ppicks)
        expert_picks.extend(epicks)
        policy_picks.extend(ppicks)
        width = max(ex[0].shape[1], po[0].shape[1])
        ex, po = _pad_batch(ex, width), _pad_batch(po, width)
        near_ex, near_po = _flag_batch(ex_flags, epicks, width), _flag_batch(po_flags, ppicks, width)
        states = torch.cat((ex[0], po[0]), 0)
        points = torch.cat((ex[1], po[1]), 0)
        actions = torch.cat((ex[2], po[2]), 0)
        mask = torch.cat((ex[3], po[3]), 0)
        near = torch.cat((near_ex, near_po), 0) & mask
        batch = {"states": states, "points": points, "actions": actions, "mask": mask, "near": near}
        for key, value in batch.items():
            digest.update(key.encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.contiguous().numpy().tobytes())
        batches.append(batch)
        expert_valid += int(ex[3].sum())
        policy_valid += int(po[3].sum())
        expert_contact += int((near_ex & ex[3]).sum())
        policy_contact += int((near_po & po[3]).sum())
    expert_pick_hash = hashlib.sha256(np.asarray(expert_picks, dtype=np.int64).tobytes()).hexdigest()
    policy_counts = {str(i): policy_picks.count(i) for i in sorted(set(policy_picks))}
    if expert_pick_hash != historical["sampled_expert_episode_index_sha256"]:
        raise AssertionError("Expert episode draw hash differs from controlled Uniform.")
    if policy_counts != historical["sampled_policy_episode_index_counts"]:
        raise AssertionError("Policy episode draw counts differ from controlled Uniform.")
    if (expert_valid, policy_valid) != (historical["expert_valid_supervised_timesteps"],
                                      historical["policy_valid_supervised_timesteps"]):
        raise AssertionError("Eligible sample counts differ from controlled Uniform.")
    if (sorted(ex_lengths), sorted(po_lengths)) != (
            sorted([int(k) for k, v in historical["sampled_expert_sequence_lengths"]["histogram"].items() for _ in range(v)]),
            sorted([int(k) for k, v in historical["sampled_policy_sequence_lengths"]["histogram"].items() for _ in range(v)])):
        raise AssertionError("Sampled sequence lengths differ from controlled Uniform.")
    audit = {"updates": UPDATES, "batch_size": temporal.sequence_batch_size,
             "maximum_length": temporal.sequence_length, "all_windows_are_complete_episodes": True,
             "expert_episode_index_sha256": expert_pick_hash,
             "policy_episode_index_sha256": hashlib.sha256(np.asarray(policy_picks, dtype=np.int64).tobytes()).hexdigest(),
             "schedule_tensor_sha256": digest.hexdigest(),
             "expert_valid_supervised_timesteps": expert_valid,
             "policy_valid_supervised_timesteps": policy_valid,
             "near_contact_supervised_timesteps": expert_contact + policy_contact,
             "ordinary_supervised_timesteps": expert_valid + policy_valid - expert_contact - policy_contact,
             "expert_near_contact_supervised_timesteps": expert_contact,
             "policy_near_contact_supervised_timesteps": policy_contact,
             "historical_uniform_draw_replay_verified": True}
    return batches, audit


@torch.no_grad()
def validation_metrics(model: rf.GraphGRUPolicy, validation: dict[str, Any],
                       val_near: list[torch.Tensor], config: sf.SurfaceFeasibilityConfig,
                       device: torch.device) -> dict[str, float]:
    episodes = rf._episodes(validation)
    metrics = {key: 0.0 for key in ("unweighted_mse", "weighted_objective")}
    count = 0
    for ep, near in zip(episodes, val_near, strict=True):
        states = ep["states"][None].to(device)
        points = ep["surface_points"][None].to(device)
        target = ep["actions"][None].to(device)
        pred, _ = model.forward_sequence(states, points)
        mask = torch.ones_like(target[..., 0], dtype=torch.bool)
        size = len(ep["actions"])
        metrics["unweighted_mse"] += float(rf.sequence_loss(pred, target, mask, config).cpu()) * size
        metrics["weighted_objective"] += float(weighted_sequence_loss(pred, target, mask, near[None].to(device), config).cpu()) * size
        count += size
    return {key: value / count for key, value in metrics.items()}


def train_weighted(start: Path, config: sf.SurfaceFeasibilityConfig, temporal: rf.TemporalConfig,
                   validation: dict[str, Any], val_near: list[torch.Tensor],
                   batches: list[dict[str, torch.Tensor]], schedule: dict[str, Any],
                   device: torch.device, output: Path, start_model_hash: str) -> dict[str, Any]:
    set_seed(CONTINUATION_SEED)
    model = rf._load_gru(start, config, temporal, device)
    if hd._model_digest(model) != start_model_hash:
        raise AssertionError("Weighted branch did not load exact Gate 2 weights.")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    checkpoint = ensure_dir(output) / "gru.pt"
    history = []
    best, selected = float("inf"), None
    for update, cpu_batch in enumerate(batches, 1):
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        pred, _ = model.forward_sequence(batch["states"], batch["points"])
        loss = mixed_weighted_loss(pred, batch["actions"], batch["mask"], batch["near"],
                                   temporal.sequence_batch_size // 2, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update % 25 == 0:
            model.eval()
            score = validation_metrics(model, validation, val_near, config, device)
            history.append({"update": update, "train_weighted_loss": float(loss.detach().cpu()), **score})
            # Same checkpoint-selection rule as controlled Uniform: common expert MSE.
            if score["unweighted_mse"] < best:
                best, selected = score["unweighted_mse"], update
                torch.save({"model_state": model.state_dict(), "seed": SEED, "updates": update,
                            "best_validation_loss": best, "hidden_size": temporal.gru_hidden,
                            "topology": "consistent_intended_100", "config": asdict(config),
                            "stage": "near_contact_translation_weighted_alpha2"}, checkpoint)
            model.train()
        if update % 100 == 0:
            print(f"[objective] weighted update {update}/{UPDATES}; common val MSE {history[-1]['unweighted_mse']:.5f}", flush=True)
    result = {"checkpoint": str(checkpoint), "checkpoint_sha256": hd._sha256(checkpoint),
              "starting_model_sha256": start_model_hash,
              "updates": UPDATES, "selected_update": selected,
              "best_common_unweighted_validation_mse": best,
              "selection_rule": "lowest common unweighted expert validation MSE every 25 updates",
              "optimizer": "fresh AdamW", "learning_rate": config.learning_rate,
              "weight_decay": config.weight_decay,
              "schedule_tensor_sha256": schedule["schedule_tensor_sha256"],
              "exposure": {key: schedule[key] for key in ("expert_valid_supervised_timesteps",
                         "policy_valid_supervised_timesteps", "near_contact_supervised_timesteps",
                         "ordinary_supervised_timesteps")},
              "history": history}
    _write(output / "training.json", result)
    return result


def fixed_replay(models: dict[str, rf.GraphGRUPolicy], pool: dict[str, Any],
                 validation: dict[str, Any], metadata: list[dict[str, Any]],
                 config: sf.SurfaceFeasibilityConfig, device: torch.device,
                 root: Path) -> dict[str, Any]:
    result = {}
    for name, model in models.items():
        policy = hd.replay_fixed_sequences(model, pool, config, device, policies=("normal",))
        expert = hd.replay_fixed_sequences(model, validation, config, device, policies=("normal",))
        records = policy["records"]["normal"]
        categories = priority._category_errors(records, metadata)
        _write(root / "fixed_sequence" / name / "policy_records.json", records)
        _write(root / "fixed_sequence" / name / "expert_records.json", expert["records"]["normal"])
        result[name] = {"policy_pool": policy["summary"]["normal"],
                        "expert_validation": expert["summary"]["normal"],
                        "categories": categories,
                        "policy_input_sha256": policy["input_sha256"],
                        "expert_input_sha256": expert["input_sha256"]}
    if result[BRANCHES[0]]["policy_input_sha256"] != result[BRANCHES[1]]["policy_input_sha256"] or \
       result[BRANCHES[0]]["expert_input_sha256"] != result[BRANCHES[1]]["expert_input_sha256"]:
        raise AssertionError("Fixed-sequence inputs differ between branches.")
    _write(root / "fixed_sequence" / "summary.json", result)
    return result


def fresh_evaluation(models: dict[str, rf.GraphGRUPolicy],
                     config: sf.SurfaceFeasibilityConfig, temporal: rf.TemporalConfig,
                     device: torch.device, root: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    specs = sf.sample_episode_specs(16, SEED + 60_000, config, "iid")
    dagger._verify_eval_specs(specs, dagger.GATE2)
    expected = [final.spec_signature(s) for s in specs]
    rows = {name: [] for name in models}
    traces = {name: [] for name in models}
    env = sf.make_env(config, SEED + 864_211)
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
        if [row["spec_signature"] for row in rows[name]] != expected:
            raise AssertionError("Held-out specifications differ.")
        if any(step["oracle_action"] is not None for ep in traces[name] for step in ep["steps"]):
            raise AssertionError("Fresh evaluation called the scripted expert.")
        _write(root / "fresh_eval" / name / "episodes.json", rows[name])
        _write(root / "traces" / f"{name}.json", traces[name])
    old_baseline = json.loads((CONTROLLED / "fresh_eval/uniform/episodes.json").read_text())
    for before, after in zip(old_baseline, rows[BRANCHES[0]], strict=True):
        if (before["collision"] != after["collision"] or before["success"] != after["success"] or
                not np.isclose(before["final_position_error"], after["final_position_error"], atol=1e-7)):
            raise AssertionError("Reused baseline did not reproduce its historical fresh rollout.")
    return rows, traces


def paired_summary(rows: dict[str, list[dict]], config: sf.SurfaceFeasibilityConfig) -> dict[str, Any]:
    baseline, weighted = rows[BRANCHES[0]], rows[BRANCHES[1]]
    rng = np.random.default_rng(CONTINUATION_SEED)
    result = {"episodes": 16, "weighted_minus_baseline": {}, "discordant": {}}
    for key in METRICS:
        delta = np.asarray([float(b[key]) - float(a[key]) for a, b in zip(baseline, weighted, strict=True)])
        indices = rng.integers(0, len(delta), size=(5000, len(delta)))
        means = delta[indices].mean(axis=1)
        result["weighted_minus_baseline"][key] = {"mean": float(delta.mean()),
                                                   "episode_bootstrap_95_ci": [float(x) for x in np.quantile(means, [.025, .975])]}
    for key in ("collision", "success"):
        baseline_only = sum(bool(a[key] and not b[key]) for a, b in zip(baseline, weighted, strict=True))
        weighted_only = sum(bool(b[key] and not a[key]) for a, b in zip(baseline, weighted, strict=True))
        result["discordant"][key] = {"baseline_only": baseline_only, "weighted_only": weighted_only,
                                     "mcnemar_exact_p": sf._mcnemar_exact_p(baseline_only, weighted_only)}
    return result


def _oracle_from_step(step: dict, shape: sf.SurfaceShape, config: sf.SurfaceFeasibilityConfig) -> np.ndarray:
    ee = np.asarray(step["true_ee_pose"], dtype=np.float64)
    yaw = float(step["true_yaw"])
    target, _, target_yaw, target_width, _ = sf._surface_target(shape, ee, config.pregrasp_clearance)
    local = sf.rotate_xz(torch.as_tensor((target - ee).reshape(1, 3), dtype=torch.float32), -yaw)[0]
    local = sf.clamp_delta(local, config.max_delta_ee)
    dg = np.clip(sf.opening_fraction(target_width) - sf.opening_fraction(step["true_aperture"]),
                 -config.max_delta_gripper, config.max_delta_gripper)
    return np.asarray([float(local[0]), 0, float(local[2]),
                       np.clip(sf.wrap_angle(target_yaw - yaw), -config.max_delta_rotation, config.max_delta_rotation), dg],
                      dtype=np.float32)


def focused_transitions(traces: dict[str, list[dict]], config: sf.SurfaceFeasibilityConfig,
                        root: Path) -> dict[str, Any]:
    records = []
    summary = {}
    for name, episodes in traces.items():
        for ep in episodes:
            signature = json.loads(ep["spec_signature"])
            shape_data = signature["shape"]
            shape = sf.SurfaceShape(**{**shape_data, "object_center": tuple(shape_data["object_center"])})
            steps = ep["steps"]
            for i, step in enumerate(steps[:-3]):
                if step["severe_penetration_excluded"] or step["collision"]:
                    continue
                oracle = _oracle_from_step(step, shape, config)
                action = np.asarray(step["processed_action"])
                near = NEAR_LOW <= step["clearance_m"] <= NEAR_HIGH
                future = steps[i+1:i+4]
                records.append({"controller": name, "episode": ep["episode_id"], "t": step["t"],
                                "near_contact": near, "clearance": step["clearance_m"],
                                "translation_error": float(np.linalg.norm(action[:3] - oracle[:3])),
                                "minimum_future_clearance_3": float(min(s["clearance_m"] for s in future)),
                                "clearance_drop_3": float(step["clearance_m"] - min(s["clearance_m"] for s in future)),
                                "future_collision_3": int(any(s["collision"] for s in future)),
                                "position_progress_1": float(step["position_error"] - steps[i+1]["position_error"]),
                                "yaw_progress_1": float(step["orientation_error"] - steps[i+1]["orientation_error"])})
        near_rows = [r for r in records if r["controller"] == name and r["near_contact"]]
        association = {}
        for outcome in ("future_collision_3", "clearance_drop_3", "minimum_future_clearance_3"):
            if len(near_rows) > 2:
                rho, _ = spearmanr([r["translation_error"] for r in near_rows], [r[outcome] for r in near_rows])
                association[outcome] = None if np.isnan(rho) else float(rho)
            else:
                association[outcome] = None
        summary[name] = {"near_contact_rows": len(near_rows),
                         "near_contact_future_collision_rate": float(np.mean([r["future_collision_3"] for r in near_rows])) if near_rows else None,
                         "near_contact_mean_translation_error": float(np.mean([r["translation_error"] for r in near_rows])) if near_rows else None,
                         "translation_error_association": association}
    _write(root / "state_transition" / "records.json", records)
    _write(root / "state_transition" / "summary.json", summary)
    return summary


def trajectory_style(traces: dict[str, list[dict]], rows: dict[str, list[dict]],
                     config: sf.SurfaceFeasibilityConfig, root: Path) -> dict[str, Any]:
    summary = {}
    for name, episodes in traces.items():
        stats = []
        by_id = {row["episode_id"]: row for row in rows[name]}
        for ep in episodes:
            actions = np.asarray([s["processed_action"] for s in ep["steps"]], dtype=float)
            trans = np.linalg.norm(actions[:, :3], axis=1)
            stats.append({"episode": ep["episode_id"],
                          "mean_translation": float(trans.mean()), "max_translation": float(trans.max()),
                          "mean_abs_yaw": float(np.abs(actions[:, 3]).mean()),
                          "mean_normalized_action_variation": float(np.linalg.norm(np.diff(actions / np.array(
                              [config.max_delta_ee] * 3 + [config.max_delta_rotation, config.max_delta_gripper]), axis=0), axis=1).mean()) if len(actions) > 1 else None,
                          "near_contact_steps": sum(NEAR_LOW <= s["clearance_m"] <= NEAR_HIGH for s in ep["steps"]),
                          "minimum_safe_clearance": by_id[ep["episode_id"]]["minimum_safe_clearance"]})
        summary[name] = {"episodes": stats, "means": {k: float(np.mean([r[k] for r in stats])) for k in stats[0] if k != "episode"}}
    _write(root / "state_transition" / "trajectory_style.json", summary)
    return summary


def plots(fixed: dict[str, Any], rows: dict[str, list[dict]], traces: dict[str, list[dict]],
          transitions: dict[str, Any], root: Path) -> None:
    out = ensure_dir(root / "plots")
    labels = ["Baseline", "Weighted"]
    errors = [fixed[name]["categories"]["near_contact"]["translation_l2"] for name in BRANCHES]
    fig, ax = plt.subplots(figsize=(5, 4)); ax.bar(labels, errors, color=["tab:blue", "tab:orange"])
    ax.set(ylabel="Near-contact translation L2 (m)"); fig.tight_layout()
    fig.savefig(out / "01_near_contact_translation_error.png", dpi=160); plt.close(fig)
    base = {r["episode_id"]: r for r in rows[BRANCHES[0]]}
    weighted = {r["episode_id"]: r for r in rows[BRANCHES[1]]}
    ids = sorted(base)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(ids, [base[i]["minimum_safe_clearance"] for i in ids], "o-", label="Baseline")
    ax.plot(ids, [weighted[i]["minimum_safe_clearance"] for i in ids], "o-", label="Weighted")
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Held-out episode", ylabel="Minimum safe clearance (m)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "02_paired_minimum_clearance.png", dpi=160); plt.close(fig)
    discordant = [i for i in ids if base[i]["collision"] != weighted[i]["collision"] or
                  base[i]["success"] != weighted[i]["success"]]
    if not discordant:
        discordant = [max(ids, key=lambda i: abs(weighted[i]["minimum_safe_clearance"] - base[i]["minimum_safe_clearance"]))]
    for episode_id in discordant[:3]:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for name, color in zip(BRANCHES, ("tab:blue", "tab:orange"), strict=True):
            step = next(ep["steps"] for ep in traces[name] if ep["episode_id"] == episode_id)
            axes[0].plot([s["t"] for s in step], [s["position_error"] for s in step], label=name, color=color)
            axes[1].plot([s["t"] for s in step], [s["orientation_error"] for s in step], label=name, color=color)
        axes[0].set(ylabel="Position error (m)"); axes[1].set(xlabel="Step", ylabel="Yaw error (rad)")
        axes[0].legend(); fig.tight_layout(); fig.savefig(out / f"03_disagreement_episode{episode_id}.png", dpi=160); plt.close(fig)
    record = json.loads((root / "state_transition/records.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, name in zip(axes, BRANCHES, strict=True):
        subset = [r for r in record if r["controller"] == name and r["near_contact"]]
        for collision, color in ((0, "tab:blue"), (1, "tab:red")):
            selected = [r for r in subset if r["future_collision_3"] == collision]
            ax.scatter([r["translation_error"] for r in selected], [r["minimum_future_clearance_3"] for r in selected],
                       s=20, alpha=.7, color=color, label="Collision ≤3" if collision else "No collision")
        ax.set(xlabel="Near-contact translation error (m)", title=name)
        ax.axhline(0, color="black", linewidth=.8)
    axes[0].set_ylabel("Minimum next-3 clearance (m)"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "04_near_contact_error_future_clearance.png", dpi=160); plt.close(fig)


def run(output: Path = OUTPUT, device_preference: DevicePreference = "mps",
        eval_device_preference: DevicePreference = "cpu", eval_workers: int = 1) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Objective artifact already exists: {output}")
    if eval_workers != 1:
        raise ValueError("Paired MuJoCo evaluation is serial and uses one process-local environment.")
    root = ensure_dir(output)
    device, eval_device = select_device(device_preference), select_device(eval_device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    config = rf._task_config(root, (SEED,), device_preference, eval_device_preference, 16)
    temporal = replace(rf.TemporalConfig(), dagger_updates=UPDATES)
    if config.point_count != 32 or not config.symmetric_robot_edges:
        raise AssertionError("Canonical N=32 100-edge graph configuration changed.")
    controlled = json.loads((CONTROLLED / "config.json").read_text())
    baseline_training = json.loads((CONTROLLED / "uniform/training.json").read_text())
    start = Path(controlled["starting_checkpoint"])
    baseline_checkpoint = Path(baseline_training["checkpoint"])
    if hd._sha256(start) != controlled["starting_checkpoint_sha256"]:
        raise AssertionError("Gate 2 starting checkpoint hash changed.")
    if str(device) != controlled["training_device"] or str(eval_device) != controlled["evaluation_device"]:
        raise ValueError("Reusing historical Uniform requires its exact MPS/CPU device protocol.")
    if (controlled["updates_per_branch"] != UPDATES or controlled["sampling_seed"] != CONTINUATION_SEED or
            controlled["sequence_batch_size"] != temporal.sequence_batch_size or
            controlled["sequence_length"] != temporal.sequence_length or
            controlled["learning_rate"] != config.learning_rate or
            controlled["weight_decay"] != config.weight_decay or
            baseline_training["updates"] != UPDATES or
            baseline_training["validation_interval_updates"] != 25):
        raise AssertionError("Historical Uniform training protocol differs.")
    expert, validation, train_specs, val_specs = final._current_datasets(SEED, config)
    pool = torch.load(CONTROLLED / "policy_pool.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((CONTROLLED / "policy_pool_metadata.json").read_text())
    if hd._tensor_digest(pool) != controlled["policy_pool_sha256"] or len(rf._episodes(pool)) != 144:
        raise AssertionError("Combined 72+72 policy pool changed.")
    if json.loads(json.dumps([asdict(s) for s in train_specs])) != controlled["training_episode_specs"]:
        raise AssertionError("Expert training EpisodeSpecs differ from controlled Uniform.")
    policy_mask = pool["loss_mask"].numpy().astype(bool)
    policy_clearance = np.asarray([r["clearance_m"] for r in metadata])
    policy_near = near_contact(policy_clearance, policy_mask)
    if not np.array_equal(policy_near, np.asarray([r["categories"]["near_contact"] for r in metadata])):
        raise AssertionError("Near-contact labels differ from existing deterministic definition.")
    ex_clearance = expert_clearance(expert, train_specs, config, SEED + 91)
    val_clearance = expert_clearance(validation, val_specs, config, SEED + 92)
    expert_near = near_contact(ex_clearance, np.ones(len(ex_clearance), dtype=bool))
    val_near = near_contact(val_clearance, np.ones(len(val_clearance), dtype=bool))
    schedule_batches, schedule = build_schedule(expert, pool, expert_near, policy_near, temporal, baseline_training)
    _write(root / "schedule_audit.json", schedule)
    _write(root / "contact_metadata.json", {"source": "saved expert qpos and surface shape; saved policy clearance metadata",
                                                 "expert_near_states": int(expert_near.sum()),
                                                 "validation_near_states": int(val_near.sum()),
                                                 "policy_near_eligible_states": int(policy_near.sum()),
                                                 "policy_masked_states": int((~policy_mask).sum()),
                                                 "near_contact_range_m": [NEAR_LOW, NEAR_HIGH]})
    start_model = rf._load_gru(start, config, temporal, device)
    start_model_hash = hd._model_digest(start_model)
    if start_model_hash != controlled["starting_model_sha256"]:
        raise AssertionError("Gate 2 starting model weights changed.")
    del start_model
    baseline = {"checkpoint": str(baseline_checkpoint), "checkpoint_sha256": hd._sha256(baseline_checkpoint),
                "reused_historical_controlled_uniform": True,
                "starting_checkpoint_sha256": hd._sha256(start),
                "starting_model_sha256": start_model_hash,
                "schedule_tensor_sha256": schedule["schedule_tensor_sha256"],
                "exposure": {key: schedule[key] for key in ("expert_valid_supervised_timesteps", "policy_valid_supervised_timesteps",
                             "near_contact_supervised_timesteps", "ordinary_supervised_timesteps")},
                "historical_training": str(CONTROLLED / "uniform/training.json")}
    _write(root / "baseline_mse" / "provenance.json", baseline)
    _write(root / "config.json", {"seed": SEED, "alpha": ALPHA, "near_contact_range_m": [NEAR_LOW, NEAR_HIGH],
                                  "starting_checkpoint": str(start), "starting_checkpoint_sha256": hd._sha256(start),
                                  "starting_model_sha256": start_model_hash,
                                  "expert_trajectories": len(rf._episodes(expert)), "policy_trajectories": len(rf._episodes(pool)),
                                  "expert_validation_states": len(validation["actions"]),
                                  "schedule_tensor_sha256": schedule["schedule_tensor_sha256"],
                                  "training_device": str(device), "evaluation_device": str(eval_device),
                                  "optimizer": "fresh AdamW", "learning_rate": config.learning_rate,
                                  "weight_decay": config.weight_decay, "updates": UPDATES,
                                  "sequence_batch_size": temporal.sequence_batch_size,
                                  "sequence_length": temporal.sequence_length,
                                  "selection_rule": "lowest common unweighted expert validation MSE every 25 updates",
                                  "graph": "N=32 symmetric 100-edge no local surface edges"})
    weighted_training = train_weighted(start, config, temporal, validation,
                                       episode_flags(validation, val_near), schedule_batches, schedule,
                                       device, root / "translation_weighted", start_model_hash)
    del schedule_batches
    models = {BRANCHES[0]: rf._load_gru(baseline_checkpoint, config, temporal, eval_device),
              BRANCHES[1]: rf._load_gru(Path(weighted_training["checkpoint"]), config, temporal, eval_device)}
    validation_objectives = {name: validation_metrics(model, validation, episode_flags(validation, val_near),
                                                      config, eval_device) for name, model in models.items()}
    _write(root / "fixed_sequence" / "validation_objectives.json", validation_objectives)
    fixed = fixed_replay(models, pool, validation, metadata, config, eval_device, root)
    rows, traces = fresh_evaluation(models, config, temporal, eval_device, root)
    paired = paired_summary(rows, config)
    _write(root / "fresh_eval" / "paired.json", paired)
    transition = focused_transitions(traces, config, root)
    style = trajectory_style(traces, rows, config, root)
    plots(fixed, rows, traces, transition, root)
    fresh_means = {name: rf._metric_means(rows[name]) for name in BRANCHES}
    summary = {"seed": SEED, "starting_checkpoint_sha256": hd._sha256(start),
               "schedule": schedule, "baseline": baseline, "weighted_training": weighted_training,
               "fixed_sequence": fixed, "validation_objectives": validation_objectives,
               "fresh": fresh_means,
               "ff_context": rf._metric_means(json.loads((CONTROLLED / "fresh_eval/ff_reference/episodes.json").read_text())),
               "paired": paired, "state_transition": transition,
               "trajectory_style_means": {name: style[name]["means"] for name in BRANCHES},
               "temporal_corruption_evaluated": False, "training_seed_count": 1}
    _write(root / "summary.json", summary)
    return summary


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
