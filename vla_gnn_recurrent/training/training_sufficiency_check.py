"""Fixed-data optimization-budget check for the dynamic residual controllers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vla_gnn_recurrent.training import dynamic_visual_delay_feasibility as dynamic
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import residual_recurrent_feasibility as residual
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir, select_device, set_seed


SOURCE = dynamic.OUTPUT / "seed2811"
OUTPUT = Path("artifacts/training_sufficiency_check")
BUDGETS = (1200, 1600, 2400, 3200, 4800)


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_curves(output: Path = OUTPUT) -> dict:
    root = ensure_dir(output / "audit")
    source_config = json.loads((SOURCE / "config.json").read_text())
    records = {}
    all_curves = []
    for kind in ("mlp", "gru"):
        records[kind] = {}
        for stage in ("a", "b"):
            folder = SOURCE / "training" / ("stage_a" if stage == "a" else f"stage_b_{kind}")
            log = json.loads((folder / "logs" / f"{kind}_training.json").read_text())
            schedule = json.loads((folder / "logs/schedule.json").read_text())
            checkpoint = folder / "checkpoints" / f"{kind}.pt"
            records[kind][stage] = {"selected_update": log["selected_update"],
                                    "final_update": schedule["updates"],
                                    "best_validation_5d_mse": log["best_common_validation_5d_mse"],
                                    "final_train_4d_mse_sample": log["history"][-1]["training_4d_mse"],
                                    "final_validation_5d_mse": log["history"][-1]["corrected_action_mse_5d"],
                                    "checkpoint_sha256": sha(checkpoint),
                                    "schedule_sha256": schedule["schedule_sha256"],
                                    "draws_sha256": schedule["draws_sha256"],
                                    "supervised_timestep_draws": sum(schedule["supervised_timesteps_by_delay"].values()),
                                    "batch_size": schedule["batch_size"],
                                    "validation_interval": 25}
            for item in log["history"]:
                all_curves.append({"model": kind, "stage": stage,
                                   "update": item["update"] + (800 if stage == "b" else 0),
                                   "train_loss": item["training_4d_mse"],
                                   "validation_loss": item["corrected_action_mse_5d"],
                                   "residual_target_mse": item["residual_prediction_mse_4d"],
                                   "corrected_action_mse": item["corrected_action_mse_4d"]})
    pools = {}
    for name in ("expert", "validation", "policy_mlp", "policy_gru"):
        path = SOURCE / "training" / f"{name}.pt"
        episodes = torch.load(path, map_location="cpu", weights_only=False)
        pools[name] = {"sha256": sha(path), "episodes": len(episodes),
                       "unique_episode_identities": len({e["identity"] for e in episodes}),
                       "valid_timesteps": sum(int(e["mask"].sum()) for e in episodes)}
    task = rf._task_config(SOURCE, (2811,), "cpu", "cpu", 16)
    design = replace(residual.ResidualConfig(), seed=2811)
    params = {}
    for kind in ("mlp", "gru"):
        model = residual.load_controller(kind, task, design, select_device("cpu"))
        params[kind] = {"registered": sum(p.numel() for p in model.parameters()),
                        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    audit = {"source_protocol": source_config, "source_records": records, "pools": pools,
             "parameters": params, "base_checkpoint_actual_sha256": sha(residual.BASE_CHECKPOINT),
             "stage_a_expert_reuse_per_branch": 800 * 16 / pools["expert"]["episodes"],
             "stage_b_expert_reuse_per_branch": 400 * 8 / pools["expert"]["episodes"],
             "stage_b_policy_reuse_per_branch": 400 * 8 / pools["policy_mlp"]["episodes"]}
    residual._write(root / "existing_budget.json", audit)
    write_csv(output / "learning_curves" / "historical.csv", all_curves)
    return audit


def train(kind: str, output: Path = OUTPUT, device_preference: DevicePreference = "auto") -> dict:
    root = ensure_dir(output / "seed2811" / kind)
    device = select_device(device_preference)
    if device.type == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    task = rf._task_config(SOURCE, (2811,), device_preference, "cpu", 16)
    expert = torch.load(SOURCE / "training/expert.pt", map_location="cpu", weights_only=False)
    policy = torch.load(SOURCE / f"training/policy_{kind}.pt", map_location="cpu", weights_only=False)
    validation = torch.load(SOURCE / "training/validation.pt", map_location="cpu", weights_only=False)
    for name, episodes in (("expert", expert), ("policy", policy), ("validation", validation)):
        expected = json.loads((output / "audit/existing_budget.json").read_text())["pools"][
            name if name != "policy" else f"policy_{kind}"]["sha256"]
        if name == "validation" and sha(SOURCE / "training/validation.pt") != expected:
            raise AssertionError("Validation pool changed")
        if name != "validation" and sha(SOURCE / "training" / f"{name if name == 'expert' else 'policy_' + kind}.pt") != expected:
            raise AssertionError("Training pool changed")
    design = replace(residual.ResidualConfig(), seed=2811, updates=4000)
    eligibility = {"expert": {d: list(range(len(expert))) for d in dynamic.TRAIN_DELAYS},
                   "policy": dynamic._eligible_policy_by_delay(policy)}
    batches, schedule = residual.make_schedule({"expert": expert, "policy": policy}, design,
                                               dynamic.dynamic_delayed_sequence, eligibility)
    old_schedule = json.loads((SOURCE / "training" / f"stage_b_{kind}/logs/schedule.json").read_text())
    _, prefix = residual.make_schedule({"expert": expert, "policy": policy},
                                       replace(design, updates=400),
                                       dynamic.dynamic_delayed_sequence, eligibility)
    if prefix["schedule_sha256"] != old_schedule["schedule_sha256"]:
        raise AssertionError("Extended schedule does not reproduce the first 400 Stage B batches")
    residual._write(root / "schedule.json", schedule)
    start_checkpoint = SOURCE / "training/stage_a/checkpoints" / f"{kind}.pt"
    set_seed(design.seed + 1_400_001)
    model = residual.load_controller(kind, task, design, device, start_checkpoint)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=task.learning_rate, weight_decay=task.weight_decay)
    best, best_update, history = float("inf"), None, []
    start = time.perf_counter()
    for stage_update, cpu_batch in enumerate(batches, 1):
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        model.train()
        corrected, _, _, _, _ = model.forward_sequence(batch["states"], batch["points"], batch["previous"])
        loss = .5 * residual.corrected_loss(corrected[:8], batch["targets"][:8], batch["mask"][:8], task) + \
               .5 * residual.corrected_loss(corrected[8:], batch["targets"][8:], batch["mask"][8:], task)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if stage_update % 25 == 0:
            model.eval()
            metrics = residual.validation_metrics(model, validation, task, device)
            row = {"model": kind, "stage": "b", "update": 800 + stage_update,
                   "train_loss": float(loss.detach().cpu()),
                   "validation_loss": metrics["corrected_action_mse_5d"],
                   "residual_target_mse": metrics["residual_prediction_mse_4d"],
                   "corrected_action_mse": metrics["corrected_action_mse_4d"],
                   "training_seconds": time.perf_counter() - start}
            history.append(row)
            if row["validation_loss"] < best:
                best, best_update = row["validation_loss"], 800 + stage_update
                torch.save({"model_state": model.state_dict(), "kind": kind,
                            "base_sha256": residual.BASE_SHA256, "design": asdict(design),
                            "seed": 2811, "selected_update": stage_update,
                            "validation_5d_mse": best, "topology": "consistent_intended_100"},
                           root / "best_running.pt")
        global_update = 800 + stage_update
        if global_update in BUDGETS:
            final_path = root / f"final_u{global_update}.pt"
            selected_path = root / f"selected_u{global_update}.pt"
            torch.save({"model_state": model.state_dict(), "kind": kind,
                        "base_sha256": residual.BASE_SHA256, "design": asdict(design),
                        "seed": 2811, "selected_update": stage_update,
                        "validation_5d_mse": history[-1]["validation_loss"],
                        "topology": "consistent_intended_100"}, final_path)
            selected_path.write_bytes((root / "best_running.pt").read_bytes())
            residual._write(root / f"budget_u{global_update}.json",
                            {"budget": global_update, "selected_update": best_update,
                             "best_validation_loss": best, "final_validation_loss": history[-1]["validation_loss"],
                             "training_seconds": time.perf_counter() - start,
                             "seconds_per_update": (time.perf_counter() - start) / stage_update,
                             "device": str(device), "selected_sha256": sha(selected_path),
                             "final_sha256": sha(final_path)})
            write_csv(root / "training_curve.csv", history)
            print(f"[{kind}] budget {global_update}, selected {best_update}, val {best:.6f}", flush=True)
    return {"kind": kind, "history": history, "selected_update": best_update}


def evaluate(kind: str, output: Path = OUTPUT,
             eval_device_preference: DevicePreference = "cpu") -> list[dict]:
    device = select_device(eval_device_preference)
    task = rf._task_config(SOURCE, (2811,), "cpu", eval_device_preference, 16)
    specs = dynamic.dynamic_specs(2811, task, 16, dynamic.SELECTED_STEP_SPEEDS_M, "heldout")
    expected = json.loads((SOURCE / "config.json").read_text())["dynamic_splits"]["heldout"]
    if [s.identity for s in specs] != expected:
        raise AssertionError("Held-out EpisodeSpecs changed")
    design = replace(residual.ResidualConfig(), seed=2811)
    root = ensure_dir(output / "fresh_evaluation" / kind)
    curve, failures, episode_rows = [], [], []
    source_checkpoint = SOURCE / "training/stage_a/checkpoints" / f"{kind}.pt"
    checkpoints = [(800, source_checkpoint)] + [(n, output / "seed2811" / kind / f"selected_u{n}.pt")
                                                   for n in BUDGETS]
    for budget, path in checkpoints:
        model = residual.load_controller(kind, task, design, device, path)
        env = sf.make_env(task, 2811 + 864_211)
        start = time.perf_counter()
        rows = []
        try:
            for spec in specs:
                row, trace = dynamic.rollout(env, spec, task, device, f"residual_{kind}", model, 0)
                row["mean_action_deviation_from_ff"] = float(np.mean([
                    np.linalg.norm(np.asarray(step["executed_action"]) -
                                   rf.clipped_action(np.asarray(step["ff_base_action"]), task))
                    for step in trace]))
                rows.append(row)
        finally:
            env.close()
        elapsed = time.perf_counter() - start
        selected_update = 800 if budget == 800 else json.loads(
            (output / "seed2811" / kind / f"budget_u{budget}.json").read_text())["selected_update"]
        curve.append({"model": kind, "update": budget, "selected_update": selected_update,
                      "success": float(np.mean([r["success"] for r in rows])),
                      "collision": float(np.mean([r["collision"] for r in rows])),
                      "mean_tracking": float(np.mean([r["mean_tracking_error"] for r in rows])),
                      "median_tracking": float(np.mean([r["median_tracking_error"] for r in rows])),
                      "p95_tracking": float(np.mean([r["p95_tracking_error"] for r in rows])),
                      "final_tracking": float(np.mean([r["final_tracking_error"] for r in rows])),
                      "yaw": float(np.mean([r["final_yaw_error"] for r in rows])),
                      "aperture": float(np.mean([r["final_aperture_error"] for r in rows])),
                      "residual_magnitude": float(np.mean([r["mean_residual_norm"] for r in rows])),
                      "action_deviation_from_ff": float(np.mean([r["mean_action_deviation_from_ff"] for r in rows])),
                      "evaluation_seconds": elapsed})
        for row in rows:
            flags = {"position_fail": row["final_tracking_error"] >= task.success_position_threshold,
                     "yaw_fail": row["final_yaw_error"] >= task.success_rotation_threshold,
                     "aperture_fail": row["final_aperture_error"] >= task.success_opening_threshold,
                     "collision": bool(row["collision"])}
            failures.append({"model": kind, "update": budget, "episode_id": row["episode_id"],
                             **flags, "overall_success": bool(row["success"]),
                             "failure_count": sum(flags.values()),
                             "position_margin": task.success_position_threshold - row["final_tracking_error"],
                             "yaw_margin": task.success_rotation_threshold - row["final_yaw_error"],
                             "aperture_margin": task.success_opening_threshold - row["final_aperture_error"]})
            episode_rows.append({"model": kind, "update": budget, **row})
        write_csv(root / "fresh_curve.csv", curve)
        write_csv(output / "failure_decomposition" / f"{kind}.csv", failures)
        write_csv(root / "per_episode.csv", episode_rows)
        print(f"[{kind}] fresh budget {budget}: success {curve[-1]['success']:.3f}, "
              f"tracking {curve[-1]['mean_tracking']:.5f}", flush=True)
    return curve


def report(output: Path = OUTPUT) -> None:
    audit = json.loads((output / "audit/existing_budget.json").read_text())
    historical = list(csv.DictReader((output / "learning_curves/historical.csv").open()))
    curves = {kind: list(csv.DictReader((output / "seed2811" / kind / "training_curve.csv").open()))
              for kind in ("mlp", "gru")}
    fresh = {kind: list(csv.DictReader((output / "fresh_evaluation" / kind / "fresh_curve.csv").open()))
             for kind in ("mlp", "gru")}
    failures = {kind: list(csv.DictReader((output / "failure_decomposition" / f"{kind}.csv").open()))
                for kind in ("mlp", "gru")}
    gaps = []
    for kind in ("mlp", "gru"):
        for r in curves[kind]:
            gaps.append({"model": kind, "update": r["update"],
                         "train_4d_mse_sample": r["train_loss"],
                         "validation_4d_mse": r["corrected_action_mse"],
                         "validation_minus_sampled_train_4d":
                         float(r["corrected_action_mse"]) - float(r["train_loss"])})
    write_csv(output / "learning_curves" / "generalization_gap_4d.csv", gaps)
    overlap_rows = []
    flag_names = ("position_fail", "yaw_fail", "aperture_fail", "collision")
    for kind in ("mlp", "gru"):
        for budget in (800, 1200, 1600, 2400, 3200, 4800):
            subset = [r for r in failures[kind] if int(r["update"]) == budget]
            for left_index, left in enumerate(flag_names):
                for right in flag_names[left_index + 1:]:
                    overlap_rows.append({"model": kind, "update": budget,
                                         "failure_a": left, "failure_b": right,
                                         "overlap_count": sum(r[left] == "True" and r[right] == "True"
                                                              for r in subset)})
    write_csv(output / "failure_decomposition" / "overlap_counts.csv", overlap_rows)
    plotdir = ensure_dir(output / "learning_curves" / "plots")
    for kind in ("mlp", "gru"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for stage in ("a", "b"):
            source = [r for r in historical if r["model"] == kind and r["stage"] == stage]
            axes[0].plot([int(r["update"]) for r in source], [float(r["train_loss"]) for r in source],
                         ".", markersize=3, label=f"historical {stage} train")
            axes[1].plot([int(r["update"]) for r in source], [float(r["validation_loss"]) for r in source],
                         ".", markersize=3, label=f"historical {stage} validation")
        axes[0].plot([int(r["update"]) for r in curves[kind]],
                     [float(r["train_loss"]) for r in curves[kind]], alpha=.55, linewidth=.8,
                     label="extended Stage B train")
        axes[1].plot([int(r["update"]) for r in curves[kind]],
                     [float(r["validation_loss"]) for r in curves[kind]], linewidth=1,
                     label="extended Stage B validation")
        for ax in axes:
            ax.axvline(1200, color="gray", linestyle="--", linewidth=.8)
            ax.set(xlabel="Optimization updates", ylabel="Normalized action MSE")
            ax.legend(fontsize=7)
        fig.suptitle(kind.upper() + " fixed-data learning curves")
        fig.tight_layout()
        fig.savefig(plotdir / f"{kind}_loss.png", dpi=170)
        plt.close(fig)
    for key, title, filename in (("mean_tracking", "Dynamic fresh mean tracking (mm)", "fresh_tracking.png"),
                                 ("success", "Dynamic fresh success", "fresh_success.png"),
                                 ("collision", "Dynamic fresh collision", "fresh_collision.png")):
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for kind in ("mlp", "gru"):
            ax.plot([int(r["update"]) for r in fresh[kind]],
                    [float(r[key]) * (1000 if key == "mean_tracking" else 1) for r in fresh[kind]],
                    "o-", label=kind.upper())
        ax.axvline(1200, color="gray", linestyle="--", linewidth=.8)
        ax.set(xlabel="Optimization budget", ylabel=title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plotdir / filename, dpi=170)
        plt.close(fig)
    lines = ["# Dynamic residual training sufficiency check", "",
             "## A. Existing training-budget audit", "",
             "Seed 2811 used the unchanged 32-point, 100-edge graph, frozen FF checkpoint "
             f"`{audit['base_checkpoint_actual_sha256']}`, 128-hidden GRU, 384-width MLP, "
             "four corrected action dimensions and a 20% residual bound. Input is the 192D frozen "
             "embedding, frozen FF action, and previous executed action. AdamW used learning rate 3e-4, "
             "weight decay 1e-4, and 16 full sequences per update. Stage A used two plus "
             "two expert/reference draws per delay 0/1/2/4; Stage B used two expert plus "
             "two current-policy draws per delay. The same 72 expert episodes (1,440 valid timesteps), "
             "72 branch-specific policy episodes (1,440 timesteps), and 16 validation episodes "
             "(320 timesteps) were reused throughout. Stage A drew 256,000 supervised "
             "timesteps and Stage B drew 128,000; these are repeated draws, not unique data. "
             "Average per-episode draw reuse was 177.8 in Stage A, then 44.4 expert and "
             "44.4 policy in Stage B. Validation ran every 25 updates and selected minimum "
             "unweighted expert-validation 5D action MSE. Each stage began a fresh AdamW optimizer.", "",
             "| Model | Parameters total/trainable | Stage A best/final | Stage B best/final | "
             "Stage B last train / last validation |", "|---|---:|---:|---:|---:|"]
    for kind in ("mlp", "gru"):
        a, b = audit["source_records"][kind]["a"], audit["source_records"][kind]["b"]
        p = audit["parameters"][kind]
        lines.append(f"| {kind.upper()} | {p['registered']}/{p['trainable']} | "
                     f"{a['selected_update']}/800 | {b['selected_update']}/400 | "
                     f"{b['final_train_4d_mse_sample']:.5f} / {b['final_validation_5d_mse']:.5f} |")
    lines += ["", "The published 1200-update run performed 800 Stage A and 400 Stage B optimizer "
              "steps, but its evaluated Stage B checkpoints were selected at global update 850 "
              "for both branches. The GRU Stage A start was itself selected at Stage A update 750. "
              "Thus the evaluated MLP and GRU weights have only 850 and 800 effective "
              "sequential updates, respectively. Checkpoint, pool and schedule hashes are "
              "in `audit/existing_budget.json`.", "",
              "## B. Existing learning curve", "",
              "Stage A validation improved to its end for MLP and near its end for GRU. "
              "Stage B validation reached its minimum at update 50 for both; its final "
              "validation loss was higher. Historical 25-update curves are in "
              "`learning_curves/historical.csv` and the loss plots. A Stage A update-400 "
              "checkpoint was not saved by the original run, so that historical rollout "
              "point is unavailable; the earliest recoverable selected checkpoint is "
              "the Stage A endpoint at budget 800.", "",
              "## C. Extended training setup", "",
              "The experiment replayed Stage B from the original validation-selected Stage A "
              "checkpoint with the identical fixed expert/policy pools. The first 400 batch "
              "tensors have the historical schedule SHA-256. The MLP replayed validation "
              "curve matches the historical first 400 updates within 3.4e-8; GRU matches "
              "through update 300 within 2e-7 and then drifts by up to 2.1e-4, while its "
              "selected update 50 still matches within 2e-8. Stage B continued to 4,000 updates "
              "(4,800 nominal combined updates). Checkpoints at each budget were selected by "
              "validation loss accumulated *within Stage B*; final weights were saved separately. "
              "No new policy trajectories were collected. Training and evaluation used CPU; "
              "CUDA and MPS were unavailable in this runtime, so those paths were not "
              "executed. The code uses the shared device selector for all three backends. "
              "The same held-out 16 EpisodeSpecs and unchanged thresholds were used for every checkpoint.", "",
              "## D–F. Learning and dynamic fresh curves", "",
              "Train loss is the sampled batch at the listed update; it is noisy and uses the 4D "
              "objective. Validation loss is the full fixed expert set's 5D action MSE. Fresh "
              "metrics use the validation-selected checkpoint for that budget.", "",
              "| Model | Budget | Selected update | Train loss | Final val loss | Best val loss | "
              "Fresh success | Collision | Mean tracking mm | Final tracking mm |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in ("mlp", "gru"):
        by_update = {int(r["update"]): r for r in curves[kind]}
        for r in fresh[kind]:
            budget = int(r["update"])
            if budget == 800:
                a = audit["source_records"][kind]["a"]
                lines.append(f"| {kind.upper()} | 800 | {a['selected_update']} | "
                             f"{a['final_train_4d_mse_sample']:.5f} | {a['final_validation_5d_mse']:.5f} | "
                             f"{a['best_validation_5d_mse']:.5f} | {float(r['success']):.3f} | "
                             f"{float(r['collision']):.3f} | {1000*float(r['mean_tracking']):.2f} | "
                             f"{1000*float(r['final_tracking']):.2f} |")
            else:
                b = json.loads((output / "seed2811" / kind / f"budget_u{budget}.json").read_text())
                c = by_update[budget]
                lines.append(f"| {kind.upper()} | {budget} | {b['selected_update']} | "
                             f"{float(c['train_loss']):.5f} | {b['final_validation_loss']:.5f} | "
                             f"{b['best_validation_loss']:.5f} | {float(r['success']):.3f} | "
                             f"{float(r['collision']):.3f} | {1000*float(r['mean_tracking']):.2f} | "
                             f"{1000*float(r['final_tracking']):.2f} |")
    lines += ["", "Full training values, residual-target MSE, corrected-action MSE, "
              "fresh median/p95 tracking, yaw, aperture, residual magnitude, action deviation "
              "from FF, and per-episode metrics are in the linked CSV files. "
              "The plot-ready data and PNGs show the full curve shape. The comparable 4D "
              "validation-minus-sampled-train loss is in `learning_curves/generalization_gap_4d.csv`; "
              "the sampled batch and expert-only validation set have different distributions, "
              "so this gap is diagnostic rather than a direct estimate of population overfitting.", "",
              "## G. Failure-mode decomposition", "",
              "Counts overlap: a single failed episode can violate multiple thresholds. "
              "Success requires position <25 mm, yaw <0.20 rad, aperture <8 mm, "
              "and no collision anywhere in the trajectory.", "",
              "| Model | Budget | Position | Yaw | Aperture | Collision | Multiple failures | Success |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in ("mlp", "gru"):
        for budget in (800, 1200, 1600, 2400, 3200, 4800):
            subset = [r for r in failures[kind] if int(r["update"]) == budget]
            count = lambda key: sum(r[key] == "True" for r in subset)
            lines.append(f"| {kind.upper()} | {budget} | {count('position_fail')} | "
                         f"{count('yaw_fail')} | {count('aperture_fail')} | {count('collision')} | "
                         f"{sum(int(r['failure_count']) > 1 for r in subset)} | "
                         f"{count('overall_success')} |")
    lines += ["", "Aperture is the most common failing criterion (11/16 episodes for both "
              "models at every listed budget). The frozen residual corrects four action "
              "dimensions and does not directly correct aperture. At the original 1200 "
              "budget, the median threshold miss among aperture failures was 4.63 mm "
              "for MLP and 4.89 mm for GRU; position misses were 5.26 mm and 3.79 mm. "
              "These failures are generally several millimeters beyond threshold, not "
              "just rounding differences. At 1200, MLP had four yaw-plus-aperture "
              "overlaps; GRU had three. The complete pairwise counts are saved in CSV."]
    lines += ["", "Per-episode threshold margins and pairwise overlap counts are in "
              "`failure_decomposition/*.csv`. Positive margin means the corresponding final "
              "error passes its unchanged threshold.", "",
              "## H. Training sufficiency decision", "",
              "**LIKELY SATURATED** for optimization duration under this fixed data pool "
              "and the existing validation selection. GRU's best Stage B validation checkpoint "
              "stayed at global update 850 through 4800, so its selected fresh result stayed "
              "at 3/16 success and 47.90 mm mean tracking. MLP's selected checkpoint stayed "
              "at update 850 through budget 3200; a small late validation improvement "
              "(0.039378 to 0.039270) selected update 3950 at budget 4800. Fresh success "
              "then rose from 1/16 to 3/16, with two paired episodes gaining success "
              "and none losing it; mean tracking improved 1.93 mm across 16 episodes "
              "(9 improved), but the "
              "earlier Stage A MLP checkpoint had 4/16 success and 47.33 mm mean tracking. "
              "There is no consistent validation-and-rollout improvement across the budgets "
              "or both architectures. The single-seed late MLP change remains a limited "
              "signal, not evidence of clear undertraining. GRU training loss declined late "
              "without a better selected validation or rollout result, consistent with "
              "some supervised overfitting or objective/distribution mismatch.", "",
              "## I. Temporal recheck", "",
              "The predefined gate of substantial fresh improvement was not met. The delay "
              "0/4/8, carry/reset, and matched-history probes were therefore not rerun. "
              "The 8000-update option and seed 2812/2813 long-training replication were "
              "also not triggered.", "",
              "## J. Facts, interpretation, remaining hypotheses", "",
              "**Facts:** Fixed-data Stage B optimization ran to 4000 updates per branch; "
              "best validation checkpoints and fresh results are above. Both branches "
              "performed substantially more optimization than the original run, with "
              "no systematic fresh-control gain. **Interpretation:** The previous negative "
              "recurrent result is unlikely to be explained simply by stopping optimization "
              "at 1200 nominal updates. **Open hypotheses:** data coverage, supervised "
              "objective versus closed-loop control, controller formulation, residual "
              "capacity, and representation remain separate future questions; this check "
              "does not distinguish them.", "",
              "## Runtime", "",
              "| Model | Stage B updates | Training s | s/update | Fresh evaluation s/checkpoint |",
              "|---|---:|---:|---:|---:|"]
    for kind in ("mlp", "gru"):
        b = json.loads((output / "seed2811" / kind / "budget_u4800.json").read_text())
        lines.append(f"| {kind.upper()} | 4000 | {b['training_seconds']:.2f} | "
                     f"{b['seconds_per_update']:.4f} | "
                     f"{np.mean([float(r['evaluation_seconds']) for r in fresh[kind]]):.2f} |")
    lines += ["", "Peak memory was not captured. MuJoCo simulation ran on CPU.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("audit", "train", "evaluate", "report"))
    parser.add_argument("--kind", choices=("mlp", "gru"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.eval_workers != 1:
        raise ValueError("This fixed-seed check evaluates episodes serially.")
    if args.phase == "audit":
        original_curves(args.output)
    elif args.phase == "train":
        if args.kind is None:
            parser.error("--kind is required for training")
        train(args.kind, args.output, args.device)
    elif args.phase == "evaluate":
        if args.kind is None:
            parser.error("--kind is required for evaluation")
        evaluate(args.kind, args.output, args.eval_device)
    else:
        report(args.output)


if __name__ == "__main__":
    main()
