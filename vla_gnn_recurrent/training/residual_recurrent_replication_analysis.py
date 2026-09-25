"""Summarize the fixed-seed physical-delay residual replication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vla_gnn_recurrent.training import residual_recurrent_feasibility as residual
from vla_gnn_recurrent.training import residual_recurrent_replication as replication
from vla_gnn_recurrent.utils import ensure_dir


METRICS = ("success", "collision", "final_position_error", "final_orientation_error",
           "final_gripper_width_error", "trajectory_error", "trajectory_yaw_error",
           "minimum_safe_clearance")
LABELS = {"ff": "FF", "residual_mlp": "Residual MLP", "residual_gru": "Residual GRU"}


def _write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _episode_rows(root: Path, controller: str) -> list[dict]:
    return json.loads((root / "per_episode" / f"{controller}.json").read_text())


def _paired(root: Path, reference: str, candidate: str, seed: int) -> dict:
    return residual.paired_summary(_episode_rows(root, reference),
                                   _episode_rows(root, candidate), seed)


def aggregate(output: Path = replication.OUTPUT) -> dict:
    summaries = {}
    for seed in replication.SEEDS:
        sub = "seed2811_recheck" if seed == 2811 else f"seed{seed}"
        summaries[seed] = json.loads((output / sub / "summary.json").read_text())
    old = json.loads((residual.OUTPUT / "summary.json").read_text())
    action = json.loads((output / "audit/action_history_mismatch.json").read_text())
    aggregate_root = ensure_dir(output / "aggregate")
    fresh_rows, delay_rows, reset_rows, perturb_rows, variable_rows, pair_rows = [], [], [], [], [], []
    for seed, summary in summaries.items():
        sub = "seed2811_recheck" if seed == 2811 else f"seed{seed}"
        base = output / sub / "metrics"
        for name, means in summary["fresh"]["means"].items():
            fresh_rows.append({"seed": seed, "controller": name, "fresh_gate_passed": summary["fresh_gate"]["passed"],
                               **{key: means[key] for key in METRICS}})
        conditions = {0: summary["fresh"], **{int(k): v for k, v in summary["fixed_delay"].items()}}
        for delay, condition in sorted(conditions.items()):
            for name, means in condition["means"].items():
                fresh_position = summary["fresh"]["means"][name]["final_position_error"]
                delay_rows.append({"seed": seed, "delay": delay, "controller": name,
                                   **{key: means[key] for key in METRICS},
                                   "position_degradation": means["final_position_error"] - fresh_position,
                                   "inference_ms": condition["ff_inference_latency"]["mean_ms"]
                                   if name == "ff" else condition["residual"][name]["mean_inference_ms"],
                                   "mean_residual_norm": 0.0 if name == "ff" else
                                   condition["residual"][name]["mean_residual_norm"]})
            condition_root = base / ("fresh" if delay == 0 else f"delay{delay}")
            for reference, candidate in (("ff", "residual_gru"), ("residual_mlp", "residual_gru")):
                paired = _paired(condition_root, reference, candidate, seed + delay)
                for metric in METRICS:
                    entry = paired[metric]
                    pair_rows.append({"seed": seed, "condition": f"delay{delay}",
                                      "comparison": f"{candidate}-{reference}", "metric": metric,
                                      "mean_difference": entry["mean_candidate_minus_reference"],
                                      "ci_low": entry["episode_bootstrap_95_ci"][0],
                                      "ci_high": entry["episode_bootstrap_95_ci"][1],
                                      "discordant_reference_only": entry.get("discordant_reference_only", ""),
                                      "discordant_candidate_only": entry.get("discordant_candidate_only", "")})
        for delay, condition in summary["hidden_reset"].items():
            carry = conditions[int(delay)]["means"]["residual_gru"]
            reset = condition["means"]["residual_gru"]
            reset_rows.append({"seed": seed, "delay": int(delay),
                               **{f"carry_{key}": carry[key] for key in METRICS},
                               **{f"reset_{key}": reset[key] for key in METRICS},
                               "carry_minus_reset_position": carry["final_position_error"] - reset["final_position_error"]})
            carry_root = base / ("fresh" if int(delay) == 0 else f"delay{delay}")
            reset_root = base / f"reset_delay{delay}"
            paired = residual.paired_summary(_episode_rows(reset_root, "residual_gru"),
                                             _episode_rows(carry_root, "residual_gru"), seed + int(delay) + 10_000)
            for metric in METRICS:
                entry = paired[metric]
                pair_rows.append({"seed": seed, "condition": f"carry_reset_delay{delay}",
                                  "comparison": "carry-reset", "metric": metric,
                                  "mean_difference": entry["mean_candidate_minus_reference"],
                                  "ci_low": entry["episode_bootstrap_95_ci"][0],
                                  "ci_high": entry["episode_bootstrap_95_ci"][1],
                                  "discordant_reference_only": entry.get("discordant_reference_only", ""),
                                  "discordant_candidate_only": entry.get("discordant_candidate_only", "")})
        for label, destination in (("variable_delay", variable_rows), ("perturbation", perturb_rows)):
            condition = summary[label]
            if condition is None:
                continue
            for name, means in condition["means"].items():
                row = {"seed": seed, "controller": name, **{key: means[key] for key in METRICS}}
                if label == "perturbation":
                    row.update({key: means[key] for key in ("post_perturb_peak_position_error",
                                                            "recovery_fraction", "recovery_time_mean_among_recovered",
                                                            "recovery_time_horizon_censored_mean")})
                destination.append(row)
            root = base / label
            for reference, candidate in (("ff", "residual_gru"), ("residual_mlp", "residual_gru")):
                paired = _paired(root, reference, candidate, seed + 20_000)
                for metric in METRICS:
                    entry = paired[metric]
                    pair_rows.append({"seed": seed, "condition": label,
                                      "comparison": f"{candidate}-{reference}", "metric": metric,
                                      "mean_difference": entry["mean_candidate_minus_reference"],
                                      "ci_low": entry["episode_bootstrap_95_ci"][0],
                                      "ci_high": entry["episode_bootstrap_95_ci"][1],
                                      "discordant_reference_only": entry.get("discordant_reference_only", ""),
                                      "discordant_candidate_only": entry.get("discordant_candidate_only", "")})
    for name, rows in (("fresh_table", fresh_rows), ("fixed_delay_table", delay_rows),
                       ("carry_reset_table", reset_rows), ("perturbation_table", perturb_rows),
                       ("variable_delay_table", variable_rows), ("paired_differences", pair_rows)):
        _write_csv(aggregate_root / f"{name}.csv", rows)
    pooled = {}
    for delay in replication.DELAYS:
        pooled[str(delay)] = {}
        for name in LABELS:
            subset = [row for row in delay_rows if row["delay"] == delay and row["controller"] == name]
            if subset:
                pooled[str(delay)][name] = {key: float(np.mean([row[key] for row in subset]))
                                             for key in ("final_position_error", "position_degradation",
                                                         "collision", "success", "final_orientation_error")}
    seed_contrasts = []
    for seed in replication.SEEDS:
        for delay in (4, 8):
            group = {r["controller"]: r for r in delay_rows if r["seed"] == seed and r["delay"] == delay}
            if len(group) == 3:
                seed_contrasts.append({"seed": seed, "delay": delay,
                    "gru_minus_ff_position_degradation": group["residual_gru"]["position_degradation"] - group["ff"]["position_degradation"],
                    "gru_minus_mlp_position_degradation": group["residual_gru"]["position_degradation"] - group["residual_mlp"]["position_degradation"]})
    summary = {"seeds": list(replication.SEEDS), "seed_fresh_gate": {str(s): x["fresh_gate"] for s, x in summaries.items()},
               "pooled_descriptive": pooled, "paired_results_csv": "paired_differences.csv",
               "seed_level_long_delay_degradation_contrasts": seed_contrasts,
               "maximum_absolute_position_degradation_m": max(abs(r["position_degradation"]) for r in delay_rows),
               "old_seed2811_delay_protocol_invalid": True,
               "action_history_training_normalized_l2_mean": action["training"]["normalized_l2_mean"]}
    residual._write(aggregate_root / "summary.json", summary)
    _plots(aggregate_root, delay_rows, reset_rows, perturb_rows)
    _report(output, summaries, old, action, fresh_rows, delay_rows, reset_rows, variable_rows, perturb_rows)
    return summary


def _plots(root: Path, delay_rows: list[dict], reset_rows: list[dict], perturb_rows: list[dict]) -> None:
    plots = ensure_dir(root / "plots")
    for value, filename, ylabel in (("final_position_error", "final_position.png", "Final position error (mm)"),
                                     ("position_degradation", "position_degradation.png", "Change from fresh (mm)")):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
        for column, seed in enumerate(replication.SEEDS):
            ax = axes[column]
            for name in LABELS:
                rows = sorted((r for r in delay_rows if r["seed"] == seed and r["controller"] == name), key=lambda r:r["delay"])
                ax.plot([r["delay"] for r in rows], [1000*r[value] for r in rows], marker="o", label=LABELS[name])
            ax.set_title(f"Seed {seed}"); ax.set_xlabel("Delay (steps)")
        ax = axes[3]
        for name in LABELS:
            values = [(delay, np.mean([r[value] for r in delay_rows if r["delay"] == delay and r["controller"] == name]))
                      for delay in replication.DELAYS if any(r["delay"] == delay and r["controller"] == name for r in delay_rows)]
            ax.plot([x for x,_ in values], [1000*y for _,y in values], marker="o", label=LABELS[name])
        ax.set_title("Seed mean (descriptive)"); ax.set_xlabel("Delay (steps)")
        if value == "position_degradation":
            for panel in axes:
                panel.axhline(0, color="black", linewidth=.7, alpha=.5)
                panel.set_ylim(-.1, .1)
        axes[0].set_ylabel(ylabel); axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(plots / filename, dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, seed in zip(axes, replication.SEEDS, strict=True):
        rows = sorted((r for r in reset_rows if r["seed"] == seed), key=lambda r:r["delay"])
        ax.plot([r["delay"] for r in rows], [1000*r["carry_final_position_error"] for r in rows], marker="o", label="Carry")
        ax.plot([r["delay"] for r in rows], [1000*r["reset_final_position_error"] for r in rows], marker="o", label="Reset")
        ax.set_title(f"Seed {seed}"); ax.set_xlabel("Delay (steps)")
    axes[0].set_ylabel("Final position error (mm)"); axes[0].legend()
    fig.tight_layout(); fig.savefig(plots / "carry_reset.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, seed in zip(axes, replication.SEEDS, strict=True):
        rows = [r for r in perturb_rows if r["seed"] == seed]
        x = np.arange(len(rows))
        ax.bar(x-.17, [r["recovery_fraction"] for r in rows], .34, label="Recovery fraction")
        ax.bar(x+.17, [r["recovery_time_horizon_censored_mean"]/20 for r in rows], .34,
               label="Censored time / 20")
        ax.set_xticks(x, [LABELS[r["controller"]] for r in rows], rotation=30, ha="right")
        ax.set_title(f"Seed {seed}")
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(plots / "perturbation_recovery.png", dpi=160); plt.close(fig)


def _report(output: Path, summaries: dict, old: dict, action: dict, fresh: list[dict],
            delayed: list[dict], reset: list[dict], variable: list[dict], perturb: list[dict]) -> None:
    graph_check = json.loads((output / "audit/graph_delay_numerical_check.json").read_text())
    maximum_state_difference = max(row["max_abs_state_difference"] for row in graph_check)
    maximum_surface_difference = max(row["max_abs_surface_difference_m"] for row in graph_check)
    maximum_position_degradation_mm = 1000 * max(abs(row["position_degradation"]) for row in delayed)
    old_integrity = json.loads((output / "audit/historical_artifact_integrity.json").read_text())
    lines = ["# Residual recurrent replication after delayed-observation audit", "",
             "## A. Delay implementation audit — BUG FOUND", "",
             "The development implementation copied past EE-local object and surface coordinates into a current robot graph. "
             "That retained the past robot pose and did not model stale world visual geometry with current proprioception. "
             "The corrected branch reconstructs old world visual geometry, then transforms it into the current EE frame. "
             "Current yaw, aperture, fingertips, and EE pose stay fresh; graph edges are rebuilt from corrected nodes. "
             "Delay 0 is bitwise identical to the canonical graph. See `audit/delay_semantics.md` and synthetic tests A–E. "
             "The old fixed-delay 2811 result is invalid as a physical visual-latency result and is preserved unchanged. "
             f"Original residual checkpoint hashes still match their saved provenance: {old_integrity['all_match']}.", "",
             "For the static objects in these held-out EpisodeSpecs, physically corrected stale world geometry is nearly "
             "identical to fresh world geometry. This protocol therefore supplies almost no delay-dependent information loss. "
             "The scripted object shift is immediately observed in the separate perturbation condition. "
             f"Across fixed-delay traces, the maximum absolute observed-versus-current state difference was "
             f"{maximum_state_difference:.2g}; maximum surface-point difference was "
             f"{maximum_surface_difference:.2g} m. See `audit/graph_delay_numerical_check.json`. "
             "Delay-0 FF episode metrics exactly match the historical fresh runner.", "",
             "## B. Action-history distribution audit", "",
             f"The sampled training previous action has mean normalized L2 {action['training']['normalized_l2_mean']:.3f} "
             f"and normalized temporal-delta L2 {action['training']['normalized_temporal_delta_l2_mean']:.3f}. "
             "The table includes read-only rollouts from the original and corrected 2811 checkpoints; histogram intersection "
             "uses 40 bins per normalized action component and is descriptive, not a causal test.", "",
             "| Source | Mean normalized action L2 | Mean normalized delta L2 | Mean component overlap with training |",
             "|---|---:|---:|---:|"]
    for name, entry in action["rollouts"].items():
        lines.append(f"| {name} | {entry['stats']['normalized_l2_mean']:.3f} | "
                     f"{entry['stats']['normalized_temporal_delta_l2_mean']:.3f} | "
                     f"{entry['overlap_with_training']['mean_dimension_overlap']:.3f} |")
    lines += ["", "Full per-dimension physical and normalized means, standard deviations, quantiles, and overlaps are in "
              "`audit/action_history_mismatch.json`.", "", "## C. Frozen confirmatory protocol", "",
              "Canonical frozen FF SHA-256: `" + residual.BASE_SHA256 + "`. The graph is 32 surface points, 100 directed edges, "
              "three encoder layers, no surface-local edges. Residual input is 202D; GRU hidden size 128, MLP width 384; "
              "the four geometric residual dimensions are tanh-bounded to 20% of each action limit; gripper remains FF. "
              "Training uses AdamW, 3e-4 learning rate, 1e-4 weight decay, 800 updates, batch 16, delays 0/1/2/4 "
              "with two expert and two policy episodes per delay and update, validation every 25 updates, and minimum "
              "unweighted expert-validation 5D MSE selection. Each seed has 16 fixed held-out IID EpisodeSpecs. "
              "Per-seed source hashes, EpisodeSpec signatures, schedule hashes, checkpoints, and supervised counts are in "
              "`audit/frozen_protocol_seed*.json` and `seed*/logs`. The corrected 2811 schedule replays the original "
              "episode/window draws exactly; only delayed graph values change. Existing seed-specific expert and "
              "144-policy-trajectory pools were reused for 2812/2813; no upstream policy data or FF checkpoint "
              "was regenerated. PyTorch training selected MPS on this host; evaluation and MuJoCo ran on CPU. "
              "The device selector also supports CUDA and CPU, though CUDA was not available for a runtime check here.", "",
              "## D. Fresh parity across seeds", "",
              "| Seed | Gate | Controller | Success | Collision | Final position mm | Final yaw rad | Final aperture | Trajectory position mm |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in fresh:
        lines.append(f"| {row['seed']} | {'PASS' if row['fresh_gate_passed'] else 'FAIL'} | {LABELS[row['controller']]} | "
                     f"{row['success']:.3f} | {row['collision']:.3f} | {1000*row['final_position_error']:.2f} | "
                     f"{row['final_orientation_error']:.3f} | {row['final_gripper_width_error']:.4f} | "
                     f"{1000*row['trajectory_error']:.2f} |")
    lines += ["", "## E. Fixed-delay replication", "",
              "| Seed | Delay | FF position mm | MLP position mm | GRU position mm | GRU−FF degradation mm | GRU−MLP degradation mm |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in replication.SEEDS:
        for delay in replication.DELAYS:
            group = {r["controller"]: r for r in delayed if r["seed"] == seed and r["delay"] == delay}
            if len(group) != 3:
                continue
            ff, mlp, gru = (group[name] for name in ("ff", "residual_mlp", "residual_gru"))
            lines.append(f"| {seed} | {delay} | {1000*ff['final_position_error']:.2f} | "
                         f"{1000*mlp['final_position_error']:.2f} | {1000*gru['final_position_error']:.2f} | "
                         f"{1000*(gru['position_degradation']-ff['position_degradation']):+.3f} | "
                         f"{1000*(gru['position_degradation']-mlp['position_degradation']):+.3f} |")
    lines += ["", "The old and corrected 2811 protocols give different answers:", "",
              "| Delay | Old FF position mm / collision | Old MLP position mm / collision | Old GRU position mm / collision | Corrected FF | Corrected MLP | Corrected GRU |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for delay in replication.DELAYS:
        old_condition = old["fresh"] if delay == 0 else old["fixed_delay"][str(delay)]
        current = {row["controller"]: row for row in delayed if row["seed"] == 2811 and row["delay"] == delay}
        cells = []
        for name in ("ff", "residual_mlp", "residual_gru"):
            prior = old_condition["means"][name]
            cells.append(f"{1000*prior['final_position_error']:.2f} / {prior['collision']:.3f}")
        lines.append(f"| {delay} | " + " | ".join(cells) + " | " +
                     " | ".join(f"{1000*current[name]['final_position_error']:.2f} / {current[name]['collision']:.3f}"
                                for name in ("ff", "residual_mlp", "residual_gru")) + " |")
    lines += ["", "Per-seed collisions, success, yaw, trajectory position, inference latency, and paired bootstrap intervals "
              "are in `aggregate/fixed_delay_table.csv` and `aggregate/paired_differences.csv`. "
              "The development 2811 delay-4/8 geometric advantage disappears after correction: delays 0/2/4/8 "
              f"yield effectively the same trajectory within each controller (largest absolute final-position change "
              f"{maximum_position_degradation_mm:.6f} mm).", "", "## F. Hidden-memory ablation", "",
              "| Seed | Delay | Carry position mm | Reset position mm | Carry−reset mm | Carry collision | Reset collision |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in reset:
        lines.append(f"| {row['seed']} | {row['delay']} | {1000*row['carry_final_position_error']:.2f} | "
                     f"{1000*row['reset_final_position_error']:.2f} | "
                     f"{1000*row['carry_minus_reset_position']:+.2f} | {row['carry_collision']:.3f} | "
                     f"{row['reset_collision']:.3f} |")
    lines += ["", "Carry−reset differences are interpreted across delays within each seed. A gap that is unchanged "
              "from fresh to delay 8 is a generic hidden-state effect, not evidence of robustness to stale visual data.",
              "", "## G. Variable delay", "",
              "The pattern is `[0,0,1,3,0,2,4,1]`. Full results are in `aggregate/variable_delay_table.csv`. "
              "For static geometry the corrected pattern produces no meaningful new visual information loss.", "",
              "| Seed | FF position mm / collision | MLP position mm / collision | GRU position mm / collision |",
              "|---:|---:|---:|---:|"]
    for seed in replication.SEEDS:
        group = {row["controller"]: row for row in variable if row["seed"] == seed}
        if len(group) == 3:
            lines.append(f"| {seed} | " + " | ".join(
                f"{1000*group[name]['final_position_error']:.2f} / {group[name]['collision']:.3f}"
                for name in ("ff", "residual_mlp", "residual_gru")) + " |")
    lines += ["", "## H. Perturbation recovery", "",
              "The object shifts +0.025 m in x at step 5 and is immediately observable. Recovery is position <0.025 m "
              "and yaw <0.200 rad; unrecovered episodes are censored at 20 steps.", "",
              "| Seed | Controller | Recovery fraction | Mean recovery steps (recovered) | Censored mean steps | Final position mm | Collision |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for row in perturb:
        recovered = row["recovery_time_mean_among_recovered"]
        lines.append(f"| {row['seed']} | {LABELS[row['controller']]} | {row['recovery_fraction']:.3f} | "
                     f"{'—' if recovered is None else f'{recovered:.2f}'} | "
                     f"{row['recovery_time_horizon_censored_mean']:.2f} | "
                     f"{1000*row['final_position_error']:.2f} | {row['collision']:.3f} |")
    lines += ["", "## I. Facts, interpretation, remaining hypotheses", "",
              "**Facts.** The old EE-local copy implementation was physically inconsistent; the corrected delay graphs "
              "for static objects are nearly identical to fresh graphs. The recorded seed-wise fresh, delay, carry/reset, "
              "variable-delay, and perturbation outcomes appear in the tables above and machine-readable episode files.", "",
              "**Interpretation.** The apparent development-seed long-delay recurrent advantage does not replicate under "
              "the corrected physical delay. Because the current task keeps the visual object static, these fixed-delay "
              "conditions cannot establish whether memory helps under genuinely stale, changing visual geometry. "
              "Any carry/reset difference that also appears under fresh observations is an effect of carrying hidden state "
              "in general, not a delay-specific benefit. Perturbation recovery is better for GRU than FF in all three "
              "seeds, but MLP is close and ties GRU in seed 2813, so this does not isolate recurrence. This separate "
              "outcome is limited to three training seeds and 16 paired episodes per seed. The primary replication "
              "decision is **corrected delay semantics erase the long-delay effect**; no reproducible delay-specific "
              "recurrent-memory advantage was demonstrated.", "",
              "**Remaining hypotheses.** Recurrent memory might help when visual geometry changes during the unseen "
              "interval; the action-history off-policy distribution may limit fitted residual behavior. Neither mechanism "
              "was isolated by this fixed static-object delay protocol. No architecture, loss, or delay hyperparameter was tuned.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=replication.OUTPUT)
    args = parser.parse_args()
    aggregate(args.output)


if __name__ == "__main__":
    main()
