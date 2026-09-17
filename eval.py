from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_gnn_recurrent.evaluation.evaluate import EvalConfig, evaluate_controller
from vla_gnn_recurrent.evaluation.visualize import (
    plot_action_redirection,
    plot_condition_aggregate,
    plot_error_comparison,
    plot_perturbation_recovery,
    plot_stale_aggregate,
    plot_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VLA-GNN-Recurrent toy reaching controllers.")
    parser.add_argument("--model", choices=["recurrent", "feedforward", "both"], default="both")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=36)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--perturb-step", type=int, default=10)
    parser.add_argument("--output-dir", default="artifacts/eval")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--task", choices=["static", "moving"], default="static")
    parser.add_argument("--observation-intervals", default="1")
    parser.add_argument("--target-velocity-feature", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-velocity-min", type=float, default=0.006)
    parser.add_argument("--target-velocity-max", type=float, default=0.018)
    parser.add_argument("--recurrent-modes", default="normal")
    parser.add_argument("--correction-checkpoint", default=None)
    parser.add_argument("--fixed-decay-alphas", default="0.5")
    parser.add_argument("--perturbation-level", choices=["default", "small", "medium", "large"], default="default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = _build_jobs(args.model, args.recurrent_modes, args.fixed_decay_alphas)
    intervals = _parse_intervals(args.observation_intervals)
    payloads = []

    for interval in intervals:
        interval_payloads = []
        for model_kind, recurrent_mode, condition, fixed_decay_alpha in jobs:
            checkpoint_path = Path(args.checkpoint_dir) / f"{model_kind}.pt"
            config = EvalConfig(
                model_kind=model_kind,
                checkpoint_path=str(checkpoint_path),
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                device=args.device,
                perturb=args.perturb,
                perturb_step=args.perturb_step,
                output_dir=args.output_dir,
                task=args.task,
                observation_interval=interval,
                target_velocity_feature=args.target_velocity_feature,
                target_velocity_min=args.target_velocity_min,
                target_velocity_max=args.target_velocity_max,
                recurrent_mode=recurrent_mode,
                correction_checkpoint_path=args.correction_checkpoint,
                fixed_decay_alpha=fixed_decay_alpha,
                perturbation_level=args.perturbation_level,
            )
            payload = evaluate_controller(config)
            payload["condition"] = condition
            if args.plot:
                mode_name = _mode_name(args.task, interval, args.perturb)
                plot_path = Path(args.output_dir) / f"{condition}_{mode_name}_trajectory.png"
                saved = plot_trajectory(payload["episodes"][0], plot_path, title=f"{condition} {mode_name}")
                payload["plot_path"] = str(saved)
            payloads.append(payload)
            interval_payloads.append(payload)

        if args.plot and len(interval_payloads) == 2:
            by_model = {payload["config"]["model_kind"]: payload for payload in interval_payloads}
            if "feedforward" in by_model and "recurrent" in by_model:
                mode_name = _mode_name(args.task, interval, args.perturb)
                comparison_path = Path(args.output_dir) / f"error_comparison_{mode_name}.png"
                saved = plot_error_comparison(
                    by_model["feedforward"]["episodes"][0],
                    by_model["recurrent"]["episodes"][0],
                    comparison_path,
                    title=f"error comparison {mode_name}",
                )
                for payload in interval_payloads:
                    payload["error_comparison_path"] = str(saved)

        if args.plot and args.perturb:
            by_condition = {payload["condition"]: payload["episodes"] for payload in interval_payloads}
            mode_name = _mode_name(args.task, interval, args.perturb)
            recovery_path = Path(args.output_dir) / f"perturb_recovery_{mode_name}.png"
            redirection_path = Path(args.output_dir) / f"action_redirection_{mode_name}.png"
            saved_recovery = plot_perturbation_recovery(by_condition, recovery_path)
            saved_redirection = plot_action_redirection(by_condition, redirection_path)
            for payload in interval_payloads:
                payload["perturb_recovery_plot"] = str(saved_recovery)
                payload["action_redirection_plot"] = str(saved_redirection)

    if args.plot and len(payloads) > 1:
        summaries = [
            {
                "model_kind": payload["config"]["model_kind"],
                "condition": payload["condition"],
                "observation_interval": payload["config"]["observation_interval"],
                "summary": payload["summary"],
            }
            for payload in payloads
        ]
        if len({item["condition"] for item in summaries}) > 2:
            tracking_plot = plot_condition_aggregate(
                summaries,
                Path(args.output_dir) / "aggregate_tracking_error.png",
                metric="mean_tracking_error",
            )
            success_plot = plot_condition_aggregate(
                summaries,
                Path(args.output_dir) / "aggregate_success_rate.png",
                metric="success_rate",
            )
        else:
            tracking_plot = plot_stale_aggregate(
                summaries,
                Path(args.output_dir) / "aggregate_tracking_error.png",
                metric="mean_tracking_error",
            )
            success_plot = plot_stale_aggregate(
                summaries,
                Path(args.output_dir) / "aggregate_success_rate.png",
                metric="success_rate",
            )
        for payload in payloads:
            payload["aggregate_tracking_plot"] = str(tracking_plot)
            payload["aggregate_success_plot"] = str(success_plot)

    concise = [
        {
            "model_kind": payload["config"]["model_kind"],
            "condition": payload["condition"],
            "observation_interval": payload["config"]["observation_interval"],
            "summary": payload["summary"],
            "output_path": payload["output_path"],
            "plot_path": payload.get("plot_path"),
            "error_comparison_path": payload.get("error_comparison_path"),
            "perturb_recovery_plot": payload.get("perturb_recovery_plot"),
            "action_redirection_plot": payload.get("action_redirection_plot"),
            "aggregate_tracking_plot": payload.get("aggregate_tracking_plot"),
            "aggregate_success_plot": payload.get("aggregate_success_plot"),
        }
        for payload in payloads
    ]
    print(json.dumps({"evaluation_summaries": concise}, indent=2))


def _parse_intervals(raw: str) -> tuple[int, ...]:
    intervals = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not intervals:
        raise ValueError("At least one observation interval is required.")
    if any(interval < 1 for interval in intervals):
        raise ValueError("Observation intervals must be >= 1.")
    return intervals


def _parse_recurrent_modes(raw: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in raw.split(",") if part.strip())
    aliases = {"zero_reset": "event_reset"}
    modes = tuple(aliases.get(mode, mode) for mode in modes)
    allowed = {"normal", "step_reset", "event_reset", "fixed_decay", "learned_correction"}
    if not modes:
        raise ValueError("At least one recurrent mode is required.")
    invalid = [mode for mode in modes if mode not in allowed]
    if invalid:
        raise ValueError(f"Unknown recurrent modes: {invalid}")
    return modes


def _parse_alphas(raw: str) -> tuple[float, ...]:
    alphas = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not alphas:
        raise ValueError("At least one fixed-decay alpha is required.")
    invalid = [alpha for alpha in alphas if alpha < 0.0 or alpha > 1.0]
    if invalid:
        raise ValueError(f"fixed-decay alphas must be in [0, 1], got {invalid}")
    return alphas


def _build_jobs(model: str, recurrent_modes_raw: str, fixed_decay_alphas_raw: str) -> list[tuple[str, str, str, float]]:
    modes = _parse_recurrent_modes(recurrent_modes_raw)
    alphas = _parse_alphas(fixed_decay_alphas_raw)
    jobs: list[tuple[str, str, str, float]] = []
    if model in {"feedforward", "both"}:
        jobs.append(("feedforward", "normal", "feedforward", 1.0))
    if model in {"recurrent", "both"}:
        for mode in modes:
            if mode == "fixed_decay":
                for alpha in alphas:
                    jobs.append(("recurrent", mode, f"gru_fixed_decay_{alpha:g}", alpha))
            else:
                jobs.append(("recurrent", mode, f"gru_{mode}", 1.0))
    return jobs


def _mode_name(task: str, observation_interval: int, perturb: bool) -> str:
    pieces = [task, f"N{observation_interval}"]
    if perturb:
        pieces.append("perturb")
    return "_".join(pieces)


if __name__ == "__main__":
    main()
