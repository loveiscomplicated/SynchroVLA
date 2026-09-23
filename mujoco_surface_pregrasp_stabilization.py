from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_gnn_recurrent.training.surface_graph_feasibility import SurfaceFeasibilityConfig
from vla_gnn_recurrent.training.surface_pregrasp_stabilization import (
    run_floor_postmortem_and_surface_set_resolution,
    run_overfit_stage,
    run_recovery_surface_set_trial,
    run_full_recovery_experiment,
    _write_final_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose and stabilize the existing feed-forward surface pre-grasp policies."
    )
    parser.add_argument("--stage", choices=("overfit", "recovery", "full", "postmortem"), default="overfit")
    parser.add_argument("--output-dir", default="artifacts/surface_pregrasp_stabilization")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2811, 2812, 2813])
    parser.add_argument("--overfit-episodes", type=int, default=24)
    parser.add_argument("--overfit-epochs", type=int, default=2000)
    parser.add_argument("--train-episodes", type=int, default=72)
    parser.add_argument("--validation-episodes", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    # Rollouts are intentionally sequential; each evaluation call owns one MuJoCo environment.
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--latency-warmup", type=int, default=200)
    parser.add_argument("--latency-iterations", type=int, default=2000)
    args = parser.parse_args()
    if args.eval_workers != 1:
        parser.error("This experiment currently runs sequentially; use --eval-workers 1.")
    return args


def main() -> None:
    args = parse_args()
    config = SurfaceFeasibilityConfig(
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        eval_episodes=args.eval_episodes,
        device=args.device,
        eval_device=args.eval_device,
        latency_warmup=args.latency_warmup,
        latency_iterations=args.latency_iterations,
    )
    if args.stage == "overfit":
        run_overfit_stage(config, episodes=args.overfit_episodes, epochs=args.overfit_epochs,
                          seed=args.seeds[0], device_preference=args.device,
                          eval_device_preference=args.eval_device)
    elif args.stage == "recovery":
        result = run_recovery_surface_set_trial(
            config, tuple(args.seeds), args.train_episodes, args.validation_episodes,
            args.eval_episodes, args.updates, args.device, args.eval_device,
        )
        if not result["representation_reevaluation_gate_passed"]:
            run_floor_postmortem_and_surface_set_resolution(
                config, tuple(args.seeds), args.eval_episodes, args.eval_device,
            )
            _write_final_summary(Path(args.output_dir), result, None, None, None)
    elif args.stage == "full":
        result = run_full_recovery_experiment(
            config, tuple(args.seeds), args.train_episodes, args.validation_episodes,
            args.eval_episodes, args.updates, args.device, args.eval_device,
        )
        trial = result.get("recovery_trial", result) if isinstance(result, dict) else result
        if isinstance(trial, dict) and not trial.get("representation_reevaluation_gate_passed", True):
            run_floor_postmortem_and_surface_set_resolution(
                config, tuple(args.seeds), args.eval_episodes, args.eval_device,
            )
            _write_final_summary(Path(args.output_dir), trial, None, None, None)
    else:
        result = run_floor_postmortem_and_surface_set_resolution(
            config, tuple(args.seeds), args.eval_episodes, args.eval_device,
        )
        trial_path = Path(args.output_dir) / "recovery_training" / "base_vs_recovery_summary.json"
        if trial_path.exists():
            _write_final_summary(Path(args.output_dir), json.loads(trial_path.read_text()), None, None, None)


if __name__ == "__main__":
    main()
