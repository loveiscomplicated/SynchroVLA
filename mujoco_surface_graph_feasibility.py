from __future__ import annotations

import argparse

from vla_gnn_recurrent.training.surface_graph_feasibility import (
    SurfaceFeasibilityConfig,
    run_surface_feasibility_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Surface-aware Pre-grasp Alignment feasibility.")
    parser.add_argument("--output-dir", default="artifacts/surface_graph_feasibility")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2811, 2812, 2813])
    parser.add_argument("--train-episodes", type=int, default=72)
    parser.add_argument("--validation-episodes", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--eval-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--latency-warmup", type=int, default=200)
    parser.add_argument("--latency-iterations", type=int, default=2000)
    parser.add_argument("--plot-cases", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SurfaceFeasibilityConfig(
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        eval_device=args.eval_device,
        latency_warmup=args.latency_warmup,
        latency_iterations=args.latency_iterations,
        plot_cases=args.plot_cases,
    )
    run_surface_feasibility_experiment(config)


if __name__ == "__main__":
    main()
