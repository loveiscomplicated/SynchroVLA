from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.trainer import TrainConfig, train_controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VLA-GNN-Recurrent toy reaching controllers.")
    parser.add_argument("--model", choices=["recurrent", "feedforward", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=28)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--rollout-policy", choices=["teacher", "model"], default="teacher")
    parser.add_argument("--task", choices=["static", "moving"], default="static")
    parser.add_argument("--observation-intervals", default="1")
    parser.add_argument("--target-velocity-feature", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--action-prior", choices=["none", "geometric"], default="none")
    parser.add_argument("--target-velocity-min", type=float, default=0.006)
    parser.add_argument("--target-velocity-max", type=float, default=0.018)
    parser.add_argument("--bptt-steps", type=int, default=4)
    parser.add_argument("--sequence-repeats", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_kinds = ["feedforward", "recurrent"] if args.model == "both" else [args.model]
    summaries = []
    for model_kind in model_kinds:
        config = TrainConfig(
            model_kind=model_kind,
            episodes=args.episodes,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
            log_every=args.log_every,
            rollout_policy=args.rollout_policy,
            task=args.task,
            observation_intervals=_parse_intervals(args.observation_intervals),
            target_velocity_feature=args.target_velocity_feature,
            action_prior=args.action_prior,
            target_velocity_min=args.target_velocity_min,
            target_velocity_max=args.target_velocity_max,
            bptt_steps=args.bptt_steps,
            sequence_repeats=args.sequence_repeats,
        )
        _, summary = train_controller(config)
        summaries.append(summary)

    print(json.dumps({"training_summaries": summaries}, indent=2))


def _parse_intervals(raw: str) -> tuple[int, ...]:
    intervals = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not intervals:
        raise ValueError("At least one observation interval is required.")
    if any(interval < 1 for interval in intervals):
        raise ValueError("Observation intervals must be >= 1.")
    return intervals


if __name__ == "__main__":
    main()
