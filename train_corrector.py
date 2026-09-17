from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.corrector_trainer import CorrectorTrainConfig, train_state_corrector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train event-time hidden-state corrector for frozen GRU controller.")
    parser.add_argument("--base-checkpoint", default="artifacts/stale_final/checkpoints/recurrent.pt")
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--max-steps", type=int, default=36)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="artifacts/state_corrector/checkpoints")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--observation-intervals", default="1,4")
    parser.add_argument("--perturbation-levels", default="small,medium,large")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--target-velocity-min", type=float, default=0.006)
    parser.add_argument("--target-velocity-max", type=float, default=0.018)
    parser.add_argument("--corrector-hidden-dim", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CorrectorTrainConfig(
        base_checkpoint=args.base_checkpoint,
        episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        log_every=args.log_every,
        observation_intervals=_parse_intervals(args.observation_intervals),
        perturbation_levels=_parse_levels(args.perturbation_levels),
        horizon=args.horizon,
        target_velocity_min=args.target_velocity_min,
        target_velocity_max=args.target_velocity_max,
        corrector_hidden_dim=args.corrector_hidden_dim,
    )
    _, summary = train_state_corrector(config)
    print(json.dumps({"corrector_training_summary": summary}, indent=2))


def _parse_intervals(raw: str) -> tuple[int, ...]:
    intervals = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not intervals:
        raise ValueError("At least one observation interval is required.")
    if any(interval < 1 for interval in intervals):
        raise ValueError("Observation intervals must be >= 1.")
    return intervals


def _parse_levels(raw: str) -> tuple[str, ...]:
    levels = tuple(part.strip() for part in raw.split(",") if part.strip())
    allowed = {"small", "medium", "large"}
    invalid = [level for level in levels if level not in allowed]
    if invalid:
        raise ValueError(f"Unknown perturbation levels: {invalid}")
    return levels


if __name__ == "__main__":
    main()
