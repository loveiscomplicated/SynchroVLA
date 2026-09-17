from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.pick_place_bc import (
    PickPlaceAlignmentDaggerConfig,
    aggregate_pick_place_alignment_dagger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect placement-focused Pick-and-Place DAgger labels.")
    parser.add_argument("--dataset-path", default="artifacts/mujoco_pick_place/demos/pick_place_demos_300.pt")
    parser.add_argument(
        "--policy-checkpoint-path",
        default="artifacts/mujoco_pick_place/checkpoints_gru_300_1epoch/graph_recurrent_dir_mag.pt",
    )
    parser.add_argument("--output-path", default="artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger.pt")
    parser.add_argument(
        "--metadata-path",
        default="artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger_metadata.json",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--object-x-range", nargs=2, type=float, default=(-0.0048, -0.0001))
    parser.add_argument("--destination-x-range", nargs=2, type=float, default=(-0.060, -0.053))
    parser.add_argument("--min-separation", type=float, default=0.012)
    parser.add_argument("--destination-start-threshold", type=float, default=0.120)
    parser.add_argument("--release-tail-steps", type=int, default=56)
    parser.add_argument("--correction-repeat-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = aggregate_pick_place_alignment_dagger(
        PickPlaceAlignmentDaggerConfig(
            dataset_path=args.dataset_path,
            policy_checkpoint_path=args.policy_checkpoint_path,
            output_path=args.output_path,
            metadata_path=args.metadata_path,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            max_steps=args.max_steps,
            object_x_range=tuple(args.object_x_range),
            destination_x_range=tuple(args.destination_x_range),
            min_separation=args.min_separation,
            destination_start_threshold=args.destination_start_threshold,
            release_tail_steps=args.release_tail_steps,
            correction_repeat_count=args.correction_repeat_count,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
