from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.pick_place_bc import PickPlaceDemoConfig, generate_pick_place_demonstrations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MuJoCo Pick-and-Place graph demonstration episodes.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--output-path", default="artifacts/mujoco_pick_place/demos/pick_place_demos.pt")
    parser.add_argument("--metadata-path", default="artifacts/mujoco_pick_place/demos/pick_place_demos_metadata.json")
    parser.add_argument("--object-x-range", nargs=2, type=float, default=(-0.0048, -0.0001))
    parser.add_argument("--destination-x-range", nargs=2, type=float, default=(-0.060, -0.053))
    parser.add_argument("--min-separation", type=float, default=0.012)
    parser.add_argument("--max-steps", type=int, default=340)
    parser.add_argument("--max-attempts-multiplier", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_pick_place_demonstrations(
        PickPlaceDemoConfig(
            episodes=args.episodes,
            seed=args.seed,
            output_path=args.output_path,
            metadata_path=args.metadata_path,
            object_x_range=tuple(args.object_x_range),
            destination_x_range=tuple(args.destination_x_range),
            min_separation=args.min_separation,
            max_steps=args.max_steps,
            max_attempts_multiplier=args.max_attempts_multiplier,
            render=args.render,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
