from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.pick_bc import PickDemoConfig, generate_pick_demonstrations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MuJoCo physical Pick demonstrations.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-path", default="artifacts/mujoco_learned_pick/demos/pick_demos.pt")
    parser.add_argument("--metadata-path", default="artifacts/mujoco_learned_pick/demos/pick_demos_metadata.json")
    parser.add_argument("--max-steps", type=int, default=260)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_pick_demonstrations(
        PickDemoConfig(
            episodes=args.episodes,
            seed=args.seed,
            output_path=args.output_path,
            metadata_path=args.metadata_path,
            max_steps=args.max_steps,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
