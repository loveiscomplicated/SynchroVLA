from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.pick_place_bc import PickPlaceDemoConfig, evaluate_scripted_pick_place


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scripted MuJoCo Pick-and-Place expert.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--output-path", default="artifacts/mujoco_pick_place/expert_eval/dummy.pt")
    parser.add_argument("--metadata-path", default="artifacts/mujoco_pick_place/expert_eval/dummy_metadata.json")
    parser.add_argument("--object-x-range", nargs=2, type=float, default=(-0.0048, -0.0001))
    parser.add_argument("--destination-x-range", nargs=2, type=float, default=(-0.060, -0.053))
    parser.add_argument("--min-separation", type=float, default=0.012)
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_scripted_pick_place(
        PickPlaceDemoConfig(
            episodes=args.episodes,
            seed=args.seed,
            output_path=args.output_path,
            metadata_path=args.metadata_path,
            object_x_range=tuple(args.object_x_range),
            destination_x_range=tuple(args.destination_x_range),
            min_separation=args.min_separation,
            max_steps=args.max_steps,
            render=args.render,
        )
    )
    print(json.dumps({"output_path": payload["output_path"], "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
