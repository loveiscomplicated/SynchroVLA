from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.pick_place_bc import PickPlaceEvalConfig, evaluate_pick_place_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned MuJoCo Pick-and-Place policies.")
    parser.add_argument("--checkpoint-path", default="artifacts/mujoco_pick_place/checkpoints/graph_recurrent_dir_mag.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=992)
    parser.add_argument("--output-dir", default="artifacts/mujoco_pick_place/eval")
    parser.add_argument("--recurrent-mode", choices=["normal", "step_reset"], default="normal")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, default=340)
    parser.add_argument("--object-x-range", nargs=2, type=float, default=(-0.0048, -0.0001))
    parser.add_argument("--destination-x-range", nargs=2, type=float, default=(-0.060, -0.053))
    parser.add_argument("--min-separation", type=float, default=0.012)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_pick_place_policy(
        PickPlaceEvalConfig(
            checkpoint_path=args.checkpoint_path,
            episodes=args.episodes,
            seed=args.seed,
            output_dir=args.output_dir,
            recurrent_mode=args.recurrent_mode,
            device=args.device,
            max_steps=args.max_steps,
            object_x_range=tuple(args.object_x_range),
            destination_x_range=tuple(args.destination_x_range),
            min_separation=args.min_separation,
            render=not args.no_render,
        )
    )
    print(json.dumps({"output_path": payload["output_path"], "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
