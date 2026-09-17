from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.pick_bc import PickEvalConfig, evaluate_pick_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a learned Graph-GNN Pick policy in closed-loop MuJoCo.")
    parser.add_argument("--checkpoint-path", default="artifacts/mujoco_learned_pick/checkpoints/graph_recurrent.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--output-dir", default="artifacts/mujoco_learned_pick/eval")
    parser.add_argument("--recurrent-mode", choices=["normal", "step_reset"], default="normal")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_pick_policy(
        PickEvalConfig(
            checkpoint_path=args.checkpoint_path,
            episodes=args.episodes,
            seed=args.seed,
            output_dir=args.output_dir,
            recurrent_mode=args.recurrent_mode,
            device=args.device,
            max_steps=args.max_steps,
            render=not args.no_render,
        )
    )
    print(
        json.dumps(
            {
                "output_path": payload["output_path"],
                "summary": payload["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
