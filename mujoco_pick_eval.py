from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.sim.pick_expert import ScriptedPickConfig, run_scripted_pick_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scripted MuJoCo grasp/lift expert.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--output-dir", default="artifacts/mujoco_prototype/pick")
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ScriptedPickConfig(
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        render=not args.no_render,
    )
    payload = run_scripted_pick_evaluation(config)
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
