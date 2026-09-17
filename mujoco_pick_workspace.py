from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.evaluation.pick_workspace import (
    PickWorkspaceRobustnessConfig,
    WorkspaceRange,
    evaluate_pick_workspace_robustness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pick workspace robustness for expert, Graph-FF, and Graph-GRU.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-dir", default="artifacts/mujoco_pick_place/workspace_robustness")
    parser.add_argument("--ff-checkpoint-path", default="artifacts/mujoco_pick_precision/checkpoints_ff_dir_mag/graph_feedforward_dir_mag.pt")
    parser.add_argument("--gru-checkpoint-path", default="artifacts/mujoco_pick_precision/checkpoints_gru_dir_mag_precision/graph_recurrent_dir_mag.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--range", action="append", default=[], metavar="NAME:LOW:HIGH")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranges = tuple(_parse_range(item) for item in args.range) if args.range else None
    payload = evaluate_pick_workspace_robustness(
        PickWorkspaceRobustnessConfig(
            episodes=args.episodes,
            seed=args.seed,
            output_dir=args.output_dir,
            ff_checkpoint_path=args.ff_checkpoint_path,
            gru_checkpoint_path=args.gru_checkpoint_path,
            device=args.device,
            max_steps=args.max_steps,
            render=args.render,
        ),
        ranges=ranges,  # type: ignore[arg-type]
    ) if ranges is not None else evaluate_pick_workspace_robustness(
        PickWorkspaceRobustnessConfig(
            episodes=args.episodes,
            seed=args.seed,
            output_dir=args.output_dir,
            ff_checkpoint_path=args.ff_checkpoint_path,
            gru_checkpoint_path=args.gru_checkpoint_path,
            device=args.device,
            max_steps=args.max_steps,
            render=args.render,
        )
    )
    print(json.dumps({"output_path": payload["output_path"], "rows": payload["rows"]}, indent=2))


def _parse_range(value: str) -> WorkspaceRange:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--range must be NAME:LOW:HIGH")
    return WorkspaceRange(parts[0], (float(parts[1]), float(parts[2])))


if __name__ == "__main__":
    main()
