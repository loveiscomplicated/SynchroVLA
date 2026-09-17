from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.pick_place_bc import (
    PickPlaceAlignmentDiagnosticsConfig,
    diagnose_pick_place_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose MuJoCo Pick-and-Place destination placement failures.")
    parser.add_argument("--checkpoint-path", default="artifacts/mujoco_pick_place/checkpoints_gru_300_1epoch/graph_recurrent_dir_mag.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--output-dir", default="artifacts/mujoco_pick_place_alignment/diagnosis")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--object-x-range", nargs=2, type=float, default=(-0.0048, -0.0001))
    parser.add_argument("--destination-x-range", nargs=2, type=float, default=(-0.060, -0.053))
    parser.add_argument("--min-separation", type=float, default=0.012)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = diagnose_pick_place_alignment(
        PickPlaceAlignmentDiagnosticsConfig(
            checkpoint_path=args.checkpoint_path,
            episodes=args.episodes,
            seed=args.seed,
            output_dir=args.output_dir,
            device=args.device,
            max_steps=args.max_steps,
            object_x_range=tuple(args.object_x_range),
            destination_x_range=tuple(args.destination_x_range),
            min_separation=args.min_separation,
        )
    )
    print(
        json.dumps(
            {
                "output_path": payload["output_path"],
                "policy_summary": payload["policy_summary"],
                "failure_source_counts": payload["failure_source_counts"],
                "expert_vs_policy": payload["expert_vs_policy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
