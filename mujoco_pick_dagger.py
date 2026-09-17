from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.pick_bc import PickDaggerConfig, aggregate_pick_dagger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect one DAgger-style Pick dataset aggregation.")
    parser.add_argument("--dataset-path", default="artifacts/mujoco_learned_pick/demos/pick_demos_300.pt")
    parser.add_argument(
        "--policy-checkpoint-path",
        default="artifacts/mujoco_learned_pick/checkpoints_gru_300_motion/graph_recurrent.pt",
    )
    parser.add_argument("--output-path", default="artifacts/mujoco_learned_pick/demos/pick_demos_300_dagger.pt")
    parser.add_argument(
        "--metadata-path",
        default="artifacts/mujoco_learned_pick/demos/pick_demos_300_dagger_metadata.json",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1777)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--max-steps", type=int, default=140)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = aggregate_pick_dagger(
        PickDaggerConfig(
            dataset_path=args.dataset_path,
            policy_checkpoint_path=args.policy_checkpoint_path,
            output_path=args.output_path,
            metadata_path=args.metadata_path,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            max_steps=args.max_steps,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
