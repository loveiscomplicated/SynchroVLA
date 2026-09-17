from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.pick_place_bc import PickPlaceTrainConfig, train_pick_place_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Graph-GNN policies for MuJoCo Pick-and-Place.")
    parser.add_argument("--dataset-path", default="artifacts/mujoco_pick_place/demos/pick_place_demos.pt")
    parser.add_argument(
        "--model-kind",
        choices=["graph_recurrent_dir_mag", "graph_feedforward_dir_mag", "graph_recurrent", "graph_feedforward"],
        default="graph_recurrent_dir_mag",
    )
    parser.add_argument("--output-dir", default="artifacts/mujoco_pick_place/checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--bptt-steps", type=int, default=32)
    parser.add_argument("--motion-loss-weight", type=float, default=10.0)
    parser.add_argument("--direction-loss-weight", type=float, default=1.0)
    parser.add_argument("--magnitude-loss-weight", type=float, default=1.0)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-precision-weighting", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--initial-checkpoint-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = train_pick_place_policy(
        PickPlaceTrainConfig(
            dataset_path=args.dataset_path,
            model_kind=args.model_kind,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            bptt_steps=args.bptt_steps,
            motion_loss_weight=args.motion_loss_weight,
            direction_loss_weight=args.direction_loss_weight,
            magnitude_loss_weight=args.magnitude_loss_weight,
            gripper_loss_weight=args.gripper_loss_weight,
            precision_weighting=not args.no_precision_weighting,
            device=args.device,
            seed=args.seed,
            initial_checkpoint_path=args.initial_checkpoint_path,
        )
    )
    print(
        json.dumps(
            {
                "checkpoint_path": summary["checkpoint_path"],
                "summary_path": summary["summary_path"],
                "offline": summary["offline"],
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
