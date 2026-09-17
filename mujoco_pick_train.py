from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.pick_bc import PickTrainConfig, train_pick_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Graph-GNN policy for physical MuJoCo Pick.")
    parser.add_argument("--dataset-path", default="artifacts/mujoco_learned_pick/demos/pick_demos.pt")
    parser.add_argument(
        "--model-kind",
        choices=["graph_recurrent", "graph_feedforward", "graph_recurrent_dir_mag", "graph_feedforward_dir_mag"],
        default="graph_recurrent",
    )
    parser.add_argument("--output-dir", default="artifacts/mujoco_learned_pick/checkpoints")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--bptt-steps", type=int, default=16)
    parser.add_argument("--motion-loss-weight", type=float, default=1.0)
    parser.add_argument("--direction-loss-weight", type=float, default=1.0)
    parser.add_argument("--magnitude-loss-weight", type=float, default=1.0)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--precision-weighting", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--initial-checkpoint-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = train_pick_policy(
        PickTrainConfig(
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
            precision_weighting=args.precision_weighting,
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
