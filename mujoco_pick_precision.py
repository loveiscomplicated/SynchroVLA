from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.evaluation.pick_precision import PickPrecisionConfig, analyze_pick_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Pick action precision and near-contact saturation.")
    parser.add_argument("--dataset-path", default="artifacts/mujoco_learned_pick/demos/pick_demos_300.pt")
    parser.add_argument("--output-dir", default="artifacts/mujoco_pick_precision/diagnostics")
    parser.add_argument("--ff-checkpoint-path", default="artifacts/mujoco_learned_pick/checkpoints_ff_300/graph_feedforward.pt")
    parser.add_argument("--gru-checkpoint-path", default="artifacts/mujoco_learned_pick/checkpoints_gru_300_motion/graph_recurrent.pt")
    parser.add_argument(
        "--extra-checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional policy checkpoint to include in offline precision tables. May be repeated.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-step", type=float, default=0.045)
    parser.add_argument("--saturation-fraction", type=float, default=0.90)
    parser.add_argument("--failure-seed", type=int, default=4242)
    parser.add_argument("--failure-steps", type=int, default=180)
    parser.add_argument("--no-failure-diagnostic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze_pick_precision(
        PickPrecisionConfig(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            ff_checkpoint_path=args.ff_checkpoint_path,
            gru_checkpoint_path=args.gru_checkpoint_path,
            extra_checkpoints=tuple(_parse_extra_checkpoint(item) for item in args.extra_checkpoint),
            split=args.split,
            device=args.device,
            max_step=args.max_step,
            saturation_fraction=args.saturation_fraction,
            failure_seed=args.failure_seed,
            failure_steps=args.failure_steps,
            run_failure_diagnostic=not args.no_failure_diagnostic,
        )
    )
    print(
        json.dumps(
            {
                "output_path": payload["output_path"],
                "expert_by_distance_bin": payload["expert_by_distance_bin"],
                "policy_by_distance_bin": payload["policy_by_distance_bin"],
                "failure_diagnostic": payload.get("gru_failure_diagnostic"),
            },
            indent=2,
        )
    )


def _parse_extra_checkpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--extra-checkpoint must be formatted as LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--extra-checkpoint must be formatted as LABEL=PATH")
    return label, path


if __name__ == "__main__":
    main()
