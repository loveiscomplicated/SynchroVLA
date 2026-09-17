from __future__ import annotations

import argparse
import json

from vla_gnn_recurrent.training.mujoco_reaching import MujocoReachTrainConfig, train_mujoco_reaching


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MuJoCo articulated reaching controller.")
    parser.add_argument("--model", choices=["graph_recurrent", "graph_feedforward"], default="graph_recurrent")
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="artifacts/mujoco_prototype/checkpoints")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--bptt-steps", type=int, default=8)
    parser.add_argument("--sequence-repeats", type=int, default=8)
    parser.add_argument("--observation-interval", type=int, default=1)
    parser.add_argument("--demo-modes", default="static,moving,perturb")
    parser.add_argument("--perturb-step", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MujocoReachTrainConfig(
        model_kind=args.model,
        episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        log_every=args.log_every,
        bptt_steps=args.bptt_steps,
        sequence_repeats=args.sequence_repeats,
        observation_interval=args.observation_interval,
        demo_modes=tuple(part.strip() for part in args.demo_modes.split(",") if part.strip()),
        perturb_step=args.perturb_step,
    )
    _, summary = train_mujoco_reaching(config)
    print(json.dumps({"mujoco_training_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
