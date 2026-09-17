from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_gnn_recurrent.evaluation.visualize import plot_trajectory
from vla_gnn_recurrent.training.mujoco_reaching import evaluate_mujoco_reaching


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MuJoCo articulated reaching controller.")
    parser.add_argument("--checkpoint", default="artifacts/mujoco_prototype/checkpoints/graph_recurrent.pt")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--observation-intervals", default="1")
    parser.add_argument("--moving-target", action="store_true")
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--perturb-step", type=int, default=16)
    parser.add_argument("--output-dir", default="artifacts/mujoco_prototype/eval")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--corrector-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = []
    for interval in _parse_intervals(args.observation_intervals):
        payload = evaluate_mujoco_reaching(
            checkpoint_path=args.checkpoint,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            device_preference=args.device,
            observation_interval=interval,
            moving_target=args.moving_target,
            perturb=args.perturb,
            perturb_step=args.perturb_step,
            output_dir=args.output_dir,
            render=args.render,
            corrector_checkpoint_path=args.corrector_checkpoint,
        )
        if args.plot:
            plot_payload = _trajectory_plot_payload(payload["episodes"][0])
            suffix = Path(payload["output_path"]).stem
            plot_path = Path(args.output_dir) / f"{suffix}_trajectory.png"
            plot_trajectory(plot_payload, plot_path, title=suffix)
            payload["trajectory_plot"] = str(plot_path)
        payloads.append(payload)
    concise = [
        {
            "output_path": payload["output_path"],
            "observation_interval": payload["observation_interval"],
            "moving_target": payload["moving_target"],
            "perturb": payload["perturb"],
            "summary": payload["summary"],
            "trajectory_plot": payload.get("trajectory_plot"),
            "frames": payload["episodes"][0].get("frames", [])[:3],
        }
        for payload in payloads
    ]
    print(json.dumps({"mujoco_evaluation_summaries": concise}, indent=2))


def _parse_intervals(raw: str) -> tuple[int, ...]:
    intervals = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not intervals:
        raise ValueError("At least one observation interval is required.")
    return intervals


def _trajectory_plot_payload(episode: dict) -> dict:
    return {
        "trajectory": episode["trajectory"],
        "distances": episode["distances"],
        "targets": episode["targets"],
        "true_targets": episode["targets"],
        "observed_targets": episode["observed_targets"],
        "reanchor_steps": episode["reanchor_steps"],
        "original_target": episode["targets"][0],
        "perturbed_target": episode["targets"][episode["perturb_true_step"]]
        if episode.get("perturb_true_step") is not None
        and episode["perturb_true_step"] < len(episode["targets"])
        else None,
        "perturb_step": episode.get("perturb_true_step"),
    }


if __name__ == "__main__":
    main()
