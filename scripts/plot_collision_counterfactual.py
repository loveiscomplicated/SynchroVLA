"""Render four diagnostic traces from completed collision counterfactual outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/collision_multistep_counterfactual_v1"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for episode_dir in sorted(args.output_dir.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        metadata = json.loads((episode_dir / "metadata.json").read_text())
        original = json.loads((episode_dir / "original_trace.json").read_text())
        # Prefer a successful no-inward persistent replay for a stable comparison;
        # this choice affects the figure only, never checkpoint or policy selection.
        candidates = sorted((episode_dir / "no_inward/persistent").glob("start_*.json"))
        if not any(json.loads(path.read_text())["result"]["success"] for path in candidates):
            candidates = sorted(episode_dir.glob("*/**/start_*.json"))
        candidate = None
        for path in candidates:
            payload = json.loads(path.read_text())
            if payload["result"]["success"]:
                candidate = payload
                break
        fields = (("position_error", "Position error (m)"),
                  ("yaw_error", "Yaw error (rad)"),
                  ("clearance", "Clearance (m)"),
                  ("inward_m", "Inward action (m)"))
        fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
        for ax, (field, label) in zip(axes, fields, strict=True):
            ax.plot([row["timestep"] for row in original],
                    [row[field] if row[field] is not None else float("nan") for row in original],
                    label="Original", color="black", linewidth=1.5)
            if candidate is not None:
                trace = candidate["trace"]
                ax.plot([row["timestep"] for row in trace],
                        [row[field] if row[field] is not None else float("nan") for row in trace],
                        label=(f"{candidate['result']['intervention']} "
                               f"{candidate['result']['mode']} "
                               f"from {candidate['result']['start_relative_step']:+d}"),
                        color="tab:blue", linewidth=1.5)
            ax.axvline(metadata["original_outcome"]["collision_step"],
                       color="tab:red", linestyle="--", linewidth=1)
            ax.set_ylabel(label)
            ax.grid(alpha=.25)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("Episode timestep")
        fig.suptitle(metadata["episode_id"], fontsize=10)
        fig.tight_layout()
        fig.savefig(episode_dir / "trajectory_comparison.png", dpi=140)
        plt.close(fig)
        print(episode_dir / "trajectory_comparison.png")


if __name__ == "__main__":
    main()
