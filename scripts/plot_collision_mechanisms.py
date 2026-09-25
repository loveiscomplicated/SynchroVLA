"""Render saved guard-window and near-target contact diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def _flag(value: str) -> bool:
    return value.lower() == "true"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/collision_mechanism_audit_v1"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    definition = json.loads((args.output_dir / "guard_definition.json").read_text())
    thresholds = definition["thresholds"]
    for folder in sorted((args.output_dir / "guard_audit").glob("[0-9]*_[0-9]*")):
        rows = _rows(folder / "per_step_guard_audit.csv")
        x = [int(row["relative_to_collision"]) for row in rows]
        fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
        for ax in axes:
            for step, row in zip(x, rows, strict=True):
                if _flag(row["counterfactual_no_inward_success_if_started_here"]):
                    ax.axvspan(step-.45, step+.45, color="tab:green", alpha=.13)
                if _flag(row["guard_condition"]):
                    ax.axvline(step, color="tab:orange", alpha=.75, linewidth=2)
            ax.grid(alpha=.25)
        axes[0].plot(x, [float(row["guard_position_proxy"]) for row in rows],
                     label="Observed position proxy")
        axes[0].plot(x, [float(row["position_error_true"]) for row in rows],
                     label="True position error", linestyle="--", alpha=.7)
        axes[0].axhline(thresholds["position_greater_equal_m"], color="tab:red",
                        linestyle=":", label="Guard threshold")
        axes[0].set_ylabel("Position (m)")
        axes[1].plot(x, [float(row["guard_yaw_proxy"]) for row in rows],
                     label="Observed yaw proxy")
        axes[1].plot(x, [float(row["yaw_error_true"]) for row in rows],
                     label="True yaw error", linestyle="--", alpha=.7)
        axes[1].axhline(thresholds["yaw_greater_equal_rad"], color="tab:red",
                        linestyle=":", label="Guard threshold")
        axes[1].set_ylabel("Yaw (rad)")
        axes[2].plot(x, [float(row["guard_distance_proxy"]) for row in rows],
                     label="Visible tip-to-point distance")
        axes[2].axhline(thresholds["distance_strict_less_m"], color="tab:red",
                        linestyle=":", label="Guard zone")
        axes[2].set_ylabel("Distance (m)")
        axes[3].plot(x, [float(row["raw_inward_component"]) for row in rows],
                     label="Base inward")
        axes[3].plot(x, [float(row["guarded_inward_component"]) for row in rows],
                     label="Guarded inward", linestyle="--")
        axes[3].set_ylabel("Inward (m)")
        axes[3].set_xlabel("Steps relative to collision")
        for ax in axes:
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(folder.name + " · green: rescuable start, orange: guard condition",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(folder / "guard_window.png", dpi=140)
        plt.close(fig)

    for folder in sorted((args.output_dir / "near_target").glob("[0-9]*_[0-9]*")):
        rows = _rows(folder / "precontact_trace.csv")
        x = [int(row["relative_to_collision"]) for row in rows]
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(x, [float(row["left_tip_surface_distance_m"]) for row in rows],
                     label="Left fingertip")
        axes[0].plot(x, [float(row["right_tip_surface_distance_m"]) for row in rows],
                     label="Right fingertip")
        axes[0].axhline(0, color="tab:red", linestyle=":")
        axes[0].set_ylabel("Physical clearance (m)")
        for name, label in (("inward_m", "Inward"),
                            ("outward_m", "Outward"),
                            ("tangent_norm_m", "Tangent norm")):
            axes[1].plot(x[:-1], [float(row[name]) for row in rows[:-1]], label=label)
        axes[1].set_ylabel("Translation action (m)")
        axes[2].plot(x[:-1], [float(row["yaw_command_rad"]) for row in rows[:-1]],
                     label="Yaw action")
        axes[2].set_ylabel("Yaw action (rad)")
        axes[2].set_xlabel("Steps relative to collision")
        for ax in axes:
            ax.axvline(0, color="tab:red", linestyle="--", linewidth=1)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(folder.name, fontsize=10)
        fig.tight_layout()
        fig.savefig(folder / "contact_mechanism.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
