from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vla_gnn_recurrent.evaluation.visualize import plot_gate_by_magnitude, plot_perturbation_magnitude


LEVELS = ("small", "medium", "large")
FIXED_DECAY_CONDITIONS = ("gru_fixed_decay_0.25", "gru_fixed_decay_0.5", "gru_fixed_decay_0.75")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize learned state-correction experiment artifacts.")
    parser.add_argument("--root", default="artifacts/state_corrector_final")
    parser.add_argument("--output", default="artifacts/state_corrector_final/correction_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = _load_perturb_rows(root)
    moving_rows = _load_moving_rows(root / "moving")
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for interval in (1, 4):
        recurrent_rows = [
            row
            for row in rows
            if row["observation_interval"] == interval and row["condition"] != "feedforward"
        ]
        plot_perturbation_magnitude(
            recurrent_rows,
            plot_dir / f"perturb_magnitude_recovery_N{interval}.png",
            metric="mean_recovery_time_observable",
        )

    learned_episodes = []
    for row in rows:
        if row["condition"] == "gru_learned_correction":
            learned_episodes.extend(row["episodes"])
    plot_gate_by_magnitude({"gru_learned_correction": learned_episodes}, plot_dir / "learned_gate_by_magnitude.png")

    payload = {
        "perturbation_rows": rows,
        "moving_rows": moving_rows,
        "best_fixed_decay": _best_fixed_decay(rows),
        "plots": {
            "perturb_magnitude_recovery_N1": str(plot_dir / "perturb_magnitude_recovery_N1.png"),
            "perturb_magnitude_recovery_N4": str(plot_dir / "perturb_magnitude_recovery_N4.png"),
            "learned_gate_by_magnitude": str(plot_dir / "learned_gate_by_magnitude.png"),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(_compact(payload), indent=2))


def _load_perturb_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        directory = root / f"perturb_{level}"
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = payload["config"]
            condition = _condition_from_payload(path, payload)
            rows.append(
                {
                    "level": level,
                    "perturbation_level": level,
                    "condition": condition,
                    "model_kind": config["model_kind"],
                    "observation_interval": int(config["observation_interval"]),
                    "summary": payload["summary"],
                    "episodes": payload["episodes"],
                    "path": str(path),
                }
            )
    return rows


def _load_moving_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        rows.append(
            {
                "condition": _condition_from_payload(path, payload),
                "model_kind": config["model_kind"],
                "observation_interval": int(config["observation_interval"]),
                "summary": payload["summary"],
                "path": str(path),
            }
        )
    return rows


def _condition_from_payload(path: Path, payload: dict[str, Any]) -> str:
    config = payload["config"]
    if config["model_kind"] == "feedforward":
        return "feedforward"
    mode = config.get("recurrent_mode", "normal")
    if mode == "fixed_decay":
        return f"gru_fixed_decay_{float(config.get('fixed_decay_alpha', 0.5)):g}"
    return f"gru_{mode}"


def _best_fixed_decay(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    for level in LEVELS:
        for interval in (1, 4):
            candidates = [
                row
                for row in rows
                if row["level"] == level
                and row["observation_interval"] == interval
                and row["condition"] in FIXED_DECAY_CONDITIONS
            ]
            if not candidates:
                continue
            winner = min(candidates, key=lambda row: float(row["summary"]["mean_recovery_time_observable"]))
            best.append(
                {
                    "level": level,
                    "observation_interval": interval,
                    "condition": winner["condition"],
                    "mean_recovery_time_observable": winner["summary"]["mean_recovery_time_observable"],
                }
            )
    return best


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    compact_rows = []
    for row in payload["perturbation_rows"]:
        summary = row["summary"]
        compact_rows.append(
            {
                "level": row["level"],
                "N": row["observation_interval"],
                "condition": row["condition"],
                "success": round(float(summary["success_rate"]), 3),
                "recovery": round(float(summary["mean_recovery_time_observable"]), 2),
                "redir+1": round(float(summary["mean_redirection_cosine_1"]), 3),
                "final": round(float(summary["mean_final_distance"]), 4),
                "smooth": round(float(summary["mean_action_smoothness"]), 4),
            }
        )
    return {
        "perturbation_rows": compact_rows,
        "best_fixed_decay": payload["best_fixed_decay"],
        "plots": payload["plots"],
    }


if __name__ == "__main__":
    main()
