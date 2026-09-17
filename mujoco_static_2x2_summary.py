from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


MODEL_ORDER = ("flat_ff", "flat_gru", "graph_ff", "graph_gru")
METRICS = (
    "grasp_success_rate",
    "lift_success_rate",
    "transport_success_rate",
    "valid_release_rate",
    "placement_success_rate",
    "drop_rate",
    "mean_steps",
    "mean_final_placement_error",
    "mean_action_smoothness",
    "mean_latency_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize static Flat/Graph x FF/GRU Pick-and-Place ablation.")
    parser.add_argument("--flat-ff", nargs="+", required=True)
    parser.add_argument("--flat-gru", nargs="+", required=True)
    parser.add_argument("--graph-ff", nargs="+", required=True)
    parser.add_argument("--graph-gru", nargs="+", required=True)
    parser.add_argument("--output-path", default="artifacts/static_2x2_ablation/interaction_calculations.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "flat_ff": args.flat_ff,
        "flat_gru": args.flat_gru,
        "graph_ff": args.graph_ff,
        "graph_gru": args.graph_gru,
    }
    payload = summarize_2x2(paths)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "effects": payload["effects"]}, indent=2))


def summarize_2x2(paths: dict[str, list[str]]) -> dict[str, Any]:
    model_summaries = {
        model: [_load_summary(path) for path in model_paths]
        for model, model_paths in paths.items()
    }
    aggregate = {
        model: {
            metric: _mean_std([float(summary.get(metric, 0.0)) for summary in summaries])
            for metric in METRICS
        }
        for model, summaries in model_summaries.items()
    }
    means = {
        model: {metric: aggregate[model][metric]["mean"] for metric in METRICS}
        for model in MODEL_ORDER
    }
    effects = {}
    for metric in ("placement_success_rate", "valid_release_rate"):
        flat_ff = means["flat_ff"][metric]
        flat_gru = means["flat_gru"][metric]
        graph_ff = means["graph_ff"][metric]
        graph_gru = means["graph_gru"][metric]
        effects[metric] = {
            "graph_effect_under_ff": graph_ff - flat_ff,
            "graph_effect_under_gru": graph_gru - flat_gru,
            "recurrence_effect_on_flat": flat_gru - flat_ff,
            "recurrence_effect_on_graph": graph_gru - graph_ff,
            "graph_recurrence_interaction": graph_gru - graph_ff - flat_gru + flat_ff,
        }
    return {
        "models": aggregate,
        "effects": effects,
        "source_paths": paths,
    }


def _load_summary(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload["summary"])


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0.0}
    return {
        "mean": sum(values) / len(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": float(len(values)),
    }


if __name__ == "__main__":
    main()
