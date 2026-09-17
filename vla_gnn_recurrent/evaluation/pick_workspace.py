from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vla_gnn_recurrent.sim.pick_expert import ScriptedPickConfig, run_scripted_pick_evaluation
from vla_gnn_recurrent.training.pick_bc import PickEvalConfig, evaluate_pick_policy
from vla_gnn_recurrent.utils import DevicePreference, ensure_dir


@dataclass(frozen=True)
class WorkspaceRange:
    name: str
    object_x_range: tuple[float, float]


@dataclass
class PickWorkspaceRobustnessConfig:
    episodes: int = 100
    seed: int = 2027
    output_dir: str = "artifacts/mujoco_pick_place/workspace_robustness"
    ff_checkpoint_path: str = "artifacts/mujoco_pick_precision/checkpoints_ff_dir_mag/graph_feedforward_dir_mag.pt"
    gru_checkpoint_path: str = "artifacts/mujoco_pick_precision/checkpoints_gru_dir_mag_precision/graph_recurrent_dir_mag.pt"
    device: DevicePreference = "auto"
    max_steps: int = 180
    render: bool = False


DEFAULT_WORKSPACE_RANGES = (
    WorkspaceRange("R0_current", (-0.015, 0.0)),
    WorkspaceRange("R1_modest", (-0.025, 0.010)),
    WorkspaceRange("R2_broader", (-0.035, 0.020)),
)


def evaluate_pick_workspace_robustness(
    config: PickWorkspaceRobustnessConfig,
    ranges: tuple[WorkspaceRange, ...] = DEFAULT_WORKSPACE_RANGES,
) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    rows: list[dict[str, Any]] = []
    for idx, workspace in enumerate(ranges):
        seed = config.seed + idx * 1000
        expert_payload = run_scripted_pick_evaluation(
            ScriptedPickConfig(
                episodes=config.episodes,
                seed=seed,
                max_steps=260,
                object_x_range=workspace.object_x_range,
                output_dir=str(output_dir / workspace.name / "expert"),
                render=config.render,
            )
        )
        ff_payload = evaluate_pick_policy(
            PickEvalConfig(
                checkpoint_path=config.ff_checkpoint_path,
                episodes=config.episodes,
                seed=seed,
                output_dir=str(output_dir / workspace.name / "ff"),
                device=config.device,
                max_steps=config.max_steps,
                render=config.render,
                object_x_range=workspace.object_x_range,
            )
        )
        gru_payload = evaluate_pick_policy(
            PickEvalConfig(
                checkpoint_path=config.gru_checkpoint_path,
                episodes=config.episodes,
                seed=seed,
                output_dir=str(output_dir / workspace.name / "gru"),
                device=config.device,
                max_steps=config.max_steps,
                render=config.render,
                object_x_range=workspace.object_x_range,
            )
        )
        rows.append(
            {
                "workspace": workspace.name,
                "object_x_range": workspace.object_x_range,
                "expert": expert_payload["summary"],
                "graph_ff": ff_payload["summary"],
                "graph_gru": gru_payload["summary"],
                "mechanically_valid": expert_payload["summary"]["lift_success_rate"] >= 0.90,
            }
        )
    payload = {
        "config": asdict(config),
        "ranges": [asdict(item) for item in ranges],
        "rows": rows,
    }
    path = Path(output_dir) / "workspace_robustness.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["output_path"] = str(path)
    return payload
