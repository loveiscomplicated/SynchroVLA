from pathlib import Path

from mujoco_dynamic_2x2_eval import (
    checkpoint_path_for,
    compute_effects,
    make_dynamic_episode_plan,
    static_run_dir_for,
    summarize_dynamic_results,
)


def test_dynamic_episode_plan_is_deterministic_and_moves_destination() -> None:
    plan_a = make_dynamic_episode_plan(episodes=6, seed=3701, max_steps=420)
    plan_b = make_dynamic_episode_plan(episodes=6, seed=3701, max_steps=420)

    assert plan_a == plan_b
    assert [item["episode"] for item in plan_a] == [1, 2, 3, 4, 5, 6]
    assert len({item["episode_seed"] for item in plan_a}) == 6
    assert all(item["perturbed_destination_x"] > 0.0 for item in plan_a)
    assert all(abs(item["perturbed_destination_x"] - item["destination_x"]) >= 0.090 for item in plan_a)


def test_dynamic_summary_counts_perturbation_and_recovery() -> None:
    episode = {
        "approach_success": True,
        "alignment_success": True,
        "grasp_success": True,
        "lift_success": True,
        "transport_success": True,
        "release_success": True,
        "valid_release_success": True,
        "gripper_open_event": True,
        "placement_success": True,
        "drop": False,
        "steps": 3,
        "first_grasp_step": 1,
        "valid_release_step": 2,
        "placement_step": 3,
        "failure_reason": None,
        "destination_perturbed": True,
        "dynamic_recovery_steps": 4,
        "records": [
            {"step": 1, "placement_error": 0.08, "delta_ee": [0.01, 0.0, 0.0]},
            {"step": 2, "placement_error": 0.03, "pre_placement_error": 0.03, "delta_ee": [0.0, 0.0, 0.01]},
        ],
    }

    summary = summarize_dynamic_results([episode])

    assert summary["dynamic_perturbation_rate"] == 1.0
    assert summary["dynamic_recovery_rate"] == 1.0
    assert summary["mean_dynamic_recovery_steps"] == 4.0
    assert summary["placement_success_rate"] == 1.0


def test_dynamic_factorial_interaction_uses_per_seed_complete_cells() -> None:
    runs = [
        {"model_label": "flat_ff", "seed": 1, "summary": {"placement_success_rate": 0.2, "valid_release_rate": 0.3, "transport_success_rate": 0.4, "dynamic_recovery_rate": 0.5}},
        {"model_label": "flat_gru", "seed": 1, "summary": {"placement_success_rate": 0.4, "valid_release_rate": 0.5, "transport_success_rate": 0.6, "dynamic_recovery_rate": 0.7}},
        {"model_label": "graph_ff", "seed": 1, "summary": {"placement_success_rate": 0.3, "valid_release_rate": 0.4, "transport_success_rate": 0.5, "dynamic_recovery_rate": 0.6}},
        {"model_label": "graph_gru", "seed": 1, "summary": {"placement_success_rate": 0.8, "valid_release_rate": 0.9, "transport_success_rate": 1.0, "dynamic_recovery_rate": 1.0}},
    ]

    effects = compute_effects(runs)

    assert effects["per_seed"]["1"]["placement_success_rate"]["recurrence_effect_on_flat"] == 0.2
    assert effects["per_seed"]["1"]["placement_success_rate"]["recurrence_effect_on_graph"] == 0.5
    assert effects["per_seed"]["1"]["placement_success_rate"]["graph_recurrence_interaction"] == 0.3


def test_dynamic_eval_uses_static_run_layout_and_exposes_incomplete_override() -> None:
    static_root = Path("artifacts/static_2x2_convergence")

    assert static_run_dir_for(static_root, "graph_gru", 1601) == static_root / "runs" / "graph_gru_seed1601"
    assert checkpoint_path_for(static_root, "graph_gru", 1601) == static_root / "runs" / "graph_gru_seed1601" / "checkpoints" / "best.pt"
    assert "--allow-incomplete-static" in Path("mujoco_dynamic_2x2_eval.py").read_text(encoding="utf-8")
