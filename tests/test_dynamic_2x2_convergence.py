from pathlib import Path

from mujoco_dynamic_2x2_convergence import _episode_split, dynamic_validation_score
from mujoco_static_2x2_convergence import MODEL_KINDS, MODEL_ORDER


def test_dynamic_2x2_model_set_matches_static_factorial_design() -> None:
    assert tuple(MODEL_ORDER) == ("flat_ff", "flat_gru", "graph_ff", "graph_gru")
    assert MODEL_KINDS == {
        "flat_ff": "flat_feedforward_dir_mag",
        "flat_gru": "flat_recurrent_dir_mag",
        "graph_ff": "graph_feedforward_dir_mag",
        "graph_gru": "graph_recurrent_dir_mag",
    }


def test_dynamic_validation_score_prioritizes_placement_over_recovery_and_loss() -> None:
    high_recovery_lower_placement = dynamic_validation_score(
        {
            "placement_success_rate": 0.1,
            "successful_placement_release_rate": 1.0,
            "valid_release_rate": 1.0,
            "dynamic_recovery_rate": 1.0,
            "transport_success_rate": 1.0,
            "lift_success_rate": 1.0,
            "grasp_success_rate": 1.0,
        },
        {"motion_l2_error": 0.001},
    )
    better_placement = dynamic_validation_score(
        {
            "placement_success_rate": 0.2,
            "successful_placement_release_rate": 0.0,
            "valid_release_rate": 0.0,
            "dynamic_recovery_rate": 0.0,
            "transport_success_rate": 0.0,
            "lift_success_rate": 0.0,
            "grasp_success_rate": 0.0,
        },
        {"motion_l2_error": 9.0},
    )

    assert better_placement > high_recovery_lower_placement


def test_dynamic_episode_split_is_episode_disjoint_and_deterministic() -> None:
    split_a = _episode_split(20, seed=4301)
    split_b = _episode_split(20, seed=4301)
    all_ids = split_a["train"] + split_a["val"] + split_a["test"]

    assert split_a == split_b
    assert sorted(all_ids) == list(range(20))
    assert len(all_ids) == len(set(all_ids))


def test_dynamic_convergence_cli_exposes_dataset_generation_cuda_and_parallel_eval() -> None:
    source = Path("mujoco_dynamic_2x2_convergence.py").read_text(encoding="utf-8")

    assert "generate-demos" in source
    assert '"cuda"' in source
    assert "--eval-workers" in source
    assert "--generate-dataset-if-missing" in source
