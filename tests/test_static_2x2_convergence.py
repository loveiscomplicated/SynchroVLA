import json
import sys
from pathlib import Path

from mujoco_static_2x2_convergence import (
    MODEL_KINDS,
    MODEL_ORDER,
    chunk_sequence,
    completed_run,
    make_eval_episode_plan,
    run_child_and_wait,
    validation_score,
)


def test_convergence_model_set_is_static_2x2_only() -> None:
    assert tuple(MODEL_ORDER) == ("flat_ff", "flat_gru", "graph_ff", "graph_gru")
    assert MODEL_KINDS == {
        "flat_ff": "flat_feedforward_dir_mag",
        "flat_gru": "flat_recurrent_dir_mag",
        "graph_ff": "graph_feedforward_dir_mag",
        "graph_gru": "graph_recurrent_dir_mag",
    }


def test_validation_score_is_lexicographic_with_offline_loss_last() -> None:
    lower_loss_only = validation_score(
        {
            "placement_success_rate": 0.1,
            "successful_placement_release_rate": 0.1,
            "valid_release_rate": 1.0,
            "transport_success_rate": 1.0,
            "lift_success_rate": 1.0,
            "grasp_success_rate": 1.0,
        },
        {"motion_l2_error": 0.001},
    )
    better_placement = validation_score(
        {
            "placement_success_rate": 0.2,
            "successful_placement_release_rate": 0.0,
            "valid_release_rate": 0.0,
            "transport_success_rate": 0.0,
            "lift_success_rate": 0.0,
            "grasp_success_rate": 0.0,
        },
        {"motion_l2_error": 9.0},
    )
    same_closed_loop_lower_loss = validation_score(
        {
            "placement_success_rate": 0.1,
            "successful_placement_release_rate": 0.1,
            "valid_release_rate": 1.0,
            "transport_success_rate": 1.0,
            "lift_success_rate": 1.0,
            "grasp_success_rate": 1.0,
        },
        {"motion_l2_error": 0.01},
    )

    assert better_placement > lower_loss_only
    assert lower_loss_only > same_closed_loop_lower_loss


def test_completed_run_requires_explicit_result_and_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "flat_ff_seed1601"
    run_dir.mkdir()
    (run_dir / "run_result.json").write_text(json.dumps({"completed": True}), encoding="utf-8")
    assert not completed_run(run_dir)

    (run_dir / "run_result.json").write_text(
        json.dumps({"completed": True, "best_checkpoint_path": str(run_dir / "missing.pt")}),
        encoding="utf-8",
    )
    assert not completed_run(run_dir)

    (run_dir / "best.pt").write_bytes(b"checkpoint")
    (run_dir / "run_result.json").write_text(
        json.dumps({"completed": True, "best_checkpoint_path": str(run_dir / "best.pt")}),
        encoding="utf-8",
    )
    assert completed_run(run_dir)


def test_run_child_and_wait_blocks_and_records_completion_metadata(tmp_path: Path) -> None:
    stdout_log = tmp_path / "stdout.log"
    stderr_log = tmp_path / "stderr.log"
    metadata_path = tmp_path / "metadata.json"
    exit_code = run_child_and_wait(
        [sys.executable, "-c", "print('done')"],
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        metadata_path=metadata_path,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert stdout_log.read_text(encoding="utf-8").strip() == "done"
    assert metadata["exit_code"] == 0
    assert metadata["stdout_log"] == str(stdout_log)
    assert metadata["stderr_log"] == str(stderr_log)
    assert metadata["duration_sec"] >= 0.0


def test_convergence_runner_has_no_sleep_based_polling() -> None:
    source = Path("mujoco_static_2x2_convergence.py").read_text(encoding="utf-8")
    assert "sleep(" not in source
    assert ".poll(" not in source


def test_eval_episode_plan_is_deterministic_disjoint_and_ordered() -> None:
    plan_a = make_eval_episode_plan(episodes=5, seed=2601, max_steps=420)
    plan_b = make_eval_episode_plan(episodes=5, seed=2601, max_steps=420)

    assert plan_a == plan_b
    assert [item["episode"] for item in plan_a] == [1, 2, 3, 4, 5]
    assert len({item["episode_seed"] for item in plan_a}) == 5


def test_eval_chunking_preserves_every_episode_once() -> None:
    plan = make_eval_episode_plan(episodes=7, seed=2601, max_steps=420)
    chunks = chunk_sequence(plan, chunks=3)
    flattened = sorted((item for chunk in chunks for item in chunk), key=lambda item: item["episode_idx"])

    assert flattened == plan
    assert len(chunks) == 3
    assert sum(len(chunk) for chunk in chunks) == len(plan)


def test_convergence_cli_exposes_cuda_and_parallel_eval_options() -> None:
    source = Path("mujoco_static_2x2_convergence.py").read_text(encoding="utf-8")

    assert '"cuda"' in source
    assert "--eval-device" in source
    assert "--eval-workers" in source
