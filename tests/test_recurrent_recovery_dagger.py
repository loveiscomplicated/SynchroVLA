"""Checks for the controlled three-branch recurrent recovery continuation."""

from dataclasses import replace

import pytest
import torch

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import recurrent_hidden_diagnostic as hd
from vla_gnn_recurrent.training import recurrent_recovery_dagger as recovery
from vla_gnn_recurrent.training import surface_graph_feasibility as sf


def _tiny_data(mask: torch.Tensor | None = None) -> dict:
    result = {
        "states": torch.zeros(4, sf.STATE_DIM),
        "surface_points": torch.zeros(4, 32, 3),
        "actions": torch.zeros(4, sf.ACTION_DIM),
        "episode_ids": torch.tensor([0, 0, 1, 1]),
        "steps": torch.tensor([0, 1, 0, 1]),
    }
    if mask is not None:
        result["loss_mask"] = mask
    return result


def test_branch_sources_are_disjoint_and_expert_branch_has_no_policy_supervision() -> None:
    new = _tiny_data(torch.tensor([True, False, True, True]))
    old = _tiny_data(torch.tensor([True, True, True, True]))
    assert recovery.branch_policy_data(recovery.BRANCHES[0], new, old) is None
    assert recovery.branch_policy_data(recovery.BRANCHES[1], new, old) is new
    assert recovery.branch_policy_data(recovery.BRANCHES[2], new, old) is old
    with pytest.raises(ValueError):
        recovery.branch_policy_data("unknown", new, old)


def test_policy_dataset_keeps_masked_context_and_rejects_gap_or_wrong_split() -> None:
    config = replace(sf.SurfaceFeasibilityConfig(), symmetric_robot_edges=True)
    specs = sf.sample_episode_specs(2, 17, config, "train")
    data = _tiny_data(torch.tensor([True, False, True, True]))
    data["episode_ids"] = torch.tensor([specs[0].episode_id] * 2 + [specs[1].episode_id] * 2)
    audit = recovery.validate_policy_data(data, specs, config)
    assert audit["visited"] == 4 and audit["eligible"] == 3 and audit["masked"] == 1
    assert audit["sequence_lengths"] == [2, 2]
    assert [ep["loss_mask"].tolist() for ep in rf._episodes(data)] == [[True, False], [True, True]]
    bad_steps = {**data, "steps": torch.tensor([0, 2, 0, 1])}
    with pytest.raises(ValueError):
        recovery.validate_policy_data(bad_steps, specs, config)
    with pytest.raises(ValueError):
        recovery.validate_policy_data(data, specs[:1], config)
    with pytest.raises(ValueError):
        recovery.validate_policy_data(data, specs, replace(config, symmetric_robot_edges=False))


def test_three_continuations_start_from_identical_checkpoint_weights(tmp_path) -> None:
    config = replace(sf.SurfaceFeasibilityConfig(), symmetric_robot_edges=True)
    model = rf.GraphGRUPolicy(config)
    checkpoint = tmp_path / "start.pt"
    torch.save({"model_state": model.state_dict(), "seed": 2811, "topology": "consistent_intended_100"}, checkpoint)
    file_hash = hd._sha256(checkpoint)
    expected = hd._model_digest(model)
    for _ in recovery.BRANCHES:
        loaded = rf._load_gru(checkpoint, config, rf.TemporalConfig(), torch.device("cpu"))
        assert hd._model_digest(loaded) == expected
        assert loaded.graph.use_local_edges is False
        state, points, topology = rf._graph_tensors(
            torch.zeros(sf.STATE_DIM).numpy(), torch.zeros(32, 3).numpy(), config, torch.device("cpu"))
        rf.assert_canonical_topology(topology, config)
        assert state.shape == (1, sf.STATE_DIM) and points.shape == (1, 32, 3)
    assert hd._sha256(checkpoint) == file_hash


def test_matched_update_budget_and_masked_loss_in_all_branches(tmp_path) -> None:
    config = replace(sf.SurfaceFeasibilityConfig(output_dir=str(tmp_path)), symmetric_robot_edges=True)
    temporal = rf.TemporalConfig(sequence_batch_size=4)
    expert = _tiny_data()
    new = _tiny_data(torch.tensor([True, False, True, True]))
    old = _tiny_data(torch.tensor([True, True, False, True]))
    start = rf.GraphGRUPolicy(config).state_dict()
    counts = []
    for branch in recovery.BRANCHES:
        model = rf.GraphGRUPolicy(config)
        model.load_state_dict(start)
        result = rf.train_gru(model, expert, expert, recovery.branch_policy_data(branch, new, old),
                              config, temporal, 2811, torch.device("cpu"), 2, tmp_path / branch,
                              sampling_seed=991, validation_interval=2, stage_label=branch)
        counts.append(result["updates"])
        assert result["sampling_seed"] == 991
        assert result["validation_interval_updates"] == 2
        assert result["expert_valid_supervised_timesteps"] > 0
        if branch == recovery.BRANCHES[0]:
            assert result["policy_valid_supervised_timesteps"] == 0
        else:
            assert result["policy_valid_supervised_timesteps"] > 0
    assert counts == [2, 2, 2]


def test_heldout_specs_match_gate2_artifact() -> None:
    config = rf._task_config(recovery.OUTPUT, (2811,), "cpu", "cpu", 16)
    specs = sf.sample_episode_specs(16, 2811 + 60_000, config, "iid")
    recovery._verify_eval_specs(specs, recovery.GATE2)
