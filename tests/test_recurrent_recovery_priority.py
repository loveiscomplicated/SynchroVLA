"""Sampling-composition checks for the fixed recurrent recovery pool."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import recurrent_recovery_priority as priority
from vla_gnn_recurrent.training import surface_graph_feasibility as sf


def _source(episode_id: int, masked: bool = False):
    data = {"states": torch.zeros(3, sf.STATE_DIM), "surface_points": torch.zeros(3, 32, 3),
            "actions": torch.zeros(3, sf.ACTION_DIM),
            "episode_ids": torch.full((3,), episode_id), "steps": torch.arange(3),
            "loss_mask": torch.tensor([True, not masked, True])}
    metadata = [{"episode_id": episode_id, "t": t, "clearance_m": -.03 if masked and t == 1 else .02,
                 "collision": False, "severe_penetration_excluded": bool(masked and t == 1)}
                for t in range(3)]
    return data, metadata


def test_common_pool_preserves_source_sequences_and_masked_context() -> None:
    old, old_meta = _source(0, masked=True)
    new, new_meta = _source(0)
    pool, metadata = priority.combine_pool(old, new, old_meta, new_meta)
    episodes = rf._episodes(pool)
    assert len(episodes) == 2
    assert [ep["steps"].tolist() for ep in episodes] == [[0, 1, 2], [0, 1, 2]]
    assert episodes[0]["loss_mask"].tolist() == [True, False, True]
    assert [r["pool_episode_id"] for r in metadata] == [0, 0, 0, 1, 1, 1]
    with pytest.raises(ValueError):
        priority.combine_pool(old, new, old_meta, [{**r, "t": r["t"] + 1} for r in new_meta])


def test_difficulty_labels_deterministic_and_exclude_masked_supervision() -> None:
    config = replace(sf.SurfaceFeasibilityConfig(), symmetric_robot_edges=True)
    specs = sf.sample_episode_specs(1, 19, config, "train")
    data, metadata = _source(specs[0].episode_id, masked=True)
    pool, rows = priority.combine_pool(data, data, metadata, metadata)
    first = priority.difficulty_rows(pool, rows, specs, config)
    second = priority.difficulty_rows(pool, rows, specs, config)
    assert first == second
    assert first[1]["priority_score"] == 0
    assert all(not flag for flag in first[1]["categories"].values())


def test_priority_weights_ignore_old_new_provenance_and_cover_every_trajectory() -> None:
    rows = []
    for episode in range(40):
        for t in range(3):
            late = episode < 10
            rows.append({"pool_episode_id": episode, "source_dataset_id": "old_base_gru", "eligible": True,
                         "priority_score": 2 if late else 0,
                         "categories": {"ordinary": not late, "late": late, "near_contact": late,
                                        "pre_collision": False, "recovery": False}})
    weights, audit = priority.sampling_weights(rows)
    swapped = [{**row, "source_dataset_id": "new_final_gru"} for row in rows]
    other, _ = priority.sampling_weights(swapped)
    assert np.array_equal(weights, other)
    assert np.isclose(weights.sum(), 1)
    assert np.all(weights > 0)
    assert audit["priority_trajectory_count"] == 10
    assert weights[0] > weights[-1]


def test_weighted_batch_preserves_order_and_eligibility_mask() -> None:
    old, old_meta = _source(0, masked=True)
    new, new_meta = _source(0)
    pool, _ = priority.combine_pool(old, new, old_meta, new_meta)
    episodes = rf._episodes(pool)
    picks = []
    states, points, actions, mask = rf.sample_sequence_batch(
        episodes, 6, 32, np.random.default_rng(7), torch.device("cpu"),
        episode_weights=np.array([1., 0.]), picks_out=picks)
    assert picks == [0] * 6
    assert states.shape == (6, 3, sf.STATE_DIM) and points.shape == (6, 3, 32, 3)
    assert actions.shape == (6, 3, sf.ACTION_DIM)
    assert mask.tolist() == [[True, False, True]] * 6


def test_training_budgets_and_topology_remain_fixed(tmp_path) -> None:
    config = replace(sf.SurfaceFeasibilityConfig(output_dir=str(tmp_path)), symmetric_robot_edges=True)
    temporal = rf.TemporalConfig(sequence_batch_size=4)
    old, old_meta = _source(0, masked=True)
    new, new_meta = _source(0)
    pool, _ = priority.combine_pool(old, new, old_meta, new_meta)
    start = rf.GraphGRUPolicy(config).state_dict()
    results = []
    for name, weights in (("uniform", None), ("recovery", np.array([.75, .25]))):
        model = rf.GraphGRUPolicy(config)
        model.load_state_dict(start)
        assert not model.graph.use_local_edges
        _, _, topology = rf._graph_tensors(np.zeros(sf.STATE_DIM), np.zeros((32, 3)), config,
                                            torch.device("cpu"))
        rf.assert_canonical_topology(topology, config)
        result = rf.train_gru(model, old, old, pool, config, temporal, 2811, torch.device("cpu"),
                              2, tmp_path / name, sampling_seed=99, validation_interval=2,
                              policy_episode_weights=weights, separate_source_rngs=True)
        results.append(result)
    assert [result["updates"] for result in results] == [2, 2]
    assert all(result["expert_policy_sequence_fraction"] == [.5, .5] for result in results)
    assert all(result["expert_valid_supervised_timesteps"] > 0 for result in results)
    assert all(result["policy_valid_supervised_timesteps"] > 0 for result in results)
    assert results[0]["sampled_expert_episode_index_sha256"] == results[1]["sampled_expert_episode_index_sha256"]
    assert results[0]["expert_valid_supervised_timesteps"] == results[1]["expert_valid_supervised_timesteps"]


def test_heldout_specs_are_the_existing_gate2_set() -> None:
    config = rf._task_config(priority.OUTPUT, (2811,), "cpu", "cpu", 16)
    specs = sf.sample_episode_specs(16, 2811 + 60_000, config, "iid")
    priority.previous._verify_eval_specs(specs, priority.previous.GATE2)
