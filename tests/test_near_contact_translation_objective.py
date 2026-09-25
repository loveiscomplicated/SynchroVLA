"""Contracts for the seed-2811 state-dependent component-weight ablation."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vla_gnn_recurrent.training import near_contact_translation_objective as objective
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_representation_final_comparison as final
from vla_gnn_recurrent.training import recurrent_hidden_diagnostic as hd


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    config = rf._task_config(tmp_path_factory.mktemp("near_objective"), (2811,), "cpu", "cpu", 16)
    expert, validation, train_specs, val_specs = final._current_datasets(2811, config)
    pool = torch.load(objective.CONTROLLED / "policy_pool.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((objective.CONTROLLED / "policy_pool_metadata.json").read_text())
    historical = json.loads((objective.CONTROLLED / "uniform/training.json").read_text())
    return config, expert, validation, train_specs, val_specs, pool, metadata, historical


@pytest.fixture(scope="module")
def schedule(sources):
    config, expert, validation, train_specs, _, pool, metadata, historical = sources
    expert_clearance = objective.expert_clearance(expert, train_specs, config, 2902)
    expert_near = objective.near_contact(expert_clearance, np.ones(len(expert_clearance), dtype=bool))
    policy_near = objective.near_contact(
        np.asarray([row["clearance_m"] for row in metadata]), pool["loss_mask"].numpy())
    return objective.build_schedule(expert, pool, expert_near, policy_near, rf.TemporalConfig(), historical)


def test_both_branches_have_identical_gate2_starting_weights(sources):
    config = sources[0]
    controlled = json.loads((objective.CONTROLLED / "config.json").read_text())
    start = Path(controlled["starting_checkpoint"])
    assert hd._sha256(start) == controlled["starting_checkpoint_sha256"]
    first = rf._load_gru(start, config, rf.TemporalConfig(), torch.device("cpu"))
    second = rf._load_gru(start, config, rf.TemporalConfig(), torch.device("cpu"))
    assert hd._model_digest(first) == hd._model_digest(second) == controlled["starting_model_sha256"]


def test_exact_uniform_episode_and_window_draws_match_historical(schedule, sources):
    _, audit = schedule
    historical = sources[-1]
    assert audit["historical_uniform_draw_replay_verified"]
    assert audit["expert_episode_index_sha256"] == historical["sampled_expert_episode_index_sha256"]
    assert audit["all_windows_are_complete_episodes"]
    assert audit["updates"] == historical["updates"] == 800


def test_valid_masks_and_exposure_are_shared(schedule, sources):
    batches, audit = schedule
    historical = sources[-1]
    assert sum(int(b["mask"][:8].sum()) for b in batches) == historical["expert_valid_supervised_timesteps"]
    assert sum(int(b["mask"][8:].sum()) for b in batches) == historical["policy_valid_supervised_timesteps"]
    assert sum(int(b["near"].sum()) for b in batches) == audit["near_contact_supervised_timesteps"]
    assert all(not torch.any(b["near"] & ~b["mask"]) for b in batches)


def test_near_contact_labels_are_deterministic_and_match_existing_categories(sources):
    config, expert, _, train_specs, _, pool, metadata, _ = sources
    clearance_a = objective.expert_clearance(expert, train_specs, config, 2904)
    clearance_b = objective.expert_clearance(expert, train_specs, config, 2905)
    assert np.array_equal(clearance_a, clearance_b)
    labels = objective.near_contact(np.asarray([r["clearance_m"] for r in metadata]), pool["loss_mask"].numpy())
    assert np.array_equal(labels, np.asarray([r["categories"]["near_contact"] for r in metadata]))
    assert not objective.near_contact(np.array([-.020001, .010001]), np.array([True, True])).any()
    assert objective.near_contact(np.array([-.020, .010]), np.array([True, True])).all()


def test_clearance_is_loss_metadata_and_not_model_input(sources, schedule):
    config = sources[0]
    batch = schedule[0][0]
    assert batch["states"].shape[-1] == sf.STATE_DIM == 14
    assert batch["points"].shape[-2:] == (32, 3)
    controlled = json.loads((objective.CONTROLLED / "config.json").read_text())
    model = rf._load_gru(Path(controlled["starting_checkpoint"]), config, rf.TemporalConfig(), torch.device("cpu"))
    with torch.no_grad():
        first, _ = model.forward_sequence(batch["states"], batch["points"])
        second, _ = model.forward_sequence(batch["states"].clone(), batch["points"].clone())
    assert torch.equal(first, second)
    assert "clearance" not in inspect.signature(model.forward_sequence).parameters
    assert batch["near"].shape == batch["mask"].shape


def test_masked_severe_states_have_zero_loss_gradient(sources):
    config = sources[0]
    pred = torch.tensor([[[.01, 0, 0, 0, 0], [.02, 0, 0, 0, 0]]], requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True, False]])
    near = torch.tensor([[False, True]])
    objective.weighted_sequence_loss(pred, target, mask, near, config).backward()
    assert pred.grad[0, 1].abs().sum() == 0
    assert pred.grad[0, 0].abs().sum() > 0


def test_ordinary_states_reproduce_original_loss_exactly(sources):
    config = sources[0]
    pred = torch.randn(2, 3, 5)
    target = torch.randn(2, 3, 5)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    near = torch.zeros_like(mask)
    assert torch.equal(objective.weighted_sequence_loss(pred, target, mask, near, config),
                       rf.sequence_loss(pred, target, mask, config))


def test_alpha_one_reproduces_baseline_mse(sources):
    config = sources[0]
    pred = torch.randn(2, 3, 5)
    target = torch.randn(2, 3, 5)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    near = torch.tensor([[True, False, False], [True, True, False]])
    assert torch.equal(objective.weighted_sequence_loss(pred, target, mask, near, config, alpha=1),
                       rf.sequence_loss(pred, target, mask, config))


def test_weighted_loss_is_normalized_component_mean(sources):
    config = sources[0]
    scales = torch.tensor([config.max_delta_ee] * 3 + [config.max_delta_rotation, config.max_delta_gripper])
    pred = scales.reshape(1, 1, 5)
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True]])
    near = torch.tensor([[True]])
    assert torch.allclose(objective.weighted_sequence_loss(pred, target, mask, near, config), torch.tensor(1.0))
    pred[..., 0] = 2 * scales[0]
    assert torch.allclose(objective.weighted_sequence_loss(pred, target, mask, near, config), torch.tensor(14 / 8))


def test_both_models_use_one_fresh_evaluation_call_site():
    source = inspect.getsource(objective.fresh_evaluation)
    assert source.count("rf.rollout(") == 1
    assert "for name, model in models.items()" in source
    assert 'mode="fresh", severity=0, collect=False, record_dynamics=True' in source


def test_heldout_episode_specs_match_existing_controlled_run(sources):
    config = sources[0]
    specs = sf.sample_episode_specs(16, 2811 + 60_000, config, "iid")
    expected = [r["spec_signature"] for r in json.loads((objective.CONTROLLED / "fresh_eval/uniform/episodes.json").read_text())]
    assert [final.spec_signature(s) for s in specs] == expected


def test_canonical_100_edge_topology_is_unchanged(sources):
    config = sources[0]
    state = torch.zeros(1, sf.STATE_DIM)
    points = torch.zeros(1, 32, 3)
    topology = sf._torch_topology(points, state, config, use_local_edges=False)
    assert topology.src.shape[1] == 100
    assert not torch.any(topology.edge_type == 3)


def test_existing_artifacts_are_guarded_against_overwrite():
    old = objective.CONTROLLED / "uniform/gru.pt"
    before = hashlib.sha256(old.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        objective.run(output=objective.CONTROLLED)
    assert hashlib.sha256(old.read_bytes()).hexdigest() == before
