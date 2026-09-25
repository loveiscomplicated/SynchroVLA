"""Frozen multi-seed objective replication contracts."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vla_gnn_recurrent.training import near_contact_translation_objective as original
from vla_gnn_recurrent.training import near_contact_translation_objective_multiseed as replication
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import recurrent_hidden_diagnostic as hd
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_representation_final_comparison as final


def test_frozen_objective_and_canonical_graph(tmp_path):
    config = rf._task_config(tmp_path, (2812,), "cpu", "cpu", 16)
    replication._require_canonical(config)
    assert (original.ALPHA, original.NEAR_LOW, original.NEAR_HIGH) == (2.0, -.020, .010)
    assert original.near_contact(np.array([-.020, .010]), np.array([True, True])).all()
    assert not original.near_contact(np.array([-.020001, .010001]), np.array([True, True])).any()


def test_weighted_mean_and_mask_preserve_baseline(tmp_path):
    config = rf._task_config(tmp_path, (2812,), "cpu", "cpu", 16)
    scales = torch.tensor([config.max_delta_ee] * 3 + [config.max_delta_rotation,
                                                         config.max_delta_gripper])
    pred = torch.stack([scales, 2 * scales]).reshape(1, 2, 5).requires_grad_()
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True, False]])
    near = torch.tensor([[True, True]])
    loss = original.weighted_sequence_loss(pred, target, mask, near, config)
    assert torch.allclose(loss, torch.tensor(1.0))
    loss.backward()
    assert pred.grad[0, 1].abs().sum() == 0
    assert torch.equal(original.weighted_sequence_loss(pred.detach(), target, mask,
                                                         torch.zeros_like(mask), config),
                       rf.sequence_loss(pred.detach(), target, mask, config))
    assert torch.equal(original.weighted_sequence_loss(pred.detach(), target, mask, near,
                                                         config, alpha=1),
                       rf.sequence_loss(pred.detach(), target, mask, config))


def test_shared_schedule_is_deterministic_and_uniform(tmp_path):
    config = rf._task_config(tmp_path, (2812,), "cpu", "cpu", 16)
    expert, _, _, _ = final._current_datasets(2812, config)
    # A small synthetic policy pool is sufficient to audit the draw stream.
    pool = {key: value.clone() for key, value in expert.items() if isinstance(value, torch.Tensor)}
    pool["loss_mask"] = torch.ones(len(pool["actions"]), dtype=torch.bool)
    pool_near = np.zeros(len(pool["actions"]), dtype=bool)
    expert_near = pool_near.copy()
    first, a = replication.build_schedule(2812, expert, pool, expert_near, pool_near, rf.TemporalConfig())
    second, b = replication.build_schedule(2812, expert, pool, expert_near, pool_near, rf.TemporalConfig())
    assert a == b
    assert a["uniform_policy_sampling"] and a["updates"] == 800
    assert a["schedule_tensor_sha256"] == b["schedule_tensor_sha256"]
    assert torch.equal(first[0]["mask"], second[0]["mask"])
    assert all(not torch.any(batch["near"] & ~batch["mask"]) for batch in first)


def test_branch_training_has_one_start_and_only_loss_differs():
    source = inspect.getsource(replication.train_branch)
    assert source.count("rf._load_gru(start") == 1
    assert "set_seed(seed + 1_300_001)" in source
    assert "rf.mixed_sequence_loss" in source
    assert "objective.mixed_weighted_loss" in source
    assert "for update, cpu_batch in enumerate(batches, 1)" in source
    assert '"unweighted_mse"' in source


def test_fresh_evaluation_uses_one_call_site_and_no_expert():
    source = inspect.getsource(replication.evaluate)
    assert source.count("rf.rollout(") == 1
    assert 'mode="fresh", severity=0, collect=False, record_dynamics=True' in source
    assert "for name, model in models.items()" in source


def test_heldout_train_specs_disjoint(tmp_path):
    config = rf._task_config(tmp_path, (2812,), "cpu", "cpu", 16)
    _, _, train, _ = final._current_datasets(2812, config)
    heldout = sf.sample_episode_specs(16, 2812 + 60_000, config, "iid")
    assert not {s.sample_identity for s in train} & {s.sample_identity for s in heldout}
    assert len(heldout) == 16


@pytest.mark.parametrize("seed", (2812, 2813))
def test_completed_artifacts_have_identical_starts_schedules_and_masks(seed):
    root = replication.OUTPUT / f"seed{seed}"
    if not (root / "summary.json").exists():
        pytest.skip("Replication artifacts have not been generated yet")
    summary = json.loads((root / "summary.json").read_text())
    m, t = (summary["training"][name] for name in replication.BRANCHES)
    assert m["starting_model_sha256"] == t["starting_model_sha256"]
    assert m["schedule_tensor_sha256"] == t["schedule_tensor_sha256"]
    assert m["exposure"] == t["exposure"]
    assert summary["config"]["alpha"] == 2.0
    assert summary["config"]["near_contact_range_m"] == [-.020, .010]
    assert summary["schedule"]["policy_valid_supervised_timesteps"] > 0
    assert summary["schedule"]["near_contact_supervised_timesteps"] > 0
    assert summary["schedule"]["ordinary_supervised_timesteps"] > 0
    assert summary["fixed_sequence"]["baseline_mse"]["policy_input_sha256"] == \
           summary["fixed_sequence"]["translation_weighted"]["policy_input_sha256"]
    expected = [r["spec_signature"] for r in json.loads((root / "fresh_eval" / "baseline_mse" / "episodes.json").read_text())]
    actual = [r["spec_signature"] for r in json.loads((root / "fresh_eval" / "translation_weighted" / "episodes.json").read_text())]
    assert expected == actual


def test_seed2811_history_untouched():
    summary = json.loads((original.OUTPUT / "summary.json").read_text())
    assert summary["seed"] == 2811
    assert hd._sha256(Path(summary["weighted_training"]["checkpoint"])) == \
           summary["weighted_training"]["checkpoint_sha256"]
    if (replication.OUTPUT / "aggregate" / "summary.json").exists():
        aggregate = json.loads((replication.OUTPUT / "aggregate" / "summary.json").read_text())
        for path, expected in aggregate["reference_sha256"].items():
            assert hd._sha256(Path(path)) == expected
