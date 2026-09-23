from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from torch import nn

from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_pregrasp_onpolicy_relabel as dagger
from vla_gnn_recurrent.training import surface_pregrasp_stabilization as stab


@pytest.fixture
def tiny_config(tmp_path: Path) -> sf.SurfaceFeasibilityConfig:
    return sf.SurfaceFeasibilityConfig(
        output_dir=str(tmp_path), train_episodes=1, validation_episodes=1, eval_episodes=1,
        max_steps=3, batch_size=4, bootstrap_samples=20,
    )


@pytest.fixture
def collected_policy(tiny_config: sf.SurfaceFeasibilityConfig, tmp_path: Path, monkeypatch):
    specs = sf.sample_episode_specs(1, 91_771, tiny_config, "train")
    expert_data, reference = stab.collect_supervised_rows(tiny_config, specs, 91_772, tmp_path / "expert")

    def fixed_policy(_name, _model, _state, _points, _config, _device):
        return np.asarray([-0.030, 0.0, 0.0, -0.30, -0.20], dtype=np.float32)

    monkeypatch.setattr(sf, "_policy_action", fixed_policy)
    model = nn.Identity()
    data, summary = dagger.collect_on_policy_states(
        model, specs, reference, tiny_config, 91_773, torch.device("cpu"), tmp_path / "onpolicy",
    )
    return specs, expert_data, data, summary, tmp_path / "onpolicy"


def test_training_episode_specs_are_reproducible_and_training_only(tiny_config):
    a = sf.sample_episode_specs(3, 447, tiny_config, "train")
    b = sf.sample_episode_specs(3, 447, tiny_config, "train")
    assert [sf.asdict(x) for x in a] == [sf.asdict(x) for x in b]
    assert all(spec.condition == "train" for spec in a)


def test_onpolicy_collection_rejects_validation_test_specs(tiny_config, tmp_path):
    spec = sf.sample_episode_specs(1, 881, tiny_config, "validation")[0]
    with pytest.raises(ValueError, match="training-split"):
        dagger.collect_on_policy_states(nn.Identity(), [spec], {}, tiny_config, 12,
                                       torch.device("cpu"), tmp_path)


def test_collected_state_is_live_policy_rollout_not_expert_replay(collected_policy):
    _specs, expert, policy, _summary, _out = collected_policy
    assert len(policy["steps"]) >= 2
    row = int(torch.where(policy["steps"] == 1)[0][0])
    expert_rows = torch.where((expert["episode_ids"] == policy["episode_ids"][row]) & (expert["steps"] == 1))[0]
    assert len(expert_rows) == 1
    assert not torch.allclose(policy["states"][row], expert["states"][expert_rows[0]], atol=1e-5)


def test_collected_live_qpos_reconstructs_the_stored_observation(collected_policy, tiny_config):
    specs, _expert, policy, _summary, _out = collected_policy
    row = int(torch.where(policy["steps"] == 1)[0][0])
    spec = specs[0]
    env = sf.make_env(tiny_config, 29)
    sf._reset_surface_env(env, spec)
    env.data.qpos[:] = policy["qpos"][row].numpy()
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    state, surface, _ = sf.observation_inputs(env, spec.shape, tiny_config.point_count,
                                              sf._stable_seed(spec.sample_identity))
    env.close()
    np.testing.assert_allclose(state, policy["states"][row].numpy(), atol=1e-6)
    np.testing.assert_allclose(surface, policy["surface_points"][row].numpy(), atol=1e-6)


def test_oracle_action_is_recomputed_at_the_collected_qpos(collected_policy, tiny_config):
    specs, _expert, policy, _summary, _out = collected_policy
    row = int(torch.where(policy["steps"] == 1)[0][0])
    env = sf.make_env(tiny_config, 31)
    sf._reset_surface_env(env, specs[0])
    env.data.qpos[:] = policy["qpos"][row].numpy()
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    fresh_action, _ = sf.expert_action(env, specs[0].shape, tiny_config)
    env.close()
    np.testing.assert_allclose(fresh_action, policy["actions"][row].numpy(), atol=1e-6)


def test_relabel_is_not_the_policy_action_or_original_expert_label(collected_policy):
    _specs, expert, policy, _summary, _out = collected_policy
    row = int(torch.where(policy["steps"] == 1)[0][0])
    original_idx = int(torch.where((expert["episode_ids"] == policy["episode_ids"][row]) &
                                   (expert["steps"] == policy["steps"][row]))[0][0])
    assert not torch.allclose(policy["actions"][row], policy["policy_actions"][row], atol=1e-5)
    assert not torch.allclose(policy["actions"][row], expert["actions"][original_idx], atol=1e-5)


def test_oracle_metadata_and_deviation_labels_are_not_model_inputs(collected_policy):
    _specs, _expert, policy, _summary, _out = collected_policy
    assert set(("states", "surface_points", "actions")).issubset(policy)
    assert "target_position" not in policy and "distance_to_target" not in policy
    assert "deviation_bucket" not in policy and "expert_action" not in policy
    # Inputs are only state and points; the separate actions array is a supervision target.
    assert policy["states"].shape[-1] == sf.STATE_DIM
    assert policy["surface_points"].shape[-1] == 3


def test_severe_penetration_exclusion_is_deterministic():
    assert dagger.is_severe_penetration(-0.020001)
    assert not dagger.is_severe_penetration(-0.020)
    assert not dagger.is_severe_penetration(0.0)


def test_difficulty_bucket_rules_are_stable():
    assert dagger.classify_deviation(0.005, False, 0.020) == "near_expert"
    assert dagger.classify_deviation(0.020, False, 0.020) == "moderate_deviation"
    assert dagger.classify_deviation(0.050, False, 0.020) == "large_deviation"
    assert dagger.classify_deviation(0.050, True, 0.020) == "collision_near_collision"


def _save_base_checkpoint(path: Path, config: sf.SurfaceFeasibilityConfig, seed: int = 9) -> dict[str, torch.Tensor]:
    model = sf.build_surface_model("surface_set", config)
    torch.save({"model_name": "surface_set", "model_state": model.state_dict(),
                "config": sf.asdict(config) if hasattr(sf, "asdict") else {}, "seed": seed}, path)
    return model.state_dict()


def _tiny_data(seed: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "states": torch.randn((4, sf.STATE_DIM), generator=generator),
        "surface_points": torch.randn((4, sf.SURFACE_POINTS, 3), generator=generator),
        "actions": torch.randn((4, sf.ACTION_DIM), generator=generator) * 0.01,
    }


def test_continuations_use_same_start_checkpoint_and_update_budget(tiny_config, tmp_path):
    checkpoint = tmp_path / "base.pt"
    _save_base_checkpoint(checkpoint, tiny_config)
    expert = _tiny_data(2)
    validation = _tiny_data(3)
    policy = _tiny_data(4)
    metadata = [{"deviation_bucket": "near_expert"} for _ in range(4)]
    extra = dagger.continuation_train("extra_original", checkpoint, expert, None, validation, None,
                                      tiny_config, 34, 2, "cpu", tmp_path / "extra")
    mixed = dagger.continuation_train("dagger_round1", checkpoint, expert, policy, validation, metadata,
                                      tiny_config, 34, 2, "cpu", tmp_path / "dagger")
    assert extra["optimizer_updates"] == mixed["optimizer_updates"] == 2
    assert extra["base_checkpoint_sha256"] == mixed["base_checkpoint_sha256"]
    assert extra["effective_expert_draws"] == 8 and extra["effective_policy_draws"] == 0
    assert mixed["effective_expert_draws"] == mixed["effective_policy_draws"] == 4


@pytest.mark.parametrize("batch_size", [2, 4, 128])
def test_configured_dagger_ratio_is_half_expert_half_policy(batch_size):
    expert, policy = dagger.continuation_batch_counts(batch_size)
    assert expert + policy == batch_size
    assert expert / batch_size == pytest.approx(0.5)
    assert policy / batch_size == pytest.approx(0.5)


def test_paired_comparison_requires_same_episode_ids(tiny_config):
    row = {"episode_id": 0, "success": 0.0, "collision": 0.0,
           "final_position_error": 0.1, "final_orientation_error": 0.2,
           "final_gripper_width_error": 0.01, "trajectory_error": 0.1,
           "trajectory_yaw_error": 0.2, "trajectory_aperture_error": 0.01,
           "trajectory_length": 0.1}
    dagger.paired_comparison([row], [dict(row)], tiny_config, "same", 3)
    with pytest.raises(ValueError, match="exact same EpisodeSpec"):
        dagger.paired_comparison([row], [dict(row, episode_id=1)], tiny_config, "mismatch", 3)


def test_base_checkpoint_hash_is_stable_across_loads(tiny_config, tmp_path):
    checkpoint = tmp_path / "base.pt"
    _save_base_checkpoint(checkpoint, tiny_config)
    a = sf.load_surface_model(checkpoint, "surface_set", tiny_config, torch.device("cpu"))
    b = sf.load_surface_model(checkpoint, "surface_set", tiny_config, torch.device("cpu"))
    assert dagger._state_dict_hash(a) == dagger._state_dict_hash(b)

