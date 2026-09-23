from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_pregrasp_stabilization as stab
from vla_gnn_recurrent.training import surface_representation_final_comparison as final


@pytest.fixture
def tiny_config(tmp_path: Path) -> sf.SurfaceFeasibilityConfig:
    return sf.SurfaceFeasibilityConfig(
        output_dir=str(tmp_path), train_episodes=1, validation_episodes=1, eval_episodes=1,
        max_steps=2, batch_size=4, bootstrap_samples=30,
    )


def test_failure_decomposition_matches_strict_success_conjunction(tiny_config):
    rows = []
    for collision in (False, True):
        for position in (False, True):
            for yaw in (False, True):
                for aperture in (False, True):
                    rows.append({
                        "episode_id": len(rows), "trajectory_collision": collision,
                        "final_position_error": 0.03 if position else 0.024,
                        "final_orientation_error": 0.21 if yaw else 0.19,
                        "final_gripper_width_error": 0.009 if aperture else 0.007,
                    })
    result = final.failure_decomposition(rows, tiny_config)
    assert len(result["all_16_failure_combinations"]) == 16
    assert result["failed_episodes"] == 15
    assert result["all_16_failure_combinations"]["collision_failure+position_failure+orientation_failure+aperture_failure"]["count"] == 1
    assert result["all_16_failure_combinations"]["success_all_conditions_pass"]["count"] == 1
    assert result["failure_conditions"]["aperture_failure"]["episode_count"] == 8


def test_paired_episode_validation_rejects_different_initial_states(tiny_config):
    a = sf.sample_episode_specs(2, 12, tiny_config, "iid")
    b = list(a)
    b[1] = replace(b[1], gripper_command=b[1].gripper_command + 0.01)
    with pytest.raises(ValueError, match="exactly the same"):
        final.assert_same_specs({"surface_set": a, "surface_graph": b})


def test_dagger_collection_accepts_training_specs_only(tiny_config):
    train = sf.sample_episode_specs(1, 2, tiny_config, "train")
    val = sf.sample_episode_specs(1, 3, tiny_config, "validation")
    final.validate_train_specs(train)
    with pytest.raises(ValueError, match="training-split"):
        final.validate_train_specs(val)


def test_self_policy_collection_stores_live_states_and_fresh_oracle_labels(tiny_config, tmp_path, monkeypatch):
    specs = sf.sample_episode_specs(1, 71_200, tiny_config, "train")
    expert, reference = stab.collect_supervised_rows(tiny_config, specs, 71_201, tmp_path / "expert")

    def fixed_policy(_name, _model, _state, _points, _config, _device):
        return np.asarray([-0.030, 0.0, 0.0, -0.30, -0.20], dtype=np.float32)

    monkeypatch.setattr(sf, "_policy_action", fixed_policy)
    policy, metadata, stats = final.collect_self_policy_states(
        "centerline_set", nn.Identity(), specs, reference, tiny_config, 71_202,
        torch.device("cpu"), tmp_path / "policy",
    )
    assert policy["model"] == "centerline_set"
    assert stats["training_split_only"] is True
    assert len(policy["actions"]) == len(metadata)
    assert set(policy["episode_ids"].tolist()) == {specs[0].episode_id}
    later = int(torch.where(policy["steps"] > 0)[0][0])
    source = torch.where((expert["episode_ids"] == policy["episode_ids"][later]) &
                         (expert["steps"] == policy["steps"][later]))[0]
    assert len(source) == 1
    assert not torch.allclose(policy["states"][later], expert["states"][source[0]], atol=1e-5)
    assert not torch.allclose(policy["actions"][later], expert["actions"][source[0]], atol=1e-5)

    env = sf.make_env(tiny_config, 71_203)
    sf._reset_surface_env(env, specs[0])
    env.data.qpos[:] = policy["qpos"][later].numpy()
    env.data.qvel[:] = 0.0
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    fresh, _ = sf.expert_action(env, specs[0].shape, tiny_config)
    env.close()
    np.testing.assert_allclose(fresh, policy["actions"][later].numpy(), atol=1e-6)


def test_trajectory_collision_failure_uses_any_collision_during_rollout(tiny_config):
    row = {"trajectory_collision": True, "collision": False,
           "final_position_error": 0.01, "final_orientation_error": 0.1,
           "final_gripper_width_error": 0.002}
    assert final.failure_flags(row, tiny_config)["collision_failure"] is True


def test_cached_and_rebuilt_graph_have_same_connectivity_and_prediction(tiny_config):
    torch.manual_seed(5)
    rng = np.random.default_rng(7)
    points = rng.normal(size=(sf.SURFACE_POINTS, 3)).astype(np.float32) * 0.012
    tips = rng.normal(size=(2, 3)).astype(np.float32) * 0.025
    full = sf.build_graph_topology_numpy(points, tips, tiny_config, use_local_edges=True)
    cached = sf.build_graph_topology_numpy(points, tips, tiny_config, use_local_edges=True,
                                           cached_surface_pairs=full["surface_pairs"])
    for key in ("src", "dst", "edge_type", "valid", "surface_pairs"):
        np.testing.assert_array_equal(full[key], cached[key])
    model = sf.build_surface_model("surface_graph", tiny_config).eval()
    state = torch.zeros((1, sf.STATE_DIM), dtype=torch.float32)
    state[:, 8:14] = torch.as_tensor(tips.reshape(1, 6))
    point_t = torch.as_tensor(points).unsqueeze(0)
    with torch.no_grad():
        p0 = model(state, point_t, sf.topology_from_numpy(full, torch.device("cpu")))
        p1 = model(state, point_t, sf.topology_from_numpy(cached, torch.device("cpu")))
    torch.testing.assert_close(p0, p1, rtol=0, atol=1e-7)


def test_no_local_ablation_removes_only_surface_surface_edges(tiny_config):
    rng = np.random.default_rng(9)
    points = rng.normal(size=(sf.SURFACE_POINTS, 3)).astype(np.float32) * 0.010
    tips = np.zeros((2, 3), dtype=np.float32)
    full = sf.build_graph_topology_numpy(points, tips, tiny_config, use_local_edges=True)
    no_local = sf.build_graph_topology_numpy(points, tips, tiny_config, use_local_edges=False)
    for key in ("src", "dst", "edge_type", "valid"):
        np.testing.assert_array_equal(no_local[key], full[key][: len(no_local[key])])
    assert np.all(no_local["edge_type"] != 3)
    assert np.all(full["edge_type"][len(no_local["edge_type"]):] == 3)
    assert no_local["surface_pairs"].shape[0] == 0


def _tiny_dataset(seed: int) -> dict[str, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    return {
        "states": torch.randn((8, sf.STATE_DIM), generator=rng),
        "surface_points": torch.randn((8, sf.SURFACE_POINTS, 3), generator=rng) * 0.02,
        "centerline_points": torch.randn((8, sf.SURFACE_POINTS, 3), generator=rng) * 0.02,
        "actions": torch.randn((8, sf.ACTION_DIM), generator=rng) * 0.01,
    }


def test_round1_batch_ratio_updates_and_start_checkpoint_are_auditable(tiny_config, tmp_path):
    base_path = tmp_path / "base.pt"
    base_model = sf.build_surface_model("surface_set", tiny_config)
    torch.save({"model_name": "surface_set", "model_state": base_model.state_dict(),
                "config": sf.asdict(tiny_config), "seed": 41}, base_path)
    expert = _tiny_dataset(1)
    policy = _tiny_dataset(2)
    val = _tiny_dataset(3)
    metadata = [{"deviation_bucket": "near_expert" if i % 2 == 0 else "large_deviation"} for i in range(8)]
    row = final.train_dagger_round1("surface_set", base_path, expert, val, policy, metadata,
                                   tiny_config, 41, torch.device("cpu"), tmp_path / "dagger", updates=2)
    assert row["optimizer_updates"] == 2
    assert row["batch_size"] == 4
    assert row["effective_expert_fraction"] == row["effective_policy_fraction"] == 0.5
    assert row["base_checkpoint_sha256"] == final.state_dict_hash(sf.load_surface_model(
        base_path, "surface_set", tiny_config, torch.device("cpu")))


def test_n_sweep_keeps_exact_episode_specs(tiny_config):
    specs = sf.sample_episode_specs(3, 117, tiny_config, "iid")
    final.assert_same_specs({"n16": specs, "n32": list(specs), "n64": list(specs)})
    with pytest.raises(ValueError):
        final.assert_same_specs({"n16": specs, "n64": sf.sample_episode_specs(3, 118, tiny_config, "iid")})
