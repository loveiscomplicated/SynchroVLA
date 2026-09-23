import numpy as np
import pytest
import torch

import vla_gnn_recurrent.training.surface_graph_feasibility as surface
import vla_gnn_recurrent.training.surface_pregrasp_stabilization as stabilize


def _config(tmp_path, **kwargs):
    defaults = dict(output_dir=str(tmp_path), max_steps=3, control_substeps=3,
                    latency_warmup=2, latency_iterations=3, eval_episodes=2)
    defaults.update(kwargs)
    return surface.SurfaceFeasibilityConfig(**defaults)


def test_drawn_perturbations_obey_configured_position_yaw_and_aperture_bounds():
    perturb_config = stabilize.RecoveryPerturbationConfig()
    for seed in range(40):
        perturb = stabilize.draw_perturbation(np.random.default_rng(seed), perturb_config)
        distance = np.linalg.norm(perturb["position_xz_m"])
        assert 0.010 <= distance <= 0.030
        assert 0.050 <= abs(perturb["yaw_rad"]) <= 0.200
        assert 0.002 <= abs(perturb["aperture_m"]) <= 0.008


def test_perturbed_pose_wraps_yaw_and_keeps_aperture_in_physical_range(tmp_path):
    config = _config(tmp_path)
    env = surface.make_env(config, 21)
    spec = surface.sample_episode_specs(1, 22, config)[0]
    surface._reset_surface_env(env, spec)
    surface._set_gripper_open_fraction_kinematic(env, 1.0)
    result = stabilize.apply_state_perturbation(env, {
        "position_xz_m": [0.005, -0.005], "yaw_rad": 0.20, "aperture_m": 0.008,
    })
    assert -np.pi <= result["desired_yaw_wrapped"] <= np.pi
    assert surface.MIN_GRIPPER_WIDTH <= result["aperture_after_m"] <= surface.MAX_GRIPPER_WIDTH
    assert abs(result["actual_yaw_rad"]) <= 0.25
    assert result["actual_position_norm_m"] <= 0.03
    env.close()


def test_perturbed_state_recomputes_expert_action_instead_of_reusing_label(tmp_path):
    config = _config(tmp_path)
    spec = surface.sample_episode_specs(1, 23, config, "recompute")[0]
    base, _ = stabilize.collect_supervised_rows(config, [spec], 24, tmp_path / "base")
    recovery, stats = stabilize.build_recovery_dataset(base, [spec], config, 25, tmp_path / "recovery")
    records = __import__("json").loads((tmp_path / "recovery" / "perturbation_records.json").read_text())
    assert stats["recomputed_action_changed_fraction"] > 0.0
    assert len(recovery["actions"]) == 2 * len(base["actions"])
    assert torch.equal(recovery["actions"][:len(base["actions"])], base["actions"])
    changed = [not np.allclose(row["original_expert_action"], row["recomputed_recovery_action"], atol=1e-6)
               for row in records]
    assert any(changed)
    assert torch.equal(recovery["surface_points"], recovery["surface_points"])


def test_recovery_states_are_shared_before_model_specific_geometry_selection(tmp_path):
    config = _config(tmp_path)
    spec = surface.sample_episode_specs(1, 26, config, "same_perturb")[0]
    base, _ = stabilize.collect_supervised_rows(config, [spec], 27, tmp_path / "collect")
    combined, _ = stabilize.build_recovery_dataset(base, [spec], config, 28, tmp_path / "recover")
    assert torch.equal(combined["states"][:len(base["states"])], base["states"])
    assert torch.equal(combined["surface_points"][:len(base["states"])], base["surface_points"])
    assert torch.equal(combined["centerline_points"][:len(base["states"])], base["centerline_points"])
    assert torch.equal(stabilize._model_points(combined, "surface_set"), combined["surface_points"])
    assert torch.equal(stabilize._model_points(combined, "surface_graph"), combined["surface_points"])


def test_action_scale_normalization_has_correct_inverse():
    config = surface.SurfaceFeasibilityConfig()
    actions = np.random.default_rng(29).normal(size=(10, surface.ACTION_DIM)).astype(np.float32)
    normalized = stabilize.normalize_action_targets(actions, config)
    recovered = stabilize.denormalize_action_targets(normalized, config)
    assert np.allclose(recovered, actions, atol=1e-7)
    tensor = torch.as_tensor(actions)
    assert torch.allclose(stabilize.denormalize_action_targets(
        stabilize.normalize_action_targets(tensor, config), config), tensor)


def test_local_action_application_matches_translation_yaw_and_opening_command(tmp_path):
    config = _config(tmp_path, max_delta_ee=0.035, max_delta_rotation=0.35, max_delta_gripper=0.25)
    env = surface.make_env(config, 29)
    spec = surface.sample_episode_specs(1, 30, config, "action_application")[0]
    surface._reset_surface_env(env, spec)
    before = env.robot_observation().ee_position.numpy().astype(np.float64)
    yaw_before = surface.tool_yaw(env)
    width_before = surface.gripper_width(env)
    action = np.asarray([0.010, 0.0, -0.005, 0.10, 0.10], dtype=np.float32)
    expected_delta = surface.world_action_from_local(action[:3], yaw_before)
    surface.apply_local_action(env, action, config)
    after = env.robot_observation().ee_position.numpy().astype(np.float64)
    yaw_after = surface.tool_yaw(env)
    width_after = surface.gripper_width(env)
    assert np.linalg.norm((after - before - expected_delta)[[0, 2]]) < 0.006
    assert abs(surface.wrap_rotation_delta(yaw_after - yaw_before - action[3])) < 0.03
    assert width_after - width_before == pytest.approx(0.10 * (surface.MAX_GRIPPER_WIDTH - surface.MIN_GRIPPER_WIDTH), abs=0.001)
    env.close()


def test_cached_and_rebuilt_surface_graph_connectivity_and_prediction_match(tmp_path):
    config = _config(tmp_path)
    spec = surface.sample_episode_specs(1, 30, config, "cache")[0]
    env = surface.make_env(config, 31)
    surface._reset_surface_env(env, spec)
    state, points, _ = surface.observation_inputs(env, spec.shape)
    tips = state[8:14].reshape(2, 3)
    rebuilt = surface.build_graph_topology_numpy(points, tips, config, True)
    cached = surface.build_graph_topology_numpy(points, tips, config, True,
                                                cached_surface_pairs=rebuilt["surface_pairs"])
    for key in ("src", "dst", "edge_type", "valid", "surface_pairs"):
        assert np.array_equal(rebuilt[key], cached[key])
    model = surface.build_surface_model("surface_graph", config).eval()
    state_t = torch.as_tensor(state).reshape(1, -1)
    points_t = torch.as_tensor(points).reshape(1, -1, 3)
    topo_a = surface.topology_from_numpy(rebuilt, "cpu")
    topo_b = surface.topology_from_numpy(cached, "cpu")
    with torch.no_grad():
        a = model(state_t, points_t, topo_a)
        b = model(state_t, points_t, topo_b)
    assert float(torch.max(torch.abs(a - b))) < 1e-6
    env.close()


def test_resolution_evaluations_can_reuse_the_same_episode_specs(tmp_path):
    config = _config(tmp_path)
    specs = surface.sample_episode_specs(1, 32, config, "same_n_episode")
    model = surface.build_surface_model("surface_set", config)
    device = torch.device("cpu")
    ids = []
    for count in (16, 32, 64):
        result = surface.evaluate_models_paired(specs, {"surface_set": model}, config, device,
                                                tmp_path / f"n{count}", point_count=count)
        ids.append([row["episode_id"] for row in result["episodes"]])
    assert ids[0] == ids[1] == ids[2] == [specs[0].episode_id]
