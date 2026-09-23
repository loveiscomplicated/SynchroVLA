from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

import vla_gnn_recurrent.training.surface_graph_feasibility as surface


def _config(tmp_path, **kwargs):
    defaults = dict(output_dir=str(tmp_path), max_steps=3, control_substeps=3,
                    latency_warmup=2, latency_iterations=3, eval_episodes=2)
    defaults.update(kwargs)
    return surface.SurfaceFeasibilityConfig(**defaults)


def test_surface_generator_is_deterministic_for_seed(tmp_path):
    config = _config(tmp_path)
    shape = surface.sample_shape(np.random.default_rng(18), "same", config)
    a = surface.sample_surface_points_world(shape, 32, seed=91)
    b = surface.sample_surface_points_world(shape, 32, seed=91)
    c = surface.sample_surface_points_world(shape, 32, seed=92)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_necessity_pair_keeps_centerline_but_changes_surface_target_and_opening(tmp_path):
    config = _config(tmp_path)
    thin, thick = surface.paired_surface_necessity_shapes(31, config)
    assert np.allclose(surface.sample_centerline_points_world(thin), surface.sample_centerline_points_world(thick))
    ee = np.asarray(thin.object_center) + np.array([0.0, 0.0, 0.20])
    target_a = surface._surface_target(thin, ee, config.pregrasp_clearance)
    target_b = surface._surface_target(thick, ee, config.pregrasp_clearance)
    assert np.linalg.norm(target_a[0] - target_b[0]) > 1e-3
    assert abs(target_a[3] - target_b[3]) > 1e-3


def test_target_and_surface_normal_are_not_model_input(tmp_path):
    config = _config(tmp_path)
    env = surface.make_env(config, 4)
    spec = surface.sample_episode_specs(1, 5, config)[0]
    surface._reset_surface_env(env, spec)
    inputs = surface.observation_inputs(env, spec.shape)
    env.close()
    assert len(inputs) == 3
    state, cloud, skeleton = inputs
    assert state.shape == (surface.STATE_DIM,)
    assert cloud.shape == skeleton.shape == (32, 3)
    assert not any("target" in key or "normal" in key for key in ("state", "surface_points", "centerline_points"))


def test_surface_set_and_graph_share_identical_point_tensor(tmp_path):
    config = _config(tmp_path)
    env = surface.make_env(config, 12)
    spec = surface.sample_episode_specs(1, 13, config)[0]
    surface._reset_surface_env(env, spec)
    _, cloud, skeleton = surface.observation_inputs(env, spec.shape)
    env.close()
    surface_set_input = surface.policy_geometry_points("surface_set", cloud, skeleton)
    graph_input = surface.policy_geometry_points("surface_graph", cloud, skeleton)
    assert np.array_equal(surface_set_input, graph_input)


def test_graph_uses_observed_points_not_mesh_adjacency(tmp_path):
    config = _config(tmp_path)
    rng = np.random.default_rng(7)
    points = rng.normal(0.0, 0.02, size=(32, 3)).astype(np.float32)
    tips = np.zeros((2, 3), dtype=np.float32)
    graph = surface.build_graph_topology_numpy(points, tips, config, use_local_edges=True)
    assert graph["surface_pairs"].ndim == 2
    assert graph["surface_pairs"].shape[1] == 2
    assert not hasattr(graph, "mesh_adjacency")
    assert "mesh_adjacency" not in graph


def test_knn_neighborhood_is_permutation_equivariant(tmp_path):
    config = _config(tmp_path)
    points = np.random.default_rng(9).uniform(-0.03, 0.03, size=(32, 3)).astype(np.float32)
    tips = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float32)
    original = surface.build_graph_topology_numpy(points, tips, config, True)["surface_pairs"]
    permutation = np.random.default_rng(10).permutation(len(points))
    permuted = surface.build_graph_topology_numpy(points[permutation], tips, config, True)["surface_pairs"]
    original_set = {tuple(pair) for pair in original.tolist()}
    mapped = {tuple(sorted((int(permutation[a]), int(permutation[b])))) for a, b in permuted.tolist()}
    expected = {tuple(sorted(pair)) for pair in original_set}
    assert mapped == expected


def test_no_local_ablation_removes_only_surface_surface_edges(tmp_path):
    config = _config(tmp_path)
    points = np.random.default_rng(11).uniform(-0.02, 0.02, size=(32, 3)).astype(np.float32)
    tips = np.array([[0.08, 0.0, 0.01], [-0.08, 0.0, -0.01]], dtype=np.float32)
    full = surface.build_graph_topology_numpy(points, tips, config, True)
    no_local = surface.build_graph_topology_numpy(points, tips, config, False)
    assert np.any(full["edge_type"] == 3)
    assert not np.any(no_local["edge_type"] == 3)
    keep_full = sorted(zip(full["src"][full["edge_type"] != 3], full["dst"][full["edge_type"] != 3],
                           full["edge_type"][full["edge_type"] != 3], strict=True))
    keep_no = sorted(zip(no_local["src"], no_local["dst"], no_local["edge_type"], strict=True))
    assert keep_full == keep_no


def test_ee_local_transform_round_trip_and_action_world_transform():
    yaw = -0.73
    ee = np.array([0.14, 0.0, 0.56], dtype=np.float32)
    point = np.array([[0.22, 0.0, 0.61]], dtype=np.float32)
    local = surface.localize_world(point, ee, yaw)
    reconstructed = surface.world_action_from_local(local[0], yaw) + ee
    assert np.allclose(reconstructed, point[0], atol=1e-6)
    local_action = np.array([0.02, 0.0, -0.01], dtype=np.float32)
    expected = surface.rotate_xz(torch.as_tensor(local_action).reshape(1, 3), yaw)[0].numpy()
    assert np.allclose(surface.world_action_from_local(local_action, yaw), expected)


def test_rotation_wrap_and_gripper_action_clipping():
    assert surface.wrap_rotation_delta(2 * np.pi + 0.2) == pytest.approx(0.2)
    assert surface.equivalent_orientation_error(0.2 + 4 * np.pi, 0.2) == pytest.approx(0.0, abs=1e-7)
    applied, opening = surface.clamp_gripper_action(0.8, 0.7, 0.25)
    assert applied == pytest.approx(0.25)
    assert opening == pytest.approx(0.95)
    assert surface.clamp_gripper_action(-4.0, 0.1, 0.25) == pytest.approx((-0.25, 0.0))


def test_mujoco_collision_evaluation_uses_actual_primitive_geometry(tmp_path):
    config = _config(tmp_path)
    env = surface.make_env(config, 17)
    env.reset(randomize_robot=False)
    ee = env.robot_observation().ee_position.numpy()
    shape = surface.sample_shape(np.random.default_rng(18), "collision", config)
    shape = replace(shape, object_center=(float(ee[0]), 0.0, float(ee[2])), cross_section="rounded",
                    half_width=0.04, half_depth=0.02, length=0.15)
    env.set_surface_shape(shape)
    mujoco.mj_forward(env.model, env.data)
    contacts, clearance = surface._surface_distance_metrics(env, shape)
    env.close()
    assert contacts
    assert clearance < 0.0
    assert all(row["distance"] <= 1e-4 for row in contacts)


def test_paired_evaluation_uses_same_episode_spec_for_each_policy(tmp_path):
    config = _config(tmp_path, max_steps=1, batch_size=8)
    spec = surface.sample_episode_specs(1, 20, config)[0]
    device = torch.device("cpu")
    models = {name: surface.build_surface_model(name, config).to(device).eval()
              for name in ("centerline_set", "surface_set", "surface_graph")}
    result = surface.evaluate_models_paired([spec], models, config, device, tmp_path / "paired")
    row = result["episodes"][0]
    assert {row["models"][name]["shape_id"] for name in models} == {spec.shape.shape_id}
    assert all(np.allclose(row["models"][name]["initial_ee"], row["models"]["centerline_set"]["initial_ee"]) for name in models)


def test_cached_topology_requires_stable_surface_identity():
    assert surface.cached_topology_valid("shape-a:points-v1", "shape-a:points-v1")
    assert not surface.cached_topology_valid("shape-a:points-v1", "shape-b:points-v1")
    assert not surface.cached_topology_valid(None, "shape-a:points-v1")


def test_async_backend_latency_timer_synchronizes(monkeypatch):
    if not hasattr(torch, "mps") or not hasattr(torch.mps, "synchronize"):
        pytest.skip("This PyTorch build has no MPS synchronization API")
    calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append(True))
    surface._sync_device(torch.device("mps"))
    assert calls == [True]


def test_point_counts_support_requested_resolution_curve(tmp_path):
    config = _config(tmp_path)
    shape = surface.sample_shape(np.random.default_rng(22), "resolution", config)
    for count in (16, 32, 64):
        points = surface.sample_surface_points_world(shape, count, seed=23)
        assert points.shape == (count, 3)
