"""Physical visual-latency semantics for the fixed residual replication."""

import math

import numpy as np
import torch

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import residual_recurrent_replication as replication
from vla_gnn_recurrent.training import surface_graph_feasibility as sf


def _observation(ee_x: float, object_x: float, yaw: float = 0.0,
                 object_yaw: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ee = np.array([ee_x, 0.0, 0.0], dtype=np.float32)
    world = np.zeros((32, 3), dtype=np.float32)
    world[:, 0] = object_x + np.linspace(-0.01, 0.01, 32)
    points = sf.localize_world(world, ee, yaw)
    center = sf.localize_world(np.array([[object_x, 0.0, 0.0]], dtype=np.float32), ee, yaw)[0]
    state = np.zeros(14, dtype=np.float32)
    state[:3] = center
    state[3:5] = (math.sin(object_yaw - yaw), math.cos(object_yaw - yaw))
    state[5:7] = (math.sin(yaw), math.cos(yaw))
    state[7] = 0.4
    state[8:14] = np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.02], dtype=np.float32)
    return state, points, ee


def test_static_robot_static_object():
    state, points, ee = _observation(0, .1)
    observed, visual = replication.reexpress_visual(state, points, state, points, ee, ee, 2)
    np.testing.assert_allclose(observed, state, atol=1e-7)
    np.testing.assert_allclose(visual, points, atol=1e-7)


def test_moving_robot_static_object_reexpresses_old_world_vision():
    old, old_points, old_ee = _observation(0, .1)
    now, points, ee = _observation(.04, .1)
    observed, visual = replication.reexpress_visual(now, points, old, old_points, ee, old_ee, 2)
    np.testing.assert_allclose(observed[0], .06, atol=1e-7)
    np.testing.assert_allclose(visual, points, atol=1e-7)
    np.testing.assert_array_equal(observed[5:], now[5:])


def test_static_robot_moving_object_keeps_old_visual_world_pose():
    old, old_points, old_ee = _observation(0, .1)
    now, points, ee = _observation(0, .14)
    observed, visual = replication.reexpress_visual(now, points, old, old_points, ee, old_ee, 2)
    np.testing.assert_allclose(observed[0], .1, atol=1e-7)
    np.testing.assert_allclose(visual, old_points, atol=1e-7)
    assert not np.array_equal(visual, points)


def test_robot_and_object_move_with_yaw_transform():
    old, old_points, old_ee = _observation(0, .1, yaw=.1, object_yaw=.3)
    now, _, ee = _observation(.04, .14, yaw=.4, object_yaw=.5)
    observed, visual = replication.reexpress_visual(now, old_points, old, old_points, ee, old_ee, 2)
    expected_center = sf.localize_world(np.array([[.1, 0, 0]], dtype=np.float32), ee, .4)[0]
    np.testing.assert_allclose(observed[:3], expected_center, atol=1e-7)
    np.testing.assert_allclose(observed[3:5], [math.sin(.3-.4), math.cos(.3-.4)], atol=1e-7)
    assert visual.shape == (32, 3)


def test_delay_zero_is_bitwise_canonical_and_observer_tracks_age():
    old, points, ee = _observation(0, .1)
    observed, visual = replication.reexpress_visual(old, points, old, points, ee, ee, 0)
    np.testing.assert_array_equal(observed, old)
    np.testing.assert_array_equal(visual, points)
    observer = replication.PhysicalVisualDelayObserver(delay=2)
    observer.observe(old, points, ee)
    observer.observe(old, points, ee)
    current, fresh_points, current_ee = _observation(.04, .1)
    observed, visual, age = observer.observe(current, fresh_points, current_ee)
    assert age == 2
    np.testing.assert_allclose(observed[0], .06, atol=1e-7)
    np.testing.assert_allclose(visual, fresh_points, atol=1e-7)


def test_training_transform_matches_observer_and_edges_use_current_nodes():
    from types import SimpleNamespace
    first, first_points, ee0 = _observation(0, .1)
    later, later_points, ee1 = _observation(.04, .1)
    states = torch.from_numpy(np.stack([first, first, later]))
    points = torch.from_numpy(np.stack([first_points, first_points, later_points]))
    ep = {"states": states, "points": points,
          "shape": SimpleNamespace(object_center=(.1, 0.0, 0.0))}
    observed_state, observed_points = replication.corrected_delayed_sequence(ep, 2)
    observer = replication.PhysicalVisualDelayObserver(delay=2)
    observer.observe(first, first_points, ee0)
    observer.observe(first, first_points, ee0)
    online_state, online_points, age = observer.observe(later, later_points, ee1)
    assert age == 2
    np.testing.assert_allclose(observed_state[2], online_state, atol=1e-6)
    np.testing.assert_allclose(observed_points[2], online_points, atol=1e-6)
    config = sf.SurfaceFeasibilityConfig(point_count=32, symmetric_robot_edges=True)
    _, _, topology = rf._graph_tensors(online_state, online_points, config, torch.device("cpu"))
    rf.assert_canonical_topology(topology, config)
    positions = np.concatenate((np.zeros((1, 3)), online_state[8:14].reshape(2, 3), online_points))
    src, dst = topology.src[0].numpy(), topology.dst[0].numpy()
    edge_relative = positions[dst] - positions[src]
    np.testing.assert_allclose(edge_relative[4], online_points[0], atol=1e-7)
