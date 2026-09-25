"""Deterministic moving-target delay geometry and benchmark tests."""

from dataclasses import replace

import numpy as np
import torch

from vla_gnn_recurrent.training import dynamic_visual_delay_feasibility as dynamic
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import residual_recurrent_replication as physical
from vla_gnn_recurrent.training import surface_graph_feasibility as sf


def _state(ee_x: float, object_x: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ee = np.array([ee_x, 0, 0], dtype=np.float32)
    center = np.array([[object_x, 0, 0]], dtype=np.float32)
    world_points = np.repeat(center, 32, axis=0)
    state = np.zeros(14, dtype=np.float32)
    state[:3] = sf.localize_world(center, ee, 0)[0]
    state[4], state[6] = 1, 1
    state[7] = .5
    return state, sf.localize_world(world_points, ee, 0), ee


def test_f_moving_object_static_robot_uses_old_visual_pose():
    observer = physical.PhysicalVisualDelayObserver(2)
    for object_x in (.10, .11, .12):
        state, points, ee = _state(0, object_x)
        seen, surface, age = observer.observe(state, points, ee)
    assert age == 2
    np.testing.assert_allclose(seen[0], .10, atol=1e-7)
    np.testing.assert_allclose(surface[:, 0], .10, atol=1e-7)
    np.testing.assert_allclose(state[0], .12, atol=1e-7)


def test_g_moving_robot_and_object_uses_current_robot_pose():
    observer = physical.PhysicalVisualDelayObserver(2)
    for ee_x, object_x in ((0, .10), (.02, .11), (.04, .12)):
        state, points, ee = _state(ee_x, object_x)
        seen, surface, age = observer.observe(state, points, ee)
    assert age == 2
    np.testing.assert_allclose(seen[0], .06, atol=1e-7)
    np.testing.assert_allclose(surface[:, 0], .06, atol=1e-7)
    np.testing.assert_allclose(state[0], .08, atol=1e-7)


def test_h_zero_delay_is_exact_current_graph():
    state, points, ee = _state(.04, .12)
    observed, visual, age = physical.PhysicalVisualDelayObserver(0).observe(state, points, ee)
    assert age == 0
    np.testing.assert_array_equal(observed, state)
    np.testing.assert_array_equal(visual, points)


def test_i_staleness_scales_with_speed_times_delay():
    for speed in (.001, .002, .004):
        for delay in (1, 2, 4, 8):
            observer = physical.PhysicalVisualDelayObserver(delay)
            for t in range(delay + 1):
                state, points, ee = _state(0, .10 + speed*t)
                observed, _, age = observer.observe(state, points, ee)
            assert age == delay
            np.testing.assert_allclose(state[0] - observed[0], speed*delay, atol=2e-7)


def test_motion_shape_changes_only_center():
    shape = sf.SurfaceShape("test", (.1, 0, .6), 0., 0., .2, 0., 0., .035, .015, 0., 0., 0., "flat")
    episode = sf.SurfaceEpisodeSpec(0, shape, (0., -.35, .95, -.45), .5, sample_identity="motion")
    spec = dynamic.DynamicSpec(episode, "-z", .002)
    moved = dynamic.moving_shape(spec, 4)
    np.testing.assert_allclose(np.asarray(moved.object_center) - np.asarray(shape.object_center), [0,0,-.008])
    assert replace(moved, object_center=shape.object_center) == shape


def test_saved_dynamic_sequence_matches_online_delay():
    states, points, ees = [], [], []
    observer = physical.PhysicalVisualDelayObserver(2)
    online = []
    for t in range(5):
        state, surface, ee = _state(.01*t, .10+.002*t)
        states.append(state); points.append(surface); ees.append(ee)
        online.append(observer.observe(state,surface,ee))
    episode = {"states":torch.from_numpy(np.stack(states)),
               "points":torch.from_numpy(np.stack(points)),
               "ee_world":torch.from_numpy(np.stack(ees))}
    delayed_states, delayed_points = dynamic.dynamic_delayed_sequence(episode,2)
    for t,(state,surface,age) in enumerate(online):
        np.testing.assert_allclose(delayed_states[t],state,atol=1e-7)
        np.testing.assert_allclose(delayed_points[t],surface,atol=1e-7)
    fresh_states,fresh_points = dynamic.dynamic_delayed_sequence(episode,0)
    assert torch.equal(fresh_states,episode["states"])
    assert torch.equal(fresh_points,episode["points"])


def test_moving_shape_updates_mujoco_collision_proxies_and_visual_points(tmp_path):
    config=rf._task_config(tmp_path,(2811,),"cpu","cpu",16)
    spec=sf.sample_episode_specs(1,2811,config,"moving_test")[0]
    env=sf.make_env(config,2811)
    try:
        sf._reset_surface_env(env,spec)
        ids=(env.surface_box_mocap_ids if spec.shape.cross_section=="flat"
             else env.surface_capsule_mocap_ids)
        before_mocap=env.data.mocap_pos[ids].copy()
        before_state,before_points,_=sf.observation_inputs(env,spec.shape,config.point_count,17)
        moved=replace(spec.shape,object_center=tuple(np.asarray(spec.shape.object_center)+[.004,0,0]))
        env.set_surface_shape(moved)
        after_mocap=env.data.mocap_pos[ids].copy()
        after_state,after_points,_=sf.observation_inputs(env,moved,config.point_count,17)
        np.testing.assert_allclose(after_mocap-before_mocap,
                                   np.tile(np.array([.004,0,0]),(len(ids),1)),atol=1e-9)
        assert np.linalg.norm(after_state[:3]-before_state[:3])>1e-3
        assert np.linalg.norm(after_points-before_points,axis=1).mean()>1e-3
    finally:
        env.close()
