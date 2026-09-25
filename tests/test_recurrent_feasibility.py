"""Behavioral checks for the fixed-graph recurrent experiment."""

import numpy as np
import pytest
import torch

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_representation_final_comparison as historical
from vla_gnn_recurrent.utils import set_seed


def test_graph_embedding_and_gru_dimensions_and_state_lifecycle() -> None:
    config = sf.SurfaceFeasibilityConfig()
    ff = sf.build_surface_model("surface_graph_no_local", config).eval()
    gru = rf.GraphGRUPolicy(config).eval()
    state = torch.randn(1, sf.STATE_DIM)
    points = torch.randn(1, sf.SURFACE_POINTS, 3) * 0.03
    with torch.no_grad():
        embedding = ff.encode(state, points)
        first, hidden = gru.step(state, points, None)
        second, hidden2 = gru.step(state, points, hidden)
        reset, reset_hidden = gru.step(state, points, None)
        ff_a = ff(state, points)
        ff_b = ff(state, points)
    assert embedding.shape == (1, config.hidden_dim * 3)
    assert first.shape == (1, sf.ACTION_DIM)
    assert hidden.shape == (1, 1, gru.hidden_size)
    assert hidden2.shape == hidden.shape
    assert not torch.allclose(first, second)
    assert torch.allclose(first, reset)
    assert torch.allclose(hidden, reset_hidden)
    assert torch.allclose(ff_a, ff_b)
    assert gru.graph.use_local_edges is False


def test_sequence_order_gaps_and_masked_batch() -> None:
    data = {
        "states": torch.arange(5, dtype=torch.float32)[:, None].repeat(1, sf.STATE_DIM),
        "surface_points": torch.zeros(5, sf.SURFACE_POINTS, 3),
        "actions": torch.zeros(5, sf.ACTION_DIM),
        "episode_ids": torch.tensor([0, 0, 0, 1, 1]),
        "steps": torch.tensor([0, 1, 3, 0, 1]),
    }
    episodes = rf._episodes(data)
    assert [e["steps"].tolist() for e in episodes] == [[0, 1], [3], [0, 1]]
    states, points, actions, mask = rf.sample_sequence_batch(
        episodes, 8, 32, np.random.default_rng(4), torch.device("cpu"))
    assert states.shape[:2] == mask.shape
    assert points.shape[:2] == mask.shape
    assert actions.shape[:2] == mask.shape
    for i in range(len(mask)):
        values = states[i, mask[i], 0].tolist()
        assert values in ([0.0, 1.0], [2.0], [3.0, 4.0])
    bad = {**data, "steps": torch.tensor([1, 0, 3, 0, 1])}
    with pytest.raises(ValueError, match="order"):
        rf._episodes(bad)


def test_severe_state_keeps_recurrent_context_but_has_no_training_loss() -> None:
    data = {
        "states": torch.arange(3, dtype=torch.float32)[:, None].repeat(1, sf.STATE_DIM),
        "surface_points": torch.zeros(3, sf.SURFACE_POINTS, 3),
        "actions": torch.zeros(3, sf.ACTION_DIM),
        "episode_ids": torch.zeros(3, dtype=torch.long),
        "steps": torch.arange(3),
        "loss_mask": torch.tensor([True, False, True]),
    }
    episodes = rf._episodes(data)
    assert len(episodes) == 1
    states, _, _, mask = rf.sample_sequence_batch(episodes, 1, 32, np.random.default_rng(5), torch.device("cpu"))
    assert states[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert mask[0].tolist() == [True, False, True]


def test_dagger_mixture_is_equal_by_source_despite_longer_policy_trajectory() -> None:
    config = sf.SurfaceFeasibilityConfig()
    prediction = torch.zeros(2, 4, sf.ACTION_DIM)
    target = torch.zeros_like(prediction)
    target[0, 0, 0] = config.max_delta_ee
    target[1, :, 0] = 2 * config.max_delta_ee
    mask = torch.tensor([[True, False, False, False], [True, True, True, True]])
    mixed = rf.mixed_sequence_loss(prediction, target, mask, 1, config)
    assert float(mixed) == pytest.approx(0.5 * (1 / sf.ACTION_DIM + 4 / sf.ACTION_DIM))


def test_masked_target_has_no_supervised_gradient() -> None:
    config = sf.SurfaceFeasibilityConfig()
    prediction = torch.ones(1, 3, sf.ACTION_DIM, requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False, True]])
    rf.sequence_loss(prediction, target, mask, config).backward()
    assert prediction.grad is not None
    assert torch.all(prediction.grad[0, 1] == 0)
    assert torch.any(prediction.grad[0, 0] != 0)
    assert torch.any(prediction.grad[0, 2] != 0)


def test_historical_spatial_topology_remains_available_for_provenance(monkeypatch) -> None:
    config = sf.SurfaceFeasibilityConfig()
    set_seed(2811)
    ff = sf.build_surface_model("surface_graph_no_local", config).eval()
    set_seed(2811)
    gru = rf.GraphGRUPolicy(config).eval()
    for name, tensor in ff.state_dict().items():
        assert torch.equal(tensor, gru.graph.state_dict()[name])
    state = torch.zeros(1, sf.STATE_DIM)
    points = torch.zeros(1, 32, 3)
    train_topologies = []
    original_torch = sf._torch_topology

    def watch_torch(*args, **kwargs):
        topology = original_torch(*args, **kwargs)
        train_topologies.append(topology)
        return topology

    monkeypatch.setattr(sf, "_torch_topology", watch_torch)
    with torch.no_grad():
        ff(state, points)
        gru.forward_sequence(state[:, None], points[:, None])
    assert [len(t.src[0]) for t in train_topologies] == [98, 98]
    assert all(not torch.any(t.edge_type == 3) for t in train_topologies)
    assert torch.equal(train_topologies[0].src, train_topologies[1].src)
    assert torch.equal(train_topologies[0].dst, train_topologies[1].dst)
    inference_topologies = []
    original_numpy = sf.build_graph_topology_numpy

    def watch_numpy(*args, **kwargs):
        topology = original_numpy(*args, **kwargs)
        inference_topologies.append(topology)
        return topology

    monkeypatch.setattr(sf, "build_graph_topology_numpy", watch_numpy)
    np_state = state[0].numpy()
    np_points = points[0].numpy()
    rf.controller_step(ff, "ff", np_state, np_points, None, config, torch.device("cpu"))
    rf.controller_step(gru, "gru", np_state, np_points, None, config, torch.device("cpu"))
    assert [len(t["src"]) for t in inference_topologies] == [100, 100]
    for key in ("src", "dst", "edge_type", "valid"):
        assert np.array_equal(inference_topologies[0][key], inference_topologies[1][key])
    assert not np.any(inference_topologies[0]["edge_type"] == 3)


def test_canonical_100_edge_topology_in_ff_and_gru_training_and_rollout(monkeypatch) -> None:
    from dataclasses import replace

    config = replace(sf.SurfaceFeasibilityConfig(), symmetric_robot_edges=True)
    assert config.point_count == 32
    state = torch.zeros(1, sf.STATE_DIM)
    points = torch.zeros(1, sf.SURFACE_POINTS, 3)
    ff = sf.build_surface_model(rf.GRAPH_NAME, config).eval()
    gru = rf.GraphGRUPolicy(config).eval()
    train_topologies = []
    original = sf._torch_topology

    def watch(*args, **kwargs):
        topology = original(*args, **kwargs)
        train_topologies.append(topology)
        return topology

    monkeypatch.setattr(sf, "_torch_topology", watch)
    with torch.no_grad():
        ff(state, points)
        gru.forward_sequence(state[:, None], points[:, None])
    assert len(train_topologies) == 2
    for topology in train_topologies:
        rf.assert_canonical_topology(topology, config)
    assert torch.equal(train_topologies[0].src, train_topologies[1].src)
    assert torch.equal(train_topologies[0].dst, train_topologies[1].dst)
    for model, name in ((ff, "ff"), (gru, "gru")):
        state_t, points_t, rollout_topology = rf._graph_tensors(state[0].numpy(), points[0].numpy(),
                                                                config, torch.device("cpu"))
        rf.assert_canonical_topology(rollout_topology, config)
        rf.controller_step(model, name, state_t[0].numpy(), points_t[0].numpy(), None,
                           config, torch.device("cpu"))
    assert {(0, 1), (1, 0), (0, 2), (2, 0)}.issubset(
        set(zip(train_topologies[0].src[0].tolist(), train_topologies[0].dst[0].tolist(), strict=True)))


def test_canonical_gru_training_sequence_matches_stepwise_rollout() -> None:
    from dataclasses import replace

    config = replace(sf.SurfaceFeasibilityConfig(), symmetric_robot_edges=True)
    generator = torch.Generator().manual_seed(2811)
    states = torch.randn(1, 6, sf.STATE_DIM, generator=generator) * 0.1
    points = torch.randn(1, 6, sf.SURFACE_POINTS, 3, generator=generator) * 0.03
    model = rf.GraphGRUPolicy(config).eval()
    with torch.no_grad():
        sequence_actions, _ = model.forward_sequence(states, points)
        hidden = None
        step_actions = []
        for t in range(states.shape[1]):
            state_t, points_t, topology = rf._graph_tensors(
                states[0, t].numpy(), points[0, t].numpy(), config, torch.device("cpu"))
            action, hidden = model.step(state_t, points_t, hidden, topology)
            step_actions.append(action)
    assert torch.allclose(sequence_actions, torch.stack(step_actions, dim=1), atol=1e-6)


def test_registered_trainable_and_active_parameter_counts() -> None:
    config = sf.SurfaceFeasibilityConfig()
    ff = rf.parameter_audit(sf.build_surface_model("surface_graph_no_local", config), config)
    gru = rf.parameter_audit(rf.GraphGRUPolicy(config), config)
    assert ff == {"registered": 112136, "trainable_requires_grad": 112136,
                  "active_in_forward_loss": 112136, "inactive_registered": 0}
    assert gru == {"registered": 474509, "trainable_requires_grad": 461832,
                   "active_in_forward_loss": 461832, "inactive_registered": 12677}


def test_stale_and_delay_exact_snapshot_age() -> None:
    stale = rf.ObservationCorruptor("stale", 2, start=3)
    delay = rf.ObservationCorruptor("delay", 2)
    fresh = rf.ObservationCorruptor("fresh")
    stale_seen = []
    delay_seen = []
    for step in range(7):
        state = np.full(sf.STATE_DIM, step, dtype=np.float32)
        points = np.full((sf.SURFACE_POINTS, 3), step, dtype=np.float32)
        for observer, rows in ((stale, stale_seen), (delay, delay_seen)):
            observed, observed_points, age, _ = observer.observe(state, points)
            rows.append((int(observed[0]), int(observed_points[0, 0]), age))
        observed_fresh, _, age_fresh, _ = fresh.observe(state, points)
        assert observed_fresh[0] == step and age_fresh == 0
    assert stale_seen == [(0, 0, 0), (1, 1, 0), (2, 2, 0), (2, 2, 1), (2, 2, 2), (5, 5, 0), (6, 6, 0)]
    assert delay_seen == [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 1, 2), (2, 2, 2), (3, 3, 2), (4, 4, 2)]


def test_paired_rollout_same_corruption_and_simulator_progresses(tmp_path) -> None:
    config = sf.SurfaceFeasibilityConfig(output_dir=str(tmp_path), max_steps=4)
    temporal = rf.TemporalConfig(stale_start=1)
    spec = sf.sample_episode_specs(1, 991, config)[0]
    torch.manual_seed(3)
    ff = sf.build_surface_model("surface_graph_no_local", config).eval()
    torch.manual_seed(3)
    gru = rf.GraphGRUPolicy(config).eval()
    env = sf.make_env(config, 991)
    try:
        ff_row, ff_trace = rf.rollout(env, spec, ff, "ff", config, temporal, torch.device("cpu"), "stale", 2)
        gru_row, gru_trace = rf.rollout(env, spec, gru, "gru", config, temporal, torch.device("cpu"), "stale", 2)
    finally:
        env.close()
    assert ff_row["spec_signature"] == gru_row["spec_signature"]
    assert [r["observed_graph_timestamp"] for r in ff_trace] == [r["observed_graph_timestamp"] for r in gru_trace]
    assert ff_trace[2]["observed_graph_timestamp"] == 0
    assert ff_trace[2]["true_ee_pose"] != ff_trace[0]["true_ee_pose"]
    assert all(r["oracle_action"] is None for r in ff_trace + gru_trace)


def test_historical_ff_clean_rollout_primary_metric_parity(tmp_path) -> None:
    seed = 2811
    config = rf._task_config(tmp_path, (seed,), "cpu", "cpu", 16)
    checkpoint = rf.PRIOR / "training" / f"seed{seed}" / rf.GRAPH_NAME / "dagger_round1" / f"{rf.GRAPH_NAME}.pt"
    model = sf.load_surface_model(checkpoint, rf.GRAPH_NAME, config, torch.device("cpu"))
    specs = sf.sample_episode_specs(config.eval_episodes, seed + 60_000, config, "iid")
    env = sf.make_env(config, 70113)
    try:
        for spec in specs:
            old = historical.rollout_detailed(env, spec, config, rf.GRAPH_NAME, model, torch.device("cpu"))
            new, _ = rf.rollout(env, spec, model, "ff", config, rf.TemporalConfig(), torch.device("cpu"))
            for key in ("success", "collision", "final_collision", "first_collision_timestep",
                        "steps_to_convergence"):
                assert new[key] == old[key], (spec.episode_id, key)
            for key in ("final_position_error", "final_orientation_error", "final_gripper_width_error",
                        "trajectory_error", "trajectory_yaw_error", "trajectory_aperture_error",
                        "trajectory_length", "minimum_safe_clearance"):
                assert new[key] == pytest.approx(old[key], abs=1e-7), (spec.episode_id, key)
    finally:
        env.close()


def test_initial_success_terminates_before_action(tmp_path, monkeypatch) -> None:
    config = sf.SurfaceFeasibilityConfig(output_dir=str(tmp_path), max_steps=3)
    spec = sf.sample_episode_specs(1, 77, config)[0]
    model = sf.build_surface_model("surface_graph_no_local", config).eval()
    monkeypatch.setattr(sf, "_success_from_errors", lambda errors, collision, cfg: True)
    env = sf.make_env(config, 77)
    try:
        row, trace = rf.rollout(env, spec, model, "ff", config, rf.TemporalConfig(), torch.device("cpu"))
    finally:
        env.close()
    assert trace == []
    assert row["steps_executed"] == 0
    assert row["steps_to_convergence"] == 0
    assert row["trajectory_length"] == 0


def test_recurrent_collector_serializes_full_order_and_loss_mask(tmp_path) -> None:
    config = sf.SurfaceFeasibilityConfig(output_dir=str(tmp_path), max_steps=3)
    temporal = rf.TemporalConfig()
    specs = sf.sample_episode_specs(2, 121, config, "train")
    model = rf.GraphGRUPolicy(config).eval()
    data, stats = rf.collect_recurrent_policy_data(model, specs, config, temporal, 121,
                                                     torch.device("cpu"), tmp_path / "collection")
    saved = torch.load(tmp_path / "collection" / "policy_relabelled_trajectories.pt",
                       map_location="cpu", weights_only=False)
    for key in ("states", "surface_points", "actions", "episode_ids", "steps", "loss_mask"):
        assert torch.equal(data[key], saved[key])
    assert len(data["steps"]) == stats["visited_states"]
    assert int(data["loss_mask"].sum()) == stats["eligible_states"]
    for episode in rf._episodes(saved):
        steps = episode["steps"].tolist()
        assert steps == list(range(steps[0], steps[0] + len(steps)))


def test_gru_hidden_resets_between_rollouts_of_same_episode(tmp_path) -> None:
    config = sf.SurfaceFeasibilityConfig(output_dir=str(tmp_path), max_steps=3)
    spec = sf.sample_episode_specs(1, 52, config)[0]
    model = rf.GraphGRUPolicy(config).eval()
    env = sf.make_env(config, 52)
    try:
        first, first_trace = rf.rollout(env, spec, model, "gru", config, rf.TemporalConfig(), torch.device("cpu"))
        second, second_trace = rf.rollout(env, spec, model, "gru", config, rf.TemporalConfig(), torch.device("cpu"))
    finally:
        env.close()
    assert first["success"] == second["success"]
    assert first["final_position_error"] == pytest.approx(second["final_position_error"], abs=1e-8)
    assert first_trace[0]["predicted_action"] == pytest.approx(second_trace[0]["predicted_action"], abs=1e-8)
    assert first_trace[0]["hidden_norm"] == pytest.approx(second_trace[0]["hidden_norm"], abs=1e-8)


def test_training_logs_actual_valid_exposure_and_sequence_lengths(tmp_path) -> None:
    config = sf.SurfaceFeasibilityConfig(output_dir=str(tmp_path))
    temporal = rf.TemporalConfig(sequence_batch_size=4)
    data = {
        "states": torch.zeros(4, sf.STATE_DIM),
        "surface_points": torch.zeros(4, sf.SURFACE_POINTS, 3),
        "actions": torch.zeros(4, sf.ACTION_DIM),
        "episode_ids": torch.tensor([0, 0, 1, 1]),
        "steps": torch.tensor([0, 1, 0, 1]),
    }
    policy = {**data, "loss_mask": torch.tensor([True, False, True, True])}
    model = rf.GraphGRUPolicy(config)
    result = rf.train_gru(model, data, data, policy, config, temporal, 4,
                          torch.device("cpu"), 2, tmp_path / "train")
    assert result["updates"] == 2
    assert result["expert_valid_supervised_timesteps"] == 8
    assert 4 <= result["policy_valid_supervised_timesteps"] <= 8
    assert result["actual_valid_supervised_timesteps"] == (
        result["expert_valid_supervised_timesteps"] + result["policy_valid_supervised_timesteps"])
    assert result["sampled_expert_sequence_lengths"]["histogram"] == {"2": 4}
    assert result["sampled_policy_sequence_lengths"]["histogram"] == {"2": 4}
