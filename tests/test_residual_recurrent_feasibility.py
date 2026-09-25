"""Contracts for the frozen-FF residual feasibility experiment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import residual_recurrent_feasibility as residual
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training import surface_representation_final_comparison as final
from vla_gnn_recurrent.utils import select_device


@pytest.fixture(scope="module")
def task(tmp_path_factory):
    return rf._task_config(tmp_path_factory.mktemp("residual"), (2811,), "cpu", "cpu", 16)


@pytest.fixture(scope="module")
def design():
    return residual.ResidualConfig()


@pytest.fixture(scope="module")
def inputs(task):
    expert, validation, train_specs, _ = final._current_datasets(2811, task)
    return expert, validation, train_specs


def _model(kind, task, design):
    return residual.load_controller(kind, task, design, select_device("cpu"))


def test_canonical_ff_checkpoint_and_topology_load(task, design):
    assert residual.BASE_CHECKPOINT.exists()
    assert residual.hd._sha256(residual.BASE_CHECKPOINT) == residual.BASE_SHA256
    model = _model("gru", task, design)
    assert not model.base.use_local_edges
    assert all(not p.requires_grad for p in model.base.parameters())
    state = torch.zeros(1, sf.STATE_DIM)
    points = torch.zeros(1, 32, 3)
    rf.assert_canonical_topology(sf._torch_topology(points, state, task, False), task)


@pytest.mark.parametrize("kind", ("mlp", "gru"))
def test_zero_initial_residual_is_exact_ff_and_gripper_unchanged(task, design, kind):
    model = _model(kind, task, design)
    state = torch.randn(2, sf.STATE_DIM) * .1
    points = torch.randn(2, 32, 3) * .01
    previous = torch.zeros(2, 5)
    with torch.no_grad():
        corrected, delta, _, base = model.step(state, points, previous)
        canonical = model.base(state, points)
    assert torch.equal(corrected, base)
    assert torch.equal(base, canonical)
    assert torch.count_nonzero(delta) == 0
    assert torch.equal(corrected[:, 4], base[:, 4])


def test_frozen_encoder_and_head_receive_no_gradient_or_update(task, design):
    model = _model("gru", task, design)
    before = {key: value.clone() for key, value in model.base.state_dict().items()}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
    state = torch.randn(2, 3, sf.STATE_DIM)
    points = torch.randn(2, 3, 32, 3) * .01
    previous = torch.zeros(2, 3, 5)
    prediction, _, _, _, _ = model.forward_sequence(state, points, previous)
    loss = prediction[..., :4].square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.base.parameters())
    optimizer.step()
    assert all(torch.equal(before[key], value) for key, value in model.base.state_dict().items())


def test_residual_bounds_and_optional_fifth_dimension(task, design):
    model = _model("gru", task, design)
    with torch.no_grad():
        model.residual_head.bias.fill_(100)
        state = torch.zeros(1, sf.STATE_DIM)
        points = torch.zeros(1, 32, 3)
        corrected, delta, _, base = model.step(state, points, torch.zeros(1, 5))
    assert torch.all(delta.abs() <= model.residual_bounds + 1e-8)
    assert torch.equal(corrected[:, 4], base[:, 4])
    optional = _model("mlp", task, replace(design, corrected_dimensions=5))
    assert optional.residual_head.out_features == 5


def test_sequence_matches_stepwise_and_hidden_reset(task, design):
    model = _model("gru", task, design).eval()
    with torch.no_grad():
        model.residual_head.weight.normal_(std=.05)
    states = torch.randn(2, 4, sf.STATE_DIM) * .1
    points = torch.randn(2, 4, 32, 3) * .01
    previous = torch.randn(2, 4, 5) * .01
    with torch.no_grad():
        sequence, _, final_hidden, _, _ = model.forward_sequence(states, points, previous)
        hidden, pieces = None, []
        for t in range(4):
            action, _, hidden, _ = model.step(states[:, t], points[:, t], previous[:, t], hidden)
            pieces.append(action)
        reset, _, _, _ = model.step(states[:, 0], points[:, 0], previous[:, 0], hidden, reset_hidden=True)
        fresh, _, _, _ = model.step(states[:, 0], points[:, 0], previous[:, 0], None)
    assert torch.allclose(sequence, torch.stack(pieces, 1), atol=1e-6)
    assert torch.allclose(final_hidden, hidden, atol=1e-6)
    assert torch.allclose(reset, fresh)


def test_visual_only_delay_and_zero_delay_exact(task, inputs):
    _, _, specs = inputs
    env = sf.make_env(task, 41)
    try:
        sf._reset_surface_env(env, specs[0])
        state, points, _ = sf.observation_inputs(env, specs[0].shape, 32,
                                                  sf._stable_seed(specs[0].sample_identity))
    finally:
        env.close()
    observer = residual.VisualDelayObserver(0)
    observed_state, observed_points, age = observer.observe(state, points)
    assert age == 0
    assert np.array_equal(observed_state, state)
    assert np.array_equal(observed_points, points)
    old_state, old_points = state.copy(), points.copy()
    current_state, current_points = state.copy(), points.copy()
    current_state[:5] += 1
    current_state[5:] += 2
    current_points += 3
    observed_state, observed_points = residual.visual_delay(current_state, current_points, old_state, old_points)
    assert np.array_equal(observed_state[:5], old_state[:5])
    assert np.array_equal(observed_state[5:], current_state[5:])
    assert np.array_equal(observed_points, old_points)
    state_t, points_t, topology = rf._graph_tensors(observed_state, observed_points, task, torch.device("cpu"))
    rf.assert_canonical_topology(topology, task)
    assert state_t.shape == (1, 14) and points_t.shape == (1, 32, 3)


def test_ordered_data_previous_actions_and_padding_mask(task, inputs):
    expert, _, _ = inputs
    pool = torch.load("artifacts/recurrent_recovery_priority_controlled/policy_pool.pt",
                      map_location="cpu", weights_only=False)
    episodes = residual.source_episodes(expert, pool)
    assert len(episodes["expert"]) == 72 and len(episodes["policy"]) == 144
    assert all(torch.count_nonzero(ep["previous"][0]) == 0 for source in episodes.values() for ep in source)
    assert all(torch.equal(ep["previous"][1:], ep["targets"][:-1]) for ep in episodes["expert"])
    batches, audit = residual.make_schedule(episodes, replace(residual.ResidualConfig(), updates=2))
    assert audit["supervised_timesteps_by_delay"]["0"] > 0
    assert all(batch["mask"].shape == batch["states"].shape[:2] for batch in batches)
    pred = torch.randn_like(batches[0]["targets"], requires_grad=True)
    loss = residual.corrected_loss(pred, batches[0]["targets"], batches[0]["mask"], task)
    loss.backward()
    assert torch.count_nonzero(pred.grad[~batches[0]["mask"]]) == 0


def test_zero_residual_runner_matches_historical_ff_on_one_episode(task, design, inputs):
    _, _, specs = inputs
    heldout = sf.sample_episode_specs(16, 2811 + 60_000, task, "iid")
    base = sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME, task, torch.device("cpu"))
    model = _model("mlp", task, design)
    env = sf.make_env(task, 2811 + 864_211)
    try:
        ff_row, ff_trace = rf.rollout(env, heldout[0], base, "ff", task, rf.TemporalConfig(),
                                      torch.device("cpu"), mode="fresh", severity=0,
                                      collect=False, record_dynamics=True)
        residual_row, residual_trace = residual.rollout_residual(env, heldout[0], model, task,
                                                                  rf.TemporalConfig(), torch.device("cpu"))
    finally:
        env.close()
    for key in ("success", "collision", "final_position_error", "final_orientation_error",
                "final_gripper_width_error", "trajectory_error", "minimum_safe_clearance"):
        assert residual_row[key] == pytest.approx(ff_row[key], abs=1e-7)
    assert np.allclose([r["processed_action"] for r in ff_trace],
                       [r["processed_action"] for r in residual_trace], atol=1e-7)
    historical = __import__("json").loads(residual.HISTORICAL_FRESH.read_text())
    assert ff_row["final_position_error"] == pytest.approx(historical[0]["final_position_error"], abs=1e-7)


def test_heldout_specs_and_reset_mode_are_deterministic(task, design, inputs):
    _, _, train_specs = inputs
    heldout = sf.sample_episode_specs(16, 2811 + 60_000, task, "iid")
    assert not {s.sample_identity for s in train_specs} & {s.sample_identity for s in heldout}
    model = _model("gru", task, design)
    state = torch.zeros(1, 14)
    points = torch.zeros(1, 32, 3)
    previous = torch.zeros(1, 5)
    with torch.no_grad():
        _, _, hidden, _ = model.step(state, points, previous)
        a, _, _, _ = model.step(state, points, previous, hidden, reset_hidden=True)
        b, _, _, _ = model.step(state, points, previous, hidden, reset_hidden=True)
    assert torch.equal(a, b)


def test_visual_delay_age_and_fresh_proprioception():
    observer = residual.VisualDelayObserver(2)
    for t in range(5):
        state = np.full(14, float(t), dtype=np.float32)
        points = np.full((32, 3), float(t), dtype=np.float32)
        seen_state, seen_points, age = observer.observe(state, points)
        old_t = max(0, t - 2)
        assert age == t - old_t
        assert np.all(seen_state[:5] == old_t)
        assert np.all(seen_state[5:] == t)
        assert np.all(seen_points == old_t)


def test_training_delay_matches_rollout_delay_feature_split():
    states = torch.arange(5, dtype=torch.float32)[:, None].repeat(1, 14)
    points = torch.arange(5, dtype=torch.float32)[:, None, None].repeat(1, 32, 3)
    states[:, 5:] += 100
    observed_states, observed_points = residual.delayed_sequence(states, points, 2)
    observer = residual.VisualDelayObserver(2)
    for t in range(5):
        s, p, _ = observer.observe(states[t].numpy(), points[t].numpy())
        assert np.array_equal(s, observed_states[t].numpy())
        assert np.array_equal(p, observed_points[t].numpy())


def test_saved_fresh_ff_replays_every_historical_episode():
    path = residual.OUTPUT / "metrics/fresh/per_episode/ff.json"
    if not path.exists():
        pytest.skip("Fresh experiment artifacts have not been generated")
    current = json.loads(path.read_text())
    historical = json.loads(residual.HISTORICAL_FRESH.read_text())
    assert [r["spec_signature"] for r in current] == [r["spec_signature"] for r in historical]
    for a, b in zip(current, historical, strict=True):
        for key in ("success", "collision", "final_position_error", "final_orientation_error",
                    "final_gripper_width_error", "trajectory_error", "minimum_safe_clearance"):
            assert a[key] == pytest.approx(b[key], abs=1e-7)


def test_saved_three_controllers_share_episode_specs_and_no_evaluation_oracle():
    root = residual.OUTPUT / "metrics/fresh"
    if not (root / "summary.json").exists():
        pytest.skip("Fresh experiment artifacts have not been generated")
    signatures = []
    for name in residual.CONTROLLERS:
        rows = json.loads((root / "per_episode" / f"{name}.json").read_text())
        traces = json.loads((root / "traces" / f"{name}.json").read_text())
        signatures.append([r["spec_signature"] for r in rows])
        assert all(step["oracle_action"] is None for ep in traces for step in ep["steps"])
    assert signatures[0] == signatures[1] == signatures[2]


def test_saved_frozen_weights_and_training_exposure_match():
    path = residual.OUTPUT / "summary.json"
    if not path.exists():
        pytest.skip("Residual experiment artifacts have not been generated")
    summary = json.loads(path.read_text())
    mlp, gru = (summary["training"][name] for name in ("residual_mlp", "residual_gru"))
    assert mlp["base_model_hash_before_after_equal"]
    assert gru["base_model_hash_before_after_equal"]
    assert mlp["schedule_sha256"] == gru["schedule_sha256"] == summary["schedule"]["schedule_sha256"]
    assert mlp["supervised_timesteps_by_delay"] == gru["supervised_timesteps_by_delay"]
    assert sum(mlp["supervised_timesteps_by_delay"].values()) > 0
