"""Contracts for the aperture diagnosis and conservative 5D action path."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from vla_gnn_recurrent.training import aperture_bottleneck_diagnosis as aperture
from vla_gnn_recurrent.training import dynamic_visual_delay_feasibility as dynamic
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import residual_recurrent_feasibility as residual
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.utils import select_device


@pytest.fixture(scope="module")
def task():
    return aperture.task_config()


def test_action_dimension_four_is_opening_fraction_delta(task):
    assert sf.ACTION_DIM == 5
    assert sf.MIN_GRIPPER_WIDTH == pytest.approx(0.065)
    assert sf.MAX_GRIPPER_WIDTH == pytest.approx(0.105)
    assert sf.opening_fraction(0.085) == pytest.approx(0.5)
    applied, target = sf.clamp_gripper_action(0.10, 0.5, task.max_delta_gripper)
    assert applied == pytest.approx(0.10)
    assert target == pytest.approx(0.60)
    assert sf.opening_fraction_to_command(target) == pytest.approx(0.40)
    env = sf.make_env(task, 2811)
    try:
        spec = dynamic.dynamic_specs(2811, task, 1, dynamic.SELECTED_STEP_SPEEDS_M, "heldout")[0]
        sf._reset_surface_env(env, spec.initial)
        sf._set_gripper_open_fraction_kinematic(env, 0.5)
        before = sf.gripper_width(env)
        sf.apply_local_action(env, np.array([0, 0, 0, 0, 0.1], dtype=np.float32), task)
        after = sf.gripper_width(env)
    finally:
        env.close()
    assert after > before
    assert after - before == pytest.approx(0.004, abs=0.001)


def test_oracle_width_and_aperture_success_semantics(task):
    spec = dynamic.dynamic_specs(2811, task, 1, dynamic.SELECTED_STEP_SPEEDS_M, "heldout")[0]
    target = sf._surface_target(spec.initial.shape, np.zeros(3), task.pregrasp_clearance)
    assert target[3] == pytest.approx(aperture.desired_width(spec.initial.shape))
    assert target[3] == pytest.approx(2 * sf.profile_half_width(0.5, spec.initial.shape) + 0.010)
    assert task.success_opening_threshold == pytest.approx(0.008)


@pytest.mark.parametrize("kind", ("mlp", "gru"))
def test_fifth_residual_zero_init_shape_bound_and_frozen_base(task, kind):
    device = select_device("cpu")
    design = replace(residual.ResidualConfig(), corrected_dimensions=5)
    model = residual.load_controller(kind, task, design, device)
    state = torch.randn(2, sf.STATE_DIM) * 0.1
    points = torch.randn(2, 32, 3) * 0.01
    previous = torch.zeros(2, 5)
    before = {key: value.clone() for key, value in model.base.state_dict().items()}
    with torch.no_grad():
        corrected, correction, _, base = model.step(state, points, previous)
    assert corrected.shape == (2, 5) and correction.shape == (2, 5)
    assert torch.equal(corrected, base)
    assert torch.count_nonzero(correction) == 0
    assert model.residual_bounds[4].item() == pytest.approx(0.2 * task.max_delta_gripper)
    with torch.no_grad():
        model.residual_head.bias[4] = 100
        corrected, correction, _, base = model.step(state, points, previous)
    assert torch.all(correction.abs() <= model.residual_bounds + 1e-8)
    assert torch.allclose(corrected[:, 4] - base[:, 4], correction[:, 4])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
    prediction, _, _, _, _ = model.forward_sequence(state[:, None], points[:, None], previous[:, None])
    loss = prediction.square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.base.parameters())
    optimizer.step()
    assert all(torch.equal(before[key], value) for key, value in model.base.state_dict().items())


def test_raw_surface_width_is_observable_without_privileged_input(task):
    spec = dynamic.dynamic_specs(2811, task, 1, dynamic.SELECTED_STEP_SPEEDS_M, "heldout")[0]
    thin = replace(spec.initial.shape, half_width=0.029)
    thick = replace(spec.initial.shape, half_width=0.040)
    points_thin = sf.sample_surface_points_world(thin, 32, seed=42)
    points_thick = sf.sample_surface_points_world(thick, 32, seed=42)
    target_delta = aperture.desired_width(thick) - aperture.desired_width(thin)
    observed_delta = aperture.observed_width_proxy(points_thick) - aperture.observed_width_proxy(points_thin)
    assert target_delta > 0.02
    assert observed_delta == pytest.approx(target_delta, abs=0.001)
    permutation = np.random.default_rng(51).permutation(32)
    unordered_thin = aperture.unordered_endpoint_pair_width_proxy(points_thin)
    unordered_thick = aperture.unordered_endpoint_pair_width_proxy(points_thick)
    assert unordered_thin == pytest.approx(
        aperture.unordered_endpoint_pair_width_proxy(points_thin[permutation]), abs=1e-7)
    assert unordered_thick == pytest.approx(
        aperture.unordered_endpoint_pair_width_proxy(points_thick[permutation]), abs=1e-7)
    assert unordered_thick - unordered_thin == pytest.approx(target_delta, abs=0.001)
    assert not np.array_equal(points_thin, points_thick)


def test_raw_set_probe_is_permutation_invariant():
    model = aperture.RawSetProbe().eval()
    state = torch.randn(3, sf.STATE_DIM)
    points = torch.randn(3, 32, 3)
    permuted = points[:, torch.randperm(32)]
    with torch.no_grad():
        assert torch.allclose(model(state, points), model(state, permuted), atol=1e-6)


def test_probe_episode_splits_and_raw_input_no_shape_leakage(task):
    train, validation, heldout = (aperture.load_split(name, task)
                                   for name in ("train", "validation", "heldout"))
    train_ids, val_ids, test_ids = (set(part["identity"]) for part in (train, validation, heldout))
    assert not train_ids & val_ids
    assert not train_ids & test_ids
    assert not val_ids & test_ids
    assert len(train_ids) == 72 and len(val_ids) == len(test_ids) == 16
    raw = np.concatenate([train["state"], train["points"].reshape(len(train["points"]), -1)], axis=1)
    assert raw.shape[1] == sf.STATE_DIM + 32 * 3
    assert "shape" not in train
    assert "target_width" not in train


def test_canonical_train_rollout_eval_topology_and_historical_4d(task):
    device = select_device("cpu")
    spec = dynamic.dynamic_specs(2811, task, 1, dynamic.SELECTED_STEP_SPEEDS_M, "heldout")[0]
    env = sf.make_env(task, 2811 + 864_211)
    try:
        sf._reset_surface_env(env, spec.initial)
        state, points, _ = sf.observation_inputs(env, spec.initial.shape, 32,
                                                  sf._stable_seed(spec.initial.sample_identity))
        state_t, points_t, evaluation_topology = rf._graph_tensors(state, points, task, device)
        training_topology = sf._torch_topology(points_t, state_t, task, False)
        rf.assert_canonical_topology(evaluation_topology, task)
        rf.assert_canonical_topology(training_topology, task)
        train_edges = set(zip(training_topology.src[0].tolist(), training_topology.dst[0].tolist()))
        eval_edges = set(zip(evaluation_topology.src[0].tolist(), evaluation_topology.dst[0].tolist()))
        assert train_edges == eval_edges
        assert (1, 2) not in train_edges and (2, 1) not in train_edges
        model = residual.load_controller("mlp", task, residual.ResidualConfig(), device,
                  aperture.SOURCE / "training/stage_b_mlp/checkpoints/mlp.pt")
        with torch.no_grad():
            corrected, correction, _, base = model.step(state_t, points_t, torch.zeros(1, 5),
                                                          topology=evaluation_topology)
        assert torch.equal(corrected[:, 4], base[:, 4])
        assert correction.shape[-1] == 4
        row, _ = dynamic.rollout(env, spec, task, device, "residual_mlp", model, 0)
    finally:
        env.close()
    historical = json.loads((aperture.SOURCE / "evaluation/delay0/per_episode/residual_mlp.json").read_text())[0]
    assert row["success"] == historical["success"]
    assert row["final_aperture_error"] == pytest.approx(historical["final_aperture_error"], abs=1e-7)
