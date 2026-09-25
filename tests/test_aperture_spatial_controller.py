"""Checks for the unchanged canonical graph and aperture-aware controller."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from vla_gnn_recurrent.training import aperture_spatial_controller as spatial
from vla_gnn_recurrent.training import aperture_bottleneck_diagnosis as prior
from vla_gnn_recurrent.training import dynamic_visual_delay_feasibility as dynamic
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import residual_recurrent_feasibility as residual
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.utils import select_device


def test_width_target_is_exact_oracle_label() -> None:
    task = spatial._task()
    shape = dynamic.dynamic_specs(2811, task, 1, dynamic.SELECTED_STEP_SPEEDS_M, "heldout")[0].initial.shape
    assert spatial.width_target(shape) == 2 * sf.profile_half_width(.5, shape) + .010


def test_train_only_width_normalization_roundtrip() -> None:
    train = spatial._supervised_split("train", spatial._task())
    center, scale = spatial.width_transform(train["width"])
    values = torch.as_tensor(train["width"][:8])
    assert torch.allclose(spatial.physical_width(spatial.normalized_width(values, center, scale),
                                                 center, scale), values)
    assert center == float(np.mean(train["width"]))


def test_episode_splits_are_disjoint() -> None:
    splits = spatial._assert_splits()
    ids = {name: set(value["identity"]) for name, value in splits.items()}
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["heldout"]
    assert not ids["validation"] & ids["heldout"]


def test_counterfactual_varies_surface_only() -> None:
    task = spatial._task()
    spec = dynamic.dynamic_specs(2811, task, 1, dynamic.SELECTED_STEP_SPEEDS_M, "heldout")[0]
    shape_a = replace(spec.initial.shape, half_width=.029)
    shape_b = replace(spec.initial.shape, half_width=.040)
    state = spatial._supervised_split("heldout", task)["state"][0].copy()
    point_a = sf.sample_surface_points_world(shape_a, 32, 47)
    point_b = sf.sample_surface_points_world(shape_b, 32, 47)
    assert np.array_equal(state, state.copy())
    assert all(asdict(shape_a)[key] == asdict(shape_b)[key]
               for key in asdict(shape_a) if key != "half_width")
    assert not np.array_equal(point_a, point_b)
    assert spatial.width_target(shape_b) > spatial.width_target(shape_a)


def test_canonical_topology_and_graph_size_unchanged() -> None:
    task = spatial._task()
    split = spatial._supervised_split("train", task)
    spatial._check_topology(task, split["state"][0], split["points"][0])
    state_t, point_t, topology = rf._graph_tensors(split["state"][0], split["points"][0],
                                                   task, select_device("cpu"))
    assert point_t.shape == (1, 32, 3)
    assert topology.src.shape[1] == 100
    assert not sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME,
                                     task, select_device("cpu")).use_local_edges


def test_pairwise_branch_is_permutation_invariant_and_has_no_shape_input() -> None:
    split = spatial._supervised_split("train", spatial._task())
    points = split["points"][0]
    first = spatial.pairwise_span_features(points)
    permuted = spatial.pairwise_span_features(points[np.random.default_rng(47).permutation(32)])
    assert np.allclose(first, permuted, atol=1e-6)
    assert "shape" not in spatial.pairwise_span_features.__code__.co_varnames
    assert spatial.ApertureSpanHead(0.08, 0.01).net[0].in_features == 8


def test_zero_initialized_5d_action_path_reproduces_ff() -> None:
    task = spatial._task()
    device = select_device("cpu")
    base = sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME, task, device)
    model = spatial.ApertureAwareController(base, task, .08, .01)
    split = spatial._supervised_split("train", task)
    state = torch.from_numpy(split["state"][:2])
    points = torch.from_numpy(split["points"][:2])
    previous = torch.zeros(2, 5)
    with torch.no_grad():
        action, correction, ff, _, _ = model(state, points, previous)
    assert torch.equal(action, ff)
    assert torch.count_nonzero(correction) == 0


def test_5d_bounds_and_normalization() -> None:
    task = spatial._task()
    base = sf.load_surface_model(residual.BASE_CHECKPOINT, rf.GRAPH_NAME,
                                 task, select_device("cpu"))
    model = spatial.ApertureAwareController(base, task, .08, .01)
    assert torch.allclose(model.action_scales,
                          torch.tensor([task.max_delta_ee] * 3 +
                                       [task.max_delta_rotation, task.max_delta_gripper]))
    assert model.residual_bounds[4].item() == torch.tensor(.05).item()
    raw = torch.full((2, 5), 100.)
    bounded = model.residual_bounds * torch.tanh(raw)
    assert torch.all(bounded <= model.residual_bounds + 1e-7)


def test_separate_branch_correction_obeys_bound() -> None:
    task = spatial._task()
    device = select_device("cpu")
    motion = residual.load_controller("mlp", task,
        replace(residual.ResidualConfig(), corrected_dimensions=5),
        device, spatial.STAGE_B_5D)
    branch = spatial.SpanBranchController(motion, spatial.OUTPUT /
        "separate_aperture_branch/selected.pt", task, device, .05)
    split = spatial._supervised_split("train", task)
    state = torch.from_numpy(split["state"][:4])
    points = torch.from_numpy(split["points"][:4])
    previous = torch.from_numpy(split["previous"][:4])
    corrected, _, _, base = branch.step(state, points, previous)
    assert torch.all((corrected[:, 4] - base[:, 4]).abs() <= .0500001)
    assert len(branch.capacity_trace) == 4


def test_counterfactual_intermediate_embeddings_are_saved() -> None:
    path = spatial.OUTPUT / "counterfactual/paired_node_embeddings.npz"
    with np.load(path) as payload:
        assert len(payload["pair_id"]) == 16
        for model in ("frozen", "aware"):
            for node in ("h_EE", "h_left", "h_right"):
                assert payload[f"{model}_{node}"].shape == (16, 2, 64)
            assert payload[f"{model}_z"].shape == (16, 2, 192)


def test_static_dynamic_target_width_semantics_agree() -> None:
    task = spatial._task()
    dynamic_shape = dynamic.dynamic_specs(2811, task, 1,
        dynamic.SELECTED_STEP_SPEEDS_M, "heldout")[0].initial.shape
    width = spatial.width_target(dynamic_shape)
    target = sf._surface_target(dynamic_shape, np.array([0.0, 0.0, .60]),
                                task.pregrasp_clearance)
    assert abs(width - target[3]) < 1e-12


def test_historical_4d_5d_baseline_hashes_and_success() -> None:
    result = json.loads((spatial.OUTPUT / "reproduction/summary.json").read_text())
    assert result["checkpoint_sha256"]["historical_4d_mlp"] == spatial._sha(spatial.STAGE_B_4D)
    assert result["checkpoint_sha256"]["selected_5d_mlp"] == spatial._sha(spatial.STAGE_B_5D)
    assert (result["prior_dynamic_4d_success"], result["prior_dynamic_5d_success"]) == (1, 5)


def test_observable_geometry_is_near_exact_without_sampler_order() -> None:
    task = spatial._task()
    split = spatial._supervised_split("heldout", task)
    offset = json.loads((spatial.OUTPUT / "reproduction/summary.json").read_text())[
        "geometry_offset_fit_train_only_m"]
    estimated = np.asarray([spatial.pairwise_span_features(p)[3:5].mean() + offset
                            for p in split["points"]])
    assert float(np.mean(np.abs(estimated - split["width"]))) < .0005


def test_heldout_is_not_used_for_checkpoint_selection() -> None:
    shared = json.loads((spatial.OUTPUT / "shared_encoder/summary.json").read_text())
    branch = json.loads((spatial.OUTPUT / "separate_aperture_branch/summary.json").read_text())
    assert shared["validation_score"] >= 0
    assert branch["validation_normalized_mse"] >= 0
    assert shared["selected_update"] <= shared["final_update"]
    assert branch["selected_update"] <= branch["final_update"]
