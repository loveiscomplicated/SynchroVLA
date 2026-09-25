"""Regression checks for the frozen aperture path and new yaw diagnostics."""

import json
import math
from dataclasses import replace

import numpy as np
import torch

from vla_gnn_recurrent.training import motion_controller_performance as motion
from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import surface_graph_feasibility as sf
from vla_gnn_recurrent.training.generalized_geometric_graph import wrap_angle
from vla_gnn_recurrent.utils import select_device


def test_wrap_boundary_is_two_degrees_and_action_uses_short_rotation() -> None:
    delta = float(wrap_angle(math.radians(179) - math.radians(-179)))
    assert math.isclose(math.degrees(abs(delta)), 2.0, abs_tol=1e-9)
    assert math.isclose(np.clip(delta, -.35, .35), delta, abs_tol=1e-12)
    assert math.isclose(abs(float(wrap_angle(-math.pi-math.pi))), 0.0, abs_tol=1e-12)


def test_yaw_spatial_features_are_permutation_invariant_and_observable() -> None:
    generator = torch.Generator().manual_seed(2811)
    state = torch.randn(2, 14, generator=generator)
    points = torch.randn(2, 32, 3, generator=generator)
    action = torch.randn(2, generator=generator)
    first = motion.yaw_spatial_features(state, points, action)
    order = torch.randperm(32, generator=generator)
    second = motion.yaw_spatial_features(state, points[:, order], action)
    assert first.shape == (2, 28)
    assert torch.allclose(first, second, atol=2e-6)
    assert "shape" not in motion.yaw_spatial_features.__code__.co_varnames


def test_yaw_head_zero_initialization_and_native_limit() -> None:
    head = motion.YawSpatialHead()
    assert torch.count_nonzero(head(torch.randn(4, 28))) == 0
    task = motion.task(2811)
    bound = min(task.max_delta_rotation, json.loads((motion.OUTPUT /
        "yaw_capacity/summary.json").read_text())["target_correction_abs_p95_rad"])
    assert 0 < bound <= task.max_delta_rotation
    raw = torch.full((4,), 100.0)
    assert torch.all(bound * torch.tanh(raw) <= bound)


def test_zero_yaw_head_reproduces_frozen_motion_and_aperture(tmp_path) -> None:
    device = select_device("cpu")
    combined = motion.load_combined(2811, device,
        motion.paths(2811)["updated_motion"])
    checkpoint = tmp_path / "zero_yaw.pt"
    torch.save({"model_state": motion.YawSpatialHead().state_dict(),
                "feature_mean": torch.zeros(1, 28),
                "feature_scale": torch.ones(1, 28),
                "correction_bound_rad": .3}, checkpoint)
    wrapper = motion.YawSpatialController(combined, checkpoint, device)
    split = motion._yaw_split("train")
    state = torch.from_numpy(split["state"][:1])
    points = torch.from_numpy(split["points"][:1])
    previous = torch.zeros(1, 5)
    with torch.no_grad():
        original = combined.step(state, points, previous)[0]
        wrapped = wrapper.step(state, points, previous)[0]
    assert torch.equal(original, wrapped)


def test_saved_final_controller_keeps_32_points_100_edges_and_aperture_checkpoint() -> None:
    task = motion.task(2811)
    device = select_device("cpu")
    state = torch.zeros(1, 14)
    points = torch.zeros(1, 32, 3)
    _, _, topology = rf._graph_tensors(state[0].numpy(), points[0].numpy(), task, device)
    assert topology.src.shape[1] == 100
    for seed in motion.SEEDS:
        expected = motion.sha(motion.paths(seed)["aperture"])
        payload = torch.load(motion.OUTPUT / f"seed{seed}/yaw_spatial_readout/selected.pt",
                             map_location="cpu", weights_only=False)
        assert payload["aperture_checkpoint_sha256"] == expected
        assert payload["topology"] == "canonical_32_points_100_edges_unchanged"


def test_yaw_probe_episode_splits_and_counterfactual_robot_state() -> None:
    splits = {name: set(motion._yaw_split(name)["identity"])
              for name in ("train", "validation", "heldout")}
    assert not splits["train"] & splits["validation"]
    assert not splits["train"] & splits["heldout"]
    assert not splits["validation"] & splits["heldout"]
    rows = list(__import__("csv").DictReader((motion.OUTPUT /
        "yaw_counterfactual/paired_response.csv").open()))
    assert len(rows) == 16
    assert all(float(r["robot_state_other_max_delta"]) == 0 for r in rows)
    assert all(r["same_approach_side"] == "True" for r in rows)


def test_final_dynamic_trace_matches_official_rollout() -> None:
    for seed in motion.SEEDS:
        result = json.loads((motion.OUTPUT / f"seed{seed}/final_diagnostics/summary.json").read_text())
        assert result["dynamic_official_parity"]
        assert result["traces"] == 640
