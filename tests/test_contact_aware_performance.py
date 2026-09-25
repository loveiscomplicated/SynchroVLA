"""Contact coordination preserves the frozen controller and observable-only inputs."""

import ast
import csv
import json
from pathlib import Path

import numpy as np
import torch

from vla_gnn_recurrent.training import contact_aware_performance as contact
from vla_gnn_recurrent.training import dynamic_visual_delay_feasibility as dynamic
from vla_gnn_recurrent.training import motion_controller_performance as motion
from vla_gnn_recurrent.training import surface_graph_feasibility as sf


class _ConstantController:
    def __init__(self, action: torch.Tensor):
        self.action = action
        self.task = motion.task(2811)
        self.motion = None

    def step(self, state, points, previous, hidden=None, topology=None,
             reset_hidden=False):
        action = self.action.to(state.device).expand(len(state), -1)
        return action, action, None, torch.zeros_like(action)


def test_observed_geometry_is_order_invariant_and_uses_no_shape_input() -> None:
    spec = dynamic.dynamic_specs(2811, motion.task(2811), 1,
        dynamic.SELECTED_STEP_SPEEDS_M, "heldout")[0]
    world = sf.sample_surface_points_world(spec.initial.shape, 32, 2811)
    points = sf.localize_world(world, np.array([0.1, 0., .8]), .3)
    first = contact.observed_geometry(points, motion.task(2811).pregrasp_clearance)
    order = np.random.default_rng(2811).permutation(32)
    second = contact.observed_geometry(points[order], motion.task(2811).pregrasp_clearance)
    assert np.allclose(first["normal_local_xz"], second["normal_local_xz"])
    assert np.allclose(first["target_local_xz"], second["target_local_xz"])
    assert "shape" not in contact.observed_geometry.__code__.co_varnames


def test_close_gate_blocks_only_closing_when_not_aligned() -> None:
    points = torch.zeros(1, 32, 3)
    points[0, :, 0] = torch.linspace(.1, .2, 32)
    points[0, :, 2] = .005 * torch.sin(torch.linspace(0, 3.14, 32))
    state = torch.zeros(1, 14)
    previous = torch.zeros(1, 5)
    for command, expected in ((-.1, 0.), (.1, .1)):
        action = torch.tensor([[0., 0., 0., 0., command]])
        controller = contact.ContactController(_ConstantController(action), "close_gate")
        output = controller.step(state, points, previous)[0]
        assert abs(float(output[0, 4])-expected) < 1e-7


def test_near_mixed_removes_only_inward_normal_component() -> None:
    task = motion.task(2813)
    event = next(r for r in csv.DictReader((contact.OUTPUT /
        "collision_audit/collision_onsets.csv").open()) if
        r["episode_id"].startswith("dynamic_heldout_112813_0001"))
    t = int(event["timestep"])-1
    trace = next(r for r in csv.DictReader((motion.OUTPUT /
        "seed2813/final_diagnostics/fixed_horizon_traces.csv").open()) if
        r["episode_id"] == event["episode_id"] and int(r["timestep"]) == t and
        r["condition"] == "dynamic")
    spec = {s.identity: s for s in dynamic.dynamic_specs(2813, task, 16,
        dynamic.SELECTED_STEP_SPEEDS_M, "heldout")}[event["episode_id"]]
    ee = np.asarray(ast.literal_eval(trace["EE_position"]))
    yaw = float(trace["EE_yaw"])
    points = contact._observed_points(2813, spec, t, ee, yaw, "dynamic")
    left = np.asarray(ast.literal_eval(trace["left_tip"]))
    right = np.asarray(ast.literal_eval(trace["right_tip"]))
    tips = sf.localize_world(np.stack((left, right)), ee, yaw)
    state = torch.zeros(1, 14)
    state[0, 8:14] = torch.from_numpy(tips.reshape(-1))
    action = torch.tensor([[-.03, 0., .02, .2, -.02]])
    geometry = contact.observed_geometry(points, task.pregrasp_clearance)
    normal = geometry["normal_local_xz"]
    inward = -np.dot(action[0, [0, 2]].numpy(), normal)
    if inward <= 0:
        action[0, [0, 2]] = torch.from_numpy((-.02*normal).astype(np.float32))
    controller = contact.ContactController(_ConstantController(action),
        "near_mixed", json.loads((contact.OUTPUT /
        "approach_safety/training_zone/zone.json").read_text())["zone_m"])
    output = controller.step(state, torch.from_numpy(points[None]),
                             torch.zeros(1, 5))[0][0].numpy()
    assert controller.approach_blocked_steps == 1
    assert abs(float(np.dot(output[[0, 2]], normal))) < 1e-6
    assert np.allclose(output[[1, 3, 4]], action[0, [1, 3, 4]].numpy())


def test_reproduction_and_one_step_contact_parity() -> None:
    reproduced = json.loads((contact.OUTPUT / "reproduction/summary.json").read_text())
    for condition, expected in (("static", (46, 0, 1, 0, 1)),
                                ("dynamic", (38, 5, 2, 0, 8))):
        keys = ("success", "position_fail", "yaw_fail", "aperture_fail", "collision_fail")
        assert tuple(sum(reproduced[str(seed)][condition][key] for seed in contact.SEEDS)
                     for key in keys) == expected
    diagnostic = json.loads((contact.OUTPUT /
        "collision_audit/one_step_counterfactual/summary.json").read_text())
    assert diagnostic["contact_remaining_by_variant"]["baseline"] == 8
    assert diagnostic["contact_remaining_by_variant"]["no_closing"] == 8
    assert diagnostic["contact_remaining_by_variant"]["no_inward_translation"] == 4


def test_final_improvement_and_aperture_are_preserved() -> None:
    result = json.loads((contact.OUTPUT / "aggregate/summary.json").read_text())
    dynamic = result["totals"]["dynamic"]
    static = result["totals"]["static"]
    assert dynamic["baseline"]["success"] == 38
    assert dynamic["near_mixed"]["success"] == 40
    assert dynamic["near_mixed"]["collision_fail"] == 5
    assert dynamic["near_mixed"]["aperture_fail"] == 0
    assert static["near_mixed"]["success"] == 46
    assert static["near_mixed"]["aperture_fail"] == 0
    assert all(json.loads((contact.OUTPUT /
        f"seed{seed}/final_diagnostics/summary.json").read_text())["dynamic_official_parity"]
        for seed in contact.SEEDS)

