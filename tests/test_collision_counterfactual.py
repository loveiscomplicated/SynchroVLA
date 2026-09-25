"""Replay validity and observable-action invariants for collision diagnosis."""

import numpy as np
import pytest
import torch

from vla_gnn_recurrent.training import collision_counterfactual as cf
from vla_gnn_recurrent.training import dynamic_visual_delay_feasibility as dynamic
from vla_gnn_recurrent.training import motion_controller_performance as motion
from vla_gnn_recurrent.training import surface_graph_feasibility as sf


def _points() -> np.ndarray:
    x = np.linspace(-0.07, 0.07, 32)
    return np.stack((x, np.zeros_like(x), .05 + .002*x*x), axis=1).astype(np.float32)


@pytest.mark.parametrize("name,expected", [
    ("no_inward", 0.0), ("inward_025", .25),
    ("inward_050", .50), ("tangent_first", 0.0)])
def test_inward_intervention_preserves_tangent_yaw_and_gripper(name: str,
                                                               expected: float) -> None:
    points = _points()
    normal = np.asarray(cf.contact.observed_geometry(points, .04)["normal_local_xz"])
    tangent = np.array([-normal[1], normal[0]])
    action = np.array([*(-.02*normal+.007*tangent), .13, -.04], dtype=np.float32)
    action = np.array([action[0], 0., action[1], action[2], action[3]], dtype=np.float32)
    changed, detail = cf.intervene(action, points, np.zeros(14, dtype=np.float32),
                                   name, cf.ReplayConfig(), .04)
    assert detail["before"]["inward_m"] > .019
    assert detail["after"]["inward_m"] == pytest.approx(
        expected*detail["before"]["inward_m"], abs=2e-7)
    assert np.allclose(changed[[1, 3, 4]], action[[1, 3, 4]])
    assert np.allclose(detail["after"]["tangent_local_xz"],
                       detail["before"]["tangent_local_xz"], atol=2e-7)


def test_yaw_first_requires_observable_yaw_and_near_surface() -> None:
    points = _points()
    normal = np.asarray(cf.contact.observed_geometry(points, .04)["normal_local_xz"])
    action = np.array([-.02*normal[0], 0., -.02*normal[1], .2, -.01], dtype=np.float32)
    state = np.zeros(14, dtype=np.float32)
    state[8:11] = points[8]
    state[11:14] = points[20]
    active_cfg = cf.ReplayConfig(yaw_threshold_rad=0., near_surface_m=.1)
    _, active = cf.intervene(action, points, state, "yaw_first", active_cfg, .04)
    assert active["active"]
    inactive_cfg = cf.ReplayConfig(yaw_threshold_rad=4., near_surface_m=.1)
    unchanged, inactive = cf.intervene(action, points, state, "yaw_first", inactive_cfg, .04)
    assert not inactive["active"]
    assert np.array_equal(unchanged, action)


class _ConstantController:
    def step(self, state, points, previous, hidden=None, topology=None):
        action = torch.tensor([[.002, 0., -.001, .02, -.005]], device=state.device)
        return action, action, None, action


def test_native_snapshot_replays_fresh_dynamic_states(tmp_path) -> None:
    task = motion.task(2811)
    task.output_dir = str(tmp_path)
    spec = dynamic.dynamic_specs(2811, task, 1, dynamic.SELECTED_STEP_SPEEDS_M,
                                 "heldout")[0]
    env = sf.make_env(task, 2811)
    try:
        controller = _ConstantController()
        original, snapshots = cf.original_rollout(env, spec, controller, task,
                                                  torch.device("cpu"))
        for start in (0, 3, 9):
            check = cf.verify_start_snapshot(env, snapshots[start], spec,
                controller, task, torch.device("cpu"), original, 2e-5)
            assert check["max_absolute_difference"] <= 2e-5
        replayed, replay_outcome = cf.replay_from(env, snapshots[3], spec,
            controller, task, torch.device("cpu"), None, "single",
            cf.ReplayConfig(), stop_on_collision=False)
        parity = cf.verify_reproduction(original[3:], replayed,
            cf._terminal(original[3:], task), replay_outcome, 2e-5)
        assert max(parity["max_absolute_differences"].values()) <= 2e-5
    finally:
        env.close()


def test_intervention_uses_new_simulator_state_for_next_policy_call(tmp_path) -> None:
    task = motion.task(2811)
    task.output_dir = str(tmp_path)
    spec = dynamic.dynamic_specs(2811, task, 1, dynamic.SELECTED_STEP_SPEEDS_M,
                                 "heldout")[0]

    class InwardController:
        def __init__(self):
            self.states = []

        def step(self, state, points, previous, hidden=None, topology=None):
            self.states.append(state[0].detach().cpu().numpy().copy())
            normal = np.asarray(cf.contact.observed_geometry(
                points[0].detach().cpu().numpy(), task.pregrasp_clearance)["normal_local_xz"])
            action = torch.tensor([[-.02*normal[0], 0., -.02*normal[1], .0, .0]],
                                  dtype=torch.float32, device=state.device)
            return action, action, None, action

    env = sf.make_env(task, 2811)
    try:
        controller = InwardController()
        original, snapshots = cf.original_rollout(env, spec, controller, task,
                                                  torch.device("cpu"))
        controller.states.clear()
        replay, outcome = cf.replay_from(env, snapshots[0], spec, controller, task,
            torch.device("cpu"), "no_inward", "single", cf.ReplayConfig(),
            stop_on_collision=False)
        assert outcome["num_intervention_steps"] == 1
        assert replay[0]["intervention"]["active"]
        assert replay[0]["executed_action"] != original[0]["executed_action"]
        assert np.linalg.norm(np.asarray(replay[1]["ee_position"])-
                              np.asarray(original[1]["ee_position"])) > 1e-5
        assert np.allclose(controller.states[1], replay[1]["state_14d"])
        assert np.allclose(replay[1]["previous_executed_action"],
                           replay[0]["executed_action"])
    finally:
        env.close()


def test_aggregate_earliest_and_latest_rescue() -> None:
    rows = [{"episode_id": "e", "intervention": "no_inward", "mode": "single",
             "start_absolute_step": t, "success": t in (3, 5),
             "collision": t == 4, "collision_avoided_task_failed": False}
            for t in (3, 4, 5)]
    episode = {"episode_id": "e", "collision": True,
               "original_outcome": {"collision_step": 6}, "results": rows}
    result = cf.aggregate([episode], cf.ReplayConfig(
        interventions=("no_inward",), modes=("single",)))
    group = result["interventions"][0]
    assert group["rescued_episodes"] == 1
    window = group["episode_rescue_windows"][0]
    assert (window["earliest_rescuable_step"],
            window["latest_rescuable_step"]) == (3, 5)
    assert (window["earliest_rescuable_relative_step"],
            window["latest_rescuable_relative_step"]) == (-3, -1)
