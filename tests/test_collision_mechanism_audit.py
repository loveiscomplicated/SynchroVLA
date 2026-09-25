"""Production-guard parity, replay causality, and official contact extraction."""

import json

import numpy as np
import torch

from vla_gnn_recurrent.training import collision_counterfactual as cf
from vla_gnn_recurrent.training import collision_mechanism_audit as audit
from vla_gnn_recurrent.training import contact_aware_performance as contact
from vla_gnn_recurrent.training import motion_controller_performance as motion
from vla_gnn_recurrent.training import surface_graph_feasibility as sf


class _ConstantInward:
    def __init__(self, task):
        self.task = task
        self.motion = None

    def step(self, state, points, previous, hidden=None, topology=None,
             reset_hidden=False):
        normal = contact.observed_geometry(points[0].detach().cpu().numpy(),
                                            self.task.pregrasp_clearance)["normal_local_xz"]
        tangent = np.array([-normal[1], normal[0]])
        translation = -.02*normal+.004*tangent
        action = torch.tensor([[translation[0], 0., translation[1], .10, -.01]],
                              dtype=torch.float32, device=state.device)
        return action, action, None, torch.zeros_like(action)


def _synthetic_points():
    x = np.linspace(-.06, .06, 32)
    return np.stack((x, np.zeros_like(x), .15+.001*x*x), axis=1).astype(np.float32)


def test_audited_predicates_match_production_guard_actions():
    task = motion.task(2811)
    points = _synthetic_points()
    state = np.zeros(14, dtype=np.float32)
    state[8:11] = points[8]
    state[11:14] = points[20]
    base = _ConstantInward(task)
    zone = .06459809315570239
    for near in (True, False):
        current = state.copy()
        if not near:
            current[8:14] += .3
        observed = audit._guard_observation(current, points, task, zone)
        guard = contact.ContactController(base, "near_mixed", zone)
        action, production = audit.production_guard_step(guard, current, points,
            np.zeros(5, dtype=np.float32), task, torch.device("cpu"))
        requested = np.asarray(production["requested_action"])
        normal = np.asarray(observed["normal_local_xz"])
        before = cf.decompose(requested, normal)
        after = cf.decompose(action, normal)
        assert observed["guard_condition"] is near
        assert production["diagnostic"]["approach_blocked"] is near
        assert np.allclose(action[[3, 4]], requested[[3, 4]])
        if near:
            assert after["inward_m"] < 2e-6
            assert np.allclose(after["tangent_local_xz"],
                               before["tangent_local_xz"], atol=2e-6)
        else:
            assert np.allclose(action, requested)


def test_rescue_window_lookup_matches_authoritative_artifact():
    config, _, rows, windows = audit._read_counterfactual(cf.OUTPUT)
    for seed, episode_id in audit.LARGE:
        identity = audit._spec(seed, episode_id).identity
        selected = audit._rescue_lookup(rows, identity)
        audit._validate_window_lookup(identity, selected, windows, config)
        assert selected


def test_same_snapshot_guard_and_no_inward_are_fresh_closed_loop():
    seed, episode_id = 2812, 1
    spec = audit._spec(seed, episode_id)
    task = motion.task(seed)
    device = torch.device("cpu")
    controller = contact._final_controller(seed, device)
    env = sf.make_env(task, seed+864_211)
    try:
        original, snapshots = cf.original_rollout(env, spec, controller, task,
                                                  device, seed=seed)
        start = 2
        baseline, base_outcome = cf.replay_from(env, snapshots[start], spec,
            controller, task, device, None, "persistent", cf.ReplayConfig(),
            stop_on_collision=False, seed=seed)
        cf.verify_reproduction(original[start:], baseline,
            cf._terminal(original[start:], task), base_outcome, 2e-5)
        guard = audit.RecordingGuard(controller, .06459809315570239)
        guarded, _ = cf.replay_from(env, snapshots[start], spec, guard,
            task, device, None, "persistent", cf.ReplayConfig(), seed=seed)
        no_inward, _ = cf.replay_from(env, snapshots[start], spec, controller,
            task, device, "no_inward", "persistent", cf.ReplayConfig(),
            seed=seed)
        assert np.allclose(guarded[0]["state_14d"], no_inward[0]["state_14d"])
        assert len(guard.history) >= 1
        for trace in (guarded, no_inward):
            if len(trace) > 1:
                assert np.allclose(trace[1]["previous_executed_action"],
                                   trace[0]["executed_action"])
    finally:
        env.close()


def test_official_contact_pairs_and_side_distances_at_onset():
    seed, episode_id = 2811, 4
    spec = audit._spec(seed, episode_id)
    task = motion.task(seed)
    device = torch.device("cpu")
    controller = contact._final_controller(seed, device)
    env = sf.make_env(task, seed+864_211)
    try:
        original, snapshots = cf.original_rollout(env, spec, controller,
                                                  task, device, seed=seed)
        onset = cf._terminal(original, task)["collision_step"]
        assert onset is not None
        cf.restore(env, snapshots[onset-1])
        pre_shape, _, _ = cf._observation(env, spec, task, onset-1)
        assert not audit.extract_contacts(env, pre_shape)["official_trigger_pairs"]
        cf.restore(env, snapshots[onset])
        shape, _, _ = cf._observation(env, spec, task, onset)
        extracted = audit.extract_contacts(env, shape)
        assert extracted["official_collision"]
        assert extracted["official_trigger_pairs"]
        assert all(row["robot_part"] == "right_finger_or_tip"
                   for row in extracted["official_trigger_pairs"])
        left = audit._tip_distance(env, shape, "left")
        right = audit._tip_distance(env, shape, "right")
        assert extracted["official_clearance_m"] <= min(left, right)+1e-7
        assert right < left
    finally:
        env.close()


def test_one_step_component_variant_only_claims_immediate_contact():
    seed, episode_id = 2811, 4
    spec = audit._spec(seed, episode_id)
    task = motion.task(seed)
    device = torch.device("cpu")
    controller = contact._final_controller(seed, device)
    env = sf.make_env(task, seed+864_211)
    try:
        original, snapshots = cf.original_rollout(env, spec, controller,
                                                  task, device, seed=seed)
        onset = cf._terminal(original, task)["collision_step"]
        assert onset is not None
        rows = audit._component_counterfactual(env, spec, original, snapshots,
                                               task, controller, device, onset)
        by_variant = {row["variant"]: row for row in rows}
        assert set(by_variant) == set(audit.ONE_STEP_VARIANTS)
        assert by_variant["original"]["immediate_collision"]
        assert not by_variant["no_tangent"]["immediate_collision"]
        assert by_variant["no_inward"]["immediate_collision"]
        assert not any("success" in row for row in rows)
    finally:
        env.close()
