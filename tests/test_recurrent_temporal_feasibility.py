"""Controlled FF continuation and temporal trace safeguards."""

from dataclasses import replace

import numpy as np
import torch

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import recurrent_temporal_feasibility as temporal_study
from vla_gnn_recurrent.training import surface_graph_feasibility as sf


def _tiny_data(mask: bool = True) -> dict:
    return {"states": torch.zeros(6, sf.STATE_DIM),
            "surface_points": torch.zeros(6, 32, 3),
            "actions": torch.zeros(6, sf.ACTION_DIM),
            "episode_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
            "steps": torch.tensor([0, 1, 2, 0, 1, 2]),
            "loss_mask": torch.tensor([True, mask, True, True, True, True])}


def test_matched_ff_continuation_uses_same_sequence_draws_and_mask_as_gru(tmp_path) -> None:
    config = replace(sf.SurfaceFeasibilityConfig(output_dir=str(tmp_path)), symmetric_robot_edges=True)
    temporal = replace(rf.TemporalConfig(), sequence_batch_size=4)
    expert = _tiny_data()
    policy = _tiny_data(mask=False)
    ff_start = tmp_path / "ff_start.pt"
    model = sf.build_surface_model(rf.GRAPH_NAME, config)
    torch.save({"model_name": rf.GRAPH_NAME, "model_state": model.state_dict(), "config": {}}, ff_start)
    gru = rf.GraphGRUPolicy(config)
    weights = np.array([.75, .25])
    ff = temporal_study.train_matched_ff(ff_start, expert, expert, policy, weights, config, temporal,
                                         torch.device("cpu"), tmp_path / "ff", 2)
    rec = rf.train_gru(gru, expert, expert, policy, config, temporal, 2811,
                       torch.device("cpu"), 2, tmp_path / "gru",
                       sampling_seed=temporal_study.CONTINUATION_SEED, validation_interval=25,
                       policy_episode_weights=weights, separate_source_rngs=True)
    assert ff["optimizer_updates"] == rec["updates"] == 2
    assert ff["expert_valid_supervised_states"] == rec["expert_valid_supervised_timesteps"]
    assert ff["policy_valid_supervised_states"] == rec["policy_valid_supervised_timesteps"]
    assert ff["expert_sequence_draw_sha256"] == rec["sampled_expert_episode_index_sha256"]
    assert ff["expert_policy_loss_weight"] == [.5, .5]
    assert not model.use_local_edges


def test_stale_recovery_uses_fresh_return_not_stale_start() -> None:
    config = sf.SurfaceFeasibilityConfig()
    temporal = rf.TemporalConfig(stale_start=3)
    trace = [{"t": t, "position_error": .08 if t < 6 else .01,
              "orientation_error": .4 if t < 6 else .1} for t in range(8)]
    result = temporal_study._stale_recovery(trace, 2, temporal, config)
    assert result["window_completed"]
    assert result["fresh_return_timestep"] == 5
    assert result["recovery_steps"] == 1
    assert result["peak_position_error"] == .08


def test_fresh_rollout_dynamics_logged_without_scripted_expert(monkeypatch) -> None:
    config = replace(sf.SurfaceFeasibilityConfig(), symmetric_robot_edges=True, max_steps=2)
    spec = sf.sample_episode_specs(1, 1881, config, "iid")[0]
    env = sf.make_env(config, 27)
    model = sf.build_surface_model(rf.GRAPH_NAME, config)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Evaluation called scripted expert")
    monkeypatch.setattr(sf, "expert_action", forbidden)
    try:
        _, trace = rf.rollout(env, spec, model, "ff", config, rf.TemporalConfig(),
                              torch.device("cpu"), record_dynamics=True)
    finally:
        env.close()
    assert trace
    assert trace[0]["oracle_action"] is None
    assert len(trace[0]["true_state"]) == sf.STATE_DIM
    assert len(trace[0]["next_true_state"]) == sf.STATE_DIM
    assert len(trace[0]["observed_surface_points"]) == 32
    assert trace[0]["observation_age"] == 0
