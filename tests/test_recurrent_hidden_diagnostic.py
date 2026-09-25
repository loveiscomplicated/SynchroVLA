"""Hidden-state reset diagnostics must change only recurrent memory persistence."""

import numpy as np
import pytest
import torch
from dataclasses import replace

from vla_gnn_recurrent.training import recurrent_feasibility as rf
from vla_gnn_recurrent.training import recurrent_hidden_diagnostic as hd
from vla_gnn_recurrent.training import surface_graph_feasibility as sf


def test_hidden_reset_schedule() -> None:
    assert [rf.hidden_reset_due("normal", t) for t in range(9)] == [True] + [False] * 8
    assert [rf.hidden_reset_due("reset_1", t) for t in range(9)] == [True] * 9
    assert [rf.hidden_reset_due("reset_4", t) for t in range(9)] == [True, False, False, False] * 2 + [True]
    assert [rf.hidden_reset_due("reset_8", t) for t in range(9)] == [True] + [False] * 7 + [True]
    with pytest.raises(ValueError):
        rf.hidden_reset_due("reset_0", 0)


def test_rollout_hidden_carry_and_resets_keep_physical_spec_identical(tmp_path, monkeypatch) -> None:
    config = replace(sf.SurfaceFeasibilityConfig(output_dir=str(tmp_path), max_steps=9),
                     symmetric_robot_edges=True)
    spec = sf.sample_episode_specs(1, 818, config, "iid")[0]
    model = rf.GraphGRUPolicy(config).eval()
    seen: list[float | None] = []

    def fake_step(model, controller, state, points, hidden, cfg, device, diagnostics=None):
        del model, controller, state, points, cfg, device
        seen.append(None if hidden is None else float(hidden.flatten()[0]))
        value = 1.0 if hidden is None else float(hidden.flatten()[0]) + 1.0
        diagnostics["raw_predicted_action"] = [0.0] * 5
        diagnostics["hidden_delta_norm"] = 1.0
        return np.zeros(5, dtype=np.float32), torch.full((1, 1, 256), value)

    monkeypatch.setattr(rf, "controller_step", fake_step)
    results = {}
    env = sf.make_env(config, 818)
    try:
        for policy in hd.POLICIES:
            seen.clear()
            row, trace = rf.rollout(env, spec, model, "gru", config, rf.TemporalConfig(),
                                    torch.device("cpu"), hidden_policy=policy)
            results[policy] = (row, trace, seen.copy())
    finally:
        env.close()
    for policy, (row, trace, _) in results.items():
        assert row["spec_signature"] == results["normal"][0]["spec_signature"]
        assert trace[0]["true_ee_pose"] == results["normal"][1][0]["true_ee_pose"]
        assert [r["observed_graph_timestamp"] for r in trace] == [r["observed_graph_timestamp"] for r in results["normal"][1]]
        assert [r["hidden_reset"] for r in trace] == [rf.hidden_reset_due(policy, t) for t in range(len(trace))]
        assert all(r["hidden_policy"] == policy for r in trace)
        assert all(r["raw_predicted_action"] == [0.0] * 5 for r in trace)
    assert results["normal"][2] == [None] + [float(i) for i in range(1, 9)]
    assert results["reset_1"][2] == [None] * 9
    assert results["reset_4"][2] == [None, 1.0, 2.0, 3.0, None, 1.0, 2.0, 3.0, None]
    assert results["reset_8"][2] == [None] + [float(i) for i in range(1, 8)] + [None]


def test_fixed_replay_identical_graph_stream_topology_and_unchanged_weights(monkeypatch) -> None:
    config = replace(sf.SurfaceFeasibilityConfig(), symmetric_robot_edges=True)
    generator = torch.Generator().manual_seed(47)
    data = {
        "states": torch.randn(4, sf.STATE_DIM, generator=generator) * 0.1,
        "surface_points": torch.randn(4, sf.SURFACE_POINTS, 3, generator=generator) * 0.03,
        "actions": torch.zeros(4, sf.ACTION_DIM),
        "episode_ids": torch.tensor([4, 4, 4, 4]),
        "steps": torch.arange(4),
        "loss_mask": torch.tensor([True, False, True, True]),
    }
    model = rf.GraphGRUPolicy(config).eval()
    before_model = hd._model_digest(model)
    before_input = hd._tensor_digest(data)
    topology_rows = []
    original = rf._graph_tensors

    def watch(state, points, cfg, device):
        result = original(state, points, cfg, device)
        topology = result[2]
        topology_rows.append((topology.src.clone(), topology.dst.clone(), topology.edge_type.clone()))
        return result

    monkeypatch.setattr(rf, "_graph_tensors", watch)
    result = hd.replay_fixed_sequences(model, data, config, torch.device("cpu"))
    assert hd._model_digest(model) == before_model
    assert hd._tensor_digest(data) == before_input == result["input_sha256"]
    carry = result["records"]["normal"]
    reset = result["records"]["reset_1"]
    assert [(r["episode_id"], r["timestep"], r["input_sha256"]) for r in carry] == [
        (r["episode_id"], r["timestep"], r["input_sha256"]) for r in reset]
    assert len(topology_rows) == 8
    assert result["summary"]["normal"]["all_context_states"] == 4
    assert result["summary"]["normal"]["samples"] == 3
    assert result["summary"]["normal"]["masked_severe_states"] == 1
    assert carry[1]["eligible_oracle_label"] is False
    for first, second in zip(topology_rows[:4], topology_rows[4:], strict=True):
        for left, right in zip(first, second, strict=True):
            assert torch.equal(left, right)
        assert first[0].shape[1] == 100
        assert not torch.any(first[2] == 3)
