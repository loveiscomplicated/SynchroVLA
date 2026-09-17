import torch

from vla_gnn_recurrent.env.reaching_env import ReachingEnv, ReachingEnvConfig
from vla_gnn_recurrent.evaluation.evaluate import run_episode
from vla_gnn_recurrent.graph.graph_builder import GraphBuilder
from vla_gnn_recurrent.models.recurrent_controller import RecurrentController


def _run_recurrent(mode: str, observation_interval: int = 1, perturb_step: int | None = None):
    env = ReachingEnv(ReachingEnvConfig(max_steps=8), seed=12)
    model = RecurrentController(action_prior="none")
    return run_episode(
        model=model,
        model_kind="recurrent",
        env=env,
        graph_builder=GraphBuilder(target_velocity_feature=False),
        device=torch.device("cpu"),
        perturb_step=perturb_step,
        distance_range=(1.05, 1.1) if perturb_step is not None else None,
        task="static",
        observation_interval=observation_interval,
        target_velocity_feature=False,
        recurrent_mode=mode,  # type: ignore[arg-type]
    )


def test_gru_normal_hidden_persists_between_timesteps() -> None:
    result = _run_recurrent("normal")
    diagnostics = result["hidden_diagnostics"]

    assert diagnostics[0]["hidden_input_norm"] == 0.0
    assert diagnostics[1]["hidden_input_norm"] > 0.0
    assert not any(item["reset_applied"] for item in diagnostics)


def test_gru_step_reset_receives_zero_hidden_every_timestep() -> None:
    result = _run_recurrent("step_reset")
    diagnostics = result["hidden_diagnostics"]

    assert len(diagnostics) > 2
    assert all(item["reset_applied"] for item in diagnostics)
    assert all(item["hidden_input_norm"] == 0.0 for item in diagnostics)


def test_gru_event_reset_waits_until_perturbation_is_observable() -> None:
    result = _run_recurrent("event_reset", observation_interval=4, perturb_step=2)
    diagnostics = result["hidden_diagnostics"]
    reset_steps = [item["step"] for item in diagnostics if item["reset_applied"]]

    assert result["perturb_true_step"] == 2
    assert result["observable_event_step"] == 4
    assert reset_steps == [4]
    assert next(item for item in diagnostics if item["step"] == 4)["hidden_input_norm"] == 0.0
    assert all(not item["reset_applied"] for item in diagnostics if item["step"] < 4)


def test_gru_event_reset_preserves_memory_before_event() -> None:
    result = _run_recurrent("event_reset", observation_interval=4, perturb_step=2)
    diagnostics = result["hidden_diagnostics"]

    step_1 = next(item for item in diagnostics if item["step"] == 1)
    step_2 = next(item for item in diagnostics if item["step"] == 2)
    assert step_1["hidden_input_norm"] > 0.0
    assert step_2["hidden_input_norm"] > 0.0
    assert not step_2["reset_applied"]
