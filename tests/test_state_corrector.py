import torch

from vla_gnn_recurrent.env.reaching_env import ReachingEnv, ReachingEnvConfig
from vla_gnn_recurrent.evaluation.evaluate import apply_fixed_decay, run_episode
from vla_gnn_recurrent.graph.graph_builder import GraphBuilder
from vla_gnn_recurrent.graph.types import ObjectObservation
from vla_gnn_recurrent.models.recurrent_controller import RecurrentController
from vla_gnn_recurrent.models.state_corrector import StateCorrector


def _graph():
    builder = GraphBuilder(target_velocity_feature=False)
    target = ObjectObservation.from_position("target", [0.4, 0.1, -0.2])
    return builder.build(ee_position=[0.0, 0.0, 0.0], target=target)


def test_state_corrector_preserves_gru_hidden_shape_and_corrects_both_layers() -> None:
    corrector = StateCorrector(hidden_dim=256, graph_dim=128, num_layers=2, mlp_hidden_dim=16)
    with torch.no_grad():
        for parameter in corrector.parameters():
            parameter.zero_()
        corrector.candidate_mlp[-1].bias.fill_(1.0)  # type: ignore[attr-defined]

    hidden = torch.zeros(2, 1, 256)
    graph_embedding = torch.zeros(1, 128)
    corrected, gate = corrector(hidden, graph_embedding)

    assert corrected.shape == (2, 1, 256)
    assert gate.shape == (2, 1, 256)
    assert torch.allclose(corrected[0], torch.full((1, 256), 0.5))
    assert torch.allclose(corrected[1], torch.full((1, 256), 0.5))


def test_fixed_decay_alpha_is_applied_exactly() -> None:
    hidden = torch.randn(2, 1, 256)
    decayed = apply_fixed_decay(hidden, 0.25)

    assert decayed is not None
    assert torch.allclose(decayed, hidden * 0.25)


def test_learned_correction_runs_only_when_perturbation_is_observable() -> None:
    env = ReachingEnv(ReachingEnvConfig(max_steps=8), seed=12)
    model = RecurrentController(action_prior="none")
    corrector = StateCorrector()

    result = run_episode(
        model=model,
        model_kind="recurrent",
        env=env,
        graph_builder=GraphBuilder(target_velocity_feature=False),
        device=torch.device("cpu"),
        perturb_step=2,
        distance_range=(1.05, 1.1),
        task="static",
        observation_interval=4,
        target_velocity_feature=False,
        recurrent_mode="learned_correction",
        corrector=corrector,
    )

    diagnostics = result["hidden_diagnostics"]
    correction_events = result["correction_events"]

    assert result["perturb_true_step"] == 2
    assert result["observable_event_step"] == 4
    assert [event["step"] for event in correction_events] == [4]
    assert all(not item["correction_applied"] for item in diagnostics if item["step"] < 4)
    event_diag = next(item for item in diagnostics if item["step"] == 4)
    assert event_diag["correction_applied"]
    assert not event_diag["reset_applied"]
    assert event_diag["correction_kind"] == "learned_correction"


def test_learned_correction_occurs_before_action_generation() -> None:
    env = ReachingEnv(ReachingEnvConfig(max_steps=8), seed=12)
    model = RecurrentController(action_prior="none")
    corrector = StateCorrector()

    result = run_episode(
        model=model,
        model_kind="recurrent",
        env=env,
        graph_builder=GraphBuilder(target_velocity_feature=False),
        device=torch.device("cpu"),
        perturb_step=2,
        distance_range=(1.05, 1.1),
        task="static",
        observation_interval=4,
        target_velocity_feature=False,
        recurrent_mode="learned_correction",
        corrector=corrector,
    )

    event = result["correction_events"][0]
    event_diag = next(item for item in result["hidden_diagnostics"] if item["step"] == event["step"])
    assert event_diag["hidden_input_norm"] == event["hidden_post_correction_norm"]


def test_frozen_controller_stays_fixed_while_corrector_gets_gradients() -> None:
    graph = _graph()
    controller = RecurrentController(action_prior="none")
    final = controller.action_head[-1]
    with torch.no_grad():
        if isinstance(final, torch.nn.Linear):
            final.weight.fill_(0.01)
            final.bias.fill_(0.02)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    before = {name: parameter.detach().clone() for name, parameter in controller.named_parameters()}

    corrector = StateCorrector()
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=1e-3)
    hidden = controller.initial_hidden(torch.device("cpu"))
    with torch.no_grad():
        graph_embedding = controller.gnn(graph)
    corrected, _ = corrector(hidden, graph_embedding)
    prediction, _ = controller.raw_action(graph, corrected)
    loss = torch.nn.functional.mse_loss(prediction, torch.tensor([0.04, -0.02, 0.01]))
    loss.backward()

    controller_grads = [parameter.grad for parameter in controller.parameters()]
    corrector_grad_norm = sum(
        float(parameter.grad.detach().norm().item())
        for parameter in corrector.parameters()
        if parameter.grad is not None
    )
    optimizer.step()

    assert all(grad is None for grad in controller_grads)
    assert corrector_grad_norm > 0.0
    for name, parameter in controller.named_parameters():
        assert torch.allclose(parameter.detach(), before[name])
