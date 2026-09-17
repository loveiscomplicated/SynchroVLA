import torch

from vla_gnn_recurrent.graph.graph_builder import GraphBuilder
from vla_gnn_recurrent.graph.types import ObjectObservation
from vla_gnn_recurrent.models.feedforward_controller import FeedForwardController
from vla_gnn_recurrent.models.recurrent_controller import RecurrentController


def _graph():
    builder = GraphBuilder()
    target = ObjectObservation.from_position("target", [0.4, 0.1, -0.2])
    return builder.build(ee_position=[0.0, 0.0, 0.0], target=target)


def test_feedforward_controller_forward_shape_and_bound() -> None:
    model = FeedForwardController(max_step=0.08, action_prior="none")
    action = model(_graph())

    assert action.shape == (3,)
    assert float(action.detach().norm()) <= 0.08001


def test_recurrent_controller_forward_shape_hidden_and_bound() -> None:
    model = RecurrentController(max_step=0.08, action_prior="none")
    hidden = model.initial_hidden(torch.device("cpu"))
    action, next_hidden = model(_graph(), hidden)

    assert action.shape == (3,)
    assert next_hidden.shape == (2, 1, 256)
    assert float(action.detach().norm()) <= 0.08001


def test_gru_hidden_changes_and_resets() -> None:
    model = RecurrentController(max_step=0.08, action_prior="none")
    hidden0 = model.initial_hidden(torch.device("cpu"))
    _, hidden1 = model(_graph(), hidden0)
    hidden_reset = model.initial_hidden(torch.device("cpu"))

    assert not torch.allclose(hidden0, hidden1)
    assert torch.allclose(hidden0, hidden_reset)


def test_learned_only_mode_has_no_direct_geometric_shortcut() -> None:
    graph = _graph()
    learned_only = FeedForwardController(max_step=0.08, action_prior="none")
    geometric = FeedForwardController(max_step=0.08, action_prior="geometric")
    for model in (learned_only, geometric):
        for parameter in model.parameters():
            parameter.data.zero_()

    learned_action = learned_only(graph)
    geometric_action = geometric(graph)

    assert torch.allclose(learned_action, torch.zeros(3), atol=1e-7)
    assert float(geometric_action.detach().norm()) > 0.0
    assert float(geometric_action.detach().norm()) <= 0.08001
