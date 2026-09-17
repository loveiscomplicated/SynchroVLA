from __future__ import annotations

from typing import Literal

import torch

from vla_gnn_recurrent.graph.types import GraphData
from vla_gnn_recurrent.utils import clamp_delta

ActionPrior = Literal["none", "geometric"]


def bounded_action(raw_action: torch.Tensor, max_step: float) -> torch.Tensor:
    """Bound action vectors to max_step L2 norm without a sign-saturating tanh."""
    return clamp_delta(raw_action, max_step)


def geometric_action_prior(graph: GraphData, max_step: float) -> torch.Tensor:
    """Closed-loop proportional prior from the graph's current EE-target geometry."""
    relative = graph.node_features[graph.target_node_index, 0:3] - graph.node_features[graph.ee_node_index, 0:3]
    return clamp_delta(relative, max_step)


def compose_action(
    raw_action: torch.Tensor,
    graph: GraphData,
    max_step: float,
    action_prior: ActionPrior,
    residual_scale: float,
) -> torch.Tensor:
    """Create the final bounded Delta EE action.

    action_prior="none": learned-only controller output.
    action_prior="geometric": graph geometric prior plus learned residual.
    """
    if action_prior == "none":
        return bounded_action(raw_action, max_step).squeeze(0)
    if action_prior == "geometric":
        residual = bounded_action(raw_action, max_step * residual_scale).squeeze(0)
        prior = geometric_action_prior(graph, max_step)
        return clamp_delta(prior + residual, max_step)
    raise ValueError(f"Unknown action prior: {action_prior}")
