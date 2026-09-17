import math

import torch

from vla_gnn_recurrent.graph.manipulation_graph_builder import ManipulationGraphBuilder
from vla_gnn_recurrent.graph.types import GraphData
from vla_gnn_recurrent.models.pick_controller import PickRecurrentController, decode_pick_action
from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv, MujocoReachConfig
from vla_gnn_recurrent.training.pick_bc import (
    _alignment_success,
    _direction_magnitude_loss,
    _precision_weight,
    _saturation_ratio,
)


def test_direction_magnitude_action_respects_bound_and_can_approach_zero() -> None:
    bounded = decode_pick_action(torch.tensor([1.0, 0.0, 20.0, 0.0]), max_step=0.045, action_head_type="direction_magnitude")
    tiny = decode_pick_action(torch.tensor([1.0, 0.0, -100.0, 0.0]), max_step=0.045, action_head_type="direction_magnitude")

    assert float(bounded.delta_ee.norm()) <= 0.045 + 1e-6
    assert tiny.delta_ee.norm().item() < 1e-6
    assert 0.0 <= float(bounded.gripper.item()) <= 1.0


def test_zero_vector_expert_action_has_finite_direction_magnitude_loss() -> None:
    raw = torch.zeros(4, requires_grad=True)
    action = decode_pick_action(raw, max_step=0.045, action_head_type="direction_magnitude")
    pred_delta = torch.stack([action.delta_ee[0], action.delta_ee[2]])
    loss = _direction_magnitude_loss(
        raw=raw,
        pred_delta=pred_delta,
        target_delta=torch.zeros(2),
        max_step=0.045,
        direction_loss_weight=1.0,
        magnitude_loss_weight=1.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_precision_weight_uses_observable_graph_distance_only() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=4, control_substeps=5, pick_scene=True), seed=73)
    graph = ManipulationGraphBuilder(task="pick", include_object=True).build(
        env.reset_pick_scene(object_x=-0.01, randomize_robot=False)
    )
    near_graph = _with_object_offset(graph, torch.tensor([0.01, 0.0, 0.0]))
    medium_graph = _with_object_offset(graph, torch.tensor([0.07, 0.0, 0.0]))
    far_graph = _with_object_offset(graph, torch.tensor([0.15, 0.0, 0.0]))

    assert _precision_weight(near_graph, True, 0.05, 0.10, 4.0, 2.0).item() == 4.0
    assert _precision_weight(medium_graph, True, 0.05, 0.10, 4.0, 2.0).item() == 2.0
    assert _precision_weight(far_graph, True, 0.05, 0.10, 4.0, 2.0).item() == 1.0
    assert _precision_weight(near_graph, False, 0.05, 0.10, 4.0, 2.0).item() == 1.0


def test_saturation_metric_and_alignment_success_are_deterministic() -> None:
    results = [
        {
            "records": [
                {"predicted_action_magnitude": 0.041, "object_to_ee": 0.040},
                {"predicted_action_magnitude": 0.010, "object_to_ee": 0.030},
                {"predicted_action_magnitude": 0.045, "object_to_ee": 0.100},
            ]
        }
    ]

    assert math.isclose(_saturation_ratio(results, near_only=False), 2.0 / 3.0)
    assert math.isclose(_saturation_ratio(results, near_only=True), 0.5)
    assert _alignment_success(torch.tensor([0.02, 0.0, -0.03]).numpy(), 0.0)
    assert not _alignment_success(torch.tensor([0.08, 0.0, -0.03]).numpy(), 0.0)


def test_direction_magnitude_controller_has_no_geometric_shortcut() -> None:
    model = PickRecurrentController(max_step=0.045, action_head_type="direction_magnitude")

    assert not hasattr(model, "action_prior")
    assert model.action_head_type == "direction_magnitude"


def _with_object_offset(graph: GraphData, offset: torch.Tensor) -> GraphData:
    features = graph.node_features.clone()
    features[graph.target_node_index, 0:3] = features[graph.ee_node_index, 0:3] + offset
    return GraphData(
        node_features=features,
        edge_index=graph.edge_index,
        edge_features=graph.edge_features,
        node_types=graph.node_types,
        node_names=graph.node_names,
        ee_node_index=graph.ee_node_index,
        target_node_index=graph.target_node_index,
        task_node_index=graph.task_node_index,
    )
