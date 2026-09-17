import torch

from vla_gnn_recurrent.graph.graph_builder import GraphBuilder
from vla_gnn_recurrent.graph.types import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, ObjectObservation


def test_graph_builder_spatial_edge_features() -> None:
    builder = GraphBuilder()
    target = ObjectObservation.from_position("target", [1.0, 2.0, 3.0])
    graph = builder.build(ee_position=[0.0, 0.0, 0.0], ee_velocity=[0.0, 0.0, 0.0], target=target)

    assert graph.node_features.shape == (3, NODE_FEATURE_DIM)
    assert graph.edge_features.shape == (6, EDGE_FEATURE_DIM)
    assert graph.edge_index.shape == (2, 6)
    assert graph.node_types == ("end_effector", "object", "task")

    first_src, first_dst = graph.edge_index[:, 0].tolist()
    assert (first_src, first_dst) == (graph.ee_node_index, graph.target_node_index)
    assert torch.allclose(graph.edge_features[0, 0:3], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isclose(graph.edge_features[0, 3], torch.tensor((1.0 + 4.0 + 9.0) ** 0.5))


def test_graph_builder_optional_joint_edges() -> None:
    builder = GraphBuilder(include_joints=True, num_joints=2)
    target = ObjectObservation.from_position("target", [0.5, 0.0, 0.0])
    graph = builder.build(ee_position=[0.0, 0.2, 0.0], target=target)

    assert graph.num_nodes == 5
    assert "joint" in graph.node_types
    assert graph.num_edges == 10

