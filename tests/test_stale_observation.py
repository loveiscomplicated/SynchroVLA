import torch

from vla_gnn_recurrent.env.reaching_env import ReachingEnv, ReachingEnvConfig
from vla_gnn_recurrent.env.stale_observation import StaleTargetObserver
from vla_gnn_recurrent.graph.graph_builder import GraphBuilder


def test_n1_observed_target_equals_true_target() -> None:
    env = ReachingEnv(ReachingEnvConfig(max_steps=8), seed=1)
    env.reset(
        ee_position=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
        target_velocity=[0.01, 0.02, 0.0],
    )
    observer = StaleTargetObserver(observation_interval=1, target_velocity_feature=False)

    for _ in range(4):
        observed = observer.observe(env.observe())
        assert torch.allclose(observed.observed_target.position, observed.true_target.position)
        env.step(torch.zeros(3))


def test_n4_observed_target_stays_stale_between_reanchors() -> None:
    env = ReachingEnv(ReachingEnvConfig(max_steps=8), seed=2)
    env.reset(
        ee_position=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
        target_velocity=[0.01, 0.0, 0.0],
    )
    observer = StaleTargetObserver(observation_interval=4, target_velocity_feature=False)

    first = observer.observe(env.observe()).observed_target.position.clone()
    for _ in range(3):
        env.step(torch.zeros(3))
        observed = observer.observe(env.observe())
        assert torch.allclose(observed.observed_target.position, first)

    env.step(torch.zeros(3))
    reanchored = observer.observe(env.observe())
    assert reanchored.reanchored
    assert torch.allclose(reanchored.observed_target.position, reanchored.true_target.position)
    assert not torch.allclose(reanchored.observed_target.position, first)


def test_stale_graph_does_not_leak_true_target_features() -> None:
    env = ReachingEnv(ReachingEnvConfig(max_steps=8), seed=3)
    env.reset(
        ee_position=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
        target_velocity=[0.02, 0.0, 0.0],
    )
    observer = StaleTargetObserver(observation_interval=4, target_velocity_feature=False)
    graph_builder = GraphBuilder(target_velocity_feature=False)

    first = observer.observe(env.observe())
    env.step(torch.tensor([0.01, 0.0, 0.0]))
    stale = observer.observe(env.observe())
    graph = graph_builder.build(
        ee_position=stale.ee_position,
        ee_velocity=stale.ee_velocity,
        target=stale.observed_target,
    )

    assert torch.allclose(graph.node_features[graph.target_node_index, 0:3], first.observed_target.position)
    assert not torch.allclose(graph.node_features[graph.target_node_index, 0:3], stale.true_target.position)
    expected_relative = first.observed_target.position - stale.ee_position
    assert torch.allclose(graph.edge_features[0, 0:3], expected_relative)
    true_relative = stale.true_target.position - stale.ee_position
    assert not torch.allclose(graph.edge_features[0, 0:3], true_relative)
    assert torch.allclose(graph.node_features[graph.target_node_index, 3:6], torch.zeros(3))
