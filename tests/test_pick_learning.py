import torch

from vla_gnn_recurrent.graph.manipulation_graph_builder import ManipulationGraphBuilder
from vla_gnn_recurrent.models.pick_controller import PickFeedForwardController, PickRecurrentController, decode_pick_action
from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv, MujocoReachConfig
from vla_gnn_recurrent.training.pick_bc import (
    PickDemoConfig,
    generate_pick_demonstrations,
    load_pick_dataset,
    run_closed_loop_pick_episode,
)


def test_pick_demonstrations_are_sequential_and_split_episode_disjoint(tmp_path) -> None:
    dataset_path = tmp_path / "demos.pt"
    metadata_path = tmp_path / "metadata.json"
    generate_pick_demonstrations(
        PickDemoConfig(
            episodes=6,
            seed=55,
            output_path=str(dataset_path),
            metadata_path=str(metadata_path),
        )
    )
    dataset = load_pick_dataset(dataset_path)
    split_sets = {name: set(indices) for name, indices in dataset["split"].items()}

    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    first = dataset["episodes"][0]
    assert len(first["graphs"]) == int(first["steps"])
    assert first["delta_xz"].shape == (first["steps"], 2)
    assert first["gripper"].shape == (first["steps"],)


def test_pick_policy_inputs_do_not_contain_expert_phase_or_waypoint(tmp_path) -> None:
    dataset_path = tmp_path / "demos.pt"
    metadata_path = tmp_path / "metadata.json"
    generate_pick_demonstrations(
        PickDemoConfig(episodes=2, seed=56, output_path=str(dataset_path), metadata_path=str(metadata_path))
    )
    dataset = load_pick_dataset(dataset_path)
    episode = dataset["episodes"][0]
    graph = episode["graphs"][0]

    assert "phase_labels_diagnostics_only" in episode
    assert "ee_target" not in episode
    assert "waypoint" not in episode
    assert all("PRE_GRASP" not in name and "ALIGN" not in name and "CLOSE" not in name for name in graph.node_names)
    assert all("target_pose" not in name and "waypoint" not in name for name in graph.node_names)


def test_pick_graph_uses_object_as_control_target_and_contains_gripper_state() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=16, control_substeps=20, pick_scene=True), seed=57)
    obs = env.reset_pick_scene(object_x=-0.01, randomize_robot=False)
    graph = ManipulationGraphBuilder(task="pick", include_object=True).build(obs)
    gripper_idx = graph.node_names.index("gripper")

    assert graph.node_types[graph.target_node_index] == "object"
    assert graph.node_names[graph.target_node_index] == "ball"
    assert torch.isclose(graph.node_features[gripper_idx, -2], obs.robot.gripper_state.reshape(()))
    interaction_edges = [
        idx
        for idx in range(graph.num_edges)
        if graph.node_types[int(graph.edge_index[0, idx])] == "gripper"
        and graph.node_types[int(graph.edge_index[1, idx])] == "object"
    ]
    assert interaction_edges
    assert graph.edge_features[interaction_edges[0], -1].item() in {0.0, 1.0}


def test_pick_action_bounds_and_no_geometric_prior_shortcut() -> None:
    recurrent = PickRecurrentController(max_step=0.045)
    feedforward = PickFeedForwardController(max_step=0.045)
    action = decode_pick_action(torch.tensor([100.0, -100.0, 100.0]), max_step=0.045)

    assert not hasattr(recurrent, "action_prior")
    assert not hasattr(feedforward, "action_prior")
    assert float(action.delta_ee.norm()) <= 0.045 + 1e-6
    assert action.delta_ee[1].item() == 0.0
    assert 0.0 <= float(action.gripper.item()) <= 1.0


def test_pick_gru_hidden_persists_and_resets_between_episodes() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=8, pick_scene=True), seed=58)
    obs = env.reset_pick_scene(object_x=-0.01, randomize_robot=False)
    graph = ManipulationGraphBuilder(task="pick", include_object=True).build(obs)
    model = PickRecurrentController()
    hidden0 = model.initial_hidden(torch.device("cpu"))
    _, hidden1 = model(graph, hidden0)
    _, hidden2 = model(graph, hidden1)
    reset_hidden = model.initial_hidden(torch.device("cpu"))

    assert hidden1.norm().item() > 0.0
    assert not torch.allclose(hidden1, hidden2)
    assert torch.allclose(reset_hidden, torch.zeros_like(reset_hidden))


def test_closed_loop_pick_evaluation_does_not_call_expert(monkeypatch) -> None:
    def forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("expert should not be called during closed-loop learned evaluation")

    monkeypatch.setattr("vla_gnn_recurrent.sim.pick_expert.run_scripted_pick_episode", forbidden)
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=4, control_substeps=5, pick_scene=True), seed=59)
    model = PickFeedForwardController(max_step=0.045)
    result = run_closed_loop_pick_episode(
        model=model,
        model_kind="graph_feedforward",
        env=env,
        graph_builder=ManipulationGraphBuilder(task="pick", include_object=True),
        device=torch.device("cpu"),
        object_x=-0.01,
    )

    assert "records" in result
    assert len(result["records"]) <= 4
