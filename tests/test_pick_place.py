import torch

from vla_gnn_recurrent.graph.manipulation_graph_builder import ManipulationGraphBuilder
from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv, MujocoReachConfig
from vla_gnn_recurrent.training.pick_place_bc import (
    PickPlaceDemoConfig,
    PickPlaceExpertConfig,
    classify_pick_place_failure,
    generate_pick_place_demonstrations,
    load_pick_policy,
    run_closed_loop_pick_place_episode,
    run_scripted_pick_place_episode,
)
from vla_gnn_recurrent.training.pick_bc import load_pick_dataset
from vla_gnn_recurrent.models.pick_controller import PickFeedForwardController


def test_pick_place_destination_observation_and_graph_edges() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=8, control_substeps=5, pick_scene=True), seed=81)
    obs = env.reset_pick_place_scene(object_x=-0.003, destination_x=-0.056, randomize_robot=False)
    graph = ManipulationGraphBuilder(task="pick_and_place", include_object=True).build(obs)
    destination_idx = graph.node_names.index("destination")
    object_idx = graph.node_names.index("ball")

    assert obs.destination is not None
    assert obs.destination.object_id == "destination"
    assert graph.node_types[destination_idx] == "destination"
    assert graph.node_types[graph.target_node_index] == "destination"
    rows = [
        idx
        for idx in range(graph.num_edges)
        if int(graph.edge_index[0, idx]) == object_idx and int(graph.edge_index[1, idx]) == destination_idx
    ]
    assert rows
    expected = graph.node_features[destination_idx, 0:3] - graph.node_features[object_idx, 0:3]
    assert torch.allclose(graph.edge_features[rows[0], 0:3], expected)


def test_pick_place_demonstrations_hide_expert_phase_and_split_by_episode(tmp_path) -> None:
    dataset_path = tmp_path / "pp.pt"
    metadata_path = tmp_path / "pp.json"
    generate_pick_place_demonstrations(
        PickPlaceDemoConfig(
            episodes=4,
            seed=82,
            output_path=str(dataset_path),
            metadata_path=str(metadata_path),
            max_steps=420,
        )
    )
    dataset = load_pick_dataset(dataset_path)
    split_sets = {name: set(indices) for name, indices in dataset["split"].items()}
    graph = dataset["episodes"][0]["graphs"][0]

    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert "phase_labels_diagnostics_only" in dataset["episodes"][0]
    assert all("APPROACH" not in name and "RELEASE" not in name for name in graph.node_names)
    assert "ee_target" not in dataset["episodes"][0]
    assert dataset["episodes"][0]["delta_xz"].shape[-1] == 2


def test_scripted_pick_place_uses_physical_stack_and_releases() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=420, control_substeps=40, pick_scene=True), seed=83)
    result = run_scripted_pick_place_episode(
        env=env,
        expert_config=PickPlaceExpertConfig(max_steps=420),
        object_x=-0.003,
        destination_x=-0.056,
        graph_builder=ManipulationGraphBuilder(task="pick_and_place", include_object=True),
    )

    assert not env.config.kinematic_joint_control
    assert result["grasp_success"]
    assert result["lift_success"]
    assert result["release_success"]
    assert result["placement_success"]
    assert result["ik_failures"] == 0


def test_pick_place_placement_failure_classification_is_deterministic() -> None:
    reason = classify_pick_place_failure(
        approach_success=True,
        grasp_success=True,
        lift_success=True,
        transport_success=True,
        release_success=True,
        records=[{"placement_error": 0.2, "gripper": 0.0}],
        ik_failures=0,
    )

    assert reason == "placement miss"


def test_closed_loop_pick_place_evaluation_does_not_call_expert(monkeypatch) -> None:
    def forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("expert should not be called during learned pick-place evaluation")

    monkeypatch.setattr("vla_gnn_recurrent.training.pick_place_bc.run_scripted_pick_place_episode", forbidden)
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=4, control_substeps=5, pick_scene=True), seed=84)
    model = PickFeedForwardController(max_step=0.045, action_head_type="direction_magnitude")
    result = run_closed_loop_pick_place_episode(
        model=model,
        model_kind="graph_feedforward_dir_mag",
        env=env,
        graph_builder=ManipulationGraphBuilder(task="pick_and_place", include_object=True),
        device=torch.device("cpu"),
        object_x=-0.003,
        destination_x=-0.056,
    )

    assert "records" in result
    assert len(result["records"]) <= 4
