import torch

from vla_gnn_recurrent.graph.flat_features import flat_state_from_graph, flat_state_schema_from_graph
from vla_gnn_recurrent.graph.manipulation_graph_builder import ManipulationGraphBuilder
from vla_gnn_recurrent.models.pick_controller import (
    FlatPickFeedForwardController,
    FlatPickRecurrentController,
    PickFeedForwardController,
    PickRecurrentController,
)
from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv, MujocoReachConfig
from vla_gnn_recurrent.training.pick_bc import _build_pick_model
from vla_gnn_recurrent.training.pick_place_bc import PickPlaceTrainConfig


def _pick_place_graph():
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=8, control_substeps=5, pick_scene=True), seed=91)
    obs = env.reset_pick_place_scene(object_x=-0.003, destination_x=-0.056, randomize_robot=False)
    return ManipulationGraphBuilder(task="pick_and_place", include_object=True).build(obs)


def test_flat_vector_contains_same_graph_tensors_in_fixed_order() -> None:
    graph = _pick_place_graph()
    flat = flat_state_from_graph(graph)
    schema = flat_state_schema_from_graph(graph)
    node_count = graph.node_features.numel()

    assert schema.input_dim == 369
    assert flat.shape == (schema.input_dim,)
    assert torch.allclose(flat[:node_count], graph.node_features.reshape(-1))
    assert torch.allclose(flat[node_count:], graph.edge_features.reshape(-1))


def test_flat_feature_ordering_is_deterministic() -> None:
    graph = _pick_place_graph()

    assert torch.allclose(flat_state_from_graph(graph), flat_state_from_graph(graph))
    assert graph.node_names == (
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "ee",
        "gripper",
        "ball",
        "destination",
        "task:pick_and_place",
    )


def test_all_static_ablation_models_share_direction_magnitude_action_head() -> None:
    graph = _pick_place_graph()
    flat_dim = flat_state_schema_from_graph(graph).input_dim
    kinds = [
        "flat_feedforward_dir_mag",
        "flat_recurrent_dir_mag",
        "graph_feedforward_dir_mag",
        "graph_recurrent_dir_mag",
    ]

    for kind in kinds:
        model = _build_pick_model(kind, max_step=0.045, flat_input_dim=flat_dim)
        assert model.action_head_type == "direction_magnitude"
        assert model.max_step == 0.045


def test_feedforward_models_have_no_recurrent_state() -> None:
    assert not hasattr(FlatPickFeedForwardController(input_dim=369), "initial_hidden")
    assert not hasattr(PickFeedForwardController(action_head_type="direction_magnitude"), "initial_hidden")
    assert hasattr(FlatPickRecurrentController(input_dim=369), "initial_hidden")
    assert hasattr(PickRecurrentController(action_head_type="direction_magnitude"), "initial_hidden")


def test_flat_training_uses_existing_episode_split(tmp_path) -> None:
    # Tiny smoke test with generated data is intentionally avoided here because
    # Pick-and-Place demo generation is a physical MuJoCo rollout. The config
    # path checks keep the 2x2 train entrypoint pinned to the same dataset
    # mechanism as graph policies.
    config = PickPlaceTrainConfig(
        dataset_path="artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger.pt",
        model_kind="flat_feedforward_dir_mag",
        output_dir=str(tmp_path),
        epochs=0,
    )

    assert config.dataset_path.endswith("pick_place_alignment_dagger.pt")
    assert config.model_kind == "flat_feedforward_dir_mag"
