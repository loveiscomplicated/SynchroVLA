import numpy as np
import pytest
import torch

from vla_gnn_recurrent.training.generalized_geometric_graph import (
    GeneralizedBatch,
    GeneralizedGeometryConfig,
    GeometricGNN,
    LocalResampledConcat,
    LocalSetAttention,
    OrientationSanityConfig,
    _sample_shape,
    _sample_orientation_episode,
    build_observation_from_state,
    curve_world,
    expert_action_from_state,
    interaction_frame,
    sample_observation_parameters,
)

from vla_gnn_recurrent.training.representation_feasibility import (
    KeypointConcatMLP,
    KeypointDemoConfig,
    KeypointMPNN,
    KeypointTrainConfig,
    PartNodeDemoConfig,
    PartNodeMPNN,
    RepresentationDemoConfig,
    TinyRepresentationMPNN,
    _keypoint_model_dims,
    _write_json,
    frozen_visual_feature,
    generate_keypoints_local,
    generate_part_node_demonstrations,
    generate_representation_demonstrations,
    graph_batch_from_feature_tensors,
    keypoint_local_tangent_yaw,
    keypoint_observation_from_feature_tensors,
    sample_keypoint_episode_specs,
    load_representation_dataset,
    part_graph_batch_from_feature_tensors,
    paired_contingency_counts,
    paired_episode_metrics,
    sample_part_episode_specs,
    summarize_paired_episode_results,
    render_handle_crop,
    wrap_angle,
)


def test_frozen_visual_feature_distinguishes_handle_side() -> None:
    left = frozen_visual_feature(
        render_handle_crop(
            handle_side=-1,
            handle_offset=0.07,
            object_half_extent=(0.06, 0.034),
            object_yaw=0.0,
            camera_yaw=0.0,
            crop_size=32,
        )
    )
    right = frozen_visual_feature(
        render_handle_crop(
            handle_side=1,
            handle_offset=0.07,
            object_half_extent=(0.06, 0.034),
            object_yaw=0.0,
            camera_yaw=0.0,
            crop_size=32,
        )
    )

    assert left.shape == right.shape == (67,)
    assert not torch.allclose(left, right)


def test_generate_representation_demo_payload(tmp_path) -> None:
    dataset_path = tmp_path / "demo.pt"
    metadata_path = tmp_path / "metadata.json"
    result = generate_representation_demonstrations(
        RepresentationDemoConfig(
            episodes=4,
            max_steps=3,
            seed=11,
            output_path=str(dataset_path),
            metadata_path=str(metadata_path),
        )
    )
    dataset = load_representation_dataset(result["dataset_path"])
    episode = dataset["episodes"][0]

    assert metadata_path.exists()
    assert dataset["task"] == "handle_contact_representation_feasibility"
    assert set(dataset["split"]) == {"train", "val", "test"}
    assert episode["pose_features"].shape[-1] == dataset["feature_dims"]["pose"]
    assert episode["visual_features"].shape[-1] == dataset["feature_dims"]["visual"]
    assert episode["ee_geometry_features"].shape[-1] == dataset["feature_dims"]["ee_geometry"]
    assert episode["actions"].shape[-1] == 3


def test_representation_graph_batch_shapes_and_directed_edges() -> None:
    pose = torch.zeros(2, 14)
    pose[:, 0:3] = torch.tensor([[1.0, 0.0, 2.0], [0.5, 0.0, 0.6]])
    pose[:, 3:5] = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    pose[:, 5:7] = torch.tensor([[0.1, 0.04], [0.08, 0.03]])
    pose[:, 7:10] = torch.tensor([[0.2, 0.0, 0.3], [0.1, 0.0, 0.2]])
    pose[:, 10:13] = pose[:, 0:3] - pose[:, 7:10]
    pose[:, 13] = pose[:, 10:13].norm(dim=-1)
    visual = torch.ones(2, 67)
    ee_geometry = torch.ones(2, 21) * 2.0

    graph = graph_batch_from_feature_tensors(pose, visual, ee_geometry)

    assert graph.object_features.shape == (2, 74)
    assert graph.ee_features.shape == (2, 24)
    assert graph.edge_features.shape == (2, 2, 4)
    assert torch.allclose(graph.edge_features[:, 0, 0:3], pose[:, 10:13])
    assert torch.allclose(graph.edge_features[:, 1, 0:3], -pose[:, 10:13])
    assert torch.allclose(graph.edge_features[:, 0, 3], graph.edge_features[:, 1, 3])


def test_representation_graph_batch_index_select_and_mpnn_forward() -> None:
    torch.manual_seed(3)
    pose = torch.randn(5, 14)
    pose[:, 13] = pose[:, 10:13].norm(dim=-1)
    visual = torch.randn(5, 67)
    ee_geometry = torch.randn(5, 21)
    graph = graph_batch_from_feature_tensors(pose, visual, ee_geometry)
    subset = graph.index_select(torch.tensor([1, 3]))
    model = TinyRepresentationMPNN(
        object_feature_dim=subset.object_features.shape[-1],
        ee_feature_dim=subset.ee_features.shape[-1],
        hidden_dim=64,
        num_layers=2,
    )

    output = model(subset)

    assert subset.num_samples == 2
    assert output.shape == (2, 3)


def test_part_graph_variants_preserve_part_information_fairly() -> None:
    object_features = torch.zeros(2, 74)
    object_features[:, 0:3] = torch.tensor([[0.2, 0.0, 0.6], [-0.1, 0.0, 0.7]])
    ee_features = torch.zeros(2, 8)
    ee_features[:, 0:3] = torch.tensor([[0.0, 0.0, 0.5], [0.1, 0.0, 0.4]])
    part_features = torch.zeros(2, 10)
    part_features[:, 0:3] = torch.tensor([[0.23, 0.0, 0.64], [-0.16, 0.0, 0.76]])
    part_features[:, 3:6] = torch.tensor([[0.03, 0.0, 0.04], [-0.06, 0.0, 0.06]])
    part_features[:, 6:8] = torch.tensor([[0.0, 1.0], [0.1, 0.99]])
    part_features[:, 8:10] = torch.tensor([[0.02, 0.01], [0.025, 0.012]])

    object_only = part_graph_batch_from_feature_tensors(object_features, ee_features, part_features, "object_only")
    part_concat = part_graph_batch_from_feature_tensors(object_features, ee_features, part_features, "part_concat")
    part_node = part_graph_batch_from_feature_tensors(object_features, ee_features, part_features, "part_node")

    assert object_only.object_features.shape == (2, 74)
    assert object_only.part_features is None
    assert part_concat.object_features.shape == (2, 84)
    assert torch.allclose(part_concat.object_features[:, -10:], part_node.part_features)
    assert part_node.object_features.shape == (2, 74)
    assert part_node.part_features.shape == (2, 10)
    assert part_node.edge_features.shape == (2, 6, 4)


def test_part_graph_directed_edge_signs() -> None:
    object_features = torch.zeros(1, 74)
    object_features[:, 0:3] = torch.tensor([[0.2, 0.0, 0.6]])
    ee_features = torch.zeros(1, 8)
    ee_features[:, 0:3] = torch.tensor([[0.1, 0.0, 0.5]])
    part_features = torch.zeros(1, 10)
    part_features[:, 0:3] = torch.tensor([[0.25, 0.0, 0.62]])
    part_features[:, 3:6] = torch.tensor([[0.05, 0.0, 0.02]])

    graph = part_graph_batch_from_feature_tensors(object_features, ee_features, part_features, "part_node")

    assert torch.equal(graph.edge_index, torch.tensor([[1, 0, 1, 2, 0, 2], [0, 1, 2, 1, 2, 0]]))
    assert torch.allclose(graph.edge_features[:, 0, 0:3], torch.tensor([[0.1, 0.0, 0.1]]))
    assert torch.allclose(graph.edge_features[:, 1, 0:3], torch.tensor([[-0.1, -0.0, -0.1]]))
    assert torch.allclose(graph.edge_features[:, 2, 0:3], torch.tensor([[0.15, 0.0, 0.12]]))
    assert torch.allclose(graph.edge_features[:, 3, 0:3], torch.tensor([[-0.15, -0.0, -0.12]]))
    assert torch.allclose(graph.edge_features[:, 4, 0:3], part_features[:, 3:6])
    assert torch.allclose(graph.edge_features[:, 5, 0:3], -part_features[:, 3:6])


def test_part_node_mpnn_forward_for_all_variants() -> None:
    torch.manual_seed(5)
    object_features = torch.randn(4, 74)
    ee_features = torch.randn(4, 8)
    part_features = torch.randn(4, 10)
    for variant in ("object_only", "part_concat", "part_node"):
        graph = part_graph_batch_from_feature_tensors(object_features, ee_features, part_features, variant)
        model = PartNodeMPNN(
            object_feature_dim=graph.object_features.shape[-1],
            ee_feature_dim=graph.ee_features.shape[-1],
            part_feature_dim=0 if graph.part_features is None else graph.part_features.shape[-1],
            hidden_dim=32,
            num_layers=2,
        )

        output = model(graph)

        assert output.shape == (4, 3)


def test_generate_part_node_demo_payload_includes_ood(tmp_path) -> None:
    dataset_path = tmp_path / "part_demo.pt"
    metadata_path = tmp_path / "part_metadata.json"
    result = generate_part_node_demonstrations(
        PartNodeDemoConfig(
            episodes=5,
            ood_episodes=3,
            max_steps=3,
            seed=17,
            output_path=str(dataset_path),
            metadata_path=str(metadata_path),
        )
    )
    dataset = load_representation_dataset(result["dataset_path"])
    episode = dataset["episodes"][0]
    ood_episode = dataset["ood_episodes"][0]

    assert metadata_path.exists()
    assert dataset["task"] == "oracle_part_node_feasibility"
    assert set(dataset["split"]) == {"train", "val", "test"}
    assert len(dataset["ood_episodes"]) == 3
    assert episode["object_features"].shape[-1] == dataset["feature_dims"]["object"]
    assert episode["ee_features"].shape[-1] == dataset["feature_dims"]["ee"]
    assert episode["part_features"].shape[-1] == dataset["feature_dims"]["part"]
    assert episode["actions"].shape[-1] == 3
    assert abs(float(ood_episode["part_local"][0])) > abs(float(episode["part_local"][0]))


def test_part_episode_specs_are_reproducible_and_shared() -> None:
    config = PartNodeDemoConfig(episodes=2, ood_episodes=1, max_steps=3)
    specs_a = sample_part_episode_specs(config, episodes=4, seed=123, layout="ood")
    specs_b = sample_part_episode_specs(config, episodes=4, seed=123, layout="ood")

    assert [spec.arm_qpos for spec in specs_a] == [spec.arm_qpos for spec in specs_b]
    assert [spec.task_state.part_local.tolist() for spec in specs_a] == [
        spec.task_state.part_local.tolist() for spec in specs_b
    ]
    assert specs_a[0].task_state.layout == "ood"


def test_paired_contingency_and_metric_deltas() -> None:
    episodes = [
        _fake_paired_episode(True, True, 0.01, 0.02, 0.4, 0.5, 4, 5),
        _fake_paired_episode(True, False, 0.02, 0.30, 0.5, 0.9, 3, 8),
        _fake_paired_episode(False, True, 0.25, 0.03, 0.8, 0.4, 8, 4),
        _fake_paired_episode(False, False, 0.20, 0.18, 0.7, 0.6, 8, 8),
    ]

    counts = paired_contingency_counts(episodes)
    summary = summarize_paired_episode_results(episodes, bootstrap_samples=64, seed=9)

    assert counts == {
        "both_success": 1,
        "concat_only_success": 1,
        "part_node_only_success": 1,
        "both_fail": 1,
    }
    assert abs(episodes[1]["paired"]["delta_final_distance"] - 0.28) < 1e-8
    assert summary["num_episodes"] == 4
    assert summary["contingency"]["concat_only_success"] == 1
    assert abs(summary["final_distance"]["paired_delta_part_node_minus_concat"]["median"] + 0.005) < 1e-8


def test_paired_artifact_serialization(tmp_path) -> None:
    payload = {"episodes": [_fake_paired_episode(True, False, 0.01, 0.35, 0.2, 0.7, 2, 8)]}
    output_path = tmp_path / "episode_results.json"

    _write_json(output_path, payload)

    assert output_path.exists()
    assert '"episodes"' in output_path.read_text(encoding="utf-8")


def test_keypoint_graph_arc_position_and_permutation_preserve_identity() -> None:
    object_features = torch.zeros(1, 7)
    ee_features = torch.zeros(1, 5)
    part_features = torch.zeros(1, 10)
    keypoints = torch.zeros(1, 5, 6)
    keypoints[0, :, 0] = torch.arange(5, dtype=torch.float32)
    keypoints[0, :, 3] = torch.arange(5, dtype=torch.float32) * 0.1
    permutation = [3, 0, 4, 1, 2]

    graph = keypoint_observation_from_feature_tensors(
        object_features, ee_features, part_features, keypoints, "keypoint_graph_arc", permutation
    )

    assert graph.keypoint_features.shape == (1, 5, 7)
    assert torch.allclose(graph.keypoint_features[0, :, 6], torch.tensor([0.5, -1.0, 1.0, -0.5, 0.0]))
    assert torch.allclose(graph.keypoint_features[0, :, :6], keypoints[:, permutation, :][0])

    physical_to_storage = {physical: storage for storage, physical in enumerate(permutation)}
    adjacent_pairs = {(2 + physical_to_storage[i], 2 + physical_to_storage[i + 1]) for i in range(4)}
    observed_pairs = {(int(src), int(dst)) for src, dst in graph.edge_index.t().tolist()}
    assert all((src, dst) in observed_pairs and (dst, src) in observed_pairs for src, dst in adjacent_pairs)


def test_keypoint_graph_arc_readouts_share_input_and_emit_actions() -> None:
    torch.manual_seed(13)
    object_features = torch.randn(3, 7)
    ee_features = torch.randn(3, 5)
    part_features = torch.randn(3, 10)
    keypoints = torch.randn(3, 5, 6)
    graph = keypoint_observation_from_feature_tensors(
        object_features, ee_features, part_features, keypoints, "keypoint_graph_arc"
    )
    global_model = KeypointMPNN(
        object_feature_dim=7, ee_feature_dim=5, keypoint_feature_dim=7, hidden_dim=32, num_layers=2, readout="global"
    )
    ee_model = KeypointMPNN(
        object_feature_dim=7, ee_feature_dim=5, keypoint_feature_dim=7, hidden_dim=32, num_layers=2, readout="ee"
    )

    assert torch.equal(graph.keypoint_features, graph.keypoint_features.clone())
    assert global_model(graph).shape == (3, 4)
    assert ee_model(graph).shape == (3, 4)


def test_generalized_physical_target_is_independent_of_observed_sampling() -> None:
    config = GeneralizedGeometryConfig()
    rng = np.random.default_rng(41)
    shape = _sample_shape(config, rng, "shape")
    target_a, yaw_a = interaction_frame(shape, config.target_s)
    target_b, yaw_b = interaction_frame(shape, config.target_s)
    s_a = sample_observation_parameters(rng, k_values=(5,))
    s_b = sample_observation_parameters(rng, k_values=(10,))
    assert s_a.shape[0] != s_b.shape[0]
    assert torch.allclose(target_a, target_b)
    assert abs(yaw_a - yaw_b) < 1e-8
    _, _, points_a = build_observation_from_state(shape, torch.tensor([0.0, 0.0, 0.52]), 0.2, s_a)
    _, _, points_b = build_observation_from_state(shape, torch.tensor([0.0, 0.0, 0.52]), 0.2, s_b)
    assert points_a.shape[-1] == points_b.shape[-1] == 4
    assert not torch.allclose(points_a[:, 3].mean(), torch.tensor(config.target_s)) or not torch.allclose(
        points_b[:, 3].mean(), torch.tensor(config.target_s)
    )


def test_variable_sampling_models_are_storage_permutation_invariant() -> None:
    torch.manual_seed(19)
    batch = GeneralizedBatch(
        object_features=torch.randn(1, 7),
        ee_features=torch.zeros(1, 1),
        point_features=torch.tensor([[[0.1, 0.0, 0.01, 0.0], [0.2, 0.0, -0.01, 0.5], [0.3, 0.0, 0.02, 1.0]]]),
        point_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    permutation = torch.tensor([2, 0, 1])
    permuted = GeneralizedBatch(
        batch.object_features,
        batch.ee_features,
        batch.point_features[:, permutation],
        batch.point_mask[:, permutation],
    )
    concat = LocalResampledConcat(hidden_dim=16, resample_points=8)
    set_model = LocalSetAttention(hidden_dim=16, heads=4)
    graph = GeometricGNN(hidden_dim=16, layers=2, use_local_edges=True)
    assert torch.allclose(concat(batch), concat(permuted), atol=1e-6)
    assert torch.allclose(set_model(batch), set_model(permuted), atol=1e-6)
    assert torch.allclose(graph(batch), graph(permuted), atol=1e-6)


def test_generalized_observation_uses_ee_local_coordinates() -> None:
    config = GeneralizedGeometryConfig()
    rng = np.random.default_rng(7)
    shape = _sample_shape(config, rng, "shape")
    s_values = sample_observation_parameters(rng, k_values=(6,))
    object_features, ee_features, point_features = build_observation_from_state(
        shape, torch.tensor([0.0, 0.0, 0.52]), 0.0, s_values
    )
    assert object_features.shape == (7,)
    assert ee_features.shape == (1,)
    assert point_features.shape == (6, 4)
    assert torch.allclose(point_features[:, 3], s_values)
    assert torch.all(point_features[:, 3] >= 0.0)
    assert torch.all(point_features[:, 3] <= 1.0)


def test_relative_orientation_sanity_controls_bin_and_yaw_clipping() -> None:
    config = GeneralizedGeometryConfig(max_delta_yaw=0.35)
    shape = _sample_shape(config, np.random.default_rng(101), "orientation_shape")
    episode, relative_yaw = _sample_orientation_episode(
        shape, 0, np.random.default_rng(102), config, (1.80, 2.40), "orientation_ood"
    )
    _, target_yaw = interaction_frame(shape, config.target_s)
    assert 1.80 <= abs(relative_yaw) <= 2.40
    assert abs(float(wrap_angle(target_yaw - episode.ee_yaw))) == pytest.approx(abs(relative_yaw), abs=1e-6)
    action = expert_action_from_state(torch.tensor([0.0, 0.0, 0.52]), episode, config)
    assert action.shape == (4,)
    assert abs(float(action[3])) == pytest.approx(config.max_delta_yaw)
    assert int(np.ceil(abs(relative_yaw) / config.max_delta_yaw)) <= config.max_steps


def test_orientation_sanity_bins_are_serializable_and_reproducible() -> None:
    sanity = OrientationSanityConfig()
    config = GeneralizedGeometryConfig()
    shape = _sample_shape(config, np.random.default_rng(7), "shape")
    episode_a, rel_a = _sample_orientation_episode(shape, 0, np.random.default_rng(9), config, (0.45, 0.90), "bin")
    episode_b, rel_b = _sample_orientation_episode(shape, 0, np.random.default_rng(9), config, (0.45, 0.90), "bin")
    assert sanity.orientation_bins[1] == (0.45, 0.90)
    assert episode_a == episode_b
    assert rel_a == rel_b


def _fake_paired_episode(
    concat_success: bool,
    node_success: bool,
    concat_final: float,
    node_final: float,
    concat_traj_len: float,
    node_traj_len: float,
    concat_steps: int,
    node_steps: int,
) -> dict[str, object]:
    concat = {
        "success": concat_success,
        "final_distance": concat_final,
        "trajectory_length": concat_traj_len,
        "max_distance_from_target": max(concat_final, 0.1),
        "steps": concat_steps,
        "trajectory": [[0.0, 0.0, 0.0], [concat_traj_len, 0.0, 0.0]],
    }
    node = {
        "success": node_success,
        "final_distance": node_final,
        "trajectory_length": node_traj_len,
        "max_distance_from_target": max(node_final, 0.1),
        "steps": node_steps,
        "trajectory": [[0.0, 0.0, 0.0], [node_traj_len, 0.0, 0.0]],
    }
    return {
        "episode_id": 0,
        "spec": {
            "part_local": [0.1, 0.0, 0.02],
            "object_center": [0.0, 0.0, 0.6],
            "object_yaw": 0.0,
            "part_world_position": [0.1, 0.0, 0.62],
            "part_extent": [0.02, 0.01],
        },
        "part_concat": concat,
        "part_node": node,
        "paired": paired_episode_metrics(concat, node),
    }


def test_keypoint_geometry_generation_is_deterministic() -> None:
    config = KeypointDemoConfig(keypoints=5, episodes=2, ood_episodes=1)
    specs_a = sample_keypoint_episode_specs(config, episodes=3, seed=55, split="shape_ood")
    specs_b = sample_keypoint_episode_specs(config, episodes=3, seed=55, split="shape_ood")

    assert specs_a[0].task_state.keypoints_world.tolist() == specs_b[0].task_state.keypoints_world.tolist()
    assert specs_a[0].task_state.length == specs_b[0].task_state.length
    assert specs_a[0].arm_qpos == specs_b[0].arm_qpos


def test_keypoint_physical_order_and_local_tangent() -> None:
    keypoints = generate_keypoints_local(
        keypoints=5,
        length=0.10,
        curvature=0.02,
        spacing_jitter=0.0,
        rng=np.random.default_rng(1),
    )
    x_values = keypoints[:, 0]
    yaw = keypoint_local_tangent_yaw(keypoints)

    assert torch.all(x_values[1:] > x_values[:-1])
    assert abs(yaw) > 0.1


def test_keypoint_yaw_wrapping() -> None:
    assert abs(wrap_angle(3.5) + 2.7831853071795862) < 1e-6
    assert abs(wrap_angle(-3.5) - 2.7831853071795862) < 1e-6


def test_keypoint_concat_and_graph_use_same_raw_coordinates() -> None:
    object_features = torch.randn(2, 7)
    ee_features = torch.randn(2, 5)
    part_features = torch.randn(2, 11)
    keypoint_features = torch.randn(2, 5, 6)
    concat = keypoint_observation_from_feature_tensors(
        object_features, ee_features, part_features, keypoint_features, "keypoint_concat"
    )
    graph = keypoint_observation_from_feature_tensors(
        object_features, ee_features, part_features, keypoint_features, "keypoint_graph"
    )

    assert torch.allclose(concat[:, -30:].reshape(2, 5, 6), graph.keypoint_features)


def test_keypoint_graph_permutation_preserves_physical_adjacency() -> None:
    object_features = torch.zeros(1, 7)
    ee_features = torch.zeros(1, 5)
    part_features = torch.zeros(1, 11)
    keypoint_features = torch.zeros(1, 5, 6)
    keypoint_features[0, :, 0] = torch.arange(5, dtype=torch.float32)
    keypoint_features[0, :, 3] = torch.arange(5, dtype=torch.float32)
    permutation = [3, 0, 4, 1, 2]
    graph = keypoint_observation_from_feature_tensors(
        object_features,
        ee_features,
        part_features,
        keypoint_features,
        "keypoint_graph",
        permutation=permutation,
    )
    storage_for_physical_0 = permutation.index(0)
    storage_for_physical_1 = permutation.index(1)
    src = 2 + storage_for_physical_0
    dst = 2 + storage_for_physical_1
    edges = graph.edge_index.t().tolist()

    assert [src, dst] in edges
    assert [dst, src] in edges


def test_single_part_node_does_not_include_local_tangent_or_anchor() -> None:
    config = KeypointDemoConfig(keypoints=5)
    spec = sample_keypoint_episode_specs(config, episodes=1, seed=77, split="shape_ood")[0]
    object_features = torch.zeros(1, 7)
    ee_features = torch.zeros(1, 5)
    keypoint_features = torch.cat(
        [spec.task_state.keypoints_world, spec.task_state.keypoints_local],
        dim=-1,
    ).unsqueeze(0)
    # part feature format: centroid world/local + principal yaw sin/cos + length/extents.
    centroid_world = spec.task_state.keypoints_world.mean(dim=0)
    centroid_local = spec.task_state.keypoints_local.mean(dim=0)
    part_features = torch.cat(
        [
            centroid_world,
            centroid_local,
            torch.tensor([0.0, 1.0]),
            torch.tensor([spec.task_state.length, 0.1, 0.02]),
        ],
        dim=0,
    ).reshape(1, -1)
    graph = keypoint_observation_from_feature_tensors(
        object_features, ee_features, part_features, keypoint_features, "single_part_node"
    )

    assert graph.part_features.shape[-1] == 11
    assert not torch.allclose(graph.part_features[0, 0:3], spec.task_state.target_position)


def test_keypoint_models_output_dx_dy_dz_dyaw() -> None:
    object_features = torch.randn(3, 7)
    ee_features = torch.randn(3, 5)
    part_features = torch.randn(3, 11)
    keypoint_features = torch.randn(3, 5, 6)
    concat_obs = keypoint_observation_from_feature_tensors(
        object_features, ee_features, part_features, keypoint_features, "keypoint_concat"
    )
    graph_obs = keypoint_observation_from_feature_tensors(
        object_features, ee_features, part_features, keypoint_features, "keypoint_graph"
    )
    concat_model = KeypointConcatMLP(input_dim=concat_obs.shape[-1], hidden_dim=32)
    graph_model = KeypointMPNN(
        object_feature_dim=graph_obs.object_features.shape[-1],
        ee_feature_dim=graph_obs.ee_features.shape[-1],
        keypoint_feature_dim=graph_obs.keypoint_features.shape[-1],
        hidden_dim=32,
        num_layers=1,
    )

    assert concat_model(concat_obs).shape == (3, 4)
    assert graph_model(graph_obs).shape == (3, 4)
