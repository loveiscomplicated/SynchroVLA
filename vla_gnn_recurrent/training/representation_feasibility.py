from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv, MujocoReachConfig
from vla_gnn_recurrent.utils import DevicePreference, clamp_delta, ensure_dir, select_device, set_seed

RepresentationVariant = Literal["pose", "pose_visual", "pose_visual_ee"]
RepresentationModelKind = Literal["mlp", "mpnn"]
PartNodeVariant = Literal["object_only", "part_concat", "part_node"]
PartLayout = Literal["iid", "ood"]
KeypointVariant = Literal["single_part_node", "keypoint_concat", "keypoint_graph", "keypoint_graph_arc"]
KeypointReadout = Literal["global", "ee"]
KeypointEvalSplit = Literal["iid", "length_ood", "orientation_ood", "shape_ood"]

VARIANT_FLAGS: dict[RepresentationVariant, dict[str, bool]] = {
    "pose": {"pose": True, "visual": False, "ee_geometry": False},
    "pose_visual": {"pose": True, "visual": True, "ee_geometry": False},
    "pose_visual_ee": {"pose": True, "visual": True, "ee_geometry": True},
}


@dataclass
class RepresentationDemoConfig:
    episodes: int = 120
    max_steps: int = 24
    seed: int = 2401
    output_path: str = "artifacts/representation_feasibility/demos/handle_contact.pt"
    metadata_path: str = "artifacts/representation_feasibility/demos/handle_contact_metadata.json"
    object_x_range: tuple[float, float] = (-0.14, 0.14)
    object_z_range: tuple[float, float] = (0.50, 0.74)
    object_yaw_range: tuple[float, float] = (-0.9, 0.9)
    handle_offset_range: tuple[float, float] = (0.075, 0.115)
    object_half_extent: tuple[float, float] = (0.095, 0.040)
    target_radius: float = 0.022
    max_delta_ee: float = 0.035
    gripper_command_range: tuple[float, float] = (0.0, 1.0)
    camera_yaw: float = 0.0
    crop_size: int = 32
    visual_grid: int = 8
    render_first_episode: bool = False


@dataclass
class RepresentationTrainConfig:
    dataset_path: str = "artifacts/representation_feasibility/demos/handle_contact.pt"
    variant: RepresentationVariant = "pose"
    model_kind: RepresentationModelKind = "mlp"
    output_dir: str = "artifacts/representation_feasibility/checkpoints"
    epochs: int = 24
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    mpnn_hidden_dim: int = 64
    mpnn_layers: int = 2
    seed: int = 2401
    device: DevicePreference = "auto"
    log_every: int = 10


@dataclass
class RepresentationEvalConfig:
    checkpoint_path: str = "artifacts/representation_feasibility/checkpoints/pose.pt"
    episodes: int = 48
    max_steps: int = 24
    seed: int = 3401
    output_dir: str = "artifacts/representation_feasibility/eval"
    device: DevicePreference = "auto"
    camera_yaw: float | None = None
    render_first_episode: bool = False


@dataclass
class PartNodeDemoConfig:
    episodes: int = 180
    ood_episodes: int = 72
    max_steps: int = 24
    seed: int = 2601
    output_path: str = "artifacts/part_node_feasibility/demos/oracle_part_reach.pt"
    metadata_path: str = "artifacts/part_node_feasibility/demos/oracle_part_reach_metadata.json"
    object_x_range: tuple[float, float] = (-0.14, 0.14)
    object_z_range: tuple[float, float] = (0.50, 0.74)
    object_yaw_range: tuple[float, float] = (-0.9, 0.9)
    object_half_extent: tuple[float, float] = (0.095, 0.040)
    train_part_local_x_range: tuple[float, float] = (-0.035, 0.035)
    train_part_local_z_range: tuple[float, float] = (-0.014, 0.026)
    ood_part_local_abs_x_range: tuple[float, float] = (0.075, 0.120)
    ood_part_local_z_range: tuple[float, float] = (-0.052, 0.052)
    part_extent_range: tuple[float, float] = (0.014, 0.028)
    part_local_yaw_range: tuple[float, float] = (-0.7, 0.7)
    target_radius: float = 0.022
    max_delta_ee: float = 0.035
    gripper_command_range: tuple[float, float] = (0.0, 1.0)
    camera_yaw: float = 0.0
    crop_size: int = 32
    visual_grid: int = 8
    render_first_episode: bool = False


@dataclass
class PartNodeTrainConfig:
    dataset_path: str = "artifacts/part_node_feasibility/demos/oracle_part_reach.pt"
    variant: PartNodeVariant = "object_only"
    output_dir: str = "artifacts/part_node_feasibility/checkpoints"
    epochs: int = 36
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    message_passing_layers: int = 2
    seed: int = 2601
    device: DevicePreference = "auto"
    log_every: int = 12


@dataclass
class PartNodeEvalConfig:
    checkpoint_path: str = "artifacts/part_node_feasibility/checkpoints/object_only.pt"
    episodes: int = 64
    max_steps: int = 24
    seed: int = 3601
    output_dir: str = "artifacts/part_node_feasibility/eval"
    layout: PartLayout = "iid"
    eval_device: DevicePreference = "auto"
    eval_workers: int = 1
    render_first_episode: bool = False


@dataclass
class PartNodePairedEvalConfig:
    artifact_dir: str = "artifacts/part_node_feasibility"
    output_dir: str = "artifacts/part_node_feasibility/paired_ood_validation"
    seeds: tuple[int, ...] = (2811, 2812, 2813)
    episodes: int = 500
    max_steps: int = 24
    layout: PartLayout = "ood"
    eval_seed_base: int = 3811
    eval_device: DevicePreference = "auto"
    eval_workers: int = 1
    bootstrap_samples: int = 2000
    plot_worst: int = 5
    catastrophic_distance: float = 0.3


@dataclass
class KeypointDemoConfig:
    episodes: int = 160
    ood_episodes: int = 80
    max_steps: int = 24
    seed: int = 2811
    output_path: str = "artifacts/keypoint_graph_feasibility/seed2811/dataset/keypoint_alignment.pt"
    metadata_path: str = "artifacts/keypoint_graph_feasibility/seed2811/dataset/geometry_statistics.json"
    keypoints: int = 5
    object_x_range: tuple[float, float] = (-0.14, 0.14)
    object_z_range: tuple[float, float] = (0.50, 0.74)
    object_yaw_range: tuple[float, float] = (-0.45, 0.45)
    object_half_extent: tuple[float, float] = (0.095, 0.040)
    train_length_range: tuple[float, float] = (0.08, 0.12)
    length_ood_range: tuple[float, float] = (0.13, 0.17)
    train_part_yaw_range: tuple[float, float] = (-0.55, 0.55)
    orientation_ood_yaw_range: tuple[float, float] = (1.05, 2.10)
    train_offset_x_range: tuple[float, float] = (-0.035, 0.035)
    train_offset_z_range: tuple[float, float] = (-0.020, 0.030)
    train_curvature_range: tuple[float, float] = (-0.004, 0.004)
    shape_ood_curvature_abs_range: tuple[float, float] = (0.018, 0.032)
    train_spacing_jitter: float = 0.04
    shape_ood_spacing_jitter: float = 0.24
    target_radius: float = 0.030
    yaw_success_threshold: float = 0.35
    max_delta_ee: float = 0.035
    max_delta_yaw: float = 0.35
    yaw_loss_weight: float = 0.25
    gripper_command_range: tuple[float, float] = (0.0, 1.0)


@dataclass
class KeypointTrainConfig:
    dataset_path: str = "artifacts/keypoint_graph_feasibility/seed2811/dataset/keypoint_alignment.pt"
    variant: KeypointVariant = "single_part_node"
    output_dir: str = "artifacts/keypoint_graph_feasibility/seed2811/checkpoints"
    epochs: int = 36
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    mlp_hidden_dim: int = 256
    message_passing_layers: int = 2
    readout: KeypointReadout = "global"
    yaw_loss_weight: float = 0.25
    seed: int = 2811
    device: DevicePreference = "auto"
    log_every: int = 12


@dataclass
class KeypointExperimentConfig:
    artifact_dir: str = "artifacts/keypoint_graph_feasibility"
    seeds: tuple[int, ...] = (2811, 2812, 2813)
    episodes: int = 160
    ood_episodes: int = 80
    eval_episodes: int = 64
    max_steps: int = 24
    epochs: int = 36
    batch_size: int = 256
    device: DevicePreference = "auto"
    eval_device: DevicePreference = "auto"
    eval_workers: int = 1
    bootstrap_samples: int = 1000
    plot_worst: int = 3


@dataclass
class KeypointDiagnosticConfig:
    dataset_path: str
    checkpoint_dir: str
    output_dir: str = "artifacts/keypoint_graph_feasibility/diagnostics/seed2811"
    seed: int = 2811
    target_epochs: int = 300
    overfit_epochs: int = 500
    overfit_episodes: int = 100
    batch_size: int = 128
    device: DevicePreference = "auto"
    eval_device: DevicePreference = "auto"


@dataclass
class KeypointStructuralDiagnosticConfig:
    artifact_dir: str = "artifacts/keypoint_graph_feasibility/parallel_mps"
    output_dir: str = "artifacts/keypoint_graph_feasibility/structural_diagnostics"
    seeds: tuple[int, ...] = (2811, 2812, 2813)
    epochs: int = 36
    target_decode_epochs: int = 300
    batch_size: int = 256
    eval_episodes: int = 64
    max_steps: int = 24
    device: DevicePreference = "mps"
    eval_device: DevicePreference = "mps"
    bootstrap_samples: int = 1000
    plot_worst: int = 3


class RepresentationMLP(nn.Module):
    """Small feed-forward Delta EE policy for representation ablations."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return self.net(features)


@dataclass(frozen=True)
class RepresentationGraphBatch:
    """Two-node C-representation graph batch with node order [object, EE]."""

    object_features: torch.Tensor
    ee_features: torch.Tensor
    edge_features: torch.Tensor

    def to(self, device: torch.device | str) -> "RepresentationGraphBatch":
        return RepresentationGraphBatch(
            object_features=self.object_features.to(device),
            ee_features=self.ee_features.to(device),
            edge_features=self.edge_features.to(device),
        )

    def index_select(self, indices: torch.Tensor) -> "RepresentationGraphBatch":
        return RepresentationGraphBatch(
            object_features=self.object_features.index_select(0, indices),
            ee_features=self.ee_features.index_select(0, indices),
            edge_features=self.edge_features.index_select(0, indices),
        )

    @property
    def num_samples(self) -> int:
        return int(self.object_features.shape[0])


class EdgeMessagePassingLayer(nn.Module):
    """One tiny directed edge-aware message passing block for two-node graphs."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_state: torch.Tensor, edge_state: torch.Tensor) -> torch.Tensor:
        # node_state: [B, 2, H] with [object, EE]
        # edge_state: [B, 2, H] with [EE->object, object->EE]
        object_state = node_state[:, 0]
        ee_state = node_state[:, 1]
        src_state = torch.stack([ee_state, object_state], dim=1)
        dst_state = torch.stack([object_state, ee_state], dim=1)
        messages = self.message_mlp(torch.cat([src_state, dst_state, edge_state], dim=-1))
        aggregated = torch.stack([messages[:, 0], messages[:, 1]], dim=1)
        update = self.update_mlp(torch.cat([node_state, aggregated], dim=-1))
        return self.norm(node_state + update)


class TinyRepresentationMPNN(nn.Module):
    """Two-node edge-aware MPNN for fixed representation C."""

    def __init__(
        self,
        object_feature_dim: int,
        ee_feature_dim: int,
        edge_feature_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.object_feature_dim = int(object_feature_dim)
        self.ee_feature_dim = int(ee_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.object_projector = nn.Sequential(
            nn.Linear(object_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ee_projector = nn.Sequential(
            nn.Linear(ee_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_projector = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([EdgeMessagePassingLayer(hidden_dim) for _ in range(num_layers)])
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, graph: RepresentationGraphBatch) -> torch.Tensor:
        object_state = self.object_projector(graph.object_features)
        ee_state = self.ee_projector(graph.ee_features)
        node_state = torch.stack([object_state, ee_state], dim=1)
        edge_state = self.edge_projector(graph.edge_features)
        for layer in self.layers:
            node_state = layer(node_state, edge_state)
        object_state = node_state[:, 0]
        ee_state = node_state[:, 1]
        global_state = node_state.mean(dim=1)
        return self.action_head(torch.cat([object_state, ee_state, global_state], dim=-1))


@dataclass(frozen=True)
class PartGraphBatch:
    """Tiny oracle-part graph batch.

    Node order is [object, EE] for object-only / part-concat and
    [object, EE, part] for part-node.
    """

    object_features: torch.Tensor
    ee_features: torch.Tensor
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    part_features: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "PartGraphBatch":
        return PartGraphBatch(
            object_features=self.object_features.to(device),
            ee_features=self.ee_features.to(device),
            edge_features=self.edge_features.to(device),
            edge_index=self.edge_index.to(device),
            part_features=None if self.part_features is None else self.part_features.to(device),
        )

    def index_select(self, indices: torch.Tensor) -> "PartGraphBatch":
        return PartGraphBatch(
            object_features=self.object_features.index_select(0, indices),
            ee_features=self.ee_features.index_select(0, indices),
            edge_features=self.edge_features.index_select(0, indices),
            edge_index=self.edge_index,
            part_features=None if self.part_features is None else self.part_features.index_select(0, indices),
        )

    @property
    def num_samples(self) -> int:
        return int(self.object_features.shape[0])

    @property
    def has_part_node(self) -> bool:
        return self.part_features is not None


class PartEdgeMessagePassingLayer(nn.Module):
    """Directed edge-aware message passing over two or three typed nodes."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_state: torch.Tensor, edge_state: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        src_state = node_state.index_select(1, src)
        dst_state = node_state.index_select(1, dst)
        messages = self.message_mlp(torch.cat([src_state, dst_state, edge_state], dim=-1))
        aggregated = torch.zeros_like(node_state)
        counts = torch.zeros(node_state.shape[1], dtype=node_state.dtype, device=node_state.device)
        for edge_idx in range(int(edge_index.shape[1])):
            dst_idx = int(dst[edge_idx].item())
            aggregated[:, dst_idx] = aggregated[:, dst_idx] + messages[:, edge_idx]
            counts[dst_idx] = counts[dst_idx] + 1.0
        aggregated = aggregated / counts.clamp_min(1.0).reshape(1, -1, 1)
        update = self.update_mlp(torch.cat([node_state, aggregated], dim=-1))
        return self.norm(node_state + update)


class PartNodeMPNN(nn.Module):
    """Small shared MPNN for object-only, part-concat, and part-node variants."""

    def __init__(
        self,
        object_feature_dim: int,
        ee_feature_dim: int,
        edge_feature_dim: int = 4,
        part_feature_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.object_feature_dim = int(object_feature_dim)
        self.ee_feature_dim = int(ee_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.part_feature_dim = int(part_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.object_projector = nn.Sequential(
            nn.Linear(object_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ee_projector = nn.Sequential(
            nn.Linear(ee_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.part_projector = (
            nn.Sequential(
                nn.Linear(part_feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if part_feature_dim > 0
            else None
        )
        self.edge_projector = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([PartEdgeMessagePassingLayer(hidden_dim) for _ in range(num_layers)])
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, graph: PartGraphBatch) -> torch.Tensor:
        object_state = self.object_projector(graph.object_features)
        ee_state = self.ee_projector(graph.ee_features)
        node_states = [object_state, ee_state]
        if graph.part_features is not None:
            if self.part_projector is None:
                raise ValueError("Received part node features, but this model has no part projector.")
            part_state = self.part_projector(graph.part_features)
            node_states.append(part_state)
        else:
            part_state = torch.zeros_like(object_state)
        node_state = torch.stack(node_states, dim=1)
        edge_state = self.edge_projector(graph.edge_features)
        for layer in self.layers:
            node_state = layer(node_state, edge_state, graph.edge_index)
        object_state = node_state[:, 0]
        ee_state = node_state[:, 1]
        if graph.part_features is not None:
            part_state = node_state[:, 2]
        global_state = node_state.mean(dim=1)
        return self.action_head(torch.cat([object_state, ee_state, part_state, global_state], dim=-1))


@dataclass(frozen=True)
class KeypointGraphBatch:
    object_features: torch.Tensor
    ee_features: torch.Tensor
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    part_features: torch.Tensor | None = None
    keypoint_features: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "KeypointGraphBatch":
        return KeypointGraphBatch(
            object_features=self.object_features.to(device),
            ee_features=self.ee_features.to(device),
            part_features=None if self.part_features is None else self.part_features.to(device),
            keypoint_features=None if self.keypoint_features is None else self.keypoint_features.to(device),
            edge_features=self.edge_features.to(device),
            edge_index=self.edge_index.to(device),
        )

    def index_select(self, indices: torch.Tensor) -> "KeypointGraphBatch":
        return KeypointGraphBatch(
            object_features=self.object_features.index_select(0, indices),
            ee_features=self.ee_features.index_select(0, indices),
            part_features=None if self.part_features is None else self.part_features.index_select(0, indices),
            keypoint_features=None if self.keypoint_features is None else self.keypoint_features.index_select(0, indices),
            edge_features=self.edge_features.index_select(0, indices),
            edge_index=self.edge_index,
        )

    @property
    def num_samples(self) -> int:
        return int(self.object_features.shape[0])


class KeypointConcatMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 4) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return self.net(features)


class KeypointMPNN(nn.Module):
    def __init__(
        self,
        object_feature_dim: int,
        ee_feature_dim: int,
        edge_feature_dim: int = 4,
        part_feature_dim: int = 0,
        keypoint_feature_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 4,
        readout: KeypointReadout = "global",
    ) -> None:
        super().__init__()
        self.object_projector = nn.Sequential(
            nn.Linear(object_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ee_projector = nn.Sequential(
            nn.Linear(ee_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.part_projector = (
            nn.Sequential(nn.Linear(part_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            if part_feature_dim > 0
            else None
        )
        self.keypoint_projector = (
            nn.Sequential(nn.Linear(keypoint_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            if keypoint_feature_dim > 0
            else None
        )
        self.edge_projector = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([PartEdgeMessagePassingLayer(hidden_dim) for _ in range(num_layers)])
        if readout not in {"global", "ee"}:
            raise ValueError(f"Unknown keypoint readout: {readout}")
        self.readout = readout
        readout_dim = hidden_dim * (4 if readout == "global" else 3)
        self.action_head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, graph: KeypointGraphBatch) -> torch.Tensor:
        object_state = self.object_projector(graph.object_features)
        ee_state = self.ee_projector(graph.ee_features)
        node_states = [object_state, ee_state]
        geometry_state = torch.zeros_like(object_state)
        if graph.part_features is not None:
            if self.part_projector is None:
                raise ValueError("Received part features but this model has no part projector.")
            geometry_state = self.part_projector(graph.part_features)
            node_states.append(geometry_state)
        if graph.keypoint_features is not None:
            if self.keypoint_projector is None:
                raise ValueError("Received keypoint features but this model has no keypoint projector.")
            keypoint_state = self.keypoint_projector(graph.keypoint_features)
            geometry_state = keypoint_state.mean(dim=1)
            node_states.extend(torch.unbind(keypoint_state, dim=1))
        node_state = torch.stack(node_states, dim=1)
        edge_state = self.edge_projector(graph.edge_features)
        for layer in self.layers:
            node_state = layer(node_state, edge_state, graph.edge_index)
        object_state = node_state[:, 0]
        ee_state = node_state[:, 1]
        if graph.part_features is not None:
            geometry_state = node_state[:, 2]
        elif graph.keypoint_features is not None:
            geometry_state = node_state[:, 2:].mean(dim=1)
        global_state = node_state.mean(dim=1)
        if self.readout == "ee":
            readout_state = torch.cat([ee_state, object_state, geometry_state], dim=-1)
        else:
            readout_state = torch.cat([object_state, ee_state, geometry_state, global_state], dim=-1)
        return self.action_head(readout_state)


@dataclass(frozen=True)
class HandleTaskState:
    object_center: torch.Tensor
    object_yaw: float
    handle_side: int
    handle_offset: float
    object_half_extent: tuple[float, float]
    gripper_command: float
    camera_yaw: float

    @property
    def handle_position(self) -> torch.Tensor:
        local = torch.tensor(
            [self.handle_side * self.handle_offset, 0.0, 0.0],
            dtype=torch.float32,
        )
        return self.object_center + _rotate_xz(local, self.object_yaw)


@dataclass(frozen=True)
class PartTaskState:
    object_center: torch.Tensor
    object_yaw: float
    part_local: torch.Tensor
    part_local_yaw: float
    part_extent: tuple[float, float]
    object_half_extent: tuple[float, float]
    gripper_command: float
    camera_yaw: float
    layout: PartLayout

    @property
    def part_world_position(self) -> torch.Tensor:
        return self.object_center + _rotate_xz(self.part_local.float(), self.object_yaw)

    @property
    def part_world_yaw(self) -> float:
        return float(self.object_yaw + self.part_local_yaw)


@dataclass(frozen=True)
class PartEpisodeSpec:
    episode_id: int
    task_state: PartTaskState
    arm_qpos: tuple[float, float, float, float]


@dataclass(frozen=True)
class KeypointTaskState:
    object_center: torch.Tensor
    object_yaw: float
    object_half_extent: tuple[float, float]
    part_offset_local: torch.Tensor
    part_yaw_local: float
    length: float
    curvature: float
    spacing_jitter: float
    keypoints_local: torch.Tensor
    keypoints_world: torch.Tensor
    gripper_command: float
    ee_yaw: float
    split: KeypointEvalSplit

    @property
    def target_position(self) -> torch.Tensor:
        return self.keypoints_world[self.keypoints_world.shape[0] // 2].float()

    @property
    def target_yaw(self) -> float:
        return keypoint_local_tangent_yaw(self.keypoints_world)

    @property
    def principal_yaw(self) -> float:
        delta = self.keypoints_world[-1] - self.keypoints_world[0]
        return float(math.atan2(float(delta[2]), float(delta[0])))


@dataclass(frozen=True)
class KeypointEpisodeSpec:
    episode_id: int
    task_state: KeypointTaskState
    arm_qpos: tuple[float, float, float, float]


def generate_representation_demonstrations(config: RepresentationDemoConfig) -> dict[str, Any]:
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    env_config = _env_config(config)
    env = MujocoManipulatorEnv(env_config, seed=config.seed)
    episodes: list[dict[str, Any]] = []
    render_paths: list[str] = []

    for episode_id in range(config.episodes):
        state = _sample_task_state(config, rng)
        episode = _collect_expert_episode(
            env=env,
            config=config,
            task_state=state,
            episode_id=episode_id,
            render=config.render_first_episode and episode_id == 0,
            render_dir=Path(config.output_path).parent / "expert_frames",
        )
        render_paths.extend(episode.pop("render_frames", []))
        episodes.append(episode)

    env.close()
    split = _episode_split(len(episodes), config.seed)
    feature_dims = _feature_dims(episodes[0])
    payload = {
        "task": "handle_contact_representation_feasibility",
        "config": asdict(config),
        "env_config": asdict(env_config),
        "episodes": episodes,
        "split": split,
        "feature_dims": feature_dims,
    }
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    metadata = _dataset_metadata(payload)
    metadata["render_frames"] = render_paths
    metadata_path = Path(config.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"dataset_path": str(output_path), "metadata_path": str(metadata_path), "metadata": metadata}


def generate_part_node_demonstrations(config: PartNodeDemoConfig) -> dict[str, Any]:
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    env_config = _part_env_config(config)
    env = MujocoManipulatorEnv(env_config, seed=config.seed)
    episodes: list[dict[str, Any]] = []
    ood_episodes: list[dict[str, Any]] = []
    render_paths: list[str] = []

    for episode_id in range(config.episodes):
        state = _sample_part_task_state(config, rng, layout="iid")
        episode = _collect_part_expert_episode(
            env=env,
            config=config,
            task_state=state,
            episode_id=episode_id,
            render=config.render_first_episode and episode_id == 0,
            render_dir=Path(config.output_path).parent / "part_expert_frames",
        )
        render_paths.extend(episode.pop("render_frames", []))
        episodes.append(episode)

    for episode_id in range(config.ood_episodes):
        state = _sample_part_task_state(config, rng, layout="ood")
        ood_episodes.append(
            _collect_part_expert_episode(
                env=env,
                config=config,
                task_state=state,
                episode_id=episode_id,
            )
        )

    env.close()
    split = _episode_split(len(episodes), config.seed)
    feature_dims = _part_feature_dims(episodes[0])
    payload = {
        "task": "oracle_part_node_feasibility",
        "config": asdict(config),
        "env_config": asdict(env_config),
        "episodes": episodes,
        "ood_episodes": ood_episodes,
        "split": split,
        "feature_dims": feature_dims,
    }
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    metadata = _part_dataset_metadata(payload)
    metadata["render_frames"] = render_paths
    metadata_path = Path(config.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"dataset_path": str(output_path), "metadata_path": str(metadata_path), "metadata": metadata}


def generate_keypoint_alignment_dataset(config: KeypointDemoConfig) -> dict[str, Any]:
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    env_config = _keypoint_env_config(config)
    env = MujocoManipulatorEnv(env_config, seed=config.seed)
    episodes: list[dict[str, Any]] = []
    ood: dict[str, list[dict[str, Any]]] = {"length_ood": [], "orientation_ood": [], "shape_ood": []}
    for episode_id in range(config.episodes):
        state = _sample_keypoint_task_state(config, rng, "iid")
        episodes.append(_collect_keypoint_expert_episode(env, config, state, episode_id))
    for split_name in ood:
        for episode_id in range(config.ood_episodes):
            state = _sample_keypoint_task_state(config, rng, split_name)  # type: ignore[arg-type]
            ood[split_name].append(_collect_keypoint_expert_episode(env, config, state, episode_id))
    env.close()
    split = _episode_split(len(episodes), config.seed)
    feature_dims = _keypoint_feature_dims(episodes[0])
    payload = {
        "task": "sparse_keypoint_alignment_feasibility",
        "config": asdict(config),
        "env_config": asdict(env_config),
        "episodes": episodes,
        "ood_episodes": ood,
        "split": split,
        "feature_dims": feature_dims,
        "geometry_statistics": keypoint_geometry_statistics(episodes, ood),
    }
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    metadata_path = Path(config.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload["geometry_statistics"], indent=2), encoding="utf-8")
    return {
        "dataset_path": str(output_path),
        "metadata_path": str(metadata_path),
        "metadata": {
            "task": payload["task"],
            "num_episodes": len(episodes),
            "num_ood_episodes": {key: len(value) for key, value in ood.items()},
            "split": split,
            "feature_dims": feature_dims,
            "geometry_statistics": payload["geometry_statistics"],
        },
    }


def train_representation_policy(config: RepresentationTrainConfig) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(config.seed)
    device = select_device(config.device)
    dataset = load_representation_dataset(config.dataset_path)
    if config.model_kind == "mpnn" and config.variant != "pose_visual_ee":
        raise ValueError("The MPNN architecture ablation is defined only for fixed representation C: pose_visual_ee.")
    input_dim = _variant_feature_dim(dataset, config.variant)
    model = _build_representation_model(config, dataset).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_observations, train_actions = _split_observations(dataset, "train", config.variant, config.model_kind)
    val_observations, val_actions = _split_observations(dataset, "val", config.variant, config.model_kind)
    train_observations = _observation_to(train_observations, device)
    train_actions = train_actions.to(device)
    val_observations = _observation_to(val_observations, device)
    val_actions = val_actions.to(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    start_time = time.perf_counter()
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(_observation_num_samples(train_observations), generator=generator)
        losses: list[float] = []
        for start in range(0, _observation_num_samples(train_observations), config.batch_size):
            batch_ids = permutation[start : start + config.batch_size].to(device)
            prediction = model(_observation_index_select(train_observations, batch_ids))
            target = train_actions.index_select(0, batch_ids)
            loss = torch.nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        validation = _evaluate_observation_split(model, val_observations, val_actions)
        record = {
            "epoch": epoch,
            "train_mse": float(np.mean(losses)) if losses else 0.0,
            "val_mse": validation["mse"],
            "val_l2_error": validation["l2_error"],
            "val_cosine": validation["cosine"],
        }
        history.append(record)
        if config.log_every and (epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs):
            print(
                f"[repr:{config.variant}] epoch={epoch} train_mse={record['train_mse']:.6f} "
                f"val_l2={record['val_l2_error']:.4f} val_cos={record['val_cosine']:.3f}"
            )

    offline = {
        split: evaluate_representation_offline(model, dataset, split, config.variant, device, config.model_kind)
        for split in ("train", "val", "test")
    }
    output_dir = ensure_dir(config.output_dir)
    checkpoint_stem = config.variant if config.model_kind == "mlp" else f"{config.variant}_{config.model_kind}"
    checkpoint_path = output_dir / f"{checkpoint_stem}.pt"
    checkpoint = {
        "model_kind": config.model_kind,
        "variant": config.variant,
        "model_state": model.state_dict(),
        "config": asdict(config),
        "demo_config": dataset["config"],
        "env_config": dataset["env_config"],
        "feature_dims": dataset["feature_dims"],
        "input_dim": input_dim,
        "graph_dims": _graph_feature_dims(dataset) if config.model_kind == "mpnn" else None,
        "split": dataset["split"],
        "offline": offline,
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "runtime_seconds": time.perf_counter() - start_time,
    }
    torch.save(checkpoint, checkpoint_path)
    summary_path = output_dir / f"{checkpoint_stem}_summary.json"
    serializable = dict(checkpoint)
    serializable.pop("model_state", None)
    summary_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return model, {"checkpoint_path": str(checkpoint_path), "summary_path": str(summary_path), **serializable}


def train_part_node_policy(config: PartNodeTrainConfig) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(config.seed)
    device = select_device(config.device)
    dataset = load_representation_dataset(config.dataset_path)
    model = _build_part_node_model(config, dataset).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_observations, train_actions = _part_split_observations(dataset, "train", config.variant)
    val_observations, val_actions = _part_split_observations(dataset, "val", config.variant)
    train_observations = _observation_to(train_observations, device)
    train_actions = train_actions.to(device)
    val_observations = _observation_to(val_observations, device)
    val_actions = val_actions.to(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    start_time = time.perf_counter()
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(_observation_num_samples(train_observations), generator=generator)
        losses: list[float] = []
        for start in range(0, _observation_num_samples(train_observations), config.batch_size):
            batch_ids = permutation[start : start + config.batch_size].to(device)
            prediction = model(_observation_index_select(train_observations, batch_ids))
            target = train_actions.index_select(0, batch_ids)
            loss = torch.nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        validation = _evaluate_observation_split(model, val_observations, val_actions)
        record = {
            "epoch": epoch,
            "train_mse": float(np.mean(losses)) if losses else 0.0,
            "val_mse": validation["mse"],
            "val_l2_error": validation["l2_error"],
            "val_cosine": validation["cosine"],
        }
        history.append(record)
        if config.log_every and (epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs):
            print(
                f"[part:{config.variant}] epoch={epoch} train_mse={record['train_mse']:.6f} "
                f"val_l2={record['val_l2_error']:.4f} val_cos={record['val_cosine']:.3f}"
            )

    offline = {
        split: evaluate_part_node_offline(model, dataset, split, config.variant, device)
        for split in ("train", "val", "test", "ood")
    }
    output_dir = ensure_dir(config.output_dir)
    checkpoint_path = output_dir / f"{config.variant}.pt"
    graph_dims = _part_graph_feature_dims(dataset, config.variant)
    checkpoint = {
        "experiment": "oracle_part_node_feasibility",
        "variant": config.variant,
        "model_state": model.state_dict(),
        "config": asdict(config),
        "demo_config": dataset["config"],
        "env_config": dataset["env_config"],
        "feature_dims": dataset["feature_dims"],
        "graph_dims": graph_dims,
        "split": dataset["split"],
        "offline": offline,
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "runtime_seconds": time.perf_counter() - start_time,
    }
    torch.save(checkpoint, checkpoint_path)
    summary_path = output_dir / f"{config.variant}_summary.json"
    serializable = dict(checkpoint)
    serializable.pop("model_state", None)
    summary_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return model, {"checkpoint_path": str(checkpoint_path), "summary_path": str(summary_path), **serializable}


def train_keypoint_alignment_policy(config: KeypointTrainConfig) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(config.seed)
    device = select_device(config.device)
    dataset = load_representation_dataset(config.dataset_path)
    model = _build_keypoint_model(config, dataset).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_observations, train_actions = _keypoint_split_observations(dataset, "train", config.variant)
    val_observations, val_actions = _keypoint_split_observations(dataset, "val", config.variant)
    train_observations = _observation_to(train_observations, device)
    train_actions = train_actions.to(device)
    val_observations = _observation_to(val_observations, device)
    val_actions = val_actions.to(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(_observation_num_samples(train_observations), generator=generator)
        losses: list[float] = []
        for start in range(0, _observation_num_samples(train_observations), config.batch_size):
            batch_ids = permutation[start : start + config.batch_size].to(device)
            prediction = model(_observation_index_select(train_observations, batch_ids))
            target = train_actions.index_select(0, batch_ids)
            loss = keypoint_action_loss(prediction, target, config.yaw_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        validation = _evaluate_keypoint_observation_split(model, val_observations, val_actions)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_translation_l2": validation["translation_l2"],
            "val_yaw_abs_error": validation["yaw_abs_error"],
        }
        history.append(record)
        if config.log_every and (epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs):
            print(
                f"[keypoint:{config.variant}] epoch={epoch} train_loss={record['train_loss']:.6f} "
                f"val_xyz={record['val_translation_l2']:.4f} val_yaw={record['val_yaw_abs_error']:.4f}"
            )
    offline = {
        split: evaluate_keypoint_offline(model, dataset, split, config.variant, device)
        for split in ("train", "val", "test", "length_ood", "orientation_ood", "shape_ood")
    }
    output_dir = ensure_dir(config.output_dir)
    checkpoint_path = output_dir / f"{config.variant}.pt"
    dims = _keypoint_model_dims(dataset, config.variant)
    checkpoint = {
        "experiment": "sparse_keypoint_alignment_feasibility",
        "variant": config.variant,
        "model_state": model.state_dict(),
        "config": asdict(config),
        "demo_config": dataset["config"],
        "env_config": dataset["env_config"],
        "feature_dims": dataset["feature_dims"],
        "model_dims": dims,
        "split": dataset["split"],
        "offline": offline,
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "runtime_seconds": time.perf_counter() - start_time,
    }
    torch.save(checkpoint, checkpoint_path)
    summary_path = output_dir / f"{config.variant}_summary.json"
    serializable = dict(checkpoint)
    serializable.pop("model_state", None)
    summary_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return model, {"checkpoint_path": str(checkpoint_path), "summary_path": str(summary_path), **serializable}


def run_keypoint_diagnostics(config: KeypointDiagnosticConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    dataset = load_representation_dataset(config.dataset_path)
    device = select_device(config.device)
    target_decode = {
        variant: _run_keypoint_target_decode_diagnostic(dataset, variant, config, device)
        for variant in ("keypoint_concat", "keypoint_graph")
    }
    overfit = {
        variant: _run_keypoint_action_overfit_diagnostic(dataset, variant, config, device)
        for variant in ("keypoint_concat", "keypoint_graph")
    }
    cpu_mps = _run_keypoint_cpu_mps_consistency_check(dataset, config)
    payload = {
        "config": asdict(config),
        "device": str(device),
        "target_decode": target_decode,
        "action_overfit": overfit,
        "cpu_mps_consistency": cpu_mps,
    }
    _write_json(output_dir / "diagnostics.json", payload)
    (output_dir / "summary.md").write_text(keypoint_diagnostics_summary_markdown(payload), encoding="utf-8")
    payload["output_path"] = str(output_dir / "diagnostics.json")
    payload["summary_path"] = str(output_dir / "summary.md")
    return payload


def _run_keypoint_target_decode_diagnostic(
    dataset: dict[str, Any],
    variant: KeypointVariant,
    config: KeypointDiagnosticConfig,
    device: torch.device,
    readout: KeypointReadout = "global",
) -> dict[str, Any]:
    model = _build_keypoint_diagnostic_model(dataset, variant, output_dim=5, readout=readout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_episodes = _keypoint_episodes_for_split(dataset, "train")
    val_episodes = _keypoint_episodes_for_split(dataset, "val")
    train_observations = _observation_to(_keypoint_observations_from_episodes(train_episodes, variant), device)
    train_targets = _keypoint_target_decode_targets(train_episodes).to(device)
    val_observations = _observation_to(_keypoint_observations_from_episodes(val_episodes, variant), device)
    val_targets = _keypoint_target_decode_targets(val_episodes).to(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 17)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.target_epochs + 1):
        model.train()
        permutation = torch.randperm(_observation_num_samples(train_observations), generator=generator)
        losses: list[float] = []
        for start in range(0, _observation_num_samples(train_observations), config.batch_size):
            batch_ids = permutation[start : start + config.batch_size].to(device)
            prediction = model(_observation_index_select(train_observations, batch_ids))
            target = train_targets.index_select(0, batch_ids)
            loss = keypoint_target_decode_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch == 1 or epoch == config.target_epochs:
            validation = evaluate_keypoint_target_decode(model, val_observations, val_targets)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **validation})
    return {
        "variant": variant,
        "epochs": config.target_epochs,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train": evaluate_keypoint_target_decode(model, train_observations, train_targets),
        "val": evaluate_keypoint_target_decode(model, val_observations, val_targets),
        "history": history,
    }


def _run_keypoint_action_overfit_diagnostic(
    dataset: dict[str, Any],
    variant: KeypointVariant,
    config: KeypointDiagnosticConfig,
    device: torch.device,
) -> dict[str, Any]:
    model = _build_keypoint_diagnostic_model(dataset, variant, output_dim=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_episodes = _keypoint_episodes_for_split(dataset, "train")[: config.overfit_episodes]
    observations = _observation_to(_keypoint_observations_from_episodes(train_episodes, variant), device)
    actions = torch.cat([episode["actions"].float() for episode in train_episodes], dim=0).to(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 23)
    initial = _evaluate_keypoint_observation_split(model, observations, actions)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.overfit_epochs + 1):
        model.train()
        permutation = torch.randperm(_observation_num_samples(observations), generator=generator)
        losses: list[float] = []
        for start in range(0, _observation_num_samples(observations), config.batch_size):
            batch_ids = permutation[start : start + config.batch_size].to(device)
            prediction = model(_observation_index_select(observations, batch_ids))
            target = actions.index_select(0, batch_ids)
            loss = keypoint_action_loss(prediction, target, yaw_loss_weight=0.25)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch in {1, config.overfit_epochs}:
            metrics = _evaluate_keypoint_observation_split(model, observations, actions)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
    return {
        "variant": variant,
        "episodes": len(train_episodes),
        "samples": int(actions.shape[0]),
        "epochs": config.overfit_epochs,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial": initial,
        "final": _evaluate_keypoint_observation_split(model, observations, actions),
        "history": history,
    }


def _run_keypoint_cpu_mps_consistency_check(dataset: dict[str, Any], config: KeypointDiagnosticConfig) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"skipped": True, "reason": "MPS is not available."}
    checkpoint_dir = Path(config.checkpoint_dir)
    episodes = _keypoint_episodes_for_split(dataset, "val")
    results: dict[str, Any] = {"skipped": False}
    for variant in ("keypoint_concat", "keypoint_graph"):
        checkpoint_path = checkpoint_dir / f"{variant}.pt"
        if not checkpoint_path.exists():
            results[variant] = {"skipped": True, "reason": f"Missing checkpoint: {checkpoint_path}"}
            continue
        cpu_device = select_device("cpu")
        mps_device = select_device("mps")
        cpu_model, _ = load_keypoint_policy(checkpoint_path, cpu_device)
        mps_model, _ = load_keypoint_policy(checkpoint_path, mps_device)
        observations = _keypoint_observations_from_episodes(episodes, variant)
        sample_count = min(64, _observation_num_samples(observations))
        sample_ids = torch.arange(sample_count, dtype=torch.long)
        cpu_obs = _observation_index_select(observations, sample_ids)
        mps_obs = _observation_to(cpu_obs, mps_device)
        with torch.no_grad():
            cpu_output = cpu_model(_observation_to(cpu_obs, cpu_device)).detach().cpu()
            mps_output = mps_model(mps_obs).detach().cpu()
        diff = (cpu_output - mps_output).abs()
        results[variant] = {
            "samples": sample_count,
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
        }
    return results


def _build_keypoint_diagnostic_model(
    dataset: dict[str, Any],
    variant: KeypointVariant,
    output_dim: int,
    readout: KeypointReadout = "global",
) -> nn.Module:
    dims = _keypoint_model_dims(dataset, variant)
    if variant == "keypoint_concat":
        return KeypointConcatMLP(input_dim=dims["input"], hidden_dim=256, output_dim=output_dim)
    return KeypointMPNN(
        object_feature_dim=dims["object"],
        ee_feature_dim=dims["ee"],
        edge_feature_dim=dims["edge"],
        part_feature_dim=dims.get("part", 0),
        keypoint_feature_dim=dims.get("keypoint", 0),
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        readout=readout,
    )


def _keypoint_episodes_for_split(dataset: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    if split_name in {"length_ood", "orientation_ood", "shape_ood"}:
        return list(dataset["ood_episodes"][split_name])
    return [dataset["episodes"][idx] for idx in dataset["split"][split_name]]


def _keypoint_observations_from_episodes(
    episodes: list[dict[str, Any]],
    variant: KeypointVariant,
) -> torch.Tensor | KeypointGraphBatch:
    object_features = torch.cat([episode["object_features"].float() for episode in episodes], dim=0)
    ee_features = torch.cat([episode["ee_features"].float() for episode in episodes], dim=0)
    part_features = torch.cat([episode["part_features"].float() for episode in episodes], dim=0)
    keypoint_features = torch.cat([episode["keypoint_features"].float() for episode in episodes], dim=0)
    return keypoint_observation_from_feature_tensors(object_features, ee_features, part_features, keypoint_features, variant)


def _keypoint_target_decode_targets(episodes: list[dict[str, Any]]) -> torch.Tensor:
    rows = []
    for episode in episodes:
        steps = int(episode["actions"].shape[0])
        target_position = torch.tensor(episode["target_position"], dtype=torch.float32).reshape(1, 3)
        target_yaw = float(episode["target_yaw"])
        target = torch.cat(
            [
                target_position,
                torch.tensor([[math.sin(target_yaw), math.cos(target_yaw)]], dtype=torch.float32),
            ],
            dim=-1,
        )
        rows.append(target.repeat(steps, 1))
    return torch.cat(rows, dim=0)


def keypoint_target_decode_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    position_loss = torch.nn.functional.mse_loss(prediction[:, :3], target[:, :3])
    pred_yaw_vec = torch.nn.functional.normalize(prediction[:, 3:5], dim=-1)
    yaw_loss = torch.nn.functional.mse_loss(pred_yaw_vec, target[:, 3:5])
    return position_loss + yaw_loss


@torch.no_grad()
def evaluate_keypoint_target_decode(
    model: nn.Module,
    observations: torch.Tensor | KeypointGraphBatch,
    targets: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    prediction = model(observations)
    position_l2 = (prediction[:, :3] - targets[:, :3]).norm(dim=-1)
    pred_vec = torch.nn.functional.normalize(prediction[:, 3:5], dim=-1)
    pred_yaw = torch.atan2(pred_vec[:, 0], pred_vec[:, 1])
    target_yaw = torch.atan2(targets[:, 3], targets[:, 4])
    directed_yaw = torch.remainder(pred_yaw - target_yaw + math.pi, 2.0 * math.pi) - math.pi
    directed_abs = directed_yaw.abs()
    axis_abs = torch.minimum(directed_abs, (math.pi - directed_abs).abs())
    return {
        "position_l2": float(position_l2.mean().detach().cpu().item()),
        "position_l2_p90": float(torch.quantile(position_l2.detach().cpu(), 0.90).item()),
        "directed_yaw_abs_error": float(directed_abs.mean().detach().cpu().item()),
        "directed_yaw_abs_error_p90": float(torch.quantile(directed_abs.detach().cpu(), 0.90).item()),
        "axis_yaw_abs_error": float(axis_abs.mean().detach().cpu().item()),
        "axis_yaw_abs_error_p90": float(torch.quantile(axis_abs.detach().cpu(), 0.90).item()),
    }


@torch.no_grad()
def evaluate_representation_policy(config: RepresentationEvalConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = select_device(config.device)
    model, checkpoint = load_representation_policy(config.checkpoint_path, device)
    variant: RepresentationVariant = checkpoint["variant"]
    model_kind: RepresentationModelKind = checkpoint.get("model_kind", "mlp")
    demo_config = _demo_config_from_checkpoint(checkpoint)
    if config.max_steps != demo_config.max_steps:
        demo_config.max_steps = config.max_steps
    if config.camera_yaw is not None:
        demo_config.camera_yaw = config.camera_yaw
    env = MujocoManipulatorEnv(_env_config(demo_config), seed=config.seed)
    rng = np.random.default_rng(config.seed)
    results: list[dict[str, Any]] = []
    output_dir = ensure_dir(config.output_dir)

    for episode_idx in range(config.episodes):
        task_state = _sample_task_state(demo_config, rng)
        result = run_representation_rollout(
            model=model,
            variant=variant,
            model_kind=model_kind,
            env=env,
            config=demo_config,
            task_state=task_state,
            device=device,
            episode_id=episode_idx,
            render=config.render_first_episode and episode_idx == 0,
            render_dir=output_dir / f"{variant}_frames_ep{episode_idx:03d}",
        )
        results.append(result)

    env.close()
    summary = summarize_rollouts(results)
    payload = {
        "checkpoint_path": str(config.checkpoint_path),
        "model_kind": model_kind,
        "variant": variant,
        "device": str(device),
        "config": asdict(config),
        "demo_config": asdict(demo_config),
        "summary": summary,
        "episodes": results,
    }
    suffix = "view_shift" if config.camera_yaw is not None else "normal"
    output_stem = variant if model_kind == "mlp" else f"{variant}_{model_kind}"
    output_path = output_dir / f"{output_stem}_{suffix}_eval.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["output_path"] = str(output_path)
    return payload


@torch.no_grad()
def evaluate_part_node_policy(config: PartNodeEvalConfig) -> dict[str, Any]:
    if config.eval_workers != 1:
        raise ValueError("Part-node rollout currently supports eval_workers=1 only; each run creates one MuJoCo env.")
    set_seed(config.seed)
    device = select_device(config.eval_device)
    model, checkpoint = load_part_node_policy(config.checkpoint_path, device)
    variant: PartNodeVariant = checkpoint["variant"]
    demo_config = _part_demo_config_from_checkpoint(checkpoint)
    if config.max_steps != demo_config.max_steps:
        demo_config.max_steps = config.max_steps
    env = MujocoManipulatorEnv(_part_env_config(demo_config), seed=config.seed)
    rng = np.random.default_rng(config.seed)
    results: list[dict[str, Any]] = []
    output_dir = ensure_dir(config.output_dir)

    for episode_idx in range(config.episodes):
        task_state = _sample_part_task_state(demo_config, rng, layout=config.layout)
        result = run_part_node_rollout(
            model=model,
            variant=variant,
            env=env,
            config=demo_config,
            task_state=task_state,
            device=device,
            episode_id=episode_idx,
            render=config.render_first_episode and episode_idx == 0,
            render_dir=output_dir / f"{variant}_{config.layout}_frames_ep{episode_idx:03d}",
        )
        results.append(result)

    env.close()
    summary = summarize_rollouts(results)
    payload = {
        "checkpoint_path": str(config.checkpoint_path),
        "variant": variant,
        "layout": config.layout,
        "device": str(device),
        "config": asdict(config),
        "demo_config": asdict(demo_config),
        "summary": summary,
        "episodes": results,
    }
    output_path = output_dir / f"{variant}_{config.layout}_eval.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["output_path"] = str(output_path)
    return payload


@torch.no_grad()
def evaluate_part_node_paired_rollouts(config: PartNodePairedEvalConfig) -> dict[str, Any]:
    if config.eval_workers != 1:
        raise ValueError("Paired MuJoCo validation currently supports eval_workers=1 only.")
    set_seed(config.eval_seed_base)
    device = select_device(config.eval_device)
    artifact_dir = Path(config.artifact_dir)
    output_dir = ensure_dir(config.output_dir)
    seed_payloads: list[dict[str, Any]] = []

    for seed_idx, seed in enumerate(config.seeds):
        seed_name = f"seed{seed}"
        seed_output_dir = ensure_dir(output_dir / seed_name)
        concat_ckpt = artifact_dir / seed_name / "checkpoints" / "part_concat.pt"
        node_ckpt = artifact_dir / seed_name / "checkpoints" / "part_node.pt"
        if not concat_ckpt.exists() or not node_ckpt.exists():
            raise FileNotFoundError(f"Missing part_concat/part_node checkpoints under {artifact_dir / seed_name}")
        concat_model, concat_checkpoint = load_part_node_policy(concat_ckpt, device)
        node_model, node_checkpoint = load_part_node_policy(node_ckpt, device)
        concat_config = _part_demo_config_from_checkpoint(concat_checkpoint)
        node_config = _part_demo_config_from_checkpoint(node_checkpoint)
        if concat_checkpoint["variant"] != "part_concat" or node_checkpoint["variant"] != "part_node":
            raise ValueError(f"Unexpected checkpoint variants for {seed_name}.")
        if asdict(concat_config) != asdict(node_config):
            raise ValueError(f"Demo configs differ between paired checkpoints for {seed_name}.")
        demo_config = concat_config
        demo_config.max_steps = config.max_steps
        eval_seed = config.eval_seed_base + seed_idx
        specs = sample_part_episode_specs(demo_config, config.episodes, eval_seed, config.layout)
        concat_env = MujocoManipulatorEnv(_part_env_config(demo_config), seed=eval_seed)
        node_env = MujocoManipulatorEnv(_part_env_config(demo_config), seed=eval_seed)
        episode_results: list[dict[str, Any]] = []
        for spec in specs:
            concat_result = run_part_node_rollout_from_spec(
                model=concat_model,
                variant="part_concat",
                env=concat_env,
                config=demo_config,
                spec=spec,
                device=device,
            )
            node_result = run_part_node_rollout_from_spec(
                model=node_model,
                variant="part_node",
                env=node_env,
                config=demo_config,
                spec=spec,
                device=device,
            )
            paired_result = _paired_episode_result(spec, concat_result, node_result)
            paired_result["seed"] = seed_name
            paired_result["training_seed"] = seed
            paired_result["eval_seed"] = eval_seed
            episode_results.append(paired_result)
        concat_env.close()
        node_env.close()

        seed_summary = summarize_paired_episode_results(
            episode_results,
            bootstrap_samples=config.bootstrap_samples,
            seed=eval_seed,
        )
        worst_cases = paired_worst_cases(
            episode_results,
            catastrophic_distance=config.catastrophic_distance,
        )
        _write_json(seed_output_dir / "episode_results.json", {"seed": seed, "episodes": episode_results})
        _write_json(seed_output_dir / "contingency.json", seed_summary)
        _write_json(seed_output_dir / "worst_cases.json", worst_cases)
        if config.plot_worst > 0:
            plots_dir = ensure_dir(seed_output_dir / "plots")
            for case in worst_cases["top_part_node_final_distance"][: config.plot_worst]:
                episode_id = int(case["episode_id"])
                episode = episode_results[episode_id]
                plot_path = plots_dir / f"episode_{episode_id:04d}_trajectory.png"
                plot_paired_episode_trajectory(episode, plot_path)
                case["plot_path"] = str(plot_path)
            _write_json(seed_output_dir / "worst_cases.json", worst_cases)
        seed_payloads.append(
            {
                "seed": seed,
                "seed_name": seed_name,
                "eval_seed": eval_seed,
                "output_dir": str(seed_output_dir),
                "summary": seed_summary,
                "worst_cases": worst_cases,
            }
        )

    all_episodes = []
    for item in seed_payloads:
        episode_payload = json.loads((Path(item["output_dir"]) / "episode_results.json").read_text(encoding="utf-8"))
        all_episodes.extend(episode_payload["episodes"])
    aggregate = summarize_paired_episode_results(
        all_episodes,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.eval_seed_base + 100_000,
    )
    aggregate["geometry_breakdown"] = paired_geometry_breakdown(all_episodes)
    aggregate["top_part_node_final_distance"] = paired_worst_cases(
        all_episodes,
        catastrophic_distance=config.catastrophic_distance,
    )["top_part_node_final_distance"][:10]
    payload = {
        "config": asdict(config),
        "device": str(device),
        "seeds": seed_payloads,
        "aggregate": aggregate,
        "reference_latency_ms": {
            "part_concat_policy": 0.504,
            "part_node_policy": 0.578,
        },
    }
    _write_json(output_dir / "aggregated_results.json", payload)
    (output_dir / "summary.md").write_text(paired_validation_summary_markdown(payload), encoding="utf-8")
    payload["output_path"] = str(output_dir / "aggregated_results.json")
    payload["summary_path"] = str(output_dir / "summary.md")
    return payload


def run_keypoint_feasibility_experiment(config: KeypointExperimentConfig) -> dict[str, Any]:
    artifact_dir = ensure_dir(config.artifact_dir)
    _write_json(artifact_dir / "config.json", asdict(config))
    if config.eval_workers > 1 and len(config.seeds) > 1:
        worker_count = min(int(config.eval_workers), len(config.seeds))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            seed_payloads = list(executor.map(_run_keypoint_seed_experiment, config.seeds, [config] * len(config.seeds)))
        seed_payloads.sort(key=lambda item: int(item["seed"]))
    else:
        seed_payloads = [_run_keypoint_seed_experiment(seed, config) for seed in config.seeds]
    aggregate = aggregate_keypoint_experiment(seed_payloads)
    aggregate["config"] = asdict(config)
    aggregate["geometry_statistics"] = {
        f"seed{item['seed']}": item["dataset"]["metadata"]["geometry_statistics"] for item in seed_payloads
    }
    _write_json(artifact_dir / "aggregated_results.json", aggregate)
    (artifact_dir / "summary.md").write_text(keypoint_summary_markdown(aggregate), encoding="utf-8")
    aggregate["output_path"] = str(artifact_dir / "aggregated_results.json")
    aggregate["summary_path"] = str(artifact_dir / "summary.md")
    return aggregate


def run_keypoint_structural_diagnostics(config: KeypointStructuralDiagnosticConfig) -> dict[str, Any]:
    """Run the narrow arc-position/readout follow-up without touching prior artifacts."""
    output_dir = ensure_dir(config.output_dir)
    _write_json(output_dir / "config.json", asdict(config))
    seed_payloads = []
    for seed in config.seeds:
        seed_payloads.append(_run_keypoint_structural_seed(seed, config))
    aggregate = _aggregate_keypoint_structural(seed_payloads)
    aggregate["config"] = asdict(config)
    _write_json(output_dir / "aggregated_results.json", aggregate)
    (output_dir / "summary.md").write_text(keypoint_structural_summary_markdown(aggregate), encoding="utf-8")
    (output_dir / "final_summary.md").write_text(keypoint_structural_summary_markdown(aggregate), encoding="utf-8")
    aggregate["output_path"] = str(output_dir / "aggregated_results.json")
    aggregate["summary_path"] = str(output_dir / "summary.md")
    return aggregate


def _run_keypoint_structural_seed(seed: int, config: KeypointStructuralDiagnosticConfig) -> dict[str, Any]:
    base = Path(config.artifact_dir) / f"seed{seed}_run" / f"seed{seed}"
    dataset_path = base / "dataset" / "keypoint_alignment.pt"
    old_checkpoint_dir = base / "checkpoints"
    dataset = load_representation_dataset(dataset_path)
    seed_dir = ensure_dir(Path(config.output_dir) / f"seed{seed}")
    checkpoint_dir = ensure_dir(seed_dir / "checkpoints")
    arc_checkpoints: dict[str, Path] = {}
    train_summaries: dict[str, Any] = {}
    for readout in ("global", "ee"):
        _, summary = train_keypoint_alignment_policy(
            KeypointTrainConfig(
                dataset_path=str(dataset_path),
                variant="keypoint_graph_arc",
                output_dir=str(checkpoint_dir / readout),
                epochs=config.epochs,
                batch_size=config.batch_size,
                seed=seed,
                device=config.device,
                readout=readout,  # type: ignore[arg-type]
                log_every=max(config.epochs // 3, 1),
            )
        )
        arc_checkpoints[readout] = Path(summary["checkpoint_path"])
        train_summaries[f"keypoint_graph_arc_{readout}"] = {
            "offline": summary["offline"],
            "parameter_count": summary["parameter_count"],
            "runtime_seconds": summary["runtime_seconds"],
        }

    target_config = KeypointDiagnosticConfig(
        dataset_path=str(dataset_path),
        checkpoint_dir=str(old_checkpoint_dir),
        output_dir=str(seed_dir / "target_decode"),
        seed=seed,
        target_epochs=config.target_decode_epochs,
        batch_size=config.batch_size,
        device=config.device,
        eval_device=config.eval_device,
    )
    target_decode = {
        "keypoint_concat": _run_keypoint_target_decode_diagnostic(dataset, "keypoint_concat", target_config, select_device(config.device)),
        "keypoint_graph": _run_keypoint_target_decode_diagnostic(dataset, "keypoint_graph", target_config, select_device(config.device)),
        "keypoint_graph_arc": _run_keypoint_target_decode_diagnostic(
            dataset, "keypoint_graph_arc", target_config, select_device(config.device)
        ),
    }
    _write_json(seed_dir / "target_decode" / "results.json", target_decode)

    checkpoint_paths: dict[str, Path] = {
        "keypoint_concat": old_checkpoint_dir / "keypoint_concat.pt",
        "keypoint_graph": old_checkpoint_dir / "keypoint_graph.pt",
        "keypoint_graph_arc": arc_checkpoints["global"],
        "keypoint_graph_arc_ee": arc_checkpoints["ee"],
    }
    eval_payload = _evaluate_structural_paired_rollouts(
        checkpoint_paths=checkpoint_paths,
        output_dir=seed_dir / "paired_eval",
        episodes=config.eval_episodes,
        max_steps=config.max_steps,
        seed=seed + 1000,
        device_preference=config.eval_device,
        bootstrap_samples=config.bootstrap_samples,
        plot_worst=config.plot_worst,
    )
    return {
        "seed": seed,
        "dataset_path": str(dataset_path),
        "checkpoints": {name: str(path) for name, path in checkpoint_paths.items()},
        "train_summaries": train_summaries,
        "target_decode": target_decode,
        "eval": eval_payload,
        "geometry_statistics": dataset.get("metadata", {}).get("geometry_statistics", {}),
    }


def _evaluate_structural_paired_rollouts(
    checkpoint_paths: dict[str, Path],
    output_dir: Path,
    episodes: int,
    max_steps: int,
    seed: int,
    device_preference: DevicePreference,
    bootstrap_samples: int,
    plot_worst: int,
) -> dict[str, Any]:
    device = select_device(device_preference)
    output_dir = ensure_dir(output_dir)
    model_specs: dict[str, tuple[nn.Module, KeypointVariant]] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for label, path in checkpoint_paths.items():
        model, checkpoint = load_keypoint_policy(path, device)
        variant: KeypointVariant = checkpoint["variant"]
        model_specs[label] = (model, variant)
        checkpoints[label] = checkpoint
    demo_config = _keypoint_demo_config_from_checkpoint(checkpoints["keypoint_concat"])
    demo_config.max_steps = max_steps
    envs = {
        label: MujocoManipulatorEnv(_keypoint_env_config(demo_config), seed=seed + idx)
        for idx, label in enumerate(model_specs)
    }
    split_payloads: dict[str, Any] = {}
    for split_idx, split_name in enumerate(("iid", "length_ood", "orientation_ood", "shape_ood")):
        specs = sample_keypoint_episode_specs(demo_config, episodes, seed + split_idx * 1000, split_name)
        split_results = []
        for spec in specs:
            model_results = {
                label: run_keypoint_rollout_from_spec(
                    model=model,
                    variant=variant,
                    env=envs[label],
                    config=demo_config,
                    spec=spec,
                    device=device,
                    permutation=None,
                )
                for label, (model, variant) in model_specs.items()
            }
            split_results.append(_keypoint_paired_episode_result(spec, model_results))
        split_dir = ensure_dir(output_dir / split_name)
        _write_json(split_dir / "episode_results.json", {"split": split_name, "episodes": split_results})
        summary = _summarize_structural_split(split_results, bootstrap_samples, seed + split_idx)
        _write_json(split_dir / "summary.json", summary)
        if plot_worst > 0:
            plots_dir = ensure_dir(split_dir / "plots")
            for label in ("keypoint_graph", "keypoint_graph_arc", "keypoint_graph_arc_ee"):
                cases = sorted(
                    split_results,
                    key=lambda item: item["models"][label]["final_position_error"],
                    reverse=True,
                )[:plot_worst]
                for case in cases:
                    plot_path = plots_dir / f"{label}_episode_{case['episode_id']:04d}.png"
                    plot_keypoint_episode(case, plot_path)
        split_payloads[split_name] = summary
    for env in envs.values():
        env.close()
    payload = {
        "device": str(device),
        "splits": split_payloads,
        "parameter_count": {
            label: int(checkpoint["parameter_count"]) for label, checkpoint in checkpoints.items()
        },
    }
    _write_json(output_dir / "paired_eval_summary.json", payload)
    return payload


def _summarize_structural_split(episodes: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    labels = list(episodes[0]["models"].keys()) if episodes else []
    models = {
        label: summarize_keypoint_rollouts([episode["models"][label] for episode in episodes])
        for label in labels
    }
    pair_labels = [
        ("keypoint_concat", "keypoint_graph"),
        ("keypoint_concat", "keypoint_graph_arc"),
        ("keypoint_graph_arc", "keypoint_graph_arc_ee"),
    ]
    comparisons = {
        f"{baseline}_vs_{candidate}": _keypoint_pair_metrics(
            episodes, baseline, candidate, bootstrap_samples, seed + idx
        )
        for idx, (baseline, candidate) in enumerate(pair_labels)
        if baseline in labels and candidate in labels
    }
    return {"num_episodes": len(episodes), "models": models, "paired_comparisons": comparisons}


def _aggregate_keypoint_structural(seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    split_names = ("iid", "length_ood", "orientation_ood", "shape_ood")
    labels = ("keypoint_concat", "keypoint_graph", "keypoint_graph_arc", "keypoint_graph_arc_ee")
    splits: dict[str, Any] = {}
    for split in split_names:
        splits[split] = {}
        for label in labels:
            values = [item["eval"]["splits"][split]["models"][label] for item in seed_payloads]
            splits[split][label] = {
                metric: _mean_std([float(value[metric]) for value in values])
                for metric in ("success_rate", "final_position_error", "final_orientation_error", "mean_position_error", "mean_orientation_error")
            }
    target_decode = {}
    for label in ("keypoint_concat", "keypoint_graph", "keypoint_graph_arc"):
        vals = [item["target_decode"][label]["val"] for item in seed_payloads]
        target_decode[label] = {
            metric: _mean_std([float(value[metric]) for value in vals])
            for metric in ("position_l2", "directed_yaw_abs_error")
        }
    return {
        "seeds": seed_payloads,
        "splits": splits,
        "target_decode_validation": target_decode,
        "parameter_counts": {
            label: [item["eval"]["parameter_count"][label] for item in seed_payloads] for label in labels
        },
    }


def keypoint_structural_summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Keypoint Graph Structural Diagnostics",
        "",
        "Arc-position was added as one scalar to each keypoint node. The EE readout uses [EE, Object, mean-keypoint] states; the global readout preserves the prior [Object, EE, mean-keypoint, global] head.",
        "",
        "## Target Decode Validation",
        "",
        "| Model | Position L2 | Directed yaw error |",
        "|---|---:|---:|",
    ]
    for label, metrics in payload["target_decode_validation"].items():
        lines.append(f"| {label} | {metrics['position_l2']['mean']:.5f} +/- {metrics['position_l2']['std']:.5f} | {metrics['directed_yaw_abs_error']['mean']:.5f} +/- {metrics['directed_yaw_abs_error']['std']:.5f} |")
    lines.extend(["", "## Closed-loop Success", "", "| Model | IID | Length OOD | Orientation OOD | Shape OOD |", "|---|---:|---:|---:|---:|"])
    for label in ("keypoint_concat", "keypoint_graph", "keypoint_graph_arc", "keypoint_graph_arc_ee"):
        values = [payload["splits"][split][label]["success_rate"] for split in ("iid", "length_ood", "orientation_ood", "shape_ood")]
        lines.append("| " + label + " | " + " | ".join(f"{v['mean']:.3f} +/- {v['std']:.3f}" for v in values) + " |")
    lines.extend(["", "## Parameter Counts", ""])
    for label, values in payload["parameter_counts"].items():
        lines.append(f"- `{label}`: {values}")
    return "\n".join(lines) + "\n"


def _run_keypoint_seed_experiment(seed: int, config: KeypointExperimentConfig) -> dict[str, Any]:
    variants: tuple[KeypointVariant, ...] = ("single_part_node", "keypoint_concat", "keypoint_graph")
    eval_splits: tuple[KeypointEvalSplit, ...] = ("iid", "length_ood", "orientation_ood", "shape_ood")
    seed_dir = ensure_dir(Path(config.artifact_dir) / f"seed{seed}")
    dataset_path = seed_dir / "dataset" / "keypoint_alignment.pt"
    metadata_path = seed_dir / "dataset" / "geometry_statistics.json"
    demo = generate_keypoint_alignment_dataset(
        KeypointDemoConfig(
            episodes=config.episodes,
            ood_episodes=config.ood_episodes,
            max_steps=config.max_steps,
            seed=seed,
            output_path=str(dataset_path),
            metadata_path=str(metadata_path),
        )
    )
    checkpoints = {}
    train_summaries = {}
    for variant in variants:
        _, summary = train_keypoint_alignment_policy(
            KeypointTrainConfig(
                dataset_path=str(dataset_path),
                variant=variant,
                output_dir=str(seed_dir / "checkpoints"),
                epochs=config.epochs,
                batch_size=config.batch_size,
                seed=seed,
                device=config.device,
                log_every=max(config.epochs // 3, 1),
            )
        )
        checkpoints[variant] = summary["checkpoint_path"]
        train_summaries[variant] = summary
    eval_payload = evaluate_keypoint_models_paired(
        checkpoint_paths={variant: Path(path) for variant, path in checkpoints.items()},
        output_dir=seed_dir / "eval",
        eval_splits=eval_splits,
        episodes=config.eval_episodes,
        max_steps=config.max_steps,
        seed=seed + 1000,
        device_preference=config.eval_device,
        bootstrap_samples=config.bootstrap_samples,
        plot_worst=config.plot_worst,
    )
    return {
        "seed": seed,
        "dataset": demo,
        "checkpoints": checkpoints,
        "train_summaries": {
            variant: {
                "offline": summary["offline"],
                "parameter_count": summary["parameter_count"],
                "runtime_seconds": summary["runtime_seconds"],
            }
            for variant, summary in train_summaries.items()
        },
        "eval": eval_payload,
    }


@torch.no_grad()
def run_representation_rollout(
    model: nn.Module,
    variant: RepresentationVariant,
    model_kind: RepresentationModelKind,
    env: MujocoManipulatorEnv,
    config: RepresentationDemoConfig,
    task_state: HandleTaskState,
    device: torch.device,
    episode_id: int = 0,
    render: bool = False,
    render_dir: Path | None = None,
) -> dict[str, Any]:
    _reset_env_for_task(env, task_state)
    if render:
        render_dir = ensure_dir(render_dir or Path("artifacts/representation_feasibility/eval/frames"))
    frames: list[str] = []
    records: list[dict[str, Any]] = []
    policy_latencies_ms: list[float] = []
    visual_latencies_ms: list[float] = []
    state_latencies_ms: list[float] = []
    forward_latencies_ms: list[float] = []
    sim_latencies_ms: list[float] = []
    success = False

    for step_idx in range(config.max_steps):
        state_start = time.perf_counter()
        feature_parts, diagnostics, timing = _build_feature_parts(
            env,
            task_state,
            config,
            measure_visual=True,
            include_visual=VARIANT_FLAGS[variant]["visual"],
        )
        state_latencies_ms.append((time.perf_counter() - state_start) * 1000.0)
        visual_latencies_ms.append(timing["visual_ms"])
        observation = _observation_from_feature_parts(feature_parts, variant, model_kind)
        observation = _observation_to(observation, device)

        forward_start = time.perf_counter()
        raw_delta = model(observation).squeeze(0)
        if device.type != "cpu":
            _sync_if_needed(device)
        forward_ms = (time.perf_counter() - forward_start) * 1000.0
        forward_latencies_ms.append(forward_ms)
        policy_latencies_ms.append(state_latencies_ms[-1] + forward_ms)

        delta = clamp_delta(raw_delta.detach().cpu(), config.max_delta_ee)
        expert_delta = _expert_delta(env, task_state, config.max_delta_ee)
        sim_start = time.perf_counter()
        env.step_delta_ee(delta, gripper=task_state.gripper_command)
        sim_latencies_ms.append((time.perf_counter() - sim_start) * 1000.0)

        distance = _active_fingertip_distance(env, task_state)
        success = success or distance <= config.target_radius
        records.append(
            {
                "step": step_idx,
                "distance": distance,
                "success": distance <= config.target_radius,
                "delta_ee": delta.tolist(),
                "expert_delta_ee": expert_delta.tolist(),
                "action_l2_error": float((delta - expert_delta).norm().item()),
                "handle_position": task_state.handle_position.tolist(),
                "ee_position": env.robot_observation().ee_position.tolist(),
                "active_fingertip": _active_fingertip_position(env, task_state.handle_side).tolist(),
                "object_center": task_state.object_center.tolist(),
                "handle_side": task_state.handle_side,
                "visual_feature_sum": diagnostics["visual_feature_sum"],
            }
        )
        if render and render_dir is not None and step_idx % 4 == 0:
            frame_path = render_dir / f"frame_{step_idx:03d}.png"
            _save_frame(env, frame_path)
            frames.append(str(frame_path))
        if success:
            break

    final_distance = records[-1]["distance"] if records else _active_fingertip_distance(env, task_state)
    return {
        "episode": episode_id,
        "success": bool(success),
        "final_distance": float(final_distance),
        "steps": len(records),
        "handle_side": task_state.handle_side,
        "handle_offset": task_state.handle_offset,
        "object_yaw": task_state.object_yaw,
        "camera_yaw": task_state.camera_yaw,
        "gripper_command": task_state.gripper_command,
        "mean_policy_latency_ms": _mean(policy_latencies_ms),
        "mean_visual_latency_ms": _mean(visual_latencies_ms),
        "mean_state_latency_ms": _mean(state_latencies_ms),
        "mean_forward_latency_ms": _mean(forward_latencies_ms),
        "mean_sim_step_latency_ms": _mean(sim_latencies_ms),
        "records": records,
        "render_frames": frames,
    }


@torch.no_grad()
def run_part_node_rollout(
    model: nn.Module,
    variant: PartNodeVariant,
    env: MujocoManipulatorEnv,
    config: PartNodeDemoConfig,
    task_state: PartTaskState,
    device: torch.device,
    episode_id: int = 0,
    render: bool = False,
    render_dir: Path | None = None,
) -> dict[str, Any]:
    _reset_env_for_part_task(env, task_state)
    if render:
        render_dir = ensure_dir(render_dir or Path("artifacts/part_node_feasibility/eval/frames"))
    frames: list[str] = []
    records: list[dict[str, Any]] = []
    policy_latencies_ms: list[float] = []
    visual_latencies_ms: list[float] = []
    state_latencies_ms: list[float] = []
    forward_latencies_ms: list[float] = []
    sim_latencies_ms: list[float] = []
    success = False

    for step_idx in range(config.max_steps):
        state_start = time.perf_counter()
        feature_parts, diagnostics, timing = _build_part_feature_parts(env, task_state, config, measure_visual=True)
        state_latencies_ms.append((time.perf_counter() - state_start) * 1000.0)
        visual_latencies_ms.append(timing["visual_ms"])
        observation = _part_observation_from_feature_parts(feature_parts, variant)
        observation = _observation_to(observation, device)

        forward_start = time.perf_counter()
        raw_delta = model(observation).squeeze(0)
        if device.type != "cpu":
            _sync_if_needed(device)
        forward_ms = (time.perf_counter() - forward_start) * 1000.0
        forward_latencies_ms.append(forward_ms)
        policy_latencies_ms.append(state_latencies_ms[-1] + forward_ms)

        delta = clamp_delta(raw_delta.detach().cpu(), config.max_delta_ee)
        sim_start = time.perf_counter()
        env.step_delta_ee(delta, gripper=task_state.gripper_command)
        sim_latencies_ms.append((time.perf_counter() - sim_start) * 1000.0)

        distance = _ee_to_part_distance(env, task_state)
        success = success or distance <= config.target_radius
        records.append(
            {
                "step": step_idx,
                "distance": distance,
                "success": distance <= config.target_radius,
                "delta_ee": delta.tolist(),
                "part_world_position": task_state.part_world_position.tolist(),
                "part_local": task_state.part_local.tolist(),
                "ee_position": env.robot_observation().ee_position.tolist(),
                "object_center": task_state.object_center.tolist(),
                "object_yaw": task_state.object_yaw,
                "part_world_yaw": task_state.part_world_yaw,
                "layout": task_state.layout,
                "visual_feature_sum": diagnostics["visual_feature_sum"],
            }
        )
        if render and render_dir is not None and step_idx % 4 == 0:
            frame_path = render_dir / f"frame_{step_idx:03d}.png"
            _save_frame(env, frame_path)
            frames.append(str(frame_path))
        if success:
            break

    final_distance = records[-1]["distance"] if records else _ee_to_part_distance(env, task_state)
    return {
        "episode": episode_id,
        "success": bool(success),
        "final_distance": float(final_distance),
        "steps": len(records),
        "layout": task_state.layout,
        "object_center": task_state.object_center.tolist(),
        "object_yaw": task_state.object_yaw,
        "part_local": task_state.part_local.tolist(),
        "part_world_position": task_state.part_world_position.tolist(),
        "part_world_yaw": task_state.part_world_yaw,
        "camera_yaw": task_state.camera_yaw,
        "gripper_command": task_state.gripper_command,
        "mean_policy_latency_ms": _mean(policy_latencies_ms),
        "mean_visual_latency_ms": _mean(visual_latencies_ms),
        "mean_state_latency_ms": _mean(state_latencies_ms),
        "mean_forward_latency_ms": _mean(forward_latencies_ms),
        "mean_sim_step_latency_ms": _mean(sim_latencies_ms),
        "records": records,
        "render_frames": frames,
    }


@torch.no_grad()
def run_part_node_rollout_from_spec(
    model: nn.Module,
    variant: PartNodeVariant,
    env: MujocoManipulatorEnv,
    config: PartNodeDemoConfig,
    spec: PartEpisodeSpec,
    device: torch.device,
) -> dict[str, Any]:
    _reset_env_for_part_episode_spec(env, spec)
    task_state = spec.task_state
    initial_ee = env.robot_observation().ee_position.float()
    trajectory = [initial_ee.tolist()]
    distances = [float((initial_ee - task_state.part_world_position).norm().item())]
    records: list[dict[str, Any]] = []
    success = distances[-1] <= config.target_radius

    for step_idx in range(config.max_steps):
        if success:
            break
        feature_parts, _, _ = _build_part_feature_parts(env, task_state, config, measure_visual=False)
        observation = _part_observation_from_feature_parts(feature_parts, variant)
        observation = _observation_to(observation, device)
        raw_delta = model(observation).squeeze(0)
        if device.type != "cpu":
            _sync_if_needed(device)
        delta = clamp_delta(raw_delta.detach().cpu(), config.max_delta_ee)
        env.step_delta_ee(delta, gripper=task_state.gripper_command)
        ee_position = env.robot_observation().ee_position.float()
        distance = float((ee_position - task_state.part_world_position).norm().item())
        trajectory.append(ee_position.tolist())
        distances.append(distance)
        step_success = distance <= config.target_radius
        success = success or step_success
        records.append(
            {
                "step": step_idx,
                "distance": distance,
                "success": step_success,
                "delta_ee": delta.tolist(),
                "ee_position": ee_position.tolist(),
            }
        )

    return {
        "variant": variant,
        "success": bool(success),
        "final_distance": float(distances[-1]),
        "initial_distance": float(distances[0]),
        "max_distance_from_target": float(max(distances)),
        "min_distance_from_target": float(min(distances)),
        "steps": len(records),
        "trajectory_length": _trajectory_length(trajectory),
        "trajectory": trajectory,
        "records": records,
        "mean_action_l2_error": None,
    }


@torch.no_grad()
def evaluate_keypoint_models_paired(
    checkpoint_paths: dict[KeypointVariant, Path],
    output_dir: Path,
    eval_splits: tuple[KeypointEvalSplit, ...],
    episodes: int,
    max_steps: int,
    seed: int,
    device_preference: DevicePreference,
    bootstrap_samples: int,
    plot_worst: int,
) -> dict[str, Any]:
    device = select_device(device_preference)
    output_dir = ensure_dir(output_dir)
    models: dict[KeypointVariant, nn.Module] = {}
    checkpoints: dict[KeypointVariant, dict[str, Any]] = {}
    for variant, path in checkpoint_paths.items():
        models[variant], checkpoints[variant] = load_keypoint_policy(path, device)
    demo_config = _keypoint_demo_config_from_checkpoint(checkpoints["single_part_node"])
    demo_config.max_steps = max_steps
    envs = {
        variant: MujocoManipulatorEnv(_keypoint_env_config(demo_config), seed=seed + idx)
        for idx, variant in enumerate(models)
    }
    split_payloads: dict[str, Any] = {}
    for split_idx, split_name in enumerate(eval_splits):
        specs = sample_keypoint_episode_specs(demo_config, episodes, seed + split_idx * 1000, split_name)
        split_results = []
        for spec in specs:
            model_results = {
                variant: run_keypoint_rollout_from_spec(
                    model=model,
                    variant=variant,
                    env=envs[variant],
                    config=demo_config,
                    spec=spec,
                    device=device,
                    permutation=None,
                )
                for variant, model in models.items()
            }
            split_results.append(_keypoint_paired_episode_result(spec, model_results))
        split_dir = ensure_dir(output_dir / split_name)
        _write_json(split_dir / "episode_results.json", {"split": split_name, "episodes": split_results})
        split_summary = summarize_keypoint_eval(split_results, bootstrap_samples, seed + split_idx)
        _write_json(split_dir / "summary.json", split_summary)
        if plot_worst > 0:
            plots_dir = ensure_dir(split_dir / "plots")
            for case in keypoint_worst_cases(split_results, model_name="keypoint_graph")[:plot_worst]:
                plot_path = plots_dir / f"episode_{case['episode_id']:04d}_trajectory.png"
                plot_keypoint_episode(case["episode"], plot_path)
                case["plot_path"] = str(plot_path)
            _write_json(split_dir / "worst_cases.json", {"keypoint_graph": keypoint_worst_cases(split_results, "keypoint_graph")})
        split_payloads[split_name] = split_summary

    permutation = _fixed_keypoint_permutation(demo_config.keypoints)
    permutation_payload = {}
    for variant in ("keypoint_concat", "keypoint_graph"):
        specs = sample_keypoint_episode_specs(demo_config, episodes, seed + 9090, "shape_ood")
        normal_results = []
        permuted_results = []
        for spec in specs:
            normal = run_keypoint_rollout_from_spec(
                models[variant],
                variant,  # type: ignore[arg-type]
                envs[variant],  # type: ignore[index]
                demo_config,
                spec,
                device,
                permutation=None,
            )
            permuted = run_keypoint_rollout_from_spec(
                models[variant],
                variant,  # type: ignore[arg-type]
                envs[variant],  # type: ignore[index]
                demo_config,
                spec,
                device,
                permutation=permutation,
            )
            normal_results.append(normal)
            permuted_results.append(permuted)
        permutation_payload[variant] = {
            "permutation": permutation,
            "normal": summarize_keypoint_rollouts(normal_results),
            "permuted": summarize_keypoint_rollouts(permuted_results),
            "delta_success": _mean([1.0 if r["success"] else 0.0 for r in permuted_results])
            - _mean([1.0 if r["success"] else 0.0 for r in normal_results]),
            "delta_final_position_error": _mean([r["final_position_error"] for r in permuted_results])
            - _mean([r["final_position_error"] for r in normal_results]),
        }
    for env in envs.values():
        env.close()
    payload = {
        "device": str(device),
        "splits": split_payloads,
        "permutation": permutation_payload,
        "parameter_count": {
            variant: int(checkpoints[variant]["parameter_count"]) for variant in checkpoints
        },
    }
    _write_json(output_dir / "paired_eval_summary.json", payload)
    return payload


@torch.no_grad()
def run_keypoint_rollout_from_spec(
    model: nn.Module,
    variant: KeypointVariant,
    env: MujocoManipulatorEnv,
    config: KeypointDemoConfig,
    spec: KeypointEpisodeSpec,
    device: torch.device,
    permutation: list[int] | None = None,
) -> dict[str, Any]:
    _reset_env_for_keypoint_spec(env, spec)
    state = spec.task_state
    ee_yaw = float(state.ee_yaw)
    trajectory = [env.robot_observation().ee_position.float().tolist()]
    yaw_values = [ee_yaw]
    position_errors: list[float] = []
    yaw_errors: list[float] = []
    forward_latencies: list[float] = []
    policy_latencies: list[float] = []
    success = False
    for step_idx in range(config.max_steps):
        obs_start = time.perf_counter()
        features = _build_keypoint_feature_parts(env, state, ee_yaw)
        observation = _keypoint_observation_from_parts(features, variant, permutation=permutation)
        observation = _observation_to(observation, device)
        obs_ms = (time.perf_counter() - obs_start) * 1000.0
        forward_start = time.perf_counter()
        action = model(observation).squeeze(0)
        if device.type != "cpu":
            _sync_if_needed(device)
        forward_ms = (time.perf_counter() - forward_start) * 1000.0
        forward_latencies.append(forward_ms)
        policy_latencies.append(obs_ms + forward_ms)
        delta_xyz = clamp_delta(action[:3].detach().cpu(), config.max_delta_ee)
        delta_yaw = float(torch.clamp(action[3].detach().cpu(), -config.max_delta_yaw, config.max_delta_yaw).item())
        env.step_delta_ee(delta_xyz, gripper=state.gripper_command)
        ee_yaw = wrap_angle(ee_yaw + delta_yaw)
        ee_pos = env.robot_observation().ee_position.float()
        pos_error = float((ee_pos - state.target_position).norm().item())
        yaw_error = abs(wrap_angle(state.target_yaw - ee_yaw))
        position_errors.append(pos_error)
        yaw_errors.append(yaw_error)
        trajectory.append(ee_pos.tolist())
        yaw_values.append(ee_yaw)
        step_success = pos_error <= config.target_radius and yaw_error <= config.yaw_success_threshold
        success = success or step_success
        if step_success:
            break
    if not position_errors:
        ee_pos = env.robot_observation().ee_position.float()
        position_errors.append(float((ee_pos - state.target_position).norm().item()))
        yaw_errors.append(abs(wrap_angle(state.target_yaw - ee_yaw)))
    return {
        "variant": variant,
        "success": bool(success),
        "final_position_error": float(position_errors[-1]),
        "final_orientation_error": float(yaw_errors[-1]),
        "mean_position_error": _mean(position_errors),
        "mean_orientation_error": _mean(yaw_errors),
        "trajectory_length": _trajectory_length(trajectory),
        "steps": len(position_errors),
        "trajectory": trajectory,
        "yaw_values": yaw_values,
        "target_position": state.target_position.tolist(),
        "target_yaw": state.target_yaw,
        "mean_forward_latency_ms": _mean(forward_latencies),
        "mean_policy_latency_ms": _mean(policy_latencies),
    }


def load_representation_dataset(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_representation_policy(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_kind: RepresentationModelKind = checkpoint.get("model_kind", "mlp")
    if model_kind == "mpnn":
        graph_dims = checkpoint["graph_dims"]
        train_config = checkpoint.get("config", {})
        model = TinyRepresentationMPNN(
            object_feature_dim=int(graph_dims["object"]),
            ee_feature_dim=int(graph_dims["ee"]),
            edge_feature_dim=int(graph_dims["edge"]),
            hidden_dim=int(train_config.get("mpnn_hidden_dim", 64)),
            num_layers=int(train_config.get("mpnn_layers", 2)),
        ).to(device)
    else:
        model = RepresentationMLP(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["config"].get("hidden_dim", 128)),
        ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def load_part_node_policy(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    train_config = checkpoint.get("config", {})
    graph_dims = checkpoint["graph_dims"]
    model = PartNodeMPNN(
        object_feature_dim=int(graph_dims["object"]),
        ee_feature_dim=int(graph_dims["ee"]),
        edge_feature_dim=int(graph_dims["edge"]),
        part_feature_dim=int(graph_dims["part"]),
        hidden_dim=int(train_config.get("hidden_dim", 64)),
        num_layers=int(train_config.get("message_passing_layers", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def load_keypoint_policy(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    variant: KeypointVariant = checkpoint["variant"]
    train_config = checkpoint.get("config", {})
    dims = checkpoint["model_dims"]
    if variant == "keypoint_concat":
        model: nn.Module = KeypointConcatMLP(
            input_dim=int(dims["input"]),
            hidden_dim=int(train_config.get("mlp_hidden_dim", 128)),
            output_dim=4,
        ).to(device)
    else:
        model = KeypointMPNN(
            object_feature_dim=int(dims["object"]),
            ee_feature_dim=int(dims["ee"]),
            edge_feature_dim=int(dims["edge"]),
            part_feature_dim=int(dims.get("part", 0)),
            keypoint_feature_dim=int(dims.get("keypoint", 0)),
            hidden_dim=int(train_config.get("hidden_dim", 64)),
            num_layers=int(train_config.get("message_passing_layers", 2)),
            output_dim=4,
            readout=train_config.get("readout", "global"),
        ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate_representation_offline(
    model: nn.Module,
    dataset: dict[str, Any],
    split_name: str,
    variant: RepresentationVariant,
    device: torch.device,
    model_kind: RepresentationModelKind = "mlp",
) -> dict[str, float]:
    observations, actions = _split_observations(dataset, split_name, variant, model_kind)
    return _evaluate_observation_split(model, _observation_to(observations, device), actions.to(device))


@torch.no_grad()
def evaluate_part_node_offline(
    model: nn.Module,
    dataset: dict[str, Any],
    split_name: str,
    variant: PartNodeVariant,
    device: torch.device,
) -> dict[str, float]:
    observations, actions = _part_split_observations(dataset, split_name, variant)
    return _evaluate_observation_split(model, _observation_to(observations, device), actions.to(device))


@torch.no_grad()
def evaluate_keypoint_offline(
    model: nn.Module,
    dataset: dict[str, Any],
    split_name: str,
    variant: KeypointVariant,
    device: torch.device,
) -> dict[str, float]:
    observations, actions = _keypoint_split_observations(dataset, split_name, variant)
    return _evaluate_keypoint_observation_split(model, _observation_to(observations, device), actions.to(device))


def summarize_rollouts(results: list[dict[str, Any]]) -> dict[str, float]:
    action_errors = [
        float(record["action_l2_error"])
        for item in results
        for record in item.get("records", [])
        if "action_l2_error" in record
    ]
    return {
        "success_rate": _mean([1.0 if item["success"] else 0.0 for item in results]),
        "mean_final_distance": _mean([item["final_distance"] for item in results]),
        "mean_steps": _mean([item["steps"] for item in results]),
        "mean_policy_latency_ms": _mean([item["mean_policy_latency_ms"] for item in results]),
        "mean_visual_latency_ms": _mean([item["mean_visual_latency_ms"] for item in results]),
        "mean_state_latency_ms": _mean([item["mean_state_latency_ms"] for item in results]),
        "mean_forward_latency_ms": _mean([item["mean_forward_latency_ms"] for item in results]),
        "mean_sim_step_latency_ms": _mean([item["mean_sim_step_latency_ms"] for item in results]),
        "mean_action_l2_error": _mean(action_errors),
    }


def _collect_expert_episode(
    env: MujocoManipulatorEnv,
    config: RepresentationDemoConfig,
    task_state: HandleTaskState,
    episode_id: int,
    render: bool = False,
    render_dir: Path | None = None,
) -> dict[str, Any]:
    _reset_env_for_task(env, task_state)
    if render:
        render_dir = ensure_dir(render_dir or Path("artifacts/representation_feasibility/demos/frames"))
    pose_features: list[torch.Tensor] = []
    visual_features: list[torch.Tensor] = []
    ee_geometry_features: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    distances: list[float] = []
    frames: list[str] = []
    success = False

    for step_idx in range(config.max_steps):
        feature_parts, _, _ = _build_feature_parts(env, task_state, config, measure_visual=False)
        action = _expert_delta(env, task_state, config.max_delta_ee)
        pose_features.append(feature_parts["pose"])
        visual_features.append(feature_parts["visual"])
        ee_geometry_features.append(feature_parts["ee_geometry"])
        actions.append(action)
        env.step_delta_ee(action, gripper=task_state.gripper_command)
        distance = _active_fingertip_distance(env, task_state)
        distances.append(distance)
        if render and render_dir is not None and step_idx % 4 == 0:
            frame_path = render_dir / f"frame_{step_idx:03d}.png"
            _save_frame(env, frame_path)
            frames.append(str(frame_path))
        if distance <= config.target_radius:
            success = True
            break

    return {
        "episode_id": episode_id,
        "pose_features": torch.stack(pose_features, dim=0) if pose_features else torch.empty(0, 0),
        "visual_features": torch.stack(visual_features, dim=0) if visual_features else torch.empty(0, 0),
        "ee_geometry_features": torch.stack(ee_geometry_features, dim=0) if ee_geometry_features else torch.empty(0, 0),
        "actions": torch.stack(actions, dim=0) if actions else torch.empty(0, 3),
        "distances": torch.tensor(distances, dtype=torch.float32),
        "steps": len(actions),
        "expert_success": success,
        "object_center": task_state.object_center.tolist(),
        "object_yaw": task_state.object_yaw,
        "handle_side": task_state.handle_side,
        "handle_offset": task_state.handle_offset,
        "gripper_command": task_state.gripper_command,
        "camera_yaw": task_state.camera_yaw,
        "render_frames": frames,
    }


def _collect_part_expert_episode(
    env: MujocoManipulatorEnv,
    config: PartNodeDemoConfig,
    task_state: PartTaskState,
    episode_id: int,
    render: bool = False,
    render_dir: Path | None = None,
) -> dict[str, Any]:
    _reset_env_for_part_task(env, task_state)
    if render:
        render_dir = ensure_dir(render_dir or Path("artifacts/part_node_feasibility/demos/frames"))
    object_features: list[torch.Tensor] = []
    ee_features: list[torch.Tensor] = []
    part_features: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    distances: list[float] = []
    frames: list[str] = []
    success = False

    for step_idx in range(config.max_steps):
        feature_parts, _, _ = _build_part_feature_parts(env, task_state, config, measure_visual=False)
        action = _part_expert_delta(env, task_state, config.max_delta_ee)
        object_features.append(feature_parts["object"])
        ee_features.append(feature_parts["ee"])
        part_features.append(feature_parts["part"])
        actions.append(action)
        env.step_delta_ee(action, gripper=task_state.gripper_command)
        distance = _ee_to_part_distance(env, task_state)
        distances.append(distance)
        if render and render_dir is not None and step_idx % 4 == 0:
            frame_path = render_dir / f"frame_{step_idx:03d}.png"
            _save_frame(env, frame_path)
            frames.append(str(frame_path))
        if distance <= config.target_radius:
            success = True
            break

    return {
        "episode_id": episode_id,
        "layout": task_state.layout,
        "object_features": torch.stack(object_features, dim=0) if object_features else torch.empty(0, 0),
        "ee_features": torch.stack(ee_features, dim=0) if ee_features else torch.empty(0, 0),
        "part_features": torch.stack(part_features, dim=0) if part_features else torch.empty(0, 0),
        "actions": torch.stack(actions, dim=0) if actions else torch.empty(0, 3),
        "distances": torch.tensor(distances, dtype=torch.float32),
        "steps": len(actions),
        "expert_success": success,
        "object_center": task_state.object_center.tolist(),
        "object_yaw": task_state.object_yaw,
        "part_local": task_state.part_local.tolist(),
        "part_world_position": task_state.part_world_position.tolist(),
        "part_local_yaw": task_state.part_local_yaw,
        "part_world_yaw": task_state.part_world_yaw,
        "part_extent": list(task_state.part_extent),
        "gripper_command": task_state.gripper_command,
        "camera_yaw": task_state.camera_yaw,
        "render_frames": frames,
    }


def _collect_keypoint_expert_episode(
    env: MujocoManipulatorEnv,
    config: KeypointDemoConfig,
    task_state: KeypointTaskState,
    episode_id: int,
) -> dict[str, Any]:
    _reset_env_for_keypoint_state(env, task_state)
    ee_yaw = float(task_state.ee_yaw)
    object_features: list[torch.Tensor] = []
    ee_features: list[torch.Tensor] = []
    part_features: list[torch.Tensor] = []
    keypoint_features: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    position_errors: list[float] = []
    yaw_errors: list[float] = []
    success = False
    for _ in range(config.max_steps):
        parts = _build_keypoint_feature_parts(env, task_state, ee_yaw)
        action = _keypoint_expert_action(env, task_state, ee_yaw, config)
        object_features.append(parts["object"])
        ee_features.append(parts["ee"])
        part_features.append(parts["part"])
        keypoint_features.append(parts["keypoints"])
        actions.append(action)
        env.step_delta_ee(action[:3], gripper=task_state.gripper_command)
        ee_yaw = wrap_angle(ee_yaw + float(action[3].item()))
        ee_pos = env.robot_observation().ee_position.float()
        pos_error = float((ee_pos - task_state.target_position).norm().item())
        yaw_error = abs(wrap_angle(task_state.target_yaw - ee_yaw))
        position_errors.append(pos_error)
        yaw_errors.append(yaw_error)
        if pos_error <= config.target_radius and yaw_error <= config.yaw_success_threshold:
            success = True
            break
    return {
        "episode_id": episode_id,
        "split": task_state.split,
        "object_features": torch.stack(object_features, dim=0) if object_features else torch.empty(0, 0),
        "ee_features": torch.stack(ee_features, dim=0) if ee_features else torch.empty(0, 0),
        "part_features": torch.stack(part_features, dim=0) if part_features else torch.empty(0, 0),
        "keypoint_features": torch.stack(keypoint_features, dim=0) if keypoint_features else torch.empty(0, 0, 0),
        "actions": torch.stack(actions, dim=0) if actions else torch.empty(0, 4),
        "position_errors": torch.tensor(position_errors, dtype=torch.float32),
        "yaw_errors": torch.tensor(yaw_errors, dtype=torch.float32),
        "steps": len(actions),
        "expert_success": success,
        "object_center": task_state.object_center.tolist(),
        "object_yaw": task_state.object_yaw,
        "part_offset_local": task_state.part_offset_local.tolist(),
        "part_yaw_local": task_state.part_yaw_local,
        "length": task_state.length,
        "curvature": task_state.curvature,
        "spacing_jitter": task_state.spacing_jitter,
        "keypoints_local": task_state.keypoints_local.tolist(),
        "keypoints_world": task_state.keypoints_world.tolist(),
        "target_position": task_state.target_position.tolist(),
        "target_yaw": task_state.target_yaw,
        "principal_yaw": task_state.principal_yaw,
        "global_local_yaw_abs_diff": abs(wrap_angle(task_state.target_yaw - task_state.principal_yaw)),
        "spacing_asymmetry": keypoint_spacing_asymmetry(task_state.keypoints_local),
        "ee_yaw_initial": task_state.ee_yaw,
        "gripper_command": task_state.gripper_command,
    }


def _build_feature_parts(
    env: MujocoManipulatorEnv,
    task_state: HandleTaskState,
    config: RepresentationDemoConfig,
    measure_visual: bool,
    include_visual: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, float]]:
    robot = env.robot_observation()
    ee = robot.ee_position.float()
    object_center = task_state.object_center.float()
    relative_center = object_center - ee
    pose = torch.cat(
        [
            object_center,
            torch.tensor([math.sin(task_state.object_yaw), math.cos(task_state.object_yaw)], dtype=torch.float32),
            torch.tensor(list(task_state.object_half_extent), dtype=torch.float32),
            ee,
            relative_center,
            relative_center.norm().reshape(1),
        ],
        dim=0,
    ).float()

    visual_start = time.perf_counter()
    if include_visual:
        crop = render_handle_crop(
            handle_side=task_state.handle_side,
            handle_offset=task_state.handle_offset,
            object_half_extent=task_state.object_half_extent,
            object_yaw=task_state.object_yaw,
            camera_yaw=task_state.camera_yaw,
            crop_size=config.crop_size,
        )
        visual = frozen_visual_feature(crop, grid=config.visual_grid)
        visual_ms = (time.perf_counter() - visual_start) * 1000.0 if measure_visual else 0.0
    else:
        visual = torch.empty(0, dtype=torch.float32)
        visual_ms = 0.0

    thumb, finger = _fingertip_positions(env)
    span = thumb - finger
    ee_geometry = torch.cat(
        [
            robot.ee_orientation.float(),
            robot.gripper_state.float().reshape(1),
            thumb,
            finger,
            thumb - object_center,
            finger - object_center,
            span,
            span.norm().reshape(1),
        ],
        dim=0,
    ).float()
    feature_parts = {"pose": pose, "visual": visual, "ee_geometry": ee_geometry}
    diagnostics = {"visual_feature_sum": float(visual.sum().item()) if include_visual else 0.0}
    timing = {"visual_ms": visual_ms}
    return feature_parts, diagnostics, timing


def _build_part_feature_parts(
    env: MujocoManipulatorEnv,
    task_state: PartTaskState,
    config: PartNodeDemoConfig,
    measure_visual: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, float]]:
    robot = env.robot_observation()
    ee = robot.ee_position.float()
    object_center = task_state.object_center.float()
    part_world = task_state.part_world_position.float()
    part_local = task_state.part_local.float()

    visual_start = time.perf_counter()
    crop = render_object_body_crop(
        object_half_extent=task_state.object_half_extent,
        object_yaw=task_state.object_yaw,
        camera_yaw=task_state.camera_yaw,
        crop_size=config.crop_size,
    )
    visual = frozen_visual_feature(crop, grid=config.visual_grid)
    visual_ms = (time.perf_counter() - visual_start) * 1000.0 if measure_visual else 0.0

    object_features = torch.cat(
        [
            object_center,
            torch.tensor([math.sin(task_state.object_yaw), math.cos(task_state.object_yaw)], dtype=torch.float32),
            torch.tensor(list(task_state.object_half_extent), dtype=torch.float32),
            visual,
        ],
        dim=0,
    ).float()
    ee_features = torch.cat(
        [
            ee,
            robot.ee_orientation.float(),
            robot.gripper_state.float().reshape(1),
        ],
        dim=0,
    ).float()
    part_features = torch.cat(
        [
            part_world,
            part_local,
            torch.tensor([math.sin(task_state.part_world_yaw), math.cos(task_state.part_world_yaw)], dtype=torch.float32),
            torch.tensor(list(task_state.part_extent), dtype=torch.float32),
        ],
        dim=0,
    ).float()
    feature_parts = {"object": object_features, "ee": ee_features, "part": part_features}
    diagnostics = {"visual_feature_sum": float(visual.sum().item())}
    timing = {"visual_ms": visual_ms}
    return feature_parts, diagnostics, timing


def _build_keypoint_feature_parts(
    env: MujocoManipulatorEnv,
    task_state: KeypointTaskState,
    ee_yaw: float,
) -> dict[str, torch.Tensor]:
    robot = env.robot_observation()
    ee = robot.ee_position.float()
    object_center = task_state.object_center.float()
    keypoints_world = task_state.keypoints_world.float()
    keypoints_local = task_state.keypoints_local.float()
    centroid_world = keypoints_world.mean(dim=0)
    centroid_local = keypoints_local.mean(dim=0)
    local_extent = keypoints_local.max(dim=0).values - keypoints_local.min(dim=0).values
    principal_yaw = task_state.principal_yaw
    object_features = torch.cat(
        [
            object_center,
            torch.tensor([math.sin(task_state.object_yaw), math.cos(task_state.object_yaw)], dtype=torch.float32),
            torch.tensor(list(task_state.object_half_extent), dtype=torch.float32),
        ],
        dim=0,
    ).float()
    ee_features = torch.cat(
        [
            ee,
            torch.tensor([math.sin(ee_yaw), math.cos(ee_yaw)], dtype=torch.float32),
        ],
        dim=0,
    ).float()
    part_features = torch.cat(
        [
            centroid_world,
            centroid_local,
            torch.tensor([math.sin(principal_yaw), math.cos(principal_yaw)], dtype=torch.float32),
            torch.tensor([task_state.length, float(local_extent[0].item()), float(local_extent[2].item())], dtype=torch.float32),
        ],
        dim=0,
    ).float()
    keypoint_features = torch.cat([keypoints_world, keypoints_local], dim=-1).float()
    return {"object": object_features, "ee": ee_features, "part": part_features, "keypoints": keypoint_features}


def render_handle_crop(
    handle_side: int,
    handle_offset: float,
    object_half_extent: tuple[float, float],
    object_yaw: float,
    camera_yaw: float,
    crop_size: int,
) -> torch.Tensor:
    """Render a tiny RGB crop of a symmetric body with one visible handle mark."""

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, crop_size),
        torch.linspace(-1.0, 1.0, crop_size),
        indexing="ij",
    )
    view_yaw = object_yaw - camera_yaw
    c = math.cos(view_yaw)
    s = math.sin(view_yaw)
    local_x = c * xx + s * yy
    local_z = -s * xx + c * yy
    body_x = float(object_half_extent[0] / max(object_half_extent[0], object_half_extent[1]))
    body_z = float(object_half_extent[1] / max(object_half_extent[0], object_half_extent[1]))
    body = (local_x.abs() <= body_x) & (local_z.abs() <= body_z)

    handle_scale = max(float(object_half_extent[0]), 1e-6)
    handle_center_x = handle_side * float(handle_offset) / handle_scale
    handle = ((local_x - handle_center_x).abs() <= 0.20) & (local_z.abs() <= 0.16)
    crop = torch.zeros(crop_size, crop_size, 3, dtype=torch.float32)
    crop[..., 0] = torch.where(body, torch.tensor(0.22), torch.tensor(0.03))
    crop[..., 1] = torch.where(body, torch.tensor(0.36), torch.tensor(0.03))
    crop[..., 2] = torch.where(body, torch.tensor(0.58), torch.tensor(0.04))
    crop[..., 0] = torch.where(handle, torch.tensor(0.95), crop[..., 0])
    crop[..., 1] = torch.where(handle, torch.tensor(0.85), crop[..., 1])
    crop[..., 2] = torch.where(handle, torch.tensor(0.12), crop[..., 2])
    return crop


def render_object_body_crop(
    object_half_extent: tuple[float, float],
    object_yaw: float,
    camera_yaw: float,
    crop_size: int,
) -> torch.Tensor:
    """Render a compact object-only crop that intentionally excludes oracle part location."""

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, crop_size),
        torch.linspace(-1.0, 1.0, crop_size),
        indexing="ij",
    )
    view_yaw = object_yaw - camera_yaw
    c = math.cos(view_yaw)
    s = math.sin(view_yaw)
    local_x = c * xx + s * yy
    local_z = -s * xx + c * yy
    body_x = float(object_half_extent[0] / max(object_half_extent[0], object_half_extent[1]))
    body_z = float(object_half_extent[1] / max(object_half_extent[0], object_half_extent[1]))
    body = (local_x.abs() <= body_x) & (local_z.abs() <= body_z)
    edge = (local_x.abs() <= body_x + 0.04) & (local_z.abs() <= body_z + 0.04)
    crop = torch.zeros(crop_size, crop_size, 3, dtype=torch.float32)
    crop[..., 0] = torch.where(edge, torch.tensor(0.12), torch.tensor(0.03))
    crop[..., 1] = torch.where(edge, torch.tensor(0.18), torch.tensor(0.03))
    crop[..., 2] = torch.where(edge, torch.tensor(0.24), torch.tensor(0.04))
    crop[..., 0] = torch.where(body, torch.tensor(0.28), crop[..., 0])
    crop[..., 1] = torch.where(body, torch.tensor(0.43), crop[..., 1])
    crop[..., 2] = torch.where(body, torch.tensor(0.62), crop[..., 2])
    return crop


def frozen_visual_feature(crop: torch.Tensor, grid: int = 8) -> torch.Tensor:
    """Frozen, deterministic compact visual encoder.

    The pooled crop keeps coarse appearance, while the simple moments expose
    where the bright handle-like region sits without training a vision model.
    """

    if crop.ndim != 3 or crop.shape[-1] != 3:
        raise ValueError(f"Expected RGB crop [H, W, 3], got {tuple(crop.shape)}")
    grayscale = (0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]).unsqueeze(0).unsqueeze(0)
    pooled = torch.nn.functional.adaptive_avg_pool2d(grayscale, (grid, grid))
    gray = grayscale.reshape(crop.shape[0], crop.shape[1])
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, crop.shape[0]),
        torch.linspace(-1.0, 1.0, crop.shape[1]),
        indexing="ij",
    )
    mass = gray.sum().clamp_min(1e-6)
    centroid_x = (gray * xx).sum() / mass
    centroid_y = (gray * yy).sum() / mass
    return torch.cat([pooled.reshape(-1), torch.stack([centroid_x, centroid_y, mass / gray.numel()])], dim=0).float()


def _variant_features_from_parts(
    feature_parts: dict[str, torch.Tensor],
    variant: RepresentationVariant,
) -> torch.Tensor:
    if variant == "pose":
        return feature_parts["pose"].float()
    if variant == "pose_visual":
        return torch.cat([feature_parts["pose"], feature_parts["visual"]], dim=0).float()
    if variant == "pose_visual_ee":
        return torch.cat([feature_parts["pose"], feature_parts["visual"], feature_parts["ee_geometry"]], dim=0).float()
    raise ValueError(f"Unknown representation variant: {variant}")


def _variant_episode_features(episode: dict[str, Any], variant: RepresentationVariant) -> torch.Tensor:
    if variant == "pose":
        return episode["pose_features"].float()
    if variant == "pose_visual":
        return torch.cat([episode["pose_features"], episode["visual_features"]], dim=-1).float()
    if variant == "pose_visual_ee":
        return torch.cat(
            [episode["pose_features"], episode["visual_features"], episode["ee_geometry_features"]],
            dim=-1,
        ).float()
    raise ValueError(f"Unknown representation variant: {variant}")


def graph_batch_from_feature_tensors(
    pose_features: torch.Tensor,
    visual_features: torch.Tensor,
    ee_geometry_features: torch.Tensor,
) -> RepresentationGraphBatch:
    """Map fixed representation C tensors into a two-node directed graph.

    Object node receives object pose/orientation/size plus visual features.
    EE node receives EE position plus the explicit EE/fingertip geometry.
    Directed edge features are deterministic transforms of the same pose inputs.
    """

    if pose_features.ndim == 1:
        pose_features = pose_features.unsqueeze(0)
    if visual_features.ndim == 1:
        visual_features = visual_features.unsqueeze(0)
    if ee_geometry_features.ndim == 1:
        ee_geometry_features = ee_geometry_features.unsqueeze(0)
    object_features = torch.cat([pose_features[:, 0:7], visual_features], dim=-1).float()
    ee_features = torch.cat([pose_features[:, 7:10], ee_geometry_features], dim=-1).float()
    ee_to_object = torch.cat([pose_features[:, 10:13], pose_features[:, 13:14]], dim=-1)
    object_to_ee = torch.cat([-pose_features[:, 10:13], pose_features[:, 13:14]], dim=-1)
    edge_features = torch.stack([ee_to_object, object_to_ee], dim=1).float()
    return RepresentationGraphBatch(
        object_features=object_features,
        ee_features=ee_features,
        edge_features=edge_features,
    )


def part_graph_batch_from_feature_tensors(
    object_features: torch.Tensor,
    ee_features: torch.Tensor,
    part_features: torch.Tensor,
    variant: PartNodeVariant,
) -> PartGraphBatch:
    """Map identical oracle-part state into the requested graph encoding."""

    if object_features.ndim == 1:
        object_features = object_features.unsqueeze(0)
    if ee_features.ndim == 1:
        ee_features = ee_features.unsqueeze(0)
    if part_features.ndim == 1:
        part_features = part_features.unsqueeze(0)
    object_center = object_features[:, 0:3]
    ee_position = ee_features[:, 0:3]
    part_world = part_features[:, 0:3]
    part_local = part_features[:, 3:6]
    object_delta = object_center - ee_position
    part_delta = part_world - ee_position
    object_edge = torch.cat([object_delta, object_delta.norm(dim=-1, keepdim=True)], dim=-1)
    object_edge_reverse = torch.cat([-object_delta, object_delta.norm(dim=-1, keepdim=True)], dim=-1)

    if variant == "object_only":
        edge_index = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
        return PartGraphBatch(
            object_features=object_features.float(),
            ee_features=ee_features.float(),
            edge_features=torch.stack([object_edge, object_edge_reverse], dim=1).float(),
            edge_index=edge_index,
        )
    if variant == "part_concat":
        edge_index = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
        return PartGraphBatch(
            object_features=torch.cat([object_features, part_features], dim=-1).float(),
            ee_features=ee_features.float(),
            edge_features=torch.stack([object_edge, object_edge_reverse], dim=1).float(),
            edge_index=edge_index,
        )
    if variant == "part_node":
        part_edge = torch.cat([part_delta, part_delta.norm(dim=-1, keepdim=True)], dim=-1)
        part_edge_reverse = torch.cat([-part_delta, part_delta.norm(dim=-1, keepdim=True)], dim=-1)
        object_to_part = torch.cat([part_local, part_local.norm(dim=-1, keepdim=True)], dim=-1)
        part_to_object = torch.cat([-part_local, part_local.norm(dim=-1, keepdim=True)], dim=-1)
        edge_index = torch.tensor(
            [
                [1, 0, 1, 2, 0, 2],
                [0, 1, 2, 1, 2, 0],
            ],
            dtype=torch.long,
        )
        edge_features = torch.stack(
            [
                object_edge,
                object_edge_reverse,
                part_edge,
                part_edge_reverse,
                object_to_part,
                part_to_object,
            ],
            dim=1,
        ).float()
        return PartGraphBatch(
            object_features=object_features.float(),
            ee_features=ee_features.float(),
            part_features=part_features.float(),
            edge_features=edge_features,
            edge_index=edge_index,
        )
    raise ValueError(f"Unknown part-node variant: {variant}")


def keypoint_observation_from_feature_tensors(
    object_features: torch.Tensor,
    ee_features: torch.Tensor,
    part_features: torch.Tensor,
    keypoint_features: torch.Tensor,
    variant: KeypointVariant,
    permutation: list[int] | None = None,
) -> torch.Tensor | KeypointGraphBatch:
    if object_features.ndim == 1:
        object_features = object_features.unsqueeze(0)
    if ee_features.ndim == 1:
        ee_features = ee_features.unsqueeze(0)
    if part_features.ndim == 1:
        part_features = part_features.unsqueeze(0)
    if keypoint_features.ndim == 2:
        keypoint_features = keypoint_features.unsqueeze(0)
    if permutation is not None:
        keypoint_features = keypoint_features[:, permutation]
    object_center = object_features[:, 0:3]
    ee_position = ee_features[:, 0:3]
    if variant == "keypoint_concat":
        return torch.cat(
            [object_features, ee_features, keypoint_features.reshape(keypoint_features.shape[0], -1)],
            dim=-1,
        ).float()
    if variant == "single_part_node":
        part_center = part_features[:, 0:3]
        ee_object = _edge_delta(ee_position, object_center)
        object_ee = _edge_delta(object_center, ee_position)
        ee_part = _edge_delta(ee_position, part_center)
        part_ee = _edge_delta(part_center, ee_position)
        object_part = _edge_delta(object_center, part_center)
        part_object = _edge_delta(part_center, object_center)
        edge_index = torch.tensor([[1, 0, 1, 2, 0, 2], [0, 1, 2, 1, 2, 0]], dtype=torch.long)
        edge_features = torch.stack([ee_object, object_ee, ee_part, part_ee, object_part, part_object], dim=1)
        return KeypointGraphBatch(
            object_features=object_features.float(),
            ee_features=ee_features.float(),
            part_features=part_features.float(),
            edge_features=edge_features.float(),
            edge_index=edge_index,
        )
    if variant in {"keypoint_graph", "keypoint_graph_arc"}:
        if variant == "keypoint_graph_arc":
            k = keypoint_features.shape[1]
            arc = torch.linspace(-1.0, 1.0, k, dtype=keypoint_features.dtype, device=keypoint_features.device)
            if permutation is not None:
                arc = arc[torch.as_tensor(permutation, dtype=torch.long, device=arc.device)]
            keypoint_features = torch.cat(
                [keypoint_features, arc.reshape(1, k, 1).expand(keypoint_features.shape[0], -1, -1)], dim=-1
            )
        key_world = keypoint_features[:, :, 0:3]
        edge_src: list[int] = []
        edge_dst: list[int] = []
        edge_values: list[torch.Tensor] = []

        def add_edge(src_idx: int, dst_idx: int, src_pos: torch.Tensor, dst_pos: torch.Tensor) -> None:
            edge_src.append(src_idx)
            edge_dst.append(dst_idx)
            edge_values.append(_edge_delta(src_pos, dst_pos))

        add_edge(1, 0, ee_position, object_center)
        add_edge(0, 1, object_center, ee_position)
        k = key_world.shape[1]
        inv_perm = {physical: storage for storage, physical in enumerate(permutation or list(range(k)))}
        for storage_idx in range(k):
            node_idx = 2 + storage_idx
            kp_pos = key_world[:, storage_idx]
            add_edge(1, node_idx, ee_position, kp_pos)
            add_edge(node_idx, 1, kp_pos, ee_position)
            add_edge(0, node_idx, object_center, kp_pos)
            add_edge(node_idx, 0, kp_pos, object_center)
        for physical_idx in range(k - 1):
            left = inv_perm[physical_idx]
            right = inv_perm[physical_idx + 1]
            left_pos = key_world[:, left]
            right_pos = key_world[:, right]
            add_edge(2 + left, 2 + right, left_pos, right_pos)
            add_edge(2 + right, 2 + left, right_pos, left_pos)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_features = torch.stack(edge_values, dim=1)
        return KeypointGraphBatch(
            object_features=object_features.float(),
            ee_features=ee_features.float(),
            keypoint_features=keypoint_features.float(),
            edge_features=edge_features.float(),
            edge_index=edge_index,
        )
    raise ValueError(f"Unknown keypoint variant: {variant}")


def _edge_delta(src_pos: torch.Tensor, dst_pos: torch.Tensor) -> torch.Tensor:
    delta = dst_pos - src_pos
    return torch.cat([delta, delta.norm(dim=-1, keepdim=True)], dim=-1)


def _split_tensors(
    dataset: dict[str, Any],
    split_name: str,
    variant: RepresentationVariant,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = [_variant_episode_features(dataset["episodes"][idx], variant) for idx in dataset["split"][split_name]]
    actions = [dataset["episodes"][idx]["actions"].float() for idx in dataset["split"][split_name]]
    if not features:
        raise ValueError(f"Split {split_name!r} is empty.")
    return torch.cat(features, dim=0), torch.cat(actions, dim=0)


def _split_observations(
    dataset: dict[str, Any],
    split_name: str,
    variant: RepresentationVariant,
    model_kind: RepresentationModelKind,
) -> tuple[torch.Tensor | RepresentationGraphBatch, torch.Tensor]:
    actions = [dataset["episodes"][idx]["actions"].float() for idx in dataset["split"][split_name]]
    if not actions:
        raise ValueError(f"Split {split_name!r} is empty.")
    if model_kind == "mpnn":
        if variant != "pose_visual_ee":
            raise ValueError("MPNN observations require fixed representation C: pose_visual_ee.")
        pose = torch.cat([dataset["episodes"][idx]["pose_features"].float() for idx in dataset["split"][split_name]], dim=0)
        visual = torch.cat(
            [dataset["episodes"][idx]["visual_features"].float() for idx in dataset["split"][split_name]],
            dim=0,
        )
        ee_geometry = torch.cat(
            [dataset["episodes"][idx]["ee_geometry_features"].float() for idx in dataset["split"][split_name]],
            dim=0,
        )
        return graph_batch_from_feature_tensors(pose, visual, ee_geometry), torch.cat(actions, dim=0)
    return _split_tensors(dataset, split_name, variant)[0], torch.cat(actions, dim=0)


def _observation_from_feature_parts(
    feature_parts: dict[str, torch.Tensor],
    variant: RepresentationVariant,
    model_kind: RepresentationModelKind,
) -> torch.Tensor | RepresentationGraphBatch:
    if model_kind == "mpnn":
        if variant != "pose_visual_ee":
            raise ValueError("MPNN rollout requires fixed representation C: pose_visual_ee.")
        return graph_batch_from_feature_tensors(
            feature_parts["pose"],
            feature_parts["visual"],
            feature_parts["ee_geometry"],
        )
    return _variant_features_from_parts(feature_parts, variant)


def _part_split_observations(
    dataset: dict[str, Any],
    split_name: str,
    variant: PartNodeVariant,
) -> tuple[PartGraphBatch, torch.Tensor]:
    episodes = dataset["ood_episodes"] if split_name == "ood" else [dataset["episodes"][idx] for idx in dataset["split"][split_name]]
    if not episodes:
        raise ValueError(f"Split {split_name!r} is empty.")
    object_features = torch.cat([episode["object_features"].float() for episode in episodes], dim=0)
    ee_features = torch.cat([episode["ee_features"].float() for episode in episodes], dim=0)
    part_features = torch.cat([episode["part_features"].float() for episode in episodes], dim=0)
    actions = torch.cat([episode["actions"].float() for episode in episodes], dim=0)
    return part_graph_batch_from_feature_tensors(object_features, ee_features, part_features, variant), actions


def _part_observation_from_feature_parts(
    feature_parts: dict[str, torch.Tensor],
    variant: PartNodeVariant,
) -> PartGraphBatch:
    return part_graph_batch_from_feature_tensors(
        feature_parts["object"],
        feature_parts["ee"],
        feature_parts["part"],
        variant,
    )


def _keypoint_split_observations(
    dataset: dict[str, Any],
    split_name: str,
    variant: KeypointVariant,
) -> tuple[torch.Tensor | KeypointGraphBatch, torch.Tensor]:
    if split_name in {"length_ood", "orientation_ood", "shape_ood"}:
        episodes = dataset["ood_episodes"][split_name]
    else:
        episodes = [dataset["episodes"][idx] for idx in dataset["split"][split_name]]
    if not episodes:
        raise ValueError(f"Split {split_name!r} is empty.")
    object_features = torch.cat([episode["object_features"].float() for episode in episodes], dim=0)
    ee_features = torch.cat([episode["ee_features"].float() for episode in episodes], dim=0)
    part_features = torch.cat([episode["part_features"].float() for episode in episodes], dim=0)
    keypoint_features = torch.cat([episode["keypoint_features"].float() for episode in episodes], dim=0)
    actions = torch.cat([episode["actions"].float() for episode in episodes], dim=0)
    observation = keypoint_observation_from_feature_tensors(
        object_features,
        ee_features,
        part_features,
        keypoint_features,
        variant,
    )
    return observation, actions


def _keypoint_observation_from_parts(
    feature_parts: dict[str, torch.Tensor],
    variant: KeypointVariant,
    permutation: list[int] | None = None,
) -> torch.Tensor | KeypointGraphBatch:
    return keypoint_observation_from_feature_tensors(
        feature_parts["object"],
        feature_parts["ee"],
        feature_parts["part"],
        feature_parts["keypoints"],
        variant,
        permutation=permutation,
    )


def _observation_to(
    observation: torch.Tensor | RepresentationGraphBatch | PartGraphBatch | KeypointGraphBatch,
    device: torch.device | str,
) -> torch.Tensor | RepresentationGraphBatch | PartGraphBatch | KeypointGraphBatch:
    if isinstance(observation, RepresentationGraphBatch | PartGraphBatch | KeypointGraphBatch):
        return observation.to(device)
    return observation.to(device)


def _observation_index_select(
    observation: torch.Tensor | RepresentationGraphBatch | PartGraphBatch | KeypointGraphBatch,
    indices: torch.Tensor,
) -> torch.Tensor | RepresentationGraphBatch | PartGraphBatch | KeypointGraphBatch:
    if isinstance(observation, RepresentationGraphBatch | PartGraphBatch | KeypointGraphBatch):
        return observation.index_select(indices)
    return observation.index_select(0, indices)


def _observation_num_samples(observation: torch.Tensor | RepresentationGraphBatch | PartGraphBatch | KeypointGraphBatch) -> int:
    if isinstance(observation, RepresentationGraphBatch | PartGraphBatch | KeypointGraphBatch):
        return observation.num_samples
    return int(observation.shape[0])


@torch.no_grad()
def _evaluate_tensor_split(model: nn.Module, features: torch.Tensor, actions: torch.Tensor) -> dict[str, float]:
    model.eval()
    prediction = model(features)
    errors = prediction - actions
    l2 = errors.norm(dim=-1)
    mse = torch.nn.functional.mse_loss(prediction, actions)
    pred_norm = prediction.norm(dim=-1)
    action_norm = actions.norm(dim=-1)
    cosine = (prediction * actions).sum(dim=-1) / (pred_norm * action_norm).clamp_min(1e-8)
    return {
        "mse": float(mse.detach().cpu().item()),
        "l2_error": float(l2.mean().detach().cpu().item()),
        "l2_error_p90": float(torch.quantile(l2.detach().cpu(), 0.90).item()),
        "cosine": float(cosine.mean().detach().cpu().item()),
    }


@torch.no_grad()
def _evaluate_observation_split(
    model: nn.Module,
    observations: torch.Tensor | RepresentationGraphBatch,
    actions: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    prediction = model(observations)
    errors = prediction - actions
    l2 = errors.norm(dim=-1)
    mse = torch.nn.functional.mse_loss(prediction, actions)
    pred_norm = prediction.norm(dim=-1)
    action_norm = actions.norm(dim=-1)
    cosine = (prediction * actions).sum(dim=-1) / (pred_norm * action_norm).clamp_min(1e-8)
    return {
        "mse": float(mse.detach().cpu().item()),
        "l2_error": float(l2.mean().detach().cpu().item()),
        "l2_error_p90": float(torch.quantile(l2.detach().cpu(), 0.90).item()),
        "cosine": float(cosine.mean().detach().cpu().item()),
    }


def keypoint_action_loss(prediction: torch.Tensor, target: torch.Tensor, yaw_loss_weight: float) -> torch.Tensor:
    xyz_loss = torch.nn.functional.mse_loss(prediction[:, :3], target[:, :3])
    yaw_loss = torch.nn.functional.mse_loss(prediction[:, 3], target[:, 3])
    return xyz_loss + float(yaw_loss_weight) * yaw_loss


@torch.no_grad()
def _evaluate_keypoint_observation_split(
    model: nn.Module,
    observations: torch.Tensor | KeypointGraphBatch,
    actions: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    prediction = model(observations)
    translation_l2 = (prediction[:, :3] - actions[:, :3]).norm(dim=-1)
    yaw_abs = (prediction[:, 3] - actions[:, 3]).abs()
    return {
        "translation_l2": float(translation_l2.mean().detach().cpu().item()),
        "translation_l2_p90": float(torch.quantile(translation_l2.detach().cpu(), 0.90).item()),
        "yaw_abs_error": float(yaw_abs.mean().detach().cpu().item()),
        "yaw_abs_error_p90": float(torch.quantile(yaw_abs.detach().cpu(), 0.90).item()),
    }


def _build_representation_model(config: RepresentationTrainConfig, dataset: dict[str, Any]) -> nn.Module:
    if config.model_kind == "mpnn":
        dims = _graph_feature_dims(dataset)
        return TinyRepresentationMPNN(
            object_feature_dim=dims["object"],
            ee_feature_dim=dims["ee"],
            edge_feature_dim=dims["edge"],
            hidden_dim=config.mpnn_hidden_dim,
            num_layers=config.mpnn_layers,
        )
    return RepresentationMLP(
        input_dim=_variant_feature_dim(dataset, config.variant),
        hidden_dim=config.hidden_dim,
    )


def _build_part_node_model(config: PartNodeTrainConfig, dataset: dict[str, Any]) -> nn.Module:
    dims = _part_graph_feature_dims(dataset, config.variant)
    return PartNodeMPNN(
        object_feature_dim=dims["object"],
        ee_feature_dim=dims["ee"],
        edge_feature_dim=dims["edge"],
        part_feature_dim=dims["part"],
        hidden_dim=config.hidden_dim,
        num_layers=config.message_passing_layers,
    )


def _build_keypoint_model(config: KeypointTrainConfig, dataset: dict[str, Any]) -> nn.Module:
    dims = _keypoint_model_dims(dataset, config.variant)
    if config.variant == "keypoint_concat":
        return KeypointConcatMLP(
            input_dim=dims["input"],
            hidden_dim=config.mlp_hidden_dim,
            output_dim=4,
        )
    return KeypointMPNN(
        object_feature_dim=dims["object"],
        ee_feature_dim=dims["ee"],
        edge_feature_dim=dims["edge"],
        part_feature_dim=dims.get("part", 0),
        keypoint_feature_dim=dims.get("keypoint", 0),
        hidden_dim=config.hidden_dim,
        num_layers=config.message_passing_layers,
        output_dim=4,
        readout=config.readout,
    )


def _graph_feature_dims(dataset: dict[str, Any]) -> dict[str, int]:
    dims = dataset["feature_dims"]
    return {
        "object": int(7 + dims["visual"]),
        "ee": int(3 + dims["ee_geometry"]),
        "edge": 4,
    }


def _keypoint_model_dims(dataset: dict[str, Any], variant: KeypointVariant) -> dict[str, int]:
    dims = dataset["feature_dims"]
    if variant == "keypoint_concat":
        return {
            "input": int(dims["object"] + dims["ee"] + dims["keypoints"] * dims["keypoint"]),
            "object": int(dims["object"]),
            "ee": int(dims["ee"]),
            "keypoint": int(dims["keypoint"]),
            "edge": 4,
        }
    if variant == "single_part_node":
        return {
            "object": int(dims["object"]),
            "ee": int(dims["ee"]),
            "part": int(dims["part"]),
            "edge": 4,
        }
    if variant in {"keypoint_graph", "keypoint_graph_arc"}:
        return {
            "object": int(dims["object"]),
            "ee": int(dims["ee"]),
            "keypoint": int(dims["keypoint"] + (1 if variant == "keypoint_graph_arc" else 0)),
            "edge": 4,
        }
    raise ValueError(f"Unknown keypoint variant: {variant}")


def _part_graph_feature_dims(dataset: dict[str, Any], variant: PartNodeVariant) -> dict[str, int]:
    dims = dataset["feature_dims"]
    object_dim = int(dims["object"])
    part_dim = int(dims["part"])
    if variant == "part_concat":
        object_dim += part_dim
        part_dim = 0
    elif variant == "object_only":
        part_dim = 0
    elif variant != "part_node":
        raise ValueError(f"Unknown part-node variant: {variant}")
    return {
        "object": object_dim,
        "ee": int(dims["ee"]),
        "part": part_dim,
        "edge": int(dims["edge"]),
    }


def _sample_task_state(config: RepresentationDemoConfig, rng: np.random.Generator) -> HandleTaskState:
    center = torch.tensor(
        [
            float(rng.uniform(*config.object_x_range)),
            0.0,
            float(rng.uniform(*config.object_z_range)),
        ],
        dtype=torch.float32,
    )
    return HandleTaskState(
        object_center=center,
        object_yaw=float(rng.uniform(*config.object_yaw_range)),
        handle_side=1 if rng.random() >= 0.5 else -1,
        handle_offset=float(rng.uniform(*config.handle_offset_range)),
        object_half_extent=config.object_half_extent,
        gripper_command=float(rng.uniform(*config.gripper_command_range)),
        camera_yaw=float(config.camera_yaw),
    )


def _sample_part_task_state(
    config: PartNodeDemoConfig,
    rng: np.random.Generator,
    layout: PartLayout,
) -> PartTaskState:
    center = torch.tensor(
        [
            float(rng.uniform(*config.object_x_range)),
            0.0,
            float(rng.uniform(*config.object_z_range)),
        ],
        dtype=torch.float32,
    )
    if layout == "iid":
        part_x = float(rng.uniform(*config.train_part_local_x_range))
        part_z = float(rng.uniform(*config.train_part_local_z_range))
    elif layout == "ood":
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        part_x = sign * float(rng.uniform(*config.ood_part_local_abs_x_range))
        part_z = float(rng.uniform(*config.ood_part_local_z_range))
    else:
        raise ValueError(f"Unknown part layout: {layout}")
    part_extent = float(rng.uniform(*config.part_extent_range))
    return PartTaskState(
        object_center=center,
        object_yaw=float(rng.uniform(*config.object_yaw_range)),
        part_local=torch.tensor([part_x, 0.0, part_z], dtype=torch.float32),
        part_local_yaw=float(rng.uniform(*config.part_local_yaw_range)),
        part_extent=(part_extent, part_extent * 0.55),
        object_half_extent=config.object_half_extent,
        gripper_command=float(rng.uniform(*config.gripper_command_range)),
        camera_yaw=float(config.camera_yaw),
        layout=layout,
    )


def _sample_keypoint_task_state(
    config: KeypointDemoConfig,
    rng: np.random.Generator,
    split: KeypointEvalSplit,
) -> KeypointTaskState:
    center = torch.tensor(
        [
            float(rng.uniform(*config.object_x_range)),
            0.0,
            float(rng.uniform(*config.object_z_range)),
        ],
        dtype=torch.float32,
    )
    if split == "length_ood":
        length = float(rng.uniform(*config.length_ood_range))
    else:
        length = float(rng.uniform(*config.train_length_range))
    if split == "orientation_ood":
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        part_yaw = sign * float(rng.uniform(*config.orientation_ood_yaw_range))
    else:
        part_yaw = float(rng.uniform(*config.train_part_yaw_range))
    if split == "shape_ood":
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        curvature = sign * float(rng.uniform(*config.shape_ood_curvature_abs_range))
        spacing_jitter = config.shape_ood_spacing_jitter
    else:
        curvature = float(rng.uniform(*config.train_curvature_range))
        spacing_jitter = config.train_spacing_jitter
    offset = torch.tensor(
        [
            float(rng.uniform(*config.train_offset_x_range)),
            0.0,
            float(rng.uniform(*config.train_offset_z_range)),
        ],
        dtype=torch.float32,
    )
    object_yaw = float(rng.uniform(*config.object_yaw_range))
    keypoints_part = generate_keypoints_local(config.keypoints, length, curvature, spacing_jitter, rng)
    keypoints_object = offset.reshape(1, 3) + _rotate_xz_batch(keypoints_part, part_yaw)
    keypoints_world = center.reshape(1, 3) + _rotate_xz_batch(keypoints_object, object_yaw)
    return KeypointTaskState(
        object_center=center,
        object_yaw=object_yaw,
        object_half_extent=config.object_half_extent,
        part_offset_local=offset,
        part_yaw_local=part_yaw,
        length=length,
        curvature=curvature,
        spacing_jitter=spacing_jitter,
        keypoints_local=keypoints_object.float(),
        keypoints_world=keypoints_world.float(),
        gripper_command=float(rng.uniform(*config.gripper_command_range)),
        ee_yaw=float(rng.uniform(-math.pi, math.pi)),
        split=split,
    )


def generate_keypoints_local(
    keypoints: int,
    length: float,
    curvature: float,
    spacing_jitter: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    if keypoints < 3:
        raise ValueError("Keypoint alignment requires at least 3 keypoints.")
    base_gaps = np.ones(keypoints - 1, dtype=np.float64)
    if spacing_jitter > 0:
        base_gaps *= rng.uniform(1.0 - spacing_jitter, 1.0 + spacing_jitter, size=keypoints - 1)
    x = np.concatenate([[0.0], np.cumsum(base_gaps)])
    x = (x - x.mean()) / max(x[-1] - x[0], 1e-6) * length
    u = x / max(length, 1e-6)
    z = curvature * np.sin(2.0 * math.pi * u)
    return torch.tensor(np.stack([x, np.zeros_like(x), z], axis=1), dtype=torch.float32)


def sample_keypoint_episode_specs(
    config: KeypointDemoConfig,
    episodes: int,
    seed: int,
    split: KeypointEvalSplit,
) -> list[KeypointEpisodeSpec]:
    rng = np.random.default_rng(seed)
    home_qpos = np.array([0.0, -0.35, 0.95, -0.45], dtype=np.float64)
    specs = []
    for episode_id in range(episodes):
        state = _sample_keypoint_task_state(config, rng, split)
        arm_qpos = home_qpos + rng.uniform(low=-0.25, high=0.25, size=4)
        specs.append(
            KeypointEpisodeSpec(
                episode_id=episode_id,
                task_state=state,
                arm_qpos=tuple(float(value) for value in arm_qpos),
            )
        )
    return specs


def sample_part_episode_specs(
    config: PartNodeDemoConfig,
    episodes: int,
    seed: int,
    layout: PartLayout,
) -> list[PartEpisodeSpec]:
    rng = np.random.default_rng(seed)
    home_qpos = np.array([0.0, -0.35, 0.95, -0.45], dtype=np.float64)
    specs: list[PartEpisodeSpec] = []
    for episode_id in range(episodes):
        task_state = _sample_part_task_state(config, rng, layout=layout)
        arm_qpos = home_qpos + rng.uniform(low=-0.25, high=0.25, size=4)
        specs.append(
            PartEpisodeSpec(
                episode_id=episode_id,
                task_state=task_state,
                arm_qpos=tuple(float(value) for value in arm_qpos),
            )
        )
    return specs


def _reset_env_for_task(env: MujocoManipulatorEnv, task_state: HandleTaskState) -> None:
    env.reset_pick_scene(
        object_x=float(task_state.object_center[0].item()),
        object_yaw=task_state.object_yaw,
        randomize_robot=True,
    )
    env.set_object_position(task_state.object_center.detach().cpu().numpy(), yaw=task_state.object_yaw)
    for _ in range(8):
        env.step_delta_ee(torch.zeros(3), gripper=task_state.gripper_command)


def _reset_env_for_part_task(env: MujocoManipulatorEnv, task_state: PartTaskState) -> None:
    env.reset_pick_scene(
        object_x=float(task_state.object_center[0].item()),
        object_yaw=task_state.object_yaw,
        randomize_robot=True,
    )
    env.set_object_position(task_state.object_center.detach().cpu().numpy(), yaw=task_state.object_yaw)
    env.set_target_position(task_state.part_world_position.detach().cpu().numpy())
    for _ in range(8):
        env.step_delta_ee(torch.zeros(3), gripper=task_state.gripper_command)


def _reset_env_for_part_episode_spec(env: MujocoManipulatorEnv, spec: PartEpisodeSpec) -> None:
    task_state = spec.task_state
    env.reset_pick_scene(
        object_x=float(task_state.object_center[0].item()),
        object_yaw=task_state.object_yaw,
        randomize_robot=False,
    )
    env._set_arm_qpos(env._clip_arm_qpos(np.asarray(spec.arm_qpos, dtype=np.float64)))
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = 0.0
    env.set_object_position(task_state.object_center.detach().cpu().numpy(), yaw=task_state.object_yaw)
    env.set_target_position(task_state.part_world_position.detach().cpu().numpy())
    for _ in range(8):
        env.step_delta_ee(torch.zeros(3), gripper=task_state.gripper_command)


def _reset_env_for_keypoint_state(env: MujocoManipulatorEnv, task_state: KeypointTaskState) -> None:
    env.reset_pick_scene(
        object_x=float(task_state.object_center[0].item()),
        object_yaw=task_state.object_yaw,
        randomize_robot=True,
    )
    env.set_object_position(task_state.object_center.detach().cpu().numpy(), yaw=task_state.object_yaw)
    env.set_target_position(task_state.target_position.detach().cpu().numpy())
    for _ in range(8):
        env.step_delta_ee(torch.zeros(3), gripper=task_state.gripper_command)


def _reset_env_for_keypoint_spec(env: MujocoManipulatorEnv, spec: KeypointEpisodeSpec) -> None:
    task_state = spec.task_state
    env.reset_pick_scene(
        object_x=float(task_state.object_center[0].item()),
        object_yaw=task_state.object_yaw,
        randomize_robot=False,
    )
    env._set_arm_qpos(env._clip_arm_qpos(np.asarray(spec.arm_qpos, dtype=np.float64)))
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = 0.0
    env.set_object_position(task_state.object_center.detach().cpu().numpy(), yaw=task_state.object_yaw)
    env.set_target_position(task_state.target_position.detach().cpu().numpy())
    for _ in range(8):
        env.step_delta_ee(torch.zeros(3), gripper=task_state.gripper_command)


def _expert_delta(env: MujocoManipulatorEnv, task_state: HandleTaskState, max_delta_ee: float) -> torch.Tensor:
    active_tip = _active_fingertip_position(env, task_state.handle_side)
    return clamp_delta(task_state.handle_position - active_tip, max_delta_ee)


def _part_expert_delta(env: MujocoManipulatorEnv, task_state: PartTaskState, max_delta_ee: float) -> torch.Tensor:
    ee = env.robot_observation().ee_position
    return clamp_delta(task_state.part_world_position - ee, max_delta_ee)


def _keypoint_expert_action(
    env: MujocoManipulatorEnv,
    task_state: KeypointTaskState,
    ee_yaw: float,
    config: KeypointDemoConfig,
) -> torch.Tensor:
    ee = env.robot_observation().ee_position.float()
    delta = clamp_delta(task_state.target_position - ee, config.max_delta_ee)
    yaw_delta = max(-config.max_delta_yaw, min(config.max_delta_yaw, wrap_angle(task_state.target_yaw - ee_yaw)))
    return torch.cat([delta, torch.tensor([yaw_delta], dtype=torch.float32)], dim=0)


def _active_fingertip_distance(env: MujocoManipulatorEnv, task_state: HandleTaskState) -> float:
    return float((_active_fingertip_position(env, task_state.handle_side) - task_state.handle_position).norm().item())


def _ee_to_part_distance(env: MujocoManipulatorEnv, task_state: PartTaskState) -> float:
    return float((env.robot_observation().ee_position - task_state.part_world_position).norm().item())


def _active_fingertip_position(env: MujocoManipulatorEnv, handle_side: int) -> torch.Tensor:
    thumb, finger = _fingertip_positions(env)
    return thumb if handle_side > 0 else finger


def _fingertip_positions(env: MujocoManipulatorEnv) -> tuple[torch.Tensor, torch.Tensor]:
    thumb_names = {"thumbtip1", "thumbtip2"}
    finger_names = {"fingertip1", "fingertip2"}
    thumb = _mean_geom_position(env, env.thumb_geom_ids, thumb_names)
    finger = _mean_geom_position(env, env.finger_geom_ids, finger_names)
    return thumb, finger


def _mean_geom_position(env: MujocoManipulatorEnv, geom_ids: set[int], wanted_names: set[str]) -> torch.Tensor:
    positions = []
    for geom_id in geom_ids:
        name = env.model.geom(geom_id).name
        if name in wanted_names:
            positions.append(env.data.geom_xpos[geom_id].copy())
    if not positions:
        raise RuntimeError(f"Could not find fingertip geoms {sorted(wanted_names)}")
    return torch.tensor(np.mean(np.stack(positions, axis=0), axis=0), dtype=torch.float32)


def _rotate_xz(vector: torch.Tensor, yaw: float) -> torch.Tensor:
    c = math.cos(yaw)
    s = math.sin(yaw)
    x = c * vector[0] - s * vector[2]
    z = s * vector[0] + c * vector[2]
    return torch.stack([x, vector[1], z]).float()


def _rotate_xz_batch(points: torch.Tensor, yaw: float) -> torch.Tensor:
    c = math.cos(yaw)
    s = math.sin(yaw)
    x = c * points[:, 0] - s * points[:, 2]
    z = s * points[:, 0] + c * points[:, 2]
    return torch.stack([x, points[:, 1], z], dim=1).float()


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def keypoint_local_tangent_yaw(keypoints_world: torch.Tensor) -> float:
    if keypoints_world.shape[0] < 3:
        raise ValueError("Need at least 3 keypoints to compute a local tangent.")
    mid = keypoints_world.shape[0] // 2
    left = keypoints_world[mid - 1]
    right = keypoints_world[mid + 1]
    delta = right - left
    return float(math.atan2(float(delta[2]), float(delta[0])))


def keypoint_spacing_asymmetry(keypoints_local: torch.Tensor) -> float:
    diffs = keypoints_local[1:] - keypoints_local[:-1]
    dists = diffs[:, [0, 2]].norm(dim=-1)
    return float((dists.std(unbiased=False) / dists.mean().clamp_min(1e-6)).item())


def _episode_split(num_episodes: int, seed: int) -> dict[str, list[int]]:
    indices = np.arange(num_episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_end = max(1, int(0.8 * num_episodes))
    val_end = max(train_end + 1, int(0.9 * num_episodes)) if num_episodes >= 3 else train_end
    val_end = min(val_end, num_episodes)
    return {
        "train": sorted(int(idx) for idx in indices[:train_end]),
        "val": sorted(int(idx) for idx in indices[train_end:val_end]),
        "test": sorted(int(idx) for idx in indices[val_end:]),
    }


def _feature_dims(episode: dict[str, Any]) -> dict[str, int]:
    return {
        "pose": int(episode["pose_features"].shape[-1]),
        "visual": int(episode["visual_features"].shape[-1]),
        "ee_geometry": int(episode["ee_geometry_features"].shape[-1]),
    }


def _part_feature_dims(episode: dict[str, Any]) -> dict[str, int]:
    return {
        "object": int(episode["object_features"].shape[-1]),
        "ee": int(episode["ee_features"].shape[-1]),
        "part": int(episode["part_features"].shape[-1]),
        "edge": 4,
    }


def _keypoint_feature_dims(episode: dict[str, Any]) -> dict[str, int]:
    return {
        "object": int(episode["object_features"].shape[-1]),
        "ee": int(episode["ee_features"].shape[-1]),
        "part": int(episode["part_features"].shape[-1]),
        "keypoints": int(episode["keypoint_features"].shape[-2]),
        "keypoint": int(episode["keypoint_features"].shape[-1]),
        "edge": 4,
    }


def _variant_feature_dim(dataset: dict[str, Any], variant: RepresentationVariant) -> int:
    dims = dataset["feature_dims"]
    if variant == "pose":
        return int(dims["pose"])
    if variant == "pose_visual":
        return int(dims["pose"] + dims["visual"])
    if variant == "pose_visual_ee":
        return int(dims["pose"] + dims["visual"] + dims["ee_geometry"])
    raise ValueError(f"Unknown representation variant: {variant}")


def _dataset_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    episodes = payload["episodes"]
    actions = torch.cat([episode["actions"] for episode in episodes], dim=0)
    return {
        "task": payload["task"],
        "num_episodes": len(episodes),
        "num_samples": int(actions.shape[0]),
        "split": payload["split"],
        "feature_dims": payload["feature_dims"],
        "expert_success_rate": _mean([1.0 if item["expert_success"] else 0.0 for item in episodes]),
        "episode_length": {
            "mean": _mean([item["steps"] for item in episodes]),
            "min": min(int(item["steps"]) for item in episodes),
            "max": max(int(item["steps"]) for item in episodes),
        },
        "action": {
            "mean_norm": float(actions.norm(dim=-1).mean().item()),
            "max_norm": float(actions.norm(dim=-1).max().item()),
        },
        "handle_side_balance": {
            "left_fraction": _mean([1.0 if item["handle_side"] < 0 else 0.0 for item in episodes]),
            "right_fraction": _mean([1.0 if item["handle_side"] > 0 else 0.0 for item in episodes]),
        },
    }


def _part_dataset_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    episodes = payload["episodes"]
    ood_episodes = payload["ood_episodes"]
    iid_actions = torch.cat([episode["actions"] for episode in episodes], dim=0)
    ood_actions = torch.cat([episode["actions"] for episode in ood_episodes], dim=0)
    return {
        "task": payload["task"],
        "num_episodes": len(episodes),
        "num_ood_episodes": len(ood_episodes),
        "num_samples": int(iid_actions.shape[0]),
        "num_ood_samples": int(ood_actions.shape[0]),
        "split": payload["split"],
        "feature_dims": payload["feature_dims"],
        "expert_success_rate": _mean([1.0 if item["expert_success"] else 0.0 for item in episodes]),
        "ood_expert_success_rate": _mean([1.0 if item["expert_success"] else 0.0 for item in ood_episodes]),
        "episode_length": {
            "mean": _mean([item["steps"] for item in episodes]),
            "min": min(int(item["steps"]) for item in episodes),
            "max": max(int(item["steps"]) for item in episodes),
        },
        "ood_episode_length": {
            "mean": _mean([item["steps"] for item in ood_episodes]),
            "min": min(int(item["steps"]) for item in ood_episodes),
            "max": max(int(item["steps"]) for item in ood_episodes),
        },
        "action": {
            "mean_norm": float(iid_actions.norm(dim=-1).mean().item()),
            "max_norm": float(iid_actions.norm(dim=-1).max().item()),
        },
        "ood_action": {
            "mean_norm": float(ood_actions.norm(dim=-1).mean().item()),
            "max_norm": float(ood_actions.norm(dim=-1).max().item()),
        },
        "part_local_x": {
            "iid_mean": _mean([float(item["part_local"][0]) for item in episodes]),
            "ood_mean": _mean([float(item["part_local"][0]) for item in ood_episodes]),
            "iid_abs_mean": _mean([abs(float(item["part_local"][0])) for item in episodes]),
            "ood_abs_mean": _mean([abs(float(item["part_local"][0])) for item in ood_episodes]),
        },
    }


def keypoint_geometry_statistics(
    episodes: list[dict[str, Any]],
    ood_episodes: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    all_splits: dict[str, list[dict[str, Any]]] = {"iid": episodes, **ood_episodes}

    def stats(items: list[dict[str, Any]]) -> dict[str, float]:
        yaw_diff = [float(item["global_local_yaw_abs_diff"]) for item in items]
        curvature = [abs(float(item["curvature"])) for item in items]
        spacing = [float(item["spacing_asymmetry"]) for item in items]
        length = [float(item["length"]) for item in items]
        return {
            "num_episodes": float(len(items)),
            "length_mean": _mean(length),
            "abs_curvature_mean": _mean(curvature),
            "spacing_asymmetry_mean": _mean(spacing),
            "global_local_yaw_abs_diff_mean": _mean(yaw_diff),
            "global_local_yaw_abs_diff_p75": float(np.quantile(np.array(yaw_diff), 0.75)) if yaw_diff else 0.0,
            "global_local_yaw_abs_diff_p90": float(np.quantile(np.array(yaw_diff), 0.90)) if yaw_diff else 0.0,
        }

    return {name: stats(items) for name, items in all_splits.items()}


def _env_config(config: RepresentationDemoConfig) -> MujocoReachConfig:
    return MujocoReachConfig(
        max_steps=config.max_steps,
        control_substeps=25,
        max_delta_ee=config.max_delta_ee,
        target_radius=config.target_radius,
        pick_scene=True,
        kinematic_joint_control=True,
    )


def _part_env_config(config: PartNodeDemoConfig) -> MujocoReachConfig:
    return MujocoReachConfig(
        max_steps=config.max_steps,
        control_substeps=25,
        max_delta_ee=config.max_delta_ee,
        target_radius=config.target_radius,
        pick_scene=True,
        kinematic_joint_control=True,
    )


def _keypoint_env_config(config: KeypointDemoConfig) -> MujocoReachConfig:
    return MujocoReachConfig(
        max_steps=config.max_steps,
        control_substeps=25,
        max_delta_ee=config.max_delta_ee,
        target_radius=config.target_radius,
        pick_scene=True,
        kinematic_joint_control=True,
    )


def _demo_config_from_checkpoint(checkpoint: dict[str, Any]) -> RepresentationDemoConfig:
    values = dict(checkpoint["demo_config"])
    tuple_fields = [
        "object_x_range",
        "object_z_range",
        "object_yaw_range",
        "handle_offset_range",
        "object_half_extent",
        "gripper_command_range",
    ]
    for field in tuple_fields:
        if field in values:
            values[field] = tuple(values[field])
    return RepresentationDemoConfig(**values)


def _part_demo_config_from_checkpoint(checkpoint: dict[str, Any]) -> PartNodeDemoConfig:
    values = dict(checkpoint["demo_config"])
    tuple_fields = [
        "object_x_range",
        "object_z_range",
        "object_yaw_range",
        "object_half_extent",
        "train_part_local_x_range",
        "train_part_local_z_range",
        "ood_part_local_abs_x_range",
        "ood_part_local_z_range",
        "part_extent_range",
        "part_local_yaw_range",
        "gripper_command_range",
    ]
    for field in tuple_fields:
        if field in values:
            values[field] = tuple(values[field])
    return PartNodeDemoConfig(**values)


def _keypoint_demo_config_from_checkpoint(checkpoint: dict[str, Any]) -> KeypointDemoConfig:
    values = dict(checkpoint["demo_config"])
    tuple_fields = [
        "object_x_range",
        "object_z_range",
        "object_yaw_range",
        "object_half_extent",
        "train_length_range",
        "length_ood_range",
        "train_part_yaw_range",
        "orientation_ood_yaw_range",
        "train_offset_x_range",
        "train_offset_z_range",
        "train_curvature_range",
        "shape_ood_curvature_abs_range",
        "gripper_command_range",
    ]
    for field in tuple_fields:
        if field in values:
            values[field] = tuple(values[field])
    return KeypointDemoConfig(**values)


def _paired_episode_result(
    spec: PartEpisodeSpec,
    concat_result: dict[str, Any],
    node_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "episode_id": spec.episode_id,
        "spec": part_episode_spec_to_dict(spec),
        "part_concat": concat_result,
        "part_node": node_result,
        "paired": paired_episode_metrics(concat_result, node_result),
    }


def part_episode_spec_to_dict(spec: PartEpisodeSpec) -> dict[str, Any]:
    state = spec.task_state
    return {
        "episode_id": spec.episode_id,
        "arm_qpos": list(spec.arm_qpos),
        "object_center": state.object_center.tolist(),
        "object_yaw": state.object_yaw,
        "part_local": state.part_local.tolist(),
        "part_local_yaw": state.part_local_yaw,
        "part_world_position": state.part_world_position.tolist(),
        "part_world_yaw": state.part_world_yaw,
        "part_extent": list(state.part_extent),
        "object_half_extent": list(state.object_half_extent),
        "gripper_command": state.gripper_command,
        "camera_yaw": state.camera_yaw,
        "layout": state.layout,
    }


def paired_episode_metrics(concat_result: dict[str, Any], node_result: dict[str, Any]) -> dict[str, Any]:
    concat_success = bool(concat_result["success"])
    node_success = bool(node_result["success"])
    if concat_success and node_success:
        outcome = "both_success"
    elif concat_success and not node_success:
        outcome = "concat_only_success"
    elif not concat_success and node_success:
        outcome = "part_node_only_success"
    else:
        outcome = "both_fail"
    return {
        "outcome": outcome,
        "delta_final_distance": float(node_result["final_distance"] - concat_result["final_distance"]),
        "delta_trajectory_length": float(node_result["trajectory_length"] - concat_result["trajectory_length"]),
        "delta_steps": int(node_result["steps"] - concat_result["steps"]),
        "delta_max_distance_from_target": float(
            node_result["max_distance_from_target"] - concat_result["max_distance_from_target"]
        ),
    }


def paired_contingency_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"both_success": 0, "concat_only_success": 0, "part_node_only_success": 0, "both_fail": 0}
    for episode in episodes:
        outcome = episode["paired"]["outcome"]
        counts[outcome] += 1
    return counts


def summarize_paired_episode_results(
    episodes: list[dict[str, Any]],
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    counts = paired_contingency_counts(episodes)
    concat_success = np.array([1.0 if episode["part_concat"]["success"] else 0.0 for episode in episodes], dtype=float)
    node_success = np.array([1.0 if episode["part_node"]["success"] else 0.0 for episode in episodes], dtype=float)
    delta_final = np.array([episode["paired"]["delta_final_distance"] for episode in episodes], dtype=float)
    delta_traj = np.array([episode["paired"]["delta_trajectory_length"] for episode in episodes], dtype=float)
    delta_steps = np.array([episode["paired"]["delta_steps"] for episode in episodes], dtype=float)
    concat_final = np.array([episode["part_concat"]["final_distance"] for episode in episodes], dtype=float)
    node_final = np.array([episode["part_node"]["final_distance"] for episode in episodes], dtype=float)
    concat_only = counts["concat_only_success"]
    node_only = counts["part_node_only_success"]
    return {
        "num_episodes": len(episodes),
        "contingency": counts,
        "part_node_only_to_concat_only_ratio": _safe_ratio(node_only, concat_only),
        "mcnemar_exact_p": _mcnemar_exact_p(concat_only, node_only),
        "success": {
            "part_concat": _rate_with_ci(concat_success, bootstrap_samples, seed + 1),
            "part_node": _rate_with_ci(node_success, bootstrap_samples, seed + 2),
            "paired_delta_part_node_minus_concat": _mean_median_ci(
                node_success - concat_success,
                bootstrap_samples,
                seed + 3,
            ),
        },
        "final_distance": {
            "part_concat": _mean_median_ci(concat_final, bootstrap_samples, seed + 4),
            "part_node": _mean_median_ci(node_final, bootstrap_samples, seed + 5),
            "paired_delta_part_node_minus_concat": _mean_median_ci(delta_final, bootstrap_samples, seed + 6),
        },
        "trajectory_length": {
            "paired_delta_part_node_minus_concat": _mean_median_ci(delta_traj, bootstrap_samples, seed + 7),
        },
        "steps": {
            "paired_delta_part_node_minus_concat": _mean_median_ci(delta_steps, bootstrap_samples, seed + 8),
        },
    }


def paired_worst_cases(
    episodes: list[dict[str, Any]],
    catastrophic_distance: float = 0.3,
) -> dict[str, Any]:
    top = sorted(episodes, key=lambda item: float(item["part_node"]["final_distance"]), reverse=True)[:10]
    catastrophic_concat_success = [
        episode
        for episode in episodes
        if bool(episode["part_concat"]["success"])
        and not bool(episode["part_node"]["success"])
        and float(episode["part_node"]["final_distance"]) > catastrophic_distance
    ]
    node_large_failures = [
        episode for episode in episodes if float(episode["part_node"]["final_distance"]) > catastrophic_distance
    ]
    return {
        "top_part_node_final_distance": [_compact_worst_case(item) for item in top],
        "concat_success_part_node_catastrophic_fail": [
            _compact_worst_case(item)
            for item in sorted(
                catastrophic_concat_success,
                key=lambda episode: float(episode["part_node"]["final_distance"]),
                reverse=True,
            )[:10]
        ],
        "part_node_large_failures": [
            _compact_worst_case(item)
            for item in sorted(
                node_large_failures,
                key=lambda episode: float(episode["part_node"]["final_distance"]),
                reverse=True,
            )[:10]
        ],
        "counts": {
            "concat_success_part_node_catastrophic_fail": len(catastrophic_concat_success),
            "part_node_large_failures": len(node_large_failures),
        },
    }


def paired_geometry_breakdown(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    distances = np.array([_part_offset_distance(episode) for episode in episodes], dtype=float)
    q1, q2 = np.quantile(distances, [1.0 / 3.0, 2.0 / 3.0])
    buckets = {
        "small": distances <= q1,
        "medium": (distances > q1) & (distances <= q2),
        "large": distances > q2,
    }
    bucket_rows = {}
    for name, mask in buckets.items():
        selected = [episode for episode, keep in zip(episodes, mask, strict=True) if bool(keep)]
        bucket_rows[name] = _paired_subset_metrics(selected)
        bucket_rows[name]["offset_distance_range"] = [
            float(distances[mask].min()) if mask.any() else 0.0,
            float(distances[mask].max()) if mask.any() else 0.0,
        ]
    quadrants: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        local = episode["spec"]["part_local"]
        label = ("x+" if float(local[0]) >= 0.0 else "x-") + "_" + ("z+" if float(local[2]) >= 0.0 else "z-")
        quadrants.setdefault(label, {"episodes": []})["episodes"].append(episode)
    quadrant_rows = {label: _paired_subset_metrics(value["episodes"]) for label, value in sorted(quadrants.items())}
    return {
        "offset_distance_quantiles": [float(q1), float(q2)],
        "buckets": bucket_rows,
        "quadrants": quadrant_rows,
    }


def paired_validation_summary_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Paired OOD Part-node Validation", ""]
    lines.append("## Seed Contingency")
    lines.append("")
    lines.append("| Seed | Episodes | Both success | Concat only | Part-node only | Both fail | Node-only / Concat-only | McNemar p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for seed_payload in payload["seeds"]:
        summary = seed_payload["summary"]
        counts = summary["contingency"]
        lines.append(
            "| "
            + " | ".join(
                [
                    seed_payload["seed_name"],
                    str(summary["num_episodes"]),
                    str(counts["both_success"]),
                    str(counts["concat_only_success"]),
                    str(counts["part_node_only_success"]),
                    str(counts["both_fail"]),
                    _fmt_optional(summary["part_node_only_to_concat_only_ratio"]),
                    _fmt_optional(summary["mcnemar_exact_p"]),
                ]
            )
            + " |"
        )
    aggregate = payload["aggregate"]
    counts = aggregate["contingency"]
    lines.append(
        "| "
        + " | ".join(
            [
                "pooled",
                str(aggregate["num_episodes"]),
                str(counts["both_success"]),
                str(counts["concat_only_success"]),
                str(counts["part_node_only_success"]),
                str(counts["both_fail"]),
                _fmt_optional(aggregate["part_node_only_to_concat_only_ratio"]),
                _fmt_optional(aggregate["mcnemar_exact_p"]),
            ]
        )
        + " |"
    )
    lines.extend(["", "## Paired Metrics", ""])
    lines.append("| Metric | Mean | Median | 95% bootstrap CI |")
    lines.append("|---|---:|---:|---:|")
    metric = aggregate["final_distance"]["paired_delta_part_node_minus_concat"]
    lines.append(
        f"| Delta final distance, node - concat | {metric['mean']:.4f} | {metric['median']:.4f} | "
        f"[{metric['ci95'][0]:.4f}, {metric['ci95'][1]:.4f}] |"
    )
    metric = aggregate["trajectory_length"]["paired_delta_part_node_minus_concat"]
    lines.append(
        f"| Delta trajectory length, node - concat | {metric['mean']:.4f} | {metric['median']:.4f} | "
        f"[{metric['ci95'][0]:.4f}, {metric['ci95'][1]:.4f}] |"
    )
    metric = aggregate["steps"]["paired_delta_part_node_minus_concat"]
    lines.append(
        f"| Delta steps, node - concat | {metric['mean']:.4f} | {metric['median']:.4f} | "
        f"[{metric['ci95'][0]:.4f}, {metric['ci95'][1]:.4f}] |"
    )
    lines.extend(["", "## Geometry Buckets", ""])
    lines.append("| Bucket | N | Concat success | Part-node success | Concat final dist | Part-node final dist | Delta final dist |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, row in aggregate["geometry_breakdown"]["buckets"].items():
        lines.append(
            f"| {name} | {row['num_episodes']} | {row['part_concat_success']:.3f} | "
            f"{row['part_node_success']:.3f} | {row['part_concat_final_distance']:.4f} | "
            f"{row['part_node_final_distance']:.4f} | {row['delta_final_distance']:.4f} |"
        )
    lines.extend(["", "## Worst Part-node Final Distances", ""])
    lines.append("| Episode | Seed | Part local | Concat success/dist | Part-node success/dist | Plot |")
    lines.append("|---:|---|---:|---:|---:|---|")
    for case in aggregate["top_part_node_final_distance"][:10]:
        lines.append(
            f"| {case['episode_id']} | {case.get('seed', '')} | {case['part_local']} | "
            f"{case['part_concat']['success']} / {case['part_concat']['final_distance']:.4f} | "
            f"{case['part_node']['success']} / {case['part_node']['final_distance']:.4f} | "
            f"{case.get('plot_path', '')} |"
        )
    lines.extend(
        [
            "",
            "Latency reference from the previous run: part_concat policy ~= 0.504 ms, part_node policy ~= 0.578 ms.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_paired_episode_trajectory(episode: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_traj = np.array(episode["part_concat"]["trajectory"], dtype=float)
    node_traj = np.array(episode["part_node"]["trajectory"], dtype=float)
    object_center = np.array(episode["spec"]["object_center"], dtype=float)
    part_world = np.array(episode["spec"]["part_world_position"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(concat_traj[:, 0], concat_traj[:, 2], "-o", markersize=2.5, linewidth=1.2, label="Part-concat EE")
    ax.plot(node_traj[:, 0], node_traj[:, 2], "-o", markersize=2.5, linewidth=1.2, label="Part-node EE")
    ax.scatter([concat_traj[0, 0]], [concat_traj[0, 2]], marker="s", s=70, label="EE initial")
    ax.scatter([object_center[0]], [object_center[2]], marker="x", s=80, label="Object center")
    ax.scatter([part_world[0]], [part_world[2]], marker="*", s=140, label="Part target")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"Episode {episode['episode_id']} paired rollout")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _compact_worst_case(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": episode["episode_id"],
        "seed": episode.get("seed"),
        "initial_ee_position": episode["part_concat"]["trajectory"][0],
        "object_center": episode["spec"]["object_center"],
        "object_yaw": episode["spec"]["object_yaw"],
        "part_local": episode["spec"]["part_local"],
        "part_world_position": episode["spec"]["part_world_position"],
        "part_extent": episode["spec"]["part_extent"],
        "part_concat": {
            "success": bool(episode["part_concat"]["success"]),
            "final_distance": float(episode["part_concat"]["final_distance"]),
            "max_distance_from_target": float(episode["part_concat"]["max_distance_from_target"]),
            "steps": int(episode["part_concat"]["steps"]),
            "trajectory": episode["part_concat"]["trajectory"],
        },
        "part_node": {
            "success": bool(episode["part_node"]["success"]),
            "final_distance": float(episode["part_node"]["final_distance"]),
            "max_distance_from_target": float(episode["part_node"]["max_distance_from_target"]),
            "steps": int(episode["part_node"]["steps"]),
            "trajectory": episode["part_node"]["trajectory"],
        },
        "paired": episode["paired"],
    }


def _paired_subset_metrics(episodes: list[dict[str, Any]]) -> dict[str, float | int]:
    if not episodes:
        return {
            "num_episodes": 0,
            "part_concat_success": 0.0,
            "part_node_success": 0.0,
            "part_concat_final_distance": 0.0,
            "part_node_final_distance": 0.0,
            "delta_final_distance": 0.0,
        }
    return {
        "num_episodes": len(episodes),
        "part_concat_success": _mean([1.0 if episode["part_concat"]["success"] else 0.0 for episode in episodes]),
        "part_node_success": _mean([1.0 if episode["part_node"]["success"] else 0.0 for episode in episodes]),
        "part_concat_final_distance": _mean([episode["part_concat"]["final_distance"] for episode in episodes]),
        "part_node_final_distance": _mean([episode["part_node"]["final_distance"] for episode in episodes]),
        "delta_final_distance": _mean([episode["paired"]["delta_final_distance"] for episode in episodes]),
    }


def _keypoint_paired_episode_result(
    spec: KeypointEpisodeSpec,
    model_results: dict[KeypointVariant, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "episode_id": spec.episode_id,
        "spec": keypoint_episode_spec_to_dict(spec),
        "models": model_results,
    }


def keypoint_episode_spec_to_dict(spec: KeypointEpisodeSpec) -> dict[str, Any]:
    state = spec.task_state
    return {
        "episode_id": spec.episode_id,
        "arm_qpos": list(spec.arm_qpos),
        "object_center": state.object_center.tolist(),
        "object_yaw": state.object_yaw,
        "part_offset_local": state.part_offset_local.tolist(),
        "part_yaw_local": state.part_yaw_local,
        "length": state.length,
        "curvature": state.curvature,
        "spacing_jitter": state.spacing_jitter,
        "spacing_asymmetry": keypoint_spacing_asymmetry(state.keypoints_local),
        "keypoints_local": state.keypoints_local.tolist(),
        "keypoints_world": state.keypoints_world.tolist(),
        "target_position": state.target_position.tolist(),
        "target_yaw": state.target_yaw,
        "principal_yaw": state.principal_yaw,
        "global_local_yaw_abs_diff": abs(wrap_angle(state.target_yaw - state.principal_yaw)),
        "ee_yaw_initial": state.ee_yaw,
        "split": state.split,
    }


def summarize_keypoint_rollouts(results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "success_rate": _mean([1.0 if item["success"] else 0.0 for item in results]),
        "final_position_error": _mean([item["final_position_error"] for item in results]),
        "final_orientation_error": _mean([item["final_orientation_error"] for item in results]),
        "mean_position_error": _mean([item["mean_position_error"] for item in results]),
        "mean_orientation_error": _mean([item["mean_orientation_error"] for item in results]),
        "trajectory_length": _mean([item["trajectory_length"] for item in results]),
        "forward_latency_ms": _mean([item["mean_forward_latency_ms"] for item in results]),
        "policy_latency_ms": _mean([item["mean_policy_latency_ms"] for item in results]),
    }


def summarize_keypoint_eval(
    episodes: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    variants: tuple[KeypointVariant, ...] = ("single_part_node", "keypoint_concat", "keypoint_graph")
    by_model = {
        variant: summarize_keypoint_rollouts([episode["models"][variant] for episode in episodes])
        for variant in variants
    }
    comparisons = {
        "single_part_node_vs_keypoint_concat": _keypoint_pair_metrics(
            episodes, "single_part_node", "keypoint_concat", bootstrap_samples, seed + 1
        ),
        "keypoint_concat_vs_keypoint_graph": _keypoint_pair_metrics(
            episodes, "keypoint_concat", "keypoint_graph", bootstrap_samples, seed + 2
        ),
        "single_part_node_vs_keypoint_graph": _keypoint_pair_metrics(
            episodes, "single_part_node", "keypoint_graph", bootstrap_samples, seed + 3
        ),
    }
    return {
        "num_episodes": len(episodes),
        "models": by_model,
        "paired_comparisons": comparisons,
        "geometry_buckets": keypoint_geometry_buckets(episodes),
    }


def _keypoint_pair_metrics(
    episodes: list[dict[str, Any]],
    baseline: KeypointVariant,
    candidate: KeypointVariant,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    base_success = np.array([1.0 if ep["models"][baseline]["success"] else 0.0 for ep in episodes], dtype=float)
    cand_success = np.array([1.0 if ep["models"][candidate]["success"] else 0.0 for ep in episodes], dtype=float)
    delta_pos = np.array(
        [ep["models"][candidate]["final_position_error"] - ep["models"][baseline]["final_position_error"] for ep in episodes],
        dtype=float,
    )
    delta_yaw = np.array(
        [
            ep["models"][candidate]["final_orientation_error"] - ep["models"][baseline]["final_orientation_error"]
            for ep in episodes
        ],
        dtype=float,
    )
    counts = {
        "both_success": int(((base_success == 1.0) & (cand_success == 1.0)).sum()),
        "baseline_only_success": int(((base_success == 1.0) & (cand_success == 0.0)).sum()),
        "candidate_only_success": int(((base_success == 0.0) & (cand_success == 1.0)).sum()),
        "both_fail": int(((base_success == 0.0) & (cand_success == 0.0)).sum()),
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "contingency": counts,
        "mcnemar_exact_p": _mcnemar_exact_p(counts["baseline_only_success"], counts["candidate_only_success"]),
        "delta_success_candidate_minus_baseline": _mean_median_ci(cand_success - base_success, bootstrap_samples, seed),
        "delta_final_position_error_candidate_minus_baseline": _mean_median_ci(delta_pos, bootstrap_samples, seed + 10),
        "delta_final_orientation_error_candidate_minus_baseline": _mean_median_ci(delta_yaw, bootstrap_samples, seed + 20),
    }


def keypoint_geometry_buckets(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.array([_keypoint_difficulty_value(ep) for ep in episodes], dtype=float)
    if len(values) == 0:
        return {}
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    buckets = {
        "small": values <= q1,
        "medium": (values > q1) & (values <= q2),
        "large": values > q2,
    }
    rows = {}
    for name, mask in buckets.items():
        selected = [ep for ep, keep in zip(episodes, mask, strict=True) if bool(keep)]
        rows[name] = {
            "num_episodes": len(selected),
            "range": [float(values[mask].min()) if mask.any() else 0.0, float(values[mask].max()) if mask.any() else 0.0],
            "models": {
                variant: summarize_keypoint_rollouts([ep["models"][variant] for ep in selected])
                for variant in ("single_part_node", "keypoint_concat", "keypoint_graph")
            },
        }
    return {"quantiles": [float(q1), float(q2)], "buckets": rows}


def _keypoint_difficulty_value(episode: dict[str, Any]) -> float:
    split = episode["spec"]["split"]
    if split == "length_ood":
        return float(episode["spec"]["length"])
    if split == "orientation_ood":
        return abs(float(episode["spec"]["part_yaw_local"]))
    if split == "shape_ood":
        return abs(float(episode["spec"]["curvature"]))
    return float(episode["spec"]["global_local_yaw_abs_diff"])


def keypoint_worst_cases(episodes: list[dict[str, Any]], model_name: KeypointVariant) -> list[dict[str, Any]]:
    sorted_eps = sorted(
        episodes,
        key=lambda ep: float(ep["models"][model_name]["final_position_error"] + ep["models"][model_name]["final_orientation_error"]),
        reverse=True,
    )
    return [
        {
            "episode_id": ep["episode_id"],
            "episode": ep,
            "spec": ep["spec"],
            "model": model_name,
            "final_position_error": ep["models"][model_name]["final_position_error"],
            "final_orientation_error": ep["models"][model_name]["final_orientation_error"],
        }
        for ep in sorted_eps[:10]
    ]


def aggregate_keypoint_experiment(seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    variants: tuple[KeypointVariant, ...] = ("single_part_node", "keypoint_concat", "keypoint_graph")
    splits: tuple[KeypointEvalSplit, ...] = ("iid", "length_ood", "orientation_ood", "shape_ood")
    aggregate: dict[str, Any] = {"seeds": seed_payloads, "splits": {}, "efficiency": {}, "permutation": {}}
    for split in splits:
        aggregate["splits"][split] = {}
        for variant in variants:
            rows = [seed_item["eval"]["splits"][split]["models"][variant] for seed_item in seed_payloads]
            aggregate["splits"][split][variant] = _aggregate_metric_dict(rows)
    for variant in variants:
        aggregate["efficiency"][variant] = {
            "params": _mean_std([float(seed_item["eval"]["parameter_count"][variant]) for seed_item in seed_payloads]),
            "forward_latency_ms": _mean_std(
                [float(seed_item["eval"]["splits"]["iid"]["models"][variant]["forward_latency_ms"]) for seed_item in seed_payloads]
            ),
            "policy_latency_ms": _mean_std(
                [float(seed_item["eval"]["splits"]["iid"]["models"][variant]["policy_latency_ms"]) for seed_item in seed_payloads]
            ),
        }
    for variant in ("keypoint_concat", "keypoint_graph"):
        normal_success = [seed_item["eval"]["permutation"][variant]["normal"]["success_rate"] for seed_item in seed_payloads]
        perm_success = [seed_item["eval"]["permutation"][variant]["permuted"]["success_rate"] for seed_item in seed_payloads]
        normal_dist = [seed_item["eval"]["permutation"][variant]["normal"]["final_position_error"] for seed_item in seed_payloads]
        perm_dist = [seed_item["eval"]["permutation"][variant]["permuted"]["final_position_error"] for seed_item in seed_payloads]
        aggregate["permutation"][variant] = {
            "normal_success": _mean_std(normal_success),
            "permuted_success": _mean_std(perm_success),
            "delta_success": _mean_std([p - n for p, n in zip(perm_success, normal_success, strict=True)]),
            "normal_final_position_error": _mean_std(normal_dist),
            "permuted_final_position_error": _mean_std(perm_dist),
            "delta_final_position_error": _mean_std([p - n for p, n in zip(perm_dist, normal_dist, strict=True)]),
        }
    return aggregate


def _aggregate_metric_dict(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = rows[0].keys()
    return {key: _mean_std([float(row[key]) for row in rows]) for key in keys}


def keypoint_summary_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Sparse Keypoint Graph Feasibility", ""]
    lines.append("## Success / Final Error")
    lines.append("")
    lines.append("| Model | IID success | Length OOD | Orientation OOD | Shape OOD |")
    lines.append("|---|---:|---:|---:|---:|")
    labels = {
        "single_part_node": "Single Part Node",
        "keypoint_concat": "Keypoint Concat",
        "keypoint_graph": "Keypoint Graph",
    }
    for variant in ("single_part_node", "keypoint_concat", "keypoint_graph"):
        cells = [labels[variant]]
        for split in ("iid", "length_ood", "orientation_ood", "shape_ood"):
            metric = payload["splits"][split][variant]["success_rate"]
            dist = payload["splits"][split][variant]["final_position_error"]
            yaw = payload["splits"][split][variant]["final_orientation_error"]
            cells.append(f"{metric['mean']:.3f} +/- {metric['std']:.3f}; pos {dist['mean']:.3f}; yaw {yaw['mean']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Efficiency", ""])
    lines.append("| Model | Params | Forward ms | Policy ms |")
    lines.append("|---|---:|---:|---:|")
    for variant in ("single_part_node", "keypoint_concat", "keypoint_graph"):
        eff = payload["efficiency"][variant]
        lines.append(
            f"| {labels[variant]} | {eff['params']['mean']:.0f} | {eff['forward_latency_ms']['mean']:.4f} | "
            f"{eff['policy_latency_ms']['mean']:.4f} |"
        )
    lines.extend(["", "## Permutation Sanity Check", ""])
    lines.append("| Model | Normal success | Permuted success | Delta success | Normal pos | Permuted pos |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for variant in ("keypoint_concat", "keypoint_graph"):
        row = payload["permutation"][variant]
        lines.append(
            f"| {labels[variant]} | {row['normal_success']['mean']:.3f} | {row['permuted_success']['mean']:.3f} | "
            f"{row['delta_success']['mean']:.3f} | {row['normal_final_position_error']['mean']:.3f} | "
            f"{row['permuted_final_position_error']['mean']:.3f} |"
        )
    pooled = payload.get("pooled_paired_evaluation")
    if pooled:
        lines.extend(["", "## Pooled Paired Comparisons", ""])
        lines.append("| Split | Comparison | Contingency b/c | Delta success | Delta pos | Delta yaw |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for split in ("iid", "length_ood", "orientation_ood", "shape_ood"):
            comparisons = pooled[split]["paired_comparisons"]
            for comparison_name in (
                "single_part_node_vs_keypoint_concat",
                "keypoint_concat_vs_keypoint_graph",
                "single_part_node_vs_keypoint_graph",
            ):
                row = comparisons[comparison_name]
                counts = row["contingency"]
                delta_success = row["delta_success_candidate_minus_baseline"]
                delta_pos = row["delta_final_position_error_candidate_minus_baseline"]
                delta_yaw = row["delta_final_orientation_error_candidate_minus_baseline"]
                lines.append(
                    f"| {split} | {comparison_name} | "
                    f"{counts['baseline_only_success']}/{counts['candidate_only_success']} | "
                    f"{delta_success['mean']:.3f} | {delta_pos['mean']:.3f} | {delta_yaw['mean']:.3f} |"
                )
    lines.extend(["", "## Geometry Statistics", ""])
    for seed, stats in payload.get("geometry_statistics", {}).items():
        shape = stats["shape_ood"]
        lines.append(
            f"- {seed} shape_ood yaw diff mean={shape['global_local_yaw_abs_diff_mean']:.3f}, "
            f"p90={shape['global_local_yaw_abs_diff_p90']:.3f}, curvature={shape['abs_curvature_mean']:.3f}"
        )
    lines.append("")
    return "\n".join(lines)


def keypoint_diagnostics_summary_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Sparse Keypoint Graph Diagnostics", ""]
    lines.append("## Target Decode")
    lines.append("")
    lines.append("| Variant | Train pos | Train yaw | Train axis yaw | Val pos | Val yaw | Val axis yaw |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for variant in ("keypoint_concat", "keypoint_graph"):
        row = payload["target_decode"][variant]
        train = row["train"]
        val = row["val"]
        lines.append(
            f"| {variant} | {train['position_l2']:.5f} | {train['directed_yaw_abs_error']:.5f} | "
            f"{train['axis_yaw_abs_error']:.5f} | {val['position_l2']:.5f} | "
            f"{val['directed_yaw_abs_error']:.5f} | {val['axis_yaw_abs_error']:.5f} |"
        )
    lines.extend(["", "## 100-Episode Action Overfit", ""])
    lines.append("| Variant | Samples | Initial xyz | Initial yaw | Final xyz | Final yaw |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for variant in ("keypoint_concat", "keypoint_graph"):
        row = payload["action_overfit"][variant]
        initial = row["initial"]
        final = row["final"]
        lines.append(
            f"| {variant} | {row['samples']} | {initial['translation_l2']:.5f} | "
            f"{initial['yaw_abs_error']:.5f} | {final['translation_l2']:.5f} | {final['yaw_abs_error']:.5f} |"
        )
    lines.extend(["", "## CPU vs MPS", ""])
    consistency = payload["cpu_mps_consistency"]
    if consistency.get("skipped"):
        lines.append(f"- skipped: {consistency.get('reason')}")
    else:
        lines.append("| Variant | Samples | Max abs diff | Mean abs diff |")
        lines.append("|---|---:|---:|---:|")
        for variant in ("keypoint_concat", "keypoint_graph"):
            row = consistency[variant]
            if row.get("skipped"):
                lines.append(f"| {variant} | 0 | skipped | skipped |")
            else:
                lines.append(
                    f"| {variant} | {row['samples']} | {row['max_abs_diff']:.8f} | {row['mean_abs_diff']:.8f} |"
                )
    lines.append("")
    return "\n".join(lines)


def plot_keypoint_episode(episode: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    keypoints = np.array(episode["spec"]["keypoints_world"], dtype=float)
    target = np.array(episode["spec"]["target_position"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(keypoints[:, 0], keypoints[:, 2], "-*", color="black", label="Part keypoints")
    ax.scatter([target[0]], [target[2]], s=110, marker="*", label="Local anchor")
    for variant, result in episode["models"].items():
        traj = np.array(result["trajectory"], dtype=float)
        ax.plot(traj[:, 0], traj[:, 2], "-o", markersize=2, linewidth=1, label=variant)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _rate_with_ci(values: np.ndarray, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    return {
        "rate": float(values.mean()) if values.size else 0.0,
        "ci95": _bootstrap_mean_ci(values, bootstrap_samples, seed),
    }


def _mean_median_ci(values: np.ndarray, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "ci95": [0.0, 0.0]}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": _bootstrap_mean_ci(values, bootstrap_samples, seed),
    }


def _bootstrap_mean_ci(values: np.ndarray, bootstrap_samples: int, seed: int) -> list[float]:
    if values.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples, dtype=float)
    for idx in range(bootstrap_samples):
        sample_ids = rng.integers(0, values.size, size=values.size)
        means[idx] = float(values[sample_ids].mean())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _mcnemar_exact_p(concat_only: int, node_only: int) -> float:
    n = int(concat_only + node_only)
    if n == 0:
        return 1.0
    lo = min(int(concat_only), int(node_only))
    log_half = math.log(0.5)
    probability = 0.0
    for k in range(lo + 1):
        log_p = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + n * log_half
        probability += math.exp(log_p)
    return float(min(1.0, 2.0 * probability))


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _fmt_optional(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "inf"
    return f"{value:.{digits}f}"


def _part_offset_distance(episode: dict[str, Any]) -> float:
    part_local = np.array(episode["spec"]["part_local"], dtype=float)
    return float(np.linalg.norm(part_local[[0, 2]]))


def _fixed_keypoint_permutation(keypoints: int) -> list[int]:
    if keypoints == 5:
        return [3, 0, 4, 1, 2]
    rng = np.random.default_rng(12345)
    return [int(value) for value in rng.permutation(keypoints)]


def _trajectory_length(trajectory: list[list[float]]) -> float:
    if len(trajectory) < 2:
        return 0.0
    points = np.array(trajectory, dtype=float)
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _save_frame(env: MujocoManipulatorEnv, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, env.render_frame())


def _mean(values: list[float | int | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    return float(sum(clean) / max(len(clean), 1))


def _mean_std(values: list[float]) -> dict[str, float]:
    clean = [float(value) for value in values]
    if not clean:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
    }
