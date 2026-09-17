import torch
import numpy as np

from vla_gnn_recurrent.graph.manipulation_graph_builder import ManipulationGraphBuilder
from vla_gnn_recurrent.models.recurrent_controller import RecurrentController
from vla_gnn_recurrent.sim.mujoco_env import MujocoManipulatorEnv, MujocoReachConfig
from vla_gnn_recurrent.sim.pick_expert import ScriptedPickConfig, run_scripted_pick_episode
from vla_gnn_recurrent.sim.stale_observation import StaleManipulationObserver
from vla_gnn_recurrent.training.mujoco_reaching import run_mujoco_reaching_episode


def test_mujoco_reset_is_deterministic_for_same_seed() -> None:
    env_a = MujocoManipulatorEnv(MujocoReachConfig(max_steps=4), seed=123)
    env_b = MujocoManipulatorEnv(MujocoReachConfig(max_steps=4), seed=123)

    obs_a = env_a.reset()
    obs_b = env_b.reset()

    assert torch.allclose(obs_a.robot.joint_positions, obs_b.robot.joint_positions)
    assert torch.allclose(obs_a.robot.ee_position, obs_b.robot.ee_position)
    assert torch.allclose(obs_a.target.position, obs_b.target.position)


def test_mujoco_observation_contains_joint_ee_and_object_state() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=4), seed=3)
    obs = env.reset()

    assert obs.robot.joint_positions.shape == (4,)
    assert obs.robot.joint_velocities.shape == (4,)
    assert obs.robot.ee_position.shape == (3,)
    assert obs.object is not None
    assert obs.object.position.shape == (3,)


def test_delta_ee_is_bounded_and_gripper_command_is_clipped() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=4, max_delta_ee=0.02), seed=4)
    obs = env.reset()
    start = obs.robot.ee_position.clone()
    next_obs, _, _, _ = env.step_delta_ee(torch.tensor([1.0, 0.0, 1.0]), gripper=5.0)

    assert float((next_obs.robot.ee_position - start).norm()) <= 0.08
    assert float(env.data.ctrl[env.gripper_actuator_id]) <= 1.0


def test_physical_arm_control_does_not_teleport_to_ik_target() -> None:
    env = MujocoManipulatorEnv(
        MujocoReachConfig(max_steps=4, control_substeps=1, max_delta_ee=0.04, pick_scene=True),
        seed=14,
    )
    env.reset_pick_scene(object_x=-0.01, randomize_robot=False)
    start_q = env.data.qpos[env.arm_qpos_ids].copy()
    target = env.data.site_xpos[env.ee_site_id].copy() + np.array([0.02, 0.0, -0.02])
    ik = env.solve_ik(target)

    env.step_toward_ee_target(target, gripper=0.0)
    after_q = env.data.qpos[env.arm_qpos_ids].copy()

    assert not env.config.kinematic_joint_control
    assert np.allclose(env.last_arm_qpos_before_control, start_q)
    assert not np.allclose(after_q, ik.qpos.numpy(), atol=1e-3)
    assert np.linalg.norm(after_q - start_q) > 0.0


def test_pick_scene_xml_generation_recovers_from_empty_file() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=4, pick_scene=True), seed=140)
    path = env._pick_scene_xml_path()
    path.write_text("", encoding="utf-8")

    recovered = env._pick_scene_xml_path()

    assert recovered == path
    assert recovered.stat().st_size > 0
    assert "<mujoco" in recovered.read_text(encoding="utf-8")


def test_actuator_command_respects_control_ranges() -> None:
    env = MujocoManipulatorEnv(
        MujocoReachConfig(max_steps=4, control_substeps=3, max_delta_ee=0.2, motor_ctrl_limit=1.2),
        seed=15,
    )
    env.reset(randomize_robot=False)

    env.step_delta_ee(torch.tensor([2.0, 0.0, -2.0]), gripper=4.0)
    ctrl = env.data.ctrl[env.arm_actuator_ids]
    ranges = env.model.actuator_ctrlrange[env.arm_actuator_ids]

    assert np.all(ctrl >= ranges[:, 0] - 1e-6)
    assert np.all(ctrl <= ranges[:, 1] + 1e-6)
    assert float(env.data.ctrl[env.gripper_actuator_id]) == 1.0


def test_ee_converges_to_reachable_physical_target() -> None:
    env = MujocoManipulatorEnv(
        MujocoReachConfig(max_steps=24, control_substeps=20, max_delta_ee=0.04, pick_scene=True),
        seed=16,
    )
    env.reset(randomize_robot=False)
    target = env.data.site_xpos[env.ee_site_id].copy() + np.array([0.06, 0.0, -0.08])
    start_error = np.linalg.norm(env.data.site_xpos[env.ee_site_id][[0, 2]] - target[[0, 2]])

    for _ in range(20):
        env.step_toward_ee_target(target, gripper=0.0)
    final_error = np.linalg.norm(env.data.site_xpos[env.ee_site_id][[0, 2]] - target[[0, 2]])

    assert final_error < start_error
    assert final_error < 0.04


def test_ik_reachable_and_unreachable_handling() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=4), seed=5)
    obs = env.reset()

    reachable = env.solve_ik(obs.target.position)
    unreachable = env.solve_ik(torch.tensor([5.0, 0.0, 5.0]))

    assert reachable.success
    assert reachable.final_error < 0.02
    assert unreachable.qpos.shape == (4,)
    assert torch.isfinite(unreachable.qpos).all()


def test_joint_limits_are_enforced_by_ik() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=4), seed=6)
    env.reset()
    result = env.solve_ik(torch.tensor([5.0, 0.0, 5.0]))
    q = result.qpos.numpy()
    ranges = env.arm_joint_ranges
    limited = env.model.jnt_limited[env.arm_joint_ids].astype(bool)

    for idx, is_limited in enumerate(limited):
        if is_limited:
            assert ranges[idx, 0] - 1e-5 <= q[idx] <= ranges[idx, 1] + 1e-5


def test_manipulation_graph_node_edges_and_relative_geometry() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=4), seed=7)
    obs = env.reset()
    graph = ManipulationGraphBuilder(task="reach").build(obs)

    assert "joint" in graph.node_types
    assert "end_effector" in graph.node_types
    assert "gripper" in graph.node_types
    assert "destination" in graph.node_types
    assert "task" in graph.node_types
    assert graph.num_edges > 0
    ee_to_target = graph.node_features[graph.target_node_index, 0:3] - graph.node_features[graph.ee_node_index, 0:3]
    spatial_rows = [
        idx
        for idx in range(graph.num_edges)
        if graph.edge_index[0, idx] == graph.ee_node_index and graph.edge_index[1, idx] == graph.target_node_index
    ]
    assert spatial_rows
    assert torch.allclose(graph.edge_features[spatial_rows[0], 0:3], ee_to_target)


def test_stale_observer_does_not_leak_true_target_between_reanchors() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=8), seed=8)
    env.reset()
    observer = StaleManipulationObserver(observation_interval=4)
    obs0 = observer.observe(env.observe())
    env.move_target(env.sample_perturbed_target())
    observer.notify_true_event(env.step_count)
    obs_stale = observer.observe(env.observe())

    assert torch.allclose(obs_stale.target.position, obs0.target.position)
    assert not torch.allclose(obs_stale.true_target.position, obs0.target.position)
    assert not obs_stale.observable_event


def test_stale_observer_event_becomes_observable_on_reanchor() -> None:
    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=8), seed=9)
    env.reset()
    observer = StaleManipulationObserver(observation_interval=4)
    observer.observe(env.observe())
    env.move_target(env.sample_perturbed_target())
    observer.notify_true_event(env.step_count)
    for _ in range(4):
        env.step_delta_ee(torch.zeros(3))
    obs = observer.observe(env.observe())

    assert obs.reanchored
    assert obs.observable_event


def test_mujoco_state_corrector_runs_once_at_observable_event() -> None:
    class RecordingCorrector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, hidden: torch.Tensor, graph_embedding: torch.Tensor):
            self.calls += 1
            return hidden, torch.ones_like(hidden)

    env = MujocoManipulatorEnv(MujocoReachConfig(max_steps=8), seed=10)
    model = RecurrentController(max_step=env.config.max_delta_ee, action_prior="none")
    corrector = RecordingCorrector()

    result = run_mujoco_reaching_episode(
        model=model,
        model_kind="graph_recurrent",
        env=env,
        graph_builder=ManipulationGraphBuilder(task="reach"),
        device=torch.device("cpu"),
        observation_interval=4,
        perturb=True,
        perturb_step=2,
        corrector=corrector,  # type: ignore[arg-type]
    )

    assert result["perturb_true_step"] == 2
    assert result["observable_event_step"] == 4
    assert corrector.calls == 1


def test_gripper_open_close_and_contact_diagnostics() -> None:
    env = MujocoManipulatorEnv(
        MujocoReachConfig(max_steps=12, control_substeps=40, pick_scene=True),
        seed=17,
    )
    env.reset_pick_scene(object_x=-0.01, randomize_robot=False)
    open_q = env.data.qpos[env.finger_qpos_ids].copy()

    for _ in range(8):
        env.step_delta_ee(torch.zeros(3), gripper=1.0)
    closed_q = env.data.qpos[env.finger_qpos_ids].copy()

    assert np.mean(open_q) < 0.0
    assert np.mean(closed_q) > np.mean(open_q)
    assert env.data.ctrl[env.gripper_actuator_id] == 1.0


def test_scripted_pick_contacts_object_before_lift_and_lifts_physically() -> None:
    env = MujocoManipulatorEnv(
        MujocoReachConfig(max_steps=220, control_substeps=40, max_delta_ee=0.045, pick_scene=True),
        seed=18,
    )
    result = run_scripted_pick_episode(
        env,
        ScriptedPickConfig(max_steps=220, render=False),
        object_x=-0.01,
        render=False,
    )

    first_contact = next(
        idx for idx, row in enumerate(result["records"]) if row["left_contact"] or row["right_contact"]
    )
    first_lift = next(idx for idx, row in enumerate(result["records"]) if row["object_height_delta"] > 0.02)

    assert result["grasp_success"]
    assert result["lift_success"]
    assert first_contact < first_lift
    assert result["records"][-1]["object_height_delta"] > 0.10


def test_scripted_phase_order_and_failed_grasp_not_counted_success() -> None:
    env = MujocoManipulatorEnv(
        MujocoReachConfig(max_steps=160, control_substeps=40, max_delta_ee=0.045, pick_scene=True),
        seed=19,
    )
    success = run_scripted_pick_episode(
        env,
        ScriptedPickConfig(max_steps=220, render=False),
        object_x=-0.01,
        render=False,
    )
    failure = run_scripted_pick_episode(
        env,
        ScriptedPickConfig(
            max_steps=160,
            render=False,
            adaptive_grasp_offset=False,
            grasp_offset=(0.12, 0.0, 0.0),
        ),
        object_x=-0.01,
        render=False,
    )

    assert success["phase_order"] == ["PRE_GRASP", "ALIGN", "CLOSE", "VERIFY", "LIFT", "SUCCESS"]
    assert not failure["grasp_success"]
    assert not failure["lift_success"]
    assert failure["failure_reason"] == "insufficient_contact_or_alignment_error"
