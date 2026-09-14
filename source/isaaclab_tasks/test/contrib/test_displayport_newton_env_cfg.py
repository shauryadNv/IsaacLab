# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton DisplayPort insertion production configuration."""

import math
from dataclasses import fields
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import FeatherPGSSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg, NewtonSDFCollisionCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg

import isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s  # noqa: F401
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s import (
    task_space_newton_ros_inference_env_cfg as newton_ros_cfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.agents.rsl_rl_ppo_cfg import (
    Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg,
    Rizon4sGravDisplayportInsertionRNNPPORunnerCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_pos_env_cfg import (
    set_finger_joint_pos_grav,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.task_space_env_cfg import (
    Rizon4sTaskSpaceDisplayportInsertionEnvCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.task_space_newton_env_cfg import (
    DisplayportNewtonPhysicsCfg,
    Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg,
    Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg_PLAY,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    PLUG_GOAL_ROT,
    PLUG_INSERTION_OFFSET,
    SOCKET_INSERTION_OFFSET,
)
from isaaclab_tasks.contrib.deploy.mdp import DeployOperationalSpaceControllerActionCfg
from isaaclab_tasks.contrib.deploy.mdp import events as deploy_events
from isaaclab_tasks.contrib.deploy.mdp.actions import _pose_rel_action_extra
from isaaclab_tasks.contrib.deploy.mdp.events import (
    _body_link_jacobian_for_ik,
    reset_plug_at_goal_curriculum,
    set_robot_to_object_grasp_pose,
)
from isaaclab_tasks.contrib.deploy.mdp.noise_models import ResetSampledConstantNoiseModelCfg
from isaaclab_tasks.contrib.deploy.mdp.observations import eef_pos_w, rigid_object_pos_w
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
_ENV_ENTRY_POINT = "isaaclab_tasks.contrib.deploy.cable_insertion.insertion_env:DisplayportInsertionEnv"
_CFG_PACKAGE = "isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s"
_AGENT_MODULE = f"{_CFG_PACKAGE}.agents.rsl_rl_ppo_cfg"


def _observation_term_names(group: object) -> list[str]:
    """Return observation terms in their concatenation order."""
    return [field.name for field in fields(group) if isinstance(getattr(group, field.name), ObservationTermCfg)]


@pytest.mark.parametrize(
    ("is_fixed_base", "num_base_dofs", "expected_body_idx"),
    [
        pytest.param(True, 0, 2, id="fixed-base"),
        pytest.param(False, 6, 3, id="floating-base"),
    ],
)
def test_displayport_grasp_reset_selects_backend_neutral_jacobians(
    is_fixed_base: bool,
    num_base_dofs: int,
    expected_body_idx: int,
):
    """Reset IK must select the correct body and actuated columns from public data."""
    jacobians = torch.arange(3 * 5 * 6 * 10, dtype=torch.float32).reshape(3, 5, 6, 10)
    asset = SimpleNamespace(
        is_fixed_base=is_fixed_base,
        num_base_dofs=num_base_dofs,
        data=SimpleNamespace(body_link_jacobian_w=SimpleNamespace(torch=jacobians)),
    )
    env_ids = torch.tensor([2, 0])

    selected = _body_link_jacobian_for_ik(asset, env_ids, body_idx=3)

    torch.testing.assert_close(selected, jacobians[env_ids, expected_body_idx, :, num_base_dofs:])


def test_displayport_grasp_reset_holds_randomized_target_during_ik(monkeypatch: pytest.MonkeyPatch):
    """A reset must sample one grasp offset and hold it fixed across all IK iterations."""
    num_envs = 2
    num_arm_joints = 7
    identity_quaternions = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    joint_limits = torch.tensor([[[-math.pi, math.pi]] * num_arm_joints] * num_envs)

    robot_data = SimpleNamespace(
        joint_pos=wp.from_torch(torch.zeros(num_envs, num_arm_joints)),
        joint_vel=wp.from_torch(torch.zeros(num_envs, num_arm_joints)),
        joint_pos_limits=wp.from_torch(joint_limits),
        body_pos_w=wp.from_torch(torch.zeros(num_envs, 1, 3)),
        body_quat_w=wp.from_torch(identity_quaternions.unsqueeze(1)),
    )
    robot = SimpleNamespace(
        data=robot_data,
        set_joint_position_target_index=lambda **_kwargs: None,
        set_joint_velocity_target_index=lambda **_kwargs: None,
        write_joint_position_to_sim_index=lambda **_kwargs: None,
        write_joint_velocity_to_sim_index=lambda **_kwargs: None,
    )
    plug = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=wp.from_torch(torch.zeros(num_envs, 3)),
            root_link_quat_w=wp.from_torch(identity_quaternions),
        ),
        write_root_pose_to_sim=lambda *_args, **_kwargs: None,
        write_root_velocity_to_sim=lambda *_args, **_kwargs: None,
    )
    env = SimpleNamespace(device="cpu", scene={"dp_plug": plug})
    env_ids = torch.arange(num_envs)

    term = object.__new__(set_robot_to_object_grasp_pose)
    term.robot_asset = robot
    term.target_object_name = "dp_plug"
    term.grasp_offsets_buffer = torch.zeros(num_envs, 3)
    term.grasp_offset_tensor = torch.zeros(3)
    term.grasp_rot_offset_tensor = identity_quaternions
    term.eef_idx = 0
    term.num_arm_joints = num_arm_joints
    term.all_joints = list(range(num_arm_joints))
    term.finger_joints = []
    term.joint_name_to_idx = {}
    term.hand_hold_width = 0.0
    term.hand_close_width = 0.0
    term.gripper_joint_setter_func = lambda *_args, **_kwargs: None

    sample_count = 0
    ik_targets: list[torch.Tensor] = []

    def sample_uniform_once(
        _lower: torch.Tensor,
        _upper: torch.Tensor,
        size: tuple[int, int],
        device: str,
    ) -> torch.Tensor:
        nonlocal sample_count
        sample_count += 1
        return torch.full(size, float(sample_count), device=device)

    def record_pose_target(**kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        target = kwargs["ctrl_target_fingertip_midpoint_pos"]
        ik_targets.append(target.clone())
        return torch.ones_like(target), torch.zeros_like(target)

    monkeypatch.setattr(deploy_events.math_utils, "sample_uniform", sample_uniform_once)
    monkeypatch.setattr(deploy_events.fc, "get_pose_error", record_pose_target)
    monkeypatch.setattr(
        deploy_events.fc,
        "_get_delta_dof_pos",
        lambda **_kwargs: torch.zeros(num_envs, num_arm_joints),
    )
    monkeypatch.setattr(
        deploy_events,
        "_body_link_jacobian_for_ik",
        lambda *_args, **_kwargs: torch.zeros(num_envs, 6, num_arm_joints),
    )

    term(
        env,
        env_ids,
        max_iterations=3,
        pos_randomization_range={"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
    )

    assert sample_count == 1
    assert len(ik_targets) == 3
    for target in ik_targets[1:]:
        torch.testing.assert_close(target, ik_targets[0])


def test_displayport_grav_reset_sets_mimic_joints_by_name():
    """The reset must preserve Grav mimic signs when Newton reorders the joints."""
    joint_names = [
        "finger_joint",
        "left_outer_finger_joint",
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_outer_finger_joint",
    ]
    finger_joints = list(range(7, 13))
    joint_name_to_idx = dict(zip(joint_names, finger_joints))
    joint_pos = torch.full((2, 13), 42.0)

    set_finger_joint_pos_grav(
        joint_pos,
        [0],
        finger_joints,
        -0.1,
        joint_name_to_idx,
    )

    expected_gearing = {
        "finger_joint": 1.0,
        "left_inner_knuckle_joint": 1.0,
        "right_inner_knuckle_joint": 1.0,
        "right_outer_knuckle_joint": 1.0,
        "left_outer_finger_joint": -1.0,
        "right_outer_finger_joint": -1.0,
    }
    for joint_name, gearing in expected_gearing.items():
        assert joint_pos[0, joint_name_to_idx[joint_name]] == pytest.approx(-0.1 * gearing)
    torch.testing.assert_close(joint_pos[1], torch.full((13,), 42.0))


def test_rigid_object_offset_is_rotated_and_cached_as_a_batch_view():
    """Object offsets must compose in the local frame without per-step batching."""

    class Scene(dict):
        pass

    positions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    quaternions = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]])
    scene = Scene(
        dp_socket=SimpleNamespace(
            data=SimpleNamespace(root_pos_w=wp.from_torch(positions), root_quat_w=wp.from_torch(quaternions))
        )
    )
    scene.env_origins = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    env = SimpleNamespace(device="cpu", num_envs=2, scene=scene)
    cfg = ObservationTermCfg(
        func=rigid_object_pos_w,
        params={"asset_cfg": SceneEntityCfg("dp_socket"), "offset": [1.0, 0.0, 0.0]},
    )
    term = rigid_object_pos_w(cfg, env)
    offset_batch_data_ptr = term._offset_batch.data_ptr()

    observed = term(env)
    observed_again = term(env)

    expected = torch.tensor([[1.5, 1.5, 2.5], [3.0, 5.0, 5.0]])
    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(observed_again, expected)
    assert term._offset_batch.data_ptr() == offset_batch_data_ptr
    assert term._offset_batch.stride()[0] == 0


def test_zero_eef_offset_skips_orientation_lookup():
    """A static zero offset must return body position without entering rotation code."""

    class Scene(dict):
        pass

    body_positions = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    robot = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=wp.from_torch(body_positions)),
        find_bodies=lambda _name: ([0], ["flange"]),
    )
    scene = Scene(robot=robot)
    scene.env_origins = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    env = SimpleNamespace(device="cpu", num_envs=2, scene=scene)
    cfg = ObservationTermCfg(
        func=eef_pos_w,
        params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "flange", "offset": [0.0, 0.0, 0.0]},
    )

    observed = eef_pos_w(cfg, env)(env)

    torch.testing.assert_close(observed, torch.tensor([[0.5, 1.5, 2.5], [3.0, 4.0, 5.0]]))


def test_displayport_newton_tasks_route_to_dedicated_configs():
    """Newton registrations must not replace or redirect existing PhysX tasks."""
    expected = {
        "Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-v0": (
            f"{_CFG_PACKAGE}.task_space_newton_env_cfg:Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg",
            f"{_AGENT_MODULE}:Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg",
        ),
        "Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-Play-v0": (
            f"{_CFG_PACKAGE}.task_space_newton_env_cfg:Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg_PLAY",
            f"{_AGENT_MODULE}:Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg",
        ),
        "Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference-v0": (
            f"{_CFG_PACKAGE}.task_space_newton_ros_inference_env_cfg:"
            "Rizon4sTaskSpaceNewtonDisplayportInsertionROSInferenceEnvCfg",
            f"{_AGENT_MODULE}:Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg",
        ),
        "Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-v0": (
            f"{_CFG_PACKAGE}.task_space_env_cfg:Rizon4sTaskSpaceDisplayportInsertionEnvCfg",
            f"{_AGENT_MODULE}:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
        ),
        "Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Play-v0": (
            f"{_CFG_PACKAGE}.task_space_env_cfg:Rizon4sTaskSpaceDisplayportInsertionEnvCfg_PLAY",
            f"{_AGENT_MODULE}:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
        ),
        "Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference-v0": (
            f"{_CFG_PACKAGE}.task_space_ros_inference_env_cfg:Rizon4sTaskSpaceDisplayportInsertionROSInferenceEnvCfg",
            f"{_AGENT_MODULE}:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
        ),
    }

    for task_id, (env_cfg_entry_point, runner_cfg_entry_point) in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == _ENV_ENTRY_POINT
        assert spec.disable_env_checker
        assert spec.kwargs["env_cfg_entry_point"] == env_cfg_entry_point
        assert spec.kwargs["rsl_rl_cfg_entry_point"] == runner_cfg_entry_point


@pytest.mark.parametrize(
    ("task_id", "expected_num_envs"),
    [
        pytest.param("Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-v0", 256, id="training"),
        pytest.param("Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-Play-v0", 50, id="play"),
        pytest.param(
            "Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference-v0",
            256,
            id="ros-inference",
        ),
    ],
)
def test_registered_displayport_newton_tasks_resolve_compatible_safe_defaults(task_id: str, expected_num_envs: int):
    """Every Newton variant must resolve to Newton with its supported shard size."""
    cfg = resolve_presets(load_cfg_from_registry(task_id, "env_cfg_entry_point"))
    runner = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")

    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.actions.arm_action, DeployOperationalSpaceControllerActionCfg)
    assert cfg.scene.num_envs == expected_num_envs
    assert cfg.sim.physics.collision_cfg.max_triangle_pairs == 2**25
    assert runner.seed == 123
    assert runner.clip_actions == pytest.approx(1.0)
    cfg.validate()


def test_displayport_newton_timing_solver_and_point_sdf_assets():
    """Newton defaults must retain the functional timing and point-SDF contracts."""
    cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()

    assert cfg.scene.num_envs == 256
    assert isinstance(cfg.sim.physics, DisplayportNewtonPhysicsCfg)

    cfg = resolve_presets(cfg)
    assert cfg.sim.dt == pytest.approx(0.01)
    assert cfg.decimation == 3
    assert cfg.sim.render_interval == cfg.decimation

    physics = cfg.sim.physics
    assert isinstance(physics, NewtonCfg)
    assert physics.num_substeps == 20
    assert physics.collision_decimation == 10
    assert physics.default_shape_cfg.gap == pytest.approx(0.005)
    assert physics.num_substeps / cfg.sim.dt == pytest.approx(2000.0)
    assert physics.num_substeps / cfg.sim.dt / physics.collision_decimation == pytest.approx(200.0)
    assert 1.0 / (cfg.sim.dt * cfg.decimation) == pytest.approx(100.0 / 3.0)

    solver = physics.solver_cfg
    assert isinstance(solver, MJWarpSolverCfg)
    assert solver.solver == "newton"
    assert solver.integrator == "implicitfast"
    assert solver.use_mujoco_contacts is False

    collision = physics.collision_cfg
    assert collision is not None
    assert collision.reduce_contacts is True
    assert collision.max_triangle_pairs == 2**25

    assert cfg.scene.dp_plug.spawn.usd_path.endswith("/displayport_plug.usd")
    assert cfg.scene.dp_socket.spawn.usd_path.endswith("/displayport_socket_no_protrusions.usd")
    plug_sdf_paths = ("/collision_mesh",)
    socket_sdf_paths = tuple(f"/tn__2584N111_DisplayportCord_jP/Body{body_id}/Mesh" for body_id in (5, 6, 8, 12, 13))
    collision_fragment_sets = (
        (cfg.scene.dp_plug.spawn.collision_props, 0.00001, -0.00005, plug_sdf_paths),
        (cfg.scene.dp_socket.spawn.collision_props, 0.0001, -0.0001, socket_sdf_paths),
    )
    for mapping, contact_offset, rest_offset, sdf_paths in collision_fragment_sets:
        assert set(mapping) == {"/.*", *sdf_paths}

        physx_fragments = mapping["/.*"]
        assert not any(isinstance(fragment, NewtonCollisionCfg) for fragment in physx_fragments)
        assert not any(isinstance(fragment, NewtonSDFCollisionCfg) for fragment in physx_fragments)
        physx = next(fragment for fragment in physx_fragments if isinstance(fragment, PhysxCollisionCfg))
        assert physx.contact_offset == pytest.approx(contact_offset)
        assert physx.rest_offset == pytest.approx(rest_offset)

        for prim_path in sdf_paths:
            fragments = mapping[prim_path]
            assert not any(isinstance(fragment, PhysxCollisionCfg) for fragment in fragments)
            collision = next(fragment for fragment in fragments if isinstance(fragment, NewtonCollisionCfg))
            sdf = next(fragment for fragment in fragments if isinstance(fragment, NewtonSDFCollisionCfg))
            assert collision.contact_margin == pytest.approx(0.0)
            assert collision.contact_gap == pytest.approx(0.005)
            assert sdf.sdf_max_resolution == 256
            assert sdf.sdf_narrow_band_inner == pytest.approx(-0.005)
            assert sdf.sdf_narrow_band_outer == pytest.approx(0.005)
            assert sdf.sdf_texture_format == "uint16"
            assert sdf.sdf_padding == pytest.approx(0.005)
            assert sdf.hydroelastic_enabled is False
            assert sdf.hydroelastic_stiffness == pytest.approx(1.0e8)


def test_displayport_newton_osc_abi_robot_and_gravity_settings():
    """OSC must command flange-relative pose deltas without double gravity compensation."""
    cfg = resolve_presets(Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg())
    action = cfg.actions.arm_action

    assert isinstance(action, DeployOperationalSpaceControllerActionCfg)
    assert action.asset_name == "robot"
    assert action.joint_names == _ARM_JOINTS
    assert action.body_name == "flange"
    assert action.body_offset is not None
    assert tuple(action.body_offset.pos) == pytest.approx((0.0, 0.0, 0.0))
    assert tuple(action.body_offset.rot) == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert tuple(action.position_scale) == pytest.approx((0.025, 0.025, 0.010))
    assert action.orientation_scale == pytest.approx(0.025)
    assert action.clip is None
    assert action.nullspace_joint_pos_target == "none"

    controller = action.controller_cfg
    assert isinstance(controller, OperationalSpaceControllerCfg)
    assert list(controller.target_types) == ["pose_rel"]
    assert controller.impedance_mode == "fixed"
    assert controller.inertial_dynamics_decoupling is True
    assert controller.partial_inertial_dynamics_decoupling is False
    assert controller.gravity_compensation is False
    assert tuple(controller.motion_stiffness_task) == pytest.approx((300.0, 300.0, 300.0, 30.0, 30.0, 30.0))
    assert tuple(controller.motion_damping_ratio_task) == pytest.approx((1.0,) * 6)
    assert controller.nullspace_control == "none"

    assert cfg.scene.robot.spawn.rigid_props.gravcomp == pytest.approx(1.0)
    assert cfg.scene.robot.spawn.joint_drive_props.actuatorgravcomp is False

    for actuator_name in ("shoulder", "elbow", "wrist"):
        actuator = cfg.scene.robot.actuators[actuator_name]
        assert actuator.stiffness == pytest.approx(0.0)
        assert actuator.damping == pytest.approx(0.0)

    drive = cfg.scene.robot.actuators["gripper_drive"]
    assert drive.joint_names_expr == ["finger_joint"]
    assert drive.effort_limit_sim == pytest.approx(200.0)
    assert drive.velocity_limit_sim == pytest.approx(2.0)
    assert drive.stiffness == pytest.approx(2000.0)
    assert drive.damping == pytest.approx(10.0)
    assert drive.friction == pytest.approx(0.0)
    assert drive.armature == pytest.approx(0.1)

    passive = cfg.scene.robot.actuators["gripper_passive"]
    assert passive.joint_names_expr == [".*_knuckle_joint", ".*_outer_finger_joint"]
    assert passive.effort_limit_sim == pytest.approx(20.0)
    assert passive.velocity_limit_sim == pytest.approx(1.0)
    assert passive.stiffness == pytest.approx(2000.0)
    assert passive.damping == pytest.approx(10.0)
    assert passive.friction == pytest.approx(0.0)
    assert passive.armature == pytest.approx(0.05)
    assert cfg.hand_hold_width == pytest.approx(-0.1)
    assert cfg.hand_close_width == pytest.approx(-0.1)


def test_displayport_feather_pgs_matches_physx_timing_and_gravity_contract():
    """FeatherPGS must preserve the PhysX timing, gripper, contact, and gravity contracts."""
    raw_cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()
    cfg = resolve_presets(Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg(), selected=("feather_pgs",))
    cfg.validate()

    assert cfg.sim.dt == pytest.approx(1.0 / 240.0)
    assert cfg.decimation == 8
    assert cfg.sim.render_interval == cfg.decimation
    assert 1.0 / (cfg.sim.dt * cfg.decimation) == pytest.approx(30.0)

    physics = cfg.sim.physics
    assert isinstance(physics, NewtonCfg)
    assert physics.num_substeps == 1
    assert physics.collision_decimation == 0
    assert physics.use_cuda_graph is True
    assert physics.default_shape_cfg.gap == pytest.approx(0.005)
    assert physics.collision_cfg.reduce_contacts is True
    assert physics.collision_cfg.max_triangle_pairs == 2**25

    solver = physics.solver_cfg
    assert isinstance(solver, FeatherPGSSolverCfg)
    assert solver.pgs_mode == "matrix_free"
    assert solver.pgs_iterations == 128
    assert solver.pgs_velocity_iterations == 0
    assert solver.enable_bilateral_preelimination is True
    assert solver.update_mass_matrix_interval == 1
    assert solver.enable_joint_limits is True
    assert solver.enable_joint_velocity_limits is True
    assert solver.velocity_limit_activation_fraction == pytest.approx(0.7)
    assert solver.dense_max_constraints >= 192
    assert solver.mf_max_constraints >= 1024

    assert cfg.actions.arm_action.controller_cfg.gravity_compensation is True
    assert cfg.scene.robot.spawn.rigid_props.disable_gravity is False
    assert cfg.scene.robot.spawn.joint_drive_props is None

    drive = cfg.scene.robot.actuators["gripper_drive"]
    passive = cfg.scene.robot.actuators["gripper_passive"]
    assert drive.effort_limit_sim == pytest.approx(2.0)
    assert drive.velocity_limit_sim == pytest.approx(1.0)
    assert drive.stiffness == pytest.approx(2000.0)
    assert drive.damping == pytest.approx(10.0)
    assert drive.armature == pytest.approx(0.0)
    assert passive.effort_limit_sim == pytest.approx(1.0)
    assert passive.stiffness == pytest.approx(0.0)
    assert passive.damping == pytest.approx(0.0)
    assert passive.armature == pytest.approx(0.0)
    assert cfg.hand_hold_width == pytest.approx(-0.05)
    assert cfg.hand_close_width == pytest.approx(-0.155)

    plug_friction = cfg.events.plug_physics_material.params["dynamic_friction_range"][0]
    socket_friction = cfg.events.socket_physics_material.params["dynamic_friction_range"][0]
    finger_friction = cfg.events.robot_physics_material.params["dynamic_friction_range"][0]
    assert (plug_friction + socket_friction) / 2.0 == pytest.approx(0.003)
    assert (plug_friction + finger_friction) / 2.0 == pytest.approx(3.0)

    raw_plug_defaults = raw_cfg.scene.dp_plug.spawn.collision_props["/.*"]
    plug_physx = next(fragment for fragment in raw_plug_defaults if isinstance(fragment, PhysxCollisionCfg))
    plug_collision = cfg.scene.dp_plug.spawn.collision_props["/collision_mesh"]
    plug_sdf = next(fragment for fragment in plug_collision if isinstance(fragment, NewtonCollisionCfg))
    assert plug_physx.rest_offset == pytest.approx(-0.00005)
    assert plug_physx.contact_offset == pytest.approx(0.00001)
    assert plug_sdf.contact_margin is None
    assert plug_sdf.contact_gap is None

    socket_sdf_paths = tuple(f"/tn__2584N111_DisplayportCord_jP/Body{body_id}/Mesh" for body_id in (5, 6, 8, 12, 13))
    raw_socket_defaults = raw_cfg.scene.dp_socket.spawn.collision_props["/.*"]
    socket_physx = next(fragment for fragment in raw_socket_defaults if isinstance(fragment, PhysxCollisionCfg))
    for prim_path in socket_sdf_paths:
        socket_collision = cfg.scene.dp_socket.spawn.collision_props[prim_path]
        socket_sdf = next(fragment for fragment in socket_collision if isinstance(fragment, NewtonCollisionCfg))
        assert socket_physx.rest_offset == pytest.approx(-0.0001)
        assert socket_physx.contact_offset == pytest.approx(0.0001)
        assert socket_sdf.contact_margin is None
        assert socket_sdf.contact_gap is None


def test_displayport_newton_observation_abi_noise_and_deployment_metadata():
    """Actor order, flange frame, reset-held noise, and dimensions form the checkpoint ABI."""
    cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()
    actor = cfg.observations.policy
    critic = cfg.observations.critic
    expected_actor_order = ["socket_pos", "tool_pos", "tool_rot_6d", "socket_rot_6d"]

    assert cfg.task_space_obs_order == expected_actor_order
    assert _observation_term_names(actor) == expected_actor_order
    assert actor.concatenate_terms is True
    assert actor.enable_corruption is True
    assert actor.socket_pos.params["asset_cfg"].name == "dp_socket"
    assert tuple(actor.socket_pos.params["offset"]) == pytest.approx(tuple(SOCKET_INSERTION_OFFSET))
    assert isinstance(actor.socket_pos.noise, ResetSampledConstantNoiseModelCfg)
    assert isinstance(actor.socket_pos.noise.noise_cfg, UniformNoiseCfg)
    assert actor.socket_pos.noise.noise_cfg.n_min == pytest.approx(-0.01)
    assert actor.socket_pos.noise.noise_cfg.n_max == pytest.approx(0.01)
    assert actor.socket_pos.noise.noise_cfg.operation == "add"
    assert actor.tool_pos.params["asset_cfg"].name == "robot"
    assert actor.tool_pos.params["body_name"] == "flange"
    assert tuple(actor.tool_pos.params["offset"]) == pytest.approx((0.0, 0.0, 0.0))
    assert actor.tool_rot_6d.params["body_name"] == "flange"
    assert actor.socket_rot_6d.params["asset_cfg"].name == "dp_socket"

    assert _observation_term_names(critic) == [
        "joint_pos",
        "joint_vel",
        "socket_pos",
        "socket_quat",
        "plug_pos",
        "plug_quat",
    ]
    assert critic.joint_pos.params["asset_cfg"].joint_names == [".*"]
    assert critic.joint_vel.params["asset_cfg"].joint_names == [".*"]

    ros_cfg = newton_ros_cfg.Rizon4sTaskSpaceNewtonDisplayportInsertionROSInferenceEnvCfg()
    assert ros_cfg.obs_order == ["eef_pos", "eef_rot_6d", "socket_kp_pos", "socket_kp_rot_6d"]
    assert ros_cfg.policy_action_space == "task"
    assert ros_cfg.arm_joint_names == _ARM_JOINTS
    assert ros_cfg.action_space == 6
    assert ros_cfg.observation_space == 18
    assert ros_cfg.state_space == 40
    assert ros_cfg.fixed_asset_init_pos_range == pytest.approx([0.01, 0.01, 0.02])
    assert ros_cfg.fixed_asset_init_orn_deg_range == pytest.approx([2.0, 2.0, 2.0])
    assert ros_cfg.fixed_asset_pos_obs_noise_level == pytest.approx([0.01, 0.01, 0.01])


def test_displayport_newton_domain_randomization_curriculum_and_rewards():
    """Checkpoint material, reset, friction, curriculum, and reward contracts must remain exact."""
    cfg = resolve_presets(Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg())
    events = cfg.events

    material_coefficients = {
        "plug_physics_material": 3.0,
        "socket_physics_material": 0.001,
        "robot_physics_material": 1.0,
    }
    for event_name, coefficient in material_coefficients.items():
        event = getattr(events, event_name)
        assert event.mode == "startup"
        assert event.params["static_friction_range"] == pytest.approx((coefficient, coefficient))
        assert event.params["dynamic_friction_range"] == pytest.approx((coefficient, coefficient))
        assert event.params["restitution_range"] == pytest.approx((0.0, 0.0))
        assert event.params["num_buckets"] == 16

    arm_friction = events.randomize_arm_joint_friction
    assert arm_friction.mode == "reset"
    assert arm_friction.params["asset_cfg"].name == "robot"
    assert arm_friction.params["asset_cfg"].joint_names == _ARM_JOINTS
    assert arm_friction.params["friction_distribution_params"] == pytest.approx((0.0, 0.15))
    assert arm_friction.params["operation"] == "add"
    assert arm_friction.params["distribution"] == "uniform"
    assert events.randomize_arm_pd_gains is None

    pose_range = events.randomize_socket_pose.params["pose_range"]
    assert pose_range["x"] == pytest.approx([-0.01, 0.01])
    assert pose_range["y"] == pytest.approx([-0.01, 0.01])
    assert pose_range["z"] == pytest.approx([-0.02, 0.02])
    for axis in ("roll", "pitch", "yaw"):
        assert pose_range[axis] == pytest.approx([-math.radians(2.0), math.radians(2.0)])

    curriculum = events.reset_plug_curriculum.params
    assert curriculum["at_goal_prob"] == pytest.approx(0.8)
    assert curriculum["at_goal_prob_final"] == pytest.approx(0.0)
    assert curriculum["anneal_start_iter"] == pytest.approx(0.0)
    assert curriculum["anneal_end_iter"] == pytest.approx(500.0)
    assert curriculum["num_steps_per_env"] == 512
    assert curriculum["insertion_axis"] == pytest.approx([1.0, 0.0, 0.0])
    assert curriculum["insertion_length"] == pytest.approx(0.011)
    assert curriculum["at_goal_depth_range"] == pytest.approx([0.0, 0.015])
    assert curriculum["approach_depth_range"] == pytest.approx([0.02, 0.06])
    assert curriculum["socket_insertion_offset"] == pytest.approx(SOCKET_INSERTION_OFFSET)
    assert curriculum["plug_insertion_offset"] == pytest.approx(PLUG_INSERTION_OFFSET)
    assert curriculum["goal_rot"] == pytest.approx(PLUG_GOAL_ROT)
    assert curriculum["normal_pose_range"]["x"] == pytest.approx([-0.02, 0.02])
    assert curriculum["normal_pose_range"]["y"] == pytest.approx([-0.02, 0.02])
    assert curriculum["normal_pose_range"]["z"] == pytest.approx([0.0, 0.0])

    rewards = cfg.rewards
    assert rewards.plug_socket_keypoint_tracking.weight == pytest.approx(-1.5)
    assert rewards.plug_socket_keypoint_tracking_exp.weight == pytest.approx(1.5)
    assert rewards.action_rate.weight == pytest.approx(-5.0e-6)
    assert rewards.plug_socket_keypoint_tracking_exp.params["kp_exp_coeffs"] == [
        (50, 0.0001),
        (300, 0.0001),
        (600, 0.0001),
        (2000, 0.0001),
    ]
    assert rewards.plug_socket_keypoint_tracking_exp.params["kp_use_sum_of_exps"] is False


@pytest.mark.parametrize(
    ("iteration", "expected_probability"),
    [
        pytest.param(0, 0.8, id="start"),
        pytest.param(250, 0.4, id="midpoint"),
        pytest.param(500, 0.0, id="end"),
        pytest.param(750, 0.0, id="after-end"),
    ],
)
def test_displayport_newton_curriculum_anneals_at_goal_probability(iteration: int, expected_probability: float):
    """The reset curriculum must linearly anneal from 0.8 to zero over 500 iterations."""
    term = object.__new__(reset_plug_at_goal_curriculum)
    term.at_goal_prob = 0.8
    term.at_goal_prob_final = 0.0
    term.anneal_start_iter = 0.0
    term.anneal_end_iter = 500.0
    term.num_steps_per_env = 512
    env = SimpleNamespace(common_step_counter=iteration * term.num_steps_per_env)

    assert term._current_at_goal_prob(env) == pytest.approx(expected_probability)


def test_displayport_newton_runner_and_play_preserve_physx_defaults():
    """Newton uses its checkpoint horizon and safe shard size without mutating PhysX defaults."""
    newton_runner = Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg()
    physx_runner = Rizon4sGravDisplayportInsertionRNNPPORunnerCfg()
    assert newton_runner.seed == 123
    assert newton_runner.max_iterations == 1000
    assert newton_runner.experiment_name == "displayport_insertion_rizon4s_newton_osc"
    assert physx_runner.max_iterations == 1500
    assert physx_runner.experiment_name == "displayport_insertion_rizon4s"
    assert newton_runner.num_steps_per_env == physx_runner.num_steps_per_env == 512
    assert newton_runner.clip_actions == physx_runner.clip_actions == pytest.approx(1.0)
    assert newton_runner.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert physx_runner.obs_groups == {"actor": ["policy"], "critic": ["critic"]}

    train_cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()
    play_cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg_PLAY()
    assert train_cfg.scene.num_envs == 256
    assert train_cfg.observations.policy.enable_corruption is True
    assert train_cfg.events.reset_plug_curriculum.params["at_goal_prob"] == pytest.approx(0.8)
    assert play_cfg.observations.policy.enable_corruption is False
    assert play_cfg.events.reset_plug_curriculum.params["at_goal_prob"] == pytest.approx(0.0)
    assert play_cfg.events.reset_plug_curriculum.params["at_goal_prob_final"] == pytest.approx(0.0)
    assert (
        play_cfg.events.randomize_socket_pose.params["pose_range"]
        == (train_cfg.events.randomize_socket_pose.params["pose_range"])
    )
    assert play_cfg.scene.num_envs == 50

    physx_cfg = Rizon4sTaskSpaceDisplayportInsertionEnvCfg()
    assert physx_cfg.scene.num_envs == 4096
    assert isinstance(physx_cfg.sim.physics, PhysxCfg)
    assert physx_cfg.sim.dt == pytest.approx(1.0 / 240.0)
    assert physx_cfg.decimation == 8
    assert 1.0 / (physx_cfg.sim.dt * physx_cfg.decimation) == pytest.approx(30.0)
    actor = physx_cfg.observations.policy
    assert _observation_term_names(actor) == ["eef_pos", "eef_rot_6d", "socket_kp_pos", "socket_kp_rot_6d"]
    assert actor.enable_corruption is False
    assert actor.socket_kp_pos.noise is None
    assert tuple(actor.eef_pos.params["offset"]) == pytest.approx((0.0, 0.0, 0.1925))
    assert physx_cfg.actions.arm_action.controller_cfg.inertial_dynamics_decoupling is False
    assert physx_cfg.scene.robot.spawn.usd_path == train_cfg.scene.robot.spawn.usd_path
    assert physx_cfg.events.plug_physics_material.params["static_friction_range"] == pytest.approx((0.001, 0.001))
    assert physx_cfg.events.robot_physics_material.params["static_friction_range"] == pytest.approx((0.75, 0.75))


def test_osc_leapp_metadata_expands_per_axis_action_scales():
    """OSC export metadata must serialize scalar and per-axis scales consistently."""
    extra = _pose_rel_action_extra(
        position_scale=[0.025, 0.025, 0.01],
        orientation_scale=0.025,
        target_types=["pose_rel"],
    )

    assert extra["scale"] == pytest.approx([0.025, 0.025, 0.01, 0.025, 0.025, 0.025])
    assert extra["position_scale"] == pytest.approx([0.025, 0.025, 0.01])
    assert extra["orientation_scale"] == pytest.approx(0.025)
