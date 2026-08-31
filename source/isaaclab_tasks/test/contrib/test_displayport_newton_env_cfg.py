# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton DisplayPort insertion production configuration."""

import inspect
import math
from dataclasses import fields
from pathlib import Path

import gymnasium as gym
import pytest
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.managers import ObservationTermCfg
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
    _RIZON4S_CALIBRATED_USD_PATH,
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
from isaaclab_tasks.contrib.deploy.mdp.events import set_robot_to_object_grasp_pose
from isaaclab_tasks.contrib.deploy.mdp.noise_models import ResetSampledConstantNoiseModelCfg

_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
_ENV_ENTRY_POINT = "isaaclab_tasks.contrib.deploy.cable_insertion.insertion_env:DisplayportInsertionEnv"
_CFG_PACKAGE = "isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s"
_AGENT_MODULE = f"{_CFG_PACKAGE}.agents.rsl_rl_ppo_cfg"


def _observation_term_names(group: object) -> list[str]:
    """Return observation terms in their concatenation order."""
    return [field.name for field in fields(group) if isinstance(getattr(group, field.name), ObservationTermCfg)]


def test_displayport_grasp_reset_uses_backend_neutral_jacobians():
    """Reset IK must use public articulation data shared by PhysX and Newton."""
    init_source = inspect.getsource(set_robot_to_object_grasp_pose.__init__)
    source = inspect.getsource(set_robot_to_object_grasp_pose.__call__)

    assert "is_fixed_base" in init_source
    assert "body_link_jacobian_w" in source
    assert "num_base_dofs" in source
    assert "root_view" not in source


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


def test_displayport_newton_timing_solver_and_point_sdf_assets():
    """Newton defaults must retain the validated 2 kHz/200 Hz point-SDF profile."""
    cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()

    assert cfg.sim.dt == pytest.approx(0.01)
    assert cfg.decimation == 3
    assert cfg.sim.render_interval == cfg.decimation
    assert isinstance(cfg.sim.physics, DisplayportNewtonPhysicsCfg)

    physics = cfg.sim.physics.default
    assert isinstance(physics, NewtonCfg)
    assert physics.num_substeps == 20
    assert physics.collision_decimation == 10
    assert physics.debug_mode is False
    assert physics.use_cuda_graph is True
    assert physics.default_shape_cfg.gap == pytest.approx(0.005)
    assert physics.default_shape_cfg.margin == pytest.approx(0.0)
    assert physics.num_substeps / cfg.sim.dt == pytest.approx(2000.0)
    assert physics.num_substeps / cfg.sim.dt / physics.collision_decimation == pytest.approx(200.0)
    assert 1.0 / (cfg.sim.dt * cfg.decimation) == pytest.approx(100.0 / 3.0)

    solver = physics.solver_cfg
    assert isinstance(solver, MJWarpSolverCfg)
    assert solver.solver == "newton"
    assert solver.integrator == "implicitfast"
    assert solver.njmax == 8192
    assert solver.nconmax == 8192
    assert solver.iterations == 100
    assert solver.ls_iterations == 50
    assert solver.update_data_interval == 10
    assert solver.impratio == pytest.approx(10.0)
    assert solver.cone == "elliptic"
    assert solver.ccd_iterations == 35
    assert solver.use_mujoco_contacts is False

    collision = physics.collision_cfg
    assert collision is not None
    assert collision.broad_phase == "explicit"
    assert collision.reduce_contacts is True
    assert collision.max_triangle_pairs == 2**25

    asset_sublayers = {
        "display_port_plug_newton_sdf.usda": "display_port_plug_fixed_sdf.usd",
        "display_port_socket_newton_sdf.usda": "display_port_socket_fixed_sdf_noprotrusions.usd",
    }
    asset_paths = (Path(cfg.scene.dp_plug.spawn.usd_path), Path(cfg.scene.dp_socket.spawn.usd_path))
    assert {path.name for path in asset_paths} == set(asset_sublayers)

    for path in asset_paths:
        assert path.is_file()
        source = path.read_text(encoding="utf-8")
        assert f"@./{asset_sublayers[path.name]}@" in source
        assert "NewtonCollisionAPI" in source
        assert "NewtonSDFCollisionAPI" in source
        assert "float newton:contactGap = 0.005" in source
        assert "float newton:contactMargin = 0" in source
        assert "bool newton:hydroelasticEnabled = 0" in source
        assert "int newton:sdfMaxResolution = 256" in source
        assert 'token newton:sdfTextureFormat = "uint16"' in source


def test_displayport_newton_osc_abi_robot_and_gravity_settings():
    """OSC must command flange-relative pose deltas without double gravity compensation."""
    cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()
    action = cfg.actions.arm_action

    assert isinstance(action, DeployOperationalSpaceControllerActionCfg)
    assert action.asset_name == "robot"
    assert action.joint_names == _ARM_JOINTS
    assert action.body_name == "flange"
    assert action.body_offset is not None
    assert tuple(action.body_offset.pos) == pytest.approx((0.0, 0.0, 0.0))
    assert tuple(action.body_offset.rot) == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert action.position_scale == pytest.approx(0.025)
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

    calibrated_usd = Path(cfg.scene.robot.spawn.usd_path)
    assert cfg.scene.robot.spawn.usd_path == _RIZON4S_CALIBRATED_USD_PATH
    assert calibrated_usd.name == "Rizon4s-063459_with_Grav_calibrated_kinematics.usd"
    assert calibrated_usd.is_file()
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
    assert ros_cfg.obs_order == expected_actor_order
    assert ros_cfg.policy_action_space == "task"
    assert ros_cfg.arm_joint_names == _ARM_JOINTS
    assert ros_cfg.action_space == 6
    assert ros_cfg.observation_space == 18
    assert ros_cfg.state_space == 40
    assert ros_cfg.action_scale == pytest.approx([0.025] * 6)
    assert ros_cfg.fixed_asset_init_pos_range == pytest.approx([0.01, 0.01, 0.02])
    assert ros_cfg.fixed_asset_init_orn_deg_range == pytest.approx([2.0, 2.0, 2.0])
    assert ros_cfg.fixed_asset_pos_obs_noise_level == pytest.approx([0.01, 0.01, 0.01])


def test_displayport_newton_domain_randomization_curriculum_and_rewards():
    """Validated material, reset, friction, curriculum, and reward values must remain exact."""
    cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()
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


def test_displayport_newton_runner_and_play_preserve_physx_defaults():
    """Newton uses its validated horizon and play mode without mutating PhysX defaults."""
    newton_runner = Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg()
    physx_runner = Rizon4sGravDisplayportInsertionRNNPPORunnerCfg()
    assert newton_runner.max_iterations == 1000
    assert newton_runner.experiment_name == "displayport_insertion_rizon4s_newton_osc"
    assert physx_runner.max_iterations == 1500
    assert physx_runner.experiment_name == "displayport_insertion_rizon4s"
    assert newton_runner.num_steps_per_env == physx_runner.num_steps_per_env == 512
    assert newton_runner.clip_actions == physx_runner.clip_actions == pytest.approx(1.0)

    train_cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()
    play_cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg_PLAY()
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
    assert physx_cfg.scene.robot.spawn.usd_path != train_cfg.scene.robot.spawn.usd_path
    assert physx_cfg.events.plug_physics_material.params["static_friction_range"] == pytest.approx((0.001, 0.001))
    assert physx_cfg.events.robot_physics_material.params["static_friction_range"] == pytest.approx((0.75, 0.75))
