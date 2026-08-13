# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Deploy DisplayPort insertion environment configuration."""

from pathlib import Path

from pxr import Usd

from isaaclab.envs import mdp

import isaaclab_tasks.contrib.deploy.mdp as deploy_mdp
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.agents.rsl_rl_ppo_cfg import (
    Rizon4sGravDisplayportInsertionRNNPPORunnerCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.ik_newton_env_cfg import (
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangeObsEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcpObsEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedIKNewtonEnvCfg,
    Rizon4sGravDisplayportInsertionIKNewtonEnvCfg,
    Rizon4sGravDisplayportInsertionIKNewtonFlangeObsEnvCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_pos_env_cfg import (
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNoJointVelEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedNoJointVelEnvCfg,
    Rizon4sGravDisplayportInsertionEnvCfg,
    Rizon4sGravDisplayportInsertionNoJointVelEnvCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    DISPLAY_ASSETS_DIR,
    SOCKET_INSERTION_OFFSET,
)
from isaaclab_tasks.utils.hydra import resolve_presets


def test_displayport_newton_uses_full_insertion_target_and_physx_grasp():
    """Newton should preserve the PhysX full-insertion target and grasp pose."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_hydroelastic"})

    assert env_cfg.grasp_offset == [0.0025, 0.0, -0.1875]
    hydro_cfg = env_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config
    assert hydro_cfg is not None
    assert hydro_cfg.buffer_fraction == 0.125
    assert hydro_cfg.buffer_mult_broad == 8
    assert hydro_cfg.buffer_mult_iso == 2
    assert env_cfg.sim.physics.collision_cfg.max_triangle_pairs == 2**25
    assert env_cfg.scene.dp_plug.spawn.usd_path.endswith("display_port_plug_newton_hydroelastic.usda")
    assert env_cfg.scene.dp_socket.spawn.usd_path.endswith("display_port_socket_newton_hydroelastic.usda")
    assert env_cfg.observations.policy.socket_pos.params["offset"] == SOCKET_INSERTION_OFFSET
    assert env_cfg.observations.critic.socket_pos.params["offset"] == SOCKET_INSERTION_OFFSET
    assert env_cfg.rewards.plug_socket_keypoint_tracking.params["offset_1"] == SOCKET_INSERTION_OFFSET
    assert env_cfg.rewards.plug_socket_keypoint_tracking_exp.params["offset_1"] == SOCKET_INSERTION_OFFSET


def test_displayport_preserves_physx_default_and_exposes_newton_mjwarp():
    """Existing callers should keep PhysX while Newton MJWarp uses authored SDF assets."""
    default_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"default"})
    newton_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_mjwarp"})

    assert type(default_cfg.sim.physics).__name__ == "PhysxCfg"
    assert default_cfg.scene.robot.spawn.joint_drive_props is None
    assert default_cfg.scene.robot.actuators["shoulder"].stiffness == 1320.0
    assert default_cfg.scene.robot.actuators["elbow"].stiffness == 600.0
    assert default_cfg.scene.robot.actuators["wrist"].stiffness == 216.0

    assert newton_cfg.sim.physics.solver_cfg.use_mujoco_contacts is True
    assert newton_cfg.scene.robot.spawn.joint_drive_props.actuatorgravcomp is False
    assert newton_cfg.scene.robot.actuators["shoulder"].stiffness == 6000.0
    assert newton_cfg.scene.robot.actuators["elbow"].stiffness == 4200.0
    assert newton_cfg.scene.robot.actuators["wrist"].stiffness == 1500.0
    assert newton_cfg.scene.dp_plug.spawn.usd_path.endswith("display_port_plug_newton_sdf.usda")
    assert newton_cfg.scene.dp_socket.spawn.usd_path.endswith("display_port_socket_newton_sdf.usda")


def test_displayport_newton_disables_robot_gravity_passively():
    """Newton should cancel gravity on robot bodies without using actuator forces."""
    for preset_name in ("newton_mjwarp", "newton_sdf", "newton_hydroelastic"):
        env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {preset_name})
        rigid_props = env_cfg.scene.robot.spawn.rigid_props
        joint_drive_props = env_cfg.scene.robot.spawn.joint_drive_props

        assert type(rigid_props).__name__ == "MujocoRigidBodyPropertiesCfg"
        assert rigid_props.gravcomp == 1.0
        assert rigid_props.disable_gravity is None
        assert joint_drive_props.actuatorgravcomp is False
        assert env_cfg.scene.dp_plug.spawn.rigid_props.disable_gravity is False


def test_displayport_hard_sdf_uses_point_contacts_with_precomputed_sdfs():
    """Hard SDF should cook mesh volumes without enabling hydroelastic contact."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_sdf"})

    assert env_cfg.sim.physics.collision_cfg.max_triangle_pairs == 2**25
    assert env_cfg.scene.dp_plug.spawn.usd_path.endswith("display_port_plug_newton_sdf.usda")
    assert env_cfg.scene.dp_socket.spawn.usd_path.endswith("display_port_socket_newton_sdf.usda")
    assert env_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config is None
    assert env_cfg.sim.physics.solver_cfg.use_mujoco_contacts is False


def test_displayport_assets_author_newton_sdf_per_active_collider():
    """Newton overlays should author SDF metadata only on active collision meshes."""
    socket_stage = Usd.Stage.Open(f"{DISPLAY_ASSETS_DIR}/display_port_socket_newton_sdf.usda")
    assert socket_stage.GetRootLayer().subLayerPaths == ["./display_port_socket_fixed_sdf_noprotrusions.usd"]

    expected_counts = {
        "display_port_plug_newton_sdf.usda": (1, False),
        "display_port_socket_newton_sdf.usda": (5, False),
        "display_port_plug_newton_hydroelastic.usda": (1, True),
        "display_port_socket_newton_hydroelastic.usda": (5, True),
    }
    for filename, (expected_count, hydroelastic_enabled) in expected_counts.items():
        stage = Usd.Stage.Open(f"{DISPLAY_ASSETS_DIR}/{filename}")
        sdf_prims = [prim for prim in stage.Traverse() if prim.HasAttribute("newton:sdfMaxResolution")]
        assert len(sdf_prims) == expected_count
        assert all(prim.GetAttribute("newton:sdfMaxResolution").Get() == 256 for prim in sdf_prims)
        assert all(abs(prim.GetAttribute("newton:contactGap").Get() - 0.005) < 1.0e-8 for prim in sdf_prims)
        assert all(prim.GetAttribute("newton:hydroelasticEnabled").Get() is hydroelastic_enabled for prim in sdf_prims)


def test_displayport_point_sdf_gap_overlays_override_every_mating_collider():
    """Gap ablations should override authored values on both sides of the mating pair."""
    expected_gaps = {"gap_1mm": 0.001, "gap_0p5mm": 0.0005}
    for suffix, expected_gap in expected_gaps.items():
        for asset_name, expected_count in (("plug", 1), ("socket", 5)):
            filename = f"display_port_{asset_name}_newton_sdf_{suffix}.usda"
            stage = Usd.Stage.Open(f"{DISPLAY_ASSETS_DIR}/{filename}")
            sdf_prims = [prim for prim in stage.Traverse() if prim.HasAttribute("newton:sdfMaxResolution")]

            assert len(sdf_prims) == expected_count
            assert all(abs(prim.GetAttribute("newton:contactGap").Get() - expected_gap) < 1.0e-8 for prim in sdf_prims)
            assert all(prim.GetAttribute("newton:hydroelasticEnabled").Get() is False for prim in sdf_prims)


def test_displayport_legacy_socket_comparison_preserves_newton_metadata():
    """The legacy A/B layer should differ only in its underlying socket USD."""
    stage = Usd.Stage.Open(f"{DISPLAY_ASSETS_DIR}/display_port_socket_newton_sdf_legacy.usda")
    assert stage.GetRootLayer().subLayerPaths == ["./display_port_socket_fixed_sdf_split_visuals.usd"]

    sdf_prims = [prim for prim in stage.Traverse() if prim.HasAttribute("newton:sdfMaxResolution")]
    assert len(sdf_prims) == 5
    assert all(prim.GetAttribute("newton:sdfMaxResolution").Get() == 256 for prim in sdf_prims)
    assert all(abs(prim.GetAttribute("newton:contactGap").Get() - 0.005) < 1.0e-8 for prim in sdf_prims)


def test_displayport_newton_ik_commands_flange_without_actor_velocity():
    """The deployment-oriented IK task should command the flange with a six-dimensional action."""
    env_cfg = Rizon4sGravDisplayportInsertionIKNewtonEnvCfg()
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert pose_objective.body_name == "flange"
    assert pose_objective.use_relative_mode is True
    assert pose_objective.scale == 0.01
    assert env_cfg.observations.policy.joint_vel is None


def test_displayport_newton_ik_flange_observation_replaces_actor_joint_state():
    """The flange-observation task should expose task state without actor joint state."""
    env_cfg = Rizon4sGravDisplayportInsertionIKNewtonFlangeObsEnvCfg()
    policy = env_cfg.observations.policy
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert policy.joint_pos is None
    assert policy.joint_vel is None
    assert policy.flange_pose.func is mdp.body_pose_w
    assert policy.flange_pose.params["asset_cfg"].name == "robot"
    assert policy.flange_pose.params["asset_cfg"].body_names == ["flange"]
    assert policy.socket_pos is not None
    assert policy.socket_quat is not None
    assert env_cfg.observations.critic.joint_pos is not None
    assert env_cfg.observations.critic.joint_vel is not None
    assert pose_objective.body_name == "flange"
    assert pose_objective.body_offset_pos == (0.0, 0.0, 0.0)


def test_displayport_calibrated_dr_flange_observation_preserves_randomization():
    """The calibrated flange-observation task should retain arm domain randomization."""
    env_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangeObsEnvCfg()

    assert env_cfg.observations.policy.joint_pos is None
    assert env_cfg.observations.policy.flange_pose is not None
    assert env_cfg.events.randomize_arm_joint_friction is not None
    assert env_cfg.events.randomize_arm_pd_gains is not None


def test_displayport_calibrated_dr_tcp_observation_offsets_only_actor_pose():
    """The TCP observation should use a local offset while the action controls the flange origin."""
    env_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcpObsEnvCfg()
    policy = env_cfg.observations.policy
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert policy.joint_pos is None
    assert policy.joint_vel is None
    assert policy.tcp_pose.func is deploy_mdp.body_pose_w_with_offset
    assert policy.tcp_pose.params["asset_cfg"].body_names == ["flange"]
    assert policy.tcp_pose.params["offset"] == (0.0, 0.0, 0.1925)
    assert pose_objective.body_name == "flange"
    assert pose_objective.body_offset_pos == (0.0, 0.0, 0.0)
    assert env_cfg.events.randomize_arm_joint_friction is not None
    assert env_cfg.events.randomize_arm_pd_gains is not None


def test_displayport_calibrated_domain_randomized_newton_ik_preserves_training_configuration():
    """The calibrated DR task should change only the arm action to relative flange IK."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg(),
        {"newton_sdf"},
    )
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert Path(env_cfg.scene.robot.spawn.usd_path).name == "Rizon4s-063459_with_Grav_calibrated_kinematics.usd"
    assert pose_objective.body_name == "flange"
    assert pose_objective.use_relative_mode is True
    assert pose_objective.scale == 0.01
    assert env_cfg.observations.policy.joint_vel is None
    assert env_cfg.events.randomize_arm_joint_friction is not None
    assert env_cfg.events.randomize_arm_pd_gains is not None
    assert env_cfg.scene.robot.spawn.rigid_props.gravcomp == 1.0


def test_displayport_calibrated_newton_ik_omits_domain_randomization():
    """The calibrated IK task should preserve nominal arm dynamics."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionCalibratedIKNewtonEnvCfg(),
        {"newton_sdf"},
    )
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert Path(env_cfg.scene.robot.spawn.usd_path).name == "Rizon4s-063459_with_Grav_calibrated_kinematics.usd"
    assert pose_objective.body_name == "flange"
    assert env_cfg.observations.policy.joint_vel is None
    assert env_cfg.events.randomize_arm_joint_friction is None
    assert env_cfg.events.randomize_arm_pd_gains is None


def test_displayport_no_joint_velocity_hides_velocity_from_actor_only():
    """No-joint-velocity training should retain privileged critic velocity."""
    env_cfg = Rizon4sGravDisplayportInsertionNoJointVelEnvCfg()

    assert env_cfg.observations.policy.joint_vel is None
    assert env_cfg.observations.critic.joint_vel is not None


def test_displayport_calibrated_robot_preserves_newton_training_configuration():
    """Calibrated kinematics should change only the robot USD in the Newton task."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionCalibratedNoJointVelEnvCfg(),
        {"newton_sdf"},
    )

    robot_usd_path = Path(env_cfg.scene.robot.spawn.usd_path)
    curriculum = env_cfg.events.reset_plug_curriculum.params

    assert robot_usd_path.name == "Rizon4s-063459_with_Grav_calibrated_kinematics.usd"
    assert robot_usd_path.is_file()
    robot_stage = Usd.Stage.Open(str(robot_usd_path))
    required_prim_paths = (
        "/Rizon4s/base_link",
        "/Rizon4s/link7",
        "/Rizon4s/flange",
        "/Rizon4s/joints/joint1",
        "/Rizon4s/joints/joint7",
        "/Rizon4s/Grav_gripper/gripper_base",
        "/Rizon4s/Grav_gripper/left_finger_tip",
        "/Rizon4s/Grav_gripper/right_finger_tip",
    )
    assert robot_stage.GetDefaultPrim().GetPath() == "/Rizon4s"
    assert all(robot_stage.GetPrimAtPath(path).IsValid() for path in required_prim_paths)
    assert env_cfg.scene.robot.spawn.rigid_props.gravcomp == 1.0
    assert env_cfg.scene.robot.spawn.joint_drive_props.actuatorgravcomp is False
    assert type(env_cfg.actions.arm_action).__name__ == "RelativeJointPositionActionCfg"
    assert env_cfg.observations.policy.joint_vel is None
    assert env_cfg.observations.critic.joint_vel is not None
    assert curriculum["at_goal_prob"] == 0.8
    assert curriculum["at_goal_prob_final"] == 0.0


def test_displayport_socket_observation_uses_reset_sampled_ten_millimeter_noise():
    """The actor should observe one reset-sampled socket-position offset per episode."""
    env_cfg = Rizon4sGravDisplayportInsertionCalibratedNoJointVelEnvCfg()

    noise_model = env_cfg.observations.policy.socket_pos.noise
    assert type(noise_model).__name__ == "ResetSampledConstantNoiseModelCfg"
    assert noise_model.noise_cfg.n_min == -0.01
    assert noise_model.noise_cfg.n_max == 0.01
    assert noise_model.noise_cfg.operation == "add"
    assert env_cfg.observations.critic.socket_pos.noise is None


def test_displayport_calibrated_domain_randomization_targets_arm_joints():
    """The calibrated DR task should randomize only joint1 through joint7 at reset."""
    baseline_cfg = Rizon4sGravDisplayportInsertionCalibratedNoJointVelEnvCfg()
    env_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNoJointVelEnvCfg()

    assert baseline_cfg.events.randomize_arm_joint_friction is None
    assert baseline_cfg.events.randomize_arm_pd_gains is None

    friction = env_cfg.events.randomize_arm_joint_friction
    assert friction.mode == "reset"
    assert friction.params["asset_cfg"].joint_names == [f"joint{joint_id}" for joint_id in range(1, 8)]
    assert friction.params["friction_distribution_params"] == (0.0, 0.15)
    assert friction.params["operation"] == "add"
    assert friction.params["distribution"] == "uniform"

    gains = env_cfg.events.randomize_arm_pd_gains
    assert gains.mode == "reset"
    assert gains.params["asset_cfg"].joint_names == [f"joint{joint_id}" for joint_id in range(1, 8)]
    assert gains.params["stiffness_distribution_params"] == (0.8, 1.2)
    assert gains.params["damping_distribution_params"] == (0.8, 1.2)
    assert gains.params["operation"] == "scale"
    assert gains.params["distribution"] == "uniform"


def test_displayport_play_mode_uses_approach_resets():
    """Playback should not initialize the plug at the seated target."""
    env_cfg = Rizon4sGravDisplayportInsertionNoJointVelEnvCfg()
    env_cfg.play_mode()

    curriculum = env_cfg.events.reset_plug_curriculum.params
    assert curriculum["at_goal_prob"] == 0.0
    assert curriculum["at_goal_prob_final"] == 0.0
    assert env_cfg.observations.policy.enable_corruption is False


def test_displayport_newton_matches_reference_training_curriculum():
    """Newton should preserve the referenced PhysX curriculum and reward shaping."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_hydroelastic"})

    curriculum = env_cfg.events.reset_plug_curriculum.params
    assert curriculum["at_goal_prob"] == 0.8
    assert curriculum["at_goal_prob_final"] == 0.0
    assert curriculum["anneal_start_iter"] == 0.0
    assert curriculum["anneal_end_iter"] == 500.0
    assert curriculum["num_steps_per_env"] == 512
    assert curriculum["at_goal_depth_range"] == [0.0, 0.015]
    assert curriculum["approach_depth_range"] == [0.02, 0.06]

    linear_weight = env_cfg.rewards.plug_socket_keypoint_tracking.weight
    exponential = env_cfg.rewards.plug_socket_keypoint_tracking_exp
    assert exponential.weight == abs(linear_weight)
    assert exponential.params["kp_exp_coeffs"][-1] == (2000, 0.0001)

    assert Rizon4sGravDisplayportInsertionRNNPPORunnerCfg().max_iterations == 1500
