# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Deploy DisplayPort insertion environment configuration."""

from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.agents.rsl_rl_ppo_cfg import (
    Rizon4sGravDisplayportInsertionRNNPPORunnerCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_pos_env_cfg import (
    Rizon4sGravDisplayportInsertionEnvCfg,
    Rizon4sGravDisplayportInsertionNoJointVelEnvCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import SOCKET_INSERTION_OFFSET
from isaaclab_tasks.utils.hydra import resolve_presets


def test_displayport_newton_uses_full_insertion_target_and_physx_grasp():
    """Newton should preserve the PhysX full-insertion target and grasp pose."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_hydroelastic"})

    assert env_cfg.grasp_offset == [0.0025, 0.0, -0.1875]
    assert env_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config.sdf_max_resolution == 256
    assert env_cfg.observations.policy.socket_pos.params["offset"] == SOCKET_INSERTION_OFFSET
    assert env_cfg.observations.critic.socket_pos.params["offset"] == SOCKET_INSERTION_OFFSET
    assert env_cfg.rewards.plug_socket_keypoint_tracking.params["offset_1"] == SOCKET_INSERTION_OFFSET
    assert env_cfg.rewards.plug_socket_keypoint_tracking_exp.params["offset_1"] == SOCKET_INSERTION_OFFSET


def test_displayport_hard_sdf_uses_point_contacts_with_precomputed_sdfs():
    """Hard SDF should cook mesh volumes without enabling hydroelastic contact."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_sdf"})

    assert env_cfg.sim.physics.collision_cfg.mesh_sdf_max_resolution == 256
    assert env_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config is None
    assert env_cfg.sim.physics.solver_cfg.use_mujoco_contacts is False


def test_displayport_selective_hydroelastic_only_cooks_mating_colliders():
    """Selective hydroelastic should keep the gripper on convex point contacts."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionEnvCfg(),
        {"newton_hydroelastic_selective"},
    )

    collision_cfg = env_cfg.sim.physics.collision_cfg
    assert collision_cfg.mesh_sdf_shape_path_exprs == (
        r".*/collision_mesh",
        r".*/Body(5|6|8)/Mesh",
        r".*/colliders/sdf_.*",
    )
    assert collision_cfg.preserve_concave_shape_path_exprs == ()
    assert collision_cfg.sdf_hydroelastic_config.sdf_max_resolution == 256


def test_displayport_filtered_hydroelastic_excludes_non_tip_robot_asset_pairs():
    """Filtered hydroelastic should preserve fingertip contacts with the plug."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionEnvCfg(),
        {"newton_hydroelastic_selective_filtered"},
    )

    pair_exprs = env_cfg.sim.physics.collision_cfg.shape_collision_filter_pair_path_exprs
    assert pair_exprs == (
        (
            r".*/Robot/(?!Grav_gripper/(?:left|right)_finger_tip/).*",
            r".*/DisplayPort(?:Plug|Socket)/.*",
        ),
    )


def test_displayport_selective_hard_sdf_only_cooks_mating_colliders():
    """Selective hard SDF should use point contacts and convex fingertips."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_sdf_selective"})

    collision_cfg = env_cfg.sim.physics.collision_cfg
    assert collision_cfg.mesh_sdf_shape_path_exprs == (
        r".*/collision_mesh",
        r".*/Body(5|6|8)/Mesh",
        r".*/colliders/sdf_.*",
    )
    assert collision_cfg.mesh_sdf_max_resolution == 256
    assert collision_cfg.preserve_concave_shape_path_exprs == ()
    assert collision_cfg.sdf_hydroelastic_config is None


def test_displayport_filtered_sdf_excludes_non_tip_robot_asset_pairs():
    """The filter ablation should preserve fingertip contacts with the plug."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_sdf_selective_filtered"})

    pair_exprs = env_cfg.sim.physics.collision_cfg.shape_collision_filter_pair_path_exprs
    assert pair_exprs == (
        (
            r".*/Robot/(?!Grav_gripper/(?:left|right)_finger_tip/).*",
            r".*/DisplayPort(?:Plug|Socket)/.*",
        ),
    )


def test_displayport_kamino_uses_selective_newton_sdf_contacts():
    """Kamino should consume selective point-SDF contacts from Newton."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_kamino_sdf_selective"})

    assert env_cfg.sim.physics.solver_cfg.use_collision_detector is False
    assert env_cfg.sim.physics.collision_cfg.mesh_sdf_max_resolution == 256
    assert env_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config is None


def test_displayport_no_joint_velocity_hides_velocity_from_actor_only():
    """No-joint-velocity training should retain privileged critic velocity."""
    env_cfg = Rizon4sGravDisplayportInsertionNoJointVelEnvCfg()

    assert env_cfg.observations.policy.joint_vel is None
    assert env_cfg.observations.critic.joint_vel is not None


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

    assert Rizon4sGravDisplayportInsertionRNNPPORunnerCfg().max_iterations == 1000
