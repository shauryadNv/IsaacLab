# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Deploy DisplayPort insertion environment configuration."""

from pxr import Usd

from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.agents.rsl_rl_ppo_cfg import (
    Rizon4sGravDisplayportInsertionRNNPPORunnerCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.ik_newton_env_cfg import (
    Rizon4sGravDisplayportInsertionIKNewtonEnvCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_pos_env_cfg import (
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
    assert env_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config is not None
    assert env_cfg.sim.physics.collision_cfg.max_triangle_pairs == 1_000_000
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
    assert newton_cfg.sim.physics.solver_cfg.use_mujoco_contacts is True
    assert newton_cfg.scene.dp_plug.spawn.usd_path.endswith("display_port_plug_newton_sdf.usda")
    assert newton_cfg.scene.dp_socket.spawn.usd_path.endswith("display_port_socket_newton_sdf.usda")


def test_displayport_hard_sdf_uses_point_contacts_with_precomputed_sdfs():
    """Hard SDF should cook mesh volumes without enabling hydroelastic contact."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_sdf"})

    assert env_cfg.sim.physics.collision_cfg.max_triangle_pairs == 1_000_000
    assert env_cfg.scene.dp_plug.spawn.usd_path.endswith("display_port_plug_newton_sdf.usda")
    assert env_cfg.scene.dp_socket.spawn.usd_path.endswith("display_port_socket_newton_sdf.usda")
    assert env_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config is None
    assert env_cfg.sim.physics.solver_cfg.use_mujoco_contacts is False


def test_displayport_assets_author_newton_sdf_per_active_collider():
    """Newton overlays should author SDF metadata only on active collision meshes."""
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


def test_displayport_newton_ik_commands_flange_without_actor_velocity():
    """The deployment-oriented IK task should command the flange with a six-dimensional action."""
    env_cfg = Rizon4sGravDisplayportInsertionIKNewtonEnvCfg()
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert pose_objective.body_name == "flange"
    assert pose_objective.use_relative_mode is True
    assert pose_objective.scale == 0.01
    assert env_cfg.observations.policy.joint_vel is None


def test_displayport_no_joint_velocity_hides_velocity_from_actor_only():
    """No-joint-velocity training should retain privileged critic velocity."""
    env_cfg = Rizon4sGravDisplayportInsertionNoJointVelEnvCfg()

    assert env_cfg.observations.policy.joint_vel is None
    assert env_cfg.observations.critic.joint_vel is not None


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
