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
