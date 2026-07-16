# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Deploy DisplayPort insertion environment configuration."""

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
