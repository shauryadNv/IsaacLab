# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Deploy DisplayPort insertion environment."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_pos_env_cfg import (
    Rizon4sGravDisplayportInsertionEnvCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    PLUG_GOAL_ROT_INV,
    PLUG_INSERTION_OFFSET,
    SOCKET_INSERTION_OFFSET,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.insertion_env import (
    DisplayportInsertionEnv,
    _keypoint_offsets_6d,
)
from isaaclab_tasks.utils.hydra import resolve_presets


class _TensorView:
    """Minimal tensor wrapper matching an asset-data field."""

    def __init__(self, value: torch.Tensor):
        self.torch = value


class _Scene(dict):
    """Minimal scene mapping exposing the environment count."""

    def __init__(self, num_envs: int, **assets):
        super().__init__(assets)
        self.num_envs = num_envs


def _asset(
    position: torch.Tensor,
    linear_velocity: torch.Tensor | None = None,
    angular_velocity: torch.Tensor | None = None,
) -> SimpleNamespace:
    quaternion = torch.zeros((position.shape[0], 4), dtype=torch.float32)
    quaternion[:, 3] = 1.0
    if linear_velocity is None:
        linear_velocity = torch.zeros_like(position)
    if angular_velocity is None:
        angular_velocity = torch.zeros_like(position)
    return SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=_TensorView(position),
            root_link_quat_w=_TensorView(quaternion),
            root_link_lin_vel_w=_TensorView(linear_velocity),
            root_link_ang_vel_w=_TensorView(angular_velocity),
        )
    )


def test_displayport_success_uses_mate_frame_position_threshold():
    """The metric should match the PhysX three-millimetre success criterion."""
    socket_pos = torch.zeros((2, 3), dtype=torch.float32)
    plug_pos = torch.tensor([[0.002, 0.0, 0.0], [0.004, 0.0, 0.0]], dtype=torch.float32)

    env = DisplayportInsertionEnv.__new__(DisplayportInsertionEnv)
    env._is_closed = True
    env.scene = _Scene(2, socket=_asset(socket_pos), plug=_asset(plug_pos))
    env._success_socket_asset = "socket"
    env._success_plug_asset = "plug"
    env._success_pos_threshold = 0.003
    env._success_socket_offset = torch.zeros(3)
    env._success_plug_offset = torch.zeros(3)
    env._success_plug_goal_rot_inv = torch.tensor([0.0, 0.0, 0.0, 1.0])
    env._success_identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    env._success_kp_offsets = _keypoint_offsets_6d("cpu") * 0.15

    is_success, position_error, keypoint_error = env._compute_success()

    assert is_success.tolist() == [True, False]
    torch.testing.assert_close(position_error, torch.tensor([0.002, 0.004]))
    torch.testing.assert_close(keypoint_error, position_error)


def test_displayport_physics_watchdog_separates_telemetry_from_fail_fast():
    """The watchdog should always log and only reject when fail-fast is enabled."""
    socket_pos = torch.zeros((2, 3), dtype=torch.float32)
    plug_pos = torch.tensor([[-0.004, 0.0, 0.0], [0.010, 0.0, 0.0]], dtype=torch.float32)

    env = DisplayportInsertionEnv.__new__(DisplayportInsertionEnv)
    env._is_closed = True
    env.scene = _Scene(2, socket=_asset(socket_pos), plug=_asset(plug_pos))
    env._success_socket_asset = "socket"
    env._success_plug_asset = "plug"
    env._success_socket_offset = torch.zeros(3)
    env._success_plug_offset = torch.zeros(3)
    env._success_plug_goal_rot_inv = torch.tensor([0.0, 0.0, 0.0, 1.0])
    env._success_identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    env._physics_watchdog_insertion_axis = torch.tensor([1.0, 0.0, 0.0])
    env._physics_watchdog_max_overtravel = 0.003
    env._physics_watchdog_max_plug_linear_speed = 2.0
    env._physics_watchdog_max_plug_angular_speed = 50.0
    env._physics_watchdog_max_violation_fraction = 0.25
    env._physics_watchdog_check_interval = 1
    env._physics_watchdog_consecutive_checks = 2
    env._physics_watchdog_step_count = 0
    env._physics_watchdog_failed_checks = 0
    env._physics_watchdog_fail_fast = False
    log = {}

    env._update_physics_watchdog(log)

    torch.testing.assert_close(log["Metrics/physics_watchdog_violation_rate"], torch.tensor(0.5))
    torch.testing.assert_close(log["Metrics/physics_watchdog_min_insertion_depth_m"], torch.tensor(-0.004))
    env._update_physics_watchdog(log)

    env._physics_watchdog_fail_fast = True
    env._physics_watchdog_failed_checks = 0
    env._update_physics_watchdog(log)
    with pytest.raises(RuntimeError, match="persistent instability"):
        env._update_physics_watchdog(log)


def test_displayport_success_config_matches_reward_frames():
    """The metric should use exactly the insertion frames used by the reward."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_hydroelastic"})
    reward_params = env_cfg.rewards.plug_socket_keypoint_tracking.params

    assert env_cfg.success_pos_threshold == 0.003
    assert env_cfg.success_socket_offset == SOCKET_INSERTION_OFFSET == reward_params["offset_1"]
    assert env_cfg.success_plug_offset == PLUG_INSERTION_OFFSET == reward_params["offset_2"]
    assert env_cfg.success_plug_goal_rot_inv == PLUG_GOAL_ROT_INV == reward_params["rot_offset_2"]


def test_displayport_task_variants_use_success_logging_environment():
    """Every DisplayPort variant should publish the same success metrics."""
    expected_entry_point = "isaaclab_tasks.contrib.deploy.cable_insertion.insertion_env:DisplayportInsertionEnv"
    task_ids = (
        "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav",
        "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel",
        "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-NoJointVel",
        "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Newton-IK",
    )

    for task_id in task_ids:
        assert gym.spec(task_id).entry_point == expected_entry_point
