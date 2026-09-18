# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the DisplayPort geometric-fabric experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def fabric_joint_pos(env: ManagerBasedEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Return artificial joint positions [rad], ordered by the fabric chain."""
    action = env.action_manager.get_term(action_name)
    if not hasattr(action, "fabric_joint_position"):
        raise TypeError(f"Action term '{action_name}' does not expose fabric joint position.")
    return action.fabric_joint_position


def fabric_joint_vel(env: ManagerBasedEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Return artificial joint velocities [rad/s], ordered by the fabric chain."""
    action = env.action_manager.get_term(action_name)
    if not hasattr(action, "fabric_joint_velocity"):
        raise TypeError(f"Action term '{action_name}' does not expose fabric joint velocity.")
    return action.fabric_joint_velocity
