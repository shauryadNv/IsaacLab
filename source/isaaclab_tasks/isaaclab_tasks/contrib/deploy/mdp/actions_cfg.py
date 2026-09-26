# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-specific action configuration classes."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    OperationalSpaceControllerActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.utils.configclass import configclass


@configclass
class DeployRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for deploy relative joint actions with explicit LEAPP current-joint input."""

    class_type: type | str = "isaaclab_tasks.contrib.deploy.mdp.actions:DeployRelativeJointPositionAction"


@configclass
class DeployOperationalSpaceControllerActionCfg(OperationalSpaceControllerActionCfg):
    """OSC action that exports scaled pose deltas for LEAPP instead of joint efforts."""

    class_type: type | str = "isaaclab_tasks.contrib.deploy.mdp.actions:DeployOperationalSpaceControllerAction"

    payload_gravity_compensation: bool = False
    """Whether to add a feed-forward joint torque that cancels the weight of a held payload.

    The arm links are gravity-free in the task-space environments, but a grasped rigid object is not,
    and with ``pose_rel`` targets re-based on the measured pose every step nothing restores the pose
    the payload's weight pulls the end effector away from, so the arm slowly sinks. The real robot's
    stiff position servo holds its target instead. Defaults to False.
    """

    payload_asset_name: str | None = None
    """Name of the rigid object whose weight is compensated. Required when
    :attr:`payload_gravity_compensation` is True."""

    payload_mass_scale: float = 1.0
    """Scale on the payload's current mass used for compensation (1.0 cancels its weight exactly)."""


@configclass
class DeployDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """DiffIK action that exports scaled pose deltas for LEAPP instead of joint targets."""

    class_type: type | str = "isaaclab_tasks.contrib.deploy.mdp.actions:DeployDifferentialInverseKinematicsAction"
