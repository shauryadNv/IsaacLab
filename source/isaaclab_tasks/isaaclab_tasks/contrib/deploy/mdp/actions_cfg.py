# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-specific action configuration classes."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import (
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
    """OSC action that exports scaled pose deltas for LEAPP instead of joint efforts.

    Training / play still run the full operational-space controller in simulation.
    During LEAPP export the graph output is the scaled ``pose_rel`` command
    (meters + axis-angle radians), matching on-robot Cartesian impedance deploy.
    """

    class_type: type | str = (
        "isaaclab_tasks.contrib.deploy.mdp.actions:DeployOperationalSpaceControllerAction"
    )
