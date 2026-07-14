# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-specific action configuration classes."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.utils import configclass


@configclass
class DeployRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Relative joint-position action that exports absolute joint targets."""

    class_type: type | str = "{DIR}.actions:DeployRelativeJointPositionAction"
