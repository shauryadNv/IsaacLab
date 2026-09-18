# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-private termination terms for the DisplayPort fabric experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fabric_faulted(env: ManagerBasedRLEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Terminate environments whose fabric faulted during the current policy step."""
    action = env.action_manager.get_term(action_name)
    if not hasattr(action, "policy_telemetry"):
        raise TypeError(f"Action term '{action_name}' does not expose fabric policy telemetry.")
    return action.policy_telemetry.faulted
