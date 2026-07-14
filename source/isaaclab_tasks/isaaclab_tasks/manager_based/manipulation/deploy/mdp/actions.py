# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-specific action terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import DeployRelativeJointPositionActionCfg


_LEAPP_TRACED_OBSERVATION_INPUTS = "_leapp_traced_observation_inputs"


def _real_env(env):
    return object.__getattribute__(env, "_real_env") if type(env).__name__ == "_EnvProxy" else env


def _to_torch(data):
    return data.torch if hasattr(data, "torch") else wp.to_torch(data)


class DeployRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint action that exports absolute joint targets for deploy.

    During normal Isaac Lab execution this behaves like
    :class:`RelativeJointPositionAction`. During LEAPP export it reuses the
    traced policy joint-position observation so the exported graph computes:

    ``target_joint_pos = robot_joint_pos + raw_action * scale``.
    """

    cfg: DeployRelativeJointPositionActionCfg

    def __init__(self, cfg: DeployRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        current_joint_pos = None
        if type(self._asset).__name__ == "_ArticulationWriteProxy":
            traced_inputs = getattr(_real_env(self._env), _LEAPP_TRACED_OBSERVATION_INPUTS, {})
            current_joint_pos = traced_inputs.get(f"{self.cfg.asset_name}_joint_pos")

        if current_joint_pos is None:
            current_joint_pos = _to_torch(self._asset.data.joint_pos)[:, self._joint_ids]

        current_actions = self.processed_actions + current_joint_pos
        self._asset.set_joint_position_target_index(target=current_actions, joint_ids=self._joint_ids)
