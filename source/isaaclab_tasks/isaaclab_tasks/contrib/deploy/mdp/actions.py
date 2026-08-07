# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-specific action terms for LEAPP export workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.utils.leapp.leapp_semantics import POSE6_ELEMENT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import (
        DeployOperationalSpaceControllerActionCfg,
        DeployRelativeJointPositionActionCfg,
    )

_LEAPP_TRACED_OBSERVATION_INPUTS = "_leapp_traced_observation_inputs"
_LEAPP_CONSUMED_OBSERVATION_INPUTS = "_leapp_consumed_observation_inputs"


def _leapp_real_env(env):
    real_env = object.__getattribute__(env, "_real_env") if type(env).__name__ == "_EnvProxy" else env
    return real_env


def _tensor_data_to_torch(data):
    """Return a torch tensor view for Isaac Lab data stored as torch or Warp-backed data."""
    return data.torch if hasattr(data, "torch") else wp.to_torch(data)


def _get_observation_term_from_buffer(env, group_name: str, term_name: str):
    """Return a term slice from the cached observation buffer."""
    obs_buffer = getattr(env, "obs_buf", None)
    if obs_buffer is None:
        obs_buffer = getattr(getattr(env, "observation_manager", None), "_obs_buffer", None)
    if not obs_buffer or group_name not in obs_buffer:
        return None

    group_obs = obs_buffer[group_name]
    if isinstance(group_obs, dict):
        return group_obs.get(term_name)

    obs_manager = getattr(env, "observation_manager", None)
    if obs_manager is None:
        return None

    term_names = obs_manager.active_terms.get(group_name, [])
    if term_name not in term_names:
        return None

    term_index = term_names.index(term_name)
    term_dims = obs_manager.group_obs_term_dim[group_name]
    concat_dim = obs_manager._group_obs_concatenate_dim[group_name]
    if concat_dim > 0:
        concat_dim -= 1

    start = sum(dim[concat_dim] for dim in term_dims[:term_index])
    length = term_dims[term_index][concat_dim]
    return group_obs.narrow(dim=concat_dim, start=start, length=length)


def _pop_leapp_traced_observation_input(env, name: str, *, group_name: str, term_name: str):
    """Consume one traced observation tensor for the current LEAPP action trace."""
    real_env = _leapp_real_env(env)
    consumed_inputs = getattr(real_env, _LEAPP_CONSUMED_OBSERVATION_INPUTS, None)
    if consumed_inputs is None:
        consumed_inputs = set()
        setattr(real_env, _LEAPP_CONSUMED_OBSERVATION_INPUTS, consumed_inputs)

    if name in consumed_inputs:
        return None

    traced_inputs = getattr(real_env, _LEAPP_TRACED_OBSERVATION_INPUTS, {})
    traced_tensor = traced_inputs.pop(name, None)
    if traced_tensor is None:
        traced_tensor = _get_observation_term_from_buffer(real_env, group_name, term_name)

    if traced_tensor is not None:
        consumed_inputs.add(name)
    return traced_tensor


def _is_leapp_observation_input_consumed(env, name: str) -> bool:
    real_env = _leapp_real_env(env)
    return name in getattr(real_env, _LEAPP_CONSUMED_OBSERVATION_INPUTS, set())


class DeployRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint action that reuses traced current joint observations during LEAPP export."""

    def __init__(self, cfg: DeployRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        asset = self._asset
        if type(asset).__name__ == "_ArticulationWriteProxy":
            observation_input_name = f"{self.cfg.asset_name}_joint_pos"
            current_joint_pos = _pop_leapp_traced_observation_input(
                self._env,
                observation_input_name,
                group_name="policy",
                term_name="joint_pos",
            )
            if current_joint_pos is None:
                if not _is_leapp_observation_input_consumed(self._env, observation_input_name):
                    raise RuntimeError(
                        "DeployRelativeJointPositionAction requires the traced "
                        f"'{self.cfg.asset_name}_joint_pos' observation during LEAPP export."
                    )
                real_asset = object.__getattribute__(asset, "_real_asset")
                current_joint_pos = _tensor_data_to_torch(real_asset.data.joint_pos)[:, self._joint_ids]
        else:
            current_joint_pos = _tensor_data_to_torch(asset.data.joint_pos)[:, self._joint_ids]

        current_actions = self.processed_actions + current_joint_pos
        self._asset.set_joint_position_target_index(target=current_actions, joint_ids=self._joint_ids)


class DeployOperationalSpaceControllerAction(OperationalSpaceControllerAction):
    """OSC action that exports scaled pose_rel deltas during LEAPP export.

    On-robot task-space deploy runs Cartesian impedance / OSC *outside* the policy.
    The exported LEAPP graph must therefore emit the policy's scaled 6-D flange
    delta (``pose_rel``), not the intermediate joint-effort writes that OSC uses
    inside the simulator.

    During export the term still applies OSC through the real articulation so
    simulation keeps moving, but annotated asset writes are skipped so
    ``processed_actions`` is captured as ``arm_action``.
    """

    def __init__(self, cfg: DeployOperationalSpaceControllerActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Consumed by LEAPP ``_collect_processed_action_fallbacks`` when this term
        # deliberately avoids annotated joint-effort writes during export.
        self._leapp_processed_action_element_names = list(POSE6_ELEMENT_NAMES)
        self._leapp_processed_action_kind = None
        self._leapp_processed_action_extra = {
            "isaaclab_connection": "action:arm_action:pose_rel",
            "target_types": list(cfg.controller_cfg.target_types),
            "position_scale": float(cfg.position_scale),
            "orientation_scale": float(cfg.orientation_scale),
        }

    def apply_actions(self):
        asset = self._asset
        if type(asset).__name__ != "_ArticulationWriteProxy":
            super().apply_actions()
            return

        # Drive the simulator via the underlying articulation so joint-effort
        # writes are not captured as LEAPP outputs (those would be 7-D torques).
        real_asset = object.__getattribute__(asset, "_real_asset")
        self._asset = real_asset
        try:
            super().apply_actions()
        finally:
            self._asset = asset
