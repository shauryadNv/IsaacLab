# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms specific to the deploy manipulation environments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions import RelativeJointPositionAction, RelativeJointPositionActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv


class GraspStabilizedRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint-position action that keeps the selected gear attached to the grasp frame.

    The gear-assembly policy controls only the arm joints; it does not command the gripper during
    insertion. This action term preserves the standard relative arm-joint action and, before each
    simulator substep, rewrites the selected gear pose from the fingertip-midpoint grasp target
    used at reset. This removes reset-time contact slip from the held payload while keeping the
    gear geometry active for contacts with the base and shafts.
    """

    cfg: GraspStabilizedRelativeJointPositionActionCfg

    def __init__(self, cfg: GraspStabilizedRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        """Initialize the grasp-stabilized joint action.

        Args:
            cfg: Action configuration.
            env: Environment instance.
        """
        super().__init__(cfg, env)

        eef_indices, _ = self._asset.find_bodies([cfg.end_effector_body_name])
        if len(eef_indices) == 0:
            raise ValueError(f"End-effector body '{cfg.end_effector_body_name}' not found in robot.")
        self._eef_idx = eef_indices[0]
        left_indices, _ = self._asset.find_bodies([cfg.left_fingertip_body_name])
        right_indices, _ = self._asset.find_bodies([cfg.right_fingertip_body_name])
        if len(left_indices) == 0 or len(right_indices) == 0:
            raise ValueError(
                "Fingertip bodies used for grasp stabilization were not found: "
                f"{cfg.left_fingertip_body_name!r}, {cfg.right_fingertip_body_name!r}."
            )
        self._left_fingertip_idx = left_indices[0]
        self._right_fingertip_idx = right_indices[0]

        self._gear_assets: tuple[RigidObject, RigidObject, RigidObject] = (
            env.scene["factory_gear_small"],
            env.scene["factory_gear_medium"],
            env.scene["factory_gear_large"],
        )
        self._gear_offsets_grasp_hub = torch.tensor(
            [
                cfg.gear_offsets_grasp_hub["gear_small"],
                cfg.gear_offsets_grasp_hub["gear_medium"],
                cfg.gear_offsets_grasp_hub["gear_large"],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._grasp_rot_offset = torch.tensor(cfg.grasp_rot_offset, device=self.device, dtype=torch.float32)
        self._grasp_rot_offset = self._grasp_rot_offset.unsqueeze(0).repeat(self.num_envs, 1)
        self._env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._zero_root_vel = torch.zeros((self.num_envs, 6), device=self.device)

    def apply_actions(self) -> None:
        """Apply arm targets and stabilize the selected gear at the fingertip midpoint."""
        super().apply_actions()
        self._write_selected_gear_grasp_pose()

    def _write_selected_gear_grasp_pose(self) -> None:
        """Write the selected gear pose from the current fingertip-midpoint target."""
        if not hasattr(self._env, "_gear_type_manager"):
            return

        gear_type_indices = self._env._gear_type_manager.get_all_gear_type_indices()
        eef_quat = self._asset.data.body_link_quat_w.torch[:, self._eef_idx]
        fingertip_midpoint_pos = 0.5 * (
            self._asset.data.body_link_pos_w.torch[:, self._left_fingertip_idx]
            + self._asset.data.body_link_pos_w.torch[:, self._right_fingertip_idx]
        )

        grasp_offsets = self._gear_offsets_grasp_hub[gear_type_indices]
        gear_pos = fingertip_midpoint_pos - math_utils.quat_apply(eef_quat, grasp_offsets)
        gear_quat = math_utils.quat_mul(eef_quat, math_utils.quat_conjugate(self._grasp_rot_offset))
        gear_vel = self._zero_root_vel

        for gear_idx, gear_asset in enumerate(self._gear_assets):
            mask = gear_type_indices == gear_idx
            if not torch.any(mask):
                continue
            env_ids = self._env_ids[mask]
            root_pose = torch.cat((gear_pos[mask], gear_quat[mask]), dim=-1)
            gear_asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            gear_asset.write_root_velocity_to_sim_index(root_velocity=gear_vel[mask], env_ids=env_ids)


@configclass
class GraspStabilizedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`GraspStabilizedRelativeJointPositionAction`."""

    class_type: type[GraspStabilizedRelativeJointPositionAction] = GraspStabilizedRelativeJointPositionAction

    end_effector_body_name: str = MISSING
    """End-effector body used as the grasp frame."""

    grasp_rot_offset: Sequence[float] = MISSING
    """Gear-to-end-effector grasp rotation as a quaternion ``(x, y, z, w)``."""

    gear_offsets_grasp_hub: dict[str, Sequence[float]] = MISSING
    """Per-gear offsets [m] from gear origin to the fingertip-midpoint grasp target."""

    left_fingertip_body_name: str = "left_finger_tip"
    """Left fingertip body used to compute the grasp midpoint."""

    right_fingertip_body_name: str = "right_finger_tip"
    """Right fingertip body used to compute the grasp midpoint."""
