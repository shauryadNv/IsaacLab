# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based DisplayPort insertion environment with success logging.

The success calculation only reads scene state, so it is shared by PhysX and
Newton task variants and does not change the task MDP.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn
from isaaclab.utils.math import combine_frame_transforms


def _keypoint_offsets_6d(device: str | torch.device) -> torch.Tensor:
    """Return the seven unit keypoint offsets used by the insertion reward."""
    corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32)
    return torch.cat((corners, -corners[-3:]), dim=0)


class DisplayportInsertionEnv(ManagerBasedRLEnv):
    """DisplayPort insertion environment that logs task-success metrics.

    The environment adds the following scalars to ``extras["log"]``:

    * ``Metrics/success_rate``: Fraction of environments whose mate-frame
      origins are within the configured success threshold.
    * ``Metrics/terminal_success_rate``: Success fraction immediately before
      completed environments are reset.
    * ``Metrics/plug_socket_pos_error_m``: Mean mate-frame origin error [m].
    * ``Metrics/plug_socket_keypoint_dist_m``: Mean keypoint error [m].

    Success logging does not modify observations, actions, rewards, or
    terminations.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: Environment configuration.
            render_mode: Gym render mode.
            **kwargs: Additional arguments forwarded to
                :class:`~isaaclab.envs.ManagerBasedRLEnv`.
        """
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self._log_success_metrics = bool(getattr(cfg, "log_success_metrics", True))
        self._success_socket_asset = str(getattr(cfg, "success_socket_asset", "dp_socket"))
        self._success_plug_asset = str(getattr(cfg, "success_plug_asset", "dp_plug"))
        self._success_pos_threshold = float(getattr(cfg, "success_pos_threshold", 0.003))
        self._success_keypoint_scale = float(getattr(cfg, "success_keypoint_scale", 0.15))

        self._success_socket_offset = torch.tensor(
            getattr(cfg, "success_socket_offset", [0.0, 0.0, 0.0]), device=self.device, dtype=torch.float32
        )
        self._success_plug_offset = torch.tensor(
            getattr(cfg, "success_plug_offset", [0.0, 0.0, 0.0]), device=self.device, dtype=torch.float32
        )
        self._success_plug_goal_rot_inv = torch.tensor(
            getattr(cfg, "success_plug_goal_rot_inv", [0.0, 0.0, 0.0, 1.0]),
            device=self.device,
            dtype=torch.float32,
        )
        self._success_identity_quat = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], device=self.device, dtype=torch.float32
        ).repeat(self.num_envs, 1)
        self._success_kp_offsets = _keypoint_offsets_6d(self.device) * self._success_keypoint_scale

    def _compute_success(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute success, mate-frame position error, and keypoint error."""
        socket = self.scene[self._success_socket_asset]
        plug = self.scene[self._success_plug_asset]

        socket_pos = socket.data.root_link_pos_w.torch
        socket_quat = socket.data.root_link_quat_w.torch
        plug_pos = plug.data.root_link_pos_w.torch
        plug_quat = plug.data.root_link_quat_w.torch

        socket_offset = self._success_socket_offset.unsqueeze(0).expand(self.num_envs, -1)
        plug_offset = self._success_plug_offset.unsqueeze(0).expand(self.num_envs, -1)
        plug_goal_rot_inv = self._success_plug_goal_rot_inv.unsqueeze(0).expand(self.num_envs, -1)

        socket_frame_pos, socket_frame_quat = combine_frame_transforms(
            socket_pos, socket_quat, socket_offset, self._success_identity_quat
        )
        plug_frame_pos, plug_frame_quat = combine_frame_transforms(plug_pos, plug_quat, plug_offset, plug_goal_rot_inv)
        position_error = torch.linalg.norm(plug_frame_pos - socket_frame_pos, dim=-1)

        num_keypoints = self._success_kp_offsets.shape[0]
        offsets = self._success_kp_offsets.unsqueeze(0).expand(self.num_envs, -1, -1).reshape(-1, 3)
        identity_quats = self._success_identity_quat.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 4)

        socket_keypoints = combine_frame_transforms(
            socket_frame_pos.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 3),
            socket_frame_quat.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 4),
            offsets,
            identity_quats,
        )[0].reshape(self.num_envs, num_keypoints, 3)
        plug_keypoints = combine_frame_transforms(
            plug_frame_pos.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 3),
            plug_frame_quat.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 4),
            offsets,
            identity_quats,
        )[0].reshape(self.num_envs, num_keypoints, 3)
        keypoint_error = torch.linalg.norm(plug_keypoints - socket_keypoints, dim=-1).mean(dim=-1)

        return position_error < self._success_pos_threshold, position_error, keypoint_error

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Advance the environment and append instantaneous success metrics."""
        obs_buf, reward_buf, terminated, time_outs, extras = super().step(action)
        if self._log_success_metrics:
            is_success, position_error, keypoint_error = self._compute_success()
            log = extras.setdefault("log", {})
            log["Metrics/success_rate"] = is_success.float().mean()
            log["Metrics/plug_socket_pos_error_m"] = position_error.mean()
            log["Metrics/plug_socket_keypoint_dist_m"] = keypoint_error.mean()
        return obs_buf, reward_buf, terminated, time_outs, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Record terminal success before resetting selected environments."""
        terminal_success = None
        if getattr(self, "_log_success_metrics", False):
            is_success, _, _ = self._compute_success()
            terminal_success = is_success[env_ids].float().mean()

        super()._reset_idx(env_ids)

        if terminal_success is not None:
            self.extras["log"]["Metrics/terminal_success_rate"] = terminal_success
