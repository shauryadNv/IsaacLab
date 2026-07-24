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
from isaaclab.utils.math import combine_frame_transforms, quat_apply


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
    * ``Metrics/physics_watchdog_violation_rate``: Fraction of environments
      violating an enabled physics-watchdog limit.
    * ``Metrics/physics_watchdog_min_insertion_depth_m``: Minimum signed plug
      depth relative to the seated mate plane [m].

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

        self._physics_watchdog_enabled = bool(getattr(cfg, "physics_watchdog_enabled", False))
        self._physics_watchdog_fail_fast = bool(getattr(cfg, "physics_watchdog_fail_fast", False))
        watchdog_axis = torch.tensor(
            getattr(cfg, "physics_watchdog_insertion_axis", (1.0, 0.0, 0.0)),
            device=self.device,
            dtype=torch.float32,
        )
        watchdog_axis_norm = torch.linalg.vector_norm(watchdog_axis)
        if not bool(torch.isfinite(watchdog_axis_norm)) or watchdog_axis_norm.item() <= 0.0:
            raise ValueError("physics_watchdog_insertion_axis must be finite and non-zero.")
        self._physics_watchdog_insertion_axis = watchdog_axis / watchdog_axis_norm
        self._physics_watchdog_max_overtravel = float(getattr(cfg, "physics_watchdog_max_overtravel", 0.003))
        self._physics_watchdog_max_plug_linear_speed = float(
            getattr(cfg, "physics_watchdog_max_plug_linear_speed", 2.0)
        )
        self._physics_watchdog_max_plug_angular_speed = float(
            getattr(cfg, "physics_watchdog_max_plug_angular_speed", 50.0)
        )
        self._physics_watchdog_max_violation_fraction = float(
            getattr(cfg, "physics_watchdog_max_violation_fraction", 0.05)
        )
        self._physics_watchdog_check_interval = int(getattr(cfg, "physics_watchdog_check_interval", 8))
        self._physics_watchdog_consecutive_checks = int(getattr(cfg, "physics_watchdog_consecutive_checks", 3))
        if self._physics_watchdog_check_interval <= 0 or self._physics_watchdog_consecutive_checks <= 0:
            raise ValueError("Physics-watchdog check intervals must be positive.")
        if not 0.0 <= self._physics_watchdog_max_violation_fraction <= 1.0:
            raise ValueError("physics_watchdog_max_violation_fraction must be in [0, 1].")
        self._physics_watchdog_step_count = 0
        self._physics_watchdog_failed_checks = 0

    def _compute_mate_frames(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute socket and plug mate-frame poses in world coordinates."""
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
        return socket_frame_pos, socket_frame_quat, plug_frame_pos, plug_frame_quat

    def _compute_success(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute success, mate-frame position error, and keypoint error."""
        socket_frame_pos, socket_frame_quat, plug_frame_pos, plug_frame_quat = self._compute_mate_frames()
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

    def _update_physics_watchdog(self, log: dict[str, torch.Tensor]) -> None:
        """Log severe plug-state violations and optionally stop unstable runs."""
        socket_frame_pos, socket_frame_quat, plug_frame_pos, plug_frame_quat = self._compute_mate_frames()
        plug = self.scene[self._success_plug_asset]
        plug_linear_velocity = plug.data.root_link_lin_vel_w.torch
        plug_angular_velocity = plug.data.root_link_ang_vel_w.torch

        insertion_axis = self._physics_watchdog_insertion_axis.unsqueeze(0).expand(self.num_envs, -1)
        insertion_axis_w = quat_apply(socket_frame_quat, insertion_axis)
        insertion_depth = torch.sum((plug_frame_pos - socket_frame_pos) * insertion_axis_w, dim=-1)
        plug_linear_speed = torch.linalg.vector_norm(plug_linear_velocity, dim=-1)
        plug_angular_speed = torch.linalg.vector_norm(plug_angular_velocity, dim=-1)
        state_is_finite = torch.isfinite(
            torch.cat(
                (
                    socket_frame_pos,
                    socket_frame_quat,
                    plug_frame_pos,
                    plug_frame_quat,
                    plug_linear_velocity,
                    plug_angular_velocity,
                ),
                dim=-1,
            )
        ).all(dim=-1)
        violation = (
            ~state_is_finite
            | (insertion_depth < -self._physics_watchdog_max_overtravel)
            | (plug_linear_speed > self._physics_watchdog_max_plug_linear_speed)
            | (plug_angular_speed > self._physics_watchdog_max_plug_angular_speed)
        )

        violation_rate = violation.float().mean()
        safe_depth = torch.nan_to_num(insertion_depth, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        safe_linear_speed = torch.nan_to_num(plug_linear_speed, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
        safe_angular_speed = torch.nan_to_num(plug_angular_speed, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
        log["Metrics/physics_watchdog_violation_rate"] = violation_rate
        log["Metrics/physics_watchdog_min_insertion_depth_m"] = safe_depth.min()
        log["Metrics/physics_watchdog_max_plug_linear_speed_m_s"] = safe_linear_speed.max()
        log["Metrics/physics_watchdog_max_plug_angular_speed_rad_s"] = safe_angular_speed.max()

        if not self._physics_watchdog_fail_fast:
            return

        self._physics_watchdog_step_count += 1
        if self._physics_watchdog_step_count % self._physics_watchdog_check_interval != 0:
            return

        if violation_rate.item() > self._physics_watchdog_max_violation_fraction:
            self._physics_watchdog_failed_checks += 1
        else:
            self._physics_watchdog_failed_checks = 0
        if self._physics_watchdog_failed_checks >= self._physics_watchdog_consecutive_checks:
            raise RuntimeError(
                "DisplayPort physics watchdog detected persistent instability: "
                f"violation_rate={violation_rate.item():.3f}, "
                f"min_insertion_depth={safe_depth.min().item():.6f} m, "
                f"max_linear_speed={safe_linear_speed.max().item():.3f} m/s, "
                f"max_angular_speed={safe_angular_speed.max().item():.3f} rad/s."
            )

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Advance the environment and append instantaneous success metrics."""
        obs_buf, reward_buf, terminated, time_outs, extras = super().step(action)
        log = extras.setdefault("log", {})
        if self._log_success_metrics:
            is_success, position_error, keypoint_error = self._compute_success()
            log["Metrics/success_rate"] = is_success.float().mean()
            log["Metrics/plug_socket_pos_error_m"] = position_error.mean()
            log["Metrics/plug_socket_keypoint_dist_m"] = keypoint_error.mean()
        if self._physics_watchdog_enabled:
            self._update_physics_watchdog(log)
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
