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
from isaaclab.utils.math import apply_delta_pose, combine_frame_transforms, quat_apply, subtract_frame_transforms


def _keypoint_offsets_6d(device: str | torch.device) -> torch.Tensor:
    """Return the seven unit keypoint offsets used by the insertion reward."""
    corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32)
    return torch.cat((corners, -corners[-3:]), dim=0)


def _expand_action_scale(scale: float | Sequence[float], size: int) -> tuple[float, ...]:
    """Expand a scalar or validate a task-space action scale."""
    if isinstance(scale, int | float):
        return (float(scale),) * size
    if len(scale) < size:
        raise ValueError(f"Task-space action scale must contain at least {size} values.")
    return tuple(float(value) for value in scale[:size])


def _task_space_pose_scale(cfg: ManagerBasedRLEnvCfg) -> tuple[float, ...]:
    """Resolve translation and rotation scales from the configured arm action."""
    arm_action_cfg = cfg.actions.arm_action
    position_scale = getattr(arm_action_cfg, "position_scale", None)
    orientation_scale = getattr(arm_action_cfg, "orientation_scale", None)
    if position_scale is None:
        for objective_cfg in getattr(arm_action_cfg, "objectives", []):
            if hasattr(objective_cfg, "body_name"):
                return _expand_action_scale(objective_cfg.scale, 6)
    if position_scale is None or orientation_scale is None:
        raise ValueError("Task-space action diagnostics require position and orientation scales.")
    return _expand_action_scale(position_scale, 3) + _expand_action_scale(orientation_scale, 3)


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
    * ``Metrics/task_space_lateral_action_rms_m``: RMS lateral translation
      command after applying the action scale [m].
    * ``Metrics/task_space_lateral_action_delta_rms_m``: RMS step-to-step
      change in the lateral translation command [m].
    * ``Metrics/task_space_lateral_spectrum_*``: Relative lateral action power
      in low, search, and high-frequency bands plus the dominant frequency.
    * ``Metrics/task_space_tcp_tracking_error_m``: Mean position error between
      the commanded and realized 150 mm TCP after one policy step [m].
    * ``Metrics/terminal_success_rate_xy_*``: Cumulative terminal success
      grouped by the reset's initial XY-position offset.

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

        self._log_task_space_action_metrics = bool(getattr(cfg, "log_task_space_action_metrics", False))
        pose_scale = _task_space_pose_scale(cfg) if self._log_task_space_action_metrics else (1.0,) * 6
        self._task_space_pose_scale = torch.tensor(
            pose_scale,
            device=self.device,
            dtype=torch.float32,
        )
        self._task_space_position_scale = self._task_space_pose_scale[:3]
        task_space_axis = torch.tensor(
            getattr(cfg, "task_space_action_insertion_axis", (1.0, 0.0, 0.0)),
            device=self.device,
            dtype=torch.float32,
        )
        task_space_axis_norm = torch.linalg.vector_norm(task_space_axis)
        if not bool(torch.isfinite(task_space_axis_norm)) or task_space_axis_norm.item() <= 0.0:
            raise ValueError("task_space_action_insertion_axis must be finite and non-zero.")
        self._task_space_action_insertion_axis = task_space_axis / task_space_axis_norm
        self._previous_lateral_action_w = torch.zeros(self.num_envs, 3, device=self.device)
        robot = self.scene["robot"]
        flange_body_ids, _ = robot.find_bodies("flange")
        if self._log_task_space_action_metrics and len(flange_body_ids) != 1:
            raise ValueError(f"Expected one flange body for task-space diagnostics, found {flange_body_ids}.")
        self._task_space_flange_body_id = flange_body_ids[0] if flange_body_ids else 0
        self._task_space_tcp_offset = torch.tensor(
            getattr(cfg, "task_space_diagnostic_tcp_offset", (0.0, 0.0, 0.15)),
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.num_envs, 1)
        self._task_space_target_tcp_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._task_space_action_spectrum_window = int(getattr(cfg, "task_space_action_spectrum_window", 64))
        if self._task_space_action_spectrum_window < 4:
            raise ValueError("task_space_action_spectrum_window must be at least four policy steps.")
        self._task_space_action_history_w = torch.zeros(
            self.num_envs, self._task_space_action_spectrum_window, 3, device=self.device
        )
        self._task_space_action_history_index = 0
        self._task_space_action_history_samples = 0
        self._task_space_action_spectrum_low_hz = float(getattr(cfg, "task_space_action_spectrum_low_hz", 2.0))
        self._task_space_action_spectrum_high_hz = float(getattr(cfg, "task_space_action_spectrum_high_hz", 8.0))
        self._policy_dt = float(cfg.sim.dt * cfg.decimation)
        nyquist_hz = 0.5 / self._policy_dt
        if not 0.0 < self._task_space_action_spectrum_low_hz < self._task_space_action_spectrum_high_hz < nyquist_hz:
            raise ValueError("Task-space spectrum bands must be ordered below the policy Nyquist frequency.")

        offset_bin_edges = tuple(float(value) for value in getattr(cfg, "success_xy_offset_bin_edges", ()))
        if any(value <= 0.0 for value in offset_bin_edges) or any(
            left >= right for left, right in zip(offset_bin_edges, offset_bin_edges[1:])
        ):
            raise ValueError("success_xy_offset_bin_edges must be positive and strictly increasing.")
        self._success_xy_offset_bin_edges = torch.tensor(offset_bin_edges, device=self.device)
        num_offset_bins = len(offset_bin_edges) + 1
        self._terminal_success_by_xy_offset = torch.zeros(num_offset_bins, device=self.device)
        self._terminal_count_by_xy_offset = torch.zeros(num_offset_bins, device=self.device)
        edge_labels = [f"{edge * 1000:g}".replace(".", "p") for edge in offset_bin_edges]
        self._success_xy_offset_bin_labels = []
        lower_label = "0"
        for upper_label in edge_labels:
            self._success_xy_offset_bin_labels.append(f"{lower_label}_{upper_label}mm")
            lower_label = upper_label
        self._success_xy_offset_bin_labels.append(f"over_{lower_label}mm")
        if not hasattr(self, "_displayport_initial_xy_offset_m"):
            self._displayport_initial_xy_offset_m = torch.zeros(self.num_envs, device=self.device)
            self._displayport_initial_xy_offset_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

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

    def _compute_flange_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the flange-origin pose in the robot root frame."""
        robot = self.scene["robot"]
        return subtract_frame_transforms(
            robot.data.root_link_pos_w.torch,
            robot.data.root_link_quat_w.torch,
            robot.data.body_pos_w.torch[:, self._task_space_flange_body_id],
            robot.data.body_quat_w.torch[:, self._task_space_flange_body_id],
        )

    def _compute_tcp_pos_b(self, flange_pos_b: torch.Tensor, flange_quat_b: torch.Tensor) -> torch.Tensor:
        """Return the configured TCP position in the robot root frame [m]."""
        return combine_frame_transforms(
            flange_pos_b,
            flange_quat_b,
            self._task_space_tcp_offset,
            self._success_identity_quat,
        )[0]

    def _compute_task_space_action_spectrum(self, lateral_action_w: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute lateral action power by frequency band over a rolling window."""
        self._task_space_action_history_w[:, self._task_space_action_history_index] = lateral_action_w
        self._task_space_action_history_index = (
            self._task_space_action_history_index + 1
        ) % self._task_space_action_spectrum_window
        self._task_space_action_history_samples = min(
            self._task_space_action_history_samples + 1, self._task_space_action_spectrum_window
        )

        zero = torch.zeros((), device=self.device)
        metrics = {
            "Metrics/task_space_lateral_spectrum_peak_hz": zero,
            "Metrics/task_space_lateral_spectrum_low_fraction": zero,
            "Metrics/task_space_lateral_spectrum_search_fraction": zero,
            "Metrics/task_space_lateral_spectrum_high_fraction": zero,
        }
        if self._task_space_action_history_samples < self._task_space_action_spectrum_window:
            return metrics

        history = self._task_space_action_history_w - self._task_space_action_history_w.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft(history, dim=1)
        mean_power = torch.square(torch.abs(spectrum)).sum(dim=-1).mean(dim=0)
        mean_power[0] = 0.0
        frequencies = torch.fft.rfftfreq(self._task_space_action_spectrum_window, d=self._policy_dt, device=self.device)
        total_power = torch.clamp(mean_power.sum(), min=torch.finfo(mean_power.dtype).eps)
        low_mask = (frequencies > 0.0) & (frequencies < self._task_space_action_spectrum_low_hz)
        search_mask = (frequencies >= self._task_space_action_spectrum_low_hz) & (
            frequencies < self._task_space_action_spectrum_high_hz
        )
        high_mask = frequencies >= self._task_space_action_spectrum_high_hz
        metrics["Metrics/task_space_lateral_spectrum_peak_hz"] = frequencies[torch.argmax(mean_power)]
        metrics["Metrics/task_space_lateral_spectrum_low_fraction"] = mean_power[low_mask].sum() / total_power
        metrics["Metrics/task_space_lateral_spectrum_search_fraction"] = mean_power[search_mask].sum() / total_power
        metrics["Metrics/task_space_lateral_spectrum_high_fraction"] = mean_power[high_mask].sum() / total_power
        return metrics

    def _compute_task_space_action_metrics(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute lateral command metrics and the next-step TCP target."""
        if action.shape[-1] < 6:
            raise ValueError("Task-space action diagnostics require six pose action dimensions.")

        delta_pose_b = torch.clamp(action[:, :6], min=-0.5, max=0.5) * self._task_space_pose_scale
        translation_r = delta_pose_b[:, :3]
        robot = self.scene["robot"]
        socket = self.scene[self._success_socket_asset]
        translation_w = quat_apply(robot.data.root_link_quat_w.torch, translation_r)
        insertion_axis = self._task_space_action_insertion_axis.unsqueeze(0).expand(self.num_envs, -1)
        insertion_axis_w = quat_apply(socket.data.root_link_quat_w.torch, insertion_axis)
        lateral_action_w = (
            translation_w - torch.sum(translation_w * insertion_axis_w, dim=-1, keepdim=True) * insertion_axis_w
        )
        lateral_delta_w = lateral_action_w - self._previous_lateral_action_w
        self._previous_lateral_action_w.copy_(lateral_action_w)

        flange_pos_b, flange_quat_b = self._compute_flange_pose_b()
        target_flange_pos_b, target_flange_quat_b = apply_delta_pose(flange_pos_b, flange_quat_b, delta_pose_b)
        self._task_space_target_tcp_pos_b.copy_(self._compute_tcp_pos_b(target_flange_pos_b, target_flange_quat_b))

        lateral_norm = torch.linalg.vector_norm(lateral_action_w, dim=-1)
        lateral_delta_norm = torch.linalg.vector_norm(lateral_delta_w, dim=-1)
        metrics = {
            "Metrics/task_space_lateral_action_rms_m": torch.sqrt(torch.mean(torch.square(lateral_norm))),
            "Metrics/task_space_lateral_action_delta_rms_m": torch.sqrt(torch.mean(torch.square(lateral_delta_norm))),
        }
        metrics.update(self._compute_task_space_action_spectrum(lateral_action_w))
        return metrics

    def _compute_task_space_tracking_metrics(self) -> dict[str, torch.Tensor]:
        """Compute realized 150 mm TCP error against this step's command."""
        flange_pos_b, flange_quat_b = self._compute_flange_pose_b()
        tcp_pos_b = self._compute_tcp_pos_b(flange_pos_b, flange_quat_b)
        tracking_error = torch.linalg.vector_norm(tcp_pos_b - self._task_space_target_tcp_pos_b, dim=-1)
        return {"Metrics/task_space_tcp_tracking_error_m": tracking_error.mean()}

    def _log_terminal_success_by_xy_offset(self, log: dict[str, torch.Tensor]) -> None:
        """Publish cumulative terminal success grouped by initial XY offset."""
        for index, label in enumerate(self._success_xy_offset_bin_labels):
            if self._terminal_count_by_xy_offset[index].item() > 0.0:
                log[f"Metrics/terminal_success_rate_xy_{label}"] = (
                    self._terminal_success_by_xy_offset[index] / self._terminal_count_by_xy_offset[index]
                )

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
        task_space_action_metrics = {}
        if self._log_task_space_action_metrics:
            task_space_action_metrics = self._compute_task_space_action_metrics(action)

        obs_buf, reward_buf, terminated, time_outs, extras = super().step(action)
        log = extras.setdefault("log", {})
        if self._log_success_metrics:
            is_success, position_error, keypoint_error = self._compute_success()
            log["Metrics/success_rate"] = is_success.float().mean()
            log["Metrics/plug_socket_pos_error_m"] = position_error.mean()
            log["Metrics/plug_socket_keypoint_dist_m"] = keypoint_error.mean()
        log.update(task_space_action_metrics)
        if self._log_task_space_action_metrics:
            log.update(self._compute_task_space_tracking_metrics())
        if self._physics_watchdog_enabled:
            self._update_physics_watchdog(log)
        return obs_buf, reward_buf, terminated, time_outs, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Record terminal success before resetting selected environments."""
        terminal_success = None
        terminal_success_by_offset_available = False
        if getattr(self, "_log_success_metrics", False):
            is_success, _, _ = self._compute_success()
            terminal_success = is_success[env_ids].float().mean()
            if hasattr(self, "_terminal_success_by_xy_offset"):
                initial_offset = self._displayport_initial_xy_offset_m[env_ids]
                valid_offset = self._displayport_initial_xy_offset_valid[env_ids]
                offset_bins = torch.bucketize(initial_offset, self._success_xy_offset_bin_edges)
                terminal_is_success = is_success[env_ids].float()
                for index in range(len(self._success_xy_offset_bin_labels)):
                    in_bin = valid_offset & (offset_bins == index)
                    count = in_bin.sum()
                    if count.item() > 0:
                        self._terminal_success_by_xy_offset[index] += terminal_is_success[in_bin].sum()
                        self._terminal_count_by_xy_offset[index] += count
                        terminal_success_by_offset_available = True

        super()._reset_idx(env_ids)

        if hasattr(self, "_previous_lateral_action_w"):
            self._previous_lateral_action_w[env_ids] = 0.0
        if terminal_success is not None:
            self.extras["log"]["Metrics/terminal_success_rate"] = terminal_success
        if terminal_success_by_offset_available:
            self._log_terminal_success_by_xy_offset(self.extras["log"])
