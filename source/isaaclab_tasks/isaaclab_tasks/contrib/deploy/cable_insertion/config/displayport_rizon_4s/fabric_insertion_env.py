# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DisplayPort insertion environment with geometric-fabric health telemetry."""

from __future__ import annotations

import torch

from isaaclab_tasks.contrib.deploy.cable_insertion.insertion_env import DisplayportInsertionEnv


class DisplayportFabricInsertionEnv(DisplayportInsertionEnv):
    """Preserve insertion metrics and add per-step fabric/tracker health metrics."""

    def step(self, action: torch.Tensor):
        obs_buf, reward_buf, terminated, time_outs, extras = super().step(action)
        fabric_action = self.action_manager.get_term("arm_action")
        telemetry = fabric_action.policy_telemetry
        log = self.extras.setdefault("log", {})

        self._log_rate(log, "Fabric/fault_rate", telemetry.faulted)
        self._log_rate(log, "Fabric/nonfinite_rate", telemetry.nonfinite)
        self._log_rate(log, "Fabric/infeasible_rate", telemetry.infeasible)
        self._log_rate(log, "Fabric/cholesky_failure_rate", telemetry.cholesky_failed)
        self._log_count(log, "Fabric/action_saturation", telemetry.action_saturation_count)
        self._log_count(log, "Fabric/acceleration_bound_activation", telemetry.acceleration_bound_active_count)
        self._log_count(log, "Fabric/jerk_bound_activation", telemetry.jerk_bound_active_count)
        self._log_count(log, "Fabric/velocity_bound_activation", telemetry.velocity_bound_active_count)
        self._log_count(log, "Fabric/position_bound_activation", telemetry.position_bound_active_count)
        self._log_count(log, "Fabric/roundoff_guard_activation", telemetry.roundoff_guard_active_count)
        self._log_count(
            log,
            "Fabric/tracker_position_limit_violation",
            telemetry.tracker_position_limit_violation_count,
        )
        self._log_count(
            log,
            "Fabric/tracker_velocity_limit_violation",
            telemetry.tracker_velocity_limit_violation_count,
        )

        self._log_mean_max(
            log,
            "Fabric/tracking_position_error_rad",
            telemetry.tracking_position_error_norm,
        )
        self._log_mean_max(
            log,
            "Fabric/tracking_velocity_error_rad_s",
            telemetry.tracking_velocity_error_norm,
        )
        self._log_mean_max(log, "Fabric/maximum_acceleration_rad_s2", telemetry.maximum_acceleration)
        self._log_mean_max(log, "Fabric/maximum_jerk_rad_s3", telemetry.maximum_jerk)
        self._log_mean_max(
            log,
            "Fabric/maximum_measured_joint_acceleration_rad_s2",
            telemetry.maximum_measured_joint_acceleration,
        )
        self._log_mean_max(
            log,
            "Fabric/maximum_measured_joint_jerk_rad_s3",
            telemetry.maximum_measured_joint_jerk,
        )
        self._log_mean_max(log, "Fabric/solve_residual", telemetry.solve_residual_norm)
        log["Fabric/minimum_joint_margin_rad_mean"] = telemetry.minimum_joint_margin.mean()
        log["Fabric/minimum_joint_margin_rad_min"] = telemetry.minimum_joint_margin.amin()
        log["Fabric/metric_minimum_cholesky_diagonal_mean"] = telemetry.metric_min_cholesky_diagonal.mean()
        log["Fabric/metric_minimum_cholesky_diagonal_min"] = telemetry.metric_min_cholesky_diagonal.amin()

        self._log_count(
            log,
            "Fabric/predicted_tracker_effort_saturation",
            telemetry.tracker_effort_saturation_count,
        )
        self._log_mean_max(
            log,
            "Fabric/predicted_tracker_computed_effort_nm",
            telemetry.predicted_tracker_computed_effort,
        )
        self._log_mean_max(
            log,
            "Fabric/predicted_tracker_applied_effort_nm",
            telemetry.predicted_tracker_applied_effort,
        )
        self._log_mean_max(
            log,
            "Fabric/actual_applied_effort_nm",
            telemetry.maximum_actual_applied_effort,
        )
        self._log_mean_max(
            log,
            "Fabric/actual_applied_effort_slew_nm_s",
            telemetry.maximum_actual_applied_effort_slew,
        )
        return obs_buf, reward_buf, terminated, time_outs, extras

    @staticmethod
    def _log_rate(log: dict, name: str, value: torch.Tensor) -> None:
        """Log the fraction of environments with a Boolean condition."""
        log[name] = value.float().mean()

    @staticmethod
    def _log_count(log: dict, name: str, value: torch.Tensor) -> None:
        """Log a per-environment count's mean, maximum, and nonzero rate."""
        log[f"{name}_count_mean"] = value.float().mean()
        log[f"{name}_count_max"] = value.amax()
        log[f"{name}_rate"] = (value > 0).float().mean()

    @staticmethod
    def _log_mean_max(log: dict, name: str, value: torch.Tensor) -> None:
        """Log the mean and maximum of a per-environment scalar."""
        log[f"{name}_mean"] = value.mean()
        log[f"{name}_max"] = value.amax()
