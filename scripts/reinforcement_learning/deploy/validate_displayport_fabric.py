# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gate DisplayPort geometric-fabric training with live PhysX checks.

This script is intentionally stricter than a normal environment smoke test. It
checks the exact Isaac Sim runtime, resolved observation contract, dynamics and
kinematics parity, bounded zero-input behavior, stability of a grasped,
partially inserted plug, and a meaningful scripted search/insertion command. It
writes a JSON report and exits nonzero on any failed invariant.

Run headless (the default)::

    ./isaaclab.sh -p scripts/reinforcement_learning/deploy/validate_displayport_fabric.py \
        --task IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Fabric \
        --num_envs 4 --visualizer none

Pass ``--video --visualizer kit`` to write the required human-inspection MP4
before a training submission. This validator does not validate a real-robot
controller.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import setup_preset_cli

_TASK = "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Fabric"

parser = argparse.ArgumentParser(description="Validate the experimental DisplayPort geometric-fabric environment.")
parser.add_argument("--task", type=str, default=_TASK, help="Registered fabric task to validate.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel validation environments.")
parser.add_argument("--seed", type=int, default=123, help="Deterministic environment seed.")
parser.add_argument(
    "--parity_perturbation", type=float, default=0.02, help="Joint perturbation [rad] for parity poses."
)
parser.add_argument("--zero_hold_steps", type=int, default=30, help="Zero-action policy steps.")
parser.add_argument("--inserted_steps", type=int, default=30, help="Pre-inserted stability policy steps.")
parser.add_argument("--scripted_steps", type=int, default=180, help="Total scripted search/insertion policy steps.")
parser.add_argument(
    "--scripted_search_steps",
    type=int,
    default=60,
    help="Initial scripted transverse-search steps; remaining steps command insertion only.",
)
parser.add_argument(
    "--inserted_depth",
    type=float,
    default=0.004,
    help="Fixed shallow-engagement plug mate-point depth along positive socket x [m].",
)
parser.add_argument(
    "--scripted_approach_depth",
    type=float,
    default=0.015,
    help="Fixed scripted-start plug mate-point depth along positive socket x [m].",
)
parser.add_argument("--insertion_action", type=float, default=0.35, help="Normalized insertion-force amplitude.")
parser.add_argument("--search_action", type=float, default=0.20, help="Normalized transverse-search amplitude.")
parser.add_argument("--search_period", type=int, default=20, help="Transverse-search period [policy steps].")
parser.add_argument(
    "--max_tracking_position_error",
    type=float,
    default=0.03,
    help="Maximum physical-to-artificial joint-position error norm [rad].",
)
parser.add_argument(
    "--max_tracking_velocity_error",
    type=float,
    default=0.25,
    help="Maximum physical-to-artificial joint-velocity error norm [rad/s].",
)
parser.add_argument(
    "--max_effort_saturation_rate",
    type=float,
    default=0.02,
    help="Maximum fraction of tracker joint commands clipped by effort limits.",
)
parser.add_argument(
    "--max_measured_joint_acceleration",
    type=float,
    default=15.0,
    help="Provisional conservative maximum measured arm-joint acceleration [rad/s^2].",
)
parser.add_argument(
    "--max_measured_joint_jerk",
    type=float,
    default=750.0,
    help="Provisional conservative maximum measured arm-joint jerk [rad/s^3].",
)
parser.add_argument(
    "--max_actual_effort_slew",
    type=float,
    default=5000.0,
    help="Provisional conservative maximum actual applied-effort slew [N*m/s].",
)
parser.add_argument("--max_solve_residual", type=float, default=1.0e-3, help="Maximum fabric solve residual norm.")
parser.add_argument(
    "--min_metric_cholesky_diagonal",
    type=float,
    default=1.0e-5,
    help="Minimum accepted fabric metric Cholesky diagonal.",
)
parser.add_argument(
    "--max_reset_position_error",
    type=float,
    default=0.001,
    help="Maximum deterministic-reset axial or transverse position error [m].",
)
parser.add_argument(
    "--max_reset_orientation_error",
    type=float,
    default=math.radians(1.0),
    help="Maximum deterministic-reset plug goal-orientation error [rad].",
)
parser.add_argument(
    "--max_inserted_translation_drift",
    type=float,
    default=0.002,
    help="Maximum pre-inserted plug-to-flange translation drift [m].",
)
parser.add_argument(
    "--max_inserted_rotation_drift",
    type=float,
    default=math.radians(2.0),
    help="Maximum pre-inserted plug-to-flange rotation drift [rad].",
)
parser.add_argument(
    "--max_inserted_mate_point_drift",
    type=float,
    default=0.002,
    help="Maximum pre-inserted plug mate-point drift relative to the socket [m].",
)
parser.add_argument(
    "--max_zero_hold_joint_speed",
    type=float,
    default=0.25,
    help="Maximum measured arm-joint speed during zero-input hold [rad/s].",
)
parser.add_argument(
    "--min_scripted_insertion_displacement",
    type=float,
    default=0.010,
    help="Minimum per-environment plug mate-point progress toward the socket [m].",
)
parser.add_argument(
    "--min_scripted_search_range",
    type=float,
    default=0.001,
    help="Minimum per-environment plug mate-point range in each transverse search direction [m].",
)
parser.add_argument(
    "--max_scripted_final_mate_error",
    type=float,
    default=0.003,
    help="Maximum per-environment final plug-to-socket mate-point error [m].",
)
parser.add_argument(
    "--max_scripted_orientation_error",
    type=float,
    default=math.radians(5.0),
    help="Maximum final plug-to-socket goal-orientation error [rad].",
)
parser.add_argument(
    "--max_scripted_grasp_translation_drift",
    type=float,
    default=0.003,
    help="Maximum scripted plug-to-flange translation drift [m].",
)
parser.add_argument(
    "--max_scripted_grasp_rotation_drift",
    type=float,
    default=math.radians(5.0),
    help="Maximum scripted plug-to-flange rotation drift [rad].",
)
parser.add_argument(
    "--min_scripted_success_rate",
    type=float,
    default=1.0,
    help="Minimum final fraction of environments satisfying the task mate-point success threshold.",
)
parser.add_argument(
    "--required_success_hold_steps",
    type=int,
    default=15,
    help="Final consecutive policy steps over which scripted success must remain above threshold.",
)
parser.add_argument("--video", action="store_true", help="Record the complete validation sequence with Kit.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Recorded environment steps; defaults to all zero-hold, pre-inserted, and scripted steps.",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="artifacts/displayport_fabric_validation_video",
    help="Directory for the optional validation MP4 artifact.",
)
parser.add_argument(
    "--report",
    type=str,
    default="artifacts/displayport_fabric_validation.json",
    help="JSON validation report path.",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


import gymnasium as gym
import torch
from packaging.version import Version

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnvCfg, VideoRecorderCfg
from isaaclab.utils import get_isaac_sim_version

# Import the package explicitly so its Gym task registrations are available.
import isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s  # noqa: F401, E402
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (  # noqa: E402
    PLUG_GOAL_ROT,
    PLUG_INSERTION_OFFSET,
    SOCKET_INSERTION_OFFSET,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

_EXPECTED_ISAAC_SIM = Version("6.0.1")
_POLICY_TERMS = (
    "eef_pos",
    "eef_rot_6d",
    "socket_kp_pos",
    "socket_kp_rot_6d",
    "joint_pos",
    "joint_vel",
    "fabric_joint_pos",
    "fabric_joint_vel",
)
_CRITIC_TERMS = (
    "joint_pos",
    "joint_vel",
    "fabric_joint_pos",
    "fabric_joint_vel",
    "socket_kp_pos",
    "socket_kp_rot_6d",
    "plug_kp_pos",
    "plug_kp_rot_6d",
)
_REQUIRED_LOG_KEYS = (
    "Fabric/fault_rate",
    "Fabric/nonfinite_rate",
    "Fabric/infeasible_rate",
    "Fabric/cholesky_failure_rate",
    "Fabric/tracking_position_error_rad_max",
    "Fabric/tracking_velocity_error_rad_s_max",
    "Fabric/solve_residual_max",
    "Fabric/minimum_joint_margin_rad_min",
    "Fabric/metric_minimum_cholesky_diagonal_min",
    "Fabric/predicted_tracker_effort_saturation_rate",
    "Fabric/tracker_position_limit_violation_count_max",
    "Fabric/tracker_velocity_limit_violation_count_max",
    "Fabric/maximum_measured_joint_acceleration_rad_s2_max",
    "Fabric/maximum_measured_joint_jerk_rad_s3_max",
    "Fabric/actual_applied_effort_nm_max",
    "Fabric/actual_applied_effort_slew_nm_s_max",
)


def _scalar_max(value: torch.Tensor) -> float:
    return float(torch.as_tensor(value).detach().amax().cpu())


def _scalar_min(value: torch.Tensor) -> float:
    return float(torch.as_tensor(value).detach().amin().cpu())


def _write_report(report: dict[str, Any]) -> None:
    path = Path(args_cli.report).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote validation report: {path}")


def _validate_observations(env, observations: dict[str, torch.Tensor]) -> dict[str, Any]:
    manager = env.observation_manager
    expected_terms = {"policy": _POLICY_TERMS, "critic": _CRITIC_TERMS}
    result: dict[str, Any] = {}
    for group, terms in expected_terms.items():
        actual_terms = tuple(manager.active_terms[group])
        if actual_terms != terms:
            raise RuntimeError(f"{group} observation order is {actual_terms}, expected {terms}.")
        declared_dim = tuple(manager.group_obs_dim[group])
        if declared_dim != (46,):
            raise RuntimeError(f"{group} declared dimension is {declared_dim}, expected (46,).")
        value = observations[group]
        if value.shape != (env.num_envs, 46):
            raise RuntimeError(f"{group} observation shape is {tuple(value.shape)}, expected {(env.num_envs, 46)}.")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{group} observations contain a non-finite value.")
        result[group] = {
            "terms": list(actual_terms),
            "term_dimensions": [list(dim) for dim in manager.group_obs_term_dim[group]],
            "shape": list(value.shape),
        }
    return result


@dataclass
class _TelemetryAccumulator:
    """Aggregate hard failures and health extrema across policy steps."""

    steps: int = 0
    environment_samples: int = 0
    tracker_joint_samples: int = 0
    fabric_joint_samples: int = 0
    effort_saturation_count: int = 0
    action_saturation_count: int = 0
    roundoff_guard_count: int = 0
    acceleration_bound_count: int = 0
    jerk_bound_count: int = 0
    velocity_bound_count: int = 0
    position_bound_count: int = 0
    tracker_position_limit_violation_count: int = 0
    tracker_velocity_limit_violation_count: int = 0
    maximum_tracking_position_error_rad: float = 0.0
    maximum_tracking_velocity_error_rad_s: float = 0.0
    maximum_acceleration_rad_s2: float = 0.0
    maximum_jerk_rad_s3: float = 0.0
    maximum_measured_joint_acceleration_rad_s2: float = 0.0
    maximum_measured_joint_jerk_rad_s3: float = 0.0
    maximum_actual_applied_effort_nm: float = 0.0
    maximum_actual_applied_effort_slew_nm_s: float = 0.0
    maximum_solve_residual: float = 0.0
    minimum_joint_margin_rad: float = math.inf
    minimum_metric_cholesky_diagonal: float = math.inf
    maximum_predicted_applied_effort_nm: float = 0.0
    phases: set[str] = field(default_factory=set)

    def update(self, phase: str, action_term, extras: dict[str, Any]) -> None:
        telemetry = action_term.policy_telemetry
        self.steps += 1
        self.phases.add(phase)
        self.environment_samples += action_term.num_envs
        joint_count = len(action_term.cfg.joint_names)
        self.tracker_joint_samples += action_term.num_envs * joint_count * action_term.cfg.expected_policy_decimation
        policy_dt = action_term.cfg.expected_policy_decimation * action_term.cfg.expected_physics_dt
        fabric_updates = round(policy_dt / action_term.cfg.fabric_cfg.dt)
        self.fabric_joint_samples += action_term.num_envs * joint_count * fabric_updates

        hard_failures = {
            "faulted": telemetry.faulted,
            "nonfinite": telemetry.nonfinite,
            "infeasible": telemetry.infeasible,
            "cholesky_failed": telemetry.cholesky_failed,
        }
        failed = [name for name, value in hard_failures.items() if bool(torch.any(value))]
        if failed:
            raise RuntimeError(f"Fabric telemetry reported {', '.join(failed)} during {phase}.")
        self.effort_saturation_count += int(telemetry.tracker_effort_saturation_count.sum().item())
        self.action_saturation_count += int(telemetry.action_saturation_count.sum().item())
        self.roundoff_guard_count += int(telemetry.roundoff_guard_active_count.sum().item())
        self.acceleration_bound_count += int(telemetry.acceleration_bound_active_count.sum().item())
        self.jerk_bound_count += int(telemetry.jerk_bound_active_count.sum().item())
        self.velocity_bound_count += int(telemetry.velocity_bound_active_count.sum().item())
        self.position_bound_count += int(telemetry.position_bound_active_count.sum().item())
        self.tracker_position_limit_violation_count += int(
            telemetry.tracker_position_limit_violation_count.sum().item()
        )
        self.tracker_velocity_limit_violation_count += int(
            telemetry.tracker_velocity_limit_violation_count.sum().item()
        )
        self.maximum_tracking_position_error_rad = max(
            self.maximum_tracking_position_error_rad, _scalar_max(telemetry.tracking_position_error_norm)
        )
        self.maximum_tracking_velocity_error_rad_s = max(
            self.maximum_tracking_velocity_error_rad_s, _scalar_max(telemetry.tracking_velocity_error_norm)
        )
        self.maximum_acceleration_rad_s2 = max(
            self.maximum_acceleration_rad_s2, _scalar_max(telemetry.maximum_acceleration)
        )
        self.maximum_jerk_rad_s3 = max(self.maximum_jerk_rad_s3, _scalar_max(telemetry.maximum_jerk))
        self.maximum_measured_joint_acceleration_rad_s2 = max(
            self.maximum_measured_joint_acceleration_rad_s2,
            _scalar_max(telemetry.maximum_measured_joint_acceleration),
        )
        self.maximum_measured_joint_jerk_rad_s3 = max(
            self.maximum_measured_joint_jerk_rad_s3,
            _scalar_max(telemetry.maximum_measured_joint_jerk),
        )
        self.maximum_actual_applied_effort_nm = max(
            self.maximum_actual_applied_effort_nm,
            _scalar_max(telemetry.maximum_actual_applied_effort),
        )
        self.maximum_actual_applied_effort_slew_nm_s = max(
            self.maximum_actual_applied_effort_slew_nm_s,
            _scalar_max(telemetry.maximum_actual_applied_effort_slew),
        )
        self.maximum_solve_residual = max(self.maximum_solve_residual, _scalar_max(telemetry.solve_residual_norm))
        self.minimum_joint_margin_rad = min(self.minimum_joint_margin_rad, _scalar_min(telemetry.minimum_joint_margin))
        self.minimum_metric_cholesky_diagonal = min(
            self.minimum_metric_cholesky_diagonal,
            _scalar_min(telemetry.metric_min_cholesky_diagonal),
        )
        self.maximum_predicted_applied_effort_nm = max(
            self.maximum_predicted_applied_effort_nm,
            _scalar_max(telemetry.predicted_tracker_applied_effort),
        )

        log = extras.get("log", {})
        missing = [key for key in _REQUIRED_LOG_KEYS if key not in log]
        if missing:
            raise RuntimeError(f"Missing fabric environment telemetry keys: {missing}.")
        for key in _REQUIRED_LOG_KEYS:
            if not bool(torch.isfinite(torch.as_tensor(log[key])).all()):
                raise RuntimeError(f"Environment telemetry '{key}' is non-finite during {phase}.")

    def validate(self, action_term) -> None:
        cfg = action_term.cfg.fabric_cfg
        acceleration_limit = max(cfg.max_joint_acceleration) + cfg.bound_tolerance
        jerk_limit = max(cfg.max_joint_jerk) + cfg.bound_tolerance / cfg.dt
        effort_rate = self.effort_saturation_count / max(self.tracker_joint_samples, 1)
        failures: list[str] = []
        if self.action_saturation_count:
            failures.append(f"action saturation count={self.action_saturation_count}")
        if self.roundoff_guard_count:
            failures.append(f"roundoff guard count={self.roundoff_guard_count}")
        if self.tracker_position_limit_violation_count:
            failures.append(f"tracker position-limit violation count={self.tracker_position_limit_violation_count}")
        if self.tracker_velocity_limit_violation_count:
            failures.append(f"tracker velocity-limit violation count={self.tracker_velocity_limit_violation_count}")
        if self.maximum_tracking_position_error_rad > args_cli.max_tracking_position_error:
            failures.append(
                f"tracking position error={self.maximum_tracking_position_error_rad:.4g} rad"
                f">{args_cli.max_tracking_position_error:.4g} rad"
            )
        if self.maximum_tracking_velocity_error_rad_s > args_cli.max_tracking_velocity_error:
            failures.append(
                f"tracking velocity error={self.maximum_tracking_velocity_error_rad_s:.4g} rad/s"
                f">{args_cli.max_tracking_velocity_error:.4g} rad/s"
            )
        if effort_rate > args_cli.max_effort_saturation_rate:
            failures.append(f"effort saturation rate={effort_rate:.4%}>{args_cli.max_effort_saturation_rate:.4%}")
        if self.maximum_solve_residual > args_cli.max_solve_residual:
            failures.append(f"solve residual={self.maximum_solve_residual:.4g}>{args_cli.max_solve_residual:.4g}")
        if self.minimum_metric_cholesky_diagonal < args_cli.min_metric_cholesky_diagonal:
            failures.append(
                f"metric Cholesky diagonal={self.minimum_metric_cholesky_diagonal:.4g}"
                f"<{args_cli.min_metric_cholesky_diagonal:.4g}"
            )
        if self.minimum_joint_margin_rad < -cfg.bound_tolerance:
            failures.append(f"joint margin={self.minimum_joint_margin_rad:.4g} rad")
        if self.maximum_acceleration_rad_s2 > acceleration_limit:
            failures.append(f"acceleration={self.maximum_acceleration_rad_s2:.4g}>{acceleration_limit:.4g} rad/s^2")
        if self.maximum_jerk_rad_s3 > jerk_limit:
            failures.append(f"jerk={self.maximum_jerk_rad_s3:.4g}>{jerk_limit:.4g} rad/s^3")
        if self.maximum_measured_joint_acceleration_rad_s2 > args_cli.max_measured_joint_acceleration:
            failures.append(
                f"measured acceleration={self.maximum_measured_joint_acceleration_rad_s2:.4g} rad/s^2"
                f">{args_cli.max_measured_joint_acceleration:.4g} rad/s^2"
            )
        if self.maximum_measured_joint_jerk_rad_s3 > args_cli.max_measured_joint_jerk:
            failures.append(
                f"measured jerk={self.maximum_measured_joint_jerk_rad_s3:.4g} rad/s^3"
                f">{args_cli.max_measured_joint_jerk:.4g} rad/s^3"
            )
        if self.maximum_actual_applied_effort_slew_nm_s > args_cli.max_actual_effort_slew:
            failures.append(
                f"actual applied-effort slew={self.maximum_actual_applied_effort_slew_nm_s:.4g} N*m/s"
                f">{args_cli.max_actual_effort_slew:.4g} N*m/s"
            )
        if failures:
            raise RuntimeError("Fabric health thresholds failed: " + "; ".join(failures))

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "phases": sorted(self.phases),
            "effort_saturation_rate": self.effort_saturation_count / max(self.tracker_joint_samples, 1),
            "action_saturation_count": self.action_saturation_count,
            "roundoff_guard_count": self.roundoff_guard_count,
            "acceleration_bound_activation_rate": self.acceleration_bound_count / max(self.fabric_joint_samples, 1),
            "jerk_bound_activation_rate": self.jerk_bound_count / max(self.fabric_joint_samples, 1),
            "velocity_bound_activation_rate": self.velocity_bound_count / max(self.fabric_joint_samples, 1),
            "position_bound_activation_rate": self.position_bound_count / max(self.fabric_joint_samples, 1),
            "tracker_position_limit_violation_count": self.tracker_position_limit_violation_count,
            "tracker_velocity_limit_violation_count": self.tracker_velocity_limit_violation_count,
            "maximum_tracking_position_error_rad": self.maximum_tracking_position_error_rad,
            "maximum_tracking_velocity_error_rad_s": self.maximum_tracking_velocity_error_rad_s,
            "maximum_artificial_acceleration_rad_s2": self.maximum_acceleration_rad_s2,
            "maximum_artificial_jerk_rad_s3": self.maximum_jerk_rad_s3,
            "maximum_measured_joint_acceleration_rad_s2": self.maximum_measured_joint_acceleration_rad_s2,
            "maximum_measured_joint_jerk_rad_s3": self.maximum_measured_joint_jerk_rad_s3,
            "maximum_predicted_applied_effort_nm": self.maximum_predicted_applied_effort_nm,
            "maximum_actual_applied_effort_nm": self.maximum_actual_applied_effort_nm,
            "maximum_actual_applied_effort_slew_nm_s": self.maximum_actual_applied_effort_slew_nm_s,
            "maximum_solve_residual": self.maximum_solve_residual,
            "minimum_joint_margin_rad": self.minimum_joint_margin_rad,
            "minimum_metric_cholesky_diagonal": self.minimum_metric_cholesky_diagonal,
            "provisional_physical_thresholds": {
                "maximum_measured_joint_acceleration_rad_s2": args_cli.max_measured_joint_acceleration,
                "maximum_measured_joint_jerk_rad_s3": args_cli.max_measured_joint_jerk,
                "maximum_actual_applied_effort_slew_nm_s": args_cli.max_actual_effort_slew,
            },
        }


def _step(env, action: torch.Tensor, phase: str, accumulator: _TelemetryAccumulator):
    observations, _, terminated, time_outs, extras = env.step(action)
    action_term = env.unwrapped.action_manager.get_term("arm_action")
    accumulator.update(phase, action_term, extras)
    if bool(torch.any(terminated)) or bool(torch.any(time_outs)):
        raise RuntimeError(f"Unexpected episode reset during validation phase '{phase}'.")
    _validate_observations(env.unwrapped, observations)
    return observations


def _refresh_after_joint_write(env) -> None:
    env.scene["robot"].write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)


def _validate_dynamics_and_kinematics(env, action_term) -> list[dict[str, Any]]:
    robot = env.scene["robot"]
    joint_ids, names = robot.find_joints(action_term.cfg.joint_names, preserve_order=True)
    if names != action_term.cfg.joint_names:
        raise RuntimeError(f"Resolved arm joint order {names} does not match {action_term.cfg.joint_names}.")

    original_position = robot.data.joint_pos.torch[:, joint_ids].clone()
    limits = robot.data.joint_pos_limits.torch[:, joint_ids]
    signs = torch.tensor((1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0), device=env.device)
    perturbation = args_cli.parity_perturbation * signs.unsqueeze(0)
    poses = (
        ("reset", original_position),
        ("positive", original_position + perturbation),
        ("negative", original_position - perturbation),
    )
    results: list[dict[str, Any]] = []
    for name, requested in poses:
        target = torch.maximum(torch.minimum(requested, limits[..., 1] - 0.02), limits[..., 0] + 0.02)
        robot.write_joint_position_to_sim_index(position=target, joint_ids=joint_ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(target), joint_ids=joint_ids)
        _refresh_after_joint_write(env)
        action_term.reset()
        dynamics = action_term.validate_dynamics()
        parity = action_term.validate_kinematics_parity()
        results.append({"pose": name, "dynamics": dynamics, "kinematics": parity})
    return results


@dataclass(frozen=True)
class _InsertionState:
    """Measured plug/socket insertion geometry for every environment."""

    socket_keypoint_w: torch.Tensor
    plug_keypoint_w: torch.Tensor
    plug_to_socket_w: torch.Tensor
    insertion_axis_w: torch.Tensor
    transverse_y_axis_w: torch.Tensor
    transverse_z_axis_w: torch.Tensor
    signed_depth_m: torch.Tensor
    transverse_y_m: torch.Tensor
    transverse_z_m: torch.Tensor
    mate_error_m: torch.Tensor
    orientation_error_rad: torch.Tensor


def _configure_curriculum_reset(env, *, at_goal: bool, depth_m: float) -> None:
    """Configure the resolved reset event for a deterministic aligned reset."""
    term_cfg = env.event_manager.get_term_cfg("reset_plug_curriculum")
    curriculum = term_cfg.func
    required_attributes = (
        "at_goal_prob",
        "at_goal_prob_final",
        "at_goal_depth_range",
        "approach_depth_range",
        "normal_pose_range",
    )
    missing = [name for name in required_attributes if not hasattr(curriculum, name)]
    if missing:
        raise RuntimeError(f"Resolved reset_plug_curriculum is missing attributes: {missing}.")

    curriculum.at_goal_prob = 1.0 if at_goal else 0.0
    curriculum.at_goal_prob_final = None
    curriculum.at_goal_depth_range = [depth_m, depth_m]
    curriculum.approach_depth_range = [depth_m, depth_m]
    curriculum.normal_pose_range = {axis: [0.0, 0.0] for axis in ("x", "y", "z")}


def _insertion_state(env) -> _InsertionState:
    """Return insertion keypoint geometry and goal-orientation error."""
    socket = env.scene["dp_socket"]
    plug = env.scene["dp_plug"]
    socket_position = socket.data.root_pos_w.torch
    socket_quaternion = socket.data.root_quat_w.torch
    plug_position = plug.data.root_pos_w.torch
    plug_quaternion = plug.data.root_quat_w.torch
    socket_offset = torch.tensor(SOCKET_INSERTION_OFFSET, device=env.device).expand(env.num_envs, -1)
    plug_offset = torch.tensor(PLUG_INSERTION_OFFSET, device=env.device).expand(env.num_envs, -1)
    goal_rotation = torch.tensor(PLUG_GOAL_ROT, device=env.device).expand(env.num_envs, -1)
    local_axes = torch.eye(3, device=env.device).expand(env.num_envs, -1, -1)

    socket_keypoint = socket_position + math_utils.quat_apply(socket_quaternion, socket_offset)
    plug_keypoint = plug_position + math_utils.quat_apply(plug_quaternion, plug_offset)
    plug_to_socket = plug_keypoint - socket_keypoint
    insertion_axis = math_utils.quat_apply(socket_quaternion, local_axes[:, 0])
    transverse_y_axis = math_utils.quat_apply(socket_quaternion, local_axes[:, 1])
    transverse_z_axis = math_utils.quat_apply(socket_quaternion, local_axes[:, 2])
    desired_plug_quaternion = math_utils.quat_mul(socket_quaternion, goal_rotation)

    return _InsertionState(
        socket_keypoint_w=socket_keypoint,
        plug_keypoint_w=plug_keypoint,
        plug_to_socket_w=plug_to_socket,
        insertion_axis_w=insertion_axis,
        transverse_y_axis_w=transverse_y_axis,
        transverse_z_axis_w=transverse_z_axis,
        signed_depth_m=torch.sum(plug_to_socket * insertion_axis, dim=-1),
        transverse_y_m=torch.sum(plug_to_socket * transverse_y_axis, dim=-1),
        transverse_z_m=torch.sum(plug_to_socket * transverse_z_axis, dim=-1),
        mate_error_m=torch.linalg.vector_norm(plug_to_socket, dim=-1),
        orientation_error_rad=math_utils.quat_error_magnitude(plug_quaternion, desired_plug_quaternion),
    )


def _plug_pose_in_flange(env, action_term) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the plug-root pose expressed in the flange frame."""
    robot = env.scene["robot"]
    plug = env.scene["dp_plug"]
    body_ids, _ = robot.find_bodies(action_term.cfg.body_name)
    return math_utils.subtract_frame_transforms(
        robot.data.body_pos_w.torch[:, body_ids[0]],
        robot.data.body_quat_w.torch[:, body_ids[0]],
        plug.data.root_pos_w.torch,
        plug.data.root_quat_w.torch,
    )


def _socket_axes_b(env) -> torch.Tensor:
    """Return socket-frame axes expressed in the robot base frame."""
    robot = env.scene["robot"]
    socket = env.scene["dp_socket"]
    _, socket_quaternion_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        socket.data.root_pos_w.torch,
        socket.data.root_quat_w.torch,
    )
    return math_utils.matrix_from_quat(socket_quaternion_b)


def _validate_reset_alignment(state: _InsertionState, expected_depth_m: float, phase: str) -> None:
    """Verify that curriculum reset and grasp placement produced an aligned held plug."""
    depth_error = _scalar_max(torch.abs(state.signed_depth_m - expected_depth_m))
    transverse_error = _scalar_max(torch.sqrt(state.transverse_y_m.square() + state.transverse_z_m.square()))
    orientation_error = _scalar_max(state.orientation_error_rad)
    failures: list[str] = []
    if depth_error > args_cli.max_reset_position_error:
        failures.append(f"depth error={depth_error:.4g} m>{args_cli.max_reset_position_error:.4g} m")
    if transverse_error > args_cli.max_reset_position_error:
        failures.append(f"transverse error={transverse_error:.4g} m>{args_cli.max_reset_position_error:.4g} m")
    if orientation_error > args_cli.max_reset_orientation_error:
        failures.append(f"orientation error={orientation_error:.4g} rad>{args_cli.max_reset_orientation_error:.4g} rad")
    if failures:
        raise RuntimeError(f"Deterministic {phase} reset alignment failed: " + "; ".join(failures))


def _run_live_checks(env) -> dict[str, Any]:
    observations, _ = env.reset(seed=args_cli.seed)
    observation_contract = _validate_observations(env, observations)
    action_term = env.action_manager.get_term("arm_action")
    parity = _validate_dynamics_and_kinematics(env, action_term)
    accumulator = _TelemetryAccumulator()
    zero_action = torch.zeros(env.num_envs, action_term.action_dim, device=env.device)

    # Use an aligned, grasped approach reset so zero-input health is deterministic.
    _configure_curriculum_reset(env, at_goal=False, depth_m=args_cli.scripted_approach_depth)
    observations, _ = env.reset(seed=args_cli.seed)
    _validate_observations(env, observations)
    _validate_reset_alignment(_insertion_state(env), args_cli.scripted_approach_depth, "zero-hold")
    max_zero_speed = 0.0
    for _ in range(args_cli.zero_hold_steps):
        _step(env, zero_action, "zero_hold", accumulator)
        arm_ids, _ = env.scene["robot"].find_joints(action_term.cfg.joint_names, preserve_order=True)
        measured_velocity = env.scene["robot"].data.joint_vel.torch[:, arm_ids]
        max_zero_speed = max(max_zero_speed, _scalar_max(torch.abs(measured_velocity)))
    if max_zero_speed > args_cli.max_zero_hold_joint_speed:
        raise RuntimeError(
            f"Zero-input hold reached {max_zero_speed:.4g} rad/s, above {args_cli.max_zero_hold_joint_speed:.4g} rad/s."
        )

    # Exercise the task's real reset path: curriculum placement followed by the
    # configured IK/grasp event. Do not teleport the free plug after reset.
    _configure_curriculum_reset(env, at_goal=True, depth_m=args_cli.inserted_depth)
    observations, _ = env.reset(seed=args_cli.seed)
    _validate_observations(env, observations)
    inserted_initial = _insertion_state(env)
    _validate_reset_alignment(inserted_initial, args_cli.inserted_depth, "pre-inserted")
    initial_plug_in_flange_position, initial_plug_in_flange_quaternion = _plug_pose_in_flange(env, action_term)
    initial_plug_in_flange_position = initial_plug_in_flange_position.clone()
    initial_plug_in_flange_quaternion = initial_plug_in_flange_quaternion.clone()
    initial_plug_to_socket = inserted_initial.plug_to_socket_w.clone()
    max_grasp_translation_drift = 0.0
    max_grasp_rotation_drift = 0.0
    max_mate_point_drift = 0.0
    max_inserted_orientation_error = _scalar_max(inserted_initial.orientation_error_rad)
    for _ in range(args_cli.inserted_steps):
        _step(env, zero_action, "preinserted_stability", accumulator)
        state = _insertion_state(env)
        plug_in_flange_position, plug_in_flange_quaternion = _plug_pose_in_flange(env, action_term)
        max_grasp_translation_drift = max(
            max_grasp_translation_drift,
            _scalar_max(torch.linalg.vector_norm(plug_in_flange_position - initial_plug_in_flange_position, dim=-1)),
        )
        max_grasp_rotation_drift = max(
            max_grasp_rotation_drift,
            _scalar_max(math_utils.quat_error_magnitude(plug_in_flange_quaternion, initial_plug_in_flange_quaternion)),
        )
        max_mate_point_drift = max(
            max_mate_point_drift,
            _scalar_max(torch.linalg.vector_norm(state.plug_to_socket_w - initial_plug_to_socket, dim=-1)),
        )
        max_inserted_orientation_error = max(max_inserted_orientation_error, _scalar_max(state.orientation_error_rad))
    inserted_final = _insertion_state(env)
    if max_grasp_translation_drift > args_cli.max_inserted_translation_drift:
        raise RuntimeError(
            f"Pre-inserted plug-to-flange translation drift is {max_grasp_translation_drift:.4g} m, above "
            f"{args_cli.max_inserted_translation_drift:.4g} m."
        )
    if max_grasp_rotation_drift > args_cli.max_inserted_rotation_drift:
        raise RuntimeError(
            f"Pre-inserted plug-to-flange rotation drift is {max_grasp_rotation_drift:.4g} rad, above "
            f"{args_cli.max_inserted_rotation_drift:.4g} rad."
        )
    if max_mate_point_drift > args_cli.max_inserted_mate_point_drift:
        raise RuntimeError(
            f"Pre-inserted mate-point drift is {max_mate_point_drift:.4g} m, above "
            f"{args_cli.max_inserted_mate_point_drift:.4g} m."
        )

    # Start the scripted check aligned outside the socket and grasped by the
    # robot. Search in both socket-transverse directions, then apply insertion
    # only. Measurements use the plug/socket mate frames rather than TCP motion.
    _configure_curriculum_reset(env, at_goal=False, depth_m=args_cli.scripted_approach_depth)
    observations, _ = env.reset(seed=args_cli.seed)
    _validate_observations(env, observations)
    scripted_initial = _insertion_state(env)
    _validate_reset_alignment(scripted_initial, args_cli.scripted_approach_depth, "scripted approach")
    scripted_plug_in_flange_position, scripted_plug_in_flange_quaternion = _plug_pose_in_flange(env, action_term)
    scripted_plug_in_flange_position = scripted_plug_in_flange_position.clone()
    scripted_plug_in_flange_quaternion = scripted_plug_in_flange_quaternion.clone()

    socket_axes_b = _socket_axes_b(env)
    insertion_axis_b = socket_axes_b[:, :, 0]
    transverse_y_b = socket_axes_b[:, :, 1]
    transverse_z_b = socket_axes_b[:, :, 2]
    transverse_y_positions = [scripted_initial.transverse_y_m.clone()]
    transverse_z_positions = [scripted_initial.transverse_z_m.clone()]
    success_history: list[torch.Tensor] = []
    minimum_mate_error = scripted_initial.mate_error_m.clone()
    max_scripted_grasp_translation_drift = 0.0
    max_scripted_grasp_rotation_drift = 0.0
    for step in range(args_cli.scripted_steps):
        if step < args_cli.scripted_search_steps:
            phase = 2.0 * math.pi * step / args_cli.search_period
            search = args_cli.search_action * (math.sin(phase) * transverse_y_b + math.cos(phase) * transverse_z_b)
        else:
            search = torch.zeros_like(insertion_axis_b)
        linear_action = -args_cli.insertion_action * insertion_axis_b + search
        action = torch.cat((linear_action, torch.zeros_like(linear_action)), dim=-1)
        _step(env, action, "scripted_insertion_search", accumulator)

        state = _insertion_state(env)
        if step < args_cli.scripted_search_steps:
            transverse_y_positions.append(state.transverse_y_m.clone())
            transverse_z_positions.append(state.transverse_z_m.clone())
        plug_in_flange_position, plug_in_flange_quaternion = _plug_pose_in_flange(env, action_term)
        max_scripted_grasp_translation_drift = max(
            max_scripted_grasp_translation_drift,
            _scalar_max(torch.linalg.vector_norm(plug_in_flange_position - scripted_plug_in_flange_position, dim=-1)),
        )
        max_scripted_grasp_rotation_drift = max(
            max_scripted_grasp_rotation_drift,
            _scalar_max(math_utils.quat_error_magnitude(plug_in_flange_quaternion, scripted_plug_in_flange_quaternion)),
        )
        minimum_mate_error = torch.minimum(minimum_mate_error, state.mate_error_m)
        if not hasattr(env, "_compute_success"):
            raise RuntimeError("Fabric task environment does not expose insertion success computation.")
        success, _, _ = env._compute_success()
        success_history.append(success.clone())

    scripted_final = _insertion_state(env)
    insertion_progress = scripted_initial.signed_depth_m - scripted_final.signed_depth_m
    y_trajectory = torch.stack(transverse_y_positions)
    z_trajectory = torch.stack(transverse_z_positions)
    y_range = y_trajectory.amax(dim=0) - y_trajectory.amin(dim=0)
    z_range = z_trajectory.amax(dim=0) - z_trajectory.amin(dim=0)
    final_success = success_history[-1].float()
    hold_steps = min(args_cli.required_success_hold_steps, len(success_history))
    sustained_success_rate = torch.stack(success_history[-hold_steps:]).float().mean()

    minimum_progress = _scalar_min(insertion_progress)
    minimum_y_range = _scalar_min(y_range)
    minimum_z_range = _scalar_min(z_range)
    maximum_final_mate_error = _scalar_max(scripted_final.mate_error_m)
    maximum_final_orientation_error = _scalar_max(scripted_final.orientation_error_rad)
    final_success_rate = float(final_success.mean().cpu())
    failures: list[str] = []
    if minimum_progress < args_cli.min_scripted_insertion_displacement:
        failures.append(
            f"minimum insertion progress={minimum_progress:.4g} m<{args_cli.min_scripted_insertion_displacement:.4g} m"
        )
    if minimum_y_range < args_cli.min_scripted_search_range:
        failures.append(
            f"minimum transverse-y search range={minimum_y_range:.4g} m<{args_cli.min_scripted_search_range:.4g} m"
        )
    if minimum_z_range < args_cli.min_scripted_search_range:
        failures.append(
            f"minimum transverse-z search range={minimum_z_range:.4g} m<{args_cli.min_scripted_search_range:.4g} m"
        )
    if maximum_final_mate_error > args_cli.max_scripted_final_mate_error:
        failures.append(
            f"maximum final mate error={maximum_final_mate_error:.4g} m>{args_cli.max_scripted_final_mate_error:.4g} m"
        )
    if maximum_final_orientation_error > args_cli.max_scripted_orientation_error:
        failures.append(
            f"maximum final orientation error={maximum_final_orientation_error:.4g} rad"
            f">{args_cli.max_scripted_orientation_error:.4g} rad"
        )
    if max_scripted_grasp_translation_drift > args_cli.max_scripted_grasp_translation_drift:
        failures.append(
            f"plug-to-flange translation drift={max_scripted_grasp_translation_drift:.4g} m"
            f">{args_cli.max_scripted_grasp_translation_drift:.4g} m"
        )
    if max_scripted_grasp_rotation_drift > args_cli.max_scripted_grasp_rotation_drift:
        failures.append(
            f"plug-to-flange rotation drift={max_scripted_grasp_rotation_drift:.4g} rad"
            f">{args_cli.max_scripted_grasp_rotation_drift:.4g} rad"
        )
    if final_success_rate < args_cli.min_scripted_success_rate:
        failures.append(f"final success rate={final_success_rate:.2%}<{args_cli.min_scripted_success_rate:.2%}")
    if float(sustained_success_rate) < args_cli.min_scripted_success_rate:
        failures.append(
            f"last-{hold_steps}-step success rate={float(sustained_success_rate):.2%}"
            f"<{args_cli.min_scripted_success_rate:.2%}"
        )
    try:
        accumulator.validate(action_term)
    except RuntimeError as error:
        failures.append(str(error))
    if failures:
        raise RuntimeError("Scripted held-plug insertion thresholds failed: " + "; ".join(failures))
    return {
        "observation_contract": observation_contract,
        "dynamics_and_kinematics": parity,
        "zero_hold": {"maximum_measured_joint_speed_rad_s": max_zero_speed},
        "preinserted_stability": {
            "requested_depth_m": args_cli.inserted_depth,
            "initial_mean_signed_depth_m": float(inserted_initial.signed_depth_m.mean().cpu()),
            "final_mean_signed_depth_m": float(inserted_final.signed_depth_m.mean().cpu()),
            "maximum_plug_to_flange_translation_drift_m": max_grasp_translation_drift,
            "maximum_plug_to_flange_rotation_drift_rad": max_grasp_rotation_drift,
            "maximum_mate_point_drift_m": max_mate_point_drift,
            "maximum_goal_orientation_error_rad": max_inserted_orientation_error,
        },
        "scripted_insertion_search": {
            "requested_approach_depth_m": args_cli.scripted_approach_depth,
            "initial_mean_signed_depth_m": float(scripted_initial.signed_depth_m.mean().cpu()),
            "final_mean_signed_depth_m": float(scripted_final.signed_depth_m.mean().cpu()),
            "search_steps": args_cli.scripted_search_steps,
            "insertion_only_steps": args_cli.scripted_steps - args_cli.scripted_search_steps,
            "minimum_insertion_progress_m": minimum_progress,
            "mean_insertion_progress_m": float(insertion_progress.mean().cpu()),
            "minimum_transverse_y_range_m": minimum_y_range,
            "minimum_transverse_z_range_m": minimum_z_range,
            "mean_transverse_y_range_m": float(y_range.mean().cpu()),
            "mean_transverse_z_range_m": float(z_range.mean().cpu()),
            "mean_minimum_mate_error_reached_m": float(minimum_mate_error.mean().cpu()),
            "maximum_minimum_mate_error_reached_m": _scalar_max(minimum_mate_error),
            "mean_final_mate_error_m": float(scripted_final.mate_error_m.mean().cpu()),
            "maximum_final_mate_error_m": maximum_final_mate_error,
            "maximum_final_orientation_error_rad": maximum_final_orientation_error,
            "maximum_plug_to_flange_translation_drift_m": max_scripted_grasp_translation_drift,
            "maximum_plug_to_flange_rotation_drift_rad": max_scripted_grasp_rotation_drift,
            "final_success_rate": final_success_rate,
            "success_hold_steps": hold_steps,
            "sustained_success_rate": float(sustained_success_rate),
        },
        "telemetry": accumulator.as_dict(),
    }


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg, _agent_cfg: None) -> None:
    """Launch the exact task configuration and run the training gate."""
    if args_cli.task.split(":")[-1] not in {_TASK, f"{_TASK}-Play"}:
        raise ValueError(f"This validator only supports {_TASK} and {_TASK}-Play.")
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    for name in (
        "zero_hold_steps",
        "inserted_steps",
        "scripted_steps",
        "scripted_search_steps",
        "search_period",
        "required_success_hold_steps",
    ):
        if getattr(args_cli, name) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if args_cli.scripted_search_steps >= args_cli.scripted_steps:
        raise ValueError("--scripted_search_steps must be smaller than --scripted_steps.")
    if args_cli.required_success_hold_steps > args_cli.scripted_steps:
        raise ValueError("--required_success_hold_steps cannot exceed --scripted_steps.")
    if not 0.0 <= args_cli.min_scripted_success_rate <= 1.0:
        raise ValueError("--min_scripted_success_rate must be in [0, 1].")
    if not 0.0 <= args_cli.insertion_action <= 1.0 or not 0.0 <= args_cli.search_action <= 1.0:
        raise ValueError("--insertion_action and --search_action must be normalized amplitudes in [0, 1].")
    if args_cli.inserted_depth < 0.0 or args_cli.scripted_approach_depth <= 0.0:
        raise ValueError("Reset depths must be non-negative, with a positive scripted approach depth.")
    for name in (
        "max_measured_joint_acceleration",
        "max_measured_joint_jerk",
        "max_actual_effort_slew",
        "max_reset_position_error",
        "max_reset_orientation_error",
        "max_scripted_final_mate_error",
    ):
        if getattr(args_cli, name) <= 0.0:
            raise ValueError(f"--{name} must be positive.")
    if args_cli.video_length is not None and args_cli.video_length <= 0:
        raise ValueError("--video_length must be positive when provided.")
    required_video_length = args_cli.zero_hold_steps + args_cli.inserted_steps + args_cli.scripted_steps
    if args_cli.video_length is not None and args_cli.video_length < required_video_length:
        raise ValueError(
            f"--video_length must be at least {required_video_length} steps to cover every validation phase."
        )

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), 30.0)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    video_report = None
    video_baseline: dict[str, int] = {}
    if args_cli.video:
        selected_visualizers = getattr(args_cli, "visualizer", None)
        if isinstance(selected_visualizers, str):
            selected_visualizers = {name.strip() for name in selected_visualizers.split(",")}
        elif selected_visualizers is None:
            selected_visualizers = set()
        else:
            selected_visualizers = set(selected_visualizers)
        if "kit" not in selected_visualizers:
            raise ValueError("--video requires --visualizer kit for the visualizer:kit recording source.")
        video_dir = Path(args_cli.video_dir).expanduser().resolve()
        video_length = args_cli.video_length or required_video_length
        video_baseline = {
            str(path): path.stat().st_mtime_ns for path in video_dir.glob("displayport_fabric_validation_*.mp4")
        }
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:kit",
                output_dir=str(video_dir),
                output_filename_prefix="displayport_fabric_validation",
                video_interval=0,
                video_length=video_length,
            )
        ]
        video_report = {
            "output_dir": str(video_dir),
            "output_filename_prefix": "displayport_fabric_validation",
            "video_length_steps": video_length,
        }
    env_cfg.validate()

    report: dict[str, Any] = {
        "status": "running",
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
    }
    if video_report is not None:
        report["video"] = video_report
    env = None
    try:
        with launch_simulation(env_cfg, args_cli):
            try:
                actual_version = get_isaac_sim_version()
                report["isaac_sim_version"] = str(actual_version)
                report["expected_isaac_sim_version"] = str(_EXPECTED_ISAAC_SIM)
                if actual_version != _EXPECTED_ISAAC_SIM:
                    raise RuntimeError(
                        f"Fabric validation requires Isaac Sim {_EXPECTED_ISAAC_SIM} (wheel pin 6.0.1.0), "
                        f"but the active runtime is {actual_version}."
                    )
                env = gym.make(args_cli.task, cfg=env_cfg)
                report.update(_run_live_checks(env.unwrapped))
            finally:
                if env is not None:
                    env.close()
            if args_cli.video:
                video_dir = Path(args_cli.video_dir).expanduser().resolve()
                video_files = sorted(video_dir.glob("displayport_fabric_validation_*.mp4"))
                fresh_video_files = [
                    path for path in video_files if video_baseline.get(str(path)) != path.stat().st_mtime_ns
                ]
                if not fresh_video_files:
                    raise RuntimeError(f"Validation video was enabled but no new MP4 was written to {video_dir}.")
                report["video"]["artifacts"] = [str(path) for path in fresh_video_files]
            report["status"] = "passed"
            print("[PASS] DisplayPort geometric-fabric live validation passed.")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _write_report(report)


if __name__ == "__main__":
    main()
