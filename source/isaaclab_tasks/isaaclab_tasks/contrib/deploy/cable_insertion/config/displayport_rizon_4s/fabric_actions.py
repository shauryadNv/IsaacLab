# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab action term for the DisplayPort geometric-fabric experiment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from isaaclab_physx.physics import PhysxCfg

import isaaclab.utils.math as math_utils
from isaaclab.actuators import IdealPDActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
from isaaclab.managers.action_manager import ActionTerm

from .fabric_kinematics import BatchedSerialChainKinematics
from .joint_fabric_controller import JointFabricController, JointFabricTaskKinematics, JointFabricTelemetry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .fabric_actions_cfg import DisplayportFabricActionCfg


_NUM_ARM_JOINTS = 7
_ACTION_DIM = 6


@dataclass(frozen=True)
class DisplayportFabricPolicyTelemetry:
    """Immutable health snapshot accumulated across one 30 Hz policy step."""

    faulted: torch.Tensor
    nonfinite: torch.Tensor
    infeasible: torch.Tensor
    cholesky_failed: torch.Tensor
    action_saturation_count: torch.Tensor
    acceleration_bound_active_count: torch.Tensor
    jerk_bound_active_count: torch.Tensor
    velocity_bound_active_count: torch.Tensor
    position_bound_active_count: torch.Tensor
    roundoff_guard_active_count: torch.Tensor
    tracker_effort_saturation_count: torch.Tensor
    tracker_position_limit_violation_count: torch.Tensor
    tracker_velocity_limit_violation_count: torch.Tensor
    tracking_position_error_norm: torch.Tensor
    tracking_velocity_error_norm: torch.Tensor
    maximum_acceleration: torch.Tensor
    maximum_jerk: torch.Tensor
    solve_residual_norm: torch.Tensor
    predicted_tracker_computed_effort: torch.Tensor
    predicted_tracker_applied_effort: torch.Tensor
    maximum_measured_joint_acceleration: torch.Tensor
    maximum_measured_joint_jerk: torch.Tensor
    maximum_actual_applied_effort: torch.Tensor
    maximum_actual_applied_effort_slew: torch.Tensor
    minimum_joint_margin: torch.Tensor
    metric_min_cholesky_diagonal: torch.Tensor


def _sample_constant_jerk_segment(
    position_0: torch.Tensor,
    velocity_0: torch.Tensor,
    acceleration_0: torch.Tensor,
    acceleration_1: torch.Tensor,
    duration: float,
    sample_time: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the exact polynomial joining two endpoint accelerations.

    Args:
        position_0: Segment-start joint position [rad].
        velocity_0: Segment-start joint velocity [rad/s].
        acceleration_0: Segment-start joint acceleration [rad/s^2].
        acceleration_1: Segment-end joint acceleration [rad/s^2].
        duration: Segment duration [s].
        sample_time: Time from the segment start [s].

    Returns:
        Position [rad], velocity [rad/s], and acceleration [rad/s^2].
    """
    if duration <= 0.0:
        raise ValueError("Constant-jerk segment duration must be positive.")
    if sample_time < 0.0 or sample_time > duration:
        raise ValueError("Constant-jerk sample time must lie within the segment.")
    jerk = (acceleration_1 - acceleration_0) / duration
    time_squared = sample_time * sample_time
    position = (
        position_0
        + sample_time * velocity_0
        + 0.5 * time_squared * acceleration_0
        + (sample_time * time_squared / 6.0) * jerk
    )
    velocity = velocity_0 + sample_time * acceleration_0 + 0.5 * time_squared * jerk
    acceleration = acceleration_0 + sample_time * jerk
    return position, velocity, acceleration


def _socket_transverse_error(tcp_offset_s: torch.Tensor, tcp_goal_offset_s: torch.Tensor) -> torch.Tensor:
    """Return socket-local y/z TCP error while leaving insertion x unconstrained."""
    return (tcp_offset_s - tcp_goal_offset_s)[:, 1:3]


def _rizon4s_nominal_kinematics(
    tcp_offset: tuple[float, float, float], *, device: str | torch.device
) -> BatchedSerialChainKinematics:
    """Construct the nominal Rizon 4s chain used by the shipping robot asset.

    The transforms follow the URDF convention ``T_parent_joint``. The final
    transform composes link7-to-flange (0.124 m, ``Rz(-pi)``) with the configured
    flange-to-TCP translation.
    """

    dtype = torch.float32
    device = torch.device(device)
    identity = torch.eye(4, dtype=dtype, device=device)
    rz_pi = torch.diag(torch.tensor((-1.0, -1.0, 1.0), dtype=dtype, device=device))
    ry_minus_half_pi = torch.tensor(
        ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        dtype=dtype,
        device=device,
    )

    translations = (
        (0.0, 0.0, 0.155),
        (0.0, 0.030, 0.210),
        (0.0, 0.035, 0.205),
        (-0.020, -0.030, 0.190),
        (-0.020, 0.025, 0.195),
        (0.0, 0.030, 0.190),
        (-0.015, 0.073, 0.110),
    )
    rotations = (rz_pi, identity[:3, :3], identity[:3, :3], rz_pi, rz_pi, identity[:3, :3], ry_minus_half_pi)
    parent_to_joint = identity.repeat(_NUM_ARM_JOINTS, 1, 1)
    for index, (translation, rotation) in enumerate(zip(translations, rotations, strict=True)):
        parent_to_joint[index, :3, :3] = rotation
        parent_to_joint[index, :3, 3] = torch.tensor(translation, dtype=dtype, device=device)

    child_to_joint = identity.repeat(_NUM_ARM_JOINTS, 1, 1)
    joint_axes = torch.tensor(
        (
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=dtype,
        device=device,
    )
    child_to_tcp = identity.clone()
    child_to_tcp[:3, :3] = rz_pi
    flange_offset = torch.tensor(tcp_offset, dtype=dtype, device=device)
    child_to_tcp[:3, 3] = torch.tensor((0.0, 0.0, 0.124), dtype=dtype, device=device) + rz_pi @ flange_offset
    return BatchedSerialChainKinematics(parent_to_joint, child_to_joint, joint_axes, child_to_tcp)


class DisplayportFabricAction(ActionTerm):
    """Map a 30 Hz base-frame wrench action through a 60 Hz joint fabric.

    The six normalized policy components represent a TCP wrench direction in the
    robot base frame, ordered as force xyz followed by torque xyz. The geometric
    fabric owns an independent artificial joint state. An explicit joint-space
    impedance tracker applies its targets at the 240 Hz PhysX rate. Its optional
    inverse-dynamics feed-forward contains ``M(q) qdd_f + g(q)``; Coriolis effort
    is intentionally omitted because this Isaac Lab PhysX data API does not
    expose it.
    """

    cfg: DisplayportFabricActionCfg
    _asset: Articulation

    def __init__(self, cfg: DisplayportFabricActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._validate_runtime_contract()

        joint_ids, joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(joint_ids) != _NUM_ARM_JOINTS or joint_names != cfg.joint_names:
            raise ValueError(
                "The fabric requires exactly the ordered arm joints "
                f"{cfg.joint_names}; resolved ids={joint_ids}, names={joint_names}."
            )
        if not self._asset.is_fixed_base:
            raise ValueError("The DisplayPort fabric experiment currently supports only a fixed-base robot.")
        self._joint_ids = list(joint_ids)
        self._joint_names = list(joint_names)
        self._dof_ids = [joint_id + self._asset.num_base_dofs for joint_id in self._joint_ids]

        body_ids, body_names = self._asset.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body named '{cfg.body_name}', resolved {body_names}.")
        self._body_idx = body_ids[0]
        self._jacobian_body_idx = self._body_idx - 1

        socket = env.scene[cfg.socket_asset_name]
        if not isinstance(socket, RigidObject):
            raise TypeError(f"Scene asset '{cfg.socket_asset_name}' must be a RigidObject.")
        self._socket: RigidObject = socket

        joint_limits = self._asset.data.joint_pos_limits.torch[:, self._joint_ids]
        self._controller = JointFabricController(
            cfg.fabric_cfg,
            num_envs=self.num_envs,
            device=self.device,
            joint_lower_limits=joint_limits[..., 0],
            joint_upper_limits=joint_limits[..., 1],
        )
        self._kinematics = _rizon4s_nominal_kinematics(cfg.tcp_offset, device=self.device)

        self._raw_actions = torch.zeros(self.num_envs, _ACTION_DIM, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._socket_offset = torch.tensor(cfg.socket_insertion_offset, device=self.device).repeat(self.num_envs, 1)
        self._socket_tcp_goal_offset = torch.tensor(cfg.socket_tcp_goal_offset, device=self.device).repeat(
            self.num_envs, 1
        )
        self._tcp_offset = torch.tensor(cfg.tcp_offset, device=self.device).repeat(self.num_envs, 1)
        self._identity_quaternion = torch.tensor((0.0, 0.0, 0.0, 1.0), device=self.device).repeat(self.num_envs, 1)
        self._fabric_interval = round(cfg.fabric_cfg.dt / env.physics_dt)
        self._physics_step_in_policy = 0

        self._segment_position_0 = torch.zeros(self.num_envs, _NUM_ARM_JOINTS, device=self.device)
        self._segment_velocity_0 = torch.zeros_like(self._segment_position_0)
        self._segment_acceleration_0 = torch.zeros_like(self._segment_position_0)
        self._segment_acceleration_1 = torch.zeros_like(self._segment_position_0)
        self._previous_measured_velocity = torch.zeros_like(self._segment_position_0)
        self._previous_measured_acceleration = torch.zeros_like(self._segment_position_0)
        self._previous_actual_applied_effort = torch.zeros_like(self._segment_position_0)
        self._tracker_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        (
            self._tracker_stiffness,
            self._tracker_damping,
            self._tracker_effort_limit,
        ) = self._resolve_tracker_parameters()
        self._policy_or = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in ("faulted", "nonfinite", "infeasible", "cholesky_failed")
        }
        self._policy_sum = {
            name: torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            for name in (
                "action_saturation_count",
                "acceleration_bound_active_count",
                "jerk_bound_active_count",
                "velocity_bound_active_count",
                "position_bound_active_count",
                "roundoff_guard_active_count",
                "tracker_effort_saturation_count",
                "tracker_position_limit_violation_count",
                "tracker_velocity_limit_violation_count",
            )
        }
        self._policy_max = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "tracking_position_error_norm",
                "tracking_velocity_error_norm",
                "maximum_acceleration",
                "maximum_jerk",
                "solve_residual_norm",
                "predicted_tracker_computed_effort",
                "predicted_tracker_applied_effort",
                "maximum_measured_joint_acceleration",
                "maximum_measured_joint_jerk",
                "maximum_actual_applied_effort",
                "maximum_actual_applied_effort_slew",
            )
        }
        self._policy_min = {
            name: torch.full((self.num_envs,), torch.inf, device=self.device)
            for name in ("minimum_joint_margin", "metric_min_cholesky_diagonal")
        }
        self._policy_telemetry = self._empty_policy_telemetry()

    @property
    def action_dim(self) -> int:
        """Dimension of the normalized base-frame TCP wrench action."""
        return _ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        """Unprocessed normalized policy actions."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Normalized policy actions after componentwise clipping."""
        return self._processed_actions

    @property
    def fabric_joint_position(self) -> torch.Tensor:
        """Artificial joint positions [rad]."""
        return self._controller.joint_position

    @property
    def fabric_joint_velocity(self) -> torch.Tensor:
        """Artificial joint velocities [rad/s]."""
        return self._controller.joint_velocity

    @property
    def fabric_joint_acceleration(self) -> torch.Tensor:
        """Artificial joint accelerations [rad/s^2]."""
        return self._controller.joint_acceleration

    @property
    def policy_telemetry(self) -> DisplayportFabricPolicyTelemetry:
        """Health telemetry finalized before policy-step terminations and resets."""
        return self._policy_telemetry

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """Describe the normalized base-frame wrench policy contract."""
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "TaskSpaceWrenchAction"
        self._IO_descriptor.extras.update(
            {
                "element_names": ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"],
                "units": "normalized",
                "reference_frame": "robot_base",
                "fabric_rate_hz": 1.0 / self.cfg.fabric_cfg.dt,
                "policy_rate_hz": 1.0 / self._env.step_dt,
                "wrench_scale": list(self.cfg.fabric_cfg.wrench_scale),
            }
        )
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        """Store one 30 Hz action without advancing the artificial state."""
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"Expected action shape {tuple(self._raw_actions.shape)}, received {tuple(actions.shape)}."
            )
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, min=-1.0, max=1.0)
        self._physics_step_in_policy = 0
        self._reset_policy_telemetry_accumulators()

    def apply_actions(self) -> None:
        """Advance at 60 Hz and apply explicit tracking commands at 240 Hz."""
        if self._physics_step_in_policy >= self.cfg.expected_policy_decimation:
            raise RuntimeError("Fabric action was applied more than eight times without a new policy action.")
        if self._physics_step_in_policy % self._fabric_interval == 0:
            self._advance_fabric()
        self._apply_tracker()
        if self._physics_step_in_policy == self.cfg.expected_policy_decimation - 1:
            self._finalize_policy_telemetry()
        self._physics_step_in_policy += 1

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Initialize selected artificial states from the post-event robot state."""
        indices = self._normalize_env_ids(env_ids)
        joint_position = self._asset.data.joint_pos.torch[indices][:, self._joint_ids]
        measured_joint_velocity = self._asset.data.joint_vel.torch[indices][:, self._joint_ids]
        self._assert_finite(joint_position, "Cannot reset fabric from non-finite joint position.")
        self._assert_finite(measured_joint_velocity, "Cannot reset fabric from non-finite joint velocity.")
        if self.cfg.reset_velocity_from_robot:
            joint_velocity = measured_joint_velocity
        else:
            joint_velocity = torch.zeros_like(joint_position)
        self._controller.reset(joint_position, joint_velocity, env_ids=indices)
        self._segment_position_0[indices] = joint_position
        self._segment_velocity_0[indices] = joint_velocity
        self._segment_acceleration_0[indices] = 0.0
        self._segment_acceleration_1[indices] = 0.0
        self._raw_actions[indices] = 0.0
        self._processed_actions[indices] = 0.0
        actual_applied_effort = self._asset.actuators.applied_effort.torch[indices][:, self._joint_ids]
        self._reset_tracker_histories(indices, measured_joint_velocity, actual_applied_effort)

    @torch.no_grad()
    def validate_dynamics(self) -> dict[str, float]:
        """Validate finite, symmetric, positive-definite arm dynamics tensors."""
        mass_matrix, gravity = self._read_dynamics()
        symmetry_error = torch.abs(mass_matrix - mass_matrix.transpose(-1, -2)).amax()
        minimum_eigenvalue = torch.linalg.eigvalsh(mass_matrix).amin()
        if not bool(torch.isfinite(mass_matrix).all() and torch.isfinite(gravity).all()):
            raise RuntimeError("PhysX returned non-finite mass-matrix or gravity-compensation data.")
        if float(symmetry_error) > 1.0e-4:
            raise RuntimeError(f"Arm mass matrix is not symmetric; maximum error is {float(symmetry_error):.3e}.")
        if float(minimum_eigenvalue) <= 0.0:
            raise RuntimeError(
                f"Arm mass matrix is not positive definite; minimum eigenvalue is {float(minimum_eigenvalue):.3e}."
            )
        return {
            "mass_matrix_symmetry_error": float(symmetry_error),
            "mass_matrix_minimum_eigenvalue": float(minimum_eigenvalue),
            "gravity_maximum_absolute_effort": float(torch.abs(gravity).amax()),
        }

    @torch.no_grad()
    def validate_kinematics_parity(self) -> dict[str, float]:
        """Compare nominal-chain TCP pose and Jacobian against current PhysX data.

        The method requires the physical robot to be tracking :math:`q_f` closely;
        call it immediately after reset or after a hold has settled.
        """
        measured_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        tracking_error = torch.abs(measured_position - self.fabric_joint_position).amax()
        if float(tracking_error) > self.cfg.parity_tracking_tolerance:
            raise RuntimeError(
                "Kinematics parity requires measured q and q_f to match; maximum error is "
                f"{float(tracking_error):.3e} rad."
            )

        nominal = self._kinematics(self.fabric_joint_position, self.fabric_joint_velocity)
        simulator_pose, simulator_jacobian = self._simulator_tcp_kinematics()
        position_error = torch.abs(nominal.tcp_pose[:, :3] - simulator_pose[:, :3]).amax()
        nominal_rotation = math_utils.matrix_from_quat(nominal.tcp_pose[:, 3:7])
        simulator_rotation = math_utils.matrix_from_quat(simulator_pose[:, 3:7])
        rotation_error = torch.abs(nominal_rotation - simulator_rotation).amax()
        jacobian_error = torch.abs(nominal.jacobian - simulator_jacobian).amax()

        errors = {
            "tracking_position_error": float(tracking_error),
            "tcp_position_error": float(position_error),
            "tcp_rotation_matrix_error": float(rotation_error),
            "tcp_jacobian_error": float(jacobian_error),
        }
        limits = {
            "tcp_position_error": self.cfg.parity_position_tolerance,
            "tcp_rotation_matrix_error": self.cfg.parity_rotation_tolerance,
            "tcp_jacobian_error": self.cfg.parity_jacobian_tolerance,
        }
        failures = [f"{name}={errors[name]:.3e}>{limit:.3e}" for name, limit in limits.items() if errors[name] > limit]
        if failures:
            raise RuntimeError("Nominal Rizon kinematics do not match PhysX: " + ", ".join(failures))
        return errors

    def _advance_fabric(self) -> None:
        self._assert_true(
            self._controller.initialized.all(),
            "Fabric state must be reset before the first simulation step.",
        )
        self._segment_position_0[:] = self.fabric_joint_position
        self._segment_velocity_0[:] = self.fabric_joint_velocity
        self._segment_acceleration_0[:] = self.fabric_joint_acceleration

        result = self._kinematics(self.fabric_joint_position, self.fabric_joint_velocity)
        socket_position_b, socket_quaternion_b = self._socket_insertion_pose_b()
        socket_rotation_b = math_utils.matrix_from_quat(socket_quaternion_b)
        base_to_socket_rotation = socket_rotation_b.transpose(-1, -2)

        position_error_b = result.tcp_pose[:, :3] - socket_position_b
        tcp_offset_s = torch.bmm(base_to_socket_rotation, position_error_b.unsqueeze(-1)).squeeze(-1)
        linear_jacobian_s = torch.bmm(base_to_socket_rotation, result.jacobian[:, :3])
        jacobian_dot_velocity_s = torch.bmm(
            base_to_socket_rotation,
            result.jacobian_dot_times_velocity[:, :3].unsqueeze(-1),
        ).squeeze(-1)
        task_kinematics = JointFabricTaskKinematics(
            transverse_error=_socket_transverse_error(tcp_offset_s, self._socket_tcp_goal_offset),
            transverse_jacobian=linear_jacobian_s[:, 1:3],
            transverse_jacobian_dot_velocity=jacobian_dot_velocity_s[:, 1:3],
            tcp_jacobian=result.jacobian,
        )
        self._controller.step(
            self._raw_actions,
            task_kinematics,
            measured_joint_position=self._asset.data.joint_pos.torch[:, self._joint_ids],
            measured_joint_velocity=self._asset.data.joint_vel.torch[:, self._joint_ids],
        )
        self._segment_acceleration_1[:] = self.fabric_joint_acceleration
        self._aggregate_controller_telemetry(self._controller.telemetry)

    def _reset_tracker_histories(
        self,
        indices: torch.Tensor,
        measured_velocity: torch.Tensor,
        actual_applied_effort: torch.Tensor,
    ) -> None:
        """Baseline finite-difference histories for selected environments."""
        self._assert_finite(measured_velocity, "Cannot reset tracker history from non-finite joint velocity.")
        self._assert_finite(actual_applied_effort, "Cannot reset tracker history from non-finite applied effort.")
        self._previous_measured_velocity[indices] = measured_velocity
        self._previous_measured_acceleration[indices] = 0.0
        self._previous_actual_applied_effort[indices] = actual_applied_effort
        self._tracker_history_valid[indices] = True

    def _update_tracker_health_telemetry(
        self, measured_position: torch.Tensor, measured_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate measured motion and the most recently applied actuator effort.

        The collection-level applied effort is the output processed for the
        preceding physics tick; it is intentionally distinct from the effort
        predicted below for the command being written on this tick.
        """
        actual_applied_effort = self._asset.actuators.applied_effort.torch[:, self._joint_ids]
        history_valid = self._tracker_history_valid.unsqueeze(-1)
        inverse_dt = 1.0 / self._env.physics_dt
        measured_acceleration = torch.where(
            history_valid,
            (measured_velocity - self._previous_measured_velocity) * inverse_dt,
            torch.zeros_like(measured_velocity),
        )
        measured_jerk = torch.where(
            history_valid,
            (measured_acceleration - self._previous_measured_acceleration) * inverse_dt,
            torch.zeros_like(measured_acceleration),
        )
        actual_effort_slew = torch.where(
            history_valid,
            (actual_applied_effort - self._previous_actual_applied_effort) * inverse_dt,
            torch.zeros_like(actual_applied_effort),
        )

        sample_finite = (
            torch.all(torch.isfinite(measured_position), dim=-1)
            & torch.all(torch.isfinite(measured_velocity), dim=-1)
            & torch.all(torch.isfinite(measured_acceleration), dim=-1)
            & torch.all(torch.isfinite(measured_jerk), dim=-1)
            & torch.all(torch.isfinite(actual_applied_effort), dim=-1)
            & torch.all(torch.isfinite(actual_effort_slew), dim=-1)
        )
        nonfinite = ~sample_finite
        self._controller.mark_faulted(nonfinite)
        self._policy_or["nonfinite"] |= nonfinite
        self._policy_or["faulted"] |= self._controller.faulted

        values = {
            "maximum_measured_joint_acceleration": measured_acceleration,
            "maximum_measured_joint_jerk": measured_jerk,
            "maximum_actual_applied_effort": actual_applied_effort,
            "maximum_actual_applied_effort_slew": actual_effort_slew,
        }
        for name, value in values.items():
            magnitude = torch.nan_to_num(torch.abs(value), nan=0.0, posinf=0.0, neginf=0.0).amax(dim=-1)
            torch.maximum(self._policy_max[name], magnitude, out=self._policy_max[name])

        finite_environment = sample_finite.unsqueeze(-1)
        self._previous_measured_velocity[:] = torch.where(
            finite_environment, measured_velocity, self._previous_measured_velocity
        )
        self._previous_measured_acceleration[:] = torch.where(
            finite_environment, measured_acceleration, self._previous_measured_acceleration
        )
        self._previous_actual_applied_effort[:] = torch.where(
            finite_environment, actual_applied_effort, self._previous_actual_applied_effort
        )
        self._tracker_history_valid |= sample_finite

        fallback_position = torch.maximum(
            torch.minimum(self._segment_position_0, self._controller.safe_joint_upper_limits),
            self._controller.safe_joint_lower_limits,
        )
        safe_position = torch.where(torch.isfinite(measured_position), measured_position, fallback_position)
        safe_velocity = torch.where(
            torch.isfinite(measured_velocity), measured_velocity, torch.zeros_like(measured_velocity)
        )
        return safe_position, safe_velocity

    def _guard_tracker_targets(
        self,
        position_target: torch.Tensor,
        velocity_target: torch.Tensor,
        acceleration_target: torch.Tensor,
        measured_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Clamp sampled targets and fault any material envelope violation.

        Out-of-envelope samples beyond the configured numerical tolerance are
        counted and faulted. Smaller roundoff is clamped to the envelope.
        Material or non-finite violations persistently fault the environment and replace
        the command with a measured-position, zero-velocity hold.
        """
        active = ~self._controller.faulted
        safe_lower = self._controller.safe_joint_lower_limits
        safe_upper = self._controller.safe_joint_upper_limits
        maximum_velocity = self._controller.maximum_joint_velocity

        position_finite = torch.isfinite(position_target)
        velocity_finite = torch.isfinite(velocity_target)
        acceleration_finite = torch.isfinite(acceleration_target)
        position_outside = active.unsqueeze(-1) & (
            ~position_finite | (position_target < safe_lower) | (position_target > safe_upper)
        )
        velocity_outside = active.unsqueeze(-1) & (
            ~velocity_finite | (velocity_target < -maximum_velocity) | (velocity_target > maximum_velocity)
        )
        clamped_position = torch.maximum(torch.minimum(position_target, safe_upper), safe_lower)
        clamped_velocity = torch.maximum(torch.minimum(velocity_target, maximum_velocity), -maximum_velocity)
        tolerance = self.cfg.fabric_cfg.bound_tolerance
        position_limit_violation = position_outside & (
            ~position_finite | (torch.abs(position_target - clamped_position) > tolerance)
        )
        velocity_limit_violation = velocity_outside & (
            ~velocity_finite | (torch.abs(velocity_target - clamped_velocity) > tolerance)
        )
        self._policy_sum["tracker_position_limit_violation_count"].add_(position_limit_violation.sum(dim=-1))
        self._policy_sum["tracker_velocity_limit_violation_count"].add_(velocity_limit_violation.sum(dim=-1))
        material_position_violation = torch.any(position_limit_violation, dim=-1)
        material_velocity_violation = torch.any(velocity_limit_violation, dim=-1)
        target_nonfinite = active & ~(
            torch.all(position_finite, dim=-1)
            & torch.all(velocity_finite, dim=-1)
            & torch.all(acceleration_finite, dim=-1)
        )
        tracker_fault = material_position_violation | material_velocity_violation | target_nonfinite
        self._controller.mark_faulted(tracker_fault)
        self._policy_or["nonfinite"] |= target_nonfinite
        self._policy_or["faulted"] |= self._controller.faulted

        faulted = self._controller.faulted.unsqueeze(-1)
        guarded_position = torch.where(faulted, measured_position, clamped_position)
        guarded_velocity = torch.where(faulted, torch.zeros_like(clamped_velocity), clamped_velocity)
        guarded_acceleration = torch.where(faulted, torch.zeros_like(acceleration_target), acceleration_target)
        return guarded_position, guarded_velocity, guarded_acceleration

    def _apply_tracker(self) -> None:
        segment_tick = self._physics_step_in_policy % self._fabric_interval
        sample_time = min((segment_tick + 1) * self._env.physics_dt, self.cfg.fabric_cfg.dt)
        position_target, velocity_target, acceleration_target = _sample_constant_jerk_segment(
            self._segment_position_0,
            self._segment_velocity_0,
            self._segment_acceleration_0,
            self._segment_acceleration_1,
            self.cfg.fabric_cfg.dt,
            sample_time,
        )

        measured_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        measured_velocity = self._asset.data.joint_vel.torch[:, self._joint_ids]
        measured_position, measured_velocity = self._update_tracker_health_telemetry(
            measured_position, measured_velocity
        )
        position_target, velocity_target, acceleration_target = self._guard_tracker_targets(
            position_target, velocity_target, acceleration_target, measured_position
        )

        mass_matrix, gravity = self._read_dynamics()
        feedforward = torch.zeros_like(position_target)
        if self.cfg.use_inertial_feedforward:
            feedforward += torch.bmm(mass_matrix, acceleration_target.unsqueeze(-1)).squeeze(-1)
        if self.cfg.use_gravity_compensation:
            feedforward += gravity
        self._assert_finite(feedforward, "Fabric tracker generated non-finite feed-forward efforts.")

        predicted_computed_effort = (
            self._tracker_stiffness * (position_target - measured_position)
            + self._tracker_damping * (velocity_target - measured_velocity)
            + feedforward
        )
        predicted_applied_effort = torch.clamp(
            predicted_computed_effort,
            min=-self._tracker_effort_limit,
            max=self._tracker_effort_limit,
        )
        self._policy_sum["tracker_effort_saturation_count"].add_(
            torch.sum(torch.abs(predicted_computed_effort - predicted_applied_effort) > 1.0e-5, dim=-1)
        )
        torch.maximum(
            self._policy_max["tracking_position_error_norm"],
            torch.linalg.vector_norm(position_target - measured_position, dim=-1),
            out=self._policy_max["tracking_position_error_norm"],
        )
        torch.maximum(
            self._policy_max["tracking_velocity_error_norm"],
            torch.linalg.vector_norm(velocity_target - measured_velocity, dim=-1),
            out=self._policy_max["tracking_velocity_error_norm"],
        )
        torch.maximum(
            self._policy_max["predicted_tracker_computed_effort"],
            torch.abs(predicted_computed_effort).amax(dim=-1),
            out=self._policy_max["predicted_tracker_computed_effort"],
        )
        torch.maximum(
            self._policy_max["predicted_tracker_applied_effort"],
            torch.abs(predicted_applied_effort).amax(dim=-1),
            out=self._policy_max["predicted_tracker_applied_effort"],
        )

        commands = self._asset.actuators.target_command
        commands.set_position_index(value=position_target, joint_ids=self._joint_ids)
        commands.set_velocity_index(value=velocity_target, joint_ids=self._joint_ids)
        commands.set_effort_index(value=feedforward, joint_ids=self._joint_ids)

    def _aggregate_controller_telemetry(self, telemetry: JointFabricTelemetry) -> None:
        for name in self._policy_or:
            self._policy_or[name] |= getattr(telemetry, name)
        for name in (
            "action_saturation_count",
            "acceleration_bound_active_count",
            "jerk_bound_active_count",
            "velocity_bound_active_count",
            "position_bound_active_count",
            "roundoff_guard_active_count",
        ):
            self._policy_sum[name].add_(getattr(telemetry, name))
        for name in ("maximum_acceleration", "maximum_jerk", "solve_residual_norm"):
            torch.maximum(self._policy_max[name], getattr(telemetry, name), out=self._policy_max[name])
        for name in self._policy_min:
            torch.minimum(self._policy_min[name], getattr(telemetry, name), out=self._policy_min[name])

    def _reset_policy_telemetry_accumulators(self) -> None:
        for value in self._policy_or.values():
            value.zero_()
        for value in self._policy_sum.values():
            value.zero_()
        for value in self._policy_max.values():
            value.zero_()
        for value in self._policy_min.values():
            value.fill_(torch.inf)

    def _finalize_policy_telemetry(self) -> None:
        telemetry = self._policy_telemetry
        telemetry.faulted.copy_(self._policy_or["faulted"])
        telemetry.nonfinite.copy_(self._policy_or["nonfinite"])
        telemetry.infeasible.copy_(self._policy_or["infeasible"])
        telemetry.cholesky_failed.copy_(self._policy_or["cholesky_failed"])
        telemetry.action_saturation_count.copy_(self._policy_sum["action_saturation_count"])
        telemetry.acceleration_bound_active_count.copy_(self._policy_sum["acceleration_bound_active_count"])
        telemetry.jerk_bound_active_count.copy_(self._policy_sum["jerk_bound_active_count"])
        telemetry.velocity_bound_active_count.copy_(self._policy_sum["velocity_bound_active_count"])
        telemetry.position_bound_active_count.copy_(self._policy_sum["position_bound_active_count"])
        telemetry.roundoff_guard_active_count.copy_(self._policy_sum["roundoff_guard_active_count"])
        telemetry.tracker_effort_saturation_count.copy_(self._policy_sum["tracker_effort_saturation_count"])
        telemetry.tracker_position_limit_violation_count.copy_(
            self._policy_sum["tracker_position_limit_violation_count"]
        )
        telemetry.tracker_velocity_limit_violation_count.copy_(
            self._policy_sum["tracker_velocity_limit_violation_count"]
        )
        telemetry.tracking_position_error_norm.copy_(self._policy_max["tracking_position_error_norm"])
        telemetry.tracking_velocity_error_norm.copy_(self._policy_max["tracking_velocity_error_norm"])
        telemetry.maximum_acceleration.copy_(self._policy_max["maximum_acceleration"])
        telemetry.maximum_jerk.copy_(self._policy_max["maximum_jerk"])
        telemetry.solve_residual_norm.copy_(self._policy_max["solve_residual_norm"])
        telemetry.predicted_tracker_computed_effort.copy_(self._policy_max["predicted_tracker_computed_effort"])
        telemetry.predicted_tracker_applied_effort.copy_(self._policy_max["predicted_tracker_applied_effort"])
        telemetry.maximum_measured_joint_acceleration.copy_(self._policy_max["maximum_measured_joint_acceleration"])
        telemetry.maximum_measured_joint_jerk.copy_(self._policy_max["maximum_measured_joint_jerk"])
        telemetry.maximum_actual_applied_effort.copy_(self._policy_max["maximum_actual_applied_effort"])
        telemetry.maximum_actual_applied_effort_slew.copy_(self._policy_max["maximum_actual_applied_effort_slew"])
        torch.nan_to_num(
            self._policy_min["minimum_joint_margin"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            out=telemetry.minimum_joint_margin,
        )
        torch.nan_to_num(
            self._policy_min["metric_min_cholesky_diagonal"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            out=telemetry.metric_min_cholesky_diagonal,
        )

    def _empty_policy_telemetry(self) -> DisplayportFabricPolicyTelemetry:
        scalar = torch.zeros(self.num_envs, device=self.device)
        integer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        boolean = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return DisplayportFabricPolicyTelemetry(
            faulted=boolean.clone(),
            nonfinite=boolean.clone(),
            infeasible=boolean.clone(),
            cholesky_failed=boolean.clone(),
            action_saturation_count=integer.clone(),
            acceleration_bound_active_count=integer.clone(),
            jerk_bound_active_count=integer.clone(),
            velocity_bound_active_count=integer.clone(),
            position_bound_active_count=integer.clone(),
            roundoff_guard_active_count=integer.clone(),
            tracker_effort_saturation_count=integer.clone(),
            tracker_position_limit_violation_count=integer.clone(),
            tracker_velocity_limit_violation_count=integer.clone(),
            tracking_position_error_norm=scalar.clone(),
            tracking_velocity_error_norm=scalar.clone(),
            maximum_acceleration=scalar.clone(),
            maximum_jerk=scalar.clone(),
            solve_residual_norm=scalar.clone(),
            predicted_tracker_computed_effort=scalar.clone(),
            predicted_tracker_applied_effort=scalar.clone(),
            maximum_measured_joint_acceleration=scalar.clone(),
            maximum_measured_joint_jerk=scalar.clone(),
            maximum_actual_applied_effort=scalar.clone(),
            maximum_actual_applied_effort_slew=scalar.clone(),
            minimum_joint_margin=scalar.clone(),
            metric_min_cholesky_diagonal=scalar.clone(),
        )

    def _resolve_tracker_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collect exact IdealPD gains and clipping limits in arm-joint order."""
        stiffness = torch.zeros(self.num_envs, _NUM_ARM_JOINTS, device=self.device)
        damping = torch.zeros_like(stiffness)
        effort_limit = torch.zeros_like(stiffness)
        covered: set[str] = set()
        arm_column = {name: index for index, name in enumerate(self._joint_names)}
        for group_name in self.cfg.tracker_actuator_names:
            actuator = self._asset.actuators[group_name]
            if not isinstance(actuator, IdealPDActuator):
                raise TypeError(f"Fabric tracker group '{group_name}' must use IdealPDActuator.")
            for group_column, joint_name in enumerate(actuator.joint_names):
                if joint_name not in arm_column:
                    continue
                if joint_name in covered:
                    raise ValueError(f"Fabric tracker joint '{joint_name}' is covered more than once.")
                target_column = arm_column[joint_name]
                stiffness[:, target_column] = actuator.stiffness[:, group_column]
                damping[:, target_column] = actuator.damping[:, group_column]
                effort_limit[:, target_column] = actuator.actuator_effort_limit[:, group_column]
                covered.add(joint_name)
        missing = set(self._joint_names) - covered
        if missing:
            raise ValueError(f"Fabric tracker actuator groups do not cover arm joints: {sorted(missing)}.")
        return stiffness, damping, effort_limit

    def _read_dynamics(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            full_mass_matrix = self._asset.data.mass_matrix.torch
            full_gravity = self._asset.data.gravity_compensation_forces.torch
        except (AttributeError, NotImplementedError) as error:
            raise RuntimeError("The fabric tracker requires PhysX mass-matrix and gravity data.") from error
        mass_matrix = full_mass_matrix[:, self._dof_ids, :][:, :, self._dof_ids]
        gravity = full_gravity[:, self._dof_ids]
        self._assert_finite(mass_matrix, "PhysX returned a non-finite arm mass matrix.")
        self._assert_finite(gravity, "PhysX returned non-finite gravity-compensation efforts.")
        return mass_matrix, gravity

    def _socket_insertion_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        socket_position_w = self._socket.data.root_pos_w.torch
        socket_quaternion_w = self._socket.data.root_quat_w.torch
        insertion_position_w = socket_position_w + math_utils.quat_apply(socket_quaternion_w, self._socket_offset)
        return math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w.torch,
            self._asset.data.root_quat_w.torch,
            insertion_position_w,
            socket_quaternion_w,
        )

    def _simulator_tcp_kinematics(self) -> tuple[torch.Tensor, torch.Tensor]:
        flange_position_b, flange_quaternion_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w.torch,
            self._asset.data.root_quat_w.torch,
            self._asset.data.body_pos_w.torch[:, self._body_idx],
            self._asset.data.body_quat_w.torch[:, self._body_idx],
        )
        tcp_position_b, tcp_quaternion_b = math_utils.combine_frame_transforms(
            flange_position_b,
            flange_quaternion_b,
            self._tcp_offset,
            self._identity_quaternion,
        )

        jacobian_w = self._asset.data.body_link_jacobian_w.torch[:, self._jacobian_body_idx, :, self._dof_ids].clone()
        world_to_base = math_utils.matrix_from_quat(math_utils.quat_inv(self._asset.data.root_quat_w.torch))
        jacobian_b = torch.empty_like(jacobian_w)
        jacobian_b[:, :3] = torch.bmm(world_to_base, jacobian_w[:, :3])
        jacobian_b[:, 3:] = torch.bmm(world_to_base, jacobian_w[:, 3:])
        offset_b = math_utils.quat_apply(flange_quaternion_b, self._tcp_offset)
        jacobian_b[:, :3] -= torch.bmm(math_utils.skew_symmetric_matrix(offset_b), jacobian_b[:, 3:])
        return torch.cat((tcp_position_b, tcp_quaternion_b), dim=-1), jacobian_b

    def _validate_runtime_contract(self) -> None:
        if not isinstance(self._env.cfg.sim.physics, PhysxCfg):
            raise ValueError("The DisplayPort fabric action is currently validated only with PhysX.")
        if self._env._physics_handles_decimation:
            raise ValueError("The fabric action requires per-physics-step apply_actions() calls.")
        if not math.isclose(self._env.physics_dt, self.cfg.expected_physics_dt, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"Expected physics dt {self.cfg.expected_physics_dt}, received {self._env.physics_dt}.")
        if self._env.cfg.decimation != self.cfg.expected_policy_decimation:
            raise ValueError(
                f"Expected policy decimation {self.cfg.expected_policy_decimation}, "
                f"received {self._env.cfg.decimation}."
            )
        ratio = self.cfg.fabric_cfg.dt / self._env.physics_dt
        interval = round(ratio)
        if interval <= 0 or not math.isclose(ratio, interval, rel_tol=0.0, abs_tol=1.0e-10):
            raise ValueError("Fabric dt must be an integer multiple of the physics dt.")
        if self._env.cfg.decimation % interval != 0 or self._env.cfg.decimation // interval != 2:
            raise ValueError("The policy step must contain exactly two evenly spaced fabric updates.")
        if not self.cfg.use_gravity_compensation:
            raise ValueError("Physical gravity is enabled for this experiment; gravity compensation must be enabled.")

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)[env_ids]
        indices = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if indices.numel() and bool(torch.any((indices < 0) | (indices >= self.num_envs))):
            raise IndexError("env_ids contains an environment outside the action batch.")
        return indices

    @staticmethod
    def _assert_finite(value: torch.Tensor, message: str) -> None:
        DisplayportFabricAction._assert_true(torch.all(torch.isfinite(value)), message)

    @staticmethod
    def _assert_true(condition: torch.Tensor, message: str) -> None:
        if hasattr(torch, "_assert_async"):
            torch._assert_async(condition, message)
        elif not bool(condition):
            raise RuntimeError(message)
