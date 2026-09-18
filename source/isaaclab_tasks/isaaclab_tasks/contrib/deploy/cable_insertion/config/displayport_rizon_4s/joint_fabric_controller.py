# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clean-room joint-state geometric fabric dynamics for DisplayPort experiments.

This module contains only batched controller mathematics and artificial state. It
does not depend on a simulator, a robot model library, or the separately licensed
NVLabs FABRICS implementation. Callers provide task errors and kinematic terms
evaluated at the artificial configuration. This experiment integrates only with
PhysX in simulation; a hardware kinematics/dynamics adapter and safety validation
are explicitly out of scope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as functional

from isaaclab.utils.configclass import configclass

_NUM_JOINTS = 7
_WRENCH_DIM = 6
_TRANSVERSE_DIM = 2


@configclass
class JointFabricControllerCfg:
    """Configuration for the seven-DoF artificial joint-state fabric.

    The defaults are deliberately conservative smoke-test values. They are not
    claimed to be tuned for the Flexiv Rizon 4s or for hardware deployment.
    """

    dt: float = 1.0 / 60.0
    """Artificial-state integration timestep [s]."""

    kinetic_energy_weights: tuple[float, ...] = (1.0,) * _NUM_JOINTS
    """Diagonal weights in the artificial kinetic energy [unitless]."""

    root_metric: tuple[float, ...] = (0.05,) * _NUM_JOINTS
    """Strictly positive root metric diagonal [artificial inertia]."""

    transverse_metric: tuple[float, float] = (1.0, 1.0)
    """Socket-transverse task metric diagonal [artificial mass]."""

    transverse_funnel_gain: float = 0.5
    """Gain of the homogeneous-degree-two transverse funnel [unitless]."""

    transverse_funnel_length_scale: float = 0.010
    """Distance scale of the transverse funnel [m]."""

    transverse_damping: tuple[float, float] = (2.0, 2.0)
    """Socket-transverse damping diagonal [artificial force*s/m]."""

    posture_stiffness: tuple[float, ...] = (0.02,) * _NUM_JOINTS
    """Weak potential stiffness toward the reset posture [artificial torque/rad]."""

    root_damping: tuple[float, ...] = (0.05,) * _NUM_JOINTS
    """Joint-space damping diagonal [artificial torque*s/rad]."""

    geometric_damping: float = 2.5
    """Geometry-preserving damping coefficient [1/s]."""

    wrench_scale: tuple[float, ...] = (0.5, 0.5, 0.5, 0.05, 0.05, 0.05)
    """Scale applied to clamped policy actions [N, N, N, N*m, N*m, N*m]."""

    joint_limit_barrier_metric: float = 0.10
    """Maximum scale of each smooth joint-limit barrier metric [artificial inertia]."""

    joint_limit_barrier_stiffness: float = 1.0
    """Smooth joint-limit potential stiffness [artificial torque/rad]."""

    joint_limit_barrier_damping: float = 0.5
    """Damping applied in proportion to the barrier metric [1/s]."""

    joint_limit_barrier_fraction: float = 0.15
    """Fraction of each joint range at which the smooth barrier begins to activate."""

    joint_limit_softness_fraction: float = 0.02
    """Fraction of each joint range used as the softplus transition width."""

    joint_limit_safety_fraction: float = 0.05
    """Fraction of each joint range reserved as a hard artificial-state margin."""

    max_joint_velocity: tuple[float, ...] = (
        0.4188,
        0.4188,
        0.4886,
        0.4886,
        0.9774,
        0.9774,
        0.9774,
    )
    """Artificial joint velocity limits [rad/s]."""

    max_joint_acceleration: tuple[float, ...] = (1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0)
    """Artificial joint acceleration limits [rad/s^2]."""

    max_joint_jerk: tuple[float, ...] = (10.0, 10.0, 10.0, 15.0, 15.0, 20.0, 20.0)
    """Artificial joint jerk limits [rad/s^3]."""

    solve_regularization: float = 1.0e-6
    """Kinetic-metric regularization added to the acceleration solve [artificial inertia]."""

    radius_epsilon: float = 1.0e-5
    """Regularization used in the transverse radial direction [m]."""

    energy_epsilon: float = 1.0e-8
    """Squared-speed threshold below which energization is disabled [rad^2/s^2]."""

    bound_tolerance: float = 1.0e-6
    """Tolerance used when reporting active safety bounds [rad/s^2]."""


@dataclass(frozen=True)
class JointFabricTaskKinematics:
    """Kinematic quantities evaluated at the current artificial joint state.

    Attributes:
        transverse_error: TCP position error in the socket transverse plane [m],
            shape ``(num_envs, 2)``. Zero is the socket axis.
        transverse_jacobian: Jacobian mapping artificial joint velocity to
            transverse error velocity [m/rad], shape ``(num_envs, 2, 7)``.
        transverse_jacobian_dot_velocity: Product ``J_dot * q_f_dot`` [m/s^2],
            shape ``(num_envs, 2)``.
        tcp_jacobian: Base-aligned TCP geometric Jacobian, ordered as linear then
            angular velocity, shape ``(num_envs, 6, 7)``.
    """

    transverse_error: torch.Tensor
    transverse_jacobian: torch.Tensor
    transverse_jacobian_dot_velocity: torch.Tensor
    tcp_jacobian: torch.Tensor


@dataclass(frozen=True)
class JointFabricTelemetry:
    """Per-environment diagnostics from the most recent fabric step."""

    metric_min_cholesky_diagonal: torch.Tensor
    cholesky_failed: torch.Tensor
    energy_residual: torch.Tensor
    action_saturation_count: torch.Tensor
    acceleration_bound_active_count: torch.Tensor
    jerk_bound_active_count: torch.Tensor
    velocity_bound_active_count: torch.Tensor
    position_bound_active_count: torch.Tensor
    roundoff_guard_active_count: torch.Tensor
    infeasible: torch.Tensor
    nonfinite: torch.Tensor
    faulted: torch.Tensor
    minimum_joint_margin: torch.Tensor
    tracking_valid: torch.Tensor
    tracking_position_error_norm: torch.Tensor
    tracking_velocity_error_norm: torch.Tensor
    nominal_acceleration: torch.Tensor
    commanded_acceleration: torch.Tensor
    maximum_acceleration: torch.Tensor
    maximum_jerk: torch.Tensor
    solve_residual_norm: torch.Tensor
    barrier_max_weight: torch.Tensor
    wrench_norm: torch.Tensor
    generalized_force_norm: torch.Tensor
    funnel_acceleration_norm: torch.Tensor
    acceleration_bound_activation_total: torch.Tensor
    jerk_bound_activation_total: torch.Tensor
    velocity_bound_activation_total: torch.Tensor
    position_bound_activation_total: torch.Tensor
    infeasible_total: torch.Tensor


class JointFabricController:
    r"""Integrate a seven-DoF artificial joint state through a geometric fabric.

    The nominal geometry is a homogeneous-degree-two funnel in the socket's two
    transverse directions. Given transverse error :math:`e`, error velocity
    :math:`\dot e = J_e \dot q_f`, and length scale :math:`\ell`, it uses

    .. math::

        h_e = -\frac{\kappa}{\ell}\|\dot e\|^2
              \tanh\left(\frac{\|e\|}{\ell}\right)
              \frac{e}{\sqrt{e^T e + \epsilon^2}}.

    The task geometry is pulled back into joint space in natural form. The
    resulting joint geometry is energized with the configured quadratic kinetic
    energy, then combined with a weak reset-posture potential, smooth joint-limit
    barriers, damping, and a bounded policy wrench pulled back through the TCP
    Jacobian. A componentwise feasible endpoint-acceleration box enforces
    acceleration, constant-jerk, velocity, and one-step position bounds. The
    artificial state advances over each fabric period :math:`T` with

    .. math::

        v_1 &= v_0 + \frac{T}{2}(a_0 + a_1), \\
        q_1 &= q_0 + T v_0 + T^2\left(\frac{a_0}{3} + \frac{a_1}{6}\right),

    which is the exact integral of the constant-jerk segment joining
    :math:`a_0` and :math:`a_1`. In addition to bounding the endpoint, the
    projection requires every Bernstein control point of the quadratic
    velocity and cubic position curves to lie inside the configured velocity
    and safe-position limits. The convex-hull property therefore bounds the
    complete continuous segment, not only its 60 Hz endpoints.

    This class owns only artificial state. Robot tracking torques or position
    targets are the responsibility of the caller.
    """

    def __init__(
        self,
        cfg: JointFabricControllerCfg,
        num_envs: int,
        device: str | torch.device,
        joint_lower_limits: torch.Tensor | Sequence[float],
        joint_upper_limits: torch.Tensor | Sequence[float],
    ):
        """Initialize the batched fabric.

        Args:
            cfg: Fabric dynamics and safety configuration.
            num_envs: Number of parallel artificial states.
            device: Torch device on which state is stored.
            joint_lower_limits: Physical joint lower limits [rad], shape ``(7,)``
                or ``(num_envs, 7)``.
            joint_upper_limits: Physical joint upper limits [rad], shape ``(7,)``
                or ``(num_envs, 7)``.
        """
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, received {num_envs}.")

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.empty((), device=device).device
        self._validate_cfg()

        dtype = torch.float32
        self._energy_weights = self._row_tensor(cfg.kinetic_energy_weights, dtype)
        self._root_metric = self._row_tensor(cfg.root_metric, dtype)
        self._transverse_metric = self._row_tensor(cfg.transverse_metric, dtype)
        self._transverse_damping = self._row_tensor(cfg.transverse_damping, dtype)
        self._posture_stiffness = self._row_tensor(cfg.posture_stiffness, dtype)
        self._root_damping = self._row_tensor(cfg.root_damping, dtype)
        self._wrench_scale = self._row_tensor(cfg.wrench_scale, dtype)
        self._max_velocity = self._row_tensor(cfg.max_joint_velocity, dtype)
        self._max_acceleration = self._row_tensor(cfg.max_joint_acceleration, dtype)
        self._max_jerk = self._row_tensor(cfg.max_joint_jerk, dtype)

        self._joint_lower = self._limit_tensor(joint_lower_limits, "joint_lower_limits", dtype)
        self._joint_upper = self._limit_tensor(joint_upper_limits, "joint_upper_limits", dtype)
        if torch.any(self._joint_upper <= self._joint_lower).item():
            raise ValueError("Every joint upper limit must be greater than its lower limit.")

        joint_range = self._joint_upper - self._joint_lower
        safety_margin = cfg.joint_limit_safety_fraction * joint_range
        self._safe_lower = self._joint_lower + safety_margin
        self._safe_upper = self._joint_upper - safety_margin
        self._barrier_distance = cfg.joint_limit_barrier_fraction * joint_range
        self._barrier_softness = cfg.joint_limit_softness_fraction * joint_range

        self._joint_position = torch.zeros(num_envs, _NUM_JOINTS, dtype=dtype, device=self.device)
        self._joint_velocity = torch.zeros_like(self._joint_position)
        self._joint_acceleration = torch.zeros_like(self._joint_position)
        self._reset_posture = torch.zeros_like(self._joint_position)
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._faulted = torch.zeros_like(self._initialized)

        counter_shape = (num_envs,)
        self._acceleration_bound_total = torch.zeros(counter_shape, dtype=torch.long, device=self.device)
        self._jerk_bound_total = torch.zeros_like(self._acceleration_bound_total)
        self._velocity_bound_total = torch.zeros_like(self._acceleration_bound_total)
        self._position_bound_total = torch.zeros_like(self._acceleration_bound_total)
        self._infeasible_total = torch.zeros_like(self._acceleration_bound_total)

        self._telemetry = self._empty_telemetry()

    @property
    def joint_position(self) -> torch.Tensor:
        """Artificial joint positions [rad], shape ``(num_envs, 7)``."""
        return self._joint_position

    @property
    def joint_velocity(self) -> torch.Tensor:
        """Artificial joint velocities [rad/s], shape ``(num_envs, 7)``."""
        return self._joint_velocity

    @property
    def joint_acceleration(self) -> torch.Tensor:
        """Artificial joint accelerations [rad/s^2], shape ``(num_envs, 7)``."""
        return self._joint_acceleration

    @property
    def reset_posture(self) -> torch.Tensor:
        """Per-environment weak-potential target [rad], shape ``(num_envs, 7)``."""
        return self._reset_posture

    @property
    def initialized(self) -> torch.Tensor:
        """Flags indicating which artificial states have been reset."""
        return self._initialized

    @property
    def faulted(self) -> torch.Tensor:
        """Persistent fault flags; reset clears the selected flags."""
        return self._faulted

    @property
    def safe_joint_lower_limits(self) -> torch.Tensor:
        """Artificial-state lower limits including the safety margin [rad]."""
        return self._safe_lower

    @property
    def safe_joint_upper_limits(self) -> torch.Tensor:
        """Artificial-state upper limits including the safety margin [rad]."""
        return self._safe_upper

    @property
    def maximum_joint_velocity(self) -> torch.Tensor:
        """Artificial-state absolute joint-velocity limits [rad/s]."""
        return self._max_velocity

    @property
    def telemetry(self) -> JointFabricTelemetry:
        """Diagnostics from the most recent call to :meth:`step`."""
        return self._telemetry

    @torch.no_grad()
    def mark_faulted(self, fault_mask: torch.Tensor) -> None:
        """Persistently fault selected environments until their next reset.

        Args:
            fault_mask: Boolean environment mask, shape ``(num_envs,)``.
        """
        if not isinstance(fault_mask, torch.Tensor):
            raise TypeError("fault_mask must be a torch.Tensor.")
        if fault_mask.shape != (self.num_envs,):
            raise ValueError(f"Expected fault_mask shape {(self.num_envs,)}, received {tuple(fault_mask.shape)}.")
        if fault_mask.device != self.device:
            raise ValueError(f"Expected fault_mask on {self.device}, received {fault_mask.device}.")
        if fault_mask.dtype != torch.bool:
            raise TypeError("fault_mask must have Boolean dtype.")
        self._faulted |= fault_mask
        self._telemetry.faulted.copy_(self._faulted)

    @torch.no_grad()
    def reset(
        self,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        posture_target: torch.Tensor | None = None,
    ) -> None:
        """Reset selected artificial states and clear their fault/counter state.

        Args:
            joint_position: Initial artificial joint positions [rad]. The first
                dimension may be either ``num_envs`` or the selected environment
                count.
            joint_velocity: Initial artificial joint velocities [rad/s]. Zero is
                used when omitted.
            env_ids: Environment indices to reset. All environments are selected
                when omitted.
            posture_target: Optional weak-potential target [rad]. The reset joint
                position is used when omitted.
        """
        indices = self._resolve_env_ids(env_ids)
        selected_position = self._select_reset_value(joint_position, indices, "joint_position")
        if joint_velocity is None:
            selected_velocity = torch.zeros_like(selected_position)
        else:
            selected_velocity = self._select_reset_value(joint_velocity, indices, "joint_velocity")
        if posture_target is None:
            selected_posture = selected_position
        else:
            selected_posture = self._select_reset_value(posture_target, indices, "posture_target")

        if not torch.all(torch.isfinite(selected_position)).item():
            raise ValueError("Reset joint positions must be finite.")
        if not torch.all(torch.isfinite(selected_velocity)).item():
            raise ValueError("Reset joint velocities must be finite.")
        if not torch.all(torch.isfinite(selected_posture)).item():
            raise ValueError("Posture targets must be finite.")

        safe_lower = self._safe_lower[indices]
        safe_upper = self._safe_upper[indices]
        if torch.any((selected_position < safe_lower) | (selected_position > safe_upper)).item():
            raise ValueError("Reset joint positions must lie inside the configured artificial-state safety limits.")
        if torch.any((selected_posture < safe_lower) | (selected_posture > safe_upper)).item():
            raise ValueError("Posture targets must lie inside the configured artificial-state safety limits.")
        if torch.any(torch.abs(selected_velocity) > self._max_velocity).item():
            raise ValueError("Reset joint velocities exceed max_joint_velocity.")

        self._joint_position[indices] = selected_position
        self._joint_velocity[indices] = selected_velocity
        self._joint_acceleration[indices] = 0.0
        self._reset_posture[indices] = selected_posture
        self._initialized[indices] = True
        self._faulted[indices] = False
        self._acceleration_bound_total[indices] = 0
        self._jerk_bound_total[indices] = 0
        self._velocity_bound_total[indices] = 0
        self._position_bound_total[indices] = 0
        self._infeasible_total[indices] = 0
        for telemetry_field in fields(self._telemetry):
            getattr(self._telemetry, telemetry_field.name)[indices] = 0

    def compute_transverse_natural_form(
        self,
        transverse_error: torch.Tensor,
        transverse_jacobian: torch.Tensor,
        transverse_jacobian_dot_velocity: torch.Tensor,
        joint_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the pulled-back socket-transverse HD2 geometry.

        Args:
            transverse_error: Socket-transverse error [m], shape ``(N, 2)``.
            transverse_jacobian: Error Jacobian [m/rad], shape ``(N, 2, 7)``.
            transverse_jacobian_dot_velocity: ``J_dot * q_dot`` [m/s^2], shape ``(N, 2)``.
            joint_velocity: Artificial joint velocity [rad/s], shape ``(N, 7)``.

        Returns:
            Tuple containing the pulled-back metric, pulled-back natural force,
            and transverse funnel acceleration. Their shapes are ``(N, 7, 7)``,
            ``(N, 7)``, and ``(N, 2)`` respectively.
        """
        batch_size = transverse_error.shape[0]
        self._validate_tensor(transverse_error, (batch_size, _TRANSVERSE_DIM), "transverse_error")
        self._validate_tensor(
            transverse_jacobian,
            (batch_size, _TRANSVERSE_DIM, _NUM_JOINTS),
            "transverse_jacobian",
        )
        self._validate_tensor(
            transverse_jacobian_dot_velocity,
            (batch_size, _TRANSVERSE_DIM),
            "transverse_jacobian_dot_velocity",
        )
        self._validate_tensor(joint_velocity, (batch_size, _NUM_JOINTS), "joint_velocity")

        task_velocity = torch.bmm(transverse_jacobian, joint_velocity.unsqueeze(-1)).squeeze(-1)
        speed_squared = torch.sum(task_velocity.square(), dim=-1, keepdim=True)
        radius = torch.sqrt(torch.sum(transverse_error.square(), dim=-1, keepdim=True) + self.cfg.radius_epsilon**2)
        direction = transverse_error / radius
        funnel_acceleration = (
            -self.cfg.transverse_funnel_gain
            / self.cfg.transverse_funnel_length_scale
            * speed_squared
            * torch.tanh(radius / self.cfg.transverse_funnel_length_scale)
            * direction
        )

        weighted_jacobian = self._transverse_metric.unsqueeze(-1) * transverse_jacobian
        metric = torch.bmm(transverse_jacobian.transpose(1, 2), weighted_jacobian)
        task_natural_force = self._transverse_metric * (transverse_jacobian_dot_velocity - funnel_acceleration)
        natural_force = torch.bmm(transverse_jacobian.transpose(1, 2), task_natural_force.unsqueeze(-1)).squeeze(-1)
        return metric, natural_force, funnel_acceleration

    def compute_bounded_wrench(
        self, action: torch.Tensor, tcp_jacobian: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clamp and scale a policy action, then pull the TCP wrench into joint space.

        Args:
            action: Normalized policy action, shape ``(N, 6)``.
            tcp_jacobian: Base-aligned TCP Jacobian, shape ``(N, 6, 7)``.

        Returns:
            Tuple containing the bounded TCP wrench [N, N*m] and generalized
            artificial force, with shapes ``(N, 6)`` and ``(N, 7)``.
        """
        batch_size = action.shape[0]
        self._validate_tensor(action, (batch_size, _WRENCH_DIM), "action")
        self._validate_tensor(tcp_jacobian, (batch_size, _WRENCH_DIM, _NUM_JOINTS), "tcp_jacobian")
        wrench = torch.clamp(action, min=-1.0, max=1.0) * self._wrench_scale
        generalized_force = torch.bmm(tcp_jacobian.transpose(1, 2), wrench.unsqueeze(-1)).squeeze(-1)
        return wrench, generalized_force

    def compute_joint_limit_terms(
        self, joint_position: torch.Tensor, joint_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute smooth joint-limit metric, potential gradient, and damping.

        Args:
            joint_position: Artificial joint positions [rad], shape ``(N, 7)``.
            joint_velocity: Artificial joint velocities [rad/s], shape ``(N, 7)``.

        Returns:
            Tuple of non-negative metric diagonal, potential gradient, and
            damping force, each shaped ``(N, 7)``.
        """
        batch_size = joint_position.shape[0]
        self._validate_tensor(joint_position, (batch_size, _NUM_JOINTS), "joint_position")
        self._validate_tensor(joint_velocity, (batch_size, _NUM_JOINTS), "joint_velocity")
        if batch_size != self.num_envs:
            raise ValueError(
                "compute_joint_limit_terms requires the controller batch because joint limits may be per environment."
            )

        lower_distance = joint_position - self._joint_lower
        upper_distance = self._joint_upper - joint_position
        lower_argument = (self._barrier_distance - lower_distance) / self._barrier_softness
        upper_argument = (self._barrier_distance - upper_distance) / self._barrier_softness
        lower_gate = torch.sigmoid(lower_argument)
        upper_gate = torch.sigmoid(upper_argument)
        lower_penetration = functional.softplus(lower_argument) * self._barrier_softness
        upper_penetration = functional.softplus(upper_argument) * self._barrier_softness

        metric_diagonal = self.cfg.joint_limit_barrier_metric * (lower_gate.square() + upper_gate.square())
        potential_gradient = self.cfg.joint_limit_barrier_stiffness * (
            -lower_penetration * lower_gate + upper_penetration * upper_gate
        )
        damping_force = self.cfg.joint_limit_barrier_damping * metric_diagonal * joint_velocity
        return metric_diagonal, potential_gradient, damping_force

    @torch.no_grad()
    def step(
        self,
        action: torch.Tensor,
        kinematics: JointFabricTaskKinematics,
        measured_joint_position: torch.Tensor | None = None,
        measured_joint_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance the artificial state by one fabric timestep.

        Args:
            action: Normalized base-frame TCP wrench action, shape ``(num_envs, 6)``.
            kinematics: Task errors and Jacobians evaluated at :attr:`joint_position`.
            measured_joint_position: Optional physical joint positions [rad] used
                only for tracking telemetry, shape ``(num_envs, 7)``.
            measured_joint_velocity: Optional physical joint velocities [rad/s]
                used only for tracking telemetry, shape ``(num_envs, 7)``.

        Returns:
            Updated artificial joint positions [rad], shape ``(num_envs, 7)``.

        Note:
            An infeasible constraint intersection or numerical failure sets a
            persistent fault and holds the affected artificial state. The caller
            should stop or hold the physical tracker until that environment is
            explicitly reset.
        """
        self._validate_step_inputs(action, kinematics, measured_joint_position, measured_joint_velocity)

        q = self._joint_position
        v = self._joint_velocity
        # Preserve the prior command for jerk limiting and telemetry. The
        # controller state is updated in place later in this method.
        previous_acceleration = self._joint_acceleration.clone()

        task_metric, task_force, funnel_acceleration = self.compute_transverse_natural_form(
            kinematics.transverse_error,
            kinematics.transverse_jacobian,
            kinematics.transverse_jacobian_dot_velocity,
            v,
        )
        barrier_metric, barrier_gradient, barrier_damping = self.compute_joint_limit_terms(q, v)

        metric_diagonal = self._root_metric + barrier_metric
        metric = task_metric + torch.diag_embed(metric_diagonal)
        identity = torch.eye(_NUM_JOINTS, dtype=metric.dtype, device=self.device).expand_as(metric)
        geometry_finite = torch.all(torch.isfinite(metric), dim=(-2, -1))
        finite_geometry_metric = torch.where(geometry_finite[:, None, None], metric, identity)
        geometry_cholesky, geometry_info = torch.linalg.cholesky_ex(finite_geometry_metric, check_errors=False)
        geometry_failed = (geometry_info != 0) | ~geometry_finite
        geometry_cholesky = torch.where(geometry_failed[:, None, None], identity, geometry_cholesky)
        metric_min_cholesky_diagonal = torch.diagonal(geometry_cholesky, dim1=-2, dim2=-1).amin(dim=-1)
        metric_min_cholesky_diagonal = torch.where(
            geometry_failed,
            torch.zeros_like(metric_min_cholesky_diagonal),
            metric_min_cholesky_diagonal,
        )
        raw_geometry = -self._solve_cholesky(geometry_cholesky, task_force)

        weighted_velocity = self._energy_weights * v
        energy = torch.sum(v * weighted_velocity, dim=-1, keepdim=True)
        energy_numerator = torch.sum(weighted_velocity * raw_geometry, dim=-1, keepdim=True)
        energization = torch.where(
            energy > self.cfg.energy_epsilon,
            -energy_numerator / torch.clamp(energy, min=self.cfg.energy_epsilon),
            torch.zeros_like(energy),
        )
        energized_geometry = raw_geometry + energization * v
        energy_residual = torch.sum(weighted_velocity * energized_geometry, dim=-1)

        wrench, generalized_force = self.compute_bounded_wrench(action, kinematics.tcp_jacobian)
        posture_gradient = self._posture_stiffness * (q - self._reset_posture)
        task_velocity = torch.bmm(kinematics.transverse_jacobian, v.unsqueeze(-1)).squeeze(-1)
        task_damping = torch.bmm(
            kinematics.transverse_jacobian.transpose(1, 2),
            (self._transverse_damping * task_velocity).unsqueeze(-1),
        ).squeeze(-1)
        metric_velocity = torch.bmm(metric, v.unsqueeze(-1)).squeeze(-1)
        metric_geometry = torch.bmm(metric, energized_geometry.unsqueeze(-1)).squeeze(-1)
        right_hand_side = (
            metric_geometry
            - posture_gradient
            - barrier_gradient
            - self._root_damping * v
            - task_damping
            - barrier_damping
            - self.cfg.geometric_damping * metric_velocity
            + generalized_force
        )

        regularized_metric = metric + torch.diag_embed(
            self.cfg.solve_regularization * self._energy_weights.expand(self.num_envs, -1)
        )
        regularized_finite = torch.all(torch.isfinite(regularized_metric), dim=(-2, -1))
        finite_regularized_metric = torch.where(regularized_finite[:, None, None], regularized_metric, identity)
        solve_cholesky, solve_info = torch.linalg.cholesky_ex(finite_regularized_metric, check_errors=False)
        solve_failed = (solve_info != 0) | ~regularized_finite
        solve_cholesky = torch.where(solve_failed[:, None, None], identity, solve_cholesky)
        nominal_acceleration = self._solve_cholesky(solve_cholesky, right_hand_side)
        solve_residual = torch.bmm(regularized_metric, nominal_acceleration.unsqueeze(-1)).squeeze(-1) - right_hand_side
        solve_residual_norm = torch.linalg.vector_norm(solve_residual, dim=-1)

        (
            commanded_acceleration,
            infeasible,
            acceleration_active,
            jerk_active,
            velocity_active,
            position_active,
        ) = self._project_acceleration(nominal_acceleration)

        finite_inputs = self._finite_step_inputs(action, kinematics)
        finite_dynamics = (
            torch.all(torch.isfinite(metric), dim=(-2, -1))
            & torch.all(torch.isfinite(right_hand_side), dim=-1)
            & torch.all(torch.isfinite(nominal_acceleration), dim=-1)
            & torch.all(torch.isfinite(commanded_acceleration), dim=-1)
        )
        nonfinite = ~(finite_inputs & finite_dynamics)
        cholesky_failed = geometry_failed | solve_failed
        new_fault = (~self._initialized) | infeasible | nonfinite | cholesky_failed
        self._faulted |= new_fault
        active = ~self._faulted

        dt = self.cfg.dt
        next_velocity = v + 0.5 * dt * (previous_acceleration + commanded_acceleration)
        next_position = q + dt * v + dt * dt * (previous_acceleration / 3.0 + commanded_acceleration / 6.0)
        guarded_position = torch.maximum(torch.minimum(next_position, self._safe_upper), self._safe_lower)
        guarded_velocity = torch.maximum(torch.minimum(next_velocity, self._max_velocity), -self._max_velocity)
        roundoff_guard_active = torch.sum(
            (torch.abs(guarded_position - next_position) > self.cfg.bound_tolerance)
            | (torch.abs(guarded_velocity - next_velocity) > self.cfg.bound_tolerance),
            dim=-1,
        )

        self._joint_position[:] = torch.where(active.unsqueeze(-1), guarded_position, q)
        self._joint_velocity[:] = torch.where(active.unsqueeze(-1), guarded_velocity, v)
        self._joint_acceleration[:] = torch.where(active.unsqueeze(-1), commanded_acceleration, previous_acceleration)

        self._acceleration_bound_total.add_(acceleration_active)
        self._jerk_bound_total.add_(jerk_active)
        self._velocity_bound_total.add_(velocity_active)
        self._position_bound_total.add_(position_active)
        self._infeasible_total.add_(infeasible.to(torch.long))

        tracking_valid = torch.full(
            (self.num_envs,),
            measured_joint_position is not None and measured_joint_velocity is not None,
            dtype=torch.bool,
            device=self.device,
        )
        if measured_joint_position is None:
            tracking_position_error = torch.zeros(self.num_envs, dtype=q.dtype, device=self.device)
        else:
            tracking_position_error = torch.linalg.vector_norm(q - measured_joint_position, dim=-1)
        if measured_joint_velocity is None:
            tracking_velocity_error = torch.zeros(self.num_envs, dtype=q.dtype, device=self.device)
        else:
            tracking_velocity_error = torch.linalg.vector_norm(v - measured_joint_velocity, dim=-1)

        jerk = (self._joint_acceleration - previous_acceleration) / dt
        minimum_margin = torch.minimum(
            self._joint_position - self._joint_lower,
            self._joint_upper - self._joint_position,
        ).amin(dim=-1)
        self._telemetry = JointFabricTelemetry(
            metric_min_cholesky_diagonal=self._sanitize_float(metric_min_cholesky_diagonal),
            cholesky_failed=cholesky_failed,
            energy_residual=self._sanitize_float(energy_residual),
            action_saturation_count=torch.sum(torch.abs(action) > 1.0, dim=-1),
            acceleration_bound_active_count=acceleration_active,
            jerk_bound_active_count=jerk_active,
            velocity_bound_active_count=velocity_active,
            position_bound_active_count=position_active,
            roundoff_guard_active_count=roundoff_guard_active,
            infeasible=infeasible,
            nonfinite=nonfinite,
            faulted=self._faulted.clone(),
            minimum_joint_margin=self._sanitize_float(minimum_margin),
            tracking_valid=tracking_valid,
            tracking_position_error_norm=self._sanitize_float(tracking_position_error),
            tracking_velocity_error_norm=self._sanitize_float(tracking_velocity_error),
            nominal_acceleration=self._sanitize_float(nominal_acceleration),
            commanded_acceleration=self._sanitize_float(self._joint_acceleration.clone()),
            maximum_acceleration=self._sanitize_float(torch.abs(self._joint_acceleration).amax(dim=-1)),
            maximum_jerk=self._sanitize_float(torch.abs(jerk).amax(dim=-1)),
            solve_residual_norm=self._sanitize_float(solve_residual_norm),
            barrier_max_weight=self._sanitize_float(barrier_metric.amax(dim=-1)),
            wrench_norm=self._sanitize_float(torch.linalg.vector_norm(wrench, dim=-1)),
            generalized_force_norm=self._sanitize_float(torch.linalg.vector_norm(generalized_force, dim=-1)),
            funnel_acceleration_norm=self._sanitize_float(torch.linalg.vector_norm(funnel_acceleration, dim=-1)),
            acceleration_bound_activation_total=self._acceleration_bound_total.clone(),
            jerk_bound_activation_total=self._jerk_bound_total.clone(),
            velocity_bound_activation_total=self._velocity_bound_total.clone(),
            position_bound_activation_total=self._position_bound_total.clone(),
            infeasible_total=self._infeasible_total.clone(),
        )
        return self._joint_position

    def _project_acceleration(
        self, nominal_acceleration: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project acceleration into a continuous constant-jerk safety envelope.

        Velocity is quadratic and position is cubic over a constant-jerk
        segment. Their Bernstein control points are, respectively,

        ``(v0, v0 + T*a0/2, v1)`` and
        ``(q0, q0 + T*v0/3, q0 + 2*T*v0/3 + T**2*a0/6, q1)``.

        The endpoint control points depend on the new acceleration and produce
        the componentwise bounds below. If any fixed control point is outside
        its envelope, no endpoint acceleration can satisfy this conservative
        convex-hull certificate, so the intersection is marked infeasible.
        """
        dt = self.cfg.dt
        q = self._joint_position
        v = self._joint_velocity
        previous_acceleration = self._joint_acceleration

        acceleration_lower = -self._max_acceleration
        acceleration_upper = self._max_acceleration
        jerk_lower = previous_acceleration - self._max_jerk * dt
        jerk_upper = previous_acceleration + self._max_jerk * dt
        velocity_lower = 2.0 * (-self._max_velocity - v) / dt - previous_acceleration
        velocity_upper = 2.0 * (self._max_velocity - v) / dt - previous_acceleration
        position_lower = 6.0 * (self._safe_lower - q - dt * v) / (dt * dt) - 2.0 * previous_acceleration
        position_upper = 6.0 * (self._safe_upper - q - dt * v) / (dt * dt) - 2.0 * previous_acceleration

        velocity_control_0 = v
        velocity_control_1 = v + 0.5 * dt * previous_acceleration
        velocity_envelope_violation = (
            (velocity_control_0 < -self._max_velocity)
            | (velocity_control_0 > self._max_velocity)
            | (velocity_control_1 < -self._max_velocity)
            | (velocity_control_1 > self._max_velocity)
        )
        position_control_0 = q
        position_control_1 = q + (dt / 3.0) * v
        position_control_2 = q + (2.0 * dt / 3.0) * v + (dt * dt / 6.0) * previous_acceleration
        position_envelope_violation = (
            (position_control_0 < self._safe_lower)
            | (position_control_0 > self._safe_upper)
            | (position_control_1 < self._safe_lower)
            | (position_control_1 > self._safe_upper)
            | (position_control_2 < self._safe_lower)
            | (position_control_2 > self._safe_upper)
        )

        lower = torch.maximum(
            torch.maximum(acceleration_lower, jerk_lower), torch.maximum(velocity_lower, position_lower)
        )
        upper = torch.minimum(
            torch.minimum(acceleration_upper, jerk_upper), torch.minimum(velocity_upper, position_upper)
        )
        infeasible = torch.any(
            (lower > upper) | velocity_envelope_violation | position_envelope_violation,
            dim=-1,
        )

        projected = torch.maximum(torch.minimum(nominal_acceleration, upper), lower)
        tolerance = self.cfg.bound_tolerance
        acceleration_active = torch.sum(
            (nominal_acceleration < acceleration_lower - tolerance)
            | (nominal_acceleration > acceleration_upper + tolerance),
            dim=-1,
        )
        jerk_active = torch.sum(
            (nominal_acceleration < jerk_lower - tolerance) | (nominal_acceleration > jerk_upper + tolerance),
            dim=-1,
        )
        velocity_active = torch.sum(
            (nominal_acceleration < velocity_lower - tolerance)
            | (nominal_acceleration > velocity_upper + tolerance)
            | velocity_envelope_violation,
            dim=-1,
        )
        position_active = torch.sum(
            (nominal_acceleration < position_lower - tolerance)
            | (nominal_acceleration > position_upper + tolerance)
            | position_envelope_violation,
            dim=-1,
        )
        return projected, infeasible, acceleration_active, jerk_active, velocity_active, position_active

    @staticmethod
    def _solve_cholesky(cholesky_factor: torch.Tensor, right_hand_side: torch.Tensor) -> torch.Tensor:
        """Solve an SPD system from its Cholesky factor using graph-safe triangular solves."""
        intermediate = torch.linalg.solve_triangular(cholesky_factor, right_hand_side.unsqueeze(-1), upper=False)
        solution = torch.linalg.solve_triangular(cholesky_factor.transpose(-1, -2), intermediate, upper=True)
        return solution.squeeze(-1)

    def _finite_step_inputs(self, action: torch.Tensor, kinematics: JointFabricTaskKinematics) -> torch.Tensor:
        finite = torch.all(torch.isfinite(action), dim=-1)
        finite &= torch.all(torch.isfinite(kinematics.transverse_error), dim=-1)
        finite &= torch.all(torch.isfinite(kinematics.transverse_jacobian), dim=(-2, -1))
        finite &= torch.all(torch.isfinite(kinematics.transverse_jacobian_dot_velocity), dim=-1)
        finite &= torch.all(torch.isfinite(kinematics.tcp_jacobian), dim=(-2, -1))
        return finite

    @staticmethod
    def _sanitize_float(value: torch.Tensor) -> torch.Tensor:
        """Replace non-finite diagnostics after the associated fault flag is set."""
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    def _validate_step_inputs(
        self,
        action: torch.Tensor,
        kinematics: JointFabricTaskKinematics,
        measured_joint_position: torch.Tensor | None,
        measured_joint_velocity: torch.Tensor | None,
    ) -> None:
        self._validate_tensor(action, (self.num_envs, _WRENCH_DIM), "action")
        self._validate_tensor(
            kinematics.transverse_error,
            (self.num_envs, _TRANSVERSE_DIM),
            "kinematics.transverse_error",
        )
        self._validate_tensor(
            kinematics.transverse_jacobian,
            (self.num_envs, _TRANSVERSE_DIM, _NUM_JOINTS),
            "kinematics.transverse_jacobian",
        )
        self._validate_tensor(
            kinematics.transverse_jacobian_dot_velocity,
            (self.num_envs, _TRANSVERSE_DIM),
            "kinematics.transverse_jacobian_dot_velocity",
        )
        self._validate_tensor(
            kinematics.tcp_jacobian,
            (self.num_envs, _WRENCH_DIM, _NUM_JOINTS),
            "kinematics.tcp_jacobian",
        )
        if measured_joint_position is not None:
            self._validate_tensor(
                measured_joint_position,
                (self.num_envs, _NUM_JOINTS),
                "measured_joint_position",
            )
        if measured_joint_velocity is not None:
            self._validate_tensor(
                measured_joint_velocity,
                (self.num_envs, _NUM_JOINTS),
                "measured_joint_velocity",
            )

    def _validate_tensor(self, value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.shape != shape:
            raise ValueError(f"Expected {name} shape {shape}, received {tuple(value.shape)}.")
        if value.device != self.device:
            raise ValueError(f"Expected {name} on {self.device}, received {value.device}.")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype.")
        if value.dtype != self._joint_position.dtype:
            raise TypeError(f"Expected {name} dtype {self._joint_position.dtype}, received {value.dtype}.")

    def _row_tensor(self, values: Sequence[float], dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=self.device).reshape(1, -1)

    def _limit_tensor(self, values: torch.Tensor | Sequence[float], name: str, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=dtype, device=self.device)
        if tensor.shape == (_NUM_JOINTS,):
            return tensor.reshape(1, _NUM_JOINTS).expand(self.num_envs, -1).clone()
        if tensor.shape == (self.num_envs, _NUM_JOINTS):
            return tensor.clone()
        raise ValueError(
            f"Expected {name} shape {(_NUM_JOINTS,)} or {(self.num_envs, _NUM_JOINTS)}, received {tuple(tensor.shape)}."
        )

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        indices = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if indices.numel() == 0:
            return indices
        if torch.any((indices < 0) | (indices >= self.num_envs)).item():
            raise IndexError("env_ids contains an index outside the controller batch.")
        return indices

    def _select_reset_value(self, value: torch.Tensor, indices: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.device != self.device:
            raise ValueError(f"Expected {name} on {self.device}, received {value.device}.")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype.")
        if value.dtype != self._joint_position.dtype:
            raise TypeError(f"Expected {name} dtype {self._joint_position.dtype}, received {value.dtype}.")
        if value.shape == (self.num_envs, _NUM_JOINTS):
            return value[indices]
        if value.shape == (indices.numel(), _NUM_JOINTS):
            return value
        raise ValueError(
            f"Expected {name} shape {(self.num_envs, _NUM_JOINTS)} or "
            f"{(indices.numel(), _NUM_JOINTS)}, received {tuple(value.shape)}."
        )

    def _validate_cfg(self) -> None:
        cfg = self.cfg
        if cfg.dt <= 0.0:
            raise ValueError(f"Fabric integration dt must be positive, received {cfg.dt}.")
        self._validate_tuple(cfg.kinetic_energy_weights, _NUM_JOINTS, "kinetic_energy_weights", positive=True)
        self._validate_tuple(cfg.root_metric, _NUM_JOINTS, "root_metric", positive=True)
        self._validate_tuple(cfg.transverse_metric, _TRANSVERSE_DIM, "transverse_metric", positive=True)
        self._validate_tuple(cfg.transverse_damping, _TRANSVERSE_DIM, "transverse_damping", non_negative=True)
        self._validate_tuple(cfg.posture_stiffness, _NUM_JOINTS, "posture_stiffness", non_negative=True)
        self._validate_tuple(cfg.root_damping, _NUM_JOINTS, "root_damping", non_negative=True)
        self._validate_tuple(cfg.wrench_scale, _WRENCH_DIM, "wrench_scale", non_negative=True)
        self._validate_tuple(cfg.max_joint_velocity, _NUM_JOINTS, "max_joint_velocity", positive=True)
        self._validate_tuple(cfg.max_joint_acceleration, _NUM_JOINTS, "max_joint_acceleration", positive=True)
        self._validate_tuple(cfg.max_joint_jerk, _NUM_JOINTS, "max_joint_jerk", positive=True)
        if cfg.transverse_funnel_gain < 0.0:
            raise ValueError("transverse_funnel_gain must be non-negative.")
        if cfg.transverse_funnel_length_scale <= 0.0:
            raise ValueError("transverse_funnel_length_scale must be positive.")
        if cfg.geometric_damping < 0.0:
            raise ValueError("geometric_damping must be non-negative.")
        barrier_values = (
            cfg.joint_limit_barrier_metric,
            cfg.joint_limit_barrier_stiffness,
            cfg.joint_limit_barrier_damping,
        )
        if min(barrier_values) < 0.0:
            raise ValueError("Joint-limit barrier metric, stiffness, and damping must be non-negative.")
        if not 0.0 < cfg.joint_limit_safety_fraction < cfg.joint_limit_barrier_fraction < 0.5:
            raise ValueError("Joint-limit fractions must satisfy 0 < safety_fraction < barrier_fraction < 0.5.")
        if not 0.0 < cfg.joint_limit_softness_fraction < cfg.joint_limit_barrier_fraction:
            raise ValueError("joint_limit_softness_fraction must lie between zero and barrier_fraction.")
        if cfg.solve_regularization < 0.0:
            raise ValueError("solve_regularization must be non-negative.")
        if min(cfg.radius_epsilon, cfg.energy_epsilon, cfg.bound_tolerance) <= 0.0:
            raise ValueError("Numerical epsilons and bound_tolerance must be positive.")

    @staticmethod
    def _validate_tuple(
        values: Sequence[float],
        expected_length: int,
        name: str,
        *,
        positive: bool = False,
        non_negative: bool = False,
    ) -> None:
        if len(values) != expected_length:
            raise ValueError(f"{name} must contain {expected_length} values.")
        if positive and min(values) <= 0.0:
            raise ValueError(f"{name} must contain strictly positive values.")
        if non_negative and min(values) < 0.0:
            raise ValueError(f"{name} must contain non-negative values.")

    def _empty_telemetry(self) -> JointFabricTelemetry:
        scalar = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        integer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        boolean = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        vector = torch.zeros(self.num_envs, _NUM_JOINTS, dtype=torch.float32, device=self.device)
        return JointFabricTelemetry(
            metric_min_cholesky_diagonal=scalar.clone(),
            cholesky_failed=boolean.clone(),
            energy_residual=scalar.clone(),
            action_saturation_count=integer.clone(),
            acceleration_bound_active_count=integer.clone(),
            jerk_bound_active_count=integer.clone(),
            velocity_bound_active_count=integer.clone(),
            position_bound_active_count=integer.clone(),
            roundoff_guard_active_count=integer.clone(),
            infeasible=boolean.clone(),
            nonfinite=boolean.clone(),
            faulted=boolean.clone(),
            minimum_joint_margin=scalar.clone(),
            tracking_valid=boolean.clone(),
            tracking_position_error_norm=scalar.clone(),
            tracking_velocity_error_norm=scalar.clone(),
            nominal_acceleration=vector.clone(),
            commanded_acceleration=vector.clone(),
            maximum_acceleration=scalar.clone(),
            maximum_jerk=scalar.clone(),
            solve_residual_norm=scalar.clone(),
            barrier_max_weight=scalar.clone(),
            wrench_norm=scalar.clone(),
            generalized_force_norm=scalar.clone(),
            funnel_acceleration_norm=scalar.clone(),
            acceleration_bound_activation_total=integer.clone(),
            jerk_bound_activation_total=integer.clone(),
            velocity_bound_activation_total=integer.clone(),
            position_bound_activation_total=integer.clone(),
            infeasible_total=integer.clone(),
        )
