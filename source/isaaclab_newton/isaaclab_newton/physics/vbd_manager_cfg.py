# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Newton VBD solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class VBDSolverCfg(NewtonSolverCfg):
    """Configuration for the Vertex Block Descent solver."""

    class_type: type[NewtonManager] | str = "{DIR}.vbd_manager:NewtonVBDManager"
    """Manager class for the VBD solver."""

    iterations: int = 10
    """Number of VBD iterations per substep."""

    friction_epsilon: float = 0.01
    """Relative-speed threshold used to regularize contact friction [m/s]."""

    integrate_with_external_rigid_solver: bool = False
    """Whether an external solver integrates rigid bodies."""

    particle_enable_self_contact: bool = False
    """Whether to enable particle self-contact."""

    particle_self_contact_radius: float = 0.005
    """Particle radius used for self-contact detection [m]."""

    particle_self_contact_margin: float = 0.005
    """Self-contact detection margin [m]."""

    particle_collision_detection_interval: int = -1
    """How often particle self-contact detection is applied.

    ``< 0``: once before initialization. ``0``: once before and once after
    initialization. ``k >= 1``: before every ``k`` VBD iterations.
    """

    particle_vertex_contact_buffer_size: int = 32
    """Preallocation size for each vertex contact buffer."""

    particle_edge_contact_buffer_size: int = 64
    """Preallocation size for each edge contact buffer."""

    particle_topological_contact_filter_threshold: int = 2
    """Topological distance below which self-contacts are discarded."""

    particle_rest_shape_contact_exclusion_radius: float = 0.0
    """Rest-shape separation threshold for filtering contacts [m]."""

    rigid_compliant_alm: bool | None = None
    """Whether rigid constraints use Newton's unified compliant-ALM formulation.

    ``None`` preserves the installed Newton version's default behavior.
    """

    rigid_contact_k_start: float = 1.0e2
    """Initial stiffness seed for rigid-body contacts [N/m]."""

    rigid_avbd_alpha: float = 0.95
    """Default C0 stabilization strength for rigid joints and contacts."""

    rigid_avbd_joint_alpha: float | None = None
    """Joint-specific C0 stabilization strength, or ``None`` to use :attr:`rigid_avbd_alpha`."""

    rigid_avbd_contact_alpha: float | None = None
    """Contact-specific C0 stabilization strength, or ``None`` to use :attr:`rigid_avbd_alpha`."""

    rigid_avbd_beta: float = 0.0
    """Fallback AVBD penalty ramp per iteration.

    Its units depend on the constraint: [N/m] for linear constraints and
    [N*m/rad] for angular constraints. Use the linear and angular overrides
    when enabling penalty ramping in production.
    """

    rigid_avbd_linear_beta: float | None = None
    """Linear AVBD penalty ramp per iteration [N/m], or ``None`` to use :attr:`rigid_avbd_beta`."""

    rigid_avbd_angular_beta: float | None = None
    """Angular AVBD penalty ramp per iteration [N*m/rad], or ``None`` to use :attr:`rigid_avbd_beta`."""

    rigid_avbd_gamma: float = 0.999
    """Per-step decay applied to AVBD penalties and persisted hard-contact duals."""

    rigid_contact_hard: bool = True
    """Whether rigid contacts use augmented-Lagrangian hard constraints instead of penalty-only contact."""

    rigid_contact_history: bool = False
    """Whether to warm-start rigid contact penalties and duals across steps.

    This requires a Newton collision pipeline configured with contact matching.
    """

    rigid_body_contact_buffer_size: int = 64
    """Maximum body-body contacts stored per rigid body."""

    rigid_body_particle_contact_buffer_size: int = 256
    """Per-body capacity of the particle, edge, and face soft-contact list.

    Increase this value when Newton reports a per-body particle contact buffer overflow.
    Only used when :attr:`integrate_with_external_rigid_solver` is ``False``.
    """

    rigid_joint_linear_ke: float = 1.0e5
    """Penalty stiffness ceiling for structural linear joint constraints [N/m]."""

    rigid_joint_angular_ke: float = 1.0e5
    """Penalty stiffness ceiling for structural angular joint constraints [N*m/rad]."""

    rigid_joint_linear_k_start: float = 1.0e2
    """Initial linear joint penalty used when linear AVBD ramping is enabled [N/m]."""

    rigid_joint_angular_k_start: float = 1.0e1
    """Initial angular joint penalty used when angular AVBD ramping is enabled [N*m/rad]."""

    rigid_joint_linear_kd: float = 0.0
    """Damping for structural linear joint constraints [N*s/m]."""

    rigid_joint_angular_kd: float = 0.0
    """Damping for structural angular joint constraints [N*m*s/rad]."""
