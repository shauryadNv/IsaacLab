# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton's experimental FeatherPGS solver."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from isaaclab.utils.configclass import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class FeatherPGSSolverCfg(NewtonSolverCfg):
    """Configuration for Newton's reduced-coordinate FeatherPGS solver.

    FeatherPGS evaluates articulated-body dynamics in generalized coordinates
    and resolves rigid contacts and joint constraints with projected
    Gauss-Seidel iterations. It always uses Newton's collision pipeline.
    """

    class_type: type[NewtonManager] | str = "{DIR}.feather_pgs_manager:NewtonFeatherPGSManager"
    """Manager class for the FeatherPGS solver."""

    solver_type: str = "feather_pgs"
    """Solver type metadata."""

    angular_damping: float = 0.05
    """Angular velocity damping rate [s^-1]."""

    update_mass_matrix_interval: int = 1
    """Number of simulation steps between mass-matrix updates."""

    enable_contact_friction: bool = True
    """Whether to solve Coulomb-friction contact rows."""

    contact_friction_gap_threshold: float = math.inf
    """Maximum contact gap at which friction rows are enabled [m]."""

    contact_friction_position_iterations: int = -1
    """Number of final position iterations that solve friction.

    ``-1`` solves friction in every position iteration and ``0`` skips it in
    the position pass.
    """

    contact_friction_shared_anchor: bool = False
    """Whether friction rows share one anchor per contact pair."""

    contact_friction_anchor_limit: int = 0
    """Deprecated compatibility field with no effect.

    Use :attr:`friction_anchor_beta` to select persistent or point friction.
    """

    contact_friction_scale: float = 1.0
    """Scale applied to Coulomb friction limits [dimensionless]."""

    contact_shared_anchor: bool = False
    """Whether normal contact rows share one anchor per contact pair."""

    enable_joint_limits: bool = False
    """Whether to enforce joint position limits."""

    enable_bilateral_preelimination: bool = False
    """Whether to project eligible mimic and connect constraints before PGS.

    This matrix-free feature requires :attr:`pgs_velocity_iterations` to be
    zero; unsupported configurations warn and use iterative bilateral rows.
    """

    joint_limit_activation_gap: float = math.inf
    """Distance from a joint limit at which its row activates [m or rad, depending on joint type]."""

    enable_joint_velocity_limits: bool = False
    """Whether to enforce per-DOF joint velocity limits."""

    velocity_limit_activation_fraction: float = 0.0
    """Fraction of each joint velocity limit at which its row activates [dimensionless]."""

    fuse_joint_velocity_limits: bool = True
    """Whether compatible PhysX-style drive rows fuse their velocity-limit clamp."""

    pgs_iterations: int = 12
    """Number of position-level Gauss-Seidel iterations per simulation step."""

    pgs_velocity_iterations: int = 0
    """Number of velocity-only Gauss-Seidel iterations per simulation step.

    Positive values require CUDA matrix-free mode.
    """

    pgs_beta: float = 0.2
    """ERP-style position correction factor [dimensionless]."""

    pgs_cfm: float = 1.0e-6
    """Constraint-force-mixing regularization.

    Units are [kg^-1] for contact and prismatic rows and
    [(kg·m²)^-1] for revolute and angular rows.
    """

    pgs_omega: float = 1.0
    """Successive over-relaxation factor [dimensionless]."""

    pgs_contact_regularization: float = 0.0
    """Numerical contact regularization [dimensionless]."""

    pgs_velocity_drive_mode: Literal["active", "freeze"] = "freeze"
    """Whether drive rows remain active during velocity-only iterations."""

    dense_max_constraints: int = 32
    """Maximum dense constraint rows per world."""

    pgs_warmstart: bool = False
    """Whether to reuse contact impulses matched from the previous step.

    External contacts require collision-pipeline contact matching.
    """

    pgs_warmstart_decay: float = 1.0
    """Scale applied to matched impulses before reuse [dimensionless]."""

    mf_warmstart: bool = False
    """Deprecated matrix-free warm-start alias."""

    mf_warmstart_decay: float = 1.0
    """Deprecated matrix-free warm-start decay alias [dimensionless]."""

    pgs_mode: Literal["split", "matrix_free"] = "split"
    """Constraint solve layout; ``"matrix_free"`` requires CUDA."""

    articulated_contact_response: Literal["immediate", "propagation", "propagation-fused", "propagation-colored"] = (
        "immediate"
    )
    """Articulated contact-response implementation used in matrix-free mode."""

    propagation_same_articulation_rows: bool = False
    """Whether propagation handles contacts between links of one articulation."""

    propagation_cached_response: bool = True
    """Whether eligible propagation paths cache per-body response matrices."""

    propagation_cached_response_max_bodies: int = 8
    """Maximum contact-active articulation bodies cached per world."""

    pgs_schedule: Literal["interleaved", "contact_then_internal", "physx_grasp"] = "interleaved"
    """Ordering of contact and internal-constraint sweeps in matrix-free mode."""

    friction_mode: Literal["current", "bisection", "bisection_desaxce", "coulomb_newton"] = "current"
    """Coulomb friction strategy used in matrix-free mode.

    Only ``"current"`` is supported in split mode.
    """

    mf_max_constraints: int = 512
    """Maximum matrix-free constraint rows per world."""

    pgs_kernel: Literal["loop", "tiled_row", "tiled_contact", "streaming"] = "tiled_row"
    """Dense-path PGS kernel."""

    pgs_chunk_size: int | None = None
    """Streaming PGS chunk size; ``None`` selects the solver default."""

    use_parallel_streams: bool = True
    """Whether independent articulation groups use parallel CUDA streams."""

    double_buffer: bool = True
    """Whether mass-matrix and Jacobian maintenance uses double buffering."""

    nvtx: bool = False
    """Whether to emit NVTX profiling ranges."""

    pgs_debug: bool = False
    """Whether to collect PGS convergence diagnostics."""

    drive_mode: Literal["augmented", "physx_pgs"] = "augmented"
    """Joint-drive formulation."""

    serial_kernel_block_dim: int = 256
    """Thread-block size for serial matrix-free kernels."""

    tile_threads: int = 64
    """Thread count for tiled kernels."""

    row_watermark: bool = False
    """Whether to collect constraint-row capacity watermarks."""

    restitution_velocity_threshold: float = 0.5
    """Minimum incident speed for restitution [m/s]."""

    contact_speculative_scale: float = 1.0
    """Scale applied to speculative contact closing velocity [dimensionless]."""

    contact_gap_gate: float = 0.0
    """Gap beyond which normal contact rows are omitted [m]."""

    contact_friction_articulation_pairs_only: bool = False
    """Whether friction rows are limited to articulation-involving contacts."""

    enable_restitution: bool = True
    """Whether qualifying contacts apply restitution."""

    same_articulation_contact_gap_gate: float = 0.0
    """Gap beyond which same-articulation contact rows are omitted [m]."""

    articulation_pair_contact_gap_gate: float = 0.0
    """Gap beyond which articulation-pair contact rows are omitted [m]."""

    warn_constraint_overflow: bool = True
    """Whether to emit device-side warnings when row capacity is exceeded."""

    friction_anchor_beta: float | None = None
    """Persistent friction-anchor position correction factor [dimensionless].

    ``None`` selects persistent patch friction with a factor of 0.2 for the
    default solver, while zero explicitly selects velocity-only point friction.
    """

    contact_torsion_radius: float = 0.0
    """Experimental effective spin-friction radius [m]; zero disables torsion.

    A positive value has restricted solver combinations and does not support
    CUDA graph capture.
    """

    contact_torsion_shape_indices: tuple[int, ...] | None = None
    """Optional global shape indices selected for torsional friction."""

    contact_torsion_shape_patterns: tuple[str, ...] | None = None
    """Optional shape-label regular expressions selected for torsional friction."""

    contact_compliance: bool = False
    """Whether to use experimental implicit material contact compliance.

    This mode has restricted matrix-free contact settings and does not support
    contact reduction or CUDA graph capture.
    """
