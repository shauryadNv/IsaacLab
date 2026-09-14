# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FeatherPGS Newton manager."""

from __future__ import annotations

import warp as wp
from newton import Model
from newton.solvers import SolverFeatherPGS

from .feather_pgs_manager_cfg import FeatherPGSSolverCfg
from .newton_manager import NewtonManager


class NewtonFeatherPGSManager(NewtonManager):
    """:class:`NewtonManager` specialization for the FeatherPGS solver.

    FeatherPGS uses Newton's collision pipeline and separate input/output
    states. It consumes both joint efforts and applied rigid-body forces.
    """

    _builder_attribute_solvers = (SolverFeatherPGS,)

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: FeatherPGSSolverCfg) -> SolverFeatherPGS:
        """Construct the configured FeatherPGS solver."""
        # FeatherPGS allocates its constraint buffers during construction, so
        # publish an explicitly configured collision capacity before creating
        # the solver. The collision pipeline later uses this same model field.
        collision_cfg = NewtonManager._collision_cfg
        if collision_cfg is not None and collision_cfg.rigid_contact_max is not None:
            model.rigid_contact_max = int(collision_cfg.rigid_contact_max)
        return SolverFeatherPGS(model, **cls._filter_solver_kwargs(SolverFeatherPGS, solver_cfg))

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: FeatherPGSSolverCfg) -> None:
        """Construct :class:`SolverFeatherPGS` and publish its capabilities."""
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_rigid_body_force_input = True

    @classmethod
    def _prepare_cuda_graph_capture(cls) -> None:
        """Seed FeatherPGS double-buffer events inside the capture stream."""
        cls._solver.seed_double_buffer_events()

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        """Return whether the active FeatherPGS features are capture-safe."""
        contact_compliance = bool(getattr(cls._solver, "contact_compliance", False))
        contact_torsion_radius = float(getattr(cls._solver, "contact_torsion_radius", 0.0))
        return not contact_compliance and contact_torsion_radius <= 0.0

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array(dtype=wp.bool) | None) -> None:
        """Reset FeatherPGS state for the worlds represented by the model."""
        if world_mask is None:
            return
        cls._solver.reset(cls._state_0, world_mask=world_mask[: cls._model.world_count], flags=0)
