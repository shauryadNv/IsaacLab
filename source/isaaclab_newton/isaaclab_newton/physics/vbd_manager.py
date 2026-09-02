# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from newton import Contacts, Control, Model, State, eval_ik
from newton.solvers import SolverVBD

from .newton_manager import NewtonManager
from .vbd_manager_cfg import VBDSolverCfg

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext


class NewtonVBDManager(NewtonManager):
    """Newton manager specialization for the VBD solver."""

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize VBD deformable integration when contrib is available."""
        try:
            from isaaclab_contrib.deformable.deformable_object import install_deformable_builder_hooks
        except ModuleNotFoundError as exc:
            if exc.name not in {"isaaclab_contrib", "isaaclab_contrib.deformable"}:
                raise
        else:
            install_deformable_builder_hooks()
        super().initialize(sim_context)

    @classmethod
    def start_simulation(cls) -> None:
        """Start simulation and bind registered deformables to Fabric."""
        if cls._builder is not None:
            cls._builder.color()
        super().start_simulation()
        try:
            from isaaclab_contrib.deformable.deformable_object import setup_registered_deformable_fabric_sync
        except ModuleNotFoundError as exc:
            if exc.name not in {"isaaclab_contrib", "isaaclab_contrib.deformable"}:
                raise
        else:
            setup_registered_deformable_fabric_sync(cls)

    @classmethod
    def instantiate_builder_from_stage(cls) -> None:
        """Create and color the VBD builder from the USD stage."""
        super().instantiate_builder_from_stage()
        if cls._builder is None:
            raise RuntimeError("Newton stage import did not create a builder.")
        cls._builder.color()

    @classmethod
    def _get_usd_import_ignore_paths(cls) -> list[str]:
        """Return registered deformable mesh paths excluded from USD import."""
        return [
            path for entry in cls._deformable_registry for path in (entry.sim_mesh_prim_path, entry.vis_mesh_prim_path)
        ]

    @classmethod
    def _validate_solver_compatibility(cls, solver_cfg: VBDSolverCfg) -> None:
        """Validate options whose availability depends on the installed Newton version."""
        solver_kwargs = cls._filter_solver_kwargs(SolverVBD, solver_cfg)
        if solver_cfg.rigid_compliant_alm is not None and "rigid_compliant_alm" not in solver_kwargs:
            raise RuntimeError(
                "VBDSolverCfg.rigid_compliant_alm requires a Newton version whose SolverVBD constructor "
                "supports that option."
            )

    @classmethod
    def _validate_contact_history_config(cls, solver_cfg: VBDSolverCfg) -> None:
        """Require collision matching when rigid contact history is enabled."""
        collision_cfg = NewtonManager._collision_cfg
        contact_matching = getattr(collision_cfg, "contact_matching", "disabled")
        if solver_cfg.rigid_contact_history and contact_matching not in ("latest", "sticky"):
            raise ValueError(
                "VBDSolverCfg.rigid_contact_history requires "
                "NewtonCollisionPipelineCfg.contact_matching='latest' or 'sticky'."
            )

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> SolverVBD:
        """Construct the configured VBD solver."""
        cls._validate_solver_compatibility(solver_cfg)
        return SolverVBD(model, **cls._filter_solver_kwargs(SolverVBD, solver_cfg))

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> None:
        """Construct VBD and configure its base-manager state."""
        # Preflight before allocating the collision pipeline or mutating the
        # shared manager lifecycle state.
        cls._validate_solver_compatibility(solver_cfg)
        cls._validate_contact_history_config(solver_cfg)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_rigid_body_force_input = not solver_cfg.integrate_with_external_rigid_solver
        # Contact history is allocated by SolverVBD at construction. Build the
        # externally managed pipeline first so CUDA graph capture never needs
        # to resize those history buffers.
        cls._initialize_contacts()
        NewtonManager._solver = cls._create_solver(model, solver_cfg)

    @classmethod
    def _step_solver(
        cls, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Run VBD and recover reduced articulation state from the solved body state."""
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)
        if cls._model.joint_count > 0:
            eval_ik(cls._model, state_1, state_1.joint_q, state_1.joint_qd)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Clear contrib deformable integration when available."""
        try:
            from isaaclab_contrib.deformable.deformable_object import clear_deformable_builder_hooks
        except ModuleNotFoundError as exc:
            if exc.name not in {"isaaclab_contrib", "isaaclab_contrib.deformable"}:
                raise
        else:
            clear_deformable_builder_hooks()

    @classmethod
    def _simulate_physics_only(cls) -> None:
        """Rebuild the VBD particle BVH before stepping physics."""
        if cls._model.particle_count > 0 and hasattr(cls._solver, "rebuild_bvh"):
            cls._solver.rebuild_bvh(cls._state_0)
        super()._simulate_physics_only()
