# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-randomization event terms for deployment tasks.

Covers the two randomizations that Isaac Lab's stock event terms do not reach:
operational-space controller gains, and rigid-body materials whose sampling range
changes at runtime under a curriculum.
"""

from __future__ import annotations

__all__ = ["AdrRigidBodyMaterial", "randomize_osc_task_gains"]

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import events as core_events
from isaaclab.managers import ManagerTermBase
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


class randomize_osc_task_gains(ManagerTermBase):
    """Scale the operational-space controller's task-space gains per environment.

    The task-space env zeroes the arm joint PD so the OSC can drive the joints with pure
    torque, which makes :func:`~isaaclab.envs.mdp.events.randomize_actuator_gains` a
    no-op. The gains that are actually in the loop live on the controller instead, and
    with ``inertial_dynamics_decoupling=False`` they carry physical units (N/m and
    N*m/rad), so scaling them spans genuinely softer and stiffer Cartesian behavior.

    Writing the gains is safe because :class:`~isaaclab.controllers.OperationalSpaceController`
    builds them once in its constructor and, in ``impedance_mode="fixed"``, neither
    ``set_command`` nor ``reset`` touches them again. The root-frame gains are recomputed
    from these task-frame tensors on every ``compute``, so the new values take effect
    immediately.

    Args:
        stiffness_scale_range: ``(low, high)`` multiplier on the nominal task stiffness,
            sampled independently per axis.
        damping_ratio_scale_range: ``(low, high)`` multiplier on the nominal damping
            ratio, sampled independently per axis.
        action_term_name: Name of the OSC action term to target.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # The event manager is constructed before the action manager, so the controller
        # cannot be resolved here.
        self._osc = None
        self._nominal_stiffness: torch.Tensor | None = None
        self._nominal_damping_ratio: torch.Tensor | None = None

    def _resolve(self, env: ManagerBasedEnv, action_term_name: str) -> None:
        action_term = env.action_manager.get_term(action_term_name)
        osc = getattr(action_term, "_osc", None)
        if osc is None:
            raise ValueError(
                f"Action term '{action_term_name}' does not own an operational-space controller;"
                " randomize_osc_task_gains only applies to OSC action terms."
            )
        if osc.cfg.impedance_mode != "fixed":
            raise ValueError(
                "randomize_osc_task_gains requires impedance_mode='fixed'. In the variable modes the"
                " controller overwrites its gains from the action vector on every step."
            )
        self._osc = osc
        self._nominal_stiffness = torch.as_tensor(
            osc.cfg.motion_stiffness_task, dtype=torch.float32, device=env.device
        ).reshape(1, 6)
        self._nominal_damping_ratio = torch.as_tensor(
            osc.cfg.motion_damping_ratio_task, dtype=torch.float32, device=env.device
        ).reshape(1, 6)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        stiffness_scale_range: tuple[float, float] = (1.0, 1.0),
        damping_ratio_scale_range: tuple[float, float] = (1.0, 1.0),
        action_term_name: str = "arm_action",
    ) -> None:
        if self._osc is None:
            self._resolve(env, action_term_name)

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        num_envs = len(env_ids)

        stiffness_scale = math_utils.sample_uniform(
            stiffness_scale_range[0], stiffness_scale_range[1], (num_envs, 6), device=env.device
        )
        damping_scale = math_utils.sample_uniform(
            damping_ratio_scale_range[0], damping_ratio_scale_range[1], (num_envs, 6), device=env.device
        )

        stiffness = self._nominal_stiffness * stiffness_scale
        damping_ratio = self._nominal_damping_ratio * damping_scale

        # Mirror the controller's own construction: zero the unselected axes before
        # deriving damping, so coupling cannot reintroduce them.
        p_gains = self._osc._selection_matrix_motion_task[env_ids] @ torch.diag_embed(stiffness)
        d_gains = torch.diag_embed(2.0 * torch.diagonal(p_gains, dim1=-2, dim2=-1).sqrt() * damping_ratio)

        self._osc._motion_p_gains_task[env_ids] = p_gains
        self._osc._motion_d_gains_task[env_ids] = d_gains


class AdrRigidBodyMaterial(ManagerTermBase):
    """Rigid-body material randomization whose sampling range may change at runtime.

    :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_material` samples its PhysX
    material buckets once in its constructor, because PhysX caps the scene at 64000
    unique materials. A curriculum that widens the range therefore never reaches the
    simulator: the term keeps reassigning the same stale buckets and silently does
    nothing. This wrapper rebuilds the delegate whenever the range changes.

    Rebuilds are rare -- at most once per curriculum level -- so the total number of
    sampled materials stays far below the PhysX cap.

    Accepts and forwards the same parameters as the wrapped term.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._delegate: core_events.randomize_rigid_body_material | None = None
        self._signature: tuple | None = None

    def _build_delegate(self, env: ManagerBasedEnv, params: dict) -> None:
        delegate_cfg = self.cfg.replace(func=core_events.randomize_rigid_body_material, params=dict(params))
        self._delegate = core_events.randomize_rigid_body_material(delegate_cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg,
        make_consistent: bool = False,
    ) -> None:
        params = {
            "static_friction_range": static_friction_range,
            "dynamic_friction_range": dynamic_friction_range,
            "restitution_range": restitution_range,
            "num_buckets": num_buckets,
            "asset_cfg": asset_cfg,
            "make_consistent": make_consistent,
        }
        signature = (
            tuple(static_friction_range),
            tuple(dynamic_friction_range),
            tuple(restitution_range),
            int(num_buckets),
            bool(make_consistent),
        )
        if signature != self._signature:
            self._build_delegate(env, params)
            self._signature = signature

        self._delegate(env, env_ids, **params)
