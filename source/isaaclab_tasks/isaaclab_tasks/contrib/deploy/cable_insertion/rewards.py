# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-private reward terms for DisplayPort insertion retention."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_apply_inverse, quat_error_magnitude


def _pose_is_stable(
    socket_pos: torch.Tensor,
    socket_quat: torch.Tensor,
    plug_pos: torch.Tensor,
    plug_quat: torch.Tensor,
    position_threshold: float,
    orientation_threshold: float,
) -> torch.Tensor:
    """Return the environments whose mate frames satisfy both pose tolerances."""
    position_error = torch.linalg.vector_norm(plug_pos - socket_pos, dim=-1)
    orientation_error = quat_error_magnitude(socket_quat, plug_quat)
    return (position_error <= position_threshold) & (orientation_error <= orientation_threshold)


def _update_success_latch(
    stable: torch.Tensor,
    dwell_count: torch.Tensor,
    latched: torch.Tensor,
    dwell_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a consecutive-step success counter and sticky success latch."""
    next_count = torch.where(stable, dwell_count + 1, torch.zeros_like(dwell_count))
    return next_count, latched | (next_count >= dwell_steps)


def _normalized_outward_action(
    translation_action_root: torch.Tensor,
    socket_quat_w: torch.Tensor,
    robot_root_quat_w: torch.Tensor,
    insertion_axis_socket: torch.Tensor,
    position_scale: torch.Tensor,
) -> torch.Tensor:
    """Project a scaled translation action onto the socket's outward axis.

    The returned value is normalized by the largest projection permitted by
    the configured anisotropic translation scale. Positive values command
    motion out of the socket; inward and lateral commands are non-positive or
    zero after the caller applies :func:`torch.relu`.
    """
    insertion_axis_w = quat_apply(socket_quat_w, insertion_axis_socket.expand_as(translation_action_root))
    insertion_axis_root = quat_apply_inverse(robot_root_quat_w, insertion_axis_w)
    max_projection = torch.sum(torch.abs(insertion_axis_root) * position_scale, dim=-1).clamp_min(1.0e-8)
    return torch.sum(translation_action_root * insertion_axis_root, dim=-1) / max_projection


class _MateFrameTerm(ManagerTermBase):
    """Cache scene objects and transforms shared by insertion reward terms."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._socket = env.scene[cfg.params["socket_cfg"].name]
        self._plug = env.scene[cfg.params["plug_cfg"].name]
        self._socket_offset = torch.tensor(
            cfg.params.get("socket_offset", (0.0, 0.0, 0.0)), device=env.device, dtype=torch.float32
        ).expand(env.num_envs, -1)
        self._plug_offset = torch.tensor(
            cfg.params.get("plug_offset", (0.0, 0.0, 0.0)), device=env.device, dtype=torch.float32
        ).expand(env.num_envs, -1)
        self._plug_rot_offset = torch.tensor(
            cfg.params.get("plug_rot_offset", (0.0, 0.0, 0.0, 1.0)), device=env.device, dtype=torch.float32
        ).expand(env.num_envs, -1)
        self._identity_quat = torch.zeros(env.num_envs, 4, device=env.device)
        self._identity_quat[:, 3] = 1.0
        self._position_threshold = float(cfg.params.get("position_threshold", 0.003))
        self._orientation_threshold = float(cfg.params.get("orientation_threshold", 0.0872664626))
        if self._position_threshold <= 0.0 or self._orientation_threshold <= 0.0:
            raise ValueError("Insertion retention pose thresholds must be positive.")

    def _mate_frames(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        socket_pos, socket_quat = combine_frame_transforms(
            self._socket.data.root_link_pos_w.torch,
            self._socket.data.root_link_quat_w.torch,
            self._socket_offset,
            self._identity_quat,
        )
        plug_pos, plug_quat = combine_frame_transforms(
            self._plug.data.root_link_pos_w.torch,
            self._plug.data.root_link_quat_w.torch,
            self._plug_offset,
            self._plug_rot_offset,
        )
        return socket_pos, socket_quat, plug_pos, plug_quat

    def _is_stable(self) -> torch.Tensor:
        return _pose_is_stable(*self._mate_frames(), self._position_threshold, self._orientation_threshold)


class stable_insertion(_MateFrameTerm):
    """Reward every policy step spent within strict insertion pose tolerances.

    Unlike the existing position-only success diagnostic, this term also
    requires the configured mate-frame orientation tolerance.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        socket_cfg: SceneEntityCfg,
        plug_cfg: SceneEntityCfg,
        socket_offset: Sequence[float],
        plug_offset: Sequence[float],
        plug_rot_offset: Sequence[float],
        position_threshold: float = 0.003,
        orientation_threshold: float = 0.0872664626,
    ) -> torch.Tensor:
        return self._is_stable().float()


class post_success_retraction_action_l2(_MateFrameTerm):
    """Penalize outward translation commands after a stable insertion.

    Success latches only after the mate pose remains within tolerance for a
    configurable number of consecutive policy steps. Once latched, it remains
    active for the rest of the episode, including after the plug leaves the
    success region. Inward preload and lateral search actions are not penalized.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._dwell_steps = int(cfg.params.get("dwell_steps", 1))
        if self._dwell_steps <= 0:
            raise ValueError("Insertion retention dwell_steps must be positive.")
        self._dwell_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._latched = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._action_term = env.action_manager.get_term(cfg.params.get("action_name", "arm_action"))
        if self._action_term.cfg.task_frame_rel_path is not None:
            raise ValueError("Insertion retention supports only OSC actions expressed in the robot-root frame.")
        position_scale = self._action_term.cfg.position_scale
        if isinstance(position_scale, int | float):
            position_scale = (float(position_scale),) * 3
        if len(position_scale) != 3:
            raise ValueError("Insertion retention requires a three-element OSC position_scale.")
        self._position_scale = torch.tensor(position_scale, device=env.device, dtype=torch.float32)
        insertion_axis = torch.tensor(
            cfg.params.get("insertion_axis", (1.0, 0.0, 0.0)), device=env.device, dtype=torch.float32
        )
        insertion_axis_norm = torch.linalg.vector_norm(insertion_axis)
        if not bool(torch.isfinite(insertion_axis_norm)) or insertion_axis_norm.item() <= 0.0:
            raise ValueError("Insertion retention insertion_axis must be finite and non-zero.")
        self._insertion_axis_socket = insertion_axis / insertion_axis_norm
        self._robot = env.scene[cfg.params.get("robot_cfg", SceneEntityCfg("robot")).name]

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear dwell and latch state for reset environments."""
        if env_ids is None:
            env_ids = slice(None)
        self._dwell_count[env_ids] = 0
        self._latched[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        socket_cfg: SceneEntityCfg,
        plug_cfg: SceneEntityCfg,
        socket_offset: Sequence[float],
        plug_offset: Sequence[float],
        plug_rot_offset: Sequence[float],
        position_threshold: float = 0.003,
        orientation_threshold: float = 0.0872664626,
        dwell_steps: int = 1,
        action_name: str = "arm_action",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        insertion_axis: Sequence[float] = (1.0, 0.0, 0.0),
    ) -> torch.Tensor:
        mate_frames = self._mate_frames()
        stable = _pose_is_stable(*mate_frames, self._position_threshold, self._orientation_threshold)
        self._dwell_count, self._latched = _update_success_latch(
            stable, self._dwell_count, self._latched, self._dwell_steps
        )
        socket_quat_w = mate_frames[1]
        outward_action = _normalized_outward_action(
            self._action_term.processed_actions[:, :3],
            socket_quat_w,
            self._robot.data.root_link_quat_w.torch,
            self._insertion_axis_socket,
            self._position_scale,
        )
        return self._latched.float() * torch.square(torch.relu(outward_action))
