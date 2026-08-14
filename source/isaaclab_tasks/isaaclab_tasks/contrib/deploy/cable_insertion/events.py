# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific reset events for cable insertion environments."""

from __future__ import annotations

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


class ResetPlugAtGoalCurriculum(ManagerTermBase):
    """Reset plugs in near-goal and approach bands along the insertion axis.

    The near-goal sampling probability is linearly annealed over policy
    iterations. Both bands share the socket insertion frame, which avoids a
    discontinuity between curriculum and approach initial states.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.plug: RigidObject = env.scene[cfg.params["plug_cfg"].name]
        self.socket: RigidObject = env.scene[cfg.params["socket_cfg"].name]

        self.at_goal_prob = float(cfg.params.get("at_goal_prob", 0.8))
        self.at_goal_prob_final = cfg.params.get("at_goal_prob_final")
        self.anneal_start_iter = float(cfg.params.get("anneal_start_iter", 0.0))
        self.anneal_end_iter = cfg.params.get("anneal_end_iter")
        self.num_steps_per_env = cfg.params.get("num_steps_per_env")

        insertion_axis = cfg.params.get("insertion_axis", [0.0, 0.0, 1.0])
        self.insertion_axis = torch.tensor(insertion_axis, device=env.device, dtype=torch.float32)
        self.insertion_axis /= torch.linalg.vector_norm(self.insertion_axis)

        self.insertion_length = float(cfg.params.get("insertion_length", 0.02))
        self.at_goal_depth_range = cfg.params.get("at_goal_depth_range")
        self.approach_depth_range = cfg.params.get("approach_depth_range")
        self.normal_pose_range = cfg.params.get("normal_pose_range", {})

        self.socket_insertion_offset = torch.tensor(
            cfg.params.get("socket_insertion_offset", [0.0, 0.0, 0.0]),
            device=env.device,
            dtype=torch.float32,
        )
        self.plug_insertion_offset = torch.tensor(
            cfg.params.get("plug_insertion_offset", [0.0, 0.0, 0.0]),
            device=env.device,
            dtype=torch.float32,
        )
        self.goal_rot = torch.tensor(
            cfg.params.get("goal_rot", [0.0, 0.0, 0.0, 1.0]),
            device=env.device,
            dtype=torch.float32,
        )
        self.identity_quat = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=env.device,
            dtype=torch.float32,
        )

    def _current_at_goal_prob(self, env: ManagerBasedEnv) -> float:
        """Return the near-goal reset probability at the current iteration."""
        if self.at_goal_prob_final is None or self.anneal_end_iter is None or not self.num_steps_per_env:
            return self.at_goal_prob

        current_iter = env.common_step_counter / float(self.num_steps_per_env)
        span = max(float(self.anneal_end_iter) - self.anneal_start_iter, 1.0e-9)
        fraction = (current_iter - self.anneal_start_iter) / span
        fraction = min(max(fraction, 0.0), 1.0)
        return self.at_goal_prob + fraction * (float(self.at_goal_prob_final) - self.at_goal_prob)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        plug_cfg: SceneEntityCfg | None = None,
        socket_cfg: SceneEntityCfg | None = None,
        at_goal_prob: float = 0.8,
        insertion_axis: list[float] | None = None,
        insertion_length: float = 0.02,
        socket_insertion_offset: list[float] | None = None,
        plug_insertion_offset: list[float] | None = None,
        goal_rot: list[float] | None = None,
        normal_pose_range: dict[str, list[float]] | None = None,
        at_goal_prob_final: float | None = None,
        anneal_start_iter: float = 0.0,
        anneal_end_iter: float | None = None,
        num_steps_per_env: int | None = None,
        at_goal_depth_range: list[float] | None = None,
        approach_depth_range: list[float] | None = None,
    ) -> None:
        """Place resetting plugs in the configured insertion-depth bands.

        Args:
            env: Environment whose plugs are reset.
            env_ids: Environment indices to reset.
            plug_cfg: Plug scene entity configuration.
            socket_cfg: Socket scene entity configuration.
            at_goal_prob: Initial probability of sampling the near-goal band.
            insertion_axis: Socket-local insertion axis.
            insertion_length: Nominal insertion length [m].
            socket_insertion_offset: Socket-root to insertion-point offset [m].
            plug_insertion_offset: Plug-root to insertion-point offset [m].
            goal_rot: Plug orientation relative to the socket.
            normal_pose_range: Approach-state position randomization ranges [m].
            at_goal_prob_final: Final near-goal sampling probability.
            anneal_start_iter: Iteration at which annealing starts.
            anneal_end_iter: Iteration at which annealing ends.
            num_steps_per_env: Environment steps in one policy iteration.
            at_goal_depth_range: Near-goal insertion-depth range [m].
            approach_depth_range: Approach insertion-depth range [m].
        """
        del (
            plug_cfg,
            socket_cfg,
            at_goal_prob,
            insertion_axis,
            insertion_length,
            socket_insertion_offset,
            plug_insertion_offset,
            goal_rot,
            normal_pose_range,
            at_goal_prob_final,
            anneal_start_iter,
            anneal_end_iter,
            num_steps_per_env,
            at_goal_depth_range,
            approach_depth_range,
        )

        num_envs = len(env_ids)
        if num_envs == 0:
            return

        socket_pos = wp.to_torch(self.socket.data.root_pos_w)[env_ids]
        socket_quat = wp.to_torch(self.socket.data.root_quat_w)[env_ids]
        socket_offset = self.socket_insertion_offset.unsqueeze(0).expand(num_envs, -1)
        identity_quat = self.identity_quat.unsqueeze(0).expand(num_envs, -1)
        keypoint_origin_w, _ = math_utils.combine_frame_transforms(
            socket_pos,
            socket_quat,
            socket_offset,
            identity_quat,
        )

        insertion_axis_w = math_utils.quat_apply(
            socket_quat,
            self.insertion_axis.unsqueeze(0).expand(num_envs, -1),
        )
        goal_quat_w = math_utils.quat_mul(
            socket_quat,
            self.goal_rot.unsqueeze(0).expand(num_envs, -1),
        )
        plug_offset_w = math_utils.quat_apply(
            goal_quat_w,
            self.plug_insertion_offset.unsqueeze(0).expand(num_envs, -1),
        )

        at_goal_mask = torch.rand(num_envs, device=env.device) < self._current_at_goal_prob(env)
        near_range = self.at_goal_depth_range or [0.0, self.insertion_length]
        if self.approach_depth_range is None:
            raise ValueError("approach_depth_range must be configured for unified insertion resets.")

        near_depth = torch.empty(num_envs, device=env.device).uniform_(float(near_range[0]), float(near_range[1]))
        approach_depth = torch.empty(num_envs, device=env.device).uniform_(
            float(self.approach_depth_range[0]),
            float(self.approach_depth_range[1]),
        )
        depth = torch.where(at_goal_mask, near_depth, approach_depth)

        position_noise = torch.zeros(num_envs, 3, device=env.device)
        for index, key in enumerate(("x", "y", "z")):
            bounds = self.normal_pose_range.get(key, [0.0, 0.0])
            position_noise[:, index].uniform_(float(bounds[0]), float(bounds[1]))
        position_noise[at_goal_mask] = 0.0

        if not hasattr(env, "_displayport_initial_xy_offset_m"):
            env._displayport_initial_xy_offset_m = torch.zeros(env.num_envs, device=env.device)
            env._displayport_initial_xy_offset_valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._displayport_initial_xy_offset_m[env_ids] = torch.linalg.vector_norm(position_noise[:, :2], dim=-1)
        env._displayport_initial_xy_offset_valid[env_ids] = True

        plug_keypoint_w = keypoint_origin_w + depth.unsqueeze(-1) * insertion_axis_w
        plug_pos_w = plug_keypoint_w - plug_offset_w + position_noise
        root_pose = torch.cat((plug_pos_w, goal_quat_w), dim=-1)
        root_velocity = torch.zeros(num_envs, 6, device=env.device, dtype=torch.float32)
        self.plug.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.plug.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
