# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-randomized deploy action terms."""

from __future__ import annotations

__all__ = ["NoisyDelayedOperationalSpaceControllerAction"]

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .actions import DeployOperationalSpaceControllerAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .dr_actions_cfg import NoisyDelayedOperationalSpaceControllerActionCfg


class NoisyDelayedOperationalSpaceControllerAction(DeployOperationalSpaceControllerAction):
    """OSC action with action noise and per-environment command latency.

    Models two effects the deployment path has and the simulator does not:

    * **Action noise**, split into a per-episode bias and per-step jitter, following the
      correlated / uncorrelated decomposition used by DextrAH-G.
    * **Command latency**, because the policy output crosses ROS and the Flexiv RDK
      before it reaches the joint controller. Neither reference paper models this; it is
      motivated by the deployment stack.

    Latency is applied once per control step, which is where the real transport delay
    lives. It is not a physics-substep delay.
    """

    cfg: NoisyDelayedOperationalSpaceControllerActionCfg

    def __init__(self, cfg: NoisyDelayedOperationalSpaceControllerActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if cfg.max_latency_steps < 0:
            raise ValueError("max_latency_steps must be non-negative.")

        self._max_latency = int(cfg.max_latency_steps)
        # Ring of past commands; index 0 is the most recent.
        self._command_history = torch.zeros(
            self._max_latency + 1, self.num_envs, self.action_dim, device=self.device
        )
        self._latency_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._action_bias = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._env_index = torch.arange(self.num_envs, device=self.device)

    def _sample_latency(self, env_ids: Sequence[int] | slice) -> None:
        low, high = self.cfg.latency_steps_range
        if high <= 0.0:
            self._latency_steps[env_ids] = 0
            return
        num_envs = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
        sampled = torch.empty(num_envs, device=self.device).uniform_(float(low), float(high))
        self._latency_steps[env_ids] = sampled.round().long().clamp_(0, self._max_latency)

    def _sample_bias(self, env_ids: Sequence[int] | slice) -> None:
        halfwidth = float(self.cfg.action_bias_halfwidth)
        if halfwidth <= 0.0:
            self._action_bias[env_ids] = 0.0
            return
        num_envs = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
        self._action_bias[env_ids] = torch.empty(
            num_envs, self.action_dim, device=self.device
        ).uniform_(-halfwidth, halfwidth)

    def process_actions(self, actions: torch.Tensor):
        command = actions
        if self.cfg.action_bias_halfwidth > 0.0:
            command = command + self._action_bias
        noise_halfwidth = float(self.cfg.action_noise_halfwidth)
        if noise_halfwidth > 0.0:
            command = command + torch.empty_like(command).uniform_(-noise_halfwidth, noise_halfwidth)

        if self._max_latency > 0:
            self._command_history = torch.roll(self._command_history, shifts=1, dims=0)
            self._command_history[0] = command
            command = self._command_history[self._latency_steps, self._env_index]

        super().process_actions(command)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        reset_ids = slice(None) if env_ids is None else env_ids
        self._sample_latency(reset_ids)
        self._sample_bias(reset_ids)
        # Clear stale commands so a reset env cannot act on the previous episode's tail.
        self._command_history[:, reset_ids, :] = 0.0
