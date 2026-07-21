# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum helpers for deploy manipulation environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs import mdp as core_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ramp_action_rate_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data,
    num_steps_per_env: int,
    end_iter: float,
    weight_start: float,
    weight_end: float,
    start_iter: float = 0.0,
):
    """Linearly ramp ``rewards.action_rate.weight`` over training iterations.

    Uses the same iteration accounting as ``reset_plug_at_goal_curriculum``:
    ``current_iter = env.common_step_counter / num_steps_per_env``.
    """
    del env_ids  # curriculum terms receive env_ids but this schedule is global

    if not num_steps_per_env or end_iter is None:
        new_weight = weight_end
    else:
        current_iter = env.common_step_counter / float(num_steps_per_env)
        span = max(float(end_iter) - float(start_iter), 1e-9)
        frac = (current_iter - float(start_iter)) / span
        frac = min(max(frac, 0.0), 1.0)
        new_weight = weight_start + frac * (weight_end - weight_start)

    if data != new_weight:
        return new_weight
    return core_mdp.modify_term_cfg.NO_CHANGE
