# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for domain-randomized deploy action terms."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .actions_cfg import DeployOperationalSpaceControllerActionCfg


@configclass
class NoisyDelayedOperationalSpaceControllerActionCfg(DeployOperationalSpaceControllerActionCfg):
    """OSC action with per-episode action bias, per-step action noise, and command latency.

    Behaves exactly like :class:`DeployOperationalSpaceControllerActionCfg` when the noise
    half-widths are zero and the latency range is ``(0, 0)``, including for LEAPP export.

    The noise fields are read on every step rather than cached, so a curriculum can widen
    them at runtime through ``actions.<term>.<field>``.
    """

    class_type: type | str = (
        "isaaclab_tasks.contrib.deploy.mdp.dr_actions:NoisyDelayedOperationalSpaceControllerAction"
    )

    action_bias_halfwidth: float = 0.0
    """Half-width of the per-episode uniform action bias, in action units.

    Sampled once per episode per component and held. This is the *correlated* component
    of the action-noise decomposition used by DextrAH-G.
    """

    action_noise_halfwidth: float = 0.0
    """Half-width of the per-step uniform action noise, in action units.

    Resampled every control step. This is the *uncorrelated* component.
    """

    latency_steps_range: tuple[float, float] = (0.0, 0.0)
    """Range of command latency in whole control steps, sampled per environment at reset.

    Values are rounded to integers and clamped to :attr:`max_latency_steps`.
    """

    max_latency_steps: int = 4
    """Buffer depth reserved for latency [control steps].

    Fixed at construction because it sizes the delay buffer, so it bounds how far a
    curriculum may widen :attr:`latency_steps_range`.
    """
