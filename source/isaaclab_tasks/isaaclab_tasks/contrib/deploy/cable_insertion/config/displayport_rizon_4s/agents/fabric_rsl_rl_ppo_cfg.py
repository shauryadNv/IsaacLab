# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for the isolated DisplayPort fabric experiment."""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import Rizon4sGravDisplayportInsertionRNNPPORunnerCfg


@configclass
class Rizon4sFabricDisplayportInsertionRNNPPORunnerCfg(Rizon4sGravDisplayportInsertionRNNPPORunnerCfg):
    """Keep the shipping PPO settings while isolating fabric experiment logs."""

    experiment_name = "displayport_insertion_rizon4s_fabric"
