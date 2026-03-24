# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Deploy-CableInsertion-Rizon4s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Rizon4sCableInsertionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sCableInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-CableInsertion-Rizon4s-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Rizon4sCableInsertionEnvCfg_PLAY",
    },
)

# Rizon 4s - ROS Inference
gym.register(
    id="Isaac-Deploy-CableInsertion-Rizon4s-ROS-Inference-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ros_inference_env_cfg:Rizon4sCableInsertionROSInferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sCableInsertionRNNPPORunnerCfg",
    },
)
