# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

_INSERTION_ENV_ENTRY = (
    "isaaclab_tasks.manager_based.manipulation.deploy.cable_insertion"
    ".insertion_env:DisplayportInsertionEnv"
)

##
# Register Gym environments.
##

# UR10e with Robotiq 2F-140 gripper
gym.register(
    id="Isaac-Deploy-DisplayportInsertion-UR10e-2F140-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F140DisplayportInsertionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-DisplayportInsertion-UR10e-2F140-Play-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F140DisplayportInsertionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-DisplayportInsertion-UR10e-2F140-NoJointVel-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F140DisplayportInsertionNoJointVelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-DisplayportInsertion-UR10e-2F140-NoJointVel-Play-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F140DisplayportInsertionNoJointVelEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-DisplayportInsertion-UR10e-2F140-ROS-Inference-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ros_inference_env_cfg:UR10e2F140DisplayportInsertionROSInferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-DisplayportInsertion-UR10e-2F140-NoJointVel-ROS-Inference-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ros_inference_env_cfg:UR10e2F140DisplayportInsertionNoJointVelROSInferenceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDisplayportInsertionRNNPPORunnerCfg",
    },
)

# UR10e with Robotiq 2F-85 gripper
gym.register(
    id="Isaac-Deploy-DisplayportInsertion-UR10e-2F85-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F85DisplayportInsertionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-DisplayportInsertion-UR10e-2F85-Play-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F85DisplayportInsertionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-DisplayportInsertion-UR10e-2F85-NoJointVel-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F85DisplayportInsertionNoJointVelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-DisplayportInsertion-UR10e-2F85-NoJointVel-Play-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F85DisplayportInsertionNoJointVelEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-DisplayportInsertion-UR10e-2F85-ROS-Inference-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ros_inference_env_cfg:UR10e2F85DisplayportInsertionROSInferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-DisplayportInsertion-UR10e-2F85-NoJointVel-ROS-Inference-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ros_inference_env_cfg:UR10e2F85DisplayportInsertionNoJointVelROSInferenceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10eDisplayportInsertionRNNPPORunnerCfg",
    },
)
