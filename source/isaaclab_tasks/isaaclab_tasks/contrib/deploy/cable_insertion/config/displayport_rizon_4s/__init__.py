# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

_INSERTION_ENV_ENTRY = (
    "isaaclab_tasks.contrib.deploy.cable_insertion.insertion_env:DisplayportInsertionEnv"
)

##
# Register Gym environments.
##


gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Rizon4sGravDisplayportInsertionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)


gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Rizon4sGravDisplayportInsertionNoJointVelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)


gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-NoJointVel",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.joint_pos_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedNoJointVelEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)


gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-NoJointVel",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.joint_pos_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNoJointVelEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)


gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Newton-IK",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_newton_env_cfg:Rizon4sGravDisplayportInsertionIKNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)


gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-Newton-IK",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedIKNewtonEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)


gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-IK",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Newton-IK-FlangeObs",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionIKNewtonFlangeObsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-Newton-IK-FlangeObs",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedIKNewtonFlangeObsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-IK-FlangeObs",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangeObsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)
