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

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-IK-FlangePose6D",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-IK-FlangePose6D-ActionClip1",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DActionClip1EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-IK-Tcp15cmObsPose6D",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-IK-FlangePose6D-Scale015",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DScale015EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-IK-FlangePose6D-Scale025",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DScale025EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-IK-Tcp15cmObsPose6D-Scale015",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DScale015EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-IK-Tcp15cmObsPose6D-Scale025",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DScale025EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-IK-TcpObs",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcpObsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Nominal-DR-Newton-OSC-FlangePose6D",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCFlangePose6DEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-OSC-FlangePose6D",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id=(
        "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-OSC-"
        "FlangePose6D-ArmFrictionDR"
    ),
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DArmFrictionDREnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-OSC-FlangePose6D-Scale010",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale010EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-OSC-FlangePose6D-ActionClip1",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DActionClip1EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-OSC-FlangePose6D-Scale010-ActionClip1",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale010ActionClip1EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-OSC-FlangePose6D-Scale0125-ActionClip1",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale0125ActionClip1EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-OSC-FlangePose6D-Scale025-ActionClip1",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale025ActionClip1EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-OSC-Inertial-FlangePose6D",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCInertialFlangePose6DEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-OSC-Tcp15cmObsPose6D",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCTcp15cmObsPose6DEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Nominal-DR-Newton-IK-Tcp15cmPose6D",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonTcp15cmPose6DEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Nominal-DR-Newton-IK-Tcp15cmPose6D-Scale0125",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonTcp15cmPose6DScale0125EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Nominal-DR-Newton-IK-Tcp15cmPose6D-Scale015",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonTcp15cmPose6DScale015EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-IK-Tcp15cmPose6D",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmPose6DEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Nominal-DR-Newton-OSC-Tcp15cmPose6D",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCTcp15cmPose6DEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Nominal-DR-Newton-OSC-Tcp15cmPose6D-Scale015",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCTcp15cmPose6DScale015EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-OSC-Tcp15cmPose6D",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCTcp15cmPose6DEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)


gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Nominal-DR-Newton-IK-PhysXProfile",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonPhysXProfileEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Nominal-DR-Newton-OSC-PhysXProfile",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ik_newton_env_cfg:"
            "Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCPhysXProfileEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)
