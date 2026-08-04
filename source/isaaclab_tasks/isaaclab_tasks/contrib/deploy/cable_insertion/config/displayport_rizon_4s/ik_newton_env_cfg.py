# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg

from isaaclab.utils.configclass import configclass

from . import joint_pos_env_cfg


def _flange_ik_action() -> NewtonInverseKinematicsActionCfg:
    """Create the relative flange-pose action shared by DisplayPort IK tasks."""
    return NewtonInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        controller=NewtonIKSolverCfg(optimizer="lm", jacobian_mode="analytic", iterations=24),
        clip={".*": (-0.5, 0.5)},
        objectives=[
            NewtonIKPoseObjectiveCfg(
                body_name="flange",
                command_type="pose",
                use_relative_mode=True,
                scale=0.01,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ],
    )


@configclass
class Rizon4sGravDisplayportInsertionIKNewtonEnvCfg(joint_pos_env_cfg.Rizon4sGravDisplayportInsertionNoJointVelEnvCfg):
    """DisplayPort insertion with a relative Newton IK action for the Rizon 4s flange.

    The six-dimensional policy action commands relative flange translation and
    rotation. Newton IK maps that command to seven arm-joint position targets;
    the simulated actuators then track those targets. The actor omits joint
    velocity for deployment parity with the real-robot policy interface.

    Note:
        This task requires ``presets=newton_mjwarp``, ``presets=newton_sdf``,
        or ``presets=newton_hydroelastic``; it is not compatible with
        ``presets=physx``.
    """

    def __post_init__(self):
        super().__post_init__()

        # Solve a bounded flange-frame delta pose each policy step.
        self.actions.arm_action = _flange_ik_action()


@configclass
class Rizon4sGravDisplayportInsertionCalibratedIKNewtonEnvCfg(
    joint_pos_env_cfg.Rizon4sGravDisplayportInsertionCalibratedNoJointVelEnvCfg
):
    """Calibrated DisplayPort task with relative Newton IK actions."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = _flange_ik_action()


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg(
    joint_pos_env_cfg.Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNoJointVelEnvCfg
):
    """Calibrated, domain-randomized DisplayPort task with relative Newton IK actions."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = _flange_ik_action()
