# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg

from isaaclab.envs import mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.deploy.mdp as deploy_mdp

from . import joint_pos_env_cfg

_TCP_OBSERVATION_OFFSET = (0.0, 0.0, 0.1925)


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


def _use_flange_pose_actor_observation(env_cfg) -> None:
    """Replace actor joint state with the flange pose in the environment frame."""
    env_cfg.observations.policy.joint_pos = None
    env_cfg.observations.policy.joint_vel = None
    env_cfg.observations.policy.flange_pose = ObsTerm(
        func=mdp.body_pose_w,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["flange"])},
    )


def _use_tcp_pose_actor_observation(env_cfg) -> None:
    """Replace actor joint state with a TCP-like pose offset from the flange."""
    env_cfg.observations.policy.joint_pos = None
    env_cfg.observations.policy.joint_vel = None
    env_cfg.observations.policy.tcp_pose = ObsTerm(
        func=deploy_mdp.body_pose_w_with_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["flange"]),
            "offset": _TCP_OBSERVATION_OFFSET,
        },
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
class Rizon4sGravDisplayportInsertionIKNewtonFlangeObsEnvCfg(Rizon4sGravDisplayportInsertionIKNewtonEnvCfg):
    """Newton IK task whose actor observes flange pose instead of arm joint state."""

    def __post_init__(self):
        super().__post_init__()

        _use_flange_pose_actor_observation(self)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedIKNewtonEnvCfg(
    joint_pos_env_cfg.Rizon4sGravDisplayportInsertionCalibratedNoJointVelEnvCfg
):
    """Calibrated DisplayPort task with relative Newton IK actions."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = _flange_ik_action()


@configclass
class Rizon4sGravDisplayportInsertionCalibratedIKNewtonFlangeObsEnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedIKNewtonEnvCfg
):
    """Calibrated Newton IK task whose actor observes the flange pose."""

    def __post_init__(self):
        super().__post_init__()

        _use_flange_pose_actor_observation(self)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg(
    joint_pos_env_cfg.Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNoJointVelEnvCfg
):
    """Calibrated, domain-randomized DisplayPort task with relative Newton IK actions."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = _flange_ik_action()


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangeObsEnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg
):
    """Calibrated, randomized Newton IK task whose actor observes flange pose."""

    def __post_init__(self):
        super().__post_init__()

        _use_flange_pose_actor_observation(self)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcpObsEnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg
):
    """Calibrated, randomized Newton IK task with a TCP-offset actor observation."""

    def __post_init__(self):
        super().__post_init__()

        _use_tcp_pose_actor_observation(self)
