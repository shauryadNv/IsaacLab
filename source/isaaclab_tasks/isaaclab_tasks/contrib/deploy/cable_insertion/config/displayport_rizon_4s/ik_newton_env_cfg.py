# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.deploy.mdp as deploy_mdp

from . import joint_pos_env_cfg

_LEGACY_TCP_OBSERVATION_OFFSET = (0.0, 0.0, 0.1925)
_TCP_15CM_OFFSET = (0.0, 0.0, 0.15)

_OSC_STIFFNESS = (300.0, 300.0, 300.0, 30.0, 30.0, 30.0)
_OSC_DAMPING_RATIO = (1.0,) * 6
_OSC_ACTION_SCALE = 0.005


def _ik_action(body_offset_pos: tuple[float, float, float]) -> NewtonInverseKinematicsActionCfg:
    """Create a relative Newton IK action for an offset frame on the flange."""
    return NewtonInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        controller=NewtonIKSolverCfg(optimizer="lm", jacobian_mode="analytic", iterations=24),
        clip={".*": (-0.5, 0.5)},
        objectives=[
            NewtonIKPoseObjectiveCfg(
                body_name="flange",
                body_offset_pos=body_offset_pos,
                command_type="pose",
                use_relative_mode=True,
                scale=0.01,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ],
    )


def _flange_ik_action() -> NewtonInverseKinematicsActionCfg:
    """Create a relative IK action for the flange origin."""
    return _ik_action((0.0, 0.0, 0.0))


def _set_ik_action_scale(env_cfg, scale: float) -> None:
    """Set the translation and rotation scale of the flange pose objective."""
    pose_objective = env_cfg.actions.arm_action.objectives[0]
    pose_objective.scale = scale


def _set_action_clip(env_cfg, bound: float) -> None:
    """Set the symmetric normalized action bound for a task-space action."""
    env_cfg.actions.arm_action.clip = {".*": (-bound, bound)}


def _flange_osc_action(
    action_scale: float = _OSC_ACTION_SCALE,
    inertial_dynamics_decoupling: bool = True,
    damping_ratio: tuple[float, ...] = _OSC_DAMPING_RATIO,
) -> OperationalSpaceControllerActionCfg:
    """Create a relative-pose torque controller for the flange origin."""
    return OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        body_name="flange",
        body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(),
        clip={".*": (-0.5, 0.5)},
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=inertial_dynamics_decoupling,
            partial_inertial_dynamics_decoupling=False,
            # Robot-body gravity is already canceled by the Newton rigid-body
            # gravcomp setting. Enabling this would compensate it twice.
            gravity_compensation=False,
            motion_stiffness_task=_OSC_STIFFNESS,
            motion_damping_ratio_task=damping_ratio,
            nullspace_control="none",
        ),
        nullspace_joint_pos_target="none",
        position_scale=action_scale,
        orientation_scale=action_scale,
    )


def _enable_task_space_diagnostics(env_cfg) -> None:
    """Enable backend-neutral search and tracking diagnostics."""
    env_cfg.log_task_space_action_metrics = True
    env_cfg.task_space_diagnostic_tcp_offset = _TCP_15CM_OFFSET


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
            "offset": _LEGACY_TCP_OBSERVATION_OFFSET,
        },
    )


def _use_pose_6d_actor_observation(env_cfg, tool_offset: tuple[float, float, float]) -> None:
    """Use matching tool and socket poses with continuous 6D rotations."""
    env_cfg.observations.policy.joint_pos = None
    env_cfg.observations.policy.joint_vel = None
    env_cfg.observations.policy.socket_quat = None
    env_cfg.observations.policy.tool_pos = ObsTerm(
        func=deploy_mdp.body_pos_w_with_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["flange"]),
            "offset": tool_offset,
        },
    )
    env_cfg.observations.policy.tool_rot_6d = ObsTerm(
        func=deploy_mdp.body_rot_6d_w,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["flange"])},
    )
    env_cfg.observations.policy.socket_rot_6d = ObsTerm(
        func=deploy_mdp.rigid_object_rot_6d_w,
        params={"asset_cfg": SceneEntityCfg("dp_socket")},
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
        _enable_task_space_diagnostics(self)


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
        _enable_task_space_diagnostics(self)


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
        _enable_task_space_diagnostics(self)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangeObsEnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg
):
    """Calibrated, randomized Newton IK task whose actor observes flange pose."""

    def __post_init__(self):
        super().__post_init__()

        _use_flange_pose_actor_observation(self)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DEnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg
):
    """Calibrated, randomized Newton IK task with a flange-origin pose-6D actor."""

    def __post_init__(self):
        super().__post_init__()

        _use_pose_6d_actor_observation(self, (0.0, 0.0, 0.0))


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcpObsEnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg
):
    """Calibrated, randomized Newton IK task with a TCP-offset actor observation."""

    def __post_init__(self):
        super().__post_init__()

        _use_tcp_pose_actor_observation(self)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DEnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg
):
    """Calibrated Newton task that observes the 150 mm TCP and controls the flange."""

    def __post_init__(self):
        super().__post_init__()

        _use_pose_6d_actor_observation(self, _TCP_15CM_OFFSET)


@configclass
class Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonTcp15cmPose6DEnvCfg(
    joint_pos_env_cfg.Rizon4sGravDisplayportInsertionDomainRandomizedNoJointVelEnvCfg
):
    """Nominal Newton IK task that observes and controls the 150 mm TCP."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = _ik_action(_TCP_15CM_OFFSET)
        _enable_task_space_diagnostics(self)
        _use_pose_6d_actor_observation(self, _TCP_15CM_OFFSET)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmPose6DEnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DEnvCfg
):
    """Calibrated Newton IK task that observes and controls the 150 mm TCP."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action.objectives[0].body_offset_pos = _TCP_15CM_OFFSET


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DScale015EnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DEnvCfg
):
    """Flange pose-6D Newton IK task with a 0.015 relative action scale."""

    def __post_init__(self):
        super().__post_init__()
        _set_ik_action_scale(self, 0.015)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DScale025EnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DEnvCfg
):
    """Flange pose-6D Newton IK task with a 0.025 relative action scale."""

    def __post_init__(self):
        super().__post_init__()
        _set_ik_action_scale(self, 0.025)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DActionClip1EnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DEnvCfg
):
    """Flange pose-6D Newton IK task with the full normalized action range."""

    def __post_init__(self):
        super().__post_init__()
        _set_action_clip(self, 1.0)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DScale015EnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DEnvCfg
):
    """TCP pose-6D Newton IK task with flange action scale 0.015."""

    def __post_init__(self):
        super().__post_init__()
        _set_ik_action_scale(self, 0.015)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DScale025EnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DEnvCfg
):
    """TCP pose-6D Newton IK task with flange action scale 0.025."""

    def __post_init__(self):
        super().__post_init__()
        _set_ik_action_scale(self, 0.025)


def _configure_osc_control(env_cfg) -> None:
    """Configure direct operational-space effort control for the arm."""
    env_cfg.actions.arm_action = _flange_osc_action()
    _enable_task_space_diagnostics(env_cfg)
    env_cfg.events.randomize_arm_pd_gains = None
    for actuator_name in ("shoulder", "elbow", "wrist"):
        env_cfg.scene.robot.actuators[actuator_name].stiffness = 0.0
        env_cfg.scene.robot.actuators[actuator_name].damping = 0.0


@configclass
class Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCEnvCfg(
    joint_pos_env_cfg.Rizon4sGravDisplayportInsertionDomainRandomizedNoJointVelEnvCfg
):
    """Nominal-kinematics Newton task using flange operational-space control."""

    def __post_init__(self):
        super().__post_init__()
        _configure_osc_control(self)


@configclass
class Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCFlangePose6DEnvCfg(
    Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCEnvCfg
):
    """Nominal-kinematics Newton OSC task with flange pose observations."""

    def __post_init__(self):
        super().__post_init__()
        _use_pose_6d_actor_observation(self, (0.0, 0.0, 0.0))


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCEnvCfg(
    joint_pos_env_cfg.Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNoJointVelEnvCfg
):
    """Calibrated Newton task using compliant flange operational-space control."""

    def __post_init__(self):
        super().__post_init__()
        _configure_osc_control(self)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DEnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCEnvCfg
):
    """Compliant Newton OSC task with flange-origin pose-6D observations."""

    def __post_init__(self):
        super().__post_init__()
        _use_pose_6d_actor_observation(self, (0.0, 0.0, 0.0))


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale010EnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DEnvCfg
):
    """Flange operational-space task with a 0.010 relative action scale."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.position_scale = 0.01
        self.actions.arm_action.orientation_scale = 0.01


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DActionClip1EnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DEnvCfg
):
    """Flange OSC task with the full normalized action range."""

    def __post_init__(self):
        super().__post_init__()
        _set_action_clip(self, 1.0)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale010ActionClip1EnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DActionClip1EnvCfg
):
    """Full-range flange OSC task with a 0.010 relative action scale."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.position_scale = 0.01
        self.actions.arm_action.orientation_scale = 0.01


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale0125ActionClip1EnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DActionClip1EnvCfg
):
    """Full-range flange OSC task with a 0.0125 relative action scale."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.position_scale = 0.0125
        self.actions.arm_action.orientation_scale = 0.0125


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale025ActionClip1EnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DActionClip1EnvCfg
):
    """Full-range flange OSC task with a 0.025 relative action scale."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.position_scale = 0.025
        self.actions.arm_action.orientation_scale = 0.025


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCInertialFlangePose6DEnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DEnvCfg
):
    """Compatibility task selecting the inertia-decoupled OSC default."""

    def __post_init__(self):
        super().__post_init__()
        controller_cfg = self.actions.arm_action.controller_cfg
        controller_cfg.inertial_dynamics_decoupling = True
        controller_cfg.motion_damping_ratio_task = _OSC_DAMPING_RATIO


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCTcp15cmObsPose6DEnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCEnvCfg
):
    """Compliant Newton OSC task with 150 mm TCP pose-6D observations."""

    def __post_init__(self):
        super().__post_init__()
        _use_pose_6d_actor_observation(self, _TCP_15CM_OFFSET)


@configclass
class Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCTcp15cmPose6DEnvCfg(
    Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCEnvCfg
):
    """Nominal Newton OSC task that observes and controls the 150 mm TCP."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action.body_offset.pos = _TCP_15CM_OFFSET
        _use_pose_6d_actor_observation(self, _TCP_15CM_OFFSET)


@configclass
class Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCTcp15cmPose6DEnvCfg(
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCTcp15cmObsPose6DEnvCfg
):
    """Calibrated Newton OSC task that observes and controls the 150 mm TCP."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action.body_offset.pos = _TCP_15CM_OFFSET
