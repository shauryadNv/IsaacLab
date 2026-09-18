# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isolated DisplayPort configuration for the geometric-fabric experiment."""

from __future__ import annotations

import math

from isaaclab_physx.physics import PhysxCfg

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.deploy.mdp as mdp
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    PLUG_GOAL_ROT,
    PLUG_INSERTION_OFFSET,
    SOCKET_INSERTION_OFFSET,
    _quat_apply_tuple,
)

from . import fabric_terminations
from .fabric_actions_cfg import DisplayportFabricActionCfg
from .fabric_observations import fabric_joint_pos, fabric_joint_vel
from .joint_fabric_controller import JointFabricControllerCfg
from .task_space_env_cfg import (
    _ARM_JOINTS,
    _TCP_OFFSET,
    Rizon4sTaskSpaceDisplayportInsertionEnvCfg,
)

_FABRIC_GRASP_OFFSET = (0.0025, 0.0, -0.1875)
_FABRIC_TCP_FROM_PLUG = tuple(
    grasp - plug + tcp
    for grasp, plug, tcp in zip(_FABRIC_GRASP_OFFSET, PLUG_INSERTION_OFFSET, _TCP_OFFSET, strict=True)
)
_FABRIC_SOCKET_TCP_GOAL_OFFSET = _quat_apply_tuple(PLUG_GOAL_ROT, _FABRIC_TCP_FROM_PLUG)


@configclass
class FabricObservationsCfg:
    """Actor and asymmetric-critic observations for fabric policy training."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations in a fixed 46-element order."""

        eef_pos = ObsTerm(
            func=mdp.eef_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "flange", "offset": _TCP_OFFSET},
        )
        eef_rot_6d = ObsTerm(
            func=mdp.eef_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "flange"},
        )
        socket_kp_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket"), "offset": SOCKET_INSERTION_OFFSET},
        )
        socket_kp_rot_6d = ObsTerm(
            func=mdp.rigid_object_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS, preserve_order=True)},
        )
        fabric_joint_pos = ObsTerm(func=fabric_joint_pos, params={"action_name": "arm_action"})
        fabric_joint_vel = ObsTerm(func=fabric_joint_vel, params={"action_name": "arm_action"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations in a fixed 46-element order."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS, preserve_order=True)},
        )
        fabric_joint_pos = ObsTerm(func=fabric_joint_pos, params={"action_name": "arm_action"})
        fabric_joint_vel = ObsTerm(func=fabric_joint_vel, params={"action_name": "arm_action"})
        socket_kp_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket"), "offset": SOCKET_INSERTION_OFFSET},
        )
        socket_kp_rot_6d = ObsTerm(
            func=mdp.rigid_object_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )
        plug_kp_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_plug"), "offset": PLUG_INSERTION_OFFSET},
        )
        plug_kp_rot_6d = ObsTerm(
            func=mdp.rigid_object_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("dp_plug")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Rizon4sFabricDisplayportInsertionEnvCfg(Rizon4sTaskSpaceDisplayportInsertionEnvCfg):
    """PhysX-only 30/60/240 Hz DisplayPort geometric-fabric experiment.

    Both the actor socket observation and controller geometry read the randomized
    simulation ``dp_socket`` pose directly. This is ground truth, not a noisy or
    delayed perception estimate, and must not be described as sensor-matched.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.observations = FabricObservationsCfg()
        self.terminations.fabric_faulted = DoneTerm(
            func=fabric_terminations.fabric_faulted,
            params={"action_name": "arm_action"},
        )

        self.actions.arm_action = DisplayportFabricActionCfg(
            asset_name="robot",
            joint_names=list(_ARM_JOINTS),
            body_name="flange",
            socket_asset_name="dp_socket",
            socket_insertion_offset=tuple(SOCKET_INSERTION_OFFSET),
            socket_tcp_goal_offset=_FABRIC_SOCKET_TCP_GOAL_OFFSET,
            tcp_offset=tuple(_TCP_OFFSET),
            fabric_cfg=JointFabricControllerCfg(dt=1.0 / 60.0),
            use_inertial_feedforward=True,
            use_gravity_compensation=True,
            expected_physics_dt=1.0 / 240.0,
            expected_policy_decimation=8,
        )

        # The artificial state is tracked by explicit, effort-limited joint PD.
        # These are the nominal Grav asset's existing arm gains and hardware limits.
        self.scene.robot.actuators["shoulder"] = IdealPDActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            actuator_effort_limit=123.0,
            actuator_velocity_limit=2.094,
            joint_effort_limit=123.0,
            joint_velocity_limit=2.094,
            stiffness=1320.0,
            damping=72.0,
            friction=0.0,
            armature=0.0,
        )
        self.scene.robot.actuators["elbow"] = IdealPDActuatorCfg(
            joint_names_expr=["joint[3-4]"],
            actuator_effort_limit=64.0,
            actuator_velocity_limit=2.443,
            joint_effort_limit=64.0,
            joint_velocity_limit=2.443,
            stiffness=600.0,
            damping=35.0,
            friction=0.0,
            armature=0.0,
        )
        self.scene.robot.actuators["wrist"] = IdealPDActuatorCfg(
            joint_names_expr=["joint[5-7]"],
            actuator_effort_limit=39.0,
            actuator_velocity_limit=4.887,
            joint_effort_limit=39.0,
            joint_velocity_limit=4.887,
            stiffness=216.0,
            damping=29.0,
            friction=0.0,
            armature=0.0,
        )

        # The explicit tracker consumes g(q), so physical gravity must remain on.
        self.scene.robot.spawn.rigid_props.disable_gravity = False

    def validate_config(self) -> None:
        """Reject any rate, backend, action, or gravity combination not validated here."""
        if not isinstance(self.sim.physics, PhysxCfg):
            raise ValueError("The DisplayPort fabric experiment currently supports only PhysX.")
        if not isinstance(self.actions.arm_action, DisplayportFabricActionCfg):
            raise TypeError("The fabric environment requires DisplayportFabricActionCfg.")
        action = self.actions.arm_action
        if action.joint_names != _ARM_JOINTS:
            raise ValueError(f"Fabric arm joints must be ordered exactly as {_ARM_JOINTS}.")
        if not math.isclose(self.sim.dt, action.expected_physics_dt, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Fabric physics dt does not match the action contract.")
        if self.decimation != action.expected_policy_decimation:
            raise ValueError("Fabric policy decimation does not match the action contract.")
        fabric_interval = action.fabric_cfg.dt / self.sim.dt
        rounded_interval = round(fabric_interval)
        if rounded_interval <= 0:
            raise ValueError("Fabric dt must be at least one simulation step.")
        if not math.isclose(fabric_interval, rounded_interval, rel_tol=0.0, abs_tol=1.0e-10):
            raise ValueError("Fabric dt must be an integer multiple of simulation dt.")
        if self.decimation // rounded_interval != 2 or self.decimation % rounded_interval != 0:
            raise ValueError("Every policy step must contain exactly two fabric updates.")
        if self.scene.robot.spawn.rigid_props.disable_gravity:
            raise ValueError("Fabric tracking requires physical gravity to be enabled.")
        if not action.use_gravity_compensation:
            raise ValueError("Fabric tracking must compensate enabled gravity exactly once.")
        if len(set(action.tracker_actuator_names)) != len(action.tracker_actuator_names):
            raise ValueError("Fabric tracker actuator group names must be unique.")
        for group_name in action.tracker_actuator_names:
            if group_name not in self.scene.robot.actuators:
                raise ValueError(f"Fabric tracker actuator group '{group_name}' does not exist.")
            if not isinstance(self.scene.robot.actuators[group_name], IdealPDActuatorCfg):
                raise TypeError(f"Fabric arm actuator '{group_name}' must be explicit IdealPDActuatorCfg.")


@configclass
class Rizon4sFabricDisplayportInsertionEnvCfg_PLAY(Rizon4sFabricDisplayportInsertionEnvCfg):
    """Small-scene play configuration for the geometric-fabric experiment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
