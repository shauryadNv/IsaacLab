# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-space DisplayPort insertion environment for Flexiv Rizon 4S + Grav gripper.

Mirrors the GB300 ``config/rizon_4s/joint_pos_env_cfg.py`` but builds on the
DisplayPort base env. Relative joint-position control of the 7-DoF arm; the
plug is grasped at reset and the goal is the verified seated mate.
"""

import math

import torch
from isaaclab_physx.sim.schemas import (
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.deploy.mdp as mdp
import isaaclab_tasks.contrib.deploy.mdp.terminations as cable_terminations
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    PLUG_GOAL_ROT,
    PLUG_INSERTION_OFFSET,
    SOCKET_INSERTION_OFFSET,
    DisplayportInsertionEnvCfg,
    compute_plug_pose,
    compute_socket_root,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.events import ResetPlugAtGoalCurriculum
from isaaclab_tasks.utils import PresetCfg, preset

# ---------------------------------------------------------------------------
# Flexiv workspace layout (DisplayPort insertion station)
# ---------------------------------------------------------------------------
# Insertion(mate) point in the robot workspace, socket orientation (opening up,
# matching the verified drop-test seated pose), and the vertical clearance the
# plug starts above the socket.
#
# The station pose matches the validated PhysX training configuration; only the
# collision backend and task action adapter vary between presets.
_GEOMETRY_POS = (0.475, 0.125, 0.06)
_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)  # opening faces +Z (top-down insertion)
_PLUG_CLEARANCE_Z = 0.068
_INSERTION_LENGTH = 0.011


def _newton_actuator_gain(default: float, newton: float) -> PresetCfg:
    """Select the validated actuator gain for each physics backend."""
    return preset(
        default=default,
        newton_mjwarp=newton,
        newton_sdf=newton,
        newton_hydroelastic=newton,
    )


_SOCKET_ROOT = compute_socket_root(_GEOMETRY_POS, _SOCKET_ROT)
_PLUG_ROOT, _PLUG_ROT = compute_plug_pose(
    _GEOMETRY_POS,
    _SOCKET_ROT,
    z_clearance=_PLUG_CLEARANCE_Z,
)

##
# Pre-defined configs
##
from isaaclab_assets import FLEXIV_RIZON4S_GRAV_GRIPPER_CFG, ISAACLAB_ASSETS_DATA_DIR  # isort: skip


_CALIBRATED_RIZON4S_GRAV_USD = (
    f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Flexiv/Rizon4s/Rizon4s-063459_with_Grav_calibrated_kinematics.usd"
)


##
# Gripper-specific helper functions
##


_GRAV_GRIPPER_MIMIC_GEARING = {
    "finger_joint": 1.0,
    "left_inner_knuckle_joint": 1.0,
    "right_inner_knuckle_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "left_outer_finger_joint": -1.0,
    "right_outer_finger_joint": -1.0,
}


def set_finger_joint_pos_grav(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
    joint_name_to_idx: dict[str, int] | None = None,
):
    """Set Grav gripper joint positions by joint name."""
    if joint_name_to_idx is None:
        raise ValueError("set_finger_joint_pos_grav requires 'joint_name_to_idx'.")

    missing = [name for name in _GRAV_GRIPPER_MIMIC_GEARING if name not in joint_name_to_idx]
    if missing:
        raise ValueError(f"Grav gripper joints not found on the robot: {missing}.")

    for idx in reset_ind_joint_pos:
        for joint_name, gearing in _GRAV_GRIPPER_MIMIC_GEARING.items():
            joint_pos[idx, joint_name_to_idx[joint_name]] = gearing * finger_joint_position


##
# Environment configuration
##


@configclass
class EventCfg:
    """Configuration for events."""

    plug_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("dp_plug", body_names=".*"),
            # The overmold is pinched by the fingertip pads. Near-zero friction
            # lets the 30 g plug slide through the gripper before insertion.
            "static_friction_range": (3.0, 3.0),
            "dynamic_friction_range": (3.0, 3.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    socket_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("dp_socket", body_names=".*"),
            "static_friction_range": (0.001, 0.001),
            "dynamic_friction_range": (0.001, 0.001),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_socket_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
                "z": [-0.02, 0.02],
                "roll": [-math.radians(2.0), math.radians(2.0)],
                "pitch": [-math.radians(2.0), math.radians(2.0)],
                "yaw": [-math.radians(2.0), math.radians(2.0)],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("dp_socket"),
        },
    )

    reset_plug_curriculum = EventTerm(
        func=ResetPlugAtGoalCurriculum,
        mode="reset",
        params={
            "plug_cfg": SceneEntityCfg("dp_plug"),
            "socket_cfg": SceneEntityCfg("dp_socket"),
            "at_goal_prob": 0.8,
            "at_goal_prob_final": 0.0,
            "anneal_start_iter": 0.0,
            "anneal_end_iter": 500.0,
            "num_steps_per_env": 512,
            "insertion_axis": [1.0, 0.0, 0.0],
            "insertion_length": _INSERTION_LENGTH,
            "at_goal_depth_range": [0.0, 0.015],
            "approach_depth_range": [0.02, 0.06],
            "socket_insertion_offset": SOCKET_INSERTION_OFFSET,
            "plug_insertion_offset": PLUG_INSERTION_OFFSET,
            "goal_rot": list(PLUG_GOAL_ROT),
            "normal_pose_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [0.0, 0.0],
            },
        },
    )

    set_robot_to_grasp_pose = EventTerm(
        func=mdp.set_robot_to_object_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "pos_randomization_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "target_object_name": "dp_plug",
            "grasp_offset": [0.0, 0.0, 0.0],
        },
    )


@configclass
class TerminationsCfg:
    """Configuration for termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    plug_dropped = DoneTerm(
        func=cable_terminations.reset_when_plug_dropped,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "plug_asset_cfg": SceneEntityCfg("dp_plug"),
            "distance_threshold": 0.15,
            "end_effector_body_name": "link7",
            "grasp_offset": [0.0, 0.0, 0.0],
            "grasp_rot_offset": [0.0, 0.0, 0.0, 1.0],
        },
    )

    plug_orientation_exceeded = DoneTerm(
        func=cable_terminations.reset_when_plug_orientation_exceeded,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "plug_asset_cfg": SceneEntityCfg("dp_plug"),
            "roll_threshold_deg": 15.0,
            "pitch_threshold_deg": 15.0,
            "yaw_threshold_deg": 180.0,
            "end_effector_body_name": "link7",
            "grasp_rot_offset": [0.0, 0.0, 0.0, 1.0],
        },
    )


@configclass
class Rizon4sGravDisplayportInsertionEnvCfg(DisplayportInsertionEnvCfg):
    """Flexiv Rizon 4s + Grav gripper DisplayPort insertion (joint-space)."""

    def __post_init__(self):
        super().__post_init__()

        self.end_effector_body_name = "flange"
        self.num_arm_joints = 7
        # Flange position in the DisplayPort plug's local frame. This places
        # the midpoint of the fingertip pads behind the metal connector on the
        # plug overmold, leaving the connector exposed for insertion.
        self.grasp_offset = [0.0025, 0.0, -0.1875]
        # Identity: the target EEF orientation equals the plug orientation.
        self.grasp_rot_offset = [0.0, 0.0, 0.0, 1.0]
        self.gripper_joint_setter_func = set_finger_joint_pos_grav

        self.plug_orientation_roll_threshold_deg = 15.0
        self.plug_orientation_pitch_threshold_deg = 15.0
        self.plug_orientation_yaw_threshold_deg = 180.0

        # Observation configuration for Rizon 4s arm joints only
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        self.events = EventCfg()

        self.terminations = TerminationsCfg()

        self.terminations.plug_orientation_exceeded.params["roll_threshold_deg"] = (
            self.plug_orientation_roll_threshold_deg
        )
        self.terminations.plug_orientation_exceeded.params["pitch_threshold_deg"] = (
            self.plug_orientation_pitch_threshold_deg
        )
        self.terminations.plug_orientation_exceeded.params["yaw_threshold_deg"] = (
            self.plug_orientation_yaw_threshold_deg
        )

        self.joint_action_scale = 0.025
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            scale=self.joint_action_scale,
            use_zero_offset=True,
        )

        self.scene.robot = FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.spawn.replace(
                joint_drive_props=preset(
                    default=None,
                    newton_mjwarp=sim_utils.MujocoJointDrivePropertiesCfg(actuatorgravcomp=False),
                    newton_sdf=sim_utils.MujocoJointDrivePropertiesCfg(actuatorgravcomp=False),
                    newton_hydroelastic=sim_utils.MujocoJointDrivePropertiesCfg(actuatorgravcomp=False),
                ),
                rigid_props=preset(
                    default=PhysxRigidBodyPropertiesCfg(
                        disable_gravity=True,
                        max_depenetration_velocity=5.0,
                        linear_damping=0.0,
                        angular_damping=0.0,
                        max_linear_velocity=1000.0,
                        max_angular_velocity=3666.0,
                        enable_gyroscopic_forces=True,
                        solver_position_iteration_count=4,
                        solver_velocity_iteration_count=1,
                        max_contact_impulse=1e32,
                    ),
                    newton_mjwarp=sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
                    newton_sdf=sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
                    newton_hydroelastic=sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
                ),
                articulation_props=PhysxArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
                collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "joint1": math.radians(32.44),
                    "joint2": math.radians(-16.71),
                    "joint3": math.radians(-5.69),
                    "joint4": math.radians(128.38),
                    "joint5": math.radians(6.74),
                    "joint6": math.radians(55.95),
                    "joint7": math.radians(111.54),
                },
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        # PhysX excludes each robot body from gravity. Newton MJWarp applies full passive body
        # gravity compensation without routing it through actuator force limits. The plug remains
        # under world gravity in both backends.
        self.scene.robot.actuators["shoulder"].stiffness = _newton_actuator_gain(1320.0, 6000.0)
        self.scene.robot.actuators["shoulder"].damping = _newton_actuator_gain(72.0, 108.5)
        self.scene.robot.actuators["elbow"].stiffness = _newton_actuator_gain(600.0, 4200.0)
        self.scene.robot.actuators["elbow"].damping = _newton_actuator_gain(35.0, 90.7)
        self.scene.robot.actuators["wrist"].stiffness = _newton_actuator_gain(216.0, 1500.0)
        self.scene.robot.actuators["wrist"].damping = _newton_actuator_gain(29.0, 54.2)

        # Newton needs physical PD targets on the driven and follower joints so
        # the Grav linkage remains closed around the plug under contact load.
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=2.0,
            stiffness=2000.0,
            damping=10.0,
            friction=0.0,
            armature=0.1,
        )

        self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_knuckle_joint", ".*_outer_finger_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.0,
            stiffness=2000.0,
            damping=10.0,
            friction=0.0,
            armature=0.05,
        )

        self.scene.dp_socket.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_SOCKET_ROOT,
            rot=_SOCKET_ROT,
        )
        self.scene.dp_plug.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_PLUG_ROOT,
            rot=_PLUG_ROT,
        )

        # Grav gripper widths [rad]. The calibrated hold width closes the pad
        # gap around the 11.9 mm plug thickness without initial over-closure.
        self.hand_grasp_width = 0.3
        self.hand_hold_width = -0.1
        self.hand_close_width = self.hand_hold_width

        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["grasp_offset"] = self.grasp_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func
        self.events.set_robot_to_grasp_pose.params["align_object_to_grasp_at_reset"] = True
        self.events.set_robot_to_grasp_pose.params["max_iterations"] = 150

        self.terminations.plug_dropped.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_dropped.params["grasp_offset"] = self.grasp_offset
        self.terminations.plug_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        self.terminations.plug_orientation_exceeded.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_orientation_exceeded.params["grasp_rot_offset"] = self.grasp_rot_offset

    def play_mode(self) -> None:
        """Configure deterministic approach-band resets for policy playback."""
        super().play_mode()
        self.events.reset_plug_curriculum.params["at_goal_prob"] = 0.0
        self.events.reset_plug_curriculum.params["at_goal_prob_final"] = 0.0


@configclass
class Rizon4sGravDisplayportInsertionNoJointVelEnvCfg(Rizon4sGravDisplayportInsertionEnvCfg):
    """Joint-space variant that hides joint velocity from the actor (policy).

    Identical to :class:`Rizon4sGravDisplayportInsertionEnvCfg` except the actor
    observation drops joint velocity (actor sees joint positions + socket pose
    only). The critic keeps joint velocity as privileged information, so the
    value function is unaffected. Useful for testing velocity-free deployment
    (e.g. when reliable joint-velocity estimates are not available on the real
    robot).
    """

    def __post_init__(self):
        super().__post_init__()
        # Remove joint velocity from the actor group; setting a term to None
        # disables it in the manager-based ObservationManager. The critic group
        # still includes joint_vel.
        self.observations.policy.joint_vel = None


@configclass
class Rizon4sGravDisplayportInsertionCalibratedNoJointVelEnvCfg(Rizon4sGravDisplayportInsertionNoJointVelEnvCfg):
    """Velocity-free DisplayPort task using calibrated Rizon 4S kinematics."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = _CALIBRATED_RIZON4S_GRAV_USD
