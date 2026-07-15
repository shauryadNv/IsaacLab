# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-space DisplayPort insertion environment for UR10e + Robotiq gripper.

Mirrors ``config/displayport_rizon_4s/joint_pos_env_cfg.py`` for task setup
(curriculum, socket randomization, rewards) and ``gear_assembly/config/ur_10e``
for robot-specific parameters (6-DoF arm, wrist_3_link EE, Robotiq grippers).
"""

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.deploy.mdp.terminations as cable_terminations
from isaaclab_tasks.manager_based.manipulation.deploy.cable_insertion.displayport_insertion_env_cfg import (
    PLUG_GOAL_ROT,
    PLUG_INSERTION_OFFSET,
    SOCKET_INSERTION_OFFSET,
    DisplayportInsertionEnvCfg,
    compute_plug_pose,
    compute_socket_root,
)


##
# Gripper-specific helper functions
##
# Copied from ``gear_assembly/config/ur_10e/joint_pos_env_cfg.py`` so this task
# does not import the gear-assembly package (avoids coupling to its module chain).


def set_finger_joint_pos_robotiq_2f140(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
):
    """Set finger joint positions for Robotiq 2F-140 gripper.

    Joint structure (8 joints):
        ``[finger_joint, finger_joint, outer x2, inner_finger x2, pad x2]``
    """
    for idx in reset_ind_joint_pos:
        if len(finger_joints) < 8:
            raise ValueError(f"2F-140 gripper requires at least 8 finger joints, got {len(finger_joints)}")

        joint_pos[idx, finger_joints[0]] = finger_joint_position
        joint_pos[idx, finger_joints[1]] = finger_joint_position

        joint_pos[idx, finger_joints[2]] = 0
        joint_pos[idx, finger_joints[3]] = 0

        joint_pos[idx, finger_joints[4]] = -finger_joint_position
        joint_pos[idx, finger_joints[5]] = -finger_joint_position

        joint_pos[idx, finger_joints[6]] = finger_joint_position
        joint_pos[idx, finger_joints[7]] = finger_joint_position


def set_finger_joint_pos_robotiq_2f85(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
):
    """Set finger joint positions for Robotiq 2F-85 gripper.

    Joint structure (6 joints):
        ``[finger_joint, finger_joint, inner_finger x2, inner_finger_knuckle x2]``
    """
    for idx in reset_ind_joint_pos:
        if len(finger_joints) < 6:
            raise ValueError(f"2F-85 gripper requires at least 6 finger joints, got {len(finger_joints)}")

        joint_pos[idx, finger_joints[0]] = finger_joint_position
        joint_pos[idx, finger_joints[1]] = finger_joint_position
        joint_pos[idx, finger_joints[2]] = -finger_joint_position
        joint_pos[idx, finger_joints[3]] = finger_joint_position
        joint_pos[idx, finger_joints[4]] = -finger_joint_position
        joint_pos[idx, finger_joints[5]] = -finger_joint_position

# ---------------------------------------------------------------------------
# UR10e workspace layout (DisplayPort insertion station)
# ---------------------------------------------------------------------------
# Measured on the real UR10e station with plug seated in socket and the robot
# holding the plug. Pose is read from the teach pendant with **TCP = 0** (tool
# flange / tool0). Position in metres; orientation is a UR **rotation vector**
# (axis-angle): direction = axis, magnitude = angle [rad].
_MEASURED_TCP_POS = (0.95495, 0.02454, 0.27303)
_MEASURED_TCP_ROTVEC_RAD = (2.155, 2.3, -0.033)

# Flange -> plug mate tip along tool +Z when the plug is fully seated.
# 2F-140 body height (232.8 mm) + e-Series I/O coupling (~13.5 mm) + plug tip
# past closed fingertips (~15 mm). CALIBRATE: tweak if the socket lands short/long.
_TCP_TO_PLUG_MATE_TIP_M = 0.0135 + 0.2328


def _rotvec_to_quat(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    """Convert a rotation vector (axis-angle, rad) to quaternion ``(x, y, z, w)``."""
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    s = math.sin(theta * 0.5) / theta
    return (rx * s, ry * s, rz * s, math.cos(theta * 0.5))


def _quat_to_euler_xyz_deg(q: tuple[float, float, float, float]) -> list[float]:
    """Convert quaternion ``(x, y, z, w)`` to XYZ Euler angles in degrees."""
    x, y, z, w = q
    sr_cp = 2.0 * (w * x + y * z)
    cr_cp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sr_cp, cr_cp)
    sp = 2.0 * (w * y - z * x)
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    sy_cp = 2.0 * (w * z + x * y)
    cy_cp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sy_cp, cy_cp)
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _quat_mul(q1, q2):
    """Hamilton product of two quaternions ``(x, y, z, w)``."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_rotate_vec(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate vector ``v`` by quaternion ``q`` (x, y, z, w)."""
    x, y, z, w = q
    vx, vy, vz = v
    return (
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - z * w) * vy + 2 * (x * z + y * w) * vz,
        2 * (x * y + z * w) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - x * w) * vz,
        2 * (x * z - y * w) * vx + 2 * (y * z + x * w) * vy + (1 - 2 * (x * x + y * y)) * vz,
    )


_MEASURED_TCP_ROT_RAW = _rotvec_to_quat(*_MEASURED_TCP_ROTVEC_RAD)
# At seated pose the plug mate tip coincides with the socket mate point. With
# TCP at the flange, advance along tool +Z by the flange-to-plug-tip distance.
_TCP_TO_SOCKET_OFFSET = _quat_rotate_vec(_MEASURED_TCP_ROT_RAW, (0.0, 0.0, _TCP_TO_PLUG_MATE_TIP_M))
# Socket mate point (insertion geometry) in the UR ``base`` frame. This is what
# the ``socket_pos`` observation reports.
#
# The socket sits ~246 mm below the measured TCP along the tool axis. If the tool
# were perfectly vertical only z would drop; because the real tool tilts ~1.3 deg
# the full advance also shifts x/y by ~-5 mm / -2 mm.
#
# _SOCKET_APPLY_TILT_XY selects how the mate x/y are set:
#   * False (default): set the mate x/y DIRECTLY to the measured TCP x/y (Flexiv
#     style -- the socket mean is exactly the number you set). Drops the ~5 mm
#     lateral tilt correction, i.e. assumes a vertical tool.
#   * True: apply the full tilted tool-axis advance (most sim2real-faithful; the
#     mate x/y come out ~-5 mm / -2 mm from the TCP x/y).
# z always uses the (tool-axis) vertical drop so the socket lands on the table.
_SOCKET_APPLY_TILT_XY = False
_MEASURED_SOCKET_GEOMETRY_POS = (
    _MEASURED_TCP_POS[0] + (_TCP_TO_SOCKET_OFFSET[0] if _SOCKET_APPLY_TILT_XY else 0.0),
    _MEASURED_TCP_POS[1] + (_TCP_TO_SOCKET_OFFSET[1] if _SOCKET_APPLY_TILT_XY else 0.0),
    _MEASURED_TCP_POS[2] + _TCP_TO_SOCKET_OFFSET[2],
)
# Socket USD asset-frame correction. The measured TCP rotation vector reflects
# the tool/gripper orientation at the seated pose. The socket USD is authored
# such that "cavity up" (local +X -> world +Z) needs an extra +90 deg about
# world X relative to that tool orientation.
_SOCKET_FRAME_CORRECTION = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))  # Rx(+90)

# Extra 180 deg yaw of the socket about the world/base vertical (Z) axis. Spins
# the socket in place (cavity stays facing up) so the keyed insertion
# orientation -- i.e. which way the socket/plug faces -- flips to match the real
# fixture. Applied here so it propagates to training (_SOCKET_ROT), the plug goal
# (derived from the socket rot), and ROS (_MEASURED_SOCKET_EULER_DEG). Set False
# to revert.
_SOCKET_EXTRA_YAW_RZ180 = True
_YAW_RZ180 = (0.0, 0.0, 1.0, 0.0)  # 180 deg about Z, (x, y, z, w)

_MEASURED_SOCKET_ROT = _quat_mul(_SOCKET_FRAME_CORRECTION, _MEASURED_TCP_ROT_RAW)
if _SOCKET_EXTRA_YAW_RZ180:
    _MEASURED_SOCKET_ROT = _quat_mul(_YAW_RZ180, _MEASURED_SOCKET_ROT)
_MEASURED_SOCKET_EULER_DEG = _quat_to_euler_xyz_deg(_MEASURED_SOCKET_ROT)

# 180 deg about Z: maps the UR ``base`` frame (used by the real controller and
# the measured pose) into the ``base_link`` / USD-root frame that the training
# env uses when the robot is spawned at identity.
_ROT_Z_180 = (0.0, 0.0, 1.0, 0.0)

# Training spawns the robot at identity, so the measured ``base``-frame socket
# pose is transformed into the ``base_link`` frame by a 180 deg Z rotation
# (position X/Y negated, orientation pre-multiplied by Rz(180)). This keeps the
# robot<->socket geometry identical to the ROS-deployment env, where the robot
# USD is instead rotated 180 deg about Z and the socket uses the measured pose.
_GEOMETRY_POS = (
    -_MEASURED_SOCKET_GEOMETRY_POS[0],
    -_MEASURED_SOCKET_GEOMETRY_POS[1],
    _MEASURED_SOCKET_GEOMETRY_POS[2],
)
_SOCKET_ROT = _quat_mul(_ROT_Z_180, _MEASURED_SOCKET_ROT)
_PLUG_CLEARANCE_Z = 0.068

_SOCKET_ROOT = compute_socket_root(_GEOMETRY_POS, _SOCKET_ROT)
_PLUG_ROOT, _PLUG_ROT = compute_plug_pose(
    _GEOMETRY_POS, _SOCKET_ROT, z_clearance=_PLUG_CLEARANCE_Z,
)

_INSERTION_LENGTH = 0.011

# Home joint pose measured on the real UR10e station (degrees -> rad).
_UR10E_HOME_JOINT_POS = {
    "shoulder_pan_joint": math.radians(171.39),
    "shoulder_lift_joint": math.radians(-56.66),
    "elbow_joint": math.radians(88.66),
    "wrist_1_joint": math.radians(-122.05),
    "wrist_2_joint": math.radians(-90.51),
    "wrist_3_joint": math.radians(80.87),
}

# =============================== EXPERIMENT TOGGLES ===============================
# --- EXP TOGGLES START ---
EXP_SYSID = False
# Socket placement randomization, +/- metres per axis [x, y, z] (uniform, per
# reset). +/-1 cm lateral, +/-2 cm along z.
EXP_SOCKET_POS_RANGE = [0.01, 0.01, 0.02]
EXP_SOCKET_ORN_DEG = 2.0
EXP_CURRICULUM = "anneal_80_20_500"
EXP_EQUAL_REWARD_WEIGHTS = True
# Observation noise: +/- metres of uniform noise on the socket position obs
# (sampled once per reset, held constant over the episode).
EXP_SOCKET_OBS_NOISE_M = 0.01
# Freeze the arm at its reset/grasp pose by zeroing the (relative) action scale,
# so policy actions have no effect. Useful for visually inspecting the grasp at
# launch (train or play): resets, rendering and grasp-snapping still run, but the
# robot does not move. Set back to False for real training.
EXP_FREEZE_ROBOT = False
# --- EXP TOGGLES END ---


def _exp_curriculum_params(mode: str) -> dict:
    """Map an ``EXP_CURRICULUM`` mode string to reset_plug_at_goal_curriculum params."""
    table = {
        "disabled": dict(at_goal_prob=0.0, at_goal_prob_final=None, anneal_end_iter=None),
        "fixed80": dict(at_goal_prob=0.8, at_goal_prob_final=None, anneal_end_iter=None),
        "anneal_80_0_1000": dict(at_goal_prob=0.8, at_goal_prob_final=0.0, anneal_end_iter=1000.0),
        "anneal_80_20_1000": dict(at_goal_prob=0.8, at_goal_prob_final=0.2, anneal_end_iter=1000.0),
        "anneal_80_20_500": dict(at_goal_prob=0.8, at_goal_prob_final=0.2, anneal_end_iter=500.0),
        "anneal_80_0_500": dict(at_goal_prob=0.8, at_goal_prob_final=0.0, anneal_end_iter=500.0),
    }
    if mode not in table:
        raise ValueError(f"Unknown EXP_CURRICULUM mode: {mode!r}. Options: {list(table)}")
    return table[mode]


_EXP_CURR = _exp_curriculum_params(EXP_CURRICULUM)
# =================================================================================

USE_SIM2REAL_ACTION_MODEL = EXP_SYSID
USE_SIM2REAL_PHYSICS_FREQ_HZ = 200.0
USE_SIM2REAL_DECIMATION = 4
USE_SIM2REAL_ACTION_LATENCY_S = 0.02
USE_SIM2REAL_COMMAND_VELOCITY_LIMIT = 2.0
USE_SIM2REAL_COMMAND_ACCELERATION_LIMIT = 3.0

##
# Pre-defined configs
##
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_GRIPPER_CFG, UR10e_ROBOTIQ_2F_85_CFG  # isort: skip

_UR10E_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@configclass
class EventCfg:
    """Configuration for events."""

    # Plug friction is raised to 1.0 (vs the Flexiv env's 0.001) so the Robotiq
    # pads can hold the near-frictionless DP plug by FRICTION with a gentle,
    # surface-resting grip, instead of having to squeeze tight and jam a finger
    # through it. friction_combine_mode="multiply" keeps the plug<->socket contact
    # effectively frictionless for insertion (plug 1.0 * socket 0.001 = 1e-3),
    # while the finger<->plug grip becomes 1.0 * 0.75 = 0.75 -- ample to hold the
    # 30 g plug with the pads merely touching its faces. UR/Robotiq-specific.
    plug_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("dp_plug", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
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
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
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
                "x": [-EXP_SOCKET_POS_RANGE[0], EXP_SOCKET_POS_RANGE[0]],
                "y": [-EXP_SOCKET_POS_RANGE[1], EXP_SOCKET_POS_RANGE[1]],
                "z": [-EXP_SOCKET_POS_RANGE[2], EXP_SOCKET_POS_RANGE[2]],
                "roll": [-math.radians(EXP_SOCKET_ORN_DEG), math.radians(EXP_SOCKET_ORN_DEG)],
                "pitch": [-math.radians(EXP_SOCKET_ORN_DEG), math.radians(EXP_SOCKET_ORN_DEG)],
                "yaw": [-math.radians(EXP_SOCKET_ORN_DEG), math.radians(EXP_SOCKET_ORN_DEG)],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("dp_socket"),
        },
    )

    reset_plug_curriculum = EventTerm(
        func=mdp.reset_plug_at_goal_curriculum,
        mode="reset",
        params={
            "plug_cfg": SceneEntityCfg("dp_plug"),
            "socket_cfg": SceneEntityCfg("dp_socket"),
            "at_goal_prob": _EXP_CURR["at_goal_prob"],
            "at_goal_prob_final": _EXP_CURR["at_goal_prob_final"],
            "anneal_start_iter": 0.0,
            "anneal_end_iter": _EXP_CURR["anneal_end_iter"],
            "num_steps_per_env": 512,
            "insertion_axis": [1.0, 0.0, 0.0],
            "insertion_length": _INSERTION_LENGTH,
            "at_goal_depth_range": [0.0, 0.015],
            "approach_depth_range": [0.02, 0.06],
            "socket_insertion_offset": SOCKET_INSERTION_OFFSET,
            "plug_insertion_offset": PLUG_INSERTION_OFFSET,
            "goal_rot": list(PLUG_GOAL_ROT),
            "normal_pose_range": {
                "x": [-0.00, 0.00],
                "y": [-0.00, 0.00],
                "z": [-0.00, 0.00],
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
            "end_effector_body_name": "wrist_3_link",
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
            "end_effector_body_name": "wrist_3_link",
            "grasp_rot_offset": [0.0, 0.0, 0.0, 1.0],
        },
    )


@configclass
class UR10eDisplayportInsertionEnvCfg(DisplayportInsertionEnvCfg):
    """Base UR10e DisplayPort insertion (joint-space).

    Shared robot parameters; gripper-specific subclasses configure the Robotiq
    variant, grasp offset, and finger widths.
    """

    def __post_init__(self):
        super().__post_init__()

        # Print a per-term observation breakdown (env 0) during rollout, every
        # ``debug_print_obs_interval`` steps. Read by DisplayportInsertionEnv.
        self.debug_print_obs = True
        self.debug_print_obs_interval = 100
        self.debug_print_obs_group = "policy"

        if EXP_EQUAL_REWARD_WEIGHTS:
            self.rewards.plug_socket_keypoint_tracking_exp.weight = abs(
                self.rewards.plug_socket_keypoint_tracking.weight
            )

        if USE_SIM2REAL_ACTION_MODEL:
            self.decimation = USE_SIM2REAL_DECIMATION
            self.sim.dt = 1.0 / USE_SIM2REAL_PHYSICS_FREQ_HZ
            self.sim.render_interval = self.decimation

        self.end_effector_body_name = "wrist_3_link"
        self.num_arm_joints = 6
        # Rotation from the plug frame to wrist_3_link at the real seated pose,
        # (w, x, y, z) in Isaac convention. Derived by scripts/dp_ur10e_grasp_probe.py
        # from the measured seated seed joints (a ~180 deg flip so the gripper
        # points down into the cavity-up socket). Gripper-independent (same
        # coupling/grasp), so it lives in the base class.
        self.grasp_rot_offset = [-0.023699, -0.022899, 0.702178, -0.711232]
        self.gripper_joint_setter_func = None  # set in gripper subclass

        self.plug_orientation_roll_threshold_deg = 15.0
        self.plug_orientation_pitch_threshold_deg = 15.0
        self.plug_orientation_yaw_threshold_deg = 180.0

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = list(_UR10E_ARM_JOINT_NAMES)
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = list(_UR10E_ARM_JOINT_NAMES)
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = list(_UR10E_ARM_JOINT_NAMES)
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = list(_UR10E_ARM_JOINT_NAMES)

        # Socket-position observation noise (policy group only). Base env ships it
        # at 0; enable it here (UR-local) so the policy is robust to socket-pose
        # sensing error. ResetSampledConstantNoiseModelCfg samples once per reset.
        self.observations.policy.socket_pos.noise.noise_cfg.n_min = -EXP_SOCKET_OBS_NOISE_M
        self.observations.policy.socket_pos.noise.noise_cfg.n_max = EXP_SOCKET_OBS_NOISE_M

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
        if EXP_FREEZE_ROBOT:
            # Relative action with use_zero_offset=True: target = current + scale*action.
            # scale=0 -> target = current joint pos every step -> arm holds still.
            self.joint_action_scale = 0.0
        if USE_SIM2REAL_ACTION_MODEL:
            self.actions.arm_action = mdp.ShapedDelayedRelativeJointPositionActionCfg(
                asset_name="robot",
                joint_names=_UR10E_ARM_JOINT_NAMES,
                scale=self.joint_action_scale,
                use_zero_offset=True,
                latency_s=USE_SIM2REAL_ACTION_LATENCY_S,
                command_velocity_limit=USE_SIM2REAL_COMMAND_VELOCITY_LIMIT,
                command_acceleration_limit=USE_SIM2REAL_COMMAND_ACCELERATION_LIMIT,
            )
        else:
            self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
                asset_name="robot",
                joint_names=_UR10E_ARM_JOINT_NAMES,
                scale=self.joint_action_scale,
                use_zero_offset=True,
            )

        self.scene.dp_socket.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_SOCKET_ROOT,
            rot=_SOCKET_ROT,
        )
        self.scene.dp_plug.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_PLUG_ROOT,
            rot=_PLUG_ROT,
        )

        self._configure_gripper_specific()

        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["grasp_offset"] = self.grasp_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func
        self.events.set_robot_to_grasp_pose.params["max_iterations"] = 150

        self.terminations.plug_dropped.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_dropped.params["grasp_offset"] = self.grasp_offset
        self.terminations.plug_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        self.terminations.plug_orientation_exceeded.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_orientation_exceeded.params["grasp_rot_offset"] = self.grasp_rot_offset

    def _configure_gripper_specific(self):
        """Override in subclasses to set robot asset, grasp offset, and finger widths."""
        raise NotImplementedError


@configclass
class UR10e2F140DisplayportInsertionEnvCfg(UR10eDisplayportInsertionEnvCfg):
    """UR10e + Robotiq 2F-140 DisplayPort insertion (joint-space)."""

    def _configure_gripper_specific(self):
        # wrist_3_link -> plug offset (EE frame, m). Tuned with
        # scripts/dp_ur10e_grasp_check.py, whose --gx/--gy/--gz/--hold/--close
        # sweeps report a DIRECT pad-vs-plug penetration metric (per finger) plus
        # the plug x_EE center and slip over 16 envs:
        #   z=-0.21 puts the plug body across the pad z-band so the pads grip the
        #     plug mid-body (‑0.23 grips only the top edge; deeper it drops below).
        #   x=-0.002 centers the plug BETWEEN the (symmetric) pads: settled plug
        #     x_EE = [-6, +6] mm (center 0). The old -0.008/-0.016 left the plug
        #     center at +6 mm, so ONE finger buried into it (+5..+13 mm) while the
        #     other never touched -- the "finger through the plug" artifact.
        #   y=+0.011 centers the plug along the pad width.
        # With plug friction 1.0 and hold=0.72 (below), both pads rest on the plug
        # faces at ~0.1 mm penetration and it stays caged (slip 1 mm over 16 envs).
        self.grasp_offset = [-0.002, 0.0, -0.2175]
        self.gripper_joint_setter_func = set_finger_joint_pos_robotiq_2f140

        self.scene.robot = UR10e_ROBOTIQ_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UR10e_ROBOTIQ_GRIPPER_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
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
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=dict(_UR10E_HOME_JOINT_POS),
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        self.scene.robot.actuators["gripper_finger"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=10.0,
            damping=0.05,
            friction=0.0,
            armature=0.0,
        )

        # Robotiq 2F-140: HIGHER finger_joint = tighter grip (the 4-bar linkage
        # closes the pads while the link origins spread apart). The DP plug is
        # ~12 mm on the grip axis (matches the real 12.33 mm reading while
        # holding). The pad free-gap maps as width 0.68->21 mm, 0.75->8 mm
        # (scripts/dp_ur10e_gripper_probe.py). hold=0.72 lands the pad inner faces
        # at ~+-6 mm = the centered 12 mm plug's surfaces, so the pads REST on the
        # plug (0.1 mm penetration per finger) rather than driving through it.
        # Verified with scripts/dp_ur10e_grasp_check.py: both fingers ~0 mm
        # penetration, plug centered, slip 1 mm over 16 envs. TIGHTER (0.80)
        # drives both pads through the plug body; LOOSER (<=0.70) leaves the plug
        # loose so it slips/jams to one side. Relies on plug friction 1.0 to hold.
        self.hand_grasp_width = 0.72
        self.hand_hold_width = 0.72
        self.hand_close_width = 0.74


@configclass
class UR10e2F85DisplayportInsertionEnvCfg(UR10eDisplayportInsertionEnvCfg):
    """UR10e + Robotiq 2F-85 DisplayPort insertion (joint-space)."""

    def _configure_gripper_specific(self):
        # ESTIMATE: 2F-85 fingers are ~70 mm shorter than the 2F-140, so the pad
        # zone sits ~70 mm closer to wrist_3_link: z ~= -0.21 + 0.07 = -0.14.
        # CALIBRATE with scripts/dp_ur10e_grasp_check.py pointed at a 2F-85 env
        # (sweep --gx/--gy/--gz/--hold/--close); the lateral seat differs per
        # gripper. grasp_rot_offset (base class) is gripper-independent.
        self.grasp_offset = [-0.016, -0.0086, -0.14]
        self.gripper_joint_setter_func = set_finger_joint_pos_robotiq_2f85

        self.scene.robot = UR10e_ROBOTIQ_2F_85_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UR10e_ROBOTIQ_2F_85_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
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
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=dict(_UR10E_HOME_JOINT_POS),
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        self.scene.robot.actuators["gripper_finger"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=10.0,
            damping=0.05,
            friction=0.0,
            armature=0.0,
        )
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
        )

        # Robotiq 2F-85: firm grip (plug friction is raised so friction holds it).
        # HIGHER finger_joint = tighter. CALIBRATE against a 2F-85 env; too loose
        # -> the plug rattles, too tight -> a pad penetrates the plug body.
        self.hand_grasp_width = 0.60
        self.hand_hold_width = 0.60
        self.hand_close_width = 0.64


@configclass
class UR10e2F140DisplayportInsertionEnvCfg_PLAY(UR10e2F140DisplayportInsertionEnvCfg):
    """Play configuration for UR10e 2F-140 DisplayPort insertion."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class UR10e2F85DisplayportInsertionEnvCfg_PLAY(UR10e2F85DisplayportInsertionEnvCfg):
    """Play configuration for UR10e 2F-85 DisplayPort insertion."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class UR10e2F140DisplayportInsertionNoJointVelEnvCfg(UR10e2F140DisplayportInsertionEnvCfg):
    """Joint-space variant that hides joint velocity from the actor (2F-140)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.joint_vel = None


@configclass
class UR10e2F85DisplayportInsertionNoJointVelEnvCfg(UR10e2F85DisplayportInsertionEnvCfg):
    """Joint-space variant that hides joint velocity from the actor (2F-85)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.joint_vel = None


@configclass
class UR10e2F140DisplayportInsertionNoJointVelEnvCfg_PLAY(UR10e2F140DisplayportInsertionNoJointVelEnvCfg):
    """Play configuration for the no-joint-velocity 2F-140 variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class UR10e2F85DisplayportInsertionNoJointVelEnvCfg_PLAY(UR10e2F85DisplayportInsertionNoJointVelEnvCfg):
    """Play configuration for the no-joint-velocity 2F-85 variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
