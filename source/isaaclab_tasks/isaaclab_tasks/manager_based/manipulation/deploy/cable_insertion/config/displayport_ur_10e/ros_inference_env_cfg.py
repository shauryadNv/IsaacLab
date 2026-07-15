# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-space ROS inference configuration for DisplayPort insertion with UR10e.

Inherits from the joint-space training configs and adds Isaac Manipulator
metadata fields plus deployment alignment. Mirrors
``displayport_rizon_4s/ros_inference_env_cfg.py`` for task fields and
``gear_assembly/config/ur_10e/ros_inference_env_cfg.py`` for UR mount conventions.
"""

import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.deploy.cable_insertion.displayport_insertion_env_cfg import (
    compute_plug_pose,
    compute_socket_root,
)

from .joint_pos_env_cfg import (
    UR10e2F140DisplayportInsertionEnvCfg,
    UR10e2F85DisplayportInsertionEnvCfg,
    _MEASURED_SOCKET_EULER_DEG,
    _MEASURED_SOCKET_GEOMETRY_POS,
    _MEASURED_SOCKET_ROT,
    _PLUG_CLEARANCE_Z,
)

# ---------------------------------------------------------------------------
# Deployment socket/plug positions (UR10e station)
# ---------------------------------------------------------------------------
# Use the measured real-world pose directly in the UR ``base`` frame.
_ROS_GEOMETRY_POS = _MEASURED_SOCKET_GEOMETRY_POS
_ROS_SOCKET_ROT = _MEASURED_SOCKET_ROT
_ROS_PLUG_CLEARANCE_Z = _PLUG_CLEARANCE_Z

_ROS_SOCKET_ROOT = compute_socket_root(_ROS_GEOMETRY_POS, _ROS_SOCKET_ROT)
_ROS_PLUG_ROOT, _ROS_PLUG_ROT = compute_plug_pose(
    _ROS_GEOMETRY_POS, _ROS_SOCKET_ROT, z_clearance=_ROS_PLUG_CLEARANCE_Z,
)


def _configure_ros_inference_common(cfg):
    """Set Isaac Manipulator metadata shared across UR10e gripper variants."""
    cfg.obs_order = ["arm_dof_pos", "arm_dof_vel", "socket_pos", "socket_quat"]
    cfg.policy_action_space = "joint"
    cfg.arm_joint_names = cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names
    cfg.action_space = cfg.num_arm_joints
    # State: 6 jpos + 6 jvel + 3 socket pos + 4 socket quat + 3 plug pos + 4 plug quat = 26
    cfg.state_space = 26
    # Observation: 6 jpos + 6 jvel + 3 socket pos + 4 socket quat = 19
    cfg.observation_space = 19

    cfg.joint_action_scale = cfg.actions.arm_action.scale
    cfg.action_scale_joint_space = [cfg.joint_action_scale] * cfg.action_space

    cfg.initial_joint_pos = [
        cfg.scene.robot.init_state.joint_pos[joint_name] for joint_name in cfg.arm_joint_names
    ]

    # UR policy uses the robot ``base`` frame (180 deg Z from base_link).
    cfg.scene.robot.init_state.rot = (0.0, 0.0, 1.0, 0.0)

    cfg.scene.dp_socket.init_state = RigidObjectCfg.InitialStateCfg(
        pos=_ROS_SOCKET_ROOT,
        rot=_ROS_SOCKET_ROT,
    )
    cfg.scene.dp_plug.init_state = RigidObjectCfg.InitialStateCfg(
        pos=_ROS_PLUG_ROOT,
        rot=_ROS_PLUG_ROT,
    )

    cfg.events.set_robot_to_grasp_pose.params["max_iterations"] = 150

    cfg.fixed_asset_init_pos_center = list(_ROS_GEOMETRY_POS)

    pose_range = cfg.events.randomize_socket_pose.params["pose_range"]
    cfg.fixed_asset_init_pos_range = [
        pose_range["x"][1],
        pose_range["y"][1],
        pose_range["z"][1],
    ]
    cfg.fixed_asset_init_orn_deg = list(_MEASURED_SOCKET_EULER_DEG)
    cfg.fixed_asset_init_orn_deg_range = [
        math.degrees(pose_range["roll"][1]),
        math.degrees(pose_range["pitch"][1]),
        math.degrees(pose_range["yaw"][1]),
    ]

    socket_pos_noise = cfg.observations.policy.socket_pos.noise.noise_cfg.n_max
    cfg.fixed_asset_pos_obs_noise_level = [
        socket_pos_noise,
        socket_pos_noise,
        socket_pos_noise,
    ]


@configclass
class UR10e2F140DisplayportInsertionROSInferenceEnvCfg(UR10e2F140DisplayportInsertionEnvCfg):
    """ROS / Isaac Manipulator inference for UR10e 2F-140 DisplayPort insertion."""

    def __post_init__(self):
        super().__post_init__()
        _configure_ros_inference_common(self)


@configclass
class UR10e2F85DisplayportInsertionROSInferenceEnvCfg(UR10e2F85DisplayportInsertionEnvCfg):
    """ROS / Isaac Manipulator inference for UR10e 2F-85 DisplayPort insertion."""

    def __post_init__(self):
        super().__post_init__()
        _configure_ros_inference_common(self)


@configclass
class UR10e2F140DisplayportInsertionNoJointVelROSInferenceEnvCfg(
    UR10e2F140DisplayportInsertionROSInferenceEnvCfg
):
    """ROS inference for the velocity-free joint-space policy (2F-140)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.joint_vel = None
        self.obs_order = ["arm_dof_pos", "socket_pos", "socket_quat"]
        # Observation: 6 jpos + 3 socket pos + 4 socket quat = 13
        self.observation_space = 13


@configclass
class UR10e2F85DisplayportInsertionNoJointVelROSInferenceEnvCfg(
    UR10e2F85DisplayportInsertionROSInferenceEnvCfg
):
    """ROS inference for the velocity-free joint-space policy (2F-85)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.joint_vel = None
        self.obs_order = ["arm_dof_pos", "socket_pos", "socket_quat"]
        self.observation_space = 13
