# Copyright (c) 2026-2027, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Flexiv Rizon robots.

The following configurations are available:

* :obj:`FLEXIV_RIZON4S_CFG`: The Flexiv Rizon 4s arm without a gripper.
* :obj:`FLEXIV_RIZON4S_GRAV_GRIPPER_CFG`: The Flexiv Rizon 4s arm with Grav gripper.

Reference: https://www.flexiv.com/product/rizon
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

FLEXIV_RIZON4S_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/shauryad/workspaces/rl_policy/isaac_sim_ws/exts/isaacsim.robot.manipulators.examples/data/flexiv/Rizon4s.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0497,
            "joint2": -1.5261,
            "joint3": 0.2972,
            "joint4": -0.4166,
            "joint5": 0.6128,
            "joint6": 4.3228,
            "joint7": -0.5606
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # Joints 1-2: Higher torque (123 Nm), lower speed (120°/s = 2.094 rad/s)
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=123.0,
            velocity_limit_sim=2.094,
            stiffness=None,
            damping=None,
        ),
        # Joints 3-4: Medium torque (64 Nm), medium speed (140°/s = 2.443 rad/s)
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-4]"],
            effort_limit_sim=64.0,
            velocity_limit_sim=2.443,
            stiffness=None,
            damping=None,
        ),
        # Joints 5-7: Lower torque (39 Nm), higher speed (280°/s = 4.887 rad/s)
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-7]"],
            effort_limit_sim=39.0,
            velocity_limit_sim=4.887,
            stiffness=None,
            damping=None,
        ),
    },
)
"""Configuration of Flexiv Rizon 4s arm using implicit actuator models.

Note: stiffness and damping are set to None to use USD defaults.
This config is used for the reach task.
"""


FLEXIV_RIZON4S_GRAV_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/shauryad/workspaces/rl_policy/isaac_sim_ws/exts/isaacsim.robot.manipulators.examples/data/flexiv/Rizon4s_with_Grav.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.698,
            "joint3": 0.0,
            "joint4": 1.571,
            "joint5": 0.0,
            "joint6": 0.698,
            "joint7": 0.0,
            # Grav gripper joints
            "finger_joint": 0.0,
            "left_outer_finger_joint": 0.0,
            "right_outer_finger_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators={
        # Joints 1-2: Higher torque (123 Nm), lower speed (120°/s = 2.094 rad/s)
        # Stiffness/damping tuned for stable position control
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=123.0,
            velocity_limit_sim=2.094,
            stiffness=1320.0,
            damping=72.0,
            friction=0.0,
            armature=0.0,
        ),
        # Joints 3-4: Medium torque (64 Nm), medium speed (140°/s = 2.443 rad/s)
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-4]"],
            effort_limit_sim=64.0,
            velocity_limit_sim=2.443,
            stiffness=600.0,
            damping=35.0,
            friction=0.0,
            armature=0.0,
        ),
        # Joints 5-7: Lower torque (39 Nm), higher speed (280°/s = 4.887 rad/s)
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-7]"],
            effort_limit_sim=39.0,
            velocity_limit_sim=4.887,
            stiffness=216.0,
            damping=29.0,
            friction=0.0,
            armature=0.0,
        ),
        # Grav gripper main actuator - finger_joint is the main actuation joint
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.6,
            stiffness=2e3,
            damping=1e1,
            friction=0.0,
            armature=0.0,
        ),
        # Passive/mimic joints in the gripper - set to zero stiffness/damping
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[".*_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""Configuration of Flexiv Rizon 4s arm with Grav gripper using implicit actuator models.

The Grav gripper is a parallel gripper with the following joint configuration:
- finger_joint: Main actuation joint (opened: 45°, closed: -8.88°)
- *_knuckle_joint: Passive/mimic joints (not directly actuated)

End effector body: right_finger_tip
"""
