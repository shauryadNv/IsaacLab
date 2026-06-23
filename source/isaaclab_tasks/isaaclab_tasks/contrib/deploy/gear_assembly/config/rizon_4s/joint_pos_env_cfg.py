# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import isaaclab_tasks.contrib.deploy.mdp as mdp
import isaaclab_tasks.contrib.deploy.mdp.events as gear_assembly_events
from isaaclab_tasks.contrib.deploy.gear_assembly.gear_assembly_env_cfg import GearAssemblyEnvCfg
from isaaclab_tasks.contrib.deploy.mdp.noise_models import (
    ResetSampledConstantNoiseModelCfg,
    ResetSampledQuaternionNoiseModelCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets import FLEXIV_RIZON4S_GRAV_GRIPPER_CFG  # isort: skip


##
# Gripper-specific helper functions
##


# Grav gripper mimic gearing applied to ``finger_joint`` position: each follower joint is set to
# ``gearing * finger_joint_position``. Resolved by joint *name* (not positional index) so the grasp
# is correct regardless of the backend DOF ordering (PhysX and Newton order joints differently).
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
    """Set finger joint positions for the Grav gripper, resolving joints by name.

    Args:
        joint_pos: Joint positions tensor.
        reset_ind_joint_pos: Row indices into the sliced ``joint_pos`` tensor.
        finger_joints: Gripper joint indices (kept for signature compatibility; unused).
        finger_joint_position: Target position for the main finger joint [rad].
        joint_name_to_idx: Mapping from joint name to simulation joint index. Required; the mimic
            gearing in :data:`_GRAV_GRIPPER_MIMIC_GEARING` is applied per named joint so the closed
            grasp is identical across physics backends despite differing DOF orderings.
    """
    if joint_name_to_idx is None:
        raise ValueError("set_finger_joint_pos_grav requires 'joint_name_to_idx' to resolve gripper joints by name.")

    missing = [name for name in _GRAV_GRIPPER_MIMIC_GEARING if name not in joint_name_to_idx]
    if missing:
        raise ValueError(f"Grav gripper joints not found on the robot: {missing}. Available: {list(joint_name_to_idx)}")

    for idx in reset_ind_joint_pos:
        for joint_name, gearing in _GRAV_GRIPPER_MIMIC_GEARING.items():
            joint_pos[idx, joint_name_to_idx[joint_name]] = gearing * finger_joint_position


##
# Environment configuration
##


@configclass
class EventCfg:
    """Configuration for events."""

    small_gear_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_small", body_names=".*"),
            "static_friction_range": (3.0, 3.0),
            "dynamic_friction_range": (3.0, 3.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    medium_gear_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_medium", body_names=".*"),
            "static_friction_range": (3.0, 3.0),
            "dynamic_friction_range": (3.0, 3.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    large_gear_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_large", body_names=".*"),
            "static_friction_range": (3.0, 3.0),
            "dynamic_friction_range": (3.0, 3.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    gear_base_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_base", body_names=".*"),
            "static_friction_range": (0.0, 0.0),
            "dynamic_friction_range": (0.0, 0.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
            "static_friction_range": (3.0, 3.0),
            "dynamic_friction_range": (3.0, 3.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    randomize_gear_type = EventTerm(
        func=gear_assembly_events.randomize_gear_type,
        mode="reset",
        params={"gear_types": ["gear_small", "gear_medium", "gear_large"]},
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_gears_and_base_pose = EventTerm(
        func=gear_assembly_events.randomize_gears_and_base_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.1, 0.1],
                "y": [-0.25, 0.25],
                "z": [-0.1, 0.1],
                "roll": [-math.pi / 90, math.pi / 90],  # 2 degree
                "pitch": [-math.pi / 90, math.pi / 90],  # 2 degree
                "yaw": [-math.pi / 6, math.pi / 6],  # 30 degree
            },
            "gear_pos_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [0.0575, 0.0775],
            },
            "velocity_range": {},
        },
    )

    set_robot_to_grasp_pose = EventTerm(
        func=gear_assembly_events.set_robot_to_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "pos_randomization_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
        },
    )


@configclass
class Rizon4sGearAssemblyEnvCfg(GearAssemblyEnvCfg):
    """Configuration for Flexiv Rizon 4s with Grav Gripper Gear Assembly Environment.

    The Flexiv Rizon 4s is a 7-DOF collaborative robot arm equipped with the
    Flexiv Grav parallel gripper for gear manipulation tasks.
    """

    ee_grasp_weight_ramp_steps: int = 512_000

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Flexiv-specific observation noise overrides
        self.observations.policy.gear_shaft_pos.noise = ResetSampledConstantNoiseModelCfg(
            noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
        )
        self.observations.policy.gear_shaft_quat.noise = ResetSampledQuaternionNoiseModelCfg(
            roll_range=(-0.03491, 0.03491),
            pitch_range=(-0.03491, 0.03491),
            yaw_range=(-0.03491, 0.03491),
        )

        # Robot-specific parameters for Flexiv Rizon 4s with Grav gripper
        self.end_effector_body_name = "link7"  # End effector body name for IK
        self.num_arm_joints = 7  # Number of arm joints (Rizon 4s has 7 DOF)
        # Rotation offset for grasp pose (quaternion [x, y, z, w])
        # Computed from IK convergence for downward-facing end effector
        self.grasp_rot_offset = [
            -0.707,
            0.707,
            0.0,
            0.0,
        ]
        self.gripper_joint_setter_func = set_finger_joint_pos_grav  # Grav gripper joint setter function

        # Gear orientation termination thresholds (in degrees)
        self.gear_orientation_roll_threshold_deg = 15.0  # Maximum allowed roll deviation
        self.gear_orientation_pitch_threshold_deg = 15.0  # Maximum allowed pitch deviation
        self.gear_orientation_yaw_threshold_deg = 180.0  # Maximum allowed yaw deviation

        # Common observation configuration for Rizon 4s joints (arm only, not gripper)
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

        # override events
        self.events = EventCfg()

        # Update termination thresholds from config
        self.terminations.gear_orientation_exceeded.params["roll_threshold_deg"] = (
            self.gear_orientation_roll_threshold_deg
        )
        self.terminations.gear_orientation_exceeded.params["pitch_threshold_deg"] = (
            self.gear_orientation_pitch_threshold_deg
        )
        self.terminations.gear_orientation_exceeded.params["yaw_threshold_deg"] = (
            self.gear_orientation_yaw_threshold_deg
        )

        # Action configuration for Rizon 4s arm
        # Using smaller action scale for stability
        self.joint_action_scale = 0.025
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
            ],
            scale=self.joint_action_scale,
            use_zero_offset=True,
        )

        # Switch robot to Flexiv Rizon 4s with Grav gripper
        self.scene.robot = FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.spawn.replace(
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
                    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            # Joint positions based on IK from center of distribution for randomized gear positions
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "joint1": 0.0,
                    "joint2": -0.698,
                    "joint3": 0.0,
                    "joint4": 1.571,
                    "joint5": 0.0,
                    "joint6": 0.698,
                    "joint7": 0.0,
                },
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        # Grav gripper actuator configuration for gear manipulation. The effort limit sets the clamp
        # force on the grasped gear; with concave SDF gear collision the fingers conform to the gear
        # rim, so a firm clamp holds without the convex-hull overlap that previously ejected gears.
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=2.0,
            stiffness=2e3,
            # Heavy damping + armature so the drive closes smoothly instead of overshooting the close
            # target and bouncing open (which ejects a light gear). The near-zero-inertia gripper
            # link on MJWarp (implicitfast, 2 substeps) is otherwise underdamped and oscillates
            # (driver swings -0.18 -> +0.12 -> ...), unlike PhysX's stable joint drive. Armature adds
            # effective inertia so the same force produces less overshoot.
            damping=1.5e2,
            friction=0.0,
            armature=0.1,
        )

        # Follower joints of the parallel-gripper closed-loop linkage (knuckles + outer fingers).
        # Their position is carried by the mimic constraints coupling them to ``finger_joint`` (the
        # Newton importer turns the USD PhysxMimicJointAPI into mjEQ_JOINT equality constraints), so
        # we add no stiffness -- mjwarp ignores per-joint PD on mimic-follower DOFs anyway. The key
        # stabilizer is ``armature``: the gripper links have near-degenerate inertia (the run logs
        # "Inertia validation corrected ... bodies"), and without armature the mimic-constrained
        # followers blow up to tens of radians in a single step (flapping). Armature regularizes the
        # mass matrix so the mimic constraints enforce correctly and all six joints move coherently.
        self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_knuckle_joint", ".*_outer_finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=10.0,
            friction=0.0,
            armature=0.05,
        )

        # Override gear initial states for Rizon. Each gear spawns over its own shaft on the base so
        # the three gears do not overlap. The base shafts are offset along the base-frame +x axis by
        # ``gear_offsets`` (see GearAssemblyEnvCfg); under the base rotation below that maps to world
        # -x, so each gear's world x = base_x - shaft_offset_x. (Without distinct positions the gears
        # spawned coincident and, now that they collide, exploded apart on contact.)
        _base_pos = (0.481, -0.073, 0.071)
        _base_rot = (0.0, 0.0, 0.70711, -0.70711)
        _shaft_x = {"gear_small": 0.076125, "gear_medium": 0.030375, "gear_large": -0.045375}
        self.scene.factory_gear_base.init_state = RigidObjectCfg.InitialStateCfg(pos=_base_pos, rot=_base_rot)
        for _name, _asset in (
            ("gear_small", self.scene.factory_gear_small),
            ("gear_medium", self.scene.factory_gear_medium),
            ("gear_large", self.scene.factory_gear_large),
        ):
            _asset.init_state = RigidObjectCfg.InitialStateCfg(
                pos=(_base_pos[0] - _shaft_x[_name], _base_pos[1], _base_pos[2]),
                rot=_base_rot,
            )

        # Gear grasp offset (gear frame). Two consumers:
        #  * set_robot_to_grasp_pose ignores z (the reach is derived from kinematics) and uses only
        #    the in-plane (x, y) selection -> kept at 0 so the fingertip midpoint lands on the gear.
        #  * reset_when_gear_dropped uses the full offset as the expected gear->end-effector vector;
        #    the z = -0.35 reach keeps "gear + offset" near link7 while the gear is held (the
        #    distance check would otherwise always trip, since link7 sits ~0.35 m above the gear).
        # The previous large y offset (reused gear-shaft offset) placed the fingers off the gear.
        # Grasp-point offset used by the grasp-keypoint rewards and the gear-dropped termination, which
        # measure the gear grasp point against the end-effector *link* (~0.35 m above the fingertips).
        # The -0.35 axial offset makes the grasp point coincide with the EE link when grasped, so the
        # reward error / dropped distance read ~0. (Not a physical point on the gear; see
        # ``gear_offsets_grasp_hub`` for the actual hub the fingertips target.)
        self.gear_offsets_grasp = {
            "gear_small": [0.0, 0.0, -0.35],
            "gear_medium": [0.0, 0.0, -0.35],
            "gear_large": [0.0, 0.0, -0.35],
        }

        # Grasp point the fingertip midpoint targets in ``set_robot_to_grasp_pose``: the gear's central
        # hub, which protrudes ~29 mm above the toothed disk. The grasp rotation maps offset +z to gear
        # -z, so a negative ``z`` reaches the hub. Distinct from ``gear_offsets_grasp`` because that term
        # references the EE link while the grasp pose references the fingertip midpoint.
        self.gear_offsets_grasp_hub = {
            "gear_small": [0.0, 0.0, -0.029],
            "gear_medium": [0.0, 0.0, -0.029],
            "gear_large": [0.0, 0.0, -0.029],
        }

        # Grasp widths for Grav gripper (raw radian values for finger_joint)
        self.hand_grasp_width = {
            "gear_small": 0.05,
            "gear_medium": 0.2,
            "gear_large": 0.28,
        }

        # Close target for the Grav gripper main drive [rad]. The reset writes this grip pose
        # directly, so the gear is clamped from the first step rather than slipping during a dynamic
        # close.
        self.hand_close_width = {
            "gear_small": -0.155,
            "gear_medium": -0.155,
            "gear_large": -0.155,
        }

        # Populate event term parameters
        self.events.randomize_gears_and_base_pose.params["gear_offsets"] = self.gear_offsets
        # Rest-height offset [m] for non-selected gears, applied in the base frame so they stay centered
        # on their shafts under randomized base roll/pitch.
        self.events.randomize_gears_and_base_pose.params["seated_gear_z_offset"] = 0.042
        self.events.set_robot_to_grasp_pose.params["gear_offsets_grasp"] = self.gear_offsets_grasp_hub
        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func

        # Flexiv-specific reward terms for EE-grasp keypoint tracking
        self.rewards.end_effector_grasp_keypoint_tracking = RewTerm(
            func=mdp.keypoint_ee_grasp_error,
            weight=-0.5,
            params={
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "keypoint_scale": 0.15,
                "ee_grasp_threshold": 0.00,
                "weight_ramp_start": 0.0,
                "weight_ramp_steps": self.ee_grasp_weight_ramp_steps,
                "end_effector_body_name": self.end_effector_body_name,
                "grasp_rot_offset": self.grasp_rot_offset,
                "gear_offsets_grasp": self.gear_offsets_grasp,
            },
        )
        self.rewards.end_effector_grasp_keypoint_tracking_exp = RewTerm(
            func=mdp.keypoint_ee_grasp_error_exp,
            weight=0.5,
            params={
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "kp_exp_coeffs": [(50, 0.0001), (300, 0.0001)],
                "kp_use_sum_of_exps": False,
                "keypoint_scale": 0.15,
                "ee_grasp_threshold": 0.00,
                "weight_ramp_start": 0.0,
                "weight_ramp_steps": self.ee_grasp_weight_ramp_steps,
                "end_effector_body_name": self.end_effector_body_name,
                "grasp_rot_offset": self.grasp_rot_offset,
                "gear_offsets_grasp": self.gear_offsets_grasp,
            },
        )

        # Populate termination term parameters
        self.terminations.gear_dropped.params["gear_offsets_grasp"] = self.gear_offsets_grasp
        self.terminations.gear_dropped.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.gear_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        self.terminations.gear_orientation_exceeded.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.gear_orientation_exceeded.params["grasp_rot_offset"] = self.grasp_rot_offset
