# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-space DisplayPort insertion environment for Flexiv Rizon 4S + Grav gripper.

Uses Operational Space Control (OSC) with 6D rotation observations and an
at-goal curriculum, mirroring the GB300 ``task_space_env_cfg.py``.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
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
)

from .joint_pos_env_cfg import (
    _PLUG_ROOT,
    _PLUG_ROT,
    _SOCKET_ROOT,
    _SOCKET_ROT,
    _exp_curriculum_params,
    set_finger_joint_pos_grav,
)

# ---------------------------------------------------------------------------
# Rizon 4S arm joint names (convenience)
# ---------------------------------------------------------------------------
_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

# ---------------------------------------------------------------------------
# OSC gains
# ---------------------------------------------------------------------------
# d_gain = 2 * sqrt(p_gain) * damping_ratio
_STIFFNESS = (300.0, 300.0, 300.0, 30.0, 30.0, 30.0)
_DAMPING_RATIO_TRANS = 35.0 / (2.0 * math.sqrt(300.0))  # ~1.010
_DAMPING_RATIO_ROT = 1.1 / (2.0 * math.sqrt(30.0))  # ~0.100
_DAMPING_RATIO = (
    _DAMPING_RATIO_TRANS, _DAMPING_RATIO_TRANS, _DAMPING_RATIO_TRANS,
    _DAMPING_RATIO_ROT, _DAMPING_RATIO_ROT, _DAMPING_RATIO_ROT,
)

# Production action_scale = [0.01]*6 for both position and rotation
_ACTION_SCALE = 0.025

# DisplayPort plug insertion length (blade engagement along the insertion axis).
# ~11 mm of blade nests in the socket cavity at the verified seated pose.
_INSERTION_LENGTH = 0.011

# Gripper tool-center-point (TCP) offset from the flange body, in the flange's
# local frame [m]. The policy observes the TCP pose (where the plug is actually
# held), not the raw flange. Matches IsaacLab_UR's Flexiv + Grav
# ``gripper_eef_pos_local = [0.0, 0.0, 0.2 - 0.0075]``. Direct cross-check: in
# the live grasp the flange sits ~0.1875-0.1925 m above the held plug along the
# tool axis.
_TCP_OFFSET = [0.0, 0.0, 0.2 - 0.0075]

##
# Pre-defined configs
##
from isaaclab_assets import FLEXIV_RIZON4S_GRAV_GRIPPER_CFG  # isort: skip


##
# Observation configuration
##


@configclass
class TaskSpaceObservationsCfg:
    """Task-space observations with 6D rotation representation."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations: EEF pose + socket keypoint frame (18 dims)."""

        # Observe the gripper TCP pose (flange + _TCP_OFFSET), not the raw flange —
        # mirrors the GB300 reference (obs=TCP, OSC controls the flange).
        eef_pos = ObsTerm(
            func=mdp.eef_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "flange", "offset": _TCP_OFFSET},
        )
        # TCP shares the flange orientation (pure-translation offset), so the 6D
        # rotation is read directly from the flange body.
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations: joint state + both keypoint frames (32 dims)."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS)},
        )
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

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# =============================== EXPERIMENT TOGGLES ===============================
# Parametrized training-experiment sweep for the TASK-SPACE (OSC) env. Each
# experiment commit sets these switches to a specific combination; see the
# experiment matrix / commit log. The block between START/END markers is what
# the experiment commits edit. Mirrors the joint-space env's EXP block, minus
# sysid: the joint-space sysid (identified joint-PD gains + delayed joint action)
# is invalid under OSC because the arm joint PD is zeroed for operational-space
# control, so there is no EXP_SYSID toggle here.
# --- EXP TOGGLES START ---
EXP_SOCKET_POS_RANGE = [0.01, 0.01, 0.02]  # socket position randomization, +/- m per axis [x, y, z]
EXP_SOCKET_ORN_DEG = 2.0                    # socket orientation randomization, +/- deg on roll/pitch/yaw
# modes: disabled|fixed80|fixed50|fixed20|anneal_80_0_1000|anneal_80_20_1000|anneal_80_20_500|anneal_80_0_500
EXP_CURRICULUM = "fixed50"
EXP_EQUAL_REWARD_WEIGHTS = True  # True => exp keypoint weight == linear (UR 1:1); False => 2:1
EXP_RAND = "none"          # none|friction|pd|both -- robot joint-friction and/or PD-gain domain randomization
EXP_CALIB_USD = False      # True => spawn the calibrated Rizon4s USD instead of the stock one
EXP_OBS_NOISE_M = 0.0      # socket-position observation noise [m] applied at inference (0.0=none, 0.005=5mm)
# --- EXP TOGGLES END ---

_EXP_CURR = _exp_curriculum_params(EXP_CURRICULUM)
# =================================================================================


##
# Event configuration
##


@configclass
class TaskSpaceEventCfg:
    """Events for the task-space DisplayPort insertion environment."""

    plug_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("dp_plug", body_names=".*"),
            "static_friction_range": (0.001, 0.001),
            "dynamic_friction_range": (0.001, 0.001),
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
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    # Robot joint-friction / PD-gain domain randomization. Defined here but gated by
    # EXP_RAND in the ROS-inference __post_init__ (disabled unless selected). Ranges
    # mirror the joint-space friction (refimage58) and PD-gain (refimage59) runs.
    robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            ),
            "friction_distribution_params": (0.0, 0.15),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    robot_pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            ),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_socket_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                # Driven by the EXP_SOCKET_POS_RANGE / EXP_SOCKET_ORN_DEG toggles.
                "x": [-EXP_SOCKET_POS_RANGE[0], EXP_SOCKET_POS_RANGE[0]],
                "y": [-EXP_SOCKET_POS_RANGE[1], EXP_SOCKET_POS_RANGE[1]],
                "z": [-EXP_SOCKET_POS_RANGE[2], EXP_SOCKET_POS_RANGE[2]],
                "roll": [-math.radians(EXP_SOCKET_ORN_DEG), math.radians(EXP_SOCKET_ORN_DEG)],
                "pitch": [-math.radians(EXP_SOCKET_ORN_DEG), math.radians(EXP_SOCKET_ORN_DEG)],
                "yaw": [-math.radians(EXP_SOCKET_ORN_DEG), math.radians(EXP_SOCKET_ORN_DEG)],
                # --- previous fixed values (superseded by EXP toggles) ---
                # "x": [-0.01, 0.01],
                # "y": [-0.01, 0.01],
                # "z": [-0.02, 0.02],
                # "roll": [0.0, 0.0],
                # "pitch": [0.0, 0.0],
                # "yaw": [0.0, 0.0],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("dp_socket"),
        },
    )

    # Disabled: plain uniform plug randomization. Superseded by the at-goal
    # curriculum below (re-enable this and comment out reset_plug_curriculum to
    # match the joint-space env's plug-start setup exactly).
    # randomize_plug_pose = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": [-0.02, 0.02],
    #             "y": [-0.02, 0.02],
    #             "z": [-0.01, 0.01],
    #             "roll": [0.0, 0.0],
    #             "pitch": [0.0, 0.0],
    #             "yaw": [0.0, 0.0],
    #         },
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("dp_plug"),
    #     },
    # )

    # 80% at-goal curriculum (mirrors IsaacLab_UR gb300, osmo at_goal_prob=0.8).
    # A fraction `at_goal_prob` of envs spawn the plug already inserted at a random
    # depth (0 -> insertion_length) with goal orientation; the rest get uniform
    # randomization around the approach pose via `normal_pose_range`.
    #
    # at_goal_prob is annealed linearly from `at_goal_prob` (start) down to
    # `at_goal_prob_final` between iterations `anneal_start_iter` and
    # `anneal_end_iter`, so the policy is weaned off goal-seeded starts and must
    # learn the full approach by the end of training. Iterations are derived from
    # the env step counter via `num_steps_per_env` (must match the agent cfg).
    # To disable annealing, set `at_goal_prob_final=None` (constant at_goal_prob).
    reset_plug_curriculum = EventTerm(
        func=mdp.reset_plug_at_goal_curriculum,
        mode="reset",
        params={
            "plug_cfg": SceneEntityCfg("dp_plug"),
            "socket_cfg": SceneEntityCfg("dp_socket"),
            # Driven by the EXP_CURRICULUM toggle (see _exp_curriculum_params).
            "at_goal_prob": _EXP_CURR["at_goal_prob"],
            "at_goal_prob_final": _EXP_CURR["at_goal_prob_final"],
            "anneal_start_iter": 0.0,
            "anneal_end_iter": _EXP_CURR["anneal_end_iter"],
            "num_steps_per_env": 512,
            # --- previous fixed values (superseded by EXP_CURRICULUM toggle) ---
            # "at_goal_prob": 0.0,
            # "at_goal_prob_final": 0.0,
            # "anneal_end_iter": 1000.0,
            "insertion_axis": [1.0, 0.0, 0.0],
            "insertion_length": _INSERTION_LENGTH,
            # Deadzone fix: at-goal envs seed shallow (0-15 mm) and approach envs seed
            # a deeper band (20-60 mm) along the insertion axis, from the same socket
            # keypoint origin, so there is no unsampled gap between the two bands.
            "at_goal_depth_range": [0.0, 0.015],
            "approach_depth_range": [0.02, 0.06],
            "socket_insertion_offset": SOCKET_INSERTION_OFFSET,
            "plug_insertion_offset": PLUG_INSERTION_OFFSET,
            "goal_rot": list(PLUG_GOAL_ROT),
            # Non-at-goal approach randomization. Matches gb300's
            # held_asset_init_pos_range = [0.02, 0.02, 0.01] (x, y, z) [m].
            # Lateral (x, y) only -- axial (z) spread is now handled by
            # approach_depth_range above, so z lateral is disabled to avoid
            # double-randomizing along the insertion axis.
            "normal_pose_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [0.0, 0.0],
                # "z": [-0.01, 0.01],
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


##
# Terminations
##


@configclass
class TaskSpaceTerminationsCfg:
    """Termination terms for the task-space DisplayPort insertion."""

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


##
# Main environment configuration
##


@configclass
class Rizon4sTaskSpaceDisplayportInsertionEnvCfg(DisplayportInsertionEnvCfg):
    """Task-space DisplayPort insertion with OSC control, 6D obs, and curriculum."""

    def __post_init__(self):
        super().__post_init__()

        # EEF / grasp settings (same as joint-space variant)
        self.end_effector_body_name = "flange"
        self.num_arm_joints = 7
        self.grasp_offset = [0.0025, 0.0, -0.1875]
        self.grasp_rot_offset = [0.0, 0.0, 0.0, 1.0]
        self.gripper_joint_setter_func = set_finger_joint_pos_grav

        self.plug_orientation_roll_threshold_deg = 15.0
        self.plug_orientation_pitch_threshold_deg = 15.0
        self.plug_orientation_yaw_threshold_deg = 180.0

        # ----- Observations: task-space with 6D rotation -----
        self.observations = TaskSpaceObservationsCfg()

        # ----- Actions: Operational Space Controller -----
        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=_ARM_JOINTS,
            body_name="flange",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="fixed",
                inertial_dynamics_decoupling=False,
                motion_stiffness_task=_STIFFNESS,
                motion_damping_ratio_task=_DAMPING_RATIO,
                nullspace_control="none",
            ),
            position_scale=_ACTION_SCALE,
            orientation_scale=_ACTION_SCALE,
        )

        # ----- Events: curriculum + production friction -----
        self.events = TaskSpaceEventCfg()

        # ----- Terminations -----
        self.terminations = TaskSpaceTerminationsCfg()

        self.terminations.plug_orientation_exceeded.params["roll_threshold_deg"] = (
            self.plug_orientation_roll_threshold_deg
        )
        self.terminations.plug_orientation_exceeded.params["pitch_threshold_deg"] = (
            self.plug_orientation_pitch_threshold_deg
        )
        self.terminations.plug_orientation_exceeded.params["yaw_threshold_deg"] = (
            self.plug_orientation_yaw_threshold_deg
        )

        # ----- Scene: robot -----
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
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
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

        # OSC requires the arm joints to be pure torque-controlled. The stock
        # Flexiv cfg gives the arm a stiff implicit position PD (stiffness
        # 1320/600/216); left enabled it holds every joint at its reset target
        # and completely overrides the OSC-computed joint efforts, so the arm
        # never moves. Zero the arm PD gains (damping is provided by the OSC via
        # ``motion_damping_ratio_task``). Mirrors scripts/tutorials/05_controllers/run_osc.py.
        for _arm_actuator in ("shoulder", "elbow", "wrist"):
            self.scene.robot.actuators[_arm_actuator].stiffness = 0.0
            self.scene.robot.actuators[_arm_actuator].damping = 0.0

        # Grav gripper actuator configuration
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=2.0,
            velocity_limit_sim=1.0,
            stiffness=2e3,
            damping=1e1,
        )
        self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
        )

        # ----- Workspace positions -----
        self.scene.dp_socket.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_SOCKET_ROOT,
            rot=_SOCKET_ROT,
        )
        self.scene.dp_plug.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_PLUG_ROOT,
            rot=_PLUG_ROT,
        )

        # Gripper widths (same as joint-space variant; CALIBRATE for DP plug)
        self.hand_grasp_width = 0.3
        self.hand_hold_width = -0.05
        self.hand_close_width = -0.155

        # Wire grasp event params
        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["grasp_offset"] = self.grasp_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func
        self.events.set_robot_to_grasp_pose.params["max_iterations"] = 150

        # Wire termination params
        self.terminations.plug_dropped.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_dropped.params["grasp_offset"] = self.grasp_offset
        self.terminations.plug_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        self.terminations.plug_orientation_exceeded.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_orientation_exceeded.params["grasp_rot_offset"] = self.grasp_rot_offset

        # Experiment toggle: match IsaacLab_UR's 1:1 linear:exp keypoint weighting
        # (exp weight == |linear weight| = 1.5) instead of the default 2:1 (exp = 3.0).
        # The joint-space env applies this in its own __post_init__; the task-space
        # env previously omitted it, so OSC runs were stuck at 2:1 regardless of the
        # toggle. Applied here after super() so self.rewards is populated.
        if EXP_EQUAL_REWARD_WEIGHTS:
            self.rewards.plug_socket_keypoint_tracking_exp.weight = abs(
                self.rewards.plug_socket_keypoint_tracking.weight
            )


@configclass
class Rizon4sTaskSpaceDisplayportInsertionEnvCfg_PLAY(Rizon4sTaskSpaceDisplayportInsertionEnvCfg):
    """Play configuration for task-space DisplayPort insertion."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
