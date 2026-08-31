# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OSC configuration for Rizon 4S DisplayPort insertion.

This module contains the validated point-SDF training profile only. It is kept
separate from :mod:`task_space_env_cfg` because the Newton policy observes the
flange origin in a different tensor order than the PhysX task-space policy.
"""

import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import isaaclab_tasks.contrib.deploy.mdp as deploy_mdp
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    DISPLAY_ASSETS_DIR,
    SOCKET_INSERTION_OFFSET,
    ObservationsCfg,
)
from isaaclab_tasks.contrib.deploy.mdp.noise_models import ResetSampledConstantNoiseModelCfg
from isaaclab_tasks.utils import PresetCfg

from .joint_pos_env_cfg import _RIZON4S_CALIBRATED_USD_PATH
from .task_space_env_cfg import Rizon4sTaskSpaceDisplayportInsertionEnvCfg, TaskSpaceEventCfg

_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
_OSC_ACTION_SCALE = 0.025
_OSC_STIFFNESS = (300.0, 300.0, 300.0, 30.0, 30.0, 30.0)
_OSC_DAMPING_RATIO = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

_NEWTON_PLUG_USD_PATH = os.path.join(DISPLAY_ASSETS_DIR, "display_port_plug_newton_sdf.usda")
_NEWTON_SOCKET_USD_PATH = os.path.join(DISPLAY_ASSETS_DIR, "display_port_socket_newton_sdf.usda")


@configclass
class DisplayportNewtonPhysicsCfg(PresetCfg):
    """Validated Newton point-SDF physics profile for DisplayPort insertion."""

    newton_sdf: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=8192,
            nconmax=8192,
            iterations=100,
            ls_iterations=50,
            update_data_interval=10,
            impratio=10.0,
            cone="elliptic",
            ccd_iterations=35,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            reduce_contacts=True,
            max_triangle_pairs=2**25,
        ),
        num_substeps=20,
        collision_decimation=10,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_sdf


@configclass
class NewtonTaskSpaceObservationsCfg:
    """Checkpoint-compatible actor and critic observations for Newton OSC."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations in the exact validated 18-dimensional order."""

        socket_pos = ObsTerm(
            func=deploy_mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket"), "offset": SOCKET_INSERTION_OFFSET},
            noise=ResetSampledConstantNoiseModelCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
            ),
        )
        tool_pos = ObsTerm(
            func=deploy_mdp.eef_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "flange", "offset": [0.0, 0.0, 0.0]},
        )
        tool_rot_6d = ObsTerm(
            func=deploy_mdp.eef_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "flange"},
        )
        socket_rot_6d = ObsTerm(
            func=deploy_mdp.rigid_object_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    # Keep every robot joint in the privileged critic, matching the validated
    # checkpoint's 40-dimensional critic input.
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()


@configclass
class NewtonTaskSpaceEventCfg(TaskSpaceEventCfg):
    """Newton-specific material and arm-friction randomization events."""

    randomize_arm_joint_friction = EventTerm(
        func=env_mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS),
            "friction_distribution_params": (0.0, 0.15),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    # OSC is torque controlled, so position-controller gain randomization is
    # intentionally disabled while joint-friction randomization remains active.
    randomize_arm_pd_gains: EventTerm | None = None


@configclass
class Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg(Rizon4sTaskSpaceDisplayportInsertionEnvCfg):
    """Validated Newton point-SDF OSC training configuration.

    The actor contract is ``socket_pos, flange_pos, flange_rot_6d,
    socket_rot_6d``. This differs from the TCP-first PhysX task-space contract,
    so the configurations must not share checkpoints despite both being 18-D.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # 100 Hz outer simulation ticks, 20 solver substeps, collision every 10
        # substeps, and one policy action every three outer ticks give 2 kHz,
        # 200 Hz, and 33.3 Hz respectively.
        self.sim.dt = 0.01
        self.sim.physics = DisplayportNewtonPhysicsCfg()
        self.decimation = 3
        self.sim.render_interval = self.decimation

        self.scene.dp_plug.spawn.usd_path = _NEWTON_PLUG_USD_PATH
        self.scene.dp_socket.spawn.usd_path = _NEWTON_SOCKET_USD_PATH

        self.observations = NewtonTaskSpaceObservationsCfg()
        self.task_space_obs_order = ["socket_pos", "tool_pos", "tool_rot_6d", "socket_rot_6d"]

        self.actions.arm_action = deploy_mdp.DeployOperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=_ARM_JOINTS,
            body_name="flange",
            body_offset=deploy_mdp.DeployOperationalSpaceControllerActionCfg.OffsetCfg(),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="fixed",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=_OSC_STIFFNESS,
                motion_damping_ratio_task=_OSC_DAMPING_RATIO,
                nullspace_control="none",
            ),
            nullspace_joint_pos_target="none",
            position_scale=_OSC_ACTION_SCALE,
            orientation_scale=_OSC_ACTION_SCALE,
        )

        self.events = NewtonTaskSpaceEventCfg()
        self.events.plug_physics_material.params["static_friction_range"] = (3.0, 3.0)
        self.events.plug_physics_material.params["dynamic_friction_range"] = (3.0, 3.0)
        self.events.robot_physics_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.robot_physics_material.params["dynamic_friction_range"] = (1.0, 1.0)

        grasp_params = self.events.set_robot_to_grasp_pose.params
        grasp_params["end_effector_body_name"] = self.end_effector_body_name
        grasp_params["num_arm_joints"] = self.num_arm_joints
        grasp_params["grasp_rot_offset"] = self.grasp_rot_offset
        grasp_params["grasp_offset"] = self.grasp_offset
        grasp_params["gripper_joint_setter_func"] = self.gripper_joint_setter_func
        grasp_params["max_iterations"] = 150

        # Newton cancels robot-body gravity directly. OSC gravity compensation
        # stays disabled to avoid applying gravity twice.
        self.scene.robot.spawn.usd_path = _RIZON4S_CALIBRATED_USD_PATH
        self.scene.robot.spawn.rigid_props = sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0)
        self.scene.robot.spawn.joint_drive_props = sim_utils.MujocoJointDrivePropertiesCfg(actuatorgravcomp=False)

        for actuator_name in ("shoulder", "elbow", "wrist"):
            self.scene.robot.actuators[actuator_name].stiffness = 0.0
            self.scene.robot.actuators[actuator_name].damping = 0.0

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
        self.hand_hold_width = -0.1
        self.hand_close_width = -0.1


@configclass
class Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg_PLAY(Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg):
    """Deterministic play configuration for a trained Newton OSC policy."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.reset_plug_curriculum.params["at_goal_prob"] = 0.0
        self.events.reset_plug_curriculum.params["at_goal_prob_final"] = 0.0
