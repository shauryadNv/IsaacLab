# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton point-SDF OSC configuration for Rizon 4S DisplayPort insertion.

It is separate from :mod:`task_space_env_cfg` because its checkpoint ABI observes
the flange origin in a different tensor order than the PhysX task-space policy.
"""

from isaaclab_newton.physics import (
    FeatherPGSSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_newton.sim.schemas import NewtonCollisionCfg, NewtonSDFCollisionCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg

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
    SOCKET_INSERTION_OFFSET,
    ObservationsCfg,
)
from isaaclab_tasks.contrib.deploy.mdp.noise_models import ResetSampledConstantNoiseModelCfg
from isaaclab_tasks.utils import PresetCfg, preset

from .task_space_env_cfg import Rizon4sTaskSpaceDisplayportInsertionEnvCfg, TaskSpaceEventCfg

_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
_OSC_POSITION_SCALE = (0.025, 0.025, 0.010)
_OSC_ORIENTATION_SCALE = 0.025
_OSC_STIFFNESS = (300.0, 300.0, 300.0, 30.0, 30.0, 30.0)
_OSC_DAMPING_RATIO = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
_NEWTON_NUM_ENVS = 256
_NEWTON_MAX_TRIANGLE_PAIRS = 2**25
_FPGS_POSITION_ITERATIONS = 128


def _newton_sdf_properties(
    contact_offset: float,
    rest_offset: float,
    sdf_prim_paths: tuple[str, ...],
) -> dict[str, list[PhysxCollisionCfg | NewtonCollisionCfg | NewtonSDFCollisionCfg]]:
    """Create point-SDF properties only for meshes authored as SDF colliders."""
    properties: dict[str, list[PhysxCollisionCfg | NewtonCollisionCfg | NewtonSDFCollisionCfg]] = {
        "/.*": [PhysxCollisionCfg(contact_offset=contact_offset, rest_offset=rest_offset)]
    }
    for prim_path in sdf_prim_paths:
        properties[prim_path] = [
            NewtonCollisionCfg(
                # FeatherPGS resolves these from the source PhysX rest/contact
                # offsets. MJWarp retains the validated 5 mm point-SDF envelope.
                contact_margin=preset(default=0.0, feather_pgs=None),
                contact_gap=preset(default=0.005, feather_pgs=None),
            ),
            NewtonSDFCollisionCfg(
                sdf_max_resolution=256,
                sdf_narrow_band_inner=-0.005,
                sdf_narrow_band_outer=0.005,
                sdf_texture_format="uint16",
                sdf_padding=0.005,
                hydroelastic_enabled=False,
                hydroelastic_stiffness=1.0e8,
            ),
        ]
    return properties


@configclass
class DisplayportNewtonPhysicsCfg(PresetCfg):
    """Newton point-SDF physics profile for DisplayPort insertion."""

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
            # Scene-wide candidate-pair capacity for the 256-environment default.
            # If this overflows, Newton warns and may omit candidate contacts;
            # increase it when increasing environment count or mesh complexity.
            max_triangle_pairs=_NEWTON_MAX_TRIANGLE_PAIRS,
        ),
        num_substeps=20,
        collision_decimation=10,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        debug_mode=False,
        use_cuda_graph=True,
    )
    feather_pgs: NewtonCfg = NewtonCfg(
        solver_cfg=FeatherPGSSolverCfg(
            pgs_mode="matrix_free",
            update_mass_matrix_interval=1,
            enable_joint_limits=True,
            joint_limit_activation_gap=0.1,
            enable_joint_velocity_limits=True,
            velocity_limit_activation_fraction=0.7,
            pgs_iterations=_FPGS_POSITION_ITERATIONS,
            pgs_velocity_iterations=0,
            # The Robotiq linkage contains five mimic constraints. Projecting
            # them before contact solves keeps the passive fingers coupled to
            # the single driven joint under sustained grasp loads.
            enable_bilateral_preelimination=True,
            dense_max_constraints=256,
            # Per-world capacity for free-body contact and friction rows. FeatherPGS
            # drops rows on overflow, so contact-rich insertion needs explicit headroom.
            mf_max_constraints=2048,
            serial_kernel_block_dim=64,
            # Collect whole-run high-water marks for the bounded FPGS
            # stability diagnostic. Readback happens only when the physics
            # manager closes, outside CUDA graph capture.
            row_watermark=True,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            reduce_contacts=True,
            max_triangle_pairs=_NEWTON_MAX_TRIANGLE_PAIRS,
        ),
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
        """Actor observations in the 18-dimensional checkpoint ABI order."""

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
    # Keep every robot joint in the privileged critic to preserve the
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
    """Newton point-SDF OSC training configuration.

    The actor contract is ``socket_pos, flange_pos, flange_rot_6d,
    socket_rot_6d``. This differs from the TCP-first PhysX task-space contract,
    so the configurations must not share checkpoints despite both being 18-D.

    Override ``scene.robot.spawn.usd_path`` with a USD calibrated for the
    robot that will execute the policy.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # MJWarp uses 20 internal steps per 100 Hz tick. FeatherPGS instead
        # follows the PhysX task's 240 Hz tick and performs one PGS solve per
        # tick; its iteration count controls constraint convergence.
        self.sim.dt = preset(default=0.01, feather_pgs=1.0 / 240.0)
        self.sim.physics = DisplayportNewtonPhysicsCfg()
        # The collision candidate-pair capacity is scene-wide and supports this
        # per-rank default. Scale the capacity when increasing this value.
        self.scene.num_envs = _NEWTON_NUM_ENVS
        self.decimation = preset(default=3, feather_pgs=8)
        self.sim.render_interval = preset(default=3, feather_pgs=8)

        # Preserve source PhysX offsets on every collider, but apply Newton's
        # point-SDF schema only to meshes authored for SDF collision. Applying
        # it to convex-decomposition meshes is rejected by Newton 1.5.
        self.scene.dp_plug.spawn.collision_props = _newton_sdf_properties(
            0.00001,
            -0.00005,
            ("/collision_mesh",),
        )
        self.scene.dp_socket.spawn.collision_props = _newton_sdf_properties(
            0.0001,
            -0.0001,
            tuple(f"/tn__2584N111_DisplayportCord_jP/Body{body_id}/Mesh" for body_id in (5, 6, 8, 12, 13)),
        )

        self.observations = NewtonTaskSpaceObservationsCfg()
        self.task_space_obs_order = ["socket_pos", "tool_pos", "tool_rot_6d", "socket_rot_6d"]

        # RSL-RL clips raw actor outputs to +/-1 before OSC applies these scales.
        # Keep the action-term clip unset so the checkpoint contract has one
        # effective clipping stage and cannot acquire a latent +/-0.5 limit.
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
                # MJWarp cancels robot gravity through body gravcomp. FPGS
                # does not consume that MuJoCo-only field, so OSC supplies the
                # equivalent generalized gravity term for this preset.
                gravity_compensation=preset(default=False, feather_pgs=True),
                motion_stiffness_task=_OSC_STIFFNESS,
                motion_damping_ratio_task=_OSC_DAMPING_RATIO,
                nullspace_control="none",
            ),
            nullspace_joint_pos_target="none",
            clip=None,
            position_scale=_OSC_POSITION_SCALE,
            orientation_scale=_OSC_ORIENTATION_SCALE,
        )

        # Retain the fully wired base events and replace only Newton-specific
        # randomization terms. Replacing the group would duplicate grasp wiring.
        newton_events = NewtonTaskSpaceEventCfg()
        self.events.randomize_arm_joint_friction = newton_events.randomize_arm_joint_friction
        self.events.randomize_arm_pd_gains = None
        # FeatherPGS currently averages pair friction while the source task
        # requests multiplication. These mapped coefficients preserve the two
        # task-critical effective pairs: plug/socket=0.003 and plug/finger=3.0.
        plug_friction = preset(default=(3.0, 3.0), feather_pgs=(0.005, 0.005))
        finger_friction = preset(default=(1.0, 1.0), feather_pgs=(5.995, 5.995))
        self.events.plug_physics_material.params["static_friction_range"] = plug_friction
        self.events.plug_physics_material.params["dynamic_friction_range"] = plug_friction
        self.events.robot_physics_material.params["static_friction_range"] = finger_friction
        self.events.robot_physics_material.params["dynamic_friction_range"] = finger_friction

        # MJWarp consumes MuJoCo body gravcomp. FeatherPGS keeps gravity active
        # and receives exactly one compensation term from OSC instead.
        self.scene.robot.spawn.rigid_props = preset(
            default=sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
            feather_pgs=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
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
        )
        self.scene.robot.spawn.joint_drive_props = preset(
            default=sim_utils.MujocoJointDrivePropertiesCfg(actuatorgravcomp=False),
            feather_pgs=None,
        )

        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=preset(default=200.0, feather_pgs=2.0),
            velocity_limit_sim=preset(default=2.0, feather_pgs=1.0),
            stiffness=2000.0,
            damping=10.0,
            friction=0.0,
            armature=preset(default=0.1, feather_pgs=0.0),
        )
        self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_knuckle_joint", ".*_outer_finger_joint"],
            effort_limit_sim=preset(default=20.0, feather_pgs=1.0),
            velocity_limit_sim=1.0,
            stiffness=preset(default=2000.0, feather_pgs=0.0),
            damping=preset(default=10.0, feather_pgs=0.0),
            friction=0.0,
            armature=preset(default=0.05, feather_pgs=0.0),
        )
        self.hand_hold_width = preset(default=-0.1, feather_pgs=-0.05)
        self.hand_close_width = preset(default=-0.1, feather_pgs=-0.155)


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
