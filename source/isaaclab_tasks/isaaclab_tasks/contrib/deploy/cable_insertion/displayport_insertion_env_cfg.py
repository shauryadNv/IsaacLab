# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base RL environment for inserting a DisplayPort plug into a socket.

This mirrors :mod:`cable_insertion_env_cfg` (the GB300 cable-insertion base)
but targets the right-angle DisplayPort plug/socket assets in
``display_cable_insertion_assets``.

The assets were produced from STEP source via the
``omniverse-cad-to-simready`` pipeline and post-processed offline. Geometry is
stored in metres and each asset has a single rigid-body root. The socket USD
separates its visible shell from its SDF collider prims because Newton does not
render a mesh that is also used as a collider when another visual shape exists
on the same body.

Geometry constants below were derived from the authored seated pose
of the drop-test (plug pos ``(0,0,0.2096)`` rot ``(0.70711,0.70711,0,0)``;
socket pos ``(0,0,0.15)`` rot ``(0.5,0.5,0.5,-0.5)`` in Isaac Lab quat order),
re-parameterized through the same quaternion helpers as the GB300 base so the
keypoint goal reproduces the exact verified mate used by the PhysX task.
"""

import os
from dataclasses import MISSING

from isaaclab_newton.physics import (
    HydroelasticSDFCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
)
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import isaaclab_tasks.contrib.deploy.mdp as mdp
from isaaclab_tasks.contrib.deploy.mdp.noise_models import ResetSampledConstantNoiseModelCfg
from isaaclab_tasks.utils import PresetCfg, preset

CABLE_INSERTION_DIR = os.path.dirname(os.path.abspath(__file__))
DISPLAY_ASSETS_DIR = os.path.join(CABLE_INSERTION_DIR, "display_cable_insertion_assets")

# A 256-world GPU shard reached 25.8M point-SDF triangle pairs and 26.5M
# hydroelastic broad-phase block pairs during randomized rollouts.
_DISPLAYPORT_MAX_TRIANGLE_PAIRS = 2**25
_DISPLAYPORT_HYDRO_BUFFER_FRACTION = 0.125
_DISPLAYPORT_HYDRO_BUFFER_MULT_BROAD = 8

# ---------------------------------------------------------------------------
# Pure-python quaternion helpers (for module-level constant computation)
# ---------------------------------------------------------------------------


def _quat_rotate_vec(q_xyzw, v):
    """Apply quaternion rotation to a 3D vector."""
    qx, qy, qz, qw = q_xyzw
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _quat_mul(q1_xyzw, q2_xyzw):
    """Multiply two quaternions in ``(x, y, z, w)`` format."""
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


# ---------------------------------------------------------------------------
# USD body-frame offsets (DisplayPort asset geometry)
# ---------------------------------------------------------------------------
# Root -> insertion(mate) point in each asset's local frame. Derived from the
# verified seated pose with the mating reference point chosen at the socket top
# face. See module docstring; round-trip-verified against the GB300 helpers.
SOCKET_INSERTION_OFFSET = [0.0375, 0.0, 0.0]
PLUG_INSERTION_OFFSET = [0.0, 0.0, 0.0221]
# Plug orientation relative to socket at the mated pose, (x, y, z, w).
PLUG_GOAL_ROT = [0.0, -0.70711, 0.0, 0.70711]
PLUG_GOAL_ROT_INV = [0.0, 0.70711, 0.0, 0.70711]


def compute_socket_root(geometry_pos, socket_rot):
    """Compute socket USD root position from a desired insertion-geometry world position.

    Inverts :data:`SOCKET_INSERTION_OFFSET` (expressed in the socket's local
    frame) for a given world-frame socket rotation.
    """
    rotated = _quat_rotate_vec(socket_rot, SOCKET_INSERTION_OFFSET)
    return (
        geometry_pos[0] - rotated[0],
        geometry_pos[1] - rotated[1],
        geometry_pos[2] - rotated[2],
    )


def compute_plug_pose(geometry_pos, socket_rot, z_clearance=0.0):
    """Compute plug USD root position and world-frame rotation.

    Returns ``(plug_root_pos, plug_rot)`` such that the plug insertion point
    lands at ``geometry_pos`` (plus optional vertical clearance) with the
    correct goal orientation relative to the socket.
    """
    plug_rot = _quat_mul(socket_rot, tuple(PLUG_GOAL_ROT))
    plug_offset_world = _quat_rotate_vec(plug_rot, PLUG_INSERTION_OFFSET)
    plug_root = (
        geometry_pos[0] - plug_offset_world[0],
        geometry_pos[1] - plug_offset_world[1],
        geometry_pos[2] - plug_offset_world[2] + z_clearance,
    )
    return plug_root, plug_rot


# ---------------------------------------------------------------------------
# Default socket/plug workspace positions (drop-test layout, socket opening up)
# ---------------------------------------------------------------------------
_INSERTION_POINT = [0.0, 0.0, 0.1875]
_DEFAULT_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)

_SOCKET_ROOT_POS = compute_socket_root(_INSERTION_POINT, _DEFAULT_SOCKET_ROT)
_PLUG_ROOT_POS, _DEFAULT_PLUG_ROT = compute_plug_pose(
    _INSERTION_POINT,
    _DEFAULT_SOCKET_ROT,
    z_clearance=0.033,
)


##
# Asset Configurations
##


@configclass
class DisplayPortPlug(RigidObjectCfg):
    """DisplayPort right-angle plug (held asset) - dynamic."""

    prim_path = "{ENV_REGEX_NS}/DisplayPortPlug"
    spawn = sim_utils.UsdFileCfg(
        usd_path=preset(
            default=os.path.join(DISPLAY_ASSETS_DIR, "display_port_plug_fixed_sdf.usd"),
            newton_mjwarp=os.path.join(DISPLAY_ASSETS_DIR, "display_port_plug_newton_sdf.usda"),
            newton_sdf=os.path.join(DISPLAY_ASSETS_DIR, "display_port_plug_newton_sdf.usda"),
            newton_hydroelastic=os.path.join(DISPLAY_ASSETS_DIR, "display_port_plug_newton_hydroelastic.usda"),
        ),
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=False,
            # Gentle depenetration: high values turn residual mate overlap into
            # an explosive ejection. 0.5 lets PhysX resolve overlaps smoothly.
            max_depenetration_velocity=0.5,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=1,
            # Leave uncapped at the PhysX default; pairing a cap with a high
            # depenetration velocity amplified contact blow-ups.
            max_contact_impulse=None,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        # Body5 clearance to blade is ~0.27mm. PhysX fires contact at (plug + socket) offsets combined
        # (0.27mm physical gap). Keep plug offset small so there's real clearance before repulsion starts.
        collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.00001, rest_offset=-0.00005),
    )
    init_state = RigidObjectCfg.InitialStateCfg(pos=_PLUG_ROOT_POS, rot=_DEFAULT_PLUG_ROT)


@configclass
class DisplayPortSocket(RigidObjectCfg):
    """DisplayPort socket (fixed asset) - kinematic."""

    prim_path = "{ENV_REGEX_NS}/DisplayPortSocket"
    spawn = sim_utils.UsdFileCfg(
        usd_path=preset(
            default=os.path.join(DISPLAY_ASSETS_DIR, "display_port_socket_fixed_sdf_split_visuals.usd"),
            newton_mjwarp=os.path.join(DISPLAY_ASSETS_DIR, "display_port_socket_newton_sdf.usda"),
            newton_sdf=os.path.join(DISPLAY_ASSETS_DIR, "display_port_socket_newton_sdf.usda"),
            newton_hydroelastic=os.path.join(DISPLAY_ASSETS_DIR, "display_port_socket_newton_hydroelastic.usda"),
        ),
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=False,
        rigid_props=PhysxRigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=None),
        # rest_offset negative on socket too: combined rest = -0.15mm so blade can slide in.
        collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.0001, rest_offset=-0.0001),
    )
    init_state = RigidObjectCfg.InitialStateCfg(pos=_SOCKET_ROOT_POS, rot=_DEFAULT_SOCKET_ROT)


##
# Environment configuration
##


@configclass
class DisplayportInsertionPhysicsCfg(PresetCfg):
    """Physics backend presets for DisplayPort insertion.

    The ``default`` and ``physx`` presets preserve the validated PhysX behavior.
    ``newton_mjwarp`` uses MuJoCo contacts, while ``newton_sdf`` and
    ``newton_hydroelastic`` use Newton's
    collision pipeline and the SDF colliders authored in the DisplayPort USD
    assets. The point-SDF preset emits reduced point contacts, while the
    hydroelastic preset emits distributed contact patches.
    """

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=1024,
            nconmax=1024,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=True,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    newton_hydroelastic: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=4096,
            nconmax=4096,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            max_triangle_pairs=_DISPLAYPORT_MAX_TRIANGLE_PAIRS,
            sdf_hydroelastic_config=HydroelasticSDFCfg(
                reduce_contacts=True,
                buffer_fraction=_DISPLAYPORT_HYDRO_BUFFER_FRACTION,
                buffer_mult_broad=_DISPLAYPORT_HYDRO_BUFFER_MULT_BROAD,
                normal_matching=True,
            ),
        ),
        num_substeps=2,
        debug_mode=False,
    )
    newton_sdf: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=4096,
            nconmax=4096,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            reduce_contacts=True,
            max_triangle_pairs=_DISPLAYPORT_MAX_TRIANGLE_PAIRS,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    physx: PhysxCfg = PhysxCfg(
        gpu_collision_stack_size=2**30,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
    )
    default = physx


@configclass
class DisplayportInsertionSceneCfg(InteractiveSceneCfg):
    """Configuration for the DisplayPort insertion scene."""

    replicate_physics = True

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dp_plug = DisplayPortPlug()
    dp_socket = DisplayPortSocket()

    robot: ArticulationCfg = MISSING

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        socket_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket"), "offset": SOCKET_INSERTION_OFFSET},
            noise=ResetSampledConstantNoiseModelCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
                # noise_cfg=UniformNoiseCfg(n_min=-0.00, n_max=0.00, operation="add")
            ),
        )
        socket_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        socket_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket"), "offset": SOCKET_INSERTION_OFFSET},
        )
        socket_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )
        plug_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_plug"), "offset": PLUG_INSERTION_OFFSET},
        )
        plug_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("dp_plug")},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    plug_socket_keypoint_tracking = RewTerm(
        func=mdp.keypoint_two_body_error,
        weight=-1.5,
        params={
            "asset_cfg_1": SceneEntityCfg("dp_socket"),
            "asset_cfg_2": SceneEntityCfg("dp_plug"),
            "offset_1": SOCKET_INSERTION_OFFSET,
            "offset_2": PLUG_INSERTION_OFFSET,
            "rot_offset_2": PLUG_GOAL_ROT_INV,
            "keypoint_scale": 0.15,
        },
    )

    plug_socket_keypoint_tracking_exp = RewTerm(
        func=mdp.keypoint_two_body_error_exp,
        weight=1.5,
        params={
            "asset_cfg_1": SceneEntityCfg("dp_socket"),
            "asset_cfg_2": SceneEntityCfg("dp_plug"),
            "offset_1": SOCKET_INSERTION_OFFSET,
            "offset_2": PLUG_INSERTION_OFFSET,
            "rot_offset_2": PLUG_GOAL_ROT_INV,
            "kp_exp_coeffs": [(50, 0.0001), (300, 0.0001), (600, 0.0001), (2000, 0.0001)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-06)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class DisplayportInsertionEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for DisplayPort plug/socket insertion."""

    scene: DisplayportInsertionSceneCfg = DisplayportInsertionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    log_success_metrics: bool = True
    """Whether to log DisplayPort insertion success metrics."""

    success_socket_asset: str = "dp_socket"
    """Scene asset containing the fixed socket."""

    success_plug_asset: str = "dp_plug"
    """Scene asset containing the held plug."""

    success_pos_threshold: float = 0.003
    """Maximum mate-frame origin error [m] counted as success."""

    success_keypoint_scale: float = 0.15
    """Keypoint diagnostic scale [m]."""

    success_socket_offset: list[float] = MISSING
    """Socket-local mate-frame position [m]."""

    success_plug_offset: list[float] = MISSING
    """Plug-local mate-frame position [m]."""

    success_plug_goal_rot_inv: list[float] = MISSING
    """Inverse goal rotation from the plug mate frame to the socket mate frame."""

    physics_watchdog_enabled: bool = False
    """Whether to publish plug-state stability metrics."""

    physics_watchdog_fail_fast: bool = False
    """Whether persistent watchdog violations stop the environment."""

    physics_watchdog_insertion_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Insertion direction in the socket mate frame, expressed as a unit vector."""

    physics_watchdog_max_overtravel: float = 0.003
    """Maximum allowed plug travel past the seated mate plane [m]."""

    physics_watchdog_max_plug_linear_speed: float = 2.0
    """Maximum allowed plug linear speed [m/s]."""

    physics_watchdog_max_plug_angular_speed: float = 50.0
    """Maximum allowed plug angular speed [rad/s]."""

    physics_watchdog_max_violation_fraction: float = 0.05
    """Maximum fraction of environments that may violate a watchdog limit."""

    physics_watchdog_check_interval: int = 8
    """Number of policy steps between watchdog host-side checks."""

    physics_watchdog_consecutive_checks: int = 3
    """Number of consecutive failed checks before training is stopped."""

    sim: SimulationCfg = SimulationCfg(
        physics_material=PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physics=DisplayportInsertionPhysicsCfg(),
    )

    def __post_init__(self):
        """Post initialization."""
        self.episode_length_s = 6.66
        self.viewer.eye = (0.5, -1.8, 1.2)
        self.viewer.lookat = (0.5, 0.0, 0.5)
        self.decimation = 33
        self.sim.render_interval = self.decimation
        self.sim.dt = 1.0 / 1000.0

        # Keep the metric geometry identical to the keypoint-tracking reward.
        self.success_socket_offset = list(SOCKET_INSERTION_OFFSET)
        self.success_plug_offset = list(PLUG_INSERTION_OFFSET)
        self.success_plug_goal_rot_inv = list(PLUG_GOAL_ROT_INV)
