# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Deploy DisplayPort insertion environment configuration."""

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, VBDSolverCfg

from pxr import Usd

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.envs import mdp

from isaaclab_contrib.coupling import CouplerAdmmCfg, CouplerProxyCfg

import isaaclab_tasks.contrib.deploy.mdp as deploy_mdp
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.agents.rsl_rl_ppo_cfg import (
    Rizon4sGravDisplayportInsertionRNNPPORunnerCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.ik_newton_env_cfg import (
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangeObsEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DActionClip1EnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DScale015EnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DScale025EnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DScale015EnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DScale025EnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmPose6DEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcpObsEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DActionClip1EnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DArmFrictionDREnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale010ActionClip1EnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale010EnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale0125ActionClip1EnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale025ActionClip1EnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCInertialFlangePose6DEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCTaskImpedanceFlangePose6DEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCTcp15cmObsPose6DEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCTcp15cmPose6DEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedIKNewtonEnvCfg,
    Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonPhysXProfileEnvCfg,
    Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonTcp15cmPose6DEnvCfg,
    Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonTcp15cmPose6DScale0125EnvCfg,
    Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonTcp15cmPose6DScale015EnvCfg,
    Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCFlangePose6DEnvCfg,
    Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCPhysXProfileEnvCfg,
    Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCTcp15cmPose6DEnvCfg,
    Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCTcp15cmPose6DScale015EnvCfg,
    Rizon4sGravDisplayportInsertionIKNewtonEnvCfg,
    Rizon4sGravDisplayportInsertionIKNewtonFlangeObsEnvCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_pos_env_cfg import (
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNoJointVelEnvCfg,
    Rizon4sGravDisplayportInsertionCalibratedNoJointVelEnvCfg,
    Rizon4sGravDisplayportInsertionDomainRandomizedNoJointVelEnvCfg,
    Rizon4sGravDisplayportInsertionEnvCfg,
    Rizon4sGravDisplayportInsertionNoJointVelEnvCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    DISPLAY_ASSETS_DIR,
    SOCKET_INSERTION_OFFSET,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.events import compensate_articulation_body_gravity
from isaaclab_tasks.utils.hydra import resolve_presets


def test_displayport_newton_uses_full_insertion_target_and_physx_grasp():
    """Newton should preserve the PhysX full-insertion target and grasp pose."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_hydroelastic"})

    assert env_cfg.grasp_offset == [0.0025, 0.0, -0.1875]
    hydro_cfg = env_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config
    assert hydro_cfg is not None
    assert hydro_cfg.buffer_fraction == 0.125
    assert hydro_cfg.buffer_mult_broad == 8
    assert hydro_cfg.buffer_mult_iso == 2
    assert env_cfg.sim.physics.collision_cfg.max_triangle_pairs == 2**25
    assert env_cfg.scene.dp_plug.spawn.usd_path.endswith("display_port_plug_newton_hydroelastic.usda")
    assert env_cfg.scene.dp_socket.spawn.usd_path.endswith("display_port_socket_newton_hydroelastic.usda")
    assert env_cfg.observations.policy.socket_pos.params["offset"] == SOCKET_INSERTION_OFFSET
    assert env_cfg.observations.critic.socket_pos.params["offset"] == SOCKET_INSERTION_OFFSET
    assert env_cfg.rewards.plug_socket_keypoint_tracking.params["offset_1"] == SOCKET_INSERTION_OFFSET
    assert env_cfg.rewards.plug_socket_keypoint_tracking_exp.params["offset_1"] == SOCKET_INSERTION_OFFSET


def test_displayport_preserves_physx_default_and_exposes_newton_mjwarp():
    """Existing callers should keep PhysX while Newton MJWarp uses authored SDF assets."""
    default_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"default"})
    newton_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_mjwarp"})

    assert type(default_cfg.sim.physics).__name__ == "PhysxCfg"
    assert default_cfg.scene.robot.spawn.joint_drive_props is None
    assert default_cfg.scene.robot.actuators["shoulder"].stiffness == 1320.0
    assert default_cfg.scene.robot.actuators["elbow"].stiffness == 600.0
    assert default_cfg.scene.robot.actuators["wrist"].stiffness == 216.0
    assert default_cfg.sim.dt == 1.0 / 1000.0
    assert default_cfg.decimation == 33

    assert newton_cfg.sim.physics.solver_cfg.use_mujoco_contacts is True
    assert newton_cfg.scene.robot.spawn.joint_drive_props.actuatorgravcomp is False
    assert newton_cfg.scene.robot.actuators["shoulder"].stiffness == 6000.0
    assert newton_cfg.scene.robot.actuators["elbow"].stiffness == 4200.0
    assert newton_cfg.scene.robot.actuators["wrist"].stiffness == 1500.0
    assert newton_cfg.scene.dp_plug.spawn.usd_path.endswith("display_port_plug_newton_sdf.usda")
    assert newton_cfg.scene.dp_socket.spawn.usd_path.endswith("display_port_socket_newton_sdf.usda")


def test_displayport_physx_can_use_clean_no_protrusions_socket():
    """PhysX should support the same cleaned socket surface used by Newton."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"physx_noprotrusions"})

    assert type(env_cfg.sim.physics).__name__ == "PhysxCfg"
    assert env_cfg.scene.dp_plug.spawn.usd_path.endswith("display_port_plug_fixed_sdf.usd")
    assert env_cfg.scene.dp_socket.spawn.usd_path.endswith("display_port_socket_fixed_sdf_noprotrusions.usd")


def test_displayport_newton_disables_robot_gravity_passively():
    """Newton should cancel gravity on robot bodies without using actuator forces."""
    for preset_name in ("newton_mjwarp", "newton_sdf", "newton_hydroelastic"):
        env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {preset_name})
        rigid_props = env_cfg.scene.robot.spawn.rigid_props
        joint_drive_props = env_cfg.scene.robot.spawn.joint_drive_props

        assert type(rigid_props).__name__ == "MujocoRigidBodyPropertiesCfg"
        assert rigid_props.gravcomp == 1.0
        assert rigid_props.disable_gravity is None
        assert joint_drive_props.actuatorgravcomp is False
        assert env_cfg.scene.dp_plug.spawn.rigid_props.disable_gravity is False


def test_displayport_hard_sdf_uses_point_contacts_with_precomputed_sdfs():
    """Hard SDF should cook mesh volumes without enabling hydroelastic contact."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_sdf"})

    assert env_cfg.sim.physics.collision_cfg.max_triangle_pairs == 2**25
    assert env_cfg.scene.dp_plug.spawn.usd_path.endswith("display_port_plug_newton_sdf.usda")
    assert env_cfg.scene.dp_socket.spawn.usd_path.endswith("display_port_socket_newton_sdf.usda")
    assert env_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config is None
    assert env_cfg.sim.physics.solver_cfg.use_mujoco_contacts is False


def test_displayport_body_gravity_compensation_cancels_selected_body_weight():
    """The VBD compensator should apply ``-mass * gravity`` at each selected CoM."""
    captured = {}

    def capture_wrench(**kwargs):
        captured.update(kwargs)

    body_mass = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    asset = SimpleNamespace(
        num_instances=2,
        device="cpu",
        data=SimpleNamespace(body_mass=SimpleNamespace(torch=body_mass)),
        permanent_wrench_composer=SimpleNamespace(add_forces_and_torques_index=capture_wrench),
    )
    env = SimpleNamespace(scene={"robot": asset})
    asset_cfg = SimpleNamespace(name="robot", body_ids=[0, 2])

    compensate_articulation_body_gravity(
        env,
        torch.tensor([1]),
        asset_cfg=asset_cfg,
        gravity=(0.0, 0.0, -9.81),
    )

    expected = torch.tensor([[[0.0, 0.0, 39.24], [0.0, 0.0, 58.86]]])
    assert torch.allclose(captured["forces"], expected)
    assert captured["body_ids"] == [0, 2]
    assert captured["env_ids"].dtype == torch.int32
    assert captured["is_global"] is True


def test_displayport_all_vbd_uses_hard_point_sdf_contacts():
    """The all-VBD baseline should use hard point-SDF contacts and explicit gravity."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_vbd"})
    solver_cfg = env_cfg.sim.physics.solver_cfg

    assert isinstance(solver_cfg, VBDSolverCfg)
    assert solver_cfg.rigid_contact_hard is True
    assert solver_cfg.rigid_contact_history is False
    assert solver_cfg.rigid_body_contact_buffer_size == 1024
    assert env_cfg.sim.dt == 1.0 / 100.0
    assert env_cfg.decimation == 3
    assert env_cfg.sim.render_interval == 3
    assert env_cfg.sim.physics.num_substeps == 20
    assert env_cfg.sim.physics.collision_decimation == 10
    assert env_cfg.sim.physics.collision_decimation < env_cfg.sim.physics.num_substeps
    assert env_cfg.sim.physics.default_shape_cfg.gap == 0.005
    assert env_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config is None
    assert env_cfg.scene.robot.spawn.joint_drive_props is None
    assert env_cfg.scene.robot.spawn.rigid_props.disable_gravity is False
    gravity_event = env_cfg.events.robot_body_gravity_compensation
    assert gravity_event.func is compensate_articulation_body_gravity
    assert gravity_event.mode == "reset"
    assert gravity_event.params["gravity"] == (0.0, 0.0, -9.81)
    assert env_cfg.scene.dp_plug.spawn.usd_path.endswith("display_port_plug_newton_sdf.usda")
    assert env_cfg.scene.dp_socket.spawn.usd_path.endswith("display_port_socket_newton_sdf.usda")


def test_displayport_physx_profile_keeps_reference_gains_with_vbd_proxy():
    """The VBD proxy must not restore the stronger generic Newton arm gains."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonPhysXProfileEnvCfg(),
        {"newton_mjwarp_vbd_proxy"},
    )

    expected_gains = {
        "shoulder": (1320.0, 72.0),
        "elbow": (600.0, 35.0),
        "wrist": (216.0, 29.0),
    }
    for actuator_name, (stiffness, damping) in expected_gains.items():
        actuator_cfg = env_cfg.scene.robot.actuators[actuator_name]
        assert actuator_cfg.stiffness == stiffness
        assert actuator_cfg.damping == damping
    assert env_cfg.events.randomize_arm_pd_gains is None


def test_displayport_proxy_coupling_keeps_robot_in_mjwarp_and_contacts_in_vbd():
    """Selective proxy coupling should preserve MJWarp robot dynamics and VBD mating contacts."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_mjwarp_vbd_proxy"})
    solver_cfg = env_cfg.sim.physics.solver_cfg

    assert isinstance(solver_cfg, CouplerProxyCfg)
    entries = {entry.name: entry for entry in solver_cfg.entries}
    assert isinstance(entries["robot"].solver_cfg, MJWarpSolverCfg)
    assert entries["robot"].solver_cfg.disable_contacts is True
    assert entries["robot"].solver_cfg.use_mujoco_contacts is True
    assert entries["robot"].solver_cfg.update_data_interval == 10
    assert env_cfg.sim.dt == 1.0 / 100.0
    assert env_cfg.decimation == 3
    assert env_cfg.sim.render_interval == 3
    assert env_cfg.sim.physics.num_substeps == 20
    assert env_cfg.sim.physics.default_shape_cfg.gap == 0.005
    assert isinstance(entries["environment"].solver_cfg, VBDSolverCfg)
    assert entries["environment"].solver_cfg.rigid_contact_hard is True
    assert entries["environment"].solver_cfg.rigid_body_contact_buffer_size == 1024
    assert solver_cfg.proxies[0].source == "robot"
    assert solver_cfg.proxies[0].destination == "environment"
    assert solver_cfg.proxies[0].bodies == [r"/World/envs/env_[^/]+/Robot/Grav_gripper"]
    assert solver_cfg.proxies[0].mode == "staggered"
    assert solver_cfg.proxies[0].collide_interval == 10
    assert solver_cfg.proxies[0].collision_pipeline.sdf_hydroelastic_config is None
    assert env_cfg.sim.physics.collision_cfg is None
    assert env_cfg.scene.robot.spawn.rigid_props.gravcomp == 1.0


def test_displayport_admm_coupling_keeps_robot_in_mjwarp_and_contacts_in_vbd():
    """ADMM should symmetrically couple the MJWarp robot to the VBD environment."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_mjwarp_vbd_admm"})
    solver_cfg = env_cfg.sim.physics.solver_cfg

    assert isinstance(solver_cfg, CouplerAdmmCfg)
    entries = {entry.name: entry for entry in solver_cfg.entries}
    assert isinstance(entries["robot"].solver_cfg, MJWarpSolverCfg)
    assert isinstance(entries["environment"].solver_cfg, VBDSolverCfg)
    assert entries["robot"].include_body_shapes is False
    assert entries["robot"].shape_label_patterns == [r"/World/envs/env_[^/]+/Robot/Grav_gripper/.*"]
    assert solver_cfg.contact_pairs == [("robot", "environment")]
    assert solver_cfg.iterations == 2
    assert solver_cfg.rho == 50.0
    assert solver_cfg.gamma == 0.1
    assert solver_cfg.baumgarte == 0.01
    assert env_cfg.sim.dt == 1.0 / 100.0
    assert env_cfg.decimation == 3
    assert env_cfg.sim.render_interval == 3
    assert env_cfg.sim.physics.num_substeps == 20
    assert env_cfg.sim.physics.collision_decimation == 10
    assert env_cfg.sim.physics.collision_decimation < env_cfg.sim.physics.num_substeps
    assert env_cfg.sim.physics.default_shape_cfg.gap == 0.005
    assert solver_cfg.rigid_contact_matching == "latest"
    assert env_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config is None
    assert env_cfg.sim.physics.collision_cfg.rigid_contact_max == 2**20
    assert env_cfg.scene.robot.spawn.rigid_props.gravcomp == 1.0


def test_displayport_point_sdf_gap_presets_keep_newton_backend():
    """Gap asset presets must not fall back to the default PhysX backend."""
    for preset_name, suffix in (
        ("newton_sdf_gap_1mm", "gap_1mm.usda"),
        ("newton_sdf_gap_0p5mm", "gap_0p5mm.usda"),
    ):
        env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {preset_name})

        assert type(env_cfg.sim.physics).__name__ == "NewtonCfg"
        assert env_cfg.sim.physics.solver_cfg.use_mujoco_contacts is False
        assert env_cfg.scene.dp_plug.spawn.usd_path.endswith(suffix)
        assert env_cfg.scene.dp_socket.spawn.usd_path.endswith(suffix)


def test_displayport_assets_author_newton_sdf_per_active_collider():
    """Newton overlays should author SDF metadata only on active collision meshes."""
    socket_stage = Usd.Stage.Open(f"{DISPLAY_ASSETS_DIR}/display_port_socket_newton_sdf.usda")
    assert socket_stage.GetRootLayer().subLayerPaths == ["./display_port_socket_fixed_sdf_noprotrusions.usd"]

    expected_counts = {
        "display_port_plug_newton_sdf.usda": (1, False),
        "display_port_socket_newton_sdf.usda": (5, False),
        "display_port_plug_newton_hydroelastic.usda": (1, True),
        "display_port_socket_newton_hydroelastic.usda": (5, True),
    }
    for filename, (expected_count, hydroelastic_enabled) in expected_counts.items():
        stage = Usd.Stage.Open(f"{DISPLAY_ASSETS_DIR}/{filename}")
        sdf_prims = [prim for prim in stage.Traverse() if prim.HasAttribute("newton:sdfMaxResolution")]
        assert len(sdf_prims) == expected_count
        assert all(prim.GetAttribute("newton:sdfMaxResolution").Get() == 256 for prim in sdf_prims)
        assert all(abs(prim.GetAttribute("newton:contactGap").Get() - 0.005) < 1.0e-8 for prim in sdf_prims)
        assert all(prim.GetAttribute("newton:hydroelasticEnabled").Get() is hydroelastic_enabled for prim in sdf_prims)


def test_displayport_point_sdf_gap_overlays_override_every_mating_collider():
    """Gap ablations should override authored values on both sides of the mating pair."""
    expected_gaps = {"gap_1mm": 0.001, "gap_0p5mm": 0.0005}
    for suffix, expected_gap in expected_gaps.items():
        for asset_name, expected_count in (("plug", 1), ("socket", 5)):
            filename = f"display_port_{asset_name}_newton_sdf_{suffix}.usda"
            stage = Usd.Stage.Open(f"{DISPLAY_ASSETS_DIR}/{filename}")
            sdf_prims = [prim for prim in stage.Traverse() if prim.HasAttribute("newton:sdfMaxResolution")]

            assert len(sdf_prims) == expected_count
            assert all(abs(prim.GetAttribute("newton:contactGap").Get() - expected_gap) < 1.0e-8 for prim in sdf_prims)
            assert all(prim.GetAttribute("newton:hydroelasticEnabled").Get() is False for prim in sdf_prims)


def test_displayport_legacy_socket_comparison_preserves_newton_metadata():
    """The legacy A/B layer should differ only in its underlying socket USD."""
    stage = Usd.Stage.Open(f"{DISPLAY_ASSETS_DIR}/display_port_socket_newton_sdf_legacy.usda")
    assert stage.GetRootLayer().subLayerPaths == ["./display_port_socket_fixed_sdf_split_visuals.usd"]

    sdf_prims = [prim for prim in stage.Traverse() if prim.HasAttribute("newton:sdfMaxResolution")]
    assert len(sdf_prims) == 5
    assert all(prim.GetAttribute("newton:sdfMaxResolution").Get() == 256 for prim in sdf_prims)
    assert all(abs(prim.GetAttribute("newton:contactGap").Get() - 0.005) < 1.0e-8 for prim in sdf_prims)


def test_displayport_newton_ik_commands_flange_without_actor_velocity():
    """The deployment-oriented IK task should command the flange with a six-dimensional action."""
    env_cfg = Rizon4sGravDisplayportInsertionIKNewtonEnvCfg()
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert pose_objective.body_name == "flange"
    assert pose_objective.use_relative_mode is True
    assert pose_objective.scale == 0.01
    assert env_cfg.observations.policy.joint_vel is None


def test_displayport_newton_ik_flange_observation_replaces_actor_joint_state():
    """The flange-observation task should expose task state without actor joint state."""
    env_cfg = Rizon4sGravDisplayportInsertionIKNewtonFlangeObsEnvCfg()
    policy = env_cfg.observations.policy
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert policy.joint_pos is None
    assert policy.joint_vel is None
    assert policy.flange_pose.func is mdp.body_pose_w
    assert policy.flange_pose.params["asset_cfg"].name == "robot"
    assert policy.flange_pose.params["asset_cfg"].body_names == ["flange"]
    assert policy.socket_pos is not None
    assert policy.socket_quat is not None
    assert env_cfg.observations.critic.joint_pos is not None
    assert env_cfg.observations.critic.joint_vel is not None
    assert pose_objective.body_name == "flange"
    assert pose_objective.body_offset_pos == (0.0, 0.0, 0.0)


def test_displayport_calibrated_dr_flange_observation_preserves_randomization():
    """The calibrated flange-observation task should retain arm domain randomization."""
    env_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangeObsEnvCfg()

    assert env_cfg.observations.policy.joint_pos is None
    assert env_cfg.observations.policy.flange_pose is not None
    assert env_cfg.events.randomize_arm_joint_friction is not None
    assert env_cfg.events.randomize_arm_pd_gains is not None


def test_displayport_flange_pose_6d_observes_and_controls_flange_origin():
    """The flange pose-6D task should use the flange origin on both policy interfaces."""
    env_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DEnvCfg()
    policy = env_cfg.observations.policy
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert policy.joint_pos is None
    assert policy.joint_vel is None
    assert policy.socket_quat is None
    assert policy.tool_pos.func is deploy_mdp.body_pos_w_with_offset
    assert policy.tool_pos.params["offset"] == (0.0, 0.0, 0.0)
    assert policy.tool_rot_6d.func is deploy_mdp.body_rot_6d_w
    assert policy.socket_rot_6d.func is deploy_mdp.rigid_object_rot_6d_w
    assert pose_objective.body_name == "flange"
    assert pose_objective.body_offset_pos == (0.0, 0.0, 0.0)
    assert pose_objective.scale == 0.01


def test_displayport_tcp_15cm_pose_6d_observes_tcp_and_controls_flange():
    """The TCP pose-6D task should observe 150 mm TCP while controlling the flange."""
    env_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DEnvCfg()
    policy = env_cfg.observations.policy
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert policy.joint_pos is None
    assert policy.joint_vel is None
    assert policy.socket_quat is None
    assert policy.tool_pos.func is deploy_mdp.body_pos_w_with_offset
    assert policy.tool_pos.params["offset"] == (0.0, 0.0, 0.15)
    assert policy.tool_rot_6d.func is deploy_mdp.body_rot_6d_w
    assert policy.socket_rot_6d.func is deploy_mdp.rigid_object_rot_6d_w
    assert pose_objective.body_name == "flange"
    assert pose_objective.body_offset_pos == (0.0, 0.0, 0.0)
    assert pose_objective.scale == 0.01
    assert env_cfg.events.randomize_arm_joint_friction is not None
    assert env_cfg.events.randomize_arm_pd_gains is not None


def test_displayport_pose_6d_action_scale_variants():
    """Scale-specific tasks should preserve flange control for either observation frame."""
    variants = (
        (Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DScale015EnvCfg, 0.015),
        (Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DScale025EnvCfg, 0.025),
        (Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DScale015EnvCfg, 0.015),
        (Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmObsPose6DScale025EnvCfg, 0.025),
    )

    for cfg_type, expected_scale in variants:
        env_cfg = cfg_type()
        pose_objective = env_cfg.actions.arm_action.objectives[0]
        assert pose_objective.body_name == "flange"
        assert pose_objective.body_offset_pos == (0.0, 0.0, 0.0)
        assert pose_objective.scale == expected_scale


def test_displayport_task_space_full_action_range_variants():
    """Full-range tasks should use clip one without changing legacy task semantics."""
    legacy_ik = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DEnvCfg()
    full_ik = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DActionClip1EnvCfg()
    assert legacy_ik.actions.arm_action.clip == {".*": (-0.5, 0.5)}
    assert full_ik.actions.arm_action.clip == {".*": (-1.0, 1.0)}
    assert full_ik.actions.arm_action.objectives[0].scale == 0.01

    osc_variants = (
        (Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DActionClip1EnvCfg, 0.005),
        (Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale010ActionClip1EnvCfg, 0.01),
        (
            Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale0125ActionClip1EnvCfg,
            0.0125,
        ),
        (
            Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale025ActionClip1EnvCfg,
            0.025,
        ),
    )
    legacy_osc = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DEnvCfg()
    assert legacy_osc.actions.arm_action.clip == {".*": (-0.5, 0.5)}

    for cfg_type, expected_scale in osc_variants:
        action = cfg_type().actions.arm_action
        assert action.clip == {".*": (-1.0, 1.0)}
        assert action.position_scale == expected_scale
        assert action.orientation_scale == expected_scale


def test_displayport_newton_osc_uses_effort_control_without_arm_position_pd():
    """Newton OSC should command flange effort without a competing arm PD loop."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DEnvCfg(),
        {"newton_sdf"},
    )
    action = env_cfg.actions.arm_action

    assert action.body_name == "flange"
    assert action.body_offset.pos == (0.0, 0.0, 0.0)
    assert action.position_scale == 0.005
    assert action.orientation_scale == 0.005
    assert action.controller_cfg.inertial_dynamics_decoupling is True
    assert action.controller_cfg.partial_inertial_dynamics_decoupling is False
    assert action.controller_cfg.gravity_compensation is False
    assert action.controller_cfg.motion_stiffness_task == (300.0, 300.0, 300.0, 30.0, 30.0, 30.0)
    assert action.controller_cfg.motion_damping_ratio_task == (1.0,) * 6
    assert action.controller_cfg.nullspace_control == "none"
    assert action.nullspace_joint_pos_target == "none"
    assert env_cfg.osc_randomize_arm_joint_friction is False
    assert env_cfg.events.randomize_arm_joint_friction is None
    assert env_cfg.events.randomize_arm_pd_gains is None
    assert env_cfg.scene.robot.spawn.rigid_props.gravcomp == 1.0
    expected_limits = {"shoulder": (123.0, 2.094), "elbow": (64.0, 2.443), "wrist": (39.0, 4.887)}
    for actuator_name, (effort_limit, velocity_limit) in expected_limits.items():
        actuator_cfg = env_cfg.scene.robot.actuators[actuator_name]
        assert isinstance(actuator_cfg, IdealPDActuatorCfg)
        assert actuator_cfg.effort_limit == effort_limit
        assert actuator_cfg.effort_limit_sim == effort_limit
        assert actuator_cfg.velocity_limit == velocity_limit
        assert actuator_cfg.velocity_limit_sim == velocity_limit
        assert actuator_cfg.stiffness == 0.0
        assert actuator_cfg.damping == 0.0


def test_displayport_newton_ik_preserves_implicit_arm_position_actuators():
    """The OSC effort-actuator fix should not change Newton IK joint tracking."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonFlangePose6DEnvCfg(),
        {"newton_sdf"},
    )
    for actuator_name in ("shoulder", "elbow", "wrist"):
        assert isinstance(env_cfg.scene.robot.actuators[actuator_name], ImplicitActuatorCfg)


def test_displayport_newton_task_impedance_uses_safe_axis_normalized_gains():
    """Task impedance should use effort limits and inertia-normalized gains."""
    baseline_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DEnvCfg()
    task_impedance_cfg = (
        Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCTaskImpedanceFlangePose6DEnvCfg()
    )
    baseline = baseline_cfg.actions.arm_action
    task_impedance = task_impedance_cfg.actions.arm_action

    assert baseline.controller_cfg.inertial_dynamics_decoupling is True
    assert task_impedance.controller_cfg.inertial_dynamics_decoupling is False
    assert task_impedance.controller_cfg.partial_inertial_dynamics_decoupling is False
    assert task_impedance.controller_cfg.motion_stiffness_task == (50.0, 30.0, 60.0, 3.0, 8.0, 0.3)
    assert task_impedance.controller_cfg.motion_damping_ratio_task == (3.0, 2.25, 3.25, 0.3, 0.45, 0.055)
    assert task_impedance.controller_cfg.motion_stiffness_task != baseline.controller_cfg.motion_stiffness_task
    assert task_impedance.controller_cfg.motion_damping_ratio_task != baseline.controller_cfg.motion_damping_ratio_task
    assert task_impedance.position_scale == baseline.position_scale
    assert task_impedance.orientation_scale == baseline.orientation_scale
    for actuator_name in ("shoulder", "elbow", "wrist"):
        assert isinstance(task_impedance_cfg.scene.robot.actuators[actuator_name], IdealPDActuatorCfg)

    task_id = (
        "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-OSC-TaskImpedance-FlangePose6D"
    )
    assert (
        gym.spec(task_id)
        .kwargs["env_cfg_entry_point"]
        .endswith(":Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCTaskImpedanceFlangePose6DEnvCfg")
    )


def test_displayport_nominal_and_calibrated_osc_vary_robot_kinematics_only():
    """Nominal and calibrated OSC tasks should share policy and physics semantics."""
    nominal_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCFlangePose6DEnvCfg(),
        {"newton_sdf"},
    )
    calibrated_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DEnvCfg(),
        {"newton_sdf"},
    )

    assert nominal_cfg.scene.robot.spawn.usd_path.endswith("rizon4s_with_grav.usd")
    assert calibrated_cfg.scene.robot.spawn.usd_path.endswith("Rizon4s-063459_with_Grav_calibrated_kinematics.usd")
    assert nominal_cfg.scene.robot.spawn.usd_path != calibrated_cfg.scene.robot.spawn.usd_path

    for env_cfg in (nominal_cfg, calibrated_cfg):
        action = env_cfg.actions.arm_action
        policy = env_cfg.observations.policy

        assert env_cfg.scene.dp_socket.spawn.usd_path.endswith("display_port_socket_newton_sdf.usda")
        assert action.body_name == "flange"
        assert action.body_offset.pos == (0.0, 0.0, 0.0)
        assert action.position_scale == 0.005
        assert action.orientation_scale == 0.005
        assert action.controller_cfg.inertial_dynamics_decoupling is True
        assert policy.tool_pos.params["offset"] == (0.0, 0.0, 0.0)
        assert policy.tool_pos.func is deploy_mdp.body_pos_w_with_offset
        assert policy.tool_rot_6d.func is deploy_mdp.body_rot_6d_w
        assert policy.socket_rot_6d.func is deploy_mdp.rigid_object_rot_6d_w
        assert env_cfg.osc_randomize_arm_joint_friction is False
        assert env_cfg.events.randomize_arm_joint_friction is None
        assert env_cfg.events.randomize_arm_pd_gains is None


def test_displayport_newton_osc_arm_friction_dr_is_explicit_opt_in():
    """The calibrated OSC ablation should restore arm friction DR without arm PD-gain DR."""
    baseline_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DEnvCfg()
    friction_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DArmFrictionDREnvCfg()

    assert baseline_cfg.osc_randomize_arm_joint_friction is False
    assert baseline_cfg.events.randomize_arm_joint_friction is None
    assert friction_cfg.osc_randomize_arm_joint_friction is True

    friction_event = friction_cfg.events.randomize_arm_joint_friction
    assert friction_event is not None
    assert friction_event.mode == "reset"
    assert friction_event.params["asset_cfg"].joint_names == [f"joint{joint_id}" for joint_id in range(1, 8)]
    assert friction_event.params["friction_distribution_params"] == (0.0, 0.15)
    assert friction_event.params["operation"] == "add"
    assert friction_event.params["distribution"] == "uniform"

    for env_cfg in (baseline_cfg, friction_cfg):
        action = env_cfg.actions.arm_action
        assert env_cfg.events.randomize_arm_pd_gains is None
        assert action.body_name == "flange"
        assert action.position_scale == 0.005
        assert action.orientation_scale == 0.005
        assert action.controller_cfg.inertial_dynamics_decoupling is True
        for actuator_name in ("shoulder", "elbow", "wrist"):
            assert env_cfg.scene.robot.actuators[actuator_name].stiffness == 0.0
            assert env_cfg.scene.robot.actuators[actuator_name].damping == 0.0

    task_id = (
        "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-OSC-FlangePose6D-ArmFrictionDR"
    )
    env_cfg_entry_point = gym.spec(task_id).kwargs["env_cfg_entry_point"]
    assert env_cfg_entry_point.endswith(
        ":Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DArmFrictionDREnvCfg"
    )


def test_displayport_newton_osc_action_ablation_variants():
    """OSC ablations should vary only action scale or inverse dynamics."""
    scale_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DScale010EnvCfg()
    inertial_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCInertialFlangePose6DEnvCfg()

    assert scale_cfg.actions.arm_action.position_scale == 0.01
    assert scale_cfg.actions.arm_action.orientation_scale == 0.01
    assert scale_cfg.actions.arm_action.controller_cfg.inertial_dynamics_decoupling is True
    assert inertial_cfg.actions.arm_action.position_scale == 0.005
    assert inertial_cfg.actions.arm_action.orientation_scale == 0.005
    assert inertial_cfg.actions.arm_action.controller_cfg.inertial_dynamics_decoupling is True
    assert inertial_cfg.actions.arm_action.controller_cfg.motion_damping_ratio_task == (1.0,) * 6


def test_displayport_newton_osc_tcp_observation_still_controls_flange():
    """The OSC TCP-observation ablation should retain flange-origin actions."""
    env_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCTcp15cmObsPose6DEnvCfg()
    policy = env_cfg.observations.policy
    action = env_cfg.actions.arm_action

    assert policy.tool_pos.params["offset"] == (0.0, 0.0, 0.15)
    assert action.body_name == "flange"
    assert action.body_offset.pos == (0.0, 0.0, 0.0)


def test_displayport_newton_ik_physx_profile_matches_reference_interface():
    """The tuned IK task should match the measured PhysX action and arm-gain profile."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonPhysXProfileEnvCfg(),
        {"newton_sdf"},
    )
    policy = env_cfg.observations.policy
    action = env_cfg.actions.arm_action

    assert policy.tool_pos.params["offset"] == (0.0, 0.0, 0.1925)
    assert action.objectives[0].body_name == "flange"
    assert action.objectives[0].body_offset_pos == (0.0, 0.0, 0.0)
    assert action.objectives[0].scale == 0.025
    assert action.clip == {".*": (-0.4, 0.4)}
    assert env_cfg.scene.robot.actuators["shoulder"].stiffness == 1320.0
    assert env_cfg.scene.robot.actuators["shoulder"].damping == 72.0
    assert env_cfg.scene.robot.actuators["elbow"].stiffness == 600.0
    assert env_cfg.scene.robot.actuators["elbow"].damping == 35.0
    assert env_cfg.scene.robot.actuators["wrist"].stiffness == 216.0
    assert env_cfg.scene.robot.actuators["wrist"].damping == 29.0
    assert env_cfg.events.randomize_arm_pd_gains is None


def test_displayport_newton_osc_physx_profile_matches_reference_controller():
    """The tuned OSC task should use direct effort and the PhysX OSC profile."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCPhysXProfileEnvCfg(),
        {"newton_sdf"},
    )
    policy = env_cfg.observations.policy
    action = env_cfg.actions.arm_action
    controller = action.controller_cfg

    assert policy.tool_pos.params["offset"] == (0.0, 0.0, 0.1925)
    assert action.body_name == "flange"
    assert action.body_offset.pos == (0.0, 0.0, 0.0)
    assert action.position_scale == 0.025
    assert action.orientation_scale == 0.025
    assert action.clip == {".*": (-0.4, 0.4)}
    assert controller.inertial_dynamics_decoupling is True
    assert controller.partial_inertial_dynamics_decoupling is False
    assert controller.motion_stiffness_task == (300.0, 300.0, 300.0, 30.0, 30.0, 30.0)
    assert controller.motion_damping_ratio_task == (1.0,) * 6
    for actuator_name in ("shoulder", "elbow", "wrist"):
        assert env_cfg.scene.robot.actuators[actuator_name].stiffness == 0.0
        assert env_cfg.scene.robot.actuators[actuator_name].damping == 0.0


def test_displayport_calibrated_dr_tcp_observation_offsets_only_actor_pose():
    """The TCP observation should use a local offset while the action controls the flange origin."""
    env_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcpObsEnvCfg()
    policy = env_cfg.observations.policy
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert policy.joint_pos is None
    assert policy.joint_vel is None
    assert policy.tcp_pose.func is deploy_mdp.body_pose_w_with_offset
    assert policy.tcp_pose.params["asset_cfg"].body_names == ["flange"]
    assert policy.tcp_pose.params["offset"] == (0.0, 0.0, 0.1925)
    assert pose_objective.body_name == "flange"
    assert pose_objective.body_offset_pos == (0.0, 0.0, 0.0)
    assert env_cfg.events.randomize_arm_joint_friction is not None
    assert env_cfg.events.randomize_arm_pd_gains is not None


def test_displayport_calibrated_domain_randomized_newton_ik_preserves_training_configuration():
    """The calibrated DR task should change only the arm action to relative flange IK."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonEnvCfg(),
        {"newton_sdf"},
    )
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert Path(env_cfg.scene.robot.spawn.usd_path).name == "Rizon4s-063459_with_Grav_calibrated_kinematics.usd"
    assert pose_objective.body_name == "flange"
    assert pose_objective.use_relative_mode is True
    assert pose_objective.scale == 0.01
    assert env_cfg.observations.policy.joint_vel is None
    assert env_cfg.events.randomize_arm_joint_friction is not None
    assert env_cfg.events.randomize_arm_pd_gains is not None
    assert env_cfg.scene.robot.spawn.rigid_props.gravcomp == 1.0


def test_displayport_calibrated_newton_ik_omits_domain_randomization():
    """The calibrated IK task should preserve nominal arm dynamics."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionCalibratedIKNewtonEnvCfg(),
        {"newton_sdf"},
    )
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert Path(env_cfg.scene.robot.spawn.usd_path).name == "Rizon4s-063459_with_Grav_calibrated_kinematics.usd"
    assert pose_objective.body_name == "flange"
    assert env_cfg.observations.policy.joint_vel is None
    assert env_cfg.events.randomize_arm_joint_friction is None
    assert env_cfg.events.randomize_arm_pd_gains is None


def test_displayport_no_joint_velocity_hides_velocity_from_actor_only():
    """No-joint-velocity training should retain privileged critic velocity."""
    env_cfg = Rizon4sGravDisplayportInsertionNoJointVelEnvCfg()

    assert env_cfg.observations.policy.joint_vel is None
    assert env_cfg.observations.critic.joint_vel is not None


def test_displayport_calibrated_robot_preserves_newton_training_configuration():
    """Calibrated kinematics should change only the robot USD in the Newton task."""
    env_cfg = resolve_presets(
        Rizon4sGravDisplayportInsertionCalibratedNoJointVelEnvCfg(),
        {"newton_sdf"},
    )

    robot_usd_path = Path(env_cfg.scene.robot.spawn.usd_path)
    curriculum = env_cfg.events.reset_plug_curriculum.params

    assert robot_usd_path.name == "Rizon4s-063459_with_Grav_calibrated_kinematics.usd"
    assert robot_usd_path.is_file()
    robot_stage = Usd.Stage.Open(str(robot_usd_path))
    required_prim_paths = (
        "/Rizon4s/base_link",
        "/Rizon4s/link7",
        "/Rizon4s/flange",
        "/Rizon4s/joints/joint1",
        "/Rizon4s/joints/joint7",
        "/Rizon4s/Grav_gripper/gripper_base",
        "/Rizon4s/Grav_gripper/left_finger_tip",
        "/Rizon4s/Grav_gripper/right_finger_tip",
    )
    assert robot_stage.GetDefaultPrim().GetPath() == "/Rizon4s"
    assert all(robot_stage.GetPrimAtPath(path).IsValid() for path in required_prim_paths)
    assert env_cfg.scene.robot.spawn.rigid_props.gravcomp == 1.0
    assert env_cfg.scene.robot.spawn.joint_drive_props.actuatorgravcomp is False
    assert type(env_cfg.actions.arm_action).__name__ == "RelativeJointPositionActionCfg"
    assert env_cfg.observations.policy.joint_vel is None
    assert env_cfg.observations.critic.joint_vel is not None
    assert curriculum["at_goal_prob"] == 0.8
    assert curriculum["at_goal_prob_final"] == 0.0


def test_displayport_socket_observation_uses_reset_sampled_ten_millimeter_noise():
    """The actor should observe one reset-sampled socket-position offset per episode."""
    env_cfg = Rizon4sGravDisplayportInsertionCalibratedNoJointVelEnvCfg()

    noise_model = env_cfg.observations.policy.socket_pos.noise
    assert type(noise_model).__name__ == "ResetSampledConstantNoiseModelCfg"
    assert noise_model.noise_cfg.n_min == -0.01
    assert noise_model.noise_cfg.n_max == 0.01
    assert noise_model.noise_cfg.operation == "add"
    assert env_cfg.observations.critic.socket_pos.noise is None


def test_displayport_domain_randomization_targets_arm_joints():
    """Nominal and calibrated DR tasks should randomize only arm joints."""
    baseline_cfg = Rizon4sGravDisplayportInsertionCalibratedNoJointVelEnvCfg()
    env_cfgs = (
        Rizon4sGravDisplayportInsertionDomainRandomizedNoJointVelEnvCfg(),
        Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNoJointVelEnvCfg(),
    )

    assert baseline_cfg.events.randomize_arm_joint_friction is None
    assert baseline_cfg.events.randomize_arm_pd_gains is None

    for env_cfg in env_cfgs:
        friction = env_cfg.events.randomize_arm_joint_friction
        assert friction.mode == "reset"
        assert friction.params["asset_cfg"].joint_names == [f"joint{joint_id}" for joint_id in range(1, 8)]
        assert friction.params["friction_distribution_params"] == (0.0, 0.15)
        assert friction.params["operation"] == "add"
        assert friction.params["distribution"] == "uniform"

        gains = env_cfg.events.randomize_arm_pd_gains
        assert gains.mode == "reset"
        assert gains.params["asset_cfg"].joint_names == [f"joint{joint_id}" for joint_id in range(1, 8)]
        assert gains.params["stiffness_distribution_params"] == (0.8, 1.2)
        assert gains.params["damping_distribution_params"] == (0.8, 1.2)
        assert gains.params["operation"] == "scale"
        assert gains.params["distribution"] == "uniform"


def test_displayport_play_mode_uses_approach_resets():
    """Playback should not initialize the plug at the seated target."""
    env_cfg = Rizon4sGravDisplayportInsertionNoJointVelEnvCfg()
    env_cfg.play_mode()

    curriculum = env_cfg.events.reset_plug_curriculum.params
    assert curriculum["at_goal_prob"] == 0.0
    assert curriculum["at_goal_prob_final"] == 0.0
    assert env_cfg.observations.policy.enable_corruption is False


def test_displayport_newton_matches_reference_training_curriculum():
    """Newton should preserve the referenced PhysX curriculum and reward shaping."""
    env_cfg = resolve_presets(Rizon4sGravDisplayportInsertionEnvCfg(), {"newton_hydroelastic"})

    curriculum = env_cfg.events.reset_plug_curriculum.params
    assert curriculum["at_goal_prob"] == 0.8
    assert curriculum["at_goal_prob_final"] == 0.0
    assert curriculum["anneal_start_iter"] == 0.0
    assert curriculum["anneal_end_iter"] == 500.0
    assert curriculum["num_steps_per_env"] == 512
    assert curriculum["at_goal_depth_range"] == [0.0, 0.015]
    assert curriculum["approach_depth_range"] == [0.02, 0.06]

    linear_weight = env_cfg.rewards.plug_socket_keypoint_tracking.weight
    exponential = env_cfg.rewards.plug_socket_keypoint_tracking_exp
    assert exponential.weight == abs(linear_weight)
    assert exponential.params["kp_exp_coeffs"][-1] == (2000, 0.0001)

    assert Rizon4sGravDisplayportInsertionRNNPPORunnerCfg().max_iterations == 1500


def test_displayport_tcp_15cm_pose_6d_ik_observes_and_controls_tcp():
    """Matched TCP IK tasks should observe and command the same 150 mm frame."""
    variants = (
        (
            Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonTcp15cmPose6DEnvCfg,
            "rizon4s_with_grav.usd",
        ),
        (
            Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedIKNewtonTcp15cmPose6DEnvCfg,
            "Rizon4s-063459_with_Grav_calibrated_kinematics.usd",
        ),
    )

    for cfg_type, expected_robot_usd in variants:
        env_cfg = cfg_type()
        policy = env_cfg.observations.policy
        pose_objective = env_cfg.actions.arm_action.objectives[0]

        assert env_cfg.scene.robot.spawn.usd_path.endswith(expected_robot_usd)
        assert policy.tool_pos.params["offset"] == (0.0, 0.0, 0.15)
        assert pose_objective.body_name == "flange"
        assert pose_objective.body_offset_pos == (0.0, 0.0, 0.15)
        assert pose_objective.use_relative_mode is True
        assert pose_objective.scale == 0.01


def test_displayport_newton_osc_tcp_15cm_observes_and_controls_tcp():
    """Matched TCP OSC tasks should use the same 150 mm pose and Jacobian frame."""
    variants = (
        (
            Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCTcp15cmPose6DEnvCfg,
            "rizon4s_with_grav.usd",
        ),
        (
            Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCTcp15cmPose6DEnvCfg,
            "Rizon4s-063459_with_Grav_calibrated_kinematics.usd",
        ),
    )

    for cfg_type, expected_robot_usd in variants:
        env_cfg = cfg_type()
        policy = env_cfg.observations.policy
        action = env_cfg.actions.arm_action

        assert env_cfg.scene.robot.spawn.usd_path.endswith(expected_robot_usd)
        assert policy.tool_pos.params["offset"] == (0.0, 0.0, 0.15)
        assert action.body_name == "flange"
        assert action.body_offset.pos == (0.0, 0.0, 0.15)
        assert action.controller_cfg.target_types == ["pose_rel"]


def test_displayport_nominal_tcp_15cm_action_scale_variants():
    """Nominal matched-TCP tasks should expose the intended action-scale ablations."""
    ik_variants = (
        (Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonTcp15cmPose6DScale0125EnvCfg, 0.0125),
        (Rizon4sGravDisplayportInsertionDomainRandomizedIKNewtonTcp15cmPose6DScale015EnvCfg, 0.015),
    )

    for cfg_type, expected_scale in ik_variants:
        env_cfg = cfg_type()
        pose_objective = env_cfg.actions.arm_action.objectives[0]

        assert env_cfg.scene.robot.spawn.usd_path.endswith("rizon4s_with_grav.usd")
        assert env_cfg.observations.policy.tool_pos.params["offset"] == (0.0, 0.0, 0.15)
        assert pose_objective.body_offset_pos == (0.0, 0.0, 0.15)
        assert pose_objective.scale == expected_scale

    osc_cfg = Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCTcp15cmPose6DScale015EnvCfg()
    osc_action = osc_cfg.actions.arm_action

    assert osc_cfg.scene.robot.spawn.usd_path.endswith("rizon4s_with_grav.usd")
    assert osc_cfg.observations.policy.tool_pos.params["offset"] == (0.0, 0.0, 0.15)
    assert osc_action.body_offset.pos == (0.0, 0.0, 0.15)
    assert osc_action.position_scale == 0.015
    assert osc_action.orientation_scale == 0.015
