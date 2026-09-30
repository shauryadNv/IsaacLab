# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for DisplayPort signal and physics-property randomization."""

from types import SimpleNamespace

import torch

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils

from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.domain_rand import (
    _apply_contact_properties,
    _apply_joint_properties,
    _apply_observation_noise,
    _Rotation6DNoiseModel,
    _Rotation6DNoiseModelCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.domain_rand_cfg import DomainRandCfg
from isaaclab_tasks.contrib.deploy.mdp.dr_events import _randomize_joint_coulomb_friction


def _identity_rotation_6d(num_envs: int) -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(num_envs, 1)


def _raw_rotation_matrix(data: torch.Tensor) -> torch.Tensor:
    """Form a matrix directly from encoded rows without repairing invalid observations."""
    row_0 = data[..., 0:3]
    row_1 = data[..., 3:6]
    return torch.stack((row_0, row_1, torch.cross(row_0, row_1, dim=-1)), dim=-2)


def test_rotation_noise_stays_on_so3_and_preserves_episode_bias():
    """Angular bias must be stable within an episode and produce valid rotations."""
    torch.manual_seed(7)
    cfg = _Rotation6DNoiseModelCfg(bias_halfwidth=0.2, noise_halfwidth=0.0)
    model = _Rotation6DNoiseModel(cfg, num_envs=64, device="cpu")
    clean = _identity_rotation_6d(64)

    model.reset()
    first = model(clean)
    second = model(clean)

    torch.testing.assert_close(first, second)
    rotation = _raw_rotation_matrix(first)
    identity = torch.eye(3).expand(64, -1, -1)
    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), identity, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(64), atol=1.0e-6, rtol=1.0e-6)

    angle = torch.linalg.vector_norm(
        math_utils.axis_angle_from_quat(math_utils.quat_from_matrix(rotation)),
        dim=-1,
    )
    assert torch.all(angle <= 0.2 + 1.0e-6)
    assert not torch.allclose(first, clean)


def test_rotation_noise_reads_runtime_widened_jitter():
    """Changing the live config must affect per-step jitter without rebuilding the model."""
    torch.manual_seed(11)
    cfg = _Rotation6DNoiseModelCfg(bias_halfwidth=0.0, noise_halfwidth=0.0)
    model = _Rotation6DNoiseModel(cfg, num_envs=32, device="cpu")
    clean = _identity_rotation_6d(32)
    model.reset()

    torch.testing.assert_close(model(clean), clean)
    cfg.noise_halfwidth = 0.1
    first = model(clean)
    second = model(clean)

    assert not torch.allclose(first, clean)
    assert not torch.allclose(first, second)
    for noisy in (first, second):
        rotation = _raw_rotation_matrix(noisy)
        identity = torch.eye(3).expand(32, -1, -1)
        torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), identity, atol=1.0e-6, rtol=1.0e-6)
        torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(32), atol=1.0e-6, rtol=1.0e-6)


def test_rotation_observation_knob_uses_angular_noise_model():
    """Rotation knobs must track angular bounds rather than 6D component bounds."""
    dr = DomainRandCfg()
    dr.obs_eef_rot.enable = True
    dr.obs_eef_rot.bias_initial = 0.01
    dr.obs_eef_rot.bias_final = 0.03
    dr.obs_eef_rot.noise_initial = 0.001
    dr.obs_eef_rot.noise_final = 0.004
    policy = SimpleNamespace(
        enable_corruption=False,
        eef_rot_6d=SimpleNamespace(noise=None),
    )
    env_cfg = SimpleNamespace(observations=SimpleNamespace(policy=policy))
    tracked = []

    _apply_observation_noise(dr, env_cfg, lambda address, initial, final: tracked.append((address, initial, final)))

    assert isinstance(policy.eef_rot_6d.noise, _Rotation6DNoiseModelCfg)
    assert tracked == [
        ("observations.policy.eef_rot_6d.noise.bias_halfwidth", 0.01, 0.03),
        ("observations.policy.eef_rot_6d.noise.noise_halfwidth", 0.001, 0.004),
    ]


class _FakeArticulation:
    def __init__(self):
        self.device = "cpu"
        self.num_joints = 3
        self.writes = []

    def write_joint_friction_coefficient_to_sim_index(self, **kwargs):
        self.writes.append(kwargs)


def test_physx_joint_friction_randomizes_dry_terms_not_viscous():
    """The friction knob must not write a dry-friction range into viscous damping."""
    asset = _FakeArticulation()
    asset_cfg = SceneEntityCfg("robot", joint_ids=[0, 2])
    scene = {"robot": asset}
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=scene,
        sim=SimpleNamespace(physics_manager=type("PhysxPhysicsManager", (), {})),
    )
    cfg = EventTermCfg(
        func=_randomize_joint_coulomb_friction,
        mode="reset",
        params={
            "asset_cfg": asset_cfg,
            "friction_distribution_params": (0.2, 0.4),
        },
    )
    term = _randomize_joint_coulomb_friction(cfg, env)

    term(env, torch.tensor([0, 1]), asset_cfg, (0.2, 0.4))

    assert len(asset.writes) == 1
    write = asset.writes[0]
    torch.testing.assert_close(write["joint_dynamic_friction_coeff"], write["joint_friction_coeff"])
    assert "joint_viscous_friction_coeff" not in write
    assert torch.all((write["joint_friction_coeff"] >= 0.2) & (write["joint_friction_coeff"] <= 0.4))


def test_joint_knobs_use_separate_unit_correct_events():
    """Armature and dry friction must not share the stock bundled friction event."""
    dr = DomainRandCfg()
    dr.joint_armature.enable = True
    dr.joint_friction.enable = True
    env_cfg = SimpleNamespace(
        actions=SimpleNamespace(arm_action=SimpleNamespace(joint_names=["joint1", "joint2"])),
    )
    events = SimpleNamespace()
    tracked = []

    _apply_joint_properties(dr, env_cfg, events, lambda address, initial, final: tracked.append(address))

    assert events.randomize_joint_friction.func is _randomize_joint_coulomb_friction
    assert "friction_distribution_params" not in events.randomize_joint_armature.params
    assert "events.randomize_joint_armature.params.armature_distribution_params" in tracked
    assert "events.randomize_joint_friction.params.friction_distribution_params" in tracked


def test_contact_material_randomization_enforces_consistent_friction():
    """Sampled dynamic friction must never exceed sampled static friction."""
    dr = DomainRandCfg()
    dr.finger_friction.enable = True
    events = SimpleNamespace(
        robot_physics_material=SimpleNamespace(func=None, mode="startup", params={}),
    )

    _apply_contact_properties(dr, events, lambda *args: None)

    assert events.robot_physics_material.params["make_consistent"] is True
