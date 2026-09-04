# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for DisplayPort post-insertion retention rewards."""

import math
from types import SimpleNamespace

import torch

from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.ik_newton_env_cfg import (
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DArmFrictionDREnvCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_pos_env_cfg import (
    Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNoJointVelEnvCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.rewards import (
    _normalized_outward_action,
    _pose_is_stable,
    _update_success_latch,
    post_success_retraction_action_l2,
)


def test_pose_stability_checks_position_and_orientation() -> None:
    socket_pos = torch.zeros(3, 3)
    plug_pos = torch.tensor([[0.002, 0.0, 0.0], [0.004, 0.0, 0.0], [0.002, 0.0, 0.0]])
    socket_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)
    half_angle = math.radians(6.0) / 2.0
    plug_quat = socket_quat.clone()
    plug_quat[2] = torch.tensor([0.0, 0.0, math.sin(half_angle), math.cos(half_angle)])

    stable = _pose_is_stable(socket_pos, socket_quat, plug_pos, plug_quat, 0.003, math.radians(5.0))

    assert stable.tolist() == [True, False, False]


def test_success_latch_requires_consecutive_dwell_and_remains_sticky() -> None:
    count = torch.zeros(2, dtype=torch.long)
    latched = torch.zeros(2, dtype=torch.bool)

    for stable in (
        torch.tensor([True, True]),
        torch.tensor([False, True]),
        torch.tensor([True, True]),
    ):
        count, latched = _update_success_latch(stable, count, latched, dwell_steps=3)

    assert count.tolist() == [1, 3]
    assert latched.tolist() == [False, True]

    count, latched = _update_success_latch(torch.tensor([False, False]), count, latched, dwell_steps=3)
    assert count.tolist() == [0, 0]
    assert latched.tolist() == [False, True]


def test_outward_action_projection_preserves_sign_and_ignores_lateral_motion() -> None:
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)
    actions = torch.tensor([[0.010, 0.0, 0.0], [-0.010, 0.0, 0.0], [0.0, 0.025, 0.0]])
    axis = torch.tensor([1.0, 0.0, 0.0])
    scale = torch.tensor([0.010, 0.025, 0.025])

    projection = _normalized_outward_action(actions, identity, identity, axis, scale)

    torch.testing.assert_close(projection, torch.tensor([1.0, -1.0, 0.0]))


def test_outward_action_projection_is_frame_invariant() -> None:
    half_angle = math.pi / 4.0
    rotate_z_90 = torch.tensor([[0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]])
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    axis = torch.tensor([1.0, 0.0, 0.0])
    scale = torch.tensor([0.025, 0.010, 0.025])

    baseline = _normalized_outward_action(torch.tensor([[0.025, 0.0, 0.0]]), identity, identity, axis, scale)
    rotated_world = _normalized_outward_action(torch.tensor([[0.025, 0.0, 0.0]]), rotate_z_90, rotate_z_90, axis, scale)
    socket_rotated_relative_to_robot = _normalized_outward_action(
        torch.tensor([[0.0, 0.010, 0.0]]), rotate_z_90, identity, axis, scale
    )

    torch.testing.assert_close(rotated_world, baseline)
    torch.testing.assert_close(socket_rotated_relative_to_robot, baseline)


def test_retention_terms_are_optional_and_osc_specific() -> None:
    osc_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DArmFrictionDREnvCfg()
    joint_cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNoJointVelEnvCfg()

    assert osc_cfg.rewards.stable_insertion.weight == 0.0
    assert osc_cfg.rewards.post_success_retraction_action.weight == 0.0
    assert not hasattr(joint_cfg.rewards, "stable_insertion")
    assert not hasattr(joint_cfg.rewards, "post_success_retraction_action")

    osc_cfg.rewards.stable_insertion.weight = 1.0
    osc_cfg.rewards.post_success_retraction_action.weight = -0.5
    assert osc_cfg.rewards.stable_insertion.weight == 1.0
    assert osc_cfg.rewards.post_success_retraction_action.weight == -0.5


def test_retraction_term_uses_final_action_scale_and_resets_selected_environments() -> None:
    cfg = Rizon4sGravDisplayportInsertionCalibratedDomainRandomizedNewtonOSCFlangePose6DArmFrictionDREnvCfg()
    cfg.actions.arm_action.position_scale = [0.025, 0.025, 0.010]
    action_term = SimpleNamespace(
        cfg=cfg.actions.arm_action,
        processed_actions=torch.zeros(3, 6),
    )
    identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)
    zero_pos = torch.zeros(3, 3)
    rigid_data = SimpleNamespace(
        root_link_pos_w=SimpleNamespace(torch=zero_pos),
        root_link_quat_w=SimpleNamespace(torch=identity_quat),
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene={
            "dp_socket": SimpleNamespace(data=rigid_data),
            "dp_plug": SimpleNamespace(data=rigid_data),
            "robot": SimpleNamespace(data=rigid_data),
        },
        action_manager=SimpleNamespace(get_term=lambda name: action_term),
    )

    term = post_success_retraction_action_l2(cfg.rewards.post_success_retraction_action, env)
    torch.testing.assert_close(term._position_scale, torch.tensor([0.025, 0.025, 0.010]))

    term._dwell_count[:] = torch.tensor([1, 2, 3])
    term._latched[:] = True
    term.reset(torch.tensor([1]))
    assert term._dwell_count.tolist() == [1, 0, 3]
    assert term._latched.tolist() == [True, False, True]
