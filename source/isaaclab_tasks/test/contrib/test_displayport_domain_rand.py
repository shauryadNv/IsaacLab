# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for DisplayPort reset-state domain randomization."""

import importlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.domain_rand import _apply_reset_state
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.domain_rand_cfg import DomainRandCfg

deploy_events = importlib.import_module("isaaclab_tasks.contrib.deploy.mdp.events")


class _FakeRobot:
    """Minimal robot surface needed by the grasp-reset event."""

    def __init__(self, num_envs: int):
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(num_envs, 1, 1)
        self.data = SimpleNamespace(
            joint_pos=torch.zeros(num_envs, 1),
            joint_vel=torch.zeros(num_envs, 1),
            joint_pos_limits=torch.tensor([[[-1.0, 1.0]]]).repeat(num_envs, 1, 1),
            body_pos_w=torch.zeros(num_envs, 1, 3),
            body_quat_w=identity_quat,
        )
        self.root_view = SimpleNamespace(get_jacobians=lambda: torch.zeros(num_envs, 1, 6, 1))

    def set_joint_position_target_index(self, **kwargs):
        pass

    def set_joint_velocity_target_index(self, **kwargs):
        pass

    def write_joint_position_to_sim_index(self, **kwargs):
        pass

    def write_joint_velocity_to_sim_index(self, **kwargs):
        pass


class _FakeObject:
    """Minimal rigid-object surface needed by the grasp-reset event."""

    def __init__(self, num_envs: int):
        self.data = SimpleNamespace(
            root_link_pos_w=torch.zeros(num_envs, 3),
            root_link_quat_w=torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(num_envs, 1),
        )

    def write_root_pose_to_sim(self, *args, **kwargs):
        pass

    def write_root_velocity_to_sim(self, *args, **kwargs):
        pass


def test_grasp_position_randomization_is_sampled_once_per_reset(monkeypatch):
    """Multiple IK iterations must solve toward one sampled grasp pose."""
    num_envs = 2
    robot = _FakeRobot(num_envs)
    held_object = _FakeObject(num_envs)
    env = SimpleNamespace(device="cpu", scene={"dp_plug": held_object})

    term = object.__new__(deploy_events.set_robot_to_object_grasp_pose)
    term.robot_asset = robot
    term.target_object_name = "dp_plug"
    term.eef_idx = 0
    term.jacobi_body_idx = 0
    term.num_arm_joints = 1
    term.grasp_offsets_buffer = torch.zeros(num_envs, 3)
    term.grasp_offset_tensor = torch.zeros(3)
    term.grasp_rot_offset_tensor = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(num_envs, 1)
    term.finger_joints = []
    term.all_joints = [0]
    term.hand_hold_width = 0.0
    term.hand_close_width = 0.0
    term.gripper_joint_setter_func = lambda *args: None

    samples = []

    def sample_uniform(low, high, shape, device):
        samples.append((low.clone(), high.clone(), shape, device))
        return torch.full(shape, 0.001 * len(samples), device=device)

    monkeypatch.setattr(deploy_events.wp, "to_torch", lambda value: value)
    monkeypatch.setattr(deploy_events.math_utils, "sample_uniform", sample_uniform)
    monkeypatch.setattr(
        deploy_events.fc,
        "get_pose_error",
        lambda **kwargs: (torch.ones(num_envs, 3), torch.ones(num_envs, 3)),
    )
    monkeypatch.setattr(
        deploy_events.fc,
        "_get_delta_dof_pos",
        lambda **kwargs: torch.zeros(num_envs, 1),
    )

    term(
        env,
        torch.arange(num_envs),
        max_iterations=3,
        pos_randomization_range={"x": (-0.001, 0.001), "y": (-0.001, 0.001), "z": (-0.001, 0.001)},
    )

    assert len(samples) == 1


@pytest.mark.parametrize("enabled_knob", ("socket_pos", "socket_rot"))
def test_single_socket_knob_preserves_disabled_pose_component(enabled_knob):
    """A position-only or rotation-only knob must retain the other baseline range."""
    baseline = {
        "x": [-0.010, 0.010],
        "y": [-0.011, 0.011],
        "z": [-0.020, 0.020],
        "roll": [-0.021, 0.021],
        "pitch": [-0.022, 0.022],
        "yaw": [-0.023, 0.023],
    }
    events = SimpleNamespace(
        set_robot_to_grasp_pose=SimpleNamespace(params={}),
        randomize_socket_pose=SimpleNamespace(func=object(), params={"pose_range": deepcopy(baseline)}),
    )
    dr = DomainRandCfg()
    getattr(dr, enabled_knob).enable = True
    tracked = []

    _apply_reset_state(dr, events, lambda address, initial, final: tracked.append((address, initial, final)))

    _, initial, final = tracked[0]
    disabled_keys = ("roll", "pitch", "yaw") if enabled_knob == "socket_pos" else ("x", "y", "z")
    assert {key: initial[key] for key in disabled_keys} == {key: baseline[key] for key in disabled_keys}
    assert {key: final[key] for key in disabled_keys} == {key: baseline[key] for key in disabled_keys}
