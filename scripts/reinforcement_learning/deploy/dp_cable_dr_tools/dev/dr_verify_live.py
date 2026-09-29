# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure what the SIMULATOR actually applies at a given ADR level.

Config-level tests only prove an address resolves. Two stock terms turned out to cache
their range at construction and silently ignore the curriculum, so this reads back the
effective values from the live env instead.

Usage: python verify_dr_live.py <level>     (level 0 = easy endpoint, 50 = hard endpoint)
"""

import argparse
import json
import math
import sys

import gymnasium as gym
import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

TASK = "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference"
KNOBS = [
    "osc_stiffness",
    "osc_damping_ratio",
    "joint_armature",
    "joint_friction",
    "finger_friction",
    "grasp_pos",
    "grasp_rot",
    "socket_pos",
    "socket_rot",
    "obs_socket_pos",
    "obs_eef_pos",
    "obs_eef_rot",
    "obs_socket_rot",
    "action_noise",
    "action_latency",
    "plug_wrench_force",
]


def t(x):
    return x.torch if hasattr(x, "torch") else wp.to_torch(x)


def main():
    level = int(sys.argv[1])
    sys.argv = sys.argv[:1]  # resolve_task_config would parse leftover argv as Hydra overrides
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args = parser.parse_args(["--viz", "none"])

    torch.manual_seed(0)
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.scene.num_envs = 64
    dr = env_cfg.dr
    dr.enabled = True
    dr.adr.enable = True
    dr.adr.init_level = level
    dr.adr.success_threshold = 99.0  # hold the level fixed
    dr.at_goal_schedule = "adr"
    for k in KNOBS:
        getattr(dr, k).enable = True
    args.device = env_cfg.sim.device

    out = {"level": level}
    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        u = env.unwrapped
        robot, plug, socket = u.scene["robot"], u.scene["dp_plug"], u.scene["dp_socket"]
        eef_idx = robot.find_bodies(["flange"])[0][0]
        arm_ids = robot.find_joints(["joint[1-7]"])[0]
        zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)

        sock_off, grasp_rel, at_goal, stiff, fric, arm_ = [], [], [], [], [], []
        for _ in range(6):  # several resets -> several samples per env
            env.reset()
            env.step(zero)
            default = t(socket.data.default_root_pose)[:, :3] + u.scene.env_origins
            sock_off.append(t(socket.data.root_pos_w) - default)
            fpos = t(robot.data.body_pos_w)[:, eef_idx]
            fquat = t(robot.data.body_quat_w)[:, eef_idx]
            grasp_rel.append(math_utils.quat_apply_inverse(fquat, t(plug.data.root_pos_w) - fpos))
            at_goal.append(u.event_manager.get_term_cfg("reset_plug_curriculum").func.spawned_at_goal.float())
            osc = u.action_manager.get_term("arm_action")._osc
            stiff.append(torch.diagonal(osc._motion_p_gains_task, dim1=-2, dim2=-1)[:, :3] / 300.0)
            fric.append(t(robot.data.joint_friction_coeff)[:, arm_ids])
            arm_.append(t(robot.data.joint_armature)[:, arm_ids])

        so = torch.cat(sock_off)
        gr = torch.cat(grasp_rel)
        out["socket_offset_max_abs_mm"] = (so.abs().amax(0) * 1000).round(decimals=2).tolist()
        out["grasp_rel_spread_mm"] = ((gr.amax(0) - gr.amin(0)) * 1000).round(decimals=2).tolist()
        out["at_goal_fraction"] = round(torch.cat(at_goal).mean().item(), 3)
        s = torch.cat(stiff)
        out["osc_stiffness_scale_min_max"] = [round(s.min().item(), 3), round(s.max().item(), 3)]
        out["osc_stiffness_scale_median"] = round(s.median().item(), 3)
        f = torch.cat(fric)
        out["joint_friction_min_max"] = [round(f.min().item(), 4), round(f.max().item(), 4)]
        a = torch.cat(arm_)
        out["joint_armature_min_max"] = [round(a.min().item(), 5), round(a.max().item(), 5)]

        term = u.action_manager.get_term("arm_action")
        out["action_noise_halfwidth_live"] = term.cfg.action_noise_halfwidth
        out["action_latency_steps_seen"] = sorted(set(term._latency_steps.tolist()))
        noise = u.observation_manager.cfg.policy.socket_kp_pos.noise
        out["obs_socket_bias_nmax_live"] = noise.bias_noise_cfg.n_max
        out["obs_eef_bias_nmax_live"] = u.observation_manager.cfg.policy.eef_pos.noise.bias_noise_cfg.n_max
        out["obs_eef_rot_bias_nmax_deg"] = round(
            math.degrees(u.observation_manager.cfg.policy.eef_rot_6d.noise.bias_noise_cfg.n_max), 3
        )

        # Wrench: step past the max resampling interval (2 s = 60 steps) and read it back.
        for _ in range(70):
            env.step(zero)
        fb = t(plug.permanent_wrench_composer.out_force_b)
        fb = fb.reshape(u.num_envs, -1, 3)[:, 0]
        out["wrench_force_min_max_N"] = [round(fb.min().item(), 3), round(fb.max().item(), 3)]
        out["wrench_force_has_negative_components"] = bool((fb < 0).any().item())

        try:
            mats = plug.root_view.get_material_properties()  # noqa: F841 (probe only)
            fmats = robot.root_view.get_material_properties()
            fm = torch.as_tensor(fmats)[..., 0]
            out["robot_static_friction_min_max"] = [round(fm.min().item(), 3), round(fm.max().item(), 3)]
        except Exception as exc:  # noqa: BLE001
            out["robot_static_friction_min_max"] = f"unreadable: {type(exc).__name__}"

        print("DR_LIVE_RESULT " + json.dumps(out))
        env.close()


if __name__ == "__main__":
    main()
