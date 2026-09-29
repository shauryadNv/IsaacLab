# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Does gripper-finger friction actually hold the DP plug? A pull-out test.

The scene's default material combines friction by *multiplying*, and the plug is set to
mu = 0.001. If that multiply applies at the finger-plug contact, finger friction is
irrelevant: the effective mu is ~0.001 whatever the fingers are set to.

Pulls the plug out of the gripper along the tool axis with a known force and measures how
far it slides, with the arm held by zero actions. 32 envs: 4 force levels x 8 envs.

Configured through env vars:
    SLIP_FINGER_MU   finger friction (static = dynamic)      (required)
    SLIP_PLUG_MU     plug + socket friction                  (required)
    SLIP_STEPS       control steps the pull is held          (default: 60, i.e. 2 s)
"""

import argparse
import json
import os
import sys

import gymnasium as gym
import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

TASK = "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference"
FORCES_N = [0.5, 1.0, 2.0, 4.0]


def t(x):
    return x.torch if hasattr(x, "torch") else wp.to_torch(x)


def main():
    finger_mu = float(os.environ["SLIP_FINGER_MU"])
    plug_mu = float(os.environ["SLIP_PLUG_MU"])
    steps = int(os.environ.get("SLIP_STEPS", "60"))
    sys.argv = sys.argv[:1]

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args = parser.parse_args(["--viz", "none"])

    torch.manual_seed(0)
    env_cfg, _ = resolve_task_config(TASK, "")
    n_per = 8
    env_cfg.scene.num_envs = n_per * len(FORCES_N)
    env_cfg.events.reset_plug_curriculum.params["at_goal_prob"] = 0.0
    env_cfg.events.reset_plug_curriculum.params["at_goal_prob_final"] = 0.0
    env_cfg.episode_length_s = 30.0
    dr = env_cfg.dr
    dr.enabled = True
    dr.finger_friction.enable = True
    dr.finger_friction.initial = (finger_mu, finger_mu)
    dr.mating_friction.enable = True
    dr.mating_friction.initial = (plug_mu, plug_mu)
    args.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        u = env.unwrapped
        robot, plug = u.scene["robot"], u.scene["dp_plug"]
        eef = robot.find_bodies(["flange"])[0][0]
        zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)

        env.reset()
        for _ in range(5):  # let the grip settle
            env.step(zero)

        def plug_in_flange():
            fpos = t(robot.data.body_pos_w)[:, eef]
            fquat = t(robot.data.body_quat_w)[:, eef]
            return math_utils.quat_apply_inverse(fquat, t(plug.data.root_pos_w) - fpos), fquat

        rel0, fquat = plug_in_flange()
        # Pull along the tool axis (flange +Z), i.e. straight out of the fingers, in world frame.
        tool_axis_w = math_utils.quat_apply(fquat, torch.tensor([0.0, 0.0, 1.0], device=u.device).expand(u.num_envs, 3))
        mags = torch.tensor(FORCES_N, device=u.device).repeat_interleave(n_per).unsqueeze(-1)
        forces = (tool_axis_w * mags).unsqueeze(1)
        env_ids = torch.arange(u.num_envs, device=u.device, dtype=torch.int32)
        plug.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces, torques=torch.zeros_like(forces), body_ids=[0], env_ids=env_ids, is_global=True
        )
        for _ in range(steps):
            env.step(zero)
        rel1, _ = plug_in_flange()

        d = (rel1 - rel0) * 1000.0
        axial = d[:, 2].reshape(len(FORCES_N), n_per)
        total = d.norm(dim=-1).reshape(len(FORCES_N), n_per)
        result = {
            "finger_mu": finger_mu,
            "plug_mu": plug_mu,
            "pull_seconds": round(steps / 30.0, 2),
            "by_force": {
                f"{f}N": {
                    "axial_slip_mm_mean": round(axial[i].mean().item(), 2),
                    "total_slip_mm_mean": round(total[i].mean().item(), 2),
                    "total_slip_mm_max": round(total[i].max().item(), 2),
                    "dropped_frac": round((total[i] > 20.0).float().mean().item(), 2),
                }
                for i, f in enumerate(FORCES_N)
            },
        }
        print("SLIP_RESULT " + json.dumps(result))
        env.close()


if __name__ == "__main__":
    main()
