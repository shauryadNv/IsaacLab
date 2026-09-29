# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record what the plug wrench randomization does, with the arm holding still.

Holds the arm with zero (relative pose) actions so the only thing moving the plug is the
wrench, and records a close-up of env 0. Also prints how far the TCP drifted and how far
the plug moved inside the gripper.

Configured through env vars:
    DEMO_NAME      label and output subdir            (default: demo)
    DEMO_FORCE     per-axis force half-width  [N]     (default: 0)
    DEMO_STEPS     control steps to record            (default: 300, i.e. 10 s)
    DEMO_OUT       output root dir                    (default: rl_policy/dr_tools/results/wrench_videos)
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

from isaaclab_rl.entrypoints.common import apply_video_recording, enable_cameras_for_video, pre_launch_video_config

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

TASK = "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference"


def t(x):
    return x.torch if hasattr(x, "torch") else wp.to_torch(x)


def main():
    name = os.environ.get("DEMO_NAME", "demo")
    force = float(os.environ.get("DEMO_FORCE", "0"))
    steps = int(os.environ.get("DEMO_STEPS", "300"))
    out_root = os.environ.get("DEMO_OUT", os.path.join(os.environ.get("DR_TOOLS_RESULTS", "results"), "wrench_videos"))
    sys.argv = sys.argv[:1]

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args = parser.parse_args(["--viz", "kit"])
    args.video = True
    args.video_length = steps
    args.video_interval = 10**9  # a single clip starting at step 0
    enable_cameras_for_video(args)

    torch.manual_seed(0)
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.scene.num_envs = 4
    # Plug always starts in free air above the socket, not partially inserted.
    env_cfg.events.reset_plug_curriculum.params["at_goal_prob"] = 0.0
    env_cfg.events.reset_plug_curriculum.params["at_goal_prob_final"] = 0.0
    # Long episode so no reset happens mid-clip.
    env_cfg.episode_length_s = steps / 30.0 + 5.0
    # Close-up camera on env 0's socket (same as dr_visual_test.sh).
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.75, 0.35, 0.25)
    env_cfg.viewer.lookat = (0.475, 0.125, 0.08)

    dr = env_cfg.dr
    if force > 0:
        dr.enabled = True
        dr.plug_wrench_force.enable = True
        dr.plug_wrench_force.initial = (-force, force)
    args.device = env_cfg.sim.device
    pre_launch_video_config(env_cfg, args_cli=args)

    with launch_simulation(env_cfg, args):
        apply_video_recording(env_cfg, out_root, args, subdir=name)
        env = gym.make(TASK, cfg=env_cfg)
        u = env.unwrapped
        robot, plug = u.scene["robot"], u.scene["dp_plug"]
        eef = robot.find_bodies(["flange"])[0][0]
        zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)

        env.reset()
        env.step(zero)

        def snapshot():
            fpos = t(robot.data.body_pos_w)[:, eef].clone()
            fquat = t(robot.data.body_quat_w)[:, eef].clone()
            rel = math_utils.quat_apply_inverse(fquat, t(plug.data.root_pos_w) - fpos)
            return fpos, fquat, rel

        f0, q0, rel0 = snapshot()
        max_force = torch.zeros(u.num_envs, device=u.device)
        for _ in range(steps):
            env.step(zero)
            fb = t(plug.permanent_wrench_composer.out_force_b).reshape(u.num_envs, -1, 3)[:, 0]
            max_force = torch.maximum(max_force, fb.norm(dim=-1))
        f1, q1, rel1 = snapshot()

        tcp_drift_mm = ((f1 - f0).norm(dim=-1) * 1000).tolist()
        rot_drift_deg = (torch.rad2deg(math_utils.quat_error_magnitude(q0, q1))).tolist()
        slip_mm = ((rel1 - rel0).norm(dim=-1) * 1000).tolist()
        print(
            "WRENCH_DEMO "
            + json.dumps(
                {
                    "name": name,
                    "force_halfwidth_N": force,
                    "seconds": round(steps / 30.0, 2),
                    "flange_drift_mm": [round(v, 1) for v in tcp_drift_mm],
                    "flange_rot_drift_deg": [round(v, 2) for v in rot_drift_deg],
                    "plug_slip_in_gripper_mm": [round(v, 2) for v in slip_mm],
                    "peak_force_seen_N": [round(v, 3) for v in max_force.tolist()],
                }
            )
        )
        env.close()


if __name__ == "__main__":
    main()
