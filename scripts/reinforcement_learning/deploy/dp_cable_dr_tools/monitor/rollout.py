# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out one checkpoint locally: a close-up video of env 0 plus an eval success rate.

The eval env is the nominal task (domain randomization off, no at-goal spawns), so every
run is scored on the same conditions regardless of which DR variant it trained with.

  python rollout.py <checkpoint.pt> <out_dir> [--envs 16] [--episodes 1] [--no-video]

Writes <out_dir>/iter_<N>.mp4 (unless --no-video) and prints one JSON line prefixed
"ROLLOUT_RESULT " with success rate and final errors.
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

TASK = "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("out_dir")
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--episodes", type=int, default=1, help="episodes per env to score")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("overrides", nargs="*", help="Hydra overrides the run trained with (e.g. agent.policy.*)")
    cli = p.parse_args()
    sys.argv = sys.argv[:1] + cli.overrides  # resolve_task_config reads sys.argv for Hydra overrides

    import importlib.metadata as metadata

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import add_launcher_args, launch_simulation

    from isaaclab_rl.entrypoints.common import apply_video_recording, enable_cameras_for_video, pre_launch_video_config
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    video = not cli.no_video
    it = int(re.search(r"model_(\d+)\.pt", cli.checkpoint).group(1))
    os.makedirs(cli.out_dir, exist_ok=True)
    work = os.path.join(cli.out_dir, f".work_{it}")
    shutil.rmtree(work, ignore_errors=True)

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args = parser.parse_args(["--viz", "kit" if video else "none"])
    args.video = video

    torch.manual_seed(0)
    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = cli.envs
    env_cfg.seed = 0
    env_cfg.dr.enabled = False
    pc = env_cfg.events.reset_plug_curriculum.params
    pc["at_goal_prob"] = 0.0
    pc["at_goal_prob_final"] = 0.0
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.75, 0.35, 0.25)
    env_cfg.viewer.lookat = (0.475, 0.125, 0.08)
    args.device = env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    max_len = int(round(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)))
    args.video_length = max_len
    args.video_interval = 10**9
    if video:
        enable_cameras_for_video(args)
        pre_launch_video_config(env_cfg, args_cli=args)

    t0 = time.time()
    with launch_simulation(env_cfg, args):
        if video:
            apply_video_recording(env_cfg, work, args, subdir="rollout")
        env = gym.make(TASK, cfg=env_cfg)
        u = env.unwrapped
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(cli.checkpoint)
        policy = runner.get_inference_policy(device=u.device)

        n = u.num_envs
        episodes_done = torch.zeros(n, dtype=torch.long, device=u.device)
        succ_now = torch.zeros(n, dtype=torch.bool, device=u.device)  # current episode so far
        successes = torch.zeros(n, dtype=torch.long, device=u.device)
        final_err = []
        obs = env.get_observations()
        steps = 0
        # Keep stepping until the clip is complete even if every episode ended early.
        while ((episodes_done < cli.episodes).any() or (video and steps < max_len + 2)) and steps < (
            cli.episodes + 1
        ) * max_len:
            with torch.inference_mode():
                # Success is a state check; score it before the step so the terminal step
                # (whose state is reset away inside env.step) is not lost.
                is_success, pos_err, _ = u._compute_success()
                succ_now |= is_success
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
            steps += 1
            d = dones.bool() & (episodes_done < cli.episodes)
            if d.any():
                successes[d] += succ_now[d].long()
                final_err.append(pos_err[d])
                episodes_done[d] += 1
            succ_now[dones.bool()] = False

        total = int(episodes_done.sum())
        errs = torch.cat(final_err) if final_err else torch.zeros(0)
        result = {
            "iteration": it,
            "checkpoint": cli.checkpoint,
            "episodes": total,
            "success_rate": round(successes.sum().item() / max(total, 1), 4),
            "final_pos_err_mm_median": round(errs.median().item() * 1000, 2) if len(errs) else None,
            "final_pos_err_mm_mean": round(errs.mean().item() * 1000, 2) if len(errs) else None,
            "env0_success": bool(successes[0].item() > 0),
            "seconds": round(time.time() - t0, 1),
            "video": None,
        }
        if video:
            clips = sorted(glob.glob(os.path.join(work, "**", "*.mp4"), recursive=True))
            if clips:
                dest = os.path.join(cli.out_dir, f"iter_{it:05d}.mp4")
                shutil.move(clips[0], dest)
                result["video"] = os.path.basename(dest)
        shutil.rmtree(work, ignore_errors=True)
        print("ROLLOUT_RESULT " + json.dumps(result), flush=True)
        # Kit ends the process during shutdown, so everything must be reported before this.
        env.close()


if __name__ == "__main__":
    main()
